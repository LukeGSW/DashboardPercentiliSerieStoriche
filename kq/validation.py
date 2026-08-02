"""
=============================================================================
kq.validation — Walk-forward dei setup dello screener contro un null
=============================================================================
Risponde a una sola domanda: quando lo screener ha segnalato un nome, nelle
sedute successive quel nome ha fatto meglio di una selezione casuale fatta
LO STESSO GIORNO sullo STESSO universo?

IL NULL — e' la scelta che conta piu' di tutte
---------------------------------------------
Il null NON e' l'entrata casuale nel tempo. I setup di mean reversion scattano
in modo sproporzionato durante i drawdown: confrontarli con entrate distribuite
su tutto il periodo significa accreditare al segnale il recupero del mercato.

Il null corretto e' la SEZIONE TRASVERSALE CONTEMPORANEA: stesse date, stesso
holding, nomi estratti dallo stesso universo eleggibile quel giorno. In pratica
si misura

    extra_t = media(forward dei segnalati) − media(forward dell'universo)

che e' anche il rendimento di una posizione equipesata long sui segnalati contro
l'universo: toglie beta di mercato ed effetto periodo in un colpo solo, ed e'
esattamente il valore atteso di una selezione casuale di pari numerosita'.

CAUSALITA'
----------
Ogni grandezza e' calcolata su finestre mobili o espandenti, quindi alla data t
usa solo dati fino a t. L'esecuzione e' a t+1 (si entra alla chiusura
successiva al segnale). Le soglie dello screener sono costanti fissate a priori:
non c'e' nulla di stimato sui dati, quindi la divisione in-sample / out-of-sample
serve a misurare la STABILITA' nel tempo, non a proteggere da overfitting.

IL LIMITE CHE NON SI PUO' TOGLIERE
----------------------------------
L'universo e' costruito sui membri di oggi. Le societa' dislocate che sono
risalite ci sono; quelle andate a zero sono uscite e non compaiono da nessuna
parte. Il bias spinge quindi A FAVORE della tesi contrarian, e un risultato
positivo su MR-LONG va letto come LIMITE SUPERIORE, non come stima. Il modulo
riporta la "copertura" (quota di segnalati con un forward calcolabile) proprio
per rendere visibile quanto il problema sia invisibile in questi dati.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kq import config as C
from kq import scanner as S


# Numero minimo di nomi segnalati perche' una data entri nel campione: la media
# di uno o due titoli e' rumore, non un portafoglio.
MIN_NOMI_PER_DATA = 3
# Numero minimo di date valide per considerare misurabile una cella.
MIN_DATE = 24

DIREZIONE_TESI = {"MR-LONG": +1, "TREND-UP": +1, "MR-SHORT": -1, "TREND-DN": -1}


# =============================================================================
# 1. SEGNALI CAUSALI SU TUTTO IL PANNELLO
# =============================================================================
def precompute_signals(close: pd.DataFrame, volume: pd.DataFrame,
                       universe: pd.DataFrame, horizon_label: str = "YTD",
                       min_adv_usd: float = C.UNIVERSE_MIN_ADV_USD) -> dict:
    """
    Calcola in un colpo solo, per OGNI data e OGNI ticker, le stesse grandezze
    che `scanner.run_screen` calcola per l'ultima data.

    Rifare `run_screen` a ogni data storica costerebbe minuti; qui le finestre
    mobili vengono percorse una volta sola. Il risultato e' identico perche' le
    finestre sono causali per costruzione.
    """
    universe = universe[universe["ticker"].isin(close.columns)].copy()

    # Gli strumenti che sono benchmark di se stessi (SPY, TLT, GLD...) misurano
    # la dislocazione sulla propria storia stagionale: una versione causale
    # richiederebbe una mediana espandente condizionata al trading day. Sono
    # una decina e non sono il cuore dell'universo: si escludono dalla
    # validazione, dichiarandolo.
    self_bench = universe["benchmark"] == universe["ticker"]
    universe = universe[~self_bench]
    tickers = universe["ticker"].tolist()
    if not tickers:
        return {}

    returns = close.pct_change()
    tdi = S.trading_day_index(close.index)

    horizon_days = C.HORIZONS.get(horizon_label)
    hret = S.horizon_return_series(close, horizon_days)
    # Orizzonte effettivo in sedute: per lo YTD cresce lungo l'anno
    h_eff = (tdi if horizon_days is None else pd.Series(horizon_days, index=close.index))
    h_eff = h_eff.clip(lower=1)

    # --- volatilita' -------------------------------------------------------
    sigma_ann = returns.rolling(C.WIN_VOL_LONG).std() * np.sqrt(252)
    sigma_ann = sigma_ann.where(sigma_ann > 0)
    rv = returns.rolling(C.WIN_VOL_SHORT).std() * np.sqrt(252)
    # Percentile ESPANDENTE: alla data t confronta solo con le date <= t.
    # (run_screen usa tutta la storia, che per l'ultima data e' la stessa cosa;
    #  la convenzione sui ties differisce di 1/n, irrilevante e comunque non
    #  usata nella classificazione dei setup.)
    rv_pctl = rv.expanding(min_periods=C.WIN_VOL_LONG).rank(pct=True) * 100

    # --- beta e R^2 mobili, per gruppo di benchmark ------------------------
    beta = pd.DataFrame(np.nan, index=close.index, columns=tickers)
    r2 = pd.DataFrame(np.nan, index=close.index, columns=tickers)

    for bench, grp in universe.groupby("benchmark"):
        if bench not in returns.columns:
            continue
        cols = [t for t in grp["ticker"] if t in returns.columns and t != bench]
        if not cols:
            continue
        b = returns[bench]
        var_b = b.rolling(C.WIN_BETA).var()
        cov = returns[cols].rolling(C.WIN_BETA).cov(b)
        std_x = returns[cols].rolling(C.WIN_BETA).std()
        std_b = b.rolling(C.WIN_BETA).std()

        beta[cols] = cov.div(var_b.where(var_b > 0), axis=0)
        corr = cov.div((std_x.mul(std_b, axis=0)).where(lambda d: d > 0))
        r2[cols] = corr ** 2

    beta = beta.fillna(0.0)
    r2 = r2.fillna(0.0).clip(0, 1)
    sigma_idio = sigma_ann[tickers] * np.sqrt((1 - r2).clip(lower=0.04))

    # --- dislocazione e momentum residuo -----------------------------------
    bench_map = universe.set_index("ticker")["benchmark"]
    hret_bench = pd.DataFrame(
        {t: hret[bench_map[t]] for t in tickers}, index=close.index
    )
    resid = hret[tickers] - beta * hret_bench
    scala = sigma_idio.mul(np.sqrt(h_eff / 252), axis=0)
    disloc_z = resid / scala.where(scala > 0)

    w = C.WIN_VELOCITY
    ret_w = close / close.shift(w) - 1
    ret_w_bench = pd.DataFrame(
        {t: ret_w[bench_map[t]] for t in tickers}, index=close.index
    )
    scala_w = sigma_idio * np.sqrt(w / 252)
    resid_mom = (ret_w[tickers] - beta * ret_w_bench) / scala_w.where(scala_w > 0)

    # --- eleggibilita' -----------------------------------------------------
    adv = (close * volume).rolling(C.WIN_LIQUIDITY).mean()
    e_etf = universe.set_index("ticker")["bucket"].eq("ETF").reindex(tickers)
    storia = close[tickers].notna().cumsum()

    # Serie con salti anomali: escluse per l'INTERO campione, non solo dopo il
    # salto. Un concambio non gestito corrompe la serie in modo retroattivo, e
    # su una misura ordinata per estremita' quelle serie dominerebbero.
    sporche = (returns[tickers].abs() > C.QC_MAX_ABS_DAILY_RET).any()

    eleggibile = (
        close[tickers].notna()
        & (storia >= C.QC_MIN_OBS)
        & sigma_ann[tickers].notna()
        & ((adv[tickers] >= min_adv_usd) | e_etf)
        & disloc_z.notna()
        & resid_mom.notna()
    )
    eleggibile.loc[:, sporche[sporche].index] = False

    # --- rank cross-sectional (solo fra gli eleggibili) ---------------------
    hret_el = hret[tickers].where(eleggibile)
    xs_rank = hret_el.rank(axis=1, pct=True) * 100

    in_coda = (xs_rank <= C.TH_XS_TAIL) | (xs_rank >= 100 - C.TH_XS_TAIL)
    cs = in_coda.cumsum()
    gg_coda = (cs - cs.where(~in_coda).ffill().fillna(0)).clip(upper=90)

    return {
        "tickers": tickers,
        "universe": universe,
        "eleggibile": eleggibile,
        "disloc_z": disloc_z.where(eleggibile),
        "resid_mom": resid_mom.where(eleggibile),
        "xs_rank": xs_rank,
        "gg_coda": gg_coda,
        "rv_pctl": rv_pctl[tickers].where(eleggibile),
        "hret": hret[tickers],
        "orizzonte": horizon_label,
        "n_esclusi_sporchi": int(sporche.sum()),
        "n_esclusi_self_bench": int(self_bench.sum()),
    }


def maschere_setup(sig: dict) -> dict[str, pd.DataFrame]:
    """
    Stessa logica di `scanner.run_screen`: due assi ortogonali, quattro esiti
    esaustivi e mutuamente esclusivi.
    """
    z, mom = sig["disloc_z"], sig["resid_mom"]
    strong_dn = z <= -C.TH_RESID_Z
    strong_up = z >= C.TH_RESID_Z
    peggiora = mom <= -0.5
    migliora = mom >= 0.5

    return {
        "MR-LONG": strong_dn & ~peggiora,
        "TREND-DN": strong_dn & peggiora,
        "MR-SHORT": strong_up & ~migliora,
        "TREND-UP": strong_up & migliora,
    }


def filtro_livello(sig: dict, maschera: pd.DataFrame, setup: str, liv: dict) -> pd.DataFrame:
    """Applica un livello di selettivita' (stessa congiunzione della UI)."""
    out = maschera & (sig["disloc_z"].abs() >= liv["z"]) & (sig["gg_coda"] <= liv["gg"])
    if liv["mom"] is not None:
        if DIREZIONE_TESI[setup] > 0:
            out &= sig["resid_mom"] >= liv["mom"]
        else:
            out &= sig["resid_mom"] <= -liv["mom"]
    return out


# =============================================================================
# 2. RENDIMENTI FORWARD
# =============================================================================
def forward_returns(close: pd.DataFrame, holding: int) -> pd.DataFrame:
    """
    Rendimento realizzabile da un segnale alla data t: si entra alla chiusura
    di t+1 e si esce alla chiusura di t+1+holding. Nessun dato di t+1 in poi
    entra nel segnale, quindi nessun look-ahead.
    """
    entrata = close.shift(-1)
    uscita = close.shift(-(1 + holding))
    return uscita / entrata - 1


# =============================================================================
# 3. INFERENZA
# =============================================================================
def newey_west_tstat(x: np.ndarray, lag: int) -> float:
    """
    t di Student con errore standard Newey-West.

    Serve perche' le osservazioni SI SOVRAPPONGONO: campionando ogni 5 sedute
    con holding 20, ogni trade condivide 3/4 del periodo col successivo. Il t
    ingenuo sarebbe gonfiato di circa il doppio.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan

    mu = x.mean()
    e = x - mu
    s = float(e @ e) / n
    for l in range(1, min(int(lag), n - 1) + 1):
        peso = 1.0 - l / (lag + 1.0)
        s += 2.0 * peso * float(e[l:] @ e[:-l]) / n
    if s <= 0:
        return np.nan
    return mu / np.sqrt(s / n)


def block_bootstrap_pvalue(x: np.ndarray, block: int, n_boot: int, seed: int = 42) -> float:
    """
    P-value empirico bilaterale con bootstrap a blocchi.

    Il ricampionamento a blocchi conserva l'autocorrelazione indotta dalla
    sovrapposizione dei periodi di detenzione: un bootstrap i.i.d. la
    distruggerebbe e restituirebbe p-value troppo piccoli.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < MIN_DATE:
        return np.nan

    block = max(1, min(int(block), n // 3))
    osservato = x.mean()
    centrato = x - osservato          # sotto l'ipotesi nulla la media e' zero

    rng = np.random.default_rng(seed)
    n_blocchi = int(np.ceil(n / block))
    partenze = rng.integers(0, n - block + 1, size=(n_boot, n_blocchi))
    offset = np.arange(block)
    idx = (partenze[:, :, None] + offset).reshape(n_boot, -1)[:, :n]
    medie = centrato[idx].mean(axis=1)

    return float((np.abs(medie) >= abs(osservato)).mean())


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Correzione per test multipli (controllo del False Discovery Rate).

    Testare 4 setup x 3 livelli x 4 orizzonti significa 48 celle: senza
    correzione la migliore sembra buona per costruzione.
    """
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q

    pv = p[ok]
    ordine = np.argsort(pv)
    m = len(pv)
    qv = np.empty(m)
    precedente = 1.0
    for rank in range(m - 1, -1, -1):
        i = ordine[rank]
        val = pv[i] * m / (rank + 1)
        precedente = min(precedente, val)
        qv[i] = min(precedente, 1.0)
    q[ok] = qv
    return q


# =============================================================================
# 4. EVENT STUDY
# =============================================================================
def _statistiche(lordo: pd.Series, netto: pd.Series, n_nomi: pd.Series,
                 holding: int, rebalance: int, n_boot: int, seed: int) -> dict:
    """
    L'INFERENZA GIRA SUL LORDO, il NETTO e' il risultato economico.

    Il costo di transazione e' uno spostamento deterministico della serie:
    sottrarlo prima del t-test aggiunge distorsione senza aggiungere varianza,
    e gonfia il |t| in proporzione al costo ipotizzato. Con costi abbastanza
    alti qualunque risultato negativo diventerebbe "significativo", il che non
    vuol dire nulla. La domanda statistica ("esiste un segnale distinguibile
    dalla selezione casuale?") si pone quindi sul lordo; quella economica
    ("dopo i costi resta qualcosa?") si legge sul netto.
    """
    valide = lordo.dropna()
    if len(valide) < MIN_DATE:
        return {"n_date": len(valide)}

    x = valide.to_numpy()
    # Sovrapposizione: quante osservazioni consecutive condividono il periodo
    lag = max(1, int(np.ceil(holding / rebalance)))
    media_netta = float(netto.reindex(valide.index).mean())

    return {
        "n_date": len(valide),
        "n_trade": int(n_nomi.reindex(valide.index).sum()),
        "nomi_medi": float(n_nomi.reindex(valide.index).mean()),
        "extra_lordo": float(x.mean()),
        "extra_netto": media_netta,
        "hit_rate": float((x > 0).mean() * 100),
        "t_nw": newey_west_tstat(x, lag),
        "p_boot": block_bootstrap_pvalue(x, lag * 2, n_boot, seed),
        "extra_annuo": media_netta * (252.0 / holding),
    }


# =============================================================================
# 3-bis. BANDA PLACEBO
# =============================================================================
def placebo_band(fwd_reb: pd.DataFrame, eleggibili_reb: pd.DataFrame,
                 conteggi: pd.Series, n_rip: int = 30, seed: int = 7) -> dict:
    """
    Distribuzione dell'extra-rendimento sotto selezione puramente CASUALE, con
    lo stesso numero di nomi nelle stesse date.

    E' la verifica che il null sia calcolato bene e, soprattutto, la banda di
    rumore contro cui leggere i risultati: un extra dentro questa banda non e'
    un segnale, e' quello che produce il caso.
    """
    valide = conteggi[conteggi >= MIN_NOMI_PER_DATA].index
    if len(valide) < MIN_DATE:
        return {}

    F = fwd_reb.reindex(valide).to_numpy(dtype=float)
    E = eleggibili_reb.reindex(valide).to_numpy(dtype=bool)
    k = conteggi.reindex(valide).to_numpy(dtype=int)

    with np.errstate(invalid="ignore"):
        media_universo = np.nanmean(np.where(E, F, np.nan), axis=1)

    rng = np.random.default_rng(seed)
    esiti = np.empty(n_rip)
    for i in range(n_rip):
        r = rng.random(F.shape)
        r[~E] = np.inf
        # posizione di ciascun ticker in un ordinamento casuale fra gli eleggibili
        posizione = np.argsort(np.argsort(r, axis=1), axis=1)
        scelti = posizione < k[:, None]
        with np.errstate(invalid="ignore"):
            media_scelti = np.nanmean(np.where(scelti, F, np.nan), axis=1)
        esiti[i] = np.nanmean(media_scelti - media_universo)

    return {
        "media": float(np.nanmean(esiti) * 100),
        "p05": float(np.nanpercentile(esiti, 5) * 100),
        "p95": float(np.nanpercentile(esiti, 95) * 100),
        "n_rip": n_rip,
    }


def event_study(sig: dict, close: pd.DataFrame, livelli: dict,
                holdings: tuple[int, ...] = (5, 10, 20, 60),
                rebalance: int = 5, costo_bps: float = 10.0,
                n_boot: int = 1000, n_placebo: int = 25, seed: int = 42) -> pd.DataFrame:
    """
    Esegue lo studio per ogni combinazione setup x livello x holding.

    Restituisce una riga per cella, con il segno gia' orientato secondo la tesi
    del setup: `extra_per_trade` positivo significa sempre "ha funzionato".
    """
    if not sig:
        return pd.DataFrame()

    maschere = maschere_setup(sig)
    eleggibile = sig["eleggibile"]

    # Date di ribilanciamento, dopo il riscaldamento delle finestre mobili
    inizio = max(C.QC_MIN_OBS, C.WIN_BETA) + 5
    date_reb = close.index[inizio::rebalance]

    righe = []
    for holding in holdings:
        fwd = forward_returns(close, holding)[sig["tickers"]]
        fwd_reb = fwd.reindex(date_reb)
        el_reb = eleggibile.reindex(date_reb)

        # Null: media dell'universo eleggibile, stesse date, stesso holding
        media_universo = fwd_reb.where(el_reb).mean(axis=1)
        copertura_universo = fwd_reb.where(el_reb).notna().sum(axis=1) / el_reb.sum(axis=1)

        for setup, base in maschere.items():
            segno = DIREZIONE_TESI[setup]
            for nome_liv, liv in livelli.items():
                flag = filtro_livello(sig, base, setup, liv).reindex(date_reb).fillna(False)
                n_nomi = flag.sum(axis=1)

                media_flag = fwd_reb.where(flag).mean(axis=1)
                copertura = fwd_reb.where(flag).notna().sum(axis=1) / n_nomi.where(n_nomi > 0)

                valida = n_nomi >= MIN_NOMI_PER_DATA
                lordo = (segno * (media_flag - media_universo)).where(valida)
                # Il costo di andata e ritorno grava sempre sulla gamba
                # negoziata, qualunque sia la direzione della tesi.
                netto = lordo - costo_bps / 1e4

                st = _statistiche(lordo, netto, n_nomi, holding, rebalance, n_boot, seed)
                if st.get("n_date", 0) < MIN_DATE:
                    righe.append({"Setup": setup, "Livello": nome_liv, "Holding": holding,
                                  "n_date": st.get("n_date", 0), "misurabile": False})
                    continue

                # Stabilita': primi 2/3 contro ultimo 1/3 delle date valide.
                # Non essendoci parametri stimati sui dati, questo misura la
                # PERSISTENZA nel tempo, non protegge da overfitting.
                v = netto.dropna()
                taglio = int(len(v) * 2 / 3)
                is_, oos = v.iloc[:taglio], v.iloc[taglio:]

                banda = placebo_band(fwd_reb, el_reb, n_nomi.where(valida, 0),
                                     n_rip=n_placebo, seed=seed)

                righe.append({
                    "Setup": setup,
                    "Livello": nome_liv,
                    "Holding": holding,
                    "misurabile": True,
                    "Extra %": st["extra_netto"] * 100,
                    "Extra lordo %": st["extra_lordo"] * 100,
                    "Extra annuo %": st["extra_annuo"] * 100,
                    "t (NW)": st["t_nw"],
                    "p (bootstrap)": st["p_boot"],
                    "Hit %": st["hit_rate"],
                    "IS %": float(is_.mean() * 100) if len(is_) else np.nan,
                    "OOS %": float(oos.mean() * 100) if len(oos) else np.nan,
                    "Placebo p05 %": banda.get("p05", np.nan),
                    "Placebo p95 %": banda.get("p95", np.nan),
                    "n_date": st["n_date"],
                    "n_trade": st["n_trade"],
                    "Nomi/data": st["nomi_medi"],
                    "Copertura %": float(copertura.mean() * 100),
                    "Copertura universo %": float(copertura_universo.mean() * 100),
                    "_serie": netto,
                })

    df = pd.DataFrame(righe)
    if df.empty or "p (bootstrap)" not in df.columns:
        return df

    df["q (BH)"] = benjamini_hochberg(df["p (bootstrap)"].to_numpy())
    df["Esito"] = [_semaforo(r) for _, r in df.iterrows()]
    return df.sort_values(["Holding", "Setup", "Livello"]).reset_index(drop=True)


def _semaforo(r: pd.Series) -> str:
    """
    Semaforo. Soglia sul p-value volutamente permissiva, ma il confronto col
    null e' obbligatorio e c'e' un pavimento sul numero di osservazioni.

    Perche' due condizioni distinte: il p-value dice se il segnale si distingue
    dalla selezione casuale (domanda statistica, sul lordo), l'extra netto dice
    se dopo i costi resta qualcosa (domanda economica). Servono entrambe: un
    segnale reale ma piu' piccolo dei costi non e' operativo, e un extra netto
    positivo ma indistinguibile dal caso non e' un segnale.
    """
    if not r.get("misurabile", False):
        return "⚫ non misurabile"
    if r["n_date"] < MIN_DATE or r["n_trade"] < 100:
        return "⚫ campione insufficiente"

    netto_ok = r["Extra %"] > 0
    lordo_ok = r["Extra lordo %"] > 0
    stabile = (r["OOS %"] > 0) if pd.notna(r["OOS %"]) else False
    fuori_placebo = (pd.notna(r.get("Placebo p95 %")) and
                     r["Extra lordo %"] > r["Placebo p95 %"])
    q = r.get("q (BH)", np.nan)

    if not lordo_ok:
        return "🔴 contrario alla tesi"
    if netto_ok and stabile and fuori_placebo and pd.notna(q) and q <= 0.10 and r["t (NW)"] >= 2:
        return "🟢 regge"
    if netto_ok and pd.notna(q) and q <= 0.25:
        return "🟡 indiziario" + ("" if stabile else " · OOS debole")
    if lordo_ok and not netto_ok:
        return "🟠 mangiato dai costi"
    return "🟠 non distinguibile dal null"

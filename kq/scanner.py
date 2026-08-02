"""
=============================================================================
kq.scanner — Motore dello screener multi-asset
=============================================================================
Filosofia: questo modulo NON produce uno studio, produce una LISTA DI CANDIDATI
da vagliare poi uno per uno con l'analisi single-asset. Ogni metrica e' quindi
scelta per essere robusta e a basso costo, non per essere definitiva.

TRE SCELTE DI DISEGNO CHE VALE LA PENA ESPLICITARE
--------------------------------------------------
1) LA METRICA PRIMARIA E' CROSS-SECTIONAL, NON TIME-SERIES.
   Con storia dal 2015 si hanno ~11 osservazioni per trading day: un percentile
   storico su 11 punti ha granularita' ~9 punti percentuali ed e' inservibile
   come soglia. Il rank del titolo contro i ~700 pari di OGGI ha invece ~700
   campioni, e' robusto con qualunque profondita' di storia ed e' immune al
   survivorship bias dell'universo. Il percentile storico resta come colonna
   di contesto, non come criterio di selezione.

2) LA DISLOCAZIONE E' IDIOSINCRATICA E NORMALIZZATA PER VOLATILITA'.
   Un titolo giu' del 20% non e' "dislocato" se il suo settore e' giu' del 18%:
   e' beta. Il segnale e' il RESIDUO rispetto al benchmark, diviso per la
   volatilita' idiosincratica attesa sull'orizzonte. Cosi' un ETF obbligazionario
   e un semiconduttore diventano finalmente confrontabili sulla stessa scala.

3) LA NORMALIZZAZIONE USA SIGMA GIORNALIERA, NON LA DEVIAZIONE CROSS-ANNO.
   Sigma stimata su 63 rendimenti giornalieri e' molto meglio determinata della
   deviazione standard di 11 osservazioni annuali. Si scala per sqrt(h/252).

QUELLO CHE QUESTO MODULO NON FA
-------------------------------
Non valida l'edge. Lo `score` e' un'euristica di ORDINAMENTO trasparente e
decomponibile, non una probabilita' e non un backtest. La validazione (event
study contro il null, walk-forward, netto costi) e' l'analisi single-asset che
l'utente fa a valle sui candidati.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kq import config as C


# =============================================================================
# 1. PRIMITIVE SUL PANNELLO
# =============================================================================
def trading_day_index(index: pd.DatetimeIndex) -> pd.Series:
    """Trading Day Index 1-based, ricalcolato a ogni anno solare."""
    return pd.Series(1, index=index).groupby(index.year).cumsum()


def horizon_return_series(close: pd.DataFrame, horizon_days: int | None) -> pd.DataFrame:
    """
    Serie storica completa del rendimento di riferimento.

    horizon_days = None -> YTD (base = primo prezzo valido di ciascun anno)
    horizon_days = n    -> rendimento rolling a n sedute
    """
    if horizon_days is None:
        base = close.groupby(close.index.year).transform("first")
        return close / base - 1
    return close / close.shift(horizon_days) - 1


def trailing_streak(mask: pd.DataFrame) -> pd.Series:
    """
    Giorni consecutivi in cui la condizione e' vera CONTANDO DALL'ULTIMA RIGA
    all'indietro. Vettorizzato: cumprod sul frame rovesciato si azzera al primo
    False, quindi la somma e' esattamente la lunghezza della serie corrente.
    """
    if mask.empty:
        return pd.Series(dtype=float)
    rev = mask.iloc[::-1].astype(int)
    return rev.cumprod().sum()


# =============================================================================
# 2. CONTROLLI QUALITA' DATO
# =============================================================================
def quality_control(close: pd.DataFrame, volume: pd.DataFrame,
                    returns: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Diagnostica per ticker. Serve a togliere di mezzo le serie rotte PRIMA di
    qualunque ranking.

    Su Momentum Track era emerso che le serie EODHD corrotte (concambi non
    gestiti, fusioni, split mancati) generano crolli spuri. Su uno screener che
    ordina per estremita' della dislocazione quelle serie non finiscono
    "da qualche parte" nella classifica: finiscono in CIMA. Il bias dei dati
    sporchi favorisce sistematicamente la tesi contrarian, quindi il filtro e'
    attivo di default.
    """
    last_valid = close.apply(pd.Series.last_valid_index)
    staleness = (asof - pd.to_datetime(last_valid)).dt.days

    qc = pd.DataFrame(index=close.columns)
    qc["n_obs"] = close.notna().sum()
    qc["staleness_days"] = staleness
    qc["max_abs_ret_252"] = returns.tail(252).abs().max()
    qc["zero_vol_days_20"] = (volume.tail(20) <= 0).sum()
    qc["vol_63"] = returns.tail(C.WIN_VOL_LONG).std()

    qc["ok_storia"] = qc["n_obs"] >= C.QC_MIN_OBS
    qc["ok_fresco"] = qc["staleness_days"] <= C.QC_MAX_STALENESS_DAYS
    qc["ok_serie"] = qc["max_abs_ret_252"] <= C.QC_MAX_ABS_DAILY_RET
    qc["ok_scambi"] = qc["zero_vol_days_20"] <= C.QC_MAX_ZERO_VOL_DAYS
    # Serie DEGENERE: prezzo identico giorno dopo giorno. Capita con i titoli
    # sospesi, di cui EODHD continua a riportare l'ultima chiusura. Superano
    # tutti gli altri controlli (non hanno salti, non sono "ferme", hanno un
    # volume) ma hanno varianza nulla, e a valle mandano a zero il denominatore
    # di ogni normalizzazione. Vanno intercettate qui.
    qc["ok_movimento"] = qc["vol_63"] > 0

    qc["eleggibile"] = qc[
        ["ok_storia", "ok_fresco", "ok_serie", "ok_scambi", "ok_movimento"]
    ].all(axis=1)

    # I motivi vanno formattati difendendosi dai NaN: una colonna interamente
    # vuota non ha una data di ultima quotazione, quindi staleness e' NaN e un
    # int() secco farebbe cadere la pagina ("cannot convert float NaN to integer").
    # Le voci non calcolabili si omettono invece di stampare "n/d": se non c'e'
    # nessun dato il motivo e' uno solo, non quattro.
    motivi = []
    for t in qc.index:
        r = qc.loc[t]
        if r["eleggibile"]:
            motivi.append("")
            continue
        if r["n_obs"] == 0:
            motivi.append("nessun dato nel periodo")
            continue

        m = []
        if not r["ok_storia"]:
            m.append(f"storia {int(r['n_obs'])}gg")
        if not r["ok_fresco"] and not pd.isna(r["staleness_days"]):
            m.append(f"fermo da {int(r['staleness_days'])}gg")
        if not r["ok_serie"] and not pd.isna(r["max_abs_ret_252"]):
            m.append(f"salto {r['max_abs_ret_252']:.0%}")
        if not r["ok_scambi"]:
            m.append("volumi nulli")
        if not r["ok_movimento"]:
            m.append("prezzo fermo (serie degenere)")
        motivi.append(", ".join(m) or "dati non utilizzabili")
    qc["motivo_esclusione"] = motivi

    return qc


# =============================================================================
# 3. BETA E RESIDUO IDIOSINCRATICO
# =============================================================================
def compute_beta_r2(returns: pd.DataFrame, universe: pd.DataFrame,
                    window: int = C.WIN_BETA):
    """
    Beta e R^2 di ogni strumento rispetto al proprio benchmark, su `window`
    sedute. Calcolato per gruppo di benchmark: 11-15 operazioni vettoriali
    invece di una regressione per ticker.

    beta = corr * sigma_titolo / sigma_benchmark
    """
    tail = returns.tail(window)
    beta = pd.Series(index=universe["ticker"], dtype=float)
    r2 = pd.Series(index=universe["ticker"], dtype=float)

    sd_all = tail.std()

    for bench, grp in universe.groupby("benchmark"):
        if bench not in tail.columns:
            continue
        cols = [t for t in grp["ticker"] if t in tail.columns and t != bench]
        if not cols:
            continue
        b = tail[bench]
        sd_b = b.std()
        if not np.isfinite(sd_b) or sd_b == 0:
            continue
        corr = tail[cols].corrwith(b)
        beta.loc[cols] = (corr * sd_all[cols] / sd_b).values
        r2.loc[cols] = (corr ** 2).values

    return beta, r2


# =============================================================================
# 4. MOTORE PRINCIPALE
# =============================================================================
def run_screen(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    universe: pd.DataFrame,
    horizon_label: str = "YTD",
    min_adv_usd: float = C.UNIVERSE_MIN_ADV_USD,
    min_price: float = C.UNIVERSE_MIN_PRICE,
    apply_qc: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Esegue lo screening completo sul pannello.

    Returns:
        (tabella risultati, dizionario di contesto/breadth)
    """
    if close.empty:
        return pd.DataFrame(), {}

    # Il pannello deve coprire almeno la finestra piu' lunga usata a valle
    # (SMA200, beta a 252 sedute, lag di momentum). Meglio fermarsi con un
    # messaggio chiaro che propagare indici fuori range o NaN silenziosi.
    min_righe = max(C.QC_MIN_OBS, 2 * C.WIN_VELOCITY + 1, 200)
    if len(close) < min_righe:
        return pd.DataFrame(), {
            "errore": f"Storia troppo corta: {len(close)} sedute nel pannello, "
                      f"ne servono almeno {min_righe}. Sposta indietro l'anno di inizio."
        }

    asof = close.index[-1]
    returns = close.pct_change()
    tdi = trading_day_index(close.index)
    tdi_now = int(tdi.iloc[-1])

    # --- Orizzonte di riferimento --------------------------------------------
    horizon_days = C.HORIZONS.get(horizon_label, None)
    horizon_used = horizon_label
    if horizon_days is None and tdi_now < C.YTD_MIN_TDI:
        # A gennaio l'YTD copre poche sedute: non e' informativo, si ripiega su 3 mesi
        horizon_days = C.HORIZONS["3 mesi"]
        horizon_used = "3 mesi (YTD troppo corto)"

    hret_series = horizon_return_series(close, horizon_days)
    hret = hret_series.iloc[-1]
    h_eff = tdi_now if horizon_days is None else horizon_days

    # --- Qualita' dato e liquidita' ------------------------------------------
    qc = quality_control(close, volume, returns, asof)

    price = close.iloc[-1]
    adv = (close * volume).rolling(C.WIN_LIQUIDITY).mean().iloc[-1]

    universe = universe[universe["ticker"].isin(close.columns)].copy()
    idx = universe["ticker"].tolist()

    eleggibile = pd.Series(True, index=idx)
    if apply_qc:
        eleggibile &= qc["eleggibile"].reindex(idx).fillna(False)
    eleggibile &= price.reindex(idx).fillna(0) >= min_price
    # Gli ETF non devono passare il filtro ADV azionario: molti ETF settoriali
    # scambiano meno di 30 M$/giorno ma hanno catene opzioni perfettamente liquide.
    is_etf = universe.set_index("ticker")["bucket"].reindex(idx).eq("ETF")
    eleggibile &= (adv.reindex(idx).fillna(0) >= min_adv_usd) | is_etf

    validi = [t for t in idx if bool(eleggibile.get(t, False))]
    if len(validi) < 20:
        return pd.DataFrame(), {"errore": "Universo eleggibile troppo piccolo dopo i filtri."}

    # --- Rank cross-sectional (metrica primaria) -----------------------------
    # Calcolato SOLO sui ticker eleggibili: includere serie rotte falserebbe
    # il rank di tutti gli altri.
    hs = hret_series[validi]
    xs_now = hs.iloc[-1].rank(pct=True) * 100
    xs_lag1 = hs.iloc[-1 - C.WIN_VELOCITY].rank(pct=True) * 100
    xs_lag2 = hs.iloc[-1 - 2 * C.WIN_VELOCITY].rank(pct=True) * 100

    velocity = xs_now - xs_lag1
    acceleration = (xs_now - xs_lag1) - (xs_lag1 - xs_lag2)

    # --- Persistenza in coda -------------------------------------------------
    rank_hist = hs.tail(90).rank(axis=1, pct=True) * 100
    giorni_coda_bassa = trailing_streak(rank_hist <= C.TH_XS_TAIL)
    giorni_coda_alta = trailing_streak(rank_hist >= (100 - C.TH_XS_TAIL))
    giorni_in_coda = giorni_coda_bassa.where(giorni_coda_bassa > 0, giorni_coda_alta)

    # --- Volatilita' ---------------------------------------------------------
    # Il .where(>0) e' una rete di sicurezza: i controlli qualita' scartano gia'
    # le serie a varianza nulla, ma sigma finisce a denominatore di ogni
    # normalizzazione e uno zero superstite propagherebbe inf silenziosi.
    sigma_ann = returns[validi].rolling(C.WIN_VOL_LONG).std().iloc[-1] * np.sqrt(252)
    sigma_ann = sigma_ann.where(sigma_ann > 0)
    rv = returns[validi].rolling(C.WIN_VOL_SHORT).std() * np.sqrt(252)
    rv_now = rv.iloc[-1]
    n_rv = rv.notna().sum()
    rv_pctl = (rv.lt(rv_now, axis=1).sum() / n_rv.where(n_rv > 0)) * 100

    # --- Residuo idiosincratico ----------------------------------------------
    uni_v = universe[universe["ticker"].isin(validi)].copy()
    beta, r2 = compute_beta_r2(returns, uni_v)
    # Gli strumenti che sono benchmark di se stessi (SPY, TLT, GLD...) non hanno
    # beta: si azzerano beta e R^2 cosi' il "residuo" coincide con il rendimento
    # grezzo e la volatilita' idiosincratica con quella totale. E' il
    # comportamento corretto per uno strumento che non ha nulla da neutralizzare.
    beta = beta.reindex(validi).fillna(0.0)
    r2 = r2.reindex(validi).fillna(0.0)

    bench_map = uni_v.set_index("ticker")["benchmark"].reindex(validi)
    self_bench = bench_map.eq(pd.Series(validi, index=validi))
    hret_bench = bench_map.map(hret).astype(float).where(~self_bench, 0.0).fillna(0.0)

    resid = hret.reindex(validi) - beta * hret_bench
    # Volatilita' idiosincratica: sigma * sqrt(1 - R^2), con floor per non far
    # esplodere lo z quando il benchmark spiega quasi tutto.
    sigma_idio = sigma_ann * np.sqrt((1 - r2).clip(lower=0.04))
    resid_z = resid / (sigma_idio * np.sqrt(max(h_eff, 1) / 252))

    # --- Momentum residuo di breve: la misura di stabilizzazione --------------
    # NON si usa la velocity del rank per decidere se una dislocazione si sta
    # chiudendo: il rank e' limitato in [0,100] e SATURA. Un titolo gia'
    # inchiodato all'ultimo percentile ha velocity esattamente zero, che
    # verrebbe letta come "si e' stabilizzato" mentre invece sta ancora
    # crollando. Il momentum residuo e' continuo e non limitato, quindi
    # distingue davvero "ha smesso di scendere" da "e' gia' in fondo".
    w = C.WIN_VELOCITY
    # Calcolato sull'intero pannello, non solo sui ticker eleggibili: un
    # benchmark settoriale scartato dal QC renderebbe altrimenti NaN il
    # momentum di tutti i titoli che lo usano, azzerandone i setup in silenzio.
    ret_w_all = close.iloc[-1] / close.iloc[-1 - w] - 1
    ret_w = ret_w_all.reindex(validi)
    ret_w_bench = bench_map.map(ret_w_all).astype(float).where(~self_bench, 0.0).fillna(0.0)
    resid_mom = (ret_w - beta * ret_w_bench) / (sigma_idio * np.sqrt(w / 252))

    # --- Dislocazione time-series (contesto) ---------------------------------
    if horizon_days is None:
        mask_T = (tdi == tdi_now).values
        ytd_at_T = hret_series.loc[mask_T, validi]
        ytd_at_T.index = ytd_at_T.index.year
        hist = ytd_at_T.drop(index=asof.year, errors="ignore")
    else:
        hist = hret_series[validi].iloc[:-1]

    n_hist = hist.notna().sum()
    med_hist = hist.median()
    pctl_ts = (hist.lt(hret.reindex(validi), axis=1).sum() / n_hist.where(n_hist > 0)) * 100
    z_ts = (hret.reindex(validi) - med_hist) / (sigma_ann * np.sqrt(max(h_eff, 1) / 252))

    # Metrica unificata: il residuo dove ha senso, la dislocazione storica per
    # gli strumenti che sono benchmark di se stessi (SPY, TLT, GLD, ...).
    disloc_z = resid_z.where(~self_bench, z_ts)
    tipo_disloc = pd.Series(
        np.where(self_bench, "vs storia", "vs benchmark"), index=validi
    )

    # --- Altri contesti ------------------------------------------------------
    sma200 = close[validi].rolling(200).mean().iloc[-1]
    sopra_sma200 = price.reindex(validi) > sma200
    dd_252 = price.reindex(validi) / close[validi].tail(252).max() - 1

    # =========================================================================
    # 5. CLASSIFICAZIONE SETUP
    # =========================================================================
    # Due assi ortogonali:
    #   AMPIEZZA  della dislocazione  -> disloc_z
    #   DIREZIONE in cui si sta muovendo ORA -> resid_mom (non satura)
    #
    # La combinazione e' esaustiva e mutuamente esclusiva. E' esattamente la
    # distinzione che uno screener deve fare: "dislocato e ha smesso di
    # muoversi contro" (candidato rimbalzo) NON e' "dislocato e sta ancora
    # andando giu'" (trend in corso, da non prendere in controtendenza).
    strong_dn = disloc_z <= -C.TH_RESID_Z
    strong_up = disloc_z >= C.TH_RESID_Z
    peggiora = resid_mom <= -0.5
    migliora = resid_mom >= 0.5

    setup = pd.Series("—", index=validi, dtype=object)
    setup[strong_dn & ~peggiora] = "MR-LONG"    # dislocato giu', non peggiora piu'
    setup[strong_dn & peggiora] = "TREND-DN"    # dislocato giu' e ancora in caduta
    setup[strong_up & ~migliora] = "MR-SHORT"   # dislocato su, ha smesso di salire
    setup[strong_up & migliora] = "TREND-UP"    # dislocato su e ancora in corsa

    vol_flag = pd.Series("—", index=validi, dtype=object)
    vol_flag[rv_pctl <= C.TH_VOL_LOW] = "COMPRESSA"
    vol_flag[rv_pctl >= C.TH_VOL_HIGH] = "RICCA"

    # =========================================================================
    # 6. SCORE (euristica di ordinamento, trasparente e decomponibile)
    # =========================================================================
    s_disloc = (disloc_z.abs() / 3.0).clip(0, 1)
    s_fresh = ((C.TH_STALE_DAYS - giorni_in_coda.reindex(validi).fillna(0))
               / (C.TH_STALE_DAYS - C.TH_FRESH_DAYS)).clip(0, 1)
    # Conferma direzionale: quanto il momentum residuo di breve va nella
    # direzione implicita dal setup. Usa resid_mom, non la velocity di rank.
    direzione_rialzista = setup.isin(["MR-LONG", "TREND-UP"])
    s_stab = pd.Series(
        np.where(direzione_rialzista,
                 (resid_mom / 2.0).clip(0, 1),
                 (-resid_mom / 2.0).clip(0, 1)),
        index=validi,
    ).fillna(0.0)
    s_vol = (rv_pctl / 100).fillna(0.5)
    s_liq = (np.log10(adv.reindex(validi).clip(lower=1e6) / 1e7) / 2).clip(0, 1)

    score = 100 * (
        0.40 * s_disloc
        + 0.20 * s_fresh
        + 0.20 * s_stab
        + 0.10 * s_vol
        + 0.10 * s_liq
    )
    score = score.where(setup != "—", np.nan)

    # =========================================================================
    # 7. STRUTTURA OPZIONI NATURALE (mappatura meccanica, non un consiglio)
    # =========================================================================
    def _struttura(row) -> str:
        s, v = row["setup"], row["vol_flag"]
        if s == "MR-LONG":
            return "Bull put spread / vendita put" if v == "RICCA" else "Call debit spread"
        if s == "MR-SHORT":
            return "Bear call spread / vendita call" if v == "RICCA" else "Put debit spread"
        if s == "TREND-UP":
            return "Call diagonal / long call" if v == "COMPRESSA" else "Bull call spread"
        if s == "TREND-DN":
            return "Long put / put diagonal" if v == "COMPRESSA" else "Bear put spread"
        if v == "COMPRESSA":
            return "Long straddle / strangle"
        if v == "RICCA":
            return "Short strangle / iron condor"
        return "—"

    # =========================================================================
    # 8. TABELLA FINALE
    # =========================================================================
    meta = uni_v.set_index("ticker")

    out = pd.DataFrame({
        "Ticker": [t.replace(".US", "") for t in validi],
        "ticker_eodhd": validi,
        "Nome": meta["nome"].reindex(validi).values,
        "Tipo": meta["bucket"].reindex(validi).values,
        "Categoria": meta["categoria"].reindex(validi).values,
        "Benchmark": [str(b).replace(".US", "") for b in bench_map.values],
        "setup": setup.values,
        "vol_flag": vol_flag.values,
        "Score": score.values,
        "Rend %": (hret.reindex(validi) * 100).values,
        "Disloc σ": disloc_z.values,
        "Tipo disloc": tipo_disloc.values,
        "Rank XS": xs_now.values,
        "Mom residuo": resid_mom.values,
        "Velocity": velocity.values,
        "Accel": acceleration.values,
        "GG in coda": giorni_in_coda.reindex(validi).fillna(0).values,
        "Vol %": (sigma_ann * 100).values,
        "Vol pctl": rv_pctl.values,
        "Beta": beta.values,
        "R²": r2.values,
        "Pctl storico": pctl_ts.values,
        "Anni storia": n_hist.values,
        "DD 52w %": (dd_252 * 100).values,
        ">SMA200": sopra_sma200.values,
        "Prezzo": price.reindex(validi).values,
        "ADV M$": (adv.reindex(validi) / 1e6).values,
        # componenti dello score, per rendere l'ordinamento ispezionabile
        "_s_disloc": s_disloc.values,
        "_s_fresh": s_fresh.values,
        "_s_stab": s_stab.values,
        "_s_vol": s_vol.values,
        "_s_liq": s_liq.values,
    })

    out["Struttura"] = out.apply(_struttura, axis=1)
    out = out.sort_values("Score", ascending=False, na_position="last").reset_index(drop=True)

    # =========================================================================
    # 9. CONTESTO DI MERCATO (BREADTH)
    # =========================================================================
    disp_series = (hs.tail(504).quantile(0.75, axis=1) - hs.tail(504).quantile(0.25, axis=1))
    disp_now = float(disp_series.iloc[-1])
    disp_pctl = float((disp_series < disp_now).sum() / disp_series.notna().sum() * 100) \
        if disp_series.notna().sum() > 0 else np.nan

    contesto = {
        "asof": asof,
        "tdi": tdi_now,
        "orizzonte": horizon_used,
        "n_universo": len(idx),
        "n_eleggibili": len(validi),
        "n_esclusi": len(idx) - len(validi),
        "pct_disloc_giu": float((disloc_z < -1).mean() * 100),
        "pct_disloc_su": float((disloc_z > 1).mean() * 100),
        "mediana_rend": float(hret.reindex(validi).median() * 100),
        "pct_sopra_sma200": float(sopra_sma200.mean() * 100),
        "dispersione": disp_now * 100,
        "dispersione_pctl": disp_pctl,
        "mediana_vol_pctl": float(rv_pctl.median()),
        "conteggio_setup": out["setup"].value_counts().to_dict(),
        "qc": qc,
    }

    return out, contesto


# =============================================================================
# 10. LETTURA DEL CONTESTO
# =============================================================================
def interpreta_contesto(ctx: dict) -> tuple[str, str]:
    """
    Traduce la dispersione cross-sectional in una indicazione operativa.

    E' l'informazione piu' importante del pannello breadth: quando la dispersione
    e' compressa i titoli si muovono tutti insieme, il residuo idiosincratico e'
    piccolo e i segnali dello screener sono in gran parte rumore. Quando e'
    elevata, l'ambiente premia la selezione.
    """
    p = ctx.get("dispersione_pctl", np.nan)
    if pd.isna(p):
        return "info", "Dispersione non calcolabile con la storia disponibile."

    if p >= 70:
        return "success", (
            f"**Dispersione alta** ({p:.0f}° percentile a 2 anni). I titoli si muovono "
            f"in ordine sparso: il residuo idiosincratico e' informativo e i candidati "
            f"dello screener hanno piu' probabilita' di essere dislocazioni vere."
        )
    if p <= 30:
        return "warning", (
            f"**Dispersione compressa** ({p:.0f}° percentile a 2 anni). Il mercato si muove "
            f"in blocco: gran parte della dislocazione apparente e' beta residuo e rumore. "
            f"Da trattare con scetticismo, o passare a un orizzonte piu' lungo."
        )
    return "info", (
        f"**Dispersione nella media** ({p:.0f}° percentile a 2 anni). Ambiente ordinario "
        f"per la selezione cross-sectional."
    )

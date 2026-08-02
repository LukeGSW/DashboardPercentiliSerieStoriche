"""
=============================================================================
kq.ui_scanner — Tab "Screener Multi-Asset"
=============================================================================
Lo screener e' un FILTRO DI PRIMO LIVELLO: restituisce candidati, non verdetti.
Ogni riga della tabella e' un ticker da portare poi nell'analisi single-asset,
dove si fanno i conti seri (percentili stagionali su storia lunga, regime,
forward returns condizionali).
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kq import charts
from kq import config as C
from kq import data as D
from kq import scanner as S
from kq import state
from kq import universe as U
from kq import validation as V

# Chiave di navigazione screener -> analisi single-asset.
# Non e' la chiave di un widget: e' una richiesta che app.main() consuma
# all'inizio del run successivo, prima di istanziare qualunque widget.
NAV_TICKER = "_nav_ticker_richiesto"
MODALITA_SINGOLO = "📈 Analisi singolo asset"


# =============================================================================
# CARICAMENTO
# =============================================================================
def _load_everything(start_date: str, n_stocks: int, min_price: float, api_key: str):
    """Universo + pannello prezzi. Entrambi cacheati; il pannello persiste su disco."""
    cache_day = D.cache_day_key()

    uni = U.build_universe(n_stocks, min_price, cache_day, api_key)
    tickers = sorted(set(uni["ticker"].tolist()) | set(U.required_benchmark_tickers(uni)))

    close, volume = D.load_screener_panel(tuple(tickers), start_date, cache_day, api_key)
    if close.empty:
        return uni, close, volume

    # Riassegna i benchmark dei singoli titoli per massima correlazione
    uni = U.assign_benchmarks_by_correlation(uni, close.pct_change())
    return uni, close, volume


# =============================================================================
# COMPONENTI UI
# =============================================================================
def _pannello_contesto(ctx: dict, risultati: pd.DataFrame) -> None:
    st.markdown("#### 🌐 Contesto di mercato")

    cols = st.columns(6)
    cols[0].metric("Universo eleggibile", f"{ctx['n_eleggibili']}",
                   delta=f"−{ctx['n_esclusi']} esclusi" if ctx["n_esclusi"] else None,
                   delta_color="off")
    cols[1].metric("Rendimento mediano", f"{ctx['mediana_rend']:+.1f}%")
    cols[2].metric("Sopra SMA200", f"{ctx['pct_sopra_sma200']:.0f}%")
    cols[3].metric("Dislocati ↓ (< −1σ)", f"{ctx['pct_disloc_giu']:.0f}%")
    cols[4].metric("Dislocati ↑ (> +1σ)", f"{ctx['pct_disloc_su']:.0f}%")
    cols[5].metric("Vol pctl mediano", f"{ctx['mediana_vol_pctl']:.0f}°")

    livello, testo = S.interpreta_contesto(ctx)
    getattr(st, livello)(testo)


# Livelli di selettività: agiscono sulle grandezze INTERPRETABILI, non sullo score.
# Lo score è una somma pesata arbitraria e non è calibrato su nulla: "score >= 60"
# non vuol dire niente in termini di probabilità. La dislocazione in sigma invece
# ha una coda nota, quindi è su quella che ha senso stringere.
#
#   |sigma|   P sotto normalità   attesi su ~650 strumenti
#     1.5          13.4%                  ~87
#     2.0           4.6%                  ~30
#     2.5           1.2%                   ~8
#     3.0           0.27%                  ~2
#
# I rendimenti hanno code più grasse della normale, quindi i conteggi reali sono
# più alti; l'ordine di grandezza però regge, ed è quello che serve per scegliere.
LIVELLI_SELETTIVITA = {
    "Tutti i candidati": {"z": 1.5, "gg": 90, "mom": None},
    "Selettivo": {"z": 2.0, "gg": 30, "mom": 0.3},
    "Alta convinzione": {"z": 2.5, "gg": 20, "mom": 0.5},
}

LIVELLO_KEY = "livello_selettivita"
LIVELLO_DEFAULT = "Selettivo"

# Vocabolari dei filtri: COSTANTI, mai derivati dai risultati del giorno.
# Un widget con chiave ripristina la selezione salvata, e Streamlit solleva
# se un valore salvato non e' fra le opzioni correnti: con opzioni che
# cambiano di giorno in giorno l'app si romperebbe in modo intermittente.
SETUP_VALIDI = list(C.SETUP_ORDINE)
TIPI_VALIDI = ["Azione", "ETF"]
CATEGORIE_VALIDE = sorted({cat for _, cat, _ in C.ETF_UNIVERSE} | {"Large Cap US"})


# =============================================================================
# PERSISTENZA DELLE IMPOSTAZIONI  (meccanica in kq.state)
# =============================================================================
DEFAULT_IMPOSTAZIONI = {
    LIVELLO_KEY: LIVELLO_DEFAULT,
    "f_setup": SETUP_VALIDI,
    "f_tipo": TIPI_VALIDI,
    "f_cat": [],
    "f_vol": [],
    "f_min_z": 1.0,
    "f_min_adv": 0,
    "f_max_gg": 90,
    "p_anno": 2015,
    "p_n_stocks": C.UNIVERSE_N_STOCKS,
    "p_min_adv": C.UNIVERSE_MIN_ADV_USD / 1e6,
    "p_orizzonte": next(iter(C.HORIZONS)),
    "p_qc": True,
}
_SPECCHIO = "_scr_"


def ripristina_impostazioni(screener_attivo: bool) -> None:
    """Da chiamare in app.main() PRIMA di istanziare qualunque widget."""
    state.ripristina(DEFAULT_IMPOSTAZIONI, _SPECCHIO, screener_attivo)


def salva_impostazioni() -> None:
    """Da chiamare dopo aver creato i widget dello screener."""
    state.salva(DEFAULT_IMPOSTAZIONI, _SPECCHIO)


def _applica_livello(df: pd.DataFrame, liv: dict) -> pd.DataFrame:
    """
    Filtro per congiunzione. È la congiunzione a rendere raro un candidato:
    presa singolarmente ogni condizione è comune, tutte insieme no.
    """
    out = df[df["Disloc σ"].abs() >= liv["z"]]
    out = out[out["GG in coda"].fillna(0) <= liv["gg"]]
    if liv["mom"] is not None:
        # Si stringe sulla PUREZZA DELLO STATO, non sulla direzione
        # dell'operazione: uno strumento esteso al rialzo deve avere momentum
        # nettamente positivo per essere un esemplare puro di quello stato,
        # anche se poi lo si tratta al ribasso.
        verso_alto = out["setup"].map(V.VERSO_STATO).fillna(0) > 0
        puro = pd.Series(
            np.where(verso_alto,
                     out["Mom residuo"] >= liv["mom"],
                     out["Mom residuo"] <= -liv["mom"]),
            index=out.index,
        )
        out = out[puro.fillna(False)]
    return out


def _filtri(risultati: pd.DataFrame) -> pd.DataFrame:
    """Barra filtri sopra la tabella."""
    candidati = risultati[risultati["setup"] != "—"]
    conteggi = {k: len(_applica_livello(candidati, v)) for k, v in LIVELLI_SELETTIVITA.items()}

    # Chiave esplicita + valore inizializzato in app.main(): senza, lo stato del
    # widget verrebbe riciclato da Streamlit ogni volta che si passa all'altra
    # modalita' (lo screener non e' renderizzato) e la scelta andrebbe persa.
    # Niente `index=`: il valore iniziale arriva dalla session_state.
    livello = st.radio(
        "Selettività",
        list(LIVELLI_SELETTIVITA.keys()),
        horizontal=True,
        key=LIVELLO_KEY,
        help="Stringe sulla dislocazione in σ, sulla freschezza e sulla concordanza del "
             "momentum — non sullo Score, che non è calibrato su nulla. Sotto normalità "
             "|σ|≥1.5 seleziona il 13% dell'universo, |σ|≥2 il 4.6%, |σ|≥2.5 l'1.2%.",
    )
    st.caption(" · ".join(f"**{k}**: {v}" for k, v in conteggi.items()))

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    # I vocabolari sono COSTANTI, non derivati dai dati del giorno: con una
    # chiave, Streamlit ripristina la selezione salvata e solleverebbe un errore
    # se un'opzione salvata ieri non fosse fra le opzioni di oggi.
    # Nessun `default=`: il valore iniziale arriva da session_state (vedi
    # ripristina_impostazioni). Passare entrambi farebbe emettere a Streamlit
    # un warning di conflitto fra default del widget e Session State API.
    with c1:
        sel_setup = st.multiselect(
            "Stato", SETUP_VALIDI, key="f_setup",
            help="Condizione osservata. La direzione operativa non è implicita nel nome: "
                 "arriva dalla validazione ed è nella colonna Azione.")
    with c2:
        sel_tipo = st.multiselect("Strumento", TIPI_VALIDI, key="f_tipo")
    with c3:
        sel_cat = st.multiselect("Categoria", CATEGORIE_VALIDE, key="f_cat")
    with c4:
        sel_vol = st.multiselect("Volatilità", ["COMPRESSA", "RICCA", "—"], key="f_vol")

    # Gli slider sono vincoli AGGIUNTIVI, non sovrascritture: con i valori di
    # default non vincolano, e possono solo stringere oltre il livello scelto.
    # Cosi' non entrano in conflitto con il selettore di selettività e possono
    # avere una chiave che ne fa sopravvivere il valore ai cambi di modalità.
    c5, c6, c7 = st.columns([2, 2, 2])
    with c5:
        min_z = st.slider("Dislocazione minima |σ|", 1.0, 4.0, step=0.1, key="f_min_z",
                          help="Restrizione aggiuntiva sopra il livello di selettività. "
                               "È la grandezza con una coda nota, quindi il filtro sensato: "
                               "sotto normalità |σ|≥2 seleziona il 4.6% dell'universo, |σ|≥2.5 l'1.2%.")
    with c6:
        min_adv = st.slider("ADV minimo (M$)", 0, 500, step=10, key="f_min_adv")
    with c7:
        max_gg = st.slider("Max giorni in coda", 0, 90, step=5, key="f_max_gg",
                           help="Restrizione aggiuntiva. Un titolo fermo in coda da mesi è un "
                                "trend, non un'anomalia.")

    out = risultati.copy()
    if sel_setup:
        out = out[out["setup"].isin(sel_setup)]
    else:
        out = out[out["setup"] != "—"]
    if sel_tipo:
        out = out[out["Tipo"].isin(sel_tipo)]
    if sel_cat:
        out = out[out["Categoria"].isin(sel_cat)]
    if sel_vol:
        out = out[out["vol_flag"].isin(sel_vol)]

    liv = LIVELLI_SELETTIVITA[livello]
    out = _applica_livello(out, {**liv,
                                 "z": max(liv["z"], min_z),
                                 "gg": min(liv["gg"], max_gg)})
    out = out[out["ADV M$"].fillna(0) >= min_adv]

    return out


def _tabella(df: pd.DataFrame) -> None:
    colonne = [
        "Ticker", "Nome", "setup", "Azione", "Evidenza", "vol_flag", "Struttura",
        "Score", "Rend %", "Disloc σ", "Mom residuo", "Rank XS", "Velocity",
        "GG in coda", "Vol %", "Vol pctl", "Beta", "Benchmark",
        "Pctl storico", "DD 52w %", "ADV M$",
    ]
    vis = df[colonne].rename(columns={"setup": "Stato", "vol_flag": "Vol"})

    st.dataframe(
        vis,
        width="stretch",
        hide_index=True,
        height=min(700, 40 + 35 * max(len(vis), 1)),
        column_config={
            "Stato": st.column_config.TextColumn(
                help="Descrive la CONDIZIONE osservata, non la strategia: ↑↑/↓↓ = "
                     "dislocato e ancora in movimento, ↑/↓ = dislocato ma il momentum "
                     "ha girato."),
            "Azione": st.column_config.TextColumn(
                help="Direzione operativa che esce dalla validazione walk-forward. "
                     "NESSUNA significa che su quello stato non è stato misurato alcun "
                     "extra rispetto a una selezione casuale."),
            "Evidenza": st.column_config.TextColumn(
                help="confermata = effetto misurato, monotono e stabile out-of-sample · "
                     "debole = stessa direzione ma non significativo · "
                     "assente = nessun extra · instabile = il segno cambia con l'orizzonte."),
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f",
                help="Euristica di ordinamento DENTRO uno stato, NON una probabilità. "
                     "Pesi: 40% ampiezza dislocazione, 20% freschezza, 20% intensità del "
                     "momentum, 10% percentile di volatilità, 10% liquidità.",
            ),
            "Rend %": st.column_config.NumberColumn(format="%.1f%%"),
            "Disloc σ": st.column_config.NumberColumn(
                format="%.2f",
                help="AMPIEZZA della dislocazione: rendimento in eccesso rispetto al benchmark, "
                     "diviso per la volatilità idiosincratica attesa sull'orizzonte.",
            ),
            "Mom residuo": st.column_config.NumberColumn(
                format="%+.2f",
                help="DIREZIONE attuale: momentum residuo delle ultime 10 sedute, in σ. "
                     "Distingue 'ha smesso di scendere' da 'sta ancora scendendo'. "
                     "È questa, non la Velocity, a decidere la classificazione: la Velocity "
                     "di rank è limitata in [0,100] e satura quando il titolo è già ultimo.",
            ),
            "Rank XS": st.column_config.NumberColumn(
                format="%.0f", help="Percentile cross-sectional: posizione contro tutto l'universo OGGI."),
            "Velocity": st.column_config.NumberColumn(
                format="%+.1f",
                help="Variazione del rank cross-sectional in 10 sedute. Solo informativa: "
                     "satura agli estremi, quindi non viene usata per classificare."),
            "GG in coda": st.column_config.NumberColumn(
                format="%.0f", help="Sedute consecutive nel decile estremo. Massimo rilevabile: 90."),
            "Vol %": st.column_config.NumberColumn(format="%.0f%%"),
            "Vol pctl": st.column_config.NumberColumn(
                format="%.0f", help="Percentile della volatilità realizzata a 20gg vs la propria storia."),
            "Beta": st.column_config.NumberColumn(format="%.2f"),
            "Pctl storico": st.column_config.NumberColumn(
                format="%.0f",
                help="Percentile stagionale sulla storia propria del titolo. Con poche annualità "
                     "è poco risolutivo: è contesto, non criterio di selezione."),
            "DD 52w %": st.column_config.NumberColumn(format="%.1f%%"),
            "ADV M$": st.column_config.NumberColumn(format="%.0f"),
        },
    )


def _dettaglio_candidato(df: pd.DataFrame) -> None:
    """Decomposizione dello score + passaggio all'analisi single-asset."""
    if df.empty:
        return

    st.markdown("#### 🔬 Approfondisci un candidato")
    c1, c2 = st.columns([3, 1])

    with c1:
        opzioni = [
            f"{r.Ticker} — {r.setup} · score {r.Score:.0f} · {r['Disloc σ']:+.2f}σ"
            for _, r in df.head(60).iterrows()
        ]
        # Le opzioni cambiano a ogni scansione e a ogni cambio di filtro: un
        # valore salvato che non esiste piu' fra le opzioni farebbe sollevare
        # Streamlit. Si scarta prima di creare il widget, cosi' la selezione
        # sopravvive finche' il candidato e' ancora in lista e si azzera quando
        # non lo e' piu'.
        if "f_candidato" in st.session_state and st.session_state["f_candidato"] not in opzioni:
            del st.session_state["f_candidato"]
        scelta = st.selectbox("Candidato", opzioni, key="f_candidato",
                              label_visibility="collapsed")
        ticker_sel = scelta.split(" — ")[0]

    with c2:
        if st.button("📈 Apri analisi completa", type="primary", width="stretch"):
            riga = df[df["Ticker"] == ticker_sel].iloc[0]
            # Si deposita solo una RICHIESTA di navigazione: scrivere qui
            # direttamente in session_state["modalita"] solleverebbe
            # StreamlitAPIException, perche' il radio con quella chiave e' gia'
            # stato istanziato in questo run. La richiesta viene consumata da
            # app.main() all'inizio del run successivo, prima dei widget.
            st.session_state[NAV_TICKER] = riga["ticker_eodhd"]
            st.rerun()

    riga = df[df["Ticker"] == ticker_sel].iloc[0]
    verdetto = C.SETUP_VERDETTO.get(riga["setup"], {})

    st.markdown(f"**{riga['Ticker']} — {riga['Nome']}**")

    if riga["Azione"] == "NESSUNA":
        st.warning(
            f"**{riga['setup']}** — {verdetto.get('stato', '')}. "
            f"Evidenza **{riga['Evidenza']}**: {verdetto.get('nota', '')}"
        )
    else:
        st.info(
            f"**{riga['setup']}** — {verdetto.get('stato', '')}. "
            f"Azione **{riga['Azione']}**, evidenza **{riga['Evidenza']}**, "
            f"detenzione indicata **{verdetto.get('holding', '?')} sedute** "
            f"(≈ 30-45 DTE). {verdetto.get('nota', '')}"
        )

    comp = pd.DataFrame({
        "Componente": ["Ampiezza dislocazione", "Freschezza", "Conferma direzionale",
                       "Percentile volatilità", "Liquidità"],
        "Peso": [0.40, 0.20, 0.20, 0.10, 0.10],
        "Valore (0-1)": [riga["_s_disloc"], riga["_s_fresh"], riga["_s_stab"],
                         riga["_s_vol"], riga["_s_liq"]],
    })
    comp["Contributo"] = (comp["Peso"] * comp["Valore (0-1)"] * 100).round(1)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(comp, hide_index=True, width="stretch",
                     column_config={
                         "Peso": st.column_config.NumberColumn(format="%.2f"),
                         "Valore (0-1)": st.column_config.NumberColumn(format="%.2f"),
                         "Contributo": st.column_config.NumberColumn(format="%.1f"),
                     })
    with c2:
        st.markdown(f"""
        - **Stato:** `{riga['setup']}` · volatilità **{riga['vol_flag']}**
        - **Dislocazione:** {riga['Disloc σ']:+.2f}σ ({riga['Tipo disloc']} `{riga['Benchmark']}`,
          beta {riga['Beta']:.2f}, R² {riga['R²']:.2f})
        - **Momentum residuo 10 sedute:** {riga['Mom residuo']:+.2f}σ
          → {"si sta chiudendo" if (riga['Mom residuo'] > 0) == (riga['Disloc σ'] < 0) else "si sta allargando"}
        - **Rank cross-sectional:** {riga['Rank XS']:.0f}° · in coda da {riga['GG in coda']:.0f} sedute
        - **Volatilità:** {riga['Vol %']:.0f}% annua, {riga['Vol pctl']:.0f}° percentile storico
        - **Struttura:** {riga['Struttura']}
        """)
        st.caption(
            "La struttura è una mappatura meccanica azione validata × regime di volatilità, "
            "non un consiglio. La direzione NON segue il movimento osservato: uno strumento "
            "esteso al rialzo si tratta al ribasso, perché è così che ha misurato l'event study."
        )


def _esclusi(ctx: dict) -> None:
    qc = ctx.get("qc")
    if qc is None or qc.empty:
        return
    scartati = qc[~qc["eleggibile"]]
    if scartati.empty:
        return

    with st.expander(f"🧹 Strumenti scartati dai controlli qualità ({len(scartati)})"):
        st.markdown("""
        Filtri attivi di default. Su un altro progetto era emerso che le serie EODHD
        corrotte (concambi non gestiti, fusioni, split mancati) generano crolli spuri:
        su uno screener che **ordina per estremità della dislocazione** quelle serie non
        finiscono da qualche parte nella lista, finiscono **in cima**. Il bias dei dati
        sporchi favorisce sistematicamente la tesi contrarian, quindi si tagliano prima
        di qualunque calcolo.

        Il prezzo da pagare è che anche un crollo *legittimo* oltre il 35% in una seduta
        viene escluso. È una scelta conservativa consapevole.
        """)
        vis = scartati[["n_obs", "staleness_days", "max_abs_ret_252",
                        "zero_vol_days_20", "motivo_esclusione"]].copy()
        vis.index = [t.replace(".US", "") for t in vis.index]
        vis.columns = ["Osservazioni", "Giorni fermo", "Max |ret| giorno",
                       "GG volume 0", "Motivo"]
        st.dataframe(vis.sort_values("Motivo"), width="stretch")


# =============================================================================
# ENTRY POINT DELLA TAB
# =============================================================================
def render(api_key: str) -> None:
    st.markdown("## 🔍 Screener Multi-Asset")

    st.markdown("""
    <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 18px;">
    <b>Stato osservato ≠ direzione da prendere</b><br>
    Lo screener classifica gli strumenti per <b>stato</b> — quanto si sono staccati dal proprio
    benchmark e in che verso si stanno muovendo ora. La <b>direzione operativa</b> non è implicita
    nello stato: viene dalla validazione walk-forward, ed è nella colonna <code>Azione</code>.
    <ul>
    <li><b>Dislocazione σ</b> — rendimento in eccesso sul benchmark, normalizzato per volatilità
    idiosincratica. Rende confrontabili un ETF obbligazionario e un semiconduttore.</li>
    <li><b>Mom residuo</b> — verso in cui si sta muovendo <i>adesso</i>. Separa
    &ldquo;ha smesso&rdquo; da &ldquo;sta ancora andando&rdquo;.</li>
    <li><b>Rank XS</b> — percentile contro tutto l'universo <i>oggi</i> (~600 campioni), non contro
    la propria storia (poche annualità).</li>
    </ul>
    <b>⚠️ Il punto che conta:</b> l'event study dice che gli strumenti <b>estesi al rialzo si
    trattano al RIBASSO</b> — sottoperformano l'universo a tutti gli orizzonti testati. E che sui
    dislocati al ribasso non c'è nulla da prendere. La colonna <code>Struttura</code> segue questo
    verdetto, non l'intuizione di seguire il movimento.
    </div>
    """, unsafe_allow_html=True)

    # --- Parametri ----------------------------------------------------------
    with st.expander("⚙️ Parametri universo e scansione", expanded=False):
        # Tutti con chiave: altrimenti tornando dall'analisi single-asset i
        # parametri si azzerano e lo screener riscarica l'universo con impostazioni
        # diverse da quelle scelte.
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            anno_inizio = st.slider("Storia dal", 2005, 2022, step=1, key="p_anno",
                                    help="Il costo di download è identico: una chiamata EODHD dal 2015 "
                                         "o dal 2020 è la stessa chiamata. Più storia significa stime di "
                                         "volatilità e percentili storici migliori, non più tempo di attesa.")
        with c2:
            n_stocks = st.slider("N. azioni in universo", 100, 1000, step=50, key="p_n_stocks",
                                 help="Ordinate per controvalore scambiato decrescente. "
                                      "Il dollar volume è il proxy di liquidità delle opzioni.")
        with c3:
            min_adv = st.number_input("ADV minimo (M$)", key="p_min_adv",
                                      step=5.0, min_value=0.0,
                                      help="Non applicato agli ETF: molti settoriali scambiano poco "
                                           "ma hanno catene opzioni liquide.")
        with c4:
            orizzonte = st.selectbox("Orizzonte", list(C.HORIZONS.keys()), key="p_orizzonte")

        applica_qc = st.checkbox(
            "Applica i controlli qualità dato (consigliato)", key="p_qc",
            help="Esclude serie ferme, con storia insufficiente, con salti giornalieri "
                 "oltre il 35% o senza scambi.")

    start_date = f"{anno_inizio}-01-01"

    # --- Caricamento --------------------------------------------------------
    with st.spinner("Costruzione universo e caricamento pannello prezzi…"):
        uni, close, volume = _load_everything(start_date, n_stocks, C.UNIVERSE_MIN_PRICE, api_key)

    if close.empty:
        st.error(
            "Non è stato possibile caricare il pannello prezzi. Verifica la chiave EODHD "
            "e la connettività, poi premi **Ricarica dati** nella barra laterale."
        )
        return

    # --- Screening ----------------------------------------------------------
    risultati, ctx = S.run_screen(
        close, volume, uni,
        horizon_label=orizzonte,
        min_adv_usd=min_adv * 1e6,
        min_price=C.UNIVERSE_MIN_PRICE,
        apply_qc=applica_qc,
    )

    if risultati.empty:
        st.warning(ctx.get("errore", "Nessun risultato: allarga l'universo o allenta i filtri."))
        return

    st.caption(
        f"Dati al **{ctx['asof']:%d/%m/%Y}** · trading day **{ctx['tdi']}** dell'anno · "
        f"orizzonte **{ctx['orizzonte']}** · **{ctx['n_eleggibili']}** strumenti analizzati"
    )

    _pannello_contesto(ctx, risultati)
    st.markdown("---")

    # --- Mappa --------------------------------------------------------------
    st.markdown("#### 🗺️ Mappa dei candidati")
    candidati = risultati[risultati["setup"] != "—"]
    st.plotly_chart(charts.build_screener_map(candidati), width="stretch")

    conteggi = ctx["conteggio_setup"]
    cols = st.columns(5)
    for i, s in enumerate(C.SETUP_ORDINE):
        v = C.SETUP_VERDETTO[s]
        cols[i].metric(s, conteggi.get(s, 0),
                       delta=v["azione"] if v["azione"] != "NESSUNA" else "—",
                       delta_color="off",
                       help=f"{v['stato']} · evidenza {v['evidenza']}")
    cols[4].metric("Nessuno stato", conteggi.get("—", 0))

    st.markdown("---")

    # --- Tabella ------------------------------------------------------------
    st.markdown("#### 📋 Candidati")
    filtrati = _filtri(risultati)
    # Tutti i widget dello screener esistono ora: se ne salva lo stato prima che
    # il pulsante di drill-down possa far cambiare pagina.
    salva_impostazioni()

    if filtrati.empty:
        st.info("Nessun candidato con i filtri correnti.")
    else:
        st.caption(f"**{len(filtrati)}** candidati · ordinati per score decrescente")
        _tabella(filtrati)

        st.download_button(
            "⬇️ Esporta CSV",
            data=filtrati.drop(columns=[c for c in filtrati.columns if c.startswith("_")])
                         .to_csv(index=False).encode("utf-8"),
            file_name=f"screener_kq_{ctx['asof']:%Y%m%d}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        _dettaglio_candidato(filtrati)

    # --- Contesto aggiuntivo ------------------------------------------------
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 📊 Distribuzione della dislocazione")
        st.plotly_chart(charts.build_breadth_chart(risultati), width="stretch")
    with c2:
        st.markdown("#### 🧭 Dislocazione per categoria")
        st.plotly_chart(charts.build_category_chart(risultati), width="stretch")

    _esclusi(ctx)

    st.markdown("---")
    st.warning(
        "**Limiti da tenere presenti.** (1) L'universo è costruito sui membri di oggi: "
        "è affetto da *survivorship bias*, accettato consapevolmente perché la metrica primaria "
        "è cross-sectional e quindi non lo eredita. (2) Lo `Score` è un'euristica di ordinamento "
        "trasparente, **non** un edge validato: nessun backtest, nessun confronto con un null, "
        "nessun costo di transazione. La validazione si fa a valle, sul singolo candidato."
    )

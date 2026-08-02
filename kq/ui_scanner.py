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
from kq import universe as U


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


def _filtri(risultati: pd.DataFrame) -> pd.DataFrame:
    """Barra filtri sopra la tabella."""
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    with c1:
        setups = [s for s in ["MR-LONG", "MR-SHORT", "TREND-UP", "TREND-DN"]
                  if s in risultati["setup"].unique()]
        sel_setup = st.multiselect("Setup", setups, default=setups,
                                   help="MR = mean reversion su dislocazione idiosincratica. "
                                        "TREND = continuazione con momentum confermato.")
    with c2:
        tipi = sorted(risultati["Tipo"].dropna().unique().tolist())
        sel_tipo = st.multiselect("Strumento", tipi, default=tipi)
    with c3:
        cats = sorted(risultati["Categoria"].dropna().unique().tolist())
        sel_cat = st.multiselect("Categoria", cats, default=[])
    with c4:
        sel_vol = st.multiselect("Volatilità", ["COMPRESSA", "RICCA", "—"], default=[])

    c5, c6, c7 = st.columns([2, 2, 2])
    with c5:
        min_score = st.slider("Score minimo", 0, 100, 0, step=5)
    with c6:
        min_adv = st.slider("ADV minimo (M$)", 0, 500, 0, step=10)
    with c7:
        max_gg = st.slider("Max giorni in coda", 0, 90, 90, step=5,
                           help="Un titolo fermo in coda da mesi è un trend, non un'anomalia. "
                                "Abbassa questo valore per tenere solo le dislocazioni fresche.")

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
    out = out[out["Score"].fillna(0) >= min_score]
    out = out[out["ADV M$"].fillna(0) >= min_adv]
    out = out[out["GG in coda"].fillna(0) <= max_gg]

    return out


def _tabella(df: pd.DataFrame) -> None:
    colonne = [
        "Ticker", "Nome", "setup", "vol_flag", "Score", "Rend %", "Disloc σ",
        "Mom residuo", "Rank XS", "Velocity", "GG in coda", "Vol %", "Vol pctl",
        "Beta", "Benchmark", "Struttura", "Pctl storico", "DD 52w %", "ADV M$",
    ]
    vis = df[colonne].rename(columns={"setup": "Setup", "vol_flag": "Vol"})

    st.dataframe(
        vis,
        use_container_width=True,
        hide_index=True,
        height=min(700, 40 + 35 * max(len(vis), 1)),
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f",
                help="Euristica di ordinamento, NON una probabilità e non un backtest. "
                     "Pesi: 40% ampiezza dislocazione, 20% freschezza, 20% stabilizzazione, "
                     "10% percentile di volatilità, 10% liquidità.",
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
        scelta = st.selectbox("Candidato", opzioni, index=0, label_visibility="collapsed")
        ticker_sel = scelta.split(" — ")[0]

    with c2:
        if st.button("📈 Apri analisi completa", type="primary", use_container_width=True):
            riga = df[df["Ticker"] == ticker_sel].iloc[0]
            st.session_state["ticker_input"] = riga["ticker_eodhd"]
            st.session_state["modalita"] = "📈 Analisi singolo asset"
            st.rerun()

    riga = df[df["Ticker"] == ticker_sel].iloc[0]

    st.markdown(f"**{riga['Ticker']} — {riga['Nome']}**")

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
        st.dataframe(comp, hide_index=True, use_container_width=True,
                     column_config={
                         "Peso": st.column_config.NumberColumn(format="%.2f"),
                         "Valore (0-1)": st.column_config.NumberColumn(format="%.2f"),
                         "Contributo": st.column_config.NumberColumn(format="%.1f"),
                     })
    with c2:
        st.markdown(f"""
        - **Setup:** `{riga['setup']}` · volatilità **{riga['vol_flag']}**
        - **Dislocazione:** {riga['Disloc σ']:+.2f}σ ({riga['Tipo disloc']} `{riga['Benchmark']}`,
          beta {riga['Beta']:.2f}, R² {riga['R²']:.2f})
        - **Momentum residuo 10 sedute:** {riga['Mom residuo']:+.2f}σ
          → {"si sta chiudendo" if (riga['Mom residuo'] > 0) == (riga['Disloc σ'] < 0) else "si sta allargando"}
        - **Rank cross-sectional:** {riga['Rank XS']:.0f}° · in coda da {riga['GG in coda']:.0f} sedute
        - **Volatilità:** {riga['Vol %']:.0f}% annua, {riga['Vol pctl']:.0f}° percentile storico
        - **Struttura naturale:** {riga['Struttura']}
        """)
        st.caption(
            "La struttura opzioni è una mappatura meccanica setup × regime di volatilità, "
            "non un consiglio: serve a ricordare quale payoff è coerente con la tesi."
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
        st.dataframe(vis.sort_values("Motivo"), use_container_width=True)


# =============================================================================
# ENTRY POINT DELLA TAB
# =============================================================================
def render(api_key: str) -> None:
    st.markdown("## 🔍 Screener Multi-Asset")

    st.markdown("""
    <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 18px;">
    <b>Cosa fa e cosa non fa</b><br>
    Questo screener produce una <b>lista di candidati da vagliare</b>, non uno studio.
    Cerca strumenti la cui performance si è staccata da quella del proprio benchmark
    più di quanto la loro volatilità giustifichi, e distingue chi si sta
    <i>stabilizzando</i> da chi sta <i>ancora scendendo</i>.
    <ul>
    <li><b>Dislocazione σ</b> — rendimento in eccesso sul benchmark, normalizzato per volatilità
    idiosincratica. È la metrica principale: rende confrontabili un ETF obbligazionario e un semiconduttore.</li>
    <li><b>Mom residuo</b> — direzione <i>attuale</i>: il titolo ha smesso di muoversi contro,
    o sta ancora andando giù? È questo a separare un rimbalzo da un coltello che cade.</li>
    <li><b>Rank XS</b> — percentile contro tutto l'universo <i>oggi</i> (~600 campioni), non contro
    la propria storia (poche annualità). Robusto e immune al survivorship bias dell'universo.</li>
    </ul>
    La verifica vera si fa a valle, aprendo il singolo candidato nell'analisi completa.
    </div>
    """, unsafe_allow_html=True)

    # --- Parametri ----------------------------------------------------------
    with st.expander("⚙️ Parametri universo e scansione", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            anno_inizio = st.slider("Storia dal", 2005, 2022, 2015, step=1,
                                    help="Il costo di download è identico: una chiamata EODHD dal 2015 "
                                         "o dal 2020 è la stessa chiamata. Più storia significa stime di "
                                         "volatilità e percentili storici migliori, non più tempo di attesa.")
        with c2:
            n_stocks = st.slider("N. azioni in universo", 100, 1000,
                                 C.UNIVERSE_N_STOCKS, step=50,
                                 help="Ordinate per controvalore scambiato decrescente. "
                                      "Il dollar volume è il proxy di liquidità delle opzioni.")
        with c3:
            min_adv = st.number_input("ADV minimo (M$)",
                                      value=C.UNIVERSE_MIN_ADV_USD / 1e6, step=5.0, min_value=0.0,
                                      help="Non applicato agli ETF: molti settoriali scambiano poco "
                                           "ma hanno catene opzioni liquide.")
        with c4:
            orizzonte = st.selectbox("Orizzonte", list(C.HORIZONS.keys()), index=0)

        applica_qc = st.checkbox(
            "Applica i controlli qualità dato (consigliato)", value=True,
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
    st.plotly_chart(charts.build_screener_map(candidati), use_container_width=True)

    conteggi = ctx["conteggio_setup"]
    cols = st.columns(5)
    for i, s in enumerate(["MR-LONG", "MR-SHORT", "TREND-UP", "TREND-DN"]):
        cols[i].metric(s, conteggi.get(s, 0))
    cols[4].metric("Nessun setup", conteggi.get("—", 0))

    st.markdown("---")

    # --- Tabella ------------------------------------------------------------
    st.markdown("#### 📋 Candidati")
    filtrati = _filtri(risultati)

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
        st.plotly_chart(charts.build_breadth_chart(risultati), use_container_width=True)
    with c2:
        st.markdown("#### 🧭 Dislocazione per categoria")
        st.plotly_chart(charts.build_category_chart(risultati), use_container_width=True)

    _esclusi(ctx)

    st.markdown("---")
    st.warning(
        "**Limiti da tenere presenti.** (1) L'universo è costruito sui membri di oggi: "
        "è affetto da *survivorship bias*, accettato consapevolmente perché la metrica primaria "
        "è cross-sectional e quindi non lo eredita. (2) Lo `Score` è un'euristica di ordinamento "
        "trasparente, **non** un edge validato: nessun backtest, nessun confronto con un null, "
        "nessun costo di transazione. La validazione si fa a valle, sul singolo candidato."
    )

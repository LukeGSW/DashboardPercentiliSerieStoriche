"""
=============================================================================
kq.ui_validation — Tab "Validazione setup"
=============================================================================
Misura storicamente i setup dello screener contro un null, walk-forward e al
netto dei costi. Riusa il pannello prezzi gia' in memoria: nessun download
aggiuntivo.
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kq import config as C
from kq import state
from kq import ui_scanner
from kq import validation as V

_SPECCHIO = "_val_"
DEFAULT_IMPOSTAZIONI = {
    "v_orizzonte": next(iter(C.HORIZONS)),
    "v_holdings": [5, 10, 20, 60],
    "v_rebalance": 5,
    "v_costo": 10.0,
    "v_boot": 1000,
    "v_placebo": 20,
    "v_livelli": ["Tutti i candidati", "Selettivo"],
}

HOLDING_DISPONIBILI = [3, 5, 10, 20, 40, 60, 120]


def ripristina_impostazioni(pagina_attiva: bool) -> None:
    state.ripristina(DEFAULT_IMPOSTAZIONI, _SPECCHIO, pagina_attiva)


def salva_impostazioni() -> None:
    state.salva(DEFAULT_IMPOSTAZIONI, _SPECCHIO)


# =============================================================================
def _grafico_extra(serie: pd.Series, titolo: str) -> go.Figure:
    """Extra-rendimento cumulato: la forma conta piu' del livello finale."""
    fig = go.Figure()
    s = serie.dropna()
    if len(s) == 0:
        return fig

    cum = s.cumsum() * 100
    taglio = int(len(cum) * 2 / 3)

    fig.add_trace(go.Scatter(
        x=cum.index[:taglio + 1], y=cum.iloc[:taglio + 1],
        mode="lines", line=dict(color=C.COLORS["persistence"], width=2),
        name="In-sample (2/3)"))
    fig.add_trace(go.Scatter(
        x=cum.index[taglio:], y=cum.iloc[taglio:],
        mode="lines", line=dict(color=C.COLORS["zscore_pos"], width=2),
        name="Out-of-sample (1/3)"))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C.COLORS["background"], plot_bgcolor=C.COLORS["background"],
        height=340, title=titolo,
        xaxis=dict(title="", gridcolor=C.COLORS["grid"]),
        yaxis=dict(title="Extra cumulato (%)", gridcolor=C.COLORS["grid"], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def _grafico_profilo(mis: pd.DataFrame) -> go.Figure:
    """
    Extra ANNUALIZZATO in funzione del periodo di detenzione.

    È il grafico che serve per scegliere l'holding, e va letto annualizzato,
    non per trade: l'extra per trade cresce quasi meccanicamente con l'holding
    (più tempo, più rendimento accumulato), quindi confrontando per trade si
    finirebbe sempre per prendere il più lungo. L'annualizzato incorpora invece
    anche il costo, che si paga a ogni trade e quindi penalizza gli holding
    corti: il massimo netto sta tipicamente a metà strada.

    Un effetto vero disegna un profilo LISCIO che sale, picca e decade. Se il
    segno sbatte da un holding all'altro, non c'è niente da temporizzare.
    """
    fig = go.Figure()
    if mis.empty:
        return fig

    for (setup, livello), grp in mis.groupby(["Setup", "Livello"]):
        g = grp.sort_values("Holding")
        tratto = {"Alta convinzione": "solid", "Selettivo": "dash",
                  "Tutti i candidati": "dot"}.get(livello, "solid")
        fig.add_trace(go.Scatter(
            x=g["Holding"], y=g["Extra annuo %"], mode="lines+markers",
            name=f"{setup} · {livello}",
            line=dict(color=C.SETUP_COLORS.get(setup, C.COLORS["neutral"]),
                      width=2, dash=tratto),
            marker=dict(size=7),
            hovertemplate=("<b>%{fullData.name}</b><br>holding %{x} sedute"
                           "<br>extra annuo %{y:.2f}%<extra></extra>"),
        ))

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.4)")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C.COLORS["background"], plot_bgcolor=C.COLORS["background"],
        height=440,
        xaxis=dict(title="Periodo di detenzione (sedute)", type="log",
                   gridcolor=C.COLORS["grid"],
                   tickmode="array", tickvals=sorted(mis["Holding"].unique()),
                   ticktext=[str(h) for h in sorted(mis["Holding"].unique())]),
        yaxis=dict(title="Extra annualizzato, netto costi (%)",
                   gridcolor=C.COLORS["grid"], ticksuffix="%"),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
                    font=dict(size=10)),
        margin=dict(l=70, r=30, t=30, b=60),
    )
    return fig


def _tabella(df: pd.DataFrame) -> None:
    cols = ["Setup", "Livello", "Holding", "Esito", "Extra %", "Extra lordo %",
            "Placebo p05 %", "Placebo p95 %", "t (NW)", "p (bootstrap)", "q (BH)",
            "Hit %", "IS %", "OOS %", "n_date", "n_trade", "Nomi/data", "Copertura %"]
    st.dataframe(
        df[cols], width="stretch", hide_index=True,
        height=min(700, 40 + 35 * max(len(df), 1)),
        column_config={
            "Extra %": st.column_config.NumberColumn(
                "Extra netto %", format="%+.3f",
                help="Extra-rendimento medio per trade, orientato secondo la tesi del setup "
                     "e al netto dei costi. È il risultato economico."),
            "Extra lordo %": st.column_config.NumberColumn(
                format="%+.3f",
                help="Al lordo dei costi. È su questo che gira l'inferenza: il costo è uno "
                     "spostamento deterministico e sottrarlo prima del test gonfierebbe il |t|."),
            "Placebo p05 %": st.column_config.NumberColumn(
                format="%+.3f",
                help="Estremo inferiore della banda ottenuta selezionando nomi A CASO, "
                     "stessa numerosità e stesse date. Un extra dentro la banda è rumore."),
            "Placebo p95 %": st.column_config.NumberColumn(format="%+.3f"),
            "t (NW)": st.column_config.NumberColumn(
                format="%+.2f",
                help="t di Student con errore standard Newey-West: corregge la sovrapposizione "
                     "fra periodi di detenzione, che gonfierebbe il t ingenuo."),
            "p (bootstrap)": st.column_config.NumberColumn(format="%.3f"),
            "q (BH)": st.column_config.NumberColumn(
                format="%.3f",
                help="p-value corretto per test multipli (Benjamini-Hochberg). Con decine di "
                     "celle testate, la migliore sembra buona per costruzione."),
            "Hit %": st.column_config.NumberColumn(format="%.0f"),
            "IS %": st.column_config.NumberColumn(format="%+.3f"),
            "OOS %": st.column_config.NumberColumn(
                format="%+.3f", help="Ultimo terzo delle date. Misura persistenza, non "
                                     "protezione da overfitting: non c'è nulla di stimato sui dati."),
            "Nomi/data": st.column_config.NumberColumn(format="%.1f"),
            "Copertura %": st.column_config.NumberColumn(
                format="%.1f",
                help="Quota di segnalati con un forward calcolabile. Se è ~100%% significa che "
                     "nel campione non fallisce mai nessuno: è il survivorship bias, non fortuna."),
        },
    )


# =============================================================================
def render(api_key: str) -> None:
    st.markdown("## 🧪 Validazione dei setup")

    st.markdown("""
    <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 18px;">
    <b>La domanda</b><br>
    Quando lo screener ha segnalato un nome, nelle sedute successive quel nome ha fatto
    meglio di una selezione <i>casuale fatta lo stesso giorno sullo stesso universo</i>?
    <br><br>
    <b>Perché questo null e non l'entrata casuale nel tempo.</b> I setup di mean reversion
    scattano in modo sproporzionato durante i drawdown: confrontarli con entrate distribuite
    su tutto il periodo significherebbe accreditare al segnale il recupero del mercato.
    La sezione trasversale contemporanea toglie beta ed effetto periodo in un colpo solo.
    <br><br>
    <b>Cosa è garantito:</b> ogni grandezza usa finestre mobili o espandenti (alla data t
    solo dati fino a t), l'esecuzione è a t+1, l'inferenza usa Newey-White per la
    sovrapposizione dei periodi, e i test multipli sono corretti con Benjamini-Hochberg.
    Le soglie dello screener sono costanti fissate a priori: <b>non c'è nulla di stimato
    sui dati</b>, quindi la divisione in-sample/out-of-sample misura la persistenza nel
    tempo, non protegge da overfitting.
    </div>
    """, unsafe_allow_html=True)

    # --- Parametri ----------------------------------------------------------
    with st.expander("⚙️ Parametri dello studio", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            orizzonte = st.selectbox("Orizzonte dello screener", list(C.HORIZONS.keys()),
                                     key="v_orizzonte")
            livelli_sel = st.multiselect(
                "Livelli di selettività", list(ui_scanner.LIVELLI_SELETTIVITA.keys()),
                key="v_livelli",
                help="Testarli tutti risponde alla domanda: la selettività in più si ripaga?")
        with c2:
            holdings = st.multiselect("Periodi di detenzione (sedute)", HOLDING_DISPONIBILI,
                                      key="v_holdings")
            rebalance = st.select_slider("Ogni quante sedute si rileva il segnale",
                                         options=[1, 2, 5, 10, 21], key="v_rebalance")
        with c3:
            costo = st.number_input("Costo andata+ritorno (bps)", min_value=0.0, max_value=200.0,
                                    step=1.0, key="v_costo",
                                    help="10 bps è realistico per large cap USA liquide. "
                                         "Cambia il risultato economico, non l'inferenza.")
            n_boot = st.select_slider("Ricampionamenti bootstrap",
                                      options=[200, 500, 1000, 2000], key="v_boot")
            n_placebo = st.select_slider("Ripetizioni placebo", options=[5, 10, 20, 50],
                                         key="v_placebo",
                                         help="Selezioni casuali usate per stimare la banda di rumore.")

    salva_impostazioni()

    if not holdings or not livelli_sel:
        st.info("Seleziona almeno un periodo di detenzione e un livello di selettività.")
        return

    n_celle = 4 * len(livelli_sel) * len(holdings)
    st.caption(f"Celle da testare: **{n_celle}** (4 setup × {len(livelli_sel)} livelli "
               f"× {len(holdings)} orizzonti di detenzione)")

    anno = st.session_state.get("p_anno", 2015)
    n_stocks = st.session_state.get("p_n_stocks", C.UNIVERSE_N_STOCKS)
    # Identifica in modo univoco i parametri di questa esecuzione: serve a
    # riconoscere quando i risultati mostrati non corrispondono piu' ai comandi.
    firma = (orizzonte, tuple(sorted(holdings)), tuple(sorted(livelli_sel)),
             int(rebalance), float(costo), int(n_boot), int(n_placebo), anno, n_stocks)

    if st.button("🚀 Esegui la validazione", type="primary"):
        with st.spinner("Caricamento pannello prezzi…"):
            uni, close, volume = ui_scanner._load_everything(
                f"{anno}-01-01", n_stocks, C.UNIVERSE_MIN_PRICE, api_key)

        if close.empty:
            st.error("Pannello prezzi non disponibile. Apri prima lo screener.")
            return

        with st.spinner("Calcolo dei segnali storici (una passata su tutte le finestre mobili)…"):
            sig = V.precompute_signals(close, volume, uni, horizon_label=orizzonte,
                                       min_adv_usd=st.session_state.get("p_min_adv", 30.0) * 1e6)
        if not sig:
            st.error("Nessuno strumento utilizzabile per la validazione.")
            return

        livelli = {k: ui_scanner.LIVELLI_SELETTIVITA[k] for k in livelli_sel}
        with st.spinner(f"Event study su {n_celle} celle…"):
            out = V.event_study(sig, close, livelli, holdings=tuple(sorted(holdings)),
                                rebalance=int(rebalance), costo_bps=float(costo),
                                n_boot=int(n_boot), n_placebo=int(n_placebo))

        if out.empty:
            st.warning("Nessun risultato: storia insufficiente per i parametri scelti.")
            return

        # I risultati vivono in session_state, NON dentro il blocco del pulsante:
        # altrimenti qualunque interazione successiva (il menu a tendina del
        # dettaglio, un filtro) farebbe ripartire il run col pulsante a False e
        # la pagina tornerebbe vuota.
        st.session_state["_val_out"] = out
        st.session_state["_val_meta"] = {
            "self_bench": sig["n_esclusi_self_bench"],
            "sporchi": sig["n_esclusi_sporchi"],
            "costo": float(costo),
            "rebalance": int(rebalance),
            "orizzonte": orizzonte,
        }
        st.session_state["_val_firma"] = firma

    out = st.session_state.get("_val_out")
    if out is None:
        st.info("Lo studio è pesante (decine di secondi): parte solo su richiesta.")
        return

    if st.session_state.get("_val_firma") != firma:
        st.warning("⚠️ I parametri sono stati modificati dopo l'ultima esecuzione: "
                   "la tabella qui sotto si riferisce ancora ai parametri precedenti. "
                   "Premi **Esegui la validazione** per aggiornarla.")

    _mostra(out, st.session_state.get("_val_meta", {}))


def _mostra(out: pd.DataFrame, meta: dict) -> None:
    mis = out[out["misurabile"]].copy()

    st.markdown("---")
    st.markdown("### 📋 Esiti")

    # Il costo va reso esplicito accanto ai risultati: e' il parametro che sposta
    # ogni riga di una quantita' fissa, ed e' facile leggere una tabella senza
    # ricordarsi con quale ipotesi e' stata prodotta.
    costo = meta.get("costo")
    if costo is not None:
        nota = "" if costo >= 5 else "  ⚠️ irrealisticamente basso per un andata+ritorno"
        st.caption(
            f"Orizzonte **{meta.get('orizzonte', '?')}** · rilevazione ogni "
            f"**{meta.get('rebalance', '?')}** sedute · costo **{costo:g} bps** "
            f"(sposta ogni Extra netto di **{-costo / 100:+.3f} pp**){nota}"
        )

    # Onestà obbligatoria: la direzione operativa degli stati "estesi" è stata
    # scelta DOPO aver visto questo stesso studio. Un semaforo verde su quelle
    # righe è in-sample per costruzione e non va letto come conferma.
    in_sample = [s for s, v in C.SETUP_VERDETTO.items() if v["azione"] != "NESSUNA"]
    if in_sample:
        st.warning(
            f"**Le righe di {', '.join(f'`{s}`' for s in in_sample)} sono in-sample per "
            f"costruzione.** La loro direzione operativa (`RIBASSISTA`) è stata fissata a "
            f"partire da questo stesso studio: il segno dell'extra è quindi positivo per "
            f"come è stata scelta l'ipotesi, non perché sia stata confermata su dati nuovi. "
            f"Vale come ipotesi da verificare in avanti."
        )

    c = st.columns(5)
    c[0].metric("Celle misurabili", f"{len(mis)}/{len(out)}")
    for i, (etichetta, chiave) in enumerate(
            [("🟢 Regge", "🟢"), ("🟡 Indiziario", "🟡"), ("🟠 Nullo/costi", "🟠"), ("🔴 Contrario", "🔴")]):
        c[i + 1].metric(etichetta, int(mis["Esito"].str.startswith(chiave).sum()) if len(mis) else 0)

    if len(mis) == 0:
        st.warning("Nessuna cella ha abbastanza osservazioni. Allunga la storia o allenta la selettività.")
        return

    if mis["Holding"].nunique() > 1:
        st.markdown("#### 📉 Profilo di decadimento")
        st.caption(
            "Come si sceglie il periodo di detenzione: si legge da qui, non da una cella "
            "singola. Un effetto vero disegna una curva liscia che picca e decade; se il "
            "segno sbatte da un holding all'altro non c'è niente da temporizzare. "
            "L'asse Y è **annualizzato e netto costi**, non per trade: per trade l'extra "
            "cresce quasi meccanicamente con l'holding e sceglieresti sempre il più lungo."
        )
        st.plotly_chart(_grafico_profilo(mis), width="stretch")
        st.markdown("---")

    _tabella(mis.sort_values(["Esito", "Extra %"], ascending=[True, False]))

    st.download_button(
        "⬇️ Esporta CSV",
        data=mis.drop(columns=[c for c in mis.columns if c.startswith("_")])
                .to_csv(index=False).encode("utf-8"),
        file_name="validazione_setup.csv", mime="text/csv")

    # --- Dettaglio cella ----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔬 Dettaglio")
    etichette = [f"{r.Setup} · {r.Livello} · holding {r.Holding}gg  →  {r.Esito}"
                 for _, r in mis.iterrows()]
    # Le etichette cambiano a ogni nuova esecuzione: un valore salvato che non
    # esiste piu' fra le opzioni farebbe sollevare Streamlit.
    if "v_cella" in st.session_state and st.session_state["v_cella"] not in etichette:
        del st.session_state["v_cella"]
    scelta = st.selectbox("Cella", etichette, key="v_cella")
    riga = mis.iloc[etichette.index(scelta)]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(_grafico_extra(riga["_serie"], "Extra-rendimento cumulato (netto)"),
                        width="stretch")
    with c2:
        st.markdown(f"""
        **{riga['Setup']} · {riga['Livello']} · {riga['Holding']} sedute**

        - Extra netto per trade: **{riga['Extra %']:+.3f}%** (lordo {riga['Extra lordo %']:+.3f}%)
        - Banda del caso: da {riga['Placebo p05 %']:+.3f}% a {riga['Placebo p95 %']:+.3f}%
        - t (Newey-West): **{riga['t (NW)']:+.2f}** · p {riga['p (bootstrap)']:.3f} · q {riga['q (BH)']:.3f}
        - In-sample {riga['IS %']:+.3f}% → out-of-sample **{riga['OOS %']:+.3f}%**
        - {int(riga['n_date'])} date · {int(riga['n_trade'])} trade · {riga['Nomi/data']:.1f} nomi per data
        """)
        dentro = riga["Placebo p05 %"] <= riga["Extra lordo %"] <= riga["Placebo p95 %"]
        if dentro:
            st.warning("L'extra **lordo cade dentro la banda del caso**: è indistinguibile "
                       "da una selezione casuale di pari numerosità.")
        else:
            st.success("L'extra lordo è **fuori dalla banda del caso**.")

    # --- Survivorship -------------------------------------------------------
    st.markdown("---")
    cop = mis["Copertura %"].mean()
    st.error(f"""
    **Il limite che non si può togliere: survivorship bias.**

    L'universo è costruito sui membri di oggi. Le società dislocate che sono risalite
    sono qui; quelle andate a zero sono uscite e non compaiono da nessuna parte. Il bias
    spinge quindi **a favore della tesi contrarian**: un risultato positivo sugli stati
    dislocati al ribasso (`↓ STABILIZZATO`, `↓↓ IN CADUTA`)
    va letto come **limite superiore**, non come stima.

    La copertura media dei segnalati è **{cop:.1f}%**: nel campione quasi nessun nome
    smette di quotare durante il periodo di detenzione. Non è un buon segno, è la prova
    che i fallimenti in questi dati **sono invisibili**.

    Per trasformarlo in una stima servirebbero la membership storica point-in-time e i
    prezzi dei delistati.
    """)

    with st.expander("📐 Cosa è stato escluso e perché"):
        st.markdown(f"""
        - **{meta.get('self_bench', '?')}** strumenti che sono benchmark di se stessi
          (SPY, TLT, GLD…): misurano la dislocazione sulla propria storia stagionale, e
          una versione causale richiederebbe una mediana espandente condizionata al
          trading day. Sono fuori dalla validazione, non dallo screener.
        - **{meta.get('sporchi', '?')}** serie con un salto giornaliero oltre il
          {C.QC_MAX_ABS_DAILY_RET:.0%}, escluse per l'**intero campione**: un concambio non
          gestito corrompe la serie in modo retroattivo, e su una misura ordinata per
          estremità quelle serie dominerebbero la classifica.
        - Date con meno di **{V.MIN_NOMI_PER_DATA}** nomi segnalati: la media di uno o due
          titoli è rumore, non un portafoglio.
        - Celle con meno di **{V.MIN_DATE}** date valide o **100** trade: marcate come
          campione insufficiente invece di produrre un numero che sembra una misura.
        """)

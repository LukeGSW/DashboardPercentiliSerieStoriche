"""
=============================================================================
YTD Seasonality & Anomaly Detection Dashboard — Kriterion Quant
=============================================================================
Entry point Streamlit. La logica vive nel pacchetto `kq/`.

Due modalità:
    🔍 Screener multi-asset   — filtro di primo livello su ~600 strumenti
                                liquidi e optionable, per produrre candidati
    📈 Analisi singolo asset  — lo studio completo su un ticker: percentili
                                stagionali, z-score, dinamiche, regime,
                                forward returns condizionali

Il flusso naturale è: screener → scegli un candidato → analisi completa.

Fonte dati: EODHD API (adjusted close + volume giornalieri)
Chiave API: esclusivamente via st.secrets, mai nel codice.
=============================================================================
"""

from __future__ import annotations

import warnings
from datetime import date

import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Anomaly Detection Dashboard | Kriterion Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from kq import data as D          # noqa: E402
from kq import ui_scanner         # noqa: E402
from kq import ui_single          # noqa: E402


def main() -> None:
    api_key = D.get_api_key()

    with st.sidebar:
        st.markdown("## 🔬 Kriterion Quant")
        st.markdown("**Percentile & Anomaly Dashboard**")
        st.markdown("---")

        modalita = st.radio(
            "Modalità",
            ["🔍 Screener multi-asset", "📈 Analisi singolo asset"],
            key="modalita",
        )

        st.markdown("---")

        if modalita == "📈 Analisi singolo asset":
            st.header("⚙️ Parametri")

            ticker = st.text_input(
                "Ticker (formato EODHD)",
                value=st.session_state.get("ticker_input", "SPY.US"),
                key="ticker_input",
                placeholder="es. SPY.US, BTC-USD.CC",
                help="Formato SIMBOLO.EXCHANGE — es. AAPL.US, ENI.MI, GSPC.INDX, BTC-USD.CC",
            ).strip().upper()

            start_date = st.date_input(
                "Inizio storico",
                value=date(2000, 1, 1),
                min_value=date(1990, 1, 1),
                max_value=date.today(),
            )

            st.markdown("---")
            st.subheader("🎛️ Parametri avanzati")

            lookahead_days = st.slider(
                "Forward lookahead (sedute)", 5, 60, 20,
                help="Orizzonte per l'analisi di mean reversion, in trading days.")
            pct_tolerance = st.slider(
                "Tolleranza percentile (%)", 5, 25, 10,
                help="Ampiezza della finestra per considerare simile un anno storico.")
            n_bootstrap = st.select_slider(
                "Campioni bootstrap", options=[100, 250, 500, 1000], value=500)
        else:
            ticker, start_date = None, None
            lookahead_days = pct_tolerance = n_bootstrap = None

        st.markdown("---")
        if st.button("🔄 Ricarica dati", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption("📡 Dati: [EODHD API](https://eodhd.com)")
        st.caption("🔬 Kriterion Quant © 2025")

        st.markdown("---")
        st.markdown("##### ℹ️ Note tecniche")
        st.caption("""
        - **TDI**: Trading Day Index, evita i bias da anni bisestili
        - **Base YTD**: primo prezzo dell'anno (convenzione TradingView)
        - **Max DD**: calcolato geometricamente
        - **Volatilità**: su rendimenti giornalieri veri
        - **Forward**: cross-year con compounding geometrico
        - **Screener**: metrica primaria cross-sectional, non time-series
        """)

    if modalita == "🔍 Screener multi-asset":
        ui_scanner.render(api_key)
    else:
        ui_single.render(api_key, ticker, start_date,
                         lookahead_days, pct_tolerance, n_bootstrap)

    st.markdown("---")
    st.caption("🔬 **Kriterion Quant** — Percentile & Anomaly Dashboard | Dati: EODHD API")
    st.caption("⚠️ Strumento a scopo educativo e di ricerca. Non costituisce consulenza finanziaria.")


if __name__ == "__main__":
    main()

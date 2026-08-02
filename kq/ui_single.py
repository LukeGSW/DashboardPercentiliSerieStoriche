"""
=============================================================================
kq.ui_single — Analisi single-asset (percentili, z-score, dinamiche, regime,
                forward returns)
=============================================================================
E' la parte "seria" della dashboard: qui si usa la storia lunga del singolo
strumento, e qui si portano i candidati che escono dallo screener.
=============================================================================
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from kq import charts
from kq import config as C
from kq import core
from kq import data as D


def _box(testo_html: str) -> None:
    st.markdown(
        f'<div style="background-color: rgba(100,149,237,0.1); padding: 15px; '
        f'border-radius: 10px; margin-bottom: 20px;">{testo_html}</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
def render(api_key: str, ticker: str, start_date: date,
           lookahead_days: int, pct_tolerance: int, n_bootstrap: int) -> None:

    current_year = date.today().year
    st.markdown(f"## 📈 Analisi Anomalie: `{ticker}`")
    st.caption(
        f"Analisi multi-dimensionale delle anomalie YTD **{current_year}** vs distribuzione "
        f"storica (dal **{start_date:%d/%m/%Y}**)."
    )

    with st.spinner(f"Scaricamento dati {ticker}…"):
        df = D.fetch_ohlcv_cached(ticker, start_date.strftime("%Y-%m-%d"), api_key)

    if df.empty:
        st.error(f"Nessun dato disponibile per **{ticker}**. Verifica il simbolo (formato `SIMBOLO.EXCHANGE`).")
        return

    # --- Validazione aggiornamento dati -------------------------------------
    df["year"] = df["date"].dt.year
    anni_nei_dati = sorted(df["year"].unique())
    ultimo_anno = anni_nei_dati[-1]
    ultima_data = df["date"].max()

    if current_year not in anni_nei_dati:
        st.error(
            f"**Dati non aggiornati.** Oggi è il {date.today():%d/%m/%Y} (anno {current_year}), "
            f"ma i dati EODHD arrivano al {ultima_data:%d/%m/%Y} (anno {ultimo_anno}). "
            f"Verifica la validità della chiave API e la copertura del ticker."
        )
        st.warning(f"Proseguo usando **{ultimo_anno}** come anno di riferimento.")
        current_year = ultimo_anno

    with st.expander("📅 Range dati disponibili", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Prima data", f"{df['date'].min():%Y-%m-%d}")
        c2.metric("Ultima data", f"{ultima_data:%Y-%m-%d}")
        c3.metric("Anni totali", len(anni_nei_dati))

    # --- Calcoli ------------------------------------------------------------
    with st.spinner("Elaborazione…"):
        pivot_ytd, pivot_returns, metadata = core.compute_ytd_by_trading_day(df)
        anni_disponibili = sorted(pivot_ytd.columns.tolist())

        if len(anni_disponibili) < 3:
            st.error("Storico insufficiente: servono almeno 3 anni di dati.")
            return

        perc = core.compute_percentiles(pivot_ytd, current_year)
        pct_attuale, ultimo_tdi = core.compute_current_percentile(pivot_ytd, current_year, metadata)
        zscore_series = core.compute_zscore_by_tdi(pivot_ytd, current_year, metadata)
        vol_context = core.compute_rolling_volatility_context(pivot_returns, current_year, metadata)
        dynamics = core.compute_percentile_dynamics(pivot_ytd, current_year, metadata)
        persistence = core.compute_anomaly_persistence(pivot_ytd, perc, current_year, metadata)
        cluster_df = core.cluster_historical_years(pivot_ytd, pivot_returns, current_year)
        current_regime = core.identify_current_regime(
            pivot_ytd, pivot_returns, current_year, cluster_df, metadata)
        regime_perc = core.compute_regime_conditional_percentiles(
            pivot_ytd, current_year, cluster_df, current_regime)
        forward_data = core.compute_forward_return_distribution(
            pivot_ytd, current_year, metadata,
            lookahead_days=lookahead_days, pct_tolerance=pct_tolerance)
        bootstrap_ci = core.bootstrap_percentile_bands(pivot_ytd, current_year, n_bootstrap=n_bootstrap)

    ytd_val = pivot_ytd[current_year].loc[ultimo_tdi] if ultimo_tdi > 0 else np.nan
    zscore_current = zscore_series.loc[ultimo_tdi] if ultimo_tdi in zscore_series.index else np.nan
    interpretation, _, emoji = core.get_anomaly_interpretation(pct_attuale, zscore_current)

    # --- Header metriche ----------------------------------------------------
    st.markdown("---")
    cols = st.columns(6)
    segno = "+" if (not pd.isna(ytd_val) and ytd_val >= 0) else ""
    cols[0].metric(f"YTD {current_year}", f"{segno}{ytd_val:.2f}%" if not pd.isna(ytd_val) else "N/D")
    cols[1].metric("Percentile", f"{pct_attuale:.1f}°" if not pd.isna(pct_attuale) else "N/D")
    cols[2].metric("Z-Score", f"{zscore_current:.2f}σ" if not pd.isna(zscore_current) else "N/D")
    cols[3].metric("Regime", current_regime)
    cols[4].metric("Streak fuori IQR", f"{persistence['current_streak']} gg")
    cols[5].metric("Trading Day", f"{ultimo_tdi}/{metadata.get('max_trading_days', C.DEFAULT_MAX_TRADING_DAYS)}")

    st.info(f"{emoji} **{interpretation}**")

    anni_storico = [a for a in anni_disponibili if a != current_year]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview Percentili", "📊 Z-Score & Volatilità", "⚡ Dinamiche Anomalia",
        "🎯 Analisi Regime", "🔮 Forward Returns",
    ])

    # ========== TAB 1 ==========
    with tab1:
        st.markdown("### Analisi percentile YTD")
        _box("""
        <b>📖 Come leggere il grafico</b><br>
        Le bande rappresentano la distribuzione storica dei rendimenti YTD per ogni trading day;
        la linea rossa è l'anno corrente. Quando esce dalle bande, c'è un'anomalia statistica.
        <ul>
        <li><b>Banda chiara (5°-95°)</b>: il 90% degli anni storici cade qui</li>
        <li><b>Banda scura (25°-75°)</b>: IQR, il 50% centrale</li>
        <li><b>Tratteggio</b>: mediana storica</li>
        </ul>
        L'asse X usa il <b>Trading Day Index</b> invece del giorno solare, per eliminare i
        disallineamenti da anni bisestili e festività variabili.
        """)

        st.plotly_chart(
            charts.build_main_percentile_chart(pivot_ytd, perc, current_year, ticker,
                                               metadata, bootstrap_ci),
            use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Statistiche distribuzione")
            def _at(col):
                return perc[col].loc[ultimo_tdi] if ultimo_tdi in perc.index else np.nan
            p5, p25, p50, p75, p95 = (_at(c) for c in ["p5", "p25", "p50", "p75", "p95"])
            st.dataframe(pd.DataFrame({
                "Metrica": ["Anni in analisi", "Mediana storica (TDI corrente)",
                            "IQR", "Range 5°-95°"],
                "Valore": [
                    f"{len(anni_storico)} ({min(anni_storico)}-{max(anni_storico)})",
                    f"{p50:.2f}%" if not pd.isna(p50) else "N/D",
                    f"{p25:.2f}% → {p75:.2f}%" if not pd.isna(p25) else "N/D",
                    f"{p5:.2f}% → {p95:.2f}%" if not pd.isna(p5) else "N/D",
                ],
            }), use_container_width=True, hide_index=True)

        with c2:
            st.markdown("#### Interpretazione")
            if not pd.isna(pct_attuale):
                if pct_attuale < 10 or pct_attuale > 90:
                    st.warning(
                        f"**Anomalia significativa.** Al {pct_attuale:.1f}° percentile, una posizione "
                        f"raggiunta storicamente solo nel {min(pct_attuale, 100 - pct_attuale):.1f}% "
                        f"degli anni. Può indicare un evento macro straordinario, un cambio strutturale "
                        f"o un'opportunità di mean reversion — da validare.")
                elif pct_attuale < 25 or pct_attuale > 75:
                    st.info(f"**Moderatamente anomalo.** Al {pct_attuale:.1f}° percentile è fuori "
                            f"dall'IQR ma non in territorio estremo.")
                else:
                    st.success(f"**Nella norma storica** ({pct_attuale:.1f}° percentile).")

        if bootstrap_ci:
            with st.expander("📐 Bootstrap Confidence Intervals"):
                st.markdown(f"""
                L'area gialla mostra l'incertezza nella stima delle bande, da **{n_bootstrap}**
                ricampionamenti bootstrap su {len(anni_storico)} anni.

                - **CI ampio** → alta incertezza: con più dati le bande potrebbero essere diverse
                - **CI stretto** → stime robuste

                Un'anomalia apparente che cade *dentro* il CI delle bande può essere un artefatto
                di campionamento.
                """)

    # ========== TAB 2 ==========
    with tab2:
        st.markdown("### Z-Score e contesto volatilità")
        _box("""
        <b>📖 Perché lo Z-Score</b><br>
        Il percentile dice <i>dove</i> sei nella distribuzione, lo Z-Score <i>quanto</i> sei lontano
        dalla media in deviazioni standard.
        <ul>
        <li><b>|Z| > 2</b>: anomalia statisticamente significativa (p &lt; 0.05)</li>
        <li><b>|Z| > 2.5</b>: molto significativa (p &lt; 0.01)</li>
        <li><b>|Z| > 3</b>: evento raro, ~0.3% delle osservazioni</li>
        </ul>
        La volatilità nel pannello inferiore è calcolata sui <b>rendimenti giornalieri veri</b>,
        non sulla variazione del cumulato YTD.
        """)

        st.plotly_chart(charts.build_zscore_chart(zscore_series, vol_context, current_year, metadata),
                        use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Z-Score attuale")
            if not pd.isna(zscore_current):
                p_value = 2 * (1 - stats.norm.cdf(abs(zscore_current)))
                st.metric("Z-Score", f"{zscore_current:.3f}σ")
                st.metric("P-Value (bilaterale)", f"{p_value:.4f}")
                if abs(zscore_current) > 2.5:
                    st.error("🔴 Anomalia molto significativa — evento raro")
                elif abs(zscore_current) > 2:
                    st.warning("🟠 Anomalia significativa — sotto la soglia del 5%")
                elif abs(zscore_current) > 1.5:
                    st.info("🟡 Deviazione moderata — da monitorare")
                else:
                    st.success("🟢 Nella norma statistica")

        with c2:
            st.markdown("#### Contesto volatilità")
            if not vol_context.empty and ultimo_tdi in vol_context.index:
                vz = vol_context.loc[ultimo_tdi, "vol_zscore"]
                if not pd.isna(vz):
                    st.metric("Vol Z-Score", f"{vz:.2f}σ")
                    if vz > 1.5:
                        st.warning("**Volatilità elevata.** La dispersione dei rendimenti è sopra "
                                   "la norma storica: le anomalie qui sono più rumorose.")
                    elif vz < -1.5:
                        st.info("**Volatilità bassa.** Mercato insolitamente calmo: le anomalie in "
                                "bassa volatilità tendono a essere più genuine.")
                    else:
                        st.success("Volatilità in linea con la media storica.")

    # ========== TAB 3 ==========
    with tab3:
        st.markdown("### Dinamiche dell'anomalia")
        _box("""
        <b>📖 Velocity &amp; Acceleration</b><br>
        Non basta sapere <i>dove</i> sei: conta <i>come</i> ci sei arrivato e <i>verso dove</i> stai andando.
        <ul>
        <li><b>Velocity</b>: quanto velocemente cambia il ranking percentile (Δ su 5 sedute)</li>
        <li><b>Acceleration</b>: la velocità sta aumentando o diminuendo?</li>
        <li><b>Persistenza</b>: da quante sedute sei fuori dalla zona normale?</li>
        </ul>
        Velocity negativa in accelerazione mentre sei già sotto il 25° percentile è un segnale di
        stress significativo.
        """)

        if not dynamics.empty:
            st.plotly_chart(
                charts.build_dynamics_chart(dynamics, persistence, current_year, metadata),
                use_container_width=True)

        vel = dynamics["velocity"].loc[ultimo_tdi] if ultimo_tdi in dynamics.index else np.nan
        acc = dynamics["acceleration"].loc[ultimo_tdi] if ultimo_tdi in dynamics.index else np.nan

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Velocity")
            if not pd.isna(vel):
                st.metric("Δ Percentile (5 sedute)", f"{vel:+.1f}")
                st.markdown(f"**Trend:** {'📈 Miglioramento' if vel > 0 else '📉 Peggioramento'}")
        with c2:
            st.markdown("#### Acceleration")
            if not pd.isna(acc):
                st.metric("ΔΔ Percentile", f"{acc:+.2f}")
                if acc > 2:
                    st.success("⬆️ Momentum positivo in aumento")
                elif acc < -2:
                    st.error("⬇️ Momentum negativo in aumento")
                else:
                    st.info("➡️ Momentum stabile")
        with c3:
            st.markdown("#### Persistenza")
            st.metric("Streak corrente", f"{persistence['current_streak']} sedute")
            st.metric("Max streak (anno)", f"{persistence['max_streak']} sedute")
            st.metric("% sedute fuori IQR", f"{persistence['pct_days_outside']:.1f}%")
            if persistence["current_streak"] > 10:
                st.warning("Anomalia persistente — potenziale cambio di regime")

        st.markdown("---")
        st.markdown("#### 🧠 Diagnosi combinata")
        if not pd.isna(vel) and not pd.isna(acc) and not pd.isna(pct_attuale):
            if pct_attuale < 25 and vel < -2 and acc < 0:
                st.error("**Segnale di stress critico.** Sotto il 25° percentile con velocità di "
                         "deterioramento in aumento: pattern tipico di capitolazione o crisi in corso.")
            elif pct_attuale > 75 and vel > 2 and acc > 0:
                st.warning("**Possibile euforia / blow-off top.** Sopra il 75° percentile con "
                           "accelerazione positiva: pattern spesso associato a eccessi speculativi.")
            elif abs(vel) < 1 and persistence["current_streak"] > 5:
                st.info("**Consolidamento in anomalia.** Stabilizzato in territorio anomalo: può "
                        "indicare un nuovo equilibrio o la preparazione di un movimento.")
            else:
                st.success("Dinamiche nella norma — nessun pattern critico rilevato.")

    # ========== TAB 4 ==========
    with tab4:
        st.markdown("### Analisi regime e clustering")
        _box("""
        <b>📖 Perché i regimi</b><br>
        Non tutti gli anni sono uguali: confrontare un anno bull con la media di anni misti è
        fuorviante. Il clustering raggruppa gli anni storici in <b>3 regimi</b> (Bull / Bear / Sideways)
        con un K-Means su 4 feature:
        <ul>
        <li><b>final_ret</b>: rendimento YTD finale</li>
        <li><b>path_vol</b>: volatilità sui rendimenti giornalieri</li>
        <li><b>max_dd</b>: maximum drawdown geometrico</li>
        <li><b>sharpe_proxy</b>: rendimento risk-adjusted annualizzato</li>
        </ul>
        """)

        if cluster_df.empty:
            st.warning("Dati insufficienti per l'analisi di regime (servono almeno 5 anni).")
        else:
            st.plotly_chart(
                charts.build_regime_chart(pivot_ytd, cluster_df, current_year, current_regime, metadata),
                use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Classificazione anni")
                summary = cluster_df.groupby("regime").agg({
                    "final_ret": ["count", "mean", "std"],
                    "max_dd": "mean", "sharpe_proxy": "mean",
                }).round(2)
                summary.columns = ["N anni", "Ret medio %", "Ret std", "DD medio %", "Sharpe medio"]
                st.dataframe(summary, use_container_width=True)

            with c2:
                st.markdown("#### Regime corrente")
                emoji_map = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}
                st.metric("Regime identificato",
                          f"{emoji_map.get(current_regime, '⚪')} {current_regime}")
                if current_regime in emoji_map:
                    anni_simili = cluster_df[cluster_df["regime"] == current_regime].index.tolist()
                    st.caption(f"Anni simili: {', '.join(map(str, anni_simili))}")

            if not regime_perc.empty and current_regime not in ("Unknown", "Insufficient Data"):
                st.markdown("---")
                st.markdown("#### Percentili regime-conditional")

                serie = pivot_ytd.get(current_year)
                anni_regime = [y for y in cluster_df[cluster_df["regime"] == current_regime].index
                               if y != current_year and y in pivot_ytd.columns]

                if serie is not None and ultimo_tdi > 0 and len(anni_regime) >= 3:
                    val = serie.loc[ultimo_tdi]
                    vals_regime = pivot_ytd[anni_regime].loc[ultimo_tdi].dropna()
                    if len(vals_regime) > 0:
                        pct_cond = (vals_regime < val).sum() / len(vals_regime) * 100
                        c1, c2 = st.columns(2)
                        c1.metric("Percentile (tutti gli anni)", f"{pct_attuale:.1f}°")
                        c2.metric(f"Percentile (solo {current_regime})", f"{pct_cond:.1f}°")

                        diff = pct_cond - pct_attuale
                        if abs(diff) > 15:
                            st.warning(
                                f"**Differenza significativa ({diff:+.1f} punti).** Rispetto a tutti "
                                f"gli anni sei al {pct_attuale:.1f}° percentile, ma rispetto ai soli "
                                f"anni {current_regime} sei al {pct_cond:.1f}°. Il contesto di regime "
                                f"cambia la lettura dell'anomalia.")

    # ========== TAB 5 ==========
    with tab5:
        st.markdown("### Forward returns e mean reversion")
        _box(f"""
        <b>📖 Probabilità condizionale</b><br>
        Storicamente, quando l'asset era in una posizione percentile simile (±{pct_tolerance}%)
        nello stesso periodo dell'anno, cosa è successo nelle <b>{lookahead_days} sedute</b> successive?
        <ul>
        <li>L'analisi <b>gestisce il cross-year</b>: se il lookahead supera fine anno, i rendimenti
        vengono combinati geometricamente</li>
        <li>Il lookahead è in <b>trading days</b>, non giorni di calendario</li>
        </ul>
        ⚠️ È un'analisi empirica su campioni piccoli, non una previsione. Un'osservazione per anno:
        con 20 anni di storia il campione massimo è 20, e la tolleranza percentile lo riduce ancora.
        """)

        if not forward_data:
            st.warning("Nessun dato storico comparabile. Prova ad aumentare la tolleranza percentile.")
        else:
            st.plotly_chart(charts.build_forward_returns_chart(forward_data), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Campioni trovati", forward_data["n_samples"])
            c1.metric("Percentile corrente", f"{forward_data['current_percentile']:.1f}°")
            c2.metric("Media forward", f"{forward_data['mean_forward']:+.2f}%")
            c2.metric("Mediana forward", f"{forward_data['median_forward']:+.2f}%")
            c3.metric("P(rendimento > 0)", f"{forward_data['prob_positive']:.1f}%")
            c3.metric("Std forward", f"{forward_data['std_forward']:.2f}%")

            st.markdown("---")
            st.markdown("#### Interpretazione")

            n = forward_data["n_samples"]
            if n < 5:
                st.warning(f"**Campione limitato ({n} osservazioni).** Le statistiche non sono "
                           f"affidabili: da usare con estrema cautela.")
            elif forward_data["prob_positive"] > 65:
                st.success(f"**Bias storico positivo.** In {n} casi simili, il "
                           f"{forward_data['prob_positive']:.1f}% delle volte il rendimento forward a "
                           f"{lookahead_days} sedute è stato positivo (media "
                           f"{forward_data['mean_forward']:+.2f}%, mediana "
                           f"{forward_data['median_forward']:+.2f}%).")
            elif forward_data["prob_positive"] < 35:
                st.error(f"**Bias storico negativo.** In {n} casi simili, solo il "
                         f"{forward_data['prob_positive']:.1f}% delle volte il forward è stato positivo "
                         f"(media {forward_data['mean_forward']:+.2f}%).")
            else:
                st.info(f"**Nessun bias chiaro.** Distribuzione bilanciata "
                        f"({forward_data['prob_positive']:.1f}% positivi), forward medio "
                        f"{forward_data['mean_forward']:+.2f}%.")

            with st.expander("📅 Anni con pattern simile"):
                st.write(f"Anni con percentile ±{pct_tolerance}% al trading day corrente:")
                st.write(", ".join(map(str, forward_data["matching_years"])))

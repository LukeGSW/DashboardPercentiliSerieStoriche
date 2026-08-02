"""
=============================================================================
kq.charts — Costruttori Plotly (tema scuro Kriterion Quant)
=============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kq import config as C
from kq.core import tdi_to_labels


def _layout(fig: go.Figure, height: int = 500, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C.COLORS["background"],
        plot_bgcolor=C.COLORS["background"],
        height=height,
        **kwargs,
    )
    return fig


# =============================================================================
# SINGLE ASSET
# =============================================================================
def build_main_percentile_chart(pivot, perc, current_year, ticker, metadata, bootstrap_ci=None):
    """Bande percentile stagionali + equity YTD corrente."""
    fig = go.Figure()

    valid_tdi = perc.dropna().index
    if len(valid_tdi) == 0:
        fig.add_annotation(text="Storia insufficiente per le bande percentile",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _layout(fig)

    labels = tdi_to_labels(valid_tdi)
    pv = perc.loc[valid_tdi]

    if bootstrap_ci:
        up = bootstrap_ci["p95_ci_upper"].loc[valid_tdi]
        lo = bootstrap_ci["p95_ci_lower"].loc[valid_tdi]
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=up.tolist() + lo.tolist()[::-1],
            fill="toself", fillcolor=C.COLORS["ci_band"],
            line=dict(color="rgba(0,0,0,0)"), name="95% CI (Bootstrap)", hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=pv["p95"].tolist() + pv["p5"].tolist()[::-1],
        fill="toself", fillcolor=C.COLORS["band_95"],
        line=dict(color="rgba(0,0,0,0)"), name="5° - 95° Pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=pv["p75"].tolist() + pv["p25"].tolist()[::-1],
        fill="toself", fillcolor=C.COLORS["band_iqr"],
        line=dict(color="rgba(0,0,0,0)"), name="25° - 75° Pct (IQR)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=pv["p50"].tolist(), mode="lines",
        line=dict(color=C.COLORS["median"], width=1.5, dash="dash"), name="Mediana (50° Pct)",
    ))

    serie = pivot.get(current_year)
    if serie is not None:
        ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
        sp = serie.loc[:ultimo_tdi].dropna()
        sp = sp[sp.index.isin(valid_tdi)]
        if len(sp) > 0:
            lab = tdi_to_labels(sp.index)
            fig.add_trace(go.Scatter(
                x=lab, y=sp.values, mode="lines",
                line=dict(color=C.COLORS["ytd"], width=3), name=f"YTD {current_year}",
            ))
            ultimo = sp.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[lab[-1]], y=[ultimo], mode="markers+text",
                marker=dict(color=C.COLORS["ytd"], size=10),
                text=[f"{'+' if ultimo >= 0 else ''}{ultimo:.2f}%"],
                textposition="top right",
                textfont=dict(color=C.COLORS["ytd"], size=13, family="Arial Black"),
                showlegend=False, hoverinfo="skip",
            ))

    return _layout(
        fig,
        xaxis=dict(title="Trading Day (calendario approssimato)", showgrid=True,
                   gridcolor=C.COLORS["grid"], tickangle=-45),
        yaxis=dict(title="Rendimento YTD (%)", showgrid=True, gridcolor=C.COLORS["grid"],
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.25)", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=40, t=40, b=80),
    )


def build_zscore_chart(zscore_series, vol_context, current_year, metadata):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Z-Score YTD vs Storico", "Volatilità Contestuale"),
                        row_heights=[0.6, 0.4])

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    z = zscore_series.loc[:ultimo_tdi].dropna()
    if len(z) > 0:
        labels = tdi_to_labels(z.index)
        colors = [C.COLORS["zscore_pos"] if v >= 0 else C.COLORS["zscore_neg"] for v in z.values]
        fig.add_trace(go.Bar(x=labels, y=z.values, marker_color=colors, showlegend=False),
                      row=1, col=1)

    for sigma, dash in [(2, "solid"), (1, "dash"), (-1, "dash"), (-2, "solid")]:
        fig.add_hline(y=sigma, line_dash=dash, line_color="rgba(255,255,255,0.3)",
                      annotation_text=f"{sigma}σ", annotation_position="right", row=1, col=1)

    if not vol_context.empty:
        vc = vol_context.loc[:ultimo_tdi].dropna()
        if len(vc) > 0:
            lv = tdi_to_labels(vc.index)
            fig.add_trace(go.Scatter(x=lv, y=vc["vol_corrente"], mode="lines",
                                     line=dict(color=C.COLORS["ytd"], width=2),
                                     name=f"Vol {current_year}"), row=2, col=1)
            fig.add_trace(go.Scatter(x=lv, y=vc["vol_storica_mean"], mode="lines",
                                     line=dict(color=C.COLORS["median"], width=1.5, dash="dash"),
                                     name="Vol media storica"), row=2, col=1)

    fig.update_yaxes(title_text="Z-Score (σ)", row=1, col=1, gridcolor=C.COLORS["grid"])
    fig.update_yaxes(title_text="Volatilità (%)", row=2, col=1, gridcolor=C.COLORS["grid"])
    return _layout(fig, height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02))


def build_dynamics_chart(dynamics_df, persistence_data, current_year, metadata):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Percentile Rolling", "Velocità (Δ Percentile)",
                                        "Accelerazione (ΔΔ Percentile)"),
                        row_heights=[0.4, 0.3, 0.3])

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    d = dynamics_df.loc[:ultimo_tdi]

    pct = d["percentile"].dropna()
    if len(pct) > 0:
        fig.add_trace(go.Scatter(x=tdi_to_labels(pct.index), y=pct.values, mode="lines",
                                 line=dict(color=C.COLORS["persistence"], width=2),
                                 fill="tozeroy", fillcolor="rgba(76, 201, 240, 0.2)",
                                 name="Percentile"), row=1, col=1)
    fig.add_hrect(y0=25, y1=75, fillcolor="rgba(100,149,237,0.1)", line_width=0, row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1)

    vel = d["velocity"].dropna()
    if len(vel) > 0:
        fig.add_trace(go.Bar(x=tdi_to_labels(vel.index), y=vel.values,
                             marker_color=[C.COLORS["zscore_pos"] if v >= 0 else C.COLORS["zscore_neg"]
                                           for v in vel.values], showlegend=False), row=2, col=1)

    acc = d["acceleration"].dropna()
    if len(acc) > 0:
        fig.add_trace(go.Bar(x=tdi_to_labels(acc.index), y=acc.values,
                             marker_color=[C.COLORS["velocity"] if a >= 0 else C.COLORS["acceleration"]
                                           for a in acc.values], showlegend=False), row=3, col=1)

    fig.update_yaxes(title_text="Pct", row=1, col=1, gridcolor=C.COLORS["grid"], range=[0, 100])
    fig.update_yaxes(title_text="Δ Pct", row=2, col=1, gridcolor=C.COLORS["grid"])
    fig.update_yaxes(title_text="ΔΔ Pct", row=3, col=1, gridcolor=C.COLORS["grid"])
    return _layout(fig, height=650, showlegend=False)


def build_regime_chart(pivot, cluster_df, current_year, current_regime, metadata):
    if cluster_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti per il clustering",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _layout(fig, height=450)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Anni per regime", "Traiettorie YTD per regime"),
                        column_widths=[0.4, 0.6])

    rc = {"Bull": C.COLORS["regime_bull"], "Bear": C.COLORS["regime_bear"],
          "Sideways": C.COLORS["regime_sideways"]}

    for regime in ["Bull", "Bear", "Sideways"]:
        rd = cluster_df[cluster_df["regime"] == regime]
        if len(rd) > 0:
            fig.add_trace(go.Scatter(x=rd["final_ret"], y=rd["path_vol"], mode="markers+text",
                                     marker=dict(color=rc[regime], size=12),
                                     text=rd.index.astype(str), textposition="top center",
                                     textfont=dict(size=9), name=regime), row=1, col=1)

    storico = pivot.drop(columns=[current_year], errors="ignore")
    for regime in ["Bull", "Bear", "Sideways"]:
        anni = [y for y in cluster_df[cluster_df["regime"] == regime].index if y in storico.columns]
        for i, anno in enumerate(anni):
            s = storico[anno].dropna()
            fig.add_trace(go.Scatter(x=tdi_to_labels(s.index), y=s.values, mode="lines",
                                     line=dict(color=rc[regime], width=1), opacity=0.4,
                                     name=regime if i == 0 else None, showlegend=(i == 0),
                                     legendgroup=regime), row=1, col=2)

    serie = pivot.get(current_year)
    if serie is not None:
        ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
        s = serie.loc[:ultimo_tdi].dropna()
        if len(s) > 0:
            fig.add_trace(go.Scatter(x=tdi_to_labels(s.index), y=s.values, mode="lines",
                                     line=dict(color="white", width=3),
                                     name=f"{current_year} (corrente)"), row=1, col=2)

    fig.update_xaxes(title_text="Rendimento finale (%)", row=1, col=1, gridcolor=C.COLORS["grid"])
    fig.update_yaxes(title_text="Volatilità path (%)", row=1, col=1, gridcolor=C.COLORS["grid"])
    fig.update_xaxes(title_text="Trading Day", row=1, col=2, gridcolor=C.COLORS["grid"])
    fig.update_yaxes(title_text="YTD %", row=1, col=2, gridcolor=C.COLORS["grid"])
    return _layout(fig, height=450)


def build_forward_returns_chart(forward_data):
    if not forward_data or "forward_returns" not in forward_data:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti per l'analisi forward",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _layout(fig, height=400)

    fwd = forward_data["forward_returns"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=fwd, nbinsx=20, marker_color=C.COLORS["persistence"],
                               opacity=0.7, name="Forward Returns"))
    fig.add_vline(x=forward_data["mean_forward"], line_dash="dash", line_color=C.COLORS["ytd"],
                  annotation_text=f"Media: {forward_data['mean_forward']:.2f}%")
    fig.add_vline(x=0, line_dash="solid", line_color="white", opacity=0.5)

    return _layout(fig, height=400, showlegend=False,
                   xaxis_title=f"Rendimento forward ({forward_data['lookahead_days']} sedute) %",
                   yaxis_title="Frequenza")


# =============================================================================
# SCREENER
# =============================================================================
def build_screener_map(df: pd.DataFrame, max_labels: int = 25) -> go.Figure:
    """
    Mappa dei candidati: ampiezza della dislocazione (x) contro momentum
    residuo di breve (y). Sono i due assi su cui e' costruita la
    classificazione, quindi i quadranti coincidono con i setup:

        in basso a sinistra  -> ↓↓ IN CADUTA    (giu' e ancora in discesa)
        in alto a sinistra   -> ↓ STABILIZZATO  (giu' ma ha smesso di scendere)
        in alto a destra     -> ↑↑ ESTESO       (su e ancora in salita)
        in basso a destra    -> ↑ ESAURITO      (su ma ha girato in giu')

    Il colore segue l'AZIONE validata, non il quadrante: i due stati di destra
    si operano entrambi al RIBASSO, quelli di sinistra non si operano affatto.

    L'asse Y NON usa la velocity del rank percentile: essendo limitata in
    [0,100] satura, e un titolo gia' all'ultimo posto avrebbe velocity zero
    proprio mentre sta crollando.
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="Nessun candidato", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return _layout(fig, height=520)

    fig.add_vrect(x0=-10, x1=0, fillcolor="rgba(255,107,107,0.05)", line_width=0)
    fig.add_vrect(x0=0, x1=10, fillcolor="rgba(0,210,106,0.05)", line_width=0)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)")
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.25)")

    for setup, grp in df.groupby("setup"):
        size = grp["Score"].fillna(20).clip(10, 100) / 4 + 6
        fig.add_trace(go.Scatter(
            x=grp["Disloc σ"], y=grp["Mom residuo"], mode="markers",
            marker=dict(size=size, color=C.SETUP_COLORS.get(setup, C.COLORS["neutral"]),
                        opacity=0.8, line=dict(width=0.5, color="rgba(255,255,255,0.35)")),
            name=setup,
            text=grp["Ticker"],
            customdata=np.stack([grp["Rend %"], grp["Score"].fillna(0),
                                 grp["Vol pctl"], grp["Rank XS"]], axis=-1),
            hovertemplate=("<b>%{text}</b><br>Dislocazione: %{x:.2f}σ"
                           "<br>Mom residuo: %{y:+.2f}σ"
                           "<br>Rend: %{customdata[0]:.1f}%<br>Score: %{customdata[1]:.0f}"
                           "<br>Rank XS: %{customdata[3]:.0f}°"
                           "<br>Vol pctl: %{customdata[2]:.0f}°<extra></extra>"),
        ))

    top = df.nlargest(max_labels, "Score")
    fig.add_trace(go.Scatter(
        x=top["Disloc σ"], y=top["Mom residuo"], mode="text", text=top["Ticker"],
        textposition="top center", textfont=dict(size=9, color="rgba(255,255,255,0.75)"),
        showlegend=False, hoverinfo="skip",
    ))

    return _layout(
        fig, height=520,
        xaxis=dict(title="Ampiezza dislocazione (σ)   ←  sotto il benchmark  |  sopra  →",
                   gridcolor=C.COLORS["grid"], zeroline=False),
        yaxis=dict(title="Momentum residuo 10 sedute (σ)   ↓ peggiora  |  migliora ↑",
                   gridcolor=C.COLORS["grid"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=30, t=50, b=60),
    )


def build_breadth_chart(df: pd.DataFrame) -> go.Figure:
    """Distribuzione della dislocazione sull'universo: dove si concentra lo stress."""
    fig = go.Figure()
    if df.empty:
        return _layout(fig, height=320)

    z = df["Disloc σ"].dropna()
    fig.add_trace(go.Histogram(
        x=z, nbinsx=50, marker_color=C.COLORS["persistence"], opacity=0.75, showlegend=False,
    ))
    for x, lab, col in [(-1.5, "−1.5σ", C.COLORS["zscore_neg"]),
                        (1.5, "+1.5σ", C.COLORS["zscore_pos"])]:
        fig.add_vline(x=x, line_dash="dash", line_color=col, annotation_text=lab)
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)")

    return _layout(fig, height=320, bargap=0.02,
                   xaxis=dict(title="Dislocazione idiosincratica (σ)", gridcolor=C.COLORS["grid"]),
                   yaxis=dict(title="N. strumenti", gridcolor=C.COLORS["grid"]))


def build_category_chart(df: pd.DataFrame) -> go.Figure:
    """Dislocazione mediana per categoria: dove si e' spostato il mercato."""
    fig = go.Figure()
    if df.empty:
        return _layout(fig, height=380)

    agg = (df.groupby("Categoria")["Disloc σ"]
             .agg(["median", "count"])
             .query("count >= 3")
             .sort_values("median"))
    if agg.empty:
        return _layout(fig, height=380)

    colors = [C.COLORS["zscore_neg"] if v < 0 else C.COLORS["zscore_pos"] for v in agg["median"]]
    fig.add_trace(go.Bar(
        x=agg["median"], y=agg.index, orientation="h", marker_color=colors,
        text=[f"{v:+.2f}σ  (n={int(n)})" for v, n in zip(agg["median"], agg["count"])],
        textposition="outside", showlegend=False,
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)")

    return _layout(fig, height=max(320, 40 * len(agg) + 120),
                   xaxis=dict(title="Dislocazione mediana (σ)", gridcolor=C.COLORS["grid"]),
                   yaxis=dict(title="", gridcolor=C.COLORS["grid"]),
                   margin=dict(l=160, r=90, t=30, b=50))

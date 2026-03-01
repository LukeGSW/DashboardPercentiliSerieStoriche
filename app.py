"""
=============================================================================
YTD Seasonality & Anomaly Detection Quant Dashboard — Kriterion Quant
=============================================================================
Descrizione:
    Dashboard avanzata per l'analisi quantitativa della stagionalità YTD
    e l'individuazione di anomalie statistiche multi-dimensionali.
    
Features:
    - Analisi Percentile YTD con bande bootstrap CI
    - Z-Score dinamico e contestualizzazione volatilità
    - Velocità e accelerazione delle anomalie
    - Persistenza anomalie (run-length)
    - Clustering regime-conditional
    - Probabilità mean reversion forward
    - Multi-asset anomaly scanner

Fonte dati: EODHD API (Adjusted Close giornaliero)
Deploy: Streamlit Community Cloud (chiave API via st.secrets)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAZIONE PAGINA
# =============================================================================
st.set_page_config(
    page_title="YTD Anomaly Detection Dashboard | Kriterion Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# COSTANTI E PALETTE COLORI
# =============================================================================
EODHD_BASE_URL = "https://eodhd.com/api/eod"

# Palette colori Kriterion Quant
COLORS = {
    "band_95": "rgba(100, 149, 237, 0.15)",
    "band_iqr": "rgba(100, 149, 237, 0.35)",
    "median": "rgba(173, 216, 230, 0.9)",
    "ytd": "#FF4B4B",
    "zscore_pos": "#00D26A",
    "zscore_neg": "#FF6B6B",
    "velocity": "#9D4EDD",
    "acceleration": "#F72585",
    "persistence": "#4CC9F0",
    "ci_band": "rgba(255, 193, 7, 0.2)",
    "regime_bull": "#00D26A",
    "regime_bear": "#FF6B6B",
    "regime_sideways": "#FFC107",
    "background": "#0E1117",
    "grid": "rgba(255,255,255,0.07)",
}

# =============================================================================
# 1. DATA FETCHING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Scarica la serie storica giornaliera (Adjusted Close) da EODHD API.
    """
    api_key = st.secrets["EODHD_API_KEY"]
    today = date.today().strftime("%Y-%m-%d")

    params = {
        "api_token": api_key,
        "from": start_date,
        "to": today,
        "fmt": "json",
        "period": "d",
    }

    try:
        url = f"{EODHD_BASE_URL}/{ticker}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or isinstance(data, dict):
            return pd.DataFrame()

        df = pd.DataFrame(data)[["date", "adjusted_close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df

    except Exception:
        return pd.DataFrame()


# =============================================================================
# 2. CALCOLO YTD E MAPPATURA SU DAY OF YEAR (DOY)
# =============================================================================
def compute_ytd_by_doy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i rendimenti YTD per ogni anno solare mappati su DOY 1-366.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.day_of_year

    anni = sorted(df["year"].unique())
    ytd_dict = {}

    for anno in anni:
        anno_prec = anno - 1
        df_prec = df[df["year"] == anno_prec]

        if df_prec.empty:
            base_price = df[df["year"] == anno]["adjusted_close"].iloc[0]
        else:
            base_price = df_prec["adjusted_close"].iloc[-1]

        df_anno = df[df["year"] == anno].copy()
        df_anno["ytd_pct"] = (df_anno["adjusted_close"] / base_price - 1) * 100

        serie_doy = pd.Series(
            data=df_anno["ytd_pct"].values,
            index=df_anno["doy"].values,
        )
        serie_full = serie_doy.reindex(range(1, 367))
        serie_full = serie_full.ffill()

        ytd_dict[anno] = serie_full

    pivot = pd.DataFrame(ytd_dict)
    return pivot


# =============================================================================
# 3. CALCOLO PERCENTILI STORICI
# =============================================================================
def compute_percentiles(pivot: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """
    Calcola percentili 5°, 25°, 50°, 75°, 95° escludendo l'anno corrente.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")

    perc = pd.DataFrame(index=pivot.index)
    perc["p5"] = storico.quantile(0.05, axis=1)
    perc["p25"] = storico.quantile(0.25, axis=1)
    perc["p50"] = storico.quantile(0.50, axis=1)
    perc["p75"] = storico.quantile(0.75, axis=1)
    perc["p95"] = storico.quantile(0.95, axis=1)

    return perc


# =============================================================================
# 4. CALCOLO PERCENTILE CORRENTE
# =============================================================================
def compute_current_percentile(pivot: pd.DataFrame, current_year: int) -> float:
    """
    Percentile dell'YTD corrente rispetto alla distribuzione storica.
    """
    serie_ytd_corrente = pivot.get(current_year)
    if serie_ytd_corrente is None:
        return np.nan

    ultimo_doy = serie_ytd_corrente.last_valid_index()
    if ultimo_doy is None:
        return np.nan

    valore_corrente = serie_ytd_corrente.loc[ultimo_doy]

    storico = pivot.drop(columns=[current_year], errors="ignore")
    valori_storici = storico.loc[ultimo_doy].dropna().values

    if len(valori_storici) == 0:
        return np.nan

    percentile = (np.sum(valori_storici < valore_corrente) / len(valori_storici)) * 100
    return round(percentile, 1)


# =============================================================================
# 5. Z-SCORE DINAMICO
# =============================================================================
def compute_zscore_by_doy(pivot: pd.DataFrame, current_year: int) -> pd.Series:
    """
    Z-score dell'YTD corrente vs distribuzione storica per ogni DOY.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    serie_corrente = pivot.get(current_year)
    
    if serie_corrente is None:
        return pd.Series(dtype=float)
    
    mu = storico.mean(axis=1)
    sigma = storico.std(axis=1)
    
    # Evita divisione per zero
    sigma = sigma.replace(0, np.nan)
    
    zscore = (serie_corrente - mu) / sigma
    return zscore


def compute_rolling_volatility_context(pivot: pd.DataFrame, current_year: int, window: int = 20) -> pd.DataFrame:
    """
    Calcola la volatilità rolling dell'anno corrente vs media storica.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    serie_corrente = pivot.get(current_year)
    
    if serie_corrente is None:
        return pd.DataFrame()
    
    # Volatilità rolling anno corrente (su returns giornalieri)
    returns_corrente = serie_corrente.diff()
    vol_corrente = returns_corrente.rolling(window=window, min_periods=5).std()
    
    # Volatilità media storica per DOY
    vol_storica_list = []
    for anno in storico.columns:
        returns_anno = storico[anno].diff()
        vol_anno = returns_anno.rolling(window=window, min_periods=5).std()
        vol_storica_list.append(vol_anno)
    
    vol_storica_df = pd.concat(vol_storica_list, axis=1)
    vol_storica_mean = vol_storica_df.mean(axis=1)
    vol_storica_std = vol_storica_df.std(axis=1)
    
    result = pd.DataFrame({
        "vol_corrente": vol_corrente,
        "vol_storica_mean": vol_storica_mean,
        "vol_storica_std": vol_storica_std,
        "vol_zscore": (vol_corrente - vol_storica_mean) / vol_storica_std.replace(0, np.nan),
    })
    
    return result


# =============================================================================
# 6. VELOCITÀ E ACCELERAZIONE ANOMALIA
# =============================================================================
def compute_percentile_dynamics(pivot: pd.DataFrame, current_year: int, window: int = 5) -> pd.DataFrame:
    """
    Calcola il percentile rolling, la sua velocità e accelerazione.
    """
    serie = pivot.get(current_year)
    if serie is None:
        return pd.DataFrame()
    
    storico = pivot.drop(columns=[current_year], errors="ignore")
    
    # Calcola percentile per ogni DOY
    pct_series = pd.Series(index=serie.index, dtype=float)
    for doy in serie.dropna().index:
        val = serie.loc[doy]
        hist_vals = storico.loc[doy].dropna()
        if len(hist_vals) > 0:
            pct_series.loc[doy] = (hist_vals < val).sum() / len(hist_vals) * 100
    
    velocity = pct_series.diff(window)
    acceleration = velocity.diff(window)
    
    result = pd.DataFrame({
        "percentile": pct_series,
        "velocity": velocity,
        "acceleration": acceleration,
    })
    
    return result


# =============================================================================
# 7. PERSISTENZA ANOMALIA
# =============================================================================
def compute_anomaly_persistence(pivot: pd.DataFrame, perc: pd.DataFrame, current_year: int) -> dict:
    """
    Calcola giorni consecutivi fuori dall'IQR e statistiche correlate.
    """
    serie = pivot.get(current_year)
    if serie is None:
        return {"current_streak": 0, "max_streak": 0, "total_days_outside": 0, "streaks": pd.Series(dtype=int)}
    
    serie = serie.dropna()
    p25 = perc["p25"].loc[serie.index]
    p75 = perc["p75"].loc[serie.index]
    
    below_iqr = serie < p25
    above_iqr = serie > p75
    outside_iqr = below_iqr | above_iqr
    
    # Run-length per streak corrente
    streaks = outside_iqr.astype(int).copy()
    streak_groups = (~outside_iqr).cumsum()
    streaks = outside_iqr.groupby(streak_groups).cumsum()
    
    current_streak = int(streaks.iloc[-1]) if len(streaks) > 0 and outside_iqr.iloc[-1] else 0
    max_streak = int(streaks.max()) if len(streaks) > 0 else 0
    total_days_outside = int(outside_iqr.sum())
    
    # Direzione anomalia corrente
    if len(serie) > 0:
        last_val = serie.iloc[-1]
        last_doy = serie.index[-1]
        if last_val < p25.loc[last_doy]:
            direction = "below"
        elif last_val > p75.loc[last_doy]:
            direction = "above"
        else:
            direction = "within"
    else:
        direction = "unknown"
    
    return {
        "current_streak": current_streak,
        "max_streak": max_streak,
        "total_days_outside": total_days_outside,
        "pct_days_outside": round(total_days_outside / len(serie) * 100, 1) if len(serie) > 0 else 0,
        "direction": direction,
        "streaks": streaks,
        "outside_iqr": outside_iqr,
    }


# =============================================================================
# 8. REGIME CLUSTERING
# =============================================================================
def cluster_historical_years(pivot: pd.DataFrame, current_year: int, n_clusters: int = 3) -> pd.DataFrame:
    """
    Clustering degli anni storici per regime (bull/bear/sideways).
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    
    if storico.shape[1] < n_clusters:
        return pd.DataFrame()
    
    # Features per anno
    features = pd.DataFrame(index=storico.columns)
    features["final_ret"] = storico.iloc[-1]
    features["path_vol"] = storico.diff().std()
    features["max_dd"] = storico.apply(lambda x: (x - x.cummax()).min())
    features["sharpe_proxy"] = features["final_ret"] / features["path_vol"].replace(0, np.nan)
    
    features_clean = features.dropna()
    
    if len(features_clean) < n_clusters:
        return pd.DataFrame()
    
    # Normalizzazione e clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_clean[["final_ret", "path_vol", "max_dd"]])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_clean["cluster"] = kmeans.fit_predict(features_scaled)
    
    # Assegna label semantici basati sul rendimento medio del cluster
    cluster_means = features_clean.groupby("cluster")["final_ret"].mean().sort_values()
    label_map = {cluster_means.index[0]: "Bear", cluster_means.index[-1]: "Bull"}
    for c in cluster_means.index:
        if c not in label_map:
            label_map[c] = "Sideways"
    
    features_clean["regime"] = features_clean["cluster"].map(label_map)
    
    return features_clean


def identify_current_regime(pivot: pd.DataFrame, current_year: int, cluster_df: pd.DataFrame) -> str:
    """
    Identifica il regime più probabile dell'anno corrente basandosi sulle features YTD.
    """
    if cluster_df.empty:
        return "Unknown"
    
    serie_corrente = pivot.get(current_year)
    if serie_corrente is None:
        return "Unknown"
    
    serie_corrente = serie_corrente.dropna()
    if len(serie_corrente) < 20:
        return "Insufficient Data"
    
    # Calcola features correnti
    current_ret = serie_corrente.iloc[-1]
    current_vol = serie_corrente.diff().std()
    current_dd = (serie_corrente - serie_corrente.cummax()).min()
    
    # Trova il regime più simile (nearest neighbor semplificato)
    distances = []
    for _, row in cluster_df.iterrows():
        dist = np.sqrt(
            (current_ret - row["final_ret"])**2 +
            (current_vol - row["path_vol"])**2 +
            (current_dd - row["max_dd"])**2
        )
        distances.append((row["regime"], dist))
    
    # Vota per regime più frequente tra i 3 più vicini
    distances.sort(key=lambda x: x[1])
    top_3 = [d[0] for d in distances[:3]]
    regime_counts = pd.Series(top_3).value_counts()
    
    return regime_counts.index[0]


def compute_regime_conditional_percentiles(pivot: pd.DataFrame, current_year: int, 
                                            cluster_df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """
    Calcola percentili usando solo anni dello stesso regime.
    """
    if cluster_df.empty or regime in ["Unknown", "Insufficient Data"]:
        return pd.DataFrame()
    
    same_regime_years = cluster_df[cluster_df["regime"] == regime].index.tolist()
    same_regime_years = [y for y in same_regime_years if y != current_year and y in pivot.columns]
    
    if len(same_regime_years) < 3:
        return pd.DataFrame()
    
    storico_regime = pivot[same_regime_years]
    
    perc = pd.DataFrame(index=pivot.index)
    perc["p5"] = storico_regime.quantile(0.05, axis=1)
    perc["p25"] = storico_regime.quantile(0.25, axis=1)
    perc["p50"] = storico_regime.quantile(0.50, axis=1)
    perc["p75"] = storico_regime.quantile(0.75, axis=1)
    perc["p95"] = storico_regime.quantile(0.95, axis=1)
    
    return perc


# =============================================================================
# 9. FORWARD RETURNS & MEAN REVERSION
# =============================================================================
def compute_forward_return_distribution(pivot: pd.DataFrame, current_year: int, 
                                         lookahead_days: int = 20, 
                                         pct_tolerance: float = 10) -> dict:
    """
    Distribuzione dei rendimenti forward storici quando in anomalia simile.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    serie_corrente = pivot.get(current_year)
    
    if serie_corrente is None:
        return {}
    
    serie_corrente = serie_corrente.dropna()
    if len(serie_corrente) == 0:
        return {}
    
    ultimo_doy = serie_corrente.index[-1]
    current_val = serie_corrente.iloc[-1]
    
    # Calcola percentile corrente
    hist_vals_at_doy = storico.loc[ultimo_doy].dropna()
    if len(hist_vals_at_doy) == 0:
        return {}
    
    current_pct = (hist_vals_at_doy < current_val).sum() / len(hist_vals_at_doy) * 100
    
    # Trova anni storici con percentile simile a questo DOY
    forward_rets = []
    matching_years = []
    
    for anno in storico.columns:
        val_doy = storico.loc[ultimo_doy, anno]
        if pd.isna(val_doy):
            continue
        
        hist_pct = (hist_vals_at_doy < val_doy).sum() / len(hist_vals_at_doy) * 100
        
        if abs(hist_pct - current_pct) <= pct_tolerance:
            future_doy = min(ultimo_doy + lookahead_days, 366)
            if future_doy in storico.index:
                future_val = storico.loc[future_doy, anno]
                if not pd.isna(future_val):
                    fwd_ret = future_val - val_doy
                    forward_rets.append(fwd_ret)
                    matching_years.append(anno)
    
    if len(forward_rets) == 0:
        return {}
    
    fwd_series = pd.Series(forward_rets)
    
    return {
        "forward_returns": fwd_series,
        "matching_years": matching_years,
        "current_percentile": current_pct,
        "mean_forward": fwd_series.mean(),
        "median_forward": fwd_series.median(),
        "std_forward": fwd_series.std(),
        "prob_positive": (fwd_series > 0).mean() * 100,
        "n_samples": len(fwd_series),
        "lookahead_days": lookahead_days,
    }


# =============================================================================
# 10. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================
def bootstrap_percentile_bands(pivot: pd.DataFrame, current_year: int, 
                                n_bootstrap: int = 500, alpha: float = 0.05) -> dict:
    """
    Calcola intervalli di confidenza bootstrap per le bande percentile.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    n_years = storico.shape[1]
    
    if n_years < 5:
        return {}
    
    p5_samples = []
    p95_samples = []
    p50_samples = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample_cols = np.random.choice(storico.columns, size=n_years, replace=True)
        sample = storico[sample_cols]
        p5_samples.append(sample.quantile(0.05, axis=1))
        p50_samples.append(sample.quantile(0.50, axis=1))
        p95_samples.append(sample.quantile(0.95, axis=1))
    
    p5_df = pd.concat(p5_samples, axis=1)
    p50_df = pd.concat(p50_samples, axis=1)
    p95_df = pd.concat(p95_samples, axis=1)
    
    return {
        "p5_ci_lower": p5_df.quantile(alpha/2, axis=1),
        "p5_ci_upper": p5_df.quantile(1-alpha/2, axis=1),
        "p50_ci_lower": p50_df.quantile(alpha/2, axis=1),
        "p50_ci_upper": p50_df.quantile(1-alpha/2, axis=1),
        "p95_ci_lower": p95_df.quantile(alpha/2, axis=1),
        "p95_ci_upper": p95_df.quantile(1-alpha/2, axis=1),
    }


# =============================================================================
# 11. MULTI-ASSET SCANNER
# =============================================================================
def scan_universe_for_anomalies(tickers: list, start_date: str, 
                                 threshold_pct: float = 15) -> pd.DataFrame:
    """
    Scansiona un universo di ticker per individuare anomalie estreme.
    """
    current_year = date.today().year
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Analisi {ticker}...")
        progress_bar.progress((i + 1) / len(tickers))
        
        df = fetch_ohlcv(ticker, start_date)
        if df.empty or len(df) < 252:  # Almeno 1 anno di dati
            continue
        
        pivot = compute_ytd_by_doy(df)
        
        if current_year not in pivot.columns:
            continue
        
        pct = compute_current_percentile(pivot, current_year)
        if pd.isna(pct):
            continue
        
        serie_corrente = pivot[current_year].dropna()
        if len(serie_corrente) == 0:
            continue
        
        ytd_val = serie_corrente.iloc[-1]
        
        # Calcola Z-score
        zscore_series = compute_zscore_by_doy(pivot, current_year)
        zscore_current = zscore_series.dropna().iloc[-1] if len(zscore_series.dropna()) > 0 else np.nan
        
        # Classificazione anomalia
        if pct <= threshold_pct:
            anomaly_type = "🔴 Coda Inferiore"
        elif pct >= (100 - threshold_pct):
            anomaly_type = "🟢 Coda Superiore"
        else:
            anomaly_type = "⚪ Normale"
        
        results.append({
            "Ticker": ticker,
            "YTD %": round(ytd_val, 2),
            "Percentile": round(pct, 1),
            "Z-Score": round(zscore_current, 2) if not pd.isna(zscore_current) else None,
            "Anomaly": anomaly_type,
        })
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Percentile").reset_index(drop=True)
    
    return df_results


# =============================================================================
# 12. UTILITY FUNCTIONS
# =============================================================================
def doy_to_label(doy_series: pd.Index, ref_year: int = 2024) -> list:
    """Converte DOY in etichette MM/DD."""
    labels = []
    for doy in doy_series:
        try:
            d = datetime(ref_year, 1, 1) + pd.Timedelta(days=int(doy) - 1)
            labels.append(d.strftime("%b %d"))
        except Exception:
            labels.append(str(doy))
    return labels


def get_anomaly_interpretation(pct: float, zscore: float = None) -> tuple:
    """Restituisce interpretazione testuale e colore per l'anomalia."""
    if pd.isna(pct):
        return "Dati insufficienti", "gray", "⚪"
    
    if pct >= 90:
        text = "Anomalia ESTREMA positiva - performance eccezionalmente superiore alla norma storica"
        color = COLORS["zscore_pos"]
        emoji = "🟢"
    elif pct >= 75:
        text = "Performance significativamente sopra la mediana storica"
        color = COLORS["zscore_pos"]
        emoji = "🟢"
    elif pct >= 50:
        text = "Performance nella metà superiore della distribuzione storica"
        color = "lightgreen"
        emoji = "🔵"
    elif pct >= 25:
        text = "Performance nella metà inferiore, ma entro la normalità (IQR)"
        color = COLORS["regime_sideways"]
        emoji = "🟡"
    elif pct >= 10:
        text = "Performance significativamente sotto la mediana storica"
        color = COLORS["zscore_neg"]
        emoji = "🔴"
    else:
        text = "Anomalia ESTREMA negativa - performance eccezionalmente inferiore alla norma storica"
        color = COLORS["zscore_neg"]
        emoji = "🔴"
    
    if zscore is not None and not pd.isna(zscore):
        if abs(zscore) > 2.5:
            text += f" | Z-Score: {zscore:.2f}σ (MOLTO significativo)"
        elif abs(zscore) > 2:
            text += f" | Z-Score: {zscore:.2f}σ (significativo)"
        elif abs(zscore) > 1.5:
            text += f" | Z-Score: {zscore:.2f}σ (moderato)"
    
    return text, color, emoji


# =============================================================================
# 13. CHART BUILDERS
# =============================================================================
def build_main_percentile_chart(pivot: pd.DataFrame, perc: pd.DataFrame, 
                                 current_year: int, ticker: str,
                                 bootstrap_ci: dict = None) -> go.Figure:
    """Grafico principale con bande percentile e CI bootstrap."""
    fig = go.Figure()
    labels = doy_to_label(perc.index)

    # Bootstrap CI per banda 95 (se disponibile)
    if bootstrap_ci:
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=bootstrap_ci["p95_ci_upper"].tolist() + bootstrap_ci["p95_ci_lower"].tolist()[::-1],
            fill="toself",
            fillcolor=COLORS["ci_band"],
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI (Bootstrap)",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Banda 5°-95°
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=perc["p95"].tolist() + perc["p5"].tolist()[::-1],
        fill="toself",
        fillcolor=COLORS["band_95"],
        line=dict(color="rgba(0,0,0,0)"),
        name="5° - 95° Pct",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Banda IQR
    fig.add_trace(go.Scatter(
        x=labels + labels[::-1],
        y=perc["p75"].tolist() + perc["p25"].tolist()[::-1],
        fill="toself",
        fillcolor=COLORS["band_iqr"],
        line=dict(color="rgba(0,0,0,0)"),
        name="25° - 75° Pct (IQR)",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Mediana
    fig.add_trace(go.Scatter(
        x=labels,
        y=perc["p50"].tolist(),
        mode="lines",
        line=dict(color=COLORS["median"], width=1.5, dash="dash"),
        name="Mediana (50° Pct)",
        showlegend=True,
    ))

    # Equity YTD corrente
    serie_corrente = pivot.get(current_year)
    if serie_corrente is not None:
        ultimo_doy_valido = serie_corrente.last_valid_index()
        serie_plot = serie_corrente.loc[:ultimo_doy_valido]
        labels_ytd = doy_to_label(serie_plot.index)

        fig.add_trace(go.Scatter(
            x=labels_ytd,
            y=serie_plot.values,
            mode="lines",
            line=dict(color=COLORS["ytd"], width=3),
            name=f"YTD {current_year}",
            showlegend=True,
        ))

        # Marker ultimo punto
        ultimo_val = serie_plot.iloc[-1]
        ultimo_label = labels_ytd[-1]
        segno = "+" if ultimo_val >= 0 else ""

        fig.add_trace(go.Scatter(
            x=[ultimo_label],
            y=[ultimo_val],
            mode="markers+text",
            marker=dict(color=COLORS["ytd"], size=10, symbol="circle"),
            text=[f"{segno}{ultimo_val:.2f}%"],
            textposition="top right",
            textfont=dict(color=COLORS["ytd"], size=13, family="Arial Black"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        xaxis=dict(
            title="Giorno dell'Anno",
            showgrid=True,
            gridcolor=COLORS["grid"],
            tickangle=-45,
        ),
        yaxis=dict(
            title="Rendimento YTD (%)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.25)",
            ticksuffix="%",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=40, t=40, b=80),
        height=500,
    )

    return fig


def build_zscore_chart(zscore_series: pd.Series, vol_context: pd.DataFrame, 
                       current_year: int) -> go.Figure:
    """Grafico Z-Score con contesto volatilità."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Z-Score YTD vs Storico", "Volatilità Contestuale"),
        row_heights=[0.6, 0.4],
    )
    
    zscore_clean = zscore_series.dropna()
    labels = doy_to_label(zscore_clean.index)
    
    # Colori condizionali per Z-score
    colors = [COLORS["zscore_pos"] if z >= 0 else COLORS["zscore_neg"] for z in zscore_clean.values]
    
    fig.add_trace(go.Bar(
        x=labels,
        y=zscore_clean.values,
        marker_color=colors,
        name="Z-Score",
        showlegend=False,
    ), row=1, col=1)
    
    # Linee di riferimento ±1σ, ±2σ
    for sigma, dash in [(2, "solid"), (1, "dash"), (-1, "dash"), (-2, "solid")]:
        fig.add_hline(
            y=sigma, 
            line_dash=dash, 
            line_color="rgba(255,255,255,0.3)",
            annotation_text=f"{sigma}σ" if sigma > 0 else f"{sigma}σ",
            annotation_position="right",
            row=1, col=1,
        )
    
    # Volatilità contestuale
    if not vol_context.empty:
        vol_clean = vol_context.dropna()
        labels_vol = doy_to_label(vol_clean.index)
        
        fig.add_trace(go.Scatter(
            x=labels_vol,
            y=vol_clean["vol_corrente"],
            mode="lines",
            line=dict(color=COLORS["ytd"], width=2),
            name=f"Vol {current_year}",
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=labels_vol,
            y=vol_clean["vol_storica_mean"],
            mode="lines",
            line=dict(color=COLORS["median"], width=1.5, dash="dash"),
            name="Vol Media Storica",
        ), row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    fig.update_yaxes(title_text="Z-Score (σ)", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Volatilità", row=2, col=1, gridcolor=COLORS["grid"])
    
    return fig


def build_dynamics_chart(dynamics_df: pd.DataFrame, persistence_data: dict, 
                         current_year: int) -> go.Figure:
    """Grafico dinamiche anomalia: percentile, velocity, acceleration."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Percentile Rolling",
            "Velocità (Δ Percentile)",
            "Accelerazione (ΔΔ Percentile)",
        ),
        row_heights=[0.4, 0.3, 0.3],
    )
    
    dynamics_clean = dynamics_df.dropna()
    labels = doy_to_label(dynamics_clean.index)
    
    # Percentile con zone colorate
    fig.add_trace(go.Scatter(
        x=labels,
        y=dynamics_clean["percentile"],
        mode="lines",
        line=dict(color=COLORS["persistence"], width=2),
        name="Percentile",
        fill="tozeroy",
        fillcolor="rgba(76, 201, 240, 0.2)",
    ), row=1, col=1)
    
    # Zone IQR
    fig.add_hrect(y0=25, y1=75, fillcolor="rgba(100,149,237,0.1)", 
                  line_width=0, row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1)
    
    # Velocity
    vel_colors = [COLORS["zscore_pos"] if v >= 0 else COLORS["zscore_neg"] 
                  for v in dynamics_clean["velocity"].values]
    fig.add_trace(go.Bar(
        x=labels,
        y=dynamics_clean["velocity"],
        marker_color=vel_colors,
        name="Velocity",
        showlegend=False,
    ), row=2, col=1)
    
    # Acceleration
    acc_colors = [COLORS["velocity"] if a >= 0 else COLORS["acceleration"] 
                  for a in dynamics_clean["acceleration"].values]
    fig.add_trace(go.Bar(
        x=labels,
        y=dynamics_clean["acceleration"],
        marker_color=acc_colors,
        name="Acceleration",
        showlegend=False,
    ), row=3, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        height=650,
        showlegend=False,
    )
    
    fig.update_yaxes(title_text="Pct", row=1, col=1, gridcolor=COLORS["grid"], range=[0, 100])
    fig.update_yaxes(title_text="Δ Pct", row=2, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="ΔΔ Pct", row=3, col=1, gridcolor=COLORS["grid"])
    
    return fig


def build_regime_chart(pivot: pd.DataFrame, cluster_df: pd.DataFrame, 
                       current_year: int, current_regime: str) -> go.Figure:
    """Visualizzazione cluster anni e traiettorie per regime."""
    if cluster_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti per clustering", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Scatter Anni per Regime", "Traiettorie YTD per Regime"),
        column_widths=[0.4, 0.6],
    )
    
    # Scatter plot anni
    regime_colors = {"Bull": COLORS["regime_bull"], "Bear": COLORS["regime_bear"], 
                     "Sideways": COLORS["regime_sideways"]}
    
    for regime in ["Bull", "Bear", "Sideways"]:
        regime_data = cluster_df[cluster_df["regime"] == regime]
        if len(regime_data) > 0:
            fig.add_trace(go.Scatter(
                x=regime_data["final_ret"],
                y=regime_data["path_vol"],
                mode="markers+text",
                marker=dict(color=regime_colors[regime], size=12),
                text=regime_data.index.astype(str),
                textposition="top center",
                textfont=dict(size=9),
                name=regime,
            ), row=1, col=1)
    
    # Traiettorie YTD per regime
    storico = pivot.drop(columns=[current_year], errors="ignore")
    labels = doy_to_label(pivot.index)
    
    for regime in ["Bull", "Bear", "Sideways"]:
        regime_years = cluster_df[cluster_df["regime"] == regime].index.tolist()
        regime_years = [y for y in regime_years if y in storico.columns]
        
        if len(regime_years) > 0:
            for i, anno in enumerate(regime_years):
                serie = storico[anno].dropna()
                labels_anno = doy_to_label(serie.index)
                fig.add_trace(go.Scatter(
                    x=labels_anno,
                    y=serie.values,
                    mode="lines",
                    line=dict(color=regime_colors[regime], width=1),
                    opacity=0.4,
                    name=regime if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup=regime,
                ), row=1, col=2)
    
    # Anno corrente
    serie_corrente = pivot.get(current_year)
    if serie_corrente is not None:
        serie_corrente = serie_corrente.dropna()
        labels_curr = doy_to_label(serie_corrente.index)
        fig.add_trace(go.Scatter(
            x=labels_curr,
            y=serie_corrente.values,
            mode="lines",
            line=dict(color="white", width=3),
            name=f"{current_year} (current)",
        ), row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        height=450,
    )
    
    fig.update_xaxes(title_text="Rendimento Finale (%)", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="Volatilità Path", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_xaxes(title_text="Giorno dell'Anno", row=1, col=2, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="YTD %", row=1, col=2, gridcolor=COLORS["grid"])
    
    return fig


def build_forward_returns_chart(forward_data: dict) -> go.Figure:
    """Istogramma distribuzione rendimenti forward."""
    if not forward_data or "forward_returns" not in forward_data:
        fig = go.Figure()
        fig.add_annotation(text="Dati insufficienti per analisi forward", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fwd_rets = forward_data["forward_returns"]
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=fwd_rets,
        nbinsx=20,
        marker_color=COLORS["persistence"],
        opacity=0.7,
        name="Forward Returns",
    ))
    
    # Linea media
    fig.add_vline(x=forward_data["mean_forward"], 
                  line_dash="dash", line_color=COLORS["ytd"],
                  annotation_text=f"Media: {forward_data['mean_forward']:.2f}%")
    
    # Linea zero
    fig.add_vline(x=0, line_dash="solid", line_color="white", opacity=0.5)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        xaxis_title=f"Rendimento Forward ({forward_data['lookahead_days']} giorni) %",
        yaxis_title="Frequenza",
        height=400,
        showlegend=False,
    )
    
    return fig


# =============================================================================
# 14. MAIN UI
# =============================================================================
def main():
    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## 🔬 Kriterion Quant")
        st.markdown("**YTD Anomaly Detection**")
        st.markdown("---")
        
        st.header("⚙️ Parametri")

        ticker = st.text_input(
            label="Ticker (formato EODHD)",
            value="SPY.US",
            placeholder="es. SPY.US, BTC-USD.CC",
            help="Formato: SIMBOLO.EXCHANGE",
        ).strip().upper()

        start_date = st.date_input(
            label="Inizio storico",
            value=date(2000, 1, 1),
            min_value=date(1990, 1, 1),
            max_value=date.today(),
        )
        
        st.markdown("---")
        st.subheader("🎛️ Parametri Avanzati")
        
        lookahead_days = st.slider(
            "Forward Lookahead (giorni)",
            min_value=5,
            max_value=60,
            value=20,
            help="Periodo per analisi mean reversion",
        )
        
        pct_tolerance = st.slider(
            "Tolleranza Percentile (%)",
            min_value=5,
            max_value=25,
            value=10,
            help="Range per matching anni simili",
        )
        
        n_bootstrap = st.select_slider(
            "Bootstrap Samples",
            options=[100, 250, 500, 1000],
            value=500,
        )
        
        st.markdown("---")
        st.caption("📡 Dati: [EODHD API](https://eodhd.com)")
        st.caption("🔬 Kriterion Quant © 2025")

    # ---- Header ----
    current_year = date.today().year
    st.markdown(f"# 📊 Anomaly Detection Dashboard: `{ticker}`")
    
    col_info, col_btn = st.columns([4, 1])
    with col_info:
        st.markdown(
            f"Analisi multi-dimensionale delle anomalie YTD **{current_year}** "
            f"vs distribuzione storica (dal **{start_date.strftime('%d/%m/%Y')}**)."
        )
    with col_btn:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # ---- Fetch & Validate Data ----
    with st.spinner(f"Scaricamento dati {ticker}..."):
        df = fetch_ohlcv(ticker, start_date.strftime("%Y-%m-%d"))

    if df.empty:
        st.error(f"❌ Nessun dato disponibile per **{ticker}**. Verifica il simbolo.")
        st.stop()

    # ---- Core Computations ----
    with st.spinner("Elaborazione dati..."):
        pivot = compute_ytd_by_doy(df)
        anni_disponibili = sorted(pivot.columns.tolist())
        
        if len(anni_disponibili) < 3:
            st.error("❌ Storico insufficiente: servono almeno 3 anni di dati.")
            st.stop()
        
        perc = compute_percentiles(pivot, current_year)
        pct_attuale = compute_current_percentile(pivot, current_year)
        zscore_series = compute_zscore_by_doy(pivot, current_year)
        vol_context = compute_rolling_volatility_context(pivot, current_year)
        dynamics = compute_percentile_dynamics(pivot, current_year)
        persistence = compute_anomaly_persistence(pivot, perc, current_year)
        cluster_df = cluster_historical_years(pivot, current_year)
        current_regime = identify_current_regime(pivot, current_year, cluster_df)
        regime_perc = compute_regime_conditional_percentiles(pivot, current_year, cluster_df, current_regime)
        forward_data = compute_forward_return_distribution(
            pivot, current_year, lookahead_days=lookahead_days, pct_tolerance=pct_tolerance
        )
        bootstrap_ci = bootstrap_percentile_bands(pivot, current_year, n_bootstrap=n_bootstrap)
    
    # ---- Quick Stats Header ----
    serie_corrente = pivot.get(current_year)
    ytd_val = serie_corrente.dropna().iloc[-1] if serie_corrente is not None else np.nan
    zscore_current = zscore_series.dropna().iloc[-1] if len(zscore_series.dropna()) > 0 else np.nan
    
    interpretation, color, emoji = get_anomaly_interpretation(pct_attuale, zscore_current)
    
    st.markdown("---")
    cols = st.columns(5)
    
    with cols[0]:
        segno = "+" if (not pd.isna(ytd_val) and ytd_val >= 0) else ""
        st.metric(f"YTD {current_year}", f"{segno}{ytd_val:.2f}%" if not pd.isna(ytd_val) else "N/D")
    
    with cols[1]:
        st.metric("Percentile", f"{pct_attuale:.1f}°" if not pd.isna(pct_attuale) else "N/D")
    
    with cols[2]:
        st.metric("Z-Score", f"{zscore_current:.2f}σ" if not pd.isna(zscore_current) else "N/D")
    
    with cols[3]:
        st.metric("Regime", current_regime)
    
    with cols[4]:
        st.metric("Streak Fuori IQR", f"{persistence['current_streak']} giorni")
    
    st.info(f"{emoji} **{interpretation}**")
    
    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview Percentili",
        "📊 Z-Score & Volatilità",
        "⚡ Dinamiche Anomalia",
        "🎯 Analisi Regime",
        "🔮 Forward Returns",
        "🔍 Multi-Asset Scanner",
    ])
    
    # ========== TAB 1: OVERVIEW PERCENTILI ==========
    with tab1:
        st.markdown("### 📈 Analisi Percentile YTD")
        
        st.markdown("""
        <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <b>📖 Come leggere questo grafico:</b><br>
        Le <b>bande colorate</b> rappresentano la distribuzione storica dei rendimenti YTD per ogni giorno dell'anno.
        La <b>linea rossa</b> è la performance dell'anno corrente. Quando esce dalle bande, indica un'<b>anomalia statistica</b>.
        <ul>
        <li><b>Banda chiara (5°-95°)</b>: range "normale" - il 90% degli anni storici cade qui</li>
        <li><b>Banda scura (25°-75°)</b>: range IQR - la "zona di comfort" del 50% centrale</li>
        <li><b>Linea tratteggiata</b>: mediana storica (50° percentile)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        fig_main = build_main_percentile_chart(pivot, perc, current_year, ticker, bootstrap_ci)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Insight contestuali
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistiche Distribuzione")
            anni_storico = [a for a in anni_disponibili if a != current_year]
            
            stats_data = {
                "Metrica": ["Anni in analisi", "Mediana storica (oggi)", "IQR Range", "5°-95° Range"],
                "Valore": [
                    f"{len(anni_storico)} ({min(anni_storico)}-{max(anni_storico)})",
                    f"{perc['p50'].dropna().iloc[-1]:.2f}%" if len(perc['p50'].dropna()) > 0 else "N/D",
                    f"{perc['p25'].dropna().iloc[-1]:.2f}% → {perc['p75'].dropna().iloc[-1]:.2f}%" if len(perc['p25'].dropna()) > 0 else "N/D",
                    f"{perc['p5'].dropna().iloc[-1]:.2f}% → {perc['p95'].dropna().iloc[-1]:.2f}%" if len(perc['p5'].dropna()) > 0 else "N/D",
                ],
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 🎯 Interpretazione")
            
            if not pd.isna(pct_attuale):
                if pct_attuale < 10 or pct_attuale > 90:
                    st.warning(f"""
                    ⚠️ **ANOMALIA SIGNIFICATIVA RILEVATA**
                    
                    L'asset si trova al **{pct_attuale:.1f}° percentile**, 
                    una posizione raggiunta storicamente solo nel **{min(pct_attuale, 100-pct_attuale):.1f}%** degli anni.
                    
                    Questo può indicare:
                    - Evento macro straordinario
                    - Cambio strutturale nel comportamento dell'asset
                    - Opportunità di mean reversion (da validare)
                    """)
                elif pct_attuale < 25 or pct_attuale > 75:
                    st.info(f"""
                    📍 **Posizione moderatamente anomala**
                    
                    Al **{pct_attuale:.1f}° percentile**, l'asset è fuori dall'IQR ma 
                    non ancora in territorio estremo. Monitorare l'evoluzione.
                    """)
                else:
                    st.success(f"""
                    ✅ **Performance nella norma storica**
                    
                    Al **{pct_attuale:.1f}° percentile**, l'asset si comporta 
                    in linea con le aspettative stagionali.
                    """)
        
        # Bootstrap CI insight
        if bootstrap_ci:
            with st.expander("📐 Dettagli Bootstrap Confidence Intervals"):
                st.markdown(f"""
                I **Confidence Intervals** (area gialla trasparente) mostrano l'incertezza 
                nella stima delle bande percentile, calcolata con **{n_bootstrap} resampling bootstrap**.
                
                Con {len(anni_storico)} anni di storia:
                - **CI ampio** → alta incertezza, le bande potrebbero essere diverse con più dati
                - **CI stretto** → le stime sono robuste
                
                ⚠️ Un'anomalia apparente potrebbe essere un artefatto di campionamento se cade 
                dentro il CI delle bande.
                """)
    
    # ========== TAB 2: Z-SCORE & VOLATILITÀ ==========
    with tab2:
        st.markdown("### 📊 Analisi Z-Score e Contesto Volatilità")
        
        st.markdown("""
        <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <b>📖 Perché lo Z-Score?</b><br>
        Il percentile indica <i>dove</i> sei nella distribuzione, lo Z-Score indica <i>quanto</i> sei lontano dalla media in termini di deviazioni standard.
        <ul>
        <li><b>|Z| > 2</b>: anomalia statisticamente significativa (p < 0.05)</li>
        <li><b>|Z| > 2.5</b>: anomalia molto significativa (p < 0.01)</li>
        <li><b>|Z| > 3</b>: evento raro, circa 0.3% delle osservazioni</li>
        </ul>
        Il pannello inferiore mostra se la <b>volatilità corrente</b> è alta/bassa rispetto alla media storica per questo periodo.
        </div>
        """, unsafe_allow_html=True)
        
        fig_zscore = build_zscore_chart(zscore_series, vol_context, current_year)
        st.plotly_chart(fig_zscore, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Z-Score Attuale")
            
            if not pd.isna(zscore_current):
                # Calcola p-value approssimativo
                p_value = 2 * (1 - stats.norm.cdf(abs(zscore_current)))
                
                st.metric("Z-Score", f"{zscore_current:.3f}σ")
                st.metric("P-Value (two-tailed)", f"{p_value:.4f}")
                
                if abs(zscore_current) > 2.5:
                    st.error("🔴 Anomalia MOLTO significativa - evento raro")
                elif abs(zscore_current) > 2:
                    st.warning("🟠 Anomalia significativa - sotto soglia 5%")
                elif abs(zscore_current) > 1.5:
                    st.info("🟡 Deviazione moderata - da monitorare")
                else:
                    st.success("🟢 Nella norma statistica")
        
        with col2:
            st.markdown("#### 🌊 Contesto Volatilità")
            
            if not vol_context.empty:
                vol_zscore_current = vol_context["vol_zscore"].dropna().iloc[-1] if len(vol_context["vol_zscore"].dropna()) > 0 else np.nan
                
                if not pd.isna(vol_zscore_current):
                    st.metric("Vol Z-Score", f"{vol_zscore_current:.2f}σ")
                    
                    if vol_zscore_current > 1.5:
                        st.warning("""
                        ⚠️ **Volatilità ELEVATA**
                        
                        La dispersione dei rendimenti è superiore alla norma storica.
                        Le anomalie in questo contesto potrebbero essere più "rumorose".
                        """)
                    elif vol_zscore_current < -1.5:
                        st.info("""
                        📉 **Volatilità BASSA**
                        
                        Mercato insolitamente calmo. Le anomalie in bassa volatilità 
                        tendono ad essere più "genuine" e meno dovute al rumore.
                        """)
                    else:
                        st.success("✅ Volatilità in linea con la media storica")
    
    # ========== TAB 3: DINAMICHE ANOMALIA ==========
    with tab3:
        st.markdown("### ⚡ Dinamiche dell'Anomalia")
        
        st.markdown("""
        <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <b>📖 Velocity & Acceleration:</b><br>
        Non basta sapere <i>dove</i> sei, conta anche <i>come</i> ci sei arrivato e <i>verso dove</i> stai andando.
        <ul>
        <li><b>Velocity</b>: quanto velocemente sta cambiando il tuo ranking percentile</li>
        <li><b>Acceleration</b>: la velocità sta aumentando o diminuendo?</li>
        <li><b>Persistenza</b>: da quanti giorni sei fuori dalla zona normale?</li>
        </ul>
        Un'anomalia con <b>velocity negativa in accelerazione</b> mentre sei già sotto il 25° percentile è un segnale di stress significativo.
        </div>
        """, unsafe_allow_html=True)
        
        if not dynamics.empty:
            fig_dynamics = build_dynamics_chart(dynamics, persistence, current_year)
            st.plotly_chart(fig_dynamics, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏃 Velocity")
            velocity_current = dynamics["velocity"].dropna().iloc[-1] if len(dynamics["velocity"].dropna()) > 0 else np.nan
            
            if not pd.isna(velocity_current):
                direction = "📈 Miglioramento" if velocity_current > 0 else "📉 Peggioramento"
                st.metric("Δ Percentile (5gg)", f"{velocity_current:+.1f}")
                st.markdown(f"**Trend:** {direction}")
        
        with col2:
            st.markdown("#### 🚀 Acceleration")
            acc_current = dynamics["acceleration"].dropna().iloc[-1] if len(dynamics["acceleration"].dropna()) > 0 else np.nan
            
            if not pd.isna(acc_current):
                if acc_current > 2:
                    st.success("⬆️ Momentum positivo in aumento")
                elif acc_current < -2:
                    st.error("⬇️ Momentum negativo in aumento")
                else:
                    st.info("➡️ Momentum stabile")
                st.metric("ΔΔ Percentile", f"{acc_current:+.2f}")
        
        with col3:
            st.markdown("#### ⏱️ Persistenza")
            st.metric("Streak Corrente", f"{persistence['current_streak']} giorni")
            st.metric("Max Streak (anno)", f"{persistence['max_streak']} giorni")
            st.metric("% Giorni Fuori IQR", f"{persistence['pct_days_outside']:.1f}%")
            
            if persistence['current_streak'] > 10:
                st.warning("⚠️ Anomalia persistente - potenziale cambio regime")
        
        # Insight combinato
        st.markdown("---")
        st.markdown("#### 🧠 Diagnosi Combinata")
        
        if not pd.isna(velocity_current) and not pd.isna(acc_current):
            if pct_attuale < 25 and velocity_current < -2 and acc_current < 0:
                st.error("""
                🚨 **SEGNALE DI STRESS CRITICO**
                
                L'asset è sotto il 25° percentile con velocità di deterioramento in aumento.
                Pattern tipico di capitolazione o crisi in corso.
                """)
            elif pct_attuale > 75 and velocity_current > 2 and acc_current > 0:
                st.warning("""
                ⚠️ **POSSIBILE EUFORIA / BLOW-OFF TOP**
                
                L'asset è sopra il 75° percentile con accelerazione positiva.
                Pattern spesso associato a eccessi speculativi.
                """)
            elif abs(velocity_current) < 1 and persistence['current_streak'] > 5:
                st.info("""
                📊 **CONSOLIDAMENTO IN ANOMALIA**
                
                L'asset è stabilizzato in territorio anomalo. 
                Potrebbe indicare un nuovo equilibrio o preparazione a un movimento.
                """)
            else:
                st.success("✅ Dinamiche nella norma - nessun pattern critico rilevato")
    
    # ========== TAB 4: ANALISI REGIME ==========
    with tab4:
        st.markdown("### 🎯 Analisi Regime & Clustering")
        
        st.markdown("""
        <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <b>📖 Perché i Regimi?</b><br>
        Non tutti gli anni sono uguali. Confrontare un anno bull con la media di anni misti può essere fuorviante.
        Il clustering raggruppa gli anni storici in <b>3 regimi</b>:
        <ul>
        <li>🟢 <b>Bull</b>: anni con rendimenti finali elevati</li>
        <li>🔴 <b>Bear</b>: anni con rendimenti finali negativi</li>
        <li>🟡 <b>Sideways</b>: anni con rendimenti moderati/laterali</li>
        </ul>
        I <b>percentili regime-conditional</b> confrontano l'anno corrente solo con anni dello stesso tipo.
        </div>
        """, unsafe_allow_html=True)
        
        if not cluster_df.empty:
            fig_regime = build_regime_chart(pivot, cluster_df, current_year, current_regime)
            st.plotly_chart(fig_regime, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏷️ Classificazione Anni")
                
                regime_summary = cluster_df.groupby("regime").agg({
                    "final_ret": ["count", "mean", "std"],
                    "max_dd": "mean",
                }).round(2)
                regime_summary.columns = ["N Anni", "Ret Medio %", "Ret Std", "DD Medio %"]
                st.dataframe(regime_summary, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Regime Corrente")
                
                regime_colors_map = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}
                regime_emoji = regime_colors_map.get(current_regime, "⚪")
                
                st.metric("Regime Identificato", f"{regime_emoji} {current_regime}")
                
                if current_regime in ["Bull", "Bear", "Sideways"]:
                    same_regime_years = cluster_df[cluster_df["regime"] == current_regime].index.tolist()
                    st.caption(f"Anni simili: {', '.join(map(str, same_regime_years))}")
            
            # Percentili regime-conditional
            if not regime_perc.empty:
                st.markdown("---")
                st.markdown("#### 📊 Percentili Regime-Conditional")
                
                # Calcola percentile condizionale
                serie_corrente = pivot.get(current_year)
                if serie_corrente is not None:
                    ultimo_doy = serie_corrente.dropna().index[-1]
                    val_corrente = serie_corrente.loc[ultimo_doy]
                    
                    same_regime_years = cluster_df[cluster_df["regime"] == current_regime].index.tolist()
                    same_regime_years = [y for y in same_regime_years if y != current_year and y in pivot.columns]
                    
                    if len(same_regime_years) >= 3:
                        storico_regime = pivot[same_regime_years]
                        vals_regime = storico_regime.loc[ultimo_doy].dropna()
                        pct_conditional = (vals_regime < val_corrente).sum() / len(vals_regime) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Percentile (tutti gli anni)", f"{pct_attuale:.1f}°")
                        with col2:
                            st.metric(f"Percentile (solo {current_regime})", f"{pct_conditional:.1f}°")
                        
                        diff = pct_conditional - pct_attuale
                        if abs(diff) > 15:
                            st.warning(f"""
                            ⚠️ **Differenza significativa ({diff:+.1f} punti)**
                            
                            Rispetto a tutti gli anni, sembri al {pct_attuale:.1f}° percentile.
                            Ma rispetto solo agli anni {current_regime}, sei al {pct_conditional:.1f}° percentile.
                            
                            Questo suggerisce che il contesto di regime è importante per interpretare l'anomalia.
                            """)
        else:
            st.warning("⚠️ Dati insufficienti per l'analisi regime (servono almeno 5 anni)")
    
    # ========== TAB 5: FORWARD RETURNS ==========
    with tab5:
        st.markdown("### 🔮 Analisi Forward Returns & Mean Reversion")
        
        st.markdown(f"""
        <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <b>📖 Probabilità Condizionale:</b><br>
        Storicamente, quando l'asset era in una posizione percentile simile (±{pct_tolerance}%) 
        nello stesso periodo dell'anno, cosa è successo nei {lookahead_days} giorni successivi?
        <br><br>
        ⚠️ <b>Attenzione:</b> Questa è un'analisi empirica, non una previsione. 
        I campioni potrebbero essere limitati e le condizioni di mercato diverse.
        </div>
        """, unsafe_allow_html=True)
        
        if forward_data:
            fig_forward = build_forward_returns_chart(forward_data)
            st.plotly_chart(fig_forward, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Campioni Trovati", forward_data["n_samples"])
                st.metric("📍 Percentile Corrente", f"{forward_data['current_percentile']:.1f}°")
            
            with col2:
                st.metric("📈 Media Forward", f"{forward_data['mean_forward']:+.2f}%")
                st.metric("📊 Mediana Forward", f"{forward_data['median_forward']:+.2f}%")
            
            with col3:
                st.metric("✅ P(Rendimento > 0)", f"{forward_data['prob_positive']:.1f}%")
                st.metric("📉 Std Forward", f"{forward_data['std_forward']:.2f}%")
            
            # Interpretazione
            st.markdown("---")
            st.markdown("#### 🧠 Interpretazione")
            
            if forward_data["n_samples"] < 5:
                st.warning("""
                ⚠️ **Campione limitato**
                
                Con meno di 5 osservazioni storiche, le statistiche non sono affidabili.
                Usare con estrema cautela.
                """)
            else:
                if forward_data["prob_positive"] > 65:
                    st.success(f"""
                    📈 **Bias storico POSITIVO**
                    
                    In {forward_data['n_samples']} casi simili, il {forward_data['prob_positive']:.1f}% 
                    delle volte il rendimento forward a {lookahead_days} giorni è stato positivo.
                    
                    Media: {forward_data['mean_forward']:+.2f}% | Mediana: {forward_data['median_forward']:+.2f}%
                    """)
                elif forward_data["prob_positive"] < 35:
                    st.error(f"""
                    📉 **Bias storico NEGATIVO**
                    
                    In {forward_data['n_samples']} casi simili, solo il {forward_data['prob_positive']:.1f}% 
                    delle volte il rendimento forward è stato positivo.
                    
                    Media: {forward_data['mean_forward']:+.2f}% | Mediana: {forward_data['median_forward']:+.2f}%
                    """)
                else:
                    st.info(f"""
                    ➡️ **Nessun bias storico chiaro**
                    
                    La distribuzione è relativamente bilanciata ({forward_data['prob_positive']:.1f}% positivi).
                    Il forward return medio è {forward_data['mean_forward']:+.2f}%.
                    """)
            
            # Anni matching
            with st.expander("📅 Anni con pattern simile"):
                st.write(f"Anni trovati con percentile ±{pct_tolerance}% al DOY corrente:")
                st.write(", ".join(map(str, forward_data["matching_years"])))
        else:
            st.warning("⚠️ Nessun dato storico comparabile trovato. Prova ad aumentare la tolleranza percentile.")
    
    # ========== TAB 6: MULTI-ASSET SCANNER ==========
    with tab6:
        st.markdown("### 🔍 Multi-Asset Anomaly Scanner")
        
        st.markdown("""
        <div style="background-color: rgba(100,149,237,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <b>📖 Scanner Universo:</b><br>
        Scansiona multipli asset per identificare quelli in anomalia estrema (code della distribuzione).
        Utile per:
        <ul>
        <li>Identificare opportunità di mean reversion cross-asset</li>
        <li>Monitorare la "salute" di un portafoglio</li>
        <li>Screening per strategie rotazionali</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            default_tickers = "SPY.US, QQQ.US, IWM.US, EFA.US, EEM.US, TLT.US, GLD.US, USO.US, UUP.US, VNQ.US"
            tickers_input = st.text_area(
                "Inserisci ticker (separati da virgola)",
                value=default_tickers,
                height=100,
            )
        
        with col2:
            threshold = st.slider(
                "Soglia Anomalia (%)",
                min_value=5,
                max_value=25,
                value=15,
                help="Percentile sotto/sopra il quale considerare anomalia",
            )
            
            scan_start = st.date_input(
                "Inizio storico scan",
                value=date(2005, 1, 1),
            )
        
        if st.button("🚀 Avvia Scansione", type="primary"):
            tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            
            if len(tickers_list) > 0:
                results_df = scan_universe_for_anomalies(
                    tickers_list, 
                    scan_start.strftime("%Y-%m-%d"),
                    threshold_pct=threshold,
                )
                
                if not results_df.empty:
                    st.markdown("#### 📊 Risultati Scansione")
                    
                    # Evidenzia anomalie
                    def highlight_anomaly(row):
                        if "Inferiore" in str(row["Anomaly"]):
                            return ["background-color: rgba(255,107,107,0.3)"] * len(row)
                        elif "Superiore" in str(row["Anomaly"]):
                            return ["background-color: rgba(0,210,106,0.3)"] * len(row)
                        return [""] * len(row)
                    
                    styled_df = results_df.style.apply(highlight_anomaly, axis=1)
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Summary
                    n_lower = len(results_df[results_df["Anomaly"].str.contains("Inferiore")])
                    n_upper = len(results_df[results_df["Anomaly"].str.contains("Superiore")])
                    n_normal = len(results_df) - n_lower - n_upper
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔴 Anomalie Negative", n_lower)
                    with col2:
                        st.metric("🟢 Anomalie Positive", n_upper)
                    with col3:
                        st.metric("⚪ Normali", n_normal)
                else:
                    st.warning("Nessun risultato trovato. Verifica i ticker.")
            else:
                st.warning("Inserisci almeno un ticker.")
    
    # ---- Footer ----
    st.markdown("---")
    st.caption("🔬 **Kriterion Quant** — YTD Anomaly Detection Dashboard | Dati: EODHD API")
    st.caption("⚠️ Questo strumento è solo a scopo educativo e di ricerca. Non costituisce consulenza finanziaria.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

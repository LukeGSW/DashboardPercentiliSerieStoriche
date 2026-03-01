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

NOTA: Utilizza Trading Day Index (TDI) invece di Day-of-Year per evitare
      disallineamenti da anni bisestili.
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

# Numero tipico di trading days in un anno (US market)
MAX_TRADING_DAYS = 253

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
# 2. CALCOLO YTD CON TRADING DAY INDEX (TDI)
# =============================================================================
def compute_ytd_by_trading_day(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Calcola i rendimenti YTD per ogni anno solare mappati su Trading Day Index (TDI).
    
    A differenza del DOY, il TDI conta solo i giorni di trading effettivi (1, 2, 3, ...)
    eliminando i problemi di disallineamento da anni bisestili e festività variabili.
    
    Returns:
        - pivot_ytd: DataFrame TDI × anni con rendimenti YTD %
        - pivot_returns: DataFrame TDI × anni con rendimenti giornalieri %
        - metadata: dict con informazioni su ultimo TDI valido per anno
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    
    anni = sorted(df["year"].unique())
    
    ytd_dict = {}
    returns_dict = {}
    metadata = {"last_valid_tdi": {}, "tdi_to_date": {}}
    
    for anno in anni:
        # Prezzo base: ultima chiusura dell'anno precedente
        anno_prec = anno - 1
        df_prec = df[df["year"] == anno_prec]
        
        if df_prec.empty:
            # Primo anno: usa primo prezzo come base
            base_price = df[df["year"] == anno]["adjusted_close"].iloc[0]
        else:
            base_price = df_prec["adjusted_close"].iloc[-1]
        
        df_anno = df[df["year"] == anno].copy().reset_index(drop=True)
        
        # Trading Day Index: 1-based, conta solo giorni di trading effettivi
        df_anno["tdi"] = np.arange(1, len(df_anno) + 1)
        
        # Rendimento YTD cumulato
        df_anno["ytd_pct"] = (df_anno["adjusted_close"] / base_price - 1) * 100
        
        # Rendimenti giornalieri (per volatilità) - calcolati sui PREZZI, non su YTD
        df_anno["daily_return"] = df_anno["adjusted_close"].pct_change() * 100
        
        # Crea serie con TDI come indice
        max_tdi = df_anno["tdi"].max()
        
        # Serie YTD - riempita SOLO fino all'ultimo TDI reale, poi NaN
        serie_ytd = pd.Series(index=range(1, MAX_TRADING_DAYS + 1), dtype=float)
        serie_ytd.loc[df_anno["tdi"]] = df_anno["ytd_pct"].values
        # Forward fill SOLO fino all'ultimo trading day reale (per weekend/festività infrasettimanali)
        # Ma NON oltre l'ultimo giorno di trading dell'anno
        serie_ytd.loc[:max_tdi] = serie_ytd.loc[:max_tdi].ffill()
        # I giorni oltre max_tdi rimangono NaN (cruciale per anno corrente)
        
        # Serie returns giornalieri
        serie_returns = pd.Series(index=range(1, MAX_TRADING_DAYS + 1), dtype=float)
        serie_returns.loc[df_anno["tdi"]] = df_anno["daily_return"].values
        
        ytd_dict[anno] = serie_ytd
        returns_dict[anno] = serie_returns
        
        # Metadata
        metadata["last_valid_tdi"][anno] = max_tdi
        metadata["tdi_to_date"][anno] = dict(zip(df_anno["tdi"], df_anno["date"]))
    
    pivot_ytd = pd.DataFrame(ytd_dict)
    pivot_returns = pd.DataFrame(returns_dict)
    
    return pivot_ytd, pivot_returns, metadata


# =============================================================================
# 3. CALCOLO PERCENTILI STORICI
# =============================================================================
def compute_percentiles(pivot: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """
    Calcola percentili 5°, 25°, 50°, 75°, 95° escludendo l'anno corrente.
    Considera solo i TDI con almeno N anni di dati validi.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    
    # Richiedi almeno 3 anni di dati per ogni TDI
    min_years = 3
    valid_mask = storico.notna().sum(axis=1) >= min_years
    
    perc = pd.DataFrame(index=pivot.index)
    perc["p5"] = storico.quantile(0.05, axis=1).where(valid_mask)
    perc["p25"] = storico.quantile(0.25, axis=1).where(valid_mask)
    perc["p50"] = storico.quantile(0.50, axis=1).where(valid_mask)
    perc["p75"] = storico.quantile(0.75, axis=1).where(valid_mask)
    perc["p95"] = storico.quantile(0.95, axis=1).where(valid_mask)
    
    return perc


# =============================================================================
# 4. CALCOLO PERCENTILE CORRENTE (SENZA LOOKAHEAD BIAS)
# =============================================================================
def compute_current_percentile(pivot: pd.DataFrame, current_year: int, 
                                metadata: dict) -> tuple[float, int]:
    """
    Percentile dell'YTD corrente rispetto alla distribuzione storica.
    Usa l'ultimo TDI REALE dell'anno corrente, non il forward-filled.
    
    Returns:
        - percentile: valore 0-100
        - ultimo_tdi: ultimo trading day index con dati reali
    """
    serie_ytd_corrente = pivot.get(current_year)
    if serie_ytd_corrente is None:
        return np.nan, 0
    
    # Usa metadata per trovare l'ultimo TDI reale (non forward-filled)
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    if ultimo_tdi == 0:
        return np.nan, 0
    
    valore_corrente = serie_ytd_corrente.loc[ultimo_tdi]
    if pd.isna(valore_corrente):
        return np.nan, ultimo_tdi
    
    storico = pivot.drop(columns=[current_year], errors="ignore")
    
    # Prendi valori storici allo STESSO TDI
    valori_storici = storico.loc[ultimo_tdi].dropna().values
    
    if len(valori_storici) == 0:
        return np.nan, ultimo_tdi
    
    percentile = (np.sum(valori_storici < valore_corrente) / len(valori_storici)) * 100
    return round(percentile, 1), ultimo_tdi


# =============================================================================
# 5. Z-SCORE DINAMICO (CORRETTO)
# =============================================================================
def compute_zscore_by_tdi(pivot: pd.DataFrame, current_year: int, 
                          metadata: dict) -> pd.Series:
    """
    Z-score dell'YTD corrente vs distribuzione storica per ogni TDI.
    Considera SOLO fino all'ultimo TDI reale dell'anno corrente.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    serie_corrente = pivot.get(current_year)
    
    if serie_corrente is None:
        return pd.Series(dtype=float)
    
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    
    mu = storico.mean(axis=1)
    sigma = storico.std(axis=1)
    
    # Evita divisione per zero
    sigma = sigma.replace(0, np.nan)
    
    zscore = (serie_corrente - mu) / sigma
    
    # Tronca oltre l'ultimo TDI reale
    zscore.loc[ultimo_tdi + 1:] = np.nan
    
    return zscore


def compute_rolling_volatility_context(pivot_returns: pd.DataFrame, current_year: int,
                                        metadata: dict, window: int = 20) -> pd.DataFrame:
    """
    Calcola la volatilità rolling dell'anno corrente vs media storica.
    USA I RENDIMENTI GIORNALIERI (non il YTD cumulato).
    """
    storico = pivot_returns.drop(columns=[current_year], errors="ignore")
    serie_corrente = pivot_returns.get(current_year)
    
    if serie_corrente is None:
        return pd.DataFrame()
    
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    
    # Volatilità rolling anno corrente (su returns giornalieri veri)
    vol_corrente = serie_corrente.rolling(window=window, min_periods=5).std()
    
    # Volatilità media storica per TDI
    vol_storica_list = []
    for anno in storico.columns:
        vol_anno = storico[anno].rolling(window=window, min_periods=5).std()
        vol_storica_list.append(vol_anno)
    
    if len(vol_storica_list) == 0:
        return pd.DataFrame()
    
    vol_storica_df = pd.concat(vol_storica_list, axis=1)
    vol_storica_mean = vol_storica_df.mean(axis=1)
    vol_storica_std = vol_storica_df.std(axis=1)
    
    result = pd.DataFrame({
        "vol_corrente": vol_corrente,
        "vol_storica_mean": vol_storica_mean,
        "vol_storica_std": vol_storica_std,
        "vol_zscore": (vol_corrente - vol_storica_mean) / vol_storica_std.replace(0, np.nan),
    })
    
    # Tronca oltre l'ultimo TDI reale
    result.loc[ultimo_tdi + 1:] = np.nan
    
    return result


# =============================================================================
# 6. VELOCITÀ E ACCELERAZIONE ANOMALIA
# =============================================================================
def compute_percentile_dynamics(pivot: pd.DataFrame, current_year: int, 
                                 metadata: dict, window: int = 5) -> pd.DataFrame:
    """
    Calcola il percentile rolling, la sua velocità e accelerazione.
    """
    serie = pivot.get(current_year)
    if serie is None:
        return pd.DataFrame()
    
    storico = pivot.drop(columns=[current_year], errors="ignore")
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    
    # Calcola percentile per ogni TDI (solo fino all'ultimo reale)
    pct_series = pd.Series(index=serie.index, dtype=float)
    
    for tdi in range(1, ultimo_tdi + 1):
        val = serie.loc[tdi]
        if pd.isna(val):
            continue
        hist_vals = storico.loc[tdi].dropna()
        if len(hist_vals) > 0:
            pct_series.loc[tdi] = (hist_vals < val).sum() / len(hist_vals) * 100
    
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
def compute_anomaly_persistence(pivot: pd.DataFrame, perc: pd.DataFrame, 
                                 current_year: int, metadata: dict) -> dict:
    """
    Calcola giorni consecutivi fuori dall'IQR e statistiche correlate.
    """
    serie = pivot.get(current_year)
    if serie is None:
        return {"current_streak": 0, "max_streak": 0, "total_days_outside": 0, 
                "pct_days_outside": 0, "direction": "unknown", "streaks": pd.Series(dtype=int)}
    
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    
    # Considera solo fino all'ultimo TDI reale
    serie = serie.loc[:ultimo_tdi].dropna()
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
        last_tdi = serie.index[-1]
        if last_val < p25.loc[last_tdi]:
            direction = "below"
        elif last_val > p75.loc[last_tdi]:
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
# 8. REGIME CLUSTERING (CON MAX DRAWDOWN GEOMETRICO CORRETTO)
# =============================================================================
def compute_geometric_max_drawdown(ytd_series: pd.Series) -> float:
    """
    Calcola il Maximum Drawdown GEOMETRICO corretto.
    
    Converte il rendimento YTD % in equity curve (base 100),
    poi calcola DD = (peak - current) / peak.
    """
    # Converti YTD % in equity curve
    equity = 100 * (1 + ytd_series.dropna() / 100)
    
    if len(equity) == 0:
        return 0.0
    
    # Running maximum (peak)
    running_max = equity.cummax()
    
    # Drawdown percentuale geometrico
    drawdown = (equity - running_max) / running_max * 100
    
    # Max drawdown (valore più negativo)
    max_dd = drawdown.min()
    
    return max_dd


def cluster_historical_years(pivot_ytd: pd.DataFrame, pivot_returns: pd.DataFrame,
                              current_year: int, n_clusters: int = 3) -> pd.DataFrame:
    """
    Clustering degli anni storici per regime (bull/bear/sideways).
    Usa volatilità calcolata sui RENDIMENTI GIORNALIERI e Max DD GEOMETRICO.
    """
    storico_ytd = pivot_ytd.drop(columns=[current_year], errors="ignore")
    storico_returns = pivot_returns.drop(columns=[current_year], errors="ignore")
    
    if storico_ytd.shape[1] < n_clusters:
        return pd.DataFrame()
    
    # Features per anno
    features = pd.DataFrame(index=storico_ytd.columns)
    
    # Rendimento finale (ultimo valore non-NaN)
    features["final_ret"] = storico_ytd.apply(lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan)
    
    # Volatilità calcolata sui RENDIMENTI GIORNALIERI (non sul YTD cumulato)
    features["path_vol"] = storico_returns.std()
    
    # Max Drawdown GEOMETRICO corretto
    features["max_dd"] = storico_ytd.apply(compute_geometric_max_drawdown)
    
    # Sharpe proxy annualizzato
    mean_daily_ret = storico_returns.mean()
    std_daily_ret = storico_returns.std()
    features["sharpe_proxy"] = (mean_daily_ret * 252) / (std_daily_ret * np.sqrt(252))
    
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


def identify_current_regime(pivot_ytd: pd.DataFrame, pivot_returns: pd.DataFrame,
                             current_year: int, cluster_df: pd.DataFrame,
                             metadata: dict) -> str:
    """
    Identifica il regime più probabile dell'anno corrente.
    """
    if cluster_df.empty:
        return "Unknown"
    
    serie_ytd = pivot_ytd.get(current_year)
    serie_returns = pivot_returns.get(current_year)
    
    if serie_ytd is None or serie_returns is None:
        return "Unknown"
    
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    
    if ultimo_tdi < 20:
        return "Insufficient Data"
    
    # Calcola features correnti (solo fino all'ultimo TDI reale)
    serie_ytd_valid = serie_ytd.loc[:ultimo_tdi].dropna()
    serie_returns_valid = serie_returns.loc[:ultimo_tdi].dropna()
    
    if len(serie_ytd_valid) < 20:
        return "Insufficient Data"
    
    current_ret = serie_ytd_valid.iloc[-1]
    current_vol = serie_returns_valid.std()
    current_dd = compute_geometric_max_drawdown(serie_ytd_valid)
    
    # Trova il regime più simile (nearest neighbor)
    distances = []
    for _, row in cluster_df.iterrows():
        # Normalizza le distanze
        dist = np.sqrt(
            ((current_ret - row["final_ret"]) / cluster_df["final_ret"].std())**2 +
            ((current_vol - row["path_vol"]) / cluster_df["path_vol"].std())**2 +
            ((current_dd - row["max_dd"]) / cluster_df["max_dd"].std())**2
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
# 9. FORWARD RETURNS (CON CROSS-YEAR HANDLING)
# =============================================================================
def compute_forward_return_distribution(pivot: pd.DataFrame, current_year: int, 
                                         metadata: dict,
                                         lookahead_days: int = 20, 
                                         pct_tolerance: float = 10) -> dict:
    """
    Distribuzione dei rendimenti forward storici quando in anomalia simile.
    GESTISCE CORRETTAMENTE il wrap-around tra anni.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    serie_corrente = pivot.get(current_year)
    
    if serie_corrente is None:
        return {}
    
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    if ultimo_tdi == 0:
        return {}
    
    current_val = serie_corrente.loc[ultimo_tdi]
    if pd.isna(current_val):
        return {}
    
    # Calcola percentile corrente
    hist_vals_at_tdi = storico.loc[ultimo_tdi].dropna()
    if len(hist_vals_at_tdi) == 0:
        return {}
    
    current_pct = (hist_vals_at_tdi < current_val).sum() / len(hist_vals_at_tdi) * 100
    
    # Trova anni storici con percentile simile a questo TDI
    forward_rets = []
    matching_years = []
    
    anni_storici = sorted(storico.columns)
    
    for i, anno in enumerate(anni_storici):
        val_tdi = storico.loc[ultimo_tdi, anno]
        if pd.isna(val_tdi):
            continue
        
        hist_pct = (hist_vals_at_tdi < val_tdi).sum() / len(hist_vals_at_tdi) * 100
        
        if abs(hist_pct - current_pct) <= pct_tolerance:
            # Calcola forward return
            future_tdi = ultimo_tdi + lookahead_days
            
            if future_tdi <= MAX_TRADING_DAYS:
                # Forward return nello stesso anno
                future_val = storico.loc[future_tdi, anno]
                if not pd.isna(future_val):
                    fwd_ret = future_val - val_tdi
                    forward_rets.append(fwd_ret)
                    matching_years.append(anno)
            else:
                # Cross-year: cerca nell'anno successivo
                anno_next = anno + 1
                if anno_next in storico.columns:
                    # TDI nell'anno successivo
                    tdi_next_year = future_tdi - MAX_TRADING_DAYS
                    
                    # Rendimento a fine anno corrente
                    last_val_year = storico[anno].dropna().iloc[-1] if len(storico[anno].dropna()) > 0 else np.nan
                    
                    # Rendimento al TDI target nell'anno successivo
                    if tdi_next_year in storico.index:
                        val_next_year = storico.loc[tdi_next_year, anno_next]
                        
                        if not pd.isna(last_val_year) and not pd.isna(val_next_year):
                            # Forward return cross-year:
                            # (rendimento fino a fine anno) + (rendimento nell'anno nuovo)
                            # Nota: val_next_year è già YTD del nuovo anno, quindi va combinato
                            # correttamente col rendimento residuo dell'anno vecchio
                            
                            # Approssimazione: usiamo la somma dei rendimenti
                            # (più preciso sarebbe (1+r1)*(1+r2)-1 ma per % piccole è simile)
                            fwd_ret = (last_val_year - val_tdi) + val_next_year
                            forward_rets.append(fwd_ret)
                            matching_years.append(f"{anno}-{anno_next}")
    
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
        if df.empty or len(df) < 252:
            continue
        
        pivot_ytd, pivot_returns, metadata = compute_ytd_by_trading_day(df)
        
        if current_year not in pivot_ytd.columns:
            continue
        
        pct, ultimo_tdi = compute_current_percentile(pivot_ytd, current_year, metadata)
        if pd.isna(pct) or ultimo_tdi == 0:
            continue
        
        ytd_val = pivot_ytd[current_year].loc[ultimo_tdi]
        
        # Calcola Z-score
        zscore_series = compute_zscore_by_tdi(pivot_ytd, current_year, metadata)
        zscore_current = zscore_series.loc[ultimo_tdi] if ultimo_tdi in zscore_series.index else np.nan
        
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
            "TDI": ultimo_tdi,
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
def tdi_to_approx_date_label(tdi: int, ref_year: int = 2024) -> str:
    """
    Converte TDI in etichetta approssimativa di data.
    Assume ~21 trading days per mese.
    """
    # Approssimazione: TDI 1 = inizio gennaio, TDI 21 = fine gennaio, etc.
    approx_month = min(12, max(1, (tdi - 1) // 21 + 1))
    approx_day = min(28, ((tdi - 1) % 21) + 1)
    
    try:
        d = datetime(ref_year, approx_month, approx_day)
        return d.strftime("%b %d")
    except:
        return f"TDI {tdi}"


def tdi_to_labels(tdi_series: pd.Index) -> list:
    """Converte serie di TDI in etichette leggibili."""
    return [tdi_to_approx_date_label(int(tdi)) for tdi in tdi_series]


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
                                 metadata: dict,
                                 bootstrap_ci: dict = None) -> go.Figure:
    """Grafico principale con bande percentile e CI bootstrap."""
    fig = go.Figure()
    
    # Usa solo TDI con dati validi
    valid_tdi = perc.dropna().index
    labels = tdi_to_labels(valid_tdi)
    
    perc_valid = perc.loc[valid_tdi]

    # Bootstrap CI per banda 95 (se disponibile)
    if bootstrap_ci:
        ci_valid = bootstrap_ci["p95_ci_upper"].loc[valid_tdi]
        ci_lower_valid = bootstrap_ci["p95_ci_lower"].loc[valid_tdi]
        
        fig.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=ci_valid.tolist() + ci_lower_valid.tolist()[::-1],
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
        y=perc_valid["p95"].tolist() + perc_valid["p5"].tolist()[::-1],
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
        y=perc_valid["p75"].tolist() + perc_valid["p25"].tolist()[::-1],
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
        y=perc_valid["p50"].tolist(),
        mode="lines",
        line=dict(color=COLORS["median"], width=1.5, dash="dash"),
        name="Mediana (50° Pct)",
        showlegend=True,
    ))

    # Equity YTD corrente (SOLO fino all'ultimo TDI reale)
    serie_corrente = pivot.get(current_year)
    if serie_corrente is not None:
        ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
        
        # Filtra solo TDI validi e fino all'ultimo reale
        serie_plot = serie_corrente.loc[:ultimo_tdi].dropna()
        serie_plot = serie_plot[serie_plot.index.isin(valid_tdi)]
        
        if len(serie_plot) > 0:
            labels_ytd = tdi_to_labels(serie_plot.index)

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
            title="Trading Day (approssimazione calendario)",
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
                       current_year: int, metadata: dict) -> go.Figure:
    """Grafico Z-Score con contesto volatilità."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Z-Score YTD vs Storico", "Volatilità Contestuale"),
        row_heights=[0.6, 0.4],
    )
    
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    
    zscore_clean = zscore_series.loc[:ultimo_tdi].dropna()
    labels = tdi_to_labels(zscore_clean.index)
    
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
            annotation_text=f"{sigma}σ",
            annotation_position="right",
            row=1, col=1,
        )
    
    # Volatilità contestuale
    if not vol_context.empty:
        vol_clean = vol_context.loc[:ultimo_tdi].dropna()
        labels_vol = tdi_to_labels(vol_clean.index)
        
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
    fig.update_yaxes(title_text="Volatilità (%)", row=2, col=1, gridcolor=COLORS["grid"])
    
    return fig


def build_dynamics_chart(dynamics_df: pd.DataFrame, persistence_data: dict, 
                         current_year: int, metadata: dict) -> go.Figure:
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
    
    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    dynamics_clean = dynamics_df.loc[:ultimo_tdi].dropna()
    labels = tdi_to_labels(dynamics_clean.index)
    
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
    vel_clean = dynamics_clean["velocity"].dropna()
    labels_vel = tdi_to_labels(vel_clean.index)
    vel_colors = [COLORS["zscore_pos"] if v >= 0 else COLORS["zscore_neg"] 
                  for v in vel_clean.values]
    fig.add_trace(go.Bar(
        x=labels_vel,
        y=vel_clean.values,
        marker_color=vel_colors,
        name="Velocity",
        showlegend=False,
    ), row=2, col=1)
    
    # Acceleration
    acc_clean = dynamics_clean["acceleration"].dropna()
    labels_acc = tdi_to_labels(acc_clean.index)
    acc_colors = [COLORS["velocity"] if a >= 0 else COLORS["acceleration"] 
                  for a in acc_clean.values]
    fig.add_trace(go.Bar(
        x=labels_acc,
        y=acc_clean.values,
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
                       current_year: int, current_regime: str,
                       metadata: dict) -> go.Figure:
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
    
    for regime in ["Bull", "Bear", "Sideways"]:
        regime_years = cluster_df[cluster_df["regime"] == regime].index.tolist()
        regime_years = [y for y in regime_years if y in storico.columns]
        
        if len(regime_years) > 0:
            for i, anno in enumerate(regime_years):
                serie = storico[anno].dropna()
                labels_anno = tdi_to_labels(serie.index)
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
        ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
        serie_corrente = serie_corrente.loc[:ultimo_tdi].dropna()
        labels_curr = tdi_to_labels(serie_corrente.index)
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
    fig.update_yaxes(title_text="Volatilità Path (%)", row=1, col=1, gridcolor=COLORS["grid"])
    fig.update_xaxes(title_text="Trading Day", row=1, col=2, gridcolor=COLORS["grid"])
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
            "Forward Lookahead (trading days)",
            min_value=5,
            max_value=60,
            value=20,
            help="Periodo per analisi mean reversion (in trading days)",
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
        
        st.markdown("---")
        st.markdown("##### ℹ️ Note Tecniche")
        st.caption("""
        - **TDI**: Trading Day Index (evita bias da anni bisestili)
        - **Max DD**: Calcolato geometricamente
        - **Volatilità**: Su rendimenti giornalieri (non YTD cumulato)
        - **Forward**: Gestisce cross-year
        """)

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
        pivot_ytd, pivot_returns, metadata = compute_ytd_by_trading_day(df)
        anni_disponibili = sorted(pivot_ytd.columns.tolist())
        
        if len(anni_disponibili) < 3:
            st.error("❌ Storico insufficiente: servono almeno 3 anni di dati.")
            st.stop()
        
        perc = compute_percentiles(pivot_ytd, current_year)
        pct_attuale, ultimo_tdi = compute_current_percentile(pivot_ytd, current_year, metadata)
        zscore_series = compute_zscore_by_tdi(pivot_ytd, current_year, metadata)
        vol_context = compute_rolling_volatility_context(pivot_returns, current_year, metadata)
        dynamics = compute_percentile_dynamics(pivot_ytd, current_year, metadata)
        persistence = compute_anomaly_persistence(pivot_ytd, perc, current_year, metadata)
        cluster_df = cluster_historical_years(pivot_ytd, pivot_returns, current_year)
        current_regime = identify_current_regime(pivot_ytd, pivot_returns, current_year, cluster_df, metadata)
        regime_perc = compute_regime_conditional_percentiles(pivot_ytd, current_year, cluster_df, current_regime)
        forward_data = compute_forward_return_distribution(
            pivot_ytd, current_year, metadata,
            lookahead_days=lookahead_days, pct_tolerance=pct_tolerance
        )
        bootstrap_ci = bootstrap_percentile_bands(pivot_ytd, current_year, n_bootstrap=n_bootstrap)
    
    # ---- Quick Stats Header ----
    ytd_val = pivot_ytd[current_year].loc[ultimo_tdi] if ultimo_tdi > 0 else np.nan
    zscore_current = zscore_series.loc[ultimo_tdi] if ultimo_tdi in zscore_series.index else np.nan
    
    interpretation, color, emoji = get_anomaly_interpretation(pct_attuale, zscore_current)
    
    st.markdown("---")
    cols = st.columns(6)
    
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
        st.metric("Streak Fuori IQR", f"{persistence['current_streak']} gg")
    
    with cols[5]:
        st.metric("Trading Day", f"{ultimo_tdi}/{MAX_TRADING_DAYS}")
    
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
        Le <b>bande colorate</b> rappresentano la distribuzione storica dei rendimenti YTD per ogni trading day.
        La <b>linea rossa</b> è la performance dell'anno corrente. Quando esce dalle bande, indica un'<b>anomalia statistica</b>.
        <ul>
        <li><b>Banda chiara (5°-95°)</b>: range "normale" - il 90% degli anni storici cade qui</li>
        <li><b>Banda scura (25°-75°)</b>: range IQR - la "zona di comfort" del 50% centrale</li>
        <li><b>Linea tratteggiata</b>: mediana storica (50° percentile)</li>
        </ul>
        <b>Nota tecnica:</b> L'asse X usa il <b>Trading Day Index</b> (TDI) invece del Day-of-Year per eliminare 
        disallineamenti da anni bisestili e festività variabili.
        </div>
        """, unsafe_allow_html=True)
        
        fig_main = build_main_percentile_chart(pivot_ytd, perc, current_year, ticker, metadata, bootstrap_ci)
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Insight contestuali
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistiche Distribuzione")
            anni_storico = [a for a in anni_disponibili if a != current_year]
            
            # Valori al TDI corrente
            p50_oggi = perc['p50'].loc[ultimo_tdi] if ultimo_tdi in perc.index else np.nan
            p25_oggi = perc['p25'].loc[ultimo_tdi] if ultimo_tdi in perc.index else np.nan
            p75_oggi = perc['p75'].loc[ultimo_tdi] if ultimo_tdi in perc.index else np.nan
            p5_oggi = perc['p5'].loc[ultimo_tdi] if ultimo_tdi in perc.index else np.nan
            p95_oggi = perc['p95'].loc[ultimo_tdi] if ultimo_tdi in perc.index else np.nan
            
            stats_data = {
                "Metrica": ["Anni in analisi", "Mediana storica (TDI corrente)", "IQR Range", "5°-95° Range"],
                "Valore": [
                    f"{len(anni_storico)} ({min(anni_storico)}-{max(anni_storico)})",
                    f"{p50_oggi:.2f}%" if not pd.isna(p50_oggi) else "N/D",
                    f"{p25_oggi:.2f}% → {p75_oggi:.2f}%" if not pd.isna(p25_oggi) else "N/D",
                    f"{p5_oggi:.2f}% → {p95_oggi:.2f}%" if not pd.isna(p5_oggi) else "N/D",
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
        <b>Nota tecnica:</b> La volatilità nel pannello inferiore è calcolata sui <b>rendimenti giornalieri veri</b> 
        (non sulla variazione del YTD cumulato), evitando distorsioni da scaling.
        </div>
        """, unsafe_allow_html=True)
        
        fig_zscore = build_zscore_chart(zscore_series, vol_context, current_year, metadata)
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
            
            if not vol_context.empty and ultimo_tdi in vol_context.index:
                vol_zscore_current = vol_context.loc[ultimo_tdi, "vol_zscore"]
                
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
        <li><b>Velocity</b>: quanto velocemente sta cambiando il tuo ranking percentile (Δ su 5 trading days)</li>
        <li><b>Acceleration</b>: la velocità sta aumentando o diminuendo? (ΔΔ)</li>
        <li><b>Persistenza</b>: da quanti trading days sei fuori dalla zona normale?</li>
        </ul>
        Un'anomalia con <b>velocity negativa in accelerazione</b> mentre sei già sotto il 25° percentile è un segnale di stress significativo.
        </div>
        """, unsafe_allow_html=True)
        
        if not dynamics.empty:
            fig_dynamics = build_dynamics_chart(dynamics, persistence, current_year, metadata)
            st.plotly_chart(fig_dynamics, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏃 Velocity")
            velocity_current = dynamics["velocity"].loc[ultimo_tdi] if ultimo_tdi in dynamics.index else np.nan
            
            if not pd.isna(velocity_current):
                direction = "📈 Miglioramento" if velocity_current > 0 else "📉 Peggioramento"
                st.metric("Δ Percentile (5 TDI)", f"{velocity_current:+.1f}")
                st.markdown(f"**Trend:** {direction}")
        
        with col2:
            st.markdown("#### 🚀 Acceleration")
            acc_current = dynamics["acceleration"].loc[ultimo_tdi] if ultimo_tdi in dynamics.index else np.nan
            
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
            st.metric("Streak Corrente", f"{persistence['current_streak']} TDI")
            st.metric("Max Streak (anno)", f"{persistence['max_streak']} TDI")
            st.metric("% TDI Fuori IQR", f"{persistence['pct_days_outside']:.1f}%")
            
            if persistence['current_streak'] > 10:
                st.warning("⚠️ Anomalia persistente - potenziale cambio regime")
        
        # Insight combinato
        st.markdown("---")
        st.markdown("#### 🧠 Diagnosi Combinata")
        
        velocity_current = dynamics["velocity"].loc[ultimo_tdi] if ultimo_tdi in dynamics.index else np.nan
        acc_current = dynamics["acceleration"].loc[ultimo_tdi] if ultimo_tdi in dynamics.index else np.nan
        
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
        <b>Note tecniche:</b>
        <ul>
        <li>La <b>volatilità</b> è calcolata sui rendimenti giornalieri (non sul YTD cumulato)</li>
        <li>Il <b>Max Drawdown</b> è calcolato geometricamente: (peak - current) / peak</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if not cluster_df.empty:
            fig_regime = build_regime_chart(pivot_ytd, cluster_df, current_year, current_regime, metadata)
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
            if not regime_perc.empty and current_regime not in ["Unknown", "Insufficient Data"]:
                st.markdown("---")
                st.markdown("#### 📊 Percentili Regime-Conditional")
                
                # Calcola percentile condizionale
                serie_corrente = pivot_ytd.get(current_year)
                if serie_corrente is not None and ultimo_tdi > 0:
                    val_corrente = serie_corrente.loc[ultimo_tdi]
                    
                    same_regime_years = cluster_df[cluster_df["regime"] == current_regime].index.tolist()
                    same_regime_years = [y for y in same_regime_years if y != current_year and y in pivot_ytd.columns]
                    
                    if len(same_regime_years) >= 3:
                        storico_regime = pivot_ytd[same_regime_years]
                        vals_regime = storico_regime.loc[ultimo_tdi].dropna()
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
        nello stesso periodo dell'anno, cosa è successo nei <b>{lookahead_days} trading days</b> successivi?
        <br><br>
        <b>Note tecniche:</b>
        <ul>
        <li>L'analisi <b>gestisce il cross-year</b>: se il lookahead supera fine anno, combina i rendimenti tra anni</li>
        <li>Il lookahead è espresso in <b>trading days</b>, non giorni calendario</li>
        </ul>
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
                    delle volte il rendimento forward a {lookahead_days} trading days è stato positivo.
                    
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
                st.write(f"Anni trovati con percentile ±{pct_tolerance}% al TDI corrente:")
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

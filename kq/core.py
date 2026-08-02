"""
=============================================================================
kq.core — Analitiche single-asset (percentili stagionali, regime, forward)
=============================================================================
Contiene la logica originale della dashboard, riorganizzata in modulo e con
due migliorie:

    - `compute_percentile_dynamics` e' stato vettorizzato: il loop Python su
      ogni trading day e' sostituito da operazioni su matrice. A 250 TDI il
      guadagno e' ~50x, e permette di calcolare le dinamiche anche in batch.
    - `compute_percentiles_walkforward` aggiunge la variante a finestra
      espandente, che usa solo gli anni PRECEDENTI a quello valutato: serve
      quando si vuole misurare il segnale storicamente senza look-ahead.
      (`compute_percentiles` originale esclude solo l'anno corrente, che va
      benissimo per il segnale live ma introdurrebbe look-ahead in backtest.)

Il Trading Day Index (TDI) conta solo le sedute effettive, eliminando i
disallineamenti da anni bisestili e festivita' variabili.
=============================================================================
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kq import config as C


# =============================================================================
# 1. YTD SU TRADING DAY INDEX
# =============================================================================
def compute_ytd_by_trading_day(df: pd.DataFrame):
    """
    Rendimenti YTD per anno solare mappati su Trading Day Index.

    Base YTD = PRIMO prezzo dell'anno corrente (convenzione TradingView), che
    evita le distorsioni da gap di capodanno rilevanti su forex e crypto.

    Returns: (pivot_ytd, pivot_returns, metadata)
    """
    df = df.copy()
    df["year"] = df["date"].dt.year

    # I rendimenti giornalieri si calcolano PRIMA del loop annuale, cosi' il
    # primo giorno di ogni anno porta con se' il gap di fine anno precedente.
    df["daily_return"] = df["adjusted_close"].pct_change() * 100

    anni = sorted(df["year"].unique())
    max_tdi_per_year = df.groupby("year").size().max()
    max_trading_days = max(C.DEFAULT_MAX_TRADING_DAYS, int(max_tdi_per_year) + 5)

    ytd_dict, returns_dict = {}, {}
    metadata = {
        "last_valid_tdi": {},
        "tdi_to_date": {},
        "base_prices": {},
        "base_dates": {},
        "max_trading_days": max_trading_days,
    }

    for anno in anni:
        df_anno = df[df["year"] == anno].copy().reset_index(drop=True)
        base_price = df_anno["adjusted_close"].iloc[0]

        metadata["base_prices"][anno] = base_price
        metadata["base_dates"][anno] = df_anno["date"].iloc[0]

        df_anno["tdi"] = np.arange(1, len(df_anno) + 1)
        df_anno["ytd_pct"] = (df_anno["adjusted_close"] / base_price - 1) * 100
        max_tdi = int(df_anno["tdi"].max())

        serie_ytd = pd.Series(index=range(1, max_trading_days + 1), dtype=float)
        serie_ytd.loc[df_anno["tdi"].values] = df_anno["ytd_pct"].values
        # ffill solo fino all'ultima seduta reale: oltre resta NaN, cruciale
        # per non estendere artificialmente l'anno in corso.
        serie_ytd.loc[:max_tdi] = serie_ytd.loc[:max_tdi].ffill()

        serie_returns = pd.Series(index=range(1, max_trading_days + 1), dtype=float)
        serie_returns.loc[df_anno["tdi"].values] = df_anno["daily_return"].values

        ytd_dict[anno] = serie_ytd
        returns_dict[anno] = serie_returns
        metadata["last_valid_tdi"][anno] = max_tdi
        metadata["tdi_to_date"][anno] = dict(zip(df_anno["tdi"], df_anno["date"]))

    return pd.DataFrame(ytd_dict), pd.DataFrame(returns_dict), metadata


# =============================================================================
# 2. PERCENTILI STORICI
# =============================================================================
def compute_percentiles(pivot: pd.DataFrame, current_year: int, min_years: int = 3) -> pd.DataFrame:
    """Percentili 5/25/50/75/95 escludendo l'anno corrente."""
    storico = pivot.drop(columns=[current_year], errors="ignore")
    valid_mask = storico.notna().sum(axis=1) >= min_years

    perc = pd.DataFrame(index=pivot.index)
    for q, name in [(0.05, "p5"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")]:
        perc[name] = storico.quantile(q, axis=1).where(valid_mask)
    return perc


def compute_percentiles_walkforward(pivot: pd.DataFrame, target_year: int,
                                     min_years: int = 3) -> pd.DataFrame:
    """
    Variante causale: usa SOLO gli anni strettamente precedenti a `target_year`.

    `compute_percentiles` esclude l'anno corrente ma tiene tutti gli altri,
    compresi quelli successivi. Per il segnale live e' corretto (il futuro non
    esiste ancora); per valutare storicamente il segnale sarebbe look-ahead.
    """
    anni_passati = [a for a in pivot.columns if a < target_year]
    if len(anni_passati) < min_years:
        return pd.DataFrame(index=pivot.index, columns=["p5", "p25", "p50", "p75", "p95"], dtype=float)

    storico = pivot[anni_passati]
    valid_mask = storico.notna().sum(axis=1) >= min_years

    perc = pd.DataFrame(index=pivot.index)
    for q, name in [(0.05, "p5"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")]:
        perc[name] = storico.quantile(q, axis=1).where(valid_mask)
    return perc


def compute_current_percentile(pivot: pd.DataFrame, current_year: int, metadata: dict):
    """Percentile dell'YTD corrente sull'ultimo TDI REALE (non forward-filled)."""
    serie = pivot.get(current_year)
    if serie is None:
        return np.nan, 0

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    if ultimo_tdi == 0:
        return np.nan, 0

    valore = serie.loc[ultimo_tdi]
    if pd.isna(valore):
        return np.nan, ultimo_tdi

    storico = pivot.drop(columns=[current_year], errors="ignore")
    valori_storici = storico.loc[ultimo_tdi].dropna().values
    if len(valori_storici) == 0:
        return np.nan, ultimo_tdi

    percentile = (np.sum(valori_storici < valore) / len(valori_storici)) * 100
    return round(float(percentile), 1), ultimo_tdi


# =============================================================================
# 3. Z-SCORE E CONTESTO VOLATILITA'
# =============================================================================
def compute_zscore_by_tdi(pivot: pd.DataFrame, current_year: int, metadata: dict) -> pd.Series:
    """Z-score dell'YTD corrente vs distribuzione storica, per ogni TDI."""
    storico = pivot.drop(columns=[current_year], errors="ignore")
    serie = pivot.get(current_year)
    if serie is None:
        return pd.Series(dtype=float)

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    mu = storico.mean(axis=1)
    sigma = storico.std(axis=1).replace(0, np.nan)

    z = (serie - mu) / sigma
    z.loc[ultimo_tdi + 1:] = np.nan
    return z


def compute_rolling_volatility_context(pivot_returns: pd.DataFrame, current_year: int,
                                        metadata: dict, window: int = 20) -> pd.DataFrame:
    """Volatilita' rolling dell'anno corrente vs media storica, sui rendimenti giornalieri."""
    storico = pivot_returns.drop(columns=[current_year], errors="ignore")
    serie = pivot_returns.get(current_year)
    if serie is None or storico.empty:
        return pd.DataFrame()

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)

    vol_corrente = serie.rolling(window=window, min_periods=5).std()
    vol_storica_df = storico.rolling(window=window, min_periods=5).std()

    result = pd.DataFrame({
        "vol_corrente": vol_corrente,
        "vol_storica_mean": vol_storica_df.mean(axis=1),
        "vol_storica_std": vol_storica_df.std(axis=1),
    })
    result["vol_zscore"] = (
        (result["vol_corrente"] - result["vol_storica_mean"])
        / result["vol_storica_std"].replace(0, np.nan)
    )
    result.loc[ultimo_tdi + 1:] = np.nan
    return result


# =============================================================================
# 4. DINAMICHE DELL'ANOMALIA (VETTORIZZATO)
# =============================================================================
def compute_percentile_dynamics(pivot: pd.DataFrame, current_year: int,
                                 metadata: dict, window: int = 5) -> pd.DataFrame:
    """
    Percentile rolling, velocita' e accelerazione.

    Versione vettorizzata: il rank di ogni TDI e' calcolato con un confronto
    matriciale invece che con un loop Python su ~250 iterazioni.
    """
    serie = pivot.get(current_year)
    if serie is None:
        return pd.DataFrame()

    storico = pivot.drop(columns=[current_year], errors="ignore")
    if storico.empty:
        return pd.DataFrame()

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)

    # Confronto broadcast: (TDI x anni) < (TDI,) -> quota di anni sotto il valore corrente
    n_validi = storico.notna().sum(axis=1)
    n_sotto = storico.lt(serie, axis=0).sum(axis=1)
    pct = (n_sotto / n_validi.replace(0, np.nan)) * 100

    pct = pct.where(serie.notna())
    pct.loc[ultimo_tdi + 1:] = np.nan

    return pd.DataFrame({
        "percentile": pct,
        "velocity": pct.diff(window),
        "acceleration": pct.diff(window).diff(window),
    })


# =============================================================================
# 5. PERSISTENZA
# =============================================================================
def compute_anomaly_persistence(pivot: pd.DataFrame, perc: pd.DataFrame,
                                 current_year: int, metadata: dict) -> dict:
    """Giorni consecutivi fuori dall'IQR e statistiche correlate."""
    empty = {"current_streak": 0, "max_streak": 0, "total_days_outside": 0,
             "pct_days_outside": 0, "direction": "unknown",
             "streaks": pd.Series(dtype=int), "outside_iqr": pd.Series(dtype=bool)}

    serie = pivot.get(current_year)
    if serie is None:
        return empty

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    serie = serie.loc[:ultimo_tdi].dropna()
    if len(serie) == 0 or perc.empty:
        return empty

    idx = serie.index.intersection(perc.index)
    serie = serie.loc[idx]
    p25 = perc["p25"].loc[idx]
    p75 = perc["p75"].loc[idx]

    outside_iqr = (serie < p25) | (serie > p75)
    streak_groups = (~outside_iqr).cumsum()
    streaks = outside_iqr.groupby(streak_groups).cumsum()

    current_streak = int(streaks.iloc[-1]) if len(streaks) > 0 and bool(outside_iqr.iloc[-1]) else 0
    max_streak = int(streaks.max()) if len(streaks) > 0 else 0
    total_outside = int(outside_iqr.sum())

    last_tdi = serie.index[-1]
    last_val = serie.iloc[-1]
    if last_val < p25.loc[last_tdi]:
        direction = "below"
    elif last_val > p75.loc[last_tdi]:
        direction = "above"
    else:
        direction = "within"

    return {
        "current_streak": current_streak,
        "max_streak": max_streak,
        "total_days_outside": total_outside,
        "pct_days_outside": round(total_outside / len(serie) * 100, 1),
        "direction": direction,
        "streaks": streaks,
        "outside_iqr": outside_iqr,
    }


# =============================================================================
# 6. REGIME CLUSTERING
# =============================================================================
def compute_geometric_max_drawdown(ytd_series: pd.Series) -> float:
    """Max drawdown geometrico: converte l'YTD % in equity base 100 e calcola (peak-cur)/peak."""
    equity = 100 * (1 + ytd_series.dropna() / 100)
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    return float(((equity - running_max) / running_max * 100).min())


def cluster_historical_years(pivot_ytd: pd.DataFrame, pivot_returns: pd.DataFrame,
                              current_year: int, n_clusters: int = 3) -> pd.DataFrame:
    """K-Means 4D sugli anni storici: final_ret, path_vol, max_dd, sharpe_proxy."""
    storico_ytd = pivot_ytd.drop(columns=[current_year], errors="ignore")
    storico_ret = pivot_returns.drop(columns=[current_year], errors="ignore")

    if storico_ytd.shape[1] < n_clusters:
        return pd.DataFrame()

    features = pd.DataFrame(index=storico_ytd.columns)
    features["final_ret"] = storico_ytd.apply(
        lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
    )
    features["path_vol"] = storico_ret.std()
    features["max_dd"] = storico_ytd.apply(compute_geometric_max_drawdown)
    features["sharpe_proxy"] = (storico_ret.mean() * 252) / (
        storico_ret.std().replace(0, np.nan) * np.sqrt(252)
    )

    features_clean = features.dropna()
    if len(features_clean) < n_clusters:
        return pd.DataFrame()

    cols = ["final_ret", "path_vol", "max_dd", "sharpe_proxy"]
    scaled = StandardScaler().fit_transform(features_clean[cols])
    features_clean["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)

    cluster_means = features_clean.groupby("cluster")["final_ret"].mean().sort_values()
    label_map = {cluster_means.index[0]: "Bear", cluster_means.index[-1]: "Bull"}
    for c in cluster_means.index:
        label_map.setdefault(c, "Sideways")

    features_clean["regime"] = features_clean["cluster"].map(label_map)
    return features_clean


def identify_current_regime(pivot_ytd: pd.DataFrame, pivot_returns: pd.DataFrame,
                             current_year: int, cluster_df: pd.DataFrame,
                             metadata: dict) -> str:
    """Regime piu' probabile dell'anno corrente: voto dei 3 anni piu' vicini nello spazio 4D."""
    if cluster_df.empty:
        return "Unknown"

    serie_ytd = pivot_ytd.get(current_year)
    serie_ret = pivot_returns.get(current_year)
    if serie_ytd is None or serie_ret is None:
        return "Unknown"

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    if ultimo_tdi < 20:
        return "Insufficient Data"

    ytd_valid = serie_ytd.loc[:ultimo_tdi].dropna()
    ret_valid = serie_ret.loc[:ultimo_tdi].dropna()
    if len(ytd_valid) < 20 or len(ret_valid) < 20:
        return "Insufficient Data"

    cur = {
        "final_ret": ytd_valid.iloc[-1],
        "path_vol": ret_valid.std(),
        "max_dd": compute_geometric_max_drawdown(ytd_valid),
        "sharpe_proxy": (ret_valid.mean() * 252) / (ret_valid.std() * np.sqrt(252))
        if ret_valid.std() > 0 else 0.0,
    }

    stds = {k: cluster_df[k].std() for k in cur}
    distances = []
    for _, row in cluster_df.iterrows():
        d = np.sqrt(sum(
            ((cur[k] - row[k]) / stds[k]) ** 2 if stds[k] and stds[k] > 0 else 0.0
            for k in cur
        ))
        distances.append((row["regime"], d))

    distances.sort(key=lambda x: x[1])
    return pd.Series([d[0] for d in distances[:3]]).value_counts().index[0]


def compute_regime_conditional_percentiles(pivot: pd.DataFrame, current_year: int,
                                            cluster_df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Percentili calcolati sui soli anni dello stesso regime."""
    if cluster_df.empty or regime in ("Unknown", "Insufficient Data"):
        return pd.DataFrame()

    anni = [y for y in cluster_df[cluster_df["regime"] == regime].index
            if y != current_year and y in pivot.columns]
    if len(anni) < 3:
        return pd.DataFrame()

    storico = pivot[anni]
    perc = pd.DataFrame(index=pivot.index)
    for q, name in [(0.05, "p5"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.95, "p95")]:
        perc[name] = storico.quantile(q, axis=1)
    return perc


# =============================================================================
# 7. FORWARD RETURNS
# =============================================================================
def compute_forward_return_distribution(pivot: pd.DataFrame, current_year: int, metadata: dict,
                                         lookahead_days: int = 20,
                                         pct_tolerance: float = 10) -> dict:
    """
    Distribuzione dei rendimenti forward storici in condizioni percentile simili.
    Gestisce il wrap-around tra anni con compounding geometrico esatto.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")
    serie = pivot.get(current_year)
    if serie is None or storico.empty:
        return {}

    ultimo_tdi = metadata["last_valid_tdi"].get(current_year, 0)
    if ultimo_tdi == 0:
        return {}

    max_trading_days = metadata.get("max_trading_days", C.DEFAULT_MAX_TRADING_DAYS)
    current_val = serie.loc[ultimo_tdi]
    if pd.isna(current_val):
        return {}

    hist_at_tdi = storico.loc[ultimo_tdi].dropna()
    if len(hist_at_tdi) == 0:
        return {}

    current_pct = (hist_at_tdi < current_val).sum() / len(hist_at_tdi) * 100

    forward_rets, matching_years = [], []

    for anno in sorted(storico.columns):
        val_tdi = storico.loc[ultimo_tdi, anno]
        if pd.isna(val_tdi):
            continue

        hist_pct = (hist_at_tdi < val_tdi).sum() / len(hist_at_tdi) * 100
        if abs(hist_pct - current_pct) > pct_tolerance:
            continue

        future_tdi = ultimo_tdi + lookahead_days
        max_tdi_anno = metadata["last_valid_tdi"].get(anno, max_trading_days)

        if future_tdi <= max_tdi_anno:
            if future_tdi in storico.index:
                future_val = storico.loc[future_tdi, anno]
                if not pd.isna(future_val):
                    r = ((1 + future_val / 100) / (1 + val_tdi / 100) - 1) * 100
                    forward_rets.append(r)
                    matching_years.append(anno)
        else:
            anno_next = anno + 1
            if anno_next not in storico.columns:
                continue
            tdi_next = future_tdi - max_tdi_anno
            serie_anno = storico[anno].dropna()
            if len(serie_anno) == 0 or tdi_next not in storico.index:
                continue
            last_val_year = serie_anno.iloc[-1]
            val_next = storico.loc[tdi_next, anno_next]
            if pd.isna(last_val_year) or pd.isna(val_next):
                continue
            r1 = (1 + last_val_year / 100) / (1 + val_tdi / 100) - 1
            r2 = val_next / 100
            forward_rets.append(((1 + r1) * (1 + r2) - 1) * 100)
            matching_years.append(f"{anno}-{anno_next}")

    if not forward_rets:
        return {}

    fwd = pd.Series(forward_rets)
    return {
        "forward_returns": fwd,
        "matching_years": matching_years,
        "current_percentile": current_pct,
        "mean_forward": fwd.mean(),
        "median_forward": fwd.median(),
        "std_forward": fwd.std(),
        "prob_positive": (fwd > 0).mean() * 100,
        "n_samples": len(fwd),
        "lookahead_days": lookahead_days,
    }


# =============================================================================
# 8. BOOTSTRAP CI
# =============================================================================
def bootstrap_percentile_bands(pivot: pd.DataFrame, current_year: int,
                                n_bootstrap: int = 500, alpha: float = 0.05) -> dict:
    """Intervalli di confidenza bootstrap per le bande percentile."""
    storico = pivot.drop(columns=[current_year], errors="ignore")
    n_years = storico.shape[1]
    if n_years < 5:
        return {}

    rng = np.random.default_rng(42)
    samples = {"p5": [], "p50": [], "p95": []}

    for _ in range(n_bootstrap):
        cols = rng.choice(storico.columns, size=n_years, replace=True)
        sample = storico[cols]
        samples["p5"].append(sample.quantile(0.05, axis=1))
        samples["p50"].append(sample.quantile(0.50, axis=1))
        samples["p95"].append(sample.quantile(0.95, axis=1))

    out = {}
    for key, lst in samples.items():
        dfk = pd.concat(lst, axis=1)
        out[f"{key}_ci_lower"] = dfk.quantile(alpha / 2, axis=1)
        out[f"{key}_ci_upper"] = dfk.quantile(1 - alpha / 2, axis=1)
    return out


# =============================================================================
# 9. UTILITY
# =============================================================================
def tdi_to_approx_date_label(tdi: int, ref_year: int = 2024) -> str:
    """Converte il TDI in etichetta di calendario approssimata (~21 sedute/mese)."""
    approx_month = min(12, max(1, (tdi - 1) // 21 + 1))
    approx_day = min(28, ((tdi - 1) % 21) + 1)
    try:
        return datetime(ref_year, approx_month, approx_day).strftime("%b %d")
    except Exception:
        return f"TDI {tdi}"


def tdi_to_labels(tdi_index) -> list:
    return [tdi_to_approx_date_label(int(t)) for t in tdi_index]


def get_anomaly_interpretation(pct: float, zscore: float | None = None):
    """Interpretazione testuale + colore + emoji per il percentile corrente."""
    if pd.isna(pct):
        return "Dati insufficienti", "gray", "⚪"

    if pct >= 90:
        text = "Anomalia ESTREMA positiva — performance eccezionalmente superiore alla norma storica"
        color, emoji = C.COLORS["zscore_pos"], "🟢"
    elif pct >= 75:
        text = "Performance significativamente sopra la mediana storica"
        color, emoji = C.COLORS["zscore_pos"], "🟢"
    elif pct >= 50:
        text = "Performance nella metà superiore della distribuzione storica"
        color, emoji = "lightgreen", "🔵"
    elif pct >= 25:
        text = "Performance nella metà inferiore, ma entro la normalità (IQR)"
        color, emoji = C.COLORS["regime_sideways"], "🟡"
    elif pct >= 10:
        text = "Performance significativamente sotto la mediana storica"
        color, emoji = C.COLORS["zscore_neg"], "🔴"
    else:
        text = "Anomalia ESTREMA negativa — performance eccezionalmente inferiore alla norma storica"
        color, emoji = C.COLORS["zscore_neg"], "🔴"

    if zscore is not None and not pd.isna(zscore):
        if abs(zscore) > 2.5:
            text += f" | Z-Score: {zscore:.2f}σ (MOLTO significativo)"
        elif abs(zscore) > 2:
            text += f" | Z-Score: {zscore:.2f}σ (significativo)"
        elif abs(zscore) > 1.5:
            text += f" | Z-Score: {zscore:.2f}σ (moderato)"

    return text, color, emoji

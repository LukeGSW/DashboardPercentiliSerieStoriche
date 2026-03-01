"""
=============================================================================
YTD Seasonality & Percentile Quant Dashboard — Kriterion Quant
=============================================================================
Descrizione:
    Dashboard interattiva per l'analisi quantitativa della stagionalità YTD.
    Confronta la performance dell'anno corrente con le bande di percentile
    storiche, calcolate su tutti gli anni precedenti.

Fonte dati: EODHD API (Adjusted Close giornaliero)
Deploy: Streamlit Community Cloud (chiave API via st.secrets)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import date, datetime

# =============================================================================
# CONFIGURAZIONE PAGINA
# =============================================================================
st.set_page_config(
    page_title="YTD Seasonality Dashboard | Kriterion Quant",
    page_icon="📈",
    layout="wide",
)

# =============================================================================
# COSTANTI
# =============================================================================
EODHD_BASE_URL = "https://eodhd.com/api/eod"

# Palette colori Kriterion Quant
COLOR_BAND_95   = "rgba(100, 149, 237, 0.15)"   # banda 5°-95° (azzurro trasparente)
COLOR_BAND_IQR  = "rgba(100, 149, 237, 0.35)"   # banda 25°-75° (azzurro più marcato)
COLOR_MEDIAN    = "rgba(173, 216, 230, 0.9)"     # mediana (azzurro chiaro)
COLOR_YTD       = "#FF4B4B"                      # equity anno corrente (rosso Streamlit)


# =============================================================================
# 1. DATA FETCHING
# =============================================================================
@st.cache_data(ttl=3600, show_spinner="Scaricamento dati storici...")
def fetch_ohlcv(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Scarica la serie storica giornaliera (Adjusted Close) da EODHD API.

    Args:
        ticker    : simbolo nel formato EODHD (es. 'SPY.US', 'BTC-USD.CC')
        start_date: data di inizio nel formato 'YYYY-MM-DD'

    Returns:
        DataFrame con colonne ['date', 'adjusted_close'], ordinato per data.
        Restituisce un DataFrame vuoto in caso di errore.
    """
    api_key = st.secrets["EODHD_API_KEY"]
    today   = date.today().strftime("%Y-%m-%d")

    params = {
        "api_token": api_key,
        "from":      start_date,
        "to":        today,
        "fmt":       "json",
        "period":    "d",
    }

    try:
        url      = f"{EODHD_BASE_URL}/{ticker}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or isinstance(data, dict):
            st.error(f"Nessun dato ricevuto per il ticker **{ticker}**. Controlla il simbolo.")
            return pd.DataFrame()

        df = pd.DataFrame(data)[["date", "adjusted_close"]].copy()
        df["date"]            = pd.to_datetime(df["date"])
        df["adjusted_close"]  = pd.to_numeric(df["adjusted_close"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df

    except requests.exceptions.HTTPError as e:
        st.error(f"Errore HTTP EODHD ({e.response.status_code}): verifica il ticker o la chiave API.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Errore imprevisto nel fetch: {e}")
        return pd.DataFrame()


# =============================================================================
# 2. CALCOLO YTD E MAPPATURA SU DAY OF YEAR (DOY)
# =============================================================================
def compute_ytd_by_doy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i rendimenti YTD per ogni anno solare e li mappa
    su un asse comune Day-of-Year (DOY) da 1 a 366, con forward fill
    per i giorni non scambiati (weekend/festività).

    Args:
        df: DataFrame con colonne ['date', 'adjusted_close']

    Returns:
        DataFrame pivot con DOY come indice (1-366) e anni come colonne.
        Ogni cella contiene il rendimento YTD cumulativo in percentuale.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["doy"]  = df["date"].dt.day_of_year   # pandas .day_of_year è 1-indexed

    anni = sorted(df["year"].unique())

    # Costruiamo un dizionario anno -> serie DOY -> ytd%
    ytd_dict = {}

    for anno in anni:
        # Prezzo di chiusura dell'ultimo giorno dell'anno precedente (base per YTD)
        anno_prec = anno - 1
        df_prec   = df[df["year"] == anno_prec]

        if df_prec.empty:
            # Primo anno disponibile: usiamo il primo prezzo dell'anno come base
            base_price = df[df["year"] == anno]["adjusted_close"].iloc[0]
        else:
            base_price = df_prec["adjusted_close"].iloc[-1]

        df_anno = df[df["year"] == anno].copy()
        df_anno["ytd_pct"] = (df_anno["adjusted_close"] / base_price - 1) * 100

        # Mappatura su DOY 1-366: crea serie completa e applica ffill
        serie_doy = pd.Series(
            data  = df_anno["ytd_pct"].values,
            index = df_anno["doy"].values,
        )
        serie_full = serie_doy.reindex(range(1, 367))   # DOY 1-366
        serie_full = serie_full.ffill()                 # Forward fill giorni vuoti

        ytd_dict[anno] = serie_full

    pivot = pd.DataFrame(ytd_dict)   # indice=DOY (1-366), colonne=anni
    return pivot


# =============================================================================
# 3. CALCOLO PERCENTILI STORICI
# =============================================================================
def compute_percentiles(pivot: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """
    Calcola, per ogni DOY, i percentili 5°, 25°, 50°, 75°, 95°
    usando SOLO gli anni storici (esclude l'anno corrente).

    Args:
        pivot       : DataFrame DOY × anni (output di compute_ytd_by_doy)
        current_year: anno da escludere dal calcolo statistico

    Returns:
        DataFrame con colonne [p5, p25, p50, p75, p95] e DOY come indice.
    """
    storico = pivot.drop(columns=[current_year], errors="ignore")

    perc = pd.DataFrame(index=pivot.index)
    perc["p5"]  = storico.quantile(0.05, axis=1)
    perc["p25"] = storico.quantile(0.25, axis=1)
    perc["p50"] = storico.quantile(0.50, axis=1)
    perc["p75"] = storico.quantile(0.75, axis=1)
    perc["p95"] = storico.quantile(0.95, axis=1)

    return perc


# =============================================================================
# 4. CALCOLO PERCENTILE CORRENTE (RANKING PUNTUALE)
# =============================================================================
def compute_current_percentile(pivot: pd.DataFrame, current_year: int) -> float:
    """
    Calcola a quale percentile si trova il valore YTD attuale rispetto
    alla distribuzione storica nello stesso DOY.

    Returns:
        Percentile corrente (0-100).
    """
    serie_ytd_corrente = pivot.get(current_year)
    if serie_ytd_corrente is None:
        return np.nan

    # Ultimo DOY disponibile con dato non-NaN
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
# 5. CONVERSIONE DOY → ETICHETTA MESE/GIORNO
# =============================================================================
def doy_to_label(doy_series: pd.Index, ref_year: int = 2024) -> list:
    """
    Converte indici DOY (1-366) in stringhe 'MM/DD' usando un anno
    bisestile di riferimento per garantire la copertura del DOY 366.

    Args:
        doy_series: indice intero con i DOY
        ref_year  : anno bisestile di riferimento (default 2024)

    Returns:
        Lista di stringhe nel formato 'MM/DD'.
    """
    labels = []
    for doy in doy_series:
        try:
            d = datetime(ref_year, 1, 1) + pd.Timedelta(days=int(doy) - 1)
            labels.append(d.strftime("%b %d"))
        except Exception:
            labels.append(str(doy))
    return labels


# =============================================================================
# 6. COSTRUZIONE GRAFICO PLOTLY
# =============================================================================
def build_chart(
    pivot:        pd.DataFrame,
    perc:         pd.DataFrame,
    current_year: int,
    ticker:       str,
) -> go.Figure:
    """
    Costruisce il grafico Plotly con bande di percentile (sfondo) e
    l'equity YTD dell'anno corrente (primo piano).

    Args:
        pivot       : DataFrame DOY × anni
        perc        : DataFrame percentili (p5, p25, p50, p75, p95)
        current_year: anno corrente
        ticker      : simbolo del ticker (per il titolo interno)

    Returns:
        Oggetto go.Figure pronto per st.plotly_chart().
    """
    fig  = go.Figure()
    doys = perc.index.tolist()
    labels = doy_to_label(perc.index)

    # --- Banda 5°-95° (coda della distribuzione) ---
    fig.add_trace(go.Scatter(
        x    = labels + labels[::-1],
        y    = perc["p95"].tolist() + perc["p5"].tolist()[::-1],
        fill = "toself",
        fillcolor = COLOR_BAND_95,
        line = dict(color="rgba(0,0,0,0)"),
        name = "5° - 95° Pct",
        showlegend = True,
        hoverinfo  = "skip",
    ))

    # --- Banda 25°-75° (core della distribuzione / IQR) ---
    fig.add_trace(go.Scatter(
        x    = labels + labels[::-1],
        y    = perc["p75"].tolist() + perc["p25"].tolist()[::-1],
        fill = "toself",
        fillcolor = COLOR_BAND_IQR,
        line = dict(color="rgba(0,0,0,0)"),
        name = "25° - 75° Pct (IQR)",
        showlegend = True,
        hoverinfo  = "skip",
    ))

    # --- Linea Mediana (50° percentile) ---
    fig.add_trace(go.Scatter(
        x    = labels,
        y    = perc["p50"].tolist(),
        mode = "lines",
        line = dict(color=COLOR_MEDIAN, width=1.5, dash="dash"),
        name = "Mediana (50° Pct)",
        showlegend = True,
    ))

    # --- Equity Anno Corrente (YTD) ---
    serie_corrente = pivot.get(current_year)
    if serie_corrente is not None:
        # Tronca al DOY massimo con dati reali (esclude la parte futura)
        ultimo_doy_valido = serie_corrente.last_valid_index()
        serie_plot = serie_corrente.loc[:ultimo_doy_valido]
        labels_ytd = doy_to_label(serie_plot.index)

        fig.add_trace(go.Scatter(
            x    = labels_ytd,
            y    = serie_plot.values,
            mode = "lines",
            line = dict(color=COLOR_YTD, width=3),
            name = f"YTD {current_year}",
            showlegend = True,
        ))

        # Marker all'ultimo giorno con annotazione
        ultimo_val = serie_plot.iloc[-1]
        ultimo_label = labels_ytd[-1]
        segno = "+" if ultimo_val >= 0 else ""

        fig.add_trace(go.Scatter(
            x    = [ultimo_label],
            y    = [ultimo_val],
            mode = "markers+text",
            marker = dict(color=COLOR_YTD, size=10, symbol="circle"),
            text = [f"{segno}{ultimo_val:.2f}%"],
            textposition = "top right",
            textfont = dict(color=COLOR_YTD, size=13, family="Arial Black"),
            showlegend = False,
            hoverinfo  = "skip",
        ))

    # --- Layout ---
    fig.update_layout(
        template    = "plotly_dark",
        paper_bgcolor = "#0E1117",
        plot_bgcolor  = "#0E1117",
        xaxis = dict(
            title    = "Giorno dell'Anno",
            showgrid = True,
            gridcolor = "rgba(255,255,255,0.07)",
            tickangle = -45,
            # Mostra un tick ogni ~30 giorni circa
            tickmode = "array",
            tickvals = doy_to_label(
                pd.Index([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]),
                2024
            ),
        ),
        yaxis = dict(
            title     = "Rendimento YTD (%)",
            showgrid  = True,
            gridcolor = "rgba(255,255,255,0.07)",
            zeroline  = True,
            zerolinecolor = "rgba(255,255,255,0.25)",
            zerolinewidth = 1,
            ticksuffix = "%",
        ),
        legend = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.01,
            xanchor     = "left",
            x           = 0,
            bgcolor      = "rgba(14,17,23,0.7)",
            bordercolor  = "rgba(255,255,255,0.1)",
            borderwidth  = 1,
        ),
        margin     = dict(l=60, r=40, t=60, b=80),
        hovermode  = "x unified",
        height     = 580,
    )

    return fig


# =============================================================================
# 7. UI STREAMLIT
# =============================================================================
def main():
    # ---- Sidebar ----
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/200x60/0E1117/FF4B4B?text=KriterionQuant",
            use_column_width=True,
        )
        st.markdown("---")
        st.header("⚙️ Parametri")

        ticker = st.text_input(
            label       = "Ticker (formato EODHD)",
            value       = "SPY.US",
            placeholder = "es. SPY.US, BTC-USD.CC, ENI.MI",
            help        = "Inserisci il simbolo nel formato EODHD: SIMBOLO.EXCHANGE",
        ).strip().upper()

        start_date = st.date_input(
            label   = "Inizio storico",
            value   = date(2000, 1, 1),
            min_value = date(1990, 1, 1),
            max_value = date.today(),
            help    = "Data di inizio per il download dello storico.",
        )

        st.markdown("---")
        st.caption("📡 Dati: [EODHD API](https://eodhd.com)")
        st.caption("🔬 Kriterion Quant © 2025")

    # ---- Main Area ----
    current_year = date.today().year

    st.markdown(f"## 📊 Analisi Stagionalità YTD: `{ticker}`")
    st.markdown(
        f"Confronto del rendimento YTD **{current_year}** con le bande di percentile storiche "
        f"(dati dal **{start_date.strftime('%d/%m/%Y')}** ad oggi)."
    )

    # Bottone di aggiornamento manuale (bypassa la cache)
    col_btn, col_empty = st.columns([1, 5])
    with col_btn:
        if st.button("🔄 Aggiorna Dati"):
            st.cache_data.clear()

    # ---- Fetch dati ----
    df = fetch_ohlcv(ticker, start_date.strftime("%Y-%m-%d"))

    if df.empty:
        st.warning("Nessun dato disponibile. Modifica i parametri nella sidebar.")
        st.stop()

    # ---- Elaborazione ----
    with st.spinner("Calcolo percentili storici..."):
        pivot = compute_ytd_by_doy(df)

        anni_disponibili = sorted(pivot.columns.tolist())
        if len(anni_disponibili) < 2:
            st.warning("Storico insufficiente: servono almeno 2 anni di dati.")
            st.stop()

        perc       = compute_percentiles(pivot, current_year)
        pct_attuale = compute_current_percentile(pivot, current_year)

    # ---- Grafico ----
    fig = build_chart(pivot, perc, current_year, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Metriche sotto il grafico ----
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    # Valore YTD corrente
    serie_corrente = pivot.get(current_year)
    if serie_corrente is not None:
        ultimo_doy_valido = serie_corrente.last_valid_index()
        ytd_val = serie_corrente.loc[ultimo_doy_valido] if ultimo_doy_valido else np.nan
    else:
        ytd_val = np.nan

    segno_val = "+" if (not np.isnan(ytd_val) and ytd_val >= 0) else ""

    with col1:
        st.metric(
            label = f"Rendimento YTD {current_year}",
            value = f"{segno_val}{ytd_val:.2f}%" if not np.isnan(ytd_val) else "N/D",
        )

    with col2:
        if not np.isnan(pct_attuale):
            st.metric(
                label = "Percentile Storico Attuale",
                value = f"{pct_attuale:.1f}°",
                help  = "Posizione percentuale dell'YTD corrente rispetto agli anni storici nello stesso giorno dell'anno.",
            )
        else:
            st.metric(label="Percentile Storico Attuale", value="N/D")

    with col3:
        anni_storico = [a for a in anni_disponibili if a != current_year]
        st.metric(
            label = "Anni in Analisi (storico)",
            value = f"{len(anni_storico)}",
            help  = f"Anni inclusi nel calcolo dei percentili: {min(anni_storico) if anni_storico else 'N/D'} – {max(anni_storico) if anni_storico else 'N/D'}",
        )

    # ---- Interpretazione testuale ----
    if not np.isnan(pct_attuale):
        if pct_attuale >= 75:
            interpretazione = f"🟢 L'asset **{ticker}** si trova al **{pct_attuale:.1f}° percentile**: performance significativamente superiore alla mediana storica."
        elif pct_attuale >= 50:
            interpretazione = f"🔵 L'asset **{ticker}** si trova al **{pct_attuale:.1f}° percentile**: performance nella metà superiore della distribuzione storica."
        elif pct_attuale >= 25:
            interpretazione = f"🟡 L'asset **{ticker}** si trova al **{pct_attuale:.1f}° percentile**: performance nella metà inferiore, ma nella zona normale (IQR)."
        else:
            interpretazione = f"🔴 L'asset **{ticker}** si trova al **{pct_attuale:.1f}° percentile**: performance anomala, nella coda inferiore della distribuzione storica."

        st.info(interpretazione)

    # ---- Expander con dati grezzi ----
    with st.expander("🗂️ Dati Grezzi - Percentili per DOY"):
        labels_display = doy_to_label(perc.index)
        perc_display = perc.copy()
        perc_display.index = labels_display
        perc_display.columns = ["5° Pct", "25° Pct", "Mediana", "75° Pct", "95° Pct"]
        st.dataframe(
            perc_display.style.format("{:.2f}%"),
            use_container_width=True,
            height=300,
        )


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

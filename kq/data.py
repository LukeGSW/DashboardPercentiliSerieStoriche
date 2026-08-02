"""
=============================================================================
kq.data — Accesso ai dati EODHD e costruzione del pannello
=============================================================================
Responsabilita':
    - chiamate HTTP a EODHD con rate limiting rispettoso del piano
    - download multi-thread dello storico di un intero universo
    - assemblaggio del pannello wide (date x ticker) per close e volume
    - cache Streamlit su disco, invalidata su base giornaliera

NOTA SUL BOOTSTRAP:
    L'app non usa parquet precomputati: al primo avvio scarica l'universo.
    Con ~700 ticker, 10 worker e rate limit a 900 chiamate/minuto il cold start
    e' nell'ordine dei 50-70 secondi. Le esecuzioni successive leggono la cache
    su disco (persist="disk"), che sopravvive ai rerun e ai riavvii dello
    script ma NON a un riavvio del container Streamlit Cloud.
=============================================================================
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st

from kq import config as C


# =============================================================================
# RATE LIMITER
# =============================================================================
class _RateLimiter:
    """
    Token bucket a finestra scorrevole, thread-safe.

    Serve a non superare le chiamate/minuto del piano EODHD quando si scarica
    l'universo in parallelo. Senza questo, 10 worker a ~0.35s per chiamata
    producono ~1700 chiamate/minuto e il server inizia a restituire 429.
    """

    def __init__(self, max_calls_per_min: int):
        self.max_calls = max_calls_per_min
        self.window = 60.0
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.window:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self.window - (now - self._calls[0]) + 0.01
            time.sleep(max(sleep_for, 0.01))


_limiter = _RateLimiter(C.RATE_LIMIT_PER_MIN)


def _get_json(url: str, params: dict, timeout: int = C.REQUEST_TIMEOUT):
    """GET con rate limiting e un singolo retry sui 429/5xx."""
    for attempt in range(2):
        _limiter.acquire()
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                time.sleep(2.0)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return None
    return None


def get_api_key() -> str:
    """
    Legge la chiave EODHD dai secrets. Mai hardcoded, mai committata.
    Va estratta nel thread principale e passata ai worker: st.secrets non e'
    garantito accessibile dai thread figli in tutte le versioni di Streamlit.
    """
    try:
        return st.secrets["EODHD_API_KEY"]
    except Exception:
        st.error(
            "Chiave EODHD assente. Aggiungi `EODHD_API_KEY` in "
            "**Settings → Secrets** su Streamlit Cloud, oppure in "
            "`.streamlit/secrets.toml` in locale."
        )
        st.stop()


# =============================================================================
# 1. SINGOLO TICKER — usato dall'analisi single-asset
# =============================================================================
def fetch_eod(ticker: str, start_date: str, api_key: str) -> pd.DataFrame:
    """
    Storico giornaliero di un ticker. Restituisce date / adjusted_close / volume.

    A differenza della versione originale della dashboard, qui il VOLUME viene
    conservato: senza volume non e' possibile calcolare l'ADV in dollari, che e'
    il filtro di liquidita' e il proxy di optionability dello screener.
    """
    data = _get_json(
        f"{C.EODHD_EOD}/{ticker}",
        {
            "api_token": api_key,
            "from": start_date,
            "to": date.today().strftime("%Y-%m-%d"),
            "fmt": "json",
            "period": "d",
        },
    )

    if not data or isinstance(data, dict):
        return pd.DataFrame()

    try:
        df = pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

    if "date" not in df.columns or "adjusted_close" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "adjusted_close": pd.to_numeric(df["adjusted_close"], errors="coerce"),
            "volume": pd.to_numeric(df.get("volume", np.nan), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "adjusted_close"])
    out = out[out["adjusted_close"] > 0]
    return out.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv_cached(ticker: str, start_date: str, _api_key: str) -> pd.DataFrame:
    """Wrapper cacheato per l'analisi single-asset (un ticker alla volta)."""
    return fetch_eod(ticker, start_date, _api_key)


# =============================================================================
# 2. ANAGRAFICA E SNAPSHOT DI MERCATO — costruzione universo (2 chiamate)
# =============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_symbol_list(exchange: str, _api_key: str) -> pd.DataFrame:
    """
    Anagrafica completa di un exchange EODHD (Code, Name, Type, Exchange...).
    Una sola chiamata, cacheata 24h.
    """
    data = _get_json(
        f"{C.EODHD_SYMBOLS}/{exchange}",
        {"api_token": _api_key, "fmt": "json"},
    )
    if not data or isinstance(data, dict):
        return pd.DataFrame()
    return pd.DataFrame(data)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bulk_last_day(exchange: str, _api_key: str) -> pd.DataFrame:
    """
    Ultimo giorno di contrattazione dell'INTERO exchange in una sola chiamata.

    E' il trucco che rende economica la costruzione dell'universo: invece di
    interrogare 26.000 ticker per sapere quali sono liquidi, se ne interroga
    zero e si filtra su questo snapshot.
    """
    data = _get_json(
        f"{C.EODHD_BULK}/{exchange}",
        {"api_token": _api_key, "fmt": "json"},
    )
    if not data or isinstance(data, dict):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty or "code" not in df.columns:
        return pd.DataFrame()

    for col in ("close", "adjusted_close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =============================================================================
# 3. DOWNLOAD MULTI-THREAD DELL'UNIVERSO
# =============================================================================
def download_universe(
    tickers: list[str],
    start_date: str,
    api_key: str,
    progress_cb=None,
) -> dict[str, pd.DataFrame]:
    """
    Scarica in parallelo lo storico di una lista di ticker.

    Il progress callback viene invocato SOLO nel thread principale
    (as_completed gira qui), perche' i widget Streamlit non sono thread-safe.
    """
    out: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    if total == 0:
        return out

    with ThreadPoolExecutor(max_workers=C.MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_eod, t, start_date, api_key): t for t in tickers
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            ticker = futures[fut]
            try:
                df = fut.result()
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                out[ticker] = df
            if progress_cb is not None:
                progress_cb(i, total, ticker)

    return out


def build_panel(raw: dict[str, pd.DataFrame], calendar_anchor: str = "SPY.US"):
    """
    Assembla i dict di serie in due DataFrame wide allineati (date x ticker).

    Il calendario di riferimento e' quello dell'ancora (SPY): tutti i titoli US
    condividono il calendario NYSE, quindi il Trading Day Index calcolato
    sull'indice e' valido per tutti e non serve ricalcolarlo per ticker.

    Il forward fill e' limitato a 3 sedute: serve a colmare halt e sospensioni
    brevi senza far sopravvivere artificialmente i delistati (che restano
    intercettati dal controllo di staleness).
    """
    if not raw:
        return pd.DataFrame(), pd.DataFrame()

    closes = {}
    volumes = {}
    for ticker, df in raw.items():
        s = df.set_index("date")
        closes[ticker] = s["adjusted_close"]
        volumes[ticker] = s["volume"]

    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(index=close.index, columns=close.columns)

    # Restringi al calendario dell'ancora, se presente
    if calendar_anchor in close.columns:
        cal = close[calendar_anchor].dropna().index
        close = close.reindex(cal)
        volume = volume.reindex(cal)

    close = close.ffill(limit=3)
    volume = volume.fillna(0.0)

    return close, volume


@st.cache_data(ttl=3600, persist="disk", show_spinner=False)
def load_screener_panel(
    tickers: tuple[str, ...],
    start_date: str,
    cache_day: str,
    _api_key: str,
):
    """
    Punto di ingresso cacheato per il pannello dello screener.

    `cache_day` (stringa YYYY-MM-DD) non viene usato nel corpo: serve solo a far
    ruotare la cache una volta al giorno. `tickers` e' una tupla perche' deve
    essere hashabile. `_api_key` ha l'underscore per non finire nella chiave
    di cache (e quindi su disco).
    """
    progress = st.progress(0.0, text="Preparazione download universo…")

    def _cb(done: int, total: int, ticker: str):
        progress.progress(
            done / total,
            text=f"Download storico {done}/{total} — ultimo: {ticker}",
        )

    raw = download_universe(list(tickers), start_date, _api_key, progress_cb=_cb)
    progress.progress(1.0, text="Assemblaggio pannello…")
    close, volume = build_panel(raw)
    progress.empty()

    return close, volume


# =============================================================================
# 4. UTILITY
# =============================================================================
_bad_re = re.compile(C.BAD_TICKER_PATTERN)


def is_tradable_symbol(code: str) -> bool:
    """
    Scarta preferred, warrant, unit, right e diritti vari.
    Le classi azionarie legittime (BRK-B, BF-B) restano dentro: il pattern
    esclude solo i suffissi che identificano strumenti non azionari.
    """
    if not isinstance(code, str) or not code:
        return False
    if "." in code:
        return False
    return _bad_re.search(code) is None


def cache_day_key() -> str:
    """
    Chiave di rotazione giornaliera della cache.

    Prima delle 23:00 UTC il dato EOD del giorno corrente non e' ancora
    consolidato su EODHD, quindi la chiave resta quella del giorno precedente:
    evita di invalidare la cache a meta' giornata per riscaricare gli stessi dati.
    """
    now = pd.Timestamp.utcnow()
    ref = now if now.hour >= 23 else now - pd.Timedelta(days=1)
    return ref.strftime("%Y-%m-%d")

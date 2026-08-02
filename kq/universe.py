"""
=============================================================================
kq.universe — Costruzione dell'universo investibile
=============================================================================
Obiettivo: un universo di soli strumenti LIQUIDI e TRADABILI ANCHE IN OPZIONI,
senza small cap, costruito con 2 sole chiamate API e senza liste hardcoded
che invecchiano.

Metodo:
    1. `exchange-symbol-list/US`  -> anagrafica di tutti i simboli US (1 chiamata)
    2. `eod-bulk-last-day/US`     -> prezzo e volume di ieri per TUTTO il mercato
                                      (1 chiamata)
    3. filtro Common Stock, prezzo minimo, ordinamento per dollar volume
    4. top N + universo ETF curato

PERCHE' IL DOLLAR VOLUME:
    EODHD non espone un flag "optionable" nel piano dell'utente (l'add-on
    options e' separato). Il controvalore medio scambiato e' pero' un proxy
    eccellente: negli USA praticamente ogni common stock che scambia oltre
    ~30 M$/giorno ha una catena opzioni con spread trattabili. Ordinare per
    dollar volume ed estrarre i primi 600 nomi produce, di fatto, l'universo
    optionable liquido — e si auto-mantiene senza manutenzione.

BIAS ACCETTATO (esplicito):
    L'universo e' costruito sui membri di OGGI: e' quindi affetto da
    survivorship bias. L'utente lo ha accettato consapevolmente perche' lo
    screener e' un filtro di primo livello, non uno studio. Il bias gonfia la
    storia dei sopravvissuti e spinge le metriche time-series verso il basso;
    per questo la metrica PRIMARIA dello screener e' cross-sectional (rank di
    oggi contro i pari di oggi), che al survivorship bias e' immune.
=============================================================================
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kq import config as C
from kq import data as D


# =============================================================================
# UNIVERSO DI EMERGENZA
# =============================================================================
# Usato solo se `exchange-symbol-list` non e' servito dal piano. Non pretende
# di essere l'S&P 500: e' un nucleo di mega/large cap indiscutibilmente
# optionable, sufficiente a far funzionare lo screener in modalita' degradata.
FALLBACK_LARGECAPS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK-B",
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "WMT", "PG", "HD", "COST",
    "ORCL", "CVX", "MRK", "ABBV", "KO", "PEP", "BAC", "CRM", "AMD", "NFLX",
    "ADBE", "TMO", "LIN", "MCD", "CSCO", "ACN", "ABT", "WFC", "DIS", "INTC",
    "QCOM", "TXN", "DHR", "VZ", "INTU", "AMGN", "CAT", "PFE", "NOW", "SPGI",
    "IBM", "GE", "UBER", "CMCSA", "AMAT", "UNP", "PM", "RTX", "GS", "NEE",
    "LOW", "HON", "ISRG", "BKNG", "T", "BLK", "SYK", "ELV", "LMT", "PLD",
    "MDT", "AXP", "TJX", "MU", "ADI", "VRTX", "MS", "SCHW", "CI", "BSX",
    "REGN", "PANW", "LRCX", "KLAC", "SBUX", "MDLZ", "GILD", "ADP", "CB", "ZTS",
    "BMY", "SO", "MMC", "DE", "PGR", "FI", "EOG", "SLB", "DUK", "APD",
    "CME", "ITW", "NOC", "GD", "MO", "CL", "WM", "EMR", "MCK", "TGT",
    "PYPL", "SHOP", "SQ", "COIN", "PLTR", "SNOW", "CRWD", "DDOG", "MRNA", "F",
    "GM", "DAL", "AAL", "UAL", "CCL", "NCLH", "MAR", "ABNB", "RIVN", "LCID",
]


def _etf_frame() -> pd.DataFrame:
    """Universo ETF curato come DataFrame con benchmark gia' assegnato."""
    rows = [
        {
            "ticker": f"{t}.US",
            "symbol": t,
            "bucket": "ETF",
            "categoria": cat,
            "benchmark": f"{bench}.US",
            "nome": t,
        }
        for t, cat, bench in C.ETF_UNIVERSE
    ]
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def build_universe(
    n_stocks: int,
    min_price: float,
    cache_day: str,
    _api_key: str,
) -> pd.DataFrame:
    """
    Costruisce l'universo completo (ETF curati + top N common stock liquide).

    Restituisce un DataFrame con: ticker, symbol, bucket, categoria, benchmark,
    nome, prezzo_snapshot, dollar_volume_snapshot.

    Il benchmark dei singoli titoli e' inizialmente SPY: viene poi riassegnato
    per massima correlazione in `assign_benchmarks_by_correlation`, che gira
    quando il pannello dei prezzi e' disponibile.
    """
    etf = _etf_frame()

    symbols = D.fetch_symbol_list("US", _api_key)
    bulk = D.fetch_bulk_last_day("US", _api_key)

    stocks = pd.DataFrame()

    if not symbols.empty and not bulk.empty and "Code" in symbols.columns:
        common = symbols.copy()

        # Solo azioni ordinarie sui listini principali. Se l'anagrafica non
        # espone i campi attesi si prosegue senza quel filtro invece di
        # far fallire tutta la costruzione dell'universo.
        if "Type" in common.columns:
            common = common[common["Type"].astype(str).str.strip() == "Common Stock"]
        if "Exchange" in common.columns:
            common = common[
                common["Exchange"].astype(str).str.upper().isin(
                    ["NYSE", "NASDAQ", "NYSE ARCA", "NYSE MKT", "BATS", "AMEX", "NYSE AMERICAN"]
                )
            ]

        common = common[common["Code"].apply(D.is_tradable_symbol)].copy()
        common["_code_u"] = common["Code"].astype(str).str.upper()

        snap = bulk.copy()
        snap["code"] = snap["code"].astype(str).str.upper()
        price_col = "adjusted_close" if "adjusted_close" in snap.columns else "close"
        snap = snap[["code", price_col, "volume"]].rename(columns={price_col: "price"})
        snap["dollar_volume"] = snap["price"] * snap["volume"]
        snap = snap.drop_duplicates(subset="code", keep="first")

        merged = common.merge(snap, left_on="_code_u", right_on="code", how="inner")
        merged = merged[merged["price"] >= min_price]
        merged = merged.dropna(subset=["dollar_volume"])
        merged = merged.sort_values("dollar_volume", ascending=False).head(n_stocks)

        stocks = pd.DataFrame(
            {
                "ticker": merged["Code"].astype(str) + ".US",
                "symbol": merged["Code"].astype(str),
                "bucket": "Azione",
                "categoria": "Large Cap US",
                "benchmark": f"{C.DEFAULT_BENCHMARK}.US",
                "nome": merged.get("Name", merged["Code"]).astype(str),
                "prezzo_snapshot": merged["price"].values,
                "dollar_volume_snapshot": merged["dollar_volume"].values,
            }
        ).reset_index(drop=True)

    if stocks.empty:
        # Modalita' degradata: nucleo statico di mega/large cap
        stocks = pd.DataFrame(
            {
                "ticker": [f"{t}.US" for t in FALLBACK_LARGECAPS],
                "symbol": FALLBACK_LARGECAPS,
                "bucket": "Azione",
                "categoria": "Large Cap US",
                "benchmark": f"{C.DEFAULT_BENCHMARK}.US",
                "nome": FALLBACK_LARGECAPS,
                "prezzo_snapshot": pd.NA,
                "dollar_volume_snapshot": pd.NA,
            }
        )

    universe = pd.concat([etf, stocks], ignore_index=True)
    universe = universe.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
    return universe


def required_benchmark_tickers(universe: pd.DataFrame) -> list[str]:
    """
    Tutti i ticker che devono essere presenti nel pannello anche se non fanno
    parte dell'universo scansionato: benchmark settoriali, SPY e i benchmark
    espliciti degli ETF.
    """
    needed = {f"{b}.US" for b in C.SECTOR_BENCHMARKS}
    needed.add(f"{C.DEFAULT_BENCHMARK}.US")
    needed.update(universe["benchmark"].dropna().unique().tolist())
    return sorted(needed)


def assign_benchmarks_by_correlation(
    universe: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = C.WIN_BETA,
) -> pd.DataFrame:
    """
    Assegna a ogni AZIONE il settoriale SPDR con cui e' massimamente correlata.

    Perche' non il settore GICS ufficiale:
        - il piano EODHD dell'utente non serve i fundamentals sugli indici
          (403 Forbidden), quindi il dato GICS non e' disponibile a costo zero;
        - ai fini del residuo idiosincratico serve il benchmark che SPIEGA il
          titolo, non la sua etichetta di classificazione. Una utility con
          profilo da growth va meglio neutralizzata con XLK che con XLU.

    Gli ETF mantengono il benchmark curato in config: e' un'informazione
    strutturale nota e piu' affidabile della correlazione.
    """
    out = universe.copy()

    bench_cols = [f"{b}.US" for b in C.SECTOR_BENCHMARKS if f"{b}.US" in returns.columns]
    if not bench_cols:
        return out

    recent = returns.tail(window)
    stock_mask = out["bucket"] == "Azione"
    stock_tickers = [t for t in out.loc[stock_mask, "ticker"] if t in recent.columns]
    if not stock_tickers:
        return out

    sub = recent[stock_tickers]
    corr = pd.DataFrame(index=stock_tickers, columns=bench_cols, dtype=float)
    for b in bench_cols:
        corr[b] = sub.corrwith(recent[b])

    best = corr.idxmax(axis=1)
    best_val = corr.max(axis=1)

    # Se nessun settoriale spiega il titolo in modo decente, resta su SPY
    best = best.where(best_val >= 0.30, f"{C.DEFAULT_BENCHMARK}.US")

    out.loc[out["ticker"].isin(best.index), "benchmark"] = (
        out.loc[out["ticker"].isin(best.index), "ticker"].map(best)
    )
    out["corr_benchmark"] = out["ticker"].map(best_val)
    return out

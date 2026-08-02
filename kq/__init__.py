"""
Kriterion Quant — Percentile & Anomaly Dashboard.

Pacchetto applicativo della dashboard Streamlit.

Moduli:
    config      costanti, palette, soglie
    data        accesso EODHD, rate limiting, pannello wide, cache
    universe    costruzione dell'universo investibile e assegnazione benchmark
    core        analitiche single-asset (percentili stagionali, regime, forward)
    scanner     motore dello screener cross-sectional
    charts      costruttori Plotly
    ui_single   interfaccia dell'analisi single-asset
    ui_scanner  interfaccia dello screener
"""

__version__ = "2.0.0"

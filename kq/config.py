"""
=============================================================================
kq.config — Costanti, palette e parametri di default
=============================================================================
Punto unico di configurazione della dashboard. Nessuna logica qui dentro:
solo valori che l'utente o gli altri moduli possono voler cambiare.
=============================================================================
"""

from __future__ import annotations

# =============================================================================
# ENDPOINT EODHD
# =============================================================================
EODHD_BASE = "https://eodhd.com/api"
EODHD_EOD = f"{EODHD_BASE}/eod"                          # storico per singolo ticker
EODHD_BULK = f"{EODHD_BASE}/eod-bulk-last-day"           # tutto il mercato in 1 chiamata
EODHD_SYMBOLS = f"{EODHD_BASE}/exchange-symbol-list"     # anagrafica simboli per exchange

# Rate limit del piano: 1000 chiamate/minuto, 100.000/giorno.
# Restiamo volutamente sotto: il download dell'universo non e' mai urgente
# quanto il rischio di farsi throttlare a meta' bootstrap.
RATE_LIMIT_PER_MIN = 900
MAX_WORKERS = 10
REQUEST_TIMEOUT = 30

# =============================================================================
# PARAMETRI DI DEFAULT — SCREENER
# =============================================================================
# Storia scaricata per lo screener. Il costo di una chiamata EODHD e' identico
# dal 2015 o dal 2020 (cambia solo la dimensione del JSON): l'unico costo reale
# della storia lunga e' la RAM, ed e' trascurabile a questo universo.
SCREENER_START_DEFAULT = "2015-01-01"

# Universo: quanti common stock tenere, ordinati per dollar volume decrescente.
# Il dollar volume e' il miglior proxy disponibile di "ha opzioni liquide"
# senza dover pagare l'add-on options di EODHD.
UNIVERSE_N_STOCKS = 600
UNIVERSE_MIN_PRICE = 10.0            # sotto i $10 le catene opzioni sono inutilizzabili
UNIVERSE_MIN_ADV_USD = 30_000_000    # ADV20 minimo in dollari

# Finestre di calcolo (in trading days)
WIN_VOL_SHORT = 20      # realized vol "corrente"
WIN_VOL_LONG = 63       # sigma di riferimento per la normalizzazione
WIN_BETA = 252          # stima beta/correlazione vs benchmark
WIN_VELOCITY = 10       # orizzonte per la velocita' di rank
WIN_LIQUIDITY = 20      # ADV

# Orizzonti disponibili per il rendimento di riferimento dello screener
HORIZONS = {
    "YTD": None,        # None = da inizio anno solare
    "1 mese": 21,
    "3 mesi": 63,
    "6 mesi": 126,
    "12 mesi": 252,
}
# Sotto questa soglia di trading day dell'anno, YTD e' troppo corto per
# essere informativo: si ripiega automaticamente su 3 mesi.
YTD_MIN_TDI = 25

# =============================================================================
# SOGLIE DI CLASSIFICAZIONE SETUP
# =============================================================================
# Volutamente permissive: lo screener e' un filtro di primo livello, non un
# sistema di trading. Meglio 40 candidati da vagliare che 3 gia' "validati".
TH_RESID_Z = 1.5          # |z| idiosincratico oltre cui la dislocazione e' notevole
TH_XS_TAIL = 12.0         # percentile cross-sectional che definisce la coda
TH_VOL_LOW = 15.0         # percentile di realized vol: compressione
TH_VOL_HIGH = 85.0        # percentile di realized vol: premio ricco
TH_FRESH_DAYS = 15        # giorni in coda oltre i quali l'anomalia non e' piu' "fresca"
TH_STALE_DAYS = 40        # giorni in coda oltre i quali e' un trend, non un'anomalia

# =============================================================================
# CONTROLLI QUALITA' DATO
# =============================================================================
# Su Momentum Track e' emerso che le serie EODHD rotte (concambi non gestiti,
# fusioni, split mancati) producono crolli spuri. Su uno screener che ordina
# per estremita' della dislocazione, quelle serie finiscono DIRETTAMENTE in cima
# alla classifica: il bias dei dati sporchi favorisce sempre la tesi contrarian.
# Questi filtri sono quindi attivi di default, non opzionali.
QC_MAX_ABS_DAILY_RET = 0.35   # |ret| giornaliero oltre cui la serie e' sospetta
QC_MAX_STALENESS_DAYS = 6     # giorni di calendario dall'ultima quotazione
QC_MIN_OBS = 260              # osservazioni minime per calcolare qualunque cosa
QC_MAX_ZERO_VOL_DAYS = 5      # giorni a volume zero negli ultimi 20

# =============================================================================
# UNIVERSO ETF CURATO
# =============================================================================
# Volutamente SNELLO: un ETF per ogni classe di attivo definita, non un catalogo.
# Niente tematici (ARKK, TAN, LIT...), niente industry (SMH, XBI, KRE...),
# niente fattoriali (MTUM, QUAL...): sono sotto-insiemi rumorosi che duplicano
# l'informazione gia' presente nei singoli titoli e nei settoriali.
# Tutti optionable con catene liquide.
#
# I settoriali SPDR restano tutti e 11 perche' oltre a essere una classe di
# attivo sono i BENCHMARK strutturali per il residuo idiosincratico dei singoli
# titoli: senza di loro non e' calcolabile la dislocazione depurata dal beta.
ETF_UNIVERSE: list[tuple[str, str, str]] = [
    # (ticker, categoria, benchmark)

    # --- Azionario USA ---
    ("SPY", "Azionario USA", "SPY"),
    ("QQQ", "Azionario USA", "SPY"),
    ("IWM", "Azionario USA", "SPY"),
    ("DIA", "Azionario USA", "SPY"),
    ("MDY", "Azionario USA", "SPY"),
    ("RSP", "Azionario USA", "SPY"),

    # --- Settori azionari USA (anche benchmark dei singoli titoli) ---
    ("XLK", "Settore USA", "SPY"),
    ("XLF", "Settore USA", "SPY"),
    ("XLE", "Settore USA", "SPY"),
    ("XLV", "Settore USA", "SPY"),
    ("XLI", "Settore USA", "SPY"),
    ("XLY", "Settore USA", "SPY"),
    ("XLP", "Settore USA", "SPY"),
    ("XLU", "Settore USA", "SPY"),
    ("XLB", "Settore USA", "SPY"),
    ("XLRE", "Settore USA", "SPY"),
    ("XLC", "Settore USA", "SPY"),

    # --- Azionario internazionale ---
    ("EFA", "Azionario Internazionale", "EFA"),
    ("VGK", "Azionario Internazionale", "EFA"),
    ("EWJ", "Azionario Internazionale", "EFA"),
    ("EEM", "Azionario Internazionale", "EEM"),
    ("FXI", "Azionario Internazionale", "EEM"),
    ("INDA", "Azionario Internazionale", "EEM"),
    ("EWZ", "Azionario Internazionale", "EEM"),
    ("ACWI", "Azionario Internazionale", "SPY"),

    # --- Obbligazionario (curva + credito) ---
    ("SHY", "Obbligazionario", "TLT"),
    ("IEF", "Obbligazionario", "TLT"),
    ("TLT", "Obbligazionario", "TLT"),
    ("AGG", "Obbligazionario", "TLT"),
    ("TIP", "Obbligazionario", "TLT"),
    ("MUB", "Obbligazionario", "TLT"),
    ("LQD", "Obbligazionario", "LQD"),
    ("HYG", "Obbligazionario", "HYG"),
    ("EMB", "Obbligazionario", "LQD"),

    # --- Materie prime ---
    ("GLD", "Materie prime", "GLD"),
    ("SLV", "Materie prime", "SLV"),
    ("PPLT", "Materie prime", "GLD"),
    ("CPER", "Materie prime", "DBC"),
    ("USO", "Materie prime", "USO"),
    ("UNG", "Materie prime", "USO"),
    ("DBC", "Materie prime", "DBC"),
    ("DBA", "Materie prime", "DBC"),

    # --- Valute ---
    ("UUP", "Valute", "UUP"),
    ("FXE", "Valute", "UUP"),
    ("FXY", "Valute", "UUP"),
    ("FXB", "Valute", "UUP"),
    ("FXF", "Valute", "UUP"),

    # --- Immobiliare ---
    ("VNQ", "Immobiliare", "XLRE"),
    ("IYR", "Immobiliare", "XLRE"),
]

# Settoriali usati come benchmark per l'assegnazione automatica dei singoli titoli.
# L'assegnazione avviene per massima correlazione sui rendimenti giornalieri:
# non serve il dato fondamentale GICS (che il piano EODHD dell'utente non copre,
# fundamentals/GSPC.INDX risponde 403) e in piu' il benchmark scelto e' quello
# che *spiega davvero* il titolo, non l'etichetta ufficiale.
SECTOR_BENCHMARKS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

# Benchmark di ultima istanza se la correlazione non e' calcolabile
DEFAULT_BENCHMARK = "SPY"

# =============================================================================
# SUFFISSI DA ESCLUDERE DALL'ANAGRAFICA US
# =============================================================================
# EODHD codifica preferred/warrant/unit/right come suffissi del ticker.
# Le classi azionarie legittime (BRK-B, GOOG/GOOGL) vanno invece tenute.
BAD_TICKER_PATTERN = r"-(P[A-Z]?|WT[A-Z]?|WS[A-Z]?|U|UN|RT|R|CL|CV)$"

# =============================================================================
# PALETTE COLORI KRITERION QUANT
# =============================================================================
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
    "neutral": "#8A8FA3",
}

# Colori per famiglia di setup (usati in tabella e scatter)
SETUP_COLORS = {
    "MR-LONG": "#00D26A",
    "MR-SHORT": "#FF6B6B",
    "TREND-UP": "#4CC9F0",
    "TREND-DN": "#F72585",
    "VOL-LOW": "#FFC107",
    "VOL-HIGH": "#9D4EDD",
    "—": "#8A8FA3",
}

DEFAULT_MAX_TRADING_DAYS = 260

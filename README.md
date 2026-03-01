# 📊 YTD Seasonality & Percentile Quant Dashboard

Parte del progetto **[Kriterion Quant](https://kriterionquant.com)** — piattaforma educativa e operativa dedicata alla finanza quantitativa.

---

## 🎯 Scopo

Dashboard interattiva in **Streamlit** per l'analisi quantitativa della stagionalità dei rendimenti YTD (Year-To-Date).

L'app consente di:
- Inserire qualsiasi ticker supportato da EODHD (Stocks, ETF, Crypto, Forex, Indici).
- Visualizzare se la performance dell'anno corrente rientra nella **normalità statistica storica** oppure si trova in una fase di **anomalia**.
- Confrontare l'equity YTD corrente con le **bande di percentile** (5°, 25°, 50°, 75°, 95°) calcolate su tutti gli anni precedenti.

**Non** vengono tracciate le singole equity line degli anni passati ("spaghetti chart"): lo storico serve esclusivamente per costruire le bande statistiche.

---

## 🛠️ Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Frontend / UI | Streamlit |
| Grafici | Plotly Graph Objects |
| Dati | EODHD API (Adjusted Close giornaliero) |
| Linguaggio | Python 3.10+ |

---

## 🚀 Deploy su Streamlit Cloud

### 1. Fork / Clone del repository

```bash
git clone https://github.com/<tuo-username>/<tuo-repo>.git
cd <tuo-repo>
```

### 2. Configura la chiave API EODHD nei Secrets di Streamlit Cloud

Sul portale [share.streamlit.io](https://share.streamlit.io), nella sezione **Settings → Secrets** della tua app, aggiungi:

```toml
EODHD_API_KEY = "la_tua_chiave_api_eodhd"
```

> ⚠️ **Non inserire mai la chiave API nel codice sorgente o in file committati sul repository.**  
> Il file `.streamlit/secrets.toml` è già escluso dal `.gitignore`.

### 3. Deploy

Collega il repository GitHub su Streamlit Cloud e imposta `app.py` come file principale. Il deploy avviene automaticamente ad ogni push sul branch principale.

---

## 📋 Utilizzo

| Parametro Sidebar | Descrizione | Default |
|---|---|---|
| **Ticker** | Simbolo nel formato EODHD (es. `SPY.US`, `BTC-USD.CC`, `ENI.MI`) | `SPY.US` |
| **Inizio storico** | Data di partenza per il download dei dati | `2000-01-01` |

### Formato Ticker EODHD
- **ETF / Stock US:** `SPY.US`, `AAPL.US`, `QQQ.US`
- **Azioni europee:** `ENI.MI`, `AIR.PA`, `SAN.MC`
- **Crypto:** `BTC-USD.CC`, `ETH-USD.CC`
- **Indici:** `GSPC.INDX` (S&P 500), `GDAXI.INDX` (DAX)

---

## 📐 Logica di Calcolo

1. **Download** della serie storica (Adjusted Close) via EODHD API.
2. **Calcolo YTD:** per ogni anno, rendimento cumulativo % rispetto all'ultima chiusura dell'anno precedente.
3. **Mappatura DOY:** ogni serie annuale viene mappata su un asse Day-of-Year (1–366). I giorni senza trading (weekend/festività) vengono riempiti con `ffill`.
4. **Percentili:** calcolati su tutti gli anni storici (escluso l'anno corrente) per ogni DOY.
5. **Percentile corrente:** ranking puntuale dell'YTD corrente rispetto alla distribuzione storica nello stesso DOY.

---

## 📁 Struttura Repository

```
.
├── app.py              # Applicazione principale Streamlit
├── requirements.txt    # Dipendenze Python
├── .gitignore          # File esclusi da Git
└── README.md           # Questa documentazione
```

---

## 📜 Licenza

Progetto educativo — Kriterion Quant © 2025. Tutti i diritti riservati.

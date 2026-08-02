# 📊 Percentile & Anomaly Dashboard

Parte del progetto **[Kriterion Quant](https://kriterionquant.com)** — piattaforma educativa e operativa dedicata alla finanza quantitativa.

---

## 🎯 Scopo

Dashboard Streamlit in due modalità complementari:

| Modalità | Cosa fa |
|---|---|
| 🔍 **Screener multi-asset** | Filtro di primo livello su ~650 strumenti USA liquidi e optionable. Produce una **lista di candidati**, non verdetti. |
| 📈 **Analisi singolo asset** | Lo studio completo su un ticker: percentili stagionali, z-score, dinamiche dell'anomalia, regime clustering, forward returns condizionali. |

Il flusso naturale è **screener → scegli un candidato → analisi completa**. Ogni riga della tabella dello screener ha un pulsante che apre direttamente l'analisi approfondita di quel ticker.

---

## 🔍 Come funziona lo screener

Cerca strumenti la cui performance si è staccata da quella del proprio benchmark **più di quanto la loro volatilità giustifichi**, e distingue chi si sta stabilizzando da chi sta ancora scendendo.

### Le tre scelte di disegno che contano

**1. La metrica primaria è cross-sectional, non time-series.**
Con storia dal 2015 si hanno ~11 osservazioni per trading day: un percentile storico su 11 punti ha granularità ~9 punti percentuali ed è inservibile come soglia. Il rank contro i ~650 pari di *oggi* ha ~650 campioni, è robusto con qualunque profondità di storia ed è immune al survivorship bias dell'universo. Il percentile storico resta come colonna di contesto.

**2. La dislocazione è idiosincratica e normalizzata per volatilità.**
Un titolo giù del 20% non è dislocato se il suo settore è giù del 18%: è beta. Il segnale è il **residuo** rispetto al benchmark, diviso per la volatilità idiosincratica attesa sull'orizzonte — così un ETF obbligazionario e un semiconduttore diventano confrontabili sulla stessa scala.

**3. La direzione si misura con il momentum residuo, non con la velocity di rank.**
Il rank percentile è limitato in [0, 100] e **satura**: un titolo già all'ultimo posto ha velocity esattamente zero proprio mentre sta crollando, e verrebbe letto come "stabilizzato". Il momentum residuo di breve è continuo e non limitato, quindi separa davvero *ha smesso di scendere* da *è già in fondo*.

### I quattro setup

Nascono dall'incrocio di due assi ortogonali — **ampiezza** della dislocazione e **direzione** attuale — e sono esaustivi e mutuamente esclusivi:

| Setup | Significato |
|---|---|
| `MR-LONG` | Dislocato sotto il benchmark, ma **ha smesso** di peggiorare |
| `TREND-DN` | Dislocato sotto e **ancora in caduta** — non è un rimbalzo |
| `MR-SHORT` | Dislocato sopra, ma **ha smesso** di salire |
| `TREND-UP` | Dislocato sopra e **ancora in corsa** |

A questi si affianca un flag indipendente sul regime di volatilità realizzata (`COMPRESSA` / `RICCA`) e la struttura in opzioni coerente con la combinazione setup × volatilità.

### Cosa lo screener NON fa

Lo `Score` è un'**euristica di ordinamento** trasparente e decomponibile (i pesi delle cinque componenti sono visibili per ogni candidato), **non** un edge validato: nessun backtest, nessun confronto con un null, nessun costo di transazione. La validazione si fa a valle, sul singolo candidato.

---

## 🛡️ Controlli qualità dato

Le serie EODHD corrotte (concambi non gestiti, fusioni, split mancati) producono crolli spuri. Su uno screener che **ordina per estremità della dislocazione** quelle serie non finiscono da qualche parte nella lista: finiscono **in cima**. Il bias dei dati sporchi favorisce quindi sistematicamente la tesi contrarian.

Per questo i filtri sono attivi di default e si applicano **prima** di qualunque ranking:

| Controllo | Soglia |
|---|---|
| Salto giornaliero anomalo | `\|ret\| > 35%` sugli ultimi 252 giorni |
| Serie ferma (delisting, sospensione) | ultima quotazione > 6 giorni |
| Storia insufficiente | < 260 osservazioni |
| Assenza di scambi | > 5 giorni a volume zero negli ultimi 20 |

Gli strumenti scartati sono elencati in un pannello dedicato con il motivo. Il prezzo da pagare è che anche un crollo *legittimo* oltre il 35% viene escluso: è una scelta conservativa consapevole.

---

## 🌐 Costruzione dell'universo

Nessuna lista hardcoded che invecchia: l'universo si costruisce con **2 chiamate API**.

1. `exchange-symbol-list/US` → anagrafica completa dei simboli US
2. `eod-bulk-last-day/US` → prezzo e volume di tutto il mercato in una chiamata
3. Filtro *Common Stock* sui listini principali, esclusione di preferred / warrant / unit / right, prezzo minimo
4. Ordinamento per **dollar volume** decrescente → top N

**Perché il dollar volume.** EODHD non espone un flag "optionable" (l'add-on options è separato). Il controvalore medio scambiato è però un proxy eccellente: negli USA praticamente ogni common stock sopra i ~30 M$/giorno ha una catena opzioni con spread trattabili. I primi 600 nomi per dollar volume *sono*, di fatto, l'universo optionable liquido — e si auto-mantiene.

A questi si aggiungono **49 ETF curati**, uno per ogni classe di attivo definita (azionario USA, settori USA, azionario internazionale, obbligazionario, materie prime, valute, immobiliare). Niente tematici, industry o fattoriali: duplicherebbero l'informazione già presente nei singoli titoli.

**Assegnazione del benchmark.** Ogni azione viene assegnata al settoriale SPDR con cui è massimamente correlata sui rendimenti giornalieri. Non si usa il settore GICS ufficiale per due motivi: il piano EODHD non serve i fundamentals sugli indici (403 Forbidden), e ai fini del residuo serve comunque il benchmark che *spiega* il titolo, non la sua etichetta di classificazione.

> ⚠️ **Survivorship bias.** L'universo è costruito sui membri di oggi, quindi ne è affetto. È una scelta consapevole: la metrica primaria dello screener è cross-sectional e al survivorship bias è immune, e l'analisi seria si fa comunque a valle sul singolo ticker.

---

## 🛠️ Stack tecnologico

| Componente | Tecnologia |
|---|---|
| Frontend / UI | Streamlit |
| Grafici | Plotly (tema scuro) |
| Dati | EODHD API (adjusted close + volume giornalieri) |
| Calcolo | pandas / numpy vettorizzati, scikit-learn per il clustering |
| Linguaggio | Python 3.10+ |

---

## 🚀 Deploy su Streamlit Cloud

### 1. Clona il repository

```bash
git clone https://github.com/<tuo-username>/<tuo-repo>.git
```

### 2. Configura la chiave API nei Secrets

Su [share.streamlit.io](https://share.streamlit.io), in **Settings → Secrets**:

```toml
EODHD_API_KEY = "la_tua_chiave_api_eodhd"
```

> ⚠️ **Non inserire mai la chiave nel codice sorgente o in file committati.**
> `.streamlit/secrets.toml` è già escluso dal `.gitignore`.

### 3. Deploy

Collega il repository e imposta `app.py` come file principale.

### Cold start

L'app non usa parquet precomputati: al primo avvio scarica l'universo. Con ~650 ticker, 10 worker e rate limit a 900 chiamate/minuto il cold start è nell'ordine dei **50-70 secondi**, con barra di avanzamento. Le esecuzioni successive leggono la cache su disco, che sopravvive ai rerun ma non a un riavvio del container Streamlit Cloud.

---

## 📁 Struttura del repository

```
.
├── app.py                  # entry point: sidebar, routing fra le due modalità
├── kq/
│   ├── config.py           # costanti, palette, soglie, universo ETF curato
│   ├── data.py             # EODHD, rate limiting, download multi-thread, cache
│   ├── universe.py         # costruzione universo, assegnazione benchmark
│   ├── core.py             # analitiche single-asset (percentili, regime, forward)
│   ├── scanner.py          # motore dello screener cross-sectional
│   ├── charts.py           # costruttori Plotly
│   ├── ui_single.py        # interfaccia analisi single-asset
│   └── ui_scanner.py       # interfaccia screener
├── requirements.txt
└── README.md
```

---

## 📐 Logica di calcolo (analisi single-asset)

1. **Download** della serie storica (adjusted close) via EODHD.
2. **Calcolo YTD:** base = **primo prezzo dell'anno corrente** (convenzione TradingView), che evita le distorsioni da gap di capodanno.
3. **Trading Day Index:** ogni serie annuale è mappata sulle sedute effettive (1, 2, 3, …) invece che sul giorno solare, eliminando i disallineamenti da anni bisestili e festività variabili.
4. **Percentili:** calcolati su tutti gli anni storici escluso il corrente, con un minimo di 3 anni per trading day.
5. **Percentile corrente:** ranking puntuale dell'YTD sull'**ultimo trading day reale** — mai su valori forward-filled.
6. **Volatilità:** sempre sui rendimenti giornalieri veri, mai sulla variazione del cumulato YTD.
7. **Max drawdown:** geometrico, su equity curve base 100.
8. **Forward returns:** il cross-year è gestito con compounding geometrico esatto.

### Variante causale

`compute_percentiles_walkforward` calcola le bande usando **solo gli anni strettamente precedenti** a quello valutato. `compute_percentiles` esclude l'anno corrente ma tiene tutti gli altri: per il segnale live è corretto (il futuro non esiste ancora), ma per valutare storicamente il segnale introdurrebbe look-ahead.

---

## 📋 Formato ticker EODHD

| Tipo | Esempio |
|---|---|
| ETF / azioni US | `SPY.US`, `AAPL.US`, `QQQ.US` |
| Azioni europee | `ENI.MI`, `AIR.PA`, `SAN.MC` |
| Crypto | `BTC-USD.CC`, `ETH-USD.CC` |
| Indici | `GSPC.INDX`, `GDAXI.INDX`, `VIX.INDX` |

---

## 📜 Licenza

Progetto educativo — Kriterion Quant © 2025. Tutti i diritti riservati.

⚠️ Strumento a scopo educativo e di ricerca. Non costituisce consulenza finanziaria.

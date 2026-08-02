# 📊 Percentile & Anomaly Dashboard

Parte del progetto **[Kriterion Quant](https://kriterionquant.com)** — piattaforma educativa e operativa dedicata alla finanza quantitativa.

---

## 🎯 Scopo

Dashboard Streamlit in tre modalità complementari:

| Modalità | Cosa fa |
|---|---|
| 📈 **Analisi singolo asset** | Lo studio completo su un ticker: percentili stagionali, z-score, dinamiche dell'anomalia, regime clustering, forward returns condizionali. |
| 🔍 **Screener multi-asset** | Filtro di primo livello su ~650 strumenti USA liquidi e optionable. Produce una **lista di candidati**, non verdetti. |
| 🧪 **Validazione setup** | Misura storicamente i setup dello screener contro un null, walk-forward e al netto dei costi. |

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

### I quattro stati — e perché il nome non dice la strategia

Gli stati nascono dall'incrocio di due assi ortogonali (**ampiezza** della dislocazione e **verso** attuale del movimento) e sono esaustivi e mutuamente esclusivi. Descrivono una **condizione osservata**, non una tesi: la direzione operativa arriva dalla validazione e vive in `config.SETUP_VERDETTO`.

| Stato | Condizione | Azione | Evidenza |
|---|---|---|---|
| `↑↑ ESTESO` | Dislocato sopra, **ancora in accelerazione** | **RIBASSISTA** | confermata |
| `↑ ESAURITO` | Dislocato sopra, momentum esaurito | RIBASSISTA | debole |
| `↓ STABILIZZATO` | Dislocato sotto, **ha smesso** di scendere | nessuna | assente |
| `↓↓ IN CADUTA` | Dislocato sotto, ancora in caduta | nessuna | instabile |

**Sì, gli strumenti estesi al rialzo si trattano al ribasso.** L'event study (84 celle, 7 orizzonti) lo misura: `↑↑ ESTESO` sottoperforma l'universo su 7 orizzonti su 7, con dose-risposta sia sulla selettività sia sull'orizzonte e out-of-sample più forte dell'in-sample (~−6%/anno a 20 sedute, t −2,48, q 0,105). Sui dislocati al ribasso non c'è invece nulla — ed è una conclusione robusta, perché il survivorship bias gonfia proprio quella tesi e nemmeno così emerge qualcosa.

> ⚠️ La direzione di `↑↑ ESTESO` è stata fissata **dopo** aver visto quei dati. È un'ipotesi da confermare in avanti, non un risultato out-of-sample — e l'app lo dichiara in cima alla tabella della validazione.

Il regime di volatilità realizzata (`COMPRESSA` / `RICCA`) è un flag **indipendente**: la colonna `Struttura` incrocia azione validata × volatilità, e per gli stati senza azione propone solo strutture sulla volatilità.

Per aggiornare il verdetto dopo un nuovo studio si modifica la sola tabella `SETUP_VERDETTO`: colonne, strutture in opzioni e segno della tesi nell'event study seguono da soli.

### Cosa lo screener NON fa

Lo `Score` è un'**euristica di ordinamento** trasparente e decomponibile (i pesi delle cinque componenti sono visibili per ogni candidato), **non** un edge validato: nessun backtest, nessun confronto con un null, nessun costo di transazione. La validazione si fa a valle, sul singolo candidato.

---

## 🧪 Validazione walk-forward

Risponde a una domanda sola: quando lo screener ha segnalato un nome, nelle sedute successive quel nome ha fatto meglio di una selezione casuale fatta **lo stesso giorno sullo stesso universo**?

### Il null

Non è l'entrata casuale nel tempo. I setup di mean reversion scattano in modo sproporzionato durante i drawdown: confrontarli con entrate distribuite su tutto il periodo significherebbe accreditare al segnale il recupero del mercato. Il null è la **sezione trasversale contemporanea** — stesse date, stesso holding, nomi estratti dallo stesso universo eleggibile quel giorno:

```
extra_t = media(forward dei segnalati) − media(forward dell'universo)
```

che è anche il rendimento di una posizione equipesata long sui segnalati contro l'universo. Toglie beta ed effetto periodo in un colpo solo, ed è il valore atteso di una selezione casuale di pari numerosità.

A questo si affianca una **banda placebo**: si ripete la selezione con nomi presi davvero a caso, stessa numerosità e stesse date. Un extra dentro quella banda non è un segnale, è quello che produce il caso.

### Garanzie metodologiche

| | |
|---|---|
| **Causalità** | Ogni grandezza usa finestre mobili o espandenti: alla data *t* solo dati fino a *t*. Verificato da test: troncando il futuro i segnali non cambiano di un bit. |
| **Esecuzione ritardata** | Si entra alla chiusura di *t+1*, si esce a *t+1+holding*. |
| **Sovrapposizione** | Newey-West con lag pari al rapporto holding/frequenza. Campionando ogni 5 sedute con holding 20, il *t* ingenuo sarebbe gonfiato del doppio. |
| **Test multipli** | Benjamini-Hochberg su tutte le celle testate (4 setup × livelli × orizzonti). |
| **Costi** | Andata+ritorno configurabile. **L'inferenza gira sul lordo**, il netto è il risultato economico: il costo è uno spostamento deterministico e sottrarlo prima del test gonfierebbe il \|t\| in proporzione al costo ipotizzato. |
| **Stabilità** | Split 2/3 – 1/3. Non essendoci parametri stimati sui dati (le soglie sono costanti fissate a priori), misura persistenza nel tempo, non protegge da overfitting. |

Il semaforo richiede **due** condizioni distinte: il segnale deve distinguersi dalla selezione casuale (domanda statistica, sul lordo) *e* deve restare qualcosa dopo i costi (domanda economica). Un segnale reale ma più piccolo dei costi non è operativo; un extra netto positivo ma indistinguibile dal caso non è un segnale.

### Il limite che non si può togliere

L'universo è costruito sui membri di oggi. Le società dislocate che sono risalite ci sono; quelle andate a zero sono uscite e non compaiono da nessuna parte. **Il bias spinge a favore della tesi contrarian**: un risultato positivo su `MR-LONG` va letto come **limite superiore**, non come stima.

Il modulo riporta la **copertura** (quota di segnalati con un forward calcolabile) proprio per rendere visibile quanto il problema sia invisibile: se è ~100%, nel campione non fallisce mai nessuno. Per trasformarlo in una stima servirebbero la membership storica point-in-time e i prezzi dei delistati.

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
├── app.py                  # entry point: sidebar, routing fra le tre modalità
├── kq/
│   ├── config.py           # costanti, palette, soglie, universo ETF curato
│   ├── data.py             # EODHD, rate limiting, download multi-thread, cache
│   ├── universe.py         # costruzione universo, assegnazione benchmark
│   ├── core.py             # analitiche single-asset (percentili, regime, forward)
│   ├── scanner.py          # motore dello screener cross-sectional
│   ├── validation.py       # event study walk-forward, null, Newey-West, BH
│   ├── state.py            # persistenza impostazioni fra le modalità
│   ├── charts.py           # costruttori Plotly
│   ├── ui_single.py        # interfaccia analisi single-asset
│   ├── ui_scanner.py       # interfaccia screener
│   └── ui_validation.py    # interfaccia validazione
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

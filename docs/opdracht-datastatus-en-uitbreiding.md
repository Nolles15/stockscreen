# Opdracht — datastatus zichtbaar maken, daarna pas uitbreiden

> Werkorder voor Claude Code, opgesteld 21 augustus 2026 na het lezen van
> `engine/`, `config.yaml` en `docs/DIAGNOSE_INSUFFICIENT_DATA.md`.
>
> **Lees eerst `docs/checklist-voor-uitbreiding.md` en
> `docs/landenuitbreiding-2026-08-21.md`.** Dit document is het uitvoerbare deel
> daarvan.
>
> Voer de fases in volgorde uit. Fase 0 is een meting die bepaalt of fase 1 en 2
> in deze vorm kloppen — begin dus niet met code schrijven.

---

## Waarom dit eerst moet

De screener staat op 2.812 tickers waarvan er **713 (25%) geen oordeel** hebben.
In `docs/overzicht-2026-08-21.md` staat de aanname dat dit "grotendeels dode
symbolen" zijn die de archivering opruimt. Die aanname is niet gemeten en is in
strijd met `docs/DIAGNOSE_INSUFFICIENT_DATA.md`, dat bij 799 tickers vaststelde
dat er **2** écht leeg waren en dat de rest door de eigen kwaliteitspoort werd
tegengehouden. Na route A ging dat naar 33 geblokkeerd — 4%.

Er is een derde verklaring die in geen van beide documenten staat en die
waarschijnlijk het grootste deel verklaart: **de rotatie is er nog niet langs
geweest.** Het universum groeide in augustus van 913 naar ~2.760;
`refresh_fundamentals_batch` doet er 250 per nacht; `get_refresh_queue` sorteert
op `data_quality.last_checked ASC NULLS FIRST`. Een ticker die nooit is
opgehaald heeft `last_checked IS NULL`, krijgt geen jaarcijfers, en valt in
`screener.run_ticker` in de tak `if not annual_rows` → `INSUFFICIENT DATA`.

Op het dashboard is dat niet te onderscheiden van een aandeel dat is afgekeurd.
Zolang dat zo is, kun je van geen enkele nieuwe markt vaststellen of hij werkt.
Dat is de reden dat de landenuitbreiding wacht.

---

## Fase 0 — Meten (geen code)

**Doel:** vaststellen waaruit de 713 bestaan, voordat er iets verandert.

Draai:

```bash
python scripts/gaps_analyze.py --sample 12
```

en de live-variant:

```bash
curl -k -s https://stockscreen-janco.fly.dev/api/gaps-report \
  | jq 'group_by(.primary_blocker) | map({b: .[0].primary_blocker, n: length}) | sort_by(-.n)'
```

Voeg daar één query aan toe die het instrument nu niet beantwoordt — hoeveel
tickers zijn simpelweg nog nooit geprobeerd:

```sql
SELECT
  COUNT(*) FILTER (WHERE dq.last_checked IS NULL)                    AS nooit_geprobeerd,
  COUNT(*) FILTER (WHERE dq.last_checked IS NOT NULL
                     AND dq.data_status IN ('bad','missing'))        AS poort_afgekeurd,
  COUNT(*) FILTER (WHERE dq.last_checked IS NOT NULL
                     AND dq.data_status NOT IN ('bad','missing')
                     AND cs.signal = 'INSUFFICIENT DATA')            AS model_kan_niet,
  COUNT(*)                                                           AS totaal_actief
FROM stocks s
LEFT JOIN data_quality dq       ON dq.ticker = s.ticker
LEFT JOIN calculated_scores cs  ON cs.ticker = s.ticker
WHERE s.active = 1 AND s.presumed_delisted_at IS NULL;
```

**Rapporteer de uitkomst voordat je verder gaat.** De vervolgstap hangt ervan af:

| Uitkomst | Betekenis | Doen |
|---|---|---|
| `nooit_geprobeerd` is groot (>300) | De rotatie loopt achter, niets is stuk | Fase 1 + 2 gewoon uitvoeren; het getal daalt vanzelf |
| `poort_afgekeurd` is groot | De poort keurt opnieuw te breed af | Fase 1 + 2, en daarna een nieuwe route-A-ronde vóór enige uitbreiding |
| `model_kan_niet` is groot | Nieuwe categorie onwaardeerbare bedrijven | Fase 1 + 2, en apart onderzoeken welke soort bedrijven dit zijn |

Leg de uitkomst vast in `docs/DIAGNOSE_INSUFFICIENT_DATA.md` als nieuwe sectie
met meetdatum en universumgrootte, in dezelfde stijl als de bestaande §5b.

---

## Fase 1 — `INSUFFICIENT DATA` opsplitsen in drie toestanden

**Probleem.** In `engine/screener.py` wordt hetzelfde label op vijf plaatsen
gezet, voor vijf verschillende oorzaken:

| Regel (bij benadering) | Voorwaarde | Werkelijke oorzaak |
|---|---|---|
| ~248 | `not annual_rows` | geen jaarcijfers — óf nooit opgehaald, óf echt leeg |
| ~261 | `dq_status in ("bad", "missing")` | kwaliteitspoort keurt af |
| ~337 | `fv_price_ratio < 0.1 or > 10.0` | vermoedelijke schaal-/eenheidsbug |
| ~350 | `price and not combined_fv and methods_used < 2` | te weinig valide methodes |
| — | geen koers | blijft `N/A` |

De eerste regel dekt twee volstrekt verschillende gevallen af die je niet uit
elkaar kunt houden, en dat is precies het gat.

### Wat te bouwen

**Niet** het `signal`-veld opsplitsen in vijf waarden. `INSUFFICIENT DATA` blijft
één signaal — de rest van het systeem (filters, exit-regels, kansenlijst) rekent
daarop en dat moet zo blijven. Voeg in plaats daarvan een **reden** toe naast het
signaal.

1. **Migratie in `engine/db.py`**, bij `CREATE TABLE IF NOT EXISTS
   calculated_scores` (regel ~181) en als losse `ALTER TABLE`, volgens het
   patroon dat er al staat voor `auto_suspended_at`:

   ```sql
   ALTER TABLE calculated_scores ADD COLUMN IF NOT EXISTS data_reden TEXT
   ```

   `upsert_scores(ticker, **fields)` is generiek, dus die hoeft niet aangepast.

2. **Vier waarden**, als constanten bovenaan `screener.py`:

   | Waarde | Wanneer | Betekenis voor de gebruiker |
   |---|---|---|
   | `NOG_NIET_OPGEHAALD` | `not annual_rows` **en** `data_quality.last_checked IS NULL` | Geen probleem, wacht op de rotatie |
   | `BRON_LEEG` | `not annual_rows` **en** wél eerder geprobeerd | Yahoo heeft geen jaarcijfers — kandidaat voor archivering |
   | `DATA_AFGEKEURD` | `dq_status in ("bad","missing")` | De poort zei nee; reden staat al in `warnings` |
   | `NIET_WAARDEERBAAR` | FV-plausibiliteit of `methods_used < 2` | Data is goed, het model kan er niets mee |

   Het onderscheid tussen de eerste twee is de kern van deze fase. Haal
   `last_checked` op via de bestaande `db.get_data_quality(ticker)` — die wordt
   in `run_ticker` al aangeroepen op regel ~260, dus verplaats die aanroep naar
   boven de `if not annual_rows`-tak in plaats van een tweede query te doen.

3. **Doorgeven aan de frontend.** `engine/scorebord.py` en `_dashboard_rij` /
   `_dashboard_rows` in `app.py` moeten `data_reden` meenemen. In de
   signaaltelling op het scorebord: splits "Geen oordeel" in
   "Wacht op data" (`NOG_NIET_OPGEHAALD`) en "Geen oordeel" (de rest). Een
   ticker die nog nooit is opgehaald is geen mislukking en hoort niet als
   mislukking in beeld te staan.

4. **Archiveringssignaal.** `BRON_LEEG` is precies wat de auto-suspendlogica in
   `engine/refresh.py` zoekt. Controleer of `outcome["empty"]` en
   `db.bump_failure_counter` nu al hetzelfde onderscheid maken; zo ja, gebruik
   dezelfde bron van waarheid in plaats van een tweede definitie.

### Tests

Nieuw bestand `tests/test_data_reden.py`, in de stijl van
`tests/test_oordelen.py`:

- ticker zonder jaarcijfers en zonder `last_checked` → `NOG_NIET_OPGEHAALD`
- ticker zonder jaarcijfers mét `last_checked` → `BRON_LEEG`
- ticker met `dq_status = "bad"` → `DATA_AFGEKEURD`
- ticker met koers 10 en `combined_fv` 200 → `NIET_WAARDEERBAAR`, en `signal`
  blijft `INSUFFICIENT DATA`
- een gezonde ticker → `data_reden IS NULL` en het signaal is ongewijzigd

Draai daarnaast `pytest tests/` volledig: `test_empty_results.py`,
`test_thinning.py`, `test_scores_recompute.py` en `test_suspend_regels.py`
raken hieraan.

### Acceptatiecriterium

Na `POST /api/recalculate` moet de som van de vier redenen exact gelijk zijn aan
het aantal `INSUFFICIENT DATA`-tickers, en moet het scorebord "Wacht op data"
apart tonen. Geen enkele ticker die vóór de wijziging een echt signaal had, mag
er daarna een ander hebben — verifieer dat met een read-only vergelijking vooraf
en achteraf, zoals `scripts/simulate_recompute.py` dat doet.

---

## Fase 2 — FV-plausibiliteitspoort: een waarschuwingsband erbij

**Probleem.** De poort in `screener.py` (~337) blokkeert bij een fair value
buiten `[0.1, 10]` maal de koers. Dat vangt factor-100-fouten (pence, agorot,
cent). Het vangt niet wat er in je eigen portefeuille staat:

| Aandeel | FV / koers | Door de poort |
|---|---:|---|
| Econocom (ECONB.BR) | 7,7× | ja |
| Arctic Paper (ARP.ST) | 6,6× | ja |

Beide staan in `overzicht-2026-08-21.md` als *"heeft geen bruikbare grens: het
generieke model kan deze aandelen niet waarderen"*. Dat is de poort die vertelt
dat hij te ruim staat.

Dit telt zwaarder bij uitbreiding: andere boekhoudconventies produceren geen
extra factor-100-fouten — die worden al gepakt — maar wel meer
**factor-3-tot-8**-onzin, precies de band waar de poort niets doet.

### Wat te bouwen

Een tweede, **niet-blokkerende** band. Buiten `[0.33, 3.0]` maar binnen
`[0.1, 10]`: signaal blijft staan, maar er komt een waarschuwing bij en een
markering die op de kansenlijst zichtbaar is. Zet de grenzen in `config.yaml`
onder `valuation`, niet hardcoded — dit is een drempel die je gaat bijstellen:

```yaml
valuation:
  fv_plausibel_hard: [0.1, 10.0]    # buiten deze band: geen signaal
  fv_plausibel_zacht: [0.33, 3.0]   # buiten deze band: signaal met voorbehoud
```

Let op de les uit `config.yaml` bij `sell_quality_floor`: **een knop die niets
doet is erger dan geen knop.** Als de zachte band nergens zichtbaar wordt, bouw
hem dan niet.

### Acceptatiecriterium

Draai de nieuwe poort read-only over het hele universum en rapporteer hoeveel
tickers in de zachte band vallen. Verwacht ergens tussen de 50 en 300; komt het
boven de 500, dan staat de band te strak en is de uitkomst onbruikbaar als
markering. Stel bij vóór uitrol, niet erna.

---

## Fase 3 — IJking testen vóór je een markt importeert

**Probleem.** `config.yaml` heeft één set multiples voor de hele wereld:
`Financial Services: pb: 1.4`, `Technology: pe: 25`, en `bond_yield: 5.0` voor
de Graham-herschaling. Dat is op Europa geijkt.

Japanse banken noteren al dertig jaar structureel onder boekwaarde om redenen
die niets met onderwaardering te maken hebben; Hongkongs vastgoed idem. Met
`pb: 1.4` levert elke Japanse regionale bank een schijnkorting van 60 tot 70%
op, en die verdringen de Europese namen uit de kansenlijst. Dat is hetzelfde
mechanisme als de GICS/Yahoo-fout van 6 augustus: geen foutmelding, alleen een
stille vertekening.

### Wat te bouwen

Een read-only proefdraai, in de geest van `scripts/simulate_recompute.py`:

```
scripts/probeer_markt.py --tickers <bestand> [--verslag docs/probe-<markt>.md]
```

- haalt via `data_fetcher.fetch_and_store` de opgegeven tickers op, **zonder ze
  in `stocks` te zetten** (of met `active = 0`, zodat ze niet in de rotatie
  belanden)
- draait `screener.run_ticker` erop
- rapporteert per ticker: sector, `fv_methods_used`, `combined_fv`, koers,
  FV/koers-verhouding, signaal en `data_reden`
- vat samen: hoeveel kregen een sector, hoeveel een signaal, en wat is de
  verdeling van FV/koers

Gebruik één gedeelde HTTP-sessie voor alle Yahoo-aanroepen. Per-thread sessies
geven HTTP 401 op `.info`; je krijgt dan lege sector en valuta terug terwijl
`.financials` gewoon doorkomt — dat leest als ontbrekende data terwijl het een
authenticatiefout is.

### Uitvoeren op Japan

Neem 20 tickers uit drie groepen, zodat de vertekening zichtbaar wordt:

- **exporteurs** (verdienen aan een zwakke yen): 7203.T Toyota, 6501.T Hitachi,
  6954.T Fanuc, 6273.T SMC, 7741.T Hoya
- **regionale banken en verzekeraars** (het risicogeval): 8306.T MUFG,
  8316.T Sumitomo Mitsui, 8411.T Mizuho, 8630.T Sompo, 8725.T MS&AD
- **binnenlands**: 9843.T Nitori, 3382.T Seven & i, 4661.T Oriental Land
- **smallcap**: kies vijf namen uit het Standard- en Growth-segment van
  `data_j.xls` met een beurswaarde onder ¥50 mrd. Welke precies maakt niet uit —
  het gaat erom dat het segment meedoet waar een screener zijn waarde moet
  halen. Controleer de tickers vóór gebruik tegen de JPX-lijst; verzin er geen.

**Beoordelingsvraag:** krijgen de banken een korting van meer dan 50%? Zo ja, dan
is dat een modelartefact en moet er iets veranderen vóór de import. Twee routes:

1. **Regiofactor op de multiples** — een vermenigvuldiger per markt bovenop het
   sectorprofiel in `config.yaml`. Netjes, maar hij moet ergens op geijkt worden.
2. **Observatiestand** — een markt importeren en waarderen, maar uitsluiten van
   de kansenlijst tot je hebt gecontroleerd wat het model doet. Sneller en
   eerlijker over wat je nog niet weet.

Route 2 heeft de voorkeur voor de eerste markt. Route 1 pas als je uit route 2
weet welke factor je nodig hebt.

---

## Fase 4 — Het Verenigd Koninkrijk importeren

Pas beginnen als fase 0 t/m 3 klaar zijn. Het VK is bewust de eerste markt: het
suffix `.L` staat al in `SUFFIX_INFO`, de pence-deling zit al in
`data_fetcher.py` (~regel 241-255), DEGIRO heeft de LSE, en er is 0%
bronbelasting. Je hebt er nu **44**, er zijn er **1.993**.

### Bron

[LSE Instrument list.xlsx](https://docs.londonstockexchange.com/sites/default/files/reports/Instrument%20list.xlsx)
— werkt vanaf een server, en bevat TIDM, ISIN, ICB-sector, handelsvaluta en
Main/AIM. Gebruik **niet** de Issuer list; die heeft geen tickerkolom.

Bouw de parser in `import_tickers.py` in dezelfde vorm als de bestaande
Xetra- en Baltic-parsers, en breid `tests/test_import_parsers.py` uit met een
fixture van tien regels.

### Verplichte filters

1. **`^0[A-Z0-9]{3}\.L` weggooien.** Van de 1.015 `.L`-symbolen die Yahoo kent
   zijn er **516** van deze vorm. Dat zijn geen Britse bedrijven maar LSE-lijnen
   voor buitenlandse aandelen — `0R2M.L` is Regeneron, `0HJI.L` is ADP. Eén dag
   koershistorie, nul jaarcijfers. Zonder dit filter importeer je 500 dode
   tickers en concludeer je ten onrechte dat Yahoo slechte Britse data heeft.
2. **`.XC` en `.IL` overslaan.** Dat zijn ook buitenlandse lijnen. Ze geven wél
   jaarcijfers terug, maar van het buitenlandse moederbedrijf — dus duplicaten,
   geen lege regels. Verraderlijker dan categorie 1.
3. **Handelsvaluta lezen uit de kolom in het bronbestand**, niet afleiden uit
   het suffix. Van de 4.078 equity-lijnen op de LSE handelt er **1.044 in USD**,
   317 in GBP, 160 in EUR en 2.548 in GBX. `infer_currency()` in
   `data_fetcher.py` gaat op het suffix af en zit er dus bij een kwart naast;
   `info["currency"]` van Yahoo gaat vóór, wat de code al doet — controleer dat
   dit ook zo blijft bij de import.

### Deduplicatie

`engine/dubbelingen.py` matcht op ISIN als die er is en anders op
genormaliseerde naam. De LSE-lijst hééft ISIN — gebruik hem, want de
naamnormalisatie (`_SUFFIXEN` met ` plc`, ` ltd`, ` group`) is precies waar
Britse namen elkaar gaan raken.

Let op dat `dubbelingen.py` alleen markeert en niet filtert. Dat is een bewuste
keuze en bij 2.812 tickers prima, maar met 1.950 Britse namen erbij wordt de
kans groot dat de kansenlijst hetzelfde bedrijf twee keer toont — zoals nu al
met Silvano Fashion Group op plek 3 en 4. Overweeg om **alleen op de
kansenlijst** te ontdubbelen, met de dubbeling zichtbaar als voetnoot.

### Acceptatiecriterium

Na de import en één volledige rotatie: van de geïmporteerde Britse tickers moet
minstens 90% een sector hebben en minstens 85% een echt signaal (dus geen
`INSUFFICIENT DATA`). Blijft het daaronder, dan zit er nog een lijn-type in de
lijst dat gefilterd moet worden — zoek dat uit vóór je Japan doet.

---

## Fase 5 — Pas hierna: de rotatie en de volgende markt

`fundamentals_per_night` staat op 250. Met 2.812 tickers is dat 11 nachten per
ronde. Met het VK erbij wordt het ~19 nachten. Met Japan, Hongkong en Korea er
ook bij kom je op ~12.000 tickers en **48 nachten** — dan zijn je jaarcijfers
gemiddeld zeven weken oud terwijl je koersen dagvers zijn, en geeft het
dashboard signalen op verouderde waarderingen.

**Raak dit getal niet aan voordat fase 1 klaar is.** Nu zou je alleen harder aan
een rotatie trekken waarvan je niet weet wat eruit komt. In
`refresh.py` staat bovendien al gedocumenteerd waarom 250 de bovengrens is:
elke ticker is een losse Yahoo-aanroep en daarboven loop je tegen rate-limits.

Drie routes, als het zover is:

1. **Selectief importeren** — Japan alleen Prime en Standard (3.117 in plaats
   van 3.713), Hongkong alleen de Main Board (2.465), Korea alleen KOSPI (850).
   Of een ondergrens op marktkapitalisatie.
2. **De rotatie differentiëren** — jaarcijfers veranderen per kwartaal, maar niet
   voor iedereen tegelijk. Bedrijven die net gerapporteerd hebben vaker
   verversen dan de rest. `get_refresh_queue` sorteert nu puur op
   `last_checked`; een tweede sorteersleutel op rapportagedatum zou hier het
   verschil maken.
3. **Fasegewijs** — één markt per keer, en meten wat de verversing doet voordat
   de volgende erbij komt.

De marktvolgorde daarna staat in `docs/landenuitbreiding-2026-08-21.md`: na het
VK komt Japan (alleen Prime en Standard), dan Hongkong, dan pas Korea. De
suffixen die aan `engine/markets.py` moeten worden toegevoegd staan daar ook, en
worden bewaakt door `tests/test_markets.py`.

---

## Wat je niet moet doen

- **Geen land importeren voordat fase 0 is gerapporteerd.** De hele werkorder
  hangt op wat die meting zegt.
- **Het `signal`-veld niet opsplitsen.** `INSUFFICIENT DATA` blijft één signaal;
  de reden komt ernaast te staan. Exit-regels, filters en de kansenlijst rekenen
  op de bestaande waarden.
- **Geen valutaconversie inbouwen.** `data_fetcher.py` stelt expliciet: alles
  blijft in de eigen valuta van het aandeel. Dat is juist — een waardering hoort
  in de valuta van de kasstromen. De `financialCurrency`-conversie die er al is
  (regel ~400) is iets anders en moet blijven.
- **De sectorsleutels in `config.yaml` niet aanraken.** Dat moeten exact de
  namen zijn die Yahoo teruggeeft, niet de GICS-namen. Bewaakt door
  `tests/test_sector_config.py`, en het is al één keer misgegaan.
- **Geen Indonesië, Israël, Thailand, Turkije, India of Taiwan.** De redenen
  staan in `docs/landenuitbreiding-2026-08-21.md`; bij Indonesië en Israël is
  het gemeten (23-30% van de aandelen heeft koershistorie maar een lege
  winst-en-verliesrekening bij Yahoo, tegen 0-5% elders).

---

## Volgorde in één blik

| Fase | Wat | Omvang | Blokkeert |
|---|---|---|---|
| 0 | Meten waaruit de 713 bestaan | half uur | alles |
| 1 | `data_reden` naast het signaal | middag | 3, 4 |
| 2 | Zachte FV-plausibiliteitsband | uur | — |
| 3 | `probeer_markt.py` + 20 Japanse namen | uur | 4 |
| 4 | VK importeren | dagdeel | 5 |
| 5 | Rotatie herzien, volgende markt | later | — |

Fase 0 tot en met 3 maken de screener beter, ook als er uiteindelijk geen enkel
land bij komt. Dat is de reden om ze eerst te doen: ze zijn niet verspild als de
rest afvalt.

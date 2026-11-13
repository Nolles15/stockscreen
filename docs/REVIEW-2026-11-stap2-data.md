# Kritische review november 2026 — stap 2: data en verversing

Datum: 2026-11-13. Read-only. Gelezen: `engine/refresh.py`, `engine/stilte.py`,
de scheduler en `/api/health` in `app.py`, de wachtrij-queries in `engine/db.py`,
de kern van `engine/data_fetcher.py`, `Dockerfile`, `fly.toml`.
Alleen op hoofdlijnen: `engine/data_quality.py` (structuur en drempels).
Live gemeten: `/api/health`, `/api/refresh/wachtrij`, `/api/stocks/suspended`,
`/api/dashboard`.

## Stand vandaag

| | |
|---|---|
| Laatste koersronde | 12 nov 18:57 (98,7% van de koersen jonger dan 24 uur) |
| Laatste ronde jaarcijfers | 13 nov 03:48 (98,6% jonger dan 30 dagen) |
| Scheduler | leeft; geen storm in zeven dagen |
| Herberekening | mediaan 1 dag oud, maximaal 4 dagen |
| Nieuwste boekjaar | 3.860 van 4.012 op FY2025 of FY2026; 108 lopen achter |
| Datastatus | ok 2.753 · waarschuwing 731 · afgekeurd 510 · ontbreekt 18 |
| Geen oordeel | 1.062, waarvan 900 verklaarbaar (verlies, negatieve omzet) en 162 een dataprobleem |
| Gesuspendeerd / vermoedelijk van de beurs | 113 / 0 |

De motor draait en de data is vers. Dat is na de stilstand van juni een echte
verbetering, en de zelfstoppende machine werkt zoals bedoeld.

## Bevindingen

### D1. Niemand wordt gewaarschuwd als het stilvalt

De verversing hangt aan één externe wekker (cron-job.org) die een slapende
machine wakker maakt. `/api/health` geeft altijd HTTP 200, ook als de data dagen
oud is. Valt de wekker uit, dan merk je dat pas aan de banner bij je volgende
bezoek. Dit is hetzelfde patroon als de zes weken stilstand in juni, alleen met
een kortere ontdektijd.

Goedkoopste reparatie: laat `/api/health` (of een apart `/api/health/strict`)
een 503 geven zodra de koersen ouder zijn dan 36 uur of de scheduler niet tikt,
en zet de foutmelding van cron-job.org aan. Voeg één controle per dag toe buiten
de verversvensters (bijvoorbeeld 09:00), zodat een gemiste nacht dezelfde ochtend
een mail oplevert.

### D2. Een gemiste koersronde wordt niet ingehaald vóór 18:30

`_scheduler_loop` start de koersronde alleen als het na 18:30 is én hij vandaag
nog niet draaide. Mist de wekker de avond, dan doet een bezoek de volgende
ochtend niets voor de koersen; pas na 18:30 die dag. CLAUDE.md zegt dat gemist
werk wordt ingehaald; dat klopt voor de jaarcijfers (vanaf 03:00 de hele dag),
niet voor de koersen. Reparatie: ook starten als de laatste koersronde ouder is
dan 24 uur.

### D3. Het publicatievenster werkt niet, en staat omgekeerd

`get_refresh_queue` geeft voorrang aan tickers waarvan `boekjaar_einde` twee tot
zeven maanden geleden ligt. Twee problemen:

- **Het is dode code.** De voorwaarde eist ook dat de laatste poging ouder is dan
  zeven dagen. Met 750 per nacht duurt een volledige ronde 5,4 dagen, dus niets is
  ooit ouder dan zeven dagen. De wachtrij toont dan ook 4.012 in "gewone rotatie"
  en nul in het venster.
- **De logica staat omgekeerd.** `boekjaar_einde` is de einddatum van het
  nieuwste boekjaar dat al in de database staat (`data_fetcher.py:327`). Twee tot
  zeven maanden daarna hébben we dat jaar dus al. Het volgende verslag is te
  verwachten veertien tot negentien maanden na die datum.

Bij 4.000 tickers is dit onschuldig, want alles komt binnen zes dagen langs. Het
was gebouwd voor de uitbreiding naar 19.000, en daar zou het de verkeerde
tickers voorrang geven. CLAUDE.md beschrijft het als werkend en geverifieerd.

### D4. Suspenderen gebeurde acht weken later dan gedocumenteerd, onopgemerkt

CLAUDE.md voorspelt de eerste suspensies rond 2 september en zegt: "ziet het er
ná die datum nog steeds zo uit, dán is er iets mis". De eerste suspensie is van
26 oktober. Voorbeeld: BML.WA, "11 mislukte pogingen sinds 2026-08-29",
gesuspendeerd op 12 november, terwijl de regel 30 dagen en twee pogingen is.
Op 12 november zijn er precies 40 tegelijk gesuspendeerd. Waarom het zo laat
kwam heb ik niet kunnen vaststellen zonder het activiteitenlog; het is de moeite
van uitzoeken waard, juist omdat de eigen controleregel het niet ving.

Daarnaast pakt `get_reprobe_candidates` altijd de 40 oudste. Wie niet herstelt
blijft de oudste, dus dezelfde 40 krijgen elke week een kans en de andere 73
geen. Het commentaar ("elk eens per drie weken") klopt niet.

### D5. Eén onofficiële bron, zonder versievastlegging of ijkcontrole

- Alles komt uit yfinance, een onofficiële schraper. `requirements.txt` zegt
  `yfinance>=0.2.40`: elke deploy haalt de nieuwste versie op.
- Verandert Yahoo of yfinance een regelnaam ("Total Revenue"), dan levert de
  ophaler `None`. Door de NULL-bescherming (`COALESCE`) blijven de oude waarden
  staan. Dat is goed tegen half lege antwoorden, maar het betekent ook dat een
  structurele breuk geen fout geeft: de cijfers bevriezen stil.
- Er is geen ijkcontrole die na een ronde een paar bekende waarden toetst
  (bijvoorbeeld omzet ASML FY2024 = 28,26 miljard).

Het archief (jaarcijfers stapelen op, koershistorie wordt bewaard) is een goede
verzekering tegen het wegvallen van de bron. Tegen het stil veranderen van de
bron is er niets.

Advies: versies vastpinnen, en een ijkset van tien tickers met bekende waarden
die na elke ronde jaarcijfers wordt getoetst en in `/api/health` meetelt.

### D6. Kleinere punten

- **Verouderde koers met een vers signaal.** Tien tickers tonen een HOLD of SELL
  op een koers van vier tot 75 dagen oud (COFB.BR 75, GLPG.BR 54, PEH.DE 51,
  CPX.F 43, ECK.DE 22). Het signaal wordt live berekend tegen de laatste koers,
  maar er is geen poort op de leeftijd van die koers per rij.
- **Split-wacht.** Een koersverschil van meer dan 40% wordt niet opgeslagen en
  leidt tot een volledige herophaling. Mislukt die, dan blijft de oude koers
  staan en herhaalt het zich elke avond zonder melding.
- **Amerikaanse tickers** (56) krijgen om 18:30 een koers van halverwege de
  handelsdag, die als slotkoers van die dag in `price_history` komt en daarna
  nooit meer wordt overschreven.
- **Wisselkoers niet beschikbaar** bij een afwijkende rapportagevaluta geeft
  alleen een waarschuwing; of de poort dit altijd afkeurt heb ik niet nagegaan.
- **`Net Interest Income` als terugval voor rentelasten**: bij ontbreken van een
  rentelastenregel telt netto rente-inkomen als last.

## Wat goed is

- Storm-wacht, NULL-bescherming, één ronde tegelijk, herstartbestendige
  scheduler op databasestatus: de lessen uit april en juni zijn in code omgezet.
- `/api/health` en `/api/refresh/wachtrij` maken de toestand in één blik
  zichtbaar. Het ontbreekt alleen aan iemand die kijkt (D1).
- De zelfstop is degelijk ontworpen: werk houdt de machine wakker, een hangende
  ronde hooguit drie uur, en een crash van de bewaker laat de machine aan.
- 162 echte dataproblemen op 4.012 (4%) is een nette score voor een gratis bron.

## Aanbevelingen, op volgorde

| # | Wat | Omvang |
|---|---|---|
| 1 | Health geeft 503 bij verouderde data; foutmail van de wekker aan; één controle overdag | klein |
| 2 | Koersronde ook starten als de vorige ouder is dan 24 uur | klein |
| 3 | Versies vastpinnen in `requirements.txt` | klein |
| 4 | IJkset van tien tickers met bekende waarden, getoetst na elke ronde | middel |
| 5 | Uitzoeken waarom suspenderen acht weken te laat kwam (activiteitenlog) | klein |
| 6 | Publicatievenster repareren (+12 maanden, drempel van zeven dagen herzien) of verwijderen tot de uitbreiding naar 19.000 er komt | klein |
| 7 | Geen signaal tonen bij een koers ouder dan vijf handelsdagen | klein |

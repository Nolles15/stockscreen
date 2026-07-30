# Diagnose — "INSUFFICIENT DATA" op schaal

> Read-only diagnose. Doel: vaststellen **wat** er op grote schaal misgaat en **waarom**, en de
> remediation-routes prioriteren — zonder de oplossing al uit te voeren. Eindoel blijft: de data moet er
> uiteindelijk **wél** in komen (de tickers staan er niet voor niets in).
>
> Instrument: `GET /api/gaps-report` + [scripts/gaps_analyze.py](../scripts/gaps_analyze.py), bovenop de
> bestaande `data_quality`-tabel. Classificatie via `data_quality.classify_blockers()` (raakt de gate niet aan).
>
> **Meetmoment:** live `/api/data-quality`, 799 tickers. Reproduceer met `python scripts/gaps_analyze.py`.

---

## 1. De kern in één zin

Het is **geen data-beschikbaarheidsprobleem**. Van de 799 tickers heeft de pipeline voor **515 (64,5%)
gewoon `ok`**; slechts **2 tickers** gaven écht niets terug. De ~260 "INSUFFICIENT DATA"-aandelen worden
vrijwel allemaal **door de kwaliteits-gate `evaluate()` afgekeurd terwijl de data bruikbaar is** — met één
heuristiek (split-detectie) als veruit grootste oorzaak.

> **Belangrijke correctie t.o.v. de planfase:** een eerdere snelle steekproef schatte ~494 `bad`. De harde
> meting met het nieuwe instrument geeft **255 `bad` + 6 `missing` = 261 geblokkeerd**. Het echte aantal is
> dus ~de helft van de eerste schatting — en het ok-aandeel is veel hoger dan gedacht.

---

## 2. Bucket-verdeling (799 tickers, elke ticker telt 1× op primaire blocker)

| Bucket | n | % univ. | Aard | Route |
|---|---:|---:|---|---|
| **ok** | 515 | 64,5% | werkt | — |
| **split_suspected** | **150** | 18,8% | heuristiek, grotendeels **false positive** | **A** |
| structural_neg_fcf | 53 | 6,6% | deels legitiem afgekeurd | A |
| structural_loss | 23 | 2,9% | cyclisch/echt verlies | A (deels B) |
| warning | 20 | 2,5% | net onder drempel | A |
| unit_mismatch_severe | 10 | 1,3% | **multi-share-class**, echte unit-bug | B |
| negative_equity | 9 | 1,1% | mix distress + buyback-gezond | A+B |
| negative_revenue | 8 | 1,0% | **holdings/investment co's** | B |
| non_equity | 6 | 0,8% | incl. **ticker-resolutie-bugs** | B |
| unknown | 3 | 0,4% | geen dq-record | (rerun) |
| no_years | 2 | 0,3% | **echt leeg op Yahoo** | B |

Geblokkeerd totaal = **261** (255 bad + 6 missing). **split_suspected alleen is 57% van alle blokkades.**

Geografie (suffix, geblokkeerde tickers): gelijkmatig over EU-beurzen (DE 35, OL 33, ST 32, WA 27, AS 20,
BR 19, US 17, MC 15, PA 13, LS 11, CO 6, SW 6) — **géén** beurs-specifiek databron-probleem. De oorzaak zit in
de gate, niet in een land/koppeling. (Markt/currency-kruistabel volgt zodra `/api/gaps-report` live staat;
het `markt`-veld is bovendien corrupt — zie §4 H4.)

---

## 3. Bewijs per bucket (gestratificeerde steekproef)

### A — Gate keurt bruikbare data af (de grote hefboom)

**`split_suspected` (150) — false-positive-epidemie.** De heuristiek vuurt bij een EPS YoY-ratio ≥3× (of ≤⅓).
In de praktijk treft dat geen splits maar:

| Ticker | EPS-sprong | Werkelijke oorzaak |
|---|---|---|
| ALM.MC (Almirall) | 0,05 → 0,22 (4,4×) | **kleine EPS-basis** — normale winstgroei blaast de ratio op |
| ACGL (Arch Capital) | 3,80 → 11,62 (3,1×) | echte winststijging, geen split |
| AF.PA (Air France-KLM) | 0,93 → 5,50 (5,9×) | **herstel uit verlies** |
| 11B.WA (11 bit studios) | 0,22 → 2,85 (13×) | release-gedreven winstpiek |
| APAM.AS (Aperam) | 3,17 → 0,13 (0,0×) | **cyclische winstinzinking** |
| AKER.OL / ALTR.LS | 190→43 / 0,52→0,10 | cyclische daling |

→ Diagnose: de drempel `_SPLIT_EPS_RATIO = 3.0` zonder absolute-EPS-floor en zonder kruischeck op
omzet/koers vangt **kleine bases, herstel-uit-verlies en cyclische swings** als "split". Dit is de
duidelijkste, hoogst-impact false positive.

**`structural_neg_fcf` (53) / `structural_loss` (23).** Bevat legitieme, waardeerbare bedrijven die op
EPS/FCF-cashflow toevallig 3+ jaar negatief staan: utilities met hoge capex (EDP.LS, Elia, Cloudberry),
cyclische industrie (Covestro, Outokumpu, Lenzing, Grupa Azoty). De FCF/EPS-methodes falen hier terecht,
maar de **multiples-methodes (P/B, EV/EBITDA) blijven bruikbaar** — toch wordt de hele ticker geblokkeerd.

**`warning` (20).** Net onder de completeness/jaren-drempel; vaak één ontbrekend veld van een verder gezond
bedrijf (Erste Group, Nordea-achtigen). Laagdrempelig te redden.

### B — Echte databron-/notatie-fix nodig (kleiner, maar reëel)

**`unit_mismatch_severe` (10) — multi-share-class.** Stuk voor stuk bedrijven met meerdere aandelenklassen
waar `shares_outstanding` (één klasse) × prijs niet matcht met Yahoo's totale market cap:
Roche RO.SW (7,46×), Lindt LISN.SW (4,72×), Merck KGaA (3,36×), GOOGL (2,0× A vs C), Atlas Copco, Henkel,
Industrivärden. **Geen databug maar een share-class-mismatch** — vraagt een mapping/correctie, niet een refetch.

**`negative_revenue` (8) — holdings/investment companies.** Exor, Kinnevik, Yellow Cake, Pharol: Yahoo's
`revenue`-veld is negatief/0 omdat het beleggingsresultaat i.p.v. omzet is. Vraagt een **andere
waarderingsbenadering of NAV-bron**, niet meer data.

**`non_equity` (6) — incl. ticker-resolutie-bugs.** Echte fondsen zitten erin, maar ook **ING Groep (INGA.BR)
en Galapagos (GLPG.BR)** worden als `MUTUALFUND` geclassificeerd → Yahoo lost de ticker verkeerd op /
verkeerde notatie. Vraagt remap naar de juiste (primaire) notatie.

**`no_years` (2) — echt leeg.** H2O.RO (Hidroelectrica), MTEL.BD: Yahoo levert geen jaarcijfers. Enige bucket
die een **alternatieve ingestion-route** (override via `/api/overrides/<T>` of andere bron) nodig heeft.

**`negative_equity` (9) — gemengd.** Echt distressed (Atos, Satis) naast **buyback-gezond** (HP Inc, Edenred:
negatief eigen vermogen door inkoop, maar winstgevend). De P/B-blokkade is voor de buyback-gevallen een
false positive.

---

## 4. Onderliggende oorzaken (hypotheses getoetst)

- **H1 — split-heuristiek = false-positive-epidemie. BEVESTIGD.** 150 tickers (57% van alle blokkades);
  steekproef laat overwegend kleine-basis/herstel/cyclus zien, geen echte splits. Bron:
  [engine/data_quality.py:262-274](../engine/data_quality.py#L262) (`_SPLIT_EPS_RATIO`).
- **H2 — structurele streaks blokkeren te breed. BEVESTIGD (deels).** 76 tickers; multiples blijven vaak
  bruikbaar maar de ticker wordt volledig geblokkeerd. Bron: [engine/data_quality.py:248-259](../engine/data_quality.py#L248).
- **H3 — unit/mcap/EV-mismatch is een échte bug. BEVESTIGD, maar specifiek.** Het is vrijwel altijd een
  **multi-share-class**-mismatch (10 megacaps), niet een willekeurige schaalfout. Bron:
  [engine/data_quality.py:177-200](../engine/data_quality.py#L177).
- **H4 — metadata corrupt. BEVESTIGD.** `markt` is voor de hele geblokkeerde set onbruikbaar (alle "?"/`US`),
  en `non_equity` bevat ticker-resolutie-fouten (ING/Galapagos). Veroorzaakt zelf geen INSUFFICIENT DATA maar
  blokkeert wél de juiste databron.
- **H5 — echt leeg is verwaarloosbaar. BEVESTIGD.** Slechts **2** `no_years` + enkele non-equity. Data-
  beschikbaarheid is niet het probleem.

---

## 5. Geprioriteerde aanbeveling (richting "data komt erin") — nog NIET uitgevoerd

**Route A — gate-kalibratie (TOP-PRIORITEIT: hoogste impact, laagste moeite).**
Raakt ~246 van de 261 geblokkeerde tickers (split + structureel + warning + buyback-equity). De data ís er;
alleen de drempels keuren te streng af. Kandidaat-aanpassingen om in de volgende sessie te ontwerpen/testen:
1. **Split-detectie verfijnen** — absolute EPS-floor (negeer ratio's bij EPS < ~€0,50), kruischeck tegen
   omzet/koers-continuïteit, en herstel-uit-verlies uitsluiten. Verwachte winst: leeuwendeel van 150 tickers.
2. **Structurele streaks niet-blokkerend maken** wanneer de multiples-methodes (P/B, EV/EBITDA) wél valide
   input hebben → waardeer op multiples i.p.v. de hele ticker te droppen.
3. **Buyback-gedreven negatief eigen vermogen** onderscheiden van insolventie (bv. winstgevend + positieve
   FCF → niet blokkeren op P/B).
   *Validatie vóór uitrol: draai de aangepaste `evaluate()` over de bucket en tel hoeveel `ok` worden zonder
   dat de echte garbage (no_years, holdings) doorglipt.*

**Route B — databron-/notatie-fix (kleiner, gerichter handwerk).**
- Multi-share-class (10): juiste shares-per-klasse of mcap-bron mappen.
- Holdings/investment co's (8): aparte NAV-/beleggingsbenadering of expliciet als categorie markeren.
- Ticker-resolutie/non-equity (6): remap naar primaire notatie (`INGA.BR`→`INGA.AS`, e.d.) via de
  bestaande `/api/stocks/remap`.
- Echt leeg (2): override-pipeline `/api/overrides/<T>` of alternatieve bron.

**Volgorde:** eerst A (één gate-aanpassing redt honderden tickers in één keer), daarna B per kleine bucket.

---

## 5b. Route A — UITGEVOERD & GEMETEN ✅

De gate is herkalibreerd in [engine/data_quality.py](../engine/data_quality.py):

1. **Split-detectie verscherpt** — absolute EPS-floor (`_SPLIT_EPS_MIN_ABS = 0.30`) + clean-factor-eis
   (`_clean_split_factor`: ratio moet binnen 6% van een écht split-getal 2/3/4/5/10 liggen). Elimineert 7 van
   8 bewezen false positives; de heuristiek is bovendien **niet langer blokkerend** (alleen waarschuwing).
2. **Structureel verlies / negatieve FCF** uit `has_blocker` gehaald → multiples (P/B, EV/EBITDA) blijven
   bruikbaar; de downstream-gates in [screener.py](../engine/screener.py#L320) blokkeren alsnog wat écht
   onwaardeerbaar is.
3. **Negatief eigen vermogen** blokkeert nog alleen bij een **verlieslatend** bedrijf (winstgevende
   buyback-gevallen zoals HP/Edenred worden niet meer geblokkeerd).

**Gemeten impact** (read-only simulatie over alle 799 tickers met de nieuwe `evaluate()` op de live-
opgeslagen data — [scripts/simulate_recompute.py](../scripts/simulate_recompute.py)):

| | voor | na |
|---|---:|---:|
| geblokkeerd (bad+missing) | **261** | **33** |
| ok | 515 | 690 |
| warning | 20 | 76 |

→ **228 tickers gered** (182 `bad→ok`, 46 `bad→warning`). **0 regressies** — niets dat eerst ok/warning
was, werd geblokkeerd. De resterende 33 zijn exact de Route B-gevallen (holdings met negatieve omzet,
multi-share-class-mismatch, écht insolvent, 6 fondsen, 2 leeg).

**Toepassen op live (na deploy)** — geen Yahoo-refetch nodig:
```bash
# 1. nieuwe gate op bestaande data toepassen (dry-run eerst):
curl -k -s -X POST https://stockscreen-janco.fly.dev/api/data-quality/recompute \
  -H 'Content-Type: application/json' -d '{"dry_run": true}'  | jq '{before,after,rescued_count}'
curl -k -s -X POST https://stockscreen-janco.fly.dev/api/data-quality/recompute \
  -H 'Content-Type: application/json' -d '{"dry_run": false}' | jq '{rescued_count, newly_blocked}'
# 2. signalen/FV herberekenen vanuit cache (geen netwerk):
curl -k -s -X POST https://stockscreen-janco.fly.dev/api/recalculate -d '{}' | jq 'length'
```

---

## 6. Reproduceren / verifiëren

```bash
# Volledige bucket-analyse + steekproeven (read-only, geen token):
python scripts/gaps_analyze.py --sample 12

# Inzoomen op één bucket:
python scripts/gaps_analyze.py --bucket split_suspected --sample 30

# Na deploy van /api/gaps-report (rijkere join met markt/currency):
curl -k -s https://stockscreen-janco.fly.dev/api/gaps-report \
  | jq 'group_by(.primary_blocker) | map({b: .[0].primary_blocker, n: length}) | sort_by(-.n)'
```

> **Status instrument:** `classify_blockers()` + `gaps_analyze.py` werken live (via `/api/data-quality`-
> fallback). Het rijkere `GET /api/gaps-report` endpoint is gebouwd maar **nog niet gedeployed** — na
> `fly deploy` levert het dezelfde buckets plus de markt/currency-join.

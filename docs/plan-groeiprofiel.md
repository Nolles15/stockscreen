# Plan: Groeiprofiel — snelle groeiers die écht hoopvol kunnen zijn

> Geschreven 3 oktober 2026 na literatuuronderzoek (sessie Claude Code). Uitvoering per fase in
> een lokale Claude Code-sessie op Janco's computer; **begin bij de sectie "Overdracht" onderaan**.
> Dit bestand is de enige plek waar het plan leeft: werk de statuslog bij na elke fase.

## Context

Het tabblad 🌱 Groeiers levert structureel zwakke bedrijven op. Dat is geen toeval maar
definitie: `engine/selectie.py:_groeiers` neemt alleen rijen die **verlies maken én** een
3-jaars omzet-CAGR ≥ 15% hebben, en sorteert op die CAGR. Er wordt geselecteerd op precies het
ene getal dat het makkelijkst te kopen is met verlies, en nergens gekeken naar wat de groei kost.

1. **Winstgevende snelle groeiers vallen buiten de tab** (eis `reason_code == "geen_fv"`), en
   scoren elders laag op kwaliteit omdat `quality_score` ROE/ROIC > 12% en stabiele winst eist.
2. **`revenue_cagr` is kapot bij een definitiewissel** (Adyen: bruto → netto omzet gelezen als
   33% krimp). `exit_regels.omzetbreuk()` repareert dat voor de bezitspagina, maar
   `screener._calc_revenue_cagr` loopt nog op het ongerepareerde getal (CLAUDE.md zegt dit zelf).
3. Dit idee stond al geparkeerd in `docs/REVIEW-GEVOELIGHEID.md` Deelvraag 5: "aparte screen
   (omzetgroei, brutomarge-richting, verwatering, kaspositie), niet mengen in de FV-pijplijn".

**Doel:** een groeiprofiel naast het moat-profiel — een **zeef** die de structureel zwakke
groeiers eruit haalt, géén koopoordeel. Groen betekent "de cijfers spreken niet tegen". De
ambitie is de basiskans verhogen (van een paar procent naar misschien 15–20%), niet winnaars
aanwijzen. Die zin hoort op de pagina.

## Besluiten (Janco, 3 okt 2026)

- **Tab Groeiers = alle snelle groeiers** (CAGR ≥ 15%, winstgevend of niet), gerangschikt op
  het groeiprofiel. 🌱 blijft het merkteken voor de verlieslatende; `reason:growth` op "Alles"
  blijft "verlies + groei".
- **Plan in de repo** als `docs/plan-groeiprofiel.md`.
- Janco downloadt wat vanaf dit netwerk geblokkeerd is (zie "Wat Janco downloadt").

Ontwerpkeuzes in dit plan (door Claude, te overrulen):
- Een **omzetbreuk geeft grijs, geen rood** — een definitiewissel bij Yahoo is een data-artefact,
  geen zwakte van het bedrijf; rood zou de Adyen-valse-vlag herhalen.
- De drempel "snelle groeier" blijft `screening.growth_lossmaker_cagr` (0,15) in config.yaml —
  één getal, geen tweede. De naam is historisch; niet hernoemen.
- **IJkgevallen** komen uit de eigen `analyses/` en `tussenchecks/` (38 analyses, o.a. ADYEN,
  EVO, NBIS, NVDA, PAY, HUG, SDIP-B, TMV, WTN) via `engine/oordelen`, precies zoals het
  moat-profiel op 15 bekende gevallen is geijkt. Twee invarianten: geen OVERSLAAN-groeier wordt
  groen, geen KOOP-groeier wordt rood.

## Wat de literatuur zegt (de onderbouwing van de toetsen)

Volgorde = sterkte van het bewijs, Europees waar beschikbaar. Drempels in de motor zijn
**voorlopig** tot de ijking (fase B) ze vastzet.

| # | Signaal | Richting | Bewijs (VS) | Bewijs (Europa) | In DB? |
|---|---|---|---|---|---|
| T1 | Aandelenuitgifte / verwatering | negatief | Pontiff & Woodgate 2008 | McLean, Pontiff & Watanabe 2009 (41 landen): buiten de VS gedreven door slechte rendementen ná uitgifte | ja (`shares_outstanding`) |
| T2 | Balansgroei (totale activa) t.o.v. omzetgroei | negatief | Cooper, Gulen & Schill 2008 | Artikis e.a. 2022 (21 landen); Papanastasopoulos 2017: bij **verliesbedrijven ~2× zo sterk** | ja (`total_assets`) |
| T3 | Piotroski F-score bínnen groeiaandelen | positief | Piotroski 2000 | Mohr 2012 (eurozone 1999-2010, 24,6%/jr long-short); Walkshäusl 2017/2020 (15 landen, ~9,9%/jr, alle groottes) | ja (`piotroski_score`) |
| T4 | Brutowinst / activa (en brutomarge-trend) | positief | Novy-Marx 2013 | Dimensional: 15 Europese markten 1982-2014, 3,6%/jr (VS 4,4%) | ja (`gross_profit`, `total_assets`) |
| T5 | Accruals | negatief | Sloan 1996 | Gemengd: 7,4% in 8 landen (BE/FR/DE/IT/NL/ES/SE/CH) vs. recentere studie "verdwenen"; Walkshäusl 2022 DE: **lage accruals + hoge F-score** beste combinatie | ja (`accruals_ratio`) |
| T6 | Stabiliteit van de omzetgroei (G-score G5) | positief | Mohanram 2005 (17,4% vs −4,0%) | Amor-Tapia & Tascón 2016: G-score werkt nog in de eurozone (FSCORE2 en PEIS niet) | ja (`revenue`) |
| T7 | R&D-intensiteit (R&D/omzet) | positief | Chan, Lakonishok & Sougiannis 2001 | Duqi, Jaafar & Torluccio 2015 (13 landen); continentaal Europa high-tech het meest ondergewaardeerd | **nee → nieuwe kolom `rd_expense`** |
| — | Kasrunway (nettokas / kasverbranding) | informatief | praktijk, geen academisch bewijs | — | ja (`net_cash`, `fcf`) |

Geschrapt: **Regel van 40** (alleen SaaS-bewijs, MPRA 2024), **operationele hefboom** (weinig
direct bewijs als selectiesignaal; de EBIT-margereeks staat al in het moat-profiel).

Basiskansen die de toon van de pagina bepalen: 20% omzetgroei tien jaar volhouden lukt 1 op 22
(Mauboussin), vanaf een kleine basis 18 op 100. Daunfeldt & Halvarsson (Zweden): kans dat een
snelle groeier het in de volgende driejaarsperiode herhaalt ≈ 1%. Chan, Karceski & Lakonishok
2003: winstgroei houdt niet beter aan dan toeval. Verlieslatende groeiaandelen renderen slecht
en analisten zijn er stelselmatig te optimistisch over (Mohrschladt 2024). Nordic small caps
zijn naar kwaliteitsmaatstaven grotendeels "junk" (JYX-studie 1995-2023); het universum is
Zweden-zwaar.

Twee kanttekeningen: de signalen leven juist in illiquide, volatiele small caps (beperkte
arbitrage, Amor-Tapia & Tascón) — goed voor het signaal, slecht voor de uitvoering; en buiten
de VS smelten anomalieën na publicatie niet weg (Jacobs & Müller 2020), dus Europese signalen
zijn waarschijnlijk nog intact. Wat níét gevonden is: een studie over een groeikwaliteits-screen
op verlieslatende Europese small caps als zodanig; de G-score is in Europa maar één keer getest.

Bronnen: Mohanram 2005 (SSRN 403180); Amor-Tapia & Tascón 2016 (SSRN 2729191, Czech J. Econ.
Fin. 66:70-94); Mohr 2012; Walkshäusl 2017 Review of Finance 21:845-870, 2020 J. Asset
Management 21, 2022 Fin. Markets & Portfolio Mgmt 36:321-367; Novy-Marx 2013 JFE 108:1-28;
McLean/Pontiff/Watanabe 2009 JFE; Cooper/Gulen/Schill 2008 JF; Artikis/Diamantopoulou/
Papanastasopoulos 2022 Eur. J. Finance 28:1867-1891; Papanastasopoulos 2017 Economics Letters
156:106-109; Duqi/Jaafar/Torluccio 2015 Eur. J. Finance 21:444-465; Asness/Frazzini/Pedersen
QMJ; Jensen/Kelly/Pedersen 2023 JF (jkpfactors.com); Chan/Karceski/Lakonishok 2003 JF;
Mauboussin Base Rate Book; Daunfeldt & Halvarsson 2015 Small Bus. Econ.; Mohrschladt 2024 Abacus.

## Wat Janco downloadt (geblokkeerd vanaf dit netwerk)

Zet alles in `data/jkp/` resp. `data/papers/` (beide buiten de build-context en buiten git —
`data/jkp/*.csv` en `data/papers/*` toevoegen aan `.gitignore`).

1. **jkpfactors.com → Data → "Country factors"**, maandelijks, weging `vw_cap` (en `ew` als
   robuustheid), voor **NLD DEU FRA SWE POL GBR DNK FIN NOR BEL ITA ESP**. Kenmerken:
   `sale_gr1`, `sale_gr3`, `chcsho_12m`, `eqnpo_12m`, `at_gr1`, `gp_at`, `oaccruals_at`,
   `rd_sale`, `f_score`. Eén CSV per land of één grote: beide prima.
2. **Drie papers (pdf)**: Amor-Tapia & Tascón 2016 (SSRN 2729191), Papanastasopoulos 2017
   (Economics Letters 156:106-109), Mohr 2012 (F-score op eurozone-groeiaandelen). Nodig om de
   gerapporteerde drempels en portefeuillevorming te lezen vóór de ijking.
3. Optioneel: Schroders "Small is beautiful: uncovering the Nordic 10-baggers" (praktijkstudie,
   Zweden-relevant).

## Volgorde en afhankelijkheden

| Fase | Levert | Hangt af van |
|---|---|---|
| A · Data | kolom `rd_expense` end-to-end; `_calc_revenue_cagr` respecteert `omzetbreuk`; deploy + herberekening | niets — **eerst**, zodat R&D zich vult tijdens de 6-nachtsrotatie |
| C · Motor | `engine/groei_profiel.py` met **voorlopige** drempels; `groei_*`-kolommen; tab leest ze | A |
| D · UI | tab Groeiers, blok 6 op de aandeelpagina, stap 8 op /methode, regel op /start | C |
| B · Bewijs | `scripts/jkp_check.py`, `scripts/groei_backtest.py`, `docs/groeiprofiel-ijking.md`; daarna **één constants-only commit** + herberekening | A (R&D-dekking), koersbackfill; loopt parallel aan C/D |
| E · Docs | plan, CLAUDE.md, ARCHITECTURE.md, REVIEW-GEVOELIGHEID.md | doorlopend |

Wachten op B zou de tab weken blokkeren (R&D zes nachten, backfill-dekking onbekend). Daarom
C+D met voorlopige drempels, gelabeld "voorlopig — ijking volgt" op /methode; B is zo ontworpen
dat herijking alleen constanten raakt.

---

## A · Fase "Data"

### A1. Kolom `rd_expense`
- `engine/db.py`: `financials` CREATE (103-133) `rd_expense REAL` na `inventory`; **nieuw**
  ALTER-blok direct na de CREATE (financials heeft er nog geen; kopieer het stocks-patroon
  79-100) met commentaar: Yahoo-regel "Research And Development", vult zich in één rotatie,
  afwezig = "niet gerapporteerd", nooit 0. `FINANCIAL_VELDEN` (929-935): `"rd_expense"` vóór
  `"fetched_date"` → `lege_jaarrij`, `jaarrijen_met_overrides` (SELECT *) en `_update_clause`
  (COALESCE) werken dan vanzelf.
- `engine/data_fetcher.py`: jaarmapping na `gross_profit` (324):
  `rd_expense = _df_value(inc, ["Research And Development", "ResearchAndDevelopment", "Research Development"], col_idx)`,
  opslaan als `abs()`; sleutel toevoegen aan `annual_row` (393-416); in `_fetch_ttm_row`
  (457-572) via `q_sum` + return-dict; `_MONEY_FIELDS_PER_ROW` (84-91) uitbreiden (FX-conversie
  bij dual-currency). `fetch_and_store` (656-665) geeft alle sleutels door — geen wijziging.
- `engine/data_quality.py`: **bewust niet** in `_CRITICAL_ANNUAL_FIELDS` (26-37) — zou
  `completeness_pct` van elk niet-R&D-bedrijf verlagen. Commentaarregel boven de tuple.
- `app.py` `VALID_OVERRIDE_FIELDS` (2326-2337): `"rd_expense"` erbij.
- `templates/stock.html`: `bulk_fields` (512-531) en JS `BULK_FIELDS` (1390-1396): R&D-kosten.
  (De jaarcijferstabel op die pagina toont ook de brutowinst niet; R&D komt daar pas in fase D
  via `toonGroeiCijfers`.)
- Vóór deploy de Yahoo-regelnaam verifiëren op Fly (lokaal is Yahoo geblokkeerd):
  `fly ssh console` → `python -c "import yfinance as yf; print([i for i in yf.Ticker('ASML.AS').income_stmt.index if 'Research' in i])"`.

### A2. `_calc_revenue_cagr` respecteert `omzetbreuk`
- `engine/screener.py`: `from engine import exit_regels` (geen cyclus; exit_regels importeert
  alleen datetime/typing). In `_calc_revenue_cagr` (146-162) eerste regel:
  `if exit_regels.omzetbreuk(annual_rows): return None`. In `run_ticker` na regel 242: bereken
  `breuk = exit_regels.omzetbreuk(annual_rows)` één keer; is hij gevuld, zet de tekst in
  `warnings` (anders verdwijnt 🌱 bij Adyen-achtige tickers geruisloos) en geef hem door aan het
  profiel (fase C).
- Gevolgen die we accepteren: de "value trap"-waarschuwing vuurt niet meer bij een breuk;
  `is_growth_lossmaker` wordt False voor zulke tickers. Geen signaal of FV hangt van
  `revenue_cagr` af.
- **Zo uitgevoerd (4 okt 2026), met twee verfijningen:**
  1. De breuk wordt getoetst op **hetzelfde venster** als de groei (`screener._omzetvenster`,
     laatste vier jaren met omzet), niet op de hele reeks. Een wissel van vóór het venster
     raakt de driejaarsgroei niet en hoort hem dus ook niet te wissen. Adyen (breuk
     2022→2023) valt er gewoon in. De variabele `breuk` in `run_ticker` is dus de breuk
     binnen het venster; geef díé door aan het profiel in fase C.
  2. **`exit_regels.omzetbreuk` herkent nu een echte sprong** aan de brutowinst
     (`_brutowinst_beweegt_mee`). **Gecorrigeerd op 7 okt** na de lijst van 391 breuken: de
     eerste versie mat op logschaal en eiste twee positieve brutowinsten, waardoor elke groeier
     die uit het verlies komt (Alvotech, ITM Power, Dolphin Drilling) als definitiewissel uit de
     Groeiers-tab viel. Nu: de brutowinst beweegt in dezelfde richting als de omzet, met minstens
     5% van de omzetverandering (`OMZETBREUK_MEEBEWEGING`). Onder 1 mln omzet
     (`OMZETBREUK_MINIMUM`) blijft een sprong altijd een breuk, want daar is een verdubbeling ruis.
     Zonder deze regel haalde fase A juist de snelste groeiers (omzet ×2 in één jaar) uit de
     Groeiers-tab. Zonder brutowinst in beide jaren blijft de oude regel gelden: sprong =
     breuk. Gevolg voor de bezitspagina: bij een echte halvering mét meedalende brutowinst
     gaan B4/A3 nu wél af — dat is terecht, het is echte krimp.

### A3. Deploy en draaien
1. `python -m pyflakes app.py engine/*.py`; `python tests/test_handmatig_boekjaar.py`,
   `tests/test_scores_recompute.py`, `tests/test_exit_regels.py`, nieuw `tests/test_revenue_cagr.py`.
2. `fly deploy --remote-only --depot=false`, daarna `fly apps destroy fly-builder-*`.
3. **Niet** `POST /api/recalculate` voor het hele universum (synchroon; 4.123 tickers passen
   niet in de 120 s gunicorn-timeout). Volg de **deployprocedure** onderaan dit document
   (proefrun op de oude code, deploy, proefrun op de nieuwe code, vergelijken, echte run).
4. R&D vult zich over zes nachten (`fundamentals_per_night: 750`). Dekking meten met
   `scripts/groei_backtest.py --dekking` (fase B).

---

## B · Fase "Bewijs"

### B1. `scripts/jkp_check.py` — werkten de signalen in jóúw landen?
- Invoer: de CSV's uit `data/jkp/` (`--bestanden data/jkp/*.csv`). Kolommen (hoofdletter-
  ongevoelig): `location, name, weighting, freq, date, ret` (+ optioneel `n_stocks`); bij
  afwijking: gevonden koppen printen en stoppen met exit 2.
- Vlaggen: `--landen` (default de twaalf hierboven), `--kenmerken` (default de negen),
  `--weging vw_cap|ew`, `--vanaf 2000-01`, `--tot`, `--out docs/groeiprofiel-jkp.md`, `--zelftest`.
- JKP tekent elk kenmerk zo dat de lange kant de kant met het hogere verwachte rendement is;
  het script hertekent niets maar print een vaste `LANGE_KANT`-tabel naast de uitkomst
  (`at_gr1` = lage activagroei, `chcsho_12m` = weinig uitgifte, `gp_at` = hoge brutowinst/activa,
  `oaccruals_at` = lage accruals, `rd_sale` = hoge R&D, `f_score` = hoge F-score, `sale_gr*` =
  lage omzetgroei). Leesregel in de kop: een positief `sale_gr3`-rendement betekent dat snelle
  groeiers áchterblijven — dat is het basiskans-punt, geen tegenspraak met de zeef.
- Uitvoer per land × kenmerk: n maanden, gemiddeld maandrendement, ×12, t-stat, trefkans
  (% maanden > 0), plus een gepoolde rij "Europa" (gelijkgewogen over de twaalf). Stdlib
  (`csv`, `statistics`, `math`), zoals `calibrate_report.py`.
- Beslisregel in de uitvoer: een kenmerk mag een **harde** toets zijn (rood kunnen geven) als
  de gepoolde t-stat ≥ 2 én ≥ 7 van 12 landen hetzelfde teken hebben; anders hooguit geel/
  informatief.

### B2. `scripts/groei_backtest.py` — eigen universum
- **Data via `DATABASE_URL`**, zoals `import_tickers.py` (lazy `from engine import db`). Reden:
  alle financials inclusief inactieve tickers, overrides, koersen op zes datums en
  `calculated_scores`; een open `GET /api/financials/bulk` is een permanent onderhoudsoppervlak
  voor een eenmalige klus. `--cache data/groei_backtest_invoer.json` (schrijven bij de eerste
  run, daarna lezen — het `--file`-patroon), zodat Neon één keer wordt geraakt.
- Om `jaarrijen_met_overrides` nergens na te bouwen: in `engine/db.py`
  `_pas_overrides_toe(rijen, overrides)` uit `jaarrijen_met_overrides` (946-977) lichten en
  `jaarrijen_met_overrides_bulk() -> dict[str, list[dict]]` toevoegen (twee bulk-SELECTs,
  dezelfde helper). Eén regel, twee ingangen.
- Queries (alleen lezen): `stocks` (ticker, name, sector, quote_type, active,
  auto_suspended_at, presumed_delisted_at — **active=0 meenemen**, overlevingsvertekening);
  jaarrijen via de bulk-helper; koersen per datum D ∈ {vorming, +12m, +24m}:
  `SELECT DISTINCT ON (ticker) ticker, date, close FROM price_history WHERE date <= D AND date >= D-14d ORDER BY ticker, date DESC`;
  `calculated_scores.quality_score` (alleen huidig — er is geen historie).
- **Vormingsdatums** (publicatievertraging ≥ 6 maanden na boekjaareinde):
  - **F2023 (primair)**: rijen FY ≤ 2023, vorming 2024-07-01, uitkomsten 2025-07-01 (12m) en
    2026-07-01 (24m). De meeste tickers hebben 2021–2023 = 3 rijen.
  - **F2024 (secundair)**: rijen FY ≤ 2024, vorming 2025-07-01, 12m-uitkomst 2026-07-01.
  - **F2022**: alleen tickers met ≥ 3 rijen ≤ 2022 (overrides/vroege fetches) — anekdotisch;
    de meeste tickers hebben pas vanaf FY2021 cijfers.
  - Pre-flight: dekking per datum (koers binnen 14 dagen); onder 60% stoppen en eerst
    `POST /api/price-history/backfill {"period":"5y"}` + `GET /api/price-history/stats`.
- **Universum op vorming**: ≥ 3 jaarrijen ≤ vormings-FY, positieve omzet, CAGR ≥ 15% (zelfde
  functie als de motor: `groei_profiel.omzet_cagr`), geen `omzetbreuk`, geen ETF/fonds
  (`quote_type`). Geen historische `data_status` → proxy: revenue, total_assets,
  shares_outstanding gevuld op eerste en laatste rij. Beperking in het rapport benoemen.
- **Uitkomsten**: *fundamenteel* (FY2025 = nieuwste): `net_income > 0` **én** huidige
  `quality_score ≥ 6`; secundair: groei hield aan (CAGR vorming → FY2025 ≥ 10%). *Koers*:
  slotkoers-op-slotkoers (ratio, valuta valt weg; dividend uitgesloten — noteren) over 12/24m
  **minus de mediaan van het hele geprijsde universum**. *Overleving*: vormingskoers maar geen
  uitkomstkoers → `presumed_delisted_at` gezet: −50% (conservatief) én apart gerapporteerd;
  anders "onbekend".
- **Per toets**: `groei_profiel.bouw_profiel(rijen_asof, piotroski=quality_score.piotroski_fscore(...), drempels=...)`,
  bucket per `uitkomst` (groen/geel/rood/onbekend) en per `niveau`: n, fundamentele trefkans,
  12m/24m-trefkans, mediaan relatief rendement. **Drempelraster** vooraf vastgelegd in het script
  (T1 rood op 3/5/8/12 %/jr; T2 gat op 10/15/20/30 pp; T4 gp/at op 0,10/0,15/0,20; T6 stdev op
  15/25/35 pp). Keuzeregel: de grofste drempel waarbij de rode bucket n ≥ 20 heeft en strikt
  slechtere trefkansen dan niet-rood op **beide** uitkomstfamilies en beide vormingen. Geen
  drempel voldoet → toets gedegradeerd naar geel-only (kan geen rood geven). De regel staat in
  het rapport, zodat de keuze controleerbaar is en niet gefit.
- Uitvoer `docs/groeiprofiel-ijking.md` (`--out`): datum, n per vorming, dekking, de tabellen,
  gekozen drempels met n en trefkans, gedegradeerde toetsen, en de "wat als"-telling op het
  huidige universum (groen/geel/rood/grijs). Daarna één commit met de constanten in
  `engine/groei_profiel.py` (met ijk-commentaar zoals moat_profile 37-71) en de herberekening.
- **Voorlopig tot B**: alle drempels in C, de gewichten, `GROEN_SCORE`. **Niet voorlopig**:
  de datapoorten (MIN_RIJEN, data_status, omzetbreuk), TTM-uitsluiting, "R&D ontbreekt = onbekend".

---

## C · Fase "Motor" — `engine/groei_profiel.py`

Spiegel van `engine/moat_profile.py`: puur op meegegeven data, constanten met ijk-commentaar,
`drempels` reizen mee in de uitkomst zodat /methode geen tweede kopie heeft.

### C1. API
```python
def bouw_profiel(annual: list[dict], sector: str | None = None,
                 sector_mediaan_rd: float | None = None, *,
                 piotroski: dict | None = None, data_status: str | None = None,
                 groei_drempel: float = 0.15, drempels: dict | None = None,
                 omzetbreuk: str | None = None) -> dict
def compact(profiel: dict) -> dict        # alleen scalars, voor calculated_scores.groei_profiel
def omzet_cagr(annual: list[dict]) -> float | None   # zelfde vensterregel als screener._calc_revenue_cagr
```
- `piotroski`: `q_result["piotroski"]` uit `run_ticker`, zodat het profiel dezelfde F-score
  toont als de aandeelpagina; None → `quality_score.piotroski_fscore` op de jaarrijen (backtest,
  `/api/stock`). Documenteren dat productie de TTM-rij meetelt en de backtest niet.
- `data_status`: dezelfde poort als de screener; bad/missing → grijs.
- `drempels`: override-dict over de constanten (alleen het backtest-raster); `drempels` in de
  uitkomst is wat werkelijk gebruikt is.
- Sleutels (spiegel van moat): `niveau`, `kop`, `score` (0–10 of None), `redenen` (platte
  tekst), `toetsen` (lijst `{code, naam, uitkomst, waarde, drempel, bron, uitleg}`), `drempels`,
  `series` (`omzet`, `activa`, `aandelen`, `brutomarge_pct`, `rd_pct`, `omzet_per_aandeel`, elk
  `[{jaar, waarde}]`), scalars `omzet_cagr`, `activa_cagr`, `aandelen_cagr`,
  `omzet_per_aandeel_cagr`, `groeier` (bool), `verlieslatend` (bool), `kasrunway_jaren`,
  `jaren`, `venster` ("2022–2025"), `omzetbreuk`.

### C2. Rijen en venster
`_jaarreeks(annual)`: rijen met `period_type == 'annual'` (default 'annual', zoals
`omzetbreuk`) **en** truthy `fiscal_year` — dat laat de TTM-rij (`fiscal_year=0`) per
constructie weg. Oplopend sorteren, ontdubbelen per jaar, laatste `VENSTER_RIJEN = 4` (3
intervallen). `MIN_RIJEN = 3`. Alle CAGR's over eerste/laatste rij van het venster. Een test
borgt `omzet_cagr(rows) == screener._calc_revenue_cagr(rows)` bij breukvrije invoer: één
getal voor "omzetgroei".

### C3. Toetsen (voorlopige drempels)

| Code | Naam | Formule | Onbekend als | Voorlopig | Hard? |
|---|---|---|---|---|---|
| T1 | Verwatering | CAGR `shares_outstanding`; ook `omzet_per_aandeel_cagr` | < 2 rijen met aandelen, of jaar-op-jaar sprong ≥ factor 2 ("aandelenbreuk": klassewissel/split-artefact), of `dubbel_soort == aandelenklasse` | rood ≥ 5 %/jr, geel 2–5, groen < 2 | ja |
| T2 | Balansgroei | CAGR `total_assets` − CAGR `revenue` (pp), zelfde venster | activa ontbreekt op eerste/laatste rij | rood ≥ 15 pp én activa-CAGR ≥ 20 %; geel ≥ 5 pp; anders groen. Bij verlies: zelfde drempel, kop meldt dat het effect dan ~2× zo sterk is | ja |
| T3 | F-score | `piotroski_score` (0–9) | < 7 van 9 criteria bekend | rood ≤ 3, geel 4–5, groen ≥ 6 | ja |
| T4 | Brutowinstkracht | niveau = `gross_profit`/`total_assets` laatste rij (+ 3j-gemiddelde); trend via `moat_profile.marge_reeks(rijen, "gross_profit")` → `_trend` | gross_profit ontbreekt (banken) | rood als niveau < 0,10 **of** trend ≤ `moat_profile.MARGE_EROSIE_PP` (−3 pp, geïmporteerd — één getal); groen als niveau ≥ 0,33 zonder erosie; anders geel | ja |
| T5 | Accruals | gemiddelde (`net_income` − `operating_cf`)/`total_assets` ×100 over het venster (zelfde formule als `screener._calc_accruals`, alleen boekjaren; herimplementeren om een importcyclus te vermijden) | < 2 bruikbare jaren | rood ≥ +10, geel 0–10, groen < 0 | nee (gewicht 0,5) |
| T6 | Groeistabiliteit | populatie-stdev van de jaarlijkse omzetgroei (pp) | < 2 intervallen | groen < 10 pp, geel 10–25, rood > 25 | nee |
| T7 | R&D-intensiteit | `rd_expense`/`revenue` laatste rij | `rd_expense` None **of ≤ 0** | met `sector_mediaan_rd`: groen ≥ mediaan, anders geel; zonder: groen ≥ 5 %, geel < 5 %. **Nooit rood** (bewijs is asymmetrisch) | nee (0,5) |

Informatief (in `redenen`, niet gescoord): `kasrunway_jaren = net_cash / |fcf|` bij verlies én
negatieve FCF (en "geen kasbuffer" bij net_cash ≤ 0); operationele-margetrend via
`moat_profile.marge_reeks(rijen, "ebit")`.

**Score**: groen 1, geel 0,5, rood 0; gewichten T1 1,5 · T2 1,5 · T3 1 · T4 1 · T5 0,5 · T6 1 ·
T7 0,5; `score = 10 × Σ(w·punt) / Σ(w over bekende toetsen)`; None bij < 3 bekende toetsen.

**Sector-relatieve R&D**: nu overslaan; parameter bestaat, `run_ticker` geeft None door. Blijkt
uit B dat sector-relatief beter werkt: mediaan per sector één keer per herberekening uit
`calculated_scores.groei_profiel` (`rd_pct`) via config — niet nu.

### C4. `_oordeel` — eerste treffer wint
1. `data_status` bad/missing → grijs, "Cijfers afgekeurd — geen groeiprofiel".
2. < MIN_RIJEN → grijs, "Te weinig boekjaren om groei te beoordelen".
3. `omzetbreuk` → grijs, "Definitiewissel in de omzetreeks — groei niet te meten".
4. `omzet_cagr < groei_drempel` → grijs, "Geen snelle groeier (omzet +x %/jr) — profiel ter
   informatie" (toetsen worden wél berekend en getoond).
5. T1 rood → rood, "Groei wordt betaald met nieuwe aandelen".
6. T2 rood → rood, "De balans groeit veel harder dan de omzet".
7. T3 rood → rood, "Zwakke fundamentele trend (F-score ≤ 3)".
8. T4 rood → rood, "Brutowinst te mager of brokkelt af".
9. Rood in T5/T6 → geel, "Eén zwak punt: …".
10. Geen rood, ≥ 4 toetsen bekend, `score ≥ GROEN_SCORE` (7,0) → groen, "De cijfers pleiten
    niet tegen de groei".
11. Anders geel, "De cijfers spreken zich niet uit".

Groen is bewust als afwezigheid van bewijs geformuleerd; de UI voegt de basiskans-zin toe.

### C5. Integratie in `run_ticker` en opslag
- `engine/screener.py`: na `q_result` (305):
  `groei = groei_profiel.bouw_profiel(annual_rows, sector=sector, piotroski=q_result.get("piotroski"), data_status=dq_status, groei_drempel=config["screening"].get("growth_lossmaker_cagr", 0.15), omzetbreuk=breuk)`
  — **`annual_rows`, niet `calc_rows`**. Result dict: `groei_niveau`, `groei_score`,
  `groei_profiel` (volledig, gaat mee naar `/api/trace`). `upsert_scores`: `groei_niveau`,
  `groei_score`, `groei_profiel=groei_profiel.compact(groei)`.
- Beide vroege uitgangen (248-255, 264-270) geven ook `groei_niveau="grijs", groei_score=None,
  groei_profiel=None` mee — `upsert_scores` schrijft alleen de meegegeven sleutels, anders houdt
  een ticker die `bad` wordt een oud groen.
- `engine/db.py`: ALTER-lus (218-226) + `("groei_niveau","TEXT"), ("groei_score","REAL"),
  ("groei_profiel","TEXT")`; `JSON_SCORE_KOLOMMEN` (915-918) én `_JSON_VELDEN` (1314-1315) +
  `"groei_profiel"`; `_DASHBOARD_SQL` (1253) + `c.groei_niveau, c.groei_score, c.groei_profiel`;
  `_verklein_voor_lijst` (1324-1352): `groei_profiel` vervangen door `groei_kop` (str) en
  `groei_rood` (lijst codes met uitkomst rood), rest weg — de compacte JSON is ~300 B × 4.123
  rijen, te veel voor een cache die al eens een OOM gaf.
- `app.py` `_verrijk_dashboardrijen` whitelist (730-792): `is_groeier` (= `rev_cagr ≥ drempel`
  en data_status niet bad/missing); `is_growth_lossmaker = is_groeier and reason_code == "geen_fv"`
  (betekenis ongewijzigd); `groei_niveau`, `groei_score`, `groei_kop`, `groei_rood`.
- `engine/selectie.py` `_groeiers`: filter `is_groeier`; sorteer op
  `(-NIVEAU_RANG[groei_niveau], -groei_score, -revenue_cagr)` met
  `NIVEAU_RANG = {"groen":3,"geel":2,"grijs":1,"rood":0}` — rood onderaan, zichtbaar
  ("markeren, niet wegfilteren"). `_extra_filters` `reden == "growth"` blijft op
  `is_growth_lossmaker`.
- `api_stock_detail` (3360-3396): `"groei": groei_profiel.bouw_profiel(annual, sector=..., piotroski={"score": scores.get("piotroski_score"), "criteria": scores.get("piotroski_breakdown")}, data_status=...)`
  — live mét series, zoals moat.
- `/api/trace` (2652-2751): `"groei": calc.get("groei_profiel")`. Tegelijk de eenregelige
  reparatie `run_ticker(t, cfg, persist=False)` — een GET hoort niet te schrijven; apart
  benoemen in de commit.
- `_run_scores_recompute_job` (3120-3173): `groei_overgangen` tellen in het resultaat, zodat de
  dry run het effect van de zeef laat zien.
- `tests/test_scores_recompute.py` `_NepDb`: `_jaarrij` krijgt `rd_expense`.

---

## D · Fase "UI en navigatie"

**Navigatie: `templates/base.html` verandert niet.** Groeiers blijft een dashboard-tabblad
(Dashboard → 🌱 Groeiers), het detail is blok 6 op `/stock/<T>`, de drempels staan op
`/methode` stap 8, en `/start` krijgt één regel. Geen nieuwe pagina, geen nieuw menu-item.

- `templates/index.html`
  - `TABS` groeiers (503-505) uitleg: "Bedrijven met minstens 15% omzetgroei per jaar,
    winstgevend of niet. Het groeiprofiel is een zeef, geen koopadvies: rood betekent dat de
    cijfers tegen de groei pleiten (verwatering, balans die harder groeit dan de omzet, zwakke
    F-score), groen alleen dat ze er níet tegen pleiten. Reken niet op winnaars — van de
    bedrijven die drie jaar hard groeien doet maar een klein deel dat nog eens drie jaar."
  - `renderFocus` kolommen (1076-1079): `['', 'Ticker', 'Naam', 'Sector', ['Omzetgroei','r','revenue_cagr'], ['Profiel','r','groei_score'], ['F-score','r','piotroski_score'], ['Kwaliteit','r','quality_score'], ['Waarom','r','']]`.
    Sorteersleutel Profiel = de numerieke score (niveau-tekst sorteert alfabetisch).
  - Rij (1130-1139): 🌱 in de naamcel bij `is_growth_lossmaker` (met title); `groeiPil(r)` op
    basis van `NIVEAU_STIJL` (592; `geel → oranje`, de map kent geen `geel`), bijv. `● groen 7,5`;
    laatste cel `groeiUitleg(r)` = `groei_kop` + (`groei_rood` ? " · rood op " + namen : "") +
    (verlieslatend ? " · verlieslatend" : "") + (`dubbel_soort` ? " · " + soort : ""). Kleuren
    uitsluitend via tokens (`var(--buy)` enz.), niet via de Tailwind-schaal.
  - Rijtelling (1172-1173): voor groeiers ` · n groen · n geel · n rood · n zonder profiel`.
  - `reasonTooltip` (1380-1382) ongewijzigd. Nieuwe `_naam`-globals declareren
    (template-JS-test); geen `const fmt`-achtige herdeclaraties.
- `templates/stock.html`: in `bouwBeslisboom` na het moat-blok `const g = d.groei; if (g) {...}`
  → `blok('6 · Groeiprofiel', g.niveau, g.kop, regels)`; `regels` = `g.redenen` + grijze
  slotregel: rood → "Een zeef, geen oordeel over het bedrijf: de cijfers pleiten tegen de
  groei"; groen → "Groen zegt alleen dat de cijfers er niet tegen pleiten. Van hard groeiende
  bedrijven houdt maar een klein deel het vol — dit is geen koopoordeel"; grijs met `!g.groeier`
  → "Geen snelle groeier; het profiel is ter informatie". Nieuw
  `<div id="groei-cijfers" class="hidden mb-6">` na `#moat-cijfers` (139); `toonGroeiCijfers(g)`:
  jaartabel (Omzet via `fmtBig`, Balanstotaal, Aandelen, Brutomarge %, R&D % of "—",
  Omzet/aandeel) + toetsentabel (code, naam, waarde, drempel, uitkomst-stip via `STOPLICHT`).
  Zes blokken = drie rijen in het `md:grid-cols-2`-raster. Invoer bewaken zoals
  `Array.isArray` op 877: één fout maakt de hele beslisboom leeg.
- `templates/methode.html`: `{ id: 'groei', titel: '8 · Groeiprofiel — is de groei gezond?' }`
  op index 7; signaal/grenzen worden 9/10 en **elke `STAPPEN[n]`-index** (regels 137-393;
  `bouwMoat` gebruikt `STAPPEN[6]`) opschuiven. `bouwGroei(t)` leest `t.groei`: oordeel,
  toetsentabel met `drempel` uit de motor, de formules, regel "voorlopige drempels — ijking
  volgt" tot B klaar is, de basiskans-alinea (Mauboussin 1 op 22; Daunfeldt & Halvarsson ≈ 1%),
  korte bronnenlijst met link naar de GitHub-versie van `docs/plan-groeiprofiel.md` (docs/ zit
  niet in het image).
- `/start`: `_routekaart` (480-493) `"groei": scores.get("groei_niveau")` als
  `revenue_cagr ≥ drempel` (opgeslagen waarde, geen herbouw); `templates/start.html` krijgt een
  feit "Groeiprofiel" naast Moat (172-175).

---

## E · Docs
- `docs/plan-groeiprofiel.md`: dit plan + de literatuurtabel met bronnen + een statuslog.
- `CLAUDE.md`: tab-tabel (~157-164) Groeiers-regel; nieuwe sectie "Groeiprofiel (2026-10)" met
  verwijzing naar het plan en de twee zinnen die ertoe doen (zeef, geen oordeel; R&D ontbreekt
  ≠ 0); de Adyen-valkuil (320-329) bijwerken: `revenue_cagr` **is** nu gerepareerd; nieuwe
  valkuilen "`shares_outstanding`-historie wordt door de fetcher afgevlakt" (zie G1) en
  "`/api/recalculate` past bij 4.123 tickers niet meer in de timeout".
- `docs/ARCHITECTURE.md`: Modules-tabel + `engine/groei_profiel.py` (en `engine/moat_profile.py`,
  ontbreekt nu); nieuw `## Groeiprofiel` na de Fair value-sectie (vóór regel 161); DB-schema:
  financials `+ rd_expense`, calculated_scores `+ groei_niveau/groei_score/groei_profiel`.
- `docs/REVIEW-GEVOELIGHEID.md` Deelvraag 5 (283-292): "**Status 2026-10:** opgepakt als aparte
  zeef buiten de FV-pijplijn, conform dit advies — zie docs/plan-groeiprofiel.md."

---

## F · Verificatie per fase
- **A**: nieuw `tests/test_revenue_cagr.py` (Adyen-reeks 8936→1863→2226→2647 → None; gewone
  reeks → waarde; TTM-rij genegeerd); `tests/test_handmatig_boekjaar.py` (FINANCIAL_VELDEN-lus;
  `_SCORE_KOLOMMEN` 38-45 in C uitbreiden met `groei_*`); bewaker dat `rd_expense` in
  `_MONEY_FIELDS_PER_ROW`, `FINANCIAL_VELDEN` en `VALID_OVERRIDE_FIELDS` zit; pyflakes; proefrun
  op oude en nieuwe code gelijk (zie de deployprocedure).
- **C**: nieuw `tests/test_groei_profiel.py` (helper `_jaren` als test_moat_profile 18-22): één
  test per T1–T7 incl. onbekend-paden; TTM-rij uitgesloten; omzetbreuk → grijs; `data_status='bad'`
  → grijs; < 3 rijen → grijs; niet-groeier → grijs mét toetsen; harde-rood-volgorde; groen eist
  geen rood en ≥ 4 bekend; `drempels`-override verandert de uitkomst; `compact()` zonder
  `series` en `json.dumps`-baar; `omzet_cagr == screener._calc_revenue_cagr`; ontbrekende R&D
  verlaagt de score nooit; **ijkgevallen uit analyses/tussenchecks** (groeiers met oordeel:
  geen OVERSLAAN wordt groen, geen KOOP wordt rood). Uitbreiden: `tests/test_scores_recompute.py`
  (groei-kwargs opgeslagen; dry run schrijft niets), `tests/test_selectie.py` (winstgevende
  groeier groen, verlieslatende geel, rode groeier → volgorde groen/geel/rood; `reden="growth"`
  blijft verliesmakers), `tests/test_lijst_payload.py` (`groei_profiel` weg, `groei_kop`/
  `groei_rood` aanwezig), `tests/test_trace.py` (`groei`-sleutel).
- **D**: `python tests/test_template_javascript.py`; browsercheck volgens CLAUDE.md op `/`,
  `/stock/ADYEN.AS`, `/stock/ASML.AS`, `/methode`, `/start?ticker=ASML` (6 blokken, geen
  console-fouten).
- **Meten en vastleggen** (statuslog in het plan): proefrun op oude en nieuwe code gelijk
  (het profiel raakt geen signaal en geen fair value — zie de deployprocedure);
  `groei_overgangen`; `/api/dashboard/selectie?tab=groeiers` tellingen per niveau en totaal;
  aantal tickers waarvan `revenue_cagr` None werd ("definitiewissel" in warnings); R&D-dekking
  na zes nachten; cache-bytes per rij vóór/na.
- **B**: `jkp_check.py --zelftest` (synthetische CSV met bekende mean/t); `groei_backtest.py
  --zelftest` (drie synthetische tickers, bekende buckets); het ijkrapport gecommit.

---

## G · Risico's en open punten
1. **`shares_outstanding`-historie wordt afgevlakt** (`data_fetcher.py` 353-368): wijken
   balans- en `info.sharesOutstanding` > 2× af, dan krijgt *elk* jaar het huidige aantal — een
   echte verdubbeling wordt 0% verwatering (vals groen op T1), en gemengde jaren geven valse
   sprongen. T1 ziet 2–40 %/jr maar geen ≥ 2×-gebeurtenissen; de "aandelenbreuk → onbekend"-regel
   vangt de valse sprongen. De backtest beslist of T1 überhaupt signaal draagt; zo niet →
   geel-only en leunen op Piotroski F7.
2. **A/B-aandelenklassen** vertekenen verwatering; `dubbelingen.py` markeert ze —
   `dubbel_soort == aandelenklasse` → T1 onbekend, en de soort in de één-regel-uitleg.
3. **Dual-listings**: secundaire tickers zonder financials → grijs; beide lijnen van één bedrijf
   kunnen in de tab staan — bestaande markering geldt.
4. **Regime 2022**: het 24-maandsvenster (2024-07 → 2026-07) is één regime; F2024 heeft maar
   12 maanden. In het rapport zeggen; drempels niet aanscherpen op één cyclus.
5. **`/api/trace` schrijft bij GET** en bouwt moat zonder overrides (2670, 2736) — bestaand;
   alleen de eenregelige `persist=False` wordt voorgesteld.
6. **Geen historische `data_status`/scores** → de backtest-poort is een proxy en de fundamentele
   uitkomst gebruikt de huidige `quality_score`; milde look-ahead op de poort, geen op de uitkomst.
7. **Yahoo R&D-regelnaam** verifiëren op Fly vóór deploy; IFRS-rapporteerders die R&D alleen in
   "Operating Expense" zetten → onbekend, by design.
8. **Geheugen/payload**: +4 kleine velden per cacherij ≈ 0,4 MB totaal; de volledige JSON mag
   nooit de lijst bereiken (bewaakt door test_lijst_payload).
9. **`/api/recalculate`-timeout** bij 4.123 tickers — de achtergrond-recompute gebruiken.
10. **Open voor Janco**: wil je extra ijkgevallen vastpinnen naast de analyses/tussenchecks?

## Statuslog

| Datum | Fase | Wat | Meting |
|---|---|---|---|
| 2026-10-03 | — | Plan geschreven na literatuuronderzoek; gecommit als `docs/plan-groeiprofiel.md` | — |
| 2026-10-03 | B (voorbereiding) | Janco: JKP-landfactoren (12 landen, `vw_cap`, 9 kenmerken) in `data/jkp/`, drie papers in `data/papers/` — lokaal op zijn machine, buiten git | — |
| 2026-10-04 | A | Kolom `rd_expense` end-to-end (db + migratie, fetcher jaar/TTM/FX, overrides, handmatig formulier); `_calc_revenue_cagr` geeft None bij een breuk binnen het venster, met reden in warnings; `omzetbreuk` herkent echte sprongen aan de brutowinst (zie A2). Nieuw `tests/test_revenue_cagr.py` (9 tests; 3 falen op de oude code). CLAUDE.md-valkuil Adyen bijgewerkt. | Lokaal: pyflakes schoon, template-JS 0 fouten, 35/35 testbestanden groen. **Gedeployed 4 okt 20:30, herberekend 6 okt** (Claude Code, zonder tussenkomst). R&D-regelnaam op Fly bevestigd: `['Research And Development']` voor ASML.AS. Dry run op de oude en de nieuwe code gaven **exact hetzelfde**, per overgang: `fv_gewijzigd` 0 en 22 signaalovergangen (SELL→HOLD 15, HOLD→SELL 4, BUY→HOLD 2, BUY→STRONG BUY 1) — fase A raakt dus geen signaal en geen fair value, zoals bedoeld. De 22 overgangen zijn koersdrift tegen de opgeslagen scores, geen effect van deze fase. Echte run 6 okt: 4.050 doorgerekend, 0 fouten, `fv_gewijzigd` 0, 31 signaalovergangen (HOLD→SELL 16, SELL→HOLD 13, BUY→HOLD 1, BUY→STRONG BUY 1 — meer dan de 22 van zondag omdat er twee nachtrondes tussen zaten). Daarna `/api/recalculate` op ADYEN.AS voor de cache. **Dashboard voor → na:** `revenue_cagr` null 395 → **786** (+391), `is_growth_lossmaker` 177 → **109**, tab Groeiers 177 → **109**. Die 68 verdwenen groeiers stonden dus op een omzetreeks met een factor-2-sprong erin; fase C geeft ze grijs in plaats van ze te laten vervallen. ADYEN.AS zelf: `revenue_cagr` −0,333 → None. R&D-dekking na twee nachten: 296 tickers / 1.248 jaarrijen gevuld — op koers voor de volledige rotatie van zes nachten; eindstand nog meten. |
| 2026-10-07 | A (correctie) | Lijst van de 391 omzetbreuken in `docs/fase-a-breuken.csv` (lokale sessie). Bevinding: 233 van de 391 zijn fondsen, trusts en financials met springerige "omzet" — daar is geen groeicijfer juist. Maar de brutowinstregel kon niet vuren bij negatieve brutowinst, dus elke groeier die uit het verlies komt viel weg. Regel herschreven naar absolute meebeweging (≥5% van de omzetverandering, zelfde richting) plus een ondergrens van 1 mln omzet. Twee tests op echte cijfers (Alvotech, ITM, Xbrane-patroon, kleine omzet). | Simulatie op de lijst: 52 van de 391 krijgen hun groeicijfer terug, waarvan 19 van de 68 Groeiers (o.a. Alvotech, Dolphin Drilling, Senzime, Shape Robotics, ITM Power, Northern Ocean, Nebius); de rest zijn echte krimpers (OCI, Exor). Lokaal 35/35 groen. **Gedeployed en herberekend 7 okt** (lokale sessie). Proefrun oud en nieuw gaven **exact hetzelfde**: `fv_gewijzigd` 0, 22 signaalovergangen (HOLD→SELL 13, SELL→HOLD 6, HOLD→BUY 2, INSUFFICIENT DATA→HOLD 1). Echte run: 4.050 doorgerekend, 0 fouten, `fv_gewijzigd` 0, dezelfde overgangen. **Dashboard voor → na:** `revenue_cagr` null 786 → **791**, `is_growth_lossmaker` 109 → **107**, tab Groeiers 109 → **107**. Van de 391 breuken uit `fase-a-breuken.csv` kregen **45** hun cijfer terug (simulatie: 52; het verschil is data die sinds de simulatie ververst is); ALVO-SDB.ST (+92%), DDRIL.OL (+104%), SEZI.ST (+95%) en SHAPE.CO (+155%) hebben weer een `revenue_cagr`, ADYEN.AS niet. **Let op: het aantal nulls daalde dus niet.** De 346 resterende breuken uit de lijst zijn echte definitiewissels of springerige fondsen; daarnaast staan er nu 445 nulls buiten de lijst (was 395 vóór fase A): 184 NEGATIEVE OMZET, 114 GEEN FV (VERLIES), 91 GEEN DATA — in een steekproef van 25 hadden 20 minder dan twee omzetjaren, dus geen omzetgroei te meten. Dat verschil van ~50 is niet verder uitgezocht; vermoedelijk de nachtelijke rondes sinds 4 okt. De winst van de correctie is daarom kleiner dan het grafiekje deed vermoeden: 45 van 68 verdwenen groeiers hadden dit nodig, de tab Groeiers kwam niet terug op 177 maar op 107. |
| 2026-10-07 | C, D, E | Groeiprofiel gebouwd: `engine/groei_profiel.py` (7 toetsen, voorlopige drempels), `groei_*`-kolommen, tab Groeiers = alle snelle groeiers gerangschikt op profiel, blok 6 op de aandeelpagina, stap 8 op /methode, regel op /start, `/api/trace` schrijft niet meer bij GET. Nieuw `tests/test_groei_profiel.py` (15 tests). | Lokaal: pyflakes schoon, template-JS 0 fouten, tests groen. **Nog te doen:** deploy, meting per niveau, browsercheck. |

## Overdracht: waar het werk staat en hoe je verdergaat

Bijgewerkt 7 oktober 2026, bij de overstap van de cloudsessie naar een lokale Claude
Code-sessie op Janco's eigen computer (`C:\Users\janco\stockscreen`). Lokaal zijn `fly`
(ingelogd) en de databaseverbinding beschikbaar; in de cloud niet. Werk daarom lokaal.

**Stand:** fase A is gedeployd en herberekend. De correctie op de omzetbreuk (commit
`2350333`, statuslog-rij "A (correctie)") staat op de branch maar is **nog niet gedeployd**.
Daarna volgen C → D → B → E.

**Eerstvolgende stap:** deploy de correctie volgens de procedure hieronder en vul de meting
in de rij "A (correctie)" in. Controleer daarbij dat Alvotech (ALVO-SDB.ST), Dolphin Drilling
(DDRIL.OL), Senzime (SEZI.ST) en Shape Robotics (SHAPE.CO) weer een `revenue_cagr` hebben en
Adyen (ADYEN.AS) niet.

**Werkwijze per fase**
- Branch `claude/nice-hamilton-udncm0`. Lees CLAUDE.md, dit plan en `engine/moat_profile.py`
  (het patroon voor `engine/groei_profiel.py`).
- Nederlands in commentaar, UI-tekst en commits. Bouw niets na dat al bestaat
  (`db.jaarrijen_met_overrides`, `moat_profile.marge_reeks`, `exit_regels.omzetbreuk`,
  `quality_score.piotroski_fscore`, `selectie.TABBLADEN`, `NIVEAU_STIJL`). Kleuren alleen via de
  tokens uit `templates/base.html`. Geen herdeclaratie van `fmt` e.d. in een template.
- Vóór elke commit: `python -m pyflakes app.py engine/*.py`, `python tests/test_template_javascript.py`
  en de tests uit sectie F. Let op: veel testbestanden draaien hun controles bij het laden en
  eindigen met `sys.exit`, dus `pytest tests/` in één keer werkt niet; draai elk bestand los met
  `python tests/<bestand>.py`.
- Drempels in `engine/groei_profiel.py` zijn **voorlopig** tot fase B; zo labelen op /methode.
- Na een UI-wijziging de pagina echt in de browser bekijken (CLAUDE.md).
- Werk aan het eind van elke fase de statuslog bij, commit en push.

**Deployprocedure (voor elke fase met codewijziging)**
1. Niet tussen 02:45–05:30 of 18:15–20:00 Amsterdamse tijd: dan lopen de verversrondes en is
   de vergelijking niet zuiver.
2. Meting vooraf uit `/api/dashboard`: aantal rijen met `revenue_cagr` null, aantal met
   `is_growth_lossmaker`, en `totaal` uit `/api/dashboard/selectie?tab=groeiers` (vanaf fase C
   ook de telling per `groei_niveau`).
3. Proefrun op de **oude** code: `POST https://stockscreen-janco.fly.dev/api/scores/recompute`
   met `{"dry_run": true}` (geen token nodig), wachten tot `/api/refresh/status?job_id=…` status
   `done` geeft. Bewaar `signaal_overgangen` en `fv_gewijzigd`.
4. `fly deploy --remote-only --depot=false`, daarna elke `fly-builder-*` app opruimen met
   `fly apps destroy`.
5. Wachten tot `/api/health` antwoordt; proefrun op de **nieuwe** code.
6. **Vergelijk 3 en 5, per overgang.** Ze moeten gelijk zijn. Een losse proefrun is nooit
   "leeg": hij vergelijkt met opgeslagen scores die tot zes dagen oud zijn, terwijl de koersen
   elke avond verversen. Alleen het verschil tussen twee proefruns vlak na elkaar zegt iets over
   de code. Verschillen ze: stoppen en melden.
7. Echte run: zelfde endpoint met `{"dry_run": false}`. Daarna
   `POST /api/recalculate {"tickers":["ADYEN.AS"]}` om de dashboardcache te legen.
8. Meting achteraf zoals in stap 2; uitkomst in de statuslog.

**Kosten:** een proefrun duurt ongeveer een half uur. Wacht met één achtergrondcommando dat
pas eindigt als de status `done` is, niet met losse controles per halve minuut; elke controle
is een beurt van het model.

**Fase B:** de JKP-landfactoren en de drie papers staan op Janco's werklaptop in
`C:\Users\nolj\.gemini\antigravity\scratch\stockscreen\data\` (`jkp\`, `papers\`). Kopieer
ze naar `data\` in deze repo, of download de JKP-bestanden opnieuw (12 landen, maandelijks,
`vw_cap`, kenmerken in sectie "Wat Janco downloadt").

## Kritieke bestanden
- `engine/screener.py` — `_calc_revenue_cagr` + `omzetbreuk`, profiel-aanroep in `run_ticker`,
  opgeslagen kwargs en grijs bij vroege uitgangen
- `engine/db.py` — `rd_expense` + nieuw financials-ALTER-blok, `FINANCIAL_VELDEN`, `groei_*`-
  kolommen, `JSON_SCORE_KOLOMMEN`/`_JSON_VELDEN`, `_DASHBOARD_SQL`, `_verklein_voor_lijst`,
  bulk-overrides-helper
- `engine/moat_profile.py` — het patroon (`_f`, `marge_reeks`, `_trend`, `MARGE_EROSIE_PP`,
  `_oordeel`, `drempels` in uitkomst) voor het nieuwe `engine/groei_profiel.py`
- `engine/data_fetcher.py` — mapping `rd_expense` (jaar + TTM + FX-lijst)
- `engine/selectie.py` — `_groeiers`
- `app.py` — whitelist `_verrijk_dashboardrijen`, `api_stock_detail`, `/api/trace`,
  `_routekaart`, `VALID_OVERRIDE_FIELDS`, recompute-job
- `templates/index.html`, `templates/stock.html`, `templates/methode.html`, `templates/start.html`
- `scripts/jkp_check.py`, `scripts/groei_backtest.py` (nieuw)
- `docs/plan-groeiprofiel.md` (nieuw), `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/REVIEW-GEVOELIGHEID.md`

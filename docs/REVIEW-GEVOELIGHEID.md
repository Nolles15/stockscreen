# Gevoeligheidsreview van de drietrapspijplijn

Datum: 2026-08-06 · Uitgevoerd read-only tegen productie (2.759 tickers) plus
een steekproef van 292 tickers met volledige jaarrijen. Elke bevinding draagt
het getal én het script waarmee hij is vastgesteld; de scripts staan in de
sessie-scratchpad (`review/sim1..sim6`), de brondata is één momentopname van
6 augustus 2026.

**Leeswijzer.** Per deelvraag: bevinding → cijfer → aanbeveling, of expliciet
"staat goed". De aanbevelingen zijn genummerd (A1–A13) en aan het eind
verzameld met prioriteit. Er is tijdens deze review níéts gewijzigd aan
database, config of engine-code — elke aanbeveling is een apart voorstel.

---

## Samenvatting in zeven zinnen

1. De **kerncijfers van Yahoo kloppen**: omzet, nettowinst, kasstroom en eigen
   vermogen wijken mediaan 0,0% af van de geverifieerde cijfers uit de eigen
   analyses; alleen FCF wijkt vaker af (definitieverschillen).
2. De **EV-consistentiecheck is stuk** (rekent met kas = 0) en circa een derde
   van zijn 313 meldingen is vals alarm — waaronder de meldingen die in de
   tussenchecks van SDG en INF als aandachtspunt zijn geciteerd.
3. Het aandeel **koopsignalen is 1,7%** tegen een eigen streefband van 5–15%,
   en dat is **niet** met drempels te repareren: de fair value ankert 65% op
   de eigen historische multiples, waardoor bedrijven die duurder werden dan
   hun eigen verleden per constructie overgewaardeerd ogen.
4. Daardoor staan **kwaliteits-compounders massaal onderaan** (Lifco −82%,
   Addtech −91%, ASML −173% "korting"; 10 van de 25 referentienamen op SELL)
   — de structurele vals-negatief-bron van stap 1 is hiermee bevestigd én
   verklaard.
5. Het **kwaliteitsoordeel voorspelt de analyse-uitkomst nauwelijks**
   (r = 0,28) en de kwaliteitspoort blokkeerde precies de twee bedrijven die
   de volledige analyse koopwaardig vond (PRX en ADYEN, beide kwaliteit 5,0).
6. De **moat-oordelen van de tussenchecks zijn drempelgevoelig**: het groen
   van PAY en WTN kantelt bij +20% drempelverschuiving — consistent met het
   feit dat beide voorspellingen (KOOP) te optimistisch bleken.
7. **Marktschokken domineren de lijst nú niet**: slechts 1 van de top 20
   dankt zijn korting aan een recente koersval; de scheefheid is chronisch
   (model versus markt), niet acuut.

---

## Deelvraag 1a — Is de data betrouwbaar genoeg?

### Goudstandaard: database (Yahoo) vs geverifieerde rapportcijfers — staat grotendeels goed

De 19 volledige analyses bevatten jaarcijfers uit jaarverslagen (HOOG-gelabeld,
met bronnen-inventaris). Vergeleken met de database (16 tickers vergelijkbaar;
PAY/HAFNI/PRX uitgesloten wegens valutaconversie, AMB/NVDA niet parseerbaar;
schaal per rapport automatisch gedetecteerd) — `sim2`:

| Veld | n | mediaan \|afwijking\| | ≤2% | ≤10% |
|---|---|---|---|---|
| Omzet | 53 | 0,0% | 92% | 92% |
| Nettowinst | 41 | 0,0% | 88% | 95% |
| Operationele kasstroom | 31 | 0,0% | **100%** | 100% |
| Eigen vermogen | 35 | 0,0% | 89% | 97% |
| FCF | 42 | 0,4% | **64%** | 83% |

De uitschieters zijn verklaarbaar: Adyen's omzetdefinitie (Yahoo wisselde in
2022 van basis, +585%), één echte afwijking (SDIP-B nettowinst 2025: rapport
−28 vs DB −69) en FCF-definitieverschillen inclusief een tekenwissel (BC8
2022: rapport −30, DB +35). **Conclusie: de kerncijfers zijn betrouwbaar; FCF
is het zwakke veld** — relevant omdat FCF via EV/FCF én de perpetuity direct
in de fair value zit.

### De EV-consistentiecheck meet zichzelf — A1

`data_quality.py:255` leest `total_cash`, een veld dat nergens in de jaarrij
wordt gevuld; de check rekent dus structureel met kas = 0. Herberekend met
`net_cash` (het veld dat wél bestaat) op de steekproef — `sim3`:

- 36 meldingen met de kapotte formule, waarvan **11 (31%) vals alarm**;
- daaronder exact de meldingen die de tussenchecks als aandachtspunt citeerden:
  SDG.PA 2,26× → **1,02×**, INF.PA 1,59× → **1,03×**, en portfolio-aandeel
  ECONB.BR 2,08× → 1,02×;
- universum: 313 tickers dragen de melding; geëxtrapoleerd is ruwweg
  een derde daarvan onterecht.

**A1 (klein, hoog effect):** vervang `total_cash` door `net_cash` in de
EV-check. Twee tussencheck-documenten (SDG, INF) citeren het valse alarm als
"uitzoekpunt" — dat mag bij een volgende update geschrapt.

### fy_lag en dekking

- fy_lag-verdeling: 0 = 2.666, 1 = 72, ≥2 = **21**. De hardste gate
  (achterstand ≥2 → bad) raakt 0,8% van het universum — **staat goed**.
- data_status: ok 1.949, warning 397, bad 292, missing 121 → 15% geblokkeerd.
  Redenen bij INSUFFICIENT DATA: geen_fv 342, databug 177, geen_data 152,
  verouderd 37.
- **Dekkingsgat, onverwacht**: drie bedrijven waar volledige analyses van
  bestaan — **SHEL, NBIS en SBC — staan helemaal niet in het universum**.
  Van de 11 portfolio-aandelen ontbreken er drie (CRWD, DIS, PUIG); van de
  externe kwaliteitsijkset ontbreken de Britten (Games Workshop, Halma). → A12
- `/api/stock` levert de jaarrijen **zonder overrides** (`app.py:2627`,
  `get_financials` i.p.v. `jaarrijen_met_overrides`) — de vijfde plek die de
  CLAUDE.md-valkuil mist. Het moat-profiel op dat endpoint (en dus in de
  tussenchecks, die het gebruiken) negeert handmatig ingevoerde boekjaren
  zoals LASTIK's FY2025. → A9

---

## Deelvraag 1b — Laten de regels de juiste bedrijven bovenkomen?

### De koopkant is te streng, en dat zit niet in de drempels

Vol universum (2.051 beoordeelbare tickers; lokale signaalreplicatie exact:
0 afwijkingen van productie) — `sim1`:

- **Koopsignalen: 35 van 2.051 = 1,7%** (4 STRONG BUY, 31 BUY) tegen de eigen
  streefband van 5–15% uit `calibrate_report.py`. SELL: 28,4% (binnen norm).
- Kanteltest ±20%: de kóópdrempels verschuiven verandert weinig (5–30
  tickers); de sell-drempel is de gevoeligste knop (178–240 wissels).
  Grensbewoners (±5% van een drempel): 59 / 111 / 170 / 139 voor
  strong_buy / buy / hold_upper / sell.
- Waardering-sweep (`sim5`, steekproef n=292): **geen enkele configknop
  beweegt de fair value wezenlijk** — sector-multiples ±20% geeft mediaan
  ±5,6% FV; groeiaannames +2pp geeft +8,7%; required_return −1pp +4,1%. Het
  aantal koopsignalen in de steekproef beweegt van 24 naar hooguit 27.
  De Graham-groeicap (8→10) doet exact niets: een **dode knop**.

De oorzaak is structureel: de multiples-methode ankert 0,65 op de **eigen
historische mediaan-multiple** (hardcoded, `valuation.py:172`). Een bedrijf
dat structureel duurder is geworden dan zijn eigen verleden — precies wat
kwaliteits-compounders doen — krijgt per constructie een fair value ver onder
de koers.

### Extern geijkt: de vals-negatief-bron is reëel

- Referentieset (25 bekende namen): **10 op SELL**, met "kortingen" van −36%
  tot −173% — ASML −173%, ATCO −96%, XTB −98% (kwaliteit 9!), AIR −81%,
  SAP −42%. Dat schendt het eigen kalibratiedoel (referentie-afwijking ≤60%).
- Externe kwaliteits-small/midcaps: Lifco −82%, Addtech −91%, Lagercrantz
  −91%, Kitron −93%, Rational −49% — allemaal SELL. Alleen Revenio (BUY, #85)
  overleeft.
- Banken en verzekeraars scoren kwaliteit 1–4,5 over de hele linie (KBC 1,
  DNB 1, PKO 1, PZU 4,5): D/E- en rentedekking-regels zijn betekenisloos voor
  financials, dus de sector is de facto van BUY uitgesloten. → A4
- rank-wegingsweep: korting-puur geeft maar 2/20 overlap met de huidige top
  20 — de weging doet er dus echt toe. **36 bedrijven met kwaliteit ≥9 staan
  buiten de top 50** (WKL #80, Revenio #85 mét BUY-signaal, Dassault #207);
  de KALMAR/EVO-klasse is geen incident maar een categorie. → A3
- Portfolio-ijk: de screener vindt een deel van Janco's eigen namen wél hoog
  (BETS #8, KPL #25 BUY, WAWI #109 BUY) — het model is niet blind voor alles
  wat hij goed vindt, het is blind voor één specifieke soort: duurzaam
  hooggewaardeerde kwaliteit.

**A2 (beleggingsbeslissing):** heroverweeg de kwaliteitspóórt in BUY/STRONG
BUY (zie 1c). **A3 (aanbeveling):** maak de kwaliteit-zonder-korting-klasse
zichtbaar — een aparte lijst of tabblad ("Kwaliteit", kwaliteit ≥9 ongeacht
korting), in lijn met het huisprincipe *markeren, niet wegfilteren*. Dat
repareert de vals-negatief-blindheid zonder één drempel te verzetten.

---

## Deelvraag 1c — Hoort het kwaliteitsoordeel in deze fase?

Als **informatie**: ja. Als **poort**: de empirie zegt nee.

- Correlatie screener-kwaliteit vs scorekaart van de 17 koppelbare analyses:
  **r = 0,28** — zwak (`sim6`).
- De enige twee KOOP-uitkomsten (PRX 34/45, ADYEN 33/45) hadden screener-
  kwaliteit **5,0** — onder de BUY-poort van 7. **De kwaliteitspoort had
  beide koopkansen uit de koopsignalen gehouden.** Omgekeerd scoorde BC8
  (kwaliteit 8,5) maar 24/45.
- Reproduceerbaarheid: 57% exact; bij 9% verschuift de score >1 punt zodra de
  TTM-rij en overrides meedoen (BETS-B: 10,0 zonder vs 6,5 mét). De
  TTM-als-jaar-0-constructie heeft dus groot effect op de score. → A11
- Jackknife (n=284): 86% ongewijzigd bij één jaar minder data, maar **5%
  kruist daardoor een signaalpoort** (uitschieter JET2.L: 6,0 → 9,0).
- Het Infotel-patroon (kwaliteit 10/10 naast rode moat) klopt conceptueel:
  de score meet gezondheid, niet bescherming — dat is geen fout maar moet je
  weten bij het lezen.

**A2, uitgewerkt:** houd de kwaliteitsscore als weging in rank_score en als
kolom, maar laat BUY/STRONG BUY niet langer hard op ≥7/≥8 vereisen — of
verlaag de poort en toon kwaliteit als badge. Dit is een beleggingsbeslissing:
het maakt de lijst ruiger maar aantoonbaar minder blind (beide echte KOOPs
kwamen uit het lage-kwaliteit-segment van de screener).

---

## Deelvraag 2 — De tussencheck

### Drempelgevoeligheid van het moat-oordeel — `sim4`, subset n=101

Basisverdeling: 25 groen / 36 geel / 35 rood / 5 grijs. Kanteltest ±20%:

- Meeste drempels: 1–12 wissels per variant — matig gevoelig, geen reden tot
  paniek.
- Maar de wissels raken precies de spannende gevallen: **PAY en WTN verliezen
  hun groen** bij ROIC_GROEN→18 of STABIEL→0,84, en worden zelfs rood bij
  STABIEL_ROOD→0,78. Beide volledige analyses kwamen op PASS/HOLD — het
  groen waarop de VERDIEPEN-oordelen leunden was dus inderdaad randgeval.
  Consistent met de kalibratie (0 van 2 voorspellingen raak, beide te
  optimistisch).
- INF's rood kantelt pas bij erosiedrempel −6 (nu −3): dat oordeel is robuust.
- ROIC-definitie harmoniseren (kasaftrek zoals quality) wijzigt maar 4/101
  niveaus — maar één daarvan is SDG: **geel → groen**. Twee definities naast
  elkaar blijft onwenselijk. → A8

### De cyclusregel was bij vier checks feitelijk onjuist

Met de inmiddels aangevulde koershistorie opnieuw gedraaid:

| Ticker | Stond in de check | Werkelijkheid (5 jaar) |
|---|---|---|
| WTN.WA | "staat weer op recordhoogte" | **66% onder de top**, diepste val −71% |
| INF.PA | "recordhoogte, −2%" | −41% val in maart 2026, 28% onder top |
| SDG.PA | "−1%, recordhoogte" | diepste val −44%, 28% onder top |
| HEM.ST | −84% (klopte) | −84% (bevestigd) |

De backfill-prioritering van 6 augustus voorkomt dit voortaan; voor de
bestaande checks is een addendum op zijn plaats — bij Wittchen verandert het
de context wezenlijk (cyclisch dieptepunt in plaats van topvorming). → A7

### OVERSLAAN-regel 1 leunt op de zwakste schakel

Regel 1 (koers boven FV → OVERSLAAN) gebruikt de screener-FV. Formele
betrouwbaarheidssignalen vingen 2 van de 9 gevallen (WTN: optimistisch ónder
voorzichtig — de spread-berekening negeert de scenario-inversie; HEM: low
confidence). Daarnaast bleken twee "aandachtspunten" uit de checks zelf
data-artefacten: SDG/INF's EV-melding was de kapotte check (A1), en PAY's
korting was een valutakwestie die wél echt was.

**A5:** laat regel 1 alleen automatisch gelden als fv_confidence high/medium
is én conservative < optimistic; anders degradeert het naar TWIJFEL met de
verplichting de waardering zelf op te bouwen. **A6:** OVERSLAAN blijft
per constructie ongetoetst (3 van 3 zonder analyse) — draai elke ~5e
OVERSLAAN alsnog volledig, bij voorkeur een grensgeval uit de kanteltest
(kandidaat: SFG, wiens rood bij −20% op STABIEL_ROOD kantelt).

---

## Deelvraag 3 — De volledige analyse (snelle check)

De two-stage-opzet met brondiscipline, DCF-invoerblok en validators is het
sterkste deel van de keten; niets gevonden dat richting fantasiecijfers wijst.
Twee kanttekeningen:

- De scorekaartdrempels (KOOP ≥33 én DCF ≥3; PASS <24 of DCF=1) zijn nergens
  op gekalibreerd aangetroffen; de verdeling (3 KOOP, 10 HOLD, 6 PASS, n=19)
  heeft een HOLD-zwaartepunt dat op conservatieve rubrics kán wijzen, maar
  n=19 is te klein voor een oordeel. Herzie dit bij n≈30.
- Per-framework-scores zijn uit de markdown niet betrouwbaar te oogsten
  (2 van 19 parseerbaar). Ze staan wél machine-leesbaar in de platform-JSON's
  van aandelenanalyse — gebruik die als het scorebord ooit per framework wil
  kalibreren. Geen actie nu.

---

## Deelvraag 4 — Marktschokken

Feitelijke tijdlijn (code-verkenning 6 aug):

| Gebeurtenis | Zichtbaar na |
|---|---|
| Koersval −20% → dashboard (MOS, signaal, rangorde) | zelfde avond (~18:45) |
| FV waartegen die koers wordt gehouden | 0–11 dagen oud (gem. 5,5) |
| Winstwaarschuwing in de winstbasis (TTM) | 4–12 weken, en alleen als de TTM-rij meetelt |
| Structureel in de FV (jaarverslag) | tot ~15 maanden |
| Koersval >40% | zelfde avond volledige herberekening (split-guard) |
| Koersval >90% | signaal wordt "DATABUG" — een echte crash oogt als datafout |

De gevreesde vervorming — vers gevallen aandelen die met verouderde FV de
Kansen-lijst inklimmen — is **nu geen dominant patroon**: 1 van de top 20
(WAVE.PA, −24% in 3 maanden) past erin; WLN.PA (90% "korting") steeg juist
+34%. De extreme kortingen bovenin zijn chronische model-afwijkingen (zie 1b),
geen verse ongelukken.

Oordeel over de asymmetrie: **kansdetectie snel, oordeelvorming traag past
bij een fundamentals-aanpak** — mits zichtbaar. Aanbevelingen klein houden:
**A10** werk bij de dagelijkse koersronde de afgeleide EV bij (of demp de
EV-melding na een koersbeweging), zodat een koersval geen valse
datakwaliteits-melding veroorzaakt; **A13** geef een koersval >90% een eigen
label in plaats van DATABUG. De suspend-keten is momenteel leeg (0 tickers)
— het permanente-vijver-risico van `presumed_delisted` is vandaag theorie,
maar verdient een zichtbaar lijstje in Beheer. (→ A12-buur, laag.)

---

## Deelvraag 5 — Groeibedrijven (korte mening, geparkeerd)

Het missen van verlieslatende groeiers is by design en terecht: de hele
waarderingsketen eist winst of positieve kasstroom, en een FV op een
verlieslatend bedrijf uit dit model zou schijnzekerheid zijn. De 🌱-markering
(CAGR ≥15%) is de juiste vorm: zichtbaar houden zonder oordeel. Wil je ooit
groeibriljantjes vangen, bouw dan een **aparte** screen (omzetgroei,
brutomarge-richting, verwatering, kaspositie) met een eigen lijst — meng hem
niet in deze FV-pijplijn. Niet doen vóór de kern gekalibreerd is; de
vals-negatief-bron van 1b is groter en goedkoper te dichten.

---

## Aanbevelingen, genummerd

Status bijgewerkt op 6 augustus 2026.

| # | Wat | Soort | Prioriteit | Status |
|---|---|---|---|---|
| A1 | EV-check: `total_cash` → `net_cash` (±100 valse meldingen weg) | bugfix | **hoog** | ✅ gedaan |
| A2 | Kwaliteitspóórt uit BUY/STRONG BUY (of verlagen); kwaliteit als weging/badge houden | beleggingsbeslissing | **hoog** | ⏸ **wacht op Janco** — dit verandert wat een koopsignaal betekent en is geen bouwkeuze |
| A3 | Aparte kwaliteitslijst (kwaliteit ≥9, ongeacht korting) naast Kansen | feature | **hoog** | ✅ gedaan (tabblad 💎 Kwaliteit) |
| A4 | Financials: kwaliteitsscore "n.v.t." i.p.v. 1–4 op regels die daar niets betekenen | model | middel | ✅ gedaan |
| A5 | Tussencheck-regel 1 alleen bij betrouwbare FV (confidence + cons<opt) | methodiek | **hoog** | ✅ gedaan — uitgebreid tot een vier-ankertoets in de tussencheck-skill |
| A6 | Controlegroep: elke ~5e OVERSLAAN alsnog analyseren (eerst SFG) | methodiek | middel | ⏸ **wacht op een analyseronde** — de regel staat in de skill, de eerste moet nog draaien |
| A7 | Addendum bij WTN/INF/SDG-tussenchecks: cyclusregel was onjuist | inhoud | middel | ✅ gedaan |
| A8 | Eén ROIC-definitie (kasaftrek) voor quality én moat | model | laag | ✅ gedaan (besluit Janco 2026-08-08) — moat rekent nu ook met kasaftrek; SDG kan geel → groen schuiven |
| A9 | `/api/stock` op `jaarrijen_met_overrides` (vijfde gemiste plek) | bugfix | middel | ✅ gedaan |
| A10 | Koersronde: afgeleide EV bijwerken of EV-melding dempen na koersbeweging | bugfix | laag | ✅ gedaan — EV beweegt mee met de koers (`EV_nieuw = EV_oud + aandelen × koersverschil`), níét opnieuw afgeleid uit mcap + nettoschuld, want dan toetst de check zijn eigen uitkomst |
| A11 | Dode knoppen opruimen + TTM-als-jaar-0 documenteren | opschoning | laag | ✅ gedaan — pyflakes volledig schoon; de Graham-groeicap is blijven staan als vangnet tegen een configtypo, met die reden erbij |
| A12 | Universum: SHEL/NBIS/SBC + CRWD/DIS/PUIG toevoegen; VK-dekking overwegen | data | middel | ✅ zes toegevoegd; **VK-dekking is een aparte keuze** (Games Workshop, Halma) en staat open |
| A13 | Koersval >90%: eigen label i.p.v. DATABUG | ux | laag | ✅ gedaan (FACTOR >10) |

### Buiten de lijst gevonden en opgelost

- **Sectorprofielen matchten niet op de Yahoo-namen.** `config.yaml` gebruikte
  de GICS-namen; 617 aandelen (22%) vielen stil terug op `Default`, en bij
  Basic Materials was dat een 20% royaler anker — schijnkortingen dus.
  Gerepareerd, plus een profiel voor Communication Services (186 tickers, stond
  ook op Default). Bewaakt door `tests/test_sector_config.py`.
- **`POST /api/scores/recompute`** — een configwijziging landde tot nu toe pas
  in de database als de nachtelijke ronde langskwam: ruim een maand, waarin het
  dashboard oude en nieuwe aannames door elkaar toonde.

### Goedkope documentatie-fixes — ✅ alle vijf gedaan

- ~~`CLAUDE.md` cron-blok beschrijft een geschrapte GitHub-Actions-schedule en
  een niet-bestaande scheduler-gate.~~
- ~~Suspend-drempels: docs zeggen 10/30/90, code doet 3/21/45.~~
- ~~"Negen dagen" FV-cadans is elf.~~
- ~~`docs/ARCHITECTURE.md` rondegroottes: 100/20 moet 250/40 zijn.~~
- ~~`valuation.py` docstrings: `max()` waar de code blendt, Graham zonder
  yield-scaler.~~

## Verantwoording en beperkingen

- Signaalreplicatie exact (0 afwijkingen op 2.051); kanteltellingen
  steekproefsgewijs met de hand nagerekend (PAY-standvastigheid 0,70 tegen
  drempel 0,78 → rood ✓; INF-erosie −5,3 tegen −6 → geel ✓).
- Kwaliteit en FV zijn buiten de server alleen bij benadering te reproduceren
  (57% resp. 43% exact) doordat `/api/stock` TTM-rij en overrides niet
  meelevert; alle sweeps vergelijken daarom variant tegen eigen baseline,
  nooit tegen productie.
- De kalibratie op echte uitkomsten rust op n=2 (beide voorspellingen te
  optimistisch) — richting, geen bewijs. Het scorebord
  (`scripts/scorebord.py`) meet dit voortaan automatisch mee.

# Tussencheck: PLX.PA — Pluxee N.V.

**Oordeel: TWIJFEL**
Datum: 2026-10-25 · Koers 14,92 EUR · Beurswaarde 2.172 mln

## Waarom

Pluxee heeft wat een moat lijkt: het is wereldwijd de nummer twee in maaltijd- en
personeelsvoordelen, met Edenred als enige echte rivaal op dezelfde schaal, en het
verdient op twee manieren aan elke voucher: een vergoeding van werkgever en
winkelier, en rente over het geld dat werkgevers vooruitbetalen en dat pas later
wordt uitgegeven (de *float*). Het rendement op kapitaal ligt al vier jaar tussen 16%
en 27%. Dat profiel hoort bij een sterke positie.

Maar die positie rust op de belastingwet. Maaltijdcheques bestaan omdat overheden ze
fiscaal vrijstellen, en precies daar zit het risico: in Brazilië legde een decreet in
november 2025 de vergoeding voor winkeliers aan banden (maximaal 3,6%) en halveerde
het de termijn waarbinnen winkeliers betaald worden van 30 naar 15 dagen — wat ook de
float kleiner maakt. Pluxee verlaagde daarop zijn doelen. In Frankrijk werkt het
parlement aan een hervorming van de titres-restaurant; op 8 oktober 2026 viel het
plafond op winkelierstarieven er voorlopig uit, maar de koers daalde toch. Volgens
een analyse hangt ~85% van Pluxee aan zulke fiscale regelingen, tegen ~60% bij
Edenred. Of dit een moat is of een vergunning die de overheid stukje bij beetje
terugneemt, valt in een tussencheck niet te beslissen.

Daar komt bij dat er geen bruikbaar prijsanker is. De modelwaarde (24,44) heeft een
spreiding van 130% en laag vertrouwen, de eigen historie bestaat uit één bruikbaar
jaar, het analistendoel ligt vrijwel op de koers, en de EV/EBITDA van 2,0 is een
artefact van Yahoo (zie hieronder). De groei uit de screener (+15% per jaar
2022–2025) is bovendien verleden tijd: dit boekjaar mikt Pluxee op stabiele omzet en
kromp het derde kwartaal 3,3%.

## Wat de cijfers zeggen

| | |
|---|---|
| Rendement op kapitaal | mediaan 19,3%, laagste jaar 16,4% (2024); 2022 26,8%, 2025 18,0% |
| Brutomarge | ontbreekt — Yahoo levert geen brutowinst (geen kostprijs van de omzet). De erosietoets en groeitoets T4 kijken dus nergens naar |
| Operationele marge | 28,1% → 27,1% → 23,3% → 28,4% (2022–2025) |
| Zwaarste koersval | −68% sinds de notering in februari 2024 (dal 26 maart 2026), nu 52% onder de top |
| Korting op fair value | 38,9% (vertrouwen: laag, spreiding 130%; conservatief 21,05, optimistisch 29,22 EUR; Graham 12,31, perpetuity 41,85) |
| Groeiprofiel | groen (score 10), maar op 2022–2025: +25%, +15%, +6% — afnemend |
| Kwaliteit / F-score | 6 / 6 |
| Datakwaliteit | ok, 96,9% compleet; boekjaar eindigt 31 augustus, laatste verwerkte jaar FY2025 |

**Vier prijsankers.** Ze wijzen niet dezelfde kant op:

| Anker | Waarde |
|---|---|
| Modelwaarde screener | 24,44 EUR — vertrouwen laag, spreiding 130% |
| Eigen historische K/W | alleen FY2024 bekend (20,2). × genormaliseerde winst per aandeel 0,85 ≈ 17,1 EUR; × FY2025-winst per aandeel 1,35 (197 mln / 145,6 mln aandelen) ≈ 27,3 EUR. Eén jaar is geen mediaan |
| Eigen historische EV/EBITDA | 6,8 (FY2025) en 10,1 (FY2024) tegen 2,0 nu — maar die 2,0 is onbruikbaar (zie datafouten) |
| Analistenkoersdoel | 15,88 EUR, 14 analisten, consensus *hold* — 6% boven de koers |

Het analistenanker zegt dat de markt het regulatoire risico al ongeveer tegen deze
koers prijst. Het model ziet 64% opwaarts. Dat verschil is precies wat de volledige
analyse moet verklaren.

**Datafouten en kanttekeningen in de screener:**
- **EV is fout.** Yahoo geeft een bedrijfswaarde van 929 mln, terwijl beurswaarde
  plus nettoschuld ≈ 1.836 mln geeft (factor 1,97, ook gemeld in `issues`).
  Waarschijnlijk trekt Yahoo de kas van de float af, maar dat geld is van klanten
  (vooruitbetaalde, nog niet uitgegeven tegoeden) en geen overtollige kas. Gevolg:
  EV/EBITDA 2,0 en de "−76% onder historie" in `hist_relative` betekenen niets.
  Dezelfde vraag geldt voor `net_cash` in de waardering en voor de "overtollige kas"
  in de ROIC-formule: wordt klantgeld als eigen geld geteld, dan liggen fair value en
  ROIC te hoog. Niet nagegaan hoe groot het effect is.
- **Historie vóór de afsplitsing.** Pluxee is in februari 2024 van Sodexo
  afgesplitst. FY2022 en FY2023 zijn uitgesplitste (carve-out) cijfers; het eigen
  vermogen springt van 704 mln (2022) naar 47 mln (2023) en terug naar 470 mln
  (2025) door de herstructurering rond de splitsing. De K/W-historie bestaat
  daardoor uit één jaar, en de omzetgroei van 2022–2023 (+25%) valt samen met de
  rentestijging, die de float-opbrengst opdreef.
- **K/W loopt uiteen.** `pe_ttm` 16,6 tegen 10,6 in `hist_relative` en 11,0 op de
  FY2025-winst. Welke winstmaat klopt is niet uitgezocht.

## Wat het onderzoek toevoegde

- Q3 FY2026 (tot mei): omzet €312 mln, organisch −3,3%, door Brazilië en de economie
  in Europa; negen maanden €967 mln, +2,7%. Float-opbrengst €124 mln over negen
  maanden (+4,5%), gemiddeld rendement op de belegde float 6,1% in H1. Doelen
  bevestigd: stabiele omzet, licht hogere EBITDA-marge. H1: EBITDA-marge 37,0%,
  aangepaste winst per aandeel €0,78 (+6,8%).
  ([Pluxee Q3](https://www.pluxeegroup.com/q3-fiscal-2026-revenues/),
  [Pluxee H1](https://pluxeegroup.com/first-half-fiscal-2026-results/))
- Brazilië: decreet november 2025, plafond van 3,6% op winkelierstarieven en
  uitbetaling binnen 15 in plaats van 30 dagen. Pluxee en Edenred daalden die dag
  ruim 9%; Pluxee rekent met het ongunstigste scenario. Rechtszaken lopen nog.
  ([Tradingsat](https://www.tradingsat.com/edenred-FR0010908533/actualites/edenred-edenred-le-president-bresilien-signe-un-decret-defavorable-aux-titres-restaurant-qui-plombera-les-resultats-d-edenred-et-pluxee-leurs-actions-chutent-de-plus-de-9-1150423.html),
  [Option Finance](https://www.optionfinance.fr/info-financiere-en-continu/d/2025-11-17-pluxee-revise-a-la-baisse-certains-objectifs-2026-a-cause-de-la-reglementation-au-bresil.html))
- Frankrijk: op 8 oktober 2026 nam een parlementaire commissie een wetsvoorstel aan
  zonder plafond op winkelierstarieven, maar met het einde van papieren cheques en
  van kortingen aan werkgevers aan het eind van het jaar. Beide aandelen daalden
  toch. ([Option Finance](https://www.optionfinance.fr/info-financiere-en-continu/d/2026-10-08-pluxee-et-edenred-restent-sous-pression-malgre-une-reforme-moins-severe-que-prevu-en-france.html))
- Concurrentie: nummer twee wereldwijd achter Edenred; Swile won terrein met een
  betere app en snellere uitbetaling aan winkeliers. Pluxee hangt meer aan fiscale
  regelingen dan Edenred (~85% tegen ~60%, volgens een substack, niet geverifieerd).
  ([Heavy Moat Investments](https://heavymoatinvestments.substack.com/p/edenred-how-does-it-compare-to-pluxee),
  [Sacra over Swile](https://sacra.com/chat/h/243f8504-3f9d-45fd-a630-757b5038c15d/),
  [Robin Research](https://robinresearch.substack.com/p/pluxee-nv-frplx))

Niet gevonden: de jaarcijfers FY2026 (boekjaar tot 31 augustus 2026), recente
marktaandeelcijfers per land, en of Pluxee prijsverhogingen kan doorvoeren.

## Als je dit uitdiept, zoek dan uit

1. **Hoeveel van de winst hangt aan regels die de overheid kan wijzigen?** Splits
   omzet en EBITDA naar land en product (Frankrijk, Brazilië, rest), en reken het
   Braziliaanse decreet en de Franse hervorming volledig door. Dit is de kern: is de
   moat van Pluxee of van de wetgever?
2. **Wat is de float waard, en wie is de eigenaar?** Hoe groot is de belegde float,
   hoeveel daarvan is wettelijk afgezonderd, en wat doet een rentedaling of een
   kortere uitbetalingstermijn met de float-opbrengst? Bouw de waardering op met
   klantgeld als verplichting, niet als kas — de screener doet dat waarschijnlijk niet.
3. **Bouw de waardering zelf op.** Er is geen bruikbaar prijsanker: één jaar
   K/W-historie, een kapotte EV, en een modelspreiding van 130%. Vergelijk met
   Edenred als enige echte sectorgenoot.
4. **Wint of verliest Pluxee terrein?** Tegen Edenred en tegen digitale nieuwkomers
   (Swile, Up). De afschaffing van papier in Frankrijk kan de drempel voor
   nieuwkomers verlagen.

## Voorspelling

**HOLD.** Het rendement op kapitaal en de duopoliepositie zullen op moat goed
scoren, maar management en "Buffett" waarschijnlijk minder: een bedrijf dat al
twee jaar zijn doelen bijstelt door regelgeving, met groei die stilvalt, en een
waardering die vooral goedkoop oogt door een fout in de bedrijfswaarde. Bewust de
voorzichtige kant, in lijn met de bekende neiging van deze checks om te optimistisch
te zijn (PAY, WTN). Een KOOP is denkbaar als de volledige analyse aantoont dat het
regulatoire risico na Brazilië en Frankrijk grotendeels in de koers zit.

---
Gebaseerd op opgeslagen screenerdata (Yahoo, `AGGREGATOR`-kwaliteit) plus
beperkt webonderzoek. **Geen invoer voor `research/PLX.md`** — die eist
`HOOG`-kwaliteit bronnen voor de recente vijf jaar.

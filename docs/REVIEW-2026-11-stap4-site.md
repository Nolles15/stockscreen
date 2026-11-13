# Kritische review november 2026 — stap 4: de site

Datum: 2026-11-13. De live site doorgelopen in Chrome op 1512 pixels breed, plus
een smalle weergave van 390 pixels in een ingebed venster. Er is nergens op een
knop gedrukt die iets wijzigt of een ronde start.

Bekeken: dashboard (Kansen, Kwaliteit, Groeiers, Ik bezit), aandeelpagina
(EQS.PA), `/start`, `/analyses`, `/leren`, `/scorebord`, `/settings`, `/triage`,
`/methode`. Niet bekeken: Herstellers, Mijn lijst, Alles, Geen oordeel,
`/tussenchecks`, een analyse-detailpagina, `/log`, en de uitkomst van `/start`
na het invullen van een ticker. De console gaf op geen enkele pagina een fout
van de site zelf.

## Algemene indruk

De site laadt snel, doet wat hij belooft en de teksten boven elk tabblad zijn
eerlijk over wat het model niet kan. Het is een werkinstrument voor één persoon,
en zo voelt het ook: veel informatie, weinig hiërarchie, en beheerknoppen die
voor iedereen zichtbaar zijn.

## W1. Fouten in wat de pagina toont

- **`/leren` mengt pence en ponden.** FRP Advisory: "toen 128.50 · nu 1.22". De
  koers uit de tussencheck staat in pence, de huidige in ponden. Het lijkt een
  val van 99%.
- **`/leren` vraagt twee keer om een besluit over hetzelfde bedrijf.** James
  Halstead staat er als JHD.L (toen 1.31 · nu 1.28) en als JHDA.L (toen 1.31 ·
  nu 0.68). De tweede is via de bedrijfsnaam gekoppeld en vergelijkt de koers van
  de ene notering met die van de andere, wat `oordelen.py` juist niet zou doen.
- **Het scorebord toont een intern document als analyse.** De eerste rij is
  `_VOORSTEL_BETA_EN_SCOREKAART_2026-10-13`, zonder cijfers. Het telt ook mee:
  de pagina zegt "45 analyses" in de kop en "46 bedrijven" in de tekst.
- **Een boekjaar met lege cijfers heet "compleet".** Bij Equasens is FY2024
  bijna leeg (geen omzet, EBIT of nettowinst), en blok 1 zegt "De cijfers zijn
  compleet, 88%". Gevolgen op dezelfde pagina:
  - de F-score kan maar 4 van de 9 toetsen bepalen en geeft 0 van 2 punten;
  - de groeistabiliteit toont "+11%, +3%, +8%", waarbij de laatste stap over
    twee jaar loopt (2023 → 2025) en als één jaar wordt geteld;
  - de moat- en groeitabellen slaan 2024 over zonder dat te melden.
- **De koersgrafiek heeft geen lineaire tijdas.** De jaren 2022, 2023 en 2024
  staan dicht op elkaar, 2025 en 2026 nemen bijna driekwart van de breedte. De
  punten zijn op volgnummer uitgezet, en oude jaren zijn tot weekkoersen
  uitgedund. De vorm van de koers over vijf jaar is daardoor vertekend.
- **Kansen: de kop klopt niet.** "Gesorteerd op grootste korting" staat boven
  elk tabblad, ook waar op rangscore, kwaliteit of groei is gesorteerd.
- **Analysekaarten zonder fair value.** Nimbus en tk accelis tonen een streepje;
  dit is de rode test uit stap 3.

## W2. Het model in beeld: waar de pagina te stellig is

- **Arctic Paper op de bezitspagina.** "Verkopen boven kr 107,95, nog 763% te
  gaan", bij een koers van 12,51, een omzet die 13% per jaar krimpt en een
  negatieve vrije kasstroom. De grens is 175% van een fair value die uit de
  topjaren 2021–2022 komt. Dit is de krimpbias uit stap 1 in zijn zuiverste vorm,
  gepresenteerd als verkoopgrens. Het aandeel staat bovendien twee keer als bezit
  (ARP.ST en ATC.WA).
- **"Hoog rendement dat standhoudt" bij een dalende reeks.** Equasens: rendement
  op kapitaal 23,8% → 15,7% → 14,8% → 12,1%. Het moat-oordeel is groen omdat het
  laagste jaar boven 70% van de mediaan ligt; de richting telt niet mee.
- **Drie percentages voor hetzelfde gat.** Op de aandeelpagina staat "49% onder
  de geschatte waarde", "verschil +96,3%" en "upside +88%" (dat laatste tegen het
  koersdoel van analisten). Alle drie kloppen, maar wie snel leest ziet drie
  verschillende uitkomsten.

## W3. Leesbaarheid en vormgeving

- **De blokken van de beslisboom zijn slecht leesbaar.** Donkergroene vlakken
  met grijze tekst erop; de toelichting ("88% van de benodigde velden is gevuld",
  de bandbreedte, het vertrouwen) valt bijna weg. Het zijn de enige donkere
  vlakken op een verder lichte pagina, net als de twee banners bovenaan.
- **Getalnotatie door elkaar.** "54.49" en "54,49", "15.3%" en "69,3%" op
  dezelfde pagina; bedragen soms met valutateken, soms zonder. Op de bezitspagina
  staat "vrije kasstroom 7110.00 → 8293.00" zonder eenheid.
- **Geen valuta op het dashboard.** Prijs en waarde staan als kale getallen:
  0.54 (pond), 4.75 (zloty), 865.60 (kroon).
- **Nederlands en Engels door elkaar.** BUY, HOLD, HIGH, Base, Analyst consensus
  naast Kwaliteit, Geschatte waarde, Oordeel. Het signaal heet BUY, het oordeel
  uit de analyse KOOP.
- **Twee tabbladen met hetzelfde pictogram** (Herstellers en Beheer), en drie
  verschillende paginatitels ("Wannabe Oracle Stockscreener", "Stockscreener",
  "Aandelentools").
- **De aandeelpagina is erg lang.** Hetzelfde staat er tot drie keer (blokken,
  tabellen, kaarten); het signaal alleen al drie keer. De invoerformulieren voor
  handmatige cijfers staan midden in de leesstroom.

## W4. Beheer staat tussen de inhoud

Dit is de zichtbare kant van punt C1 uit stap 3.

- Op elke pagina staat rechtsboven een knop **Refresh**; op elke aandeelpagina
  nog een, plus "Koers handmatig invoeren" en een volledig invoerformulier voor
  jaarcijfers.
- `/triage` toont per rij een prullenbak, en bovenaan "Ronde starten".
- `/settings` heeft 107 invoervelden en een knop Opslaan die niets bewaart (C2).
  Het veld "Verkoop kwal. drempel" hoort bij een regel die in augustus is
  geschrapt.
- De bezitspagina toont aan iedere bezoeker welke elf aandelen je hebt. Dat is
  een bewuste keuze volgens CLAUDE.md; ik noem het hier omdat het op de pagina
  zelf nergens staat.

## W5. Smalle weergave

Op 390 pixels werkt het dashboard zonder horizontaal schuiven. De aandeelpagina
schuift een paar pixels opzij doordat de jaartabel 453 pixels breed is, en het
menu moet horizontaal worden geschoven. Bruikbaar, niet verzorgd. Getest in een
ingebed venster, niet op een telefoon.

## Wat goed is

- De toelichting boven elk tabblad zegt wat de lijst wel en niet betekent, in
  gewone taal. Boven Kansen staat zelfs de waarschuwing dat het model op de eigen
  historische waardering rekent.
- De telling "15 van deze 20 zijn al onderzocht" koppelt de screener aan je eigen
  oordelen; dat is precies de goede terugkoppeling.
- De versheidsbanner en de dekkingsregel ("2950 beoordeeld · 900 buiten het
  model") staan op de plek waar je ze nodig hebt.
- `/methode` rekent elke stap door voor een gekozen aandeel.
- `/leren` stelt de juiste vraag: deed je wat je zelf concludeerde.

## Aanbevelingen, op volgorde

| # | Wat | Omvang |
|---|---|---|
| 1 | Pence/pond op `/leren` en de dubbele James Halstead repareren | klein |
| 2 | Het voorstel-document uit analyses en scorebord halen | klein |
| 3 | Tijdas van de koersgrafiek op datum zetten | klein |
| 4 | Een boekjaar zonder omzet of winst niet als compleet tellen; ontbrekende jaren zichtbaar melden; groei over een gat niet als één jaar tellen | middel |
| 5 | Beslisboomblokken en banners naar het lichte thema | klein |
| 6 | Beheerknoppen en invoerformulieren achter de login uit stap 3, of naar een aparte beheerpagina | middel |
| 7 | Verkoopgrens niet tonen als de fair value meer dan een factor 3 boven de koers ligt bij krimpende omzet | klein |
| 8 | Eén getalnotatie, valuta op het dashboard, één taal voor signalen | middel |

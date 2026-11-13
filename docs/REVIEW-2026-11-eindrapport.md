# Kritische review Stockscreen — eindrapport

Datum: 2026-11-13 · Code op de commit van 10 november · Live site
stockscreen-janco.fly.dev · Read-only: er is niets gewijzigd aan code, database
of site.

Deelrapporten met de onderbouwing:
[stap 1 — model](REVIEW-2026-11-stap1-model.md) ·
[stap 2 — data](REVIEW-2026-11-stap2-data.md) ·
stap 3 — code (bewust niet in de repo, zie de kanttekening onderaan) ·
[stap 4 — site](REVIEW-2026-11-stap4-site.md)

## Oordeel in vijf zinnen

1. De **omlijsting is beter dan de kern**: verversing, poorten, narekenbaarheid
   en documentatie zijn zorgvuldig; de waardering zelf is één achterlopende
   winstmaat maal een veelvoud, met een aantoonbare voorkeur voor krimpende
   bedrijven.
2. Er zitten **twee rekenfouten in de historische multiples** (bedrijfswaarde en
   vermoedelijk dividendcorrectie) die de fair value van een groot deel van het
   universum scheeftrekken; de omvang is nog niet gemeten.
3. De **site staat open**: 27 endpoints die iets wijzigen of verwijderen hebben
   geen afscherming, en de repo die ze beschrijft is publiek.
4. De **verversing draait goed, maar niemand wordt gewaarschuwd als ze stopt**;
   dat is het patroon van de stilstand in juni, alleen korter.
5. Als **voorselectie voor eigen onderzoek** is het bruikbaar, mits je weet welke
   kant het op helt; de signalen BUY en SELL en de verkoopgrenzen suggereren
   meer dan het model kan dragen.

## Het systeem in cijfers (13 november)

| | |
|---|---|
| Tickers actief | 4.012 |
| Met oordeel | 2.950 (74%) |
| Koopsignalen | 47 (1,6%) |
| Fair value met laag vertrouwen | 57% van de beoordeelde |
| Kwaliteitsscore mediaan | 3,5 van 10 |
| Echte dataproblemen | 162 (4%) |
| Koersen jonger dan 24 uur | 98,7% |
| Testbestanden groen | 36 van 37 |
| Kansen-lijst tegen eigen oordelen | 15 onderzocht: 2 KOOP, 5 HOLD, 2 TWIJFEL, 6 OVERSLAAN |

## Prioriteiten vanuit het doel (na overleg met Janco, 13 november)

**Het doel.** De site bestaat om te bepalen welk bedrijf als volgende een
tussencheck krijgt. Ideaal: een doorlopende wachtrij van bedrijven om in te
verdiepen, die uitmondt in tussencheck → snelle scan → volledige analyse. De
fair value is een wegwijzer, geen beslisgrootheid. Gezocht: goedkope kwaliteit
en compounders. Knelpunten volgens Janco: te veel missers, goede bedrijven
gemist, en geen vertrouwen in de cijfers. Afscherming is wenselijk maar geen
prioriteit. Doorontwikkelen mag.

**De maatstaf die daaruit volgt:** welk deel van de wachtrij wordt bij
onderzoek VERDIEPEN of KOOP, en welke goede bedrijven kwamen er nooit in.

| # | Wat | Waarom vanuit het doel | Bron |
|---|---|---|---|
| 1 | **Cijfers betrouwbaar maken**: EV-formule repareren, dividendcorrectie toetsen, effect meten; lege boekjaren niet "compleet" noemen; een ijkset van tien tickers; rekenvoorbeelden als test | Zonder dit is elke rangorde verdacht, en het is klein werk | A1, A2, W1, D5, C3 |
| 2 | **Van Kansen naar een onderzoekswachtrij**: alleen bedrijven zonder recent oordeel, één regel per bedrijf, krimpers en vervallen winst eruit of gemarkeerd | Van de twintig Kansen zijn er vijftien al onderzocht en twee dubbel; de lijst levert nu drie tot vijf nieuwe namen | B3, W1 |
| 3 | **Tweede spoor voor compounders**: rangschikken op kwaliteit, standvastig rendement en ingeprijsde groei in plaats van op korting tegen de eigen historie | Het model zet deze groep per constructie op SELL; dit is "goede bedrijven gemist" | B2, B1 |
| 4 | **Kwaliteitsscore repareren**: groei niet als instabiliteit tellen, F-score boekjaar tegen boekjaar, richting van het rendement meewegen | Beide zoekprofielen leunen op kwaliteit, en de score mist nu juist groeiende kwaliteit | A3, A4, W2 |
| 5 | **Trefkans per spoor bijhouden in de tijd**, plus af en toe een bedrijf van buiten de wachtrij onderzoeken als controle | Dit is de enige echte ijking van de zeef, en de gegevens liggen er al | B3, B6 |
| 6 | **Signalen terugbrengen tot rangorde**: BUY/HOLD/SELL en de kwaliteitspoort vervallen als oordeel; korting, kwaliteit en vertrouwen blijven als kolommen | De fair value is een wegwijzer; een signaal suggereert een besluit dat de site niet hoort te nemen | B4 |
| 7 | **Alarm bij stilstand** (health geeft 503, foutmail van de wekker) | Klein werk; een wachtrij op oude cijfers is waardeloos | D1 |
| 8 | Afscherming van de schrijvende endpoints | Geen prioriteit voor Janco; blijft staan als klein werk voor een rustig moment | C1 |

Naar achteren: verkoopregels en bezitspagina, groeiprofiel-ijking, behandeling
van financials, opsplitsen van `app.py`, vormgeving en getalnotatie,
publicatievenster.

## Uitgevoerd: prioriteit 1, eerste deel (13 november, branch `fix/historische-multiples`)

Gerepareerd in `engine/data_fetcher.py`, met `tests/test_historische_multiples.py`
(11 rekenvoorbeelden):

- bedrijfswaarde = beurswaarde + schuld − kas (was + min(schuld, kas));
- historische koersen zonder dividendcorrectie (`auto_adjust=False`). Het
  vermoeden A2 is bevestigd op echte data: de gecorrigeerde slotkoers van eind
  2021 lag bij Rai Way 24% en bij Ahold 16% onder de werkelijke;
- de koers van de maand waarin het boekjaar eindigt, in plaats van december.

**Gemeten effect** op 149 willekeurige beoordeelde aandelen (oude multiples uit
productie tegen opnieuw berekende, zelfde jaarcijfers, verhouding van de fair
values):

| | |
|---|---|
| Mediane verandering fair value | +1,5% |
| Middelste helft | −0,3% tot +6,0% |
| 5e tot 95e percentiel | −13% tot +18% |
| Meer dan 5% anders | 62 van 149 (42%) |
| Meer dan 15% anders | 18 van 149 (12%) |
| Signaal verandert | 7 van 149 (5%): vijf keer SELL → HOLD, twee keer HOLD → SELL |
| Nettoschuld > 20% van de beurswaarde (n=70) | mediaan +5,5% |
| Ongeveer neutraal (n=49) | mediaan +0,1% |
| Nettokas > 10% van de beurswaarde (n=23) | mediaan −1,9% |

Conclusie: het midden verschuift nauwelijks, de staarten wel. De richting is
zoals voorspeld. De eerdere formulering dat dit "een groot deel van het
universum scheeftrekt" was te zwaar: het raakt ruwweg één op de acht aandelen
wezenlijk. De rangorde aan de randen verandert wel, en daar zit de wachtrij.

Kanttekeningen: de meting rekent zonder de TTM-rij (die levert de API niet), dus
alleen de verhouding telt. De uitschieter Standard Chartered (−37%) is een bank,
waar de bedrijfswaarde niets betekent (B5).

**Nog niet gedeployed en niet gecommit.** Na een deploy worden de multiples per
ticker hersteld zodra die aan de beurt komt in de ronde jaarcijfers; een
volledige ronde duurt ruim vijf dagen, en in die tijd staan oude en nieuwe
multiples door elkaar.

Nog open binnen prioriteit 1: lege boekjaren niet "compleet" noemen, de ijkset
van tien tickers, en rekenvoorbeelden voor de fair value zelf.

## Oorspronkelijke volgorde (opgesteld vóór het overleg over het doel)

De volgorde weegt gevolg tegen moeite. De eerste zes zijn klein werk.

| # | Actie | Waarom | Bron |
|---|---|---|---|
| 1 | Schrijvende endpoints afschermen (één controle vóór elk niet-GET-verzoek, plus een inlogcookie) | Iedereen kan nu aandelen verwijderen, cijfers overschrijven en instellingen wijzigen | C1, W4 |
| 2 | EV-formule in de historische multiples repareren en het effect meten met een proefdraai | Twee van de vier multiple-methodes rusten op een verkeerde bedrijfswaarde | A1 |
| 3 | Dividendcorrectie toetsen op één ticker; zo nodig uitzetten en herberekenen | Historische K/W van dividendbetalers valt vermoedelijk te laag uit | A2 |
| 4 | `/api/health` een 503 laten geven bij verouderde data; foutmail van de wekker aan; één controle overdag | Stilstand wordt nu alleen bij toeval ontdekt | D1 |
| 5 | Rode test repareren (tk accelis mist de kansgewogen waarde); één testrunner; GitHub Action bij elke push | De laatste commit is rood gepusht en niets hield dat tegen | C3 |
| 6 | Zichtbare fouten op de site: pence/pond op `/leren`, dubbele James Halstead, voorstel-document in het scorebord, tijdas van de grafiek | Fout op het scherm, klein om te herstellen | W1 |
| 7 | Besluit nemen over de kwaliteitspoort (open sinds augustus) | Het koopsignaal is nu vooral een kwaliteitsfilter met bekende gebreken | B4 |
| 8 | Krimpbias zichtbaar maken: winstrichting als kolom of in de rangscore; geen verkoopgrens bij een fair value ver boven de koers en krimpende omzet | Krimpers krijgen mediaan +20% "korting", snelle groeiers −20% | B2, W2 |
| 9 | Versies vastpinnen en een ijkset van tien tickers na elke ronde toetsen | Een stille breuk bij Yahoo bevriest de cijfers zonder foutmelding | D5 |
| 10 | Rekenvoorbeelden als test voor fair value en historische multiples; F-score boekjaar tegen boekjaar; stabiliteit rond de trend | De rekenkern is nooit tegen een nagerekend voorbeeld gezet | C3, A3, A4 |

## Alle bevindingen per zwaarte

### Hoog

| Code | Bevinding |
|---|---|
| C1 | 27 schrijvende endpoints zonder afscherming op een publieke site met publieke repo |
| A1 | Historische bedrijfswaarde = beurswaarde + min(schuld, kas) in plaats van + nettoschuld |
| B2 | Het model beloont krimp en straft groei (gemeten over 2.740 aandelen) |
| D1 | Geen waarschuwing bij stilstand van de verversing |

### Middel

| Code | Bevinding |
|---|---|
| A2 | Historische koersen vermoedelijk dividend-gecorrigeerd; jaareinde op kalenderjaar in plaats van boekjaar |
| A3 | F-score vergelijkt de laatste twaalf maanden met het laatste boekjaar (overlappende perioden) |
| A4 | Stabiliteitscriterium bestraft gelijkmatige groei |
| B1 | Drie waarderingsmethodes zijn varianten van dezelfde som; "hoog vertrouwen" zegt weinig |
| B4 | Kwaliteitspoort wacht sinds augustus op een besluit |
| B5 | Banken en verzekeraars krijgen een signaal op maatstaven die daar niets betekenen |
| B6 | IJkingen van moat en groeiprofiel zijn in-sample; "geijkt" is te sterk |
| C2 | De instellingenpagina bewaart niets (vluchtig bestandssysteem) |
| C3 | Geen testrunner, geen CI, één rode test; rekenkern nauwelijks getest |
| C4 | `app.py` van 3.582 regels; signaal op twee plekken bepaald; documentatie compenseert voor structuur |
| D2 | Gemiste koersronde wordt niet ingehaald vóór 18:30 |
| D3 | Publicatievenster is dode code en staat omgekeerd |
| D4 | Suspenderen kwam acht weken later dan gedocumenteerd, onopgemerkt en onverklaard |
| D5 | Eén onofficiële bron zonder vastgepinde versie of ijkcontrole |
| W1 | Zichtbare fouten: pence/pond, dubbele notering, intern document als analyse, leeg boekjaar als "compleet", vertekende tijdas |
| W2 | Te stellige uitspraken: verkoopgrens Arctic Paper (+763%), "rendement houdt stand" bij dalende reeks |
| W4 | Beheerknoppen en invoerformulieren staan tussen de inhoud |

### Laag

| Code | Bevinding |
|---|---|
| A5 | ROIC telt schuld dubbel bij nettokas; rentedekking gratis bij ontbrekende rentelast; wisselkoers van nu op cijfers van toen |
| B7 | 26% zonder oordeel, 57% van de rest met laag vertrouwen |
| C5 | Nieuwe databaseverbinding per query; geen SQL-injectie gevonden |
| C6 | Vaste terugvalwaarde voor `secret_key`; geen begrenzing op verzoeken |
| D6 | Tien tickers met een signaal op een verouderde koers; Amerikaanse tickers krijgen een koers van halverwege de dag als slotkoers |
| W3 | Donkere blokken slecht leesbaar; getalnotatie, valuta en taal door elkaar |
| W5 | Smalle weergave bruikbaar, niet verzorgd |

## Wat goed is en zo moet blijven

- Het systeem zwijgt liever dan dat het gokt: datapoort, plausibiliteitsgrens,
  minimaal twee methodes, omzetbreuk.
- De lessen uit april en juni zitten in code: storm-wacht, NULL-bescherming,
  één ronde tegelijk, een scheduler die een herstart overleeft.
- `/api/trace` en `/methode` maken elke berekening narekenbaar.
- De framing is juist en staat op de pagina: zeef en geen oordeel,
  onderzoeksopdracht en geen verkoopopdracht.
- De terugkoppeling naar eigen oordelen (telling boven Kansen, `/leren`,
  scorebord) is het waardevolste onderdeel; daar zit de echte ijking.
- Elf van de dertien aanbevelingen uit de review van augustus zijn uitgevoerd.

## Wat niet is vastgesteld

- De omvang van A1 en A2 op de fair values (vraagt een herberekening).
- Of A2 werkelijk optreedt (afgeleid uit het standaardgedrag van yfinance).
- De oorzaak van D4 (vraagt het activiteitenlog).
- Of er een herstelpunt van de database is.
- C2 is uit de code afgeleid, niet live getest.
- Niet regel voor regel gelezen: `exit_regels.py`, `scorebord.py`,
  `besluiten.py`, `data_quality.py`, `app.py`, `db.py`.
- Niet bekeken op de site: Herstellers, Mijn lijst, Alles, Geen oordeel,
  `/tussenchecks`, een analyse-detailpagina, `/log`; geen test op een telefoon.

## Eén kanttekening bij publiceren

De repo is publiek. Het deelrapport van stap 3 beschrijft welke endpoints
openstaan en wat ermee kan. Het staat daarom niet in de repo; Janco heeft het
lokaal. Voeg het pas toe nadat de schrijvende endpoints zijn afgeschermd.

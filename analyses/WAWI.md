# Research: WAWI — Wallenius Wilhelmsen ASA

---

## Bronnen-inventaris (Stap 0.5)

> Opgesteld vóór het invullen van één numeriek veld, conform METHODE.md Stap 0.5.
> Overzichtspagina's die zijn geopend (niet geraden): de IR-hoofdpagina
> `https://www.walleniuswilhelmsen.com/who-we-are/investors` en de
> jaarverslagenarchiefpagina `https://www.walleniuswilhelmsen.com/who-we-are/investors/annual-report`.
> Alle onderstaande URL's komen letterlijk van die twee pagina's.

**Kernbron voor de reeks 2020–2026:** Wallenius Wilhelmsen publiceert per kwartaal
een *Factsheet* in Excel met de volledige geconsolideerde winst- en verliesrekening,
balans én kasstroomoverzicht per kwartaal vanaf Q1 2020, plus segmentcijfers,
vlootlijst, schuldvervalprofiel en ESG-data. Deze bestanden zijn opgehaald door
Janco (haallijst 2026-08-20) en staan in
`C:\Users\janco\aandelenanalyse\research\_bronnen\WAWI\`. Ze zijn primaire
bedrijfsbronnen (IR) en tellen als **HOOG**.

```
Jaar 2026 (H1 + TTM) — HOOG
  Bron: Wallenius Wilhelmsen Factsheet Q2 2026 (xlsx) + Quarterly report Q2 2026 (PDF)
  URL:  https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/Wallenius-Wilhelmsen-Factsheet-Q2-2026.xlsx
        https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/Wallenius-Wilhelmsen-Quarterly-report-Q2-2026.pdf
        https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/WAWI-Q2-26-presentation-Final.pdf
  Daadwerkelijk geopend: ja (xlsx ingelezen met openpyxl, alle tabbladen)
  Cijfers overgenomen: omzet, EBITDA, adj. EBITDA, D&A, EBIT, resultaat vóór
                       belasting, belastinglast, nettowinst, aandeel moeder, EPS,
                       volledige balans (incl. lease- en putverplichting),
                       volledige kasstroom (CFO, betaalde belasting, capex,
                       lease-aflossing, betaalde rente, dividend), segment-P&L,
                       volumes, netto vrachttarief, TC-resultaat per dag, vlootlijst
                       per schip (129 schepen, bouwjaar en eigendomsvorm),
                       schuldvervalprofiel, CO2- en veiligheidsdata
  Cijfers NIET overgenomen: (geen)

Jaar 2025 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2025 + Factsheet Q4 2025 (xlsx)
  URL:  https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/2025-Annual-report-document.pdf
        https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen-Factsheet-Q4-2025.xlsx
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: alle bovengenoemde reeksen + personeelsaantal, aantal landen,
                       vlootsamenstelling, segmentomzet en -EBITDA, dividendbeleid,
                       financiële doelstellingen, nieuwbouwprogramma
  Cijfers NIET overgenomen: (geen)

Jaar 2024 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2024 + Factsheet Q4 2024 (xlsx)
  URL:  https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/WAWI_2024-Annual-report.pdf
        https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen-Q4-2024-Fact-Sheet.xlsx
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks (de Q2 2026-factsheet bevat de door de
                       EUKOR-putoptie geherformuleerde 2023–2024 cijfers; die zijn
                       leidend, zie de toelichting bij de balans)
  Cijfers NIET overgenomen: (geen)

Jaar 2023 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2023 (PDF), aangevuld met de
        geherformuleerde reeks uit de Factsheet Q2 2026
  URL:  https://www.walleniuswilhelmsen.com/storage/images/WWAnnualReport2023.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks (geherformuleerd)
  Cijfers NIET overgenomen: de oorspronkelijk gerapporteerde eigen-vermogen- en
                       minderheidsbelangposten 2023 (vervangen door de
                       geherformuleerde cijfers; verschil toegelicht)

Jaar 2022 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2022 (PDF) + Factsheet Q2 2026
  URL:  https://www.walleniuswilhelmsen.com/storage/images/2022-wallenius-wilhelmsen-annual-report.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks
  Cijfers NIET overgenomen: (geen)

Jaar 2021 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2021 (PDF) + Factsheet Q2 2026
  URL:  https://www.walleniuswilhelmsen.com/storage/downloads/Annual-report-2021.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks
  Cijfers NIET overgenomen: (geen)

Jaar 2020 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2020 (PDF) + Factsheet Q2 2026
  URL:  https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen_Annual-Report-2020.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: volledige reeks. Kruiscontrole: AR2020 meldt CFO USD 615 mln
                       en kas USD 654 mln; de factsheet geeft 614,7 en 654,2 —
                       de twee bronnen sluiten op elkaar aan.
  Cijfers NIET overgenomen: (geen)

Jaar 2019 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2019 (PDF)
  URL:  https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius-Wilhelmsen-Annual-Report-2019.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: totale omzet, EBITDA, EBIT, resultaat vóór belasting,
                       belastinglast, nettowinst, EPS, CFO, capex, kas,
                       bruto rentedragende schuld, eigen vermogen (totaal en moeder),
                       balanstotaal, dividend per aandeel
  Cijfers NIET overgenomen: uitsplitsing leaseverplichting, betaalde belasting,
                       betaalde rente, lease-aflossing (niet in de opgehaalde
                       secties; de factsheet begint pas bij Q1 2020)

Jaar 2018 — HOOG
  Bron: Wallenius Wilhelmsen Annual report 2018 (PDF), aangevuld met de
        vergelijkende cijfers in AR2019 en de vijfjaarstabel in AR2022
  URL:  https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius_Wilhelmsen_Annual_report_2018.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: totale omzet, EBITDA, EBIT, resultaat vóór belasting,
                       belastinglast, nettowinst, EPS, CFO, kas, bruto schuld,
                       eigen vermogen, balanstotaal
  Cijfers NIET overgenomen: capex, betaalde belasting, betaalde rente,
                       kasstroomdetail (niet apart gespecificeerd in de bron)

Jaar 2017 — HOOG
  Bron: Wallenius Wilhelmsen Logistics ASA Annual report 2017 (PDF)
  URL:  https://www.walleniuswilhelmsen.com/storage/images/WWL-Annual-Report-2017.pdf
  Daadwerkelijk geopend: ja
  Cijfers overgenomen: totale omzet, EBITDA, adj. EBITDA, EBIT, resultaat vóór
                       belasting, belastinglast, nettowinst, EPS, CFO, kas,
                       bruto schuld, eigen vermogen, balanstotaal
  Cijfers NIET overgenomen: capex, kasstroomdetail. LET OP: 2017 bevat de
                       geconsolideerde cijfers vanaf de fusiedatum 4 april 2017;
                       het eerste kwartaal is niet meegeconsolideerd. De ratio's
                       van 2017 zijn daarom niet volledig vergelijkbaar met latere
                       jaren; dit is expliciet vermeld bij de tabellen.

Jaar 2016 — GEEN VERGELIJKBARE DATA (bron bestaat, perimeter wijkt af)
  Bron: WWASA Annual report 2016 (PDF)
  URL:  https://www.walleniuswilhelmsen.com/storage/downloads/wwasa-annual-report-2016.pdf
  Zoekpoging: bron staat op de archiefpagina en is gevonden; de reden om hem niet
              in de reeks op te nemen is inhoudelijk, niet praktisch.
  Conclusie: 2016 betreft Wilh. Wilhelmsen ASA (WWASA) vóór de fusie met WallRoll AB
             (4 april 2017). In dat jaar werden WWL, EUKOR en ARC nog volgens de
             vermogensmutatiemethode verwerkt in plaats van geconsolideerd. Omzet,
             EBITDA, balanstotaal en aandelenaantal (220 mln vs. 423 mln) zijn
             daardoor onvergelijkbaar met 2017 en later. ALLE cellen van 2016
             blijven LEEG. Genoteerd in ontbrekende_data.
```

**Zelf-check na de inventaris:** voor elke ingevulde numerieke cel in dit rapport
is een bron-URL uit deze inventaris aanwijsbaar. Cellen zonder bron zijn leeg
gelaten en staan in sectie 13 onder ontbrekende data.

**Haallijst — status:** afgehandeld. De drie Excel-factsheets die ik zelf niet kon
lezen (binair formaat) zijn op 2026-08-20 door Janco aangeleverd. Er staat niets meer open.

**Wat werkte en wat niet — voor de volgende analyse:**
- `walleniuswilhelmsen.com` levert alle jaarverslagen 2016–2025 als open PDF, geen botbeveiliging.
- De **kwartaal-factsheets in xlsx** zijn dé bron voor deze uitgever: zes jaar
  kwartaaldata inclusief volledige kasstroom, segment-P&L, vlootlijst per schip en
  ESG. Ze zijn niet leesbaar via fetch (binair) — zet ze meteen op de haallijst.
- `stockanalysis.com` en `finance.yahoo.com` leveren voor WAWI verouderde
  gecachte koersen (mei/juli 2026 bij een peildatum in augustus). MarketScreener
  en Investtech gaven wél de slotkoers van de peildatum, en bevestigden elkaar.
- `live.euronext.com` geeft voor Oslo-noteringen geen bruikbare koersdata.

---

## Metadata
- **Ticker (bare):** WAWI
- **Yahoo symbol:** WAWI.OL
- **Exchange:** OSL (Oslo Børs / Euronext Oslo)
- **Sector (GICS-achtig):** Industrie
- **Industrie:** Zeevaart en logistiek (autotransport / RoRo)
- **Land:** Noorwegen
- **Peildatum analyse:** 2026-08-20
- **Koers op peildatum:** 159,40
- **Valuta:** NOK
- **Marktkapitalisatie:** NOK 67,4 mld (USD 7,23 mld)
- **Marktkap in mln (lokale valuta):** 67443
- **Free float pct:** 23,88
- **Indexlidmaatschap:** Oslo Børs hoofdlijst; niet geverifieerd of WAWI in de OBX-index zit — weggelaten
- **Domein:** walleniuswilhelmsen.com

> **Valuta-waarschuwing (METHODE STAGE 1, regel 2):** Wallenius Wilhelmsen
> rapporteert in **USD**; het aandeel noteert in **NOK**. Alle kasstromen,
> balansposten en de DCF zijn in **USD mln** gemodelleerd. De fair value is
> omgerekend naar NOK tegen de spotkoers **USD/NOK 9,3280** op 2026-08-20
> (Trading Economics). De koers van NOK 159,40 komt overeen met **USD 17,088**
> per aandeel.

---

## 1. Executive summary

- **Kernthese:**
Wallenius Wilhelmsen is de grootste onafhankelijke vervoerder van auto's, vrachtwagens
en rollend materieel over zee. Het bedrijf bezit of huurt 129 schepen met samen bijna
900.000 autoplaatsen en vervoert daarmee ongeveer 54 miljoen kubieke meter lading per
jaar over vijftien vaste vaarroutes. Naast het varen zelf beheert het bedrijf acht
haventerminals en zeventig verwerkingscentra waar nieuwe auto's worden geïnspecteerd,
gepoetst en klaargemaakt voor de dealer — een dienstenlaag die concurrenten die alleen
schepen exploiteren niet hebben. Een derde poot vervoert militair materieel voor de
Amerikaanse overheid onder Amerikaanse vlag, een afgeschermde nichemarkt met vaste
contracten. De structurele groeimotor van de afgelopen jaren is de Chinese auto-export,
die in twaalf maanden tot medio 2026 met 72% steeg tot 8,5 miljoen voertuigen en die
per schip veel verder moet worden vervoerd dan Europese of Japanse export, waardoor
elke auto meer scheepsruimte per jaar opeist. Daar komt bij dat de omvaart om Kaap de
Goede Hoop, sinds de Rode Zee onveilig werd, extra vaartijd en dus extra schaarste
creëert. Het bedrijf verdient aan langlopende contracten met autofabrikanten — de
orderportefeuille in de scheepvaartdivisie bedraagt USD 6,5 miljard met een gemiddelde
looptijd van 2,9 jaar — en verhoogt bij verlenging de tarieven. Tegenover die
rugwind staat het kernrisico van elke scheepvaartsector: het orderboek voor nieuwe
autoschepen bedraagt inmiddels ongeveer 21% van de wereldvloot, met opleveringen die
vanaf 2028 op de markt komen en die de huidige schaarste kunnen omslaan in overcapaciteit.
Daarbovenop is de eigen vloot oud — de 91 schepen in eigendom zijn gemiddeld 18,3 jaar
en achttien daarvan hebben de 25 jaar al gepasseerd — waardoor de komende jaren fors
meer geld naar vlootvernieuwing moet dan in de afgelopen vijf jaar het geval was.

- **Oordeel:** **PASS**
- **Fair value basis** (basisscenario, NOK): 129,25
- **Fair value kansgewogen** (NOK): 135,27
- **EPV per aandeel** (NOK): 89,75
- **Upside pct**: −18,9
- **Fair value scenarios:**

| Scenario | Fair value | Upside % | FCF groei % | WACC % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 47,11 | −70,4 | 0,0 | 9,91 | 35 |
| Basis | 129,25 | −18,9 | 3,0 | 8,41 | 45 |
| Optimistisch | 303,12 | +90,2 | 5,0 | 7,91 | 20 |

- **Reverse-DCF impliciete groei pct:** 5,84 (de koers vereist bijna 6% jaarlijkse
  FCF-groei gedurende tien jaar vanaf een genormaliseerde vrije kasstroom van
  USD 385 mln; anders geformuleerd: de markt prijst een duurzame vrije kasstroom
  van USD 463 mln in, 20% boven mijn mid-cyclusraming)
- **Grootste kans:** de Chinese auto-export blijft sneller groeien dan de vlootuitbreiding,
  waardoor de tariefpiek langer aanhoudt dan het orderboek suggereert.
- **Grootste risico:** een orderboek van circa 21% van de wereldvloot dat vanaf 2028
  wordt opgeleverd, precies wanneer de eigen vloot van gemiddeld 18,3 jaar zelf ook
  vervangen moet worden.

---

## 2. Bedrijfsprofiel

- **Beschrijving:**
Wallenius Wilhelmsen vervoert wat niet in een container past en op wielen aan boord kan
rijden: personenauto's, bestelwagens, vrachtwagens, tractoren, graafmachines,
mijnbouwmaterieel en militair transportmaterieel. Het bedrijf zit midden in de
waardeketen van de auto-industrie. Een fabrikant in Shanghai, Ulsan, Bremerhaven of
Georgia levert een auto af aan de fabriekspoort; vanaf daar neemt Wallenius Wilhelmsen
het over. De auto gaat naar een verwerkingscentrum, wordt geïnspecteerd en zo nodig
uitgerust met accessoires, rijdt de haventerminal op, gaat aan boord van een RoRo-schip,
vaart de oceaan over, wordt in de aankomsthaven weer gelost, gaat opnieuw door een
verwerkingscentrum en wordt over de weg of het spoor naar de dealer gebracht. Het bedrijf
kan dat hele traject leveren en verdient daarmee aan meer dan alleen de zeereis. Wat het
onderscheidt van pure reders is precies die landzijde: acht haventerminals en zeventig
verwerkingscentra in 28 landen, gekoppeld aan een vloot die groot genoeg is om vijftien
vaste routes met hoge frequentie te bedienen. Voor een autofabrikant die elke week
duizenden voertuigen over de oceaan moet krijgen, is die combinatie van vaste afvaarten
en landafhandeling het product. De omzet komt overwegend uit meerjarige contracten met
fabrikanten waarin een tarief per kubieke meter of per voertuig is vastgelegd, aangevuld
met toeslagen die brandstof- en valutaschommelingen doorberekenen. In 2025 kwam
USD 3.989 mln van de USD 5.240 mln groepsomzet uit scheepvaart, USD 1.087 mln uit
logistiek en USD 411 mln uit de Amerikaanse overheidsdivisie.

- **Geschiedenis:**
De twee families achter het bedrijf varen al meer dan een eeuw. Wilh. Wilhelmsen werd in
1861 opgericht in Tønsberg, Noorwegen, en groeide uit tot een van de grootste Noorse
rederijen; Wallenius Lines begon in 1934 in Stockholm en was in de jaren zestig een van
de pioniers van het RoRo-schip, het type schip waarbij de lading zelf aan boord rijdt in
plaats van te worden gehesen. In 1999 brachten beide families hun autotransportactiviteiten
onder in de joint venture Wallenius Wilhelmsen Logistics. In 2002 volgde EUKOR Car
Carriers, gevormd rond het vervoer voor Hyundai en Kia, waarin Wallenius Wilhelmsen 80%
kreeg en de Hyundai-groep 20%. Daarnaast kwam American Roll-on Roll-off Carrier (ARC)
tot stand, dat onder Amerikaanse vlag militair materieel vervoert. De keerpunten van de
afgelopen tien jaar zijn er drie. Het eerste is het kartelonderzoek dat in september 2012
begon en waarbij mededingingsautoriteiten wereldwijd vaststelden dat autoschipreders
onderling routes en klanten hadden verdeeld; de Europese Commissie legde in februari 2018
zes vervoerders samen EUR 207 mln op, waaronder WWL en EUKOR, en er volgden schikkingen
in de Verenigde Staten, Australië, Japan en elders. Het tweede keerpunt is de fusie van
4 april 2017, waarbij Wilh. Wilhelmsen ASA en WallRoll AB samengingen; de beursgenoteerde
Noorse vennootschap bleef bestaan als overnemende partij en consolideerde vanaf dat moment
WWL, EUKOR en ARC volledig in plaats van ze als deelneming te verwerken. De naam werd eerst
Wallenius Wilhelmsen Logistics ASA en later Wallenius Wilhelmsen ASA. Het derde keerpunt is
de coronacrisis: in 2020 kelderde de omzet met 24% tot USD 2.958 mln, ging het bedrijf voor
USD 302 mln het rood in, werd het dividend geschrapt en moesten bankleningen worden
uitgesteld. Daarna volgde de sterkste periode uit de bedrijfsgeschiedenis. Autofabrikanten
liepen achterstanden in, de Chinese export explodeerde, de vloot bleek te klein en de
aangepaste EBITDA verdrievoudigde van USD 535 mln in 2020 naar USD 1.811 mln in 2025.
Het bedrijf keerde in 2023–2025 samen ruim USD 2 miljard aan dividend uit, halveerde de
nettoschuld van USD 3,4 mrd naar USD 1,7 mrd, verkocht de terminal MIRRAT met een boekwinst
van USD 135 mln en bestelde veertien methanol-gedreven Shaper Class-schepen bij China
Merchants Jinling.

- **Bedrijfsmodel:**
Het geld komt uit drie bronnen. De scheepvaartdivisie (72,7% van de segmentomzet 2025)
verkoopt scheepsruimte tegen een netto vrachttarief per kubieke meter; in 2025 lag dat
gemiddeld op USD 64,61 per cbm en over de laatste twaalf maanden op USD 62,95. Bovenop
dat tarief komen toeslagen die brandstof, valutabewegingen en havenkosten doorberekenen —
in 2025 goed voor USD 498 mln. Bijna alle omzet loopt via contracten met een looptijd van
één tot vijf jaar; de contractportefeuille stond eind Q2 2026 op USD 6,5 mrd met een
gewogen resterende looptijd van 2,9 jaar. De logistieke divisie (19,8%) verdient per
behandeld voertuig aan terminaloverslag, technische verwerking en achterlandvervoer, met
een orderportefeuille van USD 2,7 mrd en een veel langere gemiddelde looptijd van 8,1 jaar.
De overheidsdivisie (7,5%) vaart onder Amerikaanse vlag voor het Amerikaanse ministerie van
Defensie, waar buitenlandse reders per wet niet mogen komen. Terugkerende contractomzet
domineert; spotwerk is een minderheid.

- **IPO-context:**
Er is geen klassieke beursgang in de recente geschiedenis. De huidige vennootschap is de
voortzetting van Wilh. Wilhelmsen ASA (WWASA), dat op **24 juni 2010** aan Oslo Børs werd
genoteerd nadat de scheepvaart- en logistiekactiviteiten van de toenmalige groep in een
aparte beursvennootschap waren afgesplitst. Bij de fusie werden 203,1 miljoen nieuwe aandelen uitgegeven aan de
Wallenius-zijde, waardoor het aandelenaantal van 220 miljoen naar 423,1 miljoen ging en
de zeggenschap gelijkelijk over de twee families werd verdeeld. Sindsdien is de
kapitaalstructuur nagenoeg onveranderd: het aantal uitstaande aandelen bewoog tussen
422,3 en 423,1 miljoen, er is nauwelijks aandelencompensatie en er zijn geen emissies
of noemenswaardige inkoopprogramma's geweest.

- **Klantprofiel:**
Uitsluitend B2B, en de klantenlijst is kort maar zwaar. Het zijn autofabrikanten — van de
Hyundai-groep en Kia tot Europese en Chinese merken — fabrikanten van bouw-, landbouw- en
mijnbouwmachines, samen goed voor 22 tot 24% van het volume en per kubieke meter beter
betaald dan een personenauto, en het Amerikaanse ministerie van Defensie. Wereldwijd zijn
er maar een handvol concerns die jaarlijks honderdduizenden voertuigen over de oceaan
moeten krijgen; de klantenbasis is dus per definitie geconcentreerd en de onderhandelingsmacht
van de afnemer groot. Daar staat tegenover dat de relaties lang duren: één tot vijf jaar in
scheepvaart, gemiddeld 8,1 jaar in logistiek. De belangrijkste enkele relatie is die met de
Hyundai-groep via EUKOR, waar Hyundai Motor Group tegelijk 20%-minderheidsaandeelhouder is
en het oceaanvervoercontract tot december 2029 loopt.

- **Oprichtingsjaar:** 1861 (Wilh. Wilhelmsen); 1934 (Wallenius Lines); huidige groep sinds 4 april 2017
- **IPO-datum:** 2010-06-24 (eerste handelsdag WWASA op Oslo Børs na afsplitsing; geen nieuwe beursgang sindsdien)
- **IPO-koers:** — (afsplitsing, geen emissiekoers)
- **Personeel (FTE):** 8.253 (headcount ultimo 2025)
- **Landen actief:** 28
- **Klantconcentratie:** Wallenius Wilhelmsen publiceert geen omzetaandeel per klant.
  Wat wél verifieerbaar is: EUKOR is opgericht rond het vervoer voor Hyundai en Kia en
  Hyundai Motor Group houdt 20% van EUKOR, met een oceaanvervoercontract dat in december
  2029 afloopt. De overheidsdivisie (USD 411 mln in 2025, 7,5% van de segmentomzet) heeft
  in feite één afnemer, de Amerikaanse overheid. Een precies percentage per klant is niet
  verifieerbaar en is daarom weggelaten.

### Geografische spreiding (omzet)

Wallenius Wilhelmsen rapporteert geen omzet per land of regio, maar wel volumes per
handelsroute. Onderstaande verdeling is daarom op **volume** (cbm, boekjaar 2025, vóór
intercompany-eliminatie) en niet op omzet; het percentage is berekend uit de
kwartaalcijfers in de Factsheet Q2 2026.

| Regio (handelsroute) | Volume-aandeel % | Valuta-exposure |
|---|---|---|
| Azië → Noord-Amerika | 26,1 | USD |
| Overige routes | 22,5 | gemengd |
| Azië → Europa | 14,3 | EUR/USD |
| Azië → Zuid-Amerika westkust | 12,6 | USD |
| Atlantic (EU ↔ NA) | 11,2 | EUR/USD |
| Europa → Noord-Amerika / Oceanië | 7,6 | EUR/USD/AUD |
| Europa → Azië | 5,8 | EUR/USD |

**Toelichting geografie:** ruim de helft van het volume (52,9%) vertrekt uit Azië, en die
routes zijn de motor van de laatste jaren: het volume van Azië naar Noord-Amerika groeide
van 9,5 mln cbm in 2020 naar 15,0 mln cbm in 2025, terwijl de route Europa → Azië in
dezelfde periode meer dan halveerde. Het valutarisico is beperkt omdat de vrachttarieven
en de belangrijkste kosten — brandstof, charterhuur, scheepsfinanciering — vrijwel
allemaal in dollars luiden; dat is een natuurlijke hedge. Het echte valutarisico zit bij
de belegger: de kasstroom is in dollars, het aandeel noteert in kronen, en de kroon
apprecieerde in het afgelopen jaar 8,3% tegenover de dollar, wat het rendement in NOK
navenant drukt.

### Segmenten (boekjaar 2025, als % van de som van de segmentomzetten)

| Naam | Omzet % | Beschrijving |
|---|---|---|
| Shipping Services | 72,7 | Het varen zelf: 118 schepen op vijftien routes, verkocht per kubieke meter onder meerjarige contracten met toeslagen voor brandstof. Genereerde in 2025 USD 1.561 mln aangepaste EBITDA op USD 3.989 mln omzet — dit segment is het bedrijf. |
| Logistics Services | 19,8 | Acht haventerminals, zeventig verwerkingscentra en achterlandvervoer. Lage marge (USD 133 mln EBITDA op USD 1.087 mln omzet in 2025) maar een orderportefeuille met 8,1 jaar gemiddelde looptijd en de reden waarom fabrikanten het hele traject bij één partij onderbrengen. |
| Government Services | 7,5 | ARC vervoert militair materieel onder Amerikaanse vlag voor het Amerikaanse ministerie van Defensie. Tien tot elf schepen, wettelijk afgeschermd van buitenlandse concurrentie, met de hoogste EBITDA-marge van de drie segmenten (37% in 2025). |

### Aandeelhouders (top 5)

| Naam | Belang % | Type (oprichter / PE / institutioneel / retail) |
|---|---|---|
| Wilh. Wilhelmsen Holding ASA | 37,87 | Oprichtersfamilie (controlerend) |
| Soya Group AB (Wallenius Lines) | 37,82 | Oprichtersfamilie (controlerend) |
| Folketrygdfondet | 1,89 | Institutioneel |
| Nordea Investment Management (Norge) | 0,24 | Institutioneel |
| Danske Bank A/S (Investment Management Norge) | 0,22 | Institutioneel |

- **Institutioneel eigendomstrend:** stabiel en structureel klein. De twee families houden
  samen 75,7% en die verhouding is sinds de fusie van 2017 vrijwel onveranderd; de free
  float bedraagt 23,88%. Het grootste onafhankelijke belang is dat van het Noorse
  staatsfonds Folketrygdfondet met 1,89%. Een trendbreuk in institutioneel eigendom is in
  de beschikbare bronnen niet waarneembaar en is daarom niet geduid.

---

## 3. Financieel — historische data (10 jaar + TTM)

> **Twee dingen om te weten voordat u de tabellen leest.**
> **(1)** Boekjaar 2016 is leeg. Dat jaar betreft Wilh. Wilhelmsen ASA vóór de fusie van
> april 2017, toen WWL, EUKOR en ARC nog als deelneming werden verwerkt in plaats van
> geconsolideerd. Omzet, balans en aandelenaantal zijn daardoor niet vergelijkbaar; ze
> invullen zou de reeks vervalsen. Boekjaar 2017 bevat de geconsolideerde cijfers vanaf
> 4 april 2017 en is dus een gebroken jaar — ook dat is niet volledig vergelijkbaar.
> **(2)** IFRS 16 geldt vanaf 1 januari 2019. Vóór 2019 stond charterhuur in de
> bedrijfskosten en drukte die de EBITDA; vanaf 2019 staat ze als afschrijving en rente
> buiten de EBITDA. De EBITDA-marges van 2017 en 2018 zijn daarom structureel lager dan
> die van latere jaren en niet één-op-één vergelijkbaar.

### Resultatenrekening (bedragen in USD mln)

| Jaar | Omzet | Omzetgroei % | Brutowinst | Brutomarge % | EBIT | EBIT-marge % | EBITDA | EBITDA-marge % | Nettowinst | Nettomarge % | EPS | EPS-groei % | Aandelen mln |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 2017 | 2.992 | — | — | — | 253 | 8,5 | 524 | 17,5 | 146 | 4,9 | 0,37 | — | 423,1 |
| 2018 | 4.065 | 35,9 | — | — | 244 | 6,0 | 601 | 14,8 | 58 | 1,4 | 0,12 | −67,6 | 423,1 |
| 2019 | 3.909 | −3,8 | — | — | 358 | 9,2 | 805 | 20,6 | 102 | 2,6 | 0,22 | 83,3 | 423,1 |
| 2020 | 2.957 | −24,3 | — | — | −84 | −2,8 | 473 | 16,0 | −301 | −10,2 | −0,65 | n.v.t. | 422,4 |
| 2021 | 3.884 | 31,3 | — | — | 306 | 7,9 | 830 | 21,4 | 177 | 4,5 | 0,31 | n.v.t. | 422,4 |
| 2022 | 5.045 | 29,9 | — | — | 931 | 18,4 | 1.548 | 30,7 | 794 | 15,7 | 1,60 | 416,1 | 422,5 |
| 2023 | 5.149 | 2,1 | — | — | 1.225 | 23,8 | 1.807 | 35,1 | 974 | 18,9 | 2,02 | 26,3 | 422,7 |
| 2024 | 5.308 | 3,1 | — | — | 1.289 | 24,3 | 1.869 | 35,2 | 1.066 | 20,1 | 2,30 | 13,9 | 422,6 |
| 2025 | 5.240 | −1,3 | — | — | 1.285 | 24,5 | 1.801 | 34,4 | 1.104 | 21,1 | 2,41 | 4,8 | 422,8 |
| TTM | 5.150 | −1,7 | — | — | 925 | 18,0 | 1.596 | 31,0 | 770 | 14,9 | 1,65 | −31,5 | 422,9 |

*Aangepaste EBITDA (management-maatstaf, exclusief eenmalige posten): 2017 706; 2018 601;
2019 805; 2020 536; 2021 865; 2022 1.528; 2023 1.807; 2024 1.901; 2025 1.811; TTM 1.626.*

*Brutowinst en brutomarge worden door Wallenius Wilhelmsen niet gerapporteerd — de
winst- en verliesrekening kent geen kostprijs-van-omzetregel. De kolommen zijn daarom
leeg gelaten in plaats van te worden afgeleid.*

- **Toelichting resultaten:**
De reeks vertelt het verhaal van één volledige scheepvaartcyclus. Tussen 2017 en 2021
verdiende het bedrijf nauwelijks iets: de EBIT-marge schommelde rond 6 tot 9% en in
coronajaar 2020 werd USD 301 mln verlies geleden, met een omzetdaling van 24% in twaalf
maanden. Vanaf 2022 kantelde alles. De omzet sprong in twee jaar van USD 3,9 mrd naar
USD 5,0 mrd, maar veel belangrijker was dat de EBIT-marge meer dan verdrievoudigde van 7,9% naar
24,5%: er kwam niet zozeer meer lading bij — het volume was in 2025 met 53,7 mln cbm
zelfs lager dan de 59,2 mln in 2023 — maar het tarief per kubieke meter steeg van
USD 48,23 in 2021 naar USD 64,61 in 2025. In een sector met vrijwel vaste kosten per
scheepsdag valt zo'n tariefstijging vrijwel volledig door naar de winst. De keerzijde
is nu zichtbaar: over de laatste twaalf maanden liep de EBIT terug van USD 1.285 mln naar
USD 925 mln, doordat de bunkerprijs in het tweede kwartaal van 2026 met 48% opliep tot
USD 770 per ton en die kosten pas met vertraging via toeslagen worden doorberekend.
- **Omzet-CAGR:** 7,3% over 2017–2025. Vanaf het eerste volledige jaar na de fusie
  (2018–2025) is de CAGR 3,7%.

### Kasstromen (USD mln)

| Jaar | CFO | Capex | FCF | FCF na lease | FCF/aandeel | FCF-marge % | FCF-groei % | FCF-conversie % | SBC | Dividend totaal | Aandeleninkoop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | — | — | — | — | — | — | — | — | — | — | — |
| 2017 | 462 | — | — | — | — | — | — | — | — | 0 | 0 |
| 2018 | 272 | — | — | — | — | — | — | — | — | 0 | 0 |
| 2019 | 749 | −133 | 616 | — | 1,46 | 15,8 | — | 604 | — | 51 | 0 |
| 2020 | 615 | −135 | 480 | 300 | 1,14 | 16,2 | −22,0 | n.v.t. | — | 0 | 0 |
| 2021 | 623 | −141 | 483 | 279 | 1,14 | 12,4 | 0,5 | 273 | — | 0 | 0 |
| 2022 | 1.297 | −112 | 1.185 | 833 | 2,81 | 23,5 | 145,5 | 149 | — | 63 | 0 |
| 2023 | 1.771 | −163 | 1.608 | 1.289 | 3,80 | 31,2 | 35,7 | 165 | — | 362 | 4 |
| 2024 | 1.779 | −195 | 1.584 | 1.257 | 3,75 | 29,8 | −1,5 | 149 | — | 738 | 0 |
| 2025 | 1.744 | −245 | 1.499 | 1.138 | 3,54 | 28,6 | −5,4 | 136 | — | 989 | 0 |
| TTM | 1.424 | −339 | 1.085 | 711 | 2,57 | 21,1 | −27,6 | 141 | — | 892 | 0 |

*Aandelencompensatie (SBC) wordt door Wallenius Wilhelmsen niet als afzonderlijke regel
in de kasstroom of de factsheet gerapporteerd. Uit het beloningsrapport 2025 blijkt dat de
langetermijnbeloning van de directie voor 50% in aandelen en 50% in contanten wordt
afgewikkeld en dat de totale beloning van het volledige directieteam USD 9,6 mln bedroeg —
minder dan 0,15% van de marktkapitalisatie. De post is daarmee verwaarloosbaar en is als
onbekend leeg gelaten in plaats van geschat. "FCF na SBC" is om die reden niet als aparte
kolom opgenomen; "FCF na lease" is voor dit bedrijf de veel relevantere maatstaf.*

- **Toelichting kasstromen:**
Twee dingen springen eruit. Het eerste is de kolom "FCF na lease". Wallenius Wilhelmsen
huurt 38 van zijn 129 schepen in, plus terminals en vastgoed; onder IFRS 16 staan die
huurbetalingen niet in de operationele maar in de financieringskasstroom, zodat de
gebruikelijke berekening CFO minus capex een terugkerende kasuitgave van USD 374 mln
(laatste twaalf maanden) volledig buiten beeld laat. Wie die aftrekt, ziet dat de vrije
kasstroom niet USD 1.085 mln maar USD 711 mln bedraagt — een derde minder. Het tweede is
de capex. Die lag tussen 2020 en 2023 op USD 112 tot 163 mln per jaar, wat voor een
vloot van 91 eigen schepen buitengewoon weinig is, en loopt nu snel op: USD 245 mln in
2025 en USD 339 mln over de laatste twaalf maanden, met nog USD 1,4 mrd te gaan op het
nieuwbouwprogramma. De vrije kasstroom van de afgelopen vier jaar is dus deels geleend
van de toekomst. De daling van de operationele kasstroom van USD 1.744 mln naar
USD 1.424 mln over twaalf maanden — meer dan 15% en daarmee toelichtingsplichtig — komt
voor het grootste deel door de bunkerprijssprong in het tweede kwartaal van 2026 en door
brandstofvoorraden aan boord die tijdelijk werkkapitaal vastlegden; het management noemt
dat effect tijdelijk en de kasconversie daalde daardoor van 96% naar 78%.

### Balans-ratio's

| Jaar | Nettoschuld | Nettoschuld/EBITDA | Eigen vermogen | ROE % | ROIC % | ROCE % | Current ratio | Solvabiliteit % | Goodwill+imm. (USD mln) | Bruto schuld |
|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | — | — | — | — | — | — | — | — | — | — |
| 2017 | 2.968 | 5,66 | 2.796 | 5,2 | 4,3 | — | 1,0 | 35,8 | — | 3.764 |
| 2018 | 3.100 | 5,16 | 2.875 | 2,0 | 3,0 | — | 1,1 | 38,8 | — | 3.584 |
| 2019 | 3.646 | 4,53 | 2.921 | 3,5 | 5,0 | — | 1,0 | 37,5 | — | 4.044 |
| 2020 | 3.427 | 7,24 | 2.615 | −11,5 | −1,3 | −1,3 | 1,1 | 34,3 | 571 | 4.081 |
| 2021 | 3.418 | 4,12 | 2.804 | 6,3 | 4,4 | 4,5 | 1,1 | 36,0 | 455 | 4.128 |
| 2022 | 2.872 | 1,86 | 3.508 | 22,6 | 14,0 | 12,9 | 1,8 | 41,8 | 395 | 4.088 |
| 2023 | 2.008 | 1,11 | 3.080 | 31,6 | 19,2 | 16,1 | 1,4 | 36,0 | 360 | 3.713 |
| 2024 | 1.758 | 0,94 | 3.321 | 32,1 | 20,4 | 19,9 | 1,1 | 39,5 | 319 | 3.151 |
| 2025 | 1.729 | 0,96 | 3.302 | 33,4 | 20,9 | 18,4 | 0,9 | 42,2 | 241 | 2.800 |
| TTM | 2.011 | 1,26 | 3.669 | 21,0 | 14,5 | 15,6 | 1,2 | 48,8 | 236 | 2.638 |

*Nettoschuld en bruto schuld zijn hier inclusief leaseverplichtingen, zoals Wallenius
Wilhelmsen zelf rapporteert. ROIC is berekend als NOPAT (EBIT × (1 − effectief kastarief))
gedeeld door eigen vermogen plus nettoschuld plus de EUKOR-putverplichting. Merk op: in de
DCF hanteer ik een ándere, expliciet toegelichte schulddefinitie (zie sectie 12).*

- **Toelichting balans:**
De schuldafbouw is de meest indrukwekkende prestatie van de afgelopen vier jaar. De
nettoschuld daalde van USD 3.418 mln eind 2021 naar USD 1.729 mln eind 2025 en de
schuldgraad van 4,1 keer de EBITDA naar 0,96 keer, ruim onder de eigen doelstelling van
3,0. Tegelijk moet de bruto schuld apart worden gelezen: die daalde van USD 4.128 mln
naar USD 2.800 mln, dus de verbetering komt echt van schuldaflossing en niet alleen van
opgepotte kas. Sterker nog, de kaspositie liep in dezelfde periode terug van USD 1.705 mln
(2023) naar USD 627 mln medio 2026 doordat er sinds begin 2023 ruim USD 2,8 mrd aan dividend uitging (inclusief uitkeringen aan minderheidsbelangen).
Twee posten verdienen aandacht. Ten eerste de leaseverplichting van USD 1.490 mln, meer
dan de helft van de bruto schuld — dat is de ingehuurde vloot. Ten tweede de geschreven
putoptie op het 20%-minderheidsbelang in EUKOR: die stond eind 2025 nog op USD 897 mln en
werd in het tweede kwartaal van 2026 herwaardeerd naar USD 386 mln omdat de optie pas
vanaf 1 januari 2031 uitgeoefend kan worden. Die ene boeking tilde het eigen vermogen van
USD 3.064 mln naar USD 3.669 mln en de solvabiliteit van 39% naar 48,8%. Het is een
waarderingseffect, geen operationele verbetering, en de verplichting bestaat nog steeds.
Om diezelfde reden zijn de eigen-vermogencijfers van 2023 en 2024 hier de geherformuleerde
versie uit de factsheet Q2 2026 en niet wat destijds in de jaarverslagen 2023 en 2024 stond
(daar stond het eigen vermogen 2023 op USD 4.056 mln, inclusief het minderheidsbelang dat
later naar een financiële verplichting is geherrubriceerd).

### Kapitaalstructuur huidig (per 30 juni 2026, USD mln)
- **Nettoschuld (huidig, incl. lease):** 2.011
- **Nettoschuld (excl. lease, incl. EUKOR-put):** 906
- **Bruto schuld (incl. lease):** 2.638
- **Bruto schuld (excl. lease):** 1.148
- **Cash & equivalents:** 627
- **Lease-verplichtingen (IFRS-16):** 1.490 (1.097 langlopend, 393 kortlopend)
- **Geschreven putoptie EUKOR-minderheidsbelang:** 386
- **Beschikbare liquiditeit (kas + ongetrokken faciliteiten):** 1.179 (doelstelling minimaal 1.000)
- **Gemiddelde rente %:** 5,6 (betaalde rente laatste twaalf maanden USD 153 mln op gemiddeld USD 2,7 mrd bruto schuld)
- **Rente-dekking (EBIT/betaalde rente):** 6,1× (TTM EBIT 925 / betaalde rente 153)

**Schuldvervalprofiel (USD mln, incl. leases):** 2026: 280 · 2027: 577 · 2028: 483 ·
2029: 329 · 2030: 235 · 2031 en later: 738. Er staat niets getrokken onder de
kredietfaciliteiten. Van de obligaties vervalt USD 126 mln in 2027 en USD 101 mln in 2028.
Herfinancieringsrisico is beperkt: 43 schepen zijn onbezwaard en vertegenwoordigen een
groot deel van de door makelaars getaxeerde vlootwaarde van USD 5,0 mrd.

### Non-GAAP / aanpassingen
- **Gebruikt?** true (naast IFRS ook "aangepaste EBITDA")
- **Welke aanpassingen:** het management rapporteert *adjusted EBITDA* waarin eenmalige
  posten worden geëlimineerd. De grootste in de reeks: de boekwinst van USD 135 mln op
  de verkoop van de MIRRAT-terminal (Q2 2025, uit adj. EBITDA gehaald), een
  bijzondere waardevermindering van USD 76 mln (Q4 2021), USD 40 mln (Q1 2020) en
  USD 29 mln goodwill in logistiek (Q4 2022), en herstructureringsposten in 2020.
- **Waarom:** vergelijkbaarheid tussen jaren en aansluiting op de guidance, die het
  management in aangepaste EBITDA uitdrukt. Het verschil tussen IFRS-EBITDA en aangepaste
  EBITDA is in de meeste jaren klein: 0,6% in 2025, 1,7% in 2024, 0% in 2023. Alleen 2020
  (13,2%) en 2021 (4,3%) wijken materieel af. **Voor de DCF is IFRS de grondslag**; de
  aangepaste EBITDA is alleen gebruikt om het genormaliseerde middencyclusniveau te
  bepalen, omdat eenmalige boekwinsten daar niet in thuishoren.

### Earnings quality

| Jaar | Accruals ratio | Non-GAAP verschil % | Betaalde belasting (USD mln) | Belastinglast (USD mln) |
|---|---|---|---|---|
| 2020 | −0,120 | 13,2 | 9 | −4 |
| 2021 | −0,057 | 4,3 | 24 | 23 |
| 2022 | −0,060 | −1,3 | 35 | 35 |
| 2023 | −0,093 | 0,0 | 39 | 68 |
| 2024 | −0,085 | 1,7 | 84 | 73 |
| 2025 | −0,082 | 0,6 | 53 | 42 |
| TTM | −0,087 | 1,8 | 45 | 47 |

- **Toelichting earnings quality:**
De accruals-ratio is in elk jaar fors negatief, wat betekent dat de operationele kasstroom
structureel hoger is dan de nettowinst — het conservatieve uiteinde van het spectrum en
precies wat je verwacht bij een kapitaalintensieve reder waar de afschrijving op schepen
de winst drukt zonder kas te kosten. Er is geen enkel signaal van winststuring. Het
verschil tussen IFRS- en aangepaste winst is in de laatste vier jaar verwaarloosbaar
(0 tot 1,7%), op één uitzondering na: in 2025 zit een boekwinst van USD 135 mln op de
verkoop van een terminal in de IFRS-cijfers die het management er terecht uithaalt. De
**kasbelastingcontrole uit METHODE regel 5** geeft geen aanleiding tot correctie: over
2020–TTM bedroeg de betaalde belasting samen USD 289 mln tegen een gecumuleerde
belastinglast van USD 284 mln, een verschil van minder dan 2%. Dat komt doordat het
grootste deel van de vloot onder tonnagebelastingregimes valt, waar de heffing op
scheepsruimte en niet op winst is gebaseerd; het effectieve tarief ligt daardoor
structureel tussen 3,7% en 6,5% en is niet een uitgestelde-belastingvakantie die ooit
terugkomt.

### Rendementsindicatoren en ROIC–WACC spread

| Jaar | ROE % | ROIC % | ROA % | Asset turnover | WACC % (schatting) | Spread (pp) | Oordeel |
|---|---|---|---|---|---|---|---|
| 2017 | 5,2 | 4,3 | 1,9 | 0,38 | 8,4 | −4,1 | waardevernietigend |
| 2018 | 2,0 | 3,0 | 0,8 | 0,55 | 8,4 | −5,4 | waardevernietigend |
| 2019 | 3,5 | 5,0 | 1,3 | 0,50 | 8,4 | −3,4 | waardevernietigend |
| 2020 | −11,5 | −1,3 | −4,0 | 0,39 | 8,4 | −9,7 | waardevernietigend |
| 2021 | 6,3 | 4,4 | 2,3 | 0,50 | 8,4 | −4,0 | waardevernietigend |
| 2022 | 22,6 | 14,0 | 9,5 | 0,60 | 8,4 | +5,6 | waardecreërend |
| 2023 | 31,6 | 19,2 | 11,4 | 0,60 | 8,4 | +10,8 | waardecreërend |
| 2024 | 32,1 | 20,4 | 12,7 | 0,63 | 8,4 | +12,0 | waardecreërend |
| 2025 | 33,4 | 20,9 | 14,1 | 0,67 | 8,4 | +12,5 | waardecreërend |
| TTM | 21,0 | 14,5 | 10,2 | 0,69 | 8,4 | +6,1 | waardecreërend |

*De WACC is over de hele reeks op het actuele niveau van 8,41% gehouden. Dat is een
vereenvoudiging — de rente lag in 2020–2021 lager — maar het maakt de spread over de jaren
vergelijkbaar en voorkomt dat een dalende rente waardecreatie suggereert waar die er niet was.*

- **Toelichting rendement:**
Dit is de belangrijkste tabel van de hele analyse. Wallenius Wilhelmsen verdiende in vijf
van de negen gemeten jaren minder op zijn geïnvesteerde kapitaal dan dat kapitaal kost. De
gemiddelde ROIC over de volledige cyclus 2017–TTM bedraagt 10,4% tegen een
kapitaalkostenvoet van 8,4% — een marge van amper twee procentpunt, en die volledig te
danken aan de vier uitzonderlijke jaren 2022 tot en met 2025. Wie alleen naar de laatste
vier jaar kijkt ziet een prachtig bedrijf met 19% ROIC; wie de cyclus meeneemt ziet een
kapitaalintensieve vervoerder die over de rit heen net iets meer verdient dan zijn
financiers eisen. Dat onderscheid bepaalt de waardering en het is de reden dat ik de
Buffett-toets op "gedeeltelijk" laat uitkomen in plaats van op "voldoet". De teruggang in
de laatste twaalf maanden van 20,9% naar 14,5% laat bovendien zien hoe snel de spread
verdampt zodra één kostenpost — hier brandstof — tegenzit.

### Waarderingsratio's (peildatum 2026-08-20, koers NOK 159,40)

| Maatstaf | Waarde | Toelichting |
|---|---|---|
| P/E (TTM) | 10,4 | op TTM-EPS van USD 1,648 |
| P/E (boekjaar 2025) | 7,1 | op EPS USD 2,406 |
| Forward P/E 2026 | 10,9 | op consensus-EPS USD 1,573 |
| Forward P/E 2028 | 15,3 | op consensus-EPS USD 1,115 |
| P/B | 1,97 | boekwaarde per aandeel USD 8,66 |
| P/FCF (TTM) | 6,7 | op FCF USD 1.085 mln |
| P/FCF na lease (TTM) | 10,2 | op FCF na lease USD 711 mln |
| FCF-rendement na lease | 9,8% | |
| EV/EBITDA (TTM, incl. lease in EV) | 5,9 | EV USD 9.627 mln |
| EV/EBITDA (genormaliseerd) | 6,6 | op mid-cyclus adj. EBITDA USD 1.450 mln |
| EV/Omzet | 1,87 | |
| Dividendrendement (laatste 12 mnd gedeclareerd) | 9,5% | USD 1,62 = NOK 15,11 |
| Dividendrendement (consensus 2026) | 6,7% | USD 1,151 = NOK 10,74 |
| PEG | n.v.t. | consensus verwacht dalende winst t/m 2028 |

- **Toelichting waardering:**
Op het eerste gezicht is dit een goedkoop aandeel: zeven keer de winst van 2025, bijna
tien procent dividendrendement en zes keer de EBITDA. Maar dat is precies de val die Peter
Lynch beschrijft bij cyclische bedrijven — een lage koers-winstverhouding op piekwinst is
een waarschuwing, geen koopje. Op de winst van de laatste twaalf maanden staat de
verhouding al op 10,4, op de consensusverwachting voor 2028 op 15,3. Ten opzichte van de
directe concurrent Höegh Autoliners (koers-winstverhouding 6,3 in juni 2026, en met een
EBIT-marge van 34,4% tegen 24,5% bij Wallenius Wilhelmsen) is er geen korting. Een
historisch tienjaarsgemiddelde van de multiples kan ik niet geven: ik heb geen
geverifieerde koersreeks van tien jaar en heb die liever leeg gelaten dan gereconstrueerd.

### Dividendanalyse

| Jaar | DPS (USD, uitbetaald) | Groei YoY % | Uitkeringsratio winst % | FCF-dekkingsratio | FCF-dekking ná lease | Bijzonderheden |
|---|---|---|---|---|---|---|
| 2018 | 0,00 | — | 0 | — | — | USD 0,06 voorgesteld over 2018, uitbetaald in 2019 |
| 2019 | 0,12 | — | 55 | 12,1× | — | eerste dividend van de gefuseerde groep |
| 2020 | 0,00 | −100 | 0 | — | — | geschrapt; uitkering contractueel geblokkeerd door uitgestelde bankleningen |
| 2021 | 0,00 | — | 0 | — | — | USD 0,15 voorgesteld, uitbetaald in 2022 |
| 2022 | 0,15 | — | 9 | 18,8× | 13,2× | hervatting na corona |
| 2023 | 0,86 | 473 | 43 | 4,4× | 3,6× | |
| 2024 | 1,75 | 104 | 76 | 2,1× | 1,7× | inclusief extra dividend |
| 2025 | 2,34 | 34 | 97 | 1,5× | 1,2× | recordbedrag USD 989 mln uitbetaald |
| TTM | 2,11 | −10 | 128 | 1,2× | **0,80×** | uitkering overtreft vrije kasstroom na lease |

- **Dividendsoorten:** halfjaarlijks regulier dividend van 30 tot 50% van de nettowinst,
  aangevuld met extra dividenden wanneer de balans dat toelaat. Het dividend over de
  eerste helft van 2026 van USD 0,61 bestaat uit USD 0,34 regulier (50% van de winst) plus
  een extra uitkering van USD 100 mln (circa USD 0,24 per aandeel). Er is geen
  stockdividend. Uitbetaling gebeurt in NOK, declaratie in USD — het valutarisico ligt dus
  bij de aandeelhouder.
- **Dividendbeleid:** 30–50% van de jaarwinst na belasting, halfjaarlijks, met ruimte voor
  extra uitkeringen. Het bestuur weegt daarbij de minimumliquiditeit van USD 1 mrd, de
  schuldgraad (<3,0×) en de solvabiliteit (>35%). Het is uitdrukkelijk **geen** progressief
  beleid: in 2020 werd het dividend zonder omhaal geschrapt.
- **Dividendrendement:** 9,5% op de laatste twaalf maanden gedeclareerd dividend (USD 1,62),
  12,4% op wat er de laatste twaalf maanden feitelijk is uitbetaald (USD 2,11), en 6,7% op
  de consensusverwachting voor heel 2026 (USD 1,151). Ter vergelijking: de Noorse
  tienjaarsstaatslening en de Amerikaanse tienjaars (4,71%) liggen ruim daaronder, maar
  Höegh Autoliners noteerde in juni 2026 op 14,6% rendement — hoge rendementen zijn in
  deze sector de norm en weerspiegelen dat de markt de houdbaarheid betwijfelt.
  Een vijf- en tienjaarsgemiddelde van het rendement kan ik zonder geverifieerde
  koershistorie niet geven en is weggelaten.
- **Dividendgroei versus inflatie:** de samengestelde groei van USD 0,12 (2019) naar
  USD 2,34 (2025) komt op 64% per jaar uit, maar dat cijfer is misleidend omdat het start
  in een dieptepunt en eindigt in een piek. Betekenisvoller is dat het dividend in 2020 en
  2021 volledig wegviel — dit is geen dividendaristocraat maar een cyclische uitkering.
- **Eerstvolgende dividendbesluit:** ex-dividend 25 augustus 2026, betaalbaar 16 september
  2026, USD 0,61 per aandeel over de eerste helft van 2026 (tegen USD 1,10 een jaar eerder,
  een daling van 45%). Het volgende besluit valt bij de jaarcijfers op 11 februari 2027.
- **Eindoordeel dividend:** gespannen. Over de laatste twaalf maanden keerde Wallenius
  Wilhelmsen USD 892 mln uit terwijl de vrije kasstroom ná leasebetalingen USD 711 mln
  bedroeg — een dekkingsgraad van 0,80 keer, oftewel het dividend werd deels uit de kas
  betaald in plaats van uit de kasstroom, wat ook zichtbaar is in de kaspositie die van
  USD 1.071 mln naar USD 627 mln zakte. Dat is een bewuste keuze bij een lage schuldgraad
  en geen noodsignaal, maar het is niet vol te houden zodra de USD 1,4 mrd aan
  nieuwbouwtermijnen op tafel komt. Het al 45% verlaagde halfjaardividend is daarvan het
  eerste bewijs. Het dividend draagt bij aan de these voor een inkomensbelegger die een
  cyclische uitkering accepteert, maar wie een stabiele stroom zoekt is hier verkeerd.

### Sector-specifieke KPI's (transport & logistiek)

| KPI | Eenheid | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | TTM |
|---|---|---|---|---|---|---|---|---|
| Volume (na eliminatie) | mln cbm | 47,1 | 56,8 | 59,0 | 59,2 | 54,2 | 53,7 | 54,1 |
| Netto vrachttarief | USD/cbm | 41,35 | 48,23 | 55,73 | 55,41 | 61,94 | 64,61 | 62,95 |
| TC-resultaat per dag | USD 1.000 | 20,4 | 26,4 | 42,7 | 47,7 | 49,4 | 49,4 | 45,4 |
| Vlootkosten per dag | USD | 6.417 | 7.043 | 7.283 | 7.673 | 7.787 | 8.182 | 8.211 |
| Bunkerprijs | USD/ton | 401 | 501 | 789 | 629 | 634 | 557 | 589 |
| Aantal schepen (scheepvaart) | stuks | 129 | 122 | 119 | 116 | 114 | 116 | 119 |
| Aandeel High & Heavy + breakbulk | % volume | 30,2 | 31,2 | 30,5 | 27,3 | 23,4 | 22,8 | 23,9 |
| Orderportefeuille scheepvaart | USD mrd | — | — | — | — | — | 7,4 | 6,5 |

- **Toelichting sector-KPI's:**
De motor van de winstexplosie is niet volume maar prijs: het volume ligt in 2025 met
53,7 mln cbm nog altijd 9% onder het niveau van 2023, terwijl het netto vrachttarief in
diezelfde twee jaar met 17% steeg en het tijdcharterresultaat per scheepsdag ruim
verdubbelde ten opzichte van 2020. Twee KPI's kleuren dat beeld inmiddels donkerder. De
orderportefeuille van de scheepvaartdivisie liep in vier kwartalen terug van USD 8,7 mrd
(Q2 2025) naar USD 6,5 mrd, en het tijdcharterresultaat per dag daalde van een piek van
USD 52.100 in het vierde kwartaal van 2022 naar USD 41.100 in Q2 2026. Tegelijk stijgen de vlootkosten per
scheepsdag onafgebroken, van USD 6.204 in Q1 2020 naar USD 8.142 nu. Het aandeel
hoogwaardige High & Heavy-lading — machines waarvoor per kubieke meter meer wordt betaald
— kromp van 30% naar 24% van het volume, wat de mix ongunstiger maakt dan het gemiddelde
tarief suggereert.

---

## 4. Moat (concurrentievoordeel)

- **Oordeel:** **NARROW MOAT**
- **Moat-categorieën:**

| Naam | Sterkte | Toelichting |
|---|---|---|
| Immateriële activa | sterk | De echte immateriële bezitting is niet een merk of een patent maar de wettelijke afscherming van de Amerikaanse overheidsdivisie: onder de Jones Act en de daaraan gekoppelde regels mag militair materieel alleen door schepen onder Amerikaanse vlag met Amerikaanse bemanning worden vervoerd. Dat segment leverde in 2025 USD 153 mln aangepaste EBITDA op USD 411 mln omzet — een marge van 37%, de hoogste van de drie divisies — en buitenlandse concurrenten kunnen er per definitie niet in. Daarbuiten is er geen noemenswaardige merkkracht: autofabrikanten kopen vervoer op prijs en betrouwbaarheid. |
| Overstapkosten | middel | Overstappen kost een fabrikant geen geld maar wel tijd en risico. De logistieke contracten hebben een gewogen resterende looptijd van 8,1 jaar, omdat een verwerkingscentrum naast de haven fysiek is ingericht op het model en het accessoirepakket van één fabrikant. Voor het zeevervoer zelf zijn de overstapkosten veel lager: de contracten lopen gemiddeld 2,9 jaar en concurrenten bieden bij elke vernieuwing mee. |
| Netwerkeffecten | zwak | Er is geen echt netwerkeffect. Een extra klant maakt de dienst niet waardevoller voor bestaande klanten. Wat er wél is, is een dichtheidsvoordeel: vijftien vaste routes met hoge frequentie zijn alleen rendabel bij voldoende volume, en meer volume maakt hogere frequentie mogelijk. Dat is schaal, geen netwerk. |
| Kostenvoordeel | middel | Met bijna 900.000 autoplaatsen is Wallenius Wilhelmsen ongeveer 15 tot 17% van de wereldwijde autoschipvloot en daarmee de grootste onafhankelijke speler. Dat levert inkoopmacht op brandstof, spreiding van dokkosten en de mogelijkheid retourlading te vinden — de vlootkosten per scheepsdag van USD 8.142 zijn beheerst gebleven. Maar het is geen structureel kostenvoordeel dat een nieuwkomer niet kan kopen: wie vandaag schepen bestelt bij een Chinese werf krijgt zuiniger, grotere en dus goedkopere tonnage dan de gemiddeld 18 jaar oude vloot van Wallenius Wilhelmsen. |
| Efficiënte schaal | middel | De markt is geconcentreerd en de toetredingsdrempel is kapitaal: een modern schip van 9.300 autoplaatsen kost ruim USD 130 mln en werven zijn tot ver in het decennium volgeboekt. Dat beperkt het aantal spelers. De keerzijde is dat de bestaande spelers zelf massaal bestellen — het orderboek is ongeveer 21% van de vloot — waardoor de efficiënte schaal wordt opgerekt door de sector zelf en niet door nieuwkomers. |

- **Kwantitatief bewijs:** de ROIC-reeks is de scherpste toets en die valt gemengd uit.
  Over 2017–TTM gemiddeld 10,4% tegen een kapitaalkostenvoet van 8,4%; vijf van de negen
  jaren onder de kapitaalkosten; de laatste vier jaar 14% tot 21%. Marges zijn evenmin
  stabiel: de EBIT-marge liep van −2,8% (2020) naar 24,5% (2025) en alweer terug naar 18,0%
  over de laatste twaalf maanden. Het marktaandeel is stabiel maar niet groeiend: de eigen
  scheepscapaciteit ging van 778.000 autoplaatsen begin 2020 naar 817.000 medio 2026,
  ongeveer 5% groei, terwijl de wereldvloot in dezelfde periode sneller uitbreidde. Een
  bedrijf met een brede slotgracht laat geen vijf verliesjaren op kapitaalbasis zien.
- **Duurzaamheid:** vijf jaar voor de logistieke contracten en de overheidsdivisie, minder
  dan vijf jaar voor het zeevervoer. De contractportefeuille van 2,9 jaar is precies de
  horizon waarop de huidige tarieven zijn vastgelegd; daarna geldt de markt van dat moment.
- **Erosierisico's:** het orderboek van circa 21% van de wereldvloot is de duidelijkste
  bedreiging, met opleveringen die vanaf 2028 landen. Daarnaast bouwen Chinese
  autofabrikanten hun eigen vloten — BYD, Chery en SAIC hebben schepen besteld en varen
  inmiddels zelf — wat precies de groeimarkt aantast die de afgelopen jaren de winst droeg.
  Ook Hyundai Glovis, de logistieke arm van de grootste EUKOR-klant, breidt uit. Ten slotte
  is de eigen vloot met gemiddeld 18,3 jaar oud tegenover nieuwbouw die 30% zuiniger vaart:
  onder een aanscherpend CO2-regime wordt leeftijd zelf een kostennadeel.

---

## 5. Management

- **CEO-naam + tenure:** Lasse Kristoffersen, sinds juni 2022 (4 jaar). Daarvoor vijftien
  jaar bij Torvald Klaveness, waarvan elf als CEO, en een decennium bij DNV. Was president
  van de Noorse Redersvereniging en zit in de raad van de International Chamber of Shipping.
  Bezat per 31 december 2025 33.100 aandelen (0,008% van het kapitaal); die zijn in
  september 2025 overgedragen aan een volledig eigen holdingvennootschap.
- **CFO-naam + tenure:** Bjørnar Bukholm, sinds 25 april 2025 (ruim één jaar). Begon bij
  McKinsey & Company, werkte acht jaar bij Wallenius Wilhelmsen in bedrijfsontwikkeling,
  financiën en investor relations, en was daartussen groepsfinancieel directeur bij Sector
  Alarm. Hield eind 2025 geen aandelen. Zijn voorganger Jermund Lien kocht in november 2024
  nog 2.000 aandelen tegen NOK 106,10.
- **Bestuursvoorzitter:** Rune Bjerke, sinds 2020; oud-CEO van DNB ASA, Hafslund en
  Scancem International.
- **Oprichter nog betrokken?** Ja, in de raad van commissarissen. Thomas Wilhelmsen is
  bestuurder en houdt direct en indirect circa 161,4 miljoen aandelen (38,1%) via Wilh.
  Wilhelmsen Holding; de Wallenius-zijde houdt via Soya Group circa 160 miljoen aandelen
  (37,8%).
- **Insider ownership %:** 75,7% gezamenlijk in handen van de twee oprichtersfamilies; de
  uitvoerende directie zelf houdt minder dan 0,05%.

### Capital allocation track record (USD mln)

| Jaar | Dividend totaal | Aandeleninkoop | Schuldaflossing (netto) | Capex |
|---|---|---|---|---|
| 2020 | 0 | 0 | 74 | 135 |
| 2021 | 0 | 0 | 152 | 141 |
| 2022 | 63 | 0 | −41 | 112 |
| 2023 | 362 | 4 | 116 | 163 |
| 2024 | 738 | 0 | 480 | 195 |
| 2025 | 989 | 0 | 569 | 245 |
| TTM | 892 | 0 | 360 | 339 |

- **M&A-track-record:** Wallenius Wilhelmsen heeft in de onderzochte periode nauwelijks
  overgenomen en vooral opgeruimd. De grootste transacties zijn desinvesteringen: de
  verkoop van de terminal MIRRAT in het tweede kwartaal van 2025 met een boekwinst van
  USD 135 mln en een opbrengst van USD 179 mln. De goodwill en overige immateriële activa
  daalden van USD 602 mln (Q1 2020) naar USD 236 mln medio 2026, deels door afschrijving
  en deels door de bijzondere waardevermindering van USD 29 mln op de logistieke goodwill
  in 2022 — dat laatste is de enige duidelijke waardevernietiging in de reeks. Er zijn geen
  grote acquisities gedaan op de top van de cyclus, wat in deze sector een compliment is.
- **Beloning:** de CEO ontving over 2025 USD 1.721.000, opgebouwd uit USD 797.000 vast
  salaris, USD 352.000 jaarbonus en USD 413.000 langetermijnbeloning. Ten opzichte van het
  gemiddelde basissalaris in het bedrijf van USD 68.100 is dat een verhouding van ongeveer
  25 keer — laag naar internationale maatstaven. De jaarbonus bedraagt maximaal 40 tot 50%
  van het basissalaris en hangt voor 50% aan financiële doelen (EBITDA en rendement op
  geïnvesteerd kapitaal), 20% aan klant- en medewerkerstevredenheid, 10% aan veiligheid,
  10% aan klimaat en 10% aan individuele prestaties; over 2025 werd 80% van de doelstelling
  gehaald. De langetermijnbeloning is een prestatie-aandelenprogramma met drie jaar
  volledige vesting, waarvan 60% is gekoppeld aan het rendement op geïnvesteerd kapitaal,
  en wordt voor de helft in aandelen en de helft in contanten afgewikkeld, met een plafond
  op twee keer de toekenning. Voor een kapitaalintensieve reder is een beloning die voor
  het leeuwendeel aan ROCE hangt precies de juiste prikkel; het manco is dat de directie
  zelf vrijwel geen aandelen bezit en dat de helft van de langetermijnbeloning in contanten
  wordt uitgekeerd, wat de eigen blootstelling aan de koers beperkt.
- **Insider transactions (laatste 24 maanden):**

| Datum | Persoon | Functie | Type | Aantal | Koers (NOK) |
|---|---|---|---|---|---|
| 2025-09-19 | Lasse Kristoffersen | CEO | Overdracht naar eigen holding | 33.100 | n.v.t. |
| 2025-05-13 | Carl Magnus Groth | Bestuurder | Toekenning/aankoop | 13.000 | — |
| 2025-05 | Line Merethe Hestvik | Bestuurder | Toekenning/aankoop | 4.000 | — |
| 2024-11 | Jermund Lien | CFO (toenmalig) | KOOP | 2.000 | 106,10 |
| 2024-11 | Anders Karlsen | Investor Relations Manager | KOOP | 1.500 | 112,00 |
| 2024-10 | Rune Bjerke | Bestuursvoorzitter | KOOP | 5.000 | 108,58 |
| 2024-10 | Kerstin Margareta Alestig Johnson | Bestuurder | KOOP | 1.000 | 108,50 |
| 2024-05 | Erik Nøklebye | Directielid | VERKOOP | 10.102 | 112,00 |

- **Insider netto:** NETTO KOPER, maar de bedragen zijn klein. Alle open-markt aankopen
  vonden plaats tussen NOK 106 en NOK 112, ruim onder de huidige koers van NOK 159,40;
  in 2025 en 2026 zijn geen open-markt aankopen door de directie gemeld. De overdracht
  door de CEO in september 2025 is een herstructurering naar een eigen vennootschap, geen
  economische transactie. Het signaal is dus zwak: insiders kochten toen het aandeel
  goedkoop was en hebben op het huidige niveau niet bijgekocht.
- **Oordeel management:** **STERK**
- **Toelichting:** Het bestuur heeft in vier jaar gedaan wat je van een cyclisch bedrijf op
  de top wilt zien. De nettoschuld ging van USD 3,4 mrd naar USD 1,7 mrd, de schuldgraad
  van 4,1 naar onder 1,0, er werd niet overgenomen op de piek maar verkocht (MIRRAT, met
  boekwinst), en de overtollige kas ging naar aandeelhouders in plaats van naar prestige.
  De beloningsstructuur hangt met 60% van de langetermijncomponent aan het rendement op
  geïnvesteerd kapitaal, wat precies de maatstaf is die in deze sector telt. Ook de
  transparantie over tegenvallers is in orde: in mei 2026 verlaagde het management de
  guidance voor 2026 openlijk van USD 1,65–1,75 mrd naar circa USD 1,6 mrd en benoemde
  daarbij expliciet de oorzaken — het Midden-Oostenconflict, hogere bunkerkosten, een
  krappe chartermarkt en een schip dat vastzat in de Straat van Hormuz. Twee kanttekeningen
  blijven staan. De eerste is historisch: WWL en EUKOR werden in februari 2018 door de
  Europese Commissie beboet als onderdeel van een kartel van zes autoschipreders met een
  totale boete van EUR 207 mln, met vergelijkbare schikkingen in de Verenigde Staten en
  Australië. Dat speelde vóór de huidige directie en vóór de huidige groepsstructuur, en er
  is sindsdien geen herhaling, maar het hoort genoemd te worden. De tweede is dat de
  uitvoerende directie zelf nauwelijks aandelen bezit — de daadwerkelijke alignement komt
  van de twee families in de raad, niet van het management.

---

## 6. Sector & concurrentie

- **Sector-groeivooruitzicht:** de analistenconsensus voor Wallenius Wilhelmsen zelf gaat
  uit van kríimp: omzet van USD 5.327 mln (2026) naar USD 4.989 mln (2028) en EBITDA van
  USD 1.564 mln naar USD 1.287 mln. Voor de onderliggende markt is het beeld tweeledig.
  Aan de vraagzijde groeide de Chinese voertuigexport in de twaalf maanden tot medio 2026
  met 72% tot 8,5 miljoen eenheden en verwacht het management dat 2026 boven de tien
  miljoen uitkomt. Aan de aanbodzijde staat het orderboek voor autoschepen op ongeveer 21%
  van de wereldvloot, met opleveringen die grotendeels vanaf 2028 landen. Een
  onafhankelijke, betrouwbaar gedateerde raming van de sectorgroei in procenten per jaar
  heb ik niet gevonden die aan de bronvereisten voldoet; die is daarom weggelaten.

- **Porter five forces:**
  - **Rivaliteit: HOOG.** Ongeveer tien serieuze deep-sea spelers — Wallenius Wilhelmsen,
    Höegh Autoliners, NYK, MOL, K-Line, Grimaldi, Hyundai Glovis, Toyofuji, Zodiac en de
    nieuwe Chinese vloten — bedienen dezelfde fabrikanten met een grotendeels identiek
    product. Contracten worden bij elke vernieuwing opnieuw aanbesteed. Dat rivaliteit
    momenteel niet in prijsoorlog eindigt komt niet door discipline maar door schaarste;
    zodra het orderboek landt, keert de druk terug, precies zoals in 2016–2021.
  - **Nieuwe toetreders: MIDDEL.** Kapitaal is de drempel — een modern schip kost boven de
    USD 130 mln en werven zitten vol tot ver in het decennium — maar die drempel wordt
    geslecht door partijen met diepe zakken. Chinese autofabrikanten bouwen hun eigen
    vloten en dat is toetreding door de klant zelf, de vervelendste variant.
  - **Substituten: LAAG.** Een auto van Shanghai naar Rotterdam gaat over zee; spoor via
    Eurazië is marginaal en containervervoer van complete voertuigen is duurder en
    schadegevoeliger. Het echte substituut is geen ander transport maar géén transport:
    lokale productie in de afzetmarkt. Chinese fabrikanten die fabrieken bouwen in Hongarije,
    Spanje of Mexico halen structureel volume uit de vaart.
  - **Macht leveranciers: MIDDEL.** De grootste inkooppost is brandstof en die prijs is
    ongecontroleerd — in Q2 2026 steeg de bunkerprijs 48% in één kwartaal en kostte
    USD 31 mln EBITDA. Toeslagmechanismen vangen dat met vertraging op. Werven hebben
    momenteel de macht (volle orderboeken, gestegen nieuwbouwprijzen) en 38 ingehuurde
    schepen moeten tegen markttarieven worden verlengd in een chartermarkt die het
    management zelf "krap" noemt.
  - **Macht afnemers: HOOG.** De afnemers zijn enkele tientallen grote autoconcerns die
    per aanbesteding inkopen en het volume kunnen verdelen. Bij EUKOR is de grootste
    klant, Hyundai Motor Group, tegelijk 20%-aandeelhouder en heeft die een
    oceaanvervoercontract dat eind 2029 afloopt — een afnemer met die positie heeft veel
    onderhandelingsmacht.
  - **Conclusie Porter:** structureel een **onaantrekkelijke** sector voor duurzame
    bovengemiddelde winstgevendheid. Hoge rivaliteit, machtige afnemers en een aanbodzijde
    die op elke tariefpiek met nieuwbouw reageert. Dat de winstgevendheid nu uitzonderlijk
    is, komt door een tijdelijke krapte — omvaart om Kaap de Goede Hoop plus een
    exportgolf — en niet doordat de structuur veranderde. Precies dat verklaart waarom de
    ROIC in vijf van de negen gemeten jaren onder de kapitaalkosten lag.

- **Concurrenten:**

| Concurrent | Ticker | Omzetgroei % (2025) | EBIT-marge % (2025) | Nettowinst (USD mln, 2025) | P/E | Marktaandeel % |
|---|---|---|---|---|---|---|
| Wallenius Wilhelmsen | WAWI.OL | −1,3 | 24,5 | 1.104 | 10,4 (TTM) | ~15–17 (CEU) |
| Höegh Autoliners | HAUTO.OL | +4,0 | 34,4 | 513 | 6,3 (jun 2026) | — |
| NYK Line | — | — | — | — | — | — |
| Mitsui O.S.K. Lines | — | — | — | — | — | — |
| Hyundai Glovis | — | — | — | — | — | — |

*Voor de Japanse en Koreaanse concurrenten is het autoschipsegment onderdeel van veel
grotere conglomeraten en zijn segmentcijfers niet op een verifieerbare manier los te
krijgen; ROIC, EV/EBITDA, schuldgraad en marktaandeel zijn daarom leeg gelaten in plaats
van geschat. Het marktaandeel van Wallenius Wilhelmsen is berekend uit de eigen
scheepscapaciteit (897.800 autoplaatsen volgens de vlootlijst in de Factsheet Q2 2026)
gedeeld door de wereldvloot van autoschepen boven 2.000 autoplaatsen (5,2 tot 6,0 miljoen
autoplaatsen volgens de Q2 2026-presentatie).*

- **Positie van het bedrijf:** marktleider onder de onafhankelijke deep-sea autoschipreders,
  met naar schatting 15 tot 17% van de wereldwijde capaciteit en als enige grote speler een
  volledig geïntegreerde landzijde.
- **Positie-toelichting:** Wallenius Wilhelmsen is groter en breder dan Höegh Autoliners,
  maar Höegh verdient beter: 34,4% EBIT-marge tegen 24,5% in 2025, omdat Höegh puur
  scheepvaart is met een jongere vloot en geen logistieke divisie die op 12% EBITDA-marge
  draait. Toch noteert Höegh op een lagere koers-winstverhouding (6,3 in juni 2026 tegen
  10,4 voor Wallenius Wilhelmsen op TTM-basis) en een hoger dividendrendement (14,6% tegen
  9,5%). De premie die Wallenius Wilhelmsen geniet is dus niet verdiend met marge maar met
  spreiding: de logistieke portefeuille met 8,1 jaar looptijd en de wettelijk afgeschermde
  Amerikaanse overheidsdivisie maken de kasstroom minder cyclisch dan die van een pure
  reder. Of die premie 65% mag bedragen is de vraag waar deze analyse op uitkomt.

### TAM/SAM/SOM
- **TAM (mln lokale valuta):** — (geen verifieerbare, gedateerde raming gevonden; alle
  gevonden marktrapporten staan achter een betaalmuur en zijn niet controleerbaar)
- **TAM-groei %:** —
- **SAM (mln):** —
- **SAM-groei %:** —
- **Huidige penetratie %:** —
- **Impliciete penetratie na horizon %:** —
- **Groei plausibel?** false
- **Bron TAM/SAM:** —
- **Toelichting:** In plaats van een onverifieerbaar marktbedrag geef ik de twee cijfers
  die wél uit primaire bronnen komen en die dezelfde vraag beantwoorden. Wallenius
  Wilhelmsen bezit of huurt 897.800 autoplaatsen op een wereldvloot van 5,2 tot 6,0 miljoen
  autoplaatsen, oftewel 15 tot 17% van de capaciteit, en dat aandeel is sinds 2020
  nauwelijks veranderd (778.000 autoplaatsen in Q1 2020 tegen 817.000 in de
  scheepvaartdivisie nu, ongeveer 5% groei tegenover een sneller groeiende wereldvloot).
  De relevante groeivraag is daarmee niet of de markt groeit, maar of de vloot sneller
  groeit dan de lading: bij een orderboek van circa 21% van de vloot en een volume dat bij
  Wallenius Wilhelmsen sinds 2023 met 9% is gedáald, is een aanname van structurele
  volumegroei niet houdbaar. Vandaar `groei_plausibel: false` — de groei in het
  basisscenario komt van prijs en inflatie, niet van marktaandeel of volume.

---

## 7. Analyse-frameworks (9 frameworks, SCORES 1-5)

### Graham
- **Oordeel:** GEDEELTELIJK
- **Graham number:** 17,92 USD (NOK 167,15) — √(22,5 × EPS 1,648 × boekwaarde 8,66)
- **Margin of safety %:** 4,6 (ten opzichte van het Graham-getal)
- **Toelichting:** Op de klassieke Graham-toetsen komt Wallenius Wilhelmsen verrassend
  goed weg. De koers-winstverhouding van 10,4 op de laatste twaalf maanden ligt onder de
  grens van 15, de koers-boekwaardeverhouding van 1,97 net onder 2,0 maar boven Grahams
  eigen grens van 1,5, de schuldgraad is met minder dan één keer de EBITDA laag en er is
  een structureel dividend. Het Graham-getal komt uit op NOK 167,15, vier procent boven de
  koers — dus wel positief, maar zonder de veiligheidsmarge van 30% die Graham eiste.
  Waar Graham zelf zou zijn afgehaakt is de winststabiliteit: hij eiste tien jaar zonder
  verlies en Wallenius Wilhelmsen boekte in 2020 USD 301 mln verlies. De rubriek uit
  METHODE H9 laat dat criterium buiten beschouwing en geeft op koers-winst onder 15 én
  koers-boekwaarde onder 2,0 een score van 4.
- **Score (1-5):** **4**

### Buffett / Munger
- **Oordeel:** GEDEELTELIJK
- **ROIC structureel boven WACC?** false
- **Toelichting:** Buffett zoekt een uitzonderlijk bedrijf tegen een redelijke prijs, en de
  toets die daarvoor telt is of het rendement op geïnvesteerd kapitaal structureel boven de
  kapitaalkosten ligt. Bij Wallenius Wilhelmsen is dat over de volle cyclus niet zo: het
  gemiddelde over 2017 tot en met de laatste twaalf maanden bedraagt 10,4% tegen een
  kapitaalkostenvoet van 8,4%, en in vijf van de negen gemeten jaren lag het rendement
  eronder. De laatste vier jaar zijn schitterend — 14% tot 21% — maar dat is precies wat je
  bij een cyclisch bedrijf op de top verwacht. Begrijpelijk is het bedrijf zeker: het vaart
  auto's over zee en dat kan iedereen navertellen. Voorspelbaar is het niet, en dat is bij
  Buffett de zwaardere eis. De prijs is met een verhouding koers tot vrije kasstroom na
  lease van 10,2 niet extreem, maar wel op een kasstroom die zelf op de top staat.
- **Score (1-5):** **2**

### Peter Lynch
- **Categorie:** Cyclical
- **Oordeel:** NEUTRAAL
- **PEG-ratio:** n.v.t. (de analistenconsensus verwacht een dalende winst per aandeel van
  USD 1,573 in 2026 naar USD 1,115 in 2028; bij negatieve verwachte groei is de PEG-ratio
  niet gedefinieerd)
- **Toelichting:** Dit is een leerboekvoorbeeld van wat Lynch een cyclical noemt, en zijn
  belangrijkste waarschuwing bij die categorie is dat een lage koers-winstverhouding het
  einde van de rit markeert en niet het begin. Op de winst van 2025 noteert het aandeel op
  7,1 keer de winst; op de consensusverwachting voor 2028 op 15,3 keer. Het verhaal is
  helder — meer Chinese auto's, langere vaarten, te weinig schepen — maar juist de
  helderheid van dat verhaal heeft de koers in twaalf maanden 58% hoger gezet, en Lynch
  kocht cyclicals wanneer het verhaal slecht was. De rubriek in METHODE H9 kent bij een
  PEG boven 2,0 of een niet-berekenbare PEG door negatieve verwachte groei de laagste score
  toe; de eerste treffer van boven is hier score 1.
- **Score (1-5):** **1**

### Phil Fisher
- **Oordeel:** GEMIDDELD
- **Toelichting:** Fisher zocht bedrijven met producten met langjarig groeipotentieel, een
  onderzoekscultuur, verdedigde marges en een integere leiding. Van de drie harde criteria
  in de rubriek scoort Wallenius Wilhelmsen er twee. De marge wordt beschermd door de
  contractstructuur — een orderportefeuille van USD 6,5 mrd met 2,9 jaar looptijd in
  scheepvaart en USD 2,7 mrd met 8,1 jaar in logistiek — en door de wettelijk afgeschermde
  Amerikaanse overheidsdivisie. De integriteit van het management beoordeel ik als sterk,
  op grond van de transparante guidanceverlaging in mei 2026 en een consequent
  kapitaalbeleid. Het derde criterium, onderzoek en ontwikkeling boven het sectorgemiddelde,
  is niet van toepassing: een reder heeft geen R&D-budget in de klassieke zin. Wat er wél
  is, is een investering in decarbonisatie — veertien methanol-gedreven en
  ammoniak-gerede schepen, en een aandeel LNG en biobrandstof dat in Q2 2026 op 11,1% van
  het totale brandstofverbruik stond — maar dat is vlootinvestering, geen innovatiecultuur.
- **Score (1-5):** **4**

### Magic Formula (Greenblatt)
- **Oordeel:** GEMIDDELD
- **Earnings yield %:** 9,6 (EBIT laatste twaalf maanden USD 925 mln / ondernemingswaarde USD 9.627 mln)
- **Return on capital %:** 16,0 (EBIT USD 925 mln / (netto werkkapitaal USD 353 mln + netto vaste activa USD 5.409 mln))
- **Toelichting:** Greenblatt zoekt bedrijven die veel verdienen op hun operationele
  kapitaal én goedkoop zijn. Op de eerste as scoort Wallenius Wilhelmsen matig: een
  rendement op operationeel kapitaal van 16% is voor Greenblatts maatstaven laag, want zijn
  favoriete bedrijven halen 50% of meer. Dat is inherent aan de sector — een reder heeft
  bijna zes miljard aan schepen nodig om vijf miljard omzet te maken en dat is het
  tegenovergestelde van kapitaalarm. Op de tweede as, de winstopbrengst van 9,6%, doet het
  bedrijf het wel goed en ruim boven de rentestand. De combinatie levert volgens de rubriek
  een score van 3: goedkoop genoeg, maar niet het soort kapitaalefficiëntie waar de Magic
  Formula naar op zoek is.
- **Score (1-5):** **3**

### Moat
- **Score (1-5):** **3** — NARROW MOAT. Eén tot twee categorieën zijn duidelijk aanwezig
  (de wettelijk beschermde Amerikaanse overheidsdivisie en de langlopende logistieke
  contracten) en de gemiddelde ROIC-WACC-spread over de laatste vijf jaar bedraagt 7,4
  procentpunt, boven de drempel van 5 die de rubriek voor score 3 stelt. Over de volledige
  negenjarige cyclus is die spread echter slechts 2,0 procentpunt — dat is de reden dat het
  bij NARROW blijft en niet naar WIDE gaat.

### Management
- **Score (1-5):** **4** — kapitaalallocatie GOED (schuld gehalveerd, geen overnames op de
  top, verkoop met boekwinst, kas naar aandeelhouders), prikkels aligned (60% van de
  langetermijnbeloning aan rendement op geïnvesteerd kapitaal) en geen actuele controverses.
  Score 5 valt af omdat het uitstekende kapitaalbeleid pas vier jaar loopt en de
  uitvoerende directie zelf nauwelijks aandelen bezit.

### Fair Value DCF
- **Score (1-5):** **1** — het basisscenario komt uit op NOK 129,25 tegen een koers van
  NOK 159,40, een neerwaarts verschil van 18,9%. De rubriek kent bij een neerwaarts
  verschil boven 15% de laagste score toe.

### Fair Value IPO-gecorr.
- **Score (1-5):** **1** — de notering dateert van 24 juni 2010 en bestaat dus ruim langer
  dan tien jaar; de huidige vennootschap is daarvan de juridische voortzetting.
  De rubriek schrijft dan voor: score gelijk aan Fair Value DCF basis, dus 1. Er is geen
  IPO-correctie toegepast; zie de pre-IPO-check in sectie 8.

### Scorekaart totaal
- **Totaalscore:** **23**
- **Max:** 45
- **Eindoordeel:** **PASS** (totaal 23 < 24, én Fair Value DCF-score = 1 — beide regels
  wijzen dezelfde kant op)
- **Samenvatting:** Wallenius Wilhelmsen is een goed geleid, financieel sterk bedrijf in een
  structureel onaantrekkelijke sector, en het aandeel is na een koersstijging van 58% dit
  jaar vooruitgelopen op wat de kasstroom over een volle cyclus kan dragen. De kwaliteit
  scoort redelijk — management 4, moat 3, Graham 4 — maar de waardering is de doorslaggevende
  factor: het basisscenario geeft NOK 129,25 tegen een koers van NOK 159,40, en de
  waarde zonder groei (EPV) ligt op NOK 89,75, oftewel 44% onder de koers. De voornaamste
  onzekerheid is niet de vraag maar het aanbod: een orderboek van circa 21% van de
  wereldvloot dat vanaf 2028 landt, precies wanneer de eigen vloot van gemiddeld 18,3 jaar
  zelf vervangen moet worden. Twee aannames dragen het oordeel en het is eerlijk om te
  zeggen bij welke waarden het kantelt. Bij een genormaliseerde capex van USD 500 mln in
  plaats van USD 550 mln komt de basiswaarde op NOK 148,60 en wordt het oordeel HOLD (25/45);
  bij een genormaliseerde risicovrije rente van 3,10% in plaats van de spotrente van 4,71%
  komt de WACC op 7,02%, de basiswaarde op NOK 169,80 en het oordeel eveneens op HOLD
  (27/45). Onder de gekozen, gedocumenteerde aannames is het PASS. De katalysatorkalender
  helpt de these op korte termijn niet: het halfjaardividend is met 45% verlaagd, de
  orderportefeuille kromp in een jaar van USD 8,7 naar USD 6,5 mrd en de eerstvolgende
  toets is Q3 op 4 november 2026. Gezien de cyclische kasstroom is een veiligheidsmarge van
  30% het minimum; dat brengt het koopniveau op NOK 87,33.

---

## 8. Risico's

| # | Omschrijving | Kans | Impact | DCF-aanname geraakt | Toelichting |
|---|---|---|---|---|---|
| 1 | Vlootoverschot vanaf 2028: het orderboek voor autoschepen bedraagt circa 21% van de wereldvloot | HOOG | GROOT | FCF-groei fase 1 en 2, EBITDA-marge | Dit is het kernrisico en het is geen speculatie maar een zichtbaar feit: de schepen zijn besteld en de opleveringsdata liggen vast. Het management noemt zelf 21% van de wereldwijde RoRo-capaciteit met opleveringen vanaf 2030, terwijl marktbronnen voor 2028 al een aanzienlijke instroom melden. In een sector waar de vraag met enkele procenten per jaar groeit, drukt een capaciteitsuitbreiding van die orde de tarieven, en de tarieven zijn precies wat de winstexplosie van 2022–2025 heeft veroorzaakt. De analistenconsensus verwerkt dit al: EBITDA van USD 1.564 mln in 2026 naar USD 1.287 mln in 2028. |
| 2 | Vlootveroudering: 91 eigen schepen met een gemiddelde leeftijd van 18,3 jaar, waarvan 18 ouder dan 25 jaar | HOOG | GROOT | genormaliseerde capex | De capex lag tussen 2020 en 2023 op USD 112 tot 163 mln per jaar, terwijl vervanging van 650.000 eigen autoplaatsen tegen nieuwbouwprijzen over een levensduur van 25 jaar op circa USD 390 mln per jaar uitkomt. Er is dus jarenlang onder de vervangingswaarde geïnvesteerd en dat vlaagt de gerapporteerde vrije kasstroom. Er staat nog USD 1,4 mrd aan nieuwbouwtermijnen open. Deze post alleen al verlaagt de genormaliseerde vrije kasstroom met ongeveer USD 360 mln per jaar ten opzichte van het gemiddelde van de laatste zes jaar. |
| 3 | Afhankelijkheid van de Chinese auto-export en handelsbarrières | MIDDEN | GROOT | omzetgroei, volume | Ruim de helft van het volume vertrekt uit Azië en de groei van de laatste jaren komt vrijwel volledig uit China. Twee ontwikkelingen kunnen dat afknijpen: importheffingen in de Europese Unie en de Verenigde Staten op Chinese elektrische auto's, en de verplaatsing van Chinese productie naar fabrieken in Europa, Mexico en Zuidoost-Azië, waardoor de lange zeereis vervalt. Chinese fabrikanten bouwen bovendien hun eigen vloten en halen zo volume van de markt. |
| 4 | Bunkerprijs en het conflict in het Midden-Oosten | HOOG | MIDDEL | EBITDA-marge | In het tweede kwartaal van 2026 steeg de gemiddelde bunkerprijs met 48% tot USD 770 per ton en kostte dat USD 31 mln EBITDA, omdat de toeslagmechanismen pas met een kwartaal vertraging aanslaan. Het bedrijf mijdt de Rode Zee sinds december 2024 en vaart om Kaap de Goede Hoop — dat kost brandstof maar creëert tegelijk schaarste. Een de-escalatie die de Rode Zee heropent is daarmee paradoxaal genoeg negatief voor de tarieven. |
| 5 | EUKOR: aflopend Hyundai/Kia-contract eind 2029 en de put/call op het 20%-minderheidsbelang | MIDDEN | GROOT | omzet, nettoschuld | EUKOR is opgericht rond het vervoer voor Hyundai en Kia, die tegelijk 20% van EUKOR bezitten. Het oceaanvervoercontract loopt tot december 2029 en de put/call op het minderheidsbelang kan pas vanaf 1 januari 2031 worden uitgeoefend — die volgorde is bewust en beschermt de relatie tot na de contractvervaldatum, maar hij dekt de these niet daarna. De verplichting staat na herwaardering in Q2 2026 voor USD 386 mln op de balans, tegen USD 897 mln eind 2025. |
| 6 | Herziening van charterkosten voor 38 ingehuurde schepen in een krappe markt | HOOG | MIDDEL | genormaliseerde FCF | Van de 129 schepen zijn er 38 ingehuurd. De huidige lease-uitgaven bedragen circa USD 460 mln per jaar inclusief rente, gebaseerd op contracten uit een goedkopere periode. Eenjarige tijdcharters voor schepen van 6.500 autoplaatsen liggen inmiddels rond USD 90.000 per dag, een verdubbeling ten opzichte van het dieptepunt; het management noemde de chartermarkt in mei 2026 zelf als reden voor de guidanceverlaging. Bij verlenging tegen die niveaus loopt de kostenbasis structureel op. |
| 7 | Regelgeving rond CO2: EU ETS, FuelEU Maritime en het IMO Net-Zero Framework | MIDDEN | MIDDEL | EBITDA-marge, capex | De stemming van de Internationale Maritieme Organisatie over het Net-Zero Framework, met verplichte emissiereducties en afkoop via "Remedial Units", is in oktober 2025 met twaalf maanden uitgesteld tot oktober 2026; invoering kan op zijn vroegst 1 maart 2028. De onzekerheid zelf is het probleem: investeringsbeslissingen over brandstoftype moeten nu worden genomen. Een oude vloot met hoge verbruikscijfers wordt onder elk carbonregime duurder. |
| 8 | Zeggenschapsconcentratie en beperkte verhandelbaarheid | LAAG | MIDDEL | (geen directe DCF-aanname; wel waarderingskorting) | Twee families houden samen 75,7% en de free float bedraagt 23,88%. Minderheidsaandeelhouders hebben geen invloed op strategie, dividendbeleid of een eventuele overname, en het aandeel is dun verhandeld. Er is in deze analyse geen expliciete illiquiditeitskorting toegepast omdat het dagvolume ruim boven de drempel van 50.000 aandelen ligt (circa 500.000 stuks), maar de kans dat de koers een controlepremie ooit realiseert is klein. |
| 9 | Mededingingsverleden | LAAG | MIDDEL | (reputatie; geen DCF-aanname) | WWL en EUKOR waren onderdeel van het autoschipkartel waarvoor de Europese Commissie in februari 2018 zes vervoerders samen EUR 207 mln beboette, met vergelijkbare schikkingen in de Verenigde Staten en Australië. De zaken zijn afgewikkeld en volledig voorzien, en er is sindsdien geen herhaling, maar in een sector met weinig spelers en identieke klanten blijft toezicht een staand risico. |

**Verplicht risico-item — pre-IPO financial engineering:** *niet geconstateerd, en niet van
toepassing.* Er is geen recente beursgang: de notering dateert van 24 juni 2010, toen de
scheepvaart- en logistiekactiviteiten als Wilh. Wilhelmsen ASA werden afgesplitst en
zelfstandig genoteerd — een afsplitsing zonder emissie, dus zonder IPO-opbrengsten. Er zijn
geen pre-IPO schulden bij verbonden partijen aangetroffen, geen IPO-opbrengsten die aan
insiders zijn afgelost en geen dividendrecapitalisatie vóór een beursgang. De
kapitaalstructuurwijziging bij de fusie bestond uit de uitgifte van 203,1 miljoen aandelen
aan de Wallenius-zijde in ruil voor de inbreng van operationele activa, niet uit een
kasuitkering. Een IPO-gecorrigeerde fair value is daarom identiek aan de basiswaarde.

---

## 9. These invalide bij

Deze these — dat de koers vooruitloopt op de kasstroom die het bedrijf over een volledige
cyclus kan verdienen — is weerlegd wanneer het netto vrachttarief in 2027 en 2028 boven
USD 65 per kubieke meter blijft terwijl het volume weer boven 57 miljoen kubieke meter
stijgt, want dan is de krapte structureel en niet cyclisch. Zij is eveneens weerlegd
wanneer de orderportefeuille van de scheepvaartdivisie weer boven USD 8 miljard komt met
een looptijd langer dan drie jaar, of wanneer het bedrijf laat zien dat de vloot met
aanzienlijk minder dan USD 500 miljoen per jaar op peil te houden is — bijvoorbeeld doordat
de oudste achttien schepen zonder omzetverlies uit de vaart kunnen. Omgekeerd is de these
te voorzichtig gebleken als het bedrijf de veertien Shaper-schepen oplevert en de aangepaste
EBITDA daarna boven USD 1,6 miljard stabiliseert in plaats van naar de door analisten
verwachte USD 1,3 miljard te zakken.

---

## 10. ESG

### Materiële factoren (SASB-gebaseerd)

| Factor | SASB-categorie | Risiconiveau | Financiële impact | DCF-impact |
|---|---|---|---|---|
| Broeikasgasemissies van de vloot | Marine Transportation — GHG Emissions | Hoog | 1.220 duizend ton CO2 (well-to-wake) in Q2 2026; onder EU ETS en FuelEU Maritime al een directe kostenpost, onder het IMO Net-Zero Framework vanaf ten vroegste maart 2028 mogelijk een afkoopverplichting | EBITDA-marge; genormaliseerde capex (methanol- en LNG-schepen zijn duurder) |
| Energie-efficiëntie van de vloot | Marine Transportation — Air Quality / Fuel Management | Hoog | CO2-intensiteit (EEOI) 60,66 gram per ton-zeemijl in Q2 2026 tegen een doelstelling van 60,6 — precies op de streep. Verbruik 349.698 ton brandstof per kwartaal, waarvan 11,1% LNG en biobrandstof | brandstofkosten; herinvesteringsbehoefte |
| Veiligheid van werknemers | Marine Transportation — Employee Health & Safety | Middel | Ongevalsfrequentie in de logistieke divisie 12,23 per miljoen gewerkte uren in Q2 2026 — hoog, en in Q1 zelfs 13,77 na een verbetering naar 9,02 in Q4 2025. In scheepvaart 0,21, uitstekend | verzekering, boetes, reputatie bij aanbestedingen |
| Scheepsrecycling | Marine Transportation — Business Ethics | Middel | 18 eigen schepen zijn ouder dan 25 jaar en zullen op afzienbare termijn worden gesloopt; de wijze van recycling is een reputatiekwestie voor Europese klanten | geen directe DCF-impact |
| Bedrijfsethiek en mededinging | Business Ethics | Middel | historische kartelboetes (EC EUR 207 mln collectief in 2018, plus VS en Australië); nu volledig afgewikkeld | geen actuele DCF-impact |

- **Eindoordeel ESG:** **GEMIDDELD RISICO**
- **Toelichting:** Het klimaatrisico is voor een reder met een gemiddeld 18 jaar oude vloot
  reëel en meetbaar, maar Wallenius Wilhelmsen loopt op de sector voor in plaats van achter.
  De CO2-intensiteit per ton-zeemijl daalde van 32,7 gram (Q1 2020, tank-to-wake) naar
  27,62 gram nu, het aandeel LNG en biobrandstof groeide van nul in 2023 naar 11,1% van het
  verbruik in Q2 2026, en de veertien bestelde Shaper Class-schepen varen vanaf oplevering
  op methanol en zijn voorbereid op ammoniak. Het bedrijf mikt op een volledig
  emissievrije dienst in 2027 voor klanten die daarvoor betalen. Het grootste onopgeloste
  punt is de ongevalsfrequentie in de logistieke divisie, die met 12,23 per miljoen
  gewerkte uren tientallen malen hoger ligt dan aan boord van de schepen en het afgelopen
  half jaar weer opliep. Dat is een operationeel én een aanbestedingsrisico bij
  autofabrikanten die veiligheidscijfers in hun leveranciersselectie meewegen.

---

## 11. Katalysatoren

| Datum ca. | Omschrijving | Richting | Impact |
|---|---|---|---|
| 2026-08 | Ex-dividend 25 augustus, uitbetaling 16 september: USD 0,61 over de eerste helft van 2026, 45% lager dan de USD 1,10 een jaar eerder | NEGATIEF | KLEIN |
| 2026-10 | Stemming van de Internationale Maritieme Organisatie over het Net-Zero Framework, uitgesteld vanaf oktober 2025 | BINAIR | MIDDEL |
| 2026-11 | Kwartaalcijfers Q3 2026 op 4 november: eerste toets of de bunkertoeslagen de brandstofkosten daadwerkelijk terughalen en of de kasconversie herstelt van 78% | NEUTRAAL | MIDDEL |
| 2026-Q4 | Afronding van de contractonderhandelingen voor 2027; het management meldt langere looptijden en hogere tarieven bij Chinese klanten, maar de orderportefeuille kromp in een jaar van USD 8,7 naar USD 6,5 mrd | POSITIEF | GROOT |
| 2026-Q4 | Oplevering van de eerste Shaper Class-schepen bij China Merchants Jinling | POSITIEF | MIDDEL |
| 2027-02 | Jaarcijfers 2026 op 11 februari 2027, met guidance voor 2027 en het slotdividend | NEUTRAAL | GROOT |
| 2027-2028 | Instroom van sectorbrede nieuwbouw uit een orderboek van circa 21% van de wereldvloot | NEGATIEF | GROOT |
| 2027-H2 | Oplevering van de vier vergrote Shaper-schepen van 11.700 autoplaatsen, de grootste autoschepen ter wereld | POSITIEF | MIDDEL |

---

## 12. Fair value — kwantitatief (DCF)

### DCF-invoeren

```
Basis            fcf=385  shares=423.105  net_cash=-906.3  gross_debt=1147.7  revenue=5150.3
                 koers=17.088  ipo_jaar=2010
WACC             rf=4.71  erp=4.23  beta=0.99  crp=0.00  size_premium=0.00
                 cost_of_debt_pretax=5.60  tax_rate=4.50
Pessimistisch    g1=0.0  g2=1.0  gt=1.5  wacc_adj=1.50  kans=35
Basis            g1=3.0  g2=2.5  gt=2.0  wacc_adj=0.00  kans=45
Optimistisch     g1=5.0  g2=3.5  gt=2.5  wacc_adj=-0.50  kans=20
EPV              norm_ebit_margin=15.40  maintenance_capex=1010  da=671.2
                 norm_ebitda_margin=28.16
Multiples        pe=10.37  pb=1.97  p_fcf=10.18  peg=null
Rendement        roic=14.5  earnings_yield=9.6  roc_greenblatt=16.0
Kwalitatief      moat_oordeel=NARROW  moat_categorieen_sterk=1  management_oordeel=STERK
                 capital_allocation=GOED  insider_alignment_pct=0.008
                 roic_wacc_spread_5j_plus=false  structureel_dividend=true  debt_equity=0.16
Eenheid          bedragen in USD mln; koers en fair value per aandeel in USD
                 (fair value omgerekend naar NOK tegen USD/NOK 9,3280)
```

> **Let op bij het narekenen (stage 2):** `net_cash` is negatief omdat het nettoschuld is,
> en die is hier **exclusief leaseverplichtingen maar inclusief de EUKOR-putverplichting
> van USD 386 mln**: 1.147,7 (bruto schuld excl. lease) − 627,4 (kas) + 386,0 = 906,3.
> Die keuze hoort onlosmakelijk bij de FCF-definitie hieronder — zie de toelichting op de
> leasebehandeling. `p_fcf` is op de vrije kasstroom **ná** leasebetalingen.
> `moat_categorieen_sterk=1`: geen van de vijf categorieën scoort "sterk"; drie scoren
> "middel", wat het NARROW-oordeel draagt.

### WACC-componenten
- **Risicovrije rente %:** 4,71
- **Bron risicovrije rente:** Amerikaanse tienjaars staatsobligatie (constante looptijd),
  FRED-reeks DGS10, waarde per 18 augustus 2026. **De kasstromen zijn in USD gemodelleerd,
  dus is de Amerikaanse en niet de Noorse staatsrente de juiste risicovrije voet** — dit is
  METHODE H7 stap 4A: de valuta van de kasstromen bepaalt de rente, niet het vestigingsland.
- **Type:** spot (nominaal). De spotrente van 4,71% ligt ongeveer 165 basispunten boven het
  tienjaarsgemiddelde van de Amerikaanse tienjaars, waarmee de drempel van 150 basispunten
  uit METHODE H7 stap 4A wordt overschreden. Ik hanteer de spotrente als hoofdvariant, omdat
  de gebruikte impliciete risicopremie van Damodaran tegen dezelfde spotrente is gemeten en
  het inconsistent zou zijn de rente te normaliseren en de premie niet. De genormaliseerde
  variant is opgenomen in de gevoeligheidsanalyse: bij een risicovrije rente van 3,10%
  komt de WACC op 7,02% en de basiswaarde op NOK 169,80.
- **ERP (equity risk premium) %:** 4,23
- **Bron ERP:** Aswath Damodaran, "Historical Implied Equity Risk Premiums", impliciete
  premie per 1 januari 2026 (pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html)
- **Beta (adjusted, Blume):** 0,99
- **Bron beta:** ruwe vijfjaars maandbeta van 0,98 (Yahoo Finance, WAWI.OL), omgerekend met
  de Blume-aanpassing (0,67 × 0,98 + 0,33 × 1,00 = 0,99). Kruiscontrole met een bottom-up
  beta: een ongehefboomde sectorbeta voor zeevaart van circa 0,80, geherhefboomd naar de
  schuld-eigenvermogenverhouding van Wallenius Wilhelmsen (16% exclusief lease), geeft 0,90;
  inclusief lease als schuld 1,05. De regressiebeta ligt daar tussenin en is gebruikt.
- **Type beta:** vijfjaars maandelijkse regressie (Blume-aangepast). Het bedrijf is ruim
  langer dan vijf jaar genoteerd en het gemiddelde dagvolume ligt rond 500.000 aandelen,
  ruim boven de drempel van 100.000 uit METHODE H7 stap 4D — een bottom-up beta is dus niet
  verplicht.
- **Country risk premium %:** 0,00. Noorwegen heeft de hoogste kredietbeoordeling (Aaa/AAA)
  en daarmee een landenrisicopremie van nul in de systematiek van Damodaran. De blootstelling
  aan opkomende markten — ruim de helft van het volume vertrekt uit Azië, met daarnaast
  Zuid-Amerika en het Midden-Oosten — is níet via een opslag in de WACC verwerkt maar via
  de expliciet hogere kapitaalkostenvoet in het pessimistische scenario (+150 basispunten)
  en het bovengemiddelde kansgewicht dat dat scenario krijgt. Dat is een bewuste keuze: een
  omzetgewogen landenrisicopremie zou hier een schijnprecisie zijn, omdat de omzet per
  handelsroute en niet per land wordt gerapporteerd.
- **Size premium %:** null (marktkapitalisatie USD 7,23 mrd, ruim boven de drempel van EUR 2 mrd)
- **Cost of equity %:** 8,90
- **Schuldkosten na belasting %:** 5,35 (5,60% vóór belasting × (1 − 4,50%))
- **E/V gewicht %:** 86,3
- **D/V gewicht %:** 13,7
- **WACC %:** **8,41**
- **Sector WACC % (referentie Damodaran):** de wereldwijde kapitaalkostentabel voor
  zeevaart en transport ligt in de bandbreedte 7,5–8,5%; de uitkomst van 8,41% valt daar
  aan de bovenkant binnen. Een exacte waarde per 2026 is niet uit de bron te halen zonder
  het Excel-bestand en is daarom niet als getal opgenomen.
- **Illiquiditeitskorting %:** null (gemiddeld dagvolume circa 500.000 aandelen, boven de
  drempel van 50.000)

### DCF model-specs
- **Model type:** 2-fase (5 jaar + 5 jaar) met Gordon-eindwaarde
- **FCF-definitie:** FCFF (free cash flow to firm), verdisconteerd tegen de WACC
- **Basis FCF:** USD 385 mln (genormaliseerd, mid-cyclus)
- **Basis FCF na SBC:** USD 385 mln (aandelencompensatie is bij dit bedrijf niet
  afzonderlijk gerapporteerd en materieel verwaarloosbaar; zie de toelichting bij de
  kasstroomtabel)
- **FCF-type:** *Genormaliseerde FCF USD 385 mln (mid-cyclus), leases als operationele kost*
- **Groei fase 1 %:** 3,0 (jaar 1–5)
- **Groei fase 2 %:** 2,5 (jaar 6–10)
- **Terminal groei %:** 2,0
- **Terminal methode:** Gordon growth
- **Exit multiple gebruikt (EV/EBITDA):** 6,0
- **Bron exit multiple:** eigen berekening op basis van de peer-bandbreedte. Wallenius
  Wilhelmsen noteert zelf op 5,9× de aangepaste EBITDA van de laatste twaalf maanden
  (ondernemingswaarde inclusief leaseschuld); Höegh Autoliners en de Japanse lijnvaarders
  bewegen historisch tussen 4× en 6×. 6,0× is het bovenste deel van die reeks.
- **Terminal value Gordon growth:** USD 8.034 mln (nominaal, jaar 10)
- **Terminal value exit multiple:** USD 8.700 mln (6,0 × genormaliseerde aangepaste EBITDA
  van USD 1.450 mln)
- **Terminal value % van totaal:** 55,1
- **Terminal implied EV/EBITDA:** 5,5× (eindwaarde USD 8.034 mln gedeeld door de naar jaar
  10 doorgegroeide genormaliseerde EBITDA). Ruim onder de grens van 20× die METHODE als
  alarmdrempel noemt, en consistent met de exit-multiple-controle: beide methoden komen
  binnen 8% van elkaar uit.
- **Terminal groei consistentie:** een eindgroei van 2,0% vereist bij een langetermijn-ROIC
  van circa 10% een herinvesteringsvoet van 20% van de NOPAT. De genormaliseerde NOPAT
  bedraagt USD 757 mln, zodat 20% neerkomt op USD 151 mln aan netto-uitbreidingsinvestering
  bovenop de vervangingsinvestering van USD 550 mln — dat past ruimschoots binnen de USD 1,4
  mrd aan reeds vastgelegde nieuwbouwtermijnen. Bovendien ligt 2,0% onder de verwachte
  nominale BBP-groei van de Verenigde Staten, de economie waarvan de rente de discontovoet
  bepaalt. De aanname is dus haalbaar en conservatief.
- **Mid-year convention:** true
- **Aandelen uitstaand (mln):** 423,105
- **Nettoschuld huidig:** USD 906,3 mln (exclusief lease, inclusief EUKOR-put)

### DCF-toelichting

Drie keuzes bepalen deze waardering en het is belangrijk dat ze alle drie zichtbaar zijn.

**Ten eerste de leasebehandeling.** Wallenius Wilhelmsen huurt 38 van zijn 129 schepen in,
plus terminals en vastgoed, met een leaseverplichting van USD 1.490 mln op de balans en
huurbetalingen van ongeveer USD 460 mln per jaar inclusief rente. Er zijn twee correcte
manieren om daarmee om te gaan en één foute. Fout is de leaseverplichting bij de
nettoschuld optellen én de huurbetalingen ook nog van de kasstroom aftrekken — dan betaal
je twee keer voor dezelfde schepen. Correct is óf de lease als schuld behandelen (dan
telt hij mee in de nettoschuld en laat je de betalingen buiten de kasstroom), óf de lease
als operationele kost behandelen (dan trek je de betalingen af en houd je hem uit de
nettoschuld). Ik kies het tweede, om één inhoudelijke reden: de ingehuurde schepen zijn
geen eindige verplichting maar een permanent onderdeel van de vloot. De huidige contracten
lopen gemiddeld drie tot vier jaar; daarna wordt opnieuw gehuurd, tegen de tarieven van dat
moment. Wie alleen de bestaande verplichting van USD 1.490 mln als schuld aftrekt, waardeert
een bedrijf dat na 2030 gratis 38 schepen tot zijn beschikking heeft. De nettoschuld in dit
model is daarom USD 906 mln — bruto schuld exclusief lease minus kas plus de
EUKOR-putverplichting van USD 386 mln, die als reële toekomstige uitkoopverplichting wel
degelijk in de ondernemingswaarde thuishoort.

**Ten tweede de cycliciteitsnormalisatie (METHODE H7, regel 1 tot en met 5).** Dit is een
cyclisch bedrijf en de meest recente vrije kasstroom is dus uitdrukkelijk géén startpunt.
De aangepaste EBITDA bewoog over de laatste zeven jaar tussen USD 536 mln (2020, dal) en
USD 1.901 mln (2024, piek); ik hanteer een mid-cyclusniveau van USD 1.450 mln. Dat is niet
willekeurig gekozen: de gemiddelde aangepaste EBITDA-marge over 2020 tot en met de laatste
twaalf maanden bedraagt 29,7% en toegepast op de huidige omzet van USD 5.150 mln geeft dat
USD 1.530 mln, terwijl de analistenconsensus voor 2026, 2027 en 2028 gemiddeld USD 1.432
mln bedraagt. De USD 1.450 mln ligt daar precies tussenin. De omzetschaal-controle uit
regel 2c is uitgevoerd: de huidige omzet ligt 10% boven het zevenjaarsgemiddelde van
USD 4.676 mln, ruim onder de drempel van 30%, dus herschaling is niet nodig. De
margecontrole uit regel 4 klopt eveneens: de mediane vrije-kasstroommarge na lease van
16,5% toegepast op de huidige omzet geeft USD 850 mln, tegen USD 830 mln als eenvoudig
gemiddelde — een verschil van 2,4%.

**Ten derde de genormaliseerde investeringen, en dit is de zwaarste aanname van de
analyse.** Van de USD 1.450 mln genormaliseerde EBITDA gaat USD 55 mln naar kasbelasting,
USD 460 mln naar leasebetalingen en USD 550 mln naar investeringen, wat op USD 385 mln
vrije kasstroom uitkomt. Die USD 550 mln is meer dan het dubbele van wat het bedrijf sinds 2020
gemiddeld uitgaf (USD 190 mln per jaar) en dat vraagt om verantwoording. De
vlootlijst uit de factsheet levert die: de 91 eigen schepen tellen samen 650.000
autoplaatsen en zijn gemiddeld 18,3 jaar oud, met achttien schepen boven de 25 jaar. Tegen
nieuwbouwprijzen van ongeveer USD 15.000 per autoplaats kost vervanging USD 9,75 mrd,
oftewel USD 390 mln per jaar over een levensduur van 25 jaar; daar komen circa USD 110 mln
aan dokkosten en circa USD 50 mln aan logistieke investeringen bij. De USD 1,4 mrd aan
resterende nieuwbouwtermijnen bevestigt de orde van grootte. **De gerapporteerde vrije
kasstroom van de afgelopen jaren is dus voor een belangrijk deel geleend van de toekomst,
en het verschil tussen de gekozen basis van USD 385 mln en de vrije kasstroom van de
laatste twaalf maanden op dezelfde definitie (USD 782 mln) bedraagt 51%. Conform METHODE
regel 3 is dat een uitdrukkelijke rode vlag** — niet omdat de berekening onzeker is, maar
omdat wie op de gerapporteerde kasstroom afgaat een fair value krijgt die twee keer zo hoog
uitvalt. In de gevoeligheidstabel hieronder is precies zichtbaar wat een andere aanname
oplevert.

De eindwaarde draagt 55,1% van de totale ondernemingswaarde, ruim onder de grens van 75%,
en de impliciete eindmultiple van 5,5 keer de EBITDA ligt onder de multiple waarop het
aandeel vandaag noteert — de waardering leunt dus niet op een optimistisch eindbeeld.

### 5-jaars projectie (USD mln)

| Jaar | Omzet | Omzetgroei % | EBITDA | EBIT | EBIT-marge % | NOPAT | Capex | Lease | ΔNWC | Kasbelasting | FCFF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026 | 5.230 | 1,6 | 1.442 | 771 | 14,7 | 736 | −540 | −450 | 0 | −55 | 397 |
| 2027 | 5.230 | 0,0 | 1.465 | 787 | 15,0 | 751 | −545 | −455 | 0 | −56 | 409 |
| 2028 | 5.200 | −0,6 | 1.488 | 803 | 15,4 | 767 | −550 | −460 | 0 | −57 | 421 |
| 2029 | 5.250 | 1,0 | 1.511 | 819 | 15,6 | 783 | −555 | −465 | 0 | −58 | 433 |
| 2030 | 5.350 | 1,9 | 1.535 | 836 | 15,6 | 799 | −560 | −470 | 0 | −59 | 446 |

*Zo leest u deze tabel: de vrije kasstroom is berekend als **EBITDA minus kasbelasting minus
capex minus leasebetalingen**, en groeit vanaf de genormaliseerde basis van USD 385 mln met
3% per jaar. EBIT en NOPAT staan er ter controle bij; de afschrijvingen bedragen circa
USD 671 mln in 2026 oplopend naar USD 699 mln in 2030, zodat NOPAT plus afschrijvingen
minus capex, lease en het verschil tussen kas- en winstbelasting dezelfde uitkomst geeft.
De aandelencompensatiekolom uit het sjabloon is vervangen door de leasekolom, omdat
aandelencompensatie bij dit bedrijf verwaarloosbaar is en de leasebetalingen dat juist niet
zijn. De verandering in werkkapitaal staat op nul: het werkkapitaal bedraagt structureel
circa USD 353 mln en vertoont over de hele reeks geen richting. Het genormaliseerde
EBITDA-niveau van jaar 1 (USD 1.442 mln) sluit aan op de mid-cyclusraming van USD 1.450 mln
en de EBIT-marge convergeert naar de 15,4% die ook in de EPV-berekening is gebruikt.*

### Scenarios

| Scenario | FCF-groei % (fase 1) | WACC % | Fair value (NOK) | Upside % | Kans % |
|---|---|---|---|---|---|
| Pessimistisch | 0,0 | 9,91 | 47,11 | −70,4 | 35 |
| Basis | 3,0 | 8,41 | 129,25 | −18,9 | 45 |
| Optimistisch | 5,0 | 7,91 | 303,12 | +90,2 | 20 |

**Onderbouwing van de scenario's en de kansverdeling.** In het pessimistische scenario
landt het orderboek en zakt de aangepaste EBITDA naar mid-cyclus USD 1.200 mln, met een
investeringsniveau dat meedaalt naar USD 480 mln (goedkopere nieuwbouw in een zwakke markt,
uitgestelde bestellingen) en leasekosten van USD 420 mln: vrije kasstroom USD 260 mln, geen
groei in fase 1 en een kapitaalkostenvoet 150 basispunten hoger. In het optimistische
scenario absorbeert de Chinese exportgroei de nieuwe capaciteit en houdt de aangepaste
EBITDA USD 1.750 mln aan: vrije kasstroom USD 645 mln en 5% groei. De kansverdeling wijkt
bewust af van de standaard 25/50/25 naar **35/45/20**. Reden: het aanbodrisico is niet
speculatief maar contractueel vastgelegd — die schepen zijn besteld en de opleveringsdata
liggen vast — terwijl het optimistische scenario een aanhoudende exportgroei vereist die
bovendien door importheffingen en lokale productie wordt bedreigd. De koers is dit jaar al
58% gestegen, waardoor de asymmetrie in het risico naar beneden wijst.

- **Kansgewogen fair value (NOK):** 135,27 (0,35 × 47,11 + 0,45 × 129,25 + 0,20 × 303,12)

### Reverse DCF
- **Impliciete groei %:** 5,84
- **Historische FCF CAGR %:** 12,0 (vrije kasstroom na lease van USD 300 mln in 2020 naar
  USD 711 mln over de laatste twaalf maanden, zes jaar) — maar dat startpunt is het
  coronadal en de reeks is daarmee niet representatief
- **Consensus groei %:** negatief (EBITDA volgens consensus van USD 1.564 mln in 2026 naar
  USD 1.287 mln in 2028, oftewel −8,9% per jaar)
- **Interpretatie:** Om de huidige koers van NOK 159,40 te rechtvaardigen moet de
  genormaliseerde vrije kasstroom van USD 385 mln bijna zes procent per jaar groeien,
  tien jaar lang. Anders geformuleerd, en dat is de eerlijker vergelijking: de markt prijst
  een duurzame vrije kasstroom van USD 463 mln in tegen bescheiden groei — twintig procent
  meer dan mijn mid-cyclusraming. Dat is geen absurde inprijzing en dat is precies het punt:
  het verschil tussen mijn oordeel en de markt is geen fundamenteel meningsverschil over
  het bedrijf, maar een verschil van twintig procent in de schatting van wat de vloot over
  een volle cyclus verdient nadat je hem netjes hebt vervangen. Wie gelooft dat de Chinese
  exportgolf structureel is en dat vervanging goedkoper kan dan USD 550 mln per jaar, komt
  op de huidige koers uit. Wie het orderboek en de vlootleeftijd zwaarder weegt, niet.

### EPV (Bruce Greenwald)
- **Genormaliseerde EBIT-marge %:** 15,40 (gemiddelde over 2019–TTM, de periode vanaf de
  invoering van IFRS 16 waarin de marges onderling vergelijkbaar zijn)
- **Genormaliseerde NOPAT:** USD 757,5 mln
- **Maintenance capex:** USD 1.010 mln — dit is bewust hóger dan de afschrijvingen van
  USD 671 mln, omdat het naast de vervangingsinvestering van USD 550 mln voor de eigen
  vloot ook de jaarlijkse charterkosten van USD 460 mln bevat. Dat is dezelfde
  leasebehandeling als in de DCF; zou ik hier alleen de USD 550 mln aanhouden, dan zou de
  EPV een bedrijf waarderen dat 38 schepen gratis in de vaart heeft.
- **Adjusted earnings power:** USD 418,7 mln (NOPAT 757,5 + afschrijvingen 671,2 − 1.010,0)
- **EPV per aandeel:** NOK 89,75 (USD 9,62)
- **Groeipremie %:** 44,0 (basiswaarde uit de DCF ten opzichte van de EPV). De koers ligt
  77,6% boven de EPV.
- **Interpretatie:** De EPV is de waarde van het bedrijf zonder enige groei — puur wat de
  huidige vloot en het huidige contractenboek in een stabiele toestand kunnen verdienen. Die
  komt uit op NOK 89,75, ruim onder de koers. Dat betekent dat de markt op dit moment
  ongeveer NOK 70 per aandeel betaalt voor groei die nog moet komen. Bij een bedrijf met een
  ROIC die over de cyclus amper twee procentpunt boven de kapitaalkosten ligt, is groei
  bovendien maar beperkt waardevol: elke extra dollar omzet vereist bijna evenveel extra
  kapitaal als hij oplevert.

### Andere methoden
- **DDM uitgevoerd?** false. Het dividend is geen stabiele, voorspelbare stroom maar een
  restpost die in 2020 en 2021 volledig wegviel en in 2026 alweer met 45% is verlaagd. Een
  dividenddiscontomodel zou hier meer schijnzekerheid dan inzicht opleveren.
- **SOTP uitgevoerd?** false. De drie segmenten zijn operationeel verweven — de logistieke
  divisie en de overheidsdivisie leunen op dezelfde vloot en dezelfde terminals, met
  USD 248 mln aan interne eliminaties in 2025 — en een som-der-delen zou die verwevenheid
  wegdefiniëren.

### Synthese fair value
- **Bandbreedte laag:** 89,75 (EPV — waarde zonder groei)
- **Bandbreedte centraal:** 124,76 (gewogen synthese)
- **Bandbreedte hoog:** 170,90 (7,0 × genormaliseerde EBITDA; komt vrijwel samen met de
  DCF-uitkomst van NOK 169,80 bij een genormaliseerde risicovrije rente van 3,10%)
- **Methode-gewichten:**
  - DCF %: 45
  - EPV %: 20
  - Multiples %: 35
- **Margin of safety vereist %:** 30
- **Koopniveau:** 87,33 (124,76 × 0,70)
- **Synthese-toelichting:** De drie methoden komen dichter bij elkaar dan de losse cijfers
  suggereren. De discounted cashflow geeft NOK 129,25, een waardering op zes keer de
  genormaliseerde EBITDA geeft NOK 139,00 en de waarde zonder groei NOK 89,75. Ik weeg de
  DCF met 45% het zwaarst, maar geef de multiples bewust 35% in plaats van de gebruikelijke
  bijrol, omdat de kapitaalkostenvoet van 8,41% die uit de rubriek rolt voor een cyclische
  reder aan de lage kant aanvoelt en marktmultiples de rendementseis bevatten die
  beleggers werkelijk hanteren. De EPV krijgt 20% als ondergrens-anker. Dat brengt de
  synthese op NOK 124,76 tegen een koers van NOK 159,40 — de markt betaalt 28% boven de
  gewogen waardering en 78% boven de waarde zonder groei. Bij een cyclisch bedrijf met een
  ROIC die over de volle rit slechts twee procentpunt boven de kapitaalkosten ligt, is een
  veiligheidsmarge van 30% het minimum; interessant wordt het aandeel dus rond NOK 87, wat
  vrijwel samenvalt met de waarde zonder groei. Dat is geen toeval maar de kern van de
  these: pas onder de EPV krijgt een belegger de cyclus er gratis bij.

### Gevoeligheid (DCF)

**FCF-groei ↔ WACC** (fair value in NOK, basis-FCF USD 385 mln; fase 2 = 80% van fase 1)

| Groei fase 1 \ WACC | 7,0% | 7,5% | 8,0% | 8,5% | 9,0% | 9,5% |
|---|---|---|---|---|---|---|
| 0,0% | 105,4 | 97,4 | 90,3 | 84,0 | 78,5 | 73,5 |
| 1,5% | 140,4 | 127,4 | 116,4 | 107,0 | 98,9 | 91,7 |
| 3,0% | 169,9 | 152,9 | 138,8 | 126,8 | 116,5 | 107,6 |
| 4,5% | 191,8 | 172,5 | 156,4 | 142,8 | 131,1 | 121,1 |
| 6,0% | 216,1 | 194,2 | 176,0 | 160,6 | 147,4 | 136,0 |

**De tweede gevoeligheid is belangrijker dan de eerste.** Omdat de waardering staat of valt
met twee genormaliseerde grootheden — de mid-cyclus EBITDA en de vervangingsinvestering —
staat hieronder wat andere aannames opleveren (fair value in NOK, basisscenario, WACC 8,41%):

| Capex \ genorm. adj. EBITDA | 1.250 | 1.350 | 1.450 | 1.550 | 1.650 |
|---|---|---|---|---|---|
| 450 | 90,5 | 129,2 | **168,0** | 206,8 | 245,5 |
| 500 | 71,1 | 109,9 | **148,6** | 187,4 | 226,2 |
| 550 | 51,7 | 90,5 | **129,2** | 168,0 | 206,8 |
| 600 | 32,3 | 71,1 | **109,9** | 148,6 | 187,4 |
| 650 | 13,0 | 51,7 | **90,5** | 129,2 | 168,0 |

De koers van NOK 159,40 wordt gerechtvaardigd bij een combinatie van ongeveer USD 1.500 mln
mid-cyclus EBITDA en USD 500 mln investeringen, of USD 1.450 mln EBITDA en USD 460 mln
investeringen. Dat zijn geen onvoorstelbare getallen — ze zijn optimistischer dan mijn
basis, maar liggen binnen het bereik van wat een belegger redelijkerwijs kan aannemen.

---

## 13. Databronnen

### Bronnen-hiërarchie
- **Jaarverslag PDF / IR-pagina** → betrouwbaarheid **HOOG**
- **Beursmelding / kwartaalrapport / IR-factsheet (xlsx)** → betrouwbaarheid **HOOG**
- **Aggregator** (MacroTrends / StockAnalysis / Yahoo / MarketScreener) → **AGGREGATOR**

### Financiële bronnen (10 jaar historie)

| Jaar | Bron | URL | Betrouwbaarheid |
|---|---|---|---|
| 2016 | WWASA Annual Report 2016 — niet in de reeks opgenomen (andere consolidatieperimeter, zie ontbrekende data) | https://www.walleniuswilhelmsen.com/storage/downloads/wwasa-annual-report-2016.pdf | HOOG |
| 2017 | Wallenius Wilhelmsen Logistics ASA Annual Report 2017 | https://www.walleniuswilhelmsen.com/storage/images/WWL-Annual-Report-2017.pdf | HOOG |
| 2018 | Wallenius Wilhelmsen Annual Report 2018 | https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius_Wilhelmsen_Annual_report_2018.pdf | HOOG |
| 2019 | Wallenius Wilhelmsen Annual Report 2019 | https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius-Wilhelmsen-Annual-Report-2019.pdf | HOOG |
| 2020 | Wallenius Wilhelmsen Annual Report 2020 + Factsheet Q2 2026 | https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen_Annual-Report-2020.pdf | HOOG |
| 2021 | Wallenius Wilhelmsen Annual Report 2021 + Factsheet Q2 2026 | https://www.walleniuswilhelmsen.com/storage/downloads/Annual-report-2021.pdf | HOOG |
| 2022 | Wallenius Wilhelmsen Annual Report 2022 + Factsheet Q2 2026 | https://www.walleniuswilhelmsen.com/storage/images/2022-wallenius-wilhelmsen-annual-report.pdf | HOOG |
| 2023 | Wallenius Wilhelmsen Annual Report 2023 + Factsheet Q2 2026 (geherformuleerd) | https://www.walleniuswilhelmsen.com/storage/images/WWAnnualReport2023.pdf | HOOG |
| 2024 | Wallenius Wilhelmsen Annual Report 2024 + Factsheet Q4 2024 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/WAWI_2024-Annual-report.pdf | HOOG |
| 2025 | Wallenius Wilhelmsen Annual Report 2025 + Factsheet Q4 2025 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/2025-Annual-report-document.pdf | HOOG |
| 2026 (H1/TTM) | Wallenius Wilhelmsen Quarterly Report Q2 2026 + Factsheet Q2 2026 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/Wallenius-Wilhelmsen-Quarterly-report-Q2-2026.pdf | HOOG |

**Alle elf jaren zijn HOOG.** Er is geen enkel jaar op een aggregator gebaseerd.

### Jaarverslagen geraadpleegd

| Jaar | Bron | URL |
|---|---|---|
| 2025 | Annual report 2025 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/2025-Annual-report-document.pdf |
| 2025 | Executive Remuneration Report 2025 | https://www.walleniuswilhelmsen.com/storage/images/2025-Remuneration-report.pdf |
| 2024 | Annual report 2024 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/WAWI_2024-Annual-report.pdf |
| 2023 | Annual report 2023 | https://www.walleniuswilhelmsen.com/storage/images/WWAnnualReport2023.pdf |
| 2022 | Annual report 2022 | https://www.walleniuswilhelmsen.com/storage/images/2022-wallenius-wilhelmsen-annual-report.pdf |
| 2021 | Annual report 2021 | https://www.walleniuswilhelmsen.com/storage/downloads/Annual-report-2021.pdf |
| 2020 | Annual report 2020 | https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen_Annual-Report-2020.pdf |
| 2019 | Annual report 2019 | https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius-Wilhelmsen-Annual-Report-2019.pdf |
| 2018 | Annual report 2018 | https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius_Wilhelmsen_Annual_report_2018.pdf |
| 2017 | Annual report 2017 | https://www.walleniuswilhelmsen.com/storage/images/WWL-Annual-Report-2017.pdf |

### Beursmeldingen en kwartaalbronnen geraadpleegd

| Datum | Omschrijving | URL |
|---|---|---|
| 2026-08-11 | Kwartaalrapport Q2 2026 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/Wallenius-Wilhelmsen-Quarterly-report-Q2-2026.pdf |
| 2026-08-11 | Presentatie Q2 2026 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/WAWI-Q2-26-presentation-Final.pdf |
| 2026-08-11 | Factsheet Q2 2026 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/Wallenius-Wilhelmsen-Factsheet-Q2-2026.xlsx |
| 2026-08-11 | Transcript telefonische toelichting Q2 2026 | https://www.investing.com/news/transcripts/earnings-call-transcript-wallenius-wilhelmsen-lifts-2026-outlook-in-q2-2026-93CH-4850872 |
| 2026-05-06 | Verlaging guidance 2026 bij de cijfers over Q1 2026 | https://breakbulk.news/wallenius-wilhelmsen-cuts-2026-outlook-as-middle-east-conflict-and-fuel-costs-squeeze-roro-earnings/ |
| 2026-02 | Factsheet Q4 2025 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen-Factsheet-Q4-2025.xlsx |
| 2025-02 | Factsheet Q4 2024 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen-Q4-2024-Fact-Sheet.xlsx |
| 2025-09-19 | Melding transactie primaire insider — CEO draagt aandelen over aan eigen holding | https://www.walleniuswilhelmsen.com/stock-exchange-notice/wallenius-wilhelmsen-mandatory-notification-of-trade-president-and-ceo-lasse-kristoffersen-transfer-privately-held-shares-to-a-100-owned-company |
| 2018-02-21 | Schikking met de Europese Commissie in het autoschiponderzoek | https://www.walleniuswilhelmsen.com/stock-exchange-notice/wallenius-wilhelmsen-logistics-companies-reach-settlement-with-the-european-commission-in-car-carrier-investigation |

### IPO-prospectus
- **Geraadpleegd?** false
- **URL:** —
- **Pre-IPO data beschikbaar?** false
- **Pre-IPO bron:** niet van toepassing. Er is geen recente beursgang; de huidige
  vennootschap is de voortzetting van de al bestaande notering van Wilh. Wilhelmsen ASA op
  Oslo Børs. De relevante gebeurtenis is de fusie van 4 april 2017, die in het jaarverslag
  2017 is beschreven en waarvan de proforma-cijfers voor 2016 en 2017 daar staan — maar die
  proforma-cijfers zijn expliciet ongeaudit en zijn daarom niet in de tabellen gebruikt.

### Non-GAAP
- **Gebruikt?** true
- **Toelichting:** aangepaste EBITDA is de guidance-maatstaf van het management en is
  gebruikt om het genormaliseerde mid-cyclusniveau te bepalen, omdat eenmalige boekwinsten
  (met name de USD 135 mln op de verkoop van MIRRAT in 2025) daar niet in horen. De
  historische tabellen en de winstmaatstaven in de scorekaart zijn op IFRS-grondslag. Het
  verschil tussen beide is in de laatste vier jaar 0 tot 1,7%.

### Ontbrekende data (eerlijke lijst)
- **Boekjaar 2016 volledig leeg.** Wilh. Wilhelmsen ASA verwerkte WWL, EUKOR en ARC vóór de
  fusie van 4 april 2017 volgens de vermogensmutatiemethode; omzet, balans en aandelenaantal
  (220 mln versus 423 mln) zijn onvergelijkbaar. De bron bestaat en is gevonden, maar
  opnemen zou de reeks vervalsen. De analyse dekt daarmee negen boekjaren plus de laatste
  twaalf maanden — voldoende voor een volledige cyclus (dal 2020, piek 2024–2025), maar één
  jaar korter dan de norm van tien.
- **Boekjaar 2017 is een gebroken jaar**: alleen vanaf 4 april 2017 geconsolideerd. De
  ratio's van dat jaar zijn indicatief.
- **Capex, betaalde belasting, betaalde rente en lease-aflossing ontbreken voor 2017 en
  2018** en de lease-uitsplitsing ontbreekt voor 2019; de factsheets beginnen bij Q1 2020 en
  de jaarverslagen van die jaren specificeren deze regels niet apart. De vrije-kasstroomrij
  is voor die jaren daarom leeg.
- **Brutowinst en brutomarge zijn voor geen enkel jaar ingevuld**: Wallenius Wilhelmsen
  rapporteert geen kostprijs van de omzet.
- **Aandelencompensatie (SBC) is niet als afzonderlijke post gerapporteerd.** Uit het
  beloningsrapport blijkt dat het bedrag verwaarloosbaar is (totale directiebeloning
  USD 9,6 mln in 2025, waarvan de helft in aandelen), maar een exact groepscijfer ontbreekt.
- **Historische waarderingsmultiples (tienjaarsgemiddelde P/E en EV/EBITDA) ontbreken.** Ik
  heb geen geverifieerde koersreeks over tien jaar en heb die liever leeg gelaten dan
  gereconstrueerd uit aggregators.
- **Vijf- en tienjaarsgemiddelde dividendrendement ontbreekt** om dezelfde reden.
- **Klantconcentratie in procenten van de omzet ontbreekt**: het bedrijf publiceert dit niet.
- **TAM en SAM in geldbedragen ontbreken**: de gevonden marktrapporten staan achter een
  betaalmuur en zijn niet controleerbaar. In plaats daarvan is het marktaandeel op basis van
  scheepscapaciteit gegeven, uit primaire bronnen.
- **Financiële kerncijfers van de Japanse en Koreaanse concurrenten ontbreken**: het
  autoschipsegment is daar onderdeel van grotere conglomeraten en niet verifieerbaar los te
  krijgen.
- **Indexlidmaatschap niet geverifieerd**: of WAWI in de OBX-index is opgenomen kon ik niet
  uit een betrouwbare bron bevestigen; het veld is leeg gelaten.
- **De landenrisicopremie is op nul gezet** (Noorwegen, Aaa). De blootstelling aan opkomende
  markten is via de scenario-WACC verwerkt in plaats van via een omzetgewogen opslag, omdat
  de omzet per handelsroute en niet per land wordt gerapporteerd.
- **De koers is de slotkoers van 2026-08-20** volgens twee onafhankelijke bronnen
  (MarketScreener en Investtech, beide NOK 159,40). StockAnalysis en Yahoo Finance gaven op
  dat moment verouderde gecachte koersen uit mei en juli 2026 en zijn daarom niet gebruikt.

### Peildatum analyse
- 2026-08-20

---

## 14. Volledige bronnen-lijst

| Titel | URL | Type |
|---|---|---|
| Wallenius Wilhelmsen Annual report 2025 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/2025-Annual-report-document.pdf | jaarverslag |
| Wallenius Wilhelmsen Annual report 2024 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/WAWI_2024-Annual-report.pdf | jaarverslag |
| Wallenius Wilhelmsen Annual report 2023 | https://www.walleniuswilhelmsen.com/storage/images/WWAnnualReport2023.pdf | jaarverslag |
| Wallenius Wilhelmsen Annual report 2022 | https://www.walleniuswilhelmsen.com/storage/images/2022-wallenius-wilhelmsen-annual-report.pdf | jaarverslag |
| Wallenius Wilhelmsen Annual report 2021 | https://www.walleniuswilhelmsen.com/storage/downloads/Annual-report-2021.pdf | jaarverslag |
| Wallenius Wilhelmsen Annual report 2020 | https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen_Annual-Report-2020.pdf | jaarverslag |
| Wallenius Wilhelmsen Annual report 2019 | https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius-Wilhelmsen-Annual-Report-2019.pdf | jaarverslag |
| Wallenius Wilhelmsen Annual report 2018 | https://www.walleniuswilhelmsen.com/storage/downloads/Wallenius_Wilhelmsen_Annual_report_2018.pdf | jaarverslag |
| Wallenius Wilhelmsen Logistics ASA Annual report 2017 | https://www.walleniuswilhelmsen.com/storage/images/WWL-Annual-Report-2017.pdf | jaarverslag |
| WWASA Annual report 2016 | https://www.walleniuswilhelmsen.com/storage/downloads/wwasa-annual-report-2016.pdf | jaarverslag |
| Wilh. Wilhelmsen ASA (WWASA) Annual report 2013 — bevestigt beursnotering per 24 juni 2010 | https://www.walleniuswilhelmsen.com/storage/images/Wilh.-Wilhelmsen-ASA-WWASA-NO-Annual-Report-for-period-end-31-Dec-2013-English-PDF.pdf | jaarverslag |
| Executive Remuneration Report 2025 | https://www.walleniuswilhelmsen.com/storage/images/2025-Remuneration-report.pdf | jaarverslag |
| Quarterly report Q2 2026 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/Wallenius-Wilhelmsen-Quarterly-report-Q2-2026.pdf | beursmelding |
| Quarterly presentation Q2 2026 | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/WAWI-Q2-26-presentation-Final.pdf | beursmelding |
| Factsheet Q2 2026 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Investor-relations/Wallenius-Wilhelmsen-Factsheet-Q2-2026.xlsx | beursmelding |
| Factsheet Q4 2025 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen-Factsheet-Q4-2025.xlsx | beursmelding |
| Factsheet Q4 2024 (xlsx) | https://www.walleniuswilhelmsen.com/storage/images/Wallenius-Wilhelmsen-Q4-2024-Fact-Sheet.xlsx | beursmelding |
| Investor relations — financiële kalender en verslagenoverzicht | https://www.walleniuswilhelmsen.com/who-we-are/investors | beurswebsite |
| Jaarverslagenarchief | https://www.walleniuswilhelmsen.com/who-we-are/investors/annual-report | beurswebsite |
| Schikking met de Europese Commissie in het autoschiponderzoek (2018) | https://www.walleniuswilhelmsen.com/stock-exchange-notice/wallenius-wilhelmsen-logistics-companies-reach-settlement-with-the-european-commission-in-car-carrier-investigation | beursmelding |
| Melding transactie primaire insider — CEO, september 2025 | https://www.walleniuswilhelmsen.com/stock-exchange-notice/wallenius-wilhelmsen-mandatory-notification-of-trade-president-and-ceo-lasse-kristoffersen-transfer-privately-held-shares-to-a-100-owned-company | beursmelding |
| Transcript telefonische toelichting Q2 2026 | https://www.investing.com/news/transcripts/earnings-call-transcript-wallenius-wilhelmsen-lifts-2026-outlook-in-q2-2026-93CH-4850872 | nieuwsartikel |
| Slides Q2 2026 — China-boom compenseert bunkerkosten | https://www.investing.com/news/company-news/wallenius-wilhelmsen-q2-2026-slides-china-boom-offsets-bunker-costs-93CH-4850908 | nieuwsartikel |
| Verlaging guidance 2026 (mei 2026) | https://breakbulk.news/wallenius-wilhelmsen-cuts-2026-outlook-as-middle-east-conflict-and-fuel-costs-squeeze-roro-earnings/ | nieuwsartikel |
| Shaper Class-programma: vier schepen vergroot naar 11.700 CEU | https://gcaptain.com/wallenius-wilhelmsen-supersizes-shaper-class-car-carriers/ | nieuwsartikel |
| Autoschipvloot groeit met circa 40% | https://splash247.com/car-carrier-fleet-poised-to-expand-by-40/ | nieuwsartikel |
| Koers, marktkapitalisatie en analistenkoersdoelen WAWI (peildatum) | https://www.marketscreener.com/quote/stock/WALLENIUS-WILHELMSEN-ASA-6340571/ | aggregator |
| Slotkoers WAWI 2026-08-20 (bevestiging) | https://www.investtech.com/no/market.php?CompanyID=101042 | aggregator |
| Analistenconsensus 2026–2028 WAWI | https://www.marketscreener.com/quote/stock/WALLENIUS-WILHELMSEN-ASA-6340571/finances/ | analistenrapport |
| Aandeelhoudersstructuur en free float WAWI | https://www.marketscreener.com/quote/stock/WALLENIUS-WILHELMSEN-ASA-6340571/company-shareholders/ | aggregator |
| Insidertransacties WAWI | https://www.marketscreener.com/quote/stock/WALLENIUS-WILHELMSEN-ASA-6340571/company-insider-trading/ | aggregator |
| Kerncijfers en beta WAWI | https://stockanalysis.com/quote/osl/WAWI/statistics/ | aggregator |
| Höegh Autoliners kerncijfers en resultatenreeks | https://stockanalysis.com/quote/osl/HAUTO/financials/ | aggregator |
| Amerikaanse tienjaars staatsrente (DGS10) | https://fred.stlouisfed.org/series/DGS10 | databron |
| Damodaran — impliciete equity risk premium | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html | databron |
| USD/NOK wisselkoers 2026-08-20 | https://tradingeconomics.com/norway/currency | databron |
| Status IMO Net-Zero Framework na uitstel van de stemming | https://www.kslaw.com/news-and-insights/impacts-of-postponed-imo-vote-on-global-carbon-pricing-rules-on-shipping | onderzoeksrapport |

---

## 15. Update-historie

| Datum | Versie | Wijziging |
|---|---|---|
| 2026-08-20 | 1.0 | Eerste publicatie |

---

## Opmerkingen voor Claude Code

1. **Leasebehandeling wijkt bewust af van de standaardformule.** De nettoschuld in de
   DCF-invoeren (USD 906,3 mln) is *exclusief* leaseverplichtingen en *inclusief* de
   EUKOR-putoptie; de basis-FCF van USD 385 mln is navenant *ná* leasebetalingen van
   USD 460 mln. Wie de leaseverplichting van USD 1.490 mln alsnog bij de nettoschuld optelt
   zonder de FCF terug te corrigeren, telt dubbel en krijgt een fair value die circa 40%
   te laag uitkomt. De balanstabel in sectie 3 rapporteert de nettoschuld wél inclusief
   lease (USD 2.011 mln), zoals het bedrijf zelf doet — dat verschil is opzettelijk en
   staat op beide plaatsen toegelicht.
2. **Het eindoordeel ligt op de grens.** Totaal 23/45 tegen een HOLD-drempel van 24, en de
   Fair Value DCF-score van 1 komt bij een neerwaarts verschil van 18,9% tegen een drempel
   van 15%. Bij een genormaliseerde capex van USD 500 mln of een genormaliseerde risicovrije
   rente van 3,10% kantelt het oordeel naar HOLD. Dit staat expliciet in de
   scorekaart-samenvatting.
3. **Valuta:** alles in het DCF-blok is in USD; de fair values in de tabellen zijn in NOK
   tegen 9,3280. Controleer bij het narekenen dat `dcf_calculator.py` de USD-uitkomst
   teruggeeft en dat de omrekening naar NOK apart gebeurt.
4. **Nieuwe betrouwbare bron voor de bronnenlijst in METHODE.md:** de kwartaal-factsheets in
   xlsx van Wallenius Wilhelmsen bevatten zes jaar kwartaaldata met volledige kasstroom,
   segment-P&L, vlootlijst per schip en ESG-data. Ze zijn niet via fetch leesbaar (binair)
   en horen meteen op de haallijst. Datzelfde geldt waarschijnlijk voor andere Noorse
   uitgevers die factsheets publiceren.
5. **Structureel onbetrouwbaar gebleken voor koersen op de peildatum:** stockanalysis.com en
   finance.yahoo.com leverden bij WAWI gecachte koersen van weken tot maanden oud.
   MarketScreener en Investtech gaven de juiste slotkoers en bevestigden elkaar.

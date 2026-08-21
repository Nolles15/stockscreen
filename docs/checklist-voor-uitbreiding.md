# Wat je moet checken vóór je een land toevoegt

Opgesteld 21 augustus 2026, na het lezen van `engine/`, `config.yaml` en
`docs/DIAGNOSE_INSUFFICIENT_DATA.md`.

De kern: **twee van de zes punten hieronder gaan over je huidige screener, niet
over nieuwe landen — en die twee bepalen of uitbreiden überhaupt zin heeft.**
Doe die eerst.

---

## Check 1 — Wat zijn die 713 "geen oordeel" écht? (eerst, half uur werk)

Dit is de belangrijkste. In `overzicht-2026-08-21.md` staat als aanname:

> *De 713 zonder oordeel zijn grotendeels dode symbolen uit de uitbreiding naar
> 2.812 tickers.*

Die aanname is nooit gemeten, en je eigen diagnosedocument spreekt hem tegen. In
juli, bij 799 tickers, was de conclusie letterlijk: *"Het is geen
data-beschikbaarheidsprobleem."* Slechts **2 van de 799** gaven écht niets terug
bij Yahoo. De rest werd afgekeurd door je eigen kwaliteitspoort. Na route A ging
je van 261 geblokkeerd naar **33** — 4% van het universum.

Nu is het 713 van 2.812, dus **25%**. Dat is zes keer zo hoog. Er zijn drie
mogelijke verklaringen en ze vragen alle drie iets anders:

| | Verklaring | Wat je moet doen |
|---|---|---|
| **a** | Echt dode symbolen uit de uitbreiding | Niets — archivering ruimt ze op, precies zoals het overzicht zegt |
| **b** | De rotatie is er nog niet langs geweest | Niets — het lost zichzelf op, en 713 is geen probleem |
| **c** | De poort keurt ze opnieuw af, om een nieuwe reden | **Alles** — dan wordt het bij Japan alleen maar erger |

Verklaring **b** is verrassend waarschijnlijk en wordt nergens genoemd. Je
universum groeide in augustus van 913 naar ~2.760. `refresh_fundamentals_batch`
pakt "de langst niet-geprobeerde tickers", 250 per nacht. In je eigen commentaar
in `refresh.py` staat dat de inhaalslag voor de 2.000 nieuwe tickers 11 nachten
kost. Het overzicht is van 21 augustus. Een flink deel van die 713 is
waarschijnlijk gewoon **nog nooit opgehaald**.

Het instrument om dit te beslechten heb je al:

```bash
python scripts/gaps_analyze.py --sample 12
```

of live:

```bash
curl -k -s https://stockscreen-janco.fly.dev/api/gaps-report \
  | jq 'group_by(.primary_blocker) | map({b: .[0].primary_blocker, n: length}) | sort_by(-.n)'
```

**Voeg geen enkel land toe voordat je weet welke van de drie het is.** Bij (a) of
(b) kun je door. Bij (c) los je eerst dat op, want anders importeer je 3.700
Japanse aandelen in een poort die ze weggooit.

---

## Check 2 — "Geen oordeel" is één bak voor vijf verschillende dingen

Dit is het ontwerpadvies dat ik als eerste zou uitvoeren, omdat het check 1
permanent maakt in plaats van eenmalig.

In `screener.py` krijgt een aandeel het label `INSUFFICIENT DATA` in ten minste
vijf situaties:

1. `not annual_rows` — geen jaarcijfers in de database (regel ~248)
2. `dq_status in ("bad", "missing")` — de kwaliteitspoort keurt af (regel ~261)
3. `fv_price_ratio < 0.1 or > 10.0` — vermoedelijke schaal-/eenheidsbug (regel ~337)
4. `methods_used < 2` — te weinig valide waarderingsmethodes (regel ~350)
5. geen koers

Op je dashboard zien die er identiek uit. En situatie 1 dekt twee volstrekt
verschillende gevallen af die je niet uit elkaar kunt houden: **"Yahoo heeft niets"**
en **"we hebben het nog nooit gevraagd"**.

Zolang dat zo is, kun je van geen enkele nieuwe markt vaststellen of hij werkt.
Voeg je Japan toe en zie je 400 keer "geen oordeel", dan weet je niet of Yahoo
faalt, je poort te streng is, of de rotatie er simpelweg nog niet langs is.

Splits het in drie zichtbare toestanden voordat je uitbreidt:

- `NOG_NIET_OPGEHAALD` — `last_fundamentals IS NULL`, dus geen oordeel maar ook
  geen probleem
- `DATA_AFGEKEURD` — de poort zei nee, met de reden erbij
- `NIET_WAARDEERBAAR` — data is goed, het model kan er niets mee (holdings,
  te weinig methodes)

Dat is het verschil tussen een getal dat je kunt interpreteren en een getal
waar je naar staart.

---

## Check 3 — De FV-plausibiliteitspoort staat te ruim op 10×

Je poort blokkeert bij een fair value onder 0,1× of boven 10× de koers. Bedoeld
om schaalbugs te vangen — pence tegen ponden, verkeerde valuta, verkeerd aantal
aandelen. Dat werkt: een factor-100-fout wordt gepakt.

Maar kijk naar je eigen portefeuille:

| Aandeel | Modelgrens t.o.v. koers | Door de poort? |
|---|---:|---|
| Econocom | 7,7× | ja |
| Arctic Paper | 6,6× | ja |

Allebei glippen ze er onderdoor, en allebei staan ze in je overzicht als
*"heeft geen bruikbare grens: het generieke model kan deze aandelen niet
waarderen"*. Dat is de poort die je vertelt dat hij te ruim staat.

Waarom dit telt bij uitbreiding: andere boekhoudconventies leveren geen extra
factor-100-fouten op — die vang je al. Ze leveren extra **factor-3-tot-8**-
onzin op, precies de band waar je poort niets doet. Overweeg een tweede,
zachtere band (bijvoorbeeld 3× tot 10×) die niet blokkeert maar wel markeert,
zodat zulke gevallen zichzelf melden in plaats van in je kansenlijst te
belanden.

---

## Check 4 — Je sectorparameters zijn Europees geijkt (het echte risico)

Dit is het punt waar geen enkele import iets aan verandert, en het kan je
kansenlijst in één klap waardeloos maken.

In `config.yaml` staat één set multiples voor de hele wereld:

```yaml
Financial Services:
  pb: 1.4
Technology:
  pe: 25
```

Japanse banken noteren al dertig jaar structureel onder boekwaarde, om redenen
die met de Japanse economie te maken hebben en niet met onderwaardering. Hongkongse
vastgoedbedrijven idem. Zet je `pb: 1.4` op die markten los, dan krijg je bij
elke Japanse regionale bank en elke Hongkongse ontwikkelaar een korting van 60 tot
70% te zien.

Je kansenlijst wordt dan overspoeld door Japanse financials en Hongkongs vastgoed
— en dat is een **modelartefact, geen vondst**. Erger: het verdringt de Europese
namen waar je model wél op geijkt is.

Hetzelfde geldt voor `bond_yield: 5.0`, waarmee je de Graham-formule herschaalt.
Dat is één rente voor de hele wereld. Japan zit daar nergens bij in de buurt.

Twee routes:

1. **Regiofactor op de multiples** — een vermenigvuldiger per markt bovenop het
   sectorprofiel. Netjes, maar je moet hem ergens op ijken.
2. **Nieuwe markt begint in observatiestand** — importeren, waarderen, maar
   uitsluiten van de kansenlijst tot je hebt gecontroleerd wat het model doet.
   Sneller, en eerlijker over wat je wel en niet weet.

**Goedkope test die dit vooraf beslecht:** pak 20 Japanse bedrijven waarvan je
iets weet — een paar bekende exporteurs, een paar regionale banken, een paar
smallcaps — draai `screener.run_ticker` erop en kijk of de fair values ergens op
slaan. Dat kost een uur en bepaalt of je 3.700 tickers wilt importeren of eerst
je sectortabel wilt aanpassen.

---

## Check 5 — Deduplicatie leunt op ISIN, en die heb je straks niet

`dubbelingen.py` matcht op ISIN als die er is, en anders op genormaliseerde
bedrijfsnaam. In je eigen docstring staat dat ISIN *"bij ongeveer de helft van de
aandelen niet gevuld"* is.

De naamnormalisatie strijkt rechtsvormen weg: ` ab (publ)`, ` a/s`, ` s.p.a.`,
` oyj`, ` asa`. Dat is een Europese lijst. Voor Japanse, Koreaanse en Hongkongse
namen doet hij niets zinnigs — Yahoo romaniseert Japanse bedrijfsnamen bovendien
inconsistent.

Wat dat per markt betekent:

| Markt | ISIN in de officiële lijst? | Duplicatierisico |
|---|---|---|
| VK (LSE Instrument list) | **ja** | hoog — `.XC`/`.IL` zijn buitenlandse lijnen |
| Hongkong (HKEX) | **ja** | hoog — A/H-noteringen |
| Japan (JPX data_j.xls) | **nee** | laag — weinig Japanse dubbelnoteringen |
| Korea (KRX) | via KIND, moeilijk te scrapen | midden |

Voor Japan is het ontbreken van ISIN dus waarschijnlijk acceptabel. Voor
Hongkong niet — daar heb je hem, gebruik hem.

Let ook op: `dubbelingen.py` **markeert alleen, filtert niet**. Dat is een bewuste
keuze en bij 2.812 tickers prima. Bij 12.000 wordt je top-15 een lijst waarin
hetzelfde bedrijf drie keer staat — zoals nu al met Silvano Fashion Group op
plek 3 en 4. Overweeg bij de kansenlijst wél te ontdubbelen, ook al doe je het
elders niet.

---

## Check 6 — Sectordekking meten, niet aannemen

Je hebt dit al één keer aan den lijve gehad: de GICS-namen in `config.yaml`
lieten 617 aandelen (22%) stilletjes terugvallen op `Default`, en dat blaast fair
values op. `test_sector_config.py` bewaakt nu de namen.

Wat die test níet bewaakt is **dekking**. Je hebt nu 156 aandelen met sector
"Onbekend". Yahoo geeft wereldwijd dezelfde sectornamen terug, dus de namen
blijven kloppen — maar of Yahoo voor een Japanse smallcap überhaupt een sector
teruggeeft, is een andere vraag.

Meet dat op een steekproef vóór de import, niet erna. En let op de valkuil die
mij bij dit onderzoek een hele meetronde kostte: **gebruik één gedeelde
HTTP-sessie voor alle Yahoo-aanroepen.** Per-thread sessies geven HTTP 401 op
`.info` — je krijgt dan lege sector en valuta terug terwijl `.financials` gewoon
doorkomt. Dat leest als ontbrekende data terwijl het een authenticatiefout is.

---

## Volgorde

1. **`gaps_analyze.py` draaien** — wat zijn die 713? *(half uur)*
2. **`INSUFFICIENT DATA` opsplitsen** in drie zichtbare toestanden *(middag)*
3. **20 Japanse namen handmatig door het model** — slaan de fair values ergens
   op? *(uur)*
4. Op basis van 3: sectortabel aanpassen, óf observatiestand inbouwen
5. **VK importeren** met het `^0[A-Z0-9]{3}\.L`-filter — kleinste stap, en meteen
   een test van je pijplijn
6. **Meten wat de nachtverversing doet** voordat je verder gaat
7. **Japan**, alleen Prime en Standard

Stap 1 tot en met 4 gaan over de screener die je al hebt. Ze maken hem beter,
ook als je uiteindelijk besluit géén land toe te voegen. Dat is de reden om ze
eerst te doen: ze zijn niet verspild als de rest afvalt.

En raak `fundamentals_per_night` pas aan als 1 en 2 gedaan zijn. Nu zou je alleen
harder aan een rotatie trekken waarvan je niet weet wat eruit komt.

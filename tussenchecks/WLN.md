# Tussencheck: WLN — Worldline SA

**Oordeel: OVERSLAAN**
Datum: 2026-08-07 · Koers €13,28 · Beurswaarde €750 mln · Euronext Parijs

## Waarom

**Dit aandeel staat in Kansen op een kapot getal, en dat is de belangrijkste
uitkomst van deze check.**

Worldline verwerkt betalingen voor winkels, banken en overheden. De screener
zet het op de tiende plaats van de kansenlijst met **89,9% korting op de
modelwaarde** — de grootste van de hele lijst — bij een vertrouwen dat op
**hoog** staat en een spreiding van slechts 17,2%. Dat is precies het profiel
waar je op af zou gaan.

Het klopt niet. De modelwaarde van €131,13 komt tot stand uit twee methodes,
want de andere twee vielen af: de winst per aandeel is −€20,41 en de EBITDA is
−€5.650 mln. Wat overblijft leunt op de boekwaarde, en daar zit de fout. De
screener rekent met een eigen vermogen van €4.042 mln op **56,5 miljoen
aandelen**, wat neerkomt op €71,55 boekwaarde per aandeel tegen een koers van
€13,28. Yahoo's eigen koers/boekwaarde van **0,0266** impliceert daar
bovenop een boekwaarde van €499 per aandeel — een eigen vermogen van
**€28 miljard** voor een bedrijf met een beurswaarde van €750 miljoen.

Die twee getallen spreken elkaar met een factor zeven tegen, en allebei kunnen
ze niet waar zijn. Het aandelenaantal is bovendien vrijwel zeker fout:
Worldline heeft er ruim vier keer zoveel uitstaan dan de 56,5 miljoen waarmee
hier gerekend wordt.

Los daarvan is er ook zonder rekenfout niets te zoeken. Het bedrijf boekte in
2025 een **verlies van €5.157 mln** en zag zijn eigen vermogen in één jaar
halveren van €9.222 mln naar €4.042 mln. Drie jaar op rij verlies, een
rendement op kapitaal met een mediaan van **−3,3%** en een dieptepunt van
−55,6%, een operationele marge van −127,4% in het laatste jaar, en een omzet
die met 2,6% per jaar krimpt. Het cijferprofiel staat op rood met de reden
"verdient geen rendement op het geïnvesteerde kapitaal", en de datakwaliteit
op *warning*. De veertien analisten die het volgen staan op hold met een
koersdoel van €11,81 — **onder** de huidige koers.

Een bedrijf dat drie jaar verlies draait en waarvan de waardering op een
onmogelijke boekwaarde rust, is geen kans. **OVERSLAAN**, en de melding hoort
naar de screener terug.

## Wat de cijfers zeggen

| | |
|---|---|
| Rendement op kapitaal | mediaan **−3,3%**, laagste jaar −55,6% · moat: **rood** |
| Operationele marge | 7,7% → −19,0% → −7,2% → **−127,4%** |
| Nettoresultaat | 2023: −€817 mln · 2024: −€297 mln · **2025: −€5.157 mln** |
| Eigen vermogen | €9.564 mln → €9.222 mln → **€4.042 mln** — in één jaar gehalveerd |
| Omzetgroei | **−2,6% per jaar** (waarschuwing: mogelijke waardeval) |
| Korting op fair value | 89,9% — **onbruikbaar, zie hierboven** |
| Analistenconsensus | hold, koersdoel €11,81 (14 analisten) — **onder** de koers |
| Kwaliteitsscore | **4,0/10** · Piotroski 5/9 |
| Datakwaliteit | **warning** — structureel verlies, EV inconsistent (factor 3,10x) |

### Drie getallen die elkaar tegenspreken

| Bron | Impliceert |
|---|---|
| Jaarrekening in de screener | eigen vermogen €4.042 mln op 56,5 mln aandelen → **€71,55 per aandeel** |
| Yahoo's koers/boekwaarde 0,0266 | €13,28 / 0,0266 → **€499 per aandeel**, dus €28 mrd eigen vermogen |
| Beurswaarde / eigen vermogen | 750 / 4.042 → koers/boekwaarde **0,186** |

Geen twee van deze drie zijn met elkaar te rijmen. Het aandelenaantal van 56,5
miljoen is daarbij het meest verdachte cijfer.

## Wat hiermee moet gebeuren

Dit is geen inhoudelijk oordeel maar een datamelding, en het verdient
opvolging in de screener zelf:

- **De datapoort laat dit door.** `data_status` staat op *warning*, niet op
  *bad*, dus de fair value wordt gewoon berekend en het aandeel belandt in
  Kansen. Bij drie jaar aaneengesloten verlies en twee weggevallen
  FV-methodes is een vertrouwensoordeel "hoog" niet houdbaar.
- **Controleer het aandelenaantal tegen de jaarrekening.** Klopt de 56,5
  miljoen niet, dan zijn beurswaarde, boekwaarde per aandeel en winst per
  aandeel alle drie fout — en die drie voeden vrijwel elke maatstaf.
- **Overweeg een regel**: als er minder dan drie FV-methodes overblijven én de
  laatste twee jaren verlies laten zien, mag `fv_confidence` niet op hoog
  uitkomen.

## Voorspelling

Zou je hem tóch volledig laten analyseren, dan verwacht ik **PASS**, en wel op
de eerste stap: `METHODE.md` eist bruikbare cijfers over vijf jaar, en drie
verliesjaren met een afboeking van vijf miljard maken elke genormaliseerde
kasstroom een gok.

Dit is de eerste tussencheck van de reeks die niet over een bedrijf gaat maar
over de screener. Dat maakt hem niet minder nuttig: één vals positief bovenaan
Kansen kost net zoveel onderzoekstijd als een echte kandidaat.

---
Gebaseerd op opgeslagen screenerdata (Yahoo, `AGGREGATOR`-kwaliteit).
**Geen invoer voor `research/WLN.md`.**

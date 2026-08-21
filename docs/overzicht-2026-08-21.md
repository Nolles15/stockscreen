# Stockscreen — stand van zaken op 21 augustus 2026

Momentopname van de screener, het eigen onderzoek en de portefeuille.
Alle cijfers komen uit `/api/dashboard` en `/api/bezit` van
stockscreen-janco.fly.dev. Koersen in de eigen valuta van het aandeel; er wordt
nergens omgerekend.

---

## 1. De screener

2.812 actieve aandelen.

### Signalen

| Signaal | Aantal | Aandeel |
|---|---:|---:|
| STRONG BUY | 4 | 0,1% |
| BUY | 34 | 1,2% |
| HOLD | 1.458 | 51,8% |
| SELL | 603 | 21,4% |
| Geen oordeel | 713 | 25,4% |

De 713 zonder oordeel zijn grotendeels dode symbolen uit de uitbreiding naar
2.812 tickers. De archivering die op 21 augustus is gerepareerd ruimt ze de
komende weken op; dit getal hoort dus te dalen.

### Per land

| Land | Aantal | | Land | Aantal |
|---|---:|---|---|---:|
| Zweden | 422 | | Portugal | 46 |
| Polen | 406 | | Verenigd Koninkrijk | 44 |
| Duitsland | 368 | | Zwitserland | 37 |
| Frankrijk | 325 | | IJsland | 27 |
| Noorwegen | 232 | | Litouwen | 22 |
| Italië | 202 | | Estland | 19 |
| Finland | 134 | | Ierland | 15 |
| Nederland | 125 | | Oostenrijk | 12 |
| Denemarken | 114 | | Letland | 8 |
| België | 113 | | Roemenië | 8 |
| Spanje | 58 | | Griekenland | 6 |
| Verenigde Staten | 56 | | Hongarije | 5 |
| | | | Tsjechië | 5 |
| | | | Australië | 2 |
| | | | Canada | 1 |

### Per sector

| Sector | Aantal |
|---|---:|
| Industrials | 564 |
| Financial Services | 344 |
| Consumer Cyclical | 307 |
| Technology | 303 |
| Healthcare | 252 |
| Real Estate | 214 |
| Communication Services | 191 |
| Consumer Defensive | 170 |
| Onbekend | 156 |
| Basic Materials | 155 |
| Energy | 82 |
| Utilities | 74 |

---

## 2. Eigen onderzoek

62 aandelen onderzocht (2,2% van de universe): **38 volledige analyses** en
**24 tussenchecks** zonder vervolganalyse.

| Uitkomst | Aantal | Soort |
|---|---:|---|
| KOOP | 5 | volledige analyse |
| HOLD | 24 | volledige analyse |
| PASS | 9 | volledige analyse |
| VERDIEPEN | 0 | tussencheck (alle uitgediept) |
| TWIJFEL | 8 | tussencheck |
| OVERSLAAN | 16 | tussencheck |

### Wat er met die oordelen gebeurd is

| Oordeel | Totaal | Gehandeld | Bewust niet | Uitgesteld | Niet vastgelegd |
|---|---:|---:|---:|---:|---:|
| KOOP | 6 | 2 | 3 | 1 | 0 |
| HOLD | 22 | 4 | 17 | 1 | 0 |
| PASS | 9 | 1 | 8 | 0 | 0 |

Alle 37 oordelen zijn beantwoord; er staat niets meer open aan de koopkant.

---

## 3. Portefeuille

11 posities, waarvan 6 met een eigen volledige analyse.

| Ticker | Naam | Land | Koers | Conclusie | Analyse? | Verkopen boven | Bron | Afstand |
|---|---|---|---:|---|---|---:|---|---:|
| ADYEN.AS | Adyen N.V. | NL | 1.062,80 EUR | Let op, maar nog niet verkopen | ja | 1.414,04 | analyse | +33,0% |
| ARP.ST | Arctic Paper S.A. | SE | 16,75 SEK | Kijk of je these nog klopt | nee | 110,72 | model | +561,0% |
| BETS-B.ST | Betsson AB | SE | 93,25 SEK | Niets te doen | nee | 324,17 | model | +247,6% |
| CRWD | CrowdStrike Holdings | US | 194,26 USD | Kijk of je these nog klopt | nee | 90,63 | model | **−53,3%** |
| DIS | The Walt Disney Company | US | 107,36 USD | Kijk of je these nog klopt | nee | 134,55 | model | +25,3% |
| ECONB.BR | Econocom Group SE | BE | 1,48 EUR | Niets te doen | nee | 11,40 | model | +670,2% |
| KPL.WA | Kino Polska TV | PL | 19,05 PLN | Houden | ja | 42,79 | analyse | +124,6% |
| PUIG.MC | Puig Brands SA | ES | 17,12 EUR | Houden | ja | 27,31 | analyse | +59,5% |
| PZU.WA | Powszechny Zakład Ubezpieczeń | PL | 72,20 PLN | Houden | ja | 103,67 | analyse | +43,6% |
| TMUS | T-Mobile US | US | 180,86 USD | Houden | ja | 436,18 | analyse | +141,2% |
| WAWI.OL | Wallenius Wilhelmsen ASA | NO | 159,40 NOK | Let op, maar nog niet verkopen | ja | 303,12 | analyse | +90,2% |

Afstand is hoeveel de koers nog moet stijgen om de verkoopgrens te raken;
negatief betekent dat de grens al gepasseerd is.

**Verdeling:** VS 3, Polen 2, Zweden 2, Nederland 1, Noorwegen 1, België 1, Spanje 1.

**Conclusies:** 4× houden, 3× these controleren, 2× let op, 2× niets te doen.

### Aandachtspunten

- **CrowdStrike** staat 53% boven zijn verkoopgrens én heeft een afgekeurde
  concurrentiepositie. Dit is de enige positie met een openstaande vraag
  ("wat doe je hiermee?") die nog niet beantwoord is.
- **Wallenius Wilhelmsen** staat op "let op" terwijl de eigen analyse **PASS**
  concludeerde — een positie die volgens het eigen onderzoek is afgekeurd.
- **Arctic Paper (+561%) en Econocom (+670%)** hebben geen bruikbare grens: het
  generieke model kan deze aandelen niet waarderen. Een eigen analyse is hier
  het meest waard.
- Vijf posities missen nog een analyse: CrowdStrike, Disney, Arctic Paper,
  Betsson, Econocom.

---

## 4. Kansenlijst — top 15

Gerangschikt op `rank_score`: korting (50%), kwaliteit (30%) en vertrouwen in de
waardering (20%).

| # | Ticker | Naam | Sector | Land | Score | Korting | Kwaliteit | Signaal | Eigen oordeel |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 1 | TXT.WA | Text S.A. | Technology | PL | 90,1 | 46% | 9,0 | BUY | HOLD |
| 2 | WAVE.PA | Wavestone SA | Technology | FR | 86,3 | 42% | 8,0 | BUY | TWIJFEL |
| 3 | SFG.WA | AS Silvano Fashion Group | Consumer Cyclical | PL | 85,7 | 71% | 8,5 | STRONG BUY | OVERSLAAN |
| 4 | SFG1T.TL | AS Silvano Fashion Group | Consumer Cyclical | EE | 85,7 | 71% | 8,5 | STRONG BUY | OVERSLAAN |
| 5 | EQS.PA | Equasens SA | Healthcare | FR | 84,8 | 38% | 8,0 | BUY | KOOP |
| 6 | EVO.ST | Evolution AB | Consumer Cyclical | SE | 84,8 | 22% | 10,0 | HOLD | KOOP |
| 7 | BETS-B.ST | Betsson AB | Consumer Cyclical | SE | 83,9 | 50% | 6,5 | HOLD | — |
| 8 | TRUE-B.ST | Truecaller AB | Technology | SE | 83,7 | 34% | 8,0 | BUY | OVERSLAAN |
| 9 | HUG.WA | Huuuge, Inc. | Communication Services | PL | 82,7 | 60% | 8,0 | STRONG BUY | HOLD |
| 10 | CAP.PA | Capgemini SE | Technology | FR | 82,7 | 40% | 7,0 | BUY | TWIJFEL |
| 11 | VAN.BR | Van de Velde NV | Consumer Cyclical | BE | 80,3 | 26% | 8,0 | HOLD | TWIJFEL |
| 12 | SDG.PA | Synergie SE | Industrials | FR | 80,1 | 47% | 5,5 | HOLD | TWIJFEL |
| 13 | PAY.BR | Payton Planar Magnetics | Technology | BE | 79,7 | 32% | 7,0 | BUY | PASS |
| 14 | INF.PA | Infotel SA | Technology | FR | 79,7 | 29% | 10,0 | HOLD | OVERSLAAN |
| 15 | RWAY.MI | Rai Way S.p.A. | Industrials | IT | 79,5 | 10% | 10,0 | HOLD | HOLD |

Nrs. 3 en 4 zijn hetzelfde bedrijf op twee beurzen (gemarkeerd als tweede
notering). Van de top 15 zijn er 12 al onderzocht.

---

## 5. Werking van het systeem

- Koersen worden dagelijks na 18:30 opgehaald, jaarcijfers 's nachts in batches
  van 250. Op 21 augustus was 95,5% van de koersen jonger dan 24 uur.
- Analyses publiceren zichzelf: een geplande taak commit ze 's avonds, een
  GitHub Action zet ze op de site. Geen deploy nodig.
- Elk oordeel zonder bijbehorende daad verschijnt als openstaande beslissing.
  Bij een bezit geldt hetzelfde voor een geraakte harde verkoopregel.

# Groeiprofiel — ijking op het eigen universum (fase B2)

Gegenereerd 2026-10-25 09:25 met `scripts/groei_backtest.py`, data opgehaald 2026-10-25T08:20:48.

**Leeswijzer.** Per vorming is het profiel teruggerekend op de cijfers t/m dat boekjaar. *Fundament ok* = winst in FY2025 én kwaliteitsscore ≥ 6 nu. *Groei hield* = omzet groeide daarna nog ≥ 10%/jr. *Koers > mediaan* = slotkoers-op-slotkoers beter dan de mediaan van het hele geprijsde universum (geen dividend). Verdwenen tickers tellen als −50% en als fundamenteel mislukt.

**Keuzeregel (vooraf vastgelegd).** Een drempel mag rood geven als de rode groep n ≥ 20 heeft en strikt slechter scoort dan niet-rood op fundament én koers 12m. Een vorming waarin de rode groep te klein is, beslist niet mee; in de andere mag geen tegenspraak zitten. Slagen er meerdere, dan wint de **strengste** (strengste = die het meest rood geeft). Slaagt er geen, dan wordt de toets geel-only.

## Cohorten

| vorming | snelle groeiers | met koers op vorming | verdwenen | mediaan universum 12m | 24m |
|---|---:|---:|---:|---:|---:|
| F2023 (t/m FY2023, 2024-07-01) | 33 | 22 (67%) | 0 | 1.018 | 1.036 |
| F2024 (t/m FY2024, 2025-07-01) | 488 | 384 (79%) | 0 | 1.004 | – |

Uitval F2023: te weinig jaren 3426, bank/verzekeraar 563, geen snelle groeier 80, fonds 21, omzetbreuk 15, datapoort (proxy) 7
Uitval F2024: geen snelle groeier 2035, te weinig jaren 873, bank/verzekeraar 563, omzetbreuk 120, laatste boekjaar ontbreekt 37, fonds 21, datapoort (proxy) 8

## Basiskans: alle snelle groeiers

| vorming | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m |
|---|---:|---:|---:|---:|---:|---:|---:|
| F2023 | 33 | 17% (12) | 17% | 36% (22) | -11 pp | 32% (22) | -27 pp |
| F2024 | 488 | 16% (465) | 43% | 49% (384) | -1 pp | – (0) | – |

## Eindoordeel met de huidige drempels

**F2023**

| niveau | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m |
|---|---:|---:|---:|---:|---:|---:|---:|
| groen | 7 | 33% (3) | 33% | 80% (5) | +43 pp | 80% (5) | +58 pp |
| geel | 9 | 50% (2) | 0% | 14% (7) | -28 pp | 14% (7) | -49 pp |
| rood | 17 | 0% (7) | 14% | 30% (10) | -11 pp | 20% (10) | -31 pp |

| score | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≥ 7 | 18 | 29% (7) | 14% | 42% (12) | -8 pp | 42% (12) | -22 pp |
| 5 – 7 | 13 | 0% (4) | 25% | 30% (10) | -21 pp | 20% (10) | -43 pp |
| < 5 | 2 | 0% (1) | 0% | – (0) | – | – (0) | – |

**F2024**

| niveau | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m |
|---|---:|---:|---:|---:|---:|---:|---:|
| groen | 170 | 31% (167) | 50% | 51% (142) | +2 pp | – (0) | – |
| geel | 63 | 7% (58) | 29% | 52% (46) | +2 pp | – (0) | – |
| rood | 255 | 8% (240) | 42% | 46% (196) | -3 pp | – (0) | – |

| score | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≥ 7 | 263 | 25% (254) | 44% | 53% (213) | +4 pp | – (0) | – |
| 5 – 7 | 171 | 6% (162) | 42% | 45% (134) | -4 pp | – (0) | – |
| < 5 | 54 | 6% (49) | 47% | 41% (37) | -5 pp | – (0) | – |

## Drempelraster per toets

### T1 — `t1_rood` (nu 3)

| waarde | vorming | groep | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m | oordeel |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | F2023 | rood | 4 | 0% (1) | 0% | – (0) | – | – (0) | – | rood te klein (n fund 1, koers 0) |
| | | niet-rood | 29 | 18% (11) | 18% | 36% (22) | -11 pp | 32% (22) | -27 pp | onbekend 0 |
| 3 | F2024 | rood | 111 | 8% (106) | 49% | 44% (80) | -4 pp | – (0) | – | scheidt |
| | | niet-rood | 377 | 18% (359) | 42% | 50% (304) | +0 pp | – (0) | – | onbekend 0 |
| 5 | F2023 | rood | 4 | 0% (1) | 0% | – (0) | – | – (0) | – | rood te klein (n fund 1, koers 0) |
| | | niet-rood | 29 | 18% (11) | 18% | 36% (22) | -11 pp | 32% (22) | -27 pp | onbekend 0 |
| 5 | F2024 | rood | 91 | 8% (86) | 48% | 45% (65) | -4 pp | – (0) | – | scheidt |
| | | niet-rood | 397 | 18% (379) | 42% | 50% (319) | -0 pp | – (0) | – | onbekend 0 |
| 8 | F2023 | rood | 1 | – (0) | – | – (0) | – | – (0) | – | rood te klein (n fund 0, koers 0) |
| | | niet-rood | 32 | 17% (12) | 17% | 36% (22) | -11 pp | 32% (22) | -27 pp | onbekend 0 |
| 8 | F2024 | rood | 65 | 5% (62) | 47% | 39% (44) | -7 pp | – (0) | – | scheidt |
| | | niet-rood | 423 | 18% (403) | 43% | 50% (340) | +0 pp | – (0) | – | onbekend 0 |
| 12 | F2023 | rood | 0 | – (0) | – | – (0) | – | – (0) | – | rood te klein (n fund 0, koers 0) |
| | | niet-rood | 33 | 17% (12) | 17% | 36% (22) | -11 pp | 32% (22) | -27 pp | onbekend 0 |
| 12 | F2024 | rood | 37 | 6% (35) | 46% | 42% (24) | -6 pp | – (0) | – | scheidt |
| | | niet-rood | 451 | 17% (430) | 43% | 49% (360) | -1 pp | – (0) | – | onbekend 0 |

**Gekozen:** 3

### T2 — `t2_rood_gat` (nu 15)

| waarde | vorming | groep | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m | oordeel |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | F2023 | rood | 3 | 0% (1) | 0% | 0% (2) | -63 pp | 0% (2) | -89 pp | rood te klein (n fund 1, koers 2) |
| | | niet-rood | 30 | 18% (11) | 18% | 40% (20) | -9 pp | 35% (20) | -23 pp | onbekend 0 |
| 10 | F2024 | rood | 43 | 23% (39) | 56% | 32% (37) | -9 pp | – (0) | – | rood niet slechter op fundament |
| | | niet-rood | 445 | 15% (426) | 42% | 50% (347) | +1 pp | – (0) | – | onbekend 0 |
| 15 | F2023 | rood | 2 | – (0) | – | 0% (2) | -63 pp | 0% (2) | -89 pp | rood te klein (n fund 0, koers 2) |
| | | niet-rood | 31 | 17% (12) | 17% | 40% (20) | -9 pp | 35% (20) | -23 pp | onbekend 0 |
| 15 | F2024 | rood | 29 | 23% (26) | 54% | 25% (24) | -9 pp | – (0) | – | rood niet slechter op fundament |
| | | niet-rood | 459 | 16% (439) | 43% | 50% (360) | +1 pp | – (0) | – | onbekend 0 |
| 20 | F2023 | rood | 0 | – (0) | – | – (0) | – | – (0) | – | rood te klein (n fund 0, koers 0) |
| | | niet-rood | 33 | 17% (12) | 17% | 36% (22) | -11 pp | 32% (22) | -27 pp | onbekend 0 |
| 20 | F2024 | rood | 24 | 29% (21) | 62% | 21% (19) | -9 pp | – (0) | – | rood te klein (n fund 21, koers 19) |
| | | niet-rood | 464 | 16% (444) | 43% | 50% (365) | +0 pp | – (0) | – | onbekend 0 |
| 30 | F2023 | rood | 0 | – (0) | – | – (0) | – | – (0) | – | rood te klein (n fund 0, koers 0) |
| | | niet-rood | 33 | 17% (12) | 17% | 36% (22) | -11 pp | 32% (22) | -27 pp | onbekend 0 |
| 30 | F2024 | rood | 10 | 22% (9) | 56% | 25% (8) | -19 pp | – (0) | – | rood te klein (n fund 9, koers 8) |
| | | niet-rood | 478 | 16% (456) | 43% | 49% (376) | -1 pp | – (0) | – | onbekend 0 |

**Geen drempel slaagt → geel-only (kan geen rood meer geven).**

### T4 — `t4_rood_niveau` (nu 0.15)

| waarde | vorming | groep | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m | oordeel |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0.1 | F2023 | rood | 9 | 0% (2) | 0% | 14% (7) | -20 pp | 14% (7) | -34 pp | rood te klein (n fund 2, koers 7) |
| | | niet-rood | 24 | 20% (10) | 20% | 47% (15) | -9 pp | 40% (15) | -19 pp | onbekend 0 |
| 0.1 | F2024 | rood | 157 | 9% (148) | 41% | 48% (124) | -1 pp | – (0) | – | scheidt |
| | | niet-rood | 315 | 19% (304) | 44% | 49% (251) | -1 pp | – (0) | – | onbekend 16 |
| 0.15 | F2023 | rood | 15 | 0% (7) | 14% | 30% (10) | -11 pp | 20% (10) | -31 pp | rood te klein (n fund 7, koers 10) |
| | | niet-rood | 18 | 40% (5) | 20% | 42% (12) | -16 pp | 42% (12) | -20 pp | onbekend 0 |
| 0.15 | F2024 | rood | 190 | 8% (179) | 41% | 48% (151) | -2 pp | – (0) | – | scheidt |
| | | niet-rood | 282 | 21% (273) | 45% | 49% (224) | -1 pp | – (0) | – | onbekend 16 |
| 0.2 | F2023 | rood | 16 | 0% (7) | 14% | 27% (11) | -13 pp | 18% (11) | -33 pp | rood te klein (n fund 7, koers 11) |
| | | niet-rood | 17 | 40% (5) | 20% | 45% (11) | -9 pp | 45% (11) | -19 pp | onbekend 0 |
| 0.2 | F2024 | rood | 225 | 8% (213) | 41% | 50% (177) | -0 pp | – (0) | – | rood niet slechter op koers 12m |
| | | niet-rood | 247 | 23% (239) | 46% | 48% (198) | -2 pp | – (0) | – | onbekend 16 |

**Gekozen:** 0.15

### T6 — `t6_rood` (nu 15)

| waarde | vorming | groep | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m | oordeel |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 15 | F2023 | rood | 19 | 14% (7) | 14% | 23% (13) | -28 pp | 15% (13) | -49 pp | rood te klein (n fund 7, koers 13) |
| | | niet-rood | 14 | 20% (5) | 20% | 56% (9) | +8 pp | 56% (9) | +2 pp | onbekend 0 |
| 15 | F2024 | rood | 205 | 5% (192) | 37% | 47% (152) | -3 pp | – (0) | – | scheidt |
| | | niet-rood | 283 | 24% (273) | 48% | 50% (232) | +0 pp | – (0) | – | onbekend 0 |
| 25 | F2023 | rood | 14 | 0% (3) | 0% | 11% (9) | -28 pp | 0% (9) | -53 pp | rood te klein (n fund 3, koers 9) |
| | | niet-rood | 19 | 22% (9) | 22% | 54% (13) | +8 pp | 54% (13) | +2 pp | onbekend 0 |
| 25 | F2024 | rood | 128 | 4% (123) | 40% | 41% (100) | -6 pp | – (0) | – | scheidt |
| | | niet-rood | 360 | 20% (342) | 45% | 51% (284) | +2 pp | – (0) | – | onbekend 0 |
| 35 | F2023 | rood | 11 | 0% (3) | 0% | 17% (6) | -31 pp | 0% (6) | -56 pp | rood te klein (n fund 3, koers 6) |
| | | niet-rood | 22 | 22% (9) | 22% | 44% (16) | -8 pp | 44% (16) | -20 pp | onbekend 0 |
| 35 | F2024 | rood | 94 | 2% (90) | 43% | 43% (72) | -5 pp | – (0) | – | scheidt |
| | | niet-rood | 394 | 19% (375) | 43% | 50% (312) | +0 pp | – (0) | – | onbekend 0 |

**Gekozen:** 15

## Toetsen zonder raster (huidige drempel)

### T3

| vorming | groep | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m | oordeel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F2023 | rood | 1 | – (0) | – | 0% (1) | -33 pp | 0% (1) | -59 pp | rood te klein (n fund 0, koers 1) |
| | niet-rood | 32 | 17% (12) | 17% | 38% (21) | -9 pp | 33% (21) | -25 pp | onbekend 0 |
| F2024 | rood | 50 | 7% (46) | 57% | 48% (33) | -8 pp | – (0) | – | scheidt |
| | niet-rood | 435 | 17% (416) | 42% | 49% (350) | -1 pp | – (0) | – | onbekend 3 |

**Scheidt met de huidige drempel.**

### T5

| vorming | groep | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m | oordeel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F2023 | rood | 1 | – (0) | – | 0% (1) | -41 pp | 0% (1) | -84 pp | rood te klein (n fund 0, koers 1) |
| | niet-rood | 32 | 17% (12) | 17% | 38% (21) | -9 pp | 33% (21) | -25 pp | onbekend 0 |
| F2024 | rood | 19 | 12% (17) | 41% | 21% (14) | -12 pp | – (0) | – | rood te klein (n fund 17, koers 14) |
| | niet-rood | 459 | 17% (438) | 44% | 49% (363) | -1 pp | – (0) | – | onbekend 10 |

**Scheidt niet met de huidige drempel.**

### T7

| vorming | groep | n | fundament ok (n) | groei hield | koers 12m > mediaan (n) | mediaan rel. 12m | koers 24m > mediaan (n) | mediaan rel. 24m | oordeel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| F2023 | rest | 33 | 17% (12) | 17% | 36% (22) | -11 pp | 32% (22) | -27 pp | groen tegen de rest |
| | groen | 0 | – (0) | – | – (0) | – | – (0) | – | onbekend 0 |
| F2024 | rest | 421 | 15% (399) | 41% | 49% (333) | -1 pp | – (0) | – | groen tegen de rest |
| | groen | 67 | 21% (66) | 58% | 47% (51) | -9 pp | – (0) | – | onbekend 0 |

## Wat als: het huidige universum

Benadering (F-score zonder TTM-rij, zonder aandelenklasse-markering); de echte telling komt uit `/api/scores/recompute` met `dry_run`.

Gekozen drempels: `{"t1_rood": 3.0, "t1_geel": 2.0, "t4_rood_niveau": 0.15, "t6_rood": 15.0, "t6_groen": 10.0}`

| niveau | huidig | gekozen |
|---|---:|---:|
| groen | 107 | 107 |
| geel | 89 | 89 |
| rood | 250 | 250 |
| grijs | 148 | 148 |

Groen nu 18%, met de gekozen drempels 18%.

## Beperkingen

- Geen historische `data_status` en kwaliteitsscore: de poort is een proxy, de fundamentele uitkomst leunt op de kwaliteitsscore van nu (milde look-ahead op de poort, geen op de uitkomst).
- F-score alleen over boekjaren; productie telt de TTM-rij mee.
- Koers zonder dividend; verhoudingen buiten 0,01–100 zijn weggelaten als datafout.
- Eén marktregime (2024-07 → 2026-07); F2024 heeft alleen 12 maanden.
- De meeste tickers hebben pas cijfers vanaf FY2021, dus F2023 is een venster van drie jaar en F2022 is niet te vormen.

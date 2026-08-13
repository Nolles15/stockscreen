"""
oordelen.py — koppelt de uitslagen uit `analyses/` en `tussenchecks/` aan de
tickers in de screener.

Waarom dit bestaat: de screener rangschikt op cijfers, maar zodra een aandeel
onderzocht is weet jij meer dan de cijfers. Zonder deze koppeling blijft dat
onderzoek naast de database liggen. Op 4 augustus 2026 stonden er vier
onderzochte aandelen in de top twintig van Kansen — waaronder twee noteringen
van hetzelfde bedrijf dat een tussencheck op OVERSLAAN had gezet.

**Deze module verwijdert niets.** Ze hangt alleen een oordeel aan een rij, zodat
het dashboard het kan tonen. Wat er met dat oordeel gebeurt is een keuze van de
pagina, niet van de koppeling.

De rapporten blijven de enige bron: er is geen tabel en geen lijst die met de
hand bijgehouden wordt. Wat in `analyses/` en `tussenchecks/` ligt, telt.

## Koppelen

Een rapport noemt zijn eigen ticker kaal (`SFG`, `WKL`), de screener gebruikt
Yahoo-symbolen (`SFG.WA`, `WKL.AS`). Drie manieren, in deze volgorde:

1. **Yahoo-symbool** uit de metadata van een analyse. Exact, dus geen twijfel.
   SHEL noemt er drie (`SHEL.AS / SHEL.L / SHEL`); die tellen allemaal mee.
2. **Kale ticker** tegen het deel voor de punt. Alleen als er precies één
   kandidaat is; bij meer kandidaten moet ook de naam kloppen. Anders zou een
   tussencheck op `AD` (Ahold) zich aan elke `AD.*` elders hechten.
3. **Bedrijfsnaam.** Nodig, want een oordeel gaat over het bedrijf en niet over
   de notering: `SFG1T.TL` deelt geen letter met `SFG.WA` maar is hetzelfde
   bedrijf op een andere beurs. Zulke rijen krijgen `via` mee, zodat de pagina
   kan zeggen dat het oordeel op de andere notering is gemaakt.

## Verouderen

Een oordeel is een momentopname. Twee signalen zeggen dat het herzien mag
worden: het is ouder dan een jaar (`verouderd`), of de koers is er flink onder
gezakt (`koers_verschil`). Dat laatste alleen bij een directe koppeling en
alleen als de valuta klopt — de tussencheck van Silvano noteert PLN, terwijl de
Tallinnse notering van hetzelfde bedrijf in EUR staat, en dat verschil is geen
koersdaling.
"""

import re
from datetime import date, datetime

from . import analyses as analyses_mod

# Vanaf hier mag een oordeel opnieuw tegen het licht: een jaar aan
# kwartaalcijfers verder is de reden waarom je iets oversloeg misschien weg.
VEROUDERD_NA_DAGEN = 365


def _naam_key(naam: str | None) -> str | None:
    """Bedrijfsnaam vergelijkbaar maken zonder hem echt te normaliseren.

    Alleen hoofdletters en witruimte gladstrijken. Rechtsvormen weghalen
    ('Plc', 'AB (publ)') zou meer koppelen maar ook meer verkeerd koppelen, en
    een gemiste koppeling is hier goedkoper dan een verkeerde.
    """
    if not naam:
        return None
    return re.sub(r"\s+", " ", naam).strip().casefold() or None


def _kaal(ticker: str | None) -> str:
    return (ticker or "").split(".")[0].strip().upper()


def _yahoo_symbolen(waarde: str | None) -> list[str]:
    """Symbolen uit de Yahoo-bullet. SHEL noemt er drie met toelichting:
    'SHEL.AS (Amsterdam) / SHEL.L (Londen) / SHEL (NYSE)'."""
    if not waarde:
        return []
    symbolen = []
    for deel in waarde.split("/"):
        deel = re.sub(r"\([^)]*\)", "", deel).strip()
        m = re.match(r"[A-Z0-9][A-Z0-9.\-]*", deel.upper())
        if m:
            symbolen.append(m.group(0))
    return symbolen


def _rapporten() -> list[dict]:
    """Alle rapporten in koppelbare vorm, tussenchecks eerst.

    De volgorde is de voorrang: een volledige analyse overschrijft de
    tussencheck van hetzelfde bedrijf. Die tussencheck was de vraag of het
    onderzoek waard was; als de analyse er ligt is dat antwoord ingehaald.
    """
    lijst = []
    for check in analyses_mod.get_alle_tussenchecks():
        lijst.append({
            "soort": "tussencheck",
            "oordeel": check.get("oordeel"),
            "datum": check.get("datum"),
            "link": f"/tussenchecks/{check['ticker']}",
            "ticker": _kaal(check.get("ticker")),
            "naam_key": _naam_key(check.get("naam")),
            "koers_getal": check.get("koers_getal"),
            "valuta": check.get("valuta"),
            "symbolen": [],
        })
    for analyse in analyses_mod.get_all_summaries():
        lijst.append({
            "soort": "analyse",
            "oordeel": analyse.get("oordeel"),
            "datum": analyse.get("peildatum"),
            "link": f"/analyses/{analyse['ticker']}",
            "ticker": _kaal(analyse.get("ticker")),
            "naam_key": _naam_key(analyse.get("naam")),
            "koers_getal": analyse.get("koers_getal"),
            "valuta": analyse.get("valuta"),
            "symbolen": _yahoo_symbolen(analyse.get("yahoo_symbol")),
        })
    return [r for r in lijst if r["oordeel"]]


def _verouderd(datum: str | None, vandaag: date | None = None) -> bool:
    if not datum:
        return False
    try:
        gemaakt = datetime.fromisoformat(datum).date()
    except (ValueError, TypeError):
        return False
    return ((vandaag or date.today()) - gemaakt).days > VEROUDERD_NA_DAGEN


def _koers_verschil(rij: dict, rapport: dict) -> float | None:
    """Koersverandering in procenten sinds het oordeel, of None.

    None zodra er iets niet zeker is: geen koers in het rapport, een andere
    valuta, of een koppeling via de bedrijfsnaam (dan is het een andere
    notering in een andere valuta en zegt het verschil niets).
    """
    toen, nu = rapport.get("koers_getal"), rij.get("price")
    if not toen or not nu or toen <= 0:
        return None
    if not rapport.get("valuta") or rapport["valuta"] != (rij.get("currency") or "").upper():
        return None
    return round((nu - toen) / toen * 100, 1)


def verrijk(rijen: list[dict]) -> None:
    """Hang aan elke rij met een rapport een `oordeel`-blok.

    Werkt de rijen ter plekke bij; rijen zonder rapport blijven ongemoeid.
    De rijen hebben `ticker`, `name`, en voor het koersverschil `price` en
    `currency` nodig — precies wat het dashboard toch al levert.
    """
    rapporten = _rapporten()
    if not rapporten or not rijen:
        return

    per_symbool: dict[str, dict] = {}
    per_naam: dict[str, dict] = {}
    for rapport in rapporten:
        for symbool in rapport["symbolen"]:
            per_symbool[symbool] = rapport
        if rapport["naam_key"]:
            per_naam[rapport["naam_key"]] = rapport

    # id() als sleutel: de rijen zijn dicts en dus niet hashbaar, en twee rijen
    # met dezelfde inhoud bestaan hier niet (de ticker is uniek). De waarde is
    # (rapport, via, via_naam): op wélke notering het oordeel gemaakt is, en of
    # de koppeling überhaupt via de bedrijfsnaam liep.
    gekoppeld: dict[int, tuple[dict, str | None, bool]] = {}

    # 1. Exact op Yahoo-symbool.
    for rij in rijen:
        rapport = per_symbool.get((rij.get("ticker") or "").upper())
        if rapport:
            gekoppeld[id(rij)] = (rapport, None, False)

    # 2. Op de kale ticker, maar alleen als het niet dubbelzinnig is.
    per_kaal: dict[str, list[dict]] = {}
    for rij in rijen:
        per_kaal.setdefault(_kaal(rij.get("ticker")), []).append(rij)
    for rapport in rapporten:
        kandidaten = per_kaal.get(rapport["ticker"], [])
        if len(kandidaten) > 1:
            kandidaten = [r for r in kandidaten if _naam_key(r.get("name")) == rapport["naam_key"]]
        for rij in kandidaten:
            gekoppeld.setdefault(id(rij), (rapport, None, False))

    # 3. Op bedrijfsnaam — de andere noteringen van hetzelfde bedrijf. Zowel de
    #    naam uit het rapport als de naam die de screener aan de al gekoppelde
    #    notering geeft; die twee zijn niet altijd identiek gespeld.
    #
    #    `via` moet een ticker zijn die de screener kent, want de pagina noemt
    #    hem. Staat de andere notering niet in dezelfde set rijen — op een
    #    detailpagina kijken we naar één aandeel — dan blijft `via` leeg en
    #    vertelt alleen `via_naam` dát het oordeel elders gemaakt is.
    namen = dict(per_naam)
    for rij in rijen:
        koppeling = gekoppeld.get(id(rij))
        sleutel = _naam_key(rij.get("name"))
        if koppeling and sleutel and not koppeling[2]:
            namen[sleutel] = {**koppeling[0], "_bron_ticker": rij.get("ticker")}
    for rij in rijen:
        if id(rij) in gekoppeld:
            continue
        rapport = namen.get(_naam_key(rij.get("name")))
        if rapport:
            gekoppeld[id(rij)] = (rapport, rapport.get("_bron_ticker"), True)

    for rij in rijen:
        koppeling = gekoppeld.get(id(rij))
        if not koppeling:
            continue
        rapport, via, via_naam = koppeling
        rij["oordeel"] = {
            "oordeel": rapport["oordeel"],
            "soort": rapport["soort"],
            "datum": rapport["datum"],
            "link": rapport["link"],
            # Onder welke ticker het rapport zelf ligt. Nodig om er méér uit te
            # halen dan het oordeel (de verkoopregels lezen de scenario's), en
            # dat moet via dezelfde koppeling lopen als hier — een tweede
            # zoekweg naar hetzelfde rapport gaat gegarandeerd afwijken.
            "rapport_ticker": rapport["ticker"],
            "via": via,
            "via_naam": via_naam,
            "verouderd": _verouderd(rapport["datum"]),
            "koers_verschil": None if via_naam else _koers_verschil(rij, rapport),
        }


def voor_ticker(ticker: str, naam: str | None = None,
                price: float | None = None, currency: str | None = None) -> dict | None:
    """Het oordeel voor één ticker, of None.

    Loopt via dezelfde koppeling als het dashboard in plaats van een tweede
    variant — die zou stilletjes uit de pas kunnen lopen.
    """
    rij = {"ticker": ticker, "name": naam, "price": price, "currency": currency}
    verrijk([rij])
    return rij.get("oordeel")

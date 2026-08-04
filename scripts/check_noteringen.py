"""
check_noteringen.py — spoort tickers op die een vals koopsignaal geven.

Aanleiding (2026-08-04, Robertet): Robertet SA staat met twee noteringen in de
screener. RBT.PA is de gewone notering (koers 802, signaal HOLD); CBR.PA is een
investeringscertificaat met een veel kleiner aandelenkapitaal (koers 91). Yahoo
levert bij beide dezelfde groepscijfers, dus bij CBR werd de winst van het hele
bedrijf gedeeld door alleen die kleine klasse: winst per aandeel 49 tegen een
koers van 91, een koers/winst van 1,8 en een schijnbare korting van 80%. Het
dashboard zette daar een BUY neer.

Dat is gevaarlijker dan de dual-listings die CLAUDE.md al noemt (EXOR.AS,
ACOMO.BR): daar ontbreken de financials, hier zijn ze van de verkeerde entiteit.
Het resultaat oogt volkomen geloofwaardig.

Dit script zoekt drie signalen af:

1. **Dubbele bedrijfsnamen** — meerdere actieve tickers met dezelfde naam.
   Meestal een dual-listing of een tweede aandelenklasse; bij grote
   koersverschillen is er iets mis.
2. **Onmogelijk lage koers/winst** — een gezond bedrijf noteert niet op
   minder dan drie keer de winst. Zulke waarden komen vrijwel altijd uit een
   verkeerd gekoppelde winst of een valutamix.
3. **Koers/boekwaarde bijna nul** — zelfde oorzaak, andere symptoom.

Gebruik:
  python scripts/check_noteringen.py                 # tegen de live app
  python scripts/check_noteringen.py --url http://localhost:5001

Exit 0 = niets gevonden, 1 = verdachte tickers (die zijn niet per se fout,
maar verdienen een blik voordat je er een analyse op loslaat).
"""

import argparse
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STANDAARD_URL = "https://stockscreen-janco.fly.dev"

# Onder deze koers/winst is een gezond bedrijf vrijwel zeker verkeerd gekoppeld.
PE_ONDERGRENS = 3.0
# Koers/boekwaarde onder deze waarde bij een winstgevend bedrijf is even verdacht.
PB_ONDERGRENS = 0.10

# Bedrijfsnamen verschillen vaak alleen in rechtsvorm; die eraf halen voorkomt
# dat 'Robertet SA' en 'Robertet' als twee bedrijven tellen.
RECHTSVORMEN = re.compile(
    r"\b(n\.?v\.?|s\.?a\.?|a\.?g\.?|plc|ltd|limited|inc|corp|corporation|s\.?p\.?a\.?"
    r"|oyj|abp|ab|asa|as|a/s|se|kgaa|gmbh|holding|group|co)\b\.?",
    re.IGNORECASE,
)


def normaliseer(naam: str) -> str:
    naam = RECHTSVORMEN.sub("", naam or "")
    return re.sub(r"[^a-z0-9]", "", naam.lower())


def haal_dashboard(basis_url: str) -> list[dict]:
    """Haalt het dashboard op.

    Op Janco's netwerk wordt TLS onderschept, waardoor certificaatverificatie
    faalt op een zelfondertekende tussen-CA (zelfde reden waarom `fly deploy`
    de vlag `--depot=false` nodig heeft en de tussencheck-skill `curl -sk`
    gebruikt). Daarom: eerst netjes verifiëren, en alleen als dat op precies
    dat euvel stukloopt terugvallen — met een zichtbare melding, zodat een
    echt certificaatprobleem niet ongemerkt voorbijglipt.
    """
    url = f"{basis_url.rstrip('/')}/api/dashboard"
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        print("[noteringen] certificaat niet te verifiëren (TLS-onderschepping op dit "
              "netwerk) — opnieuw zonder verificatie")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=120, context=context) as r:
            data = json.loads(r.read().decode("utf-8"))
    return data if isinstance(data, list) else data.get("stocks", [])


def dubbele_noteringen(rijen: list[dict]) -> list[list[dict]]:
    per_naam = defaultdict(list)
    for rij in rijen:
        sleutel = normaliseer(rij.get("name") or "")
        if sleutel:
            per_naam[sleutel].append(rij)
    return [groep for groep in per_naam.values() if len(groep) > 1]


def verdachte_waardering(rijen: list[dict]) -> list[tuple[dict, str]]:
    gevonden = []
    for rij in rijen:
        hist = rij.get("hist_relative") or {}
        pe, pb = hist.get("current_pe"), hist.get("current_pb")
        if isinstance(pe, (int, float)) and 0 < pe < PE_ONDERGRENS:
            gevonden.append((rij, f"koers/winst {pe:.1f} — onder {PE_ONDERGRENS}"))
        elif isinstance(pb, (int, float)) and 0 < pb < PB_ONDERGRENS:
            gevonden.append((rij, f"koers/boekwaarde {pb:.3f} — onder {PB_ONDERGRENS}"))
    return gevonden


def zoek_urgent(groepen: list[list[dict]]) -> list[tuple]:
    """Groepen waarin de koers/winst uiteenloopt én er een koopsignaal staat.

    Koersen vergelijken helpt niet: een dual-listing noteert in twee valuta
    (Silvano doet 4,49 PLN in Warschau en 1,11 EUR in Tallinn — dezelfde
    waarde). De koers/winst is wél valuta-onafhankelijk. Zijn het werkelijk
    dezelfde aandelen, dan is die gelijk; loopt hij uiteen, dan hoort de winst
    niet bij die koers. Bij Robertet stond RBT op 16,8 en CBR op 1,8.
    """
    urgent = []
    for groep in groepen:
        pes = [(r, (r.get("hist_relative") or {}).get("current_pe")) for r in groep]
        pes = [(r, pe) for r, pe in pes if isinstance(pe, (int, float)) and pe > 0]
        if len(pes) < 2:
            continue
        laagste, hoogste = min(pe for _, pe in pes), max(pe for _, pe in pes)
        if hoogste / laagste < 1.5:
            continue                      # zelfde waardering, dus dezelfde aandelen
        kopers = [r for r in groep if "BUY" in (r.get("signal") or "")]
        if kopers:
            urgent.append((groep, kopers, laagste, hoogste))
    return urgent


def zelftest() -> int:
    """Controleert de detectie op de casus waarvoor dit script is geschreven.

    CBR.PA is inmiddels gedeactiveerd, dus het echte dashboard meldt niets
    meer. Zonder deze test zou een stille regressie ongemerkt blijven.
    """
    def rij(ticker, naam, koers, pe, signaal, valuta="EUR"):
        return {"ticker": ticker, "name": naam, "price": koers, "signal": signaal,
                "currency": valuta, "market_cap_m": 0,
                "hist_relative": {"current_pe": pe}}

    gevallen = [
        ("Robertet (de aanleiding)", [
            rij("RBT.PA", "Robertet SA", 802, 16.8, "HOLD"),
            rij("CBR.PA", "Robertet SA", 91, 1.9, "BUY"),
        ], True),
        ("Silvano: echte dual-listing in twee valuta", [
            rij("SFG.WA", "AS Silvano Fashion Group", 4.49, 4.8, "STRONG BUY", "PLN"),
            rij("SFG1T.TL", "AS Silvano Fashion Group", 1.11, 4.8, "STRONG BUY", "EUR"),
        ], False),
        ("A/B-aandelen zonder koopsignaal", [
            rij("VOLV-A.ST", "AB Volvo (publ)", 370, 12.0, "SELL", "SEK"),
            rij("VOLV-B.ST", "AB Volvo (publ)", 369, 12.0, "SELL", "SEK"),
        ], False),
    ]

    fouten = 0
    for omschrijving, groep, verwacht in gevallen:
        gevonden = bool(zoek_urgent([groep]))
        goed = gevonden == verwacht
        fouten += not goed
        print(f"  {'v' if goed else 'x'} {omschrijving}: "
              f"{'gemeld' if gevonden else 'niet gemeld'} (verwacht: "
              f"{'gemeld' if verwacht else 'niet gemeld'})")

    print("\n[zelftest] " + ("detectie werkt." if not fouten else f"{fouten} fout(en)."))
    return 1 if fouten else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=STANDAARD_URL, help="basis-URL van de app")
    p.add_argument("--alles", action="store_true",
                   help="ook de onschuldige gevallen tonen (A/B-aandelen, dual-listings)")
    p.add_argument("--zelftest", action="store_true",
                   help="controleer de detectie op de Robertet-casus, zonder de app")
    args = p.parse_args()

    if args.zelftest:
        return zelftest()

    print(f"[noteringen] dashboard ophalen van {args.url} …")
    rijen = haal_dashboard(args.url)
    print(f"[noteringen] {len(rijen)} actieve tickers\n")

    groepen = dubbele_noteringen(rijen)
    verdacht = verdachte_waardering(rijen)
    verdacht_per_ticker = {r["ticker"]: reden for r, reden in verdacht}

    urgent = zoek_urgent(groepen)

    if urgent:
        print(f"── NAKIJKEN: tweede notering met een koopsignaal ({len(urgent)}) ──")
        print("   Dit is het patroon waar Robertet op strandde: de groepscijfers")
        print("   hangen aan een kleine aandelenklasse, dus de korting is fictie.\n")
        for groep, kopers, laag, hoog in sorted(urgent, key=lambda u: u[0][0].get("name") or ""):
            print(f"  {groep[0].get('name')}  — koers/winst loopt uiteen: {laag:.1f} tot {hoog:.1f}")
            for r in sorted(groep, key=lambda x: -(x.get("market_cap_m") or 0)):
                pe = (r.get("hist_relative") or {}).get("current_pe")
                pe_tekst = f"K/W {pe:>6.1f}" if isinstance(pe, (int, float)) else "K/W     —"
                merk = "  ← koopsignaal" if r in kopers else ""
                print(f"      {r['ticker']:12s} {r.get('currency') or '???':4s} "
                      f"koers {r.get('price')!s:>10}  {pe_tekst}  "
                      f"{r.get('signal'):12s}{merk}")
            print()

    # Koopsignalen met een onmogelijke waardering, buiten de groepen hierboven.
    al_gemeld = {r["ticker"] for groep, _, _, _ in urgent for r in groep}
    koop_verdacht = [
        (r, reden) for r, reden in verdacht
        if "BUY" in (r.get("signal") or "") and r["ticker"] not in al_gemeld
    ]
    if koop_verdacht:
        print(f"── NAKIJKEN: koopsignaal bij een onmogelijke waardering ({len(koop_verdacht)}) ──\n")
        for rij, reden in sorted(koop_verdacht, key=lambda v: v[0]["ticker"]):
            print(f"  {rij['ticker']:12s} {(rij.get('name') or '')[:34]:34s} "
                  f"{rij.get('signal'):12s} {reden}")
        print()

    # De rest is informatief: A/B-aandelen en dual-listings zijn normaal, en
    # een lage koers/winst bij een SELL of INSUFFICIENT DATA leidt nergens toe.
    rest_groepen = len(groepen) - len(urgent)
    rest_verdacht = len(verdacht) - len(koop_verdacht) - sum(
        1 for groep, _, _, _ in urgent for r in groep if r["ticker"] in verdacht_per_ticker)
    print("── Achtergrond ──")
    print(f"  {rest_groepen} bedrijven met meerdere noteringen zonder koopsignaal "
          f"(A/B-aandelen, dual-listings — normaal)")
    print(f"  {rest_verdacht} tickers met een vreemde waardering die al op HOLD/SELL/"
          f"INSUFFICIENT DATA staan")
    print("  Draai met --alles om die volledig te zien.\n")

    if args.alles:
        print("── Alle groepen met meerdere noteringen ──\n")
        for groep in sorted(groepen, key=lambda g: g[0].get("name") or ""):
            tickers = ", ".join(f"{r['ticker']} ({r.get('price')})" for r in groep)
            print(f"  {(groep[0].get('name') or '')[:44]:44s} {tickers}")
        print()
        print("── Alle vreemde waarderingen ──\n")
        for rij, reden in sorted(verdacht, key=lambda v: v[0]["ticker"]):
            print(f"  {rij['ticker']:12s} {(rij.get('name') or '')[:34]:34s} "
                  f"{rij.get('signal'):12s} {reden}")
        print()

    problemen = len(urgent) + len(koop_verdacht)
    if problemen:
        print(f"[noteringen] {problemen} geval(len) om na te kijken vóór je er een analyse op doet.")
        print("[noteringen] Fout gebleken? Deactiveren kan met:")
        print('             POST /api/stocks/bulk-deactivate  {"tickers":["X"],"reason":"…"}')
        return 1

    print("[noteringen] geen koopsignalen op verdachte noteringen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

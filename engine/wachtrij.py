"""
wachtrij.py — welk bedrijf verdient als volgende een tussencheck?

Waarom dit bestaat. De site heeft één doel: bepalen wat er als volgende
onderzocht wordt. Het tabblad Kansen deed dat maar half. Op 13 november 2026
waren van de twintig Kansen er vijftien al onderzocht en twee een tweede
notering van een bedrijf dat er al stond: de lijst leverde drie tot vijf nieuwe
namen. Bovendien rangschikt Kansen op korting tegen de eigen historie, en dat
zet per constructie krimpende bedrijven bovenaan (mediane "korting" +20% bij
krimp tegen −20% bij snelle groei) en laat sterke bedrijven die zelden goedkoop
zijn nooit bovenkomen. Zie docs/REVIEW-2026-11-eindrapport.md.

De wachtrij heeft daarom twee sporen (besluit Janco, 13 november 2026):

  1. Goedkope kwaliteit — fatsoenlijk bedrijf, flinke korting op de modelwaarde.
  2. Compounders tegen een redelijke prijs — sterk bedrijf dat groeit, waar de
     koers minder groei inprijst dan het bedrijf de afgelopen jaren liet zien.
     Hier speelt de fair value van het model geen rol.

Drie regels gelden voor beide sporen:

  - Een oordeel (tussencheck of analyse) van minder dan een jaar oud haalt het
    bedrijf uit de wachtrij. Daarna mag het terugkomen.
  - Eén regel per bedrijf: van meerdere noteringen blijft de hoogst geplaatste.
  - Krimpers staan onderaan spoor 1, gemarkeerd. Niet weggefilterd.

Het blijft een zeef. In de wachtrij staan betekent "de moeite van een
tussencheck waard", niet "koopwaardig".

Puur op meegegeven rijen, geen database of netwerk — zoals `selectie`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

Rij = dict[str, Any]

# --- Gedeelde regels ---------------------------------------------------------
OORDEEL_GELDIG_DAGEN = 365     # jonger dan dit: al onderzocht, niet in de wachtrij
KRIMP_GRENS = -0.02            # omzetgroei per jaar; zelfde grens als exit_regels
PER_SPOOR = 10
KRIMPERS_ONDERAAN = 3          # hooguit zoveel gemarkeerde krimpers onder spoor 1

# Voor banken en verzekeraars meet de kwaliteitsscore niets (de pagina toont daar
# "n.v.t.") en zijn de kasstroommaten betekenisloos. Ze staan in geen van beide
# sporen; dat is een bekende blinde vlek, geen oordeel over de sector.
BUITEN_BEREIK_SECTOREN = ("Financial Services",)

# --- Spoor 1: goedkope kwaliteit --------------------------------------------
GOEDKOOP_MIN_KWALITEIT = 6.0
GOEDKOOP_MIN_KORTING = 20.0    # procent onder de modelwaarde

# --- Spoor 2: compounders ----------------------------------------------------
COMPOUNDER_MIN_KWALITEIT = 8.0
COMPOUNDER_MIN_GROEI = 0.03    # omzetgroei per jaar: het bedrijf moet groeien
# Gerealiseerde groei boven dit niveau telt niet verder mee. Zonder deze kap
# winnen overnamegedreven sprongen (+40% per jaar) het van gestage groeiers,
# terwijl juist die sprongen het minst zeggen over wat houdbaar is.
COMPOUNDER_GROEI_KAP = 0.15
# De koers moet minder groei inprijzen dan het bedrijf (afgekapt) liet zien, met
# deze marge in procentpunten.
COMPOUNDER_MIN_MARGE = 1.0

REGELS = {
    "oordeel_geldig_dagen": OORDEEL_GELDIG_DAGEN,
    "krimp_grens": KRIMP_GRENS,
    "per_spoor": PER_SPOOR,
    "krimpers_onderaan": KRIMPERS_ONDERAAN,
    "goedkoop_min_kwaliteit": GOEDKOOP_MIN_KWALITEIT,
    "goedkoop_min_korting": GOEDKOOP_MIN_KORTING,
    "compounder_min_kwaliteit": COMPOUNDER_MIN_KWALITEIT,
    "compounder_min_groei": COMPOUNDER_MIN_GROEI,
    "compounder_groei_kap": COMPOUNDER_GROEI_KAP,
    "compounder_min_marge": COMPOUNDER_MIN_MARGE,
}


# ---------------------------------------------------------------------------
# Regels per rij
# ---------------------------------------------------------------------------

def dagen_sinds_oordeel(rij: Rij, vandaag: date) -> Optional[int]:
    """Hoe oud het oordeel op deze rij is; None als er geen (leesbaar) oordeel is."""
    datum = (rij.get("oordeel") or {}).get("datum")
    if not datum:
        return None
    try:
        return (vandaag - date.fromisoformat(str(datum)[:10])).days
    except ValueError:
        return None


def al_onderzocht(rij: Rij, vandaag: date) -> bool:
    """Ligt er een oordeel van minder dan een jaar oud?

    Een oordeel zonder leesbare datum telt als recent: liever een bedrijf een
    keer te weinig tonen dan een onderzocht bedrijf opnieuw aanbieden.
    """
    if not rij.get("oordeel"):
        return False
    dagen = dagen_sinds_oordeel(rij, vandaag)
    return dagen is None or dagen < OORDEEL_GELDIG_DAGEN


def is_krimper(rij: Rij) -> bool:
    groei = rij.get("revenue_cagr")
    return groei is not None and groei < KRIMP_GRENS


def beoordeelbaar(rij: Rij) -> bool:
    """De datapoort: dezelfde eisen als de rangscore, plus een vers boekjaar."""
    return (rij.get("data_status") in ("ok", "warning")
            and rij.get("signal") not in (None, "INSUFFICIENT DATA", "N/A")
            and rij.get("quality_score") is not None
            and not (rij.get("fy_lag") or 0)
            and rij.get("sector") not in BUITEN_BEREIK_SECTOREN)


def groeimarge(rij: Rij) -> Optional[float]:
    """Gerealiseerde omzetgroei (afgekapt) min ingeprijsde groei, in procentpunten.

    Positief: de koers rekent op minder groei dan het bedrijf de afgelopen drie
    jaar liet zien. De ingeprijsde groei komt uit de genormaliseerde kasstroom
    (mediaan van vijf jaar), die bij een groeier achterloopt; de marge valt
    daardoor eerder te laag dan te hoog uit.
    """
    groei, ingeprijsd = rij.get("revenue_cagr"), rij.get("implied_growth")
    if groei is None or ingeprijsd is None:
        return None
    return round(100.0 * min(groei, COMPOUNDER_GROEI_KAP) - ingeprijsd, 1)


def _past_goedkoop(rij: Rij) -> bool:
    return ((rij.get("quality_score") or 0) >= GOEDKOOP_MIN_KWALITEIT
            and (rij.get("margin_of_safety") or -999) >= GOEDKOOP_MIN_KORTING
            and rij.get("rank_score") is not None)


def _past_compounder(rij: Rij) -> bool:
    marge = groeimarge(rij)
    return ((rij.get("quality_score") or 0) >= COMPOUNDER_MIN_KWALITEIT
            and (rij.get("revenue_cagr") or 0) >= COMPOUNDER_MIN_GROEI
            and marge is not None and marge >= COMPOUNDER_MIN_MARGE)


# ---------------------------------------------------------------------------
# Eén regel per bedrijf
# ---------------------------------------------------------------------------

def _bedrijf(rij: Rij) -> frozenset:
    return frozenset([rij.get("ticker")] + list(rij.get("dubbel_van") or []))


def een_per_bedrijf(rijen: list[Rij]) -> tuple[list[Rij], int]:
    """Houd per bedrijf de eerste (hoogst geplaatste) notering. Volgorde blijft."""
    gezien: set = set()
    uit, weg = [], 0
    for r in rijen:
        groep = _bedrijf(r)
        if groep & gezien:
            weg += 1
            continue
        gezien |= groep
        uit.append(r)
    return uit, weg


# ---------------------------------------------------------------------------
# Het geheel
# ---------------------------------------------------------------------------

def _regel(rij: Rij, spoor: str, waarom: str) -> Rij:
    """Een kopie met de wachtrijvelden erbij; de rij in de cache blijft ongemoeid."""
    return {**rij, "wachtrij_spoor": spoor, "wachtrij_waarom": waarom,
            "wachtrij_krimper": is_krimper(rij), "groeimarge": groeimarge(rij)}


def _waarom_goedkoop(r: Rij) -> str:
    tekst = (f"Kwaliteit {r.get('quality_score'):g}/10 · "
             f"{r.get('margin_of_safety'):.0f}% onder de modelwaarde")
    if is_krimper(r):
        tekst += f" · omzet krimpt {100 * r['revenue_cagr']:.0f}% per jaar"
    return tekst


def _waarom_compounder(r: Rij) -> str:
    return (f"Kwaliteit {r.get('quality_score'):g}/10 · omzet "
            f"+{100 * r['revenue_cagr']:.0f}% per jaar · koers prijst "
            f"{r.get('implied_growth'):+.1f}% in")


def bouw(rijen: list[Rij], vandaag: Optional[date] = None) -> dict:
    """De wachtrij in twee sporen, met de telling van wat er afviel en waarom."""
    vandaag = vandaag or date.today()

    basis = [r for r in rijen if beoordeelbaar(r)]
    onderzocht = [r for r in basis if al_onderzocht(r, vandaag)]
    vrij = [r for r in basis if not al_onderzocht(r, vandaag)]

    # Spoor 1: op de bestaande rangscore (korting, kwaliteit, vertrouwen).
    goedkoop = sorted((r for r in vrij if _past_goedkoop(r)),
                      key=lambda r: -r["rank_score"])
    goedkoop, dubbel_1 = een_per_bedrijf(goedkoop)
    gezond = [r for r in goedkoop if not is_krimper(r)]
    krimpers = [r for r in goedkoop if is_krimper(r)]

    spoor_1 = ([_regel(r, "goedkoop", _waarom_goedkoop(r)) for r in gezond[:PER_SPOOR]]
               + [_regel(r, "goedkoop", _waarom_goedkoop(r)) for r in krimpers[:KRIMPERS_ONDERAAN]])

    # Spoor 2: wie al in spoor 1 past hoort daar; geen bedrijf in beide lijsten.
    al_in_1 = set()
    for r in goedkoop:
        al_in_1 |= _bedrijf(r)
    compounders = sorted(
        (r for r in vrij if _past_compounder(r) and not (_bedrijf(r) & al_in_1)),
        key=lambda r: (-(groeimarge(r) or 0), -(r.get("quality_score") or 0)))
    compounders, dubbel_2 = een_per_bedrijf(compounders)
    spoor_2 = [_regel(r, "compounder", _waarom_compounder(r)) for r in compounders[:PER_SPOOR]]

    return {
        "sporen": [
            {"code": "goedkoop", "titel": "Goedkope kwaliteit",
             "rijen": spoor_1, "kandidaten": len(gezond), "krimpers": len(krimpers)},
            {"code": "compounder", "titel": "Compounders tegen een redelijke prijs",
             "rijen": spoor_2, "kandidaten": len(compounders), "krimpers": 0},
        ],
        "tellingen": {
            "beoordeelbaar": len(basis),
            "al_onderzocht": len(onderzocht),
            "tweede_noteringen": dubbel_1 + dubbel_2,
        },
        "regels": REGELS,
        "peildatum": vandaag.isoformat(),
    }

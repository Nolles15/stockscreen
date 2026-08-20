"""
exit_regels.py — wanneer is een aandeel dat je bezit het bekijken waard?

De screener beantwoordt "wat verdient aandacht?". Daarna houdt de pijplijn op:
tussencheck, analyse, gekocht — en dan niets meer. Deze module vult dat gat. Ze
toetst een aandeel dat je bezit tegen een vaste set regels en zegt per regel
welke twee getallen tot die uitkomst leiden.

**Een geraakte regel is een onderzoeksopdracht, geen verkoopopdracht.** Dat is
geen slag om de arm maar de bedoeling: de waarde zit erin dat je nadenkt vóórdat
de koers beweegt, niet dat een script je portefeuille bestuurt.

## Wat hier bewust niet in zit

**De aankoopprijs.** Geen enkele regel kent hem. Wat jij betaald hebt is geen
eigenschap van het bedrijf, en eraan vasthouden is de bekendste manier om
winnaars te vroeg te verkopen en verliezers te lang te houden. De vraag is niet
"sta ik op winst" maar "zou ik het vandaag nog kopen".

**Een absolute kwaliteitsvloer.** De verleiding is groot om "kwaliteitsscore
onder de 5" als verkoopreden op te nemen. Dat is precies `sell_quality_floor`,
die in fase 2 is geschrapt: de mediane kwaliteitsscore over 742 Europese
aandelen is 3 en 72,5% scoort onder de 6, dus die regel zet driekwart van alles
op verkopen ongeacht de prijs. Kwaliteit telt hier daarom alleen als *daling ten
opzichte van het moment dat je het vastlegde*.

**"Koers boven het optimistische scenario van de screener".** Stond in het
oorspronkelijke plan als hárde regel en is na narekenen geschrapt. De multiples
wegen 60% mee en bewegen niet mee met de scenario's; alleen Graham en de
perpetuity schuiven op. Gemeten over de sectorprofielen ligt `optimistic_fv`
maar 8% (Technology) tot 27% (Consumer Defensive) boven `base_fv`. Die regel
vuurt dus rond 108-127% van de fair value — losser dan de SELL-drempel van 130%
die de screener zelf hanteert, en als hard signaal zou hij een groot deel van
een normale portefeuille rood kleuren. Wat ervoor in de plaats komt (A2) gebruikt
de drempel die al gekalibreerd is.

## Dubbeltellen

Het eindoordeel telt geraakte regels. Twee regels die hetzelfde feit meten
zouden één waarneming tot "meerdere regels geraakt" opblazen. Daarom is er één
regel voor de bedrijfskant-instorting (`moat_rood`) die het gekalibreerde
oordeel van [moat_profile.py](moat_profile.py) overneemt, in plaats van losse
regels voor ROIC-hoogte en margeërosie — dat zijn precies de criteria waaruit
dat oordeel al is opgebouwd.

## Momentopname

Bij het vastleggen van bezit wordt een `snapshot` bewaard: wat de cijfers toen
zeiden. De regels B2, B3 en B5 vergelijken daarmee. Op dag één is die
momentopname gelijk aan vandaag, dus die regels kúnnen dan niet afgaan — dat is
geen fout maar de reden dat de absolute regels er ook zijn. De ene helft werkt
meteen, de andere wordt bruikbaar naarmate je het aandeel langer houdt.

## Zuiverheid

Geen database, geen netwerk, geen configuratiebestand lezen: alles komt via
argumenten binnen. Zelfde opzet als `moat_profile.bouw_profiel`, zodat de
drempels los te testen zijn en er maar één plek is waar ze staan.
"""

from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Drempels
#
# Waar een drempel al elders bestaat wordt hij daar vandaan gehaald in plaats
# van overgeschreven. Twee plekken met hetzelfde getal lopen gegarandeerd uit
# de pas — dat is in dit project al gebeurd met de beurssuffixen en met de
# methodologie in settings.html.
# ---------------------------------------------------------------------------

# Hoeveel procentpunt het rendement op kapitaal mag zakken ten opzichte van het
# moment van vastleggen voordat het een these-breuk heet. Vijf punten is een
# derde van de grens waarboven een moat aannemelijk heet (ROIC_GROEN = 15%);
# minder dan dat is jaarruis, meer dan dat is een ander bedrijf.
ROIC_DALING_PP = 5.0

# Kwaliteitsscore loopt van 0 tot 10 en beweegt traag; twee punten eraf is geen
# ruis maar een verschoven fundament.
KWALITEIT_DALING_PUNTEN = 2.0

# Vanaf twee boekjaren achterstand zegt de vergelijking niets meer: je toetst
# dan een these tegen cijfers van voor de vorige jaarrekening.
FY_LAG_GEEN_OORDEEL = 2

# Omzetkrimp telt pas vanaf 2% per jaar. Dezelfde grens die `screener.run_ticker`
# gebruikt voor de waarschuwing "mogelijke value trap"; een krimp van een half
# procent is afronding in de omzetreeks, geen these-breuk.
OMZETKRIMP_GRENS = -0.02

# Springt de omzet tussen twee opeenvolgende jaren meer dan deze factor, dan is
# er hoogstwaarschijnlijk van definitie gewisseld en meet de meerjarige groei
# een boekhoudkundige wissel in plaats van het bedrijf.
OMZETBREUK_FACTOR = 2.0

# `rank_score` van het bezit ten opzichte van de kop van Kansen. Informatief,
# telt nooit mee in het eindoordeel.
RANK_FACTOR = 0.5

FAMILIES = {
    "waardering": "Te duur geworden",
    "these": "These gebroken",
    "analyse": "Analyse ingehaald",
    "alternatief": "Beter alternatief",
}


def _regel(rid: str, familie: str, naam: str, *, geraakt: Optional[bool],
           uitleg: str, waarden: dict, hard: bool = False) -> dict:
    """Eén regeluitslag. `geraakt=None` betekent: niet te toetsen."""
    return {
        "id": rid,
        "familie": familie,
        "familie_label": FAMILIES[familie],
        "naam": naam,
        "geraakt": geraakt,
        "hard": hard,
        "uitleg": uitleg,
        "waarden": waarden,
    }


def _getal(waarde) -> Optional[float]:
    try:
        return float(waarde) if waarde is not None else None
    except (TypeError, ValueError):
        return None


def omzetbreuk(annual: Optional[list]) -> Optional[str]:
    """Zit er een definitiewissel in de omzetreeks? Geeft de uitleg terug, of None.

    Aanleiding: Adyen (13 augustus 2026). Yahoo gaf voor boekjaar 2022 een omzet
    van 8.936 miljoen en voor 2023 van 1.863 miljoen — een sprong van bijna een
    factor vijf, niet omdat het bedrijf instortte maar omdat de reeks overging
    van **bruto** omzet (inclusief doorbetaalde kaartkosten) naar **netto**
    omzet. De driejaars-CAGR las dat als 33% krimp per jaar, terwijl Adyen in
    werkelijkheid met zo'n 19% per jaar groeide. Twee verkoopregels gingen
    daardoor af op een boekhoudkundige wissel.

    Een echte omzethalvering in één jaar bestaat, maar is zeldzaam genoeg dat
    "geen uitspraak doen" hier het goedkopere alternatief is: een gemiste
    krimpmelding kost je een kwartaal, een valse kost je het vertrouwen in de
    hele lijst. De brutowinst en EBIT liepen bij Adyen wél netjes door, dus de
    rest van het oordeel blijft gewoon staan.
    """
    if not annual:
        return None
    reeks = [(r.get("fiscal_year"), _getal(r.get("revenue")))
             for r in annual if r.get("period_type", "annual") == "annual"]
    reeks = [(j, w) for j, w in reeks if j and w and w > 0]
    reeks.sort()
    for (jaar_a, omzet_a), (jaar_b, omzet_b) in zip(reeks, reeks[1:]):
        factor = max(omzet_a, omzet_b) / min(omzet_a, omzet_b)
        if factor >= OMZETBREUK_FACTOR:
            return (f"De omzetreeks springt van {omzet_a / 1e6:.0f} mln ({jaar_a}) naar "
                    f"{omzet_b / 1e6:.0f} mln ({jaar_b}) — een factor {factor:.1f}. "
                    f"Dat is vrijwel zeker een wisseling van definitie, dus de meerjarige "
                    f"omzetgroei zegt hier niets.")
    return None


# ---------------------------------------------------------------------------
# De datapoort
# ---------------------------------------------------------------------------


def datapoort(rij: dict) -> Optional[str]:
    """Reden om helemaal geen oordeel te vellen, of None.

    Dezelfde poort als in `screener.run_ticker` en `_effective_signal`: elke
    afleiding die de motor overdoet moet dezelfde poorten passeren. Een
    verkoopsignaal op afgekeurde data is precies de fout die dit project al
    twee keer heeft gemaakt — negentien afgekeurde aandelen met een vers
    HOLD/SELL erop was er één van.
    """
    status = rij.get("data_status")
    if status in ("bad", "missing"):
        return (f"De screener beoordeelt de data als '{status}'. Zolang dat zo is "
                f"zegt geen enkele verkoopregel iets.")
    lag = rij.get("fy_lag") or 0
    if lag >= FY_LAG_GEEN_OORDEEL:
        return (f"Het nieuwste boekjaar loopt {lag} jaar achter. Een these toetsen "
                f"tegen verouderde jaarcijfers levert een oordeel op over het verleden.")
    if not rij.get("price"):
        return "Geen koers bekend, dus niets te vergelijken."
    return None


# ---------------------------------------------------------------------------
# A — Te duur geworden
# ---------------------------------------------------------------------------


def _waardering(rij: dict, config: dict, breuk: Optional[str] = None) -> list[dict]:
    sig_cfg = (config or {}).get("signals", {})
    val_cfg = (config or {}).get("valuation", {})
    hard_pct = sig_cfg.get("sell_pct_high_quality", 175)
    max_g = val_cfg.get("max_perpetuity_growth", 5)

    pvf = _getal(rij.get("price_vs_fv_pct"))
    signaal = rij.get("signal")
    groei_ingeprijsd = _getal(rij.get("implied_growth"))
    # Bij een definitiewissel in de omzetreeks is de gerealiseerde groei geen
    # bruikbare maatstaf meer; A3 heeft hem nodig en zwijgt dan.
    groei_echt = None if breuk else _getal(rij.get("revenue_cagr"))
    if groei_echt is not None:
        groei_echt *= 100.0   # revenue_cagr staat als fractie in de rij

    regels = [
        _regel(
            "A1", "waardering", "Verkoopsignaal van de screener",
            geraakt=None if not signaal or signaal == "N/A" else signaal == "SELL",
            uitleg=("De screener zet dit op SELL: de koers ligt boven de drempel waar "
                    "'te duur' begint (130% van de fair value, 175% voor compounders)."
                    if signaal == "SELL" else
                    f"De screener zegt {signaal or 'niets'} — niet te duur volgens de eigen drempels."),
            waarden={"signaal": signaal, "koers_vs_fv_pct": pvf},
        ),
        _regel(
            "A2", "waardering", "Ver boven de fair value",
            geraakt=None if pvf is None else pvf >= hard_pct,
            hard=True,
            uitleg=(f"De koers staat op {pvf:.0f}% van de fair value. Zelfs met de ruimte "
                    f"die compounders krijgen ({hard_pct:.0f}%) is dit te duur."
                    if pvf is not None and pvf >= hard_pct else
                    f"De koers staat op {pvf:.0f}% van de fair value, onder de {hard_pct:.0f}% "
                    f"waarboven ook een compounder te duur heet."
                    if pvf is not None else "Geen fair value om de koers tegen te leggen."),
            waarden={"koers_vs_fv_pct": pvf, "grens_pct": hard_pct},
        ),
    ]

    # De omgekeerde som: welke eeuwigdurende groei rechtvaardigt deze koers? Boven
    # de bovengrens die het model zelf hanteert (max_perpetuity_growth) prijst de
    # markt iets in dat het model niet eens mág aannemen. Dat alleen is niet
    # genoeg — een groeier mag dat verdienen — dus de tweede eis is dat de
    # ingeprijsde groei ook boven de gerealiseerde omzetgroei ligt.
    if groei_ingeprijsd is None or breuk:
        regels.append(_regel(
            "A3", "waardering", "Ingeprijsde groei onhaalbaar", geraakt=None,
            uitleg=(breuk if breuk else
                    "Geen ingeprijsde groei te berekenen (geen positieve genormaliseerde winst)."),
            waarden={"ingeprijsd_pct": groei_ingeprijsd, "gerealiseerd_pct": None},
        ))
    else:
        boven_model = groei_ingeprijsd > max_g
        boven_eigen = groei_echt is not None and groei_ingeprijsd > groei_echt
        geraakt = bool(boven_model and boven_eigen)
        regels.append(_regel(
            "A3", "waardering", "Ingeprijsde groei onhaalbaar",
            geraakt=geraakt if groei_echt is not None else None,
            uitleg=(f"De koers prijst {groei_ingeprijsd:.1f}% groei per jaar in, voor altijd. "
                    f"Het bedrijf groeide de laatste drie jaar {groei_echt:.1f}% per jaar, en "
                    f"boven {max_g}% rekent het model zelf niet meer."
                    if geraakt else
                    f"Ingeprijsd {groei_ingeprijsd:.1f}% per jaar" +
                    (f", gerealiseerd {groei_echt:.1f}%." if groei_echt is not None
                     else " — geen omzetgroei bekend om het tegen te leggen.")),
            waarden={"ingeprijsd_pct": groei_ingeprijsd, "gerealiseerd_pct": groei_echt,
                     "model_max_pct": max_g},
        ))
    return regels


# ---------------------------------------------------------------------------
# B — These gebroken
# ---------------------------------------------------------------------------


def _these(rij: dict, snapshot: Optional[dict], moat: Optional[dict],
           breuk: Optional[str] = None) -> list[dict]:
    snap = snapshot or {}
    moat = moat or {}

    niveau = moat.get("niveau")
    roic_nu = _getal(moat.get("roic_mediaan"))
    roic_toen = _getal(snap.get("roic_mediaan"))
    kwal_nu = _getal(rij.get("quality_score"))
    kwal_toen = _getal(snap.get("quality_score"))
    cagr_nu = _getal(rij.get("revenue_cagr"))
    fcf_nu = _getal(rij.get("normalized_fcf_m"))
    fcf_toen = _getal(snap.get("normalized_fcf_m"))

    regels = []

    # Het moat-oordeel is al gekalibreerd op zestien aandelen waarvan de volledige
    # analyse bekend is, met als harde eis dat geen KOOP rood wordt. Dat oordeel
    # hier overnemen is beter dan de onderdelen ervan (ROIC-hoogte,
    # standvastigheid, margeërosie) als losse regels herhalen: dan zou één
    # waarneming als meerdere geraakte regels tellen.
    regels.append(_regel(
        "B1", "these", "Concurrentiepositie afgekeurd",
        geraakt=None if niveau in (None, "grijs") else niveau == "rood",
        uitleg=(f"Het moat-profiel staat op rood: {moat.get('kop') or 'het rendement op kapitaal houdt geen stand'}."
                if niveau == "rood" else
                f"Moat-profiel {niveau or 'onbekend'}" +
                (f": {moat.get('kop')}" if moat.get("kop") else ".")),
        waarden={"niveau": niveau, "roic_mediaan": roic_nu,
                 "brutomarge_trend_pp": _getal(moat.get("brutomarge_trend_pp"))},
    ))

    verschil = (roic_toen - roic_nu) if (roic_nu is not None and roic_toen is not None) else None
    regels.append(_regel(
        "B2", "these", "Rendement op kapitaal ingezakt",
        geraakt=None if verschil is None else verschil >= ROIC_DALING_PP,
        uitleg=(f"Rendement op kapitaal zakte van {roic_toen:.1f}% naar {roic_nu:.1f}% "
                f"({verschil:.1f} punt) sinds je dit vastlegde."
                if verschil is not None and verschil >= ROIC_DALING_PP else
                f"Rendement op kapitaal {roic_nu:.1f}%, was {roic_toen:.1f}% bij vastleggen."
                if verschil is not None else
                "Nog geen vergelijking mogelijk — geen rendement bij vastleggen bewaard."),
        waarden={"nu_pct": roic_nu, "bij_vastleggen_pct": roic_toen,
                 "daling_pp": round(verschil, 1) if verschil is not None else None,
                 "grens_pp": ROIC_DALING_PP},
    ))

    daling = (kwal_toen - kwal_nu) if (kwal_nu is not None and kwal_toen is not None) else None
    regels.append(_regel(
        "B3", "these", "Kwaliteitsscore gezakt",
        geraakt=None if daling is None else daling >= KWALITEIT_DALING_PUNTEN,
        uitleg=(f"Kwaliteit zakte van {kwal_toen:.1f} naar {kwal_nu:.1f} van de 10."
                if daling is not None and daling >= KWALITEIT_DALING_PUNTEN else
                f"Kwaliteit {kwal_nu:.1f}/10, was {kwal_toen:.1f} bij vastleggen."
                if daling is not None else
                "Nog geen vergelijking mogelijk — geen kwaliteitsscore bij vastleggen bewaard."),
        waarden={"nu": kwal_nu, "bij_vastleggen": kwal_toen,
                 "daling": round(daling, 1) if daling is not None else None,
                 "grens": KWALITEIT_DALING_PUNTEN},
    ))

    regels.append(_regel(
        "B4", "these", "Omzet krimpt",
        geraakt=None if (cagr_nu is None or breuk) else cagr_nu < OMZETKRIMP_GRENS,
        uitleg=(breuk if breuk else
                f"De omzet kromp de laatste drie jaar met {abs(cagr_nu) * 100:.1f}% per jaar."
                if cagr_nu is not None and cagr_nu < OMZETKRIMP_GRENS else
                f"Omzetgroei {cagr_nu * 100:.1f}% per jaar over drie jaar."
                if cagr_nu is not None else "Geen omzetgroei over drie jaar bekend."),
        waarden={"cagr_pct": round(cagr_nu * 100, 1) if cagr_nu is not None else None,
                 "grens_pct": OMZETKRIMP_GRENS * 100,
                 "reeksbreuk": bool(breuk)},
    ))

    # Alleen een omslag telt. Een bedrijf dat al negatieve kasstroom had toen je
    # het kocht is geen nieuw feit — dat wist je.
    if fcf_nu is None or fcf_toen is None:
        kasstroom = None
    else:
        kasstroom = fcf_nu <= 0 < fcf_toen
    regels.append(_regel(
        "B5", "these", "Vrije kasstroom omgeslagen",
        geraakt=kasstroom,
        uitleg=(f"De genormaliseerde vrije kasstroom is negatief ({fcf_nu:.0f} mln), "
                f"terwijl hij bij vastleggen nog {fcf_toen:.0f} mln was."
                if kasstroom else
                f"Vrije kasstroom {fcf_nu:.0f} mln." if fcf_nu is not None else
                "Geen genormaliseerde vrije kasstroom bekend."),
        waarden={"nu_mln": fcf_nu, "bij_vastleggen_mln": fcf_toen},
    ))
    return regels


# ---------------------------------------------------------------------------
# C — Analyse ingehaald
# ---------------------------------------------------------------------------


def _analyse_regels(rij: dict, analyse: Optional[dict],
                    oordeel: Optional[dict]) -> list[dict]:
    """De C-regels leggen de koers langs de waardeniveaus uit het eigen rapport.

    **Alleen bij gelijke valuta en een directe koppeling.** Een rapport rekent in
    de valuta waarin het aandeel noteerde op de peildatum; hangt het oordeel via
    de bedrijfsnaam aan een andere notering, dan is dat een andere valuta en zegt
    de vergelijking niets. Silvano noteert 4,49 PLN in Warschau en 1,11 EUR in
    Tallinn — dezelfde waarde. Dit is dezelfde afweging die `oordelen._koers_verschil`
    maakt en om dezelfde reden.
    """
    koers = _getal(rij.get("price"))
    valuta_rij = (rij.get("currency") or "").upper()
    valuta_rap = (analyse or {}).get("valuta")
    via_naam = bool((oordeel or {}).get("via_naam"))

    bruikbaar = bool(analyse) and bool(koers) and not via_naam \
        and bool(valuta_rap) and valuta_rap.upper() == valuta_rij

    scenarios = (analyse or {}).get("scenarios") or {}
    kansgewogen = _getal((analyse or {}).get("fair_value_kansgewogen"))
    optimistisch = _getal(scenarios.get("optimistisch"))

    if not bruikbaar:
        reden = ("Geen eigen analyse voor dit aandeel." if not analyse else
                 "Het oordeel is op een andere notering gemaakt, in een andere valuta."
                 if via_naam else
                 f"Het rapport rekent in {valuta_rap or 'onbekende valuta'}, de koers staat in "
                 f"{valuta_rij or 'onbekende valuta'} — niet vergelijkbaar.")
        leeg = {"koers": koers, "valuta_koers": valuta_rij, "valuta_rapport": valuta_rap}
        return [
            _regel("C1", "analyse", "Voorbij de kansgewogen waarde", geraakt=None,
                   uitleg=reden, waarden=leeg),
            _regel("C2", "analyse", "Voorbij het optimistische scenario", geraakt=None,
                   hard=True, uitleg=reden, waarden=leeg),
            _regel("C3", "analyse", "Analyse verouderd",
                   geraakt=bool((oordeel or {}).get("verouderd")) if oordeel else None,
                   uitleg=("De analyse is ouder dan een jaar; een jaar aan kwartaalcijfers "
                           "verder kan de reden waarom je kocht weg zijn."
                           if (oordeel or {}).get("verouderd") else reden),
                   waarden={"datum": (oordeel or {}).get("datum")}),
            _oordeel_tegenspraak(oordeel),
        ]

    return [
        _regel(
            "C1", "analyse", "Voorbij de kansgewogen waarde",
            geraakt=None if kansgewogen is None else koers >= kansgewogen,
            uitleg=(f"De koers ({koers:.2f}) staat op of boven de kansgewogen waarde uit je "
                    f"eigen analyse ({kansgewogen:.2f} {valuta_rap})."
                    if kansgewogen is not None and koers >= kansgewogen else
                    f"Koers {koers:.2f} tegen kansgewogen waarde {kansgewogen:.2f} {valuta_rap}."
                    if kansgewogen is not None else
                    "Het rapport noemt geen kansgewogen waarde."),
            waarden={"koers": koers, "kansgewogen": kansgewogen, "valuta": valuta_rap},
        ),
        _regel(
            "C2", "analyse", "Voorbij het optimistische scenario", hard=True,
            geraakt=None if optimistisch is None else koers >= optimistisch,
            uitleg=(f"De koers ({koers:.2f}) staat op of boven het optimistische scenario van "
                    f"je eigen analyse ({optimistisch:.2f} {valuta_rap}). De markt betaalt meer "
                    f"dan jouw gunstigste aanname."
                    if optimistisch is not None and koers >= optimistisch else
                    f"Koers {koers:.2f} tegen optimistisch scenario {optimistisch:.2f} {valuta_rap}."
                    if optimistisch is not None else
                    "Het rapport noemt geen optimistisch scenario."),
            waarden={"koers": koers, "optimistisch": optimistisch, "valuta": valuta_rap},
        ),
        _regel(
            "C3", "analyse", "Analyse verouderd",
            geraakt=bool((oordeel or {}).get("verouderd")) if oordeel else None,
            uitleg=(f"De analyse dateert van {(oordeel or {}).get('datum') or 'onbekend'} en is "
                    f"ouder dan een jaar."
                    if (oordeel or {}).get("verouderd") else
                    f"Analyse van {(oordeel or {}).get('datum') or 'onbekende datum'}."),
            waarden={"datum": (oordeel or {}).get("datum")},
        ),
        _oordeel_tegenspraak(oordeel),
    ]


def _oordeel_tegenspraak(oordeel: Optional[dict]) -> dict:
    """Je bezit iets dat je eigen onderzoek afraadde.

    Dat is geen rekenfout maar wel iets om onder ogen te zien: PASS betekent dat
    de analyse concludeerde dat het niet de moeite waard is, OVERSLAAN dat de
    tussencheck het onderzoek al niet waard vond.
    """
    uitslag = (oordeel or {}).get("oordeel")
    soort = (oordeel or {}).get("soort")
    tegen = uitslag in ("PASS", "OVERSLAAN")
    return _regel(
        "C4", "analyse", "Je eigen oordeel spreekt dit bezit tegen",
        geraakt=None if not uitslag else tegen,
        uitleg=(f"Je eigen {soort or 'onderzoek'} kwam uit op {uitslag} — en je hebt het toch."
                if tegen else
                f"Je eigen {soort or 'onderzoek'} kwam uit op {uitslag}." if uitslag else
                "Geen eigen oordeel vastgelegd."),
        waarden={"oordeel": uitslag, "soort": soort},
    )


# ---------------------------------------------------------------------------
# D — Beter alternatief (informatief)
# ---------------------------------------------------------------------------


def _alternatief(rij: dict, rank_grens: Optional[float]) -> dict:
    """Staat dit bezit ver onder de kop van Kansen?

    Telt **nooit** mee in het eindoordeel. De rangorde is een relatieve maat die
    per verversing schuift; er een verkoopoordeel op bouwen levert elk kwartaal
    vals alarm. Hij staat er omdat dit in de praktijk wél de reden is waarom je
    verkoopt: niet omdat het slecht is, maar omdat er iets beters ligt.
    """
    eigen = _getal(rij.get("rank_score"))
    return _regel(
        "D1", "alternatief", "Ver onder de kop van Kansen",
        geraakt=None if (eigen is None or rank_grens is None) else eigen < rank_grens,
        uitleg=(f"Rangscore {eigen:.0f} tegen {rank_grens:.0f} voor de kop van Kansen."
                if eigen is not None and rank_grens is not None else
                "Geen rangscore beschikbaar."),
        waarden={"rank_score": eigen, "grens": round(rank_grens, 1) if rank_grens else None},
    )


# ---------------------------------------------------------------------------
# Het geheel
# ---------------------------------------------------------------------------


def toets(rij: dict, snapshot: Optional[dict] = None, moat: Optional[dict] = None,
          analyse: Optional[dict] = None, oordeel: Optional[dict] = None,
          config: Optional[dict] = None, rank_grens: Optional[float] = None,
          annual: Optional[list] = None, vandaag: Optional[date] = None) -> dict:
    """Toets één aandeel dat je bezit tegen alle regels.

    `rij` is een dashboardrij zoals `/api/dashboard` die maakt. `snapshot` is de
    momentopname uit `bezit.these_snapshot`, `moat` de uitkomst van
    `moat_profile.bouw_profiel`, `analyse` het geparste rapport en `oordeel` het
    blok dat `oordelen.verrijk` aanhangt. Alles behalve `rij` mag ontbreken; de
    bijbehorende regels melden dan dat ze niet te toetsen zijn.
    """
    getoetst_op = (vandaag or date.today()).isoformat()

    blokkade = datapoort(rij)
    # De poort geldt voor wat de screener zelf berekent. Een eigen analyse steunt
    # daar niet op: die heeft alleen de koers en het rapport nodig, en is bij een
    # aandeel dat het model niet aankan juist het enige bruikbare oordeel dat er
    # is. Puig had een volledige analyse met scenario's van 9,58 tot 27,31, en
    # kreeg toch "geen oordeel over verkopen" omdat de screener geen fair value
    # had — precies andersom dus.
    if blokkade and analyse and rij.get("price"):
        c_regels = _analyse_regels(rij, analyse, oordeel)
        bruikbaar = [r for r in c_regels if r["geraakt"] is not None]
        if bruikbaar:
            geraakt_c = [r for r in c_regels if r["geraakt"] is True]
            hard_c = [r for r in geraakt_c if r["hard"]]
            return {
                "niveau": "rood" if hard_c else ("oranje" if geraakt_c else "groen"),
                "kop": (hard_c[0]["naam"] if hard_c else
                        geraakt_c[0]["naam"] if geraakt_c else
                        "Binnen de bandbreedte van je eigen analyse"),
                "toelichting": ("Alleen je eigen analyse is getoetst. " + blokkade),
                "regels": c_regels,
                "geraakt": geraakt_c,
                "informatief": [],
                "getoetst_op": getoetst_op,
                "aantal_regels": len(c_regels),
                "aantal_getoetst": len(bruikbaar),
            }

    if blokkade:
        return {
            "niveau": "grijs",
            "kop": "Geen oordeel over verkopen",
            "toelichting": blokkade,
            "regels": [],
            "geraakt": [],
            "informatief": [],
            "getoetst_op": getoetst_op,
            "aantal_regels": 0,
            "aantal_getoetst": 0,
        }

    breuk = omzetbreuk(annual)
    regels = (_waardering(rij, config or {}, breuk)
              + _these(rij, snapshot, moat, breuk)
              + _analyse_regels(rij, analyse, oordeel))
    informatief = [_alternatief(rij, rank_grens)]

    geraakt = [r for r in regels if r["geraakt"] is True]
    getoetst = [r for r in regels if r["geraakt"] is not None]
    families = {r["familie"] for r in geraakt}
    hard = [r for r in geraakt if r["hard"]]

    if hard:
        niveau = "rood"
        kop = hard[0]["naam"]
    elif len(geraakt) >= 2 and len(families) >= 2:
        niveau = "rood"
        kop = f"{len(geraakt)} regels geraakt, in {len(families)} verschillende soorten"
    elif geraakt:
        niveau = "oranje"
        kop = geraakt[0]["naam"]
    else:
        niveau = "groen"
        kop = f"Geen van de {len(getoetst)} toetsbare regels geraakt"

    return {
        "niveau": niveau,
        "kop": kop,
        "toelichting": None,
        "regels": regels,
        "geraakt": geraakt,
        "informatief": informatief,
        "getoetst_op": getoetst_op,
        "aantal_regels": len(regels),
        "aantal_getoetst": len(getoetst),
    }


def conclusie(rij: dict, verkoop: dict, heeft_analyse: bool) -> dict:
    """
    Wat volgt er nu uit? Eén zin, met de tegenspraak opgelost.

    De kaart toonde losse bevindingen en liet het aan de lezer om ze te wegen.
    Bij T-Mobile stond bovenaan "verkoopsignaal van de screener" en eronder dat
    je pas 141% hoger moet verkopen — twee tegengestelde beweringen zonder
    uitspraak. Dat is geen overzicht maar huiswerk.

    De rangorde die hier gehanteerd wordt:

    1. **These eerst.** Is het bedrijf veranderd, dan doet de prijs er minder toe:
       een goedkoop aandeel van een verslechterend bedrijf is geen koopje.
    2. **Je eigen analyse boven het model.** Een doorgerekend scenario voor dít
       bedrijf weegt zwaarder dan een sectorgemiddelde. Spreken ze elkaar tegen,
       dan wint de analyse en wordt de tegenspraak benoemd in plaats van verstopt.
    3. **Het model alleen als er niets beters is.**
    """
    regels = {r["id"]: r for r in (verkoop.get("regels") or [])}
    def geraakt(prefix):
        return [r for r in regels.values() if r["id"].startswith(prefix) and r["geraakt"] is True]

    these, analyse_g, waardering = geraakt("B"), geraakt("C"), geraakt("A")
    # C3 (analyse verouderd) en C4 (oordeel spreekt tegen) zijn geen prijssignaal.
    prijs_analyse = [r for r in analyse_g if r["id"] in ("C1", "C2")]

    if these:
        return {"kleur": "rood", "kop": "Kijk of je these nog klopt",
                "uitleg": f"{len(these)} van de regels over het bedrijf zelf is geraakt. "
                          "Bij een veranderend bedrijf zegt de prijs weinig — begin daar."}

    if any(r["id"] == "C2" for r in prijs_analyse):
        return {"kleur": "rood", "kop": "Overweeg verkopen",
                "uitleg": "De koers staat boven het optimistische scenario van je eigen "
                          "analyse. Ook in het gunstigste geval is dit niet meer te "
                          "verdedigen met je eigen cijfers."}

    if prijs_analyse:
        return {"kleur": "oranje", "kop": "Let op, maar nog niet verkopen",
                "uitleg": "De koers is je kansgewogen waarde gepasseerd, maar blijft onder "
                          "het optimistische scenario. Dit is het gebied waarin je zou "
                          "kunnen afbouwen, niet waarin je moet."}

    if heeft_analyse and waardering:
        return {"kleur": "groen", "kop": "Houden",
                "uitleg": "De screener geeft een verkoopsignaal, maar dat komt uit het "
                          "generieke model. Je eigen analyse ziet meer waarde dan de "
                          "huidige koers, en die weegt zwaarder."}

    if heeft_analyse:
        return {"kleur": "groen", "kop": "Houden",
                "uitleg": "Niets geraakt: de koers zit onder je eigen waardering en het "
                          "bedrijf is niet wezenlijk veranderd."}

    if waardering:
        return {"kleur": "oranje", "kop": "Uitzoeken",
                "uitleg": "Het model vindt dit te duur, maar er ligt geen eigen analyse om "
                          "dat tegen af te zetten. Een analyse maken is hier de volgende stap."}

    return {"kleur": "groen", "kop": "Niets te doen",
            "uitleg": "Geen enkele verkoopregel geraakt."}


def verkoopdrempels(rij: dict, verkoop: dict) -> list[dict]:
    """
    De prijsgebonden regels omgerekend naar bedragen: bij welke koers gaat deze af?

    Aanleiding: de kaart toonde regelnamen ("Ver boven de fair value") en Janco's
    vraag was simpelweg *wat is dan de verkoopprijs?* Die stond nergens, terwijl
    elke regel zijn eigen grens al in `waarden` meedraagt — het was een kwestie
    van omrekenen, niet van een nieuwe waardering.

    Alleen regels waarvan de grens een koers ís komen hier terug. De these-regels
    (B-familie) gaan over of dit nog hetzelfde bedrijf is; daar bestaat geen prijs
    voor en er een verzinnen zou de vraag verkeerd stellen.

    Volgorde is de laagste grens eerst, zodat je leest wat je het eerst tegenkomt.
    `bron` is 'analyse' waar het uit het eigen onderzoek komt en 'model' waar het
    uit de screener komt — dat verschil hoort zichtbaar te zijn, want een grens
    uit een doorgerekend scenario weegt zwaarder dan een sectorgemiddelde.
    """
    koers = _getal(rij.get("price"))
    valuta = rij.get("currency")
    if not koers or koers <= 0:
        return []

    per_id = {r["id"]: r for r in (verkoop.get("regels") or [])}
    uit = []

    def _voeg(regel_id, grens, bron, label):
        regel = per_id.get(regel_id)
        grens = _getal(grens)
        if regel is None or not grens or grens <= 0:
            return
        uit.append({
            "id": regel_id,
            "label": label,
            "grens": round(grens, 2),
            "valuta": valuta,
            "bron": bron,
            "geraakt": regel.get("geraakt") is True,
            # Positief: zoveel procent moet de koers nog stijgen om de grens te
            # raken. Negatief: je staat er al voorbij.
            "afstand_pct": round(100 * (grens / koers - 1), 1),
            "hard": bool(regel.get("hard")),
        })

    # Uit de eigen analyse — die weegt het zwaarst.
    c1 = per_id.get("C1", {}).get("waarden") or {}
    c2 = per_id.get("C2", {}).get("waarden") or {}
    _voeg("C1", c1.get("kansgewogen"), "analyse", "aandacht boven")
    _voeg("C2", c2.get("optimistisch"), "analyse", "verkopen boven")

    # Het model alleen als er géén analyse ligt. Ligt die er wel, dan is hij
    # leidend: een doorgerekend scenario voor dít bedrijf weegt zwaarder dan een
    # sectorgemiddelde, en beide naast elkaar tonen leverde bij Adyen een derde
    # grens op met het label "geen analyse" terwijl die er juist wél was.
    if not uit:
        fv = _getal(rij.get("combined_fv"))
        a2 = per_id.get("A2", {}).get("waarden") or {}
        grens_pct = _getal(a2.get("grens_pct"))
        if fv and grens_pct:
            _voeg("A2", fv * grens_pct / 100.0, "model", "verkopen boven")

    return sorted(uit, key=lambda d: d["grens"])


def momentopname(rij: dict, moat: Optional[dict] = None) -> dict:
    """Wat er bewaard wordt op het moment dat je bezit vastlegt.

    Dit is wat je geloofde toen je het opschreef; alles daarna is vergelijking.
    Bewust klein gehouden: alleen wat een regel ook echt gebruikt. Een grotere
    momentopname suggereert een precisie die er niet is.
    """
    return {
        "op": date.today().isoformat(),
        "koers": rij.get("price"),
        "valuta": rij.get("currency"),
        "quality_score": rij.get("quality_score"),
        "revenue_cagr": rij.get("revenue_cagr"),
        "normalized_fcf_m": rij.get("normalized_fcf_m"),
        "implied_growth": rij.get("implied_growth"),
        "combined_fv": rij.get("combined_fv"),
        "roic_mediaan": (moat or {}).get("roic_mediaan"),
        "brutomarge_trend_pp": (moat or {}).get("brutomarge_trend_pp"),
        "moat_niveau": (moat or {}).get("niveau"),
    }

"""
analyses.py — leest de rapporten uit analyses/ en tussenchecks/ en rendert
ze naar HTML voor /analyses en /tussenchecks.

Twee soorten documenten, allebei markdown uit de aandelenanalyse-pipeline:

- **analyses/** — volledige fundamentele analyses, 50-90 KB, vijftien
  genummerde secties plus interne pipeline-administratie die eruit gefilterd
  wordt.
- **tussenchecks/** — beslisdocumenten van vóór zo'n analyse (verdient dit
  aandeel diepgravend onderzoek?), 5-7 KB, ongenummerde secties die allemaal
  getoond worden. Ze leunen op screenerdata van aggregator-kwaliteit en zijn
  nadrukkelijk geen analyse; de pagina zegt dat er ook bij.

De rapporten zijn de research-MD's uit de aandelenanalyse-pipeline
(stage 1). Ze zijn bewust de bron, niet de analyse-JSON's: de MD is
altijd de nieuwste versie, terwijl een JSON pas bijgewerkt wordt als
stage 2 draait.

Twee dingen gebeuren hier:

1. Parsen — uit de kop en de secties `## Metadata` en `## 1. Executive
   summary` worden de kaart- en headervelden gehaald (naam, ticker,
   sector, koers, oordeel, fair value, score). De rapporten zijn met de
   hand geschreven en de bullets varieren in vorm, dus elk veld faalt
   afzonderlijk naar None in plaats van de hele pagina te breken.

2. Filteren en renderen — interne pipeline-secties (bronnen-inventaris,
   afrondingschecklist, opmerkingen voor Claude Code) gaan eruit, de
   Metadata-sectie wordt vervangen door de header, en de rest wordt per
   sectie naar HTML gerenderd zodat de sectienavigatie ankers heeft.

Cache: bestanden wijzigen alleen bij deploy, dus resultaten worden
gecachet op mtime. De mtime-check houdt lokaal ontwikkelen wel live.
"""

import os
import re
import threading

import markdown

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSES_DIR = os.path.join(_REPO, "analyses")
TUSSENCHECKS_DIR = os.path.join(_REPO, "tussenchecks")

# Publishable token (zelfde als het aandelenanalyse-platform gebruikt);
# mag client-side staan. Override via env als hij ooit geroteerd wordt.
LOGO_DEV_TOKEN = os.environ.get("LOGO_DEV_TOKEN", "pk_WvGztbDCRYeS_qNNd36BXg")

# Welke secties zijn rapportinhoud? De rapportvorm schrijft 15 genummerde
# secties voor (1. Executive summary t/m 15. Update-historie); alles zonder
# nummer is randadministratie van de pipeline: de bronnen-inventaris, het
# metadata-blok (dat de header al toont), de afrondingschecklist, de
# opmerkingen voor Claude Code en incidentele stage-2-notities.
#
# Filteren op "heeft een nummer" in plaats van op een lijst met titels
# houdt ook toekomstige interne secties buiten de pagina.
SECTIE_NUMMER = re.compile(r"^(\d+)\.\s*(.+)")

_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# parse-helpers
# ---------------------------------------------------------------------------


def _split_secties(tekst: str) -> list[tuple[str, str]]:
    """Split op H2-koppen. Alles voor de eerste H2 (H1 + intro-blockquote)
    valt weg; dat is pipeline-context, geen rapportinhoud."""
    delen = re.split(r"^## +(.+?)\s*$", tekst, flags=re.M)
    return [(delen[i].strip(), delen[i + 1]) for i in range(1, len(delen) - 1, 2)]


def _bullet(sectie: str, key: str) -> str | None:
    """Waarde van een bullet als `- **Key:** waarde` of
    `- **Key** (toelichting): waarde`."""
    m = re.search(r"\*\*" + key + r"[^*]*\*\*\s*(?:\([^)]*\))?\s*:?\s*(.+)", sectie)
    if not m:
        return None
    waarde = m.group(1).strip().strip("*").strip()
    return waarde or None


def _schoon(waarde: str | None) -> str | None:
    """Haal redactionele annotaties als [BIJGEWERKT v1.1: was 81] weg."""
    if waarde is None:
        return None
    waarde = re.sub(r"\[[^\]]*\]", "", waarde).strip().strip("*").strip()
    return waarde or None


def _parse_oordeel(execsum: str) -> str | None:
    """KOOP/HOLD/PASS uit de oordeel-bullet van de executive summary.

    Alleen binnen die sectie zoeken: 'Oordeel' komt verderop in elk
    rapport nog tien keer voor (moat, management, frameworks).
    """
    for regel in execsum.splitlines():
        if not re.search(r"\*\*Oordeel\b(?!\s+\w)", regel):
            continue
        # Haakjes eruit: sommige rapporten zetten de hele enum-lijst
        # '(enum UITSLUITEND: **KOOP** | **HOLD** | **PASS**)' in de bullet.
        kaal = re.sub(r"\([^)]*\)", "", regel)
        m = re.search(r"\b(KOOP|HOLD|PASS)\b", kaal)
        if m:
            return m.group(1)
    return None


def _parse_score(tekst: str) -> int | None:
    """Scorekaart-totaal (0-45).

    Drie notaties komen voor:
      '31'                                  -> eerste getal
      '32 (= 5 + 2 + 3 + ...)'              -> eerste getal, som staat tussen haakjes
      '1 + 4 + 3 + ... = **33**'            -> getal na de laatste '='
    """
    m = re.search(r"\*\*Totaalscore[^\n]*", tekst)
    if not m:
        return None
    waarde = m.group(0).split(":", 1)[-1]
    waarde = re.sub(r"\([^)]*\)", "", waarde)  # uitsplitsing tussen haakjes weg
    if "=" in waarde:
        kandidaat = re.search(r"=\s*\**(\d{1,2})", waarde)
    else:
        kandidaat = re.search(r"\b(\d{1,2})\b", waarde)
    if not kandidaat:
        return None
    score = int(kandidaat.group(1))
    return score if 0 <= score <= 45 else None


def _parse_bedrag(waarde: str | None, valuta: str | None) -> str | None:
    """Fair value leesbaar maken. Bevat de waarde al een valuta-aanduiding
    ('226 NOK'), dan blijft hij zoals hij is."""
    waarde = _schoon(waarde)
    if not waarde:
        return None
    if valuta and re.fullmatch(r"[\d.,]+", waarde):
        return f"{valuta} {waarde}"
    return waarde


def _parse_getal(waarde: str | None) -> float | None:
    """Getal uit een bullet, met zowel Europese als Amerikaanse notatie.

    De rapporten mengen beide: '1.239,40' (EUR, punt = duizendtal) naast
    '93.79' (USD, punt = decimaal). Een punt zonder komma is ambigu; drie
    cijfers erachter betekent duizendtal ('1.215' = 1215), anders decimaal.
    """
    waarde = _schoon(waarde)
    if not waarde:
        return None
    m = re.search(r"[-+]?\d[\d.,]*", waarde)
    if not m:
        return None
    getal = m.group(0).rstrip(".,")

    if "," in getal:
        getal = getal.replace(".", "").replace(",", ".")
    elif re.search(r"\.\d{3}$", getal):
        getal = getal.replace(".", "")

    try:
        return float(getal)
    except ValueError:
        return None


def _sectie_id(nr: int, titel: str) -> tuple[str, str]:
    """(anker-id, kort label voor de nav)."""
    return f"sectie-{nr}", _kort(titel)


def _kort(titel: str) -> str:
    """Navigatielabel: eerste betekenisdeel, afgekapt."""
    label = re.split(r"\s+[—–(]", titel)[0].strip()
    return label if len(label) <= 26 else label[:25].rstrip() + "…"


# ---------------------------------------------------------------------------
# parsen van een heel rapport
# ---------------------------------------------------------------------------


def _parse_bestand(pad: str) -> dict:
    tekst = open(pad, encoding="utf-8").read()
    ticker_uit_naam = os.path.splitext(os.path.basename(pad))[0]

    naam = None
    m = re.search(r"^#\s*Research:\s*(.+?)\s+[—–-]\s+(.+)$", tekst, flags=re.M)
    if m:
        naam = m.group(2).strip()

    secties = _split_secties(tekst)
    meta_tekst = next((inh for t, inh in secties if t == "Metadata"), "")
    execsum = next(
        (inh for t, inh in secties if re.match(r"\d+\.\s*Executive summary", t)), ""
    )

    # Valuta-bullets bevatten soms een toelichting ('EUR (notering AEX);
    # rapportage USD'); voor prefixen is alleen de eerste code bruikbaar.
    valuta = _bullet(meta_tekst, "Valuta")
    if valuta:
        code = re.match(r"([A-Z]{3})\b", valuta)
        valuta = code.group(1) if code else valuta
    peildatum_raw = _bullet(meta_tekst, "Peildatum analyse")
    peildatum = None
    if peildatum_raw:
        # SDIP-B noemt twee datums (koersdatum en rapportdatum); de eerste telt.
        d = re.search(r"\d{4}-\d{2}-\d{2}", peildatum_raw)
        peildatum = d.group(0) if d else None

    domein = _bullet(meta_tekst, "Domein")
    if domein:
        domein = re.sub(r"^https?://", "", domein).strip().strip("/")
        domein = domein.removeprefix("www.")

    koers = _schoon(_bullet(meta_tekst, "Koers op peildatum"))
    analyse = {
        "ticker": _bullet(meta_tekst, r"Ticker \(bare\)") or ticker_uit_naam,
        "bestand_ticker": ticker_uit_naam,
        "naam": naam or ticker_uit_naam,
        "exchange": _bullet(meta_tekst, "Exchange"),
        "sector": _bullet(meta_tekst, r"Sector \(GICS-achtig\)") or _bullet(meta_tekst, "Sector"),
        "land": _bullet(meta_tekst, "Land"),
        "valuta": valuta,
        "koers": f"{valuta} {koers}" if valuta and koers else koers,
        "marktkap": _schoon(_bullet(meta_tekst, "Marktkapitalisatie")),
        "peildatum": peildatum,
        "domein": domein,
        "oordeel": _parse_oordeel(execsum),
        "fair_value": _parse_bedrag(_bullet(execsum, "Fair value basis"), valuta),
        "fair_value_getal": _parse_getal(_bullet(execsum, "Fair value basis")),
        "upside": _parse_getal(_bullet(execsum, "Upside pct")),
        "koers_getal": _parse_getal(koers),
        "yahoo_symbol": _bullet(meta_tekst, "Yahoo symbol"),
        "score": _parse_score(tekst),
        "initiaal": (ticker_uit_naam[:1] or "?").upper(),
        # Vaste kleur per ticker voor het fallback-blokje als logo.dev
        # geen logo heeft — zelfde bedrijf houdt dezelfde kleur.
        "hue": sum(ord(c) for c in ticker_uit_naam) % 360,
    }

    zichtbaar = []
    for titel, inhoud in secties:
        m = SECTIE_NUMMER.match(titel)
        if not m:
            continue
        nr, kop = int(m.group(1)), m.group(2)
        anker, label = _sectie_id(nr, kop)
        html = markdown.markdown(
            f"## {titel}\n\n{inhoud}",
            extensions=["tables", "fenced_code", "sane_lists"],
        )
        zichtbaar.append({"id": anker, "label": label, "nr": nr, "html": html})
    analyse["secties"] = zichtbaar
    return analyse


def _laad(map_pad: str, bestandsnaam: str, parser) -> dict | None:
    """Geparst document uit de cache, of opnieuw parsen als het bestand
    veranderde. De sleutel is het volledige pad, zodat analyses en
    tussenchecks dezelfde cache kunnen delen."""
    pad = os.path.join(map_pad, bestandsnaam)
    try:
        mtime = os.path.getmtime(pad)
    except OSError:
        return None

    gecachet = _cache.get(pad)
    if gecachet and gecachet[0] == mtime:
        return gecachet[1]

    document = parser(pad)
    with _cache_lock:
        _cache[pad] = (mtime, document)
    return document


def _bestanden(map_pad: str) -> list[str]:
    if not os.path.isdir(map_pad):
        return []
    return sorted(f for f in os.listdir(map_pad) if f.endswith(".md"))


# ---------------------------------------------------------------------------
# tussenchecks
# ---------------------------------------------------------------------------
#
# Een tussencheck beantwoordt één vraag vóór de volledige analyse: verdient dit
# aandeel diepgravend onderzoek? Kortere documenten met een andere kop dan een
# analyse, en zonder genummerde secties — hier worden juist álle H2's getoond,
# want er staat niets interns in.


def _parse_tussencheck(pad: str) -> dict:
    tekst = open(pad, encoding="utf-8").read()
    ticker_uit_naam = os.path.splitext(os.path.basename(pad))[0]

    ticker, naam = ticker_uit_naam, ticker_uit_naam
    m = re.search(r"^#\s*Tussencheck:\s*(.+?)\s+[—–-]\s+(.+)$", tekst, flags=re.M)
    if m:
        ticker, naam = m.group(1).strip(), m.group(2).strip()

    oordeel = None
    m = re.search(r"\*\*Oordeel:\s*(VERDIEPEN|TWIJFEL|OVERSLAAN)", tekst)
    if m:
        oordeel = m.group(1)

    # Metaregel: 'Datum: … · Koers … · Beurswaarde … · <beurs>'.
    # De eerste drie dragen een label, wat overblijft is de beurs.
    datum = koers = beurswaarde = beurs = None
    m = re.search(r"^Datum:.*$", tekst, flags=re.M)
    if m:
        overig = []
        for deel in (d.strip() for d in m.group(0).split("·")):
            if deel.startswith("Datum:"):
                d = re.search(r"\d{4}-\d{2}-\d{2}", deel)
                datum = d.group(0) if d else deel.removeprefix("Datum:").strip()
            elif deel.startswith("Koers"):
                koers = deel.removeprefix("Koers").strip(": ").strip()
            elif deel.startswith("Beurswaarde"):
                beurswaarde = deel.removeprefix("Beurswaarde").strip(": ").strip()
            elif deel:
                overig.append(deel)
        beurs = " · ".join(overig) or None

    secties = []
    for nr, (titel, inhoud) in enumerate(_split_secties(tekst), start=1):
        html = markdown.markdown(
            f"## {titel}\n\n{inhoud}",
            extensions=["tables", "fenced_code", "sane_lists"],
        )
        secties.append({"id": f"sectie-{nr}", "label": _kort(titel), "nr": nr, "html": html})

    return {
        "ticker": ticker,
        "bestand_ticker": ticker_uit_naam,
        "naam": naam,
        "oordeel": oordeel,
        "datum": datum,
        "koers": koers,
        "beurswaarde": beurswaarde,
        "beurs": beurs,
        "initiaal": (ticker_uit_naam[:1] or "?").upper(),
        "hue": sum(ord(c) for c in ticker_uit_naam) % 360,
        "secties": secties,
    }


def get_alle_tussenchecks() -> list[dict]:
    """Kaartgegevens van alle tussenchecks, nieuwste datum eerst.

    Markeert per tussencheck of er inmiddels een volledige analyse bestaat, zodat
    de pagina daarnaar kan doorlinken zodra een VERDIEPEN is uitgediept.
    """
    met_analyse = {a["ticker"].upper() for a in get_all_summaries()}
    checks = []
    for bestand in _bestanden(TUSSENCHECKS_DIR):
        check = _laad(TUSSENCHECKS_DIR, bestand, _parse_tussencheck)
        if not check:
            continue
        kaart = {k: v for k, v in check.items() if k != "secties"}
        kaart["heeft_analyse"] = check["ticker"].upper() in met_analyse
        checks.append(kaart)
    return sorted(checks, key=lambda c: c.get("datum") or "", reverse=True)


def get_tussencheck(ticker: str) -> dict | None:
    """Volledige tussencheck op ticker, hoofdletterongevoelig."""
    gezocht = (ticker or "").strip().upper()
    for bestand in _bestanden(TUSSENCHECKS_DIR):
        check = _laad(TUSSENCHECKS_DIR, bestand, _parse_tussencheck)
        if not check:
            continue
        if gezocht in (check["ticker"].upper(), check["bestand_ticker"].upper()):
            check = dict(check)
            check["heeft_analyse"] = get_analyse(check["ticker"]) is not None
            return check
    return None


# ---------------------------------------------------------------------------
# publieke API — analyses
# ---------------------------------------------------------------------------


def get_all_summaries() -> list[dict]:
    """Kaartgegevens van alle analyses, nieuwste peildatum eerst.
    Zonder de gerenderde secties — die zijn alleen voor de detailpagina."""
    analyses = []
    for bestand in _bestanden(ANALYSES_DIR):
        analyse = _laad(ANALYSES_DIR, bestand, _parse_bestand)
        if analyse:
            analyses.append({k: v for k, v in analyse.items() if k != "secties"})
    return sorted(analyses, key=lambda a: a.get("peildatum") or "", reverse=True)


def get_analyse(ticker: str) -> dict | None:
    """Volledige analyse op ticker, hoofdletterongevoelig.

    Zoekt op de geparste ticker in plaats van een pad te bouwen uit de
    URL-parameter — zo kan een verzonnen ticker nooit een pad worden.
    """
    gezocht = (ticker or "").strip().upper()
    for bestand in _bestanden(ANALYSES_DIR):
        analyse = _laad(ANALYSES_DIR, bestand, _parse_bestand)
        if not analyse:
            continue
        if gezocht in (analyse["ticker"].upper(), analyse["bestand_ticker"].upper()):
            return analyse
    return None

"""
Is het JavaScript in de templates nog geldig?

Aanleiding: op 22 augustus brak één verkeerd ontsnapte `\n` in een tooltip het
hele script van het dashboard. Alles ná die regel stopte met werken — de
dekkingsverklaring bleef leeg, de tabel bleef leeg — terwijl pyflakes en alle
Python-tests groen bleven. Die kijken immers niet in een template.

Deze test haalt de scriptblokken eruit en laat Node ze parseren. Dat vangt geen
logische fouten, maar wel de categorie die de pagina in één klap stilzet.

Jinja-uitdrukkingen ({{ ... }} en {% ... %}) worden vervangen door een
plaatshouder, want die zijn er tijdens het parseren nog niet uit gehaald.
"""

import os
import re
import subprocess
import sys
import tempfile

HIER = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(os.path.dirname(HIER), "templates")

SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)
JINJA_BLOK = re.compile(r"\{%.*?%\}", re.S)
JINJA_WAARDE = re.compile(r"\{\{.*?\}\}", re.S)
NAAM = re.compile(r"\b(_[a-zA-Z][a-zA-Z0-9]*)\b")
# `window._health = ...` telt ook als ontstaan; niet elke naam wordt met let of
# const geboren.
DECLARATIE = re.compile(
    r"\b(?:let|const|var|function)\s+(_[a-zA-Z][a-zA-Z0-9]*)"
    r"|\bwindow\.(_[a-zA-Z][a-zA-Z0-9]*)\s*=")
TEKST = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
SJABLOON = re.compile(r"`(?:\\.|[^`\\])*`", re.S)
INVULLING = re.compile(r"\$\{([^{}]*)\}")


def zonder_tekst(js: str) -> str:
    """Alles wat tekst is eruit, zodat er geen namen uit strings komen.

    Zonder dit las de controle `target="_blank"`, `'_template.csv'` en de
    HTML-attributen in sjabloonteksten als variabelenamen. Bij een sjabloon
    blijft alleen wat tussen ${...} staat, want dat is wél code.
    """
    js = SJABLOON.sub(lambda m: " ".join(INVULLING.findall(m.group(0))), js)
    return TEKST.sub("''", js)


def kaal(js: str) -> str:
    """Jinja eruit, zodat Node puur het JavaScript ziet."""
    js = JINJA_BLOK.sub("", js)
    # Een waarde kan overal staan (in een string, als getal, als object).
    # `null` past op al die plekken zonder zelf een syntaxfout te worden.
    return JINJA_WAARDE.sub("null", js)


def controleer(pad: str) -> list[str]:
    with open(pad, encoding="utf-8") as f:
        inhoud = f.read()

    fouten = []
    for n, blok in enumerate(SCRIPT.findall(inhoud), 1):
        js = kaal(blok).strip()
        if not js:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", encoding="utf-8",
                                         delete=False) as t:
            # In een module mag `await` op het hoogste niveau staan; dat scheelt
            # valse meldingen bij async-code in de templates.
            t.write(js)
            tmp = t.name
        try:
            r = subprocess.run(["node", "--check", tmp],
                               capture_output=True, text=True)
            if r.returncode != 0:
                melding = (r.stderr or "").strip().splitlines()
                kern = next((x for x in melding if "Error" in x), melding[-1] if melding else "?")
                fouten.append(f"{os.path.basename(pad)} blok {n}: {kern}")
        finally:
            os.unlink(tmp)
    return fouten


def ongedeclareerd(pad: str, bekend: set) -> list:
    """Namen als `_actiefLand` die gebruikt worden maar nergens ontstaan.

    Node --check ziet dit niet: een ongedefinieerde naam is geldige syntax en
    valt pas om tijdens het uitvoeren. Precies zo verdween op 2026-08-22 het
    blok met `_actiefLand`, `_actieveSector` en `SIGNAAL_RANG` toen de landtabel
    ernaast werd opgeruimd — de pagina laadde, en meldde dan "Fout bij laden".

    Alleen namen met een underscore ervoor, want dat is in dit project de
    afspraak voor modulestatus. Bibliotheekfuncties en DOM-namen blijven zo
    buiten beeld zonder dat er een lijst met uitzonderingen bij hoeft.
    """
    with open(pad, encoding="utf-8") as f:
        inhoud = f.read()

    gebruikt = set()
    gedeclareerd = set(bekend)
    for blok in SCRIPT.findall(inhoud):
        js = kaal(blok)
        for a, b in DECLARATIE.findall(js):
            gedeclareerd.add(a or b)
        gebruikt |= set(NAAM.findall(zonder_tekst(js)))
    return sorted(gebruikt - gedeclareerd)


def gedeeld() -> set:
    """Namen uit base.html; elke pagina erft die."""
    pad = os.path.join(TEMPLATES, "base.html")
    if not os.path.exists(pad):
        return set()
    with open(pad, encoding="utf-8") as f:
        return {a or b for a, b in DECLARATIE.findall(f.read())}


if __name__ == "__main__":
    if not subprocess.run(["node", "--version"], capture_output=True).returncode == 0:
        print("  [--] node niet gevonden — overgeslagen")
        sys.exit(0)

    alle = []
    for naam in sorted(os.listdir(TEMPLATES)):
        if naam.endswith(".html"):
            f = controleer(os.path.join(TEMPLATES, naam))
            alle += f
            print(f"  [{'FOUT' if f else 'OK '}] {naam}")
            for x in f:
                print(f"        {x}")

    basis = gedeeld()
    for naam in sorted(os.listdir(TEMPLATES)):
        if not naam.endswith(".html"):
            continue
        mist = ongedeclareerd(os.path.join(TEMPLATES, naam), basis)
        if mist:
            alle.append(f"{naam}: gebruikt maar nergens gedeclareerd: {mist}")
            print(f"  [FOUT] {naam} — ongedeclareerd: {mist}")

    print("\nFALEND:", len(alle))
    sys.exit(1 if alle else 0)

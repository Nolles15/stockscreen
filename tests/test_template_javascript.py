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

    print("\nFALEND:", len(alle))
    sys.exit(1 if alle else 0)

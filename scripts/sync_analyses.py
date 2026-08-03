"""
sync_analyses.py — kopieert de research-MD's naar de stockscreen-repo.

Haalt twee sets op uit de aandelenanalyse-repo:
  research/*.md              -> analyses/      (volledige analyses)
  research/_tussencheck/*.md -> tussenchecks/  (beslisdocumenten vooraf)

Waarom kopieren en niet runtime ophalen: de MD's moeten in het
Docker-image zitten dat naar Fly gaat. Ze staan daarom op de repo-root —
NIET onder `data/`, want die map staat in .dockerignore en zou lokaal wel
werken maar in productie stilletjes leeg zijn.

Gedrag: mirror. Nieuwe en gewijzigde bestanden worden gekopieerd,
bestanden die in de bron niet meer bestaan worden hier verwijderd.

Gebruik:
  python scripts/sync_analyses.py
  python scripts/sync_analyses.py --bron "D:/pad/naar/research"
  python scripts/sync_analyses.py --dry-run

Daarna: git add analyses && git commit && fly deploy --remote-only --depot=false
"""

import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STANDAARD_BRON = Path(r"C:\Users\janco\aandelenanalyse\research")
REPO = Path(__file__).resolve().parent.parent

# Niet-analyses in research/: methodiek, sjablonen, prompts, transparantierapporten.
EXCLUDE_NAMEN = {"TEMPLATE.md", "METHODE.md", "README.md", "_PROMPT_COWORK.md"}
EXCLUDE_PREFIX = ("TRANSPARANTIE_",)

# (submap in research/, doelmap in de repo, omschrijving)
SETS = [
    ("", REPO / "analyses", "analyses"),
    ("_tussencheck", REPO / "tussenchecks", "tussenchecks"),
]


def is_document(pad: Path) -> bool:
    return (
        pad.suffix == ".md"
        and pad.name not in EXCLUDE_NAMEN
        and not pad.name.startswith(EXCLUDE_PREFIX)
    )


def gewijzigd(bron: Path, doel: Path) -> bool:
    if not doel.exists():
        return True
    b, d = bron.stat(), doel.stat()
    return b.st_size != d.st_size or b.st_mtime > d.st_mtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bron", default=str(STANDAARD_BRON),
                        help=f"map met research-MD's (standaard: {STANDAARD_BRON})")
    parser.add_argument("--dry-run", action="store_true",
                        help="toon wat er zou gebeuren, wijzig niets")
    args = parser.parse_args()

    bron_wortel = Path(args.bron)
    if not bron_wortel.is_dir():
        print(f"[sync] FAIL: bronmap niet gevonden: {bron_wortel}", file=sys.stderr)
        return 1

    totaal_gekopieerd = totaal_verwijderd = 0

    for submap, doel, omschrijving in SETS:
        bron = bron_wortel / submap if submap else bron_wortel
        if not bron.is_dir():
            print(f"[sync] {omschrijving}: bronmap ontbreekt ({bron}) — overgeslagen")
            continue

        # glob() is niet recursief, dus submappen komen alleen mee als ze
        # hierboven expliciet als eigen set staan.
        bron_bestanden = {p.name: p for p in bron.glob("*.md") if is_document(p)}
        if not bron_bestanden:
            print(f"[sync] FAIL: geen documenten gevonden in {bron}", file=sys.stderr)
            return 1

        if not args.dry_run:
            doel.mkdir(parents=True, exist_ok=True)

        gekopieerd = ongewijzigd = verwijderd = 0

        for naam, bronpad in sorted(bron_bestanden.items()):
            doelpad = doel / naam
            if gewijzigd(bronpad, doelpad):
                print(f"  {'zou kopieren' if args.dry_run else 'gekopieerd'}: "
                      f"{omschrijving}/{naam} ({bronpad.stat().st_size:,} bytes)")
                if not args.dry_run:
                    shutil.copy2(bronpad, doelpad)
                gekopieerd += 1
            else:
                ongewijzigd += 1

        if doel.is_dir():
            for doelpad in sorted(doel.glob("*.md")):
                if doelpad.name not in bron_bestanden:
                    print(f"  {'zou verwijderen' if args.dry_run else 'verwijderd'}: "
                          f"{omschrijving}/{doelpad.name} (niet meer in bron)")
                    if not args.dry_run:
                        doelpad.unlink()
                    verwijderd += 1

        print(f"[sync] {omschrijving}: {gekopieerd} gekopieerd, {ongewijzigd} ongewijzigd, "
              f"{verwijderd} verwijderd ({len(bron_bestanden)} in {doel.name}/)")
        totaal_gekopieerd += gekopieerd
        totaal_verwijderd += verwijderd

    if totaal_gekopieerd or totaal_verwijderd:
        print("[sync] vergeet niet: git add analyses tussenchecks && git commit "
              "&& fly deploy --remote-only --depot=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())

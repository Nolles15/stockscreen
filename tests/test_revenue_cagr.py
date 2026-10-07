"""De omzetgroei en de nieuwe R&D-kolom (groeiprofiel, fase A).

Twee dingen die bij elkaar horen omdat ze het fundament onder het groeiprofiel
zijn (docs/plan-groeiprofiel.md):

1. `screener._calc_revenue_cagr` gaf bij een definitiewissel in de omzetreeks
   een getal dat het tegendeel zei. Adyen stond op 33% krimp terwijl het ~19%
   per jaar groeide, en de Groeiers-tab, de 🌱-markering en de value-trap-
   waarschuwing lazen dat getal allemaal. Nu: geen getal.
2. Diezelfde breukregel mocht de snelste groeiers niet wegvegen. Een jong
   bedrijf dat in één jaar verdubbelt is geen definitiewissel zolang de
   brutowinst meegroeit.

Plus een bewaker op de R&D-kolom: die moet op elke plek staan waar een
jaarcijferveld moet staan, anders valt hij ergens stil weg (FX-conversie,
handmatige jaarrij, override).

Geen database, geen netwerk. Draaien met `python tests/test_revenue_cagr.py`.
"""
import ast
import os
import sys

WORTEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORTEL)

from engine import screener, exit_regels, db, data_fetcher  # noqa: E402


def _jaren(omzet, bruto=None, vanaf=2022):
    """Jaarrijen, oudste eerst, zoals db.jaarrijen_met_overrides ze levert."""
    rijen = []
    for i, w in enumerate(omzet):
        rij = {"fiscal_year": vanaf + i, "period_type": "annual", "revenue": w}
        if bruto is not None:
            rij["gross_profit"] = bruto[i]
        rijen.append(rij)
    return rijen


def _ongeveer(a, b, marge=1e-9):
    return a is not None and b is not None and abs(a - b) < marge


# ---------------------------------------------------------------------------
# Omzetgroei
# ---------------------------------------------------------------------------

def test_adyen_geeft_geen_groeicijfer():
    """De bekende valse krimp: bruto (2022) naar netto (2023+) omzet."""
    adyen = _jaren([8936e6, 1863e6, 2226e6, 2647e6])
    assert screener._calc_revenue_cagr(adyen) is None

    # Met de echte brutowinst erbij blijft het een breuk: de omzet zakt een
    # factor 4,8 terwijl de brutowinst 22% stijgt.
    met_bruto = _jaren([8936e6, 1863e6, 2226e6, 2647e6],
                       bruto=[1330e6, 1626e6, 1988e6, 2352e6])
    assert screener._calc_revenue_cagr(met_bruto) is None
    assert exit_regels.omzetbreuk(met_bruto)
    print("  [OK] Adyen: geen groeicijfer bij een definitiewissel, ook mét brutowinst")


def test_gewone_reeks_ongewijzigd():
    """Zonder breuk is de uitkomst exact de oude formule."""
    rijen = _jaren([100e6, 120e6, 150e6, 180e6])
    verwacht = (180e6 / 100e6) ** (1 / 3) - 1
    assert _ongeveer(screener._calc_revenue_cagr(rijen), verwacht)

    # Vijf jaren: alleen de laatste vier tellen, zoals altijd.
    vijf = _jaren([50e6, 100e6, 120e6, 150e6, 180e6], vanaf=2021)
    assert _ongeveer(screener._calc_revenue_cagr(vijf), verwacht)

    # Krimp blijft krimp.
    dalend = _jaren([1000e6, 850e6, 700e6, 600e6])
    assert screener._calc_revenue_cagr(dalend) < -0.02
    print("  [OK] een reeks zonder breuk geeft hetzelfde getal als voorheen")


def test_ttm_rij_telt_niet_mee():
    """De TTM-rij heeft fiscal_year 0 en hoort niet in de jaarreeks."""
    rijen = _jaren([100e6, 120e6, 150e6, 180e6])
    met_ttm = rijen + [{"fiscal_year": 0, "period_type": "ttm", "revenue": 900e6}]
    assert _ongeveer(screener._calc_revenue_cagr(met_ttm),
                     screener._calc_revenue_cagr(rijen))
    print("  [OK] de TTM-rij (boekjaar 0) blijft buiten de omzetgroei")


def test_breuk_buiten_het_venster_telt_niet():
    """Een wissel van vóór de laatste vier jaren raakt de driejaarsgroei niet."""
    rijen = _jaren([500e6, 100e6, 120e6, 150e6, 180e6], vanaf=2021)
    assert exit_regels.omzetbreuk(rijen), "de reeks als geheel heeft wél een breuk"
    verwacht = (180e6 / 100e6) ** (1 / 3) - 1
    assert _ongeveer(screener._calc_revenue_cagr(rijen), verwacht)
    print("  [OK] een breuk van vóór het venster laat de groei staan")


def test_echte_verdubbeling_is_geen_breuk():
    """Een groeier die in één jaar verdubbelt, met meegroeiende brutowinst."""
    snel = _jaren([40e6, 95e6, 150e6, 210e6], bruto=[22e6, 50e6, 82e6, 118e6])
    assert exit_regels.omzetbreuk(snel) is None
    verwacht = (210e6 / 40e6) ** (1 / 3) - 1
    assert _ongeveer(screener._calc_revenue_cagr(snel), verwacht)
    assert screener._calc_revenue_cagr(snel) > 0.5

    # Echte halvering mét meedalende brutowinst is ook echt — en dus krimp.
    instorting = _jaren([400e6, 180e6, 170e6, 160e6], bruto=[160e6, 70e6, 66e6, 62e6])
    assert exit_regels.omzetbreuk(instorting) is None
    assert screener._calc_revenue_cagr(instorting) < -0.02
    print("  [OK] een verdubbeling met meegroeiende brutowinst is groei, geen breuk")


def test_zonder_brutowinst_blijft_de_voorzichtige_regel():
    """Geen brutowinst (banken, of Yahoo mist hem): een sprong is een breuk."""
    snel = _jaren([40e6, 95e6, 150e6, 210e6])
    assert exit_regels.omzetbreuk(snel)
    assert screener._calc_revenue_cagr(snel) is None

    # Brutowinst in maar één van de twee jaren bevestigt niets.
    half = _jaren([40e6, 95e6, 150e6, 210e6], bruto=[None, 50e6, 82e6, 118e6])
    assert exit_regels.omzetbreuk(half)

    # Brutowinst die nauwelijks meebeweegt: ook breuk.
    traag = _jaren([40e6, 95e6, 150e6, 210e6], bruto=[22e6, 24e6, 26e6, 28e6])
    assert exit_regels.omzetbreuk(traag)
    print("  [OK] zonder bevestiging door de brutowinst blijft een sprong een breuk")


def test_groeier_uit_het_verlies_is_geen_breuk():
    """De gaten die de lijst van fase A blootlegde (docs/fase-a-breuken.csv).

    De eerste versie vergeleek de brutowinst op logschaal en eiste daarvoor twee
    positieve getallen. Een groeier die uit het verlies komt viel daardoor
    altijd als 'definitiewissel' uit het tabblad Groeiers.
    """
    # Alvotech: omzet ×5,4, brutowinst van −697 naar +3.057 mln (echte cijfers).
    alvotech = _jaren([833.8e6, 918.2e6, 4917.7e6, 5888.2e6],
                      bruto=[190.1e6, -697.2e6, 3056.7e6, 3522.6e6])
    assert exit_regels.omzetbreuk(alvotech) is None
    assert screener._calc_revenue_cagr(alvotech) > 0.8

    # ITM Power: beide jaren negatieve brutowinst, die wel fors verbetert.
    itm = _jaren([5.6e6, 5.2e6, 16.5e6, 26.0e6], bruto=[-60e6, -79.1e6, -16.7e6, -5e6])
    assert exit_regels.omzetbreuk(itm) is None

    # Xbrane-patroon: omzet ×4 omhoog, brutowinst omláág — dat blijft een breuk.
    tegen = _jaren([57.7e6, 238.8e6, 148.1e6, 152.4e6], bruto=[57.6e6, 35.4e6, 30e6, 31e6])
    assert exit_regels.omzetbreuk(tegen)
    print("  [OK] een groeier die uit het verlies komt is geen definitiewissel")


def test_kleine_omzet_blijft_een_breuk():
    """Onder een miljoen omzet is een verdubbeling ruis, ook mét brutowinst."""
    klein = _jaren([40e3, 95e3, 150e3, 210e3], bruto=[20e3, 50e3, 80e3, 110e3])
    breuk = exit_regels.omzetbreuk(klein)
    assert breuk and "ruis" in breuk
    assert screener._calc_revenue_cagr(klein) is None
    print("  [OK] bij een omzet onder een miljoen blijft een sprong een breuk")


def test_breuk_komt_als_waarschuwing_in_beeld():
    """De groei verdwijnt niet geruisloos: run_ticker zet de reden in warnings."""
    adyen = _jaren([8936e6, 1863e6, 2226e6, 2647e6])
    breuk = exit_regels.omzetbreuk(screener._omzetvenster(adyen))
    assert breuk and "definitie" in breuk
    print("  [OK] de breuktekst is beschikbaar voor de waarschuwing op de aandeelpagina")


# ---------------------------------------------------------------------------
# De R&D-kolom staat overal waar een jaarcijferveld moet staan
# ---------------------------------------------------------------------------

def _valid_override_fields() -> set[str]:
    """Leest VALID_OVERRIDE_FIELDS uit app.py zonder de Flask-app op te starten."""
    with open(os.path.join(WORTEL, "app.py"), encoding="utf-8") as f:
        boom = ast.parse(f.read())
    for knoop in boom.body:
        if isinstance(knoop, ast.Assign) and any(
                getattr(t, "id", None) == "VALID_OVERRIDE_FIELDS" for t in knoop.targets):
            return set(ast.literal_eval(knoop.value))
    raise AssertionError("VALID_OVERRIDE_FIELDS niet gevonden in app.py")


def test_rd_expense_staat_overal():
    assert "rd_expense" in db.FINANCIAL_VELDEN, "handmatige jaarrij mist hem anders"
    assert db.lege_jaarrij(2025)["rd_expense"] is None
    assert "rd_expense" in data_fetcher._MONEY_FIELDS_PER_ROW, "geen FX-conversie anders"
    assert "rd_expense" in _valid_override_fields(), "niet handmatig te corrigeren anders"
    assert data_fetcher._als_kosten(-12.5) == 12.5
    assert data_fetcher._als_kosten(None) is None
    print("  [OK] rd_expense staat in FINANCIAL_VELDEN, de FX-lijst en de overrides")


def test_rd_expense_niet_in_datakwaliteit():
    """Bewust niet kritiek: geen R&D-regel is geen datagebrek."""
    from engine import data_quality
    assert "rd_expense" not in data_quality._CRITICAL_ANNUAL_FIELDS
    print("  [OK] rd_expense telt niet mee in de volledigheid van de cijfers")


if __name__ == "__main__":
    fout = 0
    for naam, functie in list(globals().items()):
        if naam.startswith("test_") and callable(functie):
            try:
                functie()
            except AssertionError as e:
                fout += 1
                print(f"  [FOUT] {naam}: {e}")
    print(f"FALEND: {fout}")
    sys.exit(1 if fout else 0)

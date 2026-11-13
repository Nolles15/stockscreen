"""De historische multiples, tegen met de hand nagerekende voorbeelden.

Aanleiding (review november 2026): de bedrijfswaarde werd berekend als
beurswaarde + min(schuld, kas) in plaats van beurswaarde + schuld − kas, en de
koers van toen was achteraf verlaagd met dividenden. Geen van beide gaf een
foutmelding; de multiples zagen er geloofwaardig uit. Vandaar rekenvoorbeelden
met ronde getallen, zodat de uitkomst uit het hoofd te controleren is.

Draaien: python -m pytest tests/test_historische_multiples.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.data_fetcher import (  # noqa: E402
    _boekjaar_maand,
    _historical_year_end_price,
    multiples_voor_jaar,
)


def _rij(schuld, kas, **extra):
    """100 aandelen, EBITDA 50, vrije kasstroom 25, winst 2 en boekwaarde 5 per aandeel."""
    rij = {
        "shares_outstanding": 100, "ebitda": 50, "fcf": 25,
        "eps_diluted": 2.0, "book_value_ps": 5.0,
        "total_debt": schuld,
        "net_cash": None if kas is None else kas - schuld,
    }
    rij.update(extra)
    return rij


def test_bedrijf_met_nettoschuld():
    # Koers 10 x 100 aandelen = beurswaarde 1.000. Schuld 100, kas 20 →
    # nettoschuld 80 → bedrijfswaarde 1.080. De oude regel gaf 1.020.
    m = multiples_voor_jaar(_rij(schuld=100, kas=20), 10.0)
    assert m["ev_ebitda"] == 1080 / 50
    assert m["ev_fcf"] == 1080 / 25


def test_kasrijk_bedrijf():
    # Schuld 20, kas 100 → nettoschuld −80 → bedrijfswaarde 920. De oude regel
    # telde hier 20 bij op en kwam op 1.020.
    m = multiples_voor_jaar(_rij(schuld=20, kas=100), 10.0)
    assert m["ev_ebitda"] == 920 / 50
    assert m["ev_fcf"] == 920 / 25


def test_kas_onbekend_telt_alleen_de_schuld():
    m = multiples_voor_jaar(_rij(schuld=100, kas=None), 10.0)
    assert m["ev_ebitda"] == 1100 / 50


def test_schuld_onbekend_geeft_geen_bedrijfswaarde():
    m = multiples_voor_jaar(_rij(schuld=None, kas=None), 10.0)
    assert m["ev_ebitda"] is None and m["ev_fcf"] is None
    # De koersgebonden multiples hebben de schuld niet nodig.
    assert m["pe_ratio"] == 5.0 and m["pb_ratio"] == 2.0


def test_negatieve_bedrijfswaarde_geeft_geen_veelvoud():
    # Kas 2.000 tegen een beurswaarde van 1.000: bedrijfswaarde −1.000.
    m = multiples_voor_jaar(_rij(schuld=0, kas=2000), 10.0)
    assert m["ev_ebitda"] is None and m["ev_fcf"] is None


def test_verlies_en_negatieve_kasstroom_geven_geen_veelvoud():
    m = multiples_voor_jaar(_rij(schuld=100, kas=20, eps_diluted=-1.0, fcf=-5, ebitda=0), 10.0)
    assert m["pe_ratio"] is None
    assert m["ev_fcf"] is None
    assert m["ev_ebitda"] is None


def _maandkoersen():
    # Koers = maandnummer vanaf januari 2024, zodat de gekozen maand afleesbaar is:
    # jan 2024 = 1, dec 2024 = 12, mei 2025 = 17, dec 2025 = 24.
    index = pd.date_range("2024-01-01", periods=24, freq="MS")
    return pd.DataFrame({"Close": [float(i) for i in range(1, 25)]}, index=index)


def test_kalenderboekjaar_pakt_december():
    assert _historical_year_end_price(_maandkoersen(), 2024) == 12.0
    assert _historical_year_end_price(_maandkoersen(), 2025, 12) == 24.0


def test_gebroken_boekjaar_pakt_de_eigen_slotmaand():
    # Boekjaar 2025 dat in mei eindigt hoort bij de koers van mei 2025 (17),
    # niet bij die van december 2025 (24).
    assert _historical_year_end_price(_maandkoersen(), 2025, 5) == 17.0


def test_ontbrekende_slotmaand_valt_terug_op_de_maand_ervoor():
    koersen = _maandkoersen().drop(pd.Timestamp("2025-05-01"))
    assert _historical_year_end_price(koersen, 2025, 5) == 16.0


def test_jaar_buiten_de_reeks_geeft_niets():
    assert _historical_year_end_price(_maandkoersen(), 2020) is None
    assert _historical_year_end_price(pd.DataFrame(), 2024) is None


def test_boekjaar_maand():
    assert _boekjaar_maand("2026-03-31") == 3
    assert _boekjaar_maand("2025-12-31") == 12
    assert _boekjaar_maand(None) == 12
    assert _boekjaar_maand("onzin") == 12

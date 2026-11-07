"""
De machine mag zichzelf alleen stoppen als het stil is én er geen ronde loopt.

Aanleiding: op 7 nov 2026 stopte Fly's proxy de koersronde halverwege. Wat hier
niet mag sneuvelen: een lopende ronde houdt de machine wakker, en een job die
blijft hangen houdt hem níét eeuwig wakker.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.stilte import Stiltebewaker  # noqa: E402


class Klok:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def minuten(self, m):
        self.t += m * 60


def test_stopt_niet_zolang_er_verzoeken_komen():
    klok = Klok()
    b = Stiltebewaker(10, 3, klok=klok)
    for _ in range(30):
        klok.minuten(5)
        b.verzoek()
        assert not b.moet_stoppen()


def test_stopt_na_stilte():
    klok = Klok()
    b = Stiltebewaker(10, 3, klok=klok)
    klok.minuten(9)
    assert not b.moet_stoppen()
    klok.minuten(2)
    assert b.moet_stoppen()


def test_lopende_ronde_houdt_wakker():
    klok = Klok()
    b = Stiltebewaker(10, 3, klok=klok)
    with b.werk():
        klok.minuten(60)
        assert not b.moet_stoppen()
    assert b.moet_stoppen()


def test_ander_werk_telt_mee():
    klok = Klok()
    bezig = {"ja": True}
    b = Stiltebewaker(10, 3, ander_werk=lambda: bezig["ja"], klok=klok)
    klok.minuten(30)
    assert not b.moet_stoppen()
    bezig["ja"] = False
    assert b.moet_stoppen()


def test_hangende_job_houdt_niet_eeuwig_wakker():
    klok = Klok()
    b = Stiltebewaker(10, 3, ander_werk=lambda: True, klok=klok)
    klok.minuten(179)
    assert not b.moet_stoppen()
    klok.minuten(2)
    assert b.moet_stoppen()


def test_werk_teller_herstelt_na_fout():
    klok = Klok()
    b = Stiltebewaker(10, 3, klok=klok)
    try:
        with b.werk():
            raise RuntimeError("ronde crasht")
    except RuntimeError:
        pass
    assert not b.bezig()


if __name__ == "__main__":
    for naam, f in list(globals().items()):
        if naam.startswith("test_") and callable(f):
            f()
            print(f"  [OK] {naam}")
    print("ALLE TESTS GESLAAGD")

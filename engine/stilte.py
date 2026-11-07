"""
De machine stopt zichzelf als er niets te doen is (sinds 2026-11-07).

Waarom niet Fly's eigen auto-stop: de proxy kijkt alleen of er op het moment
van controleren een verbinding openstaat, niet of er een ronde loopt. Op
7 november stopte hij zo de koersronde halverwege, terwijl de wekker elke
minuut een verzoek stuurde — een verzoek van een halve seconde valt tussen
twee controles in.

Hier beslist de app zelf: stoppen pas als er `stil_minuten` geen verzoek
binnenkwam én er geen werk loopt. Een job die als 'bezig' blijft hangen houdt
de machine hooguit `max_werk_uren` wakker; daarna wint stilte toch.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable


class Stiltebewaker:
    def __init__(self, stil_minuten: float, max_werk_uren: float,
                 ander_werk: Callable[[], bool] = lambda: False,
                 klok: Callable[[], float] = time.monotonic):
        self.stil_minuten = stil_minuten
        self.max_werk_uren = max_werk_uren
        self._ander_werk = ander_werk
        self._klok = klok
        self._laatste_verzoek = klok()
        self._bezig = 0
        self._lock = threading.Lock()

    def verzoek(self) -> None:
        """Aanroepen bij elk binnenkomend HTTP-verzoek."""
        self._laatste_verzoek = self._klok()

    @contextmanager
    def werk(self):
        """Zolang dit blok loopt, stopt de machine niet."""
        with self._lock:
            self._bezig += 1
        try:
            yield
        finally:
            with self._lock:
                self._bezig -= 1

    def bezig(self) -> bool:
        return self._bezig > 0 or bool(self._ander_werk())

    def stil_sinds_minuten(self) -> float:
        return (self._klok() - self._laatste_verzoek) / 60

    def moet_stoppen(self) -> bool:
        stil = self.stil_sinds_minuten()
        if stil < self.stil_minuten:
            return False
        if self.bezig() and stil < self.max_werk_uren * 60:
            return False
        return True

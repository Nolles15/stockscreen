"""
cache.py — houdt dure databaseresultaten even vast.

Aanleiding: Neon rekent het verkeer af dat de database over het publieke
internet verlaat, en op 21 augustus 2026 was 80% van de gratis 5 GB per maand
op. De oorzaak was niet groei maar herhaling: `get_dashboard_data()` haalt alle
2.812 rijen op (ruim 2 MB) en draaide bij élke paginalading — en zelfs bij elke
druk op een knop die maar één ticker nodig had.

Bewust klein gehouden. Geen sleutels, geen geheugengrens, geen verdringing:
er is precies één ding dat groot en vaak is, en dat is deze ene query. Een
algemene cachelaag zou meer beloven dan hier nodig is.

**Wat hier NIET in hoort.** Alleen de kale databaserijen. Signaal en korting
worden bewust live herrekend tegen de verse koers (`_effective_signal` in
app.py); die uitkomsten cachen zou een eerder opgeloste bug terugbrengen waarbij
een verse koers tegen een oude berekening werd gehouden.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


class Houder:
    """Eén waarde, opnieuw opgehaald als hij leeg of te oud is.

    Draadveilig: gunicorn draait één worker met acht threads, dus meerdere
    verzoeken kunnen tegelijk om dezelfde waarde vragen. Zonder slot halen ze
    hem dan alle acht op — precies het verkeer dat we wilden vermijden.
    """

    def __init__(self, naam: str, ttl_seconden: float = 900):
        self.naam = naam
        self.ttl = ttl_seconden
        self._waarde: Any = None
        self._gevuld = False
        self._gezet_op: float = 0.0
        self._lock = threading.Lock()
        # Alleen om te kunnen zien of het werkt; zie de tellers in /api/health.
        self.treffers = 0
        self.missers = 0

    def haal(self, maak: Callable[[], Any]) -> Any:
        """De waarde uit de cache, of opnieuw opgehaald via `maak`."""
        with self._lock:
            vers = (self._gevuld
                    and (time.monotonic() - self._gezet_op) < self.ttl)
            if vers:
                self.treffers += 1
                return self._waarde

            # Binnen het slot ophalen. Dat laat gelijktijdige verzoeken even
            # wachten, maar één keer wachten is beter dan acht keer dezelfde
            # twee megabyte over het internet trekken.
            self.missers += 1
            waarde = maak()
            # Een leeg resultaat wordt niet vastgehouden. Een echte fout gooit een
            # uitzondering en komt hier niet, dus leeg betekent in de praktijk dat
            # er iets ongewoons aan de hand is — en dan wil je geen kwartier lang
            # een leeg dashboard tonen omdat de cache het braaf onthoudt.
            self._waarde = waarde
            self._gevuld = bool(waarde)
            self._gezet_op = time.monotonic()
            return waarde

    def leeg(self, reden: str = "") -> None:
        """Gooi de waarde weg, zodat de volgende vraag hem opnieuw ophaalt."""
        with self._lock:
            was_gevuld = self._gevuld
            self._waarde = None
            self._gevuld = False
            self._gezet_op = 0.0
        if was_gevuld:
            log.info("Cache %s geleegd%s", self.naam, f" ({reden})" if reden else "")

    def stand(self) -> dict:
        with self._lock:
            leeftijd = (time.monotonic() - self._gezet_op) if self._gevuld else None
        return {
            "naam": self.naam,
            "gevuld": leeftijd is not None,
            "leeftijd_s": round(leeftijd) if leeftijd is not None else None,
            "treffers": self.treffers,
            "missers": self.missers,
        }


# De enige cache in het project. Een kwartier is de terugval voor het geval een
# invalidatie ergens vergeten wordt; normaal loopt hij leeg zodra er iets
# verandert, niet als de tijd om is.
dashboard = Houder("dashboard", ttl_seconden=900)

"""Kill switch (Architecture §7.11): hard stop independent of every other component.

Trips on: explicit `trip()`, a signal (SIGUSR1 / SIGINT if installed), a sentinel file
(`KILL` by default), or the watchdog when a registered component stops beating.
Tripping runs every registered stop handler (release inputs, cancel reflexes, disconnect)
exactly once, even if some handlers raise.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from typing import Callable


class KillSwitch:
    def __init__(self, sentinel: str | None = None) -> None:
        self.tripped = False
        self.reason = ""
        self.sentinel = sentinel
        self._handlers: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._beats: dict[str, tuple[float, float]] = {}   # name -> (last_beat, timeout_s)
        self._watchdog: threading.Thread | None = None
        self.errors: list[str] = []

    # -- wiring -------------------------------------------------------------------------
    def on_trip(self, handler: Callable[[], None]) -> None:
        self._handlers.append(handler)
        if self.tripped:          # late registration still gets stopped
            self._run(handler)

    def install_signal_handlers(self, signals=(signal.SIGUSR1,)) -> None:
        for s in signals:
            signal.signal(s, lambda signum, frame: self.trip(f"signal {signum}"))

    # -- tripping -----------------------------------------------------------------------
    def _run(self, h: Callable[[], None]) -> None:
        try:
            h()
        except Exception as e:  # a failing handler must not stop the others
            self.errors.append(f"{getattr(h, '__name__', h)}: {e!r}")

    def trip(self, reason: str) -> None:
        with self._lock:
            if self.tripped:
                return
            self.tripped, self.reason = True, reason
            handlers = list(self._handlers)
        for h in handlers:
            self._run(h)

    def check(self) -> bool:
        """Poll external triggers; returns True if tripped. Call first in every loop."""
        if not self.tripped and self.sentinel and os.path.exists(self.sentinel):
            self.trip(f"sentinel file {self.sentinel}")
        if not self.tripped:
            now = time.monotonic()
            for name, (last, timeout) in list(self._beats.items()):
                if now - last > timeout:
                    self.trip(f"watchdog: {name} silent for {now - last:.2f}s (> {timeout}s)")
                    break
        return self.tripped

    # -- watchdog -----------------------------------------------------------------------
    def watch(self, name: str, timeout_s: float) -> None:
        self._beats[name] = (time.monotonic(), timeout_s)

    def beat(self, name: str) -> None:
        if name in self._beats:
            self._beats[name] = (time.monotonic(), self._beats[name][1])

    def unwatch(self, name: str) -> None:
        self._beats.pop(name, None)

    def start_watchdog(self, period_s: float = 0.05) -> None:
        """Background thread so a hung main loop still gets stopped."""
        if self._watchdog:
            return

        def loop():
            while not self.tripped:
                self.check()
                time.sleep(period_s)

        self._watchdog = threading.Thread(target=loop, name="killswitch-watchdog", daemon=True)
        self._watchdog.start()

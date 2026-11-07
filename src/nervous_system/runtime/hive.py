"""Hive mode (Architecture §7.10; Roadmap Phase 6): many bots, one brain, one batched Laya.

- A shared `Bus` carries danger warnings, resource/danger markers in world coordinates and
  task claims (`hive.*` topics).
- Laya decisions from every bot due this cycle are scored in one `decide_batch` call.
- The single brain sees each bot's summary plus a hive status block (claims, markers).
Game-agnostic: the world stepping function is passed in by the wiring layer.
"""
from __future__ import annotations

import math
import time
from typing import Callable

from ..protocol import Event
from .bus import Bus
from .loop import BotRuntime


class DecisionBatcher:
    def __init__(self, model):
        self.model = model
        self.pending: list[tuple[BotRuntime, tuple]] = []
        self.batches = 0
        self.batched_decisions = 0
        self.max_batch = 0
        self.ms = 0.0

    def add(self, rt: BotRuntime, req: tuple) -> None:
        self.pending.append((rt, req))

    def flush(self) -> None:
        if not self.pending:
            return
        items = [(req[2], req[3]) for _, req in self.pending]
        t0 = time.perf_counter()
        dists = self.model.decide_batch(items)
        ms = (time.perf_counter() - t0) * 1000
        self.batches += 1
        self.batched_decisions += len(items)
        self.max_batch = max(self.max_batch, len(items))
        self.ms += ms
        per = ms / len(items)
        for (rt, req), d in zip(self.pending, dists):
            rt.commit_decision(req, d, per, False)
        self.pending = []


class HiveCoordinator:
    """Claims + markers on the bus; renders a status block for the brain."""

    def __init__(self, bus: Bus, marker_ttl_s: float = 20.0):
        self.bus = bus
        self.claims: dict[str, str] = {}             # bot_id -> task
        self.markers: list[dict] = []                # {kind, x, y, t, by}
        self.marker_ttl_s = marker_ttl_s
        self.messages = 0
        bus.subscribe("hive.*", self._on)

    def _on(self, topic: str, msg: dict) -> None:
        self.messages += 1
        if topic == "hive.claim":
            self.claims[msg["bot"]] = msg["task"]
        elif topic == "hive.release":
            self.claims.pop(msg["bot"], None)
        elif topic == "hive.marker":
            self.markers.append(msg)

    def claim(self, bot: str, task: str) -> None:
        self.bus.publish("hive.claim", {"bot": bot, "task": task})

    def mark(self, kind: str, x: float, y: float, t: float, by: str) -> None:
        self.bus.publish("hive.marker", {"kind": kind, "x": x, "y": y, "t": t, "by": by})

    def active_markers(self, t: float, kind: str | None = None) -> list[dict]:
        self.markers = [m for m in self.markers if t - m["t"] <= self.marker_ttl_s]
        return [m for m in self.markers if kind is None or m["kind"] == kind]

    def status(self, t: float) -> str:
        claims = ", ".join(f"{b}:{task}" for b, task in sorted(self.claims.items())) or "none"
        dangers = self.active_markers(t, "danger")
        d = ", ".join(f"({m['x']:.0f},{m['y']:.0f})" for m in dangers[-5:]) or "none"
        return f"hive claims: {claims}\nhive danger markers: {d}"


class HiveRuntime:
    def __init__(self, runtimes: list[BotRuntime], step_world: Callable[[], None], bus: Bus,
                 model, coordinator: HiveCoordinator, positions: Callable[[BotRuntime], tuple]):
        self.rts = runtimes
        self.step_world = step_world
        self.bus = bus
        self.batcher = DecisionBatcher(model)
        self.coord = coordinator
        self.positions = positions
        for rt in runtimes:
            rt.batcher = self.batcher
            rt.model = model
            rt.summary_extra = (lambda r=rt: self.coord.status(r.t))
            bus.subscribe(f"event.{rt.hdr['bot_id']}", self._on_event)

    def _on_event(self, topic: str, ev: Event) -> None:
        if ev.kind == "event" and ev.description.startswith("new hostile") and ev.salience >= 0.5:
            rt = next(r for r in self.rts if r.hdr["bot_id"] == ev.bot_id)
            x, y = self.positions(rt)
            self.coord.mark("danger", x, y, ev.t, ev.bot_id)

    def _refresh_claims(self) -> None:
        for rt in self.rts:
            cur = rt.goal.intent if (rt.goal and not rt.done) else None
            if self.coord.claims.get(rt.hdr["bot_id"]) != cur:
                if cur is None:
                    self.bus.publish("hive.release", {"bot": rt.hdr["bot_id"]})
                else:
                    self.coord.claim(rt.hdr["bot_id"], cur)

    def run(self, max_ticks: int) -> int:
        for rt in self.rts:
            rt.start()
            self._refresh_claims()
        n = 0
        while n < max_ticks:
            live = [rt for rt in self.rts if not rt.done]
            if not live:
                break
            for rt in live:
                rt.step()
            self._refresh_claims()
            self.batcher.flush()
            alive = [rt.after_decisions() for rt in live if not rt.done]
            self.step_world()
            n += 1
            if not any(alive):
                break
        for rt in self.rts:
            rt.finish()
        return n

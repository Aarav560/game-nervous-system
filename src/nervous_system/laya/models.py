"""DecisionModel helpers: brain-labelling bootstrap, decision cache, temperature calibration."""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Callable

from . import Calibrator, DecisionModel


class BrainLabelModel(DecisionModel):
    """Phase 1/2 bootstrap: the brain labels every decision (Architecture §12.2).

    Returns a near-one-hot distribution so the escalation policy always acts.
    """

    name = "brain-label"

    def __init__(self, label: Callable[[str, list[str]], str], confidence: float = 0.99):
        self.label = label
        self.confidence = confidence
        self.last_latency_ms = 0.0

    def decide(self, state_text: str, options: list[str]) -> dict[str, float]:
        t0 = time.perf_counter()
        choice = self.label(state_text, options)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        rest = (1 - self.confidence) / max(1, len(options) - 1)
        return {o: (self.confidence if o == choice else rest) for o in options}

    def decide_batch(self, items):
        return [self.decide(s, o) for s, o in items]


class CachedDecisionModel(DecisionModel):
    """LRU cache in front of a deterministic model: identical (state text, options) -> same answer.

    The serializer quantises state (Architecture §5), so repeats are common. Cache hits are still
    logged as decisions by the runtime; `last_cached` tells it this one cost no inference.
    """

    def __init__(self, inner: DecisionModel, maxsize: int = 20000):
        self.inner = inner
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self.hits = self.misses = 0
        self.last_cached = False
        self.last_latency_ms = 0.0
        self.name = getattr(inner, "name", type(inner).__name__)

    def decide(self, state_text: str, options: list[str]) -> dict[str, float]:
        key = (state_text, tuple(options))
        t0 = time.perf_counter()
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            self.last_cached = True
            out = dict(self._cache[key])
        else:
            self.misses += 1
            self.last_cached = False
            out = self.inner.decide(state_text, options)
            self._cache[key] = dict(out)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return out

    def decide_batch(self, items):
        todo = [(s, o) for s, o in items if (s, tuple(o)) not in self._cache]
        uniq = list(dict.fromkeys((s, tuple(o)) for s, o in todo))
        if uniq:
            for (s, o), d in zip(uniq, self.inner.decide_batch([(s, list(o)) for s, o in uniq])):
                self._cache[(s, o)] = d
        self.hits += len(items) - len(uniq)
        self.misses += len(uniq)
        return [dict(self._cache[(s, tuple(o))]) for s, o in items]


class TemperatureCalibrator(Calibrator):
    """Temperature scaling on log-probabilities, fitted by grid + golden-section on NLL (§12.3).

    `fit` takes log-probabilities (or logits) per decision and the index of the label.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    @staticmethod
    def _nll(logits: list[list[float]], labels: list[int], T: float) -> float:
        tot = 0.0
        for z, y in zip(logits, labels):
            m = max(v / T for v in z)
            lse = m + math.log(sum(math.exp(v / T - m) for v in z))
            tot += lse - z[y] / T
        return tot / max(1, len(labels))

    def fit(self, logits: list[list[float]], labels: list[int]) -> None:
        grid = [0.25 * 1.15 ** i for i in range(40)]           # 0.25 .. ~59
        best = min(grid, key=lambda T: self._nll(logits, labels, T))
        lo, hi = best / 1.15, best * 1.15
        g = (math.sqrt(5) - 1) / 2
        for _ in range(40):
            a, b = hi - g * (hi - lo), lo + g * (hi - lo)
            if self._nll(logits, labels, a) < self._nll(logits, labels, b):
                hi = b
            else:
                lo = a
        self.temperature = (lo + hi) / 2

    def apply(self, dist: dict[str, float]) -> dict[str, float]:
        T = self.temperature
        logp = {k: math.log(max(v, 1e-9)) / T for k, v in dist.items()}
        m = max(logp.values())
        e = {k: math.exp(v - m) for k, v in logp.items()}
        s = sum(e.values())
        return {k: v / s for k, v in e.items()}


class CalibratedModel(DecisionModel):
    def __init__(self, inner: DecisionModel, calibrator: Calibrator):
        self.inner, self.calibrator = inner, calibrator
        self.name = getattr(inner, "name", type(inner).__name__)

    def __getattr__(self, item):            # last_cached / last_latency_ms / last from inner
        return getattr(self.inner, item)

    def decide(self, state_text, options):
        return self.calibrator.apply(self.inner.decide(state_text, options))

    def decide_batch(self, items):
        return [self.calibrator.apply(d) for d in self.inner.decide_batch(items)]

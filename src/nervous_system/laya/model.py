"""Laya-backed DecisionModel (Architecture §5).

Wraps the `laya` package (Apache 2.0, convaiinnovations on Hugging Face). Each decision is one
typed `choice` question whose criteria are the shared-vocabulary descriptions of the offered
options. Requires the `laya` extra: `pip install -e .[laya]`.

This class only scores options. Acting vs escalating is `EscalationPolicy`'s job, and logging
the resulting `Decision` is the runtime's job (Architecture §12.1).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .. import vocabulary
from . import DecisionModel

QUESTION_ID = "action"
INSTRUCTIONS = "Which option should the game bot take next, given its goal and current state?"

CHECKPOINTS = {
    "laya-english": "laya",
    "laya-typed-decisions": "typed-decisions",
    "laya-multilingual": "multilingual",
}


@dataclass
class DecisionInfo:
    """Side data from the last call, for the Decision message and the experience log."""

    latency_ms: float = 0.0
    state_tokens: int = 0
    truncated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def build_question(options: list[str]) -> dict[str, dict]:
    vocabulary.validate(options)
    return {QUESTION_ID: {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {o: vocabulary.describe(o) for o in options},
    }}


def _distribution(result: dict, options: list[str]) -> dict[str, float]:
    probs = result["answers"][QUESTION_ID]["probabilities"]
    missing = set(options) - set(probs)
    if missing:
        raise RuntimeError(f"Laya returned no probability for {sorted(missing)}")
    return {o: float(probs[o]) for o in options}


class LayaDecisionModel(DecisionModel):
    def __init__(self, checkpoint: str = "laya-typed-decisions", device: str = "cpu",
                 max_state_tokens: int = 350, agent: Any = None, **load_kwargs: Any):
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {checkpoint!r}; one of {sorted(CHECKPOINTS)}")
        self.checkpoint = checkpoint
        self.max_state_tokens = max_state_tokens
        self.last = DecisionInfo()
        if agent is None:
            os.environ.setdefault("USE_TF", "0")  # avoids a known TF/abseil load deadlock
            import laya
            agent = laya.load(CHECKPOINTS[checkpoint], device=device, **load_kwargs)
        self.agent = agent

    def _check_usage(self, result: dict) -> DecisionInfo:
        usage = result.get("usage", {})
        info = DecisionInfo(state_tokens=int(usage.get("state_tokens", 0)),
                            truncated=bool(usage.get("truncated", False)))
        if info.truncated or info.state_tokens > self.max_state_tokens:
            raise ValueError(
                f"state text is {info.state_tokens} tokens (limit {self.max_state_tokens}, "
                f"truncated={info.truncated}); fix the adapter serializer (Architecture §5)")
        return info

    def decide(self, state_text: str, options: list[str]) -> dict[str, float]:
        q = build_question(options)
        t0 = time.perf_counter()
        result = self.agent.predict(state_text, q)
        latency = (time.perf_counter() - t0) * 1000
        self.last = self._check_usage(result)
        self.last.latency_ms = latency
        return _distribution(result, options)

    def decide_batch(self, items: list[tuple[str, list[str]]]) -> list[dict[str, float]]:
        """One shared forward pass per distinct option set (Architecture §5 Throughput)."""
        out: list[dict[str, float] | None] = [None] * len(items)
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, (_, opts) in enumerate(items):
            groups.setdefault(tuple(opts), []).append(i)
        t0 = time.perf_counter()
        for opts, idxs in groups.items():
            results = self.agent.predict_batch([items[i][0] for i in idxs], build_question(list(opts)))
            for i, r in zip(idxs, results):
                self._check_usage(r)
                out[i] = _distribution(r, list(opts))
        self.last = DecisionInfo(latency_ms=(time.perf_counter() - t0) * 1000)
        return out  # type: ignore[return-value]

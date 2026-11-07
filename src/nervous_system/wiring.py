"""Wiring: builds adapters, brains, decision models and runtimes from names.

This is the one place that knows about every game package; tiers 1–2 never import it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .brain import Brain
from .brain.scripted import ScriptedBrain
from .laya import DecisionModel, EscalationPolicy
from .laya.models import BrainLabelModel, CachedDecisionModel, CalibratedModel, TemperatureCalibrator
from .modules.drives import BasicDrives
from .modules.forward import PhysicsForwardModel
from .modules.salience import RuleSalience
from .runtime.log import ExperienceLog
from .runtime.loop import BotRuntime, RuntimeConfig


@dataclass
class GameSpec:
    name: str
    make_adapter: Callable[..., Any]
    make_playbook: Callable[[], Any]
    scenarios: list[str]
    context: str = "singleplayer"
    address: str | None = None
    config: dict = field(default_factory=dict)


def game_spec(name: str) -> GameSpec:
    if name == "gridcraft":
        from .adapters.gridcraft import SCENARIOS, GridCraftAdapter, GridCraftPlaybook
        from .adapters.gridcraft.adapter import load_config
        return GameSpec(name, GridCraftAdapter, GridCraftPlaybook, list(SCENARIOS), config=load_config())
    if name == "skirmish":
        from .adapters.skirmish import SCENARIOS, SkirmishAdapter, SkirmishPlaybook
        from .adapters.skirmish.adapter import load_config
        return GameSpec(name, SkirmishAdapter, SkirmishPlaybook, list(SCENARIOS), config=load_config())
    raise KeyError(f"unknown game {name!r}")


_LAYA_CACHE: dict[str, DecisionModel] = {}


def laya_model(checkpoint: str, variant: str = "int8", calibration: float | None = None,
               cache: bool = True) -> DecisionModel:
    """Load a Laya checkpoint (registry name or local fine-tuned dir) once per process."""
    key = f"{checkpoint}:{variant}"
    if key not in _LAYA_CACHE:
        from .laya.model import LayaDecisionModel
        os.environ.pop("LAYA_CPU_AMP", None)
        if variant == "bf16":
            os.environ["LAYA_CPU_AMP"] = "bf16"
        if Path(checkpoint).exists():
            import laya
            m = LayaDecisionModel(agent=laya.load(checkpoint, device="cpu"))
            m.checkpoint = Path(checkpoint).name
        else:
            m = LayaDecisionModel(checkpoint, device="cpu")
        if variant == "int8":
            import torch
            m.agent.model.encoder = torch.ao.quantization.quantize_dynamic(
                m.agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8)
        m.name = f"{m.checkpoint}:{variant}"
        _LAYA_CACHE[key] = m
    model: DecisionModel = _LAYA_CACHE[key]
    cal_file = Path(checkpoint) / "gns_calibration.json"
    if calibration is None and cal_file.exists():
        import json
        calibration = json.loads(cal_file.read_text())["temperature"]
    if calibration and abs(calibration - 1.0) > 1e-6:
        model = CalibratedModel(model, TemperatureCalibrator(calibration))
    return CachedDecisionModel(model) if cache else model


def build_runtime(spec: GameSpec, scenario: str, seed: int, log: ExperienceLog, brain: Brain,
                  model: DecisionModel | None, run_id: str, config: RuntimeConfig | None = None,
                  bot_id: str = "bot", adapter: Any = None, **adapter_kw) -> tuple[BotRuntime, Any]:
    adapter = adapter or spec.make_adapter(scenario, seed=seed, **adapter_kw)
    adapter.connect(spec.context, spec.address)
    hdr = {"bot_id": bot_id, "game": spec.name, "episode_id": f"{run_id}-{scenario}-{seed}-{bot_id}"}
    log.start_episode(hdr["episode_id"], spec.name, {"scenario": scenario, "seed": seed})
    lcfg = spec.config.get("laya", {})
    policy = EscalationPolicy(lcfg.get("act_threshold", 0.75), lcfg.get("min_margin", 0.15))
    if model is None:
        model = BrainLabelModel(brain.label)
    cfg = config or RuntimeConfig(cadence_s=lcfg.get("cadence_ms", 100) / 1000,
                                  brain_latency_s=spec.config.get("brain", {}).get("latency_s", 2.0),
                                  heartbeat_s=spec.config.get("brain", {}).get("heartbeat_s", 10.0),
                                  seed=seed)
    rt = BotRuntime(adapter, brain, model, log, hdr, adapter.scenario.mission, cfg, policy=policy,
                    forward=PhysicsForwardModel(), salience=RuleSalience(dict(hdr)),
                    drives=BasicDrives())
    return rt, adapter


def scripted_brain(spec: GameSpec) -> ScriptedBrain:
    return ScriptedBrain(spec.make_playbook())

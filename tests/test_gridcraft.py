"""GridCraft simulator + runtime loop end to end (no Laya download)."""
import pytest

from nervous_system.laya import EscalationPolicy
from nervous_system.laya.models import BrainLabelModel
from nervous_system.modules import KillSwitch
from nervous_system.runtime.log import ExperienceLog
from nervous_system.runtime.loop import RuntimeConfig
from nervous_system.wiring import build_runtime, game_spec, scripted_brain


def _run(tmp_path, scenario, model=None, seed=1, config=None, **kw):
    spec = game_spec("gridcraft")
    log = ExperienceLog(tmp_path)
    brain = scripted_brain(spec)
    rt, a = build_runtime(spec, scenario, seed, log, brain, model, "t", config, **kw)
    st = rt.run()
    return rt, a, st, brain, log


def test_play_rules_refuse_servers():
    spec = game_spec("gridcraft")
    a = spec.make_adapter("survive")
    with pytest.raises(PermissionError):
        a.connect("own_server", "127.0.0.1:1")
    with pytest.raises(RuntimeError):
        a.read_state()


@pytest.mark.parametrize("scenario", ["gather_logs", "diamonds", "duel"])
def test_brain_only_completes(tmp_path, scenario):
    rt, a, st, brain, log = _run(tmp_path, scenario)
    res = a.episode_result()
    assert res["success"], res
    n_dec = log.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert n_dec == st.decisions > 0                         # every decision logged
    assert log.db.execute("SELECT COUNT(*) FROM goals").fetchone()[0] >= 1


def test_serializer_fixed_order_and_budget(tmp_path):
    spec = game_spec("gridcraft")
    a = spec.make_adapter("duel", seed=0)
    a.connect("singleplayer")
    w = a.read_state()
    text = a.describe(w, None, {})
    keys = [l.split()[0] for l in text.splitlines()]
    assert keys == ["GAME", "GOAL", "SELF", "OPP", "ENV", "PRED", "DRIVES", "WM"]
    assert "PRED none" in text and "GOAL none" in text
    assert len(text) / 4 < 350


class Unsure:
    name = "unsure"
    def decide(self, s, o):
        return {k: 1 / len(o) for k in o}
    def decide_batch(self, items):
        return [self.decide(s, o) for s, o in items]


def test_unsure_model_escalates_and_brain_answers_land_late(tmp_path):
    cfg = RuntimeConfig(brain_latency_s=1.0)
    rt, a, st, brain, log = _run(tmp_path, "gather_logs", model=Unsure(), config=cfg)
    assert st.acted == 0 and st.escalations > 0
    # escalation is not re-sent while one is open for the same context
    assert st.escalations < st.decisions / 5
    corr = log.db.execute("SELECT COUNT(*) FROM events WHERE kind='correction'").fetchone()[0]
    assert corr == st.escalations


def test_kill_switch_stops_runtime_and_releases(tmp_path):
    spec = game_spec("gridcraft")
    log = ExperienceLog(tmp_path)
    brain = scripted_brain(spec)
    rt, a = build_runtime(spec, "survive", 1, log, brain, None, "k")
    rt.start()
    for _ in range(50):
        rt.step()
    rt.kill.trip("test")
    assert not rt.step()
    assert a.current is None and a.bot.move == (0.0, 0.0) and not a.connected
    assert log.db.execute("SELECT COUNT(*) FROM events WHERE kind='kill'").fetchone()[0] == 1

"""Skirmish (second game) and hive mode, brain-only, no model download."""
import tempfile
from pathlib import Path

import pytest

from nervous_system.runtime.log import ExperienceLog
from nervous_system.wiring import build_runtime, game_spec, scripted_brain


@pytest.mark.parametrize("scenario", ["supply_run", "gunfight", "hold_out"])
def test_skirmish_runs_and_logs(tmp_path, scenario):
    spec = game_spec("skirmish")
    log = ExperienceLog(tmp_path)
    brain = scripted_brain(spec)
    rt, a = build_runtime(spec, scenario, 1, log, brain, None, "t")
    st = rt.run()
    assert st.decisions > 0
    text = a.describe(a.read_state(), rt.goal, {})
    assert text.splitlines()[0] == "GAME skirmish"
    for opts in spec.config["option_sets"].values():
        from nervous_system import vocabulary
        vocabulary.validate(opts)


def test_hive_one_brain_batched(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_hive
    r = run_hive.episode(5000, 3, "brain", None, True, ExperienceLog(tmp_path), "t")
    assert r["brain_instances"] == 1 and r["max_batch"] == 3
    assert r["hive_messages"] > 0 and r["success"]

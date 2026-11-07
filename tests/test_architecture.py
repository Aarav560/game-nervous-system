"""CLAUDE.md ground rules as tests."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src/nervous_system"
TIER_1_2 = ["brain", "laya", "modules", "runtime", "protocol.py", "state.py", "vocabulary.py"]


def _imports(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            out.append(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            out += [a.name for a in node.names]
    return out


def test_tiers_1_2_never_import_adapters():
    files = []
    for t in TIER_1_2:
        p = SRC / t
        files += list(p.rglob("*.py")) if p.is_dir() else [p]
    bad = [(f.relative_to(SRC), m) for f in files for m in _imports(f) if "adapters" in m]
    assert not bad, bad


def test_brain_never_emits_inputs():
    """The brain only returns Goals / option labels; it never touches adapters or reflexes."""
    for f in (SRC / "brain").rglob("*.py"):
        assert not any(m.endswith(("reflex", "adapters")) for m in _imports(f)), f


def test_pt_files_loaded_weights_only():
    import re
    for f in SRC.rglob("*.py"):
        for line in f.read_text().splitlines():
            if "torch.load(" in line:
                assert "weights_only=True" in line, f

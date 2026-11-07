"""Fine-tune Laya on brain-labelled decisions, then calibrate and evaluate (§5, §12.2, §12.3).

    python scripts/train_laya.py --data data/gridcraft-v1 --base multilingual --out checkpoints/laya-gc-v1
    python scripts/train_laya.py --eval-only --out checkpoints/laya-gc-v1 --data data/gridcraft-v1

Writes `<out>/gns_calibration.json` ({temperature, metrics}); wiring.laya_model() applies the
temperature automatically. Fine-tuning uses `laya.train.finetune` (Apache 2.0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("USE_TF", "0")

from nervous_system.laya import EscalationPolicy  # noqa: E402
from nervous_system.laya.metrics import accuracy, brier, ece  # noqa: E402
from nervous_system.laya.model import QUESTION_ID  # noqa: E402
from nervous_system.laya.models import TemperatureCalibrator  # noqa: E402


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def predict(agent, rows: list[dict], batch: int = 32) -> list[dict[str, float]]:
    """Batched predictions grouped by option set."""
    out: list = [None] * len(rows)
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(json.dumps(r["questions"], sort_keys=True), []).append(i)
    for key, idx in groups.items():
        q = json.loads(key)
        for j in range(0, len(idx), batch):
            chunk = idx[j:j + batch]
            res = agent.predict_batch([rows[i]["state"] for i in chunk], q)
            for i, r in zip(chunk, res):
                out[i] = {k: float(v) for k, v in r["answers"][QUESTION_ID]["probabilities"].items()}
    return out


def metrics(dists, labels, policy: EscalationPolicy) -> dict:
    acts = [policy.should_act(d) for d in dists]
    acted = [(a, top) for a, top, _ in acts]
    n_act = sum(a for a, _ in acted)
    acc_act = sum(a and top == y for (a, top), y in zip(acted, labels)) / max(1, n_act)
    return {"n": len(labels), "accuracy": accuracy(dists, labels), "ece": ece(dists, labels),
            "brier": brier(dists, labels), "acted_frac": n_act / max(1, len(labels)),
            "accuracy_when_acted": acc_act}


def evaluate(ckpt: str, data: Path, policy: EscalationPolicy, max_rows: int = 600, variant: str = "int8") -> dict:
    """Calibrate + score the checkpoint as the runtime will run it (same quantisation)."""
    import random
    import laya
    import torch
    agent = laya.load(ckpt, device="cpu")
    if variant == "int8":
        agent.model.encoder = torch.ao.quantization.quantize_dynamic(
            agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8)
    rng = random.Random(0)
    calib, test = read(data / "calib.jsonl"), read(data / "test.jsonl")
    calib = rng.sample(calib, min(max_rows, len(calib)))
    test = rng.sample(test, min(max_rows, len(test)))
    t0 = time.time()
    dc = predict(agent, calib)
    dt = predict(agent, test)
    yc = [r["expected"][QUESTION_ID] for r in calib]
    yt = [r["expected"][QUESTION_ID] for r in test]
    cal = TemperatureCalibrator()
    keys = [list(d) for d in dc]
    cal.fit([[math.log(max(d[k], 1e-9)) for k in ks] for d, ks in zip(dc, keys)],
            [ks.index(y) for ks, y in zip(keys, yc)])
    res = {"checkpoint": ckpt, "variant": variant, "temperature": cal.temperature,
           "test_raw": metrics(dt, yt, policy),
           "test_calibrated": metrics([cal.apply(d) for d in dt], yt, policy),
           "calib_raw": metrics(dc, yc, policy), "eval_s": round(time.time() - t0, 1)}
    per = {}
    for d, y in zip([cal.apply(d) for d in dt], yt):
        p = per.setdefault(y, [0, 0])
        p[0] += 1
        p[1] += int(max(d, key=d.get) == y)
    res["test_per_label_accuracy"] = {k: round(c / n, 3) for k, (n, c) in per.items()}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/gridcraft-v1"))
    ap.add_argument("--base", default="multilingual")
    ap.add_argument("--out", default=str(ROOT / "checkpoints/laya-gc-v1"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--loss", default="soft-ce", choices=["soft-ce", "rlcd"])
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-max", type=int, default=600)
    ap.add_argument("--variant", default="int8", choices=["fp32", "int8"])
    ap.add_argument("--init", help="start from this fine-tuned checkpoint instead of --base")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    data = Path(a.data)
    policy = EscalationPolicy()
    summary: dict = {}
    if not a.eval_only:
        from laya.train import TrainConfig, finetune
        train_path = data / "train.jsonl"
        if a.max_train:
            rows = read(train_path)[: a.max_train]
            train_path = data / f"train_{a.max_train}.jsonl"
            train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg = TrainConfig(epochs=a.epochs, micro_batch=a.micro_batch, grad_accum=a.grad_accum, loss=a.loss,
                          shuffle_options=("choice",), max_len=a.max_len, freeze_encoder=a.freeze_encoder,
                          log_every=20, calib_frac=0.05, calib_max=200)
        t0 = time.time()
        summary["finetune"] = finetune(str(train_path), a.init or a.base, a.out, cfg, device="cpu")
        summary["train_s"] = round(time.time() - t0, 1)
        print(f"trained in {summary['train_s']}s", flush=True)
    summary["eval"] = evaluate(a.out, data, policy, a.eval_max, a.variant)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "gns_calibration.json").write_text(json.dumps(
        {"temperature": summary["eval"]["temperature"], "data": str(data), "base": a.base,
         "metrics": summary["eval"]}, indent=1, default=str))
    print(json.dumps(summary["eval"], indent=1, default=str))
    (Path(a.out) / "gns_train_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()

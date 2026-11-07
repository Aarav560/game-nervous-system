"""Calibration metrics (Architecture §5, §12.3): expected calibration error and Brier score."""
from __future__ import annotations


def ece(dists: list[dict[str, float]], labels: list[str], bins: int = 10) -> float:
    buckets = [[0, 0.0, 0.0] for _ in range(bins)]   # n, sum conf, sum correct
    for d, y in zip(dists, labels):
        top = max(d, key=d.get)
        conf = d[top]
        b = min(bins - 1, int(conf * bins))
        buckets[b][0] += 1
        buckets[b][1] += conf
        buckets[b][2] += float(top == y)
    n = sum(b[0] for b in buckets) or 1
    return sum(abs(s_conf - s_cor) for _, s_conf, s_cor in buckets) / n


def brier(dists: list[dict[str, float]], labels: list[str]) -> float:
    tot = 0.0
    for d, y in zip(dists, labels):
        tot += sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in d.items())
    return tot / max(1, len(labels))


def accuracy(dists: list[dict[str, float]], labels: list[str]) -> float:
    return sum(max(d, key=d.get) == y for d, y in zip(dists, labels)) / max(1, len(labels))

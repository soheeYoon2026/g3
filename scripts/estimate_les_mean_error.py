"""Estimate the uncertainty of each LES run's MEAN Cd, not its instantaneous scatter.

`cd_std` in the result file is the time-series standard deviation over the sampling
window — the fluctuation of an individual sample, which a finer grid legitimately
increases because it resolves more turbulence. Using it as the error bar on the mean
is far too conservative and would call real deformations inconclusive.

The runner logs a cumulative mean every 900 samples, so block means can be recovered
by differencing, and their scatter gives a standard error of the mean. Blocks of 900
samples are long enough to be roughly independent; if they are not, this still
overstates rather than understates the error.
"""

import argparse
import re
from pathlib import Path

import numpy as np


def running_means(log: Path):
    text = log.read_text(errors="replace")
    rows = re.findall(r"Cd \(표본 (\d+)/(\d+)\) = ([0-9.]+) \+-([0-9.]+)", text)
    return [(int(n), int(total), float(mean), float(std)) for n, total, mean, std in rows]


def block_stats(samples):
    """Recover per-block means from cumulative means and return (mean, sem, blocks)."""
    if len(samples) < 3:
        return None
    counts = np.array([s[0] for s in samples], dtype=float)
    cumulative = np.array([s[2] for s in samples], dtype=float)
    totals = counts * cumulative
    blocks = np.diff(np.concatenate([[0.0], totals])) / np.diff(np.concatenate([[0.0], counts]))
    sem = float(blocks.std(ddof=1) / np.sqrt(len(blocks)))
    return float(cumulative[-1]), sem, blocks


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--cars", nargs="+", default=["carA", "carB"])
ap.add_argument("--variants", nargs="+", default=["blunt_tail", "wide_rear", "raise_roof"])
args = ap.parse_args()

for car in args.cars:
    base_log = args.dir / f"{car}_base.out"
    if not base_log.exists():
        print(f"\n=== {car} === 기준 로그 없음")
        continue
    base = block_stats(running_means(base_log))
    if not base:
        print(f"\n=== {car} === 기준 시계열 부족")
        continue
    base_mean, base_sem, base_blocks = base
    print(f"\n=== {car} ===")
    print(f"  기준 Cd {base_mean:.4f}  블록 {len(base_blocks)}개  "
          f"평균의 표준오차 ±{base_sem:.4f}  (순간 요동 std {base_blocks.std(ddof=1):.4f})")
    print(f"  {'변형':>12s} {'Cd':>8s} {'±SEM':>8s} {'ΔCd':>9s} {'S/N':>5s} {'판정':>10s}")
    print("  " + "-" * 60)
    for name in args.variants:
        log = args.dir / f"{car}_{name}.out"
        if not log.exists():
            print(f"  {name:>12s}   로그 없음")
            continue
        stats = block_stats(running_means(log))
        if not stats:
            print(f"  {name:>12s}   시계열 부족")
            continue
        mean, sem, _ = stats
        delta = mean - base_mean
        noise = (base_sem ** 2 + sem ** 2) ** 0.5
        sn = abs(delta) / noise if noise else float("inf")
        if sn < 2:
            verdict = "판정불가"
        else:
            verdict = "항력 증가" if delta > 0 else "항력 감소"
        print(f"  {name:>12s} {mean:8.4f} {sem:8.4f} {delta:+9.4f} {sn:5.1f} {verdict:>10s}")

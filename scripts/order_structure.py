"""Print structural statistics of frame orders (50 episodes x 500 frames), to compare how they were drawn.

Usage: .venv/bin/python scripts/order_structure.py NAME=ORDER.npy [NAME=ORDER.npy ...]
"""

import sys

import numpy as np

FRAMES = 25_000
PER_EPISODE = 500
BATCH = 8
BLOCK = 400


def report(name: str, s: np.ndarray) -> None:
    print(f"{name}: {s.size} draws")
    c = np.bincount(s, minlength=FRAMES)
    print(f"  exposures per frame: min {c.min()}, max {c.max()}, s.d. {c.std():.3f}")
    windows = [np.unique(s[i:i + FRAMES]).size for i in range(0, s.size - FRAMES + 1, FRAMES)]
    print(f"  distinct frames per {FRAMES} draws: mean {np.mean(windows):.1f}, min {min(windows)}")
    order = np.argsort(s, kind="stable")
    gaps = np.diff(order)[s[order][1:] == s[order][:-1]]
    p1, p50, p99 = np.percentile(gaps, [1, 50, 99])
    print(f"  draws between exposures of a frame: mean {gaps.mean():.0f}, s.d. {gaps.std():.0f}, "
          f"p1 {p1:.0f}, median {p50:.0f}, p99 {p99:.0f}, max {gaps.max()}")
    batches = s[: s.size // BATCH * BATCH].reshape(-1, BATCH)
    distinct = np.mean([np.unique(row).size for row in (batches // PER_EPISODE)[::10]])
    print(f"  distinct episodes per batch: {distinct:.3f}")
    for label, size, part in (("batch", BATCH, batches), ("block", BLOCK, s[: s.size // BLOCK * BLOCK].reshape(-1, BLOCK))):
        t = (part % PER_EPISODE).mean(axis=1)
        d = t - t.mean()
        print(f"  mean frame-in-episode per {label} ({size} draws): mean {t.mean():.1f}, s.d. {t.std():.2f}, "
              f"lag-1 autocorrelation {(d[1:] * d[:-1]).mean() / d.var():+.4f}")


def main() -> int:
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        report(name, np.load(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

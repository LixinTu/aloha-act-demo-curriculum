"""Compare an aloha-train run's per-step training loss with a lerobot-train run's logged loss, window by window.

When both runs used the same frame order (an order replay, or a parity check), each logged window covers
the same batches in both runs, so the paired difference removes the batch-to-batch noise. In deterministic
mode with bit-identical weights the difference is only the log's rounding (3 decimals, at most 0.0005); in
normal mode it shows whether one trainer fits the training batches better or worse, and from which step.

Descriptive only: windows are strongly autocorrelated (the two models drift apart smoothly), so no p-value
is computed and no decision rests on this output.

Usage: .venv/bin/python scripts/paired_loss_curves.py <aloha_train_dir> <lerobot_train.log> --log-freq N
       [--block STEPS] [--output PATH]
<aloha_train_dir> holds loss_curve.npy (columns loss, l1, kl, grad norm; one row per step).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

LINE = re.compile(
    r"step:\S+ .*?loss:(?P<loss>[0-9.]+) grdn:(?P<grdn>[0-9.]+) lr:(?P<lr>\S+) .*?"
    r"l1_loss:(?P<l1>[0-9.]+) kld_loss:(?P<kld>[0-9.]+)"
)
COLUMNS = {"loss": 0, "l1": 1, "kld": 2, "grdn": 3}  # lerobot log name -> loss_curve.npy column


def parse_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").replace("\r", "\n").splitlines():
        m = LINE.search(line)
        if m:
            rows.append({k: float(v) for k, v in m.groupdict().items()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("aloha_train_dir", type=Path)
    ap.add_argument("lerobot_log", type=Path)
    ap.add_argument("--log-freq", type=int, required=True, help="lerobot-train's --log_freq")
    ap.add_argument("--block", type=int, default=10000, help="steps per summary block")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    curve = np.load(args.aloha_train_dir / "loss_curve.npy")
    rows = parse_log(args.lerobot_log)
    steps = curve.shape[0]
    if len(rows) != steps // args.log_freq:
        raise SystemExit(f"{len(rows)} log windows, expected {steps // args.log_freq} for {steps} steps")
    if args.block % args.log_freq:
        raise SystemExit("--block must be a multiple of --log-freq")
    lrs = sorted({r["lr"] for r in rows})

    n = len(rows)
    ours = {k: curve[: n * args.log_freq, c].reshape(n, args.log_freq).mean(axis=1) for k, c in COLUMNS.items()}
    theirs = {k: np.array([r[k] for r in rows]) for k in COLUMNS}
    diff = {k: ours[k] - theirs[k] for k in COLUMNS}

    per_block = args.block // args.log_freq
    blocks = []
    for start in range(0, n, per_block):
        sl = slice(start, min(start + per_block, n))
        rec = {"steps": [start * args.log_freq + 1, sl.stop * args.log_freq], "windows": sl.stop - start}
        for k in ("loss", "l1", "grdn"):
            d, ref = diff[k][sl], theirs[k][sl]
            rec[k] = {
                "aloha_train_mean": float(ours[k][sl].mean()),
                "lerobot_train_mean": float(ref.mean()),
                "mean_diff": float(d.mean()),
                "relative_diff": float(d.mean() / ref.mean()) if ref.mean() else None,
                "share_aloha_train_higher": float((d > 0).mean()),
                "max_abs_diff": float(np.abs(d).max()),
            }
        blocks.append(rec)

    report = {
        "aloha_train_dir": str(args.aloha_train_dir),
        "lerobot_log": str(args.lerobot_log),
        "steps": steps,
        "log_freq": args.log_freq,
        "windows": n,
        "lerobot_lr_values": lrs,
        "log_rounding": 0.0005,
        "max_abs_diff": {k: float(np.abs(diff[k]).max()) for k in COLUMNS},
        "blocks": blocks,
    }
    print(f"{n} windows of {args.log_freq} steps; lerobot-train lr values {lrs}; "
          f"max |diff| loss {report['max_abs_diff']['loss']:.4f}, grad norm {report['max_abs_diff']['grdn']:.4f}")
    print(f"{'steps':>15} {'loss ours':>10} {'loss theirs':>11} {'diff':>8} {'rel':>7} {'ours>':>6}  "
          f"{'gn ours':>8} {'gn theirs':>9} {'rel':>7}")
    for b in blocks:
        lo, gn = b["loss"], b["grdn"]
        print(f"{b['steps'][0]:>7}-{b['steps'][1]:<7} {lo['aloha_train_mean']:>10.4f} {lo['lerobot_train_mean']:>11.4f} "
              f"{lo['mean_diff']:>+8.4f} {lo['relative_diff'] * 100:>+6.1f}% {lo['share_aloha_train_higher']:>6.2f}  "
              f"{gn['aloha_train_mean']:>8.3f} {gn['lerobot_train_mean']:>9.3f} {gn['relative_diff'] * 100:>+6.1f}%")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Helpers for scripts/check_trainer_parity.sh.

order:   writes the first steps x batch frame indices of the order lerobot-train v0.6.1 uses (a pure
         function of seed and epoch) and, if asked, a copy of the fixed protocol in deterministic mode.
compare: compares two model.safetensors tensor by tensor and exits 1 unless all are bit-identical.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def order(args) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.sampler import EpisodeAwareSampler

    protocol = json.loads(args.protocol.read_text())
    ds, tr = protocol["dataset"], protocol["training"]
    meta = LeRobotDatasetMetadata(ds["repo_id"], root=args.dataset_root, revision=ds["revision"])
    sampler = EpisodeAwareSampler(
        meta.episodes["dataset_from_index"], meta.episodes["dataset_to_index"], shuffle=True, seed=tr["seed"]
    )
    need = args.steps * tr["batch_size"]
    indices: list[int] = []
    while len(indices) < need:
        indices.extend(iter(sampler))
    np.save(args.output_order, np.asarray(indices[:need], dtype=np.int64))
    if args.output_protocol:
        protocol["training"]["cudnn_deterministic"] = True
        args.output_protocol.write_text(json.dumps(protocol, indent=2) + "\n")
    return 0


def compare(args) -> int:
    from safetensors.torch import load_file

    a, b = load_file(args.first), load_file(args.second)
    if set(a) != set(b):
        print(f"different tensor names: {sorted(set(a) ^ set(b))}")
        return 1
    differing = {k: float((a[k].double() - b[k].double()).abs().max()) for k in sorted(a) if not a[k].equal(b[k])}
    identical = len(a) - len(differing)
    # Size of the difference over all elements, not just the single worst one: root-mean-square
    # difference and the relative Frobenius norm ||a - b|| / ||a|| (normal-mode noise comparisons).
    sq_diff = sum(float(((a[k].double() - b[k].double()) ** 2).sum()) for k in a)
    sq_norm = sum(float((a[k].double() ** 2).sum()) for k in a)
    elements = sum(a[k].numel() for k in a)
    result = {
        "steps": args.steps,
        "lerobot_save_freq": args.save_freq,
        "tensors": len(a),
        "bit_identical": identical,
        "differing": len(differing),
        "worst_abs_diff": max(differing.values(), default=0.0),
        "rms_diff": (sq_diff / elements) ** 0.5,
        "rel_fro": (sq_diff / sq_norm) ** 0.5 if sq_norm else 0.0,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(a)} tensors, {identical} bit-identical, {len(differing)} differ, worst abs diff "
          f"{result['worst_abs_diff']:.3e}, rms diff {result['rms_diff']:.3e}, relative Frobenius {result['rel_fro']:.3e}")
    return 1 if differing else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    o = sub.add_parser("order")
    o.add_argument("--protocol", type=Path, required=True)
    o.add_argument("--dataset-root", type=Path, required=True)
    o.add_argument("--steps", type=int, required=True)
    o.add_argument("--output-order", type=Path, required=True)
    o.add_argument("--output-protocol", type=Path, help="also write a copy of the protocol in deterministic mode")
    o.set_defaults(run=order)
    c = sub.add_parser("compare")
    c.add_argument("first", type=Path)
    c.add_argument("second", type=Path)
    c.add_argument("--steps", type=int, required=True)
    c.add_argument("--save-freq", type=int, help="how often lerobot-train saved a checkpoint, recorded in the result")
    c.add_argument("--output", type=Path, required=True)
    c.set_defaults(run=compare)
    args = ap.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())

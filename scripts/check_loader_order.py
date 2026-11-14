"""Check that a rebuilt frame order is the one lerobot-train's own loader chain produces over several epochs.

Same chain as lerobot_train.py (LeRobot v0.6.1): EpisodeAwareSampler, DataLoader (spawn, persistent, prefetch 4,
4 workers), accelerate's prepare, cycle. The dataset is a stub that returns its index, and nothing touches the GPU.
Run with the control venv, which has accelerate.

Usage: CUDA_VISIBLE_DEVICES= .venv-control/bin/python scripts/check_loader_order.py --order ORDER.npy --protocol P.json [--steps 9575]
Exit 0: the first steps x batch entries are identical. Exit 1: they differ.
"""

import argparse
import json
import sys

import numpy as np
import torch


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return int(i)


def main() -> int:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from lerobot.datasets import EpisodeAwareSampler
    from lerobot.scripts.lerobot_train import cycle

    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--steps", type=int, default=9575, help="default: a little over three epochs")
    args = ap.parse_args()

    tr = json.load(open(args.protocol))["training"]
    episodes, frames = 50, 500
    sampler = EpisodeAwareSampler([frames * e for e in range(episodes)], [frames * (e + 1) for e in range(episodes)],
                                  shuffle=True, seed=tr["seed"])
    loader = torch.utils.data.DataLoader(
        IndexDataset(episodes * frames), num_workers=4, batch_size=tr["batch_size"], shuffle=False, sampler=sampler,
        pin_memory=False, drop_last=False, prefetch_factor=4, persistent_workers=True, multiprocessing_context="spawn",
    )
    accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[DistributedDataParallelKwargs()], cpu=True)
    stream = cycle(accelerator.prepare(loader))
    real = np.concatenate([next(stream).numpy().copy() for _ in range(args.steps)]).astype(np.int64)
    rebuilt = np.load(args.order)[: real.size]
    same = np.array_equal(real, rebuilt)
    print(f"{real.size} draws ({real.size / (episodes * frames):.2f} epochs): rebuilt order {'equals' if same else 'DIFFERS FROM'} the real loader chain")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())

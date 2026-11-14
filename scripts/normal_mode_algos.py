#!/usr/bin/env python3
"""Normal-mode algorithm comparison between aloha-train and LeRobot v0.6.1's lerobot-train.

`run` launches one trainer in this process, unchanged, and from training step N on records for K
steps (a) every CUDA kernel it launches (torch.profiler; the kernel name carries the cuDNN or cuBLAS
algorithm and whether it is the TF32 or the FP32 variant), (b) every cuBLAS and cuBLASLt matmul call
through the libraries' own runtime log switches (shape, algorithm id, compute type) and (c) the
effective torch switches (TF32, benchmark, deterministic, matmul precision) in that process.
Nothing is recorded before step N, so cudnn.benchmark's timing-based choice in step 1 runs
unperturbed: cuDNN's own API log cannot be switched on at runtime on cuDNN 9.19 (cudnnSetCallback
does not reach the graph API, tried 2026-11-11), and formatting it inside the timed calls would
change the choice it is meant to observe. The trainer's numerics are untouched: the only hook counts
calls of torch.autograd.backward, which both trainers make once per step.

`compare` reads the records of several runs and reports, for every pair, whether the switches, the
conv-type kernel multiset, the matmul-type kernel multiset, the remaining kernels and the cuBLAS and
cuBLASLt call multisets are identical, with the differing entries.
Reading rule: docs/plan.md step 1b ("cuDNN/cuBLAS 算法比对").
"""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

CONV_RE = re.compile(r"conv|fprop|dgrad|wgrad|winograd|fft|implicit_gemm|nchwToNhwc|nhwcToNchw|cudnn", re.I)
GEMM_RE = re.compile(r"gemm|cutlass|splitK|gemv|cublas", re.I)
LT_RE = re.compile(r"\[Trace\]\[(cublasLt\w*Matmul)\] (.*)$")


def classify(name: str) -> str:
    if CONV_RE.search(name):
        return "conv"
    if GEMM_RE.search(name):
        return "gemm"
    return "other"


def loaded_library(fragment: str) -> ctypes.CDLL:
    for line in open("/proc/self/maps"):
        if fragment in line:
            return ctypes.CDLL(line.split()[-1])
    raise RuntimeError(f"{fragment} is not loaded in this process")


def switches() -> dict:
    import torch

    env_keys = ("NVIDIA_TF32_OVERRIDE", "CUDNN_LOGLEVEL_DBG", "CUDNN_LOGINFO_DBG", "TORCH_CUDNN_V8_API_DISABLED",
                "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE",
                "PYTORCH_CUDA_ALLOC_CONF", "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS", "PYTHONHASHSEED",
                "MUJOCO_GL", "HF_HUB_OFFLINE")
    m = torch.backends.cuda.matmul
    try:
        autocast = torch.is_autocast_enabled("cuda")
    except TypeError:
        autocast = torch.is_autocast_enabled()
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": m.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_fp16_reduced_precision_reduction": m.allow_fp16_reduced_precision_reduction,
        "matmul_allow_bf16_reduced_precision_reduction": m.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast_enabled_at_backward": autocast, "num_threads": torch.get_num_threads(),
        "env": {k: os.environ.get(k) for k in env_keys},
    }


def parse_cublas(path: Path) -> Counter:
    """Legacy cuBLAS API log -> multiset of gemm calls (function, layout, shape, types, algo, math mode)."""
    keep = ("transa", "transb", "m", "n", "k", "batchCount", "Atype", "Btype", "Ctype", "computeType", "algo",
            "strideA", "strideB", "strideC")
    calls: Counter = Counter()
    name, params, mathmode = None, {}, None

    def flush() -> None:
        if name and "gemm" in name.lower():
            key = {"fn": name, **{k: params.get(k) for k in keep}, "MathMode": mathmode}
            calls[json.dumps(key, sort_keys=True)] += 1

    if not path.exists():
        return calls
    for line in open(path, errors="replace"):
        if line.startswith("I! cuBLAS"):
            flush()
            found = re.search(r"function \S+ (\w+)\(", line)
            name, params, mathmode = (found.group(1) if found else None), {}, None
        elif line.startswith("i!  "):
            found = re.match(r"i!\s+(\w+): type=[^;]*; val=(.*)$", line.strip())
            if found:
                params[found.group(1)] = found.group(2).strip()
        elif "MathMode=" in line:
            found = re.search(r"MathMode=([A-Z_0-9]+)", line)
            mathmode = found.group(1) if found else None
    flush()
    return calls


def parse_cublaslt(path: Path) -> Counter:
    """cuBLASLt trace log -> multiset of matmul calls (layouts, compute descriptor, algorithm, workspace)."""
    calls: Counter = Counter()
    if not path.exists():
        return calls
    for line in open(path, errors="replace"):
        found = LT_RE.search(line)
        if not found:
            continue
        fn, rest = found.group(1), found.group(2)
        fields = {"fn": fn}
        for key in ("Adesc", "Bdesc", "Cdesc", "Ddesc", "computeDesc", "algo"):
            inner = re.search(key + r"=\[([^\]]*)\]", rest)
            if inner:
                fields[key] = re.sub(r"(Pointer=)0x[0-9a-fA-F]+", r"\1<ptr>", inner.group(1))
        size = re.search(r"workSpaceSizeInBytes=(\d+)", rest)
        fields["workSpaceSizeInBytes"] = size.group(1) if size else None
        calls[json.dumps(fields, sort_keys=True)] += 1
    return calls


def run(args: argparse.Namespace) -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile

    record = Path(args.record_dir)  # created at step N: lerobot-train refuses an existing output dir
    trainer_args = list(args.trainer_args)
    if trainer_args[:1] == ["--"]:
        trainer_args = trainer_args[1:]
    state = {"backward_calls": 0, "prof": None, "libs": None, "stopped": False}
    original_backward = torch.autograd.backward

    def start() -> None:
        record.mkdir(parents=True, exist_ok=True)
        (record / "switches.json").write_text(json.dumps(switches(), indent=2) + "\n")
        cublas, cublaslt = loaded_library("libcublas.so"), loaded_library("libcublasLt.so")
        cublas.cublasLoggerConfigure.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        cublaslt.cublasLtLoggerOpenFile.argtypes = [ctypes.c_char_p]
        codes = [cublas.cublasLoggerConfigure(1, 0, 0, str(record / "cublas.log").encode()),
                 cublaslt.cublasLtLoggerOpenFile(str(record / "cublaslt.log").encode()),
                 cublaslt.cublasLtLoggerSetLevel(ctypes.c_int(5))]
        if any(codes):
            print(f"normal_mode_algos: cuBLAS logger switches returned {codes}", file=sys.stderr)
        state["libs"] = (cublas, cublaslt)
        state["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        state["prof"].__enter__()
        print(f"normal_mode_algos: recording from backward call {state['backward_calls']} on", file=sys.stderr)

    def stop() -> None:
        state["stopped"] = True
        prof = state["prof"]
        prof.__exit__(None, None, None)
        cublas, cublaslt = state["libs"]
        cublas.cublasLoggerConfigure(0, 0, 0, None)
        cublaslt.cublasLtLoggerSetLevel(ctypes.c_int(0))
        kernels: Counter = Counter()
        for evt in prof.key_averages():
            if str(getattr(evt, "device_type", "")).endswith("CUDA"):
                kernels[evt.key] += evt.count
        (record / "kernels.json").write_text(json.dumps(dict(sorted(kernels.items())), indent=2) + "\n")
        try:
            table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=1000)
        except Exception:  # older column name
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=1000)
        (record / "table.txt").write_text(table + "\n")
        (record / "cublas.json").write_text(json.dumps(dict(sorted(parse_cublas(record / "cublas.log").items())), indent=2) + "\n")
        (record / "cublaslt.json").write_text(json.dumps(dict(sorted(parse_cublaslt(record / "cublaslt.log").items())), indent=2) + "\n")
        (record / "window.json").write_text(json.dumps({
            "start_after_backward": args.start_step, "stop_after_backward": args.start_step + args.steps,
            "recorded_steps": args.steps, "trainer": args.trainer, "trainer_args": trainer_args,
            "kernel_launches": sum(kernels.values()), "distinct_kernels": len(kernels)}, indent=2) + "\n")
        print(f"normal_mode_algos: recorded {sum(kernels.values())} kernel launches, "
              f"{len(kernels)} distinct kernels, to {record}", file=sys.stderr)

    def counting_backward(*a, **k):
        state["backward_calls"] += 1
        n = state["backward_calls"]
        try:
            return original_backward(*a, **k)
        finally:
            if n == args.start_step:
                start()
            elif n == args.start_step + args.steps and not state["stopped"]:
                stop()

    torch.autograd.backward = counting_backward
    sys.argv = [f"{args.trainer}-train", *trainer_args]
    if args.trainer == "aloha":
        from aloha_curriculum.cli import train_main

        rc = train_main(trainer_args)
    else:
        from lerobot.scripts.lerobot_train import main

        rc = main()
    if not state["stopped"]:
        print(f"normal_mode_algos: the trainer made only {state['backward_calls']} backward calls, "
              f"fewer than {args.start_step + args.steps}; nothing recorded", file=sys.stderr)
        return 1
    return int(rc or 0)


def load_record(record: Path) -> dict:
    kernels = json.loads((record / "kernels.json").read_text())
    by_class = {"conv": Counter(), "gemm": Counter(), "other": Counter()}
    for name, count in kernels.items():
        by_class[classify(name)][name] += count
    return {
        "switches": json.loads((record / "switches.json").read_text()),
        "window": json.loads((record / "window.json").read_text()),
        "conv": by_class["conv"], "gemm": by_class["gemm"], "other": by_class["other"],
        "cublas": Counter(json.loads((record / "cublas.json").read_text())),
        "cublaslt": Counter(json.loads((record / "cublaslt.json").read_text())),
    }


def multiset_diff(a: Counter, b: Counter) -> dict:
    return {key: [a.get(key, 0), b.get(key, 0)] for key in sorted(set(a) | set(b)) if a.get(key, 0) != b.get(key, 0)}


def compare(args: argparse.Namespace) -> int:
    runs = {}
    for path in args.record_dirs:
        path = Path(path)
        label = path.parent.name if path.name == "record" else path.name
        runs[label] = load_record(path)
    categories = ("conv", "gemm", "other", "cublas", "cublaslt")
    report = {"runs": {}, "pairs": {}}
    for label, rec in runs.items():
        report["runs"][label] = {
            "trainer": rec["window"]["trainer"], "recorded_steps": rec["window"]["recorded_steps"],
            "kernel_launches": rec["window"]["kernel_launches"], "distinct_kernels": rec["window"]["distinct_kernels"],
            **{f"{c}_launches": sum(rec[c].values()) for c in categories},
            **{f"{c}_distinct": len(rec[c]) for c in categories},
            "tf32_kernels": sum(v for k, v in itertools.chain(rec["conv"].items(), rec["gemm"].items()) if "tf32" in k.lower()),
            "switches": {k: v for k, v in rec["switches"].items() if k != "env"},
        }
    for x, y in itertools.combinations(runs, 2):
        a, b = runs[x], runs[y]
        sw_diff = {k: [a["switches"].get(k), b["switches"].get(k)] for k in sorted(set(a["switches"]) | set(b["switches"]))
                   if k != "env" and a["switches"].get(k) != b["switches"].get(k)}
        pair = {"switches_identical": not sw_diff, "switches_diff": sw_diff}
        for c in categories:
            diff = multiset_diff(a[c], b[c])
            pair[f"{c}_identical"] = not diff
            pair[f"{c}_diff"] = diff
        pair["same_trainer"] = a["window"]["trainer"] == b["window"]["trainer"]
        report["pairs"][f"{x} vs {y}"] = pair
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    for label, info in report["runs"].items():
        sw = info["switches"]
        print(f"{label:12s} {info['trainer']:8s} launches {info['kernel_launches']:6d} distinct {info['distinct_kernels']:4d} "
              f"conv {info['conv_distinct']:3d} gemm {info['gemm_distinct']:3d} other {info['other_distinct']:3d} "
              f"cublas {info['cublas_distinct']:3d} cublasLt {info['cublaslt_distinct']:3d} tf32 launches {info['tf32_kernels']:5d} | "
              f"benchmark {sw['cudnn_benchmark']} det {sw['cudnn_deterministic']} cudnn_tf32 {sw['cudnn_allow_tf32']} "
            f"matmul_tf32 {sw['matmul_allow_tf32']} precision {sw['float32_matmul_precision']}")
    print(f"{'pair':28s} {'switches':>8s} {'conv':>6s} {'gemm':>6s} {'other':>6s} {'cublas':>7s} {'cublasLt':>8s}")
    for name, pair in report["pairs"].items():
        print(f"{name:28s} {str(pair['switches_identical']):>8s} {str(pair['conv_identical']):>6s} "
              f"{str(pair['gemm_identical']):>6s} {str(pair['other_identical']):>6s} "
              f"{str(pair['cublas_identical']):>7s} {str(pair['cublaslt_identical']):>8s}")
        for c in ("switches",) + categories:
            diff = pair[f"{c}_diff"]
            if diff:
                shown = list(diff.items())[:args.show]
                print(f"  {c} differs in {len(diff)} entries; first {len(shown)}:")
                for key, (ka, kb) in shown:
                    print(f"    {ka!s:>6} | {kb!s:>6}  {key[:200]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run one trainer with the recording hook")
    p_run.add_argument("--trainer", choices=("aloha", "lerobot"), required=True)
    p_run.add_argument("--record-dir", required=True)
    p_run.add_argument("--start-step", type=int, default=200, help="record after this many backward calls")
    p_run.add_argument("--steps", type=int, default=2, help="number of complete steps to record")
    p_run.add_argument("trainer_args", nargs=argparse.REMAINDER, help="the trainer's own arguments after --")
    p_cmp = sub.add_parser("compare", help="compare the records of several runs")
    p_cmp.add_argument("record_dirs", nargs="+")
    p_cmp.add_argument("--output")
    p_cmp.add_argument("--show", type=int, default=12, help="differing entries to print per category")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())

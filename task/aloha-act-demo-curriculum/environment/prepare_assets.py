#!/usr/bin/env python3
"""Build-time asset preparation (network allowed only here).

1. Download the pinned dataset and reference checkpoint revisions from the Hub.
2. Verify every downloaded file against assets.sha256.
3. Download the torchvision ResNet18 ImageNet weights into TORCH_HOME and verify them.
4. Build the public per-frame feature table (parquet + csv).
5. Migrate the reference checkpoint (pre-#1452 normalization layers) into the current
   LeRobot processor format so agents can load it offline with lerobot v0.6.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_listed(assets_dir: Path, listing: Path) -> int:
    expected = {}
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        expected[rel.strip()] = digest
    bad = []
    for rel, digest in expected.items():
        p = assets_dir / rel
        if not p.is_file():
            bad.append(f"missing {rel}")
        elif sha256_file(p) != digest:
            bad.append(f"hash mismatch {rel}")
    if bad:
        raise SystemExit("asset verification failed: " + "; ".join(bad))
    return len(expected)


def download_hub(protocol: dict, reference: dict, assets_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    ds = protocol["dataset"]
    snapshot_download(
        ds["repo_id"], repo_type="dataset", revision=ds["revision"],
        local_dir=assets_dir / "aloha_sim_insertion_human",
    )
    snapshot_download(
        reference["repo_id"], repo_type="model", revision=reference["revision"],
        local_dir=assets_dir / "act_aloha_sim_insertion_human",
    )
    for sub in ("aloha_sim_insertion_human", "act_aloha_sim_insertion_human"):
        cache = assets_dir / sub / ".cache"
        if cache.exists():
            shutil.rmtree(cache)


def download_backbone(protocol: dict, torch_home: Path) -> Path:
    bw = protocol["backbone_weights"]
    dest = torch_home / "hub" / "checkpoints" / bw["file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        with urllib.request.urlopen(bw["url"], timeout=120) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
    digest = sha256_file(dest)
    if digest != bw["sha256"]:
        raise SystemExit(f"backbone weights hash mismatch: {digest}")
    return dest


def build_features(dataset_root: Path, public_dir: Path, chunk_size: int) -> dict:
    from aloha_curriculum import features as features_mod

    df = features_mod.build_features(dataset_root, chunk_size)
    digest = features_mod.write_features(df, public_dir / "features.parquet")
    df.to_csv(public_dir / "features.csv", index=False, float_format="%.9g")
    schema = {"rows": int(len(df)), "columns": list(df.columns), "parquet_sha256": digest}
    (public_dir / "features_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return schema


def migrate_reference(src: Path, dst: Path) -> None:
    """Same conversion as lerobot/processor/migrate_policy_normalization.py, minus the model card."""
    import torch
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
    from lerobot.processor.migrate_policy_normalization import (
        clean_state_dict,
        convert_features_to_policy_features,
        detect_features_and_norm_modes,
        extract_normalization_stats,
        load_state_dict_with_missing_key_handling,
        remove_normalization_layers,
    )
    from safetensors.torch import load_file

    state_dict = load_file(str(src / "model.safetensors"))
    config = json.loads((src / "config.json").read_text())
    stats = extract_normalization_stats(state_dict)
    features, norm_map = detect_features_and_norm_modes(config, stats)
    new_state_dict = clean_state_dict(remove_normalization_layers(state_dict), remove_str="._orig_mod")
    policy_type = config["type"]
    cleaned = {k: v for k, v in config.items() if k not in ("normalization_mapping", "type")}
    for key in ("input_features", "output_features"):
        if key in cleaned:
            cleaned[key] = convert_features_to_policy_features(cleaned[key])
    cleaned["normalization_mapping"] = norm_map
    cleaned["device"] = "cpu"
    policy_config = make_policy_config(policy_type, **cleaned)
    policy = get_policy_class(policy_type)(policy_config)
    load_state_dict_with_missing_key_handling(policy=policy, state_dict=new_state_dict, policy_type=policy_type, known_missing_keys_whitelist={})
    policy.to(torch.float32)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_config, dataset_stats=stats)
    dst.mkdir(parents=True, exist_ok=True)
    preprocessor.save_pretrained(dst)
    postprocessor.save_pretrained(dst)
    policy.save_pretrained(dst)
    for name in ("README.md", "train_config.json", "eval_info.json"):
        if (src / name).is_file():
            shutil.copyfile(src / name, dst / name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--versions", required=True, help="pins.json with the reference checkpoint revision")
    p.add_argument("--assets-sha256", required=True)
    p.add_argument("--assets-dir", required=True)
    p.add_argument("--public-dir", required=True)
    p.add_argument("--torch-home", required=True)
    p.add_argument("--skip-download", action="store_true", help="assets already present (local builds)")
    a = p.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    versions = json.loads(Path(a.versions).read_text())
    assets_dir, public_dir, torch_home = Path(a.assets_dir), Path(a.public_dir), Path(a.torch_home)
    assets_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    if not a.skip_download:
        download_hub(protocol, versions["reference_checkpoint"], assets_dir)
    n = verify_listed(assets_dir, Path(a.assets_sha256))
    print(f"verified {n} hub files")
    backbone = download_backbone(protocol, torch_home)
    print(f"backbone weights ok: {backbone}")
    schema = build_features(assets_dir / "aloha_sim_insertion_human", public_dir, int(protocol["policy"]["chunk_size"]))
    print(f"features: {schema['rows']} rows x {len(schema['columns'])} columns, sha256 {schema['parquet_sha256']}")
    migrate_reference(assets_dir / "act_aloha_sim_insertion_human", assets_dir / "reference_policy")
    print("reference checkpoint migrated to", assets_dir / "reference_policy")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    raise SystemExit(main())

#!/bin/bash
# Oracle solution: materialise the uniform-sampling baseline candidate.
# The Judge then retrains ACT with the fixed recipe and scores it on the hidden seeds.
set -euo pipefail
cd /workspace
mkdir -p /workspace/curriculum /workspace/research
rm -f /workspace/curriculum/scores.csv
install -m 0644 /opt/aloha_public/default_policy.json /workspace/curriculum/policy.json
aloha-validate --candidate-dir /workspace/curriculum --features /opt/aloha_public/features.parquet --protocol /opt/aloha_public/fixed_train.json > /workspace/research/baseline_validation.json

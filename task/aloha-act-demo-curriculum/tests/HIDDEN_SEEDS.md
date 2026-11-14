# tests/hidden_seeds.json (withheld)

The Judge (`tests/evaluate.py`) reads `/tests/hidden_seeds.json`. The file is withheld from the
public copy of this task. It names a block of consecutive evaluation seeds:

```json
{"start_seed": 1234567890, "n_episodes": 500, "n_envs": 10}
```

The value above is only an example. Episode `i` of the hidden evaluation uses seed `start_seed + i`.
The public proxy block that agents may use is seeds 100000-100099. To run the Judge, create this
file with a start seed of your own outside the proxy block. Your results then refer to your block,
not to the block behind the results in `experiments/`.

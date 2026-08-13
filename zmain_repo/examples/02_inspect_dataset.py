"""Read a canonical offline dataset without invoking a trainer."""

from pathlib import Path

import numpy as np


dataset_root = Path("artifacts/datasets/epicare/smart/dev")
metadata = dataset_root / "metadata.json"
train = dataset_root / "train.npz"

if not train.exists():
    raise SystemExit("Build it first: python scripts/bench.py dataset build epicare/smart/dev")

data = np.load(train)
print("arrays:", sorted(data.files))
print("transitions:", int(data["actions"].shape[0]))
print("episodes:", int(np.unique(data["episode_ids"]).size))
print("state-vector shape:", tuple(data["state_vectors"].shape))
print("behavior log-probabilities finite:", bool(np.isfinite(data["behavior_logprobs"]).all()))
print("metadata:", metadata)

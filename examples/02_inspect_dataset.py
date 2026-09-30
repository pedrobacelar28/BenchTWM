"""Read a built dataset without invoking a trainer."""

import numpy as np

from benchtwm.data.dataset import OfflineTransitionDataset
from benchtwm.paths import DEFAULT_ARTIFACT_ROOT

path = DEFAULT_ARTIFACT_ROOT / "datasets" / "epicare" / "smart" / "dev" / "train.npz"
if not path.exists():
    raise SystemExit("Build it first: bench dataset build epicare/smart/dev")

dataset = OfflineTransitionDataset(path)
print("arrays:", sorted(dataset.data))
print("transitions:", len(dataset), "| episodes:", dataset.num_episodes)
print("features:", dataset.feature_names)
print("actions:", dataset.num_actions)
print("behavior log-propensities finite:", bool(np.isfinite(dataset.data["behavior_logprobs"]).all()))

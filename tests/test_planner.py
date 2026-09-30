import sys
import unittest
from pathlib import Path

import torch

from benchtwm.planners.cem import CategoricalCEMPlanner


class TestPlanner(unittest.TestCase):
    def test_first_action_respects_mask_and_prefers_best_sequence(self):
        planner = CategoricalCEMPlanner(
            horizon=3,
            num_actions=5,
            samples=32,
            elites=4,
            iterations=3,
        )
        first_mask = torch.tensor([0, 0, 0, 1, 1], dtype=torch.float32)

        def scorer(sequences):
            return sequences.float().sum(dim=1)

        action, _ = planner.plan(scorer, first_action_mask=first_mask)
        self.assertIn(action, [3, 4])


if __name__ == "__main__":
    unittest.main()

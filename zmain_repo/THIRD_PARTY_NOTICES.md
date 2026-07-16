# Third-Party Notices

This artifact vendors the ICU-Sepsis benchmark packages under
`third_party/icu-sepsis/` so the experiments can run without requiring a
separate checkout.

The vendored benchmark code keeps its upstream license files:

- `third_party/icu-sepsis/LICENSE`
- `third_party/icu-sepsis/packages/icu_sepsis/LICENSE.txt`
- `third_party/icu-sepsis/packages/icu_sepsis_helpers/LICENSE.txt`

Generated experiment artifacts such as checkpoints, datasets, Optuna databases,
logs, and plots are intentionally excluded from this public source package.

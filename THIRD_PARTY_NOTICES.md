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

## DreamerV3

The PyTorch implementation in `src/models/dreamer.py` is a framework port of
the vector-observation, discrete-action path of Danijar Hafner's DreamerV3
repository:

- source: `https://github.com/danijar/dreamerv3`
- pinned source commit: `e3f02248693a79dc8b0ebd62c93683888ddaccfe`
- local audit checkout: `repos_usados/dreamerv3/`
- license: MIT

Copyright (c) 2024 Danijar Hafner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

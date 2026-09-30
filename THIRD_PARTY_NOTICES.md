# Third-Party Notices

BenchTWM vendors the runtime cores of two benchmarks under `third_party/` so
experiments run without separate checkouts. Each keeps its upstream license
and a `NOTICE` describing local modifications:

- ICU-Sepsis (MIT): `third_party/icu-sepsis/LICENSE`,
  `third_party/icu-sepsis/packages/icu_sepsis/LICENSE.txt`, and
  `third_party/icu-sepsis/NOTICE`.
- EpiCare (Apache-2.0): `third_party/epicare/LICENSE` and
  `third_party/epicare/NOTICE`.

Generated artifacts (datasets, checkpoints, Optuna databases, logs, reports)
are excluded from the source tree.

## DreamerV3

The PyTorch implementation in `benchtwm/algorithms/dreamer/agent.py` is a
framework port of the vector-observation, discrete-action path of Danijar
Hafner's DreamerV3 repository:

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

"""BenchTWM command-line interface.

Offline workflow::

    bench doctor -> bench catalog -> bench dataset build <dataset>
    -> bench train <dataset> <method> -> bench status -> bench report <run>

Every command resolves its configuration from ``configs/`` and writes only
under the artifact root (``artifacts/`` by default).
"""

from __future__ import annotations

import argparse
import sys

from benchtwm.cli import common, data, info, research, runs
from benchtwm.config import ConfigError
from benchtwm.paths import CONFIG_ROOT, DEFAULT_ARTIFACT_ROOT

GLOBAL_VALUE_FLAGS = {"--artifact-root", "--config-root", "--color"}


def _hoist_global_flags(argv: list[str]) -> list[str]:
    """Allow global flags before or after the subcommand."""

    leading, remaining, index = [], [], 0
    while index < len(argv):
        token = argv[index]
        if token == "--json":
            leading.append(token)
        elif token in GLOBAL_VALUE_FLAGS and index + 1 < len(argv):
            leading += [token, argv[index + 1]]
            index += 1
        else:
            remaining.append(token)
        index += 1
    return [*leading, *remaining]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--color", choices=["auto", "never"], default="auto")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module in (info, data, runs, research):
        module.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_hoist_global_flags(list(sys.argv[1:] if argv is None else argv)))
    try:
        return int(args.handler(args))
    except (ConfigError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        common.console(args).error(str(exc).strip("'\""))
        return 2

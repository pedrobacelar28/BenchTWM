"""Convenience entry point: ``python scripts/bench.py`` is the same as ``bench``."""

from benchtwm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

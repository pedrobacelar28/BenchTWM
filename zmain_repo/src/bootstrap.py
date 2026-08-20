"""Bootstrap local package imports for the ICU-Sepsis benchmark packages."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ICU_PACKAGES_ROOT = PROJECT_ROOT / "third_party" / "icu-sepsis" / "packages"
SIBLING_ICU_PACKAGES_ROOT = PROJECT_ROOT.parent / "icu-sepsis" / "packages"
VENDORED_EPICARE_ROOT = PROJECT_ROOT / "third_party" / "epicare"
VENDORED_DTR_ROOT = PROJECT_ROOT / "third_party" / "dtr"


def _package_root(package_name: str) -> Path:
    for packages_root in (VENDORED_ICU_PACKAGES_ROOT, SIBLING_ICU_PACKAGES_ROOT):
        package_root = packages_root / package_name
        if package_root.exists():
            return package_root
    return VENDORED_ICU_PACKAGES_ROOT / package_name


def _prepend_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def bootstrap_icu_sepsis() -> None:
    _prepend_path(_package_root("icu_sepsis"))


def bootstrap_icu_helpers() -> None:
    bootstrap_icu_sepsis()
    _prepend_path(_package_root("icu_sepsis_helpers"))


def bootstrap_epicare() -> None:
    _prepend_path(VENDORED_EPICARE_ROOT)


def bootstrap_dtr() -> None:
    """Expose only the selective, vendored DTR runtime."""

    _prepend_path(VENDORED_DTR_ROOT)

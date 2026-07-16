"""JSON sweep helpers."""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

RUN_FIXED_LIST_KEYS = {"robustness_sigmas"}
AGG_FIXED_LIST_KEYS = {"seed", "robustness_sigmas"}


def get_sorted_dict(json_name: str | Path) -> collections.OrderedDict:
    with open(json_name, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return collections.OrderedDict(sorted(data.items()))


def get_param_iterable(config: dict) -> list[dict]:
    list_keys: list[str] = []
    list_values: list[list] = []
    fixed_keys: list[str] = []
    for key, value in config.items():
        if isinstance(value, list) and key not in RUN_FIXED_LIST_KEYS:
            list_keys.append(key)
            list_values.append(value)
        else:
            fixed_keys.append(key)

    all_parameters: list[dict] = []
    for combo in itertools.product(*list_values):
        record = {}
        for idx, key in enumerate(list_keys):
            record[key] = combo[idx]
        for key in fixed_keys:
            record[key] = config[key]
        all_parameters.append(record)
    return all_parameters


def get_param_iterable_runs(config: dict) -> list[dict]:
    list_keys: list[str] = []
    list_values: list[list] = []
    fixed_keys: list[str] = []
    for key, value in config.items():
        if isinstance(value, list) and key not in AGG_FIXED_LIST_KEYS:
            list_keys.append(key)
            list_values.append(value)
        else:
            fixed_keys.append(key)

    all_parameters: list[dict] = []
    for combo in itertools.product(*list_values):
        record = {}
        for idx, key in enumerate(list_keys):
            record[key] = combo[idx]
        for key in fixed_keys:
            record[key] = config[key]
        all_parameters.append(record)
    return all_parameters

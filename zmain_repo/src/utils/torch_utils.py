"""Torch device helpers."""

from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_device_string(device: torch.device | str) -> str:
    return str(device)

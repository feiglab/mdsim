#!/usr/bin/env python3
from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_config(path: Path) -> dict[str, str]:
    """
    Read a simple key/value config file.

    Format:
        key value
        key = value

    Blank lines and lines starting with '#' are ignored. Keys are lowercased.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    cfg: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
        else:
            parts = line.split(None, 1)
            key = parts[0].strip()
            val = parts[1].strip() if len(parts) == 2 else ""

        if key:
            cfg[key.lower()] = val

    return cfg


def parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid bool {s!r}")


def split_values(s: str) -> list[str]:
    """
    Tokenize a config value into a list (e.g. for nargs='+').

    Uses shell-like splitting, so quoting is supported.
    """
    return shlex.split(s)


def format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return " ".join(format_value(x) for x in v)
    return str(v)


_ORDER = [
    "mode",
    "refsel",
    "othersel",
    "anchor",
    "rot",
    "flip",
    "bias",
    "biasdir",
    "k",
    "setup",
    "equi",
    "pdb_in",
    "refpdb",
    "capdb",
    "cadcd",
    "box",
    "conc",
    "surf",
    "orient",
    "ff",
]


def write_config(path: Path, data: Mapping[str, str]) -> None:
    path = Path(path)
    keys = list(data.keys())

    ordered: list[str] = []
    seen = set()

    for k in _ORDER:
        if k in data:
            ordered.append(k)
            seen.add(k)

    for k in sorted(keys):
        if k not in seen:
            ordered.append(k)

    lines = []
    for k in ordered:
        v = data.get(k, "")
        if v == "":
            continue
        lines.append(f"{k} {v}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

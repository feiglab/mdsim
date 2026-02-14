#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openmm.unit import nanometer

from mdsim import (
    MDSim,
    PDBReader,
)
from mdsim.mdsim_config import format_value, parse_bool, read_config, split_values, write_config


def _as_float(name: str, s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be a float, got {s!r}") from e


def _find(tdir: Path, filename: str) -> Path:
    tdir = Path(tdir).expanduser().resolve()

    # try relative to tdir then parents
    for d in (tdir, *tdir.parents):
        candidate = d / filename
        if candidate.is_file():
            return candidate.resolve()

    # try CWD
    candidate = Path.cwd() / filename
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find '{filename}' in {tdir} or its parent directories or CWD"
    )


def _parse_config_args(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="Config file (key/value) to read/write",
    )
    ns, _ = p.parse_known_args(argv)
    return Path(ns.config)


@dataclass(frozen=True)
class BoxNM:
    x: float
    y: float
    z: float

    def as_units(self) -> tuple:
        return (self.x * nanometer, self.y * nanometer, self.z * nanometer)


def _parse_box_nm(s: str) -> BoxNM:
    parts = [p.strip() for p in s.split(":") if p.strip() != ""]
    if len(parts) == 1:
        x = _as_float("box", parts[0])
        return BoxNM(x, x, x)
    if len(parts) == 2:
        x = _as_float("boxx", parts[0])
        y = _as_float("boxy", parts[1])
        return BoxNM(x, y, y)
    if len(parts) == 3:
        x = _as_float("boxx", parts[0])
        y = _as_float("boxy", parts[1])
        z = _as_float("boxz", parts[2])
        return BoxNM(x, y, z)
    raise SystemExit("ERROR: --box must be 'x', 'x:y', or 'x:y:z' in nm (e.g. 22:11:9)")


def _expand_forcefields(paths: Sequence[str]) -> list[str]:
    return [str(Path(p).expanduser().resolve()) for p in paths]


def _default_forcefields() -> list[str]:
    ffdir = Path.home() / "ff"
    return _expand_forcefields([str(ffdir / "par_all36m_prot.prm"), str(ffdir / "waters_ions.prm")])


def _validate_forcefields(ff: Sequence[str]) -> None:
    missing = [p for p in ff if not Path(p).is_file()]
    if missing:
        msg = "ERROR: forcefield file(s) not found:\n  " + "\n  ".join(missing)
        raise SystemExit(msg)


def _apply_config_defaults(
    p: argparse.ArgumentParser,
    cfg: dict[str, str],
) -> None:
    defaults: dict[str, object] = {}

    if "setup" in cfg:
        defaults["setup"] = Path(cfg["setup"])
    if "pdb_in" in cfg:
        defaults["pdb"] = cfg["pdb_in"]
    if "psf_in" in cfg:
        defaults["psf"] = cfg["psf_in"]
    if "box" in cfg:
        defaults["box"] = cfg["box"]
    if "hmass" in cfg:
        defaults["hmass"] = parse_bool(cfg["hmass"])
    if "ff" in cfg:
        defaults["ff"] = split_values(cfg["ff"])

    if defaults:
        p.set_defaults(**defaults)


def _parse_args(
    cfg: dict[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prep_charmmsystem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--setup",
        type=Path,
        default=Path("setup"),
        help="Output/setup directory",
    )
    p.add_argument(
        "--pdb",
        type=str,
        default=None,
        help="Input PDB file (searched relative to --setup and parents, then CWD)",
    )
    p.add_argument(
        "--psf",
        type=str,
        default=None,
        help="Input PSF file (searched relative to --setup and parents, then CWD)",
    )
    p.add_argument(
        "--box",
        type=str,
        default=None,
        help="Box size in nm: x, x:y, or x:y:z (e.g. 22:11:9).",
    )

    hmass_grp = p.add_mutually_exclusive_group()
    hmass_grp.add_argument(
        "--hmass",
        dest="hmass",
        action="store_true",
        help="Hydrogen mass repartioning",
    )
    hmass_grp.add_argument(
        "--no-hmass",
        dest="hmass",
        action="store_false",
        help="No hydrogen mass repartioning",
    )
    p.set_defaults(hmass=False)

    p.add_argument(
        "--ff",
        nargs="+",
        default=None,
        help="CHARMM parameter files.",
    )

    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="Config file (key/value) to read/write",
    )
    p.add_argument(
        "--no-write-config",
        dest="write_config",
        action="store_false",
        help="Disable writing updated config values",
    )
    p.set_defaults(write_config=True)

    _apply_config_defaults(p, cfg)
    return p.parse_args(argv)


def main() -> None:
    cfg_path = _parse_config_args()
    cfg = read_config(cfg_path)

    args = _parse_args(cfg)

    tdir = Path(args.setup).expanduser().resolve()
    tdir.mkdir(parents=True, exist_ok=True)

    if args.pdb is None:
        pdb_arg = None
    else:
        pdb_arg = str(args.pdb)

    if args.psf is None:
        psf_arg = None
    else:
        psf_arg = str(args.psf)

    if args.box is None:
        box_str = "10"
    else:
        box_str = str(args.box)

    box_nm = _parse_box_nm(box_str)
    boxx, boxy, boxz = box_nm.as_units()

    ff_val: list[str] | None = None
    ff_val = _default_forcefields() if args.ff is None else _expand_forcefields(args.ff)
    _validate_forcefields(ff_val)

    if bool(args.write_config):
        cfg["setup"] = format_value(args.setup)
        if pdb_arg is not None:
            cfg["pdb_in"] = format_value(pdb_arg)
        if psf_arg is not None:
            cfg["psf_in"] = format_value(psf_arg)
        cfg["box"] = format_value(box_str)
        cfg["hmass"] = format_value(bool(args.hmass))
        if ff_val is not None:
            cfg["ff"] = format_value(ff_val)
        write_config(cfg_path, cfg)

    pdb_path = _find(tdir, pdb_arg)
    s = PDBReader(str(pdb_path))
    fullsystem = s[0]
    fullsystem.write_pdb(str(tdir / "solvated.pdb"))

    psf_path = _find(tdir, psf_arg)

    sim = MDSim(
        model=fullsystem,
        psf=psf_path,
        par=ff_val,
        box=(boxx, boxy, boxz),
        hmass=bool(args.hmass),
        switching="openmm",
    )

    sim.setup_simulation()
    print(f"openmm energy: {sim.get_potentialEnergy()}")

    sim.write_system(str(tdir / "system.xml"))
    sim.write_state(str(tdir / "initial.xml"))


if __name__ == "__main__":
    main()

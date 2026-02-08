#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openmm import Vec3
from openmm.unit import nanometer

from mdsim import (
    MDSim,
    PDBReader,
    StructureSelector,
    ion_counts,
    solvate,
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
    ffdir = Path.home() / "ff" / "openmm"
    return _expand_forcefields([str(ffdir / "c36m.xml"), str(ffdir / "waters_ions_default.xml")])


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
    if "refsel" in cfg:
        defaults["refsel"] = cfg["refsel"]
    if "box" in cfg:
        defaults["box"] = cfg["box"]
    if "conc" in cfg:
        defaults["conc"] = float(cfg["conc"])
    if "orient" in cfg:
        defaults["orient"] = parse_bool(cfg["orient"])
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
        prog="prep_waterbox",
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
        "--refsel",
        type=str,
        default=None,
        help="Reference selection (mdsim StructureSelector syntax)",
    )
    p.add_argument(
        "--box",
        type=str,
        default=None,
        help="Box size in nm: x, x:y, or x:y:z (e.g. 22:11:9).",
    )

    p.add_argument(
        "--conc",
        type=float,
        default=100.0,
        help="NaCl concentration in mM (allatom only)",
    )

    orient_grp = p.add_mutually_exclusive_group()
    orient_grp.add_argument(
        "--orient",
        dest="orient",
        action="store_true",
        help="Orient using refsel plane and rotate othersel into x-axis",
    )
    orient_grp.add_argument(
        "--no-orient",
        dest="orient",
        action="store_false",
        help="Disable orientation step",
    )
    p.set_defaults(orient=True)

    hmass_grp = p.add_mutually_exclusive_group()
    hmass_grp.add_argument(
        "--hmass",
        dest="hmass",
        action="store_true",
        help="Hydrogen mass repartioning",
    )
    orient_grp.add_argument(
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
        help="OpenMM forcefield XMLs (allatom only).",
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

    if args.box is None:
        box_str = "10"
    else:
        box_str = str(args.box)

    box_nm = _parse_box_nm(box_str)
    boxx, boxy, boxz = box_nm.as_units()

    ff_val: list[str] | None = None
    ff_val = _default_forcefields() if args.ff is None else _expand_forcefields(args.ff)
    _validate_forcefields(ff_val)

    cfg_path = Path(args.config)
    if bool(args.write_config):
        cfg["setup"] = format_value(args.setup)
        if pdb_arg is not None:
            cfg["pdb_in"] = format_value(pdb_arg)
        if args.refsel is not None:
            cfg["refsel"] = format_value(args.refsel)
        cfg["box"] = format_value(box_str)
        cfg["conc"] = format_value(args.conc)
        cfg["orient"] = format_value(bool(args.orient))
        cfg["hmass"] = format_value(bool(args.hmass))
        if ff_val is not None:
            cfg["ff"] = format_value(ff_val)
        write_config(cfg_path, cfg)

    if pdb_arg is not None:
        pdb_path = _find(tdir, pdb_arg)
        s = PDBReader(str(pdb_path))
        solute = s[0]
        nominal_charge = float(solute.nominal_charge())

        if args.refsel is not None:
            refsel = str(args.refsel)
            sel_idx = StructureSelector(refsel).atom_indices(s)
        else:
            # SELECT ALL ATOMS
            sel_idx = list(range(len(solute.atoms)))

        ch = s.center(sel_idx)[0]

        if bool(args.orient):
            # Translate so the selection center is at the box center.
            # boxx/boxy/boxz are already Quantities in nanometer.
            box_center = Vec3(0.5 * boxx, 0.5 * boxy, 0.5 * boxz)
            translate = box_center - ch
            s.translate(translate)

            t_nm = [float(x.value_in_unit(nanometer)) for x in translate]
            print("translated center by " f"({t_nm[0]:.3f}, {t_nm[1]:.3f}, {t_nm[2]:.3f}) nm")

        s.write_pdb(str(tdir / "solute.pdb"))
    else:
        solute = None
        nominal_charge = 0.0

    conc = float(args.conc)
    _, nsod, ncla = ion_counts(boxx, boxy, boxz, conc, nominal_charge)

    solvated, wbox = solvate(
        solute,
        box_min=(0.0, 0.0, 0.0),
        box_max=(boxx, boxy, boxz),
        ions={"SOD": nsod, "CLA": ncla},
    )
    print(f"solvated system with {nsod} Na and {ncla} Cl ions")
    print(f"box size: {wbox} nm^3")

    solvated.write_pdb(str(tdir / "solvated.pdb"))
    sim = MDSim(
        model=solvated,
        ff=ff_val,
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

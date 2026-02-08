#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mdsim import MDSim
from mdsim.mdsim_config import format_value, parse_bool, read_config, write_config


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


def _apply_config_defaults(
    p: argparse.ArgumentParser,
    cfg: dict[str, str],
) -> None:
    defaults: dict[str, object] = {}

    if "setup" in cfg:
        defaults["setup"] = Path(cfg["setup"])
    if "equi" in cfg:
        defaults["equi"] = Path(cfg["equi"])
    if "rundir" in cfg:
        defaults["rundir"] = Path(cfg["rundir"])
    if "npt" in cfg:
        defaults["npt"] = parse_bool(cfg["npt"])
    if "temperature" in cfg:
        defaults["temperature"] = float(cfg["temperature"])
    if "pressure" in cfg:
        defaults["pressure"] = float(cfg["pressure"])
    if "tstep" in cfg:
        defaults["tstep"] = float(cfg["tstep"])
    if "gamma" in cfg:
        defaults["gamma"] = float(cfg["gamma"])

    if defaults:
        p.set_defaults(**defaults)


def _parse_args(
    cfg: dict[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="waterbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--run",
        dest="nrun",
        type=int,
        default=1,
        help="Production run index to write (expects restart from run-1)",
    )
    p.add_argument(
        "--nstep",
        type=int,
        default=100000,
        help="Number of MD steps",
    )
    p.add_argument(
        "--tstep",
        type=float,
        default=0.002,
        help="Timestep",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Langevin friction (1/ps)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Temperature (K)",
    )

    npt_grp = p.add_mutually_exclusive_group()
    npt_grp.add_argument(
        "--npt",
        dest="npt",
        action="store_true",
        help="NPT simulation",
    )
    npt_grp.add_argument(
        "--no-npt",
        dest="npt",
        action="store_false",
        help="Disable NPT",
    )
    p.set_defaults(npt=True)

    p.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Pressure (bar)",
    )
    p.add_argument(
        "--nout",
        type=int,
        default=10000,
        help="Output/report interval (steps)",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="OpenMM resource device index",
    )
    p.add_argument(
        "--resources",
        type=str,
        default="CUDA",
        help="OpenMM platform/resources string",
    )
    p.add_argument(
        "--dir",
        dest="rundir",
        type=Path,
        default=".",
        help="Run directory",
    )
    p.add_argument(
        "--equi",
        dest="equi",
        type=Path,
        default="equi",
        help="Equilibration directory",
    )
    p.add_argument(
        "--setup",
        dest="setup",
        type=Path,
        default="setup",
        help="Setup directory",
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

    rdir = Path(args.rundir).expanduser().resolve()
    edir = Path(args.equi).expanduser().resolve()
    sdir = Path(args.setup).expanduser().resolve()

    if args.nrun < 1:
        raise SystemExit("ERROR: --run must be >= 1")

    if bool(args.write_config):
        cfg["rundir"] = format_value(args.rundir)
        cfg["npt"] = format_value(bool(args.npt))
        cfg["temperature"] = format_value(args.temperature)
        cfg["pressure"] = format_value(args.pressure)
        cfg["gamma"] = format_value(args.gamma)
        cfg["tstep"] = format_value(args.tstep)
        write_config(cfg_path, cfg)

    rdir.mkdir(parents=True, exist_ok=True)

    last = args.nrun - 1
    if last == 0:
        restart = edir / "equi_final.xml"
    else:
        restart = rdir / f"prod_{last}.xml"

    if not restart.is_file():
        raise SystemExit(f"ERROR: restart file does not exist: {restart}")

    sysxml = sdir / "system.xml"
    if not sysxml.is_file():
        raise SystemExit(f"ERROR: system xml does not exist: {sysxml}")

    sim = MDSim(xml=str(sysxml), restart=str(restart))

    if bool(args.npt):
        sim.set_barostat(pressure=float(args.pressure), temperature=float(args.temperature))

    sim.setup_simulation(
        resources=str(args.resources),
        device=int(args.device),
        temperature=float(args.temperature),
        tstep=float(args.tstep),
        gamma=float(args.gamma),
    )

    nrun = int(args.nrun)
    sim.simulate(
        nstep=int(args.nstep),
        nout=int(args.nout),
        logfile=str(rdir / f"prod_{nrun}.log"),
        dcdfile=str(rdir / f"prod_{nrun}.dcd"),
    )
    sim.write_state(str(rdir / f"prod_{nrun}.xml"))


if __name__ == "__main__":
    main()

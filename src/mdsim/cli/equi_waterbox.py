#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mdsim import MDSim
from mdsim.mdsim_config import format_value, read_config, write_config


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

    if "mode" in cfg:
        defaults["mode"] = cfg["mode"]
    if "setup" in cfg:
        defaults["setup"] = Path(cfg["setup"])
    if "equi" in cfg:
        defaults["equi"] = Path(cfg["equi"])
    if defaults:
        p.set_defaults(**defaults)


def _parse_args(
    cfg: dict[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="equi_waterbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--setup",
        type=Path,
        default=Path("setup"),
        help="Setup directory",
    )
    p.add_argument(
        "--equi",
        type=Path,
        default=Path("equi"),
        help="Equilibration output directory",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device index (OpenMM platform device id)",
    )
    p.add_argument(
        "--resources",
        type=str,
        default="CUDA",
        help="OpenMM platform/resources string",
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

    sdir = Path(args.setup).expanduser().resolve()
    edir = Path(args.equi).expanduser().resolve()
    resources = str(args.resources)
    device = int(args.device)

    cfg_path = Path(args.config)
    if bool(args.write_config):
        cfg["setup"] = format_value(args.setup)
        cfg["equi"] = format_value(args.equi)
        write_config(cfg_path, cfg)

    edir.mkdir(parents=True, exist_ok=True)

    sim = MDSim(
        pdb=str(sdir / "solvated.pdb"),
        xml=str(sdir / "system.xml"),
        restart=str(sdir / "initial.xml"),
    )

    sim.set_position_restraint(selection="protein and (name CA or name CB)")
    sim.setup_simulation(resources=resources, device=device, temperature=5, tstep=0.001)

    minsteps = 1000
    sim.minimize(nstep=minsteps)
    sim.write_state(str(edir / "equi_min.xml"))
    print(f"minimized for {minsteps} steps")

    equi_schedule = [
        [5, 10000],
        [10, 10000],
        [20, 10000],
        [50, 10000],
        [100, 10000],
        [200, 10000],
        [250, 10000],
        [298, 20000],
    ]

    for temp_k, nsteps in equi_schedule:
        pos = sim.get_positions()
        sim.setup_simulation(
            resources=resources,
            device=device,
            temperature=temp_k,
            gamma=1.0,
            tstep=0.001,
            positions=pos,
            resetvelocities=True,
        )
        sim.simulate(nstep=nsteps, logfile=str(edir / f"equi_{temp_k}.log"))
        sim.write_state(str(edir / f"equi_{temp_k}.xml"))
        print(f"{nsteps} steps at {temp_k}K")

    pos = sim.get_positions()
    vel = sim.get_velocities()
    sim.set_barostat(pressure=1, temperature=298)
    sim.setup_simulation(
        resources=resources,
        device=device,
        gamma=0.01,
        tstep=0.002,
        positions=pos,
        velocities=vel,
    )
    sim.simulate(
        nstep=10000,
        logfile=str(edir / "equi_298npt.log"),
        dcdfile=str(edir / "equi_298npt.dcd"),
    )
    print("10000 steps at 298K/1bar NPT")

    sim.write_state(str(edir / "equi_final.xml"))


if __name__ == "__main__":
    main()

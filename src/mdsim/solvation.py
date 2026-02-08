from __future__ import annotations

import base64
import zlib
from collections.abc import Mapping, Sequence
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np

# Embedded TIP3P cube (18.662 Å, 216 waters) from water.pdb; coords are float32 (Å), shape (216,3,3)
_WATER_BOXWIDTH_A = 18.662
_A_PER_NM = 10.0
_NM_PER_A = 0.1
_WATER_BOXWIDTH_NM = _WATER_BOXWIDTH_A * _NM_PER_A
_WATER_N = 216
_WATER_F32_ZLIB_B64 = """\
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
OR0=
"""


@lru_cache(maxsize=1)
def _water_box() -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(_WATER_F32_ZLIB_B64.encode("ascii")))
    return np.frombuffer(raw, dtype=np.float32).reshape((_WATER_N, 3, 3))


def parse_ions(spec: str) -> list[tuple[str, int]]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: list[tuple[str, int]] = []
    for part in spec.split("="):
        name, cnt = part.split(":")
        out.append((name.strip().upper(), int(cnt)))
    return out


def _anint(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5))


def _pbc_delta(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    return d - _anint(d / box) * box


def _ion_element(name: str) -> str:
    n = name.strip().upper()
    if n in {"SOD", "NA"}:
        return "NA"
    if n in {"CLA", "CL"}:
        return "CL"
    if n in {"POT", "K"}:
        return "K"
    if n in {"CAL", "CA"}:
        return "CA"
    if n == "MG":
        return "MG"
    import re

    return re.sub(r"[^A-Za-z]", "", n)[:2].upper() or "X"


def _to_nm_float(x: Any) -> float:
    """
    Convert a scalar length-like value to float(nm).

    Supports:
      - plain numbers (assumed nm)
      - OpenMM unit.Quantity (uses value_in_unit(nanometer))
    """
    # OpenMM Quantity has "value_in_unit"
    if hasattr(x, "value_in_unit"):
        try:
            from openmm import unit  # type: ignore

            return float(x.value_in_unit(unit.nanometer))
        except Exception as exc:  # pragma: no cover
            raise TypeError(f"Could not convert {type(x)} to nm float") from exc
    return float(x)


def _vec3_nm(vals: Sequence[Any]) -> tuple[float, float, float]:
    if len(vals) != 3:
        raise ValueError("expected a 3-vector")
    return (_to_nm_float(vals[0]), _to_nm_float(vals[1]), _to_nm_float(vals[2]))


def _build_grid(
    sol_xyz: np.ndarray,
    bmin: np.ndarray,
    dim: np.ndarray,
    *,
    dx: float,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    sol_xyz = np.asarray(sol_xyz, dtype=np.float32)
    if sol_xyz.size == 0:
        sol_xyz = sol_xyz.reshape((0, 3))
    elif sol_xyz.ndim == 1:
        sol_xyz = sol_xyz.reshape((-1, 3))

    bmin = np.asarray(bmin, dtype=np.float32)
    dim = np.asarray(dim, dtype=np.float32)

    grid = (dim / float(dx)).astype(np.int32) + 1
    nx, ny, nz = int(grid[0]), int(grid[1]), int(grid[2])
    n_cells = nx * ny * nz

    if periodic:
        # wrap around box center (solvfast uses origin-centered boxes)
        ctr = bmin + 0.5 * dim
        xyz = _pbc_delta(sol_xyz - ctr, dim) + ctr
    else:
        xyz = sol_xyz

    g = ((xyz - bmin) / float(dx)).astype(np.int32)  # (n,3)
    offs = np.array(list(product([-1, 0, 1], repeat=3)), dtype=np.int32)  # (27,3)
    neigh = g[:, None, :] + offs[None, :, :]  # (n,27,3)

    if periodic:
        neigh[..., 0] %= nx
        neigh[..., 1] %= ny
        neigh[..., 2] %= nz
        valid = np.ones(neigh.shape[:2], dtype=bool)
    else:
        valid = (
            (neigh[..., 0] >= 0)
            & (neigh[..., 0] < nx)
            & (neigh[..., 1] >= 0)
            & (neigh[..., 1] < ny)
            & (neigh[..., 2] >= 0)
            & (neigh[..., 2] < nz)
        )

    neigh = neigh.reshape(-1, 3)
    valid = valid.reshape(-1)
    neigh = neigh[valid]
    atom_rep = np.repeat(np.arange(xyz.shape[0], dtype=np.int32), 27)[valid]

    cell = neigh[:, 2] * (nx * ny) + neigh[:, 1] * nx + neigh[:, 0]
    order = np.argsort(cell, kind="mergesort")
    cell = cell[order]
    atom_rep = atom_rep[order]

    uniq, start, count = np.unique(cell, return_index=True, return_counts=True)
    cell_start = np.full(n_cells, -1, dtype=np.int32)
    cell_count = np.zeros(n_cells, dtype=np.int32)
    cell_start[uniq] = start.astype(np.int32, copy=False)
    cell_count[uniq] = count.astype(np.int32, copy=False)
    return cell_start, cell_count, atom_rep, (nx, ny, nz)


def solvate(
    model,
    *,
    box_min: Sequence[float] | None = None,
    box_max: Sequence[float] | None = None,
    box_size: Sequence[float] | None = None,
    padding: float = 0.9,
    solvcut: float = 0.21,
    watcut: float = 0.17,
    tip3p: bool = True,
    ions: Mapping[str, int] | Sequence[tuple[str, int]] | str | None = None,
    periodic: bool = False,
    seed: int | None = None,
):
    """
    Python replica of:

    Units:
      - model coordinates: Å
      - box_min/box_max/box_size/padding/solvcut/watcut/boxwidth: nm
      - returned box vectors: nm

    Returns (new_model, box_vectors_nm).
    """
    from mdsim import Atom, Chain, Model, Residue  # type: ignore

    boxwidth = _WATER_BOXWIDTH_NM

    # Normalize scalar cutoffs/padding in nm (accept OpenMM Quantities)
    padding = _to_nm_float(padding)
    solvcut = _to_nm_float(solvcut)
    watcut = _to_nm_float(watcut)

    if model is None:
        from types import SimpleNamespace

        model = SimpleNamespace(model_id=0, atoms=[])

    sol_xyz = np.asarray([(a.x, a.y, a.z) for a in model.atoms], dtype=np.float32)
    if sol_xyz.size == 0:
        sol_xyz = sol_xyz.reshape((0, 3))
    elif sol_xyz.ndim == 1:
        sol_xyz = sol_xyz.reshape((-1, 3))

    # API lengths are in nm; model coordinates are assumed to be Å (PDB convention).
    padding_a = float(padding) * _A_PER_NM
    solvcut_a = float(solvcut) * _A_PER_NM
    watcut_a = float(watcut) * _A_PER_NM
    boxwidth_a = float(boxwidth) * _A_PER_NM

    if box_size is not None and box_min is None and box_max is None:
        box_min = (0.0, 0.0, 0.0)
        box_max = _vec3_nm(box_size)

    if box_min is not None and box_max is not None:
        bmin = np.asarray(_vec3_nm(box_min), dtype=np.float32) * _A_PER_NM
        bmax = np.asarray(_vec3_nm(box_max), dtype=np.float32) * _A_PER_NM
    else:
        if sol_xyz.size == 0:
            raise ValueError("box_min/box_max or box_size required when model has no atoms")
        bmin = sol_xyz.min(axis=0)
        bmax = sol_xyz.max(axis=0)
        cofm = sol_xyz.mean(axis=0)
        for k in range(3):
            if (cofm[k] - bmin[k]) > (bmax[k] - cofm[k]):
                bmax[k] = 2.0 * cofm[k] - bmin[k]
            else:
                bmin[k] = 2.0 * cofm[k] - bmax[k]
        bmax[:] = float(bmax.max())
        bmin[:] = float(bmin.min())
        bmin -= padding_a
        bmax += padding_a

    dim = (bmax - bmin).astype(np.float32)
    if np.any(dim <= 0):
        raise ValueError(f"invalid box: {bmin=} {bmax=}")

    if ions is None:
        ion_list: list[tuple[str, int]] = []
    elif isinstance(ions, str):
        ion_list = parse_ions(ions)
    elif isinstance(ions, Mapping):
        ion_list = [(str(k).upper(), int(v)) for k, v in ions.items() if int(v) > 0]
    else:
        ion_list = [(str(k).upper(), int(v)) for k, v in ions if int(v) > 0]

    cell_start, cell_count, atom_sorted, (nx, ny, nz) = _build_grid(
        sol_xyz, bmin, dim, dx=solvcut_a, periodic=bool(periodic)
    )
    nxy = nx * ny
    solvcutsq = solvcut_a * solvcut_a
    watcutsq = watcut_a * watcut_a

    tpl = _water_box()
    tpl_O = tpl[:, 0, :]

    xmult = int(dim[0] / boxwidth_a) * 2
    ymult = int(dim[1] / boxwidth_a) * 2
    zmult = int(dim[2] / boxwidth_a) * 2

    keep_res: list[int] = []
    keep_add: list[tuple[float, float, float]] = []

    # Edge-clash cache: store all atoms of accepted edge waters (N_edge_atoms,3)
    edge_atoms = np.empty((64, 3), dtype=np.float32)
    edge_n = 0

    def edge_clash(wat_atoms: np.ndarray) -> bool:
        nonlocal edge_n
        if edge_n == 0:
            return False
        d = _pbc_delta(wat_atoms[:, None, :] - edge_atoms[:edge_n][None, :, :], dim)
        return bool(np.any(np.sum(d * d, axis=-1) < watcutsq))

    def edge_add(wat_atoms: np.ndarray) -> None:
        nonlocal edge_atoms, edge_n
        need = edge_n + 3
        if need > edge_atoms.shape[0]:
            new = np.empty((max(need, int(edge_atoms.shape[0] * 1.5) + 16), 3), dtype=np.float32)
            new[:edge_n] = edge_atoms[:edge_n]
            edge_atoms = new
        edge_atoms[edge_n : edge_n + 3] = wat_atoms
        edge_n += 3

    for ix in range(xmult + 1):
        ax = float(ix) * boxwidth_a + float(bmin[0])
        for iy in range(ymult + 1):
            ay = float(iy) * boxwidth_a + float(bmin[1])
            for iz in range(zmult + 1):
                az = float(iz) * boxwidth_a + float(bmin[2])
                a = np.array([ax, ay, az], dtype=np.float32)

                Opos = tpl_O + a
                inside = (
                    (Opos[:, 0] > bmin[0])
                    & (Opos[:, 0] < bmax[0])
                    & (Opos[:, 1] > bmin[1])
                    & (Opos[:, 1] < bmax[1])
                    & (Opos[:, 2] > bmin[2])
                    & (Opos[:, 2] < bmax[2])
                )
                cand = np.nonzero(inside)[0]
                if cand.size == 0:
                    continue

                for ridx in cand.tolist():
                    wat = tpl[ridx] + a  # (3,3)

                    good = True
                    for cc in wat:
                        ig = ((cc - bmin) / solvcut_a).astype(np.int32)
                        if (
                            ig[0] < 0
                            or ig[0] >= nx
                            or ig[1] < 0
                            or ig[1] >= ny
                            or ig[2] < 0
                            or ig[2] >= nz
                        ):
                            continue
                        cell = int(ig[2] * nxy + ig[1] * nx + ig[0])
                        cnt = int(cell_count[cell])
                        if cnt <= 0:
                            continue
                        start = int(cell_start[cell])
                        idx = atom_sorted[start : start + cnt]
                        d = cc - sol_xyz[idx]
                        if periodic:
                            d = _pbc_delta(d, dim)
                        if np.any(np.sum(d * d, axis=1) < solvcutsq):
                            good = False
                            break
                    if not good:
                        continue

                    ox, oy, oz = map(float, Opos[ridx])
                    edge = (
                        ox < float(bmin[0]) + 2.0
                        or ox > float(bmax[0]) - 2.0
                        or oy < float(bmin[1]) + 2.0
                        or oy > float(bmax[1]) - 2.0
                        or oz < float(bmin[2]) + 2.0
                        or oz > float(bmax[2]) - 2.0
                    )
                    if edge and edge_clash(wat):
                        continue

                    keep_res.append(int(ridx))
                    keep_add.append((float(a[0]), float(a[1]), float(a[2])))
                    if edge:
                        edge_add(wat.astype(np.float32, copy=False))

    keep_res_a = np.asarray(keep_res, dtype=np.int32)
    keep_add_a = np.asarray(keep_add, dtype=np.float32)
    n_wat = int(keep_res_a.shape[0])

    replaced = np.zeros(n_wat, dtype=bool)
    ion_chosen: dict[str, np.ndarray] = {}

    if ion_list:
        rng = np.random.default_rng(seed)
        o_pos = tpl_O[keep_res_a] + keep_add_a
        ig = ((o_pos - bmin) / solvcut_a).astype(np.int32)
        ig[:, 0] = np.clip(ig[:, 0], 0, nx - 1)
        ig[:, 1] = np.clip(ig[:, 1], 0, ny - 1)
        ig[:, 2] = np.clip(ig[:, 2], 0, nz - 1)
        cell = (ig[:, 2] * nxy + ig[:, 1] * nx + ig[:, 0]).astype(np.int32)
        bulk_ok = cell_count[cell] == 0  # same heuristic as solvfast
        avail = np.nonzero(bulk_ok)[0]
        used = np.zeros(n_wat, dtype=bool)

        for name, cnt in ion_list:
            cnt = int(cnt)
            if cnt <= 0:
                ion_chosen[name] = np.empty((0,), dtype=np.int32)
                continue
            avail2 = avail[~used[avail]]
            if cnt > int(avail2.size):
                msg = f"not enough bulk waters for {name}: need {cnt}, have {avail2.size}"
                raise ValueError(msg)
            chosen = rng.choice(avail2, size=cnt, replace=False).astype(np.int32, copy=False)
            used[chosen] = True
            ion_chosen[name] = chosen

        replaced = used

    new = Model(model_id=model.model_id)
    serial = 1

    def add(m: Model, atom: Atom) -> None:
        m.atoms.append(atom)
        key = atom.seg.strip() or (atom.chain.strip() or " ")
        ch = m.chain.get(key)
        if ch is None:
            ch = Chain(key_id=key, residues=[], seg_id=(atom.seg.strip() or None))
            ch.chain_id = atom.chain or " "
            m.chain[key] = ch
        rid = (atom.resname, atom.chain, atom.resnum, atom.seg)
        if (
            not ch.residues
            or (
                ch.residues[-1].resname,
                ch.residues[-1].chain,
                ch.residues[-1].resnum,
                ch.residues[-1].seg,
            )
            != rid
        ):
            ch.residues.append(Residue(*rid))
        ch.residues[-1].atoms.append(atom)

    # solute
    for a in model.atoms:
        add(
            new,
            Atom(
                serial=serial,
                name=a.name,
                element=a.element,
                resname=a.resname,
                chain=a.chain,
                resnum=a.resnum,
                x=float(a.x),
                y=float(a.y),
                z=float(a.z),
                seg=a.seg,
                mass=a.mass,
            ),
        )
        serial += 1

    # ions
    if ion_list and np.any(replaced):
        o_pos = tpl_O[keep_res_a] + keep_add_a
        nseg = 0
        for ion_name, cnt in ion_list:
            cnt = int(cnt)
            seg = f"I{nseg:03d}"
            if cnt <= 0:
                nseg += 1
                continue
            el = _ion_element(ion_name)
            resnum = 1
            for wi in ion_chosen.get(ion_name, np.empty((0,), dtype=np.int32)).tolist():
                x, y, z = map(float, o_pos[wi])
                add(
                    new,
                    Atom(
                        serial=serial,
                        name=ion_name,
                        element=el,
                        resname=ion_name,
                        chain=" ",
                        resnum=resnum,
                        x=x,
                        y=y,
                        z=z,
                        seg=seg,
                        mass=None,
                    ),
                )
                serial += 1
                resnum += 1
                if resnum >= 10000:
                    resnum = 1
                    nseg += 1
                    seg = f"I{nseg:03d}"
            nseg += 1

    # waters
    wres = "TIP3" if tip3p else "HOH"
    w_names = ("OH2", "H1", "H2")
    w_elems = ("O", "H", "H")
    resnum = 1
    nseg = 0
    seg = f"W{nseg:03d}"
    for i in range(n_wat):
        if replaced[i]:
            continue
        if resnum == 1:
            seg = f"W{nseg:03d}"
        wat = tpl[int(keep_res_a[i])] + keep_add_a[i]
        for nm, el, cc in zip(w_names, w_elems, wat):
            add(
                new,
                Atom(
                    serial=serial,
                    name=nm,
                    element=el,
                    resname=wres,
                    chain=" ",
                    resnum=resnum,
                    x=float(cc[0]),
                    y=float(cc[1]),
                    z=float(cc[2]),
                    seg=seg,
                    mass=None,
                ),
            )
            serial += 1
        resnum += 1
        if resnum >= 10000:
            resnum = 1
            nseg += 1

    return new, dim.copy() * _NM_PER_A

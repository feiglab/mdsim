from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np

from .analysis import (
    _as_file_list,
    _box_lengths_nm,
    _min_image_disp_nm,
    _pair_distances_nm,
    _peek_first_box_nm,
    _sinc,
    _wrap_nm,
    atom_masses,
    group_centers_nm,
)
from .molecule_data import PDBReader, StructureSelector, iter_dcd

FileLike = Union[str, Path]


_Z_BY_EL = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "P": 15,
    "S": 16,
    "F": 9,
    "CL": 17,
    "BR": 35,
    "I": 53,
    "NA": 11,
    "K": 19,
    "MG": 12,
    "CA": 20,
    "ZN": 30,
    "FE": 26,
    "CU": 29,
    "MN": 25,
    "CO": 27,
    "NI": 28,
}

try:
    # Optional dependency; fastest path is to precompute per-element f(q).
    from periodictable import cromermann as _pt_cm  # type: ignore
except Exception:  # pragma: no cover
    _pt_cm = None  # type: ignore


@dataclass(frozen=True)
class _WeightSpec:
    mode: str  # "scalar" or "xray"
    w_sel: Optional[np.ndarray] = None  # (n_atoms,)
    el_id_sel: Optional[np.ndarray] = None  # (n_atoms,) int
    f_el_q: Optional[np.ndarray] = None  # (n_el, n_q) float


def _element_key(el: str) -> str:
    s = (el or "").strip().upper()
    if not s:
        return "C"
    # Normalize common PDB quirks
    if s == "CL":
        return "CL"
    if s == "NA":
        return "NA"
    return s


def _build_xray_tables_for_selection(
    template_model: Any,
    atom_indices_full: list[int],
    q_nm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      el_id_sel : (n_atoms,) int
      f_el_q    : (n_el, n_q) float
    """
    if _pt_cm is None:
        raise ImportError(
            "weights='xray' requires 'periodictable'. " "Install with: pip install periodictable"
        )

    # Collect element keys per selected atom
    keys: list[str] = []
    for ai in atom_indices_full:
        el = getattr(template_model.atoms[int(ai)], "element", "") or ""
        keys.append(_element_key(str(el)))

    uniq = sorted(set(keys))
    key_to_id = {k: i for i, k in enumerate(uniq)}
    el_id = np.asarray([key_to_id[k] for k in keys], dtype=np.int32)

    # periodictable expects s = sin(theta)/lambda in 1/Angstrom
    # Your q is in nm^-1 and q = 4*pi*s, with s in 1/nm.
    # Convert: s(1/A) = (q / (4*pi)) * (1 nm^-1 -> 0.1 A^-1) = q / (40*pi)
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    stol = q / (40.0 * math.pi)

    f_el_q = np.empty((len(uniq), q.size), dtype=np.float64)
    for k, i in key_to_id.items():
        # fxrayatstol returns electrons (real f0)
        f_el_q[int(i)] = np.asarray(_pt_cm.fxrayatstol(k, stol), dtype=np.float64)

    return el_id, f_el_q


def _atomic_weights_for_selection(
    template_model: Any,
    atom_indices_full: list[int],
    *,
    weights: str,
    q_nm1: Optional[np.ndarray] = None,
) -> _WeightSpec:
    mode = str(weights).strip().lower()
    if mode not in {"unity", "z", "xray"}:
        raise ValueError("weights must be 'unity', 'z', or 'xray'")

    if mode == "xray":
        if q_nm1 is None:
            raise ValueError("q_nm1 is required when weights='xray'")
        el_id, f_el_q = _build_xray_tables_for_selection(
            template_model,
            atom_indices_full,
            np.asarray(q_nm1, dtype=np.float64).reshape(-1),
        )
        return _WeightSpec(mode="xray", el_id_sel=el_id, f_el_q=f_el_q)

    if mode == "unity":
        w = np.ones(len(atom_indices_full), dtype=np.float64)
        return _WeightSpec(mode="scalar", w_sel=w)

    w = np.empty(len(atom_indices_full), dtype=np.float64)
    for i, ai in enumerate(atom_indices_full):
        el = getattr(template_model.atoms[int(ai)], "element", "") or ""
        key = _element_key(str(el))
        w[i] = float(_Z_BY_EL.get(key, 6))
    return _WeightSpec(mode="scalar", w_sel=w)


def _protein_groups_from_selection(
    pdb_file: FileLike,
    selection: Union[str, list[list[int]]],
) -> tuple[Any, list[np.ndarray], list[int], dict[str, int]]:
    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    if isinstance(selection, str):
        groups_full = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_full = [[int(i) for i in g] for g in selection]

    groups_full = [g for g in groups_full if g]
    if not groups_full:
        raise ValueError("selection produced no atoms")
    n_prot = len(groups_full)

    atom_set: set[int] = set()
    for g in groups_full:
        atom_set.update(int(i) for i in g)
    atom_indices = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices)}
    groups_sel = []
    for g in groups_full:
        groups_sel.append(np.asarray([idx_map[int(i)] for i in g], dtype=np.int32))

    return tmpl_model, groups_sel, atom_indices, {"n_proteins": n_prot}


def element_counts_for_selection(
    pdb_file: FileLike,
    selection: Union[str, list[list[int]]] = "protein",
) -> dict[str, int]:
    """
    Diagnostic: count elements in the selected atoms (after union over groups).

    Uses atom.element when present; missing/blank elements are counted as "UNK".
    """
    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    if isinstance(selection, str):
        groups_full = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_full = [[int(i) for i in g] for g in selection]

    groups_full = [g for g in groups_full if g]
    if not groups_full:
        raise ValueError("selection produced no atoms")

    atom_set: set[int] = set()
    for g in groups_full:
        atom_set.update(int(i) for i in g)
    atom_indices = sorted(atom_set)

    counts: dict[str, int] = {}
    for ai in atom_indices:
        el = getattr(tmpl_model.atoms[int(ai)], "element", "") or ""
        key = str(el).strip().upper() or "UNK"
        counts[key] = counts.get(key, 0) + 1

    # deterministic ordering for printing
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def print_element_counts(
    pdb_file: FileLike,
    selection: Union[str, list[list[int]]] = "protein",
    *,
    top_n: int | None = None,
) -> None:
    counts = element_counts_for_selection(pdb_file, selection=selection)
    items = list(counts.items())
    if top_n is not None:
        items = items[: int(top_n)]

    total = sum(counts.values())
    print(f"Selected atoms: {total}")
    for el, n in items:
        print(f"{el:>4s}  {n:>8d}  ({n/total:6.2%})")


def debye_intensity_nm(
    xyz_nm: np.ndarray,
    q_nm1: np.ndarray,
    weights: np.ndarray,
    *,
    atom_block: int = 512,
    q_block: int = 64,
) -> np.ndarray:
    """
    Debye sum:
        I(q) = sum_i sum_j w_i w_j sinc(q r_ij)

    Uses blocked accumulation to avoid huge (n_pairs x n_q) allocations.
    """
    x = np.asarray(xyz_nm, dtype=np.float64)
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("xyz_nm must have shape (n_atoms, 3)")
    if x.shape[0] != w.shape[0]:
        raise ValueError("weights length must match number of atoms")
    if q.size < 1:
        raise ValueError("q_nm1 must be non-empty")

    n = int(x.shape[0])
    nq = int(q.size)
    out = np.zeros(nq, dtype=np.float64)

    # i==j term
    out += float(np.sum(w * w))

    if n < 2:
        return out

    ab = max(32, int(atom_block))
    qb = max(16, int(q_block))

    for i0 in range(0, n, ab):
        i1 = min(n, i0 + ab)
        xi = x[i0:i1]
        wi = w[i0:i1]

        # within-block upper triangle
        bi = i1 - i0
        if bi >= 2:
            di = xi[:, None, :] - xi[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", di, di)
            iu, ju = np.triu_indices(bi, k=1)
            r = np.sqrt(d2[iu, ju])
            wp = wi[iu] * wi[ju]
            if r.size:
                for q0 in range(0, nq, qb):
                    q1 = min(nq, q0 + qb)
                    qr = r[:, None] * q[q0:q1][None, :]
                    out[q0:q1] += 2.0 * np.sum(wp[:, None] * _sinc(qr), axis=0)

        # cross blocks
        for j0 in range(i1, n, ab):
            j1 = min(n, j0 + ab)
            xj = x[j0:j1]
            wj = w[j0:j1]

            d = xi[:, None, :] - xj[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", d, d)
            r = np.sqrt(d2).reshape(-1)
            wp = (wi[:, None] * wj[None, :]).reshape(-1)

            if r.size == 0:
                continue
            for q0 in range(0, nq, qb):
                q1 = min(nq, q0 + qb)
                qr = r[:, None] * q[q0:q1][None, :]
                out[q0:q1] += 2.0 * np.sum(wp[:, None] * _sinc(qr), axis=0)

    return out


def debye_intensity_nm_xray(
    xyz_nm: np.ndarray,
    q_nm1: np.ndarray,
    el_id: np.ndarray,
    f_el_q: np.ndarray,
    *,
    atom_block: int = 512,
    q_block: int = 64,
) -> np.ndarray:
    """
    Debye sum with q-dependent atomic form factors:
        I(q) = sum_i sum_j f_i(q) f_j(q) sinc(q r_ij)

    el_id: (n_atoms,) int indexes element table
    f_el_q: (n_el, n_q) f(element, q) in electrons
    """
    x = np.asarray(xyz_nm, dtype=np.float64)
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    el = np.asarray(el_id, dtype=np.int32).reshape(-1)
    ftab = np.asarray(f_el_q, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("xyz_nm must have shape (n_atoms, 3)")
    if x.shape[0] != el.shape[0]:
        raise ValueError("el_id length must match number of atoms")
    if q.size < 1:
        raise ValueError("q_nm1 must be non-empty")
    if ftab.ndim != 2 or ftab.shape[1] != q.size:
        raise ValueError("f_el_q must have shape (n_el, n_q) matching q_nm1")

    n = int(x.shape[0])
    nq = int(q.size)
    out = np.zeros(nq, dtype=np.float64)

    ab = max(32, int(atom_block))
    qb = max(16, int(q_block))

    # i==j term: sum_i f_i(q)^2 (accumulate in q-blocks)
    for q0 in range(0, nq, qb):
        q1 = min(nq, q0 + qb)
        fi = ftab[el, q0:q1]  # (n_atoms, qb)
        out[q0:q1] += np.sum(fi * fi, axis=0)

    if n < 2:
        return out

    for i0 in range(0, n, ab):
        i1 = min(n, i0 + ab)
        xi = x[i0:i1]
        eli = el[i0:i1]
        bi = i1 - i0

        # within-block upper triangle
        if bi >= 2:
            di = xi[:, None, :] - xi[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", di, di)
            iu, ju = np.triu_indices(bi, k=1)
            r = np.sqrt(d2[iu, ju])
            if r.size:
                el_i = eli[iu]
                el_j = eli[ju]
                for q0 in range(0, nq, qb):
                    q1 = min(nq, q0 + qb)
                    fi = ftab[el_i, q0:q1]
                    fj = ftab[el_j, q0:q1]
                    wp = fi * fj  # (n_pairs, qb)
                    qr = r[:, None] * q[q0:q1][None, :]
                    out[q0:q1] += 2.0 * np.sum(wp * _sinc(qr), axis=0)

        # cross blocks: avoid repeat/tile allocations
        for j0 in range(i1, n, ab):
            j1 = min(n, j0 + ab)
            xj = x[j0:j1]
            elj = el[j0:j1]
            bj = j1 - j0

            d = xi[:, None, :] - xj[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", d, d)
            r = np.sqrt(d2)  # (bi, bj)
            if r.size == 0:
                continue

            for ii in range(bi):
                ri = r[ii]  # (bj,)
                eli_i = int(eli[ii])

                for q0 in range(0, nq, qb):
                    q1 = min(nq, q0 + qb)
                    fi = ftab[eli_i, q0:q1]  # (qb,)
                    fj = ftab[elj, q0:q1]  # (bj, qb)
                    wp = fj * fi[None, :]  # (bj, qb)
                    qr = ri[:, None] * q[q0:q1][None, :]
                    out[q0:q1] += 2.0 * np.sum(wp * _sinc(qr), axis=0)

            # bj is used implicitly via shapes; keep it for clarity
            _ = bj

    return out


def _min_image_disp_nm_3d(d: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    b = np.asarray(box_nm, dtype=np.float64).reshape(1, 1, 3)
    return d - np.rint(d / b) * b


def debye_intensity_nm_pbc(
    xyz_nm: np.ndarray,
    q_nm1: np.ndarray,
    weights: np.ndarray,
    box_nm: np.ndarray,
    *,
    atom_block: int = 512,
    q_block: int = 64,
) -> np.ndarray:
    """
    Debye sum with orthorhombic PBC (minimum image):
        I(q) = sum_i sum_j w_i w_j sinc(q r_ij)
    """
    x = np.asarray(xyz_nm, dtype=np.float64)
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    box = np.asarray(box_nm, dtype=np.float64).reshape(3)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("xyz_nm must have shape (n_atoms, 3)")
    if x.shape[0] != w.shape[0]:
        raise ValueError("weights length must match number of atoms")
    if q.size < 1:
        raise ValueError("q_nm1 must be non-empty")
    if np.any(box <= 0.0):
        raise ValueError("box lengths must be positive")

    n = int(x.shape[0])
    nq = int(q.size)
    out = np.zeros(nq, dtype=np.float64)

    out += float(np.sum(w * w))
    if n < 2:
        return out

    ab = max(32, int(atom_block))
    qb = max(16, int(q_block))

    for i0 in range(0, n, ab):
        i1 = min(n, i0 + ab)
        xi = x[i0:i1]
        wi = w[i0:i1]
        bi = i1 - i0

        if bi >= 2:
            di = xi[:, None, :] - xi[None, :, :]
            di = _min_image_disp_nm_3d(di, box)
            d2 = np.einsum("ijk,ijk->ij", di, di)
            iu, ju = np.triu_indices(bi, k=1)
            r = np.sqrt(d2[iu, ju])
            wp = wi[iu] * wi[ju]
            if r.size:
                for q0 in range(0, nq, qb):
                    q1 = min(nq, q0 + qb)
                    qr = r[:, None] * q[q0:q1][None, :]
                    out[q0:q1] += 2.0 * np.sum(wp[:, None] * _sinc(qr), axis=0)

        for j0 in range(i1, n, ab):
            j1 = min(n, j0 + ab)
            xj = x[j0:j1]
            wj = w[j0:j1]

            d = xi[:, None, :] - xj[None, :, :]
            d = _min_image_disp_nm_3d(d, box)
            d2 = np.einsum("ijk,ijk->ij", d, d)
            r = np.sqrt(d2).reshape(-1)
            wp = (wi[:, None] * wj[None, :]).reshape(-1)

            if r.size == 0:
                continue
            for q0 in range(0, nq, qb):
                q1 = min(nq, q0 + qb)
                qr = r[:, None] * q[q0:q1][None, :]
                out[q0:q1] += 2.0 * np.sum(wp[:, None] * _sinc(qr), axis=0)

    return out


def debye_intensity_nm_xray_pbc(
    xyz_nm: np.ndarray,
    q_nm1: np.ndarray,
    el_id: np.ndarray,
    f_el_q: np.ndarray,
    box_nm: np.ndarray,
    *,
    atom_block: int = 512,
    q_block: int = 64,
) -> np.ndarray:
    """
    Debye sum with q-dependent atomic form factors + orthorhombic PBC (minimum image):
        I(q) = sum_i sum_j f_i(q) f_j(q) sinc(q r_ij)
    """
    x = np.asarray(xyz_nm, dtype=np.float64)
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    el = np.asarray(el_id, dtype=np.int32).reshape(-1)
    ftab = np.asarray(f_el_q, dtype=np.float64)
    box = np.asarray(box_nm, dtype=np.float64).reshape(3)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("xyz_nm must have shape (n_atoms, 3)")
    if x.shape[0] != el.shape[0]:
        raise ValueError("el_id length must match number of atoms")
    if q.size < 1:
        raise ValueError("q_nm1 must be non-empty")
    if ftab.ndim != 2 or ftab.shape[1] != q.size:
        raise ValueError("f_el_q must have shape (n_el, n_q) matching q_nm1")
    if np.any(box <= 0.0):
        raise ValueError("box lengths must be positive")

    n = int(x.shape[0])
    nq = int(q.size)
    out = np.zeros(nq, dtype=np.float64)

    ab = max(32, int(atom_block))
    qb = max(16, int(q_block))

    # i==j: sum_i f_i(q)^2
    for q0 in range(0, nq, qb):
        q1 = min(nq, q0 + qb)
        fi = ftab[el, q0:q1]
        out[q0:q1] += np.sum(fi * fi, axis=0)

    if n < 2:
        return out

    for i0 in range(0, n, ab):
        i1 = min(n, i0 + ab)
        xi = x[i0:i1]
        eli = el[i0:i1]
        bi = i1 - i0

        if bi >= 2:
            di = xi[:, None, :] - xi[None, :, :]
            di = _min_image_disp_nm_3d(di, box)
            d2 = np.einsum("ijk,ijk->ij", di, di)
            iu, ju = np.triu_indices(bi, k=1)
            r = np.sqrt(d2[iu, ju])
            if r.size:
                el_i = eli[iu]
                el_j = eli[ju]
                for q0 in range(0, nq, qb):
                    q1 = min(nq, q0 + qb)
                    fi = ftab[el_i, q0:q1]
                    fj = ftab[el_j, q0:q1]
                    wp = fi * fj
                    qr = r[:, None] * q[q0:q1][None, :]
                    out[q0:q1] += 2.0 * np.sum(wp * _sinc(qr), axis=0)

        for j0 in range(i1, n, ab):
            j1 = min(n, j0 + ab)
            xj = x[j0:j1]
            elj = el[j0:j1]
            bj = j1 - j0

            d = xi[:, None, :] - xj[None, :, :]
            d = _min_image_disp_nm_3d(d, box)
            d2 = np.einsum("ijk,ijk->ij", d, d)
            r = np.sqrt(d2)  # (bi, bj)
            if r.size == 0:
                continue

            # Avoid repeat/tile allocations: loop small bi, vectorize over bj
            for ii in range(bi):
                ri = r[ii]
                eli_i = int(eli[ii])

                for q0 in range(0, nq, qb):
                    q1 = min(nq, q0 + qb)
                    fi = ftab[eli_i, q0:q1]
                    fj = ftab[elj, q0:q1]
                    wp = fj * fi[None, :]
                    qr = ri[:, None] * q[q0:q1][None, :]
                    out[q0:q1] += 2.0 * np.sum(wp * _sinc(qr), axis=0)

            _ = bj  # keep if you want; remove if Ruff flags it

    return out


# ---------------------------
# Parallel frame processing
# ---------------------------

# The ProcessPoolExecutor initializer stores these read-only objects in each worker.
_G_PROT_GROUPS_SEL: Optional[list[np.ndarray]] = None
_G_PROT_LENS: Optional[np.ndarray] = None
_G_Q: Optional[np.ndarray] = None
_G_ATOM_BLOCK: int = 512
_G_Q_BLOCK: int = 64
_G_BLAS_THREADS: int = 1
_G_WSPEC: Optional[_WeightSpec] = None

# --- Full-system Debye: process worker globals -----------------------------

_G_FULL_Q: Optional[np.ndarray] = None
_G_FULL_WSPEC: Optional[_WeightSpec] = None
_G_FULL_ATOM_BLOCK: int = 512
_G_FULL_Q_BLOCK: int = 64

# --- Reciprocal full intensity: worker globals --------------------------------

_G_RECIP_Q: Optional[np.ndarray] = None
_G_RECIP_QEDGES: Optional[np.ndarray] = None
_G_RECIP_WSPEC: Optional[_WeightSpec] = None
_G_RECIP_QVEC_BLOCK: int = 512


def _init_full_debye_worker(
    q_nm1: np.ndarray,
    wspec: _WeightSpec,
    atom_block: int,
    q_block: int,
) -> None:
    global _G_FULL_Q, _G_FULL_WSPEC, _G_FULL_ATOM_BLOCK, _G_FULL_Q_BLOCK
    _G_FULL_Q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    _G_FULL_WSPEC = wspec
    _G_FULL_ATOM_BLOCK = int(atom_block)
    _G_FULL_Q_BLOCK = int(q_block)


def _full_debye_one_frame_worker(
    xyz_sel_nm: np.ndarray,
    box_nm: np.ndarray,
) -> np.ndarray:
    if _G_FULL_Q is None or _G_FULL_WSPEC is None:
        raise RuntimeError("full debye worker globals not initialized")

    q = _G_FULL_Q
    wspec = _G_FULL_WSPEC
    b = _box_lengths_nm(np.asarray(box_nm, dtype=np.float64).reshape(3))
    xyz_wr = _wrap_nm(np.asarray(xyz_sel_nm, dtype=np.float64), b)

    if wspec.mode == "scalar":
        if wspec.w_sel is None:
            raise ValueError("wspec.w_sel is None for scalar mode")
        w = np.asarray(wspec.w_sel, dtype=np.float64)
        return debye_intensity_nm_pbc(
            xyz_wr,
            q,
            w,
            b,
            atom_block=_G_FULL_ATOM_BLOCK,
            q_block=_G_FULL_Q_BLOCK,
        )

    if wspec.el_id_sel is None or wspec.f_el_q is None:
        raise ValueError("wspec missing el_id_sel/f_el_q for xray mode")
    el = np.asarray(wspec.el_id_sel, dtype=np.int32)
    ftab = np.asarray(wspec.f_el_q, dtype=np.float64)
    return debye_intensity_nm_xray_pbc(
        xyz_wr,
        q,
        el,
        ftab,
        b,
        atom_block=_G_FULL_ATOM_BLOCK,
        q_block=_G_FULL_Q_BLOCK,
    )


def _init_frame_worker(
    prot_groups_sel: list[np.ndarray],
    prot_lens: np.ndarray,
    wspec: _WeightSpec,
    q_nm1: np.ndarray,
    atom_block: int,
    q_block: int,
    blas_threads: int,
) -> None:
    global _G_PROT_GROUPS_SEL, _G_PROT_LENS, _G_WSPEC, _G_Q
    global _G_ATOM_BLOCK, _G_Q_BLOCK, _G_BLAS_THREADS
    _G_PROT_GROUPS_SEL = prot_groups_sel
    _G_PROT_LENS = np.asarray(prot_lens, dtype=np.int64)
    _G_WSPEC = wspec
    _G_Q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    _G_ATOM_BLOCK = int(atom_block)
    _G_Q_BLOCK = int(q_block)
    _G_BLAS_THREADS = int(blas_threads)


def _process_frame_batch_core(
    xyz_batch_nm: np.ndarray,
    box_batch_nm: np.ndarray,
    clusters_batch: list[list[list[int]]],
    prot_groups_sel: list[np.ndarray],
    prot_lens: np.ndarray,
    wspec: _WeightSpec,
    q_nm1: np.ndarray,
    *,
    atom_block: int,
    q_block: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    """
    Returns:
        sums[m]  = sum over cluster-instances of size m of (I(q)/m)
        sums2[m] = sum over cluster-instances of size m of (I(q)/m)^2
        counts[m]= number of cluster-instances of size m
    """
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)

    sums: dict[int, np.ndarray] = {}
    sums2: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    xyz_batch_nm = np.asarray(xyz_batch_nm)
    box_batch_nm = np.asarray(box_batch_nm, dtype=np.float64)

    if xyz_batch_nm.ndim != 3 or xyz_batch_nm.shape[-1] != 3:
        raise ValueError("xyz_batch_nm must have shape (n_frames, n_atoms, 3)")
    if box_batch_nm.ndim != 2 or box_batch_nm.shape[-1] != 3:
        raise ValueError("box_batch_nm must have shape (n_frames, 3)")
    if xyz_batch_nm.shape[0] != box_batch_nm.shape[0]:
        raise ValueError("xyz_batch_nm and box_batch_nm must have the same n_frames")

    # Hoist weight tables out of inner loops
    if wspec.mode == "scalar":
        if wspec.w_sel is None:
            raise ValueError("wspec.w_sel is None for scalar mode")
        w_sel = np.asarray(wspec.w_sel, dtype=np.float64)
        el_sel = None
        f_el_q = None
    else:
        if wspec.el_id_sel is None or wspec.f_el_q is None:
            raise ValueError("wspec missing el_id_sel/f_el_q for xray mode")
        el_sel = np.asarray(wspec.el_id_sel, dtype=np.int32)
        f_el_q = np.asarray(wspec.f_el_q, dtype=np.float64)
        w_sel = None

    n_frames = int(xyz_batch_nm.shape[0])
    n_prot = int(len(prot_groups_sel))

    for fi in range(n_frames):
        b = _box_lengths_nm(box_batch_nm[fi])
        xyz_wr = _wrap_nm(np.asarray(xyz_batch_nm[fi], dtype=np.float64), b)

        centers = np.empty((n_prot, 3), dtype=np.float64)
        for pid, idx in enumerate(prot_groups_sel):
            centers[pid] = xyz_wr[idx].mean(axis=0)

        for cl in clusters_batch[fi]:
            if not cl:
                continue

            prot_ids = np.asarray(cl, dtype=np.int32)
            m = int(prot_ids.size)
            if m < 1:
                continue

            c = centers[prot_ids]
            ref = c[0:1]
            disp = _min_image_disp_nm(c - ref, b)
            shifts = (ref + disp) - c

            n_atoms_cl = int(np.sum(prot_lens[prot_ids], dtype=np.int64))
            if n_atoms_cl < 1:
                continue

            x_cl = np.empty((n_atoms_cl, 3), dtype=np.float64)
            if wspec.mode == "scalar":
                w_cl = np.empty((n_atoms_cl,), dtype=np.float64)
            else:
                el_cl = np.empty((n_atoms_cl,), dtype=np.int32)

            pos = 0
            for k, pid in enumerate(prot_ids.tolist()):
                idx = prot_groups_sel[int(pid)]
                n_i = int(idx.size)
                x_cl[pos : pos + n_i] = xyz_wr[idx] + shifts[k]
                if wspec.mode == "scalar":
                    w_cl[pos : pos + n_i] = w_sel[idx]  # type: ignore[index]
                else:
                    el_cl[pos : pos + n_i] = el_sel[idx]  # type: ignore[index]
                pos += n_i

            if wspec.mode == "scalar":
                i_q = debye_intensity_nm(
                    x_cl,
                    q,
                    w_cl,  # type: ignore[arg-type]
                    atom_block=atom_block,
                    q_block=q_block,
                )
            else:
                i_q = debye_intensity_nm_xray(
                    x_cl,
                    q,
                    el_cl,  # type: ignore[arg-type]
                    f_el_q,  # type: ignore[arg-type]
                    atom_block=atom_block,
                    q_block=q_block,
                )

            i_per = i_q / float(m)

            if m not in sums:
                sums[m] = np.zeros_like(q)
                sums2[m] = np.zeros_like(q)
                counts[m] = 0
            sums[m] += i_per
            sums2[m] += i_per * i_per
            counts[m] += 1

    return sums, sums2, counts


def _process_frame_batch_worker(
    xyz_batch_nm: np.ndarray,
    box_batch_nm: np.ndarray,
    clusters_batch: list[list[list[int]]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    """Worker entry point (requires _init_frame_worker to have run)."""
    if _G_PROT_GROUPS_SEL is None or _G_PROT_LENS is None or _G_WSPEC is None or _G_Q is None:
        raise RuntimeError("worker globals not initialized")

    try:
        from threadpoolctl import threadpool_limits
    except Exception:  # pragma: no cover
        threadpool_limits = None  # type: ignore[assignment]

    if threadpool_limits is None:
        return _process_frame_batch_core(
            xyz_batch_nm,
            box_batch_nm,
            clusters_batch,
            _G_PROT_GROUPS_SEL,
            _G_PROT_LENS,
            _G_WSPEC,
            _G_Q,
            atom_block=_G_ATOM_BLOCK,
            q_block=_G_Q_BLOCK,
        )

    # Limit BLAS/OpenMP threads inside each worker to avoid oversubscription.
    with threadpool_limits(limits=max(1, int(_G_BLAS_THREADS))):
        return _process_frame_batch_core(
            xyz_batch_nm,
            box_batch_nm,
            clusters_batch,
            _G_PROT_GROUPS_SEL,
            _G_PROT_LENS,
            _G_WSPEC,
            _G_Q,
            atom_block=_G_ATOM_BLOCK,
            q_block=_G_Q_BLOCK,
        )


@dataclass
class ClusterFFResult:
    q_nm1: np.ndarray
    sizes: np.ndarray
    i_mean: np.ndarray
    i_stderr: np.ndarray
    i_per_protein_mean: np.ndarray
    i_per_protein_stderr: np.ndarray
    counts: np.ndarray


def cluster_form_factors_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, list[FileLike]],
    clusters: dict[str, Any],
    *,
    selection: Union[str, list[list[int]]] = "protein",
    q_nm1: Optional[np.ndarray] = None,
    q_min_nm1: float = 0.2,
    q_max_nm1: float = 20.0,
    n_q: int = 200,
    box_nm: Optional[list[float]] = None,
    weights: str = "unity",
    atom_block: int = 512,
    q_block: int = 64,
    # Parallelism
    parallel: Literal["none", "frames"] = "none",
    n_workers: int = 1,
    frames_per_task: int = 2,
    use_processes: bool = True,
    blas_threads: int = 1,
    max_pending_tasks: Optional[int] = None,
    verbose: bool = False,
) -> ClusterFFResult:
    """
    For each frame: for each cluster, build a contiguous cluster by minimum-image shifts,
    compute I(q) via the Debye sum, normalize by cluster size m.

    Accumulates mean +/- stderr across all cluster *instances* of a given size.

    Parallelization:
        If parallel == "frames" and n_workers > 1, frames are processed in parallel
        (recommended when you have few clusters per frame).
    """
    dcd_list = [dcd_files] if isinstance(dcd_files, (str, Path)) else list(dcd_files)
    clusters_by_frame = clusters.get("clusters_by_frame")
    if clusters_by_frame is None:
        raise ValueError("clusters dict missing 'clusters_by_frame'")

    params = clusters.get("params", {})
    stride = int(params.get("stride", 1))
    chunk = int(params.get("chunk", 200))
    frame_start = int(params.get("frame_start", 0))
    frame_stop = params.get("frame_stop", None)

    tmpl_model, prot_groups_sel, atom_indices_full, extra = _protein_groups_from_selection(
        pdb_file,
        selection,
    )
    n_prot = int(extra["n_proteins"])
    if int(clusters.get("n_proteins", n_prot)) != n_prot:
        raise ValueError("clusters n_proteins does not match selection-derived protein count")

    prot_lens = np.asarray([int(g.size) for g in prot_groups_sel], dtype=np.int64)

    if q_nm1 is None:
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        q = np.linspace(float(q_min_nm1), float(q_max_nm1), int(n_q), dtype=np.float64)
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")

    wspec = _atomic_weights_for_selection(
        tmpl_model,
        atom_indices_full,
        weights=weights,
        q_nm1=q,
    )

    sums: dict[int, np.ndarray] = {}
    sums2: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    def _acc_init(m: int) -> None:
        if m not in sums:
            sums[m] = np.zeros_like(q)
            sums2[m] = np.zeros_like(q)
            counts[m] = 0

    def _merge_partials(psums, psums2, pcounts) -> None:
        for m, v in psums.items():
            mi = int(m)
            _acc_init(mi)
            sums[mi] += np.asarray(v, dtype=np.float64)
        for m, v in psums2.items():
            mi = int(m)
            _acc_init(mi)
            sums2[mi] += np.asarray(v, dtype=np.float64)
        for m, c in pcounts.items():
            mi = int(m)
            _acc_init(mi)
            counts[mi] += int(c)

    fi_global = 0
    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    do_parallel = (str(parallel).lower() == "frames") and int(n_workers) > 1
    if not do_parallel:
        # Sequential streaming
        for dcd in dcd_list:
            if verbose:
                print(f"reading from {dcd}")
            for fi_local, (xyz_sel_nm, box_frame_nm) in enumerate(
                iter_dcd(
                    dcd,
                    tmpl_model,
                    chunk=chunk,
                    stride=stride,
                    atom_indices=atom_indices_full,
                )
            ):
                if fi_local < frame_start:
                    continue
                if frame_stop is not None and fi_local >= int(frame_stop):
                    break

                if fi_global >= len(clusters_by_frame):
                    raise ValueError("clusters_by_frame shorter than trajectory frames")

                b = box_fallback if box_frame_nm is None else _box_lengths_nm(box_frame_nm)
                if b is None:
                    raise ValueError("no unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")

                frame_clusters = clusters_by_frame[fi_global]

                ps, ps2, pc = _process_frame_batch_core(
                    np.asarray(xyz_sel_nm)[None, :, :],
                    np.asarray(b, dtype=np.float64)[None, :],
                    [frame_clusters],
                    prot_groups_sel,
                    prot_lens,
                    wspec,
                    q,
                    atom_block=int(atom_block),
                    q_block=int(q_block),
                )
                _merge_partials(ps, ps2, pc)
                fi_global += 1
    else:
        # Parallel over frames (batching a small number of frames per task)
        fw = max(1, int(n_workers))
        fpt = max(1, int(frames_per_task))
        max_pending = int(max_pending_tasks) if max_pending_tasks is not None else (2 * fw)

        if use_processes:
            executor = ProcessPoolExecutor(
                max_workers=fw,
                initializer=_init_frame_worker,
                initargs=(
                    prot_groups_sel,
                    prot_lens,
                    wspec,
                    q,
                    int(atom_block),
                    int(q_block),
                    int(blas_threads),
                ),
            )
            submit_fn = _process_frame_batch_worker
        else:
            # Thread backend (no pickling overhead, but can be limited by the GIL)
            executor = ThreadPoolExecutor(max_workers=fw)

            def submit_fn(  # type: ignore[misc]
                xyz_batch_nm: np.ndarray,
                box_batch_nm: np.ndarray,
                clusters_batch: list[list[list[int]]],
            ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
                try:
                    from threadpoolctl import threadpool_limits
                except Exception:
                    threadpool_limits = None  # type: ignore[assignment]

                if threadpool_limits is None:
                    return _process_frame_batch_core(
                        xyz_batch_nm,
                        box_batch_nm,
                        clusters_batch,
                        prot_groups_sel,
                        prot_lens,
                        wspec,
                        q,
                        atom_block=int(atom_block),
                        q_block=int(q_block),
                    )
                with threadpool_limits(limits=max(1, int(blas_threads))):
                    return _process_frame_batch_core(
                        xyz_batch_nm,
                        box_batch_nm,
                        clusters_batch,
                        prot_groups_sel,
                        prot_lens,
                        wspec,
                        q,
                        atom_block=int(atom_block),
                        q_block=int(q_block),
                    )

        futures = []
        batch_xyz: list[np.ndarray] = []
        batch_box: list[np.ndarray] = []
        batch_clusters: list[list[list[int]]] = []

        def _submit_current_batch() -> None:
            nonlocal batch_xyz, batch_box, batch_clusters, futures
            if not batch_xyz:
                return
            xyz_b = np.ascontiguousarray(np.stack(batch_xyz, axis=0), dtype=np.float32)
            box_b = np.ascontiguousarray(np.stack(batch_box, axis=0), dtype=np.float64)
            cl_b = batch_clusters
            futures.append(executor.submit(submit_fn, xyz_b, box_b, cl_b))
            batch_xyz = []
            batch_box = []
            batch_clusters = []

        try:
            for dcd in dcd_list:
                if verbose:
                    print(f"reading from {dcd}")
                for fi_local, (xyz_sel_nm, box_frame_nm) in enumerate(
                    iter_dcd(
                        dcd,
                        tmpl_model,
                        chunk=chunk,
                        stride=stride,
                        atom_indices=atom_indices_full,
                    )
                ):
                    if fi_local < frame_start:
                        continue
                    if frame_stop is not None and fi_local >= int(frame_stop):
                        break

                    if fi_global >= len(clusters_by_frame):
                        raise ValueError("clusters_by_frame shorter than trajectory frames")

                    b = box_fallback if box_frame_nm is None else _box_lengths_nm(box_frame_nm)
                    if b is None:
                        raise ValueError("no unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")

                    frame_clusters = clusters_by_frame[fi_global]

                    batch_xyz.append(np.asarray(xyz_sel_nm))
                    batch_box.append(np.asarray(b, dtype=np.float64))
                    batch_clusters.append(frame_clusters)

                    fi_global += 1

                    if len(batch_xyz) >= fpt:
                        _submit_current_batch()

                    # Keep the number of in-flight tasks bounded (limits RAM)
                    if len(futures) >= max_pending:
                        done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                        futures = list(not_done)
                        for fut in done:
                            ps, ps2, pc = fut.result()
                            _merge_partials(ps, ps2, pc)

            # Submit any remaining frames
            _submit_current_batch()

            # Drain remaining tasks
            while futures:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                futures = list(not_done)
                for fut in done:
                    ps, ps2, pc = fut.result()
                    _merge_partials(ps, ps2, pc)
        finally:
            executor.shutdown(wait=True)

    if fi_global != len(clusters_by_frame):
        if fi_global < len(clusters_by_frame):
            raise ValueError("trajectory shorter than clusters_by_frame")
        raise ValueError("trajectory longer than clusters_by_frame")

    sizes = np.array(sorted(counts.keys()), dtype=np.int64)
    i_mean = np.zeros((sizes.size, q.size), dtype=np.float64)
    i_stderr = np.zeros_like(i_mean)

    for k, m in enumerate(sizes.tolist()):
        n = int(counts[m])
        mu = sums[m] / float(n)
        i_mean[k] = mu
        if n >= 2:
            var = (sums2[m] / float(n)) - (mu * mu)
            var = np.maximum(var, 0.0)
            i_stderr[k] = np.sqrt(var) / math.sqrt(float(n))
        else:
            i_stderr[k] = 0.0

    # For convenience: "raw" cluster intensity mean = m * (I/m)
    i_clust_mean = i_mean * sizes[:, None].astype(np.float64)
    i_clust_stderr = i_stderr * sizes[:, None].astype(np.float64)

    return ClusterFFResult(
        q_nm1=q,
        sizes=sizes,
        i_mean=i_clust_mean,
        i_stderr=i_clust_stderr,
        i_per_protein_mean=i_mean,
        i_per_protein_stderr=i_stderr,
        counts=np.array([counts[int(m)] for m in sizes.tolist()], dtype=np.int64),
    )


@dataclass
class FullDebyeResult:
    q_nm1: np.ndarray
    i_mean: np.ndarray
    i_stderr: np.ndarray
    n_frames: int


def full_debye_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, list[FileLike]],
    *,
    selection: Union[str, list[list[int]]] = "protein",
    q_nm1: Optional[np.ndarray] = None,
    q_min_nm1: float = 0.2,
    q_max_nm1: float = 20.0,
    n_q: int = 200,
    box_nm: Optional[list[float]] = None,
    weights: str = "unity",
    atom_block: int = 512,
    q_block: int = 64,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    parallel: Literal["none", "frames"] = "none",
    n_workers: int = 1,
    use_processes: bool = True,
    blas_threads: int = 1,
    max_pending_tasks: Optional[int] = None,
    verbose: bool = False,
) -> FullDebyeResult:
    """
    Full-system Debye intensity from selected atoms (includes inter-cluster correlations).

    Computes per-frame I(q) using minimum-image distances in an orthorhombic box, then
    returns mean +/- stderr across frames.

    Notes
    -----
    - For parallel="frames" + use_processes=True, this requires module-scope worker
      functions (_init_full_debye_worker, _full_debye_one_frame_worker) to avoid
      pickling closures.
    - blas_threads is accepted for API symmetry; this function does not call BLAS-heavy
      ops by default, so it is currently unused.
    """
    del blas_threads

    dcd_list = [dcd_files] if isinstance(dcd_files, (str, Path)) else list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    if isinstance(selection, str):
        groups_full = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_full = [[int(i) for i in g] for g in selection]

    groups_full = [g for g in groups_full if g]
    if not groups_full:
        raise ValueError("selection produced no atoms")

    atom_set: set[int] = set()
    for g in groups_full:
        atom_set.update(int(i) for i in g)
    atom_indices_full = sorted(atom_set)

    # Build q grid first (needed for weights="xray")
    if q_nm1 is None:
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        q = np.linspace(float(q_min_nm1), float(q_max_nm1), int(n_q), dtype=np.float64)
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")

    wspec = _atomic_weights_for_selection(
        tmpl_model,
        atom_indices_full,
        weights=weights,
        q_nm1=q,
    )

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    def _frame_iter():
        for dcd in dcd_list:
            if verbose:
                print(f"reading from {dcd}")
            for fi, (xyz_sel_nm, box_frame_nm) in enumerate(
                iter_dcd(
                    dcd,
                    tmpl_model,
                    chunk=int(chunk),
                    stride=int(stride),
                    atom_indices=atom_indices_full,
                )
            ):
                if fi < int(frame_start):
                    continue
                if frame_stop is not None and fi >= int(frame_stop):
                    break

                if box_frame_nm is None:
                    if box_fallback is None:
                        raise ValueError("DCD has no unit cell; pass box_nm=(Lx,Ly,Lz) in nm")
                    b = box_fallback
                else:
                    b = _box_lengths_nm(box_frame_nm)
                yield np.asarray(xyz_sel_nm, dtype=np.float64), np.asarray(b, dtype=np.float64)

    def _one_frame(xyz_nm: np.ndarray, b_nm: np.ndarray) -> np.ndarray:
        xyz_wr = _wrap_nm(xyz_nm, b_nm)
        if wspec.mode == "scalar":
            if wspec.w_sel is None:
                raise ValueError("wspec.w_sel is None for scalar mode")
            w = np.asarray(wspec.w_sel, dtype=np.float64)
            return debye_intensity_nm_pbc(
                xyz_wr,
                q,
                w,
                b_nm,
                atom_block=int(atom_block),
                q_block=int(q_block),
            )

        if wspec.el_id_sel is None or wspec.f_el_q is None:
            raise ValueError("wspec missing el_id_sel/f_el_q for xray mode")
        el = np.asarray(wspec.el_id_sel, dtype=np.int32)
        ftab = np.asarray(wspec.f_el_q, dtype=np.float64)
        return debye_intensity_nm_xray_pbc(
            xyz_wr,
            q,
            el,
            ftab,
            b_nm,
            atom_block=int(atom_block),
            q_block=int(q_block),
        )

    do_parallel = (str(parallel).lower() == "frames") and int(n_workers) > 1
    frames: list[np.ndarray] = []

    if not do_parallel:
        for xyz_nm, b_nm in _frame_iter():
            frames.append(_one_frame(xyz_nm, b_nm))
    else:
        fw = max(1, int(n_workers))
        max_pending = int(max_pending_tasks) if max_pending_tasks is not None else (2 * fw)

        if not use_processes:
            ex = ThreadPoolExecutor(max_workers=fw)
            try:
                futures = []
                for xyz_nm, b_nm in _frame_iter():
                    futures.append(ex.submit(_one_frame, xyz_nm, b_nm))
                    if len(futures) >= max_pending:
                        done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                        futures = list(not_done)
                        for fut in done:
                            frames.append(fut.result())
                while futures:
                    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                    futures = list(not_done)
                    for fut in done:
                        frames.append(fut.result())
            finally:
                ex.shutdown(wait=True)
        else:
            ex = ProcessPoolExecutor(
                max_workers=fw,
                initializer=_init_full_debye_worker,
                initargs=(q, wspec, int(atom_block), int(q_block)),
            )
            try:
                futures = []
                for xyz_nm, b_nm in _frame_iter():
                    futures.append(ex.submit(_full_debye_one_frame_worker, xyz_nm, b_nm))
                    if len(futures) >= max_pending:
                        done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                        futures = list(not_done)
                        for fut in done:
                            frames.append(fut.result())
                while futures:
                    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                    futures = list(not_done)
                    for fut in done:
                        frames.append(fut.result())
            finally:
                ex.shutdown(wait=True)

    if not frames:
        raise ValueError("no frames selected")

    arr = np.stack(frames, axis=0)
    i_mean = np.mean(arr, axis=0)
    n_frames = int(arr.shape[0])
    if n_frames >= 2:
        i_stderr = np.std(arr, axis=0, ddof=1) / math.sqrt(float(n_frames))
    else:
        i_stderr = np.zeros_like(i_mean)

    return FullDebyeResult(q_nm1=q, i_mean=i_mean, i_stderr=i_stderr, n_frames=n_frames)


# --- Reciprocal full intensity  --------------------------------


def _q_edges_from_centers(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size < 1:
        raise ValueError("q_nm1 must be non-empty")
    if q.size == 1:
        left = max(0.0, float(q[0]) - 1e-6)
        return np.array([left, float(q[0]) + 1e-6], dtype=np.float64)
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q_nm1 must be strictly increasing")

    mid = 0.5 * (q[:-1] + q[1:])
    left = q[0] - 0.5 * (q[1] - q[0])
    right = q[-1] + 0.5 * (q[-1] - q[-2])
    edges = np.concatenate(([left], mid, [right])).astype(np.float64)
    edges[0] = max(0.0, float(edges[0]))
    return edges


def _init_full_recip_worker(
    q_nm1: np.ndarray,
    wspec: _WeightSpec,
    qvec_block: int,
    blas_threads: int = 1,
) -> None:
    import os

    t = str(max(1, int(blas_threads)))
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t

    global _G_RECIP_Q, _G_RECIP_QEDGES, _G_RECIP_WSPEC, _G_RECIP_QVEC_BLOCK, _G_BLAS_THREADS
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    _G_RECIP_Q = q
    _G_RECIP_QEDGES = _q_edges_from_centers(q)
    _G_RECIP_WSPEC = wspec
    _G_RECIP_QVEC_BLOCK = int(qvec_block)
    _G_BLAS_THREADS = max(1, int(blas_threads))


def _full_recip_one_frame_worker(
    xyz_sel_nm: np.ndarray,
    box_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if _G_RECIP_Q is None or _G_RECIP_QEDGES is None or _G_RECIP_WSPEC is None:
        raise RuntimeError("reciprocal worker globals not initialized")

    try:
        from threadpoolctl import threadpool_limits
    except Exception:  # pragma: no cover
        threadpool_limits = None  # type: ignore[assignment]

    if threadpool_limits is None:
        return _full_recip_one_frame_worker_core(xyz_sel_nm, box_nm)

    with threadpool_limits(limits=max(1, int(_G_BLAS_THREADS))):
        return _full_recip_one_frame_worker_core(xyz_sel_nm, box_nm)


def _full_recip_one_frame_worker_core(
    xyz_sel_nm: np.ndarray,
    box_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns per-bin (sum |rho(qvec)|^2) and per-bin qvec counts for ONE frame.
    Caller can form per-bin averages by sum/count.
    """
    if _G_RECIP_Q is None or _G_RECIP_QEDGES is None or _G_RECIP_WSPEC is None:
        raise RuntimeError("reciprocal worker globals not initialized")

    q = _G_RECIP_Q
    q_edges = _G_RECIP_QEDGES
    wspec = _G_RECIP_WSPEC
    qb = max(64, int(_G_RECIP_QVEC_BLOCK))

    b = _box_lengths_nm(np.asarray(box_nm, dtype=np.float64).reshape(3))
    x = _wrap_nm(np.asarray(xyz_sel_nm, dtype=np.float64), b)

    qvecs, qmag = _q_vectors_cubic_or_ortho(b, float(q_edges[-1]))
    if qvecs.size == 0:
        return np.zeros_like(q), np.zeros_like(q, dtype=np.int64)

    m = (qmag >= float(q_edges[0])) & (qmag < float(q_edges[-1]))
    if not np.any(m):
        return np.zeros_like(q), np.zeros_like(q, dtype=np.int64)
    qvecs = qvecs[m]
    qmag = qmag[m]

    bin_idx = np.searchsorted(q_edges, qmag, side="right") - 1
    ok = (bin_idx >= 0) & (bin_idx < q.size)
    if not np.any(ok):
        return np.zeros_like(q), np.zeros_like(q, dtype=np.int64)
    qvecs = qvecs[ok]
    bin_idx = bin_idx[ok].astype(np.int64, copy=False)

    acc = np.zeros_like(q, dtype=np.float64)
    cnt = np.zeros_like(q, dtype=np.int64)

    if wspec.mode == "scalar":
        if wspec.w_sel is None:
            raise ValueError("wspec.w_sel is None for scalar mode")
        w = np.asarray(wspec.w_sel, dtype=np.float64).reshape(-1)

        for k0 in range(0, qvecs.shape[0], qb):
            k1 = min(qvecs.shape[0], k0 + qb)
            qv = qvecs[k0:k1]
            bidx = bin_idx[k0:k1]

            ph = qv @ x.T
            rho_re = np.cos(ph) @ w
            rho_im = np.sin(ph) @ w
            i_qv = rho_re * rho_re + rho_im * rho_im

            np.add.at(acc, bidx, i_qv)
            np.add.at(cnt, bidx, 1)

        return acc, cnt

    if wspec.el_id_sel is None or wspec.f_el_q is None:
        raise ValueError("wspec missing el_id_sel/f_el_q for xray mode")

    el = np.asarray(wspec.el_id_sel, dtype=np.int32).reshape(-1)
    ftab = np.asarray(wspec.f_el_q, dtype=np.float64)

    # Group q-vectors by bin so weights are constant within that group.
    order = np.argsort(bin_idx, kind="mergesort")
    qvecs = qvecs[order]
    bin_idx = bin_idx[order]

    start = 0
    while start < bin_idx.size:
        bi = int(bin_idx[start])
        end = start + 1
        while end < bin_idx.size and int(bin_idx[end]) == bi:
            end += 1

        w = ftab[el, bi].astype(np.float64, copy=False)
        qg = qvecs[start:end]

        # Block within this bin-group.
        for k0 in range(0, qg.shape[0], qb):
            k1 = min(qg.shape[0], k0 + qb)
            qv = qg[k0:k1]
            ph = qv @ x.T
            rho_re = np.cos(ph) @ w
            rho_im = np.sin(ph) @ w
            i_qv = rho_re * rho_re + rho_im * rho_im
            acc[bi] += float(np.sum(i_qv))
            cnt[bi] += int(i_qv.size)

        start = end

    return acc, cnt


def full_debye_reciprocal_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, list[FileLike]],
    *,
    selection: Union[str, list[list[int]]] = "protein",
    q_nm1: Optional[np.ndarray] = None,
    q_min_nm1: Optional[float] = 0.2,
    q_max_nm1: float = 20.0,
    n_q: int = 200,
    box_nm: Optional[list[float]] = None,
    weights: str = "unity",
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    qvec_block: int = 512,
    min_qvecs_per_bin: int = 1,
    parallel: Literal["none", "frames"] = "none",
    n_workers: int = 1,
    use_processes: bool = True,
    max_pending_tasks: Optional[int] = None,
    blas_threads: int = 1,
    verbose: bool = False,
) -> FullDebyeResult:
    """
    Full-system periodic intensity using reciprocal-space estimator, frame-parallel.

    NPT note:
      Box fluctuates, so q-vectors per bin vary by frame. This function averages
      per-frame bin means (equal frame weighting). Bins with <min_qvecs_per_bin
      q-vectors in a frame are ignored for that frame.
    """
    dcd_list = [dcd_files] if isinstance(dcd_files, (str, Path)) else list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")
    if int(min_qvecs_per_bin) < 1:
        raise ValueError("min_qvecs_per_bin must be >= 1")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    if isinstance(selection, str):
        groups_full = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_full = [[int(i) for i in g] for g in selection]
    groups_full = [g for g in groups_full if g]
    if not groups_full:
        raise ValueError("selection produced no atoms")

    atom_set: set[int] = set()
    for g in groups_full:
        atom_set.update(int(i) for i in g)
    atom_indices_full = sorted(atom_set)

    if q_nm1 is None:
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        if q_min_nm1 is None:
            b0 = _peek_first_box_nm(
                dcd_list[0],
                tmpl_model,
                atom_indices_full,
                int(stride),
                box_nm=box_nm,
            )
            q0 = 2.0 * math.pi / float(np.min(b0))
            qmin = float(q0)
        else:
            qmin = float(q_min_nm1)
        qmax = float(q_max_nm1)
        if qmax <= qmin:
            raise ValueError("q_max_nm1 must be > q_min_nm1")
        q = np.linspace(qmin, qmax, int(n_q), dtype=np.float64)
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")

    wspec = _atomic_weights_for_selection(
        tmpl_model,
        atom_indices_full,
        weights=weights,
        q_nm1=q,
    )

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    def _frame_iter():
        for dcd in dcd_list:
            if verbose:
                print(f"reading from {dcd}")
            for fi, (xyz_sel_nm, box_frame_nm) in enumerate(
                iter_dcd(
                    dcd,
                    tmpl_model,
                    chunk=int(chunk),
                    stride=int(stride),
                    atom_indices=atom_indices_full,
                )
            ):
                if fi < int(frame_start):
                    continue
                if frame_stop is not None and fi >= int(frame_stop):
                    break

                if box_frame_nm is None:
                    if box_fallback is None:
                        raise ValueError("DCD has no unit cell; pass box_nm=(Lx,Ly,Lz) in nm")
                    b = box_fallback
                else:
                    b = _box_lengths_nm(box_frame_nm)

                yield np.asarray(xyz_sel_nm, dtype=np.float64), np.asarray(b, dtype=np.float64)

    do_parallel = (str(parallel).lower() == "frames") and int(n_workers) > 1
    fw = max(1, int(n_workers))
    max_pending = int(max_pending_tasks) if max_pending_tasks is not None else (2 * fw)

    frame_means: list[np.ndarray] = []
    frame_has: list[np.ndarray] = []

    def _consume_result(acc: np.ndarray, cnt: np.ndarray) -> None:
        cnt_i = np.asarray(cnt, dtype=np.int64)
        acc_f = np.asarray(acc, dtype=np.float64)

        has = cnt_i >= int(min_qvecs_per_bin)
        mu = np.zeros_like(acc_f)
        mu[has] = acc_f[has] / cnt_i[has].astype(np.float64)

        frame_means.append(mu)
        frame_has.append(has)

    # IMPORTANT: initialize worker globals for the serial path too.
    if not do_parallel:
        _init_full_recip_worker(q, wspec, int(qvec_block), int(blas_threads))
        for xyz_nm, b_nm in _frame_iter():
            acc, cnt = _full_recip_one_frame_worker(xyz_nm, b_nm)
            _consume_result(acc, cnt)
    else:
        if not use_processes:
            ex = ThreadPoolExecutor(max_workers=fw)
            _init_full_recip_worker(q, wspec, int(qvec_block), int(blas_threads))
            submit_fn = _full_recip_one_frame_worker
        else:
            ex = ProcessPoolExecutor(
                max_workers=fw,
                initializer=_init_full_recip_worker,
                initargs=(q, wspec, int(qvec_block), int(blas_threads)),
            )
            submit_fn = _full_recip_one_frame_worker

        try:
            futures = []
            for xyz_nm, b_nm in _frame_iter():
                futures.append(ex.submit(submit_fn, xyz_nm, b_nm))
                if len(futures) >= max_pending:
                    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                    futures = list(not_done)
                    for fut in done:
                        acc, cnt = fut.result()
                        _consume_result(acc, cnt)

            while futures:
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                futures = list(not_done)
                for fut in done:
                    acc, cnt = fut.result()
                    _consume_result(acc, cnt)
        finally:
            ex.shutdown(wait=True)

    if not frame_means:
        raise ValueError("no frames selected")

    arr = np.stack(frame_means, axis=0)  # (n_frames, n_q)
    has = np.stack(frame_has, axis=0)  # (n_frames, n_q)

    denom = np.sum(has.astype(np.int64), axis=0)
    denom = np.maximum(denom, 1)
    i_mean = np.sum(arr * has.astype(np.float64), axis=0) / denom.astype(np.float64)

    i_stderr = np.zeros_like(i_mean)
    for j in range(i_mean.size):
        mj = has[:, j]
        nj = int(np.sum(mj))
        if nj >= 2:
            v = arr[mj, j]
            i_stderr[j] = float(np.std(v, ddof=1)) / math.sqrt(float(nj))

    return FullDebyeResult(
        q_nm1=q,
        i_mean=i_mean,
        i_stderr=i_stderr,
        n_frames=int(arr.shape[0]),
    )


def _sq_single_dcd(
    dcd_file: FileLike,
    template_model: Any,
    *,
    atom_indices: Sequence[int],
    groups: Sequence[np.ndarray],
    masses: Optional[np.ndarray],
    center: str,
    unwrap: bool,
    q_nm1: np.ndarray,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_nm: Optional[Sequence[float]],
) -> tuple[np.ndarray, int]:
    n_groups = int(len(groups))
    if n_groups < 2:
        raise ValueError("need >=2 groups to compute S(q)")

    acc = np.zeros_like(q_nm1, dtype=np.float64)
    n_frames = 0

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    for fi, (xyz_nm, box_frame_nm) in enumerate(
        iter_dcd(
            dcd_file,
            template_model,
            chunk=int(chunk),
            stride=int(stride),
            atom_indices=atom_indices,
        )
    ):
        if fi < int(frame_start):
            continue
        if frame_stop is not None and fi >= int(frame_stop):
            break

        if box_frame_nm is None:
            if box_fallback is None:
                raise ValueError(
                    "DCD does not include unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm"
                )
            b = box_fallback
        else:
            b = np.asarray(box_frame_nm, dtype=np.float64).reshape(3)

        if np.any(b <= 0.0):
            raise ValueError("box lengths must be positive")

        centers = group_centers_nm(
            xyz_nm,
            groups,
            masses=masses,
            box_nm=b,
            center=center,
            unwrap=bool(unwrap),
            wrap=True,
        )
        r = _pair_distances_nm(centers, b)
        if r.size == 0:
            continue

        qr = np.outer(r, q_nm1)
        sum_sinc = np.sum(_sinc(qr), axis=0)
        s_frame = 1.0 + (2.0 / float(n_groups)) * sum_sinc
        acc += s_frame
        n_frames += 1

    if n_frames <= 0:
        raise ValueError("no frames selected for S(q) computation")
    return acc / float(n_frames), int(n_frames)


def _q_vectors_cubic_or_ortho(box_nm: np.ndarray, q_max_nm1: float):
    Lx, Ly, Lz = [float(x) for x in box_nm]
    # integer bounds
    nx_max = int(math.floor(q_max_nm1 * Lx / (2.0 * math.pi)))
    ny_max = int(math.floor(q_max_nm1 * Ly / (2.0 * math.pi)))
    nz_max = int(math.floor(q_max_nm1 * Lz / (2.0 * math.pi)))

    qvecs = []
    qmag = []
    for nx in range(-nx_max, nx_max + 1):
        for ny in range(-ny_max, ny_max + 1):
            for nz in range(-nz_max, nz_max + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                qx = 2.0 * math.pi * nx / Lx
                qy = 2.0 * math.pi * ny / Ly
                qz = 2.0 * math.pi * nz / Lz
                qm = math.sqrt(qx * qx + qy * qy + qz * qz)
                if qm <= q_max_nm1:
                    qvecs.append((qx, qy, qz))
                    qmag.append(qm)
    qvecs = np.asarray(qvecs, dtype=np.float64)  # (nq,3)
    qmag = np.asarray(qmag, dtype=np.float64)  # (nq,)
    return qvecs, qmag


def structure_factor_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[Sequence[int]]] = "protein",
    center: str = "cog",
    unwrap: bool = True,
    q_nm1: Optional[np.ndarray] = None,
    q_min_nm1: Optional[float] = None,
    q_max_nm1: float = 20.0,
    n_q: int = 200,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Isotropically averaged S(q) from solute-group centers (Debye formula).

    Uncertainties
    -------------
    - If multiple DCDs are provided: stderr across DCDs (treating each as a replicate).
    - If only one DCD: uncertainties are returned as 0.
    """
    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    center_mode = str(center).strip().lower()
    if center_mode not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    if isinstance(selection, str):
        groups_global = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_global = [[int(i) for i in g] for g in selection]

    groups_global = [g for g in groups_global if g]
    if len(groups_global) < 2:
        raise ValueError("selection must yield >=2 non-empty groups")

    atom_set: set[int] = set()
    for g in groups_global:
        atom_set.update(int(i) for i in g)
    atom_indices = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices)}
    groups = [np.asarray([idx_map[int(i)] for i in g], dtype=np.int32) for g in groups_global]

    masses_sel = None
    if center_mode == "com":
        masses_all = atom_masses(tmpl_model)
        masses_sel = np.asarray(masses_all[atom_indices], dtype=np.float64)

    if q_nm1 is None:
        b0 = _peek_first_box_nm(
            dcd_list[0],
            tmpl_model,
            atom_indices,
            int(stride),
            box_nm=box_nm,
        )
        q0 = 2.0 * math.pi / float(np.min(b0))
        qmin = float(q0 if q_min_nm1 is None else q_min_nm1)
        qmax = float(q_max_nm1)
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        if qmax <= qmin:
            raise ValueError("q_max_nm1 must be > q_min_nm1")
        q = np.linspace(qmin, qmax, int(n_q), dtype=np.float64)
        q_info = {"q_box_min_nm1": float(q0)}
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")
        q_info = {}

    sq_blocks: list[np.ndarray] = []
    frames_per_block: list[int] = []

    for dcd in dcd_list:
        sq, n_frames = _sq_single_dcd(
            dcd,
            tmpl_model,
            atom_indices=atom_indices,
            groups=groups,
            masses=masses_sel,
            center=center_mode,
            unwrap=bool(unwrap),
            q_nm1=q,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=int(frame_start),
            frame_stop=frame_stop,
            box_nm=box_nm,
        )
        sq_blocks.append(sq)
        frames_per_block.append(int(n_frames))

    sq_arr = np.stack(sq_blocks, axis=0)
    sq_mean = np.mean(sq_arr, axis=0)

    n_blocks = int(sq_arr.shape[0])
    if n_blocks < 2:
        sq_err = np.zeros_like(sq_mean)
    else:
        sq_err = np.std(sq_arr, axis=0, ddof=1) / math.sqrt(float(n_blocks))

    out: dict[str, Any] = {
        "q_nm1": q,
        "q": q,  # alias
        "sq": sq_mean,
        "sq_err": sq_err,
        "sq_stderr": sq_err,  # alias
        "n_blocks": n_blocks,
        "frames_per_block": np.asarray(frames_per_block, dtype=np.int64),
        "selection": selection,
        "center": center_mode,
        "unwrap": bool(unwrap),
        "stride": int(stride),
    }
    out.update(q_info)
    return out


def structure_factor_from_dcd_npt_debye(
    dcd_file: Any,
    template: Any,
    *,
    group_spec: str = "protein",
    groups: Optional[Sequence[Sequence[int]]] = None,
    center: str = "com",
    unwrap: bool = True,
    q_min: float = 0.0,
    q_max: float = 3.0,
    dq: float = 0.05,
    stride: int = 1,
    chunk: int = 200,
    box_nm: Any = None,
    pair_block: int = 20000,
) -> dict[str, Any]:
    """Back-compat wrapper (older name/signature).

    Parameters follow the legacy helper that computed S(q) in NPT using the Debye
    expression. Internally delegates to structure_factor_from_dcd().
    """
    del pair_block

    q = np.arange(float(q_min), float(q_max) + 0.5 * float(dq), float(dq), dtype=np.float64)
    sel: Union[str, Sequence[Sequence[int]]] = group_spec if groups is None else groups

    return structure_factor_from_dcd(
        pdb_file=template,
        dcd_files=dcd_file,
        selection=sel,
        center=center,
        unwrap=bool(unwrap),
        q_nm1=q,
        stride=int(stride),
        chunk=int(chunk),
        frame_start=0,
        frame_stop=None,
        box_nm=box_nm,
    )


def _sq_single_dcd_recip(
    dcd_file: FileLike,
    template_model: Any,
    *,
    atom_indices: Sequence[int],
    groups: Sequence[np.ndarray],
    masses: Optional[np.ndarray],
    center: str,
    unwrap: bool,
    q_nm1: np.ndarray,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_nm: Optional[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Reciprocal-space S(q) estimator for a periodic box:

        S(q) = < |sum_j exp(i q·r_j)|^2 / N >  (binned isotropically by |q|)

    Parameters
    ----------
    q_nm1
        1D array of *bin centers* in nm^-1. Binning is done using midpoints between
        centers as bin edges; values outside the edge range are ignored.

    Returns
    -------
    sq_nm1 : (n_q,) float
        S(q) at the provided bin centers.
    n_qvec_per_bin : (n_q,) int
        Number of reciprocal vectors accumulated into each bin (summed across frames).
        Useful for diagnosing noisy low-q bins.
    n_frames : int
        Number of frames used.
    """
    n_groups = int(len(groups))
    if n_groups < 2:
        raise ValueError("need >=2 groups to compute S(q)")

    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    if q.size < 1:
        raise ValueError("q_nm1 must be non-empty")
    if np.any(~np.isfinite(q)):
        raise ValueError("q_nm1 contains non-finite values")

    # Enforce strictly increasing bin centers (required for robust edge construction).
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q_nm1 must be strictly increasing for reciprocal binning")

    # Construct bin edges from centers:
    # edges[0] = q0 - dq0/2 ; edges[i] = (q[i-1]+q[i])/2 ; edges[-1] = q[-1] + dqlast/2
    mid = 0.5 * (q[:-1] + q[1:])
    left = q[0] - 0.5 * (q[1] - q[0])
    right = q[-1] + 0.5 * (q[-1] - q[-2]) if q.size > 1 else q[-1] + 1e-6
    q_edges = np.concatenate(([left], mid, [right])).astype(np.float64)

    # Guard against non-positive left edge due to user q starting very near 0
    # (not mathematically illegal, but it makes binning more intuitive).
    if q_edges[0] < 0.0:
        q_edges[0] = 0.0

    acc = np.zeros_like(q, dtype=np.float64)
    cnt = np.zeros_like(q, dtype=np.int64)  # counts of q-vectors accumulated (across frames)
    n_frames = 0

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    for fi, (xyz_nm, box_frame_nm) in enumerate(
        iter_dcd(
            dcd_file,
            template_model,
            chunk=int(chunk),
            stride=int(stride),
            atom_indices=atom_indices,
        )
    ):
        if fi < int(frame_start):
            continue
        if frame_stop is not None and fi >= int(frame_stop):
            break

        if box_frame_nm is None:
            if box_fallback is None:
                raise ValueError("DCD lacks unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")
            b = box_fallback
        else:
            b = np.asarray(box_frame_nm, dtype=np.float64).reshape(3)

        if np.any(b <= 0.0):
            raise ValueError("box lengths must be positive")

        # One position per group (protein) in [0, L)
        centers = group_centers_nm(
            xyz_nm,
            groups,
            masses=masses,
            box_nm=b,
            center=center,
            unwrap=bool(unwrap),
            wrap=True,
        )  # (n_groups, 3)

        # Build allowed reciprocal vectors for this frame's box (NPT-safe)
        qvecs, qmag = _q_vectors_cubic_or_ortho(b, float(q_edges[-1]))  # include up to max edge
        if qvecs.size == 0:
            n_frames += 1
            continue

        # Keep only vectors within the bin edge range
        m = (qmag >= float(q_edges[0])) & (qmag < float(q_edges[-1]))
        if not np.any(m):
            n_frames += 1
            continue
        qvecs = qvecs[m]
        qmag = qmag[m]

        # rho(q) = sum_j exp(i q·r_j)
        phase = qvecs @ centers.T  # (nq, n_groups)
        rho_re = np.sum(np.cos(phase), axis=1)
        rho_im = np.sum(np.sin(phase), axis=1)
        sq_qvec = (rho_re * rho_re + rho_im * rho_im) / float(n_groups)  # (nq,)

        # Bin by |q| to the provided q centers
        bin_idx = np.searchsorted(q_edges, qmag, side="right") - 1
        ok = (bin_idx >= 0) & (bin_idx < acc.size)
        np.add.at(acc, bin_idx[ok], sq_qvec[ok])
        np.add.at(cnt, bin_idx[ok], 1)

        n_frames += 1

    if n_frames <= 0:
        raise ValueError("no frames selected for S(q) computation")

    # Average over all q-vectors accumulated (across all frames) per bin.
    # This is equivalent to averaging per-frame then across frames only if each frame
    # has the same set of q-vectors; in NPT the set can change slightly, and this
    # weighted average is typically what you want.
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = acc / np.maximum(cnt, 1)

    return sq, cnt, int(n_frames)


def structure_factor_from_dcd_reciprocal(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[Sequence[int]]] = "protein",
    center: str = "cog",
    unwrap: bool = True,
    q_nm1: Optional[np.ndarray] = None,
    q_min_nm1: Optional[float] = None,
    q_max_nm1: float = 20.0,
    n_q: int = 200,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """
    Isotropically averaged S(q) from solute-group centers using the *reciprocal-space*
    estimator appropriate for periodic boundary conditions.

    Callable like structure_factor_from_dcd().

    Uncertainties
    -------------
    - If multiple DCDs are provided: stderr across DCDs (treating each as a replicate).
    - If only one DCD: uncertainties are returned as 0.

    Notes
    -----
    - q_nm1 is interpreted as desired *bin centers* (must be strictly increasing).
    - If q_nm1 is None: a linear grid is constructed like structure_factor_from_dcd().
    """
    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    center_mode = str(center).strip().lower()
    if center_mode not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    if isinstance(selection, str):
        groups_global = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_global = [[int(i) for i in g] for g in selection]

    groups_global = [g for g in groups_global if g]
    if len(groups_global) < 2:
        raise ValueError("selection must yield >=2 non-empty groups")

    atom_set: set[int] = set()
    for g in groups_global:
        atom_set.update(int(i) for i in g)
    atom_indices = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices)}
    groups = [np.asarray([idx_map[int(i)] for i in g], dtype=np.int32) for g in groups_global]

    masses_sel = None
    if center_mode == "com":
        masses_all = atom_masses(tmpl_model)
        masses_sel = np.asarray(masses_all[atom_indices], dtype=np.float64)

    # Build the requested q grid exactly like structure_factor_from_dcd()
    if q_nm1 is None:
        b0 = _peek_first_box_nm(
            dcd_list[0],
            tmpl_model,
            atom_indices,
            int(stride),
            box_nm=box_nm,
        )
        q0 = 2.0 * math.pi / float(np.min(b0))
        qmin = float(q0 if q_min_nm1 is None else q_min_nm1)
        qmax = float(q_max_nm1)
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        if qmax <= qmin:
            raise ValueError("q_max_nm1 must be > q_min_nm1")
        q = np.linspace(qmin, qmax, int(n_q), dtype=np.float64)
        q_info = {"q_box_min_nm1": float(q0)}
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")
        q_info = {}

    # Replicate-by-replicate (DCD-by-DCD)
    sq_blocks: list[np.ndarray] = []
    cnt_blocks: list[np.ndarray] = []
    frames_per_block: list[int] = []

    for dcd in dcd_list:
        sq, cnt, n_frames = _sq_single_dcd_recip(
            dcd,
            tmpl_model,
            atom_indices=atom_indices,
            groups=groups,
            masses=masses_sel,
            center=center_mode,
            unwrap=bool(unwrap),
            q_nm1=q,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=int(frame_start),
            frame_stop=frame_stop,
            box_nm=box_nm,
        )
        sq_blocks.append(sq)
        cnt_blocks.append(cnt)
        frames_per_block.append(int(n_frames))

    sq_arr = np.stack(sq_blocks, axis=0)
    sq_mean = np.mean(sq_arr, axis=0)

    n_blocks = int(sq_arr.shape[0])
    if n_blocks < 2:
        sq_err = np.zeros_like(sq_mean)
    else:
        sq_err = np.std(sq_arr, axis=0, ddof=1) / math.sqrt(float(n_blocks))

    # Diagnostics: average number of q-vectors per bin (across replicates).
    cnt_mean = np.mean(np.stack(cnt_blocks, axis=0).astype(np.float64), axis=0)

    out: dict[str, Any] = {
        "q_nm1": q,
        "q": q,  # alias
        "sq": sq_mean,
        "sq_err": sq_err,
        "sq_stderr": sq_err,  # alias
        "n_blocks": n_blocks,
        "frames_per_block": np.asarray(frames_per_block, dtype=np.int64),
        "selection": selection,
        "center": center_mode,
        "unwrap": bool(unwrap),
        "stride": int(stride),
        "qvecs_per_bin_mean": cnt_mean,
    }
    out.update(q_info)
    return out


def shell_aligned_q_grid(
    l_nm: float,
    *,
    q_shell_max_nm1: float = 2.0,
    q_max_nm1: float = 6.0,
    n_linear: int = 25,
    q_min_nm1: float | None = None,
    include_m1: bool = True,
    dedup_tol: float = 1e-6,
) -> np.ndarray:
    """
    Build a q grid for cubic boxes:
      - shell-aligned |q| values up to q_shell_max_nm1 using q = q0*sqrt(m)
        where q0 = 2*pi/L and m = nx^2+ny^2+nz^2.
      - then a linear grid from max(q_shell_max, q_switch) to q_max_nm1.

    Parameters
    ----------
    l_nm:
        Average cubic box length (nm).
    q_shell_max_nm1:
        Include shell-aligned q values with q <= this cutoff (nm^-1).
    q_max_nm1:
        Maximum q for the final grid (nm^-1).
    n_linear:
        Number of linear points from q_switch..q_max_nm1 (inclusive).
    q_min_nm1:
        Optional hard minimum q. If set, drop shell points below it.
    include_m1:
        If False, start shells from m=2 (skips the 6-vector lowest shell).
    dedup_tol:
        Tolerance for merging near-duplicate q values when concatenating.

    Returns
    -------
    q_nm1 : (n,) float
        Strictly increasing q centers.
    """
    if l_nm <= 0:
        raise ValueError("l_nm must be > 0")
    if q_shell_max_nm1 <= 0 or q_max_nm1 <= 0:
        raise ValueError("q_shell_max_nm1 and q_max_nm1 must be > 0")
    if q_max_nm1 <= q_shell_max_nm1:
        raise ValueError("q_max_nm1 must be > q_shell_max_nm1")
    if n_linear < 2:
        raise ValueError("n_linear must be >= 2")

    q0 = 2.0 * np.pi / float(l_nm)

    m_min = 1 if include_m1 else 2
    if q_shell_max_nm1 < q0 * np.sqrt(m_min):
        shell_q = np.array([], dtype=float)
    else:
        m_max = int(np.floor((float(q_shell_max_nm1) / q0) ** 2))
        m_max = max(m_max, m_min)

        m_vals = np.arange(m_min, m_max + 1, dtype=int)
        shell_q = q0 * np.sqrt(m_vals.astype(float))

        if q_min_nm1 is not None:
            shell_q = shell_q[shell_q >= float(q_min_nm1) - dedup_tol]

        shell_q = shell_q[shell_q <= float(q_shell_max_nm1) + dedup_tol]

    # Decide where linear spacing should start
    q_switch = float(q_shell_max_nm1)
    if shell_q.size > 0:
        q_switch = max(q_switch, float(shell_q[-1]))

    # Build linear part (inclusive endpoints)
    q_lin = np.linspace(q_switch, float(q_max_nm1), int(n_linear), dtype=float)

    # Merge + deduplicate
    q_all = np.concatenate([shell_q, q_lin])
    q_all.sort()

    if q_all.size == 0:
        raise ValueError("empty q grid")

    keep = [0]
    for i in range(1, q_all.size):
        if q_all[i] - q_all[keep[-1]] > float(dedup_tol):
            keep.append(i)
    q_out = q_all[np.array(keep, dtype=int)]

    # Ensure strict increasing
    if np.any(np.diff(q_out) <= 0):
        raise RuntimeError("q grid is not strictly increasing (dedup_tol too small?)")

    return q_out

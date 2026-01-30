from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .molecule_data import Structure


@dataclass(frozen=True)
class SegidPairDistanceResult:
    frame_indices: np.ndarray  # (n_frames,)
    reference_segids: tuple[str, ...]
    neighbor_tiles: list[tuple[str, ...]]
    segid_pairs: list[tuple[str, str]]  # (ref_segid, neighbor_segid)
    residue_pairs: list[tuple[int, int]]  # (res_ref, res_neighbor)
    distances_nm: np.ndarray  # (n_frames, n_neighbors, n_spairs, n_rpairs)


def segid_pair_distances_to_neighbors(
    traj: Structure,
    *,
    reference_segids: Sequence[str],
    neighbor_segids: Sequence[Sequence[str]],
    residue_pairs: Sequence[tuple[int, int]],
    segid_pairs: Optional[Sequence[tuple[str, str]]] = None,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    stride: int = 1,
) -> SegidPairDistanceResult:
    """
    Per-segid-pair distances between a reference tile and user-specified neighbor tiles.

    Distances are computed for:
      - each neighbor tile (outer dimension)
      - each segid pair (ref_segid, neighbor_segid)
      - each residue pair (res_ref, res_neighbor)

    Residue indices are directional:
      - res_ref is applied to ref_segid
      - res_neighbor is applied to neighbor_segid

    Distance definition:
      - uses residue center within each segid (mean of all atoms at that resnum)
    """
    ref = _normalize_tile(reference_segids, name="reference_segids")
    nbrs = _normalize_neighbors(neighbor_segids)
    rpairs = _normalize_pairs(residue_pairs)
    _validate_ranges(frame_start, frame_stop, stride)

    coords = getattr(traj, "_coords_nm", None)
    if coords is None:
        raise ValueError("traj must be trajectory-backed (Structure._coords_nm is None)")
    xyz_all = np.asarray(coords, dtype=np.float64)
    if xyz_all.ndim != 3 or xyz_all.shape[2] != 3:
        raise ValueError("traj._coords_nm must have shape (n_frames, n_atoms, 3)")

    n_frames = int(xyz_all.shape[0])
    if int(frame_start) >= n_frames:
        raise ValueError(f"frame_start {frame_start} out of range (0..{n_frames - 1})")
    stop = n_frames if frame_stop is None else min(int(frame_stop), n_frames)

    model = traj.model
    seg_to_atom_idx = _segment_atom_indices_from_model(model)
    seg_res_to_idx = _segment_residue_atom_indices_from_model(model)

    _ensure_segids_present(ref, seg_to_atom_idx, label="reference")
    for i, t in enumerate(nbrs):
        _ensure_segids_present(t, seg_to_atom_idx, label=f"neighbor[{i}]")

    if segid_pairs is None:
        spairs_by_neighbor = [list(_all_pairs(ref, t)) for t in nbrs]
    else:
        spairs = _normalize_segid_pairs(segid_pairs)
        _ensure_pairs_valid_for_neighbors(spairs, ref, nbrs)
        spairs_by_neighbor = [list(spairs) for _ in nbrs]

    n_neighbors = len(nbrs)
    n_rpairs = len(rpairs)
    n_spairs = max(len(p) for p in spairs_by_neighbor)
    if n_spairs <= 0:
        raise ValueError("no segid pairs to compute")

    frame_idx: list[int] = []
    dist_out: list[np.ndarray] = []

    for fi in range(int(frame_start), int(stop), int(stride)):
        xyz = xyz_all[fi]
        frame_idx.append(int(fi))

        row = np.full((n_neighbors, n_spairs, n_rpairs), np.nan, dtype=np.float64)
        for ni, tile in enumerate(nbrs):
            spairs = spairs_by_neighbor[ni]
            for si, (s_ref, s_nbr) in enumerate(spairs):
                for ri, (r_ref, r_nbr) in enumerate(rpairs):
                    a = _seg_res_center_nm(xyz, s_ref, r_ref, seg_res_to_idx)
                    b = _seg_res_center_nm(xyz, s_nbr, r_nbr, seg_res_to_idx)
                    row[ni, si, ri] = float(np.linalg.norm(a - b))

        dist_out.append(row)

    distances_nm = np.stack(dist_out, axis=0)
    # expose a single segid_pairs list if user supplied it; otherwise None is ambiguous
    flat_spairs = spairs_by_neighbor[0] if segid_pairs is not None else []
    return SegidPairDistanceResult(
        frame_indices=np.asarray(frame_idx, dtype=np.int64),
        reference_segids=ref,
        neighbor_tiles=[tuple(t) for t in nbrs],
        segid_pairs=flat_spairs,
        residue_pairs=rpairs,
        distances_nm=distances_nm,
    )


def _normalize_tile(segids: Sequence[str], *, name: str) -> tuple[str, ...]:
    t = tuple(str(s).strip() for s in segids if str(s).strip())
    if not t:
        raise ValueError(f"{name} must contain at least one non-empty segid")
    seen: set[str] = set()
    out: list[str] = []
    for s in t:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


def _normalize_neighbors(neighbor_segids: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    if not neighbor_segids:
        raise ValueError("neighbor_segids must contain at least one neighbor tile")
    out: list[tuple[str, ...]] = []
    for i, t in enumerate(neighbor_segids):
        out.append(_normalize_tile(t, name=f"neighbor_segids[{i}]"))
    return tuple(out)


def _normalize_pairs(residue_pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    pairs = [(int(a), int(b)) for a, b in residue_pairs]
    if not pairs:
        raise ValueError("residue_pairs must be non-empty")
    if any(a <= 0 or b <= 0 for a, b in pairs):
        raise ValueError("residue numbers must be positive (PDB-style)")
    return pairs


def _normalize_segid_pairs(segid_pairs: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for a, b in segid_pairs:
        sa = str(a).strip()
        sb = str(b).strip()
        if not sa or not sb:
            raise ValueError("segid_pairs cannot contain empty segids")
        out.append((sa, sb))
    if not out:
        raise ValueError("segid_pairs must be non-empty when provided")
    # keep order; dedupe
    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, str]] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return tuple(uniq)


def _validate_ranges(frame_start: int, frame_stop: Optional[int], stride: int) -> None:
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")
    if frame_stop is not None and int(frame_stop) < 0:
        raise ValueError("frame_stop must be >= 0")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if frame_stop is not None and int(frame_stop) < int(frame_start):
        raise ValueError("frame_stop must be >= frame_start")


def _segment_atom_indices_from_model(model: Any) -> dict[str, np.ndarray]:
    seg_map: dict[str, list[int]] = {}
    for ai, a in enumerate(model.atoms):
        seg = str(getattr(a, "seg", "") or "").strip()
        if not seg:
            continue
        seg_map.setdefault(seg, []).append(int(ai))
    return {s: np.asarray(ix, dtype=np.int32) for s, ix in seg_map.items() if ix}


def _segment_residue_atom_indices_from_model(model: Any) -> dict[str, dict[int, np.ndarray]]:
    seg_res: dict[str, dict[int, list[int]]] = {}
    for ai, a in enumerate(model.atoms):
        seg = str(getattr(a, "seg", "") or "").strip()
        if not seg:
            continue
        resnum = int(getattr(a, "resnum", 0) or 0)
        if resnum <= 0:
            continue
        seg_res.setdefault(seg, {}).setdefault(resnum, []).append(int(ai))

    out: dict[str, dict[int, np.ndarray]] = {}
    for seg, m in seg_res.items():
        out[seg] = {r: np.asarray(ix, dtype=np.int32) for r, ix in m.items()}
    return out


def _ensure_segids_present(
    tile: Sequence[str],
    seg_to_atom_idx: dict[str, np.ndarray],
    *,
    label: str,
) -> None:
    missing = [s for s in tile if s not in seg_to_atom_idx]
    if missing:
        raise ValueError(f"{label} segids not present in topology: {missing}")


def _all_pairs(ref: Sequence[str], nbr: Sequence[str]) -> Sequence[tuple[str, str]]:
    for a in ref:
        for b in nbr:
            yield a, b


def _ensure_pairs_valid_for_neighbors(
    spairs: Sequence[tuple[str, str]],
    ref: Sequence[str],
    nbrs: Sequence[Sequence[str]],
) -> None:
    ref_set = set(ref)
    nbr_union: set[str] = set()
    for t in nbrs:
        nbr_union |= set(t)

    bad_ref = sorted({a for a, _ in spairs if a not in ref_set})
    if bad_ref:
        raise ValueError(f"segid_pairs ref segids not in reference_segids: {bad_ref}")

    bad_nbr = sorted({b for _, b in spairs if b not in nbr_union})
    if bad_nbr:
        raise ValueError(f"segid_pairs neighbor segids not in any neighbor tile: {bad_nbr}")


def _seg_res_center_nm(
    xyz_nm: np.ndarray,
    segid: str,
    resnum: int,
    seg_res_to_idx: dict[str, dict[int, np.ndarray]],
) -> np.ndarray:
    m = seg_res_to_idx.get(str(segid))
    if m is None:
        raise ValueError(f"segid not found in topology: {segid}")
    arr = m.get(int(resnum))
    if arr is None or arr.size == 0:
        raise ValueError(f"no atoms for segid/resnum: segid={segid} resnum={resnum}")
    pts = xyz_nm[arr.astype(np.int64, copy=False), :]
    return np.mean(pts, axis=0)

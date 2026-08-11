from __future__ import annotations

import io
import multiprocessing as mp
import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional, Union

import numpy as np
from openmm.unit import Quantity, nanometer

from .molecule_data import PDBReader, StructureSelector, iter_dcd

FileLike = Union[str, Path, io.BytesIO, io.StringIO]


# ---- multiprocessing globals (inter mode) -------------------------------------

_MP_POS_I: Optional[np.ndarray] = None
_MP_POS_J: Optional[np.ndarray] = None
_MP_BOX: Optional[np.ndarray] = None
_MP_SHM: list[SharedMemory] = []


def _cpu_count() -> int:
    return int(os.cpu_count() or 1)


def _pick_start_method(want: Optional[str]) -> str:
    if want is not None:
        s = str(want).strip().lower()
        if s not in mp.get_all_start_methods():
            raise ValueError(f"unsupported mp_start_method: {want}")
        return s
    if os.name == "nt":
        return "spawn"
    if "fork" in mp.get_all_start_methods():
        return "fork"
    return "spawn"


def _shm_from_array(a: np.ndarray) -> tuple[SharedMemory, tuple[int, ...], str]:
    arr = np.ascontiguousarray(a)
    shm = SharedMemory(create=True, size=int(arr.nbytes))
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    return shm, tuple(int(x) for x in arr.shape), str(arr.dtype.str)


def _attach_shm(name: str, shape: tuple[int, ...], dtype: str) -> tuple[SharedMemory, np.ndarray]:
    shm = SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, arr


def _mp_init_shm(
    pos_i_name: str,
    pos_i_shape: tuple[int, ...],
    pos_i_dtype: str,
    pos_j_name: str,
    pos_j_shape: tuple[int, ...],
    pos_j_dtype: str,
    box_name: str,
    box_shape: tuple[int, ...],
    box_dtype: str,
) -> None:
    global _MP_POS_I, _MP_POS_J, _MP_BOX, _MP_SHM

    shm_i, a_i = _attach_shm(pos_i_name, pos_i_shape, pos_i_dtype)
    shm_j, a_j = _attach_shm(pos_j_name, pos_j_shape, pos_j_dtype)
    shm_b, a_b = _attach_shm(box_name, box_shape, box_dtype)

    _MP_SHM = [shm_i, shm_j, shm_b]
    _MP_POS_I = a_i
    _MP_POS_J = a_j
    _MP_BOX = a_b


def _mp_worker_inter_chunk(
    *,
    ref_start: int,
    ref_end: int,
    pi: np.ndarray,
    pj: np.ndarray,
    ring_size: int,
    origin_stride: int,
    lag_groups: list[list[tuple[int, int]]],
    r_edges: np.ndarray,
    distance_image: str,
    n_bins: int,
    n_lags: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    pos_i_ts = _MP_POS_I
    pos_j_ts = _MP_POS_J
    box_ts = _MP_BOX
    if pos_i_ts is None or pos_j_ts is None or box_ts is None:
        raise RuntimeError("multiprocessing globals not initialized")

    di = str(distance_image).strip().lower()
    if di not in {"unwrapped", "min_image", "hybrid"}:
        raise ValueError("distance_image must be 'unwrapped', 'min_image', or 'hybrid'")

    ref0 = int(ref_start)
    ref1 = int(ref_end)
    if ref1 <= ref0:
        raise ValueError("empty ref chunk")

    pi_v = np.asarray(pi, dtype=np.int64).reshape(-1)
    pj_v = np.asarray(pj, dtype=np.int64).reshape(-1)
    if pi_v.size != pj_v.size:
        raise ValueError("pi/pj size mismatch")

    rep = pi_v - ref0
    if np.any(rep < 0) or np.any(rep >= (ref1 - ref0)):
        raise ValueError("pi outside ref chunk bounds")

    n_rep = int(ref1 - ref0)
    n_frames = int(pos_i_ts.shape[0])
    max_lag = int(ring_size - 1)
    if n_frames <= max_lag:
        raise ValueError("not enough frames for requested max lag")

    sum_rep = np.zeros((n_rep, int(n_bins), int(n_lags)), dtype=np.float64)
    cnt_rep = np.zeros((n_rep, int(n_bins), int(n_lags)), dtype=np.int64)

    if di == "hybrid":
        delta_ring = np.empty((int(ring_size), int(pi_v.size), 3), dtype=np.float64)
        box_ring = np.empty((int(ring_size), 3), dtype=np.float64)
    else:
        r_ring = np.empty((int(ring_size), int(pi_v.size)), dtype=np.float64)

    for t in range(n_frames):
        pos1 = int(t % ring_size)

        box = np.asarray(box_ts[t], dtype=np.float64).reshape(3)
        d = pos_j_ts[t, pj_v, :] - pos_i_ts[t, pi_v, :]
        if di == "min_image":
            d = _min_image_disp_nm(d, box)

        if di == "hybrid":
            delta_ring[pos1, :, :] = d
            box_ring[pos1, :] = box
        else:
            r_ring[pos1, :] = np.linalg.norm(d, axis=1)

        rem = int(t % int(origin_stride))
        for tau, li in lag_groups[rem]:
            if t < tau:
                continue
            t0 = int(t - tau)
            if (t0 % int(origin_stride)) != 0:
                continue
            pos0 = int(t0 % ring_size)

            if di == "hybrid":
                d0 = delta_ring[pos0, :, :]
                d1 = delta_ring[pos1, :, :]
                b0 = box_ring[pos0, :].reshape(1, 3)
                b1 = box_ring[pos1, :].reshape(1, 3)
                n_img = np.rint(d0 / b0).astype(np.int64)
                r0 = np.linalg.norm(d0 - n_img * b0, axis=1)
                r1 = np.linalg.norm(d1 - n_img * b1, axis=1)
            else:
                r0 = r_ring[pos0, :]
                r1 = r_ring[pos1, :]

            _accumulate_one_lag(
                r0=r0,
                r1=r1,
                rep_idx=rep,
                r_edges=r_edges,
                n_rep=n_rep,
                n_bins=int(n_bins),
                sum_rep=sum_rep,
                cnt_rep=cnt_rep,
                lag_i=int(li),
            )

    return int(ref0), sum_rep, cnt_rep


@dataclass(frozen=True)
class DistMSDBinnedResult:
    t_ns: np.ndarray  # (n_lags,)
    lags_frames: np.ndarray  # (n_lags,)
    r0_centers_nm: np.ndarray  # (n_bins,)
    r_edges_nm: np.ndarray  # (n_bins + 1,)
    msd_nm2: np.ndarray  # (n_bins, n_lags)
    msd_stderr_nm2: np.ndarray  # (n_bins, n_lags)
    msd_rep_nm2: np.ndarray  # (n_rep, n_bins, n_lags)
    counts: np.ndarray  # (n_bins, n_lags)
    n_replicates: int
    n_chains: int
    n_frames: int
    dt_ns: float
    origin_stride: int
    mode: str  # "intra" | "inter"
    distance_image: str  # "unwrapped" | "min_image" | "hybrid"
    inter_targets_per_ref: Optional[int]
    random_seed: int
    selection: object = None
    chain_labels: tuple[str, ...] = ()
    atom_labels: tuple[tuple[str, str], ...] = ()
    # Legacy metadata retained for old res_i/res_j callers.
    res_i: Optional[int] = None
    res_j: Optional[int] = None
    atom_name: Optional[str] = None


@dataclass(frozen=True)
class DistMSDPairResult:
    intra: Optional[DistMSDBinnedResult]
    inter: Optional[DistMSDBinnedResult]


@dataclass(frozen=True)
class DistDiffusionFitResult:
    r0_centers_nm: np.ndarray  # (n_bins,)
    d_nm2_per_ns: np.ndarray  # (n_bins,)
    d_stderr_nm2_per_ns: np.ndarray  # (n_bins,)
    slope_nm2_per_ns: np.ndarray  # (n_bins,)
    intercept_nm2: np.ndarray  # (n_bins,)
    fit_tmin_ns: float
    fit_tmax_ns: float
    dims: int
    mode: str
    selection: object = None
    chain_labels: tuple[str, ...] = ()
    atom_labels: tuple[tuple[str, str], ...] = ()
    res_i: Optional[int] = None
    res_j: Optional[int] = None
    atom_name: Optional[str] = None
    # Multi-set statistics. For a single-set fit, d_stderr_nm2_per_ns retains
    # the within-set replicate SEM for backward compatibility. For multi-set
    # input, it is the SEM across independent simulation sets.
    d_std_nm2_per_ns: Optional[np.ndarray] = None
    per_set_d_nm2_per_ns: Optional[np.ndarray] = None
    per_set_stderr_nm2_per_ns: Optional[np.ndarray] = None
    per_set_slope_nm2_per_ns: Optional[np.ndarray] = None
    per_set_intercept_nm2: Optional[np.ndarray] = None
    set_labels: tuple[str, ...] = ()
    n_sets_per_bin: Optional[np.ndarray] = None
    n_sets: int = 1
    aggregation: str = "single_set"


def _as_file_list(x: Union[FileLike, Sequence[FileLike]]) -> list[FileLike]:
    if isinstance(x, (str, Path, io.BytesIO, io.StringIO)):
        return [x]
    return list(x)


def _box_lengths_nm(box_nm: object) -> np.ndarray:
    if box_nm is None:
        raise ValueError("box_nm is required")
    if isinstance(box_nm, Quantity):
        arr = np.asarray(box_nm.value_in_unit(nanometer), dtype=np.float64)
    else:
        arr = np.asarray(box_nm, dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size != 3:
        raise ValueError("box_nm must be a length-3 sequence (nm)")
    if np.any(arr <= 0.0):
        raise ValueError("box lengths must be positive")
    return arr.copy()


def _min_image_disp_nm(d_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    b = np.asarray(box_nm, dtype=np.float64).reshape(1, 3)
    d = np.asarray(d_nm, dtype=np.float64)
    return d - np.rint(d / b) * b


def _site_atom_indices_by_chain(
    tmpl: object,
    *,
    resnum: int,
    atom_name: str = "CA",
) -> tuple[list[str], np.ndarray]:
    model = tmpl.model if hasattr(tmpl, "model") else tmpl
    want = str(atom_name).strip().upper()
    out_keys: list[str] = []
    out_idx: list[int] = []

    atom_to_idx = {id(a): i for i, a in enumerate(model.atoms)}

    for key, ch in model.chain.items():
        hit = None
        for r in ch.residues:
            if int(r.resnum) != int(resnum):
                continue
            for a in r.atoms:
                if (a.name or "").strip().upper() == want:
                    hit = a
                    break
            if hit is not None:
                break
        if hit is None:
            continue
        idx = atom_to_idx.get(id(hit))
        if idx is None:
            continue
        out_keys.append(str(key))
        out_idx.append(int(idx))

    return out_keys, np.asarray(out_idx, dtype=np.int64)


def _selection_to_groups(
    tmpl: object,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]],
) -> list[np.ndarray]:
    """Resolve a StructureSelector-compatible selection into atom-index groups."""
    if isinstance(selection, str):
        groups_raw = StructureSelector(selection).atom_lists(tmpl)
    elif selection and all(isinstance(x, str) for x in selection):
        groups_raw = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_raw = [[int(i) for i in g] for g in selection]  # type: ignore[arg-type]
    return [np.asarray(g, dtype=np.int64) for g in groups_raw if len(g) > 0]


def _chain_indices_from_selection(
    tmpl: object,
    chains: Union[str, Sequence[str]],
) -> tuple[list[int], tuple[str, ...], np.ndarray]:
    """Resolve a StructureSelector-compatible chain selection to physical chains."""
    model = tmpl.model if hasattr(tmpl, "model") else tmpl
    n_atoms = int(len(model.atoms))
    atom_index = {id(atom): i for i, atom in enumerate(model.atoms)}
    atom_to_chain = np.full(n_atoms, -1, dtype=np.int64)
    physical_labels: list[str] = []

    for ci, (key, chain) in enumerate(model.chain.items()):
        physical_labels.append(str(key))
        for residue in chain.residues:
            for atom in residue.atoms:
                ai = atom_index.get(id(atom))
                if ai is not None:
                    atom_to_chain[int(ai)] = int(ci)

    if isinstance(chains, str):
        specs: list[object] = [chains]
    elif isinstance(chains, Sequence):
        specs = list(chains)
    else:
        raise TypeError("chains must be a selection string or sequence of selections")
    if not specs:
        raise ValueError("chains is empty")

    selected: list[int] = []
    seen: set[int] = set()
    for spec in specs:
        groups = _selection_to_groups(tmpl, spec)  # type: ignore[arg-type]
        if not groups:
            raise ValueError(f"chain selection {spec!r} produced no atoms")
        touched: set[int] = set()
        for group in groups:
            cis = np.unique(atom_to_chain[np.asarray(group, dtype=np.int64)])
            touched.update(int(ci) for ci in cis.tolist() if int(ci) >= 0)
        if not touched:
            raise ValueError(f"chain selection {spec!r} did not resolve to a physical chain")
        for ci in sorted(touched):
            if ci not in seen:
                selected.append(ci)
                seen.add(ci)

    return selected, tuple(physical_labels[i] for i in selected), atom_to_chain


def _ordered_atom_pair_selection_specs(
    selection: Union[str, Sequence[str]],
) -> tuple[str, str]:
    """Normalize an ordered pair selection such as ``39.CE2,69.SG``."""
    if isinstance(selection, str):
        raw = selection.strip()
        if not raw:
            raise ValueError("selection is empty")
        normalized = raw.replace(";", ",").replace("_", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
    elif isinstance(selection, Sequence):
        parts = [str(part).strip() for part in selection]
    else:
        raise TypeError("selection must be a string or sequence of two strings")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(
            "selection must contain exactly two ordered atom selectors, for example "
            "'39.CE2,69.SG' or ['39.CE2', '69.SG']"
        )
    return parts[0], parts[1]


def _atom_label(model: object, atom_index: int) -> str:
    atom = model.atoms[int(atom_index)]
    resname = (getattr(atom, "resname", "") or "").strip()
    resnum = int(getattr(atom, "resnum", 0))
    name = (getattr(atom, "name", "") or "").strip()
    return f"{resname}{resnum}.{name}"


def _aligned_selected_pair_atoms(
    tmpl: object,
    *,
    chains: Union[str, Sequence[str]],
    selection: Union[str, Sequence[str]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    Optional[int],
    Optional[int],
    Optional[str],
]:
    """Resolve one ordered atom pair in every selected physical chain.

    Returns local atom indices for the reduced trajectory, the full atom-index
    list read from DCD, chain labels, per-chain atom labels, and legacy residue/
    atom metadata when it can be represented by the old interface.
    """
    model = tmpl.model if hasattr(tmpl, "model") else tmpl
    chain_indices, chain_labels, atom_to_chain = _chain_indices_from_selection(tmpl, chains)
    if not chain_indices:
        raise ValueError("chains produced no physical chains")

    first_spec, second_spec = _ordered_atom_pair_selection_specs(selection)

    first_groups = _selection_to_groups(tmpl, first_spec)
    second_groups = _selection_to_groups(tmpl, second_spec)
    if not first_groups:
        raise ValueError(f"first atom selection {first_spec!r} produced no atoms")
    if not second_groups:
        raise ValueError(f"second atom selection {second_spec!r} produced no atoms")

    first_set = {
        int(i) for group in first_groups for i in np.asarray(group, dtype=np.int64).tolist()
    }
    second_set = {
        int(i) for group in second_groups for i in np.asarray(group, dtype=np.int64).tolist()
    }

    first_full: list[int] = []
    second_full: list[int] = []
    atom_labels: list[tuple[str, str]] = []

    for ci, label in zip(chain_indices, chain_labels):
        first = sorted(i for i in first_set if int(atom_to_chain[i]) == int(ci))
        second = sorted(i for i in second_set if int(atom_to_chain[i]) == int(ci))
        if len(first) != 1:
            desc = [_atom_label(model, i) for i in first]
            raise ValueError(
                f"first atom selection {first_spec!r} must resolve to exactly one atom "
                f"in chain {label!r}; selected {len(first)}: {desc}"
            )
        if len(second) != 1:
            desc = [_atom_label(model, i) for i in second]
            raise ValueError(
                f"second atom selection {second_spec!r} must resolve to exactly one atom "
                f"in chain {label!r}; selected {len(second)}: {desc}"
            )
        first_full.append(int(first[0]))
        second_full.append(int(second[0]))
        atom_labels.append((_atom_label(model, first[0]), _atom_label(model, second[0])))

    atom_indices_full = sorted(set(first_full + second_full))
    index_map = {old: new for new, old in enumerate(atom_indices_full)}
    idx_i = np.asarray([index_map[i] for i in first_full], dtype=np.int64)
    idx_j = np.asarray([index_map[i] for i in second_full], dtype=np.int64)

    # Populate legacy metadata only when the selected atoms are consistent across chains.
    first_atoms = [model.atoms[i] for i in first_full]
    second_atoms = [model.atoms[i] for i in second_full]
    res_i_values = {int(getattr(a, "resnum", 0)) for a in first_atoms}
    res_j_values = {int(getattr(a, "resnum", 0)) for a in second_atoms}
    name_i = {(getattr(a, "name", "") or "").strip().upper() for a in first_atoms}
    name_j = {(getattr(a, "name", "") or "").strip().upper() for a in second_atoms}
    legacy_res_i = next(iter(res_i_values)) if len(res_i_values) == 1 else None
    legacy_res_j = next(iter(res_j_values)) if len(res_j_values) == 1 else None
    legacy_atom_name = next(iter(name_i)) if len(name_i) == 1 and name_i == name_j else None

    return (
        idx_i,
        idx_j,
        atom_indices_full,
        chain_labels,
        tuple(atom_labels),
        legacy_res_i,
        legacy_res_j,
        legacy_atom_name,
    )


def _validate_lags(lags_frames: Sequence[int]) -> np.ndarray:
    lag = np.asarray([int(x) for x in lags_frames], dtype=np.int64).reshape(-1)
    if lag.size < 1:
        raise ValueError("lags_frames must have >=1 element")
    if np.any(lag < 0):
        raise ValueError("lags_frames must be >=0")
    lag = np.unique(lag)
    lag.sort()
    if lag[0] != 0:
        lag = np.concatenate([np.asarray([0], dtype=np.int64), lag])
    return lag


def _make_lags(
    *,
    max_lag_frames: Optional[int],
    lag_stride: int,
    lags_frames: Optional[Sequence[int]],
) -> np.ndarray:
    if lags_frames is not None:
        return _validate_lags(lags_frames)
    if max_lag_frames is None:
        raise ValueError("provide lags_frames or max_lag_frames")
    m = int(max_lag_frames)
    if m < 0:
        raise ValueError("max_lag_frames must be >=0")
    s = int(lag_stride)
    if s <= 0:
        raise ValueError("lag_stride must be >=1")
    return _validate_lags(list(range(0, m + 1, s)))


def _make_r_edges(
    *,
    r_edges_nm: Optional[Sequence[float]],
    r_min_nm: float,
    r_max_nm: Optional[float],
    n_bins: int,
) -> np.ndarray:
    if r_edges_nm is not None:
        e = np.asarray(r_edges_nm, dtype=np.float64).reshape(-1)
        if e.size < 2:
            raise ValueError("r_edges_nm must have >=2 elements")
        if np.any(~np.isfinite(e)):
            raise ValueError("r_edges_nm must be finite")
        if np.any(np.diff(e) <= 0.0):
            raise ValueError("r_edges_nm must be strictly increasing")
        return e

    if r_max_nm is None:
        raise ValueError("provide r_edges_nm or r_max_nm")

    r0 = float(r_min_nm)
    r1 = float(r_max_nm)
    if r1 <= r0:
        raise ValueError("r_max_nm must be > r_min_nm")
    nb = int(n_bins)
    if nb <= 0:
        raise ValueError("n_bins must be >=1")
    return np.linspace(r0, r1, nb + 1, dtype=np.float64)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("need >=2 points to fit")
    a = np.vstack([x, np.ones_like(x)]).T
    sol, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _normalize_dist_fit_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    if not isinstance(mode, str):
        raise TypeError("mode must be 'intra', 'inter', or None")
    value = mode.strip().lower().replace("-", "_")
    aliases = {
        "intra": "intra",
        "intrachain": "intra",
        "inter": "inter",
        "interchain": "inter",
    }
    if value not in aliases:
        raise ValueError("mode must be 'intra' or 'inter'")
    return aliases[value]


def _select_dist_msd_for_fit(
    value: object,
    *,
    mode: Optional[str],
    label: str,
) -> DistMSDBinnedResult:
    """Extract one intra/inter binned-MSD result from a fit input value."""
    if isinstance(value, DistMSDBinnedResult):
        if mode is not None and str(value.mode) != mode:
            raise ValueError(
                f"{label!r}: requested mode={mode!r}, but the supplied "
                f"DistMSDBinnedResult has mode={value.mode!r}"
            )
        return value

    if isinstance(value, DistMSDPairResult) or (
        hasattr(value, "intra") and hasattr(value, "inter")
    ):
        intra = getattr(value, "intra", None)
        inter = getattr(value, "inter", None)
        selected_mode = mode
        if selected_mode is None:
            available = [
                name for name, result in (("intra", intra), ("inter", inter)) if result is not None
            ]
            if len(available) != 1:
                raise ValueError(
                    f"{label!r}: mode='intra' or mode='inter' is required when "
                    "the supplied DistMSDPairResult contains both components"
                )
            selected_mode = available[0]
        result = intra if selected_mode == "intra" else inter
        if result is None:
            raise ValueError(f"{label!r}: requested {selected_mode!r} component is not present")
        if not isinstance(result, DistMSDBinnedResult):
            # Permit compatible result objects from an already-imported module
            # instance, while still validating the fields used by the fitter.
            required = ("t_ns", "r0_centers_nm", "msd_nm2", "msd_rep_nm2", "n_replicates", "mode")
            if not all(hasattr(result, field) for field in required):
                raise TypeError(
                    f"{label!r}: {selected_mode} component is not a compatible "
                    "DistMSDBinnedResult"
                )
        return result  # type: ignore[return-value]

    raise TypeError(
        f"{label!r}: fit input must be DistMSDBinnedResult, DistMSDPairResult, "
        "or a mapping whose values are one of those result types"
    )


def _fit_dist_msd_linear_bins_single(
    msd: DistMSDBinnedResult,
    *,
    fit_tmin_ns: float,
    fit_tmax_ns: float,
    dims: int,
    use_rep_sem: bool,
) -> DistDiffusionFitResult:
    """Fit one binned-MSD result; internal helper for single/multi-set input."""
    if int(dims) <= 0:
        raise ValueError("dims must be >= 1")
    t = np.asarray(msd.t_ns, dtype=np.float64)
    tmin = float(fit_tmin_ns)
    tmax = float(fit_tmax_ns)
    if tmax <= tmin:
        raise ValueError("fit_tmax_ns must be > fit_tmin_ns")
    time_sel = (t >= tmin) & (t <= tmax) & np.isfinite(t)
    if int(np.sum(time_sel)) < 2:
        raise ValueError("fit window selects <2 time points")

    n_bins = int(msd.r0_centers_nm.size)
    slope = np.full((n_bins,), np.nan, dtype=np.float64)
    intercept = np.full((n_bins,), np.nan, dtype=np.float64)
    d = np.full((n_bins,), np.nan, dtype=np.float64)
    d_err = np.full((n_bins,), np.nan, dtype=np.float64)

    mean_msd = np.asarray(msd.msd_nm2, dtype=np.float64)
    for b in range(n_bins):
        valid = time_sel & np.isfinite(mean_msd[b, :])
        if int(np.sum(valid)) < 2:
            continue
        s, itc = _linear_fit(t[valid], mean_msd[b, valid])
        slope[b] = s
        intercept[b] = itc
        d[b] = s / (2.0 * float(dims))

    if use_rep_sem and msd.n_replicates >= 1:
        rep_msd = np.asarray(msd.msd_rep_nm2, dtype=np.float64)
        d_rep = np.full((msd.n_replicates, n_bins), np.nan, dtype=np.float64)
        for r in range(msd.n_replicates):
            y = rep_msd[r]
            for b in range(n_bins):
                valid = time_sel & np.isfinite(y[b, :])
                if int(np.sum(valid)) < 2:
                    continue
                s, _ = _linear_fit(t[valid], y[b, valid])
                d_rep[r, b] = s / (2.0 * float(dims))

        finite = np.isfinite(d_rep)
        n_eff_i = np.sum(finite, axis=0, dtype=np.int64)
        n_eff = n_eff_i.astype(np.float64)
        d_err[n_eff_i == 1] = 0.0

        enough = n_eff_i > 1
        if np.any(enough):
            rep_sum = np.sum(np.where(finite, d_rep, 0.0), axis=0)
            rep_mean = np.full(n_bins, np.nan, dtype=np.float64)
            np.divide(rep_sum, n_eff, out=rep_mean, where=n_eff_i > 0)
            deviations = np.where(finite, d_rep - rep_mean[None, :], 0.0)
            ss = np.sum(deviations * deviations, axis=0)
            variance = np.zeros(n_bins, dtype=np.float64)
            np.divide(ss, n_eff - 1.0, out=variance, where=enough)
            d_err[enough] = np.sqrt(variance[enough] / n_eff[enough])
    else:
        d_err[np.isfinite(d)] = 0.0

    finite_d = np.isfinite(d)
    one_set_count = finite_d.astype(np.int64)
    single_std = np.full(n_bins, np.nan, dtype=np.float64)

    return DistDiffusionFitResult(
        r0_centers_nm=np.asarray(msd.r0_centers_nm, dtype=np.float64).copy(),
        d_nm2_per_ns=d,
        d_stderr_nm2_per_ns=d_err,
        slope_nm2_per_ns=slope,
        intercept_nm2=intercept,
        fit_tmin_ns=tmin,
        fit_tmax_ns=tmax,
        dims=int(dims),
        mode=str(msd.mode),
        selection=getattr(msd, "selection", None),
        chain_labels=tuple(getattr(msd, "chain_labels", ())),
        atom_labels=tuple(getattr(msd, "atom_labels", ())),
        res_i=getattr(msd, "res_i", None),
        res_j=getattr(msd, "res_j", None),
        atom_name=getattr(msd, "atom_name", None),
        d_std_nm2_per_ns=single_std,
        per_set_d_nm2_per_ns=d.reshape(1, -1),
        per_set_stderr_nm2_per_ns=d_err.reshape(1, -1),
        per_set_slope_nm2_per_ns=slope.reshape(1, -1),
        per_set_intercept_nm2=intercept.reshape(1, -1),
        set_labels=(),
        n_sets_per_bin=one_set_count,
        n_sets=1,
        aggregation="single_set",
    )


def _nanmean_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite-value column means without all-NaN runtime warnings."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    counts = np.sum(finite, axis=0, dtype=np.int64)
    sums = np.sum(np.where(finite, array, 0.0), axis=0)
    mean = np.full(array.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean, counts


def fit_dist_msd_linear_bins(
    data: Union[
        DistMSDBinnedResult,
        DistMSDPairResult,
        Mapping[str, Union[DistMSDBinnedResult, DistMSDPairResult]],
    ],
    *,
    fit_tmin_ns: float,
    fit_tmax_ns: float,
    mode: Optional[str] = None,
    dims: int = 1,
    use_rep_sem: bool = True,
    min_sets: int = 1,
) -> DistDiffusionFitResult:
    """Fit distance-dependent diffusion for one result or multiple simulation sets.

    Parameters
    ----------
    data
        One :class:`DistMSDBinnedResult`, one :class:`DistMSDPairResult`, or a
        mapping such as the dictionary returned by ``get_values(...,
        calc='distmsd')``. Mapping entries are treated as independent simulation
        sets and receive equal statistical weight.
    mode
        Select ``'intra'`` or ``'inter'`` when ``data`` contains
        :class:`DistMSDPairResult` objects. It may be omitted for an already
        selected :class:`DistMSDBinnedResult`, or when a pair result contains
        only one component.
    use_rep_sem
        For each individual set, estimate the within-set SEM over chain/reference
        replicates. For multi-set input these values are retained in
        ``per_set_stderr_nm2_per_ns``; the returned ``d_stderr_nm2_per_ns`` is
        instead the SEM across independent simulation sets.
    min_sets
        Minimum number of finite set-level fitted D values required to report the
        averaged D for a distance bin. The default is 1. The set-level SEM remains
        NaN unless at least two sets contribute.

    Notes
    -----
    Every simulation set is fitted independently first. The final multi-set
    ``d_nm2_per_ns`` is the equal-weight mean of those fitted D(r) values, and
    ``d_stderr_nm2_per_ns`` is the standard error across independent sets. This
    deliberately does not pool chains or time origins across simulation sets.

    Backward compatibility is retained: passing one DistMSDBinnedResult returns
    the same fitted curve and within-set replicate SEM as previous versions.
    """
    selected_mode = _normalize_dist_fit_mode(mode)

    if isinstance(min_sets, (bool, np.bool_)):
        raise TypeError("min_sets must be an integer >= 1")
    try:
        min_sets_float = float(min_sets)
    except (TypeError, ValueError) as exc:
        raise TypeError("min_sets must be an integer >= 1") from exc
    if not np.isfinite(min_sets_float) or not min_sets_float.is_integer() or min_sets_float < 1:
        raise ValueError("min_sets must be an integer >= 1")
    min_sets_i = int(min_sets_float)

    if not isinstance(data, Mapping):
        msd = _select_dist_msd_for_fit(data, mode=selected_mode, label="data")
        return _fit_dist_msd_linear_bins_single(
            msd,
            fit_tmin_ns=fit_tmin_ns,
            fit_tmax_ns=fit_tmax_ns,
            dims=dims,
            use_rep_sem=use_rep_sem,
        )

    if not data:
        raise ValueError("data mapping is empty")

    set_labels = tuple(str(key) for key in data)
    fits: list[DistDiffusionFitResult] = []
    for raw_key, value in data.items():
        key = str(raw_key)
        msd = _select_dist_msd_for_fit(value, mode=selected_mode, label=key)
        fits.append(
            _fit_dist_msd_linear_bins_single(
                msd,
                fit_tmin_ns=fit_tmin_ns,
                fit_tmax_ns=fit_tmax_ns,
                dims=dims,
                use_rep_sem=use_rep_sem,
            )
        )

    if min_sets_i > len(fits):
        raise ValueError(f"min_sets={min_sets_i} exceeds the number of supplied sets ({len(fits)})")

    first_r = np.asarray(fits[0].r0_centers_nm, dtype=np.float64)
    first_mode = str(fits[0].mode)
    for key, fit in zip(set_labels[1:], fits[1:]):
        r = np.asarray(fit.r0_centers_nm, dtype=np.float64)
        if r.shape != first_r.shape or not np.allclose(r, first_r, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                f"{key!r}: distance-bin centers differ from the first set; "
                "all sets must use the same bins"
            )
        if str(fit.mode) != first_mode:
            raise ValueError("supplied sets do not share the same intra/inter mode")

    per_set_d = np.stack([fit.d_nm2_per_ns for fit in fits], axis=0)
    per_set_err = np.stack([fit.d_stderr_nm2_per_ns for fit in fits], axis=0)
    per_set_slope = np.stack([fit.slope_nm2_per_ns for fit in fits], axis=0)
    per_set_intercept = np.stack([fit.intercept_nm2 for fit in fits], axis=0)

    d_mean, n_sets_per_bin = _nanmean_rows(per_set_d)
    slope_mean, _ = _nanmean_rows(per_set_slope)
    intercept_mean, _ = _nanmean_rows(per_set_intercept)

    n_bins = int(first_r.size)
    d_std = np.full(n_bins, np.nan, dtype=np.float64)
    d_sem = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        values = per_set_d[:, b]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            std = float(np.std(values, ddof=1))
            d_std[b] = std
            d_sem[b] = std / np.sqrt(float(values.size))

    insufficient = n_sets_per_bin < min_sets_i
    d_mean[insufficient] = np.nan
    slope_mean[insufficient] = np.nan
    intercept_mean[insufficient] = np.nan
    d_std[insufficient] = np.nan
    d_sem[insufficient] = np.nan

    # Metadata are common across sets when possible. Chain labels are deliberately
    # omitted from a multi-set aggregate because they are within-set replicates,
    # not the independent contributors represented by the final SEM.
    selections = [fit.selection for fit in fits]
    selection = (
        selections[0] if all(repr(x) == repr(selections[0]) for x in selections[1:]) else None
    )
    atom_labels = (
        fits[0].atom_labels
        if all(fit.atom_labels == fits[0].atom_labels for fit in fits[1:])
        else ()
    )
    res_i = fits[0].res_i if all(fit.res_i == fits[0].res_i for fit in fits[1:]) else None
    res_j = fits[0].res_j if all(fit.res_j == fits[0].res_j for fit in fits[1:]) else None
    atom_name = (
        fits[0].atom_name if all(fit.atom_name == fits[0].atom_name for fit in fits[1:]) else None
    )

    return DistDiffusionFitResult(
        r0_centers_nm=first_r.copy(),
        d_nm2_per_ns=d_mean,
        d_stderr_nm2_per_ns=d_sem,
        slope_nm2_per_ns=slope_mean,
        intercept_nm2=intercept_mean,
        fit_tmin_ns=float(fit_tmin_ns),
        fit_tmax_ns=float(fit_tmax_ns),
        dims=int(dims),
        mode=first_mode,
        selection=selection,
        chain_labels=(),
        atom_labels=atom_labels,
        res_i=res_i,
        res_j=res_j,
        atom_name=atom_name,
        d_std_nm2_per_ns=d_std,
        per_set_d_nm2_per_ns=per_set_d,
        per_set_stderr_nm2_per_ns=per_set_err,
        per_set_slope_nm2_per_ns=per_set_slope,
        per_set_intercept_nm2=per_set_intercept,
        set_labels=set_labels,
        n_sets_per_bin=n_sets_per_bin,
        n_sets=len(fits),
        aggregation="sets_equal_weight",
    )


def _unwrap_step_nm(
    x_wr: np.ndarray,
    *,
    x_wr_prev: np.ndarray,
    x_un_prev: np.ndarray,
    box_nm: np.ndarray,
) -> np.ndarray:
    b = np.asarray(box_nm, dtype=np.float64).reshape(1, 3)
    d = np.asarray(x_wr, dtype=np.float64) - np.asarray(x_wr_prev, dtype=np.float64)
    d -= np.rint(d / b) * b
    return np.asarray(x_un_prev, dtype=np.float64) + d


def _aligned_chain_atoms(
    tmpl: object,
    *,
    res_i: int,
    res_j: int,
    atom_name: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    keys_i, idx_i_full = _site_atom_indices_by_chain(
        tmpl,
        resnum=int(res_i),
        atom_name=str(atom_name),
    )
    keys_j, idx_j_full = _site_atom_indices_by_chain(
        tmpl,
        resnum=int(res_j),
        atom_name=str(atom_name),
    )
    map_j = {k: int(v) for k, v in zip(keys_j, idx_j_full.tolist())}

    ii: list[int] = []
    jj: list[int] = []
    keep_keys: list[str] = []
    for k, vi in zip(keys_i, idx_i_full.tolist()):
        vj = map_j.get(k)
        if vj is None:
            continue
        keep_keys.append(k)
        ii.append(int(vi))
        jj.append(int(vj))

    if len(ii) < 1:
        raise ValueError("need >=1 chain with both residues present")

    atom_indices_full = sorted(set(ii + jj))
    idx_map = {old: new for new, old in enumerate(atom_indices_full)}
    idx_i = np.asarray([idx_map[int(x)] for x in ii], dtype=np.int64)
    idx_j = np.asarray([idx_map[int(x)] for x in jj], dtype=np.int64)
    return idx_i, idx_j, atom_indices_full


def _sample_inter_pairs(
    n_ch: int,
    *,
    targets_per_ref: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_ch < 2:
        raise ValueError("need >=2 chains for inter distances")

    rng = np.random.default_rng(int(seed))
    pair_i: list[np.ndarray] = []
    pair_j: list[np.ndarray] = []
    rep: list[np.ndarray] = []

    all_idx = np.arange(n_ch, dtype=np.int64)
    for a in range(n_ch):
        targets = all_idx[all_idx != a]
        if targets_per_ref is not None and int(targets_per_ref) < int(targets.size):
            sel = rng.choice(targets, size=int(targets_per_ref), replace=False)
            sel = np.asarray(sel, dtype=np.int64)
        else:
            sel = targets
        pair_i.append(np.full((int(sel.size),), a, dtype=np.int64))
        pair_j.append(np.asarray(sel, dtype=np.int64))
        rep.append(np.full((int(sel.size),), a, dtype=np.int64))

    pi = np.concatenate(pair_i, axis=0)
    pj = np.concatenate(pair_j, axis=0)
    pr = np.concatenate(rep, axis=0)
    return pi, pj, pr


def _load_site_time_series(
    dcd_list: Sequence[FileLike],
    tmpl_model: object,
    *,
    atom_indices_full: Sequence[int],
    sel_i: np.ndarray,
    sel_j: np.ndarray,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_fallback: Optional[np.ndarray],
    unwrap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_i: list[np.ndarray] = []
    pos_j: list[np.ndarray] = []
    boxes: list[np.ndarray] = []

    i_wr_prev = i_un_prev = None
    j_wr_prev = j_un_prev = None

    for dcd in dcd_list:
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
                    raise ValueError("DCD lacks box; pass box_nm=(Lx,Ly,Lz) in nm")
                box = box_fallback
            else:
                box = _box_lengths_nm(box_frame_nm)

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
            pi_wr = xyz[sel_i, :]
            pj_wr = xyz[sel_j, :]

            if unwrap:
                if i_wr_prev is None:
                    i_wr_prev = pi_wr.copy()
                    i_un_prev = pi_wr.copy()
                    j_wr_prev = pj_wr.copy()
                    j_un_prev = pj_wr.copy()
                    pi_un = i_un_prev
                    pj_un = j_un_prev
                else:
                    assert i_un_prev is not None
                    assert j_un_prev is not None
                    pi_un = _unwrap_step_nm(
                        pi_wr,
                        x_wr_prev=i_wr_prev,
                        x_un_prev=i_un_prev,
                        box_nm=box,
                    )
                    pj_un = _unwrap_step_nm(
                        pj_wr,
                        x_wr_prev=j_wr_prev,
                        x_un_prev=j_un_prev,
                        box_nm=box,
                    )
                    i_wr_prev = pi_wr
                    i_un_prev = pi_un
                    j_wr_prev = pj_wr
                    j_un_prev = pj_un
                pos_i.append(np.asarray(pi_un, dtype=np.float64))
                pos_j.append(np.asarray(pj_un, dtype=np.float64))
            else:
                pos_i.append(np.asarray(pi_wr, dtype=np.float64))
                pos_j.append(np.asarray(pj_wr, dtype=np.float64))

            boxes.append(np.asarray(box, dtype=np.float64))

    if not pos_i:
        raise ValueError("no frames selected")

    return (
        np.stack(pos_i, axis=0),
        np.stack(pos_j, axis=0),
        np.stack(boxes, axis=0),
    )


def _compute_intra_from_ts(
    *,
    pos_i_ts: np.ndarray,
    pos_j_ts: np.ndarray,
    box_ts: np.ndarray,
    distance_image: str,
    lags: np.ndarray,
    lag_groups: list[list[tuple[int, int]]],
    origin_stride: int,
    r_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    di = str(distance_image).strip().lower()
    n_frames = int(pos_i_ts.shape[0])
    n_ch = int(pos_i_ts.shape[1])
    n_lags = int(lags.size)
    n_bins = int(r_edges.size - 1)

    ring_size = int(lags[-1] + 1)
    intra_ring = np.empty((ring_size, n_ch), dtype=np.float64)

    sum_rep = np.zeros((n_ch, n_bins, n_lags), dtype=np.float64)
    cnt_rep = np.zeros((n_ch, n_bins, n_lags), dtype=np.int64)
    rep_idx = np.arange(n_ch, dtype=np.int64)

    for t in range(n_frames):
        pos1 = int(t % ring_size)
        d = pos_j_ts[t, :, :] - pos_i_ts[t, :, :]
        if di == "min_image":
            d = _min_image_disp_nm(d, box_ts[t, :])
        intra_ring[pos1, :] = np.linalg.norm(d, axis=1)

        rem = int(t % int(origin_stride))
        for tau, li in lag_groups[rem]:
            if t < tau:
                continue
            t0 = int(t - tau)
            if (t0 % int(origin_stride)) != 0:
                continue
            pos0 = int(t0 % ring_size)
            _accumulate_one_lag(
                r0=intra_ring[pos0, :],
                r1=intra_ring[pos1, :],
                rep_idx=rep_idx,
                r_edges=r_edges,
                n_rep=n_ch,
                n_bins=n_bins,
                sum_rep=sum_rep,
                cnt_rep=cnt_rep,
                lag_i=int(li),
            )

    return sum_rep, cnt_rep


def _compute_inter_process_from_ts(
    *,
    pos_i_ts: np.ndarray,
    pos_j_ts: np.ndarray,
    box_ts: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
    distance_image: str,
    lags: np.ndarray,
    lag_groups: list[list[tuple[int, int]]],
    origin_stride: int,
    r_edges: np.ndarray,
    n_jobs: int,
    mp_start_method: Optional[str],
) -> tuple[np.ndarray, np.ndarray]:
    di = str(distance_image).strip().lower()
    if di not in {"unwrapped", "min_image", "hybrid"}:
        raise ValueError("distance_image must be 'unwrapped', 'min_image', or 'hybrid'")

    n_ch = int(pos_i_ts.shape[1])
    n_lags = int(lags.size)
    n_bins = int(r_edges.size - 1)
    ring_size = int(lags[-1] + 1)

    jobs = int(n_jobs)
    if jobs <= 0:
        jobs = _cpu_count()
    jobs = min(jobs, n_ch)

    pi_v = np.asarray(pi, dtype=np.int64).reshape(-1)
    pj_v = np.asarray(pj, dtype=np.int64).reshape(-1)
    if pi_v.size != pj_v.size:
        raise ValueError("pi/pj size mismatch")
    if (pi_v.size % n_ch) != 0:
        raise ValueError("inter pair array not divisible by n_ch")
    pairs_per_ref = int(pi_v.size // n_ch)

    blocks: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for refs in np.array_split(np.arange(n_ch, dtype=np.int64), jobs):
        if refs.size == 0:
            continue
        a0 = int(refs[0])
        a1 = int(refs[-1] + 1)
        p0 = int(a0 * pairs_per_ref)
        p1 = int(a1 * pairs_per_ref)
        blocks.append((a0, a1, pi_v[p0:p1].copy(), pj_v[p0:p1].copy()))

    sum_all = np.zeros((n_ch, n_bins, n_lags), dtype=np.float64)
    cnt_all = np.zeros((n_ch, n_bins, n_lags), dtype=np.int64)

    method = _pick_start_method(mp_start_method)
    ctx = mp.get_context(method)

    global _MP_POS_I, _MP_POS_J, _MP_BOX, _MP_SHM
    _MP_POS_I = None
    _MP_POS_J = None
    _MP_BOX = None
    _MP_SHM = []

    shms: list[SharedMemory] = []
    init = None
    initargs: tuple[object, ...] = ()

    if method == "fork":
        _MP_POS_I = pos_i_ts
        _MP_POS_J = pos_j_ts
        _MP_BOX = box_ts
    else:
        shm_i, shape_i, dtype_i = _shm_from_array(pos_i_ts)
        shm_j, shape_j, dtype_j = _shm_from_array(pos_j_ts)
        shm_b, shape_b, dtype_b = _shm_from_array(box_ts)
        shms = [shm_i, shm_j, shm_b]
        init = _mp_init_shm
        initargs = (
            shm_i.name,
            shape_i,
            dtype_i,
            shm_j.name,
            shape_j,
            dtype_j,
            shm_b.name,
            shape_b,
            dtype_b,
        )

    try:
        with ProcessPoolExecutor(
            max_workers=int(jobs),
            mp_context=ctx,
            initializer=init,
            initargs=initargs,
        ) as ex:
            futs = []
            for a0, a1, pi_blk, pj_blk in blocks:
                futs.append(
                    ex.submit(
                        _mp_worker_inter_chunk,
                        ref_start=a0,
                        ref_end=a1,
                        pi=pi_blk,
                        pj=pj_blk,
                        ring_size=ring_size,
                        origin_stride=int(origin_stride),
                        lag_groups=lag_groups,
                        r_edges=r_edges,
                        distance_image=di,
                        n_bins=n_bins,
                        n_lags=n_lags,
                    )
                )
            for fut in as_completed(futs):
                a0, s_blk, c_blk = fut.result()
                a0i = int(a0)
                a1i = int(a0i + s_blk.shape[0])
                sum_all[a0i:a1i, :, :] = s_blk
                cnt_all[a0i:a1i, :, :] = c_blk
    finally:
        for shm in shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass

    _MP_POS_I = None
    _MP_POS_J = None
    _MP_BOX = None

    return sum_all, cnt_all


def _prep_lag_groups(lags: np.ndarray, origin_stride: int) -> list[list[tuple[int, int]]]:
    os_ = int(origin_stride)
    if os_ <= 0:
        raise ValueError("origin_stride must be >=1")
    groups: list[list[tuple[int, int]]] = [[] for _ in range(os_)]
    for li, tau in enumerate(lags.tolist()):
        groups[int(tau) % os_].append((int(tau), int(li)))
    return groups


def _accumulate_one_lag(
    *,
    r0: np.ndarray,
    r1: np.ndarray,
    rep_idx: np.ndarray,
    r_edges: np.ndarray,
    n_rep: int,
    n_bins: int,
    sum_rep: np.ndarray,
    cnt_rep: np.ndarray,
    lag_i: int,
) -> None:
    r0v = np.asarray(r0, dtype=np.float64).reshape(-1)
    r1v = np.asarray(r1, dtype=np.float64).reshape(-1)
    if r0v.size != r1v.size:
        raise ValueError("r0 and r1 must have same size")
    if r0v.size < 1:
        return

    dr2 = (r1v - r0v) ** 2
    b = np.searchsorted(r_edges, r0v, side="right").astype(np.int64) - 1

    m = (b >= 0) & (b < int(n_bins))
    if not np.any(m):
        return

    b = b[m]
    w = dr2[m]
    rep = np.asarray(rep_idx, dtype=np.int64).reshape(-1)[m]

    idx = rep * int(n_bins) + b
    flat_len = int(n_rep) * int(n_bins)

    c = np.bincount(idx, minlength=flat_len)
    s = np.bincount(idx, weights=w, minlength=flat_len)

    cnt_rep[:, :, int(lag_i)] += c.reshape(int(n_rep), int(n_bins))
    sum_rep[:, :, int(lag_i)] += s.reshape(int(n_rep), int(n_bins))


def _finalize_binned_msd(
    *,
    sum_rep: np.ndarray,
    cnt_rep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finalize replicate-resolved MSDs without warnings for empty bins.

    A distance-bin/lag combination can legitimately have no observations,
    especially for sparsely populated initial-distance bins or long lag times.
    Such entries remain NaN.  SEM is NaN when no replicate contributes, zero
    when exactly one replicate contributes, and is calculated across replicate
    means when at least two contribute.
    """
    sum_rep = np.asarray(sum_rep, dtype=np.float64)
    cnt_rep = np.asarray(cnt_rep, dtype=np.int64)

    msd_rep = np.full_like(sum_rep, np.nan, dtype=np.float64)
    np.divide(sum_rep, cnt_rep, out=msd_rep, where=cnt_rep > 0)

    finite = np.isfinite(msd_rep)
    n_eff_i = np.sum(finite, axis=0, dtype=np.int64)
    n_eff = n_eff_i.astype(np.float64)

    # Equal-weight mean over contributing replicates.  Avoid np.nanmean so that
    # all-empty slices remain NaN without emitting RuntimeWarning.
    msd = np.full(msd_rep.shape[1:], np.nan, dtype=np.float64)
    rep_sum = np.sum(np.where(finite, msd_rep, 0.0), axis=0)
    np.divide(rep_sum, n_eff, out=msd, where=n_eff_i > 0)

    # Preserve the distinction between no data and a single contributing
    # replicate.  The latter has zero across-replicate SEM by convention here.
    err = np.full_like(msd, np.nan, dtype=np.float64)
    err[n_eff_i == 1] = 0.0

    enough = n_eff_i > 1
    if np.any(enough):
        deviations = np.where(finite, msd_rep - msd[None, :, :], 0.0)
        ss = np.sum(deviations * deviations, axis=0)
        variance = np.zeros_like(msd, dtype=np.float64)
        np.divide(ss, n_eff - 1.0, out=variance, where=enough)
        err[enough] = np.sqrt(variance[enough] / n_eff[enough])

    return msd, err, msd_rep, cnt_rep.sum(axis=0)


def dist_msd_binned_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    chains: Union[str, Sequence[str]] = "protein",
    selection: Optional[Union[str, Sequence[str]]] = None,
    res_i: Optional[int] = None,
    res_j: Optional[int] = None,
    atom_name: str = "CA",
    mode: str = "both",  # "intra" | "inter" | "both"
    distance_image: str = "hybrid",  # "unwrapped" | "min_image" | "hybrid"
    dt_ns: float,
    r_edges_nm: Optional[Sequence[float]] = None,
    r_min_nm: float = 0.0,
    r_max_nm: Optional[float] = None,
    n_bins: int = 50,
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: int = 200,
    lag_stride: int = 1,
    origin_stride: int = 1,
    inter_targets_per_ref: Optional[int] = None,
    random_seed: int = 0,
    backend: str = "serial",  # "serial" | "process"
    n_jobs: int = 0,
    mp_start_method: Optional[str] = None,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> DistMSDPairResult:
    """
    Distance-dependent MSD for a one-dimensional atom-pair distance coordinate.

    Preferred selection interface
    -----------------------------
    ``chains`` is a StructureSelector-compatible selection identifying the physical
    chains to analyze. ``selection`` contains exactly two ordered atom selectors,
    for example ``"39.CE2,69.SG"``. The first atom is the reference atom for
    interchain distances and the second is the target atom. Each selector must
    resolve to exactly one atom in every selected chain.

    The legacy ``res_i``/``res_j``/``atom_name`` interface remains supported when
    ``selection`` is omitted.

    Modes
    -----
    - intra: r(t) between the two selected atoms within each chain
             (replicate = chain)
    - inter: r(t) from the first selected atom on a reference chain to the second
             selected atom on a different target chain (replicate = reference chain)

    Distance definition
    -------------------
    - distance_image="unwrapped" (default): time-unwrap each residue position per chain,
      then compute Euclidean distance in unwrapped space (continuous, can exceed box).
    - distance_image="min_image": compute minimum-image distance per frame (in [0,L/2]),
      but r(t) can jump when the nearest image changes.
    - distance_image="hybrid": r(t0) is minimum-image for binning, then r(t) uses
      time-unwrapped ("exploding") coordinates aligned to the same image as at t0.


    Performance knobs
    -----------------
    - origin_stride: only use time origins t0 where t0 % origin_stride == 0
      (reduces cost ~1/origin_stride)
    - lags_frames or (max_lag_frames, lag_stride): reduce number of lags computed
    - inter_targets_per_ref: subsample inter pairs (targets per reference chain)
    """
    if float(dt_ns) <= 0.0:
        raise ValueError("dt_ns must be > 0")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")
    if int(origin_stride) <= 0:
        raise ValueError("origin_stride must be >= 1")

    be = str(backend).strip().lower()
    if be not in {"serial", "process"}:
        raise ValueError("backend must be 'serial' or 'process'")

    m = str(mode).strip().lower()
    if m not in {"intra", "inter", "both"}:
        raise ValueError("mode must be 'intra', 'inter', or 'both'")

    di = str(distance_image).strip().lower()
    if di not in {"unwrapped", "min_image", "hybrid"}:
        raise ValueError("distance_image must be 'unwrapped', 'min_image', or 'hybrid'")

    lags = _make_lags(
        max_lag_frames=max_lag_frames,
        lag_stride=int(lag_stride),
        lags_frames=lags_frames,
    )
    max_lag = int(lags[-1])
    lag_groups = _prep_lag_groups(lags, int(origin_stride))

    r_edges = _make_r_edges(
        r_edges_nm=r_edges_nm,
        r_min_nm=float(r_min_nm),
        r_max_nm=r_max_nm,
        n_bins=int(n_bins),
    )
    n_bins_i = int(r_edges.size - 1)
    r0_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    if selection is None:
        if res_i is None or res_j is None:
            raise ValueError(
                "provide selection='atom1,atom2' (preferred), or legacy res_i and res_j"
            )
        resolved_selection: Union[str, Sequence[str]] = (
            f"{int(res_i)}.{str(atom_name).strip()}",
            f"{int(res_j)}.{str(atom_name).strip()}",
        )
    else:
        resolved_selection = selection

    (
        idx_i,
        idx_j,
        atom_indices_full,
        chain_labels,
        atom_labels,
        legacy_res_i,
        legacy_res_j,
        legacy_atom_name,
    ) = _aligned_selected_pair_atoms(
        tmpl,
        chains=chains,
        selection=resolved_selection,
    )
    n_ch = int(idx_i.size)
    if m in {"inter", "both"} and n_ch < 2:
        raise ValueError("need >=2 chains for inter mode")

    sel_i = idx_i
    sel_j = idx_j

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    if max_lag < 0:
        raise ValueError("invalid lags")
    ring_size = max_lag + 1

    have_intra = m in {"intra", "both"}
    have_inter = m in {"inter", "both"}

    if be == "process" and have_inter:
        unwrap_ts = di in {"unwrapped", "hybrid"}
        pos_i_ts, pos_j_ts, box_ts = _load_site_time_series(
            dcd_list,
            tmpl_model,
            atom_indices_full=atom_indices_full,
            sel_i=sel_i,
            sel_j=sel_j,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=int(frame_start),
            frame_stop=frame_stop,
            box_fallback=box_fallback,
            unwrap=unwrap_ts,
        )

        n_frames = int(pos_i_ts.shape[0])
        if n_frames < 2:
            raise ValueError("need >=2 frames selected")

        sum_intra = cnt_intra = None
        if have_intra:
            sum_intra, cnt_intra = _compute_intra_from_ts(
                pos_i_ts=pos_i_ts,
                pos_j_ts=pos_j_ts,
                box_ts=box_ts,
                distance_image=di,
                lags=lags,
                lag_groups=lag_groups,
                origin_stride=int(origin_stride),
                r_edges=r_edges,
            )

        pi, pj, _ = _sample_inter_pairs(
            n_ch,
            targets_per_ref=inter_targets_per_ref,
            seed=int(random_seed),
        )
        sum_inter, cnt_inter = _compute_inter_process_from_ts(
            pos_i_ts=pos_i_ts,
            pos_j_ts=pos_j_ts,
            box_ts=box_ts,
            pi=pi,
            pj=pj,
            distance_image=di,
            lags=lags,
            lag_groups=lag_groups,
            origin_stride=int(origin_stride),
            r_edges=r_edges,
            n_jobs=int(n_jobs),
            mp_start_method=mp_start_method,
        )

        t_ns = lags.astype(np.float64) * float(dt_ns)

        intra_res = None
        inter_res = None

        if have_intra and sum_intra is not None and cnt_intra is not None:
            msd, err, msd_rep, counts = _finalize_binned_msd(
                sum_rep=sum_intra,
                cnt_rep=cnt_intra,
            )
            lag0_populated = counts[:, 0] > 0
            msd[lag0_populated, 0] = 0.0
            err[lag0_populated, 0] = 0.0

            intra_res = DistMSDBinnedResult(
                t_ns=t_ns,
                lags_frames=lags,
                r0_centers_nm=r0_centers,
                r_edges_nm=r_edges,
                msd_nm2=msd,
                msd_stderr_nm2=err,
                msd_rep_nm2=msd_rep,
                counts=counts,
                n_replicates=int(n_ch),
                n_chains=int(n_ch),
                n_frames=int(n_frames),
                dt_ns=float(dt_ns),
                origin_stride=int(origin_stride),
                mode="intra",
                distance_image=di,
                inter_targets_per_ref=None,
                random_seed=int(random_seed),
                selection=resolved_selection,
                chain_labels=chain_labels,
                atom_labels=atom_labels,
                res_i=legacy_res_i,
                res_j=legacy_res_j,
                atom_name=legacy_atom_name,
            )

        msd, err, msd_rep, counts = _finalize_binned_msd(
            sum_rep=sum_inter,
            cnt_rep=cnt_inter,
        )
        lag0_populated = counts[:, 0] > 0
        msd[lag0_populated, 0] = 0.0
        err[lag0_populated, 0] = 0.0

        inter_res = DistMSDBinnedResult(
            t_ns=t_ns,
            lags_frames=lags,
            r0_centers_nm=r0_centers,
            r_edges_nm=r_edges,
            msd_nm2=msd,
            msd_stderr_nm2=err,
            msd_rep_nm2=msd_rep,
            counts=counts,
            n_replicates=int(n_ch),
            n_chains=int(n_ch),
            n_frames=int(n_frames),
            dt_ns=float(dt_ns),
            origin_stride=int(origin_stride),
            mode="inter",
            distance_image=di,
            inter_targets_per_ref=(
                None if inter_targets_per_ref is None else int(inter_targets_per_ref)
            ),
            random_seed=int(random_seed),
            selection=resolved_selection,
            chain_labels=chain_labels,
            atom_labels=atom_labels,
            res_i=legacy_res_i,
            res_j=legacy_res_j,
            atom_name=legacy_atom_name,
        )

        return DistMSDPairResult(intra=intra_res, inter=inter_res)

    intra_ring = None
    inter_ring = None
    inter_delta_ring = None
    box_ring = None

    rep_intra = np.arange(n_ch, dtype=np.int64)
    pi = pj = rep_inter = None
    if have_inter:
        pi, pj, rep_inter = _sample_inter_pairs(
            n_ch,
            targets_per_ref=inter_targets_per_ref,
            seed=int(random_seed),
        )
        if di == "hybrid":
            inter_delta_ring = np.empty((ring_size, int(pi.size), 3), dtype=np.float64)
            box_ring = np.empty((ring_size, 3), dtype=np.float64)
        else:
            inter_ring = np.empty((ring_size, int(pi.size)), dtype=np.float64)
    if have_intra:
        intra_ring = np.empty((ring_size, n_ch), dtype=np.float64)

    n_lags = int(lags.size)
    sum_intra = cnt_intra = None
    sum_inter = cnt_inter = None

    if have_intra:
        sum_intra = np.zeros((n_ch, n_bins_i, n_lags), dtype=np.float64)
        cnt_intra = np.zeros((n_ch, n_bins_i, n_lags), dtype=np.int64)
    if have_inter:
        sum_inter = np.zeros((n_ch, n_bins_i, n_lags), dtype=np.float64)
        cnt_inter = np.zeros((n_ch, n_bins_i, n_lags), dtype=np.int64)

    i_wr_prev = i_un_prev = None
    j_wr_prev = j_un_prev = None

    n_frames = 0
    for dcd in dcd_list:
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
                    raise ValueError("DCD lacks box; pass box_nm=(Lx,Ly,Lz) in nm")
                box = box_fallback
            else:
                box = _box_lengths_nm(box_frame_nm)

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
            pos_i_wr = xyz[sel_i, :]
            pos_j_wr = xyz[sel_j, :]

            if di in {"unwrapped", "hybrid"}:
                if i_wr_prev is None:
                    i_wr_prev = pos_i_wr.copy()
                    i_un_prev = pos_i_wr.copy()
                    j_wr_prev = pos_j_wr.copy()
                    j_un_prev = pos_j_wr.copy()
                    pos_i = i_un_prev
                    pos_j = j_un_prev
                else:
                    assert i_un_prev is not None
                    assert j_un_prev is not None
                    pos_i = _unwrap_step_nm(
                        pos_i_wr,
                        x_wr_prev=i_wr_prev,
                        x_un_prev=i_un_prev,
                        box_nm=box,
                    )
                    pos_j = _unwrap_step_nm(
                        pos_j_wr,
                        x_wr_prev=j_wr_prev,
                        x_un_prev=j_un_prev,
                        box_nm=box,
                    )
                    i_wr_prev = pos_i_wr
                    i_un_prev = pos_i
                    j_wr_prev = pos_j_wr
                    j_un_prev = pos_j
            else:
                pos_i = pos_i_wr
                pos_j = pos_j_wr

            pos1 = n_frames % ring_size

            if have_intra:
                d_intra = pos_j - pos_i
                if di == "min_image":
                    d_intra = _min_image_disp_nm(d_intra, box)
                r_intra = np.linalg.norm(d_intra, axis=1)
                intra_ring[pos1, :] = r_intra  # type: ignore[index]

            if have_inter:
                assert pi is not None and pj is not None
                if di == "hybrid":
                    assert inter_delta_ring is not None
                    assert box_ring is not None
                    delta = pos_j[pj, :] - pos_i[pi, :]
                    inter_delta_ring[pos1, :, :] = delta
                    box_ring[pos1, :] = box
                else:
                    d_inter = pos_j[pj, :] - pos_i[pi, :]
                    if di == "min_image":
                        d_inter = _min_image_disp_nm(d_inter, box)
                    r_inter = np.linalg.norm(d_inter, axis=1)
                    inter_ring[pos1, :] = r_inter  # type: ignore[index]

            rem = n_frames % int(origin_stride)
            for tau, li in lag_groups[rem]:
                if n_frames < tau:
                    continue
                t0 = n_frames - tau
                if (t0 % int(origin_stride)) != 0:
                    continue
                pos0 = t0 % ring_size

                if have_intra:
                    r0 = intra_ring[pos0, :]  # type: ignore[index]
                    r1 = intra_ring[pos1, :]  # type: ignore[index]
                    _accumulate_one_lag(
                        r0=r0,
                        r1=r1,
                        rep_idx=rep_intra,
                        r_edges=r_edges,
                        n_rep=n_ch,
                        n_bins=n_bins_i,
                        sum_rep=sum_intra,  # type: ignore[arg-type]
                        cnt_rep=cnt_intra,  # type: ignore[arg-type]
                        lag_i=li,
                    )

                if have_inter:
                    if di == "hybrid":
                        assert inter_delta_ring is not None
                        assert box_ring is not None
                        d0 = inter_delta_ring[pos0, :, :]
                        d1 = inter_delta_ring[pos1, :, :]
                        b0 = box_ring[pos0, :].reshape(1, 3)
                        b1 = box_ring[pos1, :].reshape(1, 3)
                        n_img = np.rint(d0 / b0).astype(np.int64)
                        r0 = np.linalg.norm(d0 - n_img * b0, axis=1)
                        r1 = np.linalg.norm(d1 - n_img * b1, axis=1)
                    else:
                        r0 = inter_ring[pos0, :]  # type: ignore[index]
                        r1 = inter_ring[pos1, :]  # type: ignore[index]
                    _accumulate_one_lag(
                        r0=r0,
                        r1=r1,
                        rep_idx=rep_inter,  # type: ignore[arg-type]
                        r_edges=r_edges,
                        n_rep=n_ch,
                        n_bins=n_bins_i,
                        sum_rep=sum_inter,  # type: ignore[arg-type]
                        cnt_rep=cnt_inter,  # type: ignore[arg-type]
                        lag_i=li,
                    )

            n_frames += 1

    if n_frames < 2:
        raise ValueError("need >=2 frames selected")

    t_ns = lags.astype(np.float64) * float(dt_ns)

    intra_res = None
    inter_res = None

    if have_intra:
        msd, err, msd_rep, counts = _finalize_binned_msd(
            sum_rep=sum_intra,  # type: ignore[arg-type]
            cnt_rep=cnt_intra,  # type: ignore[arg-type]
        )
        lag0_populated = counts[:, 0] > 0
        msd[lag0_populated, 0] = 0.0
        err[lag0_populated, 0] = 0.0

        intra_res = DistMSDBinnedResult(
            t_ns=t_ns,
            lags_frames=lags,
            r0_centers_nm=r0_centers,
            r_edges_nm=r_edges,
            msd_nm2=msd,
            msd_stderr_nm2=err,
            msd_rep_nm2=msd_rep,
            counts=counts,
            n_replicates=n_ch,
            n_chains=n_ch,
            n_frames=n_frames,
            dt_ns=float(dt_ns),
            origin_stride=int(origin_stride),
            mode="intra",
            distance_image=di,
            inter_targets_per_ref=None,
            random_seed=int(random_seed),
            selection=resolved_selection,
            chain_labels=chain_labels,
            atom_labels=atom_labels,
            res_i=legacy_res_i,
            res_j=legacy_res_j,
            atom_name=legacy_atom_name,
        )

    if have_inter:
        msd, err, msd_rep, counts = _finalize_binned_msd(
            sum_rep=sum_inter,  # type: ignore[arg-type]
            cnt_rep=cnt_inter,  # type: ignore[arg-type]
        )
        lag0_populated = counts[:, 0] > 0
        msd[lag0_populated, 0] = 0.0
        err[lag0_populated, 0] = 0.0

        inter_res = DistMSDBinnedResult(
            t_ns=t_ns,
            lags_frames=lags,
            r0_centers_nm=r0_centers,
            r_edges_nm=r_edges,
            msd_nm2=msd,
            msd_stderr_nm2=err,
            msd_rep_nm2=msd_rep,
            counts=counts,
            n_replicates=n_ch,
            n_chains=n_ch,
            n_frames=n_frames,
            dt_ns=float(dt_ns),
            origin_stride=int(origin_stride),
            mode="inter",
            distance_image=di,
            inter_targets_per_ref=inter_targets_per_ref,
            random_seed=int(random_seed),
            selection=resolved_selection,
            chain_labels=chain_labels,
            atom_labels=atom_labels,
            res_i=legacy_res_i,
            res_j=legacy_res_j,
            atom_name=legacy_atom_name,
        )

    return DistMSDPairResult(intra=intra_res, inter=inter_res)


def dist_msd_binned_multi_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    residue_pairs: Sequence[tuple[int, int]],
    chains: Union[str, Sequence[str]] = "protein",
    atom_name: str = "CA",
    mode: str = "both",
    distance_image: str = "unwrapped",
    dt_ns: float,
    r_edges_nm: Optional[Sequence[float]] = None,
    r_min_nm: float = 0.0,
    r_max_nm: Optional[float] = None,
    n_bins: int = 50,
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: Optional[int] = None,
    lag_stride: int = 1,
    origin_stride: int = 1,
    inter_targets_per_ref: Optional[int] = None,
    random_seed: int = 0,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
    n_jobs: int = 0,
) -> dict[tuple[int, int], DistMSDPairResult]:
    """
    Parallel wrapper over multiple residue pairs (threaded).

    Use n_jobs=0 for all cores. Threading is usually fine because the heavy work is NumPy.
    """
    pairs = [(int(a), int(b)) for a, b in residue_pairs]
    if not pairs:
        raise ValueError("residue_pairs is empty")

    jobs = int(n_jobs)
    if jobs <= 0:
        jobs = os.cpu_count() or 1
    jobs = min(jobs, len(pairs))

    def _run(pair: tuple[int, int]) -> tuple[tuple[int, int], DistMSDPairResult]:
        a, b = pair
        out = dist_msd_binned_from_dcd(
            pdb_file,
            dcd_files,
            chains=chains,
            res_i=a,
            res_j=b,
            atom_name=atom_name,
            mode=mode,
            distance_image=distance_image,
            dt_ns=dt_ns,
            r_edges_nm=r_edges_nm,
            r_min_nm=r_min_nm,
            r_max_nm=r_max_nm,
            n_bins=n_bins,
            lags_frames=lags_frames,
            max_lag_frames=max_lag_frames,
            lag_stride=lag_stride,
            origin_stride=origin_stride,
            inter_targets_per_ref=inter_targets_per_ref,
            random_seed=random_seed,
            stride=stride,
            chunk=chunk,
            frame_start=frame_start,
            frame_stop=frame_stop,
            box_nm=box_nm,
        )
        return (a, b), out

    if jobs <= 1:
        return dict(_run(p) for p in pairs)

    out: dict[tuple[int, int], DistMSDPairResult] = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_run, p): p for p in pairs}
        for fut in as_completed(futs):
            k, v = fut.result()
            out[k] = v
    return out


# ---- all-residue distance-dependent MSD maps ---------------------------------

_MAP_POS_TS: Optional[np.memmap] = None
_MAP_BOX_TS: Optional[np.memmap] = None
_MAP_TARGETS: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DistMSDMapResult:
    """Distance-window-conditioned MSDs for all residue-site pairs.

    ``intra_msd_nm2`` is populated only in the strict lower triangle and
    ``inter_msd_nm2`` in the upper triangle including the diagonal.  Each cell
    contains one MSD curve over ``t_ns``.  The conditioning window is applied to
    the initial distance r(t0); when both distance limits are None all finite
    initial distances are accepted.
    """

    t_ns: np.ndarray
    lags_frames: np.ndarray
    residue_numbers: np.ndarray
    residue_names: tuple[str, ...]
    atom_name: str
    chain_labels: tuple[str, ...]
    intra_msd_nm2: np.ndarray  # (n_res, n_res, n_lags), lower triangle
    inter_msd_nm2: np.ndarray  # (n_res, n_res, n_lags), upper triangle incl diag
    intra_counts: np.ndarray  # same shape; total observations
    inter_counts: np.ndarray  # same shape; total observations
    n_frames: int
    dt_ns: float
    origin_stride: int
    distance_image: str
    distance_min_nm: Optional[float]
    distance_max_nm: Optional[float]
    min_intra_sequence_separation: int
    inter_targets_per_ref: Optional[int]
    random_seed: int
    pair_block_size: int
    origin_chunk: int
    reference_chunk: int
    reference_chain_labels: tuple[str, ...] = ()
    aggregation: str = "reference_replicates_equal_weight"
    units: str = "nm^2"


@dataclass(frozen=True)
class DistDiffusionMapFitResult:
    """Fitted D maps for one or more independent simulation sets."""

    residue_numbers: np.ndarray
    residue_names: tuple[str, ...]
    atom_name: str
    intra_d_nm2_per_ns: np.ndarray
    inter_d_nm2_per_ns: np.ndarray
    combined_d_nm2_per_ns: np.ndarray
    intra_stderr_nm2_per_ns: np.ndarray
    inter_stderr_nm2_per_ns: np.ndarray
    combined_stderr_nm2_per_ns: np.ndarray
    intra_n_sets: np.ndarray
    inter_n_sets: np.ndarray
    fit_tmin_ns: float
    fit_tmax_ns: float
    dims: int
    distance_image: str
    distance_min_nm: Optional[float]
    distance_max_nm: Optional[float]
    min_intra_sequence_separation: int
    set_labels: tuple[str, ...]
    n_sets: int
    per_set_intra_d_nm2_per_ns: Optional[np.ndarray] = None
    per_set_inter_d_nm2_per_ns: Optional[np.ndarray] = None
    aggregation: str = "sets_equal_weight"
    units: str = "nm^2/ns"

    @property
    def intra(self) -> np.ndarray:
        return self.intra_d_nm2_per_ns

    @property
    def inter(self) -> np.ndarray:
        return self.inter_d_nm2_per_ns

    @property
    def combined(self) -> np.ndarray:
        return self.combined_d_nm2_per_ns


def _aligned_map_site_atoms(
    tmpl: object,
    *,
    chains: Union[str, Sequence[str]],
    atom_name: str,
) -> tuple[np.ndarray, list[int], tuple[str, ...], np.ndarray, tuple[str, ...]]:
    """Resolve one named site atom for every retained residue in selected chains.

    The residue ordering is taken from the first selected physical chain.  A
    residue is retained when that first chain contains exactly one ``atom_name``;
    every other selected chain must then contain exactly one matching residue
    number and atom.  This naturally omits residues such as GLY when ``CB`` is
    requested, while still enforcing homologous site alignment across chains.
    """
    model = tmpl.model if hasattr(tmpl, "model") else tmpl
    chain_indices, chain_labels, _ = _chain_indices_from_selection(tmpl, chains)
    if not chain_indices:
        raise ValueError("chains produced no physical chains")

    want = str(atom_name).strip().upper()
    if not want:
        raise ValueError("atom_name must be non-empty")

    atom_to_idx = {id(atom): i for i, atom in enumerate(model.atoms)}
    physical_chains = list(model.chain.values())

    first_chain = physical_chains[int(chain_indices[0])]
    first_sites: list[tuple[int, str, int]] = []
    seen_resnums: set[int] = set()
    for residue in first_chain.residues:
        hits = [
            atom_to_idx[id(atom)]
            for atom in residue.atoms
            if (getattr(atom, "name", "") or "").strip().upper() == want and id(atom) in atom_to_idx
        ]
        if len(hits) == 0:
            continue
        if len(hits) != 1:
            raise ValueError(
                f"chain {chain_labels[0]!r}, residue {residue.resnum}: atom name "
                f"{want!r} resolves to {len(hits)} atoms"
            )
        resnum = int(residue.resnum)
        if resnum in seen_resnums:
            raise ValueError(
                f"chain {chain_labels[0]!r} contains duplicate residue number {resnum}; "
                "all-pairs maps require unique residue numbers within a chain"
            )
        seen_resnums.add(resnum)
        first_sites.append((resnum, str(residue.resname), int(hits[0])))

    if len(first_sites) < 2:
        raise ValueError(f"selected chains contain fewer than two residues with atom {want!r}")

    residue_numbers = np.asarray([item[0] for item in first_sites], dtype=np.int64)
    residue_names = tuple(item[1] for item in first_sites)
    n_ch = len(chain_indices)
    n_res = len(first_sites)
    full_indices = np.empty((n_ch, n_res), dtype=np.int64)
    full_indices[0, :] = np.asarray([item[2] for item in first_sites], dtype=np.int64)

    for out_ci, physical_ci in enumerate(chain_indices[1:], start=1):
        chain = physical_chains[int(physical_ci)]
        by_resnum: dict[int, list[int]] = {}
        for residue in chain.residues:
            hits = [
                atom_to_idx[id(atom)]
                for atom in residue.atoms
                if (getattr(atom, "name", "") or "").strip().upper() == want
                and id(atom) in atom_to_idx
            ]
            if hits:
                by_resnum.setdefault(int(residue.resnum), []).extend(int(x) for x in hits)

        for ri, resnum in enumerate(residue_numbers.tolist()):
            hits = by_resnum.get(int(resnum), [])
            if len(hits) != 1:
                raise ValueError(
                    f"chain {chain_labels[out_ci]!r}, residue {resnum}: expected exactly "
                    f"one atom {want!r}, found {len(hits)}"
                )
            full_indices[out_ci, ri] = int(hits[0])

    atom_indices_full = sorted(set(int(x) for x in full_indices.reshape(-1).tolist()))
    local_index = {old: new for new, old in enumerate(atom_indices_full)}
    local_sites = np.asarray(
        [[local_index[int(x)] for x in row] for row in full_indices],
        dtype=np.int64,
    )
    return local_sites, atom_indices_full, chain_labels, residue_numbers, residue_names


def _stream_map_sites_to_memmap(
    dcd_list: Sequence[FileLike],
    tmpl_model: object,
    *,
    atom_indices_full: Sequence[int],
    local_sites: np.ndarray,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_fallback: Optional[np.ndarray],
    unwrap: bool,
    directory: str,
    dtype: np.dtype,
) -> tuple[str, str, int, tuple[int, int, int, int]]:
    """Stream selected map sites to disk-backed contiguous arrays."""
    pos_path = os.path.join(directory, "distmsd_map_positions.dat")
    box_path = os.path.join(directory, "distmsd_map_boxes.dat")
    dtype_np = np.dtype(dtype)

    n_ch, n_res = (int(local_sites.shape[0]), int(local_sites.shape[1]))
    n_frames = 0
    prev_wr: Optional[np.ndarray] = None
    prev_un: Optional[np.ndarray] = None

    with open(pos_path, "wb") as pos_handle, open(box_path, "wb") as box_handle:
        for dcd in dcd_list:
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
                        raise ValueError("DCD lacks box; pass box_nm=(Lx,Ly,Lz) in nm")
                    box = np.asarray(box_fallback, dtype=np.float64).reshape(3)
                else:
                    box = _box_lengths_nm(box_frame_nm)

                xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
                sites_wr = xyz[local_sites.reshape(-1), :].reshape(n_ch, n_res, 3)

                if unwrap:
                    if prev_wr is None:
                        sites = sites_wr.copy()
                    else:
                        assert prev_un is not None
                        flat = _unwrap_step_nm(
                            sites_wr.reshape(-1, 3),
                            x_wr_prev=prev_wr.reshape(-1, 3),
                            x_un_prev=prev_un.reshape(-1, 3),
                            box_nm=box,
                        )
                        sites = flat.reshape(n_ch, n_res, 3)
                    prev_wr = sites_wr.copy()
                    prev_un = sites.copy()
                else:
                    sites = sites_wr

                np.asarray(sites, dtype=dtype_np).tofile(pos_handle)
                np.asarray(box, dtype=dtype_np).tofile(box_handle)
                n_frames += 1

    if n_frames < 2:
        raise ValueError("need >=2 selected frames")

    shape = (int(n_frames), n_ch, n_res, 3)
    return pos_path, box_path, int(n_frames), shape


def _map_init_memmaps(
    pos_path: str,
    pos_shape: tuple[int, int, int, int],
    dtype_str: str,
    box_path: str,
    box_shape: tuple[int, int],
    targets: np.ndarray,
) -> None:
    global _MAP_POS_TS, _MAP_BOX_TS, _MAP_TARGETS
    dtype = np.dtype(dtype_str)
    _MAP_POS_TS = np.memmap(pos_path, mode="r", dtype=dtype, shape=pos_shape)
    _MAP_BOX_TS = np.memmap(box_path, mode="r", dtype=dtype, shape=box_shape)
    _MAP_TARGETS = np.asarray(targets, dtype=np.int64)


def _distance_window_mask(
    r0: np.ndarray,
    *,
    distance_min_nm: Optional[float],
    distance_max_nm: Optional[float],
) -> np.ndarray:
    keep = np.isfinite(r0)
    if distance_min_nm is not None:
        keep &= r0 >= float(distance_min_nm)
    if distance_max_nm is not None:
        keep &= r0 < float(distance_max_nm)
    return keep


def _map_r0_r1_from_displacements(
    d0: np.ndarray,
    d1: np.ndarray,
    *,
    box0: np.ndarray,
    box1: np.ndarray,
    distance_image: str,
) -> tuple[np.ndarray, np.ndarray]:
    di = str(distance_image)
    if di == "hybrid":
        n_img = np.rint(d0 / box0)
        r0 = np.linalg.norm(d0 - n_img * box0, axis=-1)
        r1 = np.linalg.norm(d1 - n_img * box1, axis=-1)
        return r0, r1
    if di == "min_image":
        d0 = d0 - np.rint(d0 / box0) * box0
        d1 = d1 - np.rint(d1 / box1) * box1
    return np.linalg.norm(d0, axis=-1), np.linalg.norm(d1, axis=-1)


def _map_process_pair_block(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    intra_enabled: np.ndarray,
    *,
    lags: np.ndarray,
    origin_stride: int,
    distance_image: str,
    distance_min_nm: Optional[float],
    distance_max_nm: Optional[float],
    do_intra: bool,
    do_inter: bool,
    origin_chunk: int,
    reference_chunk: int,
    reference_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker for one block of residue pairs using process-global memmaps."""
    pos = _MAP_POS_TS
    boxes = _MAP_BOX_TS
    targets = _MAP_TARGETS
    if pos is None or boxes is None:
        raise RuntimeError("map memmaps are not initialized")

    ri = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    rj = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    intra_ok = np.asarray(intra_enabled, dtype=bool).reshape(-1)
    if ri.size != rj.size or ri.size != intra_ok.size:
        raise ValueError("pair block arrays have inconsistent sizes")

    n_pairs = int(ri.size)
    n_lags = int(len(lags))
    n_frames = int(pos.shape[0])
    n_ch = int(pos.shape[1])

    if reference_indices is None:
        average_chain_idx = np.arange(n_ch, dtype=np.int64)
    else:
        average_chain_idx = np.asarray(reference_indices, dtype=np.int64).reshape(-1)
        if average_chain_idx.size == 0:
            raise ValueError("reference_indices selects no chains")
        if np.any(average_chain_idx < 0) or np.any(average_chain_idx >= n_ch):
            raise ValueError("reference_indices contains an out-of-range chain index")
        if np.unique(average_chain_idx).size != average_chain_idx.size:
            raise ValueError("reference_indices contains duplicate chain indices")
    n_average_chains = int(average_chain_idx.size)

    intra_msd = np.full((n_pairs, n_lags), np.nan, dtype=np.float64)
    inter_msd = np.full((n_pairs, n_lags), np.nan, dtype=np.float64)
    intra_counts = np.zeros((n_pairs, n_lags), dtype=np.int64)
    inter_counts = np.zeros((n_pairs, n_lags), dtype=np.int64)

    oc = max(1, int(origin_chunk))
    rc = max(1, int(reference_chunk))

    for li, tau_raw in enumerate(np.asarray(lags, dtype=np.int64).tolist()):
        tau = int(tau_raw)
        if tau >= n_frames:
            continue
        origins = np.arange(0, n_frames - tau, int(origin_stride), dtype=np.int64)
        if origins.size == 0:
            continue

        # ---- intrachain: one equal-weight replicate per chain -----------------
        if do_intra and np.any(intra_ok):
            sum_rep = np.zeros((n_average_chains, n_pairs), dtype=np.float64)
            cnt_rep = np.zeros((n_average_chains, n_pairs), dtype=np.int64)

            for o0 in range(0, int(origins.size), oc):
                t0 = origins[o0 : o0 + oc]
                t1 = t0 + tau
                x0_i = pos[
                    t0[:, None, None], average_chain_idx[None, :, None], ri[None, None, :], :
                ]
                x0_j = pos[
                    t0[:, None, None], average_chain_idx[None, :, None], rj[None, None, :], :
                ]
                x1_i = pos[
                    t1[:, None, None], average_chain_idx[None, :, None], ri[None, None, :], :
                ]
                x1_j = pos[
                    t1[:, None, None], average_chain_idx[None, :, None], rj[None, None, :], :
                ]

                d0 = np.asarray(x0_j - x0_i, dtype=np.float64)
                d1 = np.asarray(x1_j - x1_i, dtype=np.float64)
                b0 = np.asarray(boxes[t0], dtype=np.float64)[:, None, None, :]
                b1 = np.asarray(boxes[t1], dtype=np.float64)[:, None, None, :]
                r0, r1 = _map_r0_r1_from_displacements(
                    d0,
                    d1,
                    box0=b0,
                    box1=b1,
                    distance_image=distance_image,
                )
                keep = _distance_window_mask(
                    r0,
                    distance_min_nm=distance_min_nm,
                    distance_max_nm=distance_max_nm,
                )
                keep &= intra_ok[None, None, :]
                dr2 = (r1 - r0) ** 2
                sum_rep += np.sum(np.where(keep, dr2, 0.0), axis=0)
                cnt_rep += np.sum(keep, axis=0, dtype=np.int64)

            rep_values = np.full((n_average_chains, n_pairs), np.nan, dtype=np.float64)
            np.divide(sum_rep, cnt_rep, out=rep_values, where=cnt_rep > 0)
            finite = np.isfinite(rep_values)
            n_rep = np.sum(finite, axis=0, dtype=np.int64)
            sums = np.sum(np.where(finite, rep_values, 0.0), axis=0)
            np.divide(sums, n_rep, out=intra_msd[:, li], where=n_rep > 0)
            intra_counts[:, li] = np.sum(cnt_rep, axis=0, dtype=np.int64)

        # ---- interchain: targets pooled within each reference, refs equal weight
        if do_inter:
            if targets is None:
                raise RuntimeError("interchain target matrix is not initialized")
            if targets.ndim != 2 or targets.shape[0] != n_ch:
                raise RuntimeError("invalid interchain target matrix")

            sum_ref = np.zeros((n_average_chains, n_pairs), dtype=np.float64)
            cnt_ref = np.zeros((n_average_chains, n_pairs), dtype=np.int64)

            for o0 in range(0, int(origins.size), oc):
                t0 = origins[o0 : o0 + oc]
                t1 = t0 + tau
                b0 = np.asarray(boxes[t0], dtype=np.float64)[:, None, None, None, :]
                b1 = np.asarray(boxes[t1], dtype=np.float64)[:, None, None, None, :]

                for r0_index in range(0, n_average_chains, rc):
                    r1_index = min(r0_index + rc, n_average_chains)
                    refs = average_chain_idx[r0_index:r1_index]
                    target_block = targets[refs, :]

                    ref0 = pos[
                        t0[:, None, None, None],
                        refs[None, :, None, None],
                        ri[None, None, None, :],
                        :,
                    ]
                    tar0 = pos[
                        t0[:, None, None, None],
                        target_block[None, :, :, None],
                        rj[None, None, None, :],
                        :,
                    ]
                    ref1 = pos[
                        t1[:, None, None, None],
                        refs[None, :, None, None],
                        ri[None, None, None, :],
                        :,
                    ]
                    tar1 = pos[
                        t1[:, None, None, None],
                        target_block[None, :, :, None],
                        rj[None, None, None, :],
                        :,
                    ]

                    d0 = np.asarray(tar0 - ref0, dtype=np.float64)
                    d1 = np.asarray(tar1 - ref1, dtype=np.float64)
                    rr0, rr1 = _map_r0_r1_from_displacements(
                        d0,
                        d1,
                        box0=b0,
                        box1=b1,
                        distance_image=distance_image,
                    )
                    keep = _distance_window_mask(
                        rr0,
                        distance_min_nm=distance_min_nm,
                        distance_max_nm=distance_max_nm,
                    )
                    dr2 = (rr1 - rr0) ** 2
                    sum_ref[r0_index:r1_index, :] += np.sum(np.where(keep, dr2, 0.0), axis=(0, 2))
                    cnt_ref[r0_index:r1_index, :] += np.sum(keep, axis=(0, 2), dtype=np.int64)

            ref_values = np.full((n_average_chains, n_pairs), np.nan, dtype=np.float64)
            np.divide(sum_ref, cnt_ref, out=ref_values, where=cnt_ref > 0)
            finite = np.isfinite(ref_values)
            n_finite_ref = np.sum(finite, axis=0, dtype=np.int64)
            sums = np.sum(np.where(finite, ref_values, 0.0), axis=0)
            np.divide(sums, n_finite_ref, out=inter_msd[:, li], where=n_finite_ref > 0)
            inter_counts[:, li] = np.sum(cnt_ref, axis=0, dtype=np.int64)

    # Explicitly keep populated lag-zero MSDs at exactly zero.
    if n_lags and int(lags[0]) == 0:
        intra_zero = intra_counts[:, 0] > 0
        inter_zero = inter_counts[:, 0] > 0
        intra_msd[intra_zero, 0] = 0.0
        inter_msd[inter_zero, 0] = 0.0

    return ri, rj, intra_msd, inter_msd, intra_counts, inter_counts


def dist_msd_map_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    chains: Union[str, Sequence[str]] = "protein",
    reference_chains: Optional[Union[str, Sequence[str]]] = None,
    atom_name: str = "CA",
    mode: str = "both",
    distance_image: str = "hybrid",
    dt_ns: float,
    distance_min_nm: Optional[float] = None,
    distance_max_nm: Optional[float] = None,
    min_intra_sequence_separation: int = 2,
    lags_frames: Optional[Sequence[int]] = None,
    max_lag_frames: int = 200,
    lag_stride: int = 1,
    origin_stride: int = 1,
    inter_targets_per_ref: Optional[int] = None,
    random_seed: int = 0,
    backend: str = "serial",
    n_jobs: int = 0,
    mp_start_method: Optional[str] = None,
    pair_block_size: int = 8,
    origin_chunk: int = 16,
    reference_chunk: int = 2,
    work_dtype: str = "float32",
    work_dir: Optional[Union[str, Path]] = None,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> DistMSDMapResult:
    """Calculate distance-conditioned MSD curves for all residue-site pairs.

    This specialized all-pairs path is designed to keep RAM bounded. Selected
    site coordinates are streamed once to a temporary disk-backed memmap. Residue
    pairs are then processed in blocks; no all-pairs trajectory array is held in
    memory. ``backend='process'`` distributes residue-pair blocks across processes,
    all reading the same memmap without duplicating the trajectory coordinates.

    ``distance_min_nm``/``distance_max_nm`` define the single initial-distance
    conditioning window. Either bound may be omitted; with both omitted, all
    finite initial distances contribute. The lower bound is inclusive and the
    upper bound exclusive.

    ``min_intra_sequence_separation`` applies only to intrachain cells. A value of
    1 includes adjacent residues, 2 excludes immediate sequence neighbors, etc.
    Interchain cells include the diagonal and all residue pairs.

    ``reference_chains`` optionally restricts only the chain replicates that are
    averaged. For intrachain MSDs these are the selected physical chains. For
    interchain MSDs these are reference chains; their target chains still come
    from the full ``chains`` selection (excluding self) and are additionally
    conditioned by the initial-distance window. This permits chain-property
    filtering of the averaged references without restricting possible neighbors.
    """
    if float(dt_ns) <= 0.0:
        raise ValueError("dt_ns must be > 0")
    if int(stride) <= 0 or int(chunk) <= 0:
        raise ValueError("stride and chunk must be >=1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >=0")
    if int(origin_stride) <= 0:
        raise ValueError("origin_stride must be >=1")
    if int(pair_block_size) <= 0 or int(origin_chunk) <= 0 or int(reference_chunk) <= 0:
        raise ValueError("pair_block_size, origin_chunk, and reference_chunk must be >=1")
    if int(min_intra_sequence_separation) < 1:
        raise ValueError("min_intra_sequence_separation must be >=1")

    m = str(mode).strip().lower()
    aliases = {
        "intra": "intra",
        "intrachain": "intra",
        "inter": "inter",
        "interchain": "inter",
        "both": "both",
    }
    if m not in aliases:
        raise ValueError("mode must be 'intra', 'inter', or 'both'")
    m = aliases[m]
    do_intra = m in {"intra", "both"}
    do_inter = m in {"inter", "both"}

    di = str(distance_image).strip().lower()
    if di not in {"unwrapped", "min_image", "hybrid"}:
        raise ValueError("distance_image must be 'unwrapped', 'min_image', or 'hybrid'")

    be = str(backend).strip().lower()
    if be not in {"serial", "process"}:
        raise ValueError("backend must be 'serial' or 'process'")

    dmin = None if distance_min_nm is None else float(distance_min_nm)
    dmax = None if distance_max_nm is None else float(distance_max_nm)
    if dmin is not None and (not np.isfinite(dmin) or dmin < 0.0):
        raise ValueError("distance_min_nm must be finite and >=0 or None")
    if dmax is not None and (not np.isfinite(dmax) or dmax <= 0.0):
        raise ValueError("distance_max_nm must be finite and >0 or None")
    if dmin is not None and dmax is not None and dmax <= dmin:
        raise ValueError("distance_max_nm must be greater than distance_min_nm")

    lags = _make_lags(
        max_lag_frames=max_lag_frames,
        lag_stride=int(lag_stride),
        lags_frames=lags_frames,
    )
    if int(lags[-1]) < 0:
        raise ValueError("invalid lag specification")

    dtype = np.dtype(str(work_dtype))
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("work_dtype must be 'float32' or 'float64'")

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model
    local_sites, atom_indices_full, chain_labels, residue_numbers, residue_names = (
        _aligned_map_site_atoms(
            tmpl,
            chains=chains,
            atom_name=atom_name,
        )
    )
    n_ch = int(local_sites.shape[0])
    n_res = int(local_sites.shape[1])
    if do_inter and n_ch < 2:
        raise ValueError("need >=2 selected chains for interchain map")

    # The full ``chains`` selection defines all coordinates and all possible
    # interchain targets.  ``reference_chains`` only chooses which physical chains
    # contribute equal-weight replicate MSDs to the final average.
    if reference_chains is None:
        reference_chain_labels = tuple(chain_labels)
        reference_indices = np.arange(n_ch, dtype=np.int64)
    else:
        (
            _ref_local_sites,
            _ref_atom_indices,
            requested_reference_labels,
            reference_residue_numbers,
            _ref_residue_names,
        ) = _aligned_map_site_atoms(
            tmpl,
            chains=reference_chains,
            atom_name=atom_name,
        )
        if not np.array_equal(reference_residue_numbers, residue_numbers):
            raise ValueError(
                "reference_chains does not resolve to the same residue-site grid "
                "as the full chains selection"
            )
        chain_index_by_label = {str(label): i for i, label in enumerate(chain_labels)}
        missing_reference_labels = [
            str(label)
            for label in requested_reference_labels
            if str(label) not in chain_index_by_label
        ]
        if missing_reference_labels:
            raise ValueError(
                "reference_chains must be a subset of chains; missing labels: "
                f"{missing_reference_labels}"
            )
        reference_chain_labels = tuple(str(label) for label in requested_reference_labels)
        reference_indices = np.asarray(
            [chain_index_by_label[label] for label in reference_chain_labels],
            dtype=np.int64,
        )
        if reference_indices.size == 0:
            raise ValueError("reference_chains selects no chains")

    # Inter targets are sampled once and reused for every residue pair, preserving
    # comparable statistics across the entire map.
    if do_inter:
        pi, pj, _ = _sample_inter_pairs(
            n_ch,
            targets_per_ref=inter_targets_per_ref,
            seed=int(random_seed),
        )
        if (pi.size % n_ch) != 0:
            raise RuntimeError("internal inter target sampling error")
        n_targets = int(pi.size // n_ch)
        targets_matrix = np.asarray(pj, dtype=np.int64).reshape(n_ch, n_targets)
    else:
        targets_matrix = np.empty((n_ch, 0), dtype=np.int64)

    upper_i: list[int] = []
    upper_j: list[int] = []
    intra_flags: list[bool] = []
    min_sep = int(min_intra_sequence_separation)
    for i in range(n_res):
        for j in range(i, n_res):
            upper_i.append(i)
            upper_j.append(j)
            intra_flags.append(
                bool(i != j and abs(int(residue_numbers[j]) - int(residue_numbers[i])) >= min_sep)
            )

    pair_i_all = np.asarray(upper_i, dtype=np.int64)
    pair_j_all = np.asarray(upper_j, dtype=np.int64)
    intra_all = np.asarray(intra_flags, dtype=bool)

    n_lags = int(lags.size)
    intra_map = np.full((n_res, n_res, n_lags), np.nan, dtype=np.float32)
    inter_map = np.full((n_res, n_res, n_lags), np.nan, dtype=np.float32)
    intra_counts_map = np.zeros((n_res, n_res, n_lags), dtype=np.int64)
    inter_counts_map = np.zeros((n_res, n_res, n_lags), dtype=np.int64)

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)
    unwrap = di in {"unwrapped", "hybrid"}

    temp_parent = None if work_dir is None else os.fspath(work_dir)
    if temp_parent is not None:
        os.makedirs(temp_parent, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="distmsd_map_", dir=temp_parent) as temp_dir:
        pos_path, box_path, n_frames, pos_shape = _stream_map_sites_to_memmap(
            dcd_list,
            tmpl_model,
            atom_indices_full=atom_indices_full,
            local_sites=local_sites,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=int(frame_start),
            frame_stop=frame_stop,
            box_fallback=box_fallback,
            unwrap=unwrap,
            directory=temp_dir,
            dtype=dtype,
        )
        if int(lags[-1]) >= n_frames:
            raise ValueError(
                f"max lag {int(lags[-1])} frames is not smaller than the selected "
                f"trajectory length ({n_frames} frames)"
            )

        block_size = int(pair_block_size)
        tasks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for p0 in range(0, int(pair_i_all.size), block_size):
            p1 = min(p0 + block_size, int(pair_i_all.size))
            tasks.append((pair_i_all[p0:p1], pair_j_all[p0:p1], intra_all[p0:p1]))

        initargs = (
            pos_path,
            pos_shape,
            dtype.str,
            box_path,
            (int(n_frames), 3),
            targets_matrix,
        )

        def consume(
            result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
            ri, rj, intra_block, inter_block, intra_count_block, inter_count_block = result
            for k in range(int(ri.size)):
                i = int(ri[k])
                j = int(rj[k])
                if do_inter:
                    inter_map[i, j, :] = np.asarray(inter_block[k], dtype=np.float32)
                    inter_counts_map[i, j, :] = inter_count_block[k]
                if do_intra and bool(
                    i != j and abs(int(residue_numbers[j]) - int(residue_numbers[i])) >= min_sep
                ):
                    # Put intrachain values in the lower triangle to match contact maps.
                    intra_map[j, i, :] = np.asarray(intra_block[k], dtype=np.float32)
                    intra_counts_map[j, i, :] = intra_count_block[k]

        if be == "serial":
            _map_init_memmaps(*initargs)
            try:
                for ri, rj, intra_flag in tasks:
                    consume(
                        _map_process_pair_block(
                            ri,
                            rj,
                            intra_flag,
                            lags=lags,
                            origin_stride=int(origin_stride),
                            distance_image=di,
                            distance_min_nm=dmin,
                            distance_max_nm=dmax,
                            do_intra=do_intra,
                            do_inter=do_inter,
                            origin_chunk=int(origin_chunk),
                            reference_chunk=int(reference_chunk),
                            reference_indices=reference_indices,
                        )
                    )
            finally:
                global _MAP_POS_TS, _MAP_BOX_TS, _MAP_TARGETS
                _MAP_POS_TS = None
                _MAP_BOX_TS = None
                _MAP_TARGETS = None
        else:
            jobs = int(n_jobs)
            if jobs <= 0:
                jobs = _cpu_count()
            jobs = max(1, min(jobs, len(tasks)))
            method = _pick_start_method(mp_start_method)
            ctx = mp.get_context(method)
            with ProcessPoolExecutor(
                max_workers=jobs,
                mp_context=ctx,
                initializer=_map_init_memmaps,
                initargs=initargs,
            ) as ex:
                futures = [
                    ex.submit(
                        _map_process_pair_block,
                        ri,
                        rj,
                        intra_flag,
                        lags=lags,
                        origin_stride=int(origin_stride),
                        distance_image=di,
                        distance_min_nm=dmin,
                        distance_max_nm=dmax,
                        do_intra=do_intra,
                        do_inter=do_inter,
                        origin_chunk=int(origin_chunk),
                        reference_chunk=int(reference_chunk),
                        reference_indices=reference_indices,
                    )
                    for ri, rj, intra_flag in tasks
                ]
                for fut in as_completed(futures):
                    consume(fut.result())

    return DistMSDMapResult(
        t_ns=np.asarray(lags, dtype=np.float64) * float(dt_ns),
        lags_frames=np.asarray(lags, dtype=np.int64),
        residue_numbers=np.asarray(residue_numbers, dtype=np.int64),
        residue_names=tuple(residue_names),
        atom_name=str(atom_name).strip().upper(),
        chain_labels=tuple(chain_labels),
        intra_msd_nm2=intra_map,
        inter_msd_nm2=inter_map,
        intra_counts=intra_counts_map,
        inter_counts=inter_counts_map,
        n_frames=int(n_frames),
        dt_ns=float(dt_ns),
        origin_stride=int(origin_stride),
        distance_image=di,
        distance_min_nm=dmin,
        distance_max_nm=dmax,
        min_intra_sequence_separation=min_sep,
        inter_targets_per_ref=(
            None if inter_targets_per_ref is None else int(inter_targets_per_ref)
        ),
        random_seed=int(random_seed),
        pair_block_size=int(pair_block_size),
        origin_chunk=int(origin_chunk),
        reference_chunk=int(reference_chunk),
        reference_chain_labels=reference_chain_labels,
        aggregation=(
            "reference_replicates_equal_weight"
            if len(reference_chain_labels) == len(chain_labels)
            else "selected_reference_replicates_equal_weight"
        ),
    )


def _fit_map_array_linear(
    values: np.ndarray,
    t_ns: np.ndarray,
    *,
    fit_tmin_ns: float,
    fit_tmax_ns: float,
    dims: int,
) -> np.ndarray:
    """Vectorized independent linear fits for the final axis of a map array."""
    y = np.asarray(values, dtype=np.float64)
    t = np.asarray(t_ns, dtype=np.float64).reshape(-1)
    if y.ndim != 3 or y.shape[-1] != t.size:
        raise ValueError("map MSD array must have shape (n_res,n_res,n_lags)")
    if int(dims) <= 0:
        raise ValueError("dims must be >=1")
    tmin = float(fit_tmin_ns)
    tmax = float(fit_tmax_ns)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        raise ValueError("fit_tmax_ns must be greater than fit_tmin_ns")

    time_ok = np.isfinite(t) & (t >= tmin) & (t <= tmax)
    if int(np.sum(time_ok)) < 2:
        raise ValueError("fit window selects fewer than two lag times")

    y_use = y[..., time_ok]
    x = t[time_ok]
    finite = np.isfinite(y_use)
    n = np.sum(finite, axis=-1, dtype=np.int64)
    x_b = x.reshape((1, 1, -1))
    sx = np.sum(np.where(finite, x_b, 0.0), axis=-1)
    sy = np.sum(np.where(finite, y_use, 0.0), axis=-1)
    sxx = np.sum(np.where(finite, x_b * x_b, 0.0), axis=-1)
    sxy = np.sum(np.where(finite, x_b * y_use, 0.0), axis=-1)

    nf = n.astype(np.float64)
    denom = nf * sxx - sx * sx
    slope = np.full(n.shape, np.nan, dtype=np.float64)
    valid = (n >= 2) & np.isfinite(denom) & (np.abs(denom) > 0.0)
    numerator = nf * sxy - sx * sy
    np.divide(numerator, denom, out=slope, where=valid)
    return slope / (2.0 * float(dims))


def _mean_sem_maps(per_set: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-set mean, SEM, and contributor count for map arrays."""
    data = np.asarray(per_set, dtype=np.float64)
    finite = np.isfinite(data)
    n = np.sum(finite, axis=0, dtype=np.int64)
    sums = np.sum(np.where(finite, data, 0.0), axis=0)
    mean = np.full(data.shape[1:], np.nan, dtype=np.float64)
    np.divide(sums, n, out=mean, where=n > 0)

    sem = np.full(data.shape[1:], np.nan, dtype=np.float64)
    enough = n > 1
    if np.any(enough):
        deviations = np.where(finite, data - mean[None, ...], 0.0)
        ss = np.sum(deviations * deviations, axis=0)
        variance = np.zeros_like(mean)
        np.divide(ss, n - 1, out=variance, where=enough)
        sem[enough] = np.sqrt(variance[enough] / n[enough])
    return mean, sem, n


def fit_dist_msd_map_linear(
    data: Union[DistMSDMapResult, Mapping[str, DistMSDMapResult]],
    *,
    fit_tmin_ns: float,
    fit_tmax_ns: float,
    dims: int = 1,
    min_sets: int = 1,
) -> DistDiffusionMapFitResult:
    """Fit and, for mapping input, equal-weight average all-pairs D maps.

    Each simulation set is fitted independently. For multiple sets the final
    color value in each cell is the equal-weight mean of the finite set-level D
    values; uncertainty is the SEM across independent sets. ``min_sets`` controls
    how many finite set fits are required for a cell to be retained.
    """
    if isinstance(min_sets, (bool, np.bool_)):
        raise TypeError("min_sets must be an integer >=1")
    min_sets_i = int(min_sets)
    if min_sets_i < 1 or float(min_sets_i) != float(min_sets):
        raise ValueError("min_sets must be an integer >=1")

    if isinstance(data, Mapping):
        if not data:
            raise ValueError("data mapping is empty")
        labels = tuple(str(k) for k in data)
        members = list(data.values())
    else:
        labels = ()
        members = [data]

    first = members[0]
    required = (
        "t_ns",
        "residue_numbers",
        "intra_msd_nm2",
        "inter_msd_nm2",
        "distance_image",
        "distance_min_nm",
        "distance_max_nm",
        "min_intra_sequence_separation",
        "atom_name",
    )
    if not all(hasattr(first, field) for field in required):
        raise TypeError("data must contain DistMSDMapResult-compatible objects")

    first_res = np.asarray(first.residue_numbers, dtype=np.int64)
    first_t = np.asarray(first.t_ns, dtype=np.float64)
    for label, member in zip(labels[1:] if labels else (), members[1:]):
        if not np.array_equal(np.asarray(member.residue_numbers, dtype=np.int64), first_res):
            raise ValueError(f"{label!r}: residue grid differs from the first set")
        if not np.allclose(
            np.asarray(member.t_ns, dtype=np.float64), first_t, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(f"{label!r}: lag-time grid differs from the first set")
        for field in (
            "distance_image",
            "distance_min_nm",
            "distance_max_nm",
            "min_intra_sequence_separation",
            "atom_name",
        ):
            if getattr(member, field) != getattr(first, field):
                raise ValueError(
                    f"{label!r}: map metadata field {field!r} differs from the first set"
                )

    if min_sets_i > len(members):
        raise ValueError(f"min_sets={min_sets_i} exceeds supplied sets ({len(members)})")

    per_intra = np.stack(
        [
            _fit_map_array_linear(
                member.intra_msd_nm2,
                member.t_ns,
                fit_tmin_ns=fit_tmin_ns,
                fit_tmax_ns=fit_tmax_ns,
                dims=dims,
            )
            for member in members
        ],
        axis=0,
    )
    per_inter = np.stack(
        [
            _fit_map_array_linear(
                member.inter_msd_nm2,
                member.t_ns,
                fit_tmin_ns=fit_tmin_ns,
                fit_tmax_ns=fit_tmax_ns,
                dims=dims,
            )
            for member in members
        ],
        axis=0,
    )

    intra_mean, intra_sem, intra_n = _mean_sem_maps(per_intra)
    inter_mean, inter_sem, inter_n = _mean_sem_maps(per_inter)
    intra_mean[intra_n < min_sets_i] = np.nan
    intra_sem[intra_n < min_sets_i] = np.nan
    inter_mean[inter_n < min_sets_i] = np.nan
    inter_sem[inter_n < min_sets_i] = np.nan

    n_res = int(first_res.size)
    combined = np.full((n_res, n_res), np.nan, dtype=np.float64)
    combined_sem = np.full((n_res, n_res), np.nan, dtype=np.float64)
    lower = np.tril(np.ones((n_res, n_res), dtype=bool), k=-1)
    upper = np.triu(np.ones((n_res, n_res), dtype=bool), k=0)
    combined[lower] = intra_mean[lower]
    combined[upper] = inter_mean[upper]
    combined_sem[lower] = intra_sem[lower]
    combined_sem[upper] = inter_sem[upper]

    return DistDiffusionMapFitResult(
        residue_numbers=first_res.copy(),
        residue_names=tuple(first.residue_names),
        atom_name=str(first.atom_name),
        intra_d_nm2_per_ns=intra_mean,
        inter_d_nm2_per_ns=inter_mean,
        combined_d_nm2_per_ns=combined,
        intra_stderr_nm2_per_ns=intra_sem,
        inter_stderr_nm2_per_ns=inter_sem,
        combined_stderr_nm2_per_ns=combined_sem,
        intra_n_sets=intra_n,
        inter_n_sets=inter_n,
        fit_tmin_ns=float(fit_tmin_ns),
        fit_tmax_ns=float(fit_tmax_ns),
        dims=int(dims),
        distance_image=str(first.distance_image),
        distance_min_nm=first.distance_min_nm,
        distance_max_nm=first.distance_max_nm,
        min_intra_sequence_separation=int(first.min_intra_sequence_separation),
        set_labels=labels,
        n_sets=len(members),
        per_set_intra_d_nm2_per_ns=per_intra,
        per_set_inter_d_nm2_per_ns=per_inter,
        aggregation="sets_equal_weight" if len(members) > 1 else "single_set",
    )


# More descriptive alias.
fit_dist_diffusion_map = fit_dist_msd_map_linear

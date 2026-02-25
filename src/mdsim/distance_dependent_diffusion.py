from __future__ import annotations

import io
import multiprocessing as mp
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional, Union

import numpy as np
from openmm.unit import Quantity, nanometer

from .molecule_data import PDBReader, iter_dcd

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
    res_i: int
    res_j: int
    atom_name: str
    inter_targets_per_ref: Optional[int]
    random_seed: int


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
    res_i: int
    res_j: int
    atom_name: str


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


def fit_dist_msd_linear_bins(
    msd: DistMSDBinnedResult,
    *,
    fit_tmin_ns: float,
    fit_tmax_ns: float,
    dims: int = 1,
    use_rep_sem: bool = True,
) -> DistDiffusionFitResult:
    if int(dims) <= 0:
        raise ValueError("dims must be >= 1")
    t = msd.t_ns
    tmin = float(fit_tmin_ns)
    tmax = float(fit_tmax_ns)
    if tmax <= tmin:
        raise ValueError("fit_tmax_ns must be > fit_tmin_ns")
    sel = (t >= tmin) & (t <= tmax)
    if int(np.sum(sel)) < 2:
        raise ValueError("fit window selects <2 points")

    n_bins = int(msd.r0_centers_nm.size)
    slope = np.empty((n_bins,), dtype=np.float64)
    intercept = np.empty((n_bins,), dtype=np.float64)
    d = np.empty((n_bins,), dtype=np.float64)
    d_err = np.zeros((n_bins,), dtype=np.float64)

    for b in range(n_bins):
        s, itc = _linear_fit(t[sel], msd.msd_nm2[b, sel])
        slope[b] = s
        intercept[b] = itc
        d[b] = s / (2.0 * float(dims))

    if use_rep_sem and msd.n_replicates >= 2:
        d_rep = np.full((msd.n_replicates, n_bins), np.nan, dtype=np.float64)
        for r in range(msd.n_replicates):
            y = msd.msd_rep_nm2[r]
            for b in range(n_bins):
                if not np.all(np.isfinite(y[b, sel])):
                    continue
                s, _ = _linear_fit(t[sel], y[b, sel])
                d_rep[r, b] = s / (2.0 * float(dims))

        ok = np.isfinite(d_rep)
        n_eff = np.sum(ok, axis=0).astype(np.float64)
        d_std = np.nanstd(d_rep, axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d_err = np.where(n_eff > 1.0, d_std / np.sqrt(n_eff), 0.0)

    return DistDiffusionFitResult(
        r0_centers_nm=msd.r0_centers_nm,
        d_nm2_per_ns=d,
        d_stderr_nm2_per_ns=d_err,
        slope_nm2_per_ns=slope,
        intercept_nm2=intercept,
        fit_tmin_ns=tmin,
        fit_tmax_ns=tmax,
        dims=int(dims),
        mode=msd.mode,
        res_i=msd.res_i,
        res_j=msd.res_j,
        atom_name=msd.atom_name,
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
    sum_rep = np.asarray(sum_rep, dtype=np.float64)
    cnt_rep = np.asarray(cnt_rep, dtype=np.int64)

    msd_rep = np.full_like(sum_rep, np.nan, dtype=np.float64)
    np.divide(sum_rep, cnt_rep, out=msd_rep, where=cnt_rep > 0)

    msd = np.nanmean(msd_rep, axis=0)
    n_eff = np.sum(np.isfinite(msd_rep), axis=0).astype(np.float64)

    if int(sum_rep.shape[0]) < 2:
        err = np.zeros_like(msd, dtype=np.float64)
        return msd, err, msd_rep, cnt_rep.sum(axis=0)

    std = np.nanstd(msd_rep, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.where(n_eff > 1.0, std / np.sqrt(n_eff), 0.0)

    return msd, err, msd_rep, cnt_rep.sum(axis=0)


def dist_msd_binned_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    res_i: int,
    res_j: int,
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
    Distance-dependent MSD for the 1D distance r(t) between residues (res_i,res_j).

    Modes
    -----
    - intra: r(t) within each chain (replicate = chain)
    - inter: r(t) between chains, residue i on reference chain, residue j on target chain
             (replicate = reference chain)

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

    idx_i, idx_j, atom_indices_full = _aligned_chain_atoms(
        tmpl,
        res_i=int(res_i),
        res_j=int(res_j),
        atom_name=str(atom_name),
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
        atom = str(atom_name).strip().upper()

        intra_res = None
        inter_res = None

        if have_intra and sum_intra is not None and cnt_intra is not None:
            msd, err, msd_rep, counts = _finalize_binned_msd(
                sum_rep=sum_intra,
                cnt_rep=cnt_intra,
            )
            msd[:, 0] = 0.0
            err[:, 0] = 0.0
            msd_rep[:, :, 0] = 0.0

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
                res_i=int(res_i),
                res_j=int(res_j),
                atom_name=atom,
                inter_targets_per_ref=None,
                random_seed=int(random_seed),
            )

        msd, err, msd_rep, counts = _finalize_binned_msd(
            sum_rep=sum_inter,
            cnt_rep=cnt_inter,
        )
        msd[:, 0] = 0.0
        err[:, 0] = 0.0
        msd_rep[:, :, 0] = 0.0

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
            res_i=int(res_i),
            res_j=int(res_j),
            atom_name=atom,
            inter_targets_per_ref=(
                None if inter_targets_per_ref is None else int(inter_targets_per_ref)
            ),
            random_seed=int(random_seed),
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
    atom = str(atom_name).strip().upper()

    intra_res = None
    inter_res = None

    if have_intra:
        msd, err, msd_rep, counts = _finalize_binned_msd(
            sum_rep=sum_intra,  # type: ignore[arg-type]
            cnt_rep=cnt_intra,  # type: ignore[arg-type]
        )
        msd[:, 0] = 0.0
        err[:, 0] = 0.0
        msd_rep[:, :, 0] = 0.0

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
            res_i=int(res_i),
            res_j=int(res_j),
            atom_name=atom,
            inter_targets_per_ref=None,
            random_seed=int(random_seed),
        )

    if have_inter:
        msd, err, msd_rep, counts = _finalize_binned_msd(
            sum_rep=sum_inter,  # type: ignore[arg-type]
            cnt_rep=cnt_inter,  # type: ignore[arg-type]
        )
        msd[:, 0] = 0.0
        err[:, 0] = 0.0
        msd_rep[:, :, 0] = 0.0

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
            res_i=int(res_i),
            res_j=int(res_j),
            atom_name=atom,
            inter_targets_per_ref=inter_targets_per_ref,
            random_seed=int(random_seed),
        )

    return DistMSDPairResult(intra=intra_res, inter=inter_res)


def dist_msd_binned_multi_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    residue_pairs: Sequence[tuple[int, int]],
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

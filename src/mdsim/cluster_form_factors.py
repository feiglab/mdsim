from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np

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


def _sinc(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    m = np.abs(x) <= float(eps)
    out[m] = 1.0
    out[~m] = np.sin(x[~m]) / x[~m]
    return out


def _box_lengths_nm(box_nm: Any) -> np.ndarray:
    if box_nm is None:
        raise ValueError("box_nm is required (DCD must have unit cell lengths or pass box_nm)")
    arr = np.asarray(box_nm, dtype=np.float64).reshape(3)
    if np.any(arr <= 0.0):
        raise ValueError("box lengths must be positive")
    return arr


def _min_image_disp_nm(d: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    return d - np.rint(d / box_nm.reshape(1, 3)) * box_nm.reshape(1, 3)


def _wrap_nm(xyz_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    b = box_nm.reshape(1, 3)
    return xyz_nm - np.floor(xyz_nm / b) * b


def _atomic_weights_for_selection(
    template_model: Any,
    atom_indices_full: list[int],
    *,
    weights: str,
) -> np.ndarray:
    mode = str(weights).strip().lower()
    if mode not in {"unity", "z"}:
        raise ValueError("weights must be 'unity' or 'z'")

    if mode == "unity":
        return np.ones(len(atom_indices_full), dtype=np.float64)

    w = np.empty(len(atom_indices_full), dtype=np.float64)
    for i, ai in enumerate(atom_indices_full):
        el = getattr(template_model.atoms[int(ai)], "element", "") or ""
        key = str(el).strip().upper()
        w[i] = float(_Z_BY_EL.get(key, 6))
    return w


def _protein_groups_from_selection(
    pdb_file: FileLike,
    selection: Union[str, list[list[int]]],
) -> tuple[Any, list[np.ndarray], list[int], dict[int, int]]:
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

    return tmpl_model, groups_sel, atom_indices, {"n_proteins": n_prot}  # type: ignore[return-value]


def _cluster_unwrap_coords_nm(
    xyz_sel_nm: np.ndarray,
    box_nm: np.ndarray,
    prot_groups_sel: list[np.ndarray],
    cluster_prot_ids: list[int],
) -> np.ndarray:
    """
    Make a cluster contiguous by translating each protein by a minimum-image shift
    relative to the first protein in the cluster.
    """
    box = _box_lengths_nm(box_nm)
    xyz = _wrap_nm(np.asarray(xyz_sel_nm, dtype=np.float64), box).copy()

    # Centers computed from wrapped coords (COG is fine for shift vectors)
    centers = np.empty((len(cluster_prot_ids), 3), dtype=np.float64)
    for k, pid in enumerate(cluster_prot_ids):
        idx = prot_groups_sel[int(pid)].astype(np.int64)
        centers[k] = xyz[idx].mean(axis=0)

    ref = centers[0:1]
    disp = centers - ref
    disp = _min_image_disp_nm(disp, box)
    centers_unwrapped = ref + disp

    for k, pid in enumerate(cluster_prot_ids):
        shift = centers_unwrapped[k] - centers[k]
        idx = prot_groups_sel[int(pid)].astype(np.int64)
        xyz[idx] += shift

    return xyz


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


# ---------------------------
# Parallel frame processing
# ---------------------------

# The ProcessPoolExecutor initializer stores these read-only objects in each worker.
_G_PROT_GROUPS_SEL: Optional[list[np.ndarray]] = None
_G_PROT_LENS: Optional[np.ndarray] = None
_G_W_SEL: Optional[np.ndarray] = None
_G_Q: Optional[np.ndarray] = None
_G_ATOM_BLOCK: int = 512
_G_Q_BLOCK: int = 64
_G_BLAS_THREADS: int = 1


def _init_frame_worker(
    prot_groups_sel: list[np.ndarray],
    prot_lens: np.ndarray,
    w_sel: np.ndarray,
    q_nm1: np.ndarray,
    atom_block: int,
    q_block: int,
    blas_threads: int,
) -> None:
    global _G_PROT_GROUPS_SEL, _G_PROT_LENS, _G_W_SEL, _G_Q
    global _G_ATOM_BLOCK, _G_Q_BLOCK, _G_BLAS_THREADS

    _G_PROT_GROUPS_SEL = prot_groups_sel
    _G_PROT_LENS = np.asarray(prot_lens, dtype=np.int64)
    _G_W_SEL = np.asarray(w_sel, dtype=np.float64)
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
    w_sel: np.ndarray,
    q_nm1: np.ndarray,
    *,
    atom_block: int,
    q_block: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    """
    Compute partial accumulators for a batch of frames.

    Returns:
        sums[m]  = sum over cluster-instances of size m of (I(q)/m)
        sums2[m] = sum over cluster-instances of size m of (I(q)/m)^2
        counts[m]= number of cluster-instances of size m
    """
    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)

    sums: dict[int, np.ndarray] = defaultdict(lambda: np.zeros_like(q))
    sums2: dict[int, np.ndarray] = defaultdict(lambda: np.zeros_like(q))
    counts: dict[int, int] = defaultdict(int)

    xyz_batch_nm = np.asarray(xyz_batch_nm)
    box_batch_nm = np.asarray(box_batch_nm, dtype=np.float64)

    if xyz_batch_nm.ndim != 3 or xyz_batch_nm.shape[-1] != 3:
        raise ValueError("xyz_batch_nm must have shape (n_frames, n_atoms, 3)")
    if box_batch_nm.ndim != 2 or box_batch_nm.shape[-1] != 3:
        raise ValueError("box_batch_nm must have shape (n_frames, 3)")
    if xyz_batch_nm.shape[0] != box_batch_nm.shape[0]:
        raise ValueError("xyz_batch_nm and box_batch_nm must have the same n_frames")

    n_frames = int(xyz_batch_nm.shape[0])
    n_prot = int(len(prot_groups_sel))

    for fi in range(n_frames):
        b = _box_lengths_nm(box_batch_nm[fi])
        xyz_wr = _wrap_nm(np.asarray(xyz_batch_nm[fi], dtype=np.float64), b)

        # Protein centers from wrapped coords (COG is sufficient for minimum-image shifts)
        centers = np.empty((n_prot, 3), dtype=np.float64)
        for pid, idx in enumerate(prot_groups_sel):
            centers[pid] = xyz_wr[idx].mean(axis=0)

        frame_clusters = clusters_batch[fi]
        for cl in frame_clusters:
            if not cl:
                continue

            prot_ids = np.asarray(cl, dtype=np.int32)
            m = int(prot_ids.size)
            if m < 1:
                continue

            c = centers[prot_ids]  # (m,3)
            ref = c[0:1]
            disp = c - ref
            disp = _min_image_disp_nm(disp, b)
            # shift per protein: centers_unwrapped - centers_wrapped
            shifts = (ref + disp) - c  # (m,3)

            n_atoms_cl = int(np.sum(prot_lens[prot_ids], dtype=np.int64))
            if n_atoms_cl < 1:
                continue

            x_cl = np.empty((n_atoms_cl, 3), dtype=np.float64)
            w_cl = np.empty((n_atoms_cl,), dtype=np.float64)

            pos = 0
            for k, pid in enumerate(prot_ids.tolist()):
                idx = prot_groups_sel[int(pid)]
                n_i = int(idx.size)
                x_cl[pos : pos + n_i] = xyz_wr[idx] + shifts[k]
                w_cl[pos : pos + n_i] = w_sel[idx]
                pos += n_i

            i_q = debye_intensity_nm(
                x_cl,
                q,
                w_cl,
                atom_block=atom_block,
                q_block=q_block,
            )
            i_per = i_q / float(m)

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
    if _G_PROT_GROUPS_SEL is None or _G_PROT_LENS is None or _G_W_SEL is None or _G_Q is None:
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
            _G_W_SEL,
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
            _G_W_SEL,
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
    w_sel = _atomic_weights_for_selection(tmpl_model, atom_indices_full, weights=weights)

    if q_nm1 is None:
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        q = np.linspace(float(q_min_nm1), float(q_max_nm1), int(n_q), dtype=np.float64)
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")

    sums: dict[int, np.ndarray] = defaultdict(lambda: np.zeros_like(q))
    sums2: dict[int, np.ndarray] = defaultdict(lambda: np.zeros_like(q))
    counts: dict[int, int] = defaultdict(int)

    def _merge_partials(
        psums: dict[int, np.ndarray],
        psums2: dict[int, np.ndarray],
        pcounts: dict[int, int],
    ) -> None:
        for m, v in psums.items():
            sums[int(m)] += np.asarray(v, dtype=np.float64)
        for m, v in psums2.items():
            sums2[int(m)] += np.asarray(v, dtype=np.float64)
        for m, c in pcounts.items():
            counts[int(m)] += int(c)

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
                    w_sel,
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
                    w_sel,
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
                        w_sel,
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
                        w_sel,
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

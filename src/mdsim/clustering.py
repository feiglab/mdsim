from __future__ import annotations

import io
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .analysis import (
    _as_file_list,
    _box_lengths_nm,
    _fmt_hms,
    _progress_print,
    _selection_to_groups,
)
from .molecule_data import PDBReader, iter_dcd

FileLike = Union[str, Path, io.BytesIO, io.StringIO]


def _is_hydrogen_like(atom: Any) -> bool:
    """
    Best-effort hydrogen classifier for template_model atoms.
    Uses atom.element when available; falls back to atom.name patterns.
    """
    el = (getattr(atom, "element", "") or "").strip().upper()
    if el == "H":
        return True
    name = (getattr(atom, "name", "") or "").strip().upper()
    if not name:
        return False
    if name[0] == "H":
        return True
    if len(name) >= 2 and name[0].isdigit() and name[1] == "H":
        return True
    return False


def _wrap_positions_nm(xyz_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    """Wrap positions into [0, L) for orthorhombic PBC."""
    b = box_nm.reshape(1, 3)
    return xyz_nm - np.floor(xyz_nm / b) * b


def _min_image_disp_nm(d: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    """Minimum-image displacement for orthorhombic PBC."""
    return d - np.rint(d / box_nm.reshape(1, 3)) * box_nm.reshape(1, 3)


def _build_cells(
    xyz_nm_wrapped: np.ndarray,
    box_nm: np.ndarray,
    cell_size_nm: float,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Assign atoms to a 3D cell grid for neighbor search.

    Returns
    -------
    ncell : (3,) int
        number of cells along x,y,z
    cell_idx : (n_atoms, 3) int
        integer cell indices for each atom
    bins : list[list[int]]
        flattened cell -> list of atom indices
    """
    Lx, Ly, Lz = [float(v) for v in box_nm.reshape(3)]
    if cell_size_nm <= 0.0:
        raise ValueError("cell_size_nm must be > 0")

    # Ensure at least 1 cell per dimension
    nx = max(1, int(np.floor(Lx / cell_size_nm)))
    ny = max(1, int(np.floor(Ly / cell_size_nm)))
    nz = max(1, int(np.floor(Lz / cell_size_nm)))
    ncell = np.array([nx, ny, nz], dtype=np.int64)

    # Map position -> cell index
    frac = xyz_nm_wrapped / box_nm.reshape(1, 3)  # [0,1)
    c = np.floor(frac * ncell.reshape(1, 3)).astype(np.int64)
    # Guard numerical edge cases that land on nx
    c = np.minimum(c, (ncell - 1).reshape(1, 3))

    # Flattened bin list
    n_bins = int(nx * ny * nz)
    bins: list[list[int]] = [[] for _ in range(n_bins)]

    flat = (c[:, 0] * (ny * nz) + c[:, 1] * nz + c[:, 2]).astype(np.int64)
    for ai, bi in enumerate(flat.tolist()):
        bins[int(bi)].append(int(ai))

    return ncell, c, bins


def _neighbor_cell_offsets() -> list[tuple[int, int, int]]:
    """Offsets for the 27 neighbor cells (including itself)."""
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out.append((dx, dy, dz))
    return out


_NEIGHBOR_OFFSETS_27 = _neighbor_cell_offsets()


def _union_find_init(n: int) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)
    return parent, rank


def _union_find_find(parent: np.ndarray, i: int) -> int:
    # Path compression
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = int(parent[i])
    return int(i)


def _union_find_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = _union_find_find(parent, a)
    rb = _union_find_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def _clusters_from_edges(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent, rank = _union_find_init(n_nodes)
    for i, j in edges:
        _union_find_union(parent, rank, int(i), int(j))

    comp: dict[int, list[int]] = {}
    for i in range(n_nodes):
        r = _union_find_find(parent, i)
        comp.setdefault(r, []).append(i)

    # deterministic ordering: clusters by smallest member, members sorted
    clusters = [sorted(v) for v in comp.values()]
    clusters.sort(key=lambda x: x[0] if x else 10**9)
    return clusters


def _contacts_and_clusters_one_frame(
    xyz_heavy_nm: np.ndarray,  # (n_heavy,3) nm, in template-selected indexing
    box_nm: np.ndarray,  # (3,) nm
    atom_to_protein: np.ndarray,  # (n_heavy,) int protein id
    n_proteins: int,
    *,
    dist_cutoff_nm: float,
    contact_threshold: int,
    cell_size_nm: Optional[float] = None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], list[list[int]]]:
    """
    Compute protein-protein contact edges and clusters for a single frame.

    A protein-protein "contact" edge exists if the number of heavy-atom pairs
    (a in i, b in j) with min-image distance <= dist_cutoff_nm is >= contact_threshold.
    """
    if dist_cutoff_nm <= 0.0:
        raise ValueError("dist_cutoff_nm must be > 0")
    if contact_threshold <= 0:
        raise ValueError("contact_threshold must be > 0")

    # Wrap coordinates
    xyz = _wrap_positions_nm(np.asarray(xyz_heavy_nm, dtype=np.float64), box_nm)

    # Cell list
    cs = float(dist_cutoff_nm if cell_size_nm is None else cell_size_nm)
    ncell, cell_idx, bins = _build_cells(xyz, box_nm, cs)
    nx, ny, nz = [int(v) for v in ncell.tolist()]

    # Count heavy-atom contacts per protein pair
    # store only upper-triangular (i<j)
    counts: dict[tuple[int, int], int] = {}

    cut2 = float(dist_cutoff_nm * dist_cutoff_nm)

    def flat_cell(ix: int, iy: int, iz: int) -> int:
        return int(ix * (ny * nz) + iy * nz + iz)

    # Iterate over cells; for each cell, consider itself and neighbor cells
    # To avoid double counting, only consider neighbor cells in a fixed ordering:
    # (neighbor_flat > current_flat) OR (same cell, enforce atom index j>i).
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                c0 = flat_cell(ix, iy, iz)
                atoms0 = bins[c0]
                if not atoms0:
                    continue

                # Pre-fetch positions and protein ids for atoms0
                a0 = np.asarray(atoms0, dtype=np.int64)
                x0 = xyz[a0, :]
                p0 = atom_to_protein[a0]

                seen: set[int] = set()
                for dx, dy, dz in _NEIGHBOR_OFFSETS_27:
                    jx = (ix + dx) % nx
                    jy = (iy + dy) % ny
                    jz = (iz + dz) % nz
                    c1 = flat_cell(jx, jy, jz)
                    if c1 in seen:
                        continue
                    seen.add(c1)

                    # ordering to avoid double counting
                    if c1 < c0:
                        continue

                    atoms1 = bins[c1]
                    if not atoms1:
                        continue

                    a1 = np.asarray(atoms1, dtype=np.int64)
                    x1 = xyz[a1, :]
                    p1 = atom_to_protein[a1]

                    # Same cell: only pairs with index in atoms1 list with global a1 > a0
                    if c1 == c0:
                        # all pairs within atoms0
                        # compute upper triangle distances efficiently by brute force for this cell
                        # (cells are usually small if cutoff is small)
                        m = len(a0)
                        for ii in range(m - 1):
                            pi = int(p0[ii])
                            ri = x0[ii : ii + 1, :]  # (1,3)
                            # candidates jj > ii
                            rj = x0[ii + 1 :, :]
                            pj = p0[ii + 1 :]
                            d = rj - ri
                            d = _min_image_disp_nm(d, box_nm)
                            dist2 = np.einsum("ij,ij->i", d, d)
                            hit = dist2 <= cut2
                            if not np.any(hit):
                                continue
                            for pj_val in pj[hit]:
                                pj_int = int(pj_val)
                                if pj_int == pi:
                                    continue
                                i_prot = pi if pi < pj_int else pj_int
                                j_prot = pj_int if pi < pj_int else pi
                                key = (i_prot, j_prot)
                                counts[key] = counts.get(key, 0) + 1
                        continue

                    # Different cells: compare all pairs between atoms0 and atoms1
                    # Vectorized over atoms1 per atom0 (still ok because cells are small)
                    for ii in range(len(a0)):
                        pi = int(p0[ii])
                        ri = x0[ii : ii + 1, :]  # (1,3)
                        d = x1 - ri  # (n1,3)
                        d = _min_image_disp_nm(d, box_nm)
                        dist2 = np.einsum("ij,ij->i", d, d)
                        hit = dist2 <= cut2
                        if not np.any(hit):
                            continue
                        for pj_val in p1[hit]:
                            pj_int = int(pj_val)
                            if pj_int == pi:
                                continue
                            i_prot = pi if pi < pj_int else pj_int
                            j_prot = pj_int if pi < pj_int else pi
                            key = (i_prot, j_prot)
                            counts[key] = counts.get(key, 0) + 1

    # Threshold edges and clusters
    edges = [(i, j) for (i, j), c in counts.items() if int(c) >= int(contact_threshold)]
    edges.sort()
    clusters = _clusters_from_edges(n_proteins, edges)
    # Keep only counts for edges (optional: retain all counts if you prefer)
    edge_counts = {(i, j): int(counts[(i, j)]) for (i, j) in edges}

    return edges, edge_counts, clusters


def clusters_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein",
    heavy_only: bool = True,
    dist_cutoff_nm: float = 0.45,
    contact_threshold: int = 10,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
    progress: bool = False,
    progress_every: int = 200,
    progress_min_interval_s: float = 10.0,
    progress_stream=None,
) -> dict[str, Any]:
    """
    Identify transient clusters of solute copies (typically proteins) based on
    inter-protein heavy-atom contacts.

    A protein-protein edge exists if the number of heavy-atom pairs within
    dist_cutoff_nm (minimum-image, orthorhombic PBC) is >= contact_threshold.

    Clusters are connected components of this edge graph.

    Parameters
    ----------
    selection
        Same semantics as other functions in this module (e.g. "protein").
        Should yield one group per protein copy (typically one per chain).
    heavy_only
        If True (default), use heavy atoms only; otherwise uses all atoms in each group.
    dist_cutoff_nm
        Heavy-atom distance cutoff (nm) for defining an atom-atom contact.
    contact_threshold
        Minimum number of atom-atom contacts to define a protein-protein edge.
    stride, chunk, frame_start, frame_stop, box_nm
        Same meaning as in rdf_from_dcd / structure_factor_from_dcd.

    Returns
    -------
    dict with keys:
      - "clusters_by_frame": list[list[list[int]]], clusters per frame (protein indices)
      - "edges_by_frame": list[list[tuple[int,int]]], contact edges per frame
      - "contact_counts_by_frame": list[dict[(i,j)->count]] for edges only
      - "cluster_sizes_by_frame": list[list[int]]
      - "max_cluster_size_by_frame": list[int]
      - "frac_in_clusters_by_frame": list[float]  (fraction of proteins in clusters of size>=2)
      - "n_proteins": int
      - "params": dict of settings
      - "frames_used": int
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

    t0 = time.time()
    last_print = t0

    if progress:
        _progress_print(
            f"[clusters] start: n_dcd={len(dcd_list)} stride={stride} chunk={chunk} "
            f"frame_start={frame_start} frame_stop={frame_stop} "
            f"cutoff_nm={dist_cutoff_nm} contacts>={contact_threshold}",
            stream=progress_stream,
        )

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    groups_full = _selection_to_groups(tmpl, selection)

    if len(groups_full) < 2:
        raise ValueError("selection must yield >=2 non-empty groups")

    n_proteins = int(len(groups_full))

    # Build heavy-atom index list for each protein in *template* atom indexing
    heavy_groups: list[list[int]] = []
    for g in groups_full:
        if not heavy_only:
            heavy_groups.append([int(i) for i in g])
            continue
        hg = [int(i) for i in g if not _is_hydrogen_like(tmpl_model.atoms[int(i)])]
        heavy_groups.append(hg)

    if any(len(hg) == 0 for hg in heavy_groups):
        raise ValueError("one or more groups have zero heavy atoms; check selection/topology")

    # Flatten heavy atom indices and map to protein id
    heavy_atom_indices_full: list[int] = []
    heavy_atom_to_protein_full: list[int] = []
    for pi, hg in enumerate(heavy_groups):
        for ai in hg:
            heavy_atom_indices_full.append(int(ai))
            heavy_atom_to_protein_full.append(int(pi))

    # Create selection mapping into the iter_dcd atom order (0..n_heavy-1)
    atom_indices = sorted(set(heavy_atom_indices_full))
    idx_map = {old: new for new, old in enumerate(atom_indices)}  # template -> selected

    # For each selected heavy atom, assign a protein id
    atom_to_protein = np.empty(len(atom_indices), dtype=np.int64)
    # We need a reverse lookup from atom -> protein (in full indexing)
    atom_full_to_prot: dict[int, int] = {}
    for ai, pi in zip(heavy_atom_indices_full, heavy_atom_to_protein_full):
        # If an atom appears in multiple proteins something is wrong; last wins
        atom_full_to_prot[int(ai)] = int(pi)
    for old, new in idx_map.items():
        atom_to_protein[int(new)] = int(atom_full_to_prot[int(old)])

    # Iterate frames using only heavy atoms
    clusters_by_frame: list[list[list[int]]] = []
    edges_by_frame: list[list[tuple[int, int]]] = []
    counts_by_frame: list[dict[tuple[int, int], int]] = []

    cluster_sizes_by_frame: list[list[int]] = []
    max_cluster_size_by_frame: list[int] = []
    frac_in_clusters_by_frame: list[float] = []

    frames_used = 0

    for dcd in dcd_list:
        if progress:
            _progress_print(f"reading dcd: {dcd}", stream=progress_stream)

        box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

        for fi, (xyz_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                tmpl_model,
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
                b = _box_lengths_nm(box_frame_nm)

            if np.any(b <= 0.0):
                raise ValueError("box lengths must be positive")

            edges, edge_counts, clusters = _contacts_and_clusters_one_frame(
                xyz_nm,
                b,
                atom_to_protein,
                n_proteins,
                dist_cutoff_nm=float(dist_cutoff_nm),
                contact_threshold=int(contact_threshold),
            )

            clusters_by_frame.append(clusters)
            edges_by_frame.append(edges)
            counts_by_frame.append(edge_counts)

            sizes = [len(c) for c in clusters]
            cluster_sizes_by_frame.append(sizes)
            mx = int(max(sizes)) if sizes else 0
            max_cluster_size_by_frame.append(mx)

            n_in_clusters = sum(len(c) for c in clusters if len(c) >= 2)
            frac = float(n_in_clusters) / float(n_proteins)
            frac_in_clusters_by_frame.append(frac)

            frames_used += 1

            if progress:
                now = time.time()
                do_frame_print = (progress_every > 0) and (frames_used % int(progress_every) == 0)
                do_time_print = (progress_min_interval_s > 0.0) and (
                    (now - last_print) >= float(progress_min_interval_s)
                )

                if do_frame_print or do_time_print:
                    elapsed = now - t0
                    rate = frames_used / elapsed if elapsed > 0 else float("nan")

                    # Best-effort total estimate (only if we can infer it)
                    # If your iter_dcd provides a known number of frames, plug it here.
                    frames_total_est = None  # leave None unless you can compute it reliably

                    if frames_total_est is not None and frames_total_est > 0 and frames_used > 0:
                        remaining = max(0, int(frames_total_est) - frames_used)
                        eta_s = remaining / rate if rate > 0 else float("inf")
                        _progress_print(
                            f"[clusters] frames={frames_used}/{frames_total_est} "
                            f"elapsed={_fmt_hms(elapsed)} rate={rate:6.2f} f/s "
                            + f"ETA={_fmt_hms(eta_s)}",
                            stream=progress_stream,
                        )
                    else:
                        _progress_print(
                            f"[clusters] frames={frames_used} elapsed={_fmt_hms(elapsed)} "
                            + f"rate={rate:6.2f} f/s",
                            stream=progress_stream,
                        )

                    last_print = now

    if progress:
        dt = time.time() - t0
        rate = frames_used / dt if dt > 0 else float("nan")
        _progress_print(
            f"[clusters] done. frames={frames_used}  elapsed={_fmt_hms(dt)}  "
            + f"rate={rate:6.2f} frames/s",
            stream=progress_stream,
        )

    return {
        "clusters_by_frame": clusters_by_frame,
        "edges_by_frame": edges_by_frame,
        "contact_counts_by_frame": counts_by_frame,
        "cluster_sizes_by_frame": cluster_sizes_by_frame,
        "max_cluster_size_by_frame": max_cluster_size_by_frame,
        "frac_in_clusters_by_frame": frac_in_clusters_by_frame,
        "n_proteins": n_proteins,
        "frames_used": int(frames_used),
        "params": {
            "selection": selection,
            "heavy_only": bool(heavy_only),
            "dist_cutoff_nm": float(dist_cutoff_nm),
            "contact_threshold": int(contact_threshold),
            "stride": int(stride),
            "chunk": int(chunk),
            "frame_start": int(frame_start),
            "frame_stop": frame_stop,
        },
    }


@dataclass(frozen=True)
class CondensateResult:
    membership: np.ndarray  # (n_frames, n_chains) bool
    largest_size: np.ndarray  # (n_frames,) int
    n_condensates: np.ndarray  # (n_frames,) int
    condensates_by_frame: list[list[list[int]]]  # per frame: list of clusters (member indices)
    min_size: int
    params: dict[str, Any]


def condensates_from_dcd(
    pdb_file: Any,
    dcd_files: Any,
    *,
    # interaction definition (reuses clustering.py)
    dist_cutoff_nm: float,
    contact_threshold: int = 1,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein",
    heavy_only: bool = False,
    # condensate definition
    min_frac: float = 0.1,
    min_size: Optional[int] = None,
    # io/frames
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
    progress: bool = False,
) -> CondensateResult:
    """
    Identify condensate membership per frame using clustering.clusters_from_dcd.

    - First, clusters are defined by an inter-chain edge graph:
        edge(i,j) exists if >= contact_threshold atom pairs within dist_cutoff_nm
    - Then, a "condensate" is any cluster with size >= min_size, where:
        min_size = ceil(min_frac * n_chains) unless overridden.

    Returns a per-frame membership boolean mask.
    """
    if float(dist_cutoff_nm) <= 0.0:
        raise ValueError("dist_cutoff_nm must be > 0")
    if int(contact_threshold) <= 0:
        raise ValueError("contact_threshold must be >= 1")
    if float(min_frac) <= 0.0 or float(min_frac) > 1.0:
        raise ValueError("min_frac must be in (0, 1]")

    out = clusters_from_dcd(
        pdb_file=pdb_file,
        dcd_files=dcd_files,
        selection=selection,
        heavy_only=bool(heavy_only),
        dist_cutoff_nm=float(dist_cutoff_nm),
        contact_threshold=int(contact_threshold),
        stride=int(stride),
        chunk=int(chunk),
        frame_start=int(frame_start),
        frame_stop=frame_stop,
        box_nm=box_nm,
        progress=bool(progress),
    )

    clusters_by_frame = out["clusters_by_frame"]
    n_chains = int(out["n_proteins"])
    n_frames = int(out["frames_used"])

    if min_size is None:
        ms = int(math.ceil(float(min_frac) * float(n_chains)))
        min_size_i = max(2, ms)
    else:
        min_size_i = int(min_size)
        if min_size_i < 2:
            raise ValueError("min_size must be >= 2")

    membership = np.zeros((n_frames, n_chains), dtype=bool)
    largest = np.zeros((n_frames,), dtype=np.int64)
    n_cond = np.zeros((n_frames,), dtype=np.int64)
    condensates_only: list[list[list[int]]] = []

    for t, clusters in enumerate(clusters_by_frame):
        conds = [c for c in clusters if int(len(c)) >= min_size_i]
        condensates_only.append(conds)

        if conds:
            largest[t] = int(max(len(c) for c in conds))
            n_cond[t] = int(len(conds))
            for c in conds:
                membership[t, np.asarray(c, dtype=np.int64)] = True
        else:
            largest[t] = 0
            n_cond[t] = 0

    params = dict(out.get("params", {}))
    params.update(
        {
            "min_frac": float(min_frac),
            "min_size": int(min_size_i),
        }
    )

    return CondensateResult(
        membership=membership,
        largest_size=largest,
        n_condensates=n_cond,
        condensates_by_frame=condensates_only,
        min_size=int(min_size_i),
        params=params,
    )


@dataclass(frozen=True)
class ChainContactResult:
    contacts_per_chain: np.ndarray  # (n_query, n_frames)
    contacts_mean: np.ndarray  # (n_frames,)
    contacts_stderr: np.ndarray  # (n_frames,)
    n_query_chains: int
    n_partner_chains: int
    n_frames: int
    cutoff_nm: float


def _contacts_between_group_sets_one_frame(
    xyz_nm: np.ndarray,  # (n_atoms, 3) in selected-atom indexing
    box_nm: np.ndarray,  # (3,)
    atom_to_query: np.ndarray,  # (n_atoms,) int, -1 if not in query set
    atom_to_partner: np.ndarray,  # (n_atoms,) int, -1 if not in partner set
    same_group_pairs: set[tuple[int, int]],
    *,
    dist_cutoff_nm: float,
    cell_size_nm: Optional[float] = None,
) -> np.ndarray:
    """
    Count atom-atom contacts from query groups to partner groups for one frame.

    Returns
    -------
    pair_counts : (n_query, n_partner) int64
        Number of atom pairs within cutoff for each query-partner group pair.

    Notes
    -----
    - Self interactions are excluded using same_group_pairs.
    - Double counting is avoided using the same cell-ordering logic as
      _contacts_and_clusters_one_frame().
    """
    if dist_cutoff_nm <= 0.0:
        raise ValueError("dist_cutoff_nm must be > 0")

    xyz = _wrap_positions_nm(np.asarray(xyz_nm, dtype=np.float64), box_nm)
    cs = float(dist_cutoff_nm if cell_size_nm is None else cell_size_nm)
    ncell, _, bins = _build_cells(xyz, box_nm, cs)
    nx, ny, nz = [int(v) for v in ncell.tolist()]

    q_valid = atom_to_query >= 0
    p_valid = atom_to_partner >= 0
    n_query = int(np.max(atom_to_query[q_valid])) + 1 if np.any(q_valid) else 0
    n_partner = int(np.max(atom_to_partner[p_valid])) + 1 if np.any(p_valid) else 0
    out = np.zeros((n_query, n_partner), dtype=np.int64)

    cut2 = float(dist_cutoff_nm * dist_cutoff_nm)

    def flat_cell(ix: int, iy: int, iz: int) -> int:
        return int(ix * (ny * nz) + iy * nz + iz)

    def add_hits(
        q_ids_a: np.ndarray,
        p_ids_b: np.ndarray,
        hit_mask: np.ndarray,
    ) -> None:
        if not np.any(hit_mask):
            return
        q_hit = q_ids_a[hit_mask]
        p_hit = p_ids_b[hit_mask]
        for qi_val, pj_val in zip(q_hit.tolist(), p_hit.tolist()):
            qi = int(qi_val)
            pj = int(pj_val)
            if qi < 0 or pj < 0:
                continue
            if (qi, pj) in same_group_pairs:
                continue
            out[qi, pj] += 1

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                c0 = flat_cell(ix, iy, iz)
                atoms0 = bins[c0]
                if not atoms0:
                    continue

                a0 = np.asarray(atoms0, dtype=np.int64)
                x0 = xyz[a0, :]
                q0 = atom_to_query[a0]
                p0 = atom_to_partner[a0]

                seen: set[int] = set()
                for dx, dy, dz in _NEIGHBOR_OFFSETS_27:
                    jx = (ix + dx) % nx
                    jy = (iy + dy) % ny
                    jz = (iz + dz) % nz
                    c1 = flat_cell(jx, jy, jz)
                    if c1 in seen:
                        continue
                    seen.add(c1)

                    if c1 < c0:
                        continue

                    atoms1 = bins[c1]
                    if not atoms1:
                        continue

                    a1 = np.asarray(atoms1, dtype=np.int64)
                    x1 = xyz[a1, :]
                    q1 = atom_to_query[a1]
                    p1 = atom_to_partner[a1]

                    if c1 == c0:
                        m = len(a0)
                        for ii in range(m - 1):
                            ri = x0[ii : ii + 1, :]
                            d = x0[ii + 1 :, :] - ri
                            d = _min_image_disp_nm(d, box_nm)
                            dist2 = np.einsum("ij,ij->i", d, d)
                            hit = dist2 <= cut2
                            if not np.any(hit):
                                continue

                            # direction: atom ii as query, jj as partner
                            qi = np.full(m - ii - 1, q0[ii], dtype=np.int64)
                            add_hits(qi, p0[ii + 1 :], hit)

                            # reverse direction: atom jj as query, ii as partner
                            qj = q0[ii + 1 :]
                            pi = np.full(m - ii - 1, p0[ii], dtype=np.int64)
                            add_hits(qj, pi, hit)
                        continue

                    for ii in range(len(a0)):
                        ri = x0[ii : ii + 1, :]
                        d = x1 - ri
                        d = _min_image_disp_nm(d, box_nm)
                        dist2 = np.einsum("ij,ij->i", d, d)
                        hit = dist2 <= cut2
                        if not np.any(hit):
                            continue

                        qi = np.full(len(a1), q0[ii], dtype=np.int64)
                        add_hits(qi, p1, hit)

                        qj = q1
                        pi = np.full(len(a1), p0[ii], dtype=np.int64)
                        add_hits(qj, pi, hit)

    return out


def contacts_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    query_selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein.CA",
    partner_selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein.CA",
    cutoff_nm: float = 0.8,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> ChainContactResult:
    if float(cutoff_nm) <= 0.0:
        raise ValueError("cutoff_nm must be > 0")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    query_groups_full = _selection_to_groups(tmpl, query_selection)
    partner_groups_full = _selection_to_groups(tmpl, partner_selection)

    if not query_groups_full:
        raise ValueError("query_selection produced no groups")
    if not partner_groups_full:
        raise ValueError("partner_selection produced no groups")

    atom_set: set[int] = set()
    for g in query_groups_full:
        atom_set.update(int(i) for i in g.tolist())
    for g in partner_groups_full:
        atom_set.update(int(i) for i in g.tolist())
    atom_indices = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices)}

    query_groups = [
        np.asarray([idx_map[int(i)] for i in g.tolist()], dtype=np.int64) for g in query_groups_full
    ]
    partner_groups = [
        np.asarray([idx_map[int(i)] for i in g.tolist()], dtype=np.int64)
        for g in partner_groups_full
    ]

    atom_to_query = np.full(len(atom_indices), -1, dtype=np.int64)
    for qi, g in enumerate(query_groups):
        atom_to_query[g] = int(qi)

    atom_to_partner = np.full(len(atom_indices), -1, dtype=np.int64)
    for pj, g in enumerate(partner_groups):
        atom_to_partner[g] = int(pj)

    same_group_pairs: set[tuple[int, int]] = set()
    partner_lookup = {
        tuple(int(i) for i in g.tolist()): j for j, g in enumerate(partner_groups_full)
    }
    for qi, g in enumerate(query_groups_full):
        key = tuple(int(i) for i in g.tolist())
        pj = partner_lookup.get(key)
        if pj is not None:
            same_group_pairs.add((int(qi), int(pj)))

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    frames: list[np.ndarray] = []

    for dcd in dcd_list:
        for fi, (xyz_sel_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                tmpl_model,
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
                    raise ValueError("DCD lacks box; pass box_nm=(Lx,Ly,Lz) in nm")
                b = box_fallback
            else:
                b = _box_lengths_nm(box_frame_nm)

            pair_counts = _contacts_between_group_sets_one_frame(
                np.asarray(xyz_sel_nm, dtype=np.float64),
                b,
                atom_to_query,
                atom_to_partner,
                same_group_pairs,
                dist_cutoff_nm=float(cutoff_nm),
            )
            frames.append(np.sum(pair_counts, axis=1).astype(np.float64))

    if not frames:
        raise ValueError("no frames selected")

    contacts_pf = np.stack(frames, axis=1)
    contacts_mean = np.nanmean(contacts_pf, axis=0)
    if contacts_pf.shape[0] < 2:
        contacts_stderr = np.zeros_like(contacts_mean)
    else:
        contacts_stderr = np.nanstd(contacts_pf, axis=0, ddof=1) / math.sqrt(
            float(contacts_pf.shape[0])
        )

    return ChainContactResult(
        contacts_per_chain=contacts_pf,
        contacts_mean=contacts_mean,
        contacts_stderr=contacts_stderr,
        n_query_chains=int(contacts_pf.shape[0]),
        n_partner_chains=int(len(partner_groups)),
        n_frames=int(contacts_pf.shape[1]),
        cutoff_nm=float(cutoff_nm),
    )

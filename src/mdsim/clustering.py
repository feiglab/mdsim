from __future__ import annotations

import io
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .analysis import _as_file_list, _box_lengths_nm, _fmt_hms, _progress_print
from .molecule_data import PDBReader, StructureSelector, iter_dcd

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
    selection: Union[str, Sequence[Sequence[int]]] = "protein",
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

    # Protein groups (full-atom indices in template)
    if isinstance(selection, str):
        groups_full = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_full = [[int(i) for i in g] for g in selection]
    groups_full = [g for g in groups_full if g]
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
                b = np.asarray(box_frame_nm, dtype=np.float64).reshape(3)

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

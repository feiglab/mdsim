from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .analysis import (
    _as_file_list,
    _box_lengths_nm,
    _chain_indices_from_selection,
)
from .molecule_data import PDBReader, iter_dcd

FileLike = Union[str, Path, io.BytesIO, io.StringIO]


@dataclass(frozen=True)
class CAContactTrajectoryResult:
    """Compact sparse per-frame CA contact events for later map reduction.

    Events are grouped by frame through ``frame_offsets``. Each event stores
    only a flattened query-site index, flattened partner-site index, and CA
    distance. Chain and residue indices are decoded on demand as::

        chain = site_index // n_residues
        residue = site_index % n_residues

    Site indices use ``uint16`` when possible and automatically promote to
    ``uint32`` for larger selections. This substantially reduces persistent
    cache memory compared with storing four separate int32 index arrays.

    In symmetric homotypic mode, every undirected site pair is stored once.
    The reduction function restores both reference directions as needed.
    """

    frame_offsets: np.ndarray  # (n_frames + 1,), int64
    source_frame_indices: np.ndarray  # (n_frames,), indices after DCD stride
    query_site_index: np.ndarray  # (n_events,), uint16 or uint32
    partner_site_index: np.ndarray  # (n_events,), uint16 or uint32
    distance_nm: np.ndarray  # (n_events,), float32

    query_chain_labels: tuple[str, ...]
    partner_chain_labels: tuple[str, ...]
    query_physical_chain_indices: np.ndarray
    partner_physical_chain_indices: np.ndarray
    query_residue_numbers: np.ndarray
    partner_residue_numbers: np.ndarray
    query_residue_names: tuple[str, ...]
    partner_residue_names: tuple[str, ...]

    n_frames: int
    n_query_chains: int
    n_partner_chains: int
    n_query_residues: int
    n_partner_residues: int
    n_events: int
    max_cutoff_nm: float
    atom_name: str
    pbc: bool
    symmetric_homotypic: bool
    query_selection: Any
    partner_selection: Any
    stride: int
    frame_start: int
    frame_stop: Optional[int]
    query_site_chunk: int = 4096

    @property
    def chain_labels(self) -> tuple[str, ...]:
        return self.query_chain_labels

    @property
    def query_chain_index(self) -> np.ndarray:
        """Compatibility view decoded from compact site indices."""
        return (self.query_site_index.astype(np.uint64) // self.n_query_residues).astype(np.int32)

    @property
    def partner_chain_index(self) -> np.ndarray:
        """Compatibility view decoded from compact site indices."""
        return (self.partner_site_index.astype(np.uint64) // self.n_partner_residues).astype(
            np.int32
        )

    @property
    def query_residue_index(self) -> np.ndarray:
        """Compatibility view decoded from compact site indices."""
        return (self.query_site_index.astype(np.uint64) % self.n_query_residues).astype(np.int32)

    @property
    def partner_residue_index(self) -> np.ndarray:
        """Compatibility view decoded from compact site indices."""
        return (self.partner_site_index.astype(np.uint64) % self.n_partner_residues).astype(
            np.int32
        )

    def frame_slice(self, frame_index: int) -> slice:
        frame = int(frame_index)
        if frame < 0 or frame >= self.n_frames:
            raise IndexError(f"frame index {frame} outside 0..{self.n_frames - 1}")
        return slice(int(self.frame_offsets[frame]), int(self.frame_offsets[frame + 1]))


@dataclass(frozen=True)
class ResidueContactMapResult:
    """Reduced intra/inter residue contact maps.

    ``intra`` and ``inter`` use residue indices as rows/columns.  For a
    homotypic square map, ``combined`` contains intrachain values in the lower
    triangle and interchain values in the upper triangle, including the diagonal.

    ``inter_normalization='per_reference_chain'`` reports the mean number of
    partner-chain contacts per reference chain and selected frame.  It can
    exceed one.  ``'per_chain_pair'`` reports the contact probability per
    ordered chain pair and selected frame.
    """

    intra: np.ndarray
    inter: np.ndarray
    combined: Optional[np.ndarray]
    intra_stderr: np.ndarray
    inter_stderr: np.ndarray
    combined_stderr: Optional[np.ndarray]

    query_residue_numbers: np.ndarray
    partner_residue_numbers: np.ndarray
    query_residue_names: tuple[str, ...]
    partner_residue_names: tuple[str, ...]
    query_chain_labels: tuple[str, ...]
    partner_chain_labels: tuple[str, ...]

    cutoff_nm: float
    max_cutoff_nm: float
    min_intra_sequence_separation: int
    inter_normalization: str
    n_frames_selected: int
    n_intra_chain_observations: int
    n_inter_reference_observations: int
    n_inter_pair_observations: int
    symmetric_homotypic: bool
    n_sets: int = 1
    set_labels: tuple[str, ...] = ()
    aggregation: str = "individual"

    @property
    def residue_numbers(self) -> np.ndarray:
        if not self.symmetric_homotypic:
            raise AttributeError("rectangular/heterotypic maps have separate residue axes")
        return self.query_residue_numbers

    @property
    def units(self) -> str:
        if self.inter_normalization == "per_chain_pair":
            return "contact probability"
        return "contacts per reference chain per frame"


def _ca_atom_matrix_for_chains(
    template: Any,
    chain_selection: Union[str, Sequence[str]],
    *,
    atom_name: str,
    argument_name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]]:
    """Return one selected atom per residue for each selected physical chain."""
    model = template.model if hasattr(template, "model") else template
    chain_indices, chain_labels, _ = _chain_indices_from_selection(
        template,
        chain_selection,
    )
    if not chain_indices:
        raise ValueError(f"{argument_name} produced no physical chains")

    wanted = str(atom_name).strip().upper()
    if not wanted:
        raise ValueError("atom_name must be non-empty")

    atom_index = {id(atom): index for index, atom in enumerate(model.atoms)}
    chains = list(model.chain.values())

    per_chain_atoms: list[list[int]] = []
    per_chain_numbers: list[list[int]] = []
    per_chain_names: list[list[str]] = []

    for chain_index, chain_label in zip(chain_indices, chain_labels):
        chain = chains[int(chain_index)]
        atoms: list[int] = []
        numbers: list[int] = []
        names: list[str] = []

        for residue in chain.residues:
            hits = [
                atom
                for atom in residue.atoms
                if str(getattr(atom, "name", "") or "").strip().upper() == wanted
            ]
            if len(hits) != 1:
                raise ValueError(
                    f"{argument_name} chain {chain_label!r}, residue "
                    f"{getattr(residue, 'resnum', '?')}: expected exactly one "
                    f"{wanted} atom, found {len(hits)}"
                )
            index = atom_index.get(id(hits[0]))
            if index is None:
                raise RuntimeError("selected atom is missing from the model atom list")
            atoms.append(int(index))
            numbers.append(int(getattr(residue, "resnum")))
            names.append(str(getattr(residue, "resname", "") or ""))

        if not atoms:
            raise ValueError(f"{argument_name} chain {chain_label!r} contains no residues")
        per_chain_atoms.append(atoms)
        per_chain_numbers.append(numbers)
        per_chain_names.append(names)

    reference_numbers = per_chain_numbers[0]
    reference_names = per_chain_names[0]
    for chain_label, numbers, names in zip(
        chain_labels[1:],
        per_chain_numbers[1:],
        per_chain_names[1:],
    ):
        if numbers != reference_numbers:
            raise ValueError(
                f"{argument_name} chain {chain_label!r} does not share the same "
                "residue-number axis as the first selected chain"
            )
        if names != reference_names:
            raise ValueError(
                f"{argument_name} chain {chain_label!r} does not share the same "
                "residue-name sequence as the first selected chain"
            )

    return (
        np.asarray(per_chain_atoms, dtype=np.int64),
        np.asarray(chain_indices, dtype=np.int64),
        tuple(str(label) for label in chain_labels),
        np.asarray(reference_numbers, dtype=np.int64),
        tuple(reference_names),
    )


def _wrap_positions_nm(xyz_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    box = np.asarray(box_nm, dtype=np.float64).reshape(1, 3)
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    return xyz - np.floor(xyz / box) * box


def _compact_site_dtype(n_sites: int) -> np.dtype:
    """Smallest unsigned dtype that can represent ``0..n_sites-1``."""
    n = int(n_sites)
    if n <= 0:
        raise ValueError("number of sites must be > 0")
    if n - 1 <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if n - 1 <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError("contact-map site count exceeds uint32 indexing capacity")


class _CompactContactEventBuffer:
    """Growable compact event buffer without a final concatenate copy."""

    def __init__(
        self,
        query_dtype: np.dtype,
        partner_dtype: np.dtype,
        *,
        initial_capacity: int = 65536,
    ) -> None:
        cap = max(1, int(initial_capacity))
        self.query_dtype = np.dtype(query_dtype)
        self.partner_dtype = np.dtype(partner_dtype)
        self.query_site = np.empty(cap, dtype=self.query_dtype)
        self.partner_site = np.empty(cap, dtype=self.partner_dtype)
        self.distance_nm = np.empty(cap, dtype=np.float32)
        self.size = 0

    @property
    def capacity(self) -> int:
        return int(self.distance_nm.size)

    def _grow(self, required: int) -> None:
        if required <= self.capacity:
            return
        new_capacity = self.capacity
        while new_capacity < required:
            new_capacity = max(new_capacity * 2, required)
        self.query_site.resize(new_capacity, refcheck=False)
        self.partner_site.resize(new_capacity, refcheck=False)
        self.distance_nm.resize(new_capacity, refcheck=False)

    def append(
        self,
        query_site: np.ndarray,
        partner_site: np.ndarray,
        distance_nm: np.ndarray,
    ) -> None:
        n = int(np.asarray(distance_nm).size)
        if n == 0:
            return
        required = self.size + n
        self._grow(required)
        sl = slice(self.size, required)
        self.query_site[sl] = np.asarray(query_site, dtype=self.query_dtype)
        self.partner_site[sl] = np.asarray(partner_site, dtype=self.partner_dtype)
        self.distance_nm[sl] = np.asarray(distance_nm, dtype=np.float32)
        self.size = required

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Shrink the owning arrays in place so the returned cache does not retain
        # unused doubling capacity and no full-size concatenate copy is needed.
        self.query_site.resize(self.size, refcheck=False)
        self.partner_site.resize(self.size, refcheck=False)
        self.distance_nm.resize(self.size, refcheck=False)
        return self.query_site, self.partner_site, self.distance_nm


def ca_contact_trajectory_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    query_chains: Union[str, Sequence[str]] = "protein",
    partner_chains: Optional[Union[str, Sequence[str]]] = None,
    atom_name: str = "CA",
    max_cutoff_nm: float = 1.0,
    pbc: bool = True,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    workers: int = 1,
    query_site_chunk: int = 4096,
) -> CAContactTrajectoryResult:
    """Generate compact sparse CA contacts up to ``max_cutoff_nm``.

    This is the expensive trajectory-reading stage intended for caching. A
    later call to :func:`residue_contact_map_from_trajectory` may use any
    contact threshold less than or equal to ``max_cutoff_nm`` and may apply
    arbitrary frame- and chain-level masks without rereading the trajectory.

    Memory strategy
    ---------------
    * Each event stores two compact flattened site indices plus one float32
      distance rather than four int32 chain/residue indices.
    * Query sites are searched in bounded chunks, limiting temporary neighbor
      lists and flattened hit arrays.
    * Events are appended into a growable NumPy buffer, avoiding the old
      list-of-arrays plus final-concatenation peak-memory duplication.
    * In homotypic mode, only the canonical half of the neighbor relation is
      retained, so every undirected CA pair is stored once.

    ``partner_chains=None`` selects the same chains as ``query_chains`` and
    activates symmetric homotypic storage. Supplying a distinct partner chain
    selection supports rectangular heterotypic maps.
    """
    if float(max_cutoff_nm) <= 0.0 or not math.isfinite(float(max_cutoff_nm)):
        raise ValueError("max_cutoff_nm must be finite and > 0")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")
    if int(workers) == 0 or int(workers) < -1:
        raise ValueError("workers must be -1 or a positive integer")
    if int(query_site_chunk) <= 0:
        raise ValueError("query_site_chunk must be >= 1")

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise ImportError("SciPy is required for CA contact-map generation") from exc

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    template = PDBReader().read(pdb_file)
    model = template.model

    query_atoms_full, query_physical, query_labels, query_resnums, query_resnames = (
        _ca_atom_matrix_for_chains(
            template, query_chains, atom_name=atom_name, argument_name="query_chains"
        )
    )

    partner_selection = query_chains if partner_chains is None else partner_chains
    partner_atoms_full, partner_physical, partner_labels, partner_resnums, partner_resnames = (
        _ca_atom_matrix_for_chains(
            template,
            partner_selection,
            atom_name=atom_name,
            argument_name="partner_chains",
        )
    )

    symmetric_homotypic = bool(
        np.array_equal(query_atoms_full, partner_atoms_full)
        and np.array_equal(query_physical, partner_physical)
        and np.array_equal(query_resnums, partner_resnums)
    )

    atom_indices_full = sorted(
        set(query_atoms_full.reshape(-1).tolist()) | set(partner_atoms_full.reshape(-1).tolist())
    )
    selected_index = {old: new for new, old in enumerate(atom_indices_full)}

    query_atoms = np.asarray(
        [[selected_index[int(atom)] for atom in row] for row in query_atoms_full],
        dtype=np.int64,
    )
    partner_atoms = np.asarray(
        [[selected_index[int(atom)] for atom in row] for row in partner_atoms_full],
        dtype=np.int64,
    )

    q_flat = query_atoms.reshape(-1)
    p_flat = partner_atoms.reshape(-1)
    q_template_atom = query_atoms_full.reshape(-1)
    p_template_atom = partner_atoms_full.reshape(-1)

    n_query_sites = int(q_flat.size)
    n_partner_sites = int(p_flat.size)
    query_site_dtype = _compact_site_dtype(n_query_sites)
    partner_site_dtype = _compact_site_dtype(n_partner_sites)
    events = _CompactContactEventBuffer(query_site_dtype, partner_site_dtype)

    frame_offsets = [0]
    source_frame_indices: list[int] = []
    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)
    site_chunk = int(query_site_chunk)

    for dcd in dcd_list:
        for frame_index, (xyz_sel_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                model,
                chunk=int(chunk),
                stride=int(stride),
                atom_indices=atom_indices_full,
            )
        ):
            if frame_index < int(frame_start):
                continue
            if frame_stop is not None and frame_index >= int(frame_stop):
                break

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
            if bool(pbc):
                if box_frame_nm is None:
                    if box_fallback is None:
                        raise ValueError(
                            "DCD lacks unit-cell lengths; pass box_nm=(Lx,Ly,Lz) "
                            "or set pbc=False"
                        )
                    box = box_fallback
                else:
                    box = _box_lengths_nm(box_frame_nm)
                xyz_use = _wrap_positions_nm(xyz, box)
                tree = cKDTree(xyz_use[p_flat], boxsize=box)
            else:
                box = None
                xyz_use = xyz
                tree = cKDTree(xyz_use[p_flat])

            for q_start in range(0, n_query_sites, site_chunk):
                q_stop = min(q_start + site_chunk, n_query_sites)
                q_sites_chunk = np.arange(q_start, q_stop, dtype=np.int64)

                try:
                    neighbors = tree.query_ball_point(
                        xyz_use[q_flat[q_start:q_stop]],
                        r=float(max_cutoff_nm),
                        workers=int(workers),
                        return_sorted=False,
                    )
                except TypeError as exc:
                    if int(workers) != 1:
                        raise RuntimeError("parallel cKDTree queries require SciPy >= 1.6") from exc
                    neighbors = tree.query_ball_point(
                        xyz_use[q_flat[q_start:q_stop]],
                        r=float(max_cutoff_nm),
                        return_sorted=False,
                    )

                lengths = np.fromiter(
                    (len(values) for values in neighbors),
                    dtype=np.int64,
                    count=len(neighbors),
                )
                n_hits = int(np.sum(lengths))
                if n_hits == 0:
                    continue

                q_site = np.repeat(q_sites_chunk, lengths)
                p_site = np.concatenate(
                    [np.asarray(values, dtype=np.int64) for values in neighbors if values]
                )

                # Remove an atom paired with itself when query/partner sets overlap.
                keep = q_template_atom[q_site] != p_template_atom[p_site]

                if symmetric_homotypic:
                    # q_site and p_site refer to the same row-major flattened CA
                    # space. Keeping only p_site > q_site stores each undirected
                    # contact exactly once, covering both intra- and interchain
                    # pairs without generating a second persistent copy.
                    keep &= p_site > q_site

                if not np.any(keep):
                    continue

                q_site = q_site[keep]
                p_site = p_site[keep]

                displacement = xyz_use[p_flat[p_site]] - xyz_use[q_flat[q_site]]
                if bool(pbc):
                    assert box is not None
                    displacement -= np.rint(displacement / box.reshape(1, 3)) * box.reshape(1, 3)
                distance = np.linalg.norm(displacement, axis=1)
                precise = distance <= float(max_cutoff_nm) + 1.0e-12
                if not np.any(precise):
                    continue

                events.append(
                    q_site[precise],
                    p_site[precise],
                    distance[precise],
                )

            source_frame_indices.append(int(frame_index))
            frame_offsets.append(events.size)

    if not source_frame_indices:
        raise ValueError("no frames selected")

    query_site_index, partner_site_index, distance_nm = events.finalize()

    return CAContactTrajectoryResult(
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
        source_frame_indices=np.asarray(source_frame_indices, dtype=np.int64),
        query_site_index=query_site_index,
        partner_site_index=partner_site_index,
        distance_nm=distance_nm,
        query_chain_labels=query_labels,
        partner_chain_labels=partner_labels,
        query_physical_chain_indices=query_physical.copy(),
        partner_physical_chain_indices=partner_physical.copy(),
        query_residue_numbers=query_resnums.copy(),
        partner_residue_numbers=partner_resnums.copy(),
        query_residue_names=query_resnames,
        partner_residue_names=partner_resnames,
        n_frames=len(source_frame_indices),
        n_query_chains=int(query_atoms.shape[0]),
        n_partner_chains=int(partner_atoms.shape[0]),
        n_query_residues=int(query_atoms.shape[1]),
        n_partner_residues=int(partner_atoms.shape[1]),
        n_events=int(events.size),
        max_cutoff_nm=float(max_cutoff_nm),
        atom_name=str(atom_name).strip().upper(),
        pbc=bool(pbc),
        symmetric_homotypic=symmetric_homotypic,
        query_selection=query_chains,
        partner_selection=partner_selection,
        stride=int(stride),
        frame_start=int(frame_start),
        frame_stop=frame_stop,
        query_site_chunk=site_chunk,
    )


def _validate_boolean_mask(
    mask: Optional[np.ndarray],
    *,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.shape != shape:
        raise ValueError(f"{name} has shape {out.shape}, expected {shape}")
    return out.copy()


def _normalize_inter_normalization(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("inter_normalization must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "per_reference_chain": "per_reference_chain",
        "reference": "per_reference_chain",
        "per_chain": "per_reference_chain",
        "contacts_per_chain": "per_reference_chain",
        "per_chain_pair": "per_chain_pair",
        "pair": "per_chain_pair",
        "probability": "per_chain_pair",
        "contact_probability": "per_chain_pair",
    }
    if normalized not in aliases:
        raise ValueError("inter_normalization must be 'per_reference_chain' or 'per_chain_pair'")
    return aliases[normalized]


def residue_contact_map_from_trajectory(
    result: CAContactTrajectoryResult,
    *,
    cutoff_nm: Optional[float] = None,
    frame_mask: Optional[np.ndarray] = None,
    query_chain_mask: Optional[np.ndarray] = None,
    partner_chain_mask: Optional[np.ndarray] = None,
    min_intra_sequence_separation: int = 1,
    inter_normalization: str = "per_reference_chain",
) -> ResidueContactMapResult:
    """Reduce cached sparse contacts to normalized intra/inter residue maps.

    Masks have shapes ``(n_frames,)``, ``(n_frames, n_query_chains)``, and
    ``(n_frames, n_partner_chains)``.  They may encode arbitrary time windows,
    chain inclusion/exclusion, or chain-property filters.
    """
    cutoff = result.max_cutoff_nm if cutoff_nm is None else float(cutoff_nm)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_nm must be finite and > 0")
    if cutoff > result.max_cutoff_nm + 1.0e-12:
        raise ValueError(
            f"cutoff_nm={cutoff} exceeds cached max_cutoff_nm="
            f"{result.max_cutoff_nm}; regenerate the raw contact cache"
        )

    if isinstance(min_intra_sequence_separation, (bool, np.bool_)):
        raise TypeError("min_intra_sequence_separation must be an integer >= 1")
    min_sep = int(min_intra_sequence_separation)
    if min_sep < 1:
        raise ValueError("min_intra_sequence_separation must be >= 1")

    normalization = _normalize_inter_normalization(inter_normalization)

    if frame_mask is None:
        frames = np.ones(result.n_frames, dtype=bool)
    else:
        frames = np.asarray(frame_mask, dtype=bool)
        if frames.shape != (result.n_frames,):
            raise ValueError(f"frame_mask has shape {frames.shape}, expected ({result.n_frames},)")

    query_mask = _validate_boolean_mask(
        query_chain_mask,
        shape=(result.n_frames, result.n_query_chains),
        name="query_chain_mask",
    )
    partner_mask = _validate_boolean_mask(
        partner_chain_mask,
        shape=(result.n_frames, result.n_partner_chains),
        name="partner_chain_mask",
    )
    query_mask &= frames[:, None]
    partner_mask &= frames[:, None]

    nqres = result.n_query_residues
    npres = result.n_partner_residues
    intra_counts = np.zeros((nqres, npres), dtype=np.float64)
    inter_counts = np.zeros((nqres, npres), dtype=np.float64)

    # For homotypic maps, the intra denominator is one observation for every
    # chain/frame retained on both sides.  For distinct chain sets, only shared
    # physical chains can contribute intrachain events.
    if result.symmetric_homotypic:
        intra_chain_mask = query_mask & partner_mask
        n_intra_observations = int(np.sum(intra_chain_mask))
    else:
        partner_by_physical = {
            int(physical): index
            for index, physical in enumerate(result.partner_physical_chain_indices)
        }
        n_intra_observations = 0
        for query_index, physical in enumerate(result.query_physical_chain_indices):
            partner_index = partner_by_physical.get(int(physical))
            if partner_index is not None:
                n_intra_observations += int(
                    np.sum(query_mask[:, query_index] & partner_mask[:, partner_index])
                )

    n_inter_reference_observations = int(np.sum(query_mask))

    n_inter_pair_observations = 0
    for frame in np.flatnonzero(frames):
        q_active = np.flatnonzero(query_mask[int(frame)])
        p_active = np.flatnonzero(partner_mask[int(frame)])
        for qi in q_active:
            q_physical = int(result.query_physical_chain_indices[int(qi)])
            for pj in p_active:
                if q_physical == int(result.partner_physical_chain_indices[int(pj)]):
                    continue
                n_inter_pair_observations += 1

    for frame_raw in np.flatnonzero(frames):
        frame = int(frame_raw)
        start = int(result.frame_offsets[frame])
        stop = int(result.frame_offsets[frame + 1])
        if stop <= start:
            continue

        distance = result.distance_nm[start:stop]
        within = distance <= cutoff
        if not np.any(within):
            continue

        q_site = result.query_site_index[start:stop][within]
        p_site = result.partner_site_index[start:stop][within]
        q_chain = (q_site // result.n_query_residues).astype(np.int32, copy=False)
        p_chain = (p_site // result.n_partner_residues).astype(np.int32, copy=False)
        q_res = (q_site % result.n_query_residues).astype(np.int32, copy=False)
        p_res = (p_site % result.n_partner_residues).astype(np.int32, copy=False)

        q_physical = result.query_physical_chain_indices[q_chain]
        p_physical = result.partner_physical_chain_indices[p_chain]
        same_physical = q_physical == p_physical

        # Intrachain: require both chain-side masks.  Symmetrize square
        # homotypic maps so the lower triangle can be displayed directly.
        intra = same_physical & (np.abs(q_res - p_res) >= min_sep)
        if np.any(intra):
            iq = q_chain[intra]
            ip = p_chain[intra]
            ir = q_res[intra]
            jr = p_res[intra]
            accepted = query_mask[frame, iq] & partner_mask[frame, ip]
            if np.any(accepted):
                ir = ir[accepted]
                jr = jr[accepted]
                np.add.at(intra_counts, (ir, jr), 1.0)
                if result.symmetric_homotypic:
                    np.add.at(intra_counts, (jr, ir), 1.0)

        inter = ~same_physical
        if not np.any(inter):
            continue

        iq = q_chain[inter]
        ip = p_chain[inter]
        ir = q_res[inter]
        jr = p_res[inter]

        forward = query_mask[frame, iq] & partner_mask[frame, ip]
        if np.any(forward):
            np.add.at(inter_counts, (ir[forward], jr[forward]), 1.0)

        if result.symmetric_homotypic:
            # One canonical chain-pair event represents both possible reference
            # directions.  Apply masks independently to the reverse direction.
            reverse = query_mask[frame, ip] & partner_mask[frame, iq]
            if np.any(reverse):
                np.add.at(inter_counts, (jr[reverse], ir[reverse]), 1.0)

    if n_intra_observations > 0:
        intra = intra_counts / float(n_intra_observations)
    else:
        intra = np.full_like(intra_counts, np.nan)

    if normalization == "per_reference_chain":
        inter_denominator = n_inter_reference_observations
    else:
        inter_denominator = n_inter_pair_observations

    if inter_denominator > 0:
        inter = inter_counts / float(inter_denominator)
    else:
        inter = np.full_like(inter_counts, np.nan)

    combined: Optional[np.ndarray]
    if result.symmetric_homotypic and intra.shape[0] == intra.shape[1]:
        combined = np.full_like(intra, np.nan)
        lower = np.tril_indices(intra.shape[0], k=-1)
        upper = np.triu_indices(inter.shape[0], k=0)
        combined[lower] = intra[lower]
        combined[upper] = inter[upper]
    else:
        combined = None

    selected_query_labels = tuple(
        label
        for index, label in enumerate(result.query_chain_labels)
        if np.any(query_mask[:, index])
    )
    selected_partner_labels = tuple(
        label
        for index, label in enumerate(result.partner_chain_labels)
        if np.any(partner_mask[:, index])
    )

    return ResidueContactMapResult(
        intra=intra,
        inter=inter,
        combined=combined,
        intra_stderr=np.zeros_like(intra),
        inter_stderr=np.zeros_like(inter),
        combined_stderr=None if combined is None else np.zeros_like(combined),
        query_residue_numbers=result.query_residue_numbers.copy(),
        partner_residue_numbers=result.partner_residue_numbers.copy(),
        query_residue_names=result.query_residue_names,
        partner_residue_names=result.partner_residue_names,
        query_chain_labels=selected_query_labels,
        partner_chain_labels=selected_partner_labels,
        cutoff_nm=cutoff,
        max_cutoff_nm=result.max_cutoff_nm,
        min_intra_sequence_separation=min_sep,
        inter_normalization=normalization,
        n_frames_selected=int(np.sum(frames)),
        n_intra_chain_observations=n_intra_observations,
        n_inter_reference_observations=n_inter_reference_observations,
        n_inter_pair_observations=n_inter_pair_observations,
        symmetric_homotypic=result.symmetric_homotypic,
        n_sets=1,
        set_labels=(),
        aggregation="individual",
    )


def _nan_mean_stderr(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(stack, dtype=np.float64)
    finite = np.isfinite(values)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, values, 0.0), axis=0)
    mean = np.full(values.shape[1:], np.nan, dtype=np.float64)
    valid = count > 0
    mean[valid] = total[valid] / count[valid]

    stderr = np.zeros(values.shape[1:], dtype=np.float64)
    enough = count >= 2
    if np.any(enough):
        deviations = np.where(finite, values - mean[None, ...], 0.0)
        variance = np.zeros_like(mean)
        variance[enough] = np.sum(deviations * deviations, axis=0)[enough] / (count[enough] - 1)
        stderr[enough] = np.sqrt(np.maximum(variance[enough], 0.0) / count[enough])
    return mean, stderr


def average_residue_contact_maps(
    results: Sequence[ResidueContactMapResult],
    *,
    set_labels: Optional[Sequence[str]] = None,
) -> ResidueContactMapResult:
    """Equal-weight average of independently normalized set-level maps."""
    members = list(results)
    if not members:
        raise ValueError("results is empty")
    first = members[0]

    for member in members[1:]:
        if member.inter_normalization != first.inter_normalization:
            raise ValueError("contact maps use different inter_normalization values")
        if member.intra.shape != first.intra.shape or member.inter.shape != first.inter.shape:
            raise ValueError("contact maps have incompatible shapes")
        if not np.array_equal(member.query_residue_numbers, first.query_residue_numbers):
            raise ValueError("contact maps have different query residue axes")
        if not np.array_equal(member.partner_residue_numbers, first.partner_residue_numbers):
            raise ValueError("contact maps have different partner residue axes")
        if not math.isclose(member.cutoff_nm, first.cutoff_nm, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("contact maps use different contact cutoffs")
        if member.min_intra_sequence_separation != first.min_intra_sequence_separation:
            raise ValueError("contact maps use different sequence-separation filters")

    intra, intra_stderr = _nan_mean_stderr(np.stack([member.intra for member in members], axis=0))
    inter, inter_stderr = _nan_mean_stderr(np.stack([member.inter for member in members], axis=0))

    if all(member.combined is not None for member in members):
        combined, combined_stderr = _nan_mean_stderr(
            np.stack([member.combined for member in members if member.combined is not None], axis=0)
        )
    else:
        combined = None
        combined_stderr = None

    labels = (
        tuple(str(label) for label in set_labels)
        if set_labels is not None
        else tuple(f"set{index + 1}" for index in range(len(members)))
    )
    if len(labels) != len(members):
        raise ValueError("set_labels length does not match results")

    return ResidueContactMapResult(
        intra=intra,
        inter=inter,
        combined=combined,
        intra_stderr=intra_stderr,
        inter_stderr=inter_stderr,
        combined_stderr=combined_stderr,
        query_residue_numbers=first.query_residue_numbers.copy(),
        partner_residue_numbers=first.partner_residue_numbers.copy(),
        query_residue_names=first.query_residue_names,
        partner_residue_names=first.partner_residue_names,
        query_chain_labels=tuple(
            dict.fromkeys(label for member in members for label in member.query_chain_labels)
        ),
        partner_chain_labels=tuple(
            dict.fromkeys(label for member in members for label in member.partner_chain_labels)
        ),
        cutoff_nm=first.cutoff_nm,
        max_cutoff_nm=min(member.max_cutoff_nm for member in members),
        min_intra_sequence_separation=first.min_intra_sequence_separation,
        inter_normalization=first.inter_normalization,
        n_frames_selected=int(sum(member.n_frames_selected for member in members)),
        n_intra_chain_observations=int(
            sum(member.n_intra_chain_observations for member in members)
        ),
        n_inter_reference_observations=int(
            sum(member.n_inter_reference_observations for member in members)
        ),
        n_inter_pair_observations=int(sum(member.n_inter_pair_observations for member in members)),
        symmetric_homotypic=all(member.symmetric_homotypic for member in members),
        n_sets=len(members),
        set_labels=labels,
        aggregation="sets",
    )

from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import mdtraj as md
import numpy as np
from openmm import Vec3, unit
from openmm.app import Topology, element

FileLike = Union[str, Path, io.BytesIO, io.StringIO]

# --- Data containers ---------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str  # e.g. "CA"
    element: str  # 'H', 'C', 'O', 'N', 'S', 'P' 'CL', 'SOD', 'MG', 'CA'
    resname: str  # e.g. "ALA"
    chain: str  # original PDB chain ID
    resnum: int  # residue sequence number
    x: float
    y: float
    z: float
    seg: str  # segment ID (may be "")

    def __repr__(self) -> str:
        return f"<atom {self.name} {self.resname} {self.resnum} {self.chain} {self.seg}>"


@dataclass
class Residue:
    resname: str
    chain: str  # original PDB chain ID
    resnum: int
    seg: str  # segment ID
    atoms: list[Atom] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<residue {self.resname} {self.resnum} {self.chain} {self.seg}>"


@dataclass
class Chain:
    key_id: str  # key used in Structure.models[...].chains
    residues: list[Residue] = field(default_factory=list)
    seg_id: Optional[str] = None  # segment ID if grouping by seg
    chain_id: Optional[str] = None  # chain ID from PDB

    def __repr__(self) -> str:
        return f"<chain {self.key_id} : segment {self.seg_id} chain {self.chain_id}>"


@dataclass
class Model:
    model_id: int
    chain: dict[str, Chain] = field(default_factory=dict)  # key_id -> Chain
    residues: list[Residue] = field(default_factory=list)
    atoms: list[Atom] = field(default_factory=list)

    # Optional parent trajectory/structure and frame index
    _parent: Optional[Structure] = field(default=None, repr=False, compare=False)
    _frame_index: int = field(default=0, repr=False, compare=False)

    _atom_index_cache: Optional[dict[int, int]] = field(default=None, repr=False, compare=False)

    def chains(self) -> Iterator[Chain]:
        return iter(self.chain.values())

    def iter_residues(self) -> Iterator[Residue]:
        for c in self.chain.values():
            yield from c.residues

    def iter_atoms(self) -> Iterator[Atom]:
        for c in self.chain.values():
            for r in c.residues:
                yield from r.atoms

    def __repr__(self) -> str:
        n_chains = self.nchains()
        n_res = self.nresidues()
        n_atoms = self.natoms()
        return f"<{n_chains} chains, {n_res} residues, {n_atoms} atoms>"

    __str__ = __repr__

    def nchains(self):
        return len(self.chain)

    def nresidues(self):
        return sum(len(c.residues) for c in self.chain.values())

    def natoms(self):
        return len(self.atoms)

    def positions(self):
        """
        Return positions as an OpenMM Quantity[list[Vec3]].
        - Static models: internal coordinates are in Å, converted to nm.
        - Trajectory-backed models: parent._coords_nm already in nm.
        """
        if self._has_parent_coords():
            coords_nm = self._parent._coords_nm[self._frame_index]  # (natoms, 3)
            vecs = [Vec3(float(x), float(y), float(z)) for x, y, z in coords_nm]
            return unit.Quantity(vecs, unit.nanometer)

        # Static: use Atom coordinates in Å, convert to nm
        vecs = [Vec3(a.x, a.y, a.z) for a in self.atoms]
        return unit.Quantity(vecs, unit.angstrom).in_units_of(unit.nanometer)

    def topology(self):
        top = Topology()
        for c in self.chains():
            chain = top.addChain(c.key_id)
            for r in c.residues:
                rname = r.resname
                res = top.addResidue(rname, chain)
                for a in r.atoms:
                    sym = (getattr(a, "element", "") or "").upper()
                    try:
                        el = element.Element.getBySymbol(sym)
                    except Exception:
                        el = element.carbon
                    top.addAtom(a.name, element=el, residue=res)
        return top

    # ---- selections on a single Model ----

    def _select_by_index_set(self, keep: set[int]) -> Model:
        """Internal: build a new Model with only atoms whose model-local indices are in `keep`."""
        m2 = Model(model_id=self.model_id)
        if not keep or not self.atoms:
            return m2

        use_parent = self._has_parent_coords()
        parent_frame_coords = None
        if use_parent:
            parent_frame_coords = self._parent._coords_nm[self._frame_index]  # (natoms, 3)

        running_idx = -1
        for key, ch in self.chain.items():
            new_chain = Chain(
                key_id=ch.key_id,
                seg_id=getattr(ch, "seg_id", None),
                chain_id=getattr(ch, "chain_id", None),
            )
            for r in ch.residues:
                kept_atoms: list[Atom] = []
                for a in r.atoms:
                    running_idx += 1
                    if running_idx in keep:
                        if use_parent:
                            x_nm, y_nm, z_nm = parent_frame_coords[running_idx]
                            new_atom = Atom(
                                serial=a.serial,
                                name=a.name,
                                element=a.element,
                                resname=a.resname,
                                chain=a.chain,
                                resnum=a.resnum,
                                x=float(x_nm * 10.0),  # back to Å for the static model
                                y=float(y_nm * 10.0),
                                z=float(z_nm * 10.0),
                                seg=a.seg,
                            )
                        else:
                            # static structure: reuse atom object
                            new_atom = a
                        kept_atoms.append(new_atom)
                        m2.atoms.append(new_atom)
                if kept_atoms:
                    new_res = Residue(
                        resname=r.resname,
                        chain=r.chain,
                        resnum=r.resnum,
                        seg=r.seg,
                        atoms=kept_atoms,
                    )
                    new_chain.residues.append(new_res)
                    m2.residues.append(new_res)

            if new_chain.residues:
                m2.chain[key] = new_chain

        return m2

    @staticmethod
    def _flatten_indices(indices: Union[list[int], list[list[int]]]) -> list[int]:
        if not indices:
            return []
        if isinstance(indices[0], (list, tuple)):
            out: list[int] = []
            for sub in indices:  # type: ignore[assignment]
                out.extend(int(i) for i in sub)
            return out
        return [int(i) for i in indices]  # type: ignore[return-value]

    # --- internal coordinate helpers for trajectory-backed models ---

    def _has_parent_coords(self) -> bool:
        """Return True if this model is backed by parent Structure coordinates."""
        p = self._parent
        return p is not None and getattr(p, "_coords_nm", None) is not None

    def _coord_angstrom(self, idx: int) -> tuple[float, float, float]:
        """
        Return atom coordinates (Å) for atom index `idx` in this model.

        If the model is trajectory-backed, pull from parent._coords_nm[frame].
        Otherwise, use Atom.x/y/z as before.
        """
        if self._has_parent_coords():
            coords_nm = self._parent._coords_nm  # shape (n_frames, n_atoms, 3)
            x_nm, y_nm, z_nm = coords_nm[self._frame_index, idx, :]
            # nm -> Å
            return float(x_nm * 10.0), float(y_nm * 10.0), float(z_nm * 10.0)
        else:
            a = self.atoms[idx]
            return float(a.x), float(a.y), float(a.z)

    def _center_of_geometry(self, indices: list[int]) -> tuple[float, float, float]:
        """
        Internal: center of geometry (Å) for a set of atom indices in this model.
        Works both for static and trajectory-backed models.
        """
        if not indices:
            raise ValueError("center of geometry requires at least one atom index")

        n_atoms = self.natoms()
        for idx in indices:
            if idx < 0 or idx >= n_atoms:
                raise IndexError(f"Atom index {idx} is out of range for model with {n_atoms} atoms")

        if self._has_parent_coords():
            # Fast path: use parent trajectory coordinates (nm) and numpy
            coords_nm = self._parent._coords_nm[self._frame_index]  # type: ignore[union-attr]
            idx_arr = np.asarray(indices, dtype=np.int64)
            cog_ang = coords_nm[idx_arr].mean(axis=0) * 10.0  # nm → Å
            return float(cog_ang[0]), float(cog_ang[1]), float(cog_ang[2])

        # Static structure: accumulate directly from Atom coordinates (already Å)
        sx = sy = sz = 0.0
        for idx in indices:
            a = self.atoms[idx]
            sx += float(a.x)
            sy += float(a.y)
            sz += float(a.z)

        inv_n = 1.0 / float(len(indices))
        return sx * inv_n, sy * inv_n, sz * inv_n

    def distance(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
    ) -> unit.Quantity:
        """
        Center-of-geometry distance between two atom groups, as an OpenMM Quantity in nm.
        """
        flat_a = self._flatten_indices(group_a)
        flat_b = self._flatten_indices(group_b)

        if not flat_a or not flat_b:
            raise ValueError("distance requires both groups to contain at least one atom")

        cx_a, cy_a, cz_a = self._center_of_geometry(flat_a)
        cx_b, cy_b, cz_b = self._center_of_geometry(flat_b)

        dx = cx_a - cx_b
        dy = cy_a - cy_b
        dz = cz_a - cz_b

        dist_ang = (dx * dx + dy * dy + dz * dz) ** 0.5  # Å
        return unit.Quantity(dist_ang, unit.angstrom).in_units_of(unit.nanometer)

    def _group_cog(self, group: Union[list[int], list[list[int]]]) -> np.ndarray:
        """
        Center-of-geometry (Å) for a group of atoms, returned as a 3D numpy vector.
        """
        flat = self._flatten_indices(group)
        if not flat:
            raise ValueError("group must contain at least one atom index")

        cx, cy, cz = self._center_of_geometry(flat)
        return np.array((cx, cy, cz), dtype=float)

    @staticmethod
    def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
        """
        Angle between vectors u and v in radians.
        """
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0:
            raise ValueError("Cannot compute angle with zero-length vector")
        cosang = float(np.dot(u, v) / (nu * nv))
        # numerical safety
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.arccos(cosang))

    @staticmethod
    def _dihedral_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
        """
        Signed dihedral angle (−π..π) defined by four points, matching OpenMM's dihedral().
        """
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        # normals
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm == 0.0 or n2_norm == 0.0:
            raise ValueError("Cannot compute dihedral with collinear points")

        n1 /= n1_norm
        n2 /= n2_norm
        b2_unit = b2 / np.linalg.norm(b2)

        m1 = np.cross(n1, b2_unit)

        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return float(-np.arctan2(y, x))

    def angle_norm(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_a1: Union[list[int], list[list[int]]],
        group_a2: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_b1: Union[list[int], list[list[int]]],
        group_b2: Union[list[int], list[list[int]]],
    ) -> unit.Quantity:
        """
        Angle (radians) between two plane normals, matching set_umbrella_angle_norm.

        Plane A: defined by centroids of (group_a, group_a1, group_a2)
        Plane B: defined by centroids of (group_b, group_b1, group_b2)
        """
        A0 = self._group_cog(group_a)
        A1 = self._group_cog(group_a1)
        A2 = self._group_cog(group_a2)
        B0 = self._group_cog(group_b)
        B1 = self._group_cog(group_b1)
        B2 = self._group_cog(group_b2)

        vA1 = A1 - A0
        vA2 = A2 - A0
        vB1 = B1 - B0
        vB2 = B2 - B0

        nA = np.cross(vA1, vA2)
        nB = np.cross(vB1, vB2)

        nu = np.linalg.norm(nA)
        nv = np.linalg.norm(nB)
        if nu == 0.0 or nv == 0.0:
            raise ValueError("Cannot compute plane-normal angle for degenerate plane")

        cosang = float(np.dot(nA, nB) / (nu * nv))
        cosang = max(-1.0, min(1.0, cosang))
        angle = float(np.arccos(cosang))

        return unit.Quantity(angle, unit.radian)

    def dihedral(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
        group_d: Union[list[int], list[list[int]]],
    ) -> unit.Quantity:
        """
        Dihedral angle (radians, −π..π) between four centroids,
        matching set_umbrella_dihedral geometry.
        """
        p1 = self._group_cog(group_a)
        p2 = self._group_cog(group_b)
        p3 = self._group_cog(group_c)
        p4 = self._group_cog(group_d)

        angle = self._dihedral_angle(p1, p2, p3, p4)
        return unit.Quantity(angle, unit.radian)

    def angle(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
    ) -> unit.Quantity:
        """
        Angle (radians) corresponding to the angle() terms used in
        set_umbrella_angle.

        Returns
        -------
          theta = angle(group_a,  group_b,  group_c)
        """
        A = self._group_cog(group_a)
        B = self._group_cog(group_b)
        C = self._group_cog(group_c)

        # angle(g1,g2,g3) = angle between (g1 - g2) and (g3 - g2)
        theta = self._angle_between(A - B, C - B)

        return unit.Quantity(theta, unit.radian)

    def select_byindex(self, indices: Union[list[int], list[list[int]]]) -> Model:
        """
        Return a new Model containing only atoms at the given 0-based indices
        (per this model's atom order). Accepts a flat list or list-of-lists.
        Duplicates removed, negatives ignored, indices sorted before applying.
        """
        flat = self._flatten_indices(indices)
        keep = {i for i in flat if isinstance(i, int) and i >= 0}
        return self._select_by_index_set(keep)

    def select_CA(self) -> Model:
        """Return a new Model containing only CA atoms."""
        keep = {i for i, a in enumerate(self.atoms) if a.name == "CA"}
        return self._select_by_index_set(keep)

    def select_bystring(self, spec: str) -> Model:
        """
        Return a new Model using a textual selection `spec` via StructureSelector.
        This method builds a temporary single-model Structure to reuse the selector.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("select_bystring requires a non-empty selection string")

        # normalize "H271:2-91" -> "H271.2-91" (first ':' as chain/res separator)
        raw = spec.strip()
        if "." not in raw and ":" in raw:
            head, tail = raw.split(":", 1)
            if tail and tail.lstrip() and tail.lstrip()[0].isdigit():
                raw = f"{head}.{tail}"

        # Build a temporary Structure with this model only
        temp_struct = Structure(models=[self])

        sel = StructureSelector(raw)
        # list-of-lists (per selector semantics)
        idx_lists = sel.atom_lists(temp_struct, model_index=0)
        return self.select_byindex(idx_lists)

    def mdtraj_trajectory(self):
        top = md.Topology.from_openmm(self.topology())

        if self._has_parent_coords():
            # one-frame view from parent coords, already in nm
            coords_nm = self._parent._coords_nm[self._frame_index]  # (natoms, 3)
        else:
            # static model: use Atom coords (Å -> nm)
            coords_nm = [(a.x / 10.0, a.y / 10.0, a.z / 10.0) for a in self.atoms]

        if len(coords_nm) == 0:
            traj = md.Trajectory(xyz=np.zeros((1, 0, 3), dtype=float), topology=top)
            return traj

        xyz = np.array([coords_nm], dtype=np.float32)  # (1, natoms, 3) nm
        traj = md.Trajectory(xyz=xyz, topology=top)
        return traj

    def sasa_by_residue(
        self,
        *,
        probe_radius: float = 0.14,
        n_sphere_points: int = 960,
        radii: str = "bondi",
    ) -> list[float]:
        """
        Fast SASA (nm^2) per residue using MDTraj Shrake–Rupley with Bondi radii.
        Parameters
        ----------
        probe_radius : float  (nm)
        n_sphere_points : int
        radii : str   (currently 'bondi' only; MDTraj uses element radii table)
        """

        if radii.lower() != "bondi":
            raise ValueError("Only 'bondi' radii are supported with the MDTraj backend.")

        traj = self.mdtraj_trajectory()
        if traj.n_atoms == 0:
            return []

        # MDTraj expects nm for radii; returns nm^2
        sasa_nm2 = md.shrake_rupley(
            traj,
            n_sphere_points=int(n_sphere_points),
            mode="residue",
        )  # shape (1, n_residues)
        per_res_nm2 = sasa_nm2[0]

        return per_res_nm2.tolist()


@dataclass
class Structure:
    models: list[Model] = field(default_factory=list)

    # Optional trajectory coordinates (nm), shape (n_models, n_atoms, 3)
    _coords_nm: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Model, list[Model]]:
        return self.models[idx]

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self) -> Iterator[Model]:
        return iter(self.models)

    def __repr__(self) -> str:
        lenmod = len(self.models)
        if self._coords_nm is not None:
            ncoord = len(self._coords_nm)
            return f"<Structure with {ncoord} coordinate frames>"
        else:
            return f"<Structure with {lenmod} models"

    @property
    def model(self) -> Model:
        """Return the first model (common for single-model files)."""
        if not self.models:
            raise ValueError("Structure has no models")
        return self.models[0]

    def nchains(self) -> int:
        return self.models[0].nchains()

    def nresidues(self) -> int:
        return self.models[0].nresidues()

    def natoms(self) -> int:
        return self.models[0].natoms()

    def positions(self, model_index: int = 0):
        """Positions for the selected model as Quantity[list[Vec3]] in nm."""
        return self.models[model_index].positions()

    def topology(self):
        return self.models[0].topology()

    def select_CA(self) -> Structure:
        """Apply CA selection to each model; return a new Structure."""
        out = Structure()
        for m in self.models:
            out.models.append(m.select_CA())
        if not out.models:
            out.models.append(Model(model_id=1))
        return out

    def select_byindex(self, indices: Union[list[int], list[list[int]]]) -> Structure:
        """
        Apply the same index selection to each model; return a new Structure.
        Indices are interpreted per-model (0-based within each model).
        """
        out = Structure()
        for m in self.models:
            out.models.append(m.select_byindex(indices))
        if not out.models:
            out.models.append(Model(model_id=1))
        return out

    def select_bystring(self, spec: str) -> Structure:
        """
        Apply textual selection to each model independently (chains/residues resolved per model);
        return a new Structure.
        """
        out = Structure()
        for m in self.models:
            out.models.append(m.select_bystring(spec))
        if not out.models:
            out.models.append(Model(model_id=1))
        return out

    def sasa_by_residue(
        self,
        *,
        model_index: int = 0,
        probe_radius: float = 0.14,
        n_sphere_points: int = 960,
        radii: str = "bondi",
    ) -> list[float]:
        """
        Compute SASA (nm^2) by residue for a chosen model (default 0) via MDTraj.
        """
        if not self.models:
            return []
        if model_index < 0 or model_index >= len(self.models):
            raise IndexError(f"model_index {model_index} out of range (0..{len(self.models)-1})")
        return self.models[model_index].sasa_by_residue(
            probe_radius=probe_radius,
            n_sphere_points=n_sphere_points,
            radii=radii,
        )

    def distance(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
    ) -> list[unit.Quantity]:
        """
        Center-of-geometry distance between two atom groups for all models.

        If trajectory-backed (Structure._coords_nm is set), uses a vectorized
        numpy implementation over all frames. Otherwise falls back to per-model
        computation.
        """
        if not self.models:
            return []

        # Fast path for trajectory-backed structures
        if self._coords_nm is not None:
            flat_a = Model._flatten_indices(group_a)  # type: ignore[attr-defined]
            flat_b = Model._flatten_indices(group_b)  # type: ignore[attr-defined]

            if not flat_a or not flat_b:
                raise ValueError("distance requires both groups to contain at least one atom")

            n_atoms = self.natoms()
            for idx in flat_a + flat_b:
                if idx < 0 or idx >= n_atoms:
                    raise IndexError(
                        f"Atom index {idx} is out of range for structure with {n_atoms} atoms"
                    )

            idx_a = np.asarray(flat_a, dtype=np.int64)
            idx_b = np.asarray(flat_b, dtype=np.int64)

            coords_nm = self._coords_nm  # (n_frames, n_atoms, 3)
            cog_a_nm = coords_nm[:, idx_a, :].mean(axis=1)
            cog_b_nm = coords_nm[:, idx_b, :].mean(axis=1)
            diff_nm = cog_a_nm - cog_b_nm
            dist_nm = np.linalg.norm(diff_nm, axis=1)

            return [unit.Quantity(float(d), unit.nanometer) for d in dist_nm]

        # Static / multi-model fallback
        return [m.distance(group_a, group_b) for m in self.models]

    def angle_norm(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_a1: Union[list[int], list[list[int]]],
        group_a2: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_b1: Union[list[int], list[list[int]]],
        group_b2: Union[list[int], list[list[int]]],
    ) -> list[unit.Quantity]:
        """
        Plane-normal angle (radians) between two planes for all models.

        Geometry matches the umbrella in set_umbrella_angle_norm.
        """
        if not self.models:
            return []
        out: list[unit.Quantity] = []
        for m in self.models:
            out.append(
                m.angle_norm(
                    group_a,
                    group_a1,
                    group_a2,
                    group_b,
                    group_b1,
                    group_b2,
                )
            )
        return out

    def dihedral(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
        group_d: Union[list[int], list[list[int]]],
    ) -> list[unit.Quantity]:
        """
        Dihedral angle (radians, −π..π) between four centroids for all models.

        Geometry matches the umbrella in set_umbrella_dihedral
        """
        if not self.models:
            return []
        out: list[unit.Quantity] = []
        for m in self.models:
            out.append(m.dihedral(group_a, group_b, group_c, group_d))
        return out

    def angle(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
    ) -> list[unit.Quantity]:
        """
        Rotation angles (radians) per model, matching set_umbrella_angle.

        """
        if not self.models:
            return []
        out: list[unit.Quantity] = []
        for m in self.models:
            out.append(m.angle(group_a, group_b, group_c))
        return out


# --- Parser ------------------------------------------------------------------


def _ensure_template_model(
    template: Union[Structure, Model, FileLike],
) -> tuple[Structure, Model]:
    """
    Normalize a template specification into (Structure, Model).

    template can be:
      - Model       -> wrapped into a single-model Structure
      - Structure   -> returns (template, template.model)
      - str/Path    -> treated as PDB-like coordinate file, read via PDBReader
    """
    if isinstance(template, Model):
        s = Structure(models=[template])
        return s, template

    if isinstance(template, Structure):
        return template, template.model

    # Assume it's a PDB-like file path or file-like
    reader = PDBReader()
    s = reader.read(template)  # type: ignore[arg-type]
    return s, s.model


class PDBReader:
    """
    Minimal, fast PDB reader
    - Supports MODEL/ENDMDL (multiple models).
    - Parses ATOM.
    - Groups atoms into chains keyed by SEGID when available; else by PDB chain ID with
      automatic suffixing (A, A1, A2, ...) when non-contiguous repeats occur.
    """

    def __new__(cls, file: Optional[FileLike] = None):
        self = super().__new__(cls)
        if file is None:
            return self
        return cls._read_direct(file)

    def read(self, file: FileLike) -> Structure:
        text_iter = self._open_text(file)
        return self._parse(text_iter)

    def from_string(self, pdb_text: str) -> Structure:
        return self._parse(pdb_text.splitlines())

    # -- internals --
    @staticmethod
    def _open_text(file: FileLike) -> Iterable[str]:
        """
        Yield text lines from a PDB(-like) source.

        - For StringIO/BytesIO, read from the in-memory buffer.
        - For filesystem paths, stream line-by-line (no full-file read).
        """
        if isinstance(file, io.StringIO):
            for line in file.getvalue().splitlines():
                yield line
            return

        if isinstance(file, io.BytesIO):
            text = io.TextIOWrapper(file, encoding="utf-8", newline="").read()
            for line in text.splitlines():
                yield line
            return

        p = Path(file)
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
                for line in fh:
                    yield line.rstrip("\n")
            return

        with open(p, encoding="utf-8", newline="") as fh:
            for line in fh:
                yield line.rstrip("\n")

    @classmethod
    def _read_direct(cls, file: FileLike) -> Structure:
        return cls._parse(cls._open_text(file))

    @staticmethod
    def _parse(lines: Iterable[str]) -> Structure:
        s = Structure()
        current_model: Optional[Model] = None

        # State for allocating fallback chain keys when SEGID is absent
        fallback_counts: dict[str, int] = {}
        last_chain_id_seen: Optional[str] = None

        def alloc_chain_key(m: Model, atom: Atom) -> str:
            """Return chain key for this atom per rules."""
            nonlocal last_chain_id_seen

            seg = atom.seg.strip()
            if seg:
                # Primary rule: group by segment ID
                last_chain_id_seen = atom.chain
                return seg

            # Fallback: group by PDB chain ID, splitting non-contiguous repeats
            cid = (atom.chain or "").strip() or " "
            if cid not in m.chain:
                last_chain_id_seen = cid
                return cid

            # Same contiguous block
            if last_chain_id_seen == cid:
                return cid

            # Non-contiguous repeat: allocate suffixed key
            n = fallback_counts.get(cid, 0) + 1
            fallback_counts[cid] = n
            key = f"{cid}{n}"
            last_chain_id_seen = cid
            return key

        def start_chain_if_needed(m: Model, key: str, atom: Atom) -> Chain:
            ch = m.chain.get(key)
            if ch is None:
                ch = Chain(key_id=key, residues=[], seg_id=(atom.seg.strip() or None))
                m.chain[key] = ch
            # record original PDB chain id
            ch.chain_id = atom.chain or " "
            return ch

        def add_atom_to_model(m: Model, atom: Atom):
            m.atoms.append(atom)
            key = alloc_chain_key(m, atom)
            chain = start_chain_if_needed(m, key, atom)

            rid = (atom.resname, atom.chain, atom.resnum, atom.seg)
            if not chain.residues or _res_id(chain.residues[-1]) != rid:
                chain.residues.append(Residue(*rid))
            chain.residues[-1].atoms.append(atom)

        for raw in lines:
            if not raw:
                continue
            rec = raw[0:6].strip().upper()

            if rec == "MODEL":
                model_id = _safe_int(raw[10:14], default=len(s.models) + 1) or len(s.models) + 1
                current_model = Model(model_id=model_id)
                s.models.append(current_model)
                fallback_counts = {}
                last_chain_id_seen = None
                continue

            if rec == "ENDMDL":
                current_model = None
                continue

            if rec == "ATOM":
                if current_model is None:
                    current_model = Model(model_id=1)
                    s.models.append(current_model)
                    fallback_counts = {}
                    last_chain_id_seen = None
                atom = _parse_atom_line(raw)
                add_atom_to_model(current_model, atom)
                continue

            if rec == "TER":
                last_chain_id_seen = None
                continue

        if not s.models:
            s.models.append(Model(model_id=1))
        return s


# --- parsing utilities -------------------------------------------------------


def _deduce_element(atomname: str, resname: str, element_hint: str = "") -> str:
    """
    Deduce an element symbol following user rules.
    Priority:
      1) Use PDB element column if present (uppercased, non-letters removed).
      2) Special cases from atom/residue names.
      3) First-letter rules C/N/H/S/P/O (after stripping leading digits in atom name).
      4) Fallback: atom name with digits removed (uppercased).
    """

    def clean(token: str) -> str:
        # keep only letters, upcase
        return re.sub(r"[^A-Za-z]", "", token or "").upper()

    # 1) PDB element column (columns 77-78)
    if element_hint and clean(element_hint):
        return clean(element_hint)

    an = clean(atomname)
    rn = clean(resname)

    # 2) Explicit mappings
    # Chloride / sodium / potassium aliases
    if an in {"CLA", "CL"} or atomname.upper() in {"CL-", "CLA"}:
        return "CL"
    if rn in {"CLA", "CL"}:
        return "CL"

    if an in {"NA", "SOD"} or atomname.upper() == "NA+":
        return "NA"
    if rn in {"NA", "SOD"}:
        return "NA"

    if an == "POT" or rn == "POT":
        return "K"

    # direct “use-name” set
    direct = {"MG", "CAL", "K", "LI", "FE", "CO", "MB"}
    if an in direct:
        return an
    if rn in direct:
        return rn

    # 3) First-letter rules after stripping leading digits from atom name
    atom_wo_lead_digits = re.sub(r"^\d+", "", atomname or "")
    atom_wo_digits = re.sub(r"\d", "", atom_wo_lead_digits).strip()
    if atom_wo_digits:
        ch0 = atom_wo_digits[0].upper()
        if ch0 in {"C", "N", "H", "S", "P", "O"}:
            return ch0

    # 4) Fallback: atom name without any digits, uppercased (e.g., "Cl1" -> "CL")
    fb = clean(atomname)
    return fb if fb else "X"


def _parse_atom_line(line: str) -> Atom:
    # PDB v3.3 column mapping, simplified
    raw_serial = line[4:11]
    s_serial = raw_serial.strip()
    if s_serial and all(ch == "*" for ch in s_serial):
        serial = 0
    else:
        serial = _safe_int(raw_serial, required=True)
    name = line[12:16].strip()
    resname = line[17:21].strip()
    chain = (line[21] if len(line) >= 22 else " ").strip()
    resnum = _safe_int(line[22:27], required=True)
    x = _safe_float(line[30:38], required=True)
    y = _safe_float(line[38:46], required=True)
    z = _safe_float(line[46:54], required=True)
    seg = (line[72:76] if len(line) >= 76 else " ").strip()
    element_hint = (line[76:78] if len(line) >= 78 else "").strip()
    element = _deduce_element(name, resname, element_hint)
    return Atom(
        serial=serial,
        name=name,
        element=element,
        resname=resname,
        chain=chain,
        resnum=resnum,
        x=x,
        y=y,
        z=z,
        seg=seg,
    )


def _safe_int(s: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
    try:
        return int(s.strip())
    except Exception:
        if required:
            raise ValueError(f"Expected integer in field '{s}'")
        return default


def _safe_float(s: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
    try:
        return float(s.strip())
    except Exception:
        if required:
            raise ValueError(f"Expected float in field '{s}'")
        return default


def _res_id(r: Residue) -> tuple[str, str, int, str]:
    return (r.resname, r.chain, r.resnum, r.seg)


# ---- StructureSelector ----------------------------------------------------------


class SelectionError(ValueError):
    """Raised when a selection term cannot be parsed or resolved."""


# ----------------------------- selection constants ---------------------------

# Protein residue names (3-letter codes) used by the "protein" keyword.
_AMINO_ACID_RESNAMES: set[str] = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HSD",
    "HSE",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

# Simple solvent/ion classes.
_WATER_RESNAMES: set[str] = {"HOH", "TIP3", "WAT", "SPC"}
_ION_RESNAMES: set[str] = {"SOD", "POT", "CLA", "MG", "NA"}


def _is_hydrogen(atom: Atom) -> bool:
    """Return True if this atom should be treated as hydrogen.

    Uses both the deduced element and common PDB atom-name patterns
    (H*, ?H*), to be robust against imperfect element assignment.
    """
    name = (atom.name or "").strip().upper()
    if not name:
        return False
    el = (getattr(atom, "element", "") or "").upper()
    if el == "H":
        return True
    # H*, 1H*, 2H*, etc.
    if name[0] == "H":
        return True
    if len(name) >= 2 and name[1] == "H":
        return True
    return False


def _is_element_like(atom: Atom, symbol: str) -> bool:
    """Heuristic element classifier using Atom.element and atom name."""
    symbol = symbol.upper()
    el = (getattr(atom, "element", "") or "").upper()
    if el == symbol:
        return True
    name = (atom.name or "").strip().upper()
    if not name:
        return False
    if name[0] == symbol:
        return True
    if len(name) >= 2 and name[1] == symbol:
        return True
    return False


def _residue_matches_groups(res: Residue, flags: frozenset[str]) -> bool:
    """
    Apply residue-group filters ("protein", "water", "ions") to a Residue.
    If no such flags are present, always returns True.
    """
    if not {"protein", "water", "ions"} & flags:
        return True
    name = (res.resname or "").strip().upper()
    if "protein" in flags and name not in _AMINO_ACID_RESNAMES:
        return False
    if "water" in flags and name not in _WATER_RESNAMES:
        return False
    if "ions" in flags and name not in _ION_RESNAMES:
        return False
    return True


# ----------------------------- parsing primitives ----------------------------


@dataclass(frozen=True)
class ResidueSelector:
    """Represents residue selection for a chain (or all chains)."""

    all_residues: bool
    ranges: tuple[tuple[int, int], ...] = ()  # inclusive ranges; singletons are (n, n)

    @staticmethod
    def parse(spec: str) -> ResidueSelector:
        s = spec.strip().lower()
        if s == "all":
            return ResidueSelector(all_residues=True)

        # '.' and ':' kept as before; '+' added as an extra union separator.
        toks = [t for t in re.split(r"[.:+]", spec) if t.strip()]
        ranges: list[tuple[int, int]] = []
        for t in toks:
            t = t.strip()
            if "-" in t:
                a, b = t.split("-", 1)
                try:
                    lo = int(a)
                    hi = int(b)
                except ValueError as e:
                    raise SelectionError(f"Invalid residue range '{t}' in '{spec}'") from e
                if lo > hi:
                    lo, hi = hi, lo
                ranges.append((lo, hi))
            else:
                try:
                    n = int(t)
                except ValueError as e:
                    raise SelectionError(f"Invalid residue token '{t}' in '{spec}'") from e
                ranges.append((n, n))

        if not ranges:
            raise SelectionError(f"Empty residue spec '{spec}'")

        return ResidueSelector(all_residues=False, ranges=tuple(ranges))

    def contains(self, resnum: int) -> bool:
        if self.all_residues:
            return True
        return any(lo <= resnum <= hi for (lo, hi) in self.ranges)


@dataclass(frozen=True)
class AtomSelector:
    """
    Atom-level constraints for a Term.

    names:
        Specific atom names to include (e.g. ("CA", "CB")).  Comparison is
        case-insensitive against Atom.name.

    flags:
        Keyword filters applied in addition to names:
          - "heavy"      : exclude hydrogens
          - "hydrogens"  : only hydrogens
          - "carbons"    : only carbons
          - "nitrogens"  : only nitrogens
          - "oxygens"    : only oxygens
          - "protein"    : only residues in _AMINO_ACID_RESNAMES
          - "water"      : only residues in _WATER_RESNAMES
          - "ions"       : only residues in _ION_RESNAMES
    """

    names: Optional[tuple[str, ...]] = None
    flags: frozenset[str] = frozenset()

    def has_residue_filters(self) -> bool:
        return bool({"protein", "water", "ions"} & self.flags)

    def has_atom_filters(self) -> bool:
        return bool(self.names) or bool(self.flags - {"protein", "water", "ions"})


@dataclass(frozen=True)
class Term:
    """
    One selection term:
      - chains (or None for all chains)
      - residue selector (or all)
      - optional atom selector.
    """

    chains: Optional[tuple[str, ...]]  # None => all chains
    residues: ResidueSelector
    atom_selector: Optional[AtomSelector] = None


def _parse_chain_list(s: str) -> tuple[str, ...]:
    # Chains separated by ':' or '+' (e.g. 'A:B:C' or 'A+B+C').
    ids = [tok.strip() for tok in re.split(r"[:+]", s) if tok.strip()]
    if not ids:
        raise SelectionError(f"Empty chain list in '{s}'")
    return tuple(ids)


def _looks_like_residue_spec(s: str) -> bool:
    s = s.strip().lower()
    if s == "all":
        return True
    # digits, dashes, dots, colons, plus => residue expressions (e.g., "2-91:93-94").
    return bool(re.fullmatch(r"[0-9][0-9:.\-+]*", s))


# Atom / residue macro keywords ------------------------------------------------

_RESIDUE_GROUP_KEYWORDS = {"protein", "proteins", "water", "waters", "ion", "ions"}

_ATOM_FLAG_ALIASES = {
    "heavy": "heavy",
    "heavies": "heavy",
    "hydrogen": "hydrogens",
    "hydrogens": "hydrogens",
    "carbon": "carbons",
    "carbons": "carbons",
    "nitrogen": "nitrogens",
    "nitrogens": "nitrogens",
    "oxygen": "oxygens",
    "oxygens": "oxygens",
    "protein": "protein",
    "proteins": "protein",
    "water": "water",
    "waters": "water",
    "ion": "ions",
    "ions": "ions",
}


def _parse_atom_spec(spec: str) -> AtomSelector:
    """
    Parse the atom part of a term, supporting:

      - Explicit names:  'CA', 'CA:CB', 'CA+CB'
      - Keywords: 'heavy', 'carbons', 'hydrogens', 'nitrogens', 'oxygens'
      - Residue group keywords: 'protein', 'water', 'ions'
    """
    spec = spec.strip()
    if not spec:
        raise SelectionError("Empty atom spec")

    names: list[str] = []
    flags: set[str] = set()

    for raw in re.split(r"[:+]", spec):
        tok = raw.strip()
        if not tok:
            continue
        key = tok.lower()
        if key in _ATOM_FLAG_ALIASES:
            flags.add(_ATOM_FLAG_ALIASES[key])
        else:
            # treat as literal atom-name filter
            names.append(tok.upper())

    if not names and not flags:
        raise SelectionError(f"Could not parse atom spec '{spec}'")

    return AtomSelector(names=tuple(names) if names else None, flags=frozenset(flags))


# ----------------------------- public selector -------------------------------


class StructureSelector:
    """
    Parse domain spec strings and produce atom lists from your Structure/Model.

    Extended semantics (superset of original behaviour):

      • If input is a single string: commas and/or whitespace separate terms that are
        COMBINED into one selection (one atom list if any explicit chains are present).
        Example: "A:2-10,B:5-15" -> one combined list over A:2-10 and B:5-15.
      • If input is a list/tuple of strings: each element is a GROUP; each group yields
        its own atom list(s). Example: ["A:2-10", "B:2-10"] -> two separate lists.

      • Inside a group:
          – If ANY term specifies chains => return ONE atom list pooled across those chains.
          – If NO term specifies chains  => return ONE atom list PER CHAIN (same residue spec).

    Grammar (per term, forgiving):

      - Chain lists use ':' or '+' (e.g., 'A:B:C' or 'A+B+C').
      - Chain vs residues separated by first '.' (e.g., 'A:B.2-91').
      - An optional second '.' introduces atom selection:

            A:B:C.2-90        # residues only
            A:B:C.2-90.CA     # specific atom names
            A:B:C.2-40:50-60.CA:CB
            A:B:C.CA          # all residues, atom name CA
            A:B:C.heavy       # heavy atoms only in these chains
            2-90.CA           # all chains, residues 2-90, atoms CA
            protein.CA        # CA atoms in protein residues/chains
            protein           # all atoms in protein residues

      - Residue ranges support ':' or '+' as union separators ("2-10:20-30", "2-10+20-30").
      - Atom-name lists support ':' or '+' ("CA:CB", "CA+CB").
      - 'all' alone => all chains, all residues, all atoms.
      - Terms in a group separated by commas and/or whitespace.
      - ';' or '_' in a single-string spec split it into multiple groups.
    """

    def __init__(self, spec: Union[str, Iterable[str]]):
        if spec is None:
            raise SelectionError("Empty selection spec")
        self._raw = spec
        # list of term-tuples; one tuple per group
        self._groups: list[tuple[Term, ...]] = self._parse_groups(spec)
        self._group_has_explicit = [any(t.chains is not None for t in grp) for grp in self._groups]
        if not self._groups:
            raise SelectionError("Empty selection spec")

    # ----------------------------- public API --------------------------------

    def atom_lists(
        self, structure: Union[Structure, Model], model_index: int = 0
    ) -> list[list[int]]:
        """
        Return one or more atom lists (each sorted, 0-based indices into Model.atoms).
        """
        model = structure.models[model_index] if isinstance(structure, Structure) else structure

        # Cache atom id → index map on the Model
        atom_to_idx = getattr(model, "_atom_index_cache", None)
        if atom_to_idx is None:
            atom_to_idx = {id(a): i for i, a in enumerate(model.atoms)}
            model._atom_index_cache = atom_to_idx

        out_lists: list[list[int]] = []

        # Precompute chain ordering and alias mapping once per call
        chains = list(model.chains())
        chain_index: dict[int, int] = {id(ch): i for i, ch in enumerate(chains)}
        all_aliases: set[str] = set()
        chain_by_alias: dict[str, Chain] = {}
        for ch in chains:
            for k in _all_chain_aliases(ch):
                if k is not None:
                    all_aliases.add(k)
                    chain_by_alias[k] = ch

        def target_chains(term: Term) -> list[Chain]:
            if term.chains is None:
                return chains
            unknown: list[str] = []
            out: list[Chain] = []
            seen_ids: set[int] = set()
            for tok in term.chains:
                ch = chain_by_alias.get(tok)
                if ch is None:
                    unknown.append(tok)
                    continue
                cid = id(ch)
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    out.append(ch)
            if unknown:
                avail = sorted(all_aliases)
                raise SelectionError(
                    f"Unknown chain IDs in spec '{self._raw}': {unknown}. Available: {avail}"
                )
            return out or chains

        def residue_ok(res: Residue, atom_sel: Optional[AtomSelector]) -> bool:
            if atom_sel is None:
                return True
            return _residue_matches_groups(res, atom_sel.flags)

        def atom_ok(atom: Atom, atom_sel: Optional[AtomSelector]) -> bool:
            if atom_sel is None or not atom_sel.has_atom_filters():
                return True

            name = (atom.name or "").strip().upper()
            if atom_sel.names is not None and name not in atom_sel.names:
                return False

            flags = atom_sel.flags
            if "hydrogens" in flags:
                if not _is_hydrogen(atom):
                    return False
            if "heavy" in flags:
                if _is_hydrogen(atom):
                    return False
            if "carbons" in flags and not _is_element_like(atom, "C"):
                return False
            if "nitrogens" in flags and not _is_element_like(atom, "N"):
                return False
            if "oxygens" in flags and not _is_element_like(atom, "O"):
                return False
            return True

        for grp, has_explicit in zip(self._groups, self._group_has_explicit):
            if has_explicit:
                pooled: set[int] = set()
            else:
                per_chain: dict[int, set[int]] = {}

            for term in grp:
                for ch in target_chains(term):
                    for r in ch.residues:
                        if not term.residues.contains(r.resnum):
                            continue
                        if not residue_ok(r, term.atom_selector):
                            continue
                        for a in r.atoms:
                            if not atom_ok(a, term.atom_selector):
                                continue
                            idx = atom_to_idx.get(id(a))
                            if idx is None:
                                continue
                            if has_explicit:
                                pooled.add(idx)
                            else:
                                ci = chain_index[id(ch)]
                                bucket = per_chain.setdefault(ci, set())
                                bucket.add(idx)

            if has_explicit:
                if pooled:
                    out_lists.append(sorted(pooled))
            else:
                # Emit one list per chain with matches, in chain order
                for ci in sorted(per_chain):
                    lst = per_chain[ci]
                    if lst:
                        out_lists.append(sorted(lst))

        return out_lists

    def atom_indices(self, structure: Union[Structure, Model], model_index: int = 0) -> list[int]:
        """Flattened union of all lists returned by atom_lists()."""
        lists = self.atom_lists(structure, model_index=model_index)
        merged: set[int] = set()
        for lst in lists:
            merged.update(lst)
        return sorted(merged)

    def residue_keys(self, structure: Structure, model_index: int = 0) -> list[tuple[str, int]]:
        """
        Union of (chain_key_id, residue_number) across all groups.

        Respects residue-group filters ("protein", "water", "ions") when present.
        """
        model = structure.models[model_index]
        out: set[tuple[str, int]] = set()
        for grp in self._groups:
            alias_to_resnums = self._resolve_residues_for_terms(model, grp)
            for ch in model.chains():
                resnums = _union_resnums_for_chain(ch, alias_to_resnums)
                if not resnums:
                    continue
                for r in ch.residues:
                    if r.resnum in resnums:
                        out.add((ch.key_id, r.resnum))
        return sorted(out)

    # --------------------------- internals ------------------------------------

    def _resolve_residues_for_terms(
        self, model: Model, terms: tuple[Term, ...]
    ) -> dict[str, set[int]]:
        """Original residue resolver, extended with residue-group filters."""
        # Collect alias universe
        all_aliases: set[str] = set()
        chain_by_alias: dict[str, Chain] = {}
        for ch in model.chains():
            for k in _all_chain_aliases(ch):
                if k is not None:
                    all_aliases.add(k)
                    chain_by_alias[k] = ch

        selected: dict[str, set[int]] = {}

        for term in terms:
            # Determine target aliases for this term
            if term.chains is None:
                target_aliases = set(all_aliases)
            else:
                target_aliases = set()
                unknown: list[str] = []
                for tok in term.chains:
                    if tok in all_aliases:
                        target_aliases.add(tok)
                    else:
                        unknown.append(tok)
                if unknown:
                    avail = sorted(all_aliases)
                    raise SelectionError(
                        f"Unknown chain IDs in spec '{self._raw}': {unknown}. Available: {avail}"
                    )

            # Assign residue numbers per targeted alias
            for alias in target_aliases:
                ch = chain_by_alias[alias]
                bucket = selected.setdefault(alias, set())
                flags = (
                    term.atom_selector.flags
                    if getattr(term, "atom_selector", None) is not None
                    else frozenset()
                )
                if term.residues.all_residues:
                    for r in ch.residues:
                        if not _residue_matches_groups(r, flags):
                            continue
                        bucket.add(r.resnum)
                else:
                    for r in ch.residues:
                        if not term.residues.contains(r.resnum):
                            continue
                        if not _residue_matches_groups(r, flags):
                            continue
                        bucket.add(r.resnum)
        return selected

    @staticmethod
    def _parse_groups(spec: Union[str, Iterable[str]]) -> list[tuple[Term, ...]]:
        """
        Return a list of term tuples. Each element of an iterable input is a separate group.
        For a single string input, commas/whitespace split terms within one group.

        Extended behavior:
          - For a single string, ';' and '_' split into separate groups, as if a list of
            strings had been passed.
        """
        if isinstance(spec, str):
            # Treat ';' and '_' as group separators when given a single string spec
            raw_groups = re.split(r"[;_]+", spec)
            groups: list[tuple[Term, ...]] = []
            for s in raw_groups:
                if not isinstance(s, str) or not s.strip():
                    continue
                groups.append(StructureSelector._parse_terms(s))
            return groups

        # Iterable of group strings
        groups: list[tuple[Term, ...]] = []
        for s in spec:
            if not isinstance(s, str) or not s.strip():
                continue
            groups.append(StructureSelector._parse_terms(s))
        return groups

    @staticmethod
    def _parse_terms(group_spec: str) -> tuple[Term, ...]:
        """
        Parse a group specification into Term objects.

        This expands the original grammar to support optional atom parts while
        remaining backward compatible for chain/residue-only specs.
        """
        terms: list[Term] = []
        # split on commas or whitespace; '+' is now reserved for atom/residue lists
        for raw_term in re.split(r"[,\s]+", group_spec.strip()):
            t = raw_term.strip()
            if not t:
                continue
            if t.lower() == "all":
                terms.append(
                    Term(
                        chains=None, residues=ResidueSelector(all_residues=True), atom_selector=None
                    )
                )
                continue

            parts = t.split(".")
            if len(parts) == 1:
                terms.append(StructureSelector._parse_term_single(parts[0]))
            elif len(parts) == 2:
                terms.append(StructureSelector._parse_term_two(parts[0], parts[1]))
            elif len(parts) == 3:
                terms.append(StructureSelector._parse_term_three(parts[0], parts[1], parts[2]))
            else:
                raise SelectionError(f"Too many '.' segments in term '{t}'")

        if not terms:
            raise SelectionError(f"Could not parse spec '{group_spec}'")
        return tuple(terms)

    @staticmethod
    def _parse_term_single(spec: str) -> Term:
        """Handle a single-fragment term, e.g. 'A:B', '2-10', 'protein'."""
        s = spec.strip()
        if not s:
            raise SelectionError("Empty term")

        if _looks_like_residue_spec(s):
            residues = ResidueSelector.parse(s)
            return Term(chains=None, residues=residues, atom_selector=None)

        lower = s.lower()
        if lower in _RESIDUE_GROUP_KEYWORDS or lower in _ATOM_FLAG_ALIASES:
            # e.g. 'protein', 'water', 'heavy', 'carbons'
            atom_sel = _parse_atom_spec(s)
            return Term(
                chains=None, residues=ResidueSelector(all_residues=True), atom_selector=atom_sel
            )

        # Fallback: treat as chain list (original behaviour)
        chains = _parse_chain_list(s)
        residues = ResidueSelector(all_residues=True)
        return Term(chains=chains, residues=residues, atom_selector=None)

    @staticmethod
    def _parse_term_two(first: str, second: str) -> Term:
        """Handle two-part terms, e.g. 'A:B.2-10', 'A:B.CA', 'protein.CA', '2-10.CA'."""
        first = first.strip()
        second = second.strip()
        if not first or not second:
            raise SelectionError(f"Malformed term '{first}.{second}'")

        lower_first = first.lower()
        # Residue-group keywords in first position: 'protein.CA' or 'protein.2-10'
        if lower_first in _RESIDUE_GROUP_KEYWORDS:
            residues: ResidueSelector
            atom_names: Optional[str]

            if _looks_like_residue_spec(second):
                residues = ResidueSelector.parse(second)
                atom_names = None
            else:
                residues = ResidueSelector(all_residues=True)
                atom_names = second

            if atom_names is not None:
                atom_sel = _parse_atom_spec(atom_names)
                flags = set(atom_sel.flags)
            else:
                atom_sel = AtomSelector(names=None, flags=frozenset())
                flags = set()

            flags.add(_ATOM_FLAG_ALIASES[lower_first])
            atom_sel = AtomSelector(
                names=atom_sel.names,
                flags=frozenset(flags),
            )
            return Term(chains=None, residues=residues, atom_selector=atom_sel)

        # Residue-only then atom: '2-40+50-60.CA'
        if _looks_like_residue_spec(first):
            residues = ResidueSelector.parse(first)
            atom_sel = _parse_atom_spec(second)
            return Term(chains=None, residues=residues, atom_selector=atom_sel)

        # Otherwise: chains first
        chains = _parse_chain_list(first)

        # Second fragment decides residue vs atom
        if _looks_like_residue_spec(second) or second.lower() == "all":
            residues = ResidueSelector.parse(second)
            atom_sel = None
        else:
            residues = ResidueSelector(all_residues=True)
            atom_sel = _parse_atom_spec(second)

        return Term(chains=chains, residues=residues, atom_selector=atom_sel)

    @staticmethod
    def _parse_term_three(first: str, second: str, third: str) -> Term:
        """Handle three-part terms, typically 'chains.residues.atoms'."""
        first = first.strip()
        second = second.strip()
        third = third.strip()
        if not first or not second or not third:
            raise SelectionError(f"Malformed term '{first}.{second}.{third}'")

        lower_first = first.lower()
        if lower_first in _RESIDUE_GROUP_KEYWORDS:
            # e.g. 'protein.2-90.CA'
            residues = (
                ResidueSelector.parse(second)
                if _looks_like_residue_spec(second)
                else ResidueSelector(all_residues=True)
            )
            atom_sel = _parse_atom_spec(third)
            flags = set(atom_sel.flags)
            flags.add(_ATOM_FLAG_ALIASES[lower_first])
            atom_sel = AtomSelector(names=atom_sel.names, flags=frozenset(flags))
            return Term(chains=None, residues=residues, atom_selector=atom_sel)

        # Default: 'chains.residues.atoms'
        chains = _parse_chain_list(first)
        residues = (
            ResidueSelector.parse(second)
            if _looks_like_residue_spec(second)
            else ResidueSelector(all_residues=True)
        )
        atom_sel = _parse_atom_spec(third)
        return Term(chains=chains, residues=residues, atom_selector=atom_sel)


# ----------------------------- helpers ---------------------------------------


def _all_chain_aliases(ch: Chain) -> tuple[str, ...]:
    out: list[str] = []
    if getattr(ch, "key_id", None):
        out.append(str(ch.key_id))
    if getattr(ch, "seg_id", None):
        out.append(str(ch.seg_id))
    if hasattr(ch, "chain_id") and getattr(ch, "chain_id") is not None:
        out.append(str(getattr(ch, "chain_id")))
    # De-duplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return tuple(uniq)


def _union_resnums_for_chain(ch: Chain, alias_to_resnums: dict[str, set[int]]) -> set[int]:
    """Union residue sets across all aliases of a chain."""
    resnums: set[int] = set()
    for alias in _all_chain_aliases(ch):
        resnums |= alias_to_resnums.get(alias, set())
    return resnums


def _clone_model_with_coords(
    template_model: Model,
    coords_nm: np.ndarray,
    model_id: int,
) -> Model:
    """
    Clone `template_model` but replace coordinates with `coords_nm` (nm).

    coords_nm: shape (natoms, 3), units nm, same atom order as template_model.atoms.
    """
    if coords_nm.shape != (template_model.natoms(), 3):
        raise ValueError(
            f"Coordinate array shape {coords_nm.shape} does not match "
            f"template natoms={template_model.natoms()}"
        )

    new_model = Model(model_id=model_id)
    natoms = template_model.natoms()
    # index into coords_ang; must follow the same flattened atom order
    idx = 0

    for key, ch in template_model.chain.items():
        new_chain = Chain(
            key_id=ch.key_id,
            seg_id=getattr(ch, "seg_id", None),
            chain_id=getattr(ch, "chain_id", None),
        )

        for r in ch.residues:
            new_res_atoms: list[Atom] = []
            for old_atom in r.atoms:
                if idx >= natoms:
                    raise RuntimeError("Internal consistency error while cloning model coordinates")
                x_nm, y_nm, z_nm = coords_nm[idx]
                x_ang = float(x_nm * 10.0)
                y_ang = float(y_nm * 10.0)
                z_ang = float(z_nm * 10.0)

                new_atom = Atom(
                    serial=old_atom.serial,
                    name=old_atom.name,
                    element=old_atom.element,
                    resname=old_atom.resname,
                    chain=old_atom.chain,
                    resnum=old_atom.resnum,
                    x=x_ang,
                    y=y_ang,
                    z=z_ang,
                    seg=old_atom.seg,
                )
                new_res_atoms.append(new_atom)
                new_model.atoms.append(new_atom)
                idx += 1

            new_res = Residue(
                resname=r.resname,
                chain=r.chain,
                resnum=r.resnum,
                seg=r.seg,
                atoms=new_res_atoms,
            )
            new_chain.residues.append(new_res)
            new_model.residues.append(new_res)

        new_model.chain[key] = new_chain

    if idx != natoms:
        raise RuntimeError(
            f"Cloned coordinates for {idx} atoms, expected {natoms} from template model"
        )
    return new_model


def load_dcd(
    dcd_file: FileLike,
    template: Union[Structure, Model, FileLike],
) -> Structure:
    """
    Load a CHARMM DCD trajectory and represent it as a Structure with:

      - One shared topology (chains/residues/atoms) from the template Model.
      - Per-frame coordinates stored once as Structure._coords_nm (nm).
      - models[i] is a lightweight Model view of frame i using that topology.

    Parameters
    ----------
    dcd_file
        Path (str/Path) to the DCD file, or file-like object MDTraj can read.
    template
        Reference topology for the trajectory:
          - Structure: first model is used as template.
          - Model    : used directly as template.
          - str/Path : treated as a PDB-like file to be read via PDBReader.

        The atom ordering in `template` must match the DCD topology.

    Returns
    -------
    Structure
        A Structure where each frame is a Model view; topology is only stored once.
    """
    struct_ref, tmpl_model = _ensure_template_model(template)

    # Build MDTraj topology from the template OpenMM topology
    top = md.Topology.from_openmm(tmpl_model.topology())

    # MDTraj does DCD I/O, returns xyz in nm
    traj = md.load_dcd(dcd_file, top=top)  # xyz: (n_frames, n_atoms, 3), nm

    if traj.n_atoms != tmpl_model.natoms():
        raise ValueError(
            f"DCD has {traj.n_atoms} atoms but template has {tmpl_model.natoms()} atoms"
        )

    coords_nm = np.asarray(traj.xyz, copy=False)

    s = Structure()
    s._coords_nm = coords_nm

    # Share topology across all models; no per-frame copies of chains/residues/atoms
    base_chain = tmpl_model.chain
    base_residues = tmpl_model.residues
    base_atoms = tmpl_model.atoms

    n_frames = coords_nm.shape[0]
    for i in range(n_frames):
        m = Model(
            model_id=i + 1,
            chain=base_chain,
            residues=base_residues,
            atoms=base_atoms,
        )
        m._parent = s
        m._frame_index = i
        s.models.append(m)

    return s


# ---- summarize topology ---------------------------------------------------------
def summarize_topology(
    topology: Topology,
    max_residues_per_chain: int = 5,
    max_bonds: int = 20,
) -> str:
    """
    Summarize an OpenMM Topology:
      - all chains
      - up to `max_residues_per_chain` residues from the start of each chain
        (with all atoms), plus the last residue in each chain
      - up to `max_bonds` bonds

    Returns a human-readable multi-line string.
    """
    lines = []
    lines.append(
        f"Topology: {topology.getNumChains()} chains, "
        f"{topology.getNumResidues()} residues, "
        f"{topology.getNumAtoms()} atoms, "
        f"{topology.getNumBonds()} bonds"
    )

    for chain_index, chain in enumerate(topology.chains()):
        chain_id: Optional[str] = getattr(chain, "id", None)
        chain_label = chain_id if chain_id is not None else str(chain_index)
        lines.append(f"\nChain {chain_index} (id={chain_label}):")

        residues = list(chain.residues())
        n_res = len(residues)

        # Indices we will show: first N, and always the last one
        show_indices = list(range(min(max_residues_per_chain, n_res)))
        if n_res > 0 and (n_res - 1) not in show_indices:
            show_indices.append(n_res - 1)

        shown_set = set(show_indices)

        for idx in show_indices:
            residue = residues[idx]
            res_id: Optional[str] = getattr(residue, "id", None)
            res_id_str = res_id if res_id is not None else ""
            lines.append(f"  Residue {idx} (name={residue.name}, id={res_id_str}):")

            for atom in residue.atoms():
                elem = atom.element.symbol if atom.element is not None else "?"
                lines.append(f"    Atom {atom.index}: {atom.name} ({elem})")

        # If we skipped any residues between first batch and last
        skipped = n_res - len(shown_set)
        if skipped > 0:
            lines.append(f"  ... ({skipped} residues not shown in this chain)")

    # Bonds
    bonds = list(topology.bonds())
    if bonds:
        lines.append(f"\nBonds (showing up to {max_bonds}):")

        for i, bond in enumerate(bonds[:max_bonds]):
            atom1, atom2 = bond

            def atom_label(a):
                res = a.residue
                chain = res.chain
                chain_id = getattr(chain, "id", None) or ""
                res_id = getattr(res, "id", None) or ""
                return f"{a.index}:{a.name}" f"({res.name}{res_id}, chain={chain_id})"

            lines.append(f"  {i}: {atom_label(atom1)} -- {atom_label(atom2)}")

        if len(bonds) > max_bonds:
            lines.append(f"  ... ({len(bonds) - max_bonds} more bonds not shown)")

    return "\n".join(lines)

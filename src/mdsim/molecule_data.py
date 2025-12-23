from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import mdtraj as md
import numpy as np
from openmm import Vec3
from openmm.app import Topology, element
from openmm.unit import Quantity, Unit, angstrom, dalton, nanometer, radian

FileLike = Union[str, Path, io.BytesIO, io.StringIO]

# --- mass utilities ---------------------------------------------------------

_ELEMENT_MASS_CACHE: dict[str, float] = {}


def _normalize_element_symbol(sym: str) -> str:
    sym = (sym or "").strip().upper()
    # common aliases that sometimes leak into element/resname fields
    return {
        "CAL": "CA",
        "SOD": "NA",
        "POT": "K",
        "CLA": "CL",
    }.get(sym, sym) or sym


def _mass_from_element_symbol(sym: str) -> float:
    """Return atomic mass as float (dalton-like) from an element symbol.

    Falls back to carbon if the symbol cannot be resolved.
    """
    key = _normalize_element_symbol(sym) or "C"
    cached = _ELEMENT_MASS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        el = element.Element.getBySymbol(key)
    except Exception:
        el = element.carbon
    m = el.mass
    try:
        val = float(m.value_in_unit(dalton))
    except Exception:
        val = float(getattr(m, "_value", m))
    _ELEMENT_MASS_CACHE[key] = val
    return val


# --- Data containers ---------------------------------------------------------


@dataclass
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
    mass: Optional[float] = None  # atomic mass (dalton-like); default from element

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

    def nominal_charge(self) -> int:
        """Return the nominal net charge (integer) based on residue names.

        Notes
        -----
        The PDB parser populates residues under `self.chain[...].residues`.  The
        `self.residues` list is not guaranteed to be populated for all loaders,
        so we iterate over `iter_residues()` for correctness.
        """
        charge_map = {
            "LYS": 1,
            "ARG": 1,
            "HSP": 1,
            "ASP": -1,
            "GLU": -1,
            "SOD": 1,
            "CLA": -1,
            "POT": 1,
        }
        q = 0
        for r in self.iter_residues():
            rn = (getattr(r, "resname", "") or "").strip().upper()
            q += charge_map.get(rn, 0)
        return int(q)

    def positions(self):
        """
        Return positions as an OpenMM Quantity[list[Vec3]].
        - Static models: internal coordinates are in Å, converted to nm.
        - Trajectory-backed models: parent._coords_nm already in nm.
        """
        if self._has_parent_coords():
            coords_nm = self._parent._coords_nm[self._frame_index]  # (natoms, 3)
            vecs = [Vec3(float(x), float(y), float(z)) for x, y, z in coords_nm]
            return Quantity(vecs, nanometer)

        # Static: use Atom coordinates in Å, convert to nm
        vecs = [Vec3(a.x, a.y, a.z) for a in self.atoms]
        return Quantity(vecs, angstrom).in_units_of(nanometer)

    def topology(
        self,
        *,
        bonds: Optional[bool] = None,
        auto: Optional[bool] = True,
        cutoff: Union[float, Quantity] = 0.2 * nanometer,
        disulfide_cutoff: Union[float, Quantity] = 0.25 * nanometer,
    ) -> Topology:
        """Build an OpenMM Topology for this model.

        Bonding rules:
          - Standard amino acids: rule-based heavy-atom connectivity.
          - Histidine variants: HIS/HSD/HSE/HSP are supported.
          - Peptide bonds: C(i) -- N(i+1) within each chain.
          - Disulfides: SG--SG added by distance.
          - Ions: no bonds.
          - Water (TIP3/SPC/TIP4/HOH/WAT): bonds only within each residue.
          - Unknown residues: if auto=True, bonds are inferred by distance cutoff
            within the residue and to immediate neighbors in the same chain.
        """
        top = Topology()

        do_bonds = True if bonds is None else bool(bonds)
        do_auto = True if auto is None else bool(auto)

        def _to_nm(val: Union[float, Quantity]) -> float:
            if isinstance(val, Quantity):
                return float(val.value_in_unit(nanometer))
            return float(val)

        cutoff_nm = _to_nm(cutoff)
        disulf_nm = _to_nm(disulfide_cutoff)
        h_cutoff_nm = 0.15  # nm, generous X-H covalent upper bound

        natoms = self.natoms()
        atom_id_to_idx = {id(a): i for i, a in enumerate(self.atoms)}

        if natoms == 0:
            coords_nm = np.zeros((0, 3), dtype=float)
        elif self._has_parent_coords():
            coords_nm = np.asarray(self._parent._coords_nm[self._frame_index], dtype=float)
        else:
            coords_nm = np.asarray([[a.x, a.y, a.z] for a in self.atoms], dtype=float) / 10.0

        atom_id_to_top: dict[int, Any] = {}
        res_id_to_atoms: dict[int, dict[str, Any]] = {}
        res_id_to_idx_by_name: dict[int, dict[str, int]] = {}

        for c in self.chains():
            chain = top.addChain(c.key_id)
            for r in c.residues:
                res_id = str(int(r.resnum))
                res = top.addResidue(r.resname, chain, id=res_id)
                name_map: dict[str, Any] = {}
                idx_map: dict[str, int] = {}

                for a in r.atoms:
                    sym = (getattr(a, "element", "") or "").upper()
                    try:
                        el = element.Element.getBySymbol(sym)
                    except Exception:
                        el = element.carbon

                    ta = top.addAtom(a.name, element=el, residue=res)
                    atom_id_to_top[id(a)] = ta

                    an = (a.name or "").strip().upper()
                    if an:
                        name_map[an] = ta
                        ai = atom_id_to_idx.get(id(a))
                        if ai is not None:
                            idx_map[an] = ai

                res_id_to_atoms[id(r)] = name_map
                res_id_to_idx_by_name[id(r)] = idx_map

        if not do_bonds:
            return top

        bond_set: set[tuple[int, int]] = set()

        def add_bond(a1: Any, a2: Any) -> None:
            i = int(a1.index)
            j = int(a2.index)
            if i == j:
                return
            if i > j:
                i, j = j, i
            key = (i, j)
            if key in bond_set:
                return
            top.addBond(a1, a2)
            bond_set.add(key)

        water_res = {"TIP3", "SPC", "TIP4", "HOH", "WAT"}
        ion_res = {"SOD", "POT", "CLA", "MG", "NA", "CL", "K"}

        aa_res = {
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
            "HSP",
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

        aa_side_bonds: dict[str, tuple[tuple[str, str], ...]] = {
            "ALA": (("CA", "CB"),),
            "ARG": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "CD"),
                ("CD", "NE"),
                ("NE", "CZ"),
                ("CZ", "NH1"),
                ("CZ", "NH2"),
            ),
            "ASN": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
            "ASP": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
            "CYS": (("CA", "CB"), ("CB", "SG")),
            "GLN": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
            "GLU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
            "GLY": (),
            "HIS": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "ND1"),
                ("ND1", "CE1"),
                ("CE1", "NE2"),
                ("NE2", "CD2"),
                ("CD2", "CG"),
            ),
            "HSD": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "ND1"),
                ("ND1", "CE1"),
                ("CE1", "NE2"),
                ("NE2", "CD2"),
                ("CD2", "CG"),
            ),
            "HSE": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "ND1"),
                ("ND1", "CE1"),
                ("CE1", "NE2"),
                ("NE2", "CD2"),
                ("CD2", "CG"),
            ),
            "HSP": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "ND1"),
                ("ND1", "CE1"),
                ("CE1", "NE2"),
                ("NE2", "CD2"),
                ("CD2", "CG"),
            ),
            "ILE": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
            "LEU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
            "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
            "MET": (("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
            "PHE": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "CD1"),
                ("CG", "CD2"),
                ("CD1", "CE1"),
                ("CD2", "CE2"),
                ("CE1", "CZ"),
                ("CE2", "CZ"),
            ),
            "PRO": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")),
            "SER": (("CA", "CB"), ("CB", "OG")),
            "THR": (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
            "TRP": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "CD1"),
                ("CG", "CD2"),
                ("CD1", "NE1"),
                ("NE1", "CE2"),
                ("CE2", "CD2"),
                ("CE2", "CZ2"),
                ("CZ2", "CH2"),
                ("CH2", "CZ3"),
                ("CZ3", "CE3"),
                ("CE3", "CD2"),
            ),
            "TYR": (
                ("CA", "CB"),
                ("CB", "CG"),
                ("CG", "CD1"),
                ("CG", "CD2"),
                ("CD1", "CE1"),
                ("CD2", "CE2"),
                ("CE1", "CZ"),
                ("CE2", "CZ"),
                ("CZ", "OH"),
            ),
            "VAL": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
        }

        def add_pairs(name_map: dict[str, Any], pairs: tuple[tuple[str, str], ...]) -> None:
            for n1, n2 in pairs:
                a1 = name_map.get(n1)
                a2 = name_map.get(n2)
                if a1 is None or a2 is None:
                    continue
                add_bond(a1, a2)

        def residue_indices(idx_map: dict[str, int]) -> list[int]:
            return [int(i) for i in idx_map.values()]

        def add_hydrogen_bonds(idx_list: list[int]) -> None:
            if not idx_list:
                return
            heavy = [i for i in idx_list if not _is_hydrogen(self.atoms[i])]
            if not heavy:
                return
            heavy_arr = np.asarray(heavy, dtype=np.int64)
            heavy_xyz = coords_nm[heavy_arr, :]

            for hi in idx_list:
                if not _is_hydrogen(self.atoms[hi]):
                    continue
                d = heavy_xyz - coords_nm[hi]
                dist2 = np.einsum("ij,ij->i", d, d)
                k = int(dist2.argmin())
                if float(dist2[k]) > (h_cutoff_nm * h_cutoff_nm):
                    continue
                parent = int(heavy_arr[k])
                a1 = atom_id_to_top.get(id(self.atoms[hi]))
                a2 = atom_id_to_top.get(id(self.atoms[parent]))
                if a1 is None or a2 is None:
                    continue
                add_bond(a1, a2)

        def add_heavy_bonds_by_cutoff(idx_list: list[int], cutoff_nm_: float) -> None:
            heavy = [i for i in idx_list if not _is_hydrogen(self.atoms[i])]
            if len(heavy) < 2:
                return
            idx = np.asarray(heavy, dtype=np.int64)
            xyz = coords_nm[idx, :]
            diff = xyz[:, None, :] - xyz[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", diff, diff)
            cut2 = float(cutoff_nm_ * cutoff_nm_)
            mask = np.triu(dist2 <= cut2, 1)
            ii, jj = np.where(mask)
            for a, b in zip(ii.tolist(), jj.tolist()):
                ai = int(idx[a])
                bj = int(idx[b])
                ta = atom_id_to_top.get(id(self.atoms[ai]))
                tb = atom_id_to_top.get(id(self.atoms[bj]))
                if ta is None or tb is None:
                    continue
                add_bond(ta, tb)

        def add_cross_heavy_bonds(
            idx_a: list[int],
            idx_b: list[int],
            cutoff_nm_: float,
        ) -> None:
            a = [i for i in idx_a if not _is_hydrogen(self.atoms[i])]
            b = [i for i in idx_b if not _is_hydrogen(self.atoms[i])]
            if not a or not b:
                return
            ia = np.asarray(a, dtype=np.int64)
            ib = np.asarray(b, dtype=np.int64)
            xa = coords_nm[ia, :]
            xb = coords_nm[ib, :]
            diff = xa[:, None, :] - xb[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", diff, diff)
            cut2 = float(cutoff_nm_ * cutoff_nm_)
            ii, jj = np.where(dist2 <= cut2)
            for i0, j0 in zip(ii.tolist(), jj.tolist()):
                ai = int(ia[i0])
                bj = int(ib[j0])
                ta = atom_id_to_top.get(id(self.atoms[ai]))
                tb = atom_id_to_top.get(id(self.atoms[bj]))
                if ta is None or tb is None:
                    continue
                add_bond(ta, tb)

        # Intra-residue bonds
        for c in self.chains():
            for r in c.residues:
                rn = (r.resname or "").strip().upper()
                name_map = res_id_to_atoms.get(id(r), {})
                idx_map = res_id_to_idx_by_name.get(id(r), {})

                # ILE sometimes appears with CD (instead of CD1) in PDBs.
                # Treat CD and CD1 as aliases so the ILE template bonds apply.
                if rn == "ILE":
                    if "CD" in name_map and "CD1" not in name_map:
                        name_map["CD1"] = name_map["CD"]
                    elif "CD1" in name_map and "CD" not in name_map:
                        name_map["CD"] = name_map["CD1"]

                    if "CD" in idx_map and "CD1" not in idx_map:
                        idx_map["CD1"] = idx_map["CD"]
                    elif "CD1" in idx_map and "CD" not in idx_map:
                        idx_map["CD"] = idx_map["CD1"]

                idxs = residue_indices(idx_map)

                if rn in ion_res:
                    continue

                if rn in water_res:
                    o = name_map.get("O") or name_map.get("OH2") or name_map.get("OW")
                    if o is None:
                        continue
                    for hname in ("H1", "H2", "HW1", "HW2"):
                        h = name_map.get(hname)
                        if h is not None:
                            add_bond(o, h)
                    continue

                if rn in aa_res:
                    # C-terminus alternate oxygen naming: OT1/OT2 ~ O/OXT.
                    if "OT1" in name_map and "O" not in name_map:
                        name_map["O"] = name_map["OT1"]
                    if "OT2" in name_map and "OXT" not in name_map:
                        name_map["OXT"] = name_map["OT2"]
                    add_pairs(name_map, (("N", "CA"), ("CA", "C"), ("C", "O")))
                    if "OXT" in name_map:
                        add_pairs(name_map, (("C", "OXT"),))

                    add_pairs(name_map, (("N", "CA"), ("CA", "C"), ("C", "O")))
                    if "OXT" in name_map:
                        add_pairs(name_map, (("C", "OXT"),))
                    add_pairs(name_map, aa_side_bonds.get(rn, ()))
                    add_hydrogen_bonds(idxs)
                    continue

                if do_auto:
                    add_heavy_bonds_by_cutoff(idxs, cutoff_nm)
                    add_hydrogen_bonds(idxs)

        # Inter-residue bonds (only previous/next in chain)
        for c in self.chains():
            res_list = list(c.residues)
            for i in range(len(res_list) - 1):
                r0 = res_list[i]
                r1 = res_list[i + 1]

                rn0 = (r0.resname or "").strip().upper()
                rn1 = (r1.resname or "").strip().upper()
                if rn0 in ion_res or rn1 in ion_res:
                    continue
                if rn0 in water_res or rn1 in water_res:
                    continue

                m0 = res_id_to_atoms.get(id(r0), {})
                m1 = res_id_to_atoms.get(id(r1), {})

                if rn0 in aa_res and rn1 in aa_res:
                    c_atom = m0.get("C")
                    n_atom = m1.get("N")
                    if c_atom is not None and n_atom is not None:
                        add_bond(c_atom, n_atom)
                    continue

                if not do_auto:
                    continue

                known0 = rn0 in aa_res
                known1 = rn1 in aa_res
                if known0 and known1:
                    continue

                idx0 = residue_indices(res_id_to_idx_by_name.get(id(r0), {}))
                idx1 = residue_indices(res_id_to_idx_by_name.get(id(r1), {}))
                add_cross_heavy_bonds(idx0, idx1, cutoff_nm)

        # Disulfide bonds by distance (SG--SG)
        sg_list: list[tuple[int, Any]] = []
        for r in self.iter_residues():
            rn = (r.resname or "").strip().upper()
            if rn in ion_res or rn in water_res:
                continue
            name_map = res_id_to_atoms.get(id(r), {})
            idx_map = res_id_to_idx_by_name.get(id(r), {})
            sg = name_map.get("SG")
            sgi = idx_map.get("SG")
            if sg is None or sgi is None:
                continue
            sg_list.append((int(sgi), sg))

        if len(sg_list) >= 2:
            sg_idx = np.asarray([i for i, _ in sg_list], dtype=np.int64)
            sg_xyz = coords_nm[sg_idx, :]
            n = len(sg_list)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    d = sg_xyz[i] - sg_xyz[j]
                    if float(np.dot(d, d)) <= float(disulf_nm * disulf_nm):
                        add_bond(sg_list[i][1], sg_list[j][1])

        return top

    @staticmethod
    def _positions_to_nm_array(
        positions: Any,
        natoms: int,
        *,
        assume_unit: Unit = nanometer,
    ) -> np.ndarray:
        """Normalize OpenMM-like positions into (natoms, 3) ndarray in nm."""
        if positions is None:
            raise ValueError("positions must not be None")

        if isinstance(positions, Quantity):
            q = positions.in_units_of(nanometer)
            val = q.value_in_unit(nanometer)
            if isinstance(val, (list, tuple)) and natoms and isinstance(val[0], Vec3):
                arr = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in val], dtype=float)
            else:
                arr = np.asarray(val, dtype=float)
        elif isinstance(positions, (list, tuple)) and natoms and isinstance(positions[0], Vec3):
            arr = np.array(
                [[float(v[0]), float(v[1]), float(v[2])] for v in positions], dtype=float
            )
            if assume_unit is not nanometer:
                arr = Quantity(arr, assume_unit).in_units_of(nanometer).value_in_unit(nanometer)
        else:
            arr = np.asarray(positions, dtype=float)
            if assume_unit is not nanometer:
                arr = Quantity(arr, assume_unit).in_units_of(nanometer).value_in_unit(nanometer)

        if arr.shape != (natoms, 3):
            raise ValueError(f"positions has shape {arr.shape}, expected ({natoms}, 3)")
        return arr

    def set_positions(self, positions: Any, *, assume_unit: Unit = nanometer) -> None:
        """
        Set atom x/y/z from OpenMM positions.
        - Updates parent trajectory store if present (nm).
        - Always updates Atom.x/y/z in Å.
        """
        natoms = self.natoms()
        coords_nm = self._positions_to_nm_array(positions, natoms, assume_unit=assume_unit)

        # If trajectory-backed, keep the trajectory store in sync
        if self._has_parent_coords():
            self._parent._coords_nm[self._frame_index, :, :] = coords_nm

        coords_ang = coords_nm * 10.0  # nm -> Å
        for i, a in enumerate(self.atoms):
            x, y, z = coords_ang[i]
            a.x = float(x)
            a.y = float(y)
            a.z = float(z)

        self._atom_index_cache = None

    def set_atom_masses(
        self,
        masses: Any,
        *,
        atom_indices: Optional[Sequence[int]] = None,
        assume_unit: Unit = dalton,
    ) -> None:
        """Set Atom.mass values for this model.

        Parameters
        ----------
        masses
            Either:
              - Sequence[float] (dalton-like), length == natoms or len(atom_indices)
              - Sequence[Quantity] (mass), per-atom masses
              - Quantity array-like, shape (natoms,) or (len(atom_indices),)
        atom_indices
            Optional subset of 0-based atom indices to update. If None, masses must
            be provided for all atoms in model order.
        assume_unit
            Only used when `masses` is a plain numeric array; interpreted as this unit.
            (For OpenMM `System.getParticleMass()` values, you can pass them directly.)
        """
        if not self.atoms:
            return

        if atom_indices is None:
            idx_list = list(range(self.natoms()))
        else:
            idx_list = [int(i) for i in atom_indices]

        # normalize masses -> list[float]
        mvals: list[float] = []
        if isinstance(masses, Quantity):
            q = masses.in_units_of(dalton)
            arr = np.asarray(q.value_in_unit(dalton), dtype=float).reshape(-1)
            mvals = [float(x) for x in arr]
        else:
            # list/tuple/ndarray of either floats or Quantity
            if isinstance(masses, np.ndarray):
                arr = np.asarray(masses, dtype=float).reshape(-1)
                # interpret numeric masses as assume_unit
                if assume_unit != dalton:
                    arr = Quantity(arr, assume_unit).in_units_of(dalton).value_in_unit(dalton)
                mvals = [float(x) for x in arr]
            else:
                # generic iterable
                try:
                    seq = list(masses)  # type: ignore[arg-type]
                except TypeError as e:
                    msg = "masses must be a Quantity, array, or iterable of masses"
                    raise TypeError(msg) from e

                for mv in seq:
                    if isinstance(mv, Quantity):
                        mvals.append(float(mv.in_units_of(dalton).value_in_unit(dalton)))
                    else:
                        val = float(mv)
                        if assume_unit != dalton:
                            val = val * assume_unit
                            val = float(val.in_units_of(dalton).value_in_unit(dalton))
                        mvals.append(val)

        if len(mvals) != len(idx_list):
            raise ValueError(
                f"masses length {len(mvals)} does not match number of target atoms {len(idx_list)}"
            )

        natoms = self.natoms()
        for idx, mass_val in zip(idx_list, mvals):
            if idx < 0 or idx >= natoms:
                raise IndexError(f"Atom index {idx} is out of range for model with {natoms} atoms")
            self.atoms[idx].mass = float(mass_val)

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
                                mass=a.mass,
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

    @staticmethod
    def _atom_mass(atom: Atom) -> float:
        """Atomic mass as float (dalton-like).

        Prefers Atom.mass (if set), otherwise falls back to element-derived mass.
        """
        m = getattr(atom, "mass", None)
        if m is not None:
            return float(m)
        return _mass_from_element_symbol(getattr(atom, "element", ""))

    def _center_of_group(
        self,
        indices: list[int],
        *,
        center: str = "cog",
    ) -> tuple[float, float, float]:
        """
        Internal: center (nm) for a flat list of atom indices in this model.

        center:
          - cog: center of geometry
          - com: center of mass (element-based masses; best-effort)
        """
        if not indices:
            raise ValueError("center requires at least one atom index")

        mode = (center or "cog").lower()
        if mode not in {"cog", "com"}:
            raise ValueError(f"Center option {center} is not valid. Use 'cog' or 'com'.")

        n_atoms = self.natoms()
        for idx in indices:
            if idx < 0 or idx >= n_atoms:
                raise IndexError(f"Atom index {idx} is out of range for model with {n_atoms} atoms")

        idx_arr = np.asarray(indices, dtype=np.int64)

        # Build coordinates in nm
        if self._has_parent_coords():
            coords_nm = self._parent._coords_nm[self._frame_index]  # type: ignore[union-attr]
            coords_nm = coords_nm[idx_arr, :]
        else:
            # static model stores Atom coords in Å -> nm
            coords_nm = np.array(
                [
                    [
                        float(self.atoms[i].x) / 10.0,
                        float(self.atoms[i].y) / 10.0,
                        float(self.atoms[i].z) / 10.0,
                    ]
                    for i in idx_arr
                ],
                dtype=float,
            )

        if mode == "cog":
            c = coords_nm.mean(axis=0)
            return float(c[0]), float(c[1]), float(c[2])

        masses = np.array([self._atom_mass(self.atoms[i]) for i in idx_arr], dtype=float)
        m_tot = float(masses.sum())
        if m_tot == 0.0:
            raise ValueError("Total mass of group is zero; cannot compute COM.")
        com = (masses[:, None] * coords_nm).sum(axis=0) / m_tot
        return float(com[0]), float(com[1]), float(com[2])

    def distance(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        *,
        center: str = "cog",
    ) -> Quantity:
        """
        Center distance between two atom groups, as an OpenMM Quantity in nm.

        center:
          - cog: center of geometry
          - com: center of mass
        """
        A = self.center(group_a, center=center).value_in_unit(nanometer)  # Quantity[nm]
        B = self.center(group_b, center=center).value_in_unit(nanometer)  # Quantity[nm]
        dist_nm = float(np.linalg.norm(A - B))
        return Quantity(dist_nm, nanometer)

    def distance_vector(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        *,
        center: str = "cog",
    ) -> Quantity:
        """
        Center displacement vector (from group_b -> group_a) as an OpenMM Quantity[Vec3] in nm.

        center:
          - cog: center of geometry
          - com: center of mass
        """
        A = self.center(group_a, center=center)  # Quantity[nm]
        B = self.center(group_b, center=center)  # Quantity[nm]
        d = (A - B).value_in_unit(nanometer)

        return Vec3(float(d[0]), float(d[1]), float(d[2])) * nanometer

    def center(
        self,
        group: Optional[Union[list[int], list[list[int]]]] = None,
        *,
        center="cog",
    ) -> Quantity:
        """
        Center for a group of atoms, returned as an OpenMM Quantity in nm.

        center:
           - cog: center of geometry
           - com: center of mass
        """

        if group is None:
            flat = list(range(self.natoms()))
        else:
            flat = self._flatten_indices(group)
            if not flat:
                raise ValueError("group must contain at least one atom index")

        cx, cy, cz = self._center_of_group(flat, center=center)
        return Quantity(np.array((cx, cy, cz), dtype=float), nanometer)

    @staticmethod
    @staticmethod
    def _as_nm_vector(
        v: Union[Quantity, Sequence[Union[Quantity, float, int, np.floating]], np.ndarray],
    ) -> np.ndarray:
        """Coerce translation vector into a (3,) float ndarray in nm.

        Accepts:
          - Quantity of length 3 (e.g., Vec3*unit, Quantity[list], Quantity[np.ndarray])
          - Sequence of 3 Quantities (scalars)
          - Sequence/ndarray of 3 numbers (assumed nm)
        """
        # Quantity vector (Vec3*unit or Quantity wrapping a 3-vector)
        if isinstance(v, Quantity):
            arr = np.asarray(v.value_in_unit(nanometer), dtype=float).reshape(3)
            return arr

        # Sequence (len 3): possibly scalars with units
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) != 3:
                raise ValueError(f"translate vector must have length 3, got {len(v)}")
            if any(isinstance(x, Quantity) for x in v):
                return np.array(
                    [
                        float(x.value_in_unit(nanometer)) if isinstance(x, Quantity) else float(x)
                        for x in v
                    ],
                    dtype=float,
                )
            return np.array([float(x) for x in v], dtype=float)

        raise TypeError(
            "translate expects a 3-vector: Quantity, list of 3 Quantities, or 3 numbers (nm)"
        )

    def translate(
        self,
        v: Union[Quantity, Sequence[Union[Quantity, float, int, np.floating]], np.ndarray],
    ) -> None:
        """Translate coordinates by vector `v` (nm).

        - Trajectory-backed models: shifts parent._coords_nm for this frame (in-place).
        - Static models: shifts Atom coordinates stored in Å (in-place; nm->Å conversion).
        """
        dv_nm = self._as_nm_vector(v)  # (3,) float in nm

        if self._has_parent_coords():
            self._parent._coords_nm[self._frame_index] += dv_nm
            return

        # Static: internal atom coordinates are Å
        dv_A = dv_nm * 10.0
        for a in self.atoms:
            a.x += float(dv_A[0])
            a.y += float(dv_A[1])
            a.z += float(dv_A[2])

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
        *,
        center="cog",
    ) -> Quantity:
        """
        Angle (radians) between two plane normals, matching set_umbrella_angle_norm.

        Plane A: defined by centroids of (group_a, group_a1, group_a2)
        Plane B: defined by centroids of (group_b, group_b1, group_b2)

        Angle is evaluated via atan2(sin, cos) with:
          sin = |nA × nB| / sqrt((nA·nB)^2 + |nA × nB|^2)
          cos = (nA·nB)  / sqrt((nA·nB)^2 + |nA × nB|^2)

        which matches the CustomCentroidBondForce expression used in
        MDSim.set_umbrella_angle_norm.
        """
        # centroids (nm)
        A0 = self.center(group_a, center=center).value_in_unit(nanometer)
        A1 = self.center(group_a1, center=center).value_in_unit(nanometer)
        A2 = self.center(group_a2, center=center).value_in_unit(nanometer)
        B0 = self.center(group_b, center=center).value_in_unit(nanometer)
        B1 = self.center(group_b1, center=center).value_in_unit(nanometer)
        B2 = self.center(group_b2, center=center).value_in_unit(nanometer)

        # in-plane vectors
        vA1 = A1 - A0
        vA2 = A2 - A0
        vB1 = B1 - B0
        vB2 = B2 - B0

        # plane normals (unnormalized): nA = vA1 × vA2, nB = vB1 × vB2
        nA = np.cross(vA1, vA2)
        nB = np.cross(vB1, vB2)

        # nA × nB and nA · nB
        crossAB = np.cross(nA, nB)
        magCross = float(np.linalg.norm(crossAB))
        dotAB = float(np.dot(nA, nB))

        # denom ~ |nA||nB| via dot/cross identity, +eps to avoid 0
        denom = float(np.sqrt(dotAB * dotAB + magCross * magCross) + 1.0e-8)

        # sin/cos of angle between plane normals
        sinang = magCross / denom
        cosang = dotAB / denom

        # Angle in [0, π] (since sinang >= 0)
        theta = float(np.arctan2(sinang, cosang))

        return Quantity(theta, radian)

    def dihedral(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
        group_d: Union[list[int], list[list[int]]],
        *,
        center="cog",
    ) -> Quantity:
        """
        Dihedral angle (radians, −π..π) between four centroids,
        matching set_umbrella_dihedral geometry.
        """
        p1 = self.center(group_a, center=center).value_in_unit(nanometer)
        p2 = self.center(group_b, center=center).value_in_unit(nanometer)
        p3 = self.center(group_c, center=center).value_in_unit(nanometer)
        p4 = self.center(group_d, center=center).value_in_unit(nanometer)

        angle = self._dihedral_angle(p1, p2, p3, p4)
        return Quantity(angle, radian)

    def angle(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
        *,
        center="cog",
    ) -> Quantity:
        """
        Angle (radians) corresponding to the angle() terms used in
        set_umbrella_angle.

        Returns
        -------
          theta = angle(group_a,  group_b,  group_c)
        """
        A = self.center(group_a, center=center).value_in_unit(nanometer)
        B = self.center(group_b, center=center).value_in_unit(nanometer)
        C = self.center(group_c, center=center).value_in_unit(nanometer)

        # angle(g1,g2,g3) = angle between (g1 - g2) and (g3 - g2)
        theta = self._angle_between(A - B, C - B)

        return Quantity(theta, radian)

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

    # --- I/O helpers ---------------------------------------------------------

    def write_pdb(
        self,
        file: FileLike,
        *,
        model_records: Optional[bool] = None,
        allhis: bool = False,
    ) -> None:
        """
        Write this Model to a PDB file.

        Parameters
        ----------
        file : FileLike
            Output path or file-like (str, Path, StringIO, BytesIO).
        model_records : Optional[bool], optional
            If None (default), do not emit MODEL/ENDMDL for this single model.
            If True, wrap this model in MODEL/ENDMDL.
            If False, never emit MODEL/ENDMDL.
        allhis : bool, optional
            If True, write all histidine variants (HIS/HSD/HSE/HSP) with residue
            name "HIS" in the output PDB. Does not modify the Model.
        """
        writer = PDBWriter()
        writer.write(self, file, model_records=model_records, allhis=allhis)


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
        """Return the first model"""
        if not self.models:
            raise ValueError("Structure has no models")
        return self.models[0]

    def get_model(self, model_index: int = 0) -> Model:
        """Return a single model, default: first model"""
        if not self.models:
            raise ValueError("Structure has no models")
        return self.models[model_index]

    def nchains(self) -> int:
        return self.models[0].nchains()

    def nresidues(self) -> int:
        return self.models[0].nresidues()

    def natoms(self) -> int:
        return self.models[0].natoms()

    def nominal_charge(self) -> int:
        """Return the nominal net charge (integer) based on residue names.

        Computed from the first model (self.models[0]).
        """
        if not self.models:
            return 0
        return self.models[0].nominal_charge()

    def nframes(self) -> int:
        if self._coords_nm is not None:
            return len(self._coords_nm)
        else:
            return 0

    def positions(self, model_index: int = 0):
        """Positions for the selected model as Quantity[list[Vec3]] in nm."""
        return self.models[model_index].positions()

    def set_positions(
        self,
        positions: Any,
        *,
        model_index: int = 0,
        assume_unit: Unit = nanometer,
    ) -> None:
        """Set positions for one model/frame."""
        if not self.models:
            raise ValueError("Structure has no models")
        if model_index < 0 or model_index >= len(self.models):
            raise IndexError(f"model_index {model_index} out of range (0..{len(self.models)-1})")
        self.models[model_index].set_positions(positions, assume_unit=assume_unit)

    def set_positions_all(
        self,
        positions_by_model: Sequence[Any],
        *,
        assume_unit: Unit = nanometer,
    ) -> None:
        """Set positions for all models/frames."""
        if len(positions_by_model) != len(self.models):
            has = len(positions_by_model)
            expected = len(self.models)
            raise ValueError(f"positions_by_model has length {has}, expected {expected}")
        for i, pos in enumerate(positions_by_model):
            self.models[i].set_positions(pos, assume_unit=assume_unit)

    def set_atom_masses(
        self,
        masses: Any,
        *,
        atom_indices: Optional[Sequence[int]] = None,
        model_index: int = 0,
        assume_unit: Unit = dalton,
    ) -> None:
        """Set Atom.mass values on the selected model's atoms.

        For trajectory-backed Structures (e.g. from load_dcd), all models share
        the same Atom objects, so setting masses once is sufficient.
        """
        if not self.models:
            raise ValueError("Structure has no models")
        if model_index < 0 or model_index >= len(self.models):
            raise IndexError(f"model_index {model_index} out of range (0..{len(self.models)-1})")
        self.models[model_index].set_atom_masses(
            masses, atom_indices=atom_indices, assume_unit=assume_unit
        )

    def topology(
        self,
        *,
        bonds: Optional[bool] = None,
        auto: Optional[bool] = True,
        cutoff: Optional[float, Quantity] = 0.2 * nanometer,  # nm
    ) -> Topology:
        return self.models[0].topology(bonds=bonds, auto=auto, cutoff=cutoff)

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

    def center(
        self,
        group: Optional[Union[list[int], list[list[int]]]] = None,
        *,
        center: str = "cog",
    ) -> Quantity:
        """
        Centers (nm) for all models/frames.

        Parameters
        ----------
        group
            Atom indices (0-based) as a flat list or list-of-lists. If None, uses all atoms.
        center
            "cog" (center of geometry) or "com" (center of mass).

        Returns
        -------
        Quantity
            OpenMM Quantity with unit nm; value has shape (n_models, 3).
            For an empty Structure, value shape is (0, 3).
        """
        if not self.models:
            return Quantity(np.zeros((0, 3), dtype=float), nanometer)

        mode = (center or "cog").lower()
        if mode not in {"cog", "com"}:
            raise ValueError(f"Center option {center} is not valid. Use 'cog' or 'com'.")

        if group is None:
            flat = list(range(self.natoms()))
        else:
            flat = Model._flatten_indices(group)  # type: ignore[attr-defined]
            if not flat:
                raise ValueError("group must contain at least one atom index")

        n_atoms = self.natoms()
        for idx in flat:
            if idx < 0 or idx >= n_atoms:
                msg = f"Atom index {idx} is out of range for structure with {n_atoms} atoms"
                raise IndexError(msg)

        idx_arr = np.asarray(flat, dtype=np.int64)

        # Fast path for trajectory-backed structures: vectorized over frames
        if self._coords_nm is not None:
            coords_nm = self._coords_nm  # (n_frames, n_atoms, 3)
            if mode == "cog":
                return Quantity(coords_nm[:, idx_arr, :].mean(axis=1), nanometer)

            atoms0 = self.models[0].atoms
            masses = np.asarray([Model._atom_mass(atoms0[i]) for i in idx_arr], dtype=float)
            m_tot = float(masses.sum())
            if m_tot == 0.0:
                raise ValueError("Total mass of group is zero; cannot compute COM.")
            return Quantity(
                (coords_nm[:, idx_arr, :] * masses[None, :, None]).sum(axis=1) / m_tot, nanometer
            )

        # Static / multi-model fallback
        centers_nm = [m.center(flat, center=center).value_in_unit(nanometer) for m in self.models]
        return Quantity(np.asarray(centers_nm, dtype=float), nanometer)

    def translate(
        self,
        v: Union[Quantity, Sequence[Union[Quantity, float, int, np.floating]], np.ndarray],
    ) -> None:
        """Translate all models/frames by vector `v` (nm), in-place."""
        dv_nm = Model._as_nm_vector(v)

        if self._coords_nm is not None:
            # shape (n_models, n_atoms, 3)
            self._coords_nm = self._coords_nm + dv_nm.reshape(1, 1, 3)
            return

        for m in self.models:
            m.translate(dv_nm)

    def distance(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        *,
        center: str = "cog",
    ) -> list[Quantity]:
        """
        Center distance between two atom groups for all models.

        If trajectory-backed (Structure._coords_nm is set), uses a vectorized numpy
        implementation over all frames. Otherwise falls back to per-model computation.

        center:
          - cog: center of geometry
          - com: center of mass
        """
        if not self.models:
            return []

        mode = (center or "cog").lower()
        if mode not in {"cog", "com"}:
            raise ValueError(f"Center option {center} is not valid. Use 'cog' or 'com'.")

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

            if mode == "cog":
                cen_a_nm = coords_nm[:, idx_a, :].mean(axis=1)
                cen_b_nm = coords_nm[:, idx_b, :].mean(axis=1)
            else:
                atoms0 = self.models[0].atoms
                ma = np.asarray([Model._atom_mass(atoms0[i]) for i in idx_a], dtype=float)
                mb = np.asarray([Model._atom_mass(atoms0[i]) for i in idx_b], dtype=float)
                ma_tot = float(ma.sum())
                mb_tot = float(mb.sum())
                if ma_tot == 0.0 or mb_tot == 0.0:
                    raise ValueError("Total mass of group is zero; cannot compute COM.")
                cen_a_nm = (coords_nm[:, idx_a, :] * ma[None, :, None]).sum(axis=1) / ma_tot
                cen_b_nm = (coords_nm[:, idx_b, :] * mb[None, :, None]).sum(axis=1) / mb_tot

            dist_nm = np.linalg.norm(cen_a_nm - cen_b_nm, axis=1)
            return [Quantity(float(d), nanometer) for d in dist_nm]

        # Static / multi-model fallback
        return [m.distance(group_a, group_b, center=center) for m in self.models]

    def distance_vector(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        *,
        center: str = "cog",
    ) -> list[Quantity]:
        """
        Center displacement vectors (from group_b -> group_a) for all models.

        If trajectory-backed (Structure._coords_nm is set), uses a vectorized numpy
        implementation over all frames. Otherwise falls back to per-model computation
        and returns a list of Quantity[Vec3] in nm.

        center:
          - cog: center of geometry
          - com: center of mass
        """
        if not self.models:
            return []

        mode = (center or "cog").lower()
        if mode not in {"cog", "com"}:
            raise ValueError(f"Center option {center} is not valid. Use 'cog' or 'com'.")

        # Fast path for trajectory-backed structures
        if self._coords_nm is not None:
            flat_a = Model._flatten_indices(group_a)  # type: ignore[attr-defined]
            flat_b = Model._flatten_indices(group_b)  # type: ignore[attr-defined]

            if not flat_a or not flat_b:
                raise ValueError("both groups have to contain at least one atom")

            n_atoms = self.natoms()
            for idx in flat_a + flat_b:
                if idx < 0 or idx >= n_atoms:
                    raise IndexError(
                        f"Atom index {idx} is out of range for structure with {n_atoms} atoms"
                    )

            idx_a = np.asarray(flat_a, dtype=np.int64)
            idx_b = np.asarray(flat_b, dtype=np.int64)

            coords_nm = self._coords_nm  # (n_frames, n_atoms, 3)

            if mode == "cog":
                cen_a_nm = coords_nm[:, idx_a, :].mean(axis=1)
                cen_b_nm = coords_nm[:, idx_b, :].mean(axis=1)
            else:
                atoms0 = self.models[0].atoms
                ma = np.asarray([Model._atom_mass(atoms0[i]) for i in idx_a], dtype=float)
                mb = np.asarray([Model._atom_mass(atoms0[i]) for i in idx_b], dtype=float)
                ma_tot = float(ma.sum())
                mb_tot = float(mb.sum())
                if ma_tot == 0.0 or mb_tot == 0.0:
                    raise ValueError("Total mass of group is zero; cannot compute COM.")
                cen_a_nm = (coords_nm[:, idx_a, :] * ma[None, :, None]).sum(axis=1) / ma_tot
                cen_b_nm = (coords_nm[:, idx_b, :] * mb[None, :, None]).sum(axis=1) / mb_tot

            diff_nm = cen_a_nm - cen_b_nm  # (n_frames, 3)

            out: list[Quantity] = []
            for v in diff_nm:
                out.append(Vec3(float(v[0]), float(v[1]), float(v[2])) * nanometer)
            return out

        # Static / multi-model fallback
        return [m.distance_vector(group_a, group_b, center=center) for m in self.models]

    def angle_norm(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_a1: Union[list[int], list[list[int]]],
        group_a2: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_b1: Union[list[int], list[list[int]]],
        group_b2: Union[list[int], list[list[int]]],
        *,
        center: str = "cog",
    ) -> list[Quantity]:
        """
        Plane-normal angle (radians) between two planes for all models.

        Geometry matches the umbrella in set_umbrella_angle_norm.
        """
        if not self.models:
            return []
        return [
            m.angle_norm(
                group_a,
                group_a1,
                group_a2,
                group_b,
                group_b1,
                group_b2,
                center=center,
            )
            for m in self.models
        ]

    def dihedral(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
        group_d: Union[list[int], list[list[int]]],
        *,
        center: str = "cog",
    ) -> list[Quantity]:
        """
        Dihedral angle (radians, −π..π) between four centroids for all models.

        Geometry matches the umbrella in set_umbrella_dihedral
        """
        if not self.models:
            return []
        return [m.dihedral(group_a, group_b, group_c, group_d, center=center) for m in self.models]

    def angle(
        self,
        group_a: Union[list[int], list[list[int]]],
        group_b: Union[list[int], list[list[int]]],
        group_c: Union[list[int], list[list[int]]],
        *,
        center: str = "cog",
    ) -> list[Quantity]:
        """
        Rotation angles (radians) per model, matching set_umbrella_angle.
        """
        if not self.models:
            return []
        return [m.angle(group_a, group_b, group_c, center=center) for m in self.models]

    # --- I/O helpers ---------------------------------------------------------

    def write_pdb(
        self,
        file: FileLike,
        *,
        model_records: Optional[bool] = None,
        allhis: bool = False,
    ) -> None:
        """
        Write this Structure to a PDB file.

        By default, if the Structure contains more than one model, a single
        multi-model PDB is written using MODEL/ENDMDL records for each model.
        """
        writer = PDBWriter()
        writer.write(self, file, model_records=model_records, allhis=allhis)


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
        # ensure masses are initialized from elements if missing
        for a in template.atoms:
            if getattr(a, "mass", None) is None:
                a.mass = _mass_from_element_symbol(getattr(a, "element", ""))
        s = Structure(models=[template])
        return s, template

    if isinstance(template, Structure):
        # ensure masses are initialized from elements if missing
        for a in template.model.atoms:
            if getattr(a, "mass", None) is None:
                a.mass = _mass_from_element_symbol(getattr(a, "element", ""))
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


class _PDBTextSink:
    """
    Internal helper: line-oriented text sink that understands FileLike.
    """

    def __init__(self, file: FileLike):
        self._needs_close = False
        self._binary = False

        if isinstance(file, io.StringIO):
            self._fh = file
        elif isinstance(file, io.BytesIO):
            self._fh = file
            self._binary = True
        else:
            p = Path(file)
            if p.suffix == ".gz":
                self._fh = gzip.open(p, "wt", encoding="utf-8", newline="")
            else:
                self._fh = open(p, "w", encoding="utf-8", newline="\n")
            self._needs_close = True

    def write_line(self, line: str) -> None:
        if self._binary:
            assert isinstance(self._fh, io.BytesIO)
            self._fh.write((line + "\n").encode("utf-8"))
        else:
            self._fh.write(line + "\n")

    def close(self) -> None:
        if self._needs_close:
            self._fh.close()


class PDBWriter:
    """
    Minimal PDB writer, parallel to PDBReader.

    - Accepts a Structure (possibly multi-model) or a single Model.
    - Uses MODEL/ENDMDL records when there is more than one model
      by default, so you get a single PDB with multiple models.
    - Coordinates come from the Model, including trajectory-backed
      Structures via Model._coord_angstrom().
    """

    def write(
        self,
        structure: Union[Structure, Model],
        file: FileLike,
        *,
        model_records: Optional[bool] = None,
        allhis: bool = False,
    ) -> None:
        """
        Write a Structure or Model to PDB.

        Parameters
        ----------
        structure : Structure or Model
            Data to write.
        file : FileLike
            Output path or file-like object.
        model_records : Optional[bool], optional
            - None (default): use MODEL/ENDMDL only if there is >1 model.
            - True          : always emit MODEL/ENDMDL for each model.
            - False         : never emit MODEL/ENDMDL.
        """
        if isinstance(structure, Model):
            s = Structure(models=[structure])
        else:
            s = structure

        sink = _PDBTextSink(file)
        try:
            if not s.models:
                sink.write_line("END")
                return

            if model_records is None:
                use_model_records = len(s.models) > 1
            else:
                use_model_records = bool(model_records)

            for frame_index, m in enumerate(s.models):
                if use_model_records:
                    model_id = getattr(m, "model_id", None)
                    if not isinstance(model_id, int):
                        model_id = frame_index + 1
                    sink.write_line(f"MODEL     {model_id:4d}")

                self._write_model_atoms(m, sink, allhis=allhis)

                if use_model_records:
                    sink.write_line("ENDMDL")

            sink.write_line("END")
        finally:
            sink.close()

    @staticmethod
    def _write_model_atoms(
        model: Model,
        sink: _PDBTextSink,
        *,
        allhis: bool = False,
    ) -> None:
        """
        Write all ATOM records for a single Model.

        Coordinates are taken from Model._coord_angstrom(idx), so this works
        both for static and trajectory-backed models.
        """
        natoms = model.natoms()
        if natoms == 0:
            return

        idx = 0
        serial_counter = 1

        # Deterministic chain order: respect insertion order from parsing.
        for key in model.chain:
            ch = model.chain[key]

            for res in ch.residues:
                for atom in res.atoms:
                    if idx >= natoms:
                        raise RuntimeError(
                            "Internal inconsistency while writing PDB: atom index overflow"
                        )

                    x_ang, y_ang, z_ang = model._coord_angstrom(idx)
                    idx += 1

                    # Preserve existing PDB serials when present; otherwise assign sequential.
                    atom_serial = getattr(atom, "serial", 0) or 0
                    serial = atom_serial if atom_serial > 0 else serial_counter
                    serial_counter += 1

                    raw_name = (atom.name or "").strip()
                    name = raw_name[:4]
                    if len(name) <= 3:
                        name_field = f" {name:<3s}"
                    else:
                        name_field = f"{name:<4s}"
                    resname_raw = (atom.resname or "").strip().upper()
                    if allhis and resname_raw in {"HIS", "HSD", "HSE", "HSP"}:
                        resname = "HIS"
                    else:
                        resname = resname_raw[:4]
                    chain_id = (atom.chain or " ")[:1]
                    seg = (atom.seg or "")[:4]
                    # element = (atom.element or "")[:2].upper()

                    # PDB v3.3-like formatting compatible with _parse_atom_line.

                    if int(serial) < 100000:
                        line = (
                            "{:<6s}{:>5d} {}{:1s}{:<4s}{:1s}{:>4d}{:1s}"
                            "   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:>4s}"
                        ).format(
                            "ATOM",
                            int(serial),
                            name_field,
                            "",  # altLoc
                            resname,
                            chain_id,
                            int(atom.resnum),
                            "",  # iCode
                            float(x_ang),
                            float(y_ang),
                            float(z_ang),
                            1.00,  # occupancy
                            0.00,  # tempFactor
                            seg,
                        )
                    else:
                        line = (
                            "{:<6s}***** {}{:1s}{:<4s}{:1s}{:>4d}{:1s}"
                            "   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:>4s}"
                        ).format(
                            "ATOM",
                            name_field,
                            "",  # altLoc
                            resname,
                            chain_id,
                            int(atom.resnum),
                            "",  # iCode
                            float(x_ang),
                            float(y_ang),
                            float(z_ang),
                            1.00,  # occupancy
                            0.00,  # tempFactor
                            seg,
                        )
                    sink.write_line(line)

        if idx != natoms:
            raise RuntimeError(
                f"Internal inconsistency while writing PDB: wrote {idx} atoms, expected {natoms}"
            )


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
        mass=_mass_from_element_symbol(element),
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
    "HSP",
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
_WATER_RESNAMES: set[str] = {"HOH", "TIP3", "WAT", "SPC", "TIP4"}
_ION_RESNAMES: set[str] = {"SOD", "POT", "CLA", "MG", "NA"}


def _is_hydrogen(atom: Atom) -> bool:
    """Return True if this atom should be treated as hydrogen.

    Uses both the deduced element and common PDB atom-name patterns (H*, 1H*, 2H*, ...)
    to be robust against imperfect element assignment.

    Notes
    -----
    Do *not* treat names like "NH1" or "OH2" as hydrogens; only digit-prefixed
    patterns (e.g. "1H", "2H") are considered in the second-character rule.
    """
    name = (atom.name or "").strip().upper()
    if not name:
        return False
    el = (getattr(atom, "element", "") or "").upper()
    if el == "H":
        return True
    if name[0] == "H":
        return True
    if len(name) >= 2 and name[0].isdigit() and name[1] == "H":
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
                    mass=getattr(old_atom, "mass", None),
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

    coords_nm = np.asarray(traj.xyz, dtype=np.float64)

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


def compare_topology(
    top_a: Topology,
    top_b: Topology,
    *,
    max_items: int = 200,
) -> str:
    """
    Compare two OpenMM Topology objects.

    Comparison rules
    ----------------
    - Chain IDs are ignored for chain equality, but reported separately.
    - Chains differ only if number of residues or atoms differs.
    - Residues differ only if residue name or number of atoms differs.
    - Atom diffs are reported by atom-name count changes within matching residues.
    - Bonds are compared by labeled atom paths using chain/residue indices and
      atom names (chain/residue IDs are ignored).

    Output
    ------
    1) Summary counts (chains, residues, atoms, bonds) for both.
    2) Diff-style sections for chains, residues, atoms, and bonds.
       - removed: '-' lines
       - added:   '+' lines
       - changed: '~' lines, compact "a -> b" format
    """
    if max_items <= 0:
        raise ValueError("max_items must be > 0")

    def _counts(t: Topology) -> tuple[int, int, int, int]:
        return (
            int(t.getNumChains()),
            int(t.getNumResidues()),
            int(t.getNumAtoms()),
            int(t.getNumBonds()),
        )

    def _chain_id(ch: Any) -> str:
        cid = getattr(ch, "id", None)
        s = str(cid) if cid is not None else ""
        return s.strip()

    def _norm_name(v: Any) -> str:
        return (str(v) if v is not None else "").strip().upper()

    def _name_counts(names: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in names:
            out[n] = out.get(n, 0) + 1
        return out

    def _chain_counts(ch: Any) -> tuple[int, int]:
        residues = list(ch.residues())
        n_res = len(residues)
        n_atoms = sum(sum(1 for _ in r.atoms()) for r in residues)
        return int(n_res), int(n_atoms)

    def _res_counts(res: Any) -> tuple[str, int]:
        rname = _norm_name(getattr(res, "name", ""))
        n_atoms = sum(1 for _ in res.atoms())
        return rname, int(n_atoms)

    def _res_atom_counts(res: Any) -> dict[str, int]:
        names = [_norm_name(getattr(a, "name", "")) for a in res.atoms()]
        return _name_counts(names)

    def _atom_labels(t: Topology) -> dict[int, str]:
        labels: dict[int, str] = {}
        for ci, ch in enumerate(t.chains()):
            for ri, res in enumerate(ch.residues()):
                rname = _norm_name(getattr(res, "name", ""))
                seen: dict[str, int] = {}
                for a in res.atoms():
                    aname = _norm_name(getattr(a, "name", ""))
                    seen[aname] = seen.get(aname, 0) + 1
                    suf = f"#{seen[aname]}" if seen[aname] > 1 else ""
                    labels[int(a.index)] = f"c{ci}:r{ri}:{rname}:{aname}{suf}"
        return labels

    def _bond_set(t: Topology, labels: dict[int, str]) -> set[str]:
        out: set[str] = set()
        for a1, a2 in t.bonds():
            l1 = labels.get(int(a1.index))
            l2 = labels.get(int(a2.index))
            if l1 is None or l2 is None:
                continue
            out.add(f"{l1} -- {l2}" if l1 <= l2 else f"{l2} -- {l1}")
        return out

    a_counts = _counts(top_a)
    b_counts = _counts(top_b)

    lines: list[str] = []
    lines.append(
        "Topology 1: "
        f"{a_counts[0]} chains, {a_counts[1]} residues, "
        f"{a_counts[2]} atoms, {a_counts[3]} bonds"
    )
    lines.append(
        "Topology 2: "
        f"{b_counts[0]} chains, {b_counts[1]} residues, "
        f"{b_counts[2]} atoms, {b_counts[3]} bonds"
    )
    lines.append(
        "Delta (2 - 1): "
        f"{b_counts[0] - a_counts[0]} chains, "
        f"{b_counts[1] - a_counts[1]} residues, "
        f"{b_counts[2] - a_counts[2]} atoms, "
        f"{b_counts[3] - a_counts[3]} bonds"
    )

    chains_a = list(top_a.chains())
    chains_b = list(top_b.chains())
    n_common = min(len(chains_a), len(chains_b))

    # ---- Chains ---------------------------------------------------------
    removed_ch = list(range(n_common, len(chains_a)))
    added_ch = list(range(n_common, len(chains_b)))

    cnt_a = [_chain_counts(ch) for ch in chains_a]
    cnt_b = [_chain_counts(ch) for ch in chains_b]
    changed_ch = [
        i for i in range(n_common) if (cnt_a[i][0] != cnt_b[i][0]) or (cnt_a[i][1] != cnt_b[i][1])
    ]

    id_changed = [i for i in range(n_common) if _chain_id(chains_a[i]) != _chain_id(chains_b[i])]

    lines.append("\nChains:")
    lines.append(f"  removed={len(removed_ch)}, added={len(added_ch)}, changed={len(changed_ch)}")

    shown = 0
    for i in removed_ch:
        if shown >= max_items:
            break
        cid = _chain_id(chains_a[i]) or "''"
        nres, nat = cnt_a[i]
        lines.append(f"- Chain {i}: residues={nres}, atoms={nat}, id={cid}")
        shown += 1

    for i in added_ch:
        if shown >= max_items:
            break
        cid = _chain_id(chains_b[i]) or "''"
        nres, nat = cnt_b[i]
        lines.append(f"+ Chain {i}: residues={nres}, atoms={nat}, id={cid}")
        shown += 1

    for i in changed_ch:
        if shown >= max_items:
            break
        a_res, a_at = cnt_a[i]
        b_res, b_at = cnt_b[i]
        parts: list[str] = []
        if a_res != b_res:
            parts.append(f"residues {a_res}->{b_res}")
        if a_at != b_at:
            parts.append(f"atoms {a_at}->{b_at}")
        change = ", ".join(parts) if parts else "no count changes"
        lines.append(f"~ Chain {i}: {change}")
        shown += 1

    total_chain_diffs = len(removed_ch) + len(added_ch) + len(changed_ch)
    if total_chain_diffs > max_items:
        lines.append(f"  ... ({total_chain_diffs - max_items} more chain diffs not shown)")

    lines.append("\nChain ID differences (ignored for chain equality):")
    if not id_changed:
        lines.append("  (no chain ID differences)")
    else:
        shown = 0
        for i in id_changed:
            if shown >= max_items:
                break
            a_id = _chain_id(chains_a[i]) or "''"
            b_id = _chain_id(chains_b[i]) or "''"
            lines.append(f"  ~ Chain {i}: id {a_id} -> {b_id}")
            shown += 1
        if len(id_changed) > max_items:
            lines.append(f"  ... ({len(id_changed) - max_items} more not shown)")

    # ---- Residues -------------------------------------------------------
    res_removed: list[tuple[int, int]] = []
    res_added: list[tuple[int, int]] = []
    res_changed: list[tuple[int, int]] = []

    for ci in range(n_common):
        res_a = list(chains_a[ci].residues())
        res_b = list(chains_b[ci].residues())
        nres_c = min(len(res_a), len(res_b))

        for ri in range(nres_c):
            na, aa = _res_counts(res_a[ri])
            nb, ab = _res_counts(res_b[ri])
            if na != nb or aa != ab:
                res_changed.append((ci, ri))

        for ri in range(nres_c, len(res_a)):
            res_removed.append((ci, ri))
        for ri in range(nres_c, len(res_b)):
            res_added.append((ci, ri))

    lines.append("\nResidues:")
    lines.append(
        f"  removed={len(res_removed)}, added={len(res_added)}, changed={len(res_changed)}"
    )

    shown = 0
    for ci, ri in res_removed:
        if shown >= max_items:
            break
        res = list(chains_a[ci].residues())[ri]
        rname, nat = _res_counts(res)
        lines.append(f"- c{ci} r{ri}: {rname} atoms={nat}")
        shown += 1

    for ci, ri in res_added:
        if shown >= max_items:
            break
        res = list(chains_b[ci].residues())[ri]
        rname, nat = _res_counts(res)
        lines.append(f"+ c{ci} r{ri}: {rname} atoms={nat}")
        shown += 1

    for ci, ri in res_changed:
        if shown >= max_items:
            break
        ra = list(chains_a[ci].residues())[ri]
        rb = list(chains_b[ci].residues())[ri]
        na, aa = _res_counts(ra)
        nb, ab = _res_counts(rb)
        parts: list[str] = []
        if na != nb:
            parts.append(f"name {na}->{nb}")
        if aa != ab:
            parts.append(f"atoms {aa}->{ab}")
        change = ", ".join(parts) if parts else "no changes"
        lines.append(f"~ c{ci} r{ri}: {change}")
        shown += 1

    total_res_diffs = len(res_removed) + len(res_added) + len(res_changed)
    if total_res_diffs > max_items:
        lines.append(f"  ... ({total_res_diffs - max_items} more residue diffs not shown)")

    # ---- Atoms ----------------------------------------------------------
    atom_lines: list[str] = []
    atom_removed_total = 0
    atom_added_total = 0

    for ci in range(n_common):
        res_a = list(chains_a[ci].residues())
        res_b = list(chains_b[ci].residues())
        nres_c = min(len(res_a), len(res_b))
        for ri in range(nres_c):
            ra = res_a[ri]
            rb = res_b[ri]
            rna, _ = _res_counts(ra)
            rnb, _ = _res_counts(rb)
            if rna != rnb:
                continue

            ca = _res_atom_counts(ra)
            cb = _res_atom_counts(rb)
            if ca == cb:
                continue

            keys = sorted(set(ca) | set(cb))
            removed: list[tuple[str, int]] = []
            added: list[tuple[str, int]] = []

            for k in keys:
                da = int(ca.get(k, 0))
                db = int(cb.get(k, 0))
                if da > db:
                    removed.append((k, da - db))
                elif db > da:
                    added.append((k, db - da))

            for _, n in removed:
                atom_removed_total += int(n)
            for _, n in added:
                atom_added_total += int(n)

            # Pair removals with additions as renames when possible.
            rpretty = rna.title() if rna else rna
            removed = sorted(removed)
            added = sorted(added)
            i = 0
            j = 0
            while i < len(removed) and j < len(added):
                old_name, n_old = removed[i]
                new_name, n_new = added[j]
                n = n_old if n_old < n_new else n_new

                suf = f" x{n}" if n != 1 else ""
                atom_lines.append(f"c{ci} r{ri} {rpretty}: {old_name}->{new_name}{suf}")

                n_old -= n
                n_new -= n
                if n_old == 0:
                    i += 1
                else:
                    removed[i] = (old_name, n_old)
                if n_new == 0:
                    j += 1
                else:
                    added[j] = (new_name, n_new)

            # Leftovers are true additions/removals.
            for k, n in removed[i:]:
                suf = f" x{n}" if n != 1 else ""
                atom_lines.append(f"- c{ci} r{ri} {rpretty}: {k}{suf}")
            for k, n in added[j:]:
                suf = f" x{n}" if n != 1 else ""
                atom_lines.append(f"+ c{ci} r{ri} {rpretty}: {k}{suf}")

    lines.append("\nAtoms (by name-count within matching residue names):")
    lines.append(f"  removed={atom_removed_total}, added={atom_added_total}")

    if not atom_lines:
        lines.append("  (no atom-name differences)")
    else:
        shown = 0
        for s in atom_lines:
            if shown >= max_items:
                break
            lines.append(s)
            shown += 1
        if len(atom_lines) > max_items:
            lines.append(f"  ... ({len(atom_lines) - max_items} more atom diffs not shown)")

    # ---- Bonds ----------------------------------------------------------
    labels_a = _atom_labels(top_a)
    labels_b = _atom_labels(top_b)
    bonds_a = _bond_set(top_a, labels_a)
    bonds_b = _bond_set(top_b, labels_b)

    bonds_removed = sorted(bonds_a - bonds_b)
    bonds_added = sorted(bonds_b - bonds_a)

    lines.append("\nBonds:")
    lines.append(f"  removed={len(bonds_removed)}, added={len(bonds_added)}")

    def _split_bond_str(b: str) -> tuple[str, str]:
        parts = b.split(" -- ", 1)
        if len(parts) != 2:
            return b, b
        return parts[0], parts[1]

    def _parse_atom_path(p: str) -> tuple[int, int, str, str]:
        parts = p.split(":")
        if len(parts) != 4:
            return -1, -1, "", p
        try:
            ci = int(parts[0][1:]) if parts[0].startswith("c") else int(parts[0])
            ri = int(parts[1][1:]) if parts[1].startswith("r") else int(parts[1])
        except (ValueError, IndexError):
            return -1, -1, "", parts[3]
        return ci, ri, parts[2], parts[3]

    def _res_ctx(p: str) -> tuple[int, int, str]:
        ci, ri, res, _ = _parse_atom_path(p)
        return ci, ri, res

    def _bond_ctx_desc(b: str) -> tuple[str, str]:
        a, c = _split_bond_str(b)
        ci1, ri1, r1, at1 = _parse_atom_path(a)
        ci2, ri2, r2, at2 = _parse_atom_path(c)

        key1 = (ci1, ri1, r1, at1)
        key2 = (ci2, ri2, r2, at2)
        if key2 < key1:
            ci1, ri1, r1, at1, ci2, ri2, r2, at2 = (
                ci2,
                ri2,
                r2,
                at2,
                ci1,
                ri1,
                r1,
                at1,
            )

        if (ci1, ri1, r1) == (ci2, ri2, r2):
            ctx = f"c{ci1} r{ri1} {r1.title()}"
        else:
            ctx = f"c{ci1} r{ri1} {r1.title()} <-> " f"c{ci2} r{ri2} {r2.title()}"
        return ctx, f"{at1}--{at2}"

    def _bond_key(a: str, c: str) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
        k1 = _res_ctx(a)
        k2 = _res_ctx(c)
        return (k1, k2) if k1 <= k2 else (k2, k1)

    rem_items: list[tuple[str, str, str]] = []
    for b in bonds_removed:
        a, c = _split_bond_str(b)
        rem_items.append((a, c, b))

    add_items: list[tuple[str, str, str]] = []
    add_by_ep: dict[str, list[int]] = {}
    for i, b in enumerate(bonds_added):
        a, c = _split_bond_str(b)
        add_items.append((a, c, b))
        add_by_ep.setdefault(a, []).append(i)
        add_by_ep.setdefault(c, []).append(i)

    used_add: set[int] = set()
    paired: list[tuple[str, str]] = []
    unpaired_rem: list[tuple[str, str, str]] = []

    for ra, rb, rstr in rem_items:
        match: Optional[int] = None
        for shared, other_rem in ((ra, rb), (rb, ra)):
            for j in add_by_ep.get(shared, []):
                if j in used_add:
                    continue
                aa, ab, _ = add_items[j]
                other_add = ab if aa == shared else aa
                if other_add == other_rem:
                    continue
                if _res_ctx(other_add) != _res_ctx(other_rem):
                    continue
                match = j
                break
            if match is not None:
                break

        if match is None:
            unpaired_rem.append((ra, rb, rstr))
        else:
            used_add.add(match)
            paired.append((rstr, add_items[match][2]))

    unpaired_add = [add_items[i] for i in range(len(add_items)) if i not in used_add]

    rem_by_key = {}
    add_by_key = {}

    for ra, rb, rstr in unpaired_rem:
        rem_by_key.setdefault(_bond_key(ra, rb), []).append((ra, rb, rstr))
    for aa, ab, astr in unpaired_add:
        add_by_key.setdefault(_bond_key(aa, ab), []).append((aa, ab, astr))

    leftover_rem: list[str] = []
    leftover_add: list[str] = []
    for key in sorted(set(rem_by_key) | set(add_by_key)):
        rlist = sorted(rem_by_key.get(key, []), key=lambda x: x[2])
        alist = sorted(add_by_key.get(key, []), key=lambda x: x[2])

        if rlist and alist and len(rlist) == len(alist):
            for r_it, a_it in zip(rlist, alist):
                paired.append((r_it[2], a_it[2]))
        else:
            leftover_rem.extend([x[2] for x in rlist])
            leftover_add.extend([x[2] for x in alist])

    bond_lines: list[str] = []
    for old, new in paired:
        ctx_old, desc_old = _bond_ctx_desc(old)
        ctx_new, desc_new = _bond_ctx_desc(new)
        if ctx_old == ctx_new:
            bond_lines.append(f"{ctx_old}: {desc_old}->{desc_new}")
        else:
            bond_lines.append(f"{ctx_old}: {desc_old}->{ctx_new}: {desc_new}")

    shown = 0
    for s in bond_lines:
        if shown >= max_items:
            break
        lines.append(s)
        shown += 1

    for b in sorted(leftover_rem):
        if shown >= max_items:
            break
        ctx, desc = _bond_ctx_desc(b)
        lines.append(f"- {ctx}: {desc}")
        shown += 1

    for b in sorted(leftover_add):
        if shown >= max_items:
            break
        ctx, desc = _bond_ctx_desc(b)
        lines.append(f"+ {ctx}: {desc}")
        shown += 1

    total_bond_diffs = len(bond_lines) + len(leftover_rem) + len(leftover_add)

    if total_bond_diffs > max_items:
        lines.append(f"  ... ({total_bond_diffs - max_items} more bond diffs not shown)")

    return "\n".join(lines)

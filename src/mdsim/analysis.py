from __future__ import annotations

import io
import math
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from openmm.app import Topology
from openmm.unit import Quantity, dimensionless, nanometer, nanoseconds

from .molecule_data import PDBReader, StructureSelector, iter_dcd

FileLike = Union[str, Path, io.BytesIO, io.StringIO]


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


# --- number of ions for a given concentration and net charge that needs to be neutralized ---


def ion_counts(xbox, ybox, zbox, conc_mM: float, ncharge: int):
    x = xbox.value_in_unit(nanometer)
    y = ybox.value_in_unit(nanometer)
    z = zbox.value_in_unit(nanometer)
    vol_nm3 = x * y * z
    NA = 6.02214076e23  # mol^-1
    nion = int(math.floor(vol_nm3 * conc_mM * NA * 1e-27 + 0.5))
    if ncharge < 0:
        nsod = nion + (-ncharge)  # extra Na+ to offset negative solute
        ncla = nion
    elif ncharge > 0:
        nsod = nion
        ncla = nion + ncharge  # extra Cl- to offset positive solute
    else:
        nsod = nion
        ncla = nion
    return nion, nsod, ncla


# --- geometry analysis ------------------------------------------------


def _to_xyz_nm(p: Any) -> np.ndarray:
    """Convert point to float xyz in nm."""
    if isinstance(p, Quantity):
        arr = np.asarray(p.value_in_unit(nanometer), dtype=float)
        if arr.shape == (3,):
            return arr
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr[0]
        raise ValueError(f"Unexpected Quantity point shape {arr.shape}")

    if isinstance(p, (tuple, list)) and len(p) == 3:
        if any(isinstance(x, Quantity) for x in p):
            xyz = [
                (
                    float(x.value_in_unit(nanometer))
                    if isinstance(x, Quantity)
                    else float(Quantity(x).value_in_unit(nanometer))
                )
                for x in p
            ]
            return np.asarray(xyz, dtype=float)
        return np.asarray(p, dtype=float)

    raise TypeError(f"Unsupported point type: {type(p)}")


def plane_normal(
    points: Sequence[Any],
    *,
    align_axis: Any = (0.0, 0.0, 1.0),
    normalize_each: bool = True,
    ignore_degenerate: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Average plane normal from >=3 points, enforcing each triplet normal to align
    with `align_axis` (default +z).

    - Computes normals for all point triplets (i<j<k).
    - For each triplet normal n: if dot(n, a) < 0 then n := -n.
    - Returns a unit vector (3,) float.

    `align_axis` accepts the same point-like inputs as points: tuple floats (nm),
    Quantity, or tuple of Quantities (interpreted as a direction in nm basis).
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points")

    xyz = np.stack([_to_xyz_nm(p) for p in points], axis=0)  # (n,3)

    a = _to_xyz_nm(align_axis)
    a_norm = float(np.linalg.norm(a))
    if a_norm <= eps:
        raise ValueError("align_axis must be non-zero")
    a = a / a_norm

    n_pts = xyz.shape[0]
    normals: list[np.ndarray] = []

    for i in range(n_pts - 2):
        p0 = xyz[i]
        for j in range(i + 1, n_pts - 1):
            u = xyz[j] - p0
            for k in range(j + 1, n_pts):
                v = xyz[k] - p0
                n = np.cross(u, v)
                n_norm = float(np.linalg.norm(n))
                if n_norm <= eps:
                    if ignore_degenerate:
                        continue
                    raise ValueError("Degenerate triplet produced near-zero normal")

                if normalize_each:
                    n = n / n_norm

                if float(np.dot(n, a)) < 0.0:
                    n = -n

                normals.append(n)

    if not normals:
        raise ValueError("All triplets were degenerate; cannot define a plane normal")

    avg = np.mean(np.stack(normals, axis=0), axis=0)
    avg_norm = float(np.linalg.norm(avg))
    if avg_norm <= eps:
        raise ValueError("Averaged normal is near-zero")
    return avg / avg_norm


def plane_normal_quantity(points: Sequence[Any], **kwargs: Any) -> Quantity:
    """Same as plane_normal, but returns a dimensionless Quantity."""
    n = plane_normal(points, **kwargs)
    return Quantity(n, dimensionless)


# ---- RDF / virial coefficients / structure factors -----------------------------


def _as_file_list(x: Union[FileLike, Sequence[FileLike]]) -> list[FileLike]:
    if isinstance(x, (str, Path, io.BytesIO, io.StringIO)):
        return [x]
    return list(x)


def _sinc(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    m = np.abs(x) <= float(eps)
    out[m] = 1.0
    out[~m] = np.sin(x[~m]) / x[~m]
    return out


def _box_lengths_nm(box_nm: Any) -> np.ndarray:
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


def _min_image_disp_nm(d: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    return d - np.rint(d / box_nm.reshape(1, 3)) * box_nm.reshape(1, 3)


def _wrap_nm(xyz_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    b = box_nm.reshape(1, 3)
    return xyz_nm - np.floor(xyz_nm / b) * b


def solute_groups(template: Any, *, group_spec: str = "protein") -> list[np.ndarray]:
    """Return one atom-index list per solute copy (typically one per chain).

    Notes
    -----
    For string specs, StructureSelector yields *per-chain* groups unless the
    spec contains explicit chain IDs.
    """
    if isinstance(template, (str, Path, io.BytesIO, io.StringIO)):
        tmpl = PDBReader().read(template)
    else:
        tmpl = template

    out = StructureSelector(group_spec).atom_lists(tmpl)
    return [np.asarray(g, dtype=np.int32) for g in out if g]


def atom_masses(template: Any) -> np.ndarray:
    """Per-atom masses for a template Structure/Model (dalton), as float64."""
    model = template.model if hasattr(template, "model") else template
    return np.asarray(
        [float(getattr(a, "mass", 0.0) or 0.0) for a in model.atoms], dtype=np.float64
    )


# Back-compat for older internal name.
_atom_masses = atom_masses


def group_centers_nm(
    coords_nm: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    masses: Optional[np.ndarray] = None,
    box_nm: Optional[np.ndarray] = None,
    boxes_nm: Optional[np.ndarray] = None,
    center: str = "cog",
    unwrap: bool = True,
    wrap: bool = True,
) -> np.ndarray:
    """Compute per-group centers (COG/COM) from coordinates in nm.

    Parameters
    ----------
    coords_nm
        (n_atoms, 3) or (n_frames, n_atoms, 3)
    groups
        Atom indices into the *coords* atom axis.
    masses
        Per-atom masses for coords atoms (length n_atoms). Required for COM.
    box_nm / boxes_nm
        Either a constant (3,) box or per-frame (n_frames, 3) box lengths in nm.
        Required if unwrap or wrap is True.
    unwrap
        Unwrap atoms within each group using minimum-image relative to group atom 0.
    wrap
        Wrap resulting centers back into [0, L) along each axis.

    Returns
    -------
    centers_nm
        (n_groups, 3) if single-frame input, else (n_frames, n_groups, 3)
    """
    coords = np.asarray(coords_nm, dtype=np.float64)
    if coords.ndim == 2:
        coords = coords[None, :, :]
        single = True
    elif coords.ndim == 3:
        single = False
    else:
        raise ValueError("coords_nm must have shape (n_atoms,3) or (n_frames,n_atoms,3)")

    n_frames = int(coords.shape[0])

    boxes = None
    if boxes_nm is not None:
        boxes = np.asarray(boxes_nm, dtype=np.float64)
        if boxes.shape != (n_frames, 3):
            raise ValueError("boxes_nm must have shape (n_frames, 3)")
    elif box_nm is not None:
        b = np.asarray(box_nm, dtype=np.float64).reshape(3)
        if np.any(b <= 0.0):
            raise ValueError("box lengths must be positive")
        boxes = np.broadcast_to(b, (n_frames, 3)).copy()

    center_mode = str(center).strip().lower()
    if center_mode not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    if (unwrap or wrap) and boxes is None:
        raise ValueError("box_nm/boxes_nm is required when unwrap or wrap is True")

    if center_mode == "com" and masses is None:
        raise ValueError("masses must be provided for center='com'")

    out = np.empty((n_frames, len(groups), 3), dtype=np.float64)

    for gi, idx in enumerate(groups):
        idx_i = np.asarray(idx, dtype=np.int64)
        gxyz = coords[:, idx_i, :]

        if unwrap:
            ref = gxyz[:, 0:1, :]
            delta = gxyz - ref
            delta -= np.rint(delta / boxes[:, None, :]) * boxes[:, None, :]
            gxyz = ref + delta

        if center_mode == "cog" or masses is None:
            cen = np.mean(gxyz, axis=1)
        else:
            w = np.asarray(masses[idx_i], dtype=np.float64)
            tot = float(np.sum(w))
            if tot <= 0.0:
                cen = np.mean(gxyz, axis=1)
            else:
                cen = np.sum(gxyz * w[None, :, None], axis=1) / tot

        if wrap:
            cen -= np.floor(cen / boxes) * boxes

        out[:, gi, :] = cen

    return out[0] if single else out


def _pair_distances_nm(centers_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    n = int(centers_nm.shape[0])
    if n < 2:
        return np.empty((0,), dtype=np.float64)

    ii, jj = np.triu_indices(n, k=1)
    d = centers_nm[ii] - centers_nm[jj]
    d -= np.rint(d / box_nm) * box_nm
    return np.linalg.norm(d, axis=1)


def _peek_first_box_nm(
    dcd_file: FileLike,
    template_model: Any,
    atom_indices: Sequence[int],
    stride: int,
    *,
    box_nm: Optional[Sequence[float]],
) -> np.ndarray:
    if box_nm is not None:
        return _box_lengths_nm(box_nm)

    it = iter_dcd(
        dcd_file,
        template_model,
        chunk=1,
        stride=int(stride),
        atom_indices=atom_indices,
    )
    try:
        _, b = next(it)
    except StopIteration as exc:  # pragma: no cover
        raise ValueError("DCD appears to have no frames") from exc
    if b is None:
        raise ValueError("DCD does not include unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")
    b = np.asarray(b, dtype=np.float64).reshape(3)
    if np.any(b <= 0.0):
        raise ValueError("box lengths must be positive")
    return b


def _rdf_single_dcd(
    dcd_file: FileLike,
    template_model: Any,
    *,
    atom_indices: Sequence[int],
    groups: Sequence[np.ndarray],
    masses: Optional[np.ndarray],
    center: str,
    unwrap: bool,
    r_edges_nm: np.ndarray,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_nm: Optional[Sequence[float]],
) -> tuple[np.ndarray, int, float]:
    n_bins = int(r_edges_nm.size - 1)
    shell_vol = (4.0 * math.pi / 3.0) * (np.power(r_edges_nm[1:], 3) - np.power(r_edges_nm[:-1], 3))

    n_groups = int(len(groups))
    if n_groups < 2:
        raise ValueError("need >=2 groups to compute an RDF")

    n_pairs = n_groups * (n_groups - 1) // 2
    hist_v = np.zeros(n_bins, dtype=np.float64)
    n_frames = 0
    min_half_box = float("inf")

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
            b = _box_lengths_nm(box_frame_nm)

        if np.any(b <= 0.0):
            raise ValueError("box lengths must be positive")
        min_half_box = min(min_half_box, 0.5 * float(np.min(b)))

        vol = float(b[0] * b[1] * b[2])
        if vol <= 0.0:
            raise ValueError("non-positive box volume")

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

        h, _ = np.histogram(r, bins=r_edges_nm)
        hist_v += h.astype(np.float64) * vol
        n_frames += 1

    if n_frames <= 0:
        raise ValueError("no frames selected for RDF computation")

    g_r = hist_v / (float(n_frames) * float(n_pairs) * shell_vol)
    return g_r, int(n_frames), float(min_half_box)


def _kb_from_gr(g_r: np.ndarray, r_edges_nm: np.ndarray) -> np.ndarray:
    shell_vol = (4.0 * math.pi / 3.0) * (np.power(r_edges_nm[1:], 3) - np.power(r_edges_nm[:-1], 3))
    return np.cumsum((g_r - 1.0) * shell_vol)


def rdf_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein",
    center: str = "cog",
    unwrap: bool = True,
    dr_nm: float = 0.01,
    r_max_nm: Optional[float] = None,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """RDF between multiple solute copies (group centers), plus KB and B2.

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

    groups_global = _selection_to_groups(tmpl, selection)
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

    if r_max_nm is None:
        half = []
        for dcd in dcd_list:
            b0 = _peek_first_box_nm(
                dcd,
                tmpl_model,
                atom_indices,
                int(stride),
                box_nm=box_nm,
            )
            half.append(0.5 * float(np.min(b0)))
        r_max = float(min(half))
    else:
        r_max = float(r_max_nm)

    if r_max <= 0.0:
        raise ValueError("r_max_nm must be > 0")
    if float(dr_nm) <= 0.0:
        raise ValueError("dr_nm must be > 0")

    r_edges = np.arange(0.0, r_max + float(dr_nm), float(dr_nm), dtype=np.float64)
    if r_edges.size < 2:
        raise ValueError("invalid r_max/dr combination")
    r_edges[-1] = r_max

    gr_blocks: list[np.ndarray] = []
    kb_blocks: list[np.ndarray] = []
    b2_blocks: list[np.ndarray] = []
    frames_per_block: list[int] = []
    min_half_boxes: list[float] = []

    for dcd in dcd_list:
        g_r, n_frames, min_half_box = _rdf_single_dcd(
            dcd,
            tmpl_model,
            atom_indices=atom_indices,
            groups=groups,
            masses=masses_sel,
            center=center_mode,
            unwrap=bool(unwrap),
            r_edges_nm=r_edges,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=int(frame_start),
            frame_stop=frame_stop,
            box_nm=box_nm,
        )
        kb = _kb_from_gr(g_r, r_edges)
        b2 = -0.5 * kb

        gr_blocks.append(g_r)
        kb_blocks.append(kb)
        b2_blocks.append(b2)
        frames_per_block.append(int(n_frames))
        min_half_boxes.append(float(min_half_box))

    r_keep = min(float(r_max), float(min(min_half_boxes)))
    n_keep = int(np.searchsorted(r_edges, r_keep, side="right") - 1)
    if n_keep < 1:
        raise ValueError("box is too small for the requested r_max_nm/dr_nm")
    r_edges = r_edges[: n_keep + 1]
    r_nm = 0.5 * (r_edges[:-1] + r_edges[1:])

    gr_arr = np.stack([g[:n_keep] for g in gr_blocks], axis=0)
    kb_arr = np.stack([k[:n_keep] for k in kb_blocks], axis=0)
    b2_arr = np.stack([b[:n_keep] for b in b2_blocks], axis=0)

    gr_mean = np.mean(gr_arr, axis=0)
    kb_mean = np.mean(kb_arr, axis=0)
    b2_mean = np.mean(b2_arr, axis=0)

    n_blocks = int(gr_arr.shape[0])
    if n_blocks < 2:
        gr_err = np.zeros_like(gr_mean)
        kb_err = np.zeros_like(kb_mean)
        b2_err = np.zeros_like(b2_mean)
        b2_final_err = 0.0
    else:
        denom = math.sqrt(float(n_blocks))
        gr_err = np.std(gr_arr, axis=0, ddof=1) / denom
        kb_err = np.std(kb_arr, axis=0, ddof=1) / denom
        b2_err = np.std(b2_arr, axis=0, ddof=1) / denom
        b2_final_err = float(np.std(b2_arr[:, -1], ddof=1) / denom)

    return {
        "r_nm": r_nm,
        "r_edges_nm": r_edges,
        "g_r": gr_mean,
        "g_r_err": gr_err,
        "kb_nm3": kb_mean,
        "kb_nm3_err": kb_err,
        "b2_r_nm3": b2_mean,
        "b2_r_nm3_err": b2_err,
        "b2_nm3": float(b2_mean[-1]),
        "b2_nm3_err": float(b2_final_err),
        "n_blocks": n_blocks,
        "frames_per_block": np.asarray(frames_per_block, dtype=np.int64),
        "selection": selection,
        "center": center_mode,
        "unwrap": bool(unwrap),
        "stride": int(stride),
    }


# --- site-site RDFs (coarse-grained CA per residue) ---------------------------


@dataclass(frozen=True)
class SiteRDFResult:
    r_nm: np.ndarray
    y: np.ndarray
    y_err: np.ndarray
    n_chains: int
    n_frames: int
    n_pairs_per_frame: int
    mode: str
    normalization: str
    res_i: int
    res_j: int
    atom_name: str


def _site_atom_indices_by_chain(
    tmpl: Any,
    *,
    resnum: int,
    atom_name: str = "CA",
) -> tuple[list[str], np.ndarray]:
    """
    Return (chain_keys, atom_indices) for `atom_name` at residue `resnum` in each chain.

    chain_keys are template_model.chain keys (e.g., seg IDs); indices are template atom indices.
    """
    model = tmpl.model if hasattr(tmpl, "model") else tmpl
    want = str(atom_name).strip().upper()
    out_keys: list[str] = []
    out_idx: list[int] = []

    # Need a fast atom id->index map in template ordering.
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


def _rdf_norm_from_counts(
    counts: np.ndarray,
    *,
    r_edges_nm: np.ndarray,
    n_frames: int,
    n_ref: int,
    vol_nm3: float,
    n_targets_per_ref: int,
    normalization: str,
) -> np.ndarray:
    norm = str(normalization).strip().lower()
    if norm not in {"gr", "prob", "number_density"}:
        raise ValueError("normalization must be 'gr', 'prob', or 'number_density'")

    shell_vol = (4.0 * math.pi / 3.0) * (np.power(r_edges_nm[1:], 3) - np.power(r_edges_nm[:-1], 3))

    if norm == "number_density":
        # number density around a *reference* bead
        denom = float(n_frames) * float(n_ref) * shell_vol
        return counts / denom

    if norm == "gr":
        # g(r) around a reference bead, relative to bulk target density
        rho = float(n_targets_per_ref) / float(vol_nm3)
        denom = float(n_frames) * float(n_ref) * rho * shell_vol
        return counts / denom

    # prob: radial probability density p(r) s.t. integral p(r) dr = 1
    dr = np.diff(r_edges_nm)
    tot = float(np.sum(counts))
    if tot <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    return (counts / tot) / dr


def site_rdf_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    res_i: int,
    res_j: int,
    atom_name: str = "CA",
    mode: str = "both",  # "intra" | "inter" | "both"
    normalization: str = "gr",  # "gr" | "prob" | "number_density"
    dr_nm: float = 0.02,
    r_max_nm: Optional[float] = None,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> SiteRDFResult:
    """
    Site-site RDF between residue `res_i` (reference) and `res_j` (counted)
    across chains.

    Replicates for SEM
    ------------------
    - intra: each chain contributes one distance per frame
    - inter: each reference chain contributes distances to all other chains
    - both: intra + inter combined for each reference chain
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

    m = str(mode).strip().lower()
    if m not in {"intra", "inter", "both"}:
        raise ValueError("mode must be 'intra', 'inter', or 'both'")

    norm = str(normalization).strip().lower()
    if norm not in {"gr", "prob", "number_density"}:
        raise ValueError("normalization must be 'gr', 'prob', or 'number_density'")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

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
    for k, vi in zip(keys_i, idx_i_full.tolist()):
        vj = map_j.get(k)
        if vj is None:
            continue
        ii.append(int(vi))
        jj.append(int(vj))

    if len(ii) < 2:
        raise ValueError("need >=2 chains with both residues present")

    idx_i_full = np.asarray(ii, dtype=np.int64)
    idx_j_full = np.asarray(jj, dtype=np.int64)
    n_ch = int(idx_i_full.size)

    atom_indices_full = sorted(set(idx_i_full.tolist() + idx_j_full.tolist()))
    idx_map = {old: new for new, old in enumerate(atom_indices_full)}
    idx_i = np.asarray([idx_map[int(x)] for x in idx_i_full.tolist()], dtype=np.int64)
    idx_j = np.asarray([idx_map[int(x)] for x in idx_j_full.tolist()], dtype=np.int64)

    if r_max_nm is None:
        half = []
        for dcd in dcd_list:
            b0 = _peek_first_box_nm(
                dcd,
                tmpl_model,
                atom_indices_full,
                int(stride),
                box_nm=box_nm,
            )
            half.append(0.5 * float(np.min(b0)))
        r_max = float(min(half))
    else:
        r_max = float(r_max_nm)

    if r_max <= 0.0:
        raise ValueError("r_max_nm must be > 0")
    if float(dr_nm) <= 0.0:
        raise ValueError("dr_nm must be > 0")

    r_edges = np.arange(0.0, r_max + float(dr_nm), float(dr_nm), dtype=np.float64)
    if r_edges.size < 2:
        raise ValueError("invalid r_max/dr combination")
    r_edges[-1] = r_max
    r_nm = 0.5 * (r_edges[:-1] + r_edges[1:])

    n_bins = int(r_edges.size - 1)
    h_rep = np.zeros((n_ch, n_bins), dtype=np.float64)
    h_sum = np.zeros((n_bins,), dtype=np.float64)

    n_frames = 0
    n_pairs_per_frame = 0
    vol_sum = 0.0

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)
    eye = np.eye(n_ch, dtype=bool)
    not_diag = ~eye

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
                    raise ValueError("DCD lacks unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")
                b = box_fallback
            else:
                b = _box_lengths_nm(box_frame_nm)

            vol = float(b[0] * b[1] * b[2])
            if vol <= 0.0:
                raise ValueError("non-positive box volume")
            vol_sum += vol

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
            pos_i = xyz[idx_i, :]  # (n_ch, 3)
            pos_j = xyz[idx_j, :]  # (n_ch, 3)

            if m == "intra":
                d = pos_j - pos_i
                d = _min_image_disp_nm(d, b)
                rij = np.linalg.norm(d, axis=1)  # (n_ch,)

                n_pairs_per_frame = n_ch
                for a in range(n_ch):
                    ha, _ = np.histogram(rij[a : a + 1], bins=r_edges)
                    h_rep[a] += ha
                    h_sum += ha

            elif m == "inter":
                d = pos_j[None, :, :] - pos_i[:, None, :]  # (n_ch, n_ch, 3)
                d = d - np.rint(d / b.reshape(1, 1, 3)) * b.reshape(1, 1, 3)
                rij = np.linalg.norm(d, axis=2)  # (n_ch, n_ch)

                n_pairs_per_frame = n_ch * (n_ch - 1)
                for a in range(n_ch):
                    ra = rij[a, not_diag[a]]
                    ha, _ = np.histogram(ra, bins=r_edges)
                    h_rep[a] += ha
                    h_sum += ha

            else:  # both
                d_intra = pos_j - pos_i
                d_intra = _min_image_disp_nm(d_intra, b)
                r_intra = np.linalg.norm(d_intra, axis=1)  # (n_ch,)

                d_inter = pos_j[None, :, :] - pos_i[:, None, :]  # (n_ch, n_ch, 3)
                d_inter = d_inter - np.rint(d_inter / b.reshape(1, 1, 3)) * b.reshape(1, 1, 3)
                r_inter = np.linalg.norm(d_inter, axis=2)  # (n_ch, n_ch)

                n_pairs_per_frame = n_ch * n_ch
                for a in range(n_ch):
                    ra = np.concatenate(
                        [
                            np.asarray([r_intra[a]], dtype=np.float64),
                            r_inter[a, not_diag[a]],
                        ]
                    )
                    ha, _ = np.histogram(ra, bins=r_edges)
                    h_rep[a] += ha
                    h_sum += ha

            n_frames += 1

    if n_frames <= 0:
        raise ValueError("no frames selected")

    n_ref = n_ch
    if m == "intra":
        n_targets_per_ref = 1
    elif m == "inter":
        n_targets_per_ref = n_ch - 1
    else:
        n_targets_per_ref = n_ch

    if box_nm is not None:
        b_use = _box_lengths_nm(box_nm)
        vol_use = float(b_use[0] * b_use[1] * b_use[2])
    else:
        vol_use = float(vol_sum) / float(n_frames)

    y = _rdf_norm_from_counts(
        h_sum,
        r_edges_nm=r_edges,
        n_frames=n_frames,
        n_ref=n_ref,
        vol_nm3=vol_use,
        n_targets_per_ref=n_targets_per_ref,
        normalization=norm,
    )

    y_rep = np.empty_like(h_rep, dtype=np.float64)
    for a in range(n_ch):
        y_rep[a] = _rdf_norm_from_counts(
            h_rep[a],
            r_edges_nm=r_edges,
            n_frames=n_frames,
            n_ref=1,
            vol_nm3=vol_use,
            n_targets_per_ref=n_targets_per_ref,
            normalization=norm,
        )

    if n_ch < 2:
        y_err = np.zeros_like(y)
    else:
        y_err = np.std(y_rep, axis=0, ddof=1) / math.sqrt(float(n_ch))

    return SiteRDFResult(
        r_nm=r_nm,
        y=y,
        y_err=y_err,
        n_chains=n_ch,
        n_frames=int(n_frames),
        n_pairs_per_frame=int(n_pairs_per_frame),
        mode=m,
        normalization=norm,
        res_i=int(res_i),
        res_j=int(res_j),
        atom_name=str(atom_name).strip().upper(),
    )


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds - 3600 * h) // 60)
    s = seconds - 3600 * h - 60 * m
    if h > 0:
        return f"{h:d}:{m:02d}:{s:04.1f}"
    return f"{m:d}:{s:04.1f}"


def _progress_print(msg: str, *, stream=None) -> None:
    if stream is None:
        stream = sys.stderr
    stream.write(msg + "\n")
    stream.flush()


@dataclass(frozen=True)
class MSDResult:
    t_ns: np.ndarray
    msd_nm2: np.ndarray
    msd_stderr_nm2: np.ndarray
    msd_per_chain_nm2: np.ndarray  # (n_chains, n_lags)
    n_chains: int
    n_frames: int
    dt_ns: float
    mode: str
    resnum: Optional[int]
    atom_name: str


def _unwrap_time_series_nm(x_wrapped: np.ndarray, boxes_nm: np.ndarray) -> np.ndarray:
    x = np.asarray(x_wrapped, dtype=np.float64)
    b = np.asarray(boxes_nm, dtype=np.float64)

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x_wrapped must have shape (n_frames, n_items, 3)")
    if b.ndim != 2 or b.shape[-1] != 3:
        raise ValueError("boxes_nm must have shape (n_frames, 3)")
    if x.shape[0] != b.shape[0]:
        raise ValueError("x_wrapped and boxes_nm must have the same n_frames")

    out = np.empty_like(x)
    out[0] = x[0]
    if int(x.shape[0]) == 1:
        return out

    d = x[1:] - x[:-1]
    bt = b[1:, None, :]
    d = d - np.rint(d / bt) * bt
    out[1:] = x[0] + np.cumsum(d, axis=0)
    return out


def _autocorr_fft_multi_1d(x: np.ndarray, n_fft: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    f = np.fft.rfft(x, n=int(n_fft), axis=0)
    ac = np.fft.irfft(f * np.conjugate(f), n=int(n_fft), axis=0)
    return np.asarray(ac, dtype=np.float64)


def _msd_block_fft(r_nm: np.ndarray) -> np.ndarray:
    r = np.asarray(r_nm, dtype=np.float64)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError("r_nm must have shape (n_frames, n_chains, 3)")

    n = int(r.shape[0])
    if n < 2:
        raise ValueError("need >=2 frames to compute MSD")

    n_fft = 1 << int((2 * n - 1).bit_length())

    ac = np.zeros((n, int(r.shape[1])), dtype=np.float64)
    for d in range(3):
        ac_d = _autocorr_fft_multi_1d(r[:, :, d], n_fft)[:n]
        ac += ac_d

    r2 = np.sum(r * r, axis=2)  # (n, n_ch)
    c = np.cumsum(r2, axis=0)
    c0 = np.vstack([np.zeros((1, int(r2.shape[1])), dtype=np.float64), c])

    k = np.arange(n, dtype=np.int64)
    s0 = c0[n - k]  # sum |r(t)|^2 over t=0..n-k-1
    s1 = c0[n] - c0[k]  # sum |r(t+k)|^2 over t=0..n-k-1
    denom = (n - k).astype(np.float64).reshape(n, 1)

    msd = (s0 + s1 - 2.0 * ac) / denom
    msd[0, :] = 0.0
    return msd.T  # (n_ch, n_lags)


def _msd_fft_all(
    r_nm: np.ndarray,
    *,
    n_jobs: int,
    block_chains: int,
) -> np.ndarray:
    r = np.asarray(r_nm, dtype=np.float64)
    if r.ndim != 3 or r.shape[2] != 3:
        raise ValueError("r_nm must have shape (n_frames, n_chains, 3)")
    if int(block_chains) <= 0:
        raise ValueError("block_chains must be >= 1")

    n_frames = int(r.shape[0])
    n_ch = int(r.shape[1])
    out = np.empty((n_ch, n_frames), dtype=np.float64)

    blocks = [(i, min(i + int(block_chains), n_ch)) for i in range(0, n_ch, block_chains)]
    if n_jobs <= 1 or len(blocks) == 1:
        for s, e in blocks:
            out[s:e] = _msd_block_fft(r[:, s:e, :])
        return out

    n_workers = min(int(n_jobs), len(blocks))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_msd_block_fft, r[:, s:e, :]): (s, e) for s, e in blocks}
        for fut in as_completed(futs):
            s, e = futs[fut]
            out[s:e] = fut.result()

    return out


def _centers_wrapped_nm(
    xyz_nm: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    idx2: Optional[np.ndarray],
    masses: Optional[np.ndarray],
    center: str,
) -> np.ndarray:
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    c = str(center).strip().lower()
    if c not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    if idx2 is not None:
        x = xyz[idx2, :]  # (n_ch, n_beads, 3)
        if c == "cog" or masses is None:
            return np.mean(x, axis=1)

        w = np.asarray(masses, dtype=np.float64)[idx2]  # (n_ch, n_beads)
        wsum = np.sum(w, axis=1)
        out = np.empty((int(idx2.shape[0]), 3), dtype=np.float64)
        ok = wsum > 0.0
        if np.any(ok):
            out[ok] = np.sum(x[ok] * w[ok, :, None], axis=1) / wsum[ok, None]
        if np.any(~ok):
            out[~ok] = np.mean(x[~ok], axis=1)
        return out

    out = np.empty((len(groups), 3), dtype=np.float64)
    if c == "cog" or masses is None:
        for i, idx in enumerate(groups):
            out[i] = np.mean(xyz[np.asarray(idx, dtype=np.int64), :], axis=0)
        return out

    m = np.asarray(masses, dtype=np.float64)
    for i, idx in enumerate(groups):
        ii = np.asarray(idx, dtype=np.int64)
        sel = xyz[ii, :]
        w = m[ii]
        tot = float(np.sum(w))
        if tot <= 0.0:
            out[i] = np.mean(sel, axis=0)
        else:
            out[i] = np.sum(sel * w[:, None], axis=0) / tot
    return out


def _centers_wrapped_pbc_nm(
    xyz_nm: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    idx2: Optional[np.ndarray],
    masses: Optional[np.ndarray],
    center: str,
    box_nm: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    b = np.asarray(box_nm, dtype=np.float64).reshape(3)

    c = str(center).strip().lower()
    if c not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    if idx2 is not None:
        x = xyz[idx2, :]  # (n_ch, n_beads, 3)

        ref = x[:, 0:1, :]
        d = x - ref
        d -= np.rint(d / b.reshape(1, 1, 3)) * b.reshape(1, 1, 3)
        x = ref + d

        if c == "cog" or masses is None:
            cen = np.mean(x, axis=1)
        else:
            w = np.asarray(masses, dtype=np.float64)[idx2]
            wsum = np.sum(w, axis=1)
            out = np.empty((int(idx2.shape[0]), 3), dtype=np.float64)

            ok = wsum > 0.0
            if np.any(ok):
                out[ok] = np.sum(x[ok] * w[ok, :, None], axis=1) / wsum[ok, None]
            if np.any(~ok):
                out[~ok] = np.mean(x[~ok], axis=1)

            cen = out

        cen -= np.floor(cen / b.reshape(1, 3)) * b.reshape(1, 3)
        return cen

    # fallback: uses existing per-group loop implementation
    return group_centers_nm(
        xyz,
        groups,
        masses=masses,
        box_nm=b,
        center=c,
        unwrap=True,
        wrap=True,
    )


def msd_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    mode: str = "com",  # "com" | "cog" | "residue"
    resnum: Optional[int] = None,
    atom_name: str = "CA",
    dt_ns: float,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    n_jobs: int = 0,
    block_chains: int = 16,
    unwrap_within_frame: bool = False,
) -> MSDResult:
    """
    Fast MSD for systems where each chain is wrapped as a whole.

    Algorithm
    ---------
    1) Compute wrapped chain centers (COG/COM) or a per-chain residue coordinate.
    2) Time-unwrap the (n_frames, n_chains, 3) series with minimum-image deltas.
       This lets coordinates "explode" across PBC.
    3) Compute per-chain MSD via FFT autocorrelation, in chain blocks and optionally
       in parallel threads.

    Notes
    -----
    - Orthorhombic boxes only (DCD unit-cell lengths).
    - If you pass multiple DCDs, frames are concatenated and unwrapped continuously.
      If your DCDs are independent replicates, call this per-file instead.
    """
    if float(dt_ns) <= 0.0:
        raise ValueError("dt_ns must be > 0")
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    m = str(mode).strip().lower()
    if m not in {"com", "cog", "residue"}:
        raise ValueError("mode must be 'com', 'cog', or 'residue'")
    if m == "residue" and resnum is None:
        raise ValueError("resnum is required for mode='residue'")

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    masses_sel = None
    groups_sel: list[np.ndarray] = []
    idx2: Optional[np.ndarray] = None
    atom_indices_full: list[int] = []
    center = "cog"

    if m in {"com", "cog"}:
        center = "com" if m == "com" else "cog"
        groups_full = solute_groups(tmpl, group_spec="protein")
        if not groups_full:
            raise ValueError("solute_groups returned no groups")

        atom_set: set[int] = set()
        for g in groups_full:
            atom_set.update(int(i) for i in g.tolist())
        atom_indices_full = sorted(atom_set)

        idx_map = {old: new for new, old in enumerate(atom_indices_full)}
        groups_sel = [
            np.asarray([idx_map[int(i)] for i in g.tolist()], dtype=np.int64) for g in groups_full
        ]
        idx2 = _groups_to_rect_index(groups_sel)

        if m == "com":
            masses_all = atom_masses(tmpl)
            masses_sel = np.asarray(masses_all[atom_indices_full], dtype=np.float64)

    else:
        _, idx_full = _site_atom_indices_by_chain(
            tmpl,
            resnum=int(resnum),
            atom_name=str(atom_name),
        )
        if int(idx_full.size) < 1:
            raise ValueError("no chains with requested residue/atom present")
        atom_indices_full = [int(i) for i in idx_full.tolist()]

    n_ch = int(len(groups_sel)) if m in {"com", "cog"} else int(len(atom_indices_full))
    if n_ch < 1:
        raise ValueError("need >=1 chain for MSD")
    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    centers_wr: list[np.ndarray] = []
    boxes: list[np.ndarray] = []

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
                b = box_fallback
            else:
                b = _box_lengths_nm(box_frame_nm)

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)

            if m in {"com", "cog"}:
                if unwrap_within_frame:
                    cen = _centers_wrapped_pbc_nm(
                        xyz,
                        groups_sel,
                        idx2=idx2,
                        masses=masses_sel,
                        center=center,
                        box_nm=b,
                    )
                else:
                    cen = _centers_wrapped_nm(
                        xyz,
                        groups_sel,
                        idx2=idx2,
                        masses=masses_sel,
                        center=center,
                    )

                if cen.shape != (n_ch, 3):
                    raise ValueError("unexpected centers shape")
                centers_wr.append(cen)
            else:
                if xyz.shape != (n_ch, 3):
                    raise ValueError("unexpected residue coords shape")
                centers_wr.append(xyz)

            boxes.append(np.asarray(b, dtype=np.float64))

    if not centers_wr:
        raise ValueError("no frames selected")

    x_wr = np.stack(centers_wr, axis=0)  # (n_frames, n_ch, 3)
    b_all = np.stack(boxes, axis=0)  # (n_frames, 3)
    x_un = _unwrap_time_series_nm(x_wr, b_all)

    n_frames = int(x_un.shape[0])
    t_ns = np.arange(n_frames, dtype=np.float64) * float(dt_ns)

    n_jobs_i = int(n_jobs)
    if n_jobs_i <= 0:
        n_jobs_use = os.cpu_count() or 1
    else:
        n_jobs_use = n_jobs_i

    msd_ch = _msd_fft_all(x_un, n_jobs=n_jobs_use, block_chains=int(block_chains))
    msd = np.mean(msd_ch, axis=0)

    if n_ch < 2:
        msd_stderr = np.zeros_like(msd, dtype=np.float64)
    else:
        msd_stderr = np.std(msd_ch, axis=0, ddof=1) / math.sqrt(float(n_ch))

    return MSDResult(
        t_ns=t_ns,
        msd_nm2=msd,
        msd_stderr_nm2=msd_stderr,
        msd_per_chain_nm2=msd_ch,
        n_chains=n_ch,
        n_frames=n_frames,
        dt_ns=float(dt_ns),
        mode=m,
        resnum=None if resnum is None else int(resnum),
        atom_name=str(atom_name).strip().upper(),
    )


@dataclass(frozen=True)
class MSDFitResult:
    fit_tmin_ns: float
    fit_tmax_ns: float
    slope_nm2_per_ns: float
    intercept_nm2: float
    d_nm2_per_ns: float
    d_stderr_nm2_per_ns: float
    per_chain_d_nm2_per_ns: np.ndarray


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("need >=2 points to fit")
    a = np.vstack([x, np.ones_like(x)]).T
    sol, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    return float(sol[0]), float(sol[1])


def fit_msd_linear(
    msd: MSDResult,
    *,
    fit_tmin_ns: float,
    fit_tmax_ns: float,
    dims: int = 3,
    use_chain_sem: bool = True,
) -> MSDFitResult:
    """
    Fit MSD(t) = slope * t + intercept on [fit_tmin_ns, fit_tmax_ns].

    Diffusion: D = slope / (2*dims). For 3D, D = slope/6.

    Error:
      - use_chain_sem=True: fit each chain, report mean D and SEM across chains.
      - else: no uncertainty estimate beyond 0.0 (kept simple).
    """
    if dims <= 0:
        raise ValueError("dims must be >= 1")
    t = msd.t_ns
    y = msd.msd_nm2
    tmin = float(fit_tmin_ns)
    tmax = float(fit_tmax_ns)
    if tmax <= tmin:
        raise ValueError("fit_tmax_ns must be > fit_tmin_ns")

    sel = (t >= tmin) & (t <= tmax)
    if int(np.sum(sel)) < 2:
        raise ValueError("fit window selects <2 points")

    slope, intercept = _linear_fit(t[sel], y[sel])
    d = slope / (2.0 * float(dims))

    d_stderr = 0.0
    d_ch = np.empty((msd.n_chains,), dtype=np.float64)

    if use_chain_sem:
        for a in range(msd.n_chains):
            s, _ = _linear_fit(t[sel], msd.msd_per_chain_nm2[a, sel])
            d_ch[a] = s / (2.0 * float(dims))
        d_stderr = float(np.std(d_ch, ddof=1)) / math.sqrt(float(msd.n_chains))
    else:
        d_ch[:] = np.nan

    return MSDFitResult(
        fit_tmin_ns=tmin,
        fit_tmax_ns=tmax,
        slope_nm2_per_ns=float(slope),
        intercept_nm2=float(intercept),
        d_nm2_per_ns=float(d),
        d_stderr_nm2_per_ns=float(d_stderr),
        per_chain_d_nm2_per_ns=d_ch,
    )


@dataclass(frozen=True)
class RgResult:
    rg_per_chain_nm: np.ndarray  # (n_chains, n_frames)
    rg_mean_nm: np.ndarray  # (n_frames,)
    rg_stderr_nm: np.ndarray  # (n_frames,)
    n_chains: int
    n_frames: int
    mode: str  # "cog" or "com"
    # Labels correspond one-to-one with rows of rg_per_chain_nm.  The default
    # keeps older cached/pickled results usable after this field is added; such
    # older results expose an empty label tuple.
    chain_labels: tuple[str, ...] = ()


def _group_chain_labels(
    template_model: Any,
    groups: Sequence[np.ndarray],
) -> tuple[str, ...]:
    """Return physical-chain labels for selection groups.

    Each returned label corresponds to one group, preserving group order. A
    normal per-chain selection produces one chain key such as ``P001``. If a
    group intentionally spans several physical chains, their labels are joined
    with ``+``.
    """
    n_atoms = int(len(template_model.atoms))
    atom_index = {id(atom): index for index, atom in enumerate(template_model.atoms)}
    atom_to_chain = np.full(n_atoms, -1, dtype=np.int64)
    physical_chain_labels: list[str] = []

    for chain_index, (key, chain) in enumerate(template_model.chain.items()):
        label = str(key).strip()
        if not label:
            label = str(getattr(chain, "seg_id", "") or "").strip()
        if not label:
            label = str(getattr(chain, "chain_id", "") or "").strip()
        if not label:
            label = str(chain_index)
        physical_chain_labels.append(label)

        for residue in chain.residues:
            for atom in residue.atoms:
                index = atom_index.get(id(atom))
                if index is not None:
                    atom_to_chain[int(index)] = int(chain_index)

    labels: list[str] = []
    for group_index, group in enumerate(groups):
        indices = np.asarray(group, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            labels.append(f"group {group_index}")
            continue
        if np.any(indices < 0) or np.any(indices >= n_atoms):
            raise IndexError(f"Rg group {group_index} contains an out-of-range atom index")

        touched = {
            int(chain_index)
            for chain_index in atom_to_chain[indices].tolist()
            if int(chain_index) >= 0
        }
        if not touched:
            labels.append(f"group {group_index}")
            continue

        labels.append("+".join(physical_chain_labels[index] for index in sorted(touched)))

    return tuple(labels)


def _selection_to_groups(
    tmpl: Any,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]],
) -> list[np.ndarray]:
    if isinstance(selection, str):
        groups_raw = StructureSelector(selection).atom_lists(tmpl)
    elif selection and all(isinstance(x, str) for x in selection):
        groups_raw = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_raw = [[int(i) for i in g] for g in selection]

    return [np.asarray(g, dtype=np.int64) for g in groups_raw if len(g) > 0]


def _groups_to_rect_index(groups_sel: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if not groups_sel:
        return None
    n0 = int(groups_sel[0].size)
    if n0 <= 0:
        return None
    for g in groups_sel[1:]:
        if int(g.size) != n0:
            return None
    idx2 = np.empty((len(groups_sel), n0), dtype=np.int64)
    for i, g in enumerate(groups_sel):
        idx2[i, :] = np.asarray(g, dtype=np.int64)
    return idx2


def rg_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein",
    mode: str = "cog",  # "cog" | "com"
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> RgResult:
    """
    Per-group Rg time series (nm) using coordinates exactly as stored
    in the trajectory.

    Group definition
    ----------------
    selection may be:
      - str:
          Passed to StructureSelector. Examples:
            "protein"     -> one group per chain
            "protein.CA"  -> one C-alpha group per chain
      - Sequence[str]:
          One selector per output group. Examples:
            ["SEG1.CA", "SEG2.CA"]
      - Sequence[Sequence[int]]:
          Explicit atom-index groups in template atom indexing.

    Notes
    -----
    - No PBC/min-image unwrapping is applied within a group.
    - This assumes each selected group is already whole within each frame.
    - box_nm is accepted for API compatibility but is not used here.

    Performance
    -----------
    - Uses a vectorized fast path when all groups have the same number of atoms.
    - Falls back to a per-group loop for variable-length groups.
    """
    _ = box_nm

    m = str(mode).strip().lower()
    if m not in {"cog", "com"}:
        raise ValueError("mode must be 'cog' or 'com'")
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

    groups_full = _selection_to_groups(tmpl, selection)
    if not groups_full:
        raise ValueError("selection produced no groups")

    chain_labels = _group_chain_labels(tmpl_model, groups_full)
    n_groups = int(len(groups_full))
    masses_all = atom_masses(tmpl) if m == "com" else None

    atom_set: set[int] = set()
    for g in groups_full:
        atom_set.update(int(i) for i in g.tolist())
    atom_indices_full = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices_full)}
    groups_sel = [
        np.asarray([idx_map[int(i)] for i in g.tolist()], dtype=np.int64) for g in groups_full
    ]
    idx2 = _groups_to_rect_index(groups_sel)

    masses_sel = None
    if masses_all is not None:
        masses_sel = np.asarray(masses_all[atom_indices_full], dtype=np.float64)

    rg_frames: list[np.ndarray] = []

    for dcd in dcd_list:
        for fi, (xyz_sel_nm, _box_frame_nm) in enumerate(
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

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)

            if idx2 is not None:
                x = xyz[idx2, :]  # (n_groups, n_atoms_per_group, 3)

                if masses_sel is None:
                    c = np.mean(x, axis=1)
                    d2 = np.sum((x - c[:, None, :]) ** 2, axis=2)
                    rg2 = np.mean(d2, axis=1)
                    rg_ch = np.sqrt(np.maximum(rg2, 0.0))
                else:
                    w = masses_sel[idx2]
                    wsum = np.sum(w, axis=1)
                    ok = wsum > 0.0

                    c = np.empty((n_groups, 3), dtype=np.float64)
                    if np.any(ok):
                        c[ok] = (
                            np.sum(
                                x[ok] * w[ok, :, None],
                                axis=1,
                            )
                            / wsum[ok, None]
                        )
                    if np.any(~ok):
                        c[~ok] = np.mean(x[~ok], axis=1)

                    d2 = np.sum((x - c[:, None, :]) ** 2, axis=2)
                    rg2 = np.empty((n_groups,), dtype=np.float64)
                    if np.any(ok):
                        rg2[ok] = np.sum(w[ok] * d2[ok], axis=1) / wsum[ok]
                    if np.any(~ok):
                        rg2[~ok] = np.mean(d2[~ok], axis=1)
                    rg_ch = np.sqrt(np.maximum(rg2, 0.0))

            else:
                rg_ch = np.empty((n_groups,), dtype=np.float64)
                for gi in range(n_groups):
                    idx = groups_sel[gi]
                    if idx.size == 0:
                        rg_ch[gi] = np.nan
                        continue

                    x = xyz[idx, :]

                    if masses_sel is None:
                        c = np.mean(x, axis=0)
                        d2 = np.sum((x - c) ** 2, axis=1)
                        rg2 = float(np.mean(d2))
                        rg_ch[gi] = math.sqrt(max(rg2, 0.0))
                    else:
                        w = masses_sel[idx]
                        tot = float(np.sum(w))
                        if tot <= 0.0:
                            c = np.mean(x, axis=0)
                            d2 = np.sum((x - c) ** 2, axis=1)
                            rg2 = float(np.mean(d2))
                            rg_ch[gi] = math.sqrt(max(rg2, 0.0))
                        else:
                            c = np.sum(x * w[:, None], axis=0) / tot
                            d2 = np.sum((x - c) ** 2, axis=1)
                            rg2 = float(np.sum(w * d2) / tot)
                            rg_ch[gi] = math.sqrt(max(rg2, 0.0))

            rg_frames.append(np.asarray(rg_ch, dtype=np.float64))

    if not rg_frames:
        raise ValueError("no frames selected")

    rg_pf = np.stack(rg_frames, axis=1)
    rg_mean = np.nanmean(rg_pf, axis=0)
    if n_groups < 2:
        rg_stderr = np.zeros_like(rg_mean)
    else:
        rg_stderr = np.nanstd(rg_pf, axis=0, ddof=1) / math.sqrt(float(n_groups))

    return RgResult(
        rg_per_chain_nm=rg_pf,
        rg_mean_nm=rg_mean,
        rg_stderr_nm=rg_stderr,
        n_chains=n_groups,
        n_frames=int(rg_pf.shape[1]),
        mode=m,
        chain_labels=chain_labels,
    )


@dataclass(frozen=True)
class IntrachainDistanceResult:
    """Intrachain atom-pair distance time series.

    Array conventions match ``RgResult`` and ``ChainContactResult``:
    the first axis of ``distance_per_chain_nm`` identifies a chain and the
    second axis identifies trajectory frames.
    """

    distance_per_chain_nm: np.ndarray  # (n_chains, n_frames)
    distance_mean_nm: np.ndarray  # (n_frames,)
    distance_stderr_nm: np.ndarray  # (n_frames,)
    chain_labels: tuple[str, ...]
    atom_labels: tuple[tuple[str, str], ...]  # one atom-label pair per chain
    atom_indices: np.ndarray  # (n_chains, 2), template atom indices
    n_chains: int
    n_frames: int
    selection: Any
    pbc: bool


def _chain_indices_from_selection(
    tmpl: Any,
    chains: Union[str, Sequence[str]],
) -> tuple[list[int], tuple[str, ...], np.ndarray]:
    """Resolve a chain set using StructureSelector-compatible selections.

    ``chains`` may be one selector or a sequence of selectors. A selector may
    identify one or several chains. Returned chains are de-duplicated while
    preserving selector order and topology order within each selector.
    """
    model = tmpl.model if hasattr(tmpl, "model") else tmpl
    n_atoms = int(len(model.atoms))

    atom_index = {id(atom): i for i, atom in enumerate(model.atoms)}
    atom_to_chain = np.full(n_atoms, -1, dtype=np.int64)
    chain_labels: list[str] = []

    for ci, (key, chain) in enumerate(model.chain.items()):
        chain_labels.append(str(key))
        for residue in chain.residues:
            for atom in residue.atoms:
                ai = atom_index.get(id(atom))
                if ai is not None:
                    atom_to_chain[int(ai)] = int(ci)

    if isinstance(chains, str):
        chain_specs: list[Any] = [chains]
    elif isinstance(chains, Sequence):
        chain_specs = list(chains)
    else:
        raise TypeError("chains must be a selection string or a sequence of selections")

    if not chain_specs:
        raise ValueError("chains is empty")

    selected: list[int] = []
    seen: set[int] = set()

    for spec in chain_specs:
        groups = _selection_to_groups(tmpl, spec)
        if not groups:
            raise ValueError(f"chain selection {spec!r} produced no atoms")

        touched: set[int] = set()
        for group in groups:
            ci_values = np.unique(atom_to_chain[np.asarray(group, dtype=np.int64)])
            touched.update(int(ci) for ci in ci_values.tolist() if int(ci) >= 0)

        if not touched:
            raise ValueError(f"chain selection {spec!r} did not resolve to a physical chain")

        for ci in sorted(touched):
            if ci not in seen:
                selected.append(ci)
                seen.add(ci)

    labels = tuple(chain_labels[ci] for ci in selected)
    return selected, labels, atom_to_chain


def _atom_pair_label(model: Any, atom_index: int) -> str:
    atom = model.atoms[int(atom_index)]
    resname = (getattr(atom, "resname", "") or "").strip()
    resnum = int(getattr(atom, "resnum", 0))
    name = (getattr(atom, "name", "") or "").strip()
    return f"{resname}{resnum}.{name}"


def intrachain_distances_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    chains: Union[str, Sequence[str]] = "protein",
    selection: Union[str, Sequence[str]],
    pbc: bool = True,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> IntrachainDistanceResult:
    """Calculate the same intrachain atom-pair distance for selected chains.

    Parameters
    ----------
    pdb_file, dcd_files
        Template PDB and one or more DCD trajectories.

    chains
        StructureSelector-compatible selection identifying the chains to
        analyze. It may be one selector or a sequence of selectors. Examples::

            chains="protein"
            chains="A:B:C"
            chains=["A", "B", "C"]

    selection
        StructureSelector-compatible selection that must select exactly two
        atoms in every requested chain. The same atom definition is applied to
        all chains. Examples::

            selection="10.CA,90.CA"
            selection=["10.CA", "90.CA"]

        The two forms above are equivalent. Selection order is not important
        because only the scalar distance is returned.

    pbc
        If True, calculate minimum-image distances using an orthorhombic box.
        If the DCD lacks unit-cell lengths, provide ``box_nm``.

    box_nm
        Fallback box lengths ``(Lx, Ly, Lz)`` in nm.

    stride, chunk, frame_start, frame_stop
        Same conventions as ``rg_from_dcd``.

    Returns
    -------
    IntrachainDistanceResult
        ``distance_per_chain_nm`` has shape ``(n_chains, n_frames)``.
    """
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

    chain_indices, chain_labels, atom_to_chain = _chain_indices_from_selection(
        tmpl,
        chains,
    )
    if not chain_indices:
        raise ValueError("chains produced no physical chains")

    pair_groups_full = _selection_to_groups(tmpl, selection)
    if not pair_groups_full:
        raise ValueError("selection produced no atoms")

    pair_atom_set: set[int] = set()
    for group in pair_groups_full:
        pair_atom_set.update(int(i) for i in np.asarray(group, dtype=np.int64).tolist())

    pair_indices_full = np.empty((len(chain_indices), 2), dtype=np.int64)
    atom_labels: list[tuple[str, str]] = []

    for out_i, chain_i in enumerate(chain_indices):
        selected_atoms = sorted(
            ai for ai in pair_atom_set if int(atom_to_chain[int(ai)]) == int(chain_i)
        )

        if len(selected_atoms) != 2:
            label = chain_labels[out_i]
            descriptions = [_atom_pair_label(tmpl_model, ai) for ai in selected_atoms]
            raise ValueError(
                f"selection must resolve to exactly two atoms in chain {label!r}; "
                f"it selected {len(selected_atoms)}: {descriptions}"
            )

        pair_indices_full[out_i, :] = selected_atoms
        atom_labels.append(
            (
                _atom_pair_label(tmpl_model, selected_atoms[0]),
                _atom_pair_label(tmpl_model, selected_atoms[1]),
            )
        )

    # Read only the atoms that participate in the requested distances.
    atom_indices_full = sorted(set(pair_indices_full.reshape(-1).tolist()))
    idx_map = {old: new for new, old in enumerate(atom_indices_full)}
    pair_indices_sel = np.asarray(
        [[idx_map[int(a)], idx_map[int(b)]] for a, b in pair_indices_full.tolist()],
        dtype=np.int64,
    )

    box_fallback = None
    if bool(pbc) and box_nm is not None:
        box_fallback = _box_lengths_nm(box_nm)

    distance_frames: list[np.ndarray] = []

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

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
            pair_xyz = xyz[pair_indices_sel, :]  # (n_chains, 2, 3)
            displacement = pair_xyz[:, 1, :] - pair_xyz[:, 0, :]

            if bool(pbc):
                if box_frame_nm is None:
                    if box_fallback is None:
                        raise ValueError(
                            "DCD lacks unit-cell lengths; pass box_nm=(Lx,Ly,Lz) "
                            "in nm or set pbc=False"
                        )
                    box = box_fallback
                else:
                    box = _box_lengths_nm(box_frame_nm)

                displacement -= np.rint(displacement / box.reshape(1, 3)) * box.reshape(1, 3)

            distances = np.linalg.norm(displacement, axis=1)
            distance_frames.append(np.asarray(distances, dtype=np.float64))

    if not distance_frames:
        raise ValueError("no frames selected")

    distance_per_chain = np.stack(distance_frames, axis=1)
    distance_mean = np.nanmean(distance_per_chain, axis=0)

    n_chains = int(distance_per_chain.shape[0])
    if n_chains < 2:
        distance_stderr = np.zeros_like(distance_mean)
    else:
        distance_stderr = np.nanstd(distance_per_chain, axis=0, ddof=1) / math.sqrt(float(n_chains))

    return IntrachainDistanceResult(
        distance_per_chain_nm=distance_per_chain,
        distance_mean_nm=distance_mean,
        distance_stderr_nm=distance_stderr,
        chain_labels=chain_labels,
        atom_labels=tuple(atom_labels),
        atom_indices=pair_indices_full,
        n_chains=n_chains,
        n_frames=int(distance_per_chain.shape[1]),
        selection=selection,
        pbc=bool(pbc),
    )


@dataclass(frozen=True)
class InterchainDistanceResult:
    """Distances from one reference-chain atom to one atom in target chains.

    The first axis of ``distance_per_chain_nm`` identifies a target chain and
    the second axis identifies trajectory frames. If the reference chain is
    included among the target chains and ``exclude_reference=False``, its row
    contains the corresponding intrachain distance.
    """

    distance_per_chain_nm: np.ndarray  # (n_target_chains, n_frames)
    distance_mean_nm: np.ndarray  # (n_frames,)
    distance_stderr_nm: np.ndarray  # (n_frames,)
    chain_labels: tuple[str, ...]  # target-chain labels
    reference_chain_label: str
    reference_atom_label: str
    target_atom_labels: tuple[str, ...]
    reference_atom_index: int  # template atom index
    target_atom_indices: np.ndarray  # (n_target_chains,), template atom indices
    n_chains: int  # number of retained target chains
    n_frames: int
    selection: Any
    exclude_reference: bool
    pbc: bool


def _ordered_atom_pair_selection_specs(
    selection: Union[str, Sequence[str]],
) -> tuple[str, str]:
    """Normalize an ordered two-atom selection into two selector strings.

    Accepted forms include::

        ["39.CE2", "69.SG"]
        ("39.CE2", "69.SG")
        "39.CE2,69.SG"
        "39.CE2;69.SG"
        "39.CE2_69.SG"

    The order matters: the first selector defines the reference-chain atom and
    the second selector defines the target-chain atom.
    """
    if isinstance(selection, str):
        raw = selection.strip()
        if not raw:
            raise ValueError("selection is empty")

        # StructureSelector treats ';' and '_' as group separators. Comma is
        # also accepted here for consistency with intrachain_distances_from_dcd.
        normalized = raw.replace(";", ",").replace("_", ",")
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
    elif isinstance(selection, Sequence):
        parts = [str(part).strip() for part in selection]
    else:
        raise TypeError("selection must be a string or a sequence of two strings")

    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(
            "selection must contain exactly two ordered atom selectors, for example "
            "['39.CE2', '69.SG'] or '39.CE2,69.SG'"
        )

    return parts[0], parts[1]


@dataclass(frozen=True)
class MinimumContactDistanceResult:
    """Per-chain minimum selected atom-pair distance time series.

    For every selected reference chain ``i``, the first atom selector supplies
    one reference atom and the second atom selector supplies one target atom in
    every selected chain ``j``.  The result for reference chain ``i`` is the
    minimum permitted distance ``min_j d(A_i, B_j)`` in each trajectory frame.

    Array conventions match :class:`IntrachainDistanceResult`:
    ``distance_per_chain_nm`` has shape ``(n_chains, n_frames)`` and row ``i``
    corresponds to ``chain_labels[i]``.  ``minimum_target_chain_index[i, t]``
    records which target chain supplied the minimum for reference chain ``i``
    in frame ``t``.
    """

    distance_per_chain_nm: np.ndarray  # (n_chains, n_frames)
    distance_mean_nm: np.ndarray  # (n_frames,)
    distance_stderr_nm: np.ndarray  # (n_frames,)
    minimum_target_chain_index: np.ndarray  # (n_chains, n_frames)
    chain_labels: tuple[str, ...]
    reference_atom_labels: tuple[str, ...]
    target_atom_labels: tuple[str, ...]
    reference_atom_indices: np.ndarray  # (n_chains,), template atom indices
    target_atom_indices: np.ndarray  # (n_chains,), template atom indices
    n_chains: int
    n_frames: int
    n_pairs_per_reference: int
    pair_mode: str  # "both" | "intra" | "inter"
    selection: Any
    pbc: bool

    @property
    def minimum_distance_nm(self) -> np.ndarray:
        """Alias for :attr:`distance_per_chain_nm`."""
        return self.distance_per_chain_nm

    @property
    def chain_labels_all(self) -> tuple[str, ...]:
        """Backward-compatible alias for :attr:`chain_labels`."""
        return self.chain_labels

    @property
    def n_pairs(self) -> int:
        """Total number of directed chain pairs considered per frame."""
        return int(self.n_chains * self.n_pairs_per_reference)

    @property
    def minimum_reference_chain_index(self) -> np.ndarray:
        """Reference-chain indices aligned with the per-chain result array."""
        return np.broadcast_to(
            np.arange(self.n_chains, dtype=np.int64).reshape(-1, 1),
            self.minimum_target_chain_index.shape,
        )

    @property
    def minimum_target_chain_labels(self) -> tuple[tuple[str, ...], ...]:
        """Target-chain labels for each reference-chain row and frame."""
        return tuple(
            tuple(self.chain_labels[int(index)] for index in row)
            for row in self.minimum_target_chain_index
        )


def _normalize_minimum_contact_option(option: Optional[str]) -> str:
    """Normalize notebook-facing minimum-contact pair-selection options."""
    if option is None:
        return "both"
    if not isinstance(option, str):
        raise TypeError("option must be a string or None")

    key = option.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "both",
        "all": "both",
        "both": "both",
        "intra": "intra",
        "intraonly": "intra",
        "intra_only": "intra",
        "intrachain": "intra",
        "inter": "inter",
        "interonly": "inter",
        "inter_only": "inter",
        "interchain": "inter",
    }
    if key not in aliases:
        raise ValueError("option must be None/'both', 'intraonly', or 'interonly'")
    return aliases[key]


def _minimum_contact_per_reference(
    distance_matrix_nm: np.ndarray,
    *,
    pair_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one chain-by-chain distance matrix independently by row.

    Row ``i`` contains distances from the first selected atom in reference
    chain ``i`` to the second selected atom in every target chain ``j``.
    Returns the minimum distance and the minimizing target-chain index for each
    reference-chain row.
    """
    distances = np.asarray(distance_matrix_nm, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix_nm must be a square 2D array")

    n_chains = int(distances.shape[0])
    if n_chains < 1:
        raise ValueError("distance_matrix_nm is empty")

    mode = str(pair_mode).strip().lower()
    reference_indices = np.arange(n_chains, dtype=np.int64)

    if mode == "intra":
        return np.diag(distances).copy(), reference_indices

    if mode == "both":
        permitted = distances
    elif mode == "inter":
        if n_chains < 2:
            raise ValueError("interchain minimum requires at least two chains")
        permitted = distances.copy()
        np.fill_diagonal(permitted, np.inf)
    else:
        raise ValueError("pair_mode must be 'both', 'intra', or 'inter'")

    target_indices = np.argmin(permitted, axis=1).astype(np.int64, copy=False)
    minimum_distances = permitted[reference_indices, target_indices]
    if np.any(~np.isfinite(minimum_distances)):
        raise ValueError("one or more reference chains have no permitted target")
    return minimum_distances, target_indices


def minimum_contact_distances_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    chains: Union[str, Sequence[str]] = "protein",
    selection: Union[str, Sequence[str]],
    option: Optional[str] = None,
    pbc: bool = True,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> MinimumContactDistanceResult:
    """Return a per-chain minimum selected atom-pair distance in every frame.

    For ``N`` selected chains, the first ordered atom selector supplies one
    reference atom ``A_i`` in each chain and the second supplies one target atom
    ``B_j`` in each chain.  For every reference chain ``i`` and frame, this
    routine stores the minimum permitted value from the row
    ``d(A_i, B_j)``:

    - ``option=None`` or ``"both"``: minimize over all ``j``, including ``j=i``;
    - ``option="intraonly"``: retain only ``j=i``.  This is identical to
      :func:`intrachain_distances_from_dcd` for the same selections;
    - ``option="interonly"``: minimize over ``j != i``.

    Thus the output contains one time series per selected reference chain, not
    one global minimum over the complete ``N x N`` matrix.
    """
    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    pair_mode = _normalize_minimum_contact_option(option)
    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    chain_indices, chain_labels, atom_to_chain = _chain_indices_from_selection(
        tmpl,
        chains,
    )
    if not chain_indices:
        raise ValueError("chains produced no physical chains")
    if pair_mode == "inter" and len(chain_indices) < 2:
        raise ValueError("option='interonly' requires at least two selected chains")

    reference_atom_spec, target_atom_spec = _ordered_atom_pair_selection_specs(selection)

    def atoms_for_spec(spec: str, role: str) -> tuple[np.ndarray, tuple[str, ...]]:
        groups = _selection_to_groups(tmpl, spec)
        if not groups:
            raise ValueError(f"{role} atom selection {spec!r} produced no atoms")

        selected_atom_set: set[int] = set()
        for group in groups:
            selected_atom_set.update(
                int(index) for index in np.asarray(group, dtype=np.int64).tolist()
            )

        atom_indices = np.empty(len(chain_indices), dtype=np.int64)
        labels: list[str] = []
        for output_index, (chain_index, chain_label) in enumerate(zip(chain_indices, chain_labels)):
            matches = sorted(
                atom_index
                for atom_index in selected_atom_set
                if int(atom_to_chain[int(atom_index)]) == int(chain_index)
            )
            if len(matches) != 1:
                descriptions = [_atom_pair_label(tmpl_model, atom_index) for atom_index in matches]
                raise ValueError(
                    f"{role} atom selection {spec!r} must resolve to exactly "
                    f"one atom in chain {chain_label!r}; it selected "
                    f"{len(matches)}: {descriptions}"
                )
            atom_index = int(matches[0])
            atom_indices[output_index] = atom_index
            labels.append(_atom_pair_label(tmpl_model, atom_index))

        return atom_indices, tuple(labels)

    reference_atoms_full, reference_atom_labels = atoms_for_spec(
        reference_atom_spec,
        "reference",
    )
    target_atoms_full, target_atom_labels = atoms_for_spec(
        target_atom_spec,
        "target",
    )

    atom_indices_full = sorted(set(reference_atoms_full.tolist()) | set(target_atoms_full.tolist()))
    index_map = {old: new for new, old in enumerate(atom_indices_full)}
    reference_atoms_sel = np.asarray(
        [index_map[int(atom_index)] for atom_index in reference_atoms_full],
        dtype=np.int64,
    )
    target_atoms_sel = np.asarray(
        [index_map[int(atom_index)] for atom_index in target_atoms_full],
        dtype=np.int64,
    )

    n_chains = int(len(chain_indices))
    n_pairs_per_reference = (
        1 if pair_mode == "intra" else n_chains if pair_mode == "both" else n_chains - 1
    )

    box_fallback = None
    if bool(pbc) and box_nm is not None:
        box_fallback = _box_lengths_nm(box_nm)

    distance_frames: list[np.ndarray] = []
    minimum_target_frames: list[np.ndarray] = []

    for dcd in dcd_list:
        for frame_index, (xyz_sel_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                tmpl_model,
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
            reference_positions = xyz[reference_atoms_sel, :]
            target_positions = xyz[target_atoms_sel, :]
            displacement = target_positions[None, :, :] - reference_positions[:, None, :]

            if bool(pbc):
                if box_frame_nm is None:
                    if box_fallback is None:
                        raise ValueError(
                            "DCD lacks unit-cell lengths; pass box_nm=(Lx,Ly,Lz) "
                            "in nm or set pbc=False"
                        )
                    box = box_fallback
                else:
                    box = _box_lengths_nm(box_frame_nm)

                displacement -= np.rint(displacement / box.reshape(1, 1, 3)) * box.reshape(1, 1, 3)

            distance_matrix = np.linalg.norm(displacement, axis=2)
            minimum_distances, minimum_targets = _minimum_contact_per_reference(
                distance_matrix,
                pair_mode=pair_mode,
            )
            distance_frames.append(minimum_distances)
            minimum_target_frames.append(minimum_targets)

    if not distance_frames:
        raise ValueError("no frames selected")

    distance_per_chain = np.stack(distance_frames, axis=1)
    minimum_target_array = np.stack(minimum_target_frames, axis=1)
    distance_mean = np.nanmean(distance_per_chain, axis=0)

    if n_chains < 2:
        distance_stderr = np.zeros_like(distance_mean)
    else:
        distance_stderr = np.nanstd(
            distance_per_chain,
            axis=0,
            ddof=1,
        ) / math.sqrt(float(n_chains))

    return MinimumContactDistanceResult(
        distance_per_chain_nm=distance_per_chain,
        distance_mean_nm=distance_mean,
        distance_stderr_nm=distance_stderr,
        minimum_target_chain_index=minimum_target_array,
        chain_labels=chain_labels,
        reference_atom_labels=reference_atom_labels,
        target_atom_labels=target_atom_labels,
        reference_atom_indices=reference_atoms_full,
        target_atom_indices=target_atoms_full,
        n_chains=n_chains,
        n_frames=int(distance_per_chain.shape[1]),
        n_pairs_per_reference=int(n_pairs_per_reference),
        pair_mode=pair_mode,
        selection=selection,
        pbc=bool(pbc),
    )


def interchain_distances_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    reference_chain: Union[str, Sequence[str]],
    target_chains: Union[str, Sequence[str]] = "protein",
    selection: Union[str, Sequence[str]],
    exclude_reference: bool = True,
    pbc: bool = True,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> InterchainDistanceResult:
    """Calculate distances from one reference-chain atom to target-chain atoms.

    The ordered atom pair is the same for every chain:

    - the first atom selector is evaluated in ``reference_chain``;
    - the second atom selector is evaluated in every chain selected by
      ``target_chains``.

    Parameters
    ----------
    pdb_file, dcd_files
        Template PDB and one or more DCD trajectories.

    reference_chain
        StructureSelector-compatible selection that must resolve to exactly one
        physical chain. Examples::

            reference_chain="A"
            reference_chain=["A"]

    target_chains
        StructureSelector-compatible selection identifying one or more target
        chains. Examples::

            target_chains="protein"
            target_chains="A:B:C"
            target_chains=["A", "B", "C"]

    selection
        Exactly two ordered StructureSelector atom selections. The first defines
        the reference atom and the second defines the target atom. Examples::

            selection=["39.CE2", "69.SG"]
            selection="39.CE2,69.SG"

    exclude_reference
        If True, remove the reference chain from the target set when it is
        present, yielding strictly interchain distances. If False, retain it,
        and its row contains the intrachain distance between the selected atoms.

    pbc
        If True, use minimum-image distances in an orthorhombic box. If the DCD
        lacks unit-cell lengths, provide ``box_nm``.

    box_nm
        Fallback box lengths ``(Lx, Ly, Lz)`` in nm.

    stride, chunk, frame_start, frame_stop
        Same conventions as ``rg_from_dcd``.

    Returns
    -------
    InterchainDistanceResult
        ``distance_per_chain_nm`` has shape
        ``(n_retained_target_chains, n_frames)``.
    """
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

    reference_indices, reference_labels, atom_to_chain = _chain_indices_from_selection(
        tmpl,
        reference_chain,
    )
    if len(reference_indices) != 1:
        raise ValueError(
            f"reference_chain must resolve to exactly one physical chain; "
            f"it resolved to {len(reference_indices)}: {reference_labels}"
        )

    reference_chain_index = int(reference_indices[0])
    reference_chain_label = str(reference_labels[0])

    target_indices_all, target_labels_all, _ = _chain_indices_from_selection(
        tmpl,
        target_chains,
    )

    retained_target_indices: list[int] = []
    retained_target_labels: list[str] = []
    for chain_index, chain_label in zip(target_indices_all, target_labels_all):
        if bool(exclude_reference) and int(chain_index) == reference_chain_index:
            continue
        retained_target_indices.append(int(chain_index))
        retained_target_labels.append(str(chain_label))

    if not retained_target_indices:
        if bool(exclude_reference) and reference_chain_index in target_indices_all:
            raise ValueError("no target chains remain after excluding the reference chain")
        raise ValueError("target_chains produced no physical chains")

    reference_atom_spec, target_atom_spec = _ordered_atom_pair_selection_specs(selection)

    reference_groups = _selection_to_groups(tmpl, reference_atom_spec)
    if not reference_groups:
        raise ValueError(f"reference atom selection {reference_atom_spec!r} produced no atoms")
    reference_atom_set: set[int] = set()
    for group in reference_groups:
        reference_atom_set.update(int(i) for i in np.asarray(group, dtype=np.int64).tolist())

    reference_atoms = sorted(
        atom_index
        for atom_index in reference_atom_set
        if int(atom_to_chain[int(atom_index)]) == reference_chain_index
    )
    if len(reference_atoms) != 1:
        descriptions = [_atom_pair_label(tmpl_model, atom_index) for atom_index in reference_atoms]
        raise ValueError(
            f"reference atom selection {reference_atom_spec!r} must resolve to "
            f"exactly one atom in chain {reference_chain_label!r}; it selected "
            f"{len(reference_atoms)}: {descriptions}"
        )
    reference_atom_full = int(reference_atoms[0])

    target_groups = _selection_to_groups(tmpl, target_atom_spec)
    if not target_groups:
        raise ValueError(f"target atom selection {target_atom_spec!r} produced no atoms")
    target_atom_set: set[int] = set()
    for group in target_groups:
        target_atom_set.update(int(i) for i in np.asarray(group, dtype=np.int64).tolist())

    target_atoms_full = np.empty(len(retained_target_indices), dtype=np.int64)
    target_atom_labels: list[str] = []

    for out_index, (chain_index, chain_label) in enumerate(
        zip(retained_target_indices, retained_target_labels)
    ):
        selected_atoms = sorted(
            atom_index
            for atom_index in target_atom_set
            if int(atom_to_chain[int(atom_index)]) == int(chain_index)
        )
        if len(selected_atoms) != 1:
            descriptions = [
                _atom_pair_label(tmpl_model, atom_index) for atom_index in selected_atoms
            ]
            raise ValueError(
                f"target atom selection {target_atom_spec!r} must resolve to "
                f"exactly one atom in chain {chain_label!r}; it selected "
                f"{len(selected_atoms)}: {descriptions}"
            )

        atom_index = int(selected_atoms[0])
        target_atoms_full[out_index] = atom_index
        target_atom_labels.append(_atom_pair_label(tmpl_model, atom_index))

    # Read only the reference atom and retained target atoms.
    atom_indices_full = sorted({reference_atom_full, *target_atoms_full.tolist()})
    index_map = {old: new for new, old in enumerate(atom_indices_full)}
    reference_atom_sel = int(index_map[reference_atom_full])
    target_atoms_sel = np.asarray(
        [index_map[int(atom_index)] for atom_index in target_atoms_full],
        dtype=np.int64,
    )

    box_fallback = None
    if bool(pbc) and box_nm is not None:
        box_fallback = _box_lengths_nm(box_nm)

    distance_frames: list[np.ndarray] = []

    for dcd in dcd_list:
        for frame_index, (xyz_sel_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                tmpl_model,
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
            reference_position = xyz[reference_atom_sel, :]
            target_positions = xyz[target_atoms_sel, :]
            displacement = target_positions - reference_position.reshape(1, 3)

            if bool(pbc):
                if box_frame_nm is None:
                    if box_fallback is None:
                        raise ValueError(
                            "DCD lacks unit-cell lengths; pass box_nm=(Lx,Ly,Lz) "
                            "in nm or set pbc=False"
                        )
                    box = box_fallback
                else:
                    box = _box_lengths_nm(box_frame_nm)

                displacement -= np.rint(displacement / box.reshape(1, 3)) * box.reshape(1, 3)

            distance_frames.append(
                np.linalg.norm(displacement, axis=1).astype(
                    np.float64,
                    copy=False,
                )
            )

    if not distance_frames:
        raise ValueError("no frames selected")

    distance_per_chain = np.stack(distance_frames, axis=1)
    distance_mean = np.nanmean(distance_per_chain, axis=0)

    n_chains = int(distance_per_chain.shape[0])
    if n_chains < 2:
        distance_stderr = np.zeros_like(distance_mean)
    else:
        distance_stderr = np.nanstd(distance_per_chain, axis=0, ddof=1) / math.sqrt(float(n_chains))

    return InterchainDistanceResult(
        distance_per_chain_nm=distance_per_chain,
        distance_mean_nm=distance_mean,
        distance_stderr_nm=distance_stderr,
        chain_labels=tuple(retained_target_labels),
        reference_chain_label=reference_chain_label,
        reference_atom_label=_atom_pair_label(
            tmpl_model,
            reference_atom_full,
        ),
        target_atom_labels=tuple(target_atom_labels),
        reference_atom_index=reference_atom_full,
        target_atom_indices=target_atoms_full,
        n_chains=n_chains,
        n_frames=int(distance_per_chain.shape[1]),
        selection=selection,
        exclude_reference=bool(exclude_reference),
        pbc=bool(pbc),
    )


def interchain_distances_all_references_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    reference_chains: Union[str, Sequence[str]] = "protein",
    target_chains: Union[str, Sequence[str]] = "protein",
    selection: Union[str, Sequence[str]],
    exclude_reference: bool = True,
    pbc: bool = True,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> dict[str, InterchainDistanceResult]:
    """Calculate interchain distances for many reference chains in one DCD pass.

    This is the batched counterpart of :func:`interchain_distances_from_dcd`.
    The first atom selector is resolved once for every chain in
    ``reference_chains`` and the second atom selector once for every chain in
    ``target_chains``.  For each trajectory frame a single vectorized
    ``(n_reference, n_target)`` distance matrix is evaluated.  The trajectory is
    therefore read only once, rather than once per reference chain.

    The returned mapping is keyed by the resolved physical reference-chain label.
    Each value is an ordinary :class:`InterchainDistanceResult`, so downstream
    code written for the single-reference routine can use the results unchanged.

    Notes
    -----
    This routine is primarily an I/O and vectorization optimization.  It normally
    outperforms launching one process per reference chain because those processes
    would all reread the same DCD and duplicate topology/trajectory I/O.
    """
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

    reference_indices, reference_labels, atom_to_chain = _chain_indices_from_selection(
        tmpl,
        reference_chains,
    )
    if not reference_indices:
        raise ValueError("reference_chains produced no physical chains")

    target_indices, target_labels, _ = _chain_indices_from_selection(
        tmpl,
        target_chains,
    )
    if not target_indices:
        raise ValueError("target_chains produced no physical chains")

    reference_atom_spec, target_atom_spec = _ordered_atom_pair_selection_specs(selection)

    reference_groups = _selection_to_groups(tmpl, reference_atom_spec)
    if not reference_groups:
        raise ValueError(f"reference atom selection {reference_atom_spec!r} produced no atoms")
    reference_atom_set = {
        int(atom_index)
        for group in reference_groups
        for atom_index in np.asarray(group, dtype=np.int64).tolist()
    }

    target_groups = _selection_to_groups(tmpl, target_atom_spec)
    if not target_groups:
        raise ValueError(f"target atom selection {target_atom_spec!r} produced no atoms")
    target_atom_set = {
        int(atom_index)
        for group in target_groups
        for atom_index in np.asarray(group, dtype=np.int64).tolist()
    }

    reference_atoms_full = np.empty(len(reference_indices), dtype=np.int64)
    reference_atom_labels: list[str] = []
    for out_index, (chain_index, chain_label) in enumerate(
        zip(reference_indices, reference_labels)
    ):
        selected_atoms = sorted(
            atom_index
            for atom_index in reference_atom_set
            if int(atom_to_chain[int(atom_index)]) == int(chain_index)
        )
        if len(selected_atoms) != 1:
            descriptions = [
                _atom_pair_label(tmpl_model, atom_index) for atom_index in selected_atoms
            ]
            raise ValueError(
                f"reference atom selection {reference_atom_spec!r} must resolve to "
                f"exactly one atom in chain {chain_label!r}; it selected "
                f"{len(selected_atoms)}: {descriptions}"
            )
        atom_index = int(selected_atoms[0])
        reference_atoms_full[out_index] = atom_index
        reference_atom_labels.append(_atom_pair_label(tmpl_model, atom_index))

    target_atoms_full = np.empty(len(target_indices), dtype=np.int64)
    target_atom_labels: list[str] = []
    for out_index, (chain_index, chain_label) in enumerate(zip(target_indices, target_labels)):
        selected_atoms = sorted(
            atom_index
            for atom_index in target_atom_set
            if int(atom_to_chain[int(atom_index)]) == int(chain_index)
        )
        if len(selected_atoms) != 1:
            descriptions = [
                _atom_pair_label(tmpl_model, atom_index) for atom_index in selected_atoms
            ]
            raise ValueError(
                f"target atom selection {target_atom_spec!r} must resolve to "
                f"exactly one atom in chain {chain_label!r}; it selected "
                f"{len(selected_atoms)}: {descriptions}"
            )
        atom_index = int(selected_atoms[0])
        target_atoms_full[out_index] = atom_index
        target_atom_labels.append(_atom_pair_label(tmpl_model, atom_index))

    atom_indices_full = sorted(set(reference_atoms_full.tolist()) | set(target_atoms_full.tolist()))
    index_map = {old: new for new, old in enumerate(atom_indices_full)}
    reference_atoms_sel = np.asarray(
        [index_map[int(atom_index)] for atom_index in reference_atoms_full],
        dtype=np.int64,
    )
    target_atoms_sel = np.asarray(
        [index_map[int(atom_index)] for atom_index in target_atoms_full],
        dtype=np.int64,
    )

    box_fallback = None
    if bool(pbc) and box_nm is not None:
        box_fallback = _box_lengths_nm(box_nm)

    target_index_array = np.asarray(target_indices, dtype=np.int64)
    reference_index_array = np.asarray(reference_indices, dtype=np.int64)
    if bool(exclude_reference):
        keep_matrix = target_index_array[None, :] != reference_index_array[:, None]
    else:
        keep_matrix = np.ones(
            (len(reference_indices), len(target_indices)),
            dtype=bool,
        )

    retained_counts = np.sum(keep_matrix, axis=1, dtype=np.int64)
    if np.any(retained_counts <= 0):
        bad = int(np.flatnonzero(retained_counts <= 0)[0])
        raise ValueError(f"no target chains remain for reference chain {reference_labels[bad]!r}")

    # In the usual all-reference/all-target use case every reference retains the
    # same number of targets (N-1 when self is excluded).  Store only those
    # retained distances while streaming, avoiding a second full
    # (n_reference,n_target,n_frames) tensor at finalization.
    uniform_retained = bool(np.all(retained_counts == retained_counts[0]))
    n_retained = int(retained_counts[0]) if uniform_retained else -1

    distance_frames: list[np.ndarray] = []
    for dcd in dcd_list:
        for frame_index, (xyz_sel_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                tmpl_model,
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
            reference_positions = xyz[reference_atoms_sel, :]
            target_positions = xyz[target_atoms_sel, :]
            displacement = target_positions[None, :, :] - reference_positions[:, None, :]

            if bool(pbc):
                if box_frame_nm is None:
                    if box_fallback is None:
                        raise ValueError(
                            "DCD lacks unit-cell lengths; pass box_nm=(Lx,Ly,Lz) "
                            "in nm or set pbc=False"
                        )
                    box = box_fallback
                else:
                    box = _box_lengths_nm(box_frame_nm)
                displacement -= np.rint(displacement / box.reshape(1, 1, 3)) * box.reshape(1, 1, 3)

            matrix = np.linalg.norm(displacement, axis=2).astype(
                np.float64,
                copy=False,
            )
            if uniform_retained:
                distance_frames.append(
                    matrix[keep_matrix].reshape(len(reference_indices), n_retained)
                )
            else:
                # Rare mixed case (some references are absent from target_chains).
                # Keep the full matrix so each reference can retain a different
                # number of target rows during finalization.
                distance_frames.append(matrix)

    if not distance_frames:
        raise ValueError("no frames selected")

    all_distances = np.stack(distance_frames, axis=2)

    results: dict[str, InterchainDistanceResult] = {}
    for ref_out, (reference_chain_index, reference_chain_label) in enumerate(
        zip(reference_indices, reference_labels)
    ):
        keep = keep_matrix[ref_out]
        if uniform_retained:
            distance_per_chain = all_distances[ref_out, :, :]
        else:
            distance_per_chain = all_distances[ref_out, keep, :]
        distance_mean = np.nanmean(distance_per_chain, axis=0)
        n_chains = int(distance_per_chain.shape[0])
        if n_chains < 2:
            distance_stderr = np.zeros_like(distance_mean)
        else:
            distance_stderr = np.nanstd(distance_per_chain, axis=0, ddof=1) / math.sqrt(
                float(n_chains)
            )

        results[str(reference_chain_label)] = InterchainDistanceResult(
            distance_per_chain_nm=np.asarray(distance_per_chain, dtype=np.float64),
            distance_mean_nm=np.asarray(distance_mean, dtype=np.float64),
            distance_stderr_nm=np.asarray(distance_stderr, dtype=np.float64),
            chain_labels=tuple(
                str(label) for label, retain in zip(target_labels, keep.tolist()) if retain
            ),
            reference_chain_label=str(reference_chain_label),
            reference_atom_label=reference_atom_labels[ref_out],
            target_atom_labels=tuple(
                label for label, retain in zip(target_atom_labels, keep.tolist()) if retain
            ),
            reference_atom_index=int(reference_atoms_full[ref_out]),
            target_atom_indices=np.asarray(target_atoms_full[keep], dtype=np.int64),
            n_chains=n_chains,
            n_frames=int(distance_per_chain.shape[1]),
            selection=selection,
            exclude_reference=bool(exclude_reference),
            pbc=bool(pbc),
        )

    return results


@dataclass(frozen=True)
class ReferenceCenterDistanceResult:
    """Distances of selected chain centers from a PBC-aware reference center.

    Array conventions follow ``RgResult`` and ``ChainContactResult`` where
    practical: the first axis of ``distance_per_chain_nm`` identifies a query
    chain and the second axis identifies trajectory frames. Distances may be
    evaluated in full 3D, along one Cartesian axis, or within one Cartesian
    plane, as recorded by ``distance_axes``.
    """

    distance_per_chain_nm: np.ndarray  # (n_query_chains, n_frames)
    distance_mean_nm: np.ndarray  # (n_frames,)
    distance_stderr_nm: np.ndarray  # (n_frames,)
    reference_center_nm: np.ndarray  # (n_frames, 3), wrapped into [0, L)
    reference_center_unwrapped_nm: np.ndarray  # (n_frames, 3), continuous in time
    chain_labels: tuple[str, ...]
    reference_label: str
    reference_chain_labels: tuple[str, ...]
    n_reference_chains: int
    n_query_chains: int
    n_frames: int
    mode: str  # "com" or "cog"
    # Defaults preserve access for older cached/pickled results.
    distance_axes: str = "xyz"
    reference_image_mode: str = "as_is"


def _normalize_distance_axes(distance_axes: str) -> tuple[str, np.ndarray]:
    """Validate Cartesian axes and return a canonical label and indices.

    Accepted values are ``x``, ``y``, ``z``, ``xy``, ``xz``, ``yz``, and
    ``xyz``. Axis order is ignored, so for example ``yx`` is normalized to
    ``xy``. ``3d`` is accepted as an alias for ``xyz``.
    """
    if not isinstance(distance_axes, str):
        raise TypeError("distance_axes must be a string")

    text = distance_axes.strip().lower().replace(" ", "")
    if text == "3d":
        text = "xyz"

    if not text or any(axis not in "xyz" for axis in text):
        raise ValueError(
            "distance_axes must be one of 'x', 'y', 'z', 'xy', 'xz', " "'yz', or 'xyz'"
        )
    if len(set(text)) != len(text):
        raise ValueError("distance_axes must not contain repeated axes")

    canonical = "".join(axis for axis in "xyz" if axis in text)
    if canonical not in {"x", "y", "z", "xy", "xz", "yz", "xyz"}:
        raise ValueError(
            "distance_axes must be one of 'x', 'y', 'z', 'xy', 'xz', " "'yz', or 'xyz'"
        )

    index = {"x": 0, "y": 1, "z": 2}
    indices = np.asarray([index[axis] for axis in canonical], dtype=np.int64)
    return canonical, indices


def _normalize_reference_image_mode(reference_image_mode: str) -> str:
    """Normalize how reference-chain centers are placed across PBC.

    ``"as_is"`` uses the whole-chain centers exactly as stored in the
    trajectory. All selected reference chains must already share the intended
    common image.

    ``"cluster"`` treats every selected chain as whole, wraps only the chain
    centers, and reconstructs one connected periodic cluster before averaging.
    """
    if not isinstance(reference_image_mode, str):
        raise TypeError("reference_image_mode must be a string")

    mode = reference_image_mode.strip().lower().replace("-", "_")
    aliases = {
        "asis": "as_is",
        "stored": "as_is",
        "direct": "as_is",
        "whole": "as_is",
        "common_cluster": "cluster",
        "periodic_cluster": "cluster",
        "wrapped": "cluster",
        "periodic": "cluster",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"as_is", "cluster"}:
        raise ValueError("reference_image_mode must be 'as_is' or 'cluster'")
    return mode


def _one_chain_groups_from_specs(
    tmpl: Any,
    specs: Union[str, Sequence[str]],
    *,
    argument_name: str,
) -> tuple[list[np.ndarray], tuple[str, ...]]:
    """Resolve one selection specification per physical chain.

    The returned labels are the model chain keys, which are normally segment
    IDs for structures read by ``PDBReader``. Each specification must resolve
    to exactly one selection group belonging to exactly one physical chain.
    """
    if isinstance(specs, str):
        specs_list = [specs]
    else:
        specs_list = [str(s) for s in specs]

    if not specs_list:
        raise ValueError(f"{argument_name} is empty")
    if any(not s.strip() for s in specs_list):
        raise ValueError(f"{argument_name} contains an empty selection")

    model = tmpl.model if hasattr(tmpl, "model") else tmpl
    atom_index = {id(a): i for i, a in enumerate(model.atoms)}
    atom_to_chain = np.full(len(model.atoms), -1, dtype=np.int64)
    chain_keys: list[str] = []

    for ci, (key, chain) in enumerate(model.chain.items()):
        chain_keys.append(str(key))
        for residue in chain.residues:
            for atom in residue.atoms:
                ai = atom_index.get(id(atom))
                if ai is not None:
                    atom_to_chain[int(ai)] = int(ci)

    groups: list[np.ndarray] = []
    labels: list[str] = []

    for spec in specs_list:
        resolved = _selection_to_groups(tmpl, spec)
        if len(resolved) != 1:
            raise ValueError(
                f"{argument_name} entry {spec!r} resolved to {len(resolved)} groups; "
                "provide one segment/chain selection per entry"
            )

        group = np.asarray(resolved[0], dtype=np.int64)
        if group.size == 0:
            raise ValueError(f"{argument_name} entry {spec!r} selected no atoms")

        ci = np.unique(atom_to_chain[group])
        ci = ci[ci >= 0]
        if ci.size != 1:
            raise ValueError(
                f"{argument_name} entry {spec!r} must select atoms from exactly "
                f"one physical chain; it selected {ci.size} chains"
            )

        label = chain_keys[int(ci[0])]
        groups.append(group)
        labels.append(label)

    if len(set(labels)) != len(labels):
        raise ValueError(
            f"{argument_name} contains duplicate chains after resolving segment aliases: "
            f"{labels}"
        )

    return groups, tuple(labels)


def _assemble_periodic_cluster_nm(
    centers_nm: np.ndarray,
    box_nm: np.ndarray,
) -> np.ndarray:
    """Place whole-chain centers into one connected periodic image.

    A minimum-distance spanning tree is grown from the first supplied center.
    Each unplaced center is attached through the shortest minimum-image edge to
    any center already placed. This reconstructs an extended cluster through
    local neighbors rather than forcing every center to be near one arbitrary
    anchor.

    The caller must ensure that all supplied centers belong to one connected
    physical cluster. If separate clusters are supplied, this procedure will
    still place them into one periodic image and the resulting center will not
    have a useful physical interpretation.
    """
    centers = np.asarray(centers_nm, dtype=np.float64)
    box = np.asarray(box_nm, dtype=np.float64).reshape(3)

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers_nm must have shape (n_centers, 3)")
    if centers.shape[0] < 1:
        raise ValueError("centers_nm is empty")
    if np.any(~np.isfinite(centers)):
        raise ValueError("centers_nm must contain only finite values")
    if np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box lengths must be finite and positive")

    # Work from canonical wrapped centers. Only the chain centers are wrapped;
    # atoms within each chain are never minimum-imaged or reconstructed here.
    wrapped = centers - np.floor(centers / box.reshape(1, 3)) * box.reshape(1, 3)

    n_centers = int(wrapped.shape[0])
    assembled = np.empty_like(wrapped)
    placed = np.zeros(n_centers, dtype=bool)

    assembled[0] = wrapped[0]
    placed[0] = True

    while not np.all(placed):
        best_distance2 = float("inf")
        best_source = -1
        best_target = -1
        best_displacement: Optional[np.ndarray] = None

        placed_indices = np.flatnonzero(placed)
        unplaced_indices = np.flatnonzero(~placed)

        for source_raw in placed_indices:
            source = int(source_raw)
            for target_raw in unplaced_indices:
                target = int(target_raw)
                displacement = wrapped[target] - wrapped[source]
                displacement -= np.rint(displacement / box) * box
                distance2 = float(np.dot(displacement, displacement))

                if distance2 < best_distance2:
                    best_distance2 = distance2
                    best_source = source
                    best_target = target
                    best_displacement = displacement.copy()

        if best_displacement is None or best_source < 0 or best_target < 0:
            raise RuntimeError("failed to assemble periodic reference cluster")

        assembled[best_target] = assembled[best_source] + best_displacement
        placed[best_target] = True

    return assembled


def reference_center_distances_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    reference_segments: Union[str, Sequence[str]],
    query_segments: Union[str, Sequence[str]],
    reference_label: str = "reference_center",
    mode: str = "com",
    distance_axes: str = "xyz",
    reference_image_mode: str = "as_is",
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> ReferenceCenterDistanceResult:
    """Distances of whole-chain centers from a reference-chain center.

    Every selected chain is assumed to be whole in each stored trajectory
    frame. No atom within a chain is wrapped or minimum-imaged. CA-only centers
    can be requested by supplying CA-restricted selections, as done by the
    notebook-facing helper by default.

    For each frame the routine:

    1. Computes the COM or COG of every reference and query chain directly from
       its stored coordinates.
    2. Places the reference-chain centers according to ``reference_image_mode``:

       - ``"as_is"``: preserve the stored relative images of the chains.
       - ``"cluster"``: wrap only the chain centers and assemble them into one
         connected periodic cluster using a minimum-distance spanning tree.

    3. Computes the reference center, weighting chain centers by selected mass
       for ``mode='com'`` or selected atom count for ``mode='cog'``.
    4. Applies minimum imaging only to each final query-to-reference
       displacement.
    5. Calculates the nonnegative distance using the Cartesian components
       selected by ``distance_axes``.

    ``reference_image_mode='cluster'`` assumes that all supplied reference
    chains belong to one connected physical cluster. Cluster membership itself
    is not inferred by this function.

    Parameters
    ----------
    reference_segments
        One selection per reference chain.
    query_segments
        One selection per query chain.
    mode
        ``"com"`` or ``"cog"``.
    distance_axes
        ``"x"``, ``"y"`, ``"z"``, ``"xy"``, ``"xz"``, ``"yz"``, or
        ``"xyz"``. One-dimensional results are absolute displacements; plane
        and 3D results are Euclidean magnitudes in the selected components.
    reference_image_mode
        ``"as_is"`` for chains already stored in a common image, or
        ``"cluster"`` to assemble whole-chain centers across PBC before
        calculating the reference center.
    box_nm
        Fallback orthorhombic box lengths when absent from the DCD.

    Returns
    -------
    ReferenceCenterDistanceResult
        ``distance_per_chain_nm`` has shape ``(n_query_chains, n_frames)``.
        ``reference_center_nm`` is wrapped into the primary box.
        ``reference_center_unwrapped_nm`` is continuous across frames and
        preserves the direct first-frame center for ``"as_is"`` mode.
    """
    center_mode = str(mode).strip().lower()
    if center_mode not in {"com", "cog"}:
        raise ValueError("mode must be 'com' or 'cog'")

    distance_axes_name, distance_axis_indices = _normalize_distance_axes(distance_axes)
    reference_image_mode_name = _normalize_reference_image_mode(reference_image_mode)

    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    ref_label = str(reference_label).strip()
    if not ref_label:
        raise ValueError("reference_label must be non-empty")

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    reference_groups_full, reference_chain_labels = _one_chain_groups_from_specs(
        tmpl,
        reference_segments,
        argument_name="reference_segments",
    )
    query_groups_full, query_chain_labels = _one_chain_groups_from_specs(
        tmpl,
        query_segments,
        argument_name="query_segments",
    )

    if ref_label in set(reference_chain_labels) | set(query_chain_labels):
        raise ValueError(f"reference_label {ref_label!r} conflicts with a resolved chain label")

    atom_set: set[int] = set()
    for group in reference_groups_full:
        atom_set.update(int(i) for i in group.tolist())
    for group in query_groups_full:
        atom_set.update(int(i) for i in group.tolist())
    atom_indices_full = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices_full)}
    reference_groups = [
        np.asarray([idx_map[int(i)] for i in group.tolist()], dtype=np.int64)
        for group in reference_groups_full
    ]
    query_groups = [
        np.asarray([idx_map[int(i)] for i in group.tolist()], dtype=np.int64)
        for group in query_groups_full
    ]

    masses_all = atom_masses(tmpl_model)
    masses_sel = np.asarray(masses_all[atom_indices_full], dtype=np.float64)

    if center_mode == "com":
        reference_weights = np.asarray(
            [float(np.sum(masses_sel[group])) for group in reference_groups],
            dtype=np.float64,
        )
        if np.any(reference_weights <= 0.0):
            raise ValueError(
                "one or more reference chains have non-positive selected mass; "
                "check atom masses or use mode='cog'"
            )
        center_masses: Optional[np.ndarray] = masses_sel
    else:
        reference_weights = np.asarray(
            [float(group.size) for group in reference_groups],
            dtype=np.float64,
        )
        center_masses = None

    total_reference_weight = float(np.sum(reference_weights))
    if total_reference_weight <= 0.0:
        raise ValueError("total reference-center weight must be positive")

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    distance_frames: list[np.ndarray] = []
    reference_centers_wrapped: list[np.ndarray] = []
    reference_centers_unwrapped: list[np.ndarray] = []

    previous_reference_wrapped: Optional[np.ndarray] = None
    previous_reference_raw: Optional[np.ndarray] = None
    previous_reference_unwrapped: Optional[np.ndarray] = None

    for dcd in dcd_list:
        for frame_index, (xyz_sel_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                tmpl_model,
                chunk=int(chunk),
                stride=int(stride),
                atom_indices=atom_indices_full,
            )
        ):
            if frame_index < int(frame_start):
                continue
            if frame_stop is not None and frame_index >= int(frame_stop):
                break

            if box_frame_nm is None:
                if box_fallback is None:
                    raise ValueError("DCD lacks unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")
                box = box_fallback
            else:
                box = _box_lengths_nm(box_frame_nm)

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)

            # Chains are already whole. Never minimum-image atoms within a
            # chain and never wrap a chain center before deciding how the
            # reference chains should be placed relative to one another.
            reference_chain_centers_stored = group_centers_nm(
                xyz,
                reference_groups,
                masses=center_masses,
                center=center_mode,
                unwrap=False,
                wrap=False,
            )
            query_centers = group_centers_nm(
                xyz,
                query_groups,
                masses=center_masses,
                center=center_mode,
                unwrap=False,
                wrap=False,
            )

            if reference_image_mode_name == "as_is":
                reference_chain_centers_common = reference_chain_centers_stored
            else:
                reference_chain_centers_common = _assemble_periodic_cluster_nm(
                    reference_chain_centers_stored,
                    box,
                )

            reference_center_raw = (
                np.sum(
                    reference_chain_centers_common * reference_weights[:, None],
                    axis=0,
                )
                / total_reference_weight
            )

            reference_center_wrapped = reference_center_raw.copy()
            reference_center_wrapped -= np.floor(reference_center_wrapped / box) * box

            displacement = query_centers - reference_center_raw.reshape(1, 3)
            displacement -= np.rint(displacement / box.reshape(1, 3)) * box.reshape(1, 3)
            distances = np.linalg.norm(
                displacement[:, distance_axis_indices],
                axis=1,
            )

            if previous_reference_unwrapped is None:
                reference_unwrapped = reference_center_raw.copy()
            else:
                assert previous_reference_raw is not None
                assert previous_reference_wrapped is not None

                if reference_image_mode_name == "as_is":
                    step = reference_center_raw - previous_reference_raw
                else:
                    # Cluster reconstruction may choose a raw image shifted by
                    # a whole box as the anchor crosses a boundary. Track the
                    # wrapped center so the reported unwrapped series remains
                    # continuous.
                    step = reference_center_wrapped - previous_reference_wrapped

                step -= np.rint(step / box) * box
                reference_unwrapped = previous_reference_unwrapped + step

            distance_frames.append(np.asarray(distances, dtype=np.float64))
            reference_centers_wrapped.append(reference_center_wrapped.copy())
            reference_centers_unwrapped.append(reference_unwrapped.copy())

            previous_reference_wrapped = reference_center_wrapped.copy()
            previous_reference_raw = reference_center_raw.copy()
            previous_reference_unwrapped = reference_unwrapped.copy()

    if not distance_frames:
        raise ValueError("no frames selected")

    distance_per_chain = np.stack(distance_frames, axis=1)
    distance_mean = np.nanmean(distance_per_chain, axis=0)

    n_query = int(distance_per_chain.shape[0])
    if n_query < 2:
        distance_stderr = np.zeros_like(distance_mean)
    else:
        distance_stderr = np.nanstd(distance_per_chain, axis=0, ddof=1) / math.sqrt(float(n_query))

    reference_center_arr = np.stack(reference_centers_wrapped, axis=0)
    reference_center_unwrapped_arr = np.stack(
        reference_centers_unwrapped,
        axis=0,
    )

    return ReferenceCenterDistanceResult(
        distance_per_chain_nm=distance_per_chain,
        distance_mean_nm=distance_mean,
        distance_stderr_nm=distance_stderr,
        reference_center_nm=reference_center_arr,
        reference_center_unwrapped_nm=reference_center_unwrapped_arr,
        chain_labels=query_chain_labels,
        reference_label=ref_label,
        reference_chain_labels=reference_chain_labels,
        n_reference_chains=int(len(reference_groups)),
        n_query_chains=n_query,
        n_frames=int(distance_per_chain.shape[1]),
        mode=center_mode,
        distance_axes=distance_axes_name,
        reference_image_mode=reference_image_mode_name,
    )


# --- mass-density profiles relative to a reference center --------------------

# 1 dalton / nm^3 = 1.66053906660 g/L.
_DALTON_PER_NM3_TO_G_PER_L = 1.66053906660


@dataclass(frozen=True)
class MassDensityProfileResult:
    """Mass-density profile relative to a PBC-aware reference center.

    ``coordinate_nm`` contains signed Cartesian coordinates for one-dimensional
    profiles (``profile_axes`` equal to ``"x"``, ``"y"``, or ``"z"``). For
    two- and three-dimensional profiles it contains the nonnegative radial
    distance in the selected plane or in full 3D.

    ``density_g_per_l`` is the equal-frame mean mass density. For 1D the bin
    volume is a Cartesian slab spanning the full perpendicular box area; for 2D
    it is an annular cylinder spanning the remaining box dimension; for 3D it
    is a spherical shell. ``density_stderr_g_per_l`` is the standard error over
    the selected trajectory frames.
    """

    coordinate_nm: np.ndarray
    edges_nm: np.ndarray
    density_g_per_l: np.ndarray
    density_stderr_g_per_l: np.ndarray
    n_frames: int
    n_atoms: int
    total_selected_mass_da: float
    profile_axes: str
    geometry: str
    signed: bool
    selection: Any
    reference_label: str
    reference_chain_labels: tuple[str, ...]
    n_reference_chains: int
    mode: str
    reference_image_mode: str
    frame_start: int
    frame_stop: Optional[int]
    stride: int
    element_counts: tuple[tuple[str, int], ...] = ()
    units: str = "g/L"

    @property
    def centers(self) -> np.ndarray:
        """Alias for :attr:`coordinate_nm`."""
        return self.coordinate_nm

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges_nm)


def _time_value_ns(value: Any, *, name: str) -> float:
    """Convert a numeric nanosecond value or OpenMM time Quantity to ns."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        out = float(value)
    else:
        try:
            out = float(value.value_in_unit(nanoseconds))
        except Exception as exc:
            raise TypeError(f"{name} must be a number in ns or an OpenMM time Quantity") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _density_profile_frame_limits(
    *,
    frame_start: int,
    frame_stop: Optional[int],
    time_start: Any,
    time_stop: Any,
    delta_t: Any,
) -> tuple[int, Optional[int]]:
    """Resolve optional time limits to indices in the analyzed frame stream.

    ``delta_t`` is the interval between consecutive analyzed frames, matching
    :func:`calculate_histograms` in the notebook-facing module. Stop limits are
    exclusive.
    """
    uses_time = time_start is not None or time_stop is not None
    if not uses_time:
        start = int(frame_start)
        stop = None if frame_stop is None else int(frame_stop)
        if start < 0:
            raise ValueError("frame_start must be >= 0")
        if stop is not None and stop <= start:
            raise ValueError("frame_stop must be greater than frame_start")
        return start, stop

    if int(frame_start) != 0 or frame_stop is not None:
        raise ValueError("specify either frame_start/frame_stop or time_start/time_stop, not both")
    if delta_t is None:
        raise ValueError("delta_t is required when time_start or time_stop is specified")

    dt_ns = _time_value_ns(delta_t, name="delta_t")
    if dt_ns <= 0.0:
        raise ValueError("delta_t must be > 0")
    effective_dt_ns = dt_ns

    start_time_ns = 0.0 if time_start is None else _time_value_ns(time_start, name="time_start")
    if start_time_ns < 0.0:
        raise ValueError("time_start must be >= 0")

    stop_time_ns = None if time_stop is None else _time_value_ns(time_stop, name="time_stop")
    if stop_time_ns is not None:
        if stop_time_ns < 0.0:
            raise ValueError("time_stop must be >= 0")
        if stop_time_ns <= start_time_ns:
            raise ValueError("time_stop must be greater than time_start")

    # First analyzed frame with t >= limit. The small tolerance prevents an
    # exactly represented multiple such as 4.0/0.2 from rounding just above an
    # integer and incorrectly skipping one frame.
    eps = 1.0e-12
    start = int(math.ceil(start_time_ns / effective_dt_ns - eps))
    stop = None if stop_time_ns is None else int(math.ceil(stop_time_ns / effective_dt_ns - eps))
    if stop is not None and stop <= start:
        raise ValueError("the requested time range selects no frames")
    return start, stop


def _density_profile_edges(
    bins: Union[int, Sequence[float], np.ndarray],
    *,
    value_range: Optional[tuple[float, float]],
    profile_axes: str,
    first_boxes_nm: Sequence[np.ndarray],
) -> np.ndarray:
    """Validate or construct shared profile bin edges."""
    n_dim = len(profile_axes)

    if np.isscalar(bins):
        if isinstance(bins, (bool, np.bool_)):
            raise TypeError("bins must be an integer or explicit bin edges")
        n_bins_float = float(bins)
        if not n_bins_float.is_integer():
            raise ValueError("bins must be an integer")
        n_bins = int(n_bins_float)
        if n_bins < 1:
            raise ValueError("bins must be at least 1")

        if value_range is None:
            axis_map = {"x": 0, "y": 1, "z": 2}
            selected_indices = [axis_map[a] for a in profile_axes]
            safe_half_width = min(
                0.5 * float(np.min(np.asarray(box)[selected_indices])) for box in first_boxes_nm
            )
            if n_dim == 1:
                range_min, range_max = -safe_half_width, safe_half_width
            else:
                range_min, range_max = 0.0, safe_half_width
        else:
            if len(value_range) != 2:
                raise ValueError("value_range must contain exactly two values")
            range_min = float(value_range[0])
            range_max = float(value_range[1])

        if not math.isfinite(range_min) or not math.isfinite(range_max):
            raise ValueError("value_range values must be finite")
        if range_max <= range_min:
            raise ValueError("value_range must be an increasing (min, max) pair")
        if n_dim > 1 and range_min < 0.0:
            raise ValueError("radial density profiles require value_range[0] >= 0")
        return np.linspace(range_min, range_max, n_bins + 1, dtype=np.float64)

    edges = np.asarray(bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            "explicit bin edges must be a one-dimensional array with at least two values"
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError("bin edges must all be finite")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("bin edges must be strictly increasing")
    if n_dim > 1 and edges[0] < 0.0:
        raise ValueError("radial density-profile bin edges must be >= 0")
    return edges.copy()


def _density_profile_bin_volumes_nm3(
    edges_nm: np.ndarray,
    *,
    profile_axes: str,
    box_nm: np.ndarray,
) -> np.ndarray:
    """Return slab, annular-cylinder, or spherical-shell volumes."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    selected = [axis_map[a] for a in profile_axes]
    n_dim = len(selected)
    box = np.asarray(box_nm, dtype=np.float64).reshape(3)

    if n_dim == 1:
        axis = selected[0]
        perpendicular = [i for i in range(3) if i != axis]
        return np.diff(edges_nm) * float(np.prod(box[perpendicular]))

    if n_dim == 2:
        remaining = next(i for i in range(3) if i not in selected)
        annular_area = math.pi * (np.square(edges_nm[1:]) - np.square(edges_nm[:-1]))
        return annular_area * float(box[remaining])

    return (4.0 * math.pi / 3.0) * (np.power(edges_nm[1:], 3) - np.power(edges_nm[:-1], 3))


def _validate_density_profile_range_for_box(
    edges_nm: np.ndarray,
    *,
    profile_axes: str,
    box_nm: np.ndarray,
) -> None:
    """Ensure bins lie inside the non-overlapping minimum-image domain."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    selected = [axis_map[a] for a in profile_axes]
    box = np.asarray(box_nm, dtype=np.float64).reshape(3)
    tol = 1.0e-10

    if len(selected) == 1:
        half = 0.5 * float(box[selected[0]])
        if float(edges_nm[0]) < -half - tol or float(edges_nm[-1]) > half + tol:
            raise ValueError(
                f"1D density-profile edges ({edges_nm[0]}, {edges_nm[-1]}) "
                f"exceed the minimum-image interval [-{half}, {half}] for "
                f"axis {profile_axes!r}"
            )
        return

    safe_max = 0.5 * float(np.min(box[selected]))
    if float(edges_nm[-1]) > safe_max + tol:
        raise ValueError(
            f"radial density-profile maximum {edges_nm[-1]} nm exceeds the "
            f"non-overlapping minimum-image radius {safe_max} nm for axes "
            f"{profile_axes!r}"
        )


def mass_density_profile_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    reference_segments: Union[str, Sequence[str]],
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "all",
    reference_label: str = "reference_center",
    mode: str = "cog",
    profile_axes: str = "xyz",
    reference_image_mode: str = "as_is",
    bins: Union[int, Sequence[float], np.ndarray] = 60,
    value_range: Optional[tuple[float, float]] = None,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 200,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    time_start: Any = None,
    time_stop: Any = None,
    delta_t: Any = None,
) -> MassDensityProfileResult:
    """Calculate mass-density profiles around a condensate reference center.

    The reference center is defined exactly as in
    :func:`reference_center_distances_from_dcd`: selected reference chains are
    assumed to be whole; their centers are evaluated directly from stored
    coordinates and are combined using ``reference_image_mode='as_is'`` or
    ``'cluster'``. Minimum imaging is applied only to final atom-to-reference
    displacements.

    Geometry
    --------
    ``profile_axes`` accepts ``x``, ``y``, ``z``, ``xy``, ``xz``, ``yz``, or
    ``xyz``.

    - 1D profiles retain the sign of the selected displacement component and
      use full-box slabs for normalization.
    - 2D profiles use radial distance in the selected plane and annular-cylinder
      volumes spanning the remaining box dimension.
    - 3D profiles use ordinary radial distance and spherical-shell volumes.

    Atom masses are read from the template model. The bundled PDB reader
    initializes these from element symbols, so the returned values estimate
    mass density in g/L rather than atom number density.

    Frame/time selection
    --------------------
    Specify either ``frame_start``/``frame_stop`` or
    ``time_start``/``time_stop`` with ``delta_t``. Numeric time values are in
    ns. ``delta_t`` is the spacing between analyzed frames, after any stride,
    matching :func:`calculate_histograms`. Stop limits are exclusive.
    """
    center_mode = str(mode).strip().lower()
    if center_mode not in {"com", "cog"}:
        raise ValueError("mode must be 'com' or 'cog'")

    profile_axes_name, profile_axis_indices = _normalize_distance_axes(profile_axes)
    reference_image_mode_name = _normalize_reference_image_mode(reference_image_mode)

    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")

    resolved_start, resolved_stop = _density_profile_frame_limits(
        frame_start=int(frame_start),
        frame_stop=frame_stop,
        time_start=time_start,
        time_stop=time_stop,
        delta_t=delta_t,
    )

    ref_label = str(reference_label).strip()
    if not ref_label:
        raise ValueError("reference_label must be non-empty")

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    reference_groups_full, reference_chain_labels = _one_chain_groups_from_specs(
        tmpl,
        reference_segments,
        argument_name="reference_segments",
    )

    density_groups_full = _selection_to_groups(tmpl, selection)
    if not density_groups_full:
        raise ValueError("selection produced no atoms")
    density_atom_indices_full = sorted(
        {
            int(atom_index)
            for group in density_groups_full
            for atom_index in np.asarray(group, dtype=np.int64).tolist()
        }
    )
    if not density_atom_indices_full:
        raise ValueError("selection produced no atoms")

    atom_set: set[int] = set(density_atom_indices_full)
    for group in reference_groups_full:
        atom_set.update(int(i) for i in group.tolist())
    atom_indices_full = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices_full)}
    reference_groups = [
        np.asarray([idx_map[int(i)] for i in group.tolist()], dtype=np.int64)
        for group in reference_groups_full
    ]
    density_atom_indices = np.asarray(
        [idx_map[int(i)] for i in density_atom_indices_full],
        dtype=np.int64,
    )

    masses_all = atom_masses(tmpl_model)
    masses_selected_all = np.asarray(
        masses_all[density_atom_indices_full],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(masses_selected_all)) or np.any(masses_selected_all <= 0.0):
        bad = int(np.sum((~np.isfinite(masses_selected_all)) | (masses_selected_all <= 0.0)))
        raise ValueError(
            f"{bad} selected atoms have missing or non-positive masses; "
            "check element assignments or set atom masses before analysis"
        )

    masses_subset = np.asarray(masses_all[atom_indices_full], dtype=np.float64)
    if center_mode == "com":
        reference_weights = np.asarray(
            [float(np.sum(masses_subset[group])) for group in reference_groups],
            dtype=np.float64,
        )
        if np.any(reference_weights <= 0.0):
            raise ValueError(
                "one or more reference chains have non-positive selected mass; "
                "check atom masses or use mode='cog'"
            )
        center_masses: Optional[np.ndarray] = masses_subset
    else:
        reference_weights = np.asarray(
            [float(group.size) for group in reference_groups],
            dtype=np.float64,
        )
        center_masses = None

    total_reference_weight = float(np.sum(reference_weights))
    if total_reference_weight <= 0.0:
        raise ValueError("total reference-center weight must be positive")

    first_boxes = [
        _peek_first_box_nm(
            dcd,
            tmpl_model,
            atom_indices_full,
            int(stride),
            box_nm=box_nm,
        )
        for dcd in dcd_list
    ]
    edges = _density_profile_edges(
        bins,
        value_range=value_range,
        profile_axes=profile_axes_name,
        first_boxes_nm=first_boxes,
    )
    coordinate = 0.5 * (edges[:-1] + edges[1:])
    n_bins = int(edges.size - 1)

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)
    density_sum = np.zeros(n_bins, dtype=np.float64)
    density_sumsq = np.zeros(n_bins, dtype=np.float64)
    n_frames = 0

    for dcd in dcd_list:
        for frame_index, (xyz_sel_nm, box_frame_nm) in enumerate(
            iter_dcd(
                dcd,
                tmpl_model,
                chunk=int(chunk),
                stride=int(stride),
                atom_indices=atom_indices_full,
            )
        ):
            if frame_index < resolved_start:
                continue
            if resolved_stop is not None and frame_index >= resolved_stop:
                break

            if box_frame_nm is None:
                if box_fallback is None:
                    raise ValueError("DCD lacks unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")
                box = box_fallback
            else:
                box = _box_lengths_nm(box_frame_nm)

            _validate_density_profile_range_for_box(
                edges,
                profile_axes=profile_axes_name,
                box_nm=box,
            )

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
            reference_chain_centers_stored = group_centers_nm(
                xyz,
                reference_groups,
                masses=center_masses,
                center=center_mode,
                unwrap=False,
                wrap=False,
            )

            if reference_image_mode_name == "as_is":
                reference_chain_centers_common = reference_chain_centers_stored
            else:
                reference_chain_centers_common = _assemble_periodic_cluster_nm(
                    reference_chain_centers_stored,
                    box,
                )

            reference_center_raw = (
                np.sum(
                    reference_chain_centers_common * reference_weights[:, None],
                    axis=0,
                )
                / total_reference_weight
            )

            density_xyz = xyz[density_atom_indices, :]
            displacement = density_xyz - reference_center_raw.reshape(1, 3)
            displacement -= np.rint(displacement / box.reshape(1, 3)) * box.reshape(1, 3)

            if len(profile_axis_indices) == 1:
                profile_coordinate = displacement[:, int(profile_axis_indices[0])]
            else:
                profile_coordinate = np.linalg.norm(
                    displacement[:, profile_axis_indices],
                    axis=1,
                )

            mass_per_bin_da, _ = np.histogram(
                profile_coordinate,
                bins=edges,
                weights=masses_selected_all,
            )
            bin_volumes_nm3 = _density_profile_bin_volumes_nm3(
                edges,
                profile_axes=profile_axes_name,
                box_nm=box,
            )
            if np.any(bin_volumes_nm3 <= 0.0):
                raise ValueError("density-profile bin volumes must be positive")

            density_frame = (
                np.asarray(mass_per_bin_da, dtype=np.float64)
                / bin_volumes_nm3
                * _DALTON_PER_NM3_TO_G_PER_L
            )
            density_sum += density_frame
            density_sumsq += density_frame * density_frame
            n_frames += 1

    if n_frames <= 0:
        raise ValueError("no frames selected")

    density_mean = density_sum / float(n_frames)
    if n_frames < 2:
        density_stderr = np.zeros_like(density_mean)
    else:
        variance = (density_sumsq - density_sum * density_sum / float(n_frames)) / float(
            n_frames - 1
        )
        variance = np.maximum(variance, 0.0)
        density_stderr = np.sqrt(variance / float(n_frames))

    element_counter: dict[str, int] = {}
    for atom_index in density_atom_indices_full:
        symbol = (
            str(getattr(tmpl_model.atoms[int(atom_index)], "element", "") or "?").strip().upper()
        )
        element_counter[symbol] = element_counter.get(symbol, 0) + 1

    n_dim = len(profile_axes_name)
    geometry = "slab" if n_dim == 1 else ("cylindrical" if n_dim == 2 else "spherical")

    return MassDensityProfileResult(
        coordinate_nm=coordinate,
        edges_nm=edges,
        density_g_per_l=density_mean,
        density_stderr_g_per_l=density_stderr,
        n_frames=int(n_frames),
        n_atoms=int(len(density_atom_indices_full)),
        total_selected_mass_da=float(np.sum(masses_selected_all)),
        profile_axes=profile_axes_name,
        geometry=geometry,
        signed=bool(n_dim == 1),
        selection=selection,
        reference_label=ref_label,
        reference_chain_labels=reference_chain_labels,
        n_reference_chains=int(len(reference_groups)),
        mode=center_mode,
        reference_image_mode=reference_image_mode_name,
        frame_start=int(resolved_start),
        frame_stop=resolved_stop,
        stride=int(stride),
        element_counts=tuple(sorted(element_counter.items())),
    )


# --- binding/contact probability analysis ------------------------------------


AVOGADRO = 6.02214076e23
R_KJ_MOL_K = 0.00831446261815324
LITER_PER_NM3 = 1.0e-24
STANDARD_CONCENTRATION_MOLAR = 1.0
# Molecular volume corresponding to 1 M: 1/(N_A * 1e-24) nm^3 per molecule.
STANDARD_VOLUME_NM3 = 1.0 / (AVOGADRO * LITER_PER_NM3)


@dataclass(frozen=True)
class SpecificContact:
    """Specific contact between local bead indices in two selected groups."""

    i_local: int
    j_local: int
    cutoff_nm: float


@dataclass(frozen=True)
class BindingCriterion:
    """Binding definition.

    kind:
      - "cofm": COG/COM distance between groups
      - "closest": closest bead distance between groups
      - "specific": specific local bead contacts

    Notes
    -----
    A contact criterion is a definition of a bound microstate, not a unique
    thermodynamic state. For multivalent/clustering systems, pair contacts,
    greedily assigned dimers, and connected clusters are distinct observables.
    """

    kind: str
    cutoff_nm: float = 1.0
    contacts: tuple[SpecificContact, ...] = ()
    min_contacts: int = 1


@dataclass(frozen=True)
class BindingAffinityResult:
    """Results from pair/contact binding analysis.

    Interpretation
    --------------
    - ``pair_*`` methods use a specific-pair probability and give an apparent
      two-body standard-state free energy:

          Ka_pair = V * p_pair / (1 - p_pair)
          DeltaG0_pair = -RT ln(Ka_pair / V0)

      with ``V0 = 1.66054 nm^3`` for a 1 M standard state. This is the quantity
      most directly comparable to a two-molecule umbrella calculation, provided
      many-body effects are weak enough for a pair interpretation to make sense.

    - ``kd_molar`` / ``delta_g_kj_mol`` use the per-frame ``exclusive_count``
      and a 1:1 mass-action model. They are appropriate only for systems that
      are reasonably described as independent heterotypic AB dimers or
      homotypic A2 dimers. They are not a general condensate/cluster free energy.
    """

    pair_i: np.ndarray
    pair_j: np.ndarray
    bound: np.ndarray
    metric_nm: np.ndarray
    exclusive_count: np.ndarray
    box_volume_nm3: np.ndarray
    criterion: BindingCriterion
    n_a: int
    n_b: int
    # True when group_b was omitted and the same set of molecules was self-paired.
    same_groups: bool = False
    # True when exclusive_count was generated by greedy one-partner-per-molecule matching.
    exclusive: bool = True

    @property
    def n_frames(self) -> int:
        return int(self.bound.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.bound.shape[1])

    @property
    def mean_box_volume_nm3(self) -> float:
        volume_nm3 = float(np.nanmean(self.box_volume_nm3))
        if volume_nm3 <= 0.0 or not np.isfinite(volume_nm3):
            raise ValueError("valid box volume is required")
        return volume_nm3

    @property
    def pair_probability(self) -> np.ndarray:
        return np.mean(self.bound, axis=0)

    @property
    def mean_pair_probability(self) -> float:
        """Probability obtained by pooling all specific molecular pairs."""
        if self.n_pairs <= 0 or self.n_frames <= 0:
            return float("nan")
        return float(np.mean(self.bound))

    @property
    def any_bound_probability(self) -> float:
        """Probability that at least one pair contact exists in a frame."""
        return float(np.any(self.bound, axis=1).mean())

    @property
    def mean_bound_pairs(self) -> float:
        """Mean number of pair contacts per frame; not necessarily complexes."""
        return float(self.bound.sum(axis=1).mean())

    @property
    def mean_exclusive_complexes(self) -> float:
        """Mean number of greedily assigned non-overlapping 1:1 complexes."""
        return float(self.exclusive_count.mean())

    @property
    def model_name(self) -> str:
        if not self.exclusive:
            return "nonexclusive_contact_count"
        return "homotypic_A2" if self.same_groups else "heterotypic_AB"

    def _probability_from_counts(
        self, counts: np.ndarray, trials: int, pseudocount: float
    ) -> np.ndarray:
        pc = float(pseudocount)
        if pc < 0.0 or not np.isfinite(pc):
            raise ValueError("pseudocount must be finite and >= 0")
        t = int(trials)
        if t <= 0:
            raise ValueError("number of trials must be > 0")
        counts = np.asarray(counts, dtype=np.float64)
        return (counts + pc) / (float(t) + 2.0 * pc)

    def pair_association_nm3(self, *, pseudocount: float = 0.0) -> np.ndarray:
        """Specific-pair apparent association constants in nm^3.

        For a particular pair in a box of volume V,

            Ka = V * p_bound / (1 - p_bound)

        where p_bound is the probability that that specific pair satisfies the
        contact criterion. Use a small pseudocount, e.g. 0.5, if you want finite
        estimates when a pair is never or always observed bound in a finite run.
        """
        counts = np.sum(self.bound, axis=0)
        p = self._probability_from_counts(counts, self.n_frames, pseudocount)
        return _ka_nm3_from_probability(p, self.mean_box_volume_nm3)

    def pair_association_molar_inverse(self, *, pseudocount: float = 0.0) -> np.ndarray:
        """Specific-pair apparent association constants in M^-1."""
        return self.pair_association_nm3(pseudocount=pseudocount) / STANDARD_VOLUME_NM3

    def pair_kd_molar(self, *, pseudocount: float = 0.0) -> np.ndarray:
        """Specific-pair apparent dissociation constants in mol/L."""
        ka_nm3 = self.pair_association_nm3(pseudocount=pseudocount)
        return _kd_molar_from_ka_nm3(ka_nm3)

    def pair_delta_gzero_kj_mol(
        self, temperature_k: float, *, pseudocount: float = 0.0
    ) -> np.ndarray:
        """Specific-pair apparent standard binding free energies in kJ/mol."""
        ka_nm3 = self.pair_association_nm3(pseudocount=pseudocount)
        return _delta_gzero_from_ka_nm3(ka_nm3, temperature_k)

    def mean_pair_association_nm3(self, *, pseudocount: float = 0.0) -> float:
        """Pooled-pair apparent association constant in nm^3.

        This pools all pair observations before converting to Ka. It is usually
        preferable to averaging the per-pair DeltaG values directly.
        """
        total_bound = np.asarray([float(np.sum(self.bound))], dtype=np.float64)
        p = self._probability_from_counts(
            total_bound,
            self.n_frames * self.n_pairs,
            pseudocount,
        )[0]
        return float(_ka_nm3_from_probability(np.asarray([p]), self.mean_box_volume_nm3)[0])

    def mean_pair_kd_molar(self, *, pseudocount: float = 0.0) -> float:
        ka_nm3 = self.mean_pair_association_nm3(pseudocount=pseudocount)
        return float(_kd_molar_from_ka_nm3(np.asarray([ka_nm3], dtype=np.float64))[0])

    def mean_pair_delta_gzero_kj_mol(
        self,
        temperature_k: float,
        *,
        pseudocount: float = 0.0,
    ) -> float:
        ka_nm3 = self.mean_pair_association_nm3(pseudocount=pseudocount)
        return float(
            _delta_gzero_from_ka_nm3(np.asarray([ka_nm3], dtype=np.float64), temperature_k)[0]
        )

    def kd_molar(self, *, require_exclusive: bool = True) -> float:
        """Mass-action Kd from exclusive complex counts.

        Returns
        -------
        float
            Kd in mol/L for the model implied by ``same_groups``:

            - heterotypic: A + B <-> AB
            - homotypic:  A + A <-> A2

        Notes
        -----
        This is not a general many-body cluster free energy. It assumes the
        contact network can be reduced to non-overlapping 1:1 complexes.
        """
        if require_exclusive and not self.exclusive:
            raise ValueError(
                "kd_molar requires exclusive=True because nonexclusive bound-pair "
                "counts are not molecule complexes. Use pair_kd_molar() or "
                "mean_pair_kd_molar() for pair-probability estimates."
            )

        n_complex = self.mean_exclusive_complexes
        if n_complex <= 0.0:
            return float("inf")

        volume_l = self.mean_box_volume_nm3 * LITER_PER_NM3
        if volume_l <= 0.0 or not np.isfinite(volume_l):
            raise ValueError("valid box volume is required for Kd")

        if self.same_groups:
            # A + A <-> A2. Each dimer consumes two monomers.
            max_complex = 0.5 * float(self.n_a)
            if n_complex > max_complex + 1.0e-9:
                raise ValueError(
                    "exclusive_count is inconsistent with homotypic dimerization: "
                    "mean complex count exceeds N/2"
                )
            n_free = max(float(self.n_a) - 2.0 * n_complex, 0.0)
            return n_free * n_free / (n_complex * AVOGADRO * volume_l)

        # A + B <-> AB. Each complex consumes one A and one B.
        max_complex = min(float(self.n_a), float(self.n_b))
        if n_complex > max_complex + 1.0e-9:
            raise ValueError(
                "exclusive_count is inconsistent with heterotypic dimerization: "
                "mean complex count exceeds min(n_a, n_b)"
            )
        n_a_free = max(float(self.n_a) - n_complex, 0.0)
        n_b_free = max(float(self.n_b) - n_complex, 0.0)
        return n_a_free * n_b_free / (n_complex * AVOGADRO * volume_l)

    def delta_g_kj_mol(self, temperature_k: float, *, require_exclusive: bool = True) -> float:
        """Mass-action apparent standard DeltaG in kJ/mol from Kd.

        Uses DeltaG0 = RT ln(Kd / 1 M). Since ``kd_molar`` returns Kd in mol/L
        and the standard concentration is 1 M, this is numerically ``RT ln(Kd)``.
        """
        kd = self.kd_molar(require_exclusive=require_exclusive)
        return _delta_gzero_from_kd_molar(kd, temperature_k)

    def cluster_sizes_by_frame(self, *, include_singletons: bool = True) -> list[np.ndarray]:
        """Connected-component sizes of the contact graph for self-paired systems.

        This is useful for diagnosing whether a single pairwise DeltaG is a poor
        description. It is currently defined only for ``same_groups=True``.
        """
        if not self.same_groups:
            raise ValueError("cluster_sizes_by_frame is only defined for self-paired systems")

        out: list[np.ndarray] = []
        for row in self.bound:
            out.append(
                _binding_cluster_sizes_for_frame(
                    row,
                    self.pair_i,
                    self.pair_j,
                    self.n_a,
                    include_singletons=include_singletons,
                )
            )
        return out

    def mean_cluster_counts(self, *, include_singletons: bool = False) -> dict[int, float]:
        """Mean number of clusters of each size per frame for self-paired systems."""
        sizes_by_frame = self.cluster_sizes_by_frame(include_singletons=include_singletons)
        counts: dict[int, float] = {}
        for sizes in sizes_by_frame:
            unique, cnt = np.unique(sizes.astype(np.int64), return_counts=True)
            for size, n in zip(unique.tolist(), cnt.tolist()):
                counts[int(size)] = counts.get(int(size), 0.0) + float(n)
        if self.n_frames > 0:
            for size in list(counts):
                counts[size] /= float(self.n_frames)
        return dict(sorted(counts.items()))


def binding_affinity_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    group_a: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein",
    group_b: Optional[Union[str, Sequence[str], Sequence[Sequence[int]]]] = None,
    criterion: BindingCriterion = BindingCriterion(kind="closest", cutoff_nm=0.8),
    center: str = "cog",
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
    exclusive: bool = True,
    unwrap_groups: bool = True,
) -> BindingAffinityResult:
    """Estimate contact probabilities and apparent binding quantities from DCDs.

    Parameters
    ----------
    pdb_file
        Template PDB matching the DCD atom order.
    dcd_files
        One DCD or a sequence of DCDs.
    group_a, group_b
        Selection strings or explicit atom-index groups. If group_b is None,
        group_a is self-paired with duplicate/self pairs removed; the mass-action
        model is then homotypic A + A <-> A2.
    criterion
        BindingCriterion("cofm"), BindingCriterion("closest"), or
        BindingCriterion("specific").
    center
        "cog" or "com"; used for cofm only.
    exclusive
        If True, greedily counts at most one partner per molecule per frame.
        Use this only for 1:1 complex-count/Kd-like estimates. Pair-probability
        estimates are available regardless of this setting.
    unwrap_groups
        If True, unwrap atoms within each molecule before center calculations.

    Notes
    -----
    For comparison to a two-molecule umbrella calculation, prefer
    ``result.mean_pair_delta_gzero_kj_mol(T)`` or
    ``result.pair_delta_gzero_kj_mol(T)`` over the mass-action
    ``result.delta_g_kj_mol(T)``. The latter imposes a 1:1 dimer model.
    """

    if int(stride) <= 0:
        raise ValueError("stride must be >= 1")
    if int(chunk) <= 0:
        raise ValueError("chunk must be >= 1")
    if int(frame_start) < 0:
        raise ValueError("frame_start must be >= 0")

    kind = criterion.kind.strip().lower()
    if kind not in {"cofm", "closest", "specific"}:
        raise ValueError("criterion.kind must be 'cofm', 'closest', or 'specific'")

    center_mode = center.strip().lower()
    if center_mode not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    dcd_list = _as_file_list(dcd_files)
    if not dcd_list:
        raise ValueError("no DCD files provided")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    groups_a_full = _selection_to_groups(tmpl, group_a)
    groups_b_full = groups_a_full if group_b is None else _selection_to_groups(tmpl, group_b)

    if not groups_a_full or not groups_b_full:
        raise ValueError("group selection produced no groups")

    same_groups = group_b is None
    pair_i, pair_j = _binding_pairs(len(groups_a_full), len(groups_b_full), same_groups)
    if pair_i.size == 0:
        raise ValueError("no molecule pairs to analyze")

    atom_indices_full = _binding_atom_subset(groups_a_full, groups_b_full)
    idx_map = {old: new for new, old in enumerate(atom_indices_full)}

    groups_a = _remap_groups(groups_a_full, idx_map)
    groups_b = groups_a if same_groups else _remap_groups(groups_b_full, idx_map)

    masses_sel = None
    if kind == "cofm" and center_mode == "com":
        masses_all = atom_masses(tmpl_model)
        masses_sel = np.asarray(masses_all[atom_indices_full], dtype=np.float64)

    bound_rows: list[np.ndarray] = []
    metric_rows: list[np.ndarray] = []
    count_rows: list[float] = []
    volume_rows: list[float] = []

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

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
                    raise ValueError("DCD lacks unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")
                box = box_fallback
            else:
                box = _box_lengths_nm(box_frame_nm)

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)
            bound, metric = _binding_frame(
                xyz,
                box,
                groups_a,
                groups_b,
                pair_i,
                pair_j,
                criterion,
                masses=masses_sel,
                center=center_mode,
                unwrap_groups=unwrap_groups,
            )

            if exclusive:
                n_complex = _binding_exclusive_count(
                    bound, metric, pair_i, pair_j, same_groups=same_groups
                )
            else:
                n_complex = float(bound.sum())

            bound_rows.append(bound)
            metric_rows.append(metric)
            count_rows.append(n_complex)
            volume_rows.append(float(box[0] * box[1] * box[2]))

    if not bound_rows:
        raise ValueError("no frames selected")

    return BindingAffinityResult(
        pair_i=pair_i,
        pair_j=pair_j,
        bound=np.vstack(bound_rows),
        metric_nm=np.vstack(metric_rows),
        exclusive_count=np.asarray(count_rows, dtype=np.float64),
        box_volume_nm3=np.asarray(volume_rows, dtype=np.float64),
        criterion=criterion,
        n_a=len(groups_a_full),
        n_b=len(groups_b_full),
        same_groups=bool(same_groups),
        exclusive=bool(exclusive),
    )


def save_binding_affinity(
    prefix: FileLike,
    result: BindingAffinityResult,
    *,
    temperature_k: Optional[float] = None,
    pseudocount: float = 0.0,
) -> None:
    """Write pair probabilities and frame time series.

    If ``temperature_k`` is provided, the pair table also includes pairwise
    apparent DeltaG0 values in kJ/mol.
    """

    p = Path(prefix)

    pair_p = result._probability_from_counts(
        np.sum(result.bound, axis=0),
        result.n_frames,
        pseudocount,
    )
    pair_ka_nm3 = result.pair_association_nm3(pseudocount=pseudocount)
    pair_kd_m = result.pair_kd_molar(pseudocount=pseudocount)
    mean_metric = np.nanmean(result.metric_nm, axis=0)

    if temperature_k is None:
        pair_data = np.column_stack(
            (
                result.pair_i,
                result.pair_j,
                pair_p,
                pair_ka_nm3,
                pair_kd_m,
                mean_metric,
            )
        )
        fmt = ["%d", "%d", "%.8f", "%.8g", "%.8g", "%.8f"]
        header = "pair_i pair_j p_bound ka_pair_nm3 kd_pair_M mean_metric_nm"
    else:
        pair_dg = result.pair_delta_gzero_kj_mol(
            float(temperature_k),
            pseudocount=pseudocount,
        )
        pair_data = np.column_stack(
            (
                result.pair_i,
                result.pair_j,
                pair_p,
                pair_ka_nm3,
                pair_kd_m,
                pair_dg,
                mean_metric,
            )
        )
        fmt = ["%d", "%d", "%.8f", "%.8g", "%.8g", "%.8f", "%.8f"]
        header = "pair_i pair_j p_bound ka_pair_nm3 kd_pair_M dg0_pair_kj_mol mean_metric_nm"

    np.savetxt(
        p.with_suffix(".pairs.dat"),
        pair_data,
        fmt=fmt,
        header=header,
    )

    ts_data = np.column_stack(
        (
            np.arange(result.n_frames, dtype=np.int64),
            result.bound.sum(axis=1),
            result.exclusive_count,
            np.any(result.bound, axis=1).astype(np.int32),
            result.box_volume_nm3,
        )
    )
    np.savetxt(
        p.with_suffix(".timeseries.dat"),
        ts_data,
        fmt=["%d", "%d", "%.8f", "%d", "%.8f"],
        header="frame bound_pairs exclusive_complexes any_bound box_volume_nm3",
    )


def _temperature_k(temperature_k: float) -> float:
    t = float(temperature_k)
    if t <= 0.0 or not np.isfinite(t):
        raise ValueError("temperature_k must be finite and > 0")
    return t


def _ka_nm3_from_probability(p: np.ndarray, volume_nm3: float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    if volume_nm3 <= 0.0 or not np.isfinite(volume_nm3):
        raise ValueError("valid box volume is required")

    ka = np.full_like(p_arr, np.nan, dtype=np.float64)
    ka[p_arr <= 0.0] = 0.0
    ka[p_arr >= 1.0] = np.inf
    ok = (p_arr > 0.0) & (p_arr < 1.0)
    ka[ok] = float(volume_nm3) * p_arr[ok] / (1.0 - p_arr[ok])
    return ka


def _kd_molar_from_ka_nm3(ka_nm3: np.ndarray) -> np.ndarray:
    ka = np.asarray(ka_nm3, dtype=np.float64)
    kd = np.full_like(ka, np.nan, dtype=np.float64)
    kd[ka == 0.0] = np.inf
    kd[np.isposinf(ka)] = 0.0
    ok = np.isfinite(ka) & (ka > 0.0)
    kd[ok] = STANDARD_VOLUME_NM3 / ka[ok]
    return kd


def _delta_gzero_from_ka_nm3(ka_nm3: np.ndarray, temperature_k: float) -> np.ndarray:
    t = _temperature_k(temperature_k)
    rt = R_KJ_MOL_K * t
    ka = np.asarray(ka_nm3, dtype=np.float64)

    dg = np.full_like(ka, np.nan, dtype=np.float64)
    dg[ka == 0.0] = np.inf
    dg[np.isposinf(ka)] = -np.inf
    ok = np.isfinite(ka) & (ka > 0.0)
    dg[ok] = -rt * np.log(ka[ok] / STANDARD_VOLUME_NM3)
    return dg


def _delta_gzero_from_kd_molar(kd_molar: float, temperature_k: float) -> float:
    t = _temperature_k(temperature_k)
    kd = float(kd_molar)
    if kd < 0.0 or math.isnan(kd):
        return float("nan")
    if kd == 0.0:
        return float("-inf")
    if math.isinf(kd):
        return float("inf")
    return R_KJ_MOL_K * t * math.log(kd / STANDARD_CONCENTRATION_MOLAR)


def _binding_frame(
    xyz_nm: np.ndarray,
    box_nm: np.ndarray,
    groups_a: Sequence[np.ndarray],
    groups_b: Sequence[np.ndarray],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    criterion: BindingCriterion,
    *,
    masses: Optional[np.ndarray],
    center: str,
    unwrap_groups: bool,
) -> tuple[np.ndarray, np.ndarray]:
    kind = criterion.kind.strip().lower()
    n_pairs = int(pair_i.size)

    bound = np.zeros(n_pairs, dtype=bool)
    metric = np.full(n_pairs, np.inf, dtype=np.float64)

    centers_a = None
    centers_b = None
    if kind == "cofm":
        centers_a = group_centers_nm(
            xyz_nm,
            groups_a,
            masses=masses,
            box_nm=box_nm,
            center=center,
            unwrap=unwrap_groups,
            wrap=True,
        )
        centers_b = group_centers_nm(
            xyz_nm,
            groups_b,
            masses=masses,
            box_nm=box_nm,
            center=center,
            unwrap=unwrap_groups,
            wrap=True,
        )

    for k, (ia_raw, jb_raw) in enumerate(zip(pair_i, pair_j)):
        ia = int(ia_raw)
        jb = int(jb_raw)

        if kind == "cofm":
            assert centers_a is not None
            assert centers_b is not None
            dist = _binding_distance(centers_a[ia], centers_b[jb], box_nm)
            metric[k] = dist
            bound[k] = dist <= criterion.cutoff_nm

        elif kind == "closest":
            dist = _binding_closest_distance(
                xyz_nm[groups_a[ia]],
                xyz_nm[groups_b[jb]],
                box_nm,
            )
            metric[k] = dist
            bound[k] = dist <= criterion.cutoff_nm

        elif kind == "specific":
            n_contact, score = _binding_specific_contacts(
                xyz_nm,
                groups_a[ia],
                groups_b[jb],
                box_nm,
                criterion.contacts,
            )
            metric[k] = score
            bound[k] = n_contact >= int(criterion.min_contacts)

    return bound, metric


def _binding_specific_contacts(
    xyz_nm: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray,
    box_nm: np.ndarray,
    contacts: Sequence[SpecificContact],
) -> tuple[int, float]:
    if not contacts:
        raise ValueError("specific binding requires at least one contact")

    n_contact = 0
    score = np.inf

    for contact in contacts:
        ia = int(group_a[int(contact.i_local)])
        jb = int(group_b[int(contact.j_local)])
        cutoff = float(contact.cutoff_nm)
        dist = _binding_distance(xyz_nm[ia], xyz_nm[jb], box_nm)

        if dist <= cutoff:
            n_contact += 1
        score = min(score, dist / cutoff)

    return int(n_contact), float(score)


def _binding_closest_distance(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    box_nm: np.ndarray,
) -> float:
    d = xyz_a[:, None, :] - xyz_b[None, :, :]
    d -= np.rint(d / box_nm.reshape(1, 1, 3)) * box_nm.reshape(1, 1, 3)
    r2 = np.sum(d * d, axis=2)
    return float(np.sqrt(np.min(r2)))


def _binding_distance(a_nm: np.ndarray, b_nm: np.ndarray, box_nm: np.ndarray) -> float:
    d = np.asarray(a_nm, dtype=np.float64) - np.asarray(b_nm, dtype=np.float64)
    d -= np.rint(d / box_nm) * box_nm
    return float(np.sqrt(np.dot(d, d)))


def _binding_pairs(
    n_a: int,
    n_b: int,
    same_groups: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pi: list[int] = []
    pj: list[int] = []

    for i in range(int(n_a)):
        for j in range(int(n_b)):
            if same_groups and j <= i:
                continue
            pi.append(i)
            pj.append(j)

    return np.asarray(pi, dtype=np.int32), np.asarray(pj, dtype=np.int32)


def _binding_atom_subset(
    groups_a: Sequence[np.ndarray],
    groups_b: Sequence[np.ndarray],
) -> list[int]:
    atoms: set[int] = set()
    for group in groups_a:
        atoms.update(int(i) for i in group.tolist())
    for group in groups_b:
        atoms.update(int(i) for i in group.tolist())
    return sorted(atoms)


def _remap_groups(
    groups: Sequence[np.ndarray],
    idx_map: dict[int, int],
) -> list[np.ndarray]:
    return [
        np.asarray([idx_map[int(i)] for i in group.tolist()], dtype=np.int64) for group in groups
    ]


def _binding_exclusive_count(
    bound: np.ndarray,
    metric: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    same_groups: bool,
) -> float:
    """Greedy maximum-cardinality-like count of non-overlapping contacts.

    The matching is greedy by the metric value, not globally optimal. For
    homotypic self-pairing, a single used-molecule set is required; separate
    A/B used sets would allow molecule k to be used once as i and once as j.
    """
    edges = np.flatnonzero(bound)
    if edges.size == 0:
        return 0.0

    order = edges[np.argsort(metric[edges])]
    count = 0

    if same_groups:
        used: set[int] = set()
        for edge_raw in order:
            edge = int(edge_raw)
            i = int(pair_i[edge])
            j = int(pair_j[edge])
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            count += 1
        return float(count)

    used_i: set[int] = set()
    used_j: set[int] = set()
    for edge_raw in order:
        edge = int(edge_raw)
        i = int(pair_i[edge])
        j = int(pair_j[edge])
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        count += 1

    return float(count)


def _binding_cluster_sizes_for_frame(
    bound: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    n_nodes: int,
    *,
    include_singletons: bool,
) -> np.ndarray:
    parent = np.arange(int(n_nodes), dtype=np.int64)
    size = np.ones(int(n_nodes), dtype=np.int64)

    def find(x: int) -> int:
        y = int(x)
        while int(parent[y]) != y:
            parent[y] = parent[int(parent[y])]
            y = int(parent[y])
        return y

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if int(size[ra]) < int(size[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for edge in np.flatnonzero(bound):
        union(int(pair_i[int(edge)]), int(pair_j[int(edge)]))

    roots: dict[int, int] = {}
    for node in range(int(n_nodes)):
        root = find(node)
        roots[root] = roots.get(root, 0) + 1

    values = np.asarray(list(roots.values()), dtype=np.int64)
    if not include_singletons:
        values = values[values > 1]
    return np.sort(values)[::-1]


def _cluster_centers_from_protein_centers(
    protein_centers_nm: np.ndarray,
    clusters: Sequence[Sequence[int]],
    box_nm: np.ndarray,
    *,
    min_cluster_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert protein centers into cluster centers for one frame.

    protein_centers_nm
        (n_proteins, 3), wrapped protein centers.

    clusters
        List of clusters, where each cluster is a list of protein indices.

    box_nm
        Orthorhombic box lengths in nm.

    min_cluster_size
        Minimum cluster size to include.
        Use 1 for monomers + clusters.
        Use 2 for non-singleton clusters only.

    Returns
    -------
    centers_nm
        (n_clusters_kept, 3)

    sizes
        Cluster sizes corresponding to centers_nm.
    """
    pc = np.asarray(protein_centers_nm, dtype=np.float64)
    box = np.asarray(box_nm, dtype=np.float64).reshape(3)

    centers: list[np.ndarray] = []
    sizes: list[int] = []

    for cl in clusters:
        members = np.asarray(cl, dtype=np.int64)

        if members.size < int(min_cluster_size):
            continue

        x = pc[members, :]

        if members.size == 1:
            cen = x[0].copy()
        else:
            # Make the cluster whole relative to the first member.
            ref = x[0:1, :]
            d = x - ref
            d -= np.rint(d / box.reshape(1, 3)) * box.reshape(1, 3)
            x_unwrapped = ref + d

            # Center of geometry of the member protein centers.
            cen = np.mean(x_unwrapped, axis=0)

        # Wrap cluster center back into the primary box.
        cen -= np.floor(cen / box) * box

        centers.append(cen)
        sizes.append(int(members.size))

    if not centers:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(sizes, dtype=np.int64),
    )


def cluster_rdf_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    clusters_out: dict[str, Any],
    *,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein",
    center: str = "cog",
    unwrap_proteins: bool = True,
    min_cluster_size: int = 1,
    dr_nm: float = 0.01,
    r_max_nm: Optional[float] = None,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
    tail_norm_range_nm: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """
    RDF between cluster centers, where the number of cluster particles varies by frame.

    clusters_out
        Output dictionary from clusters_from_dcd(). It must have been generated
        using the same DCDs, stride, frame_start/frame_stop, and selection.

    min_cluster_size
        1 includes monomers and clusters.
        2 includes only non-singleton clusters.

    Normalization
    -------------
    For each frame, the ideal-gas expected shell count is:

        n_pairs(frame) * shell_volume / box_volume(frame)

    This handles a variable number of cluster particles per frame.
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
    if int(min_cluster_size) < 1:
        raise ValueError("min_cluster_size must be >= 1")
    if float(dr_nm) <= 0.0:
        raise ValueError("dr_nm must be > 0")

    center_mode = str(center).strip().lower()
    if center_mode not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    clusters_by_frame = clusters_out.get("clusters_by_frame")
    if clusters_by_frame is None:
        raise KeyError("clusters_out is missing 'clusters_by_frame'")

    clusters_by_frame = list(clusters_by_frame)
    if not clusters_by_frame:
        raise ValueError("clusters_out['clusters_by_frame'] is empty")

    tmpl = PDBReader().read(pdb_file)
    tmpl_model = tmpl.model

    groups_global = _selection_to_groups(tmpl, selection)

    if len(groups_global) < 2:
        raise ValueError("selection must yield >=2 non-empty protein groups")

    n_proteins = int(len(groups_global))

    if "n_proteins" in clusters_out:
        if int(clusters_out["n_proteins"]) != n_proteins:
            raise ValueError(
                f"clusters_out has n_proteins={clusters_out['n_proteins']}, "
                f"but selection produced {n_proteins} groups"
            )

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

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    # Determine r_max.
    if r_max_nm is None:
        half_boxes = []

        for dcd in dcd_list:
            it = iter_dcd(
                dcd,
                tmpl_model,
                chunk=1,
                stride=int(stride),
                atom_indices=atom_indices,
            )

            try:
                _, b0_raw = next(it)
            except StopIteration as exc:
                raise ValueError(f"DCD appears to have no frames: {dcd}") from exc

            if b0_raw is None:
                if box_fallback is None:
                    raise ValueError(
                        "DCD does not include unit cell lengths; " "pass box_nm=(Lx,Ly,Lz) in nm"
                    )
                b0 = box_fallback
            else:
                b0 = _box_lengths_nm(b0_raw)

            half_boxes.append(0.5 * float(np.min(b0)))

        r_max = float(min(half_boxes))
    else:
        r_max = float(r_max_nm)

    if r_max <= 0.0:
        raise ValueError("r_max_nm must be > 0")

    r_edges = np.arange(
        0.0,
        r_max + float(dr_nm),
        float(dr_nm),
        dtype=np.float64,
    )

    if r_edges.size < 2:
        raise ValueError("invalid r_max/dr combination")

    r_edges[-1] = r_max

    n_bins = int(r_edges.size - 1)

    shell_vol = (4.0 * math.pi / 3.0) * (np.power(r_edges[1:], 3) - np.power(r_edges[:-1], 3))

    gr_blocks: list[np.ndarray] = []
    kb_blocks: list[np.ndarray] = []
    b2_blocks: list[np.ndarray] = []

    frames_per_block: list[int] = []
    min_half_boxes: list[float] = []

    particles_per_frame_all: list[int] = []
    pairs_per_frame_all: list[int] = []
    cluster_sizes_kept_all: list[np.ndarray] = []

    global_frame_index = 0

    for dcd in dcd_list:
        hist = np.zeros(n_bins, dtype=np.float64)
        norm = np.zeros(n_bins, dtype=np.float64)

        n_frames_block = 0
        min_half_box_block = float("inf")

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

            if global_frame_index >= len(clusters_by_frame):
                raise ValueError(
                    "More DCD frames were encountered than clusters_out contains. "
                    "Check that dcd_files, stride, frame_start, and frame_stop match "
                    "the clusters_from_dcd() call."
                )

            clusters = clusters_by_frame[global_frame_index]
            global_frame_index += 1

            if box_frame_nm is None:
                if box_fallback is None:
                    raise ValueError(
                        "DCD does not include unit cell lengths; " "pass box_nm=(Lx,Ly,Lz) in nm"
                    )
                b = box_fallback
            else:
                b = _box_lengths_nm(box_frame_nm)

            if np.any(b <= 0.0):
                raise ValueError("box lengths must be positive")

            min_half_box_block = min(
                min_half_box_block,
                0.5 * float(np.min(b)),
            )

            vol = float(b[0] * b[1] * b[2])

            if vol <= 0.0:
                raise ValueError("non-positive box volume")

            # Protein centers first, using the same machinery as rdf_from_dcd().
            protein_centers = group_centers_nm(
                xyz_nm,
                groups,
                masses=masses_sel,
                box_nm=b,
                center=center_mode,
                unwrap=bool(unwrap_proteins),
                wrap=True,
            )

            cluster_centers, cluster_sizes = _cluster_centers_from_protein_centers(
                protein_centers,
                clusters,
                b,
                min_cluster_size=int(min_cluster_size),
            )

            n_particles = int(cluster_centers.shape[0])
            n_pairs = n_particles * (n_particles - 1) // 2

            particles_per_frame_all.append(n_particles)
            pairs_per_frame_all.append(n_pairs)
            cluster_sizes_kept_all.append(cluster_sizes)

            if n_pairs > 0:
                r = _pair_distances_nm(cluster_centers, b)
                h, _ = np.histogram(r, bins=r_edges)

                hist += h.astype(np.float64)

                # Ideal-gas expected counts for this frame.
                norm += float(n_pairs) * shell_vol / vol

            n_frames_block += 1

        if n_frames_block <= 0:
            raise ValueError(f"no frames selected for DCD block: {dcd}")

        if np.all(norm <= 0.0):
            raise ValueError(
                "No frame contained at least two cluster particles. "
                "Try min_cluster_size=1, or check the clustering output."
            )

        g_r = np.zeros_like(hist)
        ok = norm > 0.0
        g_r[ok] = hist[ok] / norm[ok]

        kb = _kb_from_gr(g_r, r_edges)
        b2 = -0.5 * kb

        gr_blocks.append(g_r)
        kb_blocks.append(kb)
        b2_blocks.append(b2)

        frames_per_block.append(int(n_frames_block))
        min_half_boxes.append(float(min_half_box_block))

    if global_frame_index != len(clusters_by_frame):
        raise ValueError(
            "clusters_out contains more frames than were read from the DCDs. "
            f"Used {global_frame_index} frames, but clusters_out has "
            f"{len(clusters_by_frame)}. Check stride/frame_start/frame_stop."
        )

    # Truncate to the safe radius if the box fluctuated.
    r_keep = min(float(r_max), float(min(min_half_boxes)))
    n_keep = int(np.searchsorted(r_edges, r_keep, side="right") - 1)

    if n_keep < 1:
        raise ValueError("box is too small for the requested r_max_nm/dr_nm")

    r_edges_keep = r_edges[: n_keep + 1]
    r_nm = 0.5 * (r_edges_keep[:-1] + r_edges_keep[1:])

    gr_arr = np.stack([g[:n_keep] for g in gr_blocks], axis=0)
    kb_arr = np.stack([k[:n_keep] for k in kb_blocks], axis=0)
    b2_arr = np.stack([b[:n_keep] for b in b2_blocks], axis=0)

    gr_mean = np.mean(gr_arr, axis=0)
    kb_mean = np.mean(kb_arr, axis=0)
    b2_mean = np.mean(b2_arr, axis=0)

    n_blocks = int(gr_arr.shape[0])

    if n_blocks < 2:
        gr_err = np.zeros_like(gr_mean)
        kb_err = np.zeros_like(kb_mean)
        b2_err = np.zeros_like(b2_mean)
        b2_final_err = 0.0
    else:
        denom_blocks = math.sqrt(float(n_blocks))

        gr_err = np.std(gr_arr, axis=0, ddof=1) / denom_blocks
        kb_err = np.std(kb_arr, axis=0, ddof=1) / denom_blocks
        b2_err = np.std(b2_arr, axis=0, ddof=1) / denom_blocks
        b2_final_err = float(np.std(b2_arr[:, -1], ddof=1) / denom_blocks)

    tail_norm_factor = 1.0

    if tail_norm_range_nm is not None:
        rlo, rhi = tail_norm_range_nm
        tail_sel = (r_nm >= float(rlo)) & (r_nm <= float(rhi))

        if int(np.sum(tail_sel)) < 1:
            raise ValueError("tail_norm_range_nm selects no RDF points")

        tail_norm_factor = float(np.nanmean(gr_mean[tail_sel]))

        if not np.isfinite(tail_norm_factor) or tail_norm_factor <= 0.0:
            raise ValueError("invalid tail normalization factor")

        gr_mean = gr_mean / tail_norm_factor
        gr_err = gr_err / tail_norm_factor

        # Recompute KB/B2 from the renormalized g(r).
        kb_mean = _kb_from_gr(gr_mean, r_edges_keep)
        b2_mean = -0.5 * kb_mean

        if n_blocks < 2:
            kb_err = np.zeros_like(kb_mean)
            b2_err = np.zeros_like(b2_mean)
            b2_final_err = 0.0
        else:
            gr_arr_norm = gr_arr / tail_norm_factor
            kb_arr = np.stack([_kb_from_gr(g, r_edges_keep) for g in gr_arr_norm], axis=0)
            b2_arr = -0.5 * kb_arr

            denom_blocks = math.sqrt(float(n_blocks))
            kb_err = np.std(kb_arr, axis=0, ddof=1) / denom_blocks
            b2_err = np.std(b2_arr, axis=0, ddof=1) / denom_blocks
            b2_final_err = float(np.std(b2_arr[:, -1], ddof=1) / denom_blocks)

    return {
        "r_nm": r_nm,
        "r_edges_nm": r_edges_keep,
        "g_r": gr_mean,
        "g_r_err": gr_err,
        "kb_nm3": kb_mean,
        "kb_nm3_err": kb_err,
        "b2_r_nm3": b2_mean,
        "b2_r_nm3_err": b2_err,
        "b2_nm3": float(b2_mean[-1]),
        "b2_nm3_err": float(b2_final_err),
        "tail_norm_factor": float(tail_norm_factor),
        "tail_norm_range_nm": tail_norm_range_nm,
        "n_blocks": n_blocks,
        "frames_per_block": np.asarray(frames_per_block, dtype=np.int64),
        "particles_per_frame": np.asarray(particles_per_frame_all, dtype=np.int64),
        "pairs_per_frame": np.asarray(pairs_per_frame_all, dtype=np.int64),
        "cluster_sizes_by_frame": cluster_sizes_kept_all,
        "selection": selection,
        "center": center_mode,
        "unwrap_proteins": bool(unwrap_proteins),
        "min_cluster_size": int(min_cluster_size),
        "stride": int(stride),
    }

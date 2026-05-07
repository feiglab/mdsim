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
from openmm.unit import Quantity, dimensionless, nanometer

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
    )


# --- binding/contact probability analysis ------------------------------------


AVOGADRO = 6.02214076e23
R_KJ_MOL_K = 0.00831446261815324


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
    """

    kind: str
    cutoff_nm: float = 1.0
    contacts: tuple[SpecificContact, ...] = ()
    min_contacts: int = 1


@dataclass(frozen=True)
class BindingAffinityResult:
    pair_i: np.ndarray
    pair_j: np.ndarray
    bound: np.ndarray
    metric_nm: np.ndarray
    exclusive_count: np.ndarray
    box_volume_nm3: np.ndarray
    criterion: BindingCriterion
    n_a: int
    n_b: int

    @property
    def n_frames(self) -> int:
        return int(self.bound.shape[0])

    @property
    def pair_probability(self) -> np.ndarray:
        return np.mean(self.bound, axis=0)

    @property
    def any_bound_probability(self) -> float:
        return float(np.any(self.bound, axis=1).mean())

    @property
    def mean_bound_pairs(self) -> float:
        return float(self.bound.sum(axis=1).mean())

    @property
    def mean_exclusive_complexes(self) -> float:
        return float(self.exclusive_count.mean())

    def kd_molar(self) -> float:
        """Estimate Kd for A + B <-> AB from exclusive complex count."""

        n_ab = self.mean_exclusive_complexes
        if n_ab <= 0.0:
            return float("inf")

        volume_nm3 = float(np.nanmean(self.box_volume_nm3))
        if volume_nm3 <= 0.0 or not np.isfinite(volume_nm3):
            raise ValueError("valid box volume is required for Kd")

        n_a_free = max(float(self.n_a) - n_ab, 0.0)
        n_b_free = max(float(self.n_b) - n_ab, 0.0)
        volume_l = volume_nm3 * 1.0e-24
        return n_a_free * n_b_free / (n_ab * AVOGADRO * volume_l)

    def delta_g_kj_mol(self, temperature_k: float) -> float:
        kd = self.kd_molar()
        if kd <= 0.0 or not np.isfinite(kd):
            return float("inf")
        return R_KJ_MOL_K * float(temperature_k) * math.log(kd)


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
    """Estimate binding probability and apparent Kd from DCD contacts.

    Parameters
    ----------
    pdb_file
        Template PDB matching the DCD atom order.
    dcd_files
        One DCD or a sequence of DCDs.
    group_a, group_b
        Selection strings or explicit atom-index groups. If group_b is None,
        group_a is self-paired with duplicate/self pairs removed.
    criterion
        BindingCriterion("cofm"), BindingCriterion("closest"), or
        BindingCriterion("specific").
    center
        "cog" or "com"; used for cofm only.
    exclusive
        If True, greedily counts at most one partner per A and B molecule per frame.
        Use this for Kd-like estimates.
    unwrap_groups
        If True, unwrap atoms within each molecule before center calculations.
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
                n_complex = _binding_exclusive_count(bound, metric, pair_i, pair_j)
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
    )


def save_binding_affinity(prefix: FileLike, result: BindingAffinityResult) -> None:
    """Write pair probabilities and frame time series."""

    p = Path(prefix)

    pair_data = np.column_stack(
        (
            result.pair_i,
            result.pair_j,
            result.pair_probability,
            np.nanmean(result.metric_nm, axis=0),
        )
    )
    np.savetxt(
        p.with_suffix(".pairs.dat"),
        pair_data,
        fmt=["%d", "%d", "%.8f", "%.8f"],
        header="pair_i pair_j p_bound mean_metric_nm",
    )

    ts_data = np.column_stack(
        (
            np.arange(result.n_frames, dtype=np.int64),
            result.bound.sum(axis=1),
            result.exclusive_count,
            np.any(result.bound, axis=1).astype(np.int32),
        )
    )
    np.savetxt(
        p.with_suffix(".timeseries.dat"),
        ts_data,
        fmt=["%d", "%d", "%.8f", "%d"],
        header="frame bound_pairs exclusive_complexes any_bound",
    )


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
) -> float:
    edges = np.flatnonzero(bound)
    if edges.size == 0:
        return 0.0

    order = edges[np.argsort(metric[edges])]
    used_i: set[int] = set()
    used_j: set[int] = set()
    count = 0

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

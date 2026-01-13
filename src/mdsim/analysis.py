from __future__ import annotations

import io
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from openmm.app import Topology
from openmm.unit import Quantity, nanometer

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
            xyz = [float(Quantity(x).value_in_unit(nanometer)) for x in p]
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
    """Same as plane_normal, but returns a Quantity with length units (nm)."""
    n = plane_normal(points, **kwargs)
    return Quantity(n, nanometer)


# --- structure factors and virial coefficients -------------------------------


def _box_lengths_nm(box: Any) -> np.ndarray:
    """
    Return orthorhombic box lengths (Lx, Ly, Lz) in nm as float64.

    Accepts:
      - (3,) sequence of floats in nm
      - OpenMM Quantity of shape (3,) in length units
      - OpenMM periodic box vectors (3x3) as Quantity or array-like (orthorhombic only)
    """
    if box is None:
        raise ValueError("box must be provided")

    if isinstance(box, Quantity):
        arr = np.asarray(box.value_in_unit(nanometer), dtype=float)
    else:
        arr = np.asarray(box, dtype=float)

    arr = np.asarray(arr, dtype=float)

    if arr.shape == (3,):
        return arr.astype(np.float64, copy=False)

    if arr.shape == (3, 3):
        # interpret as box vectors; require orthorhombic
        off = arr.copy()
        off[np.diag_indices(3)] = 0.0
        if float(np.max(np.abs(off))) > 1.0e-6:
            raise ValueError("triclinic boxes are not supported (need orthorhombic)")
        return np.abs(np.diag(arr)).astype(np.float64, copy=False)

    raise ValueError(f"box has shape {arr.shape}; expected (3,) or (3,3)")


def _unwrap_group_nm(x_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    """
    Unwrap a single molecule/group into a contiguous image (orthorhombic PBC).

    x_nm
        Coordinates for this group, shape (n, 3), nm.
    box_nm
        Box lengths (Lx, Ly, Lz), shape (3,), nm.
    """
    ref = x_nm[0]
    d = x_nm - ref
    d -= np.rint(d / box_nm) * box_nm
    return ref + d


def _center_group_nm(
    xyz_nm: np.ndarray,
    idx: np.ndarray,
    *,
    masses: Optional[np.ndarray],
    center: str,
    box_nm: Optional[np.ndarray],
    unwrap: bool,
) -> np.ndarray:
    if idx.size == 0:
        raise ValueError("group is empty")

    x = xyz_nm[idx, :].astype(np.float64, copy=False)

    if unwrap:
        if box_nm is None:
            raise ValueError("unwrap=True requires a box")
        x = _unwrap_group_nm(x, box_nm)

    mode = (center or "com").lower()
    if mode == "cog":
        return x.mean(axis=0)

    if mode != "com":
        raise ValueError("center must be 'com' or 'cog'")

    if masses is None:
        w = np.ones((idx.size,), dtype=np.float64)
    else:
        w = masses[idx].astype(np.float64, copy=False)
        if not np.all(np.isfinite(w)):
            raise ValueError("non-finite masses encountered")
        if float(np.sum(w)) <= 0.0:
            raise ValueError("total mass is zero")

    return (x * w[:, None]).sum(axis=0) / float(np.sum(w))


def solute_groups(
    template: Any,
    *,
    group_spec: str = "protein",
    model_index: int = 0,
) -> list[np.ndarray]:
    """
    Build solute atom-index groups (0-based) using your StructureSelector.

    Typical use:
      - group_spec="protein" -> one group per chain containing protein residues.
      - group_spec=["A:B.protein", "C:D.protein"] -> explicit groups for multi-chain proteins.
    """
    from .molecule_data import Model, PDBReader, Structure, StructureSelector

    if isinstance(template, Model):
        model = template
    elif isinstance(template, Structure):
        model = template.get_model(model_index)
    else:
        model = PDBReader(template).get_model(model_index)

    groups = StructureSelector(group_spec).atom_lists(model)
    out = [np.asarray(g, dtype=np.int64) for g in groups if len(g) > 0]
    if not out:
        raise ValueError(f"group_spec={group_spec!r} produced no atoms")
    return out


def _atom_masses(template: Any, *, model_index: int = 0) -> np.ndarray:
    from .molecule_data import Model, PDBReader, Structure

    if isinstance(template, Model):
        model = template
    elif isinstance(template, Structure):
        model = template.get_model(model_index)
    else:
        model = PDBReader(template).get_model(model_index)

    masses = np.empty((model.natoms(),), dtype=np.float64)
    for i, a in enumerate(model.atoms):
        m = getattr(a, "mass", None)
        masses[i] = float(m) if m is not None else 0.0
    return masses


def group_centers_nm(
    coords_nm: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    masses: Optional[np.ndarray] = None,
    box_nm: Optional[np.ndarray] = None,
    boxes_nm: Optional[np.ndarray] = None,
    center: str = "com",
    unwrap: bool = True,
    wrap: bool = True,
) -> np.ndarray:
    """
    Compute per-frame centers for multiple groups.

    Parameters
    ----------
    coords_nm
        Shape (n_frames, n_atoms, 3) in nm.
    groups
        List of index arrays (one per solute copy).
    masses
        Optional per-atom masses (dalton-like), shape (n_atoms,). Required for COM.
    box_nm
        Constant box lengths in nm, shape (3,). Used if boxes_nm is None.
    boxes_nm
        Optional per-frame box lengths in nm, shape (n_frames, 3).
    center
        "com" or "cog".
    unwrap
        If True, unwrap each group under PBC before computing the center.
    wrap
        If True and a box is available, wrap output centers back into [0,L).
    """
    coords_nm = np.asarray(coords_nm, dtype=np.float64)
    if coords_nm.ndim != 3 or coords_nm.shape[2] != 3:
        raise ValueError("coords_nm must have shape (n_frames, n_atoms, 3)")

    n_frames = int(coords_nm.shape[0])
    n_groups = len(groups)
    out = np.empty((n_frames, n_groups, 3), dtype=np.float64)

    if boxes_nm is not None:
        boxes_nm = np.asarray(boxes_nm, dtype=np.float64)
        if boxes_nm.shape != (n_frames, 3):
            raise ValueError("boxes_nm must have shape (n_frames, 3)")

    if unwrap and boxes_nm is None and box_nm is None:
        raise ValueError("unwrap=True requires box_nm or boxes_nm")

    if box_nm is not None:
        box_nm = np.asarray(box_nm, dtype=np.float64).reshape(3)

    for fi in range(n_frames):
        xyz = coords_nm[fi]
        b = boxes_nm[fi] if boxes_nm is not None else box_nm
        for gi, idx in enumerate(groups):
            c = _center_group_nm(
                xyz,
                idx,
                masses=masses,
                center=center,
                box_nm=b,
                unwrap=unwrap,
            )
            if wrap and b is not None:
                c = c - np.floor(c / b) * b
            out[fi, gi, :] = c

    return out


def q_vectors_orthorhombic(
    box_nm: Any,
    *,
    q_max: Optional[float] = None,
    n_max: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enumerate reciprocal-space wavevectors for an orthorhombic box.

    q = 2*pi*(nx/Lx, ny/Ly, nz/Lz), excluding (0,0,0)

    Parameters
    ----------
    box_nm
        Box lengths (Lx,Ly,Lz) in nm.
    q_max
        Optional upper bound on |q| (1/nm). If provided, filters vectors by magnitude.
    n_max
        Optional integer triplet (nx_max, ny_max, nz_max). If omitted:
          - if q_max is provided, derived from q_max and box lengths
          - else defaults to (8, 8, 8)

    Returns
    -------
    q_vec
        (n_q, 3) float64 in 1/nm.
    q_mag
        (n_q,) float64 magnitudes in 1/nm.
    """
    box = _box_lengths_nm(box_nm)
    if np.any(box <= 0):
        raise ValueError("box lengths must be positive")

    two_pi = float(2.0 * np.pi)

    if n_max is None:
        if q_max is None:
            n_max = (8, 8, 8)
        else:
            q_max_f = float(q_max)
            n_max = tuple(int(np.ceil(q_max_f * float(L) / two_pi)) for L in box)
    if len(n_max) != 3:
        raise ValueError("n_max must be a 3-tuple")

    nx = np.arange(-int(n_max[0]), int(n_max[0]) + 1, dtype=np.int32)
    ny = np.arange(-int(n_max[1]), int(n_max[1]) + 1, dtype=np.int32)
    nz = np.arange(-int(n_max[2]), int(n_max[2]) + 1, dtype=np.int32)

    g = np.stack(np.meshgrid(nx, ny, nz, indexing="ij"), axis=-1).reshape(-1, 3)
    g = g[np.any(g != 0, axis=1)]

    q = (two_pi * g.astype(np.float64)) / box.reshape(1, 3)
    q_mag = np.linalg.norm(q, axis=1)

    if q_max is not None:
        keep = q_mag <= float(q_max) + 1.0e-12
        q = q[keep]
        q_mag = q_mag[keep]

    order = np.argsort(q_mag, kind="mergesort")
    return q[order], q_mag[order]


class SqAccumulator:
    """
    Incremental accumulator for S(q) over frames.

    S(q) = < |sum_j exp(i q·r_j)|^2 / N >
    """

    def __init__(self, q_vec: np.ndarray):
        q_vec = np.asarray(q_vec, dtype=np.float64)
        if q_vec.ndim != 2 or q_vec.shape[1] != 3:
            raise ValueError("q_vec must have shape (n_q, 3)")
        self.q_vec = q_vec
        self.sum_sq = np.zeros((q_vec.shape[0],), dtype=np.float64)
        self.sum_sq2 = np.zeros((q_vec.shape[0],), dtype=np.float64)
        self.n_frames = 0

    def add(self, centers_nm: np.ndarray) -> None:
        centers_nm = np.asarray(centers_nm, dtype=np.float64)
        if centers_nm.ndim == 2:
            centers_nm = centers_nm.reshape(1, *centers_nm.shape)

        if centers_nm.ndim != 3 or centers_nm.shape[2] != 3:
            raise ValueError("centers_nm must have shape (n_frames, n_particles, 3)")

        n_frames = int(centers_nm.shape[0])
        n_part = int(centers_nm.shape[1])
        if n_part <= 0:
            raise ValueError("need at least one particle center per frame")

        phase = np.einsum("fpc,qc->fpq", centers_nm, self.q_vec, optimize=True)
        re = np.cos(phase).sum(axis=1)
        im = np.sin(phase).sum(axis=1)
        sq = (re * re + im * im) / float(n_part)

        self.sum_sq += sq.sum(axis=0)
        self.sum_sq2 += (sq * sq).sum(axis=0)
        self.n_frames += n_frames

    def mean(self) -> np.ndarray:
        if self.n_frames <= 0:
            raise ValueError("no frames accumulated")
        return self.sum_sq / float(self.n_frames)

    def stderr(self) -> np.ndarray:
        if self.n_frames <= 1:
            return np.full_like(self.sum_sq, np.nan)
        n = float(self.n_frames)
        mean = self.sum_sq / n
        mean2 = self.sum_sq2 / n
        var = np.maximum(0.0, mean2 - mean * mean)
        return np.sqrt(var / n)


def radial_average_sq(
    q_mag: np.ndarray,
    sq: np.ndarray,
    *,
    dq: float,
) -> dict[str, np.ndarray]:
    """
    Radially average S(q) into bins of width dq.
    """
    q_mag = np.asarray(q_mag, dtype=np.float64).reshape(-1)
    sq = np.asarray(sq, dtype=np.float64).reshape(-1)
    if q_mag.shape != sq.shape:
        raise ValueError("q_mag and sq must have the same shape")
    if float(dq) <= 0:
        raise ValueError("dq must be positive")

    q_max = float(np.max(q_mag)) if q_mag.size else 0.0
    n_bins = int(np.floor(q_max / float(dq))) + 1
    if n_bins <= 0:
        raise ValueError("no bins")

    idx = np.floor(q_mag / float(dq)).astype(np.int64)
    idx = np.clip(idx, 0, n_bins - 1)

    sums = np.zeros((n_bins,), dtype=np.float64)
    counts = np.zeros((n_bins,), dtype=np.int64)
    np.add.at(sums, idx, sq)
    np.add.at(counts, idx, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg = sums / counts

    q_cent = (np.arange(n_bins, dtype=np.float64) + 0.5) * float(dq)
    mask = counts > 0
    return {
        "q": q_cent[mask],
        "sq": avg[mask],
        "counts": counts[mask],
    }


def fit_s0_ornstein_zernike(
    q: np.ndarray,
    sq: np.ndarray,
    *,
    q_max: Optional[float] = None,
    min_points: int = 4,
) -> dict[str, float]:
    """
    Estimate S(0) via Ornstein-Zernike: 1/S(q) = a + b q^2.

    Returns S0=1/a and xi=sqrt(b/a) (nm) when b,a>0.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    sq = np.asarray(sq, dtype=np.float64).reshape(-1)
    if q.shape != sq.shape:
        raise ValueError("q and sq must have the same shape")

    mask = np.isfinite(q) & np.isfinite(sq) & (q > 0) & (sq > 0)
    if q_max is not None:
        mask &= q <= float(q_max)

    qf = q[mask]
    sf = sq[mask]
    if qf.size < int(min_points):
        raise ValueError("not enough points for fit")

    x = qf * qf
    y = 1.0 / sf

    A = np.stack([np.ones_like(x), x], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    if a <= 0.0:
        raise ValueError("fit produced non-positive intercept (cannot compute S0)")

    s0 = 1.0 / a
    xi = float(np.sqrt(b / a)) if b > 0.0 else float("nan")
    return {"s0": float(s0), "xi_nm": xi, "a": a, "b": b}


def number_density_nm3(n_particles: int, box_nm: Any) -> float:
    """
    Number density rho in 1/nm^3 from N particles in a box (orthorhombic).
    """
    box = _box_lengths_nm(box_nm)
    vol = float(box[0] * box[1] * box[2])
    if vol <= 0.0:
        raise ValueError("box volume must be positive")
    return float(n_particles) / vol


def b2_from_s0(rho_nm3: float, s0: float) -> float:
    """
    Osmotic second virial coefficient B2 (nm^3) from S(0) and rho.

    Uses: 1/S(0) = 1 + 2 B2 rho + O(rho^2)
    """
    rho = float(rho_nm3)
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    s0f = float(s0)
    if s0f <= 0.0:
        raise ValueError("S0 must be positive")
    return (1.0 / s0f - 1.0) / (2.0 * rho)


def fit_virial_coefficients_from_s0(
    rho_nm3: Sequence[float],
    s0: Sequence[float],
    *,
    max_order: int = 3,
) -> dict[str, float]:
    """
    Fit virial coefficients from S(0) vs density using:

      1/S0 = 1 + 2 B2 rho + 3 B3 rho^2 + 4 B4 rho^3 + ...

    Returns B2..Bmax_order in nm^(3*(k-1)).
    """
    rho = np.asarray(rho_nm3, dtype=np.float64).reshape(-1)
    s0a = np.asarray(s0, dtype=np.float64).reshape(-1)
    if rho.shape != s0a.shape:
        raise ValueError("rho and s0 must have the same length")
    if int(max_order) < 2:
        raise ValueError("max_order must be >= 2")

    y = 1.0 / s0a - 1.0
    cols = []
    names = []
    for k in range(2, int(max_order) + 1):
        cols.append(float(k) * (rho ** (k - 1)))
        names.append(f"B{k}")

    X = np.stack(cols, axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    out: dict[str, float] = {}
    for name, c in zip(names, coef):
        out[name] = float(c)
    return out


def structure_factor_from_dcd(
    dcd_file: Any,
    template: Any,
    *,
    group_spec: str = "protein",
    groups: Optional[Sequence[Sequence[int]]] = None,
    center: str = "com",
    unwrap: bool = True,
    q_max: float = 3.0,
    dq: float = 0.05,
    n_max: Optional[Sequence[int]] = None,
    stride: int = 1,
    chunk: int = 200,
    box_nm: Any = None,
) -> dict[str, Any]:
    """
    Compute solute structure factor S(q) from a DCD trajectory.

    - Reads the template via your PDBReader when `template` is a file/path.
    - Streams the DCD via your iter_dcd() (mdtraj backend).
    - Groups solute copies using StructureSelector(group_spec) unless `groups` is provided.
    - Uses orthorhombic reciprocal vectors, radially averaged with bin width dq.
    """
    from .molecule_data import Model, PDBReader, Structure, iter_dcd

    if isinstance(template, Model):
        tmpl = template
    elif isinstance(template, Structure):
        tmpl = template.model
    else:
        tmpl = PDBReader(template).model

    if groups is None:
        grp = solute_groups(tmpl, group_spec=group_spec)
    else:
        grp = [np.asarray(g, dtype=np.int64) for g in groups]

    masses = _atom_masses(tmpl)

    it = iter_dcd(dcd_file, tmpl, chunk=int(chunk), stride=int(stride))
    try:
        coords0, boxes0 = next(it)
    except StopIteration as exc:
        raise ValueError("empty DCD stream") from exc

    if box_nm is not None:
        box_ref = _box_lengths_nm(box_nm)
    elif boxes0 is not None:
        box_ref = np.mean(boxes0, axis=0).astype(np.float64, copy=False)
    else:
        raise ValueError("box_nm must be provided (DCD has no unit cell)")

    q_vec, q_mag = q_vectors_orthorhombic(box_ref, q_max=float(q_max), n_max=n_max)
    acc = SqAccumulator(q_vec)

    c0 = group_centers_nm(
        coords0,
        grp,
        masses=masses,
        box_nm=box_ref,
        boxes_nm=boxes0,
        center=center,
        unwrap=unwrap,
        wrap=True,
    )
    acc.add(c0)

    for coords, boxes in it:
        cc = group_centers_nm(
            coords,
            grp,
            masses=masses,
            box_nm=box_ref,
            boxes_nm=boxes,
            center=center,
            unwrap=unwrap,
            wrap=True,
        )
        acc.add(cc)

    sq = acc.mean()
    err = acc.stderr()
    binned = radial_average_sq(q_mag, sq, dq=float(dq))

    return {
        "box_nm": box_ref,
        "q_vec": q_vec,
        "q_mag": q_mag,
        "sq": sq,
        "sq_stderr": err,
        "q_bin": binned["q"],
        "sq_bin": binned["sq"],
        "counts_bin": binned["counts"],
        "n_frames": int(acc.n_frames),
        "n_particles": int(len(grp)),
        "rho_nm3": number_density_nm3(len(grp), box_ref),
    }

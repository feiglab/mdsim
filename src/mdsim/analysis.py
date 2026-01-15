from __future__ import annotations

import io
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from openmm.app import Topology
from openmm.unit import Quantity, nanometer

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


# Back-compat for a typo in earlier drafts.
_box_lengthts_nm = _box_lengths_nm


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
            b = np.asarray(box_frame_nm, dtype=np.float64).reshape(3)

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
    selection: Union[str, Sequence[Sequence[int]]] = "protein",
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

    if isinstance(selection, str):
        groups_global = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_global = [[int(i) for i in g] for g in selection]

    groups_global = [g for g in groups_global if g]
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


def _sq_single_dcd(
    dcd_file: FileLike,
    template_model: Any,
    *,
    atom_indices: Sequence[int],
    groups: Sequence[np.ndarray],
    masses: Optional[np.ndarray],
    center: str,
    unwrap: bool,
    q_nm1: np.ndarray,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_nm: Optional[Sequence[float]],
) -> tuple[np.ndarray, int]:
    n_groups = int(len(groups))
    if n_groups < 2:
        raise ValueError("need >=2 groups to compute S(q)")

    acc = np.zeros_like(q_nm1, dtype=np.float64)
    n_frames = 0

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
            b = np.asarray(box_frame_nm, dtype=np.float64).reshape(3)

        if np.any(b <= 0.0):
            raise ValueError("box lengths must be positive")

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

        qr = np.outer(r, q_nm1)
        sum_sinc = np.sum(_sinc(qr), axis=0)
        s_frame = 1.0 + (2.0 / float(n_groups)) * sum_sinc
        acc += s_frame
        n_frames += 1

    if n_frames <= 0:
        raise ValueError("no frames selected for S(q) computation")
    return acc / float(n_frames), int(n_frames)


def _q_vectors_cubic_or_ortho(box_nm: np.ndarray, q_max_nm1: float):
    Lx, Ly, Lz = [float(x) for x in box_nm]
    # integer bounds
    nx_max = int(math.floor(q_max_nm1 * Lx / (2.0 * math.pi)))
    ny_max = int(math.floor(q_max_nm1 * Ly / (2.0 * math.pi)))
    nz_max = int(math.floor(q_max_nm1 * Lz / (2.0 * math.pi)))

    qvecs = []
    qmag = []
    for nx in range(-nx_max, nx_max + 1):
        for ny in range(-ny_max, ny_max + 1):
            for nz in range(-nz_max, nz_max + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                qx = 2.0 * math.pi * nx / Lx
                qy = 2.0 * math.pi * ny / Ly
                qz = 2.0 * math.pi * nz / Lz
                qm = math.sqrt(qx * qx + qy * qy + qz * qz)
                if qm <= q_max_nm1:
                    qvecs.append((qx, qy, qz))
                    qmag.append(qm)
    qvecs = np.asarray(qvecs, dtype=np.float64)  # (nq,3)
    qmag = np.asarray(qmag, dtype=np.float64)  # (nq,)
    return qvecs, qmag


def structure_factor_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[Sequence[int]]] = "protein",
    center: str = "cog",
    unwrap: bool = True,
    q_nm1: Optional[np.ndarray] = None,
    q_min_nm1: Optional[float] = None,
    q_max_nm1: float = 20.0,
    n_q: int = 200,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Isotropically averaged S(q) from solute-group centers (Debye formula).

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

    if isinstance(selection, str):
        groups_global = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_global = [[int(i) for i in g] for g in selection]

    groups_global = [g for g in groups_global if g]
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

    if q_nm1 is None:
        b0 = _peek_first_box_nm(
            dcd_list[0],
            tmpl_model,
            atom_indices,
            int(stride),
            box_nm=box_nm,
        )
        q0 = 2.0 * math.pi / float(np.min(b0))
        qmin = float(q0 if q_min_nm1 is None else q_min_nm1)
        qmax = float(q_max_nm1)
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        if qmax <= qmin:
            raise ValueError("q_max_nm1 must be > q_min_nm1")
        q = np.linspace(qmin, qmax, int(n_q), dtype=np.float64)
        q_info = {"q_box_min_nm1": float(q0)}
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")
        q_info = {}

    sq_blocks: list[np.ndarray] = []
    frames_per_block: list[int] = []

    for dcd in dcd_list:
        sq, n_frames = _sq_single_dcd(
            dcd,
            tmpl_model,
            atom_indices=atom_indices,
            groups=groups,
            masses=masses_sel,
            center=center_mode,
            unwrap=bool(unwrap),
            q_nm1=q,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=int(frame_start),
            frame_stop=frame_stop,
            box_nm=box_nm,
        )
        sq_blocks.append(sq)
        frames_per_block.append(int(n_frames))

    sq_arr = np.stack(sq_blocks, axis=0)
    sq_mean = np.mean(sq_arr, axis=0)

    n_blocks = int(sq_arr.shape[0])
    if n_blocks < 2:
        sq_err = np.zeros_like(sq_mean)
    else:
        sq_err = np.std(sq_arr, axis=0, ddof=1) / math.sqrt(float(n_blocks))

    out: dict[str, Any] = {
        "q_nm1": q,
        "q": q,  # alias
        "sq": sq_mean,
        "sq_err": sq_err,
        "sq_stderr": sq_err,  # alias
        "n_blocks": n_blocks,
        "frames_per_block": np.asarray(frames_per_block, dtype=np.int64),
        "selection": selection,
        "center": center_mode,
        "unwrap": bool(unwrap),
        "stride": int(stride),
    }
    out.update(q_info)
    return out


def structure_factor_from_dcd_npt_debye(
    dcd_file: Any,
    template: Any,
    *,
    group_spec: str = "protein",
    groups: Optional[Sequence[Sequence[int]]] = None,
    center: str = "com",
    unwrap: bool = True,
    q_min: float = 0.0,
    q_max: float = 3.0,
    dq: float = 0.05,
    stride: int = 1,
    chunk: int = 200,
    box_nm: Any = None,
    pair_block: int = 20000,
) -> dict[str, Any]:
    """Back-compat wrapper (older name/signature).

    Parameters follow the legacy helper that computed S(q) in NPT using the Debye
    expression. Internally delegates to structure_factor_from_dcd().
    """
    del pair_block

    q = np.arange(float(q_min), float(q_max) + 0.5 * float(dq), float(dq), dtype=np.float64)
    sel: Union[str, Sequence[Sequence[int]]] = group_spec if groups is None else groups

    return structure_factor_from_dcd(
        pdb_file=template,
        dcd_files=dcd_file,
        selection=sel,
        center=center,
        unwrap=bool(unwrap),
        q_nm1=q,
        stride=int(stride),
        chunk=int(chunk),
        frame_start=0,
        frame_stop=None,
        box_nm=box_nm,
    )


def _sq_single_dcd_recip(
    dcd_file: FileLike,
    template_model: Any,
    *,
    atom_indices: Sequence[int],
    groups: Sequence[np.ndarray],
    masses: Optional[np.ndarray],
    center: str,
    unwrap: bool,
    q_nm1: np.ndarray,
    stride: int,
    chunk: int,
    frame_start: int,
    frame_stop: Optional[int],
    box_nm: Optional[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Reciprocal-space S(q) estimator for a periodic box:

        S(q) = < |sum_j exp(i q·r_j)|^2 / N >  (binned isotropically by |q|)

    Parameters
    ----------
    q_nm1
        1D array of *bin centers* in nm^-1. Binning is done using midpoints between
        centers as bin edges; values outside the edge range are ignored.

    Returns
    -------
    sq_nm1 : (n_q,) float
        S(q) at the provided bin centers.
    n_qvec_per_bin : (n_q,) int
        Number of reciprocal vectors accumulated into each bin (summed across frames).
        Useful for diagnosing noisy low-q bins.
    n_frames : int
        Number of frames used.
    """
    n_groups = int(len(groups))
    if n_groups < 2:
        raise ValueError("need >=2 groups to compute S(q)")

    q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
    if q.size < 1:
        raise ValueError("q_nm1 must be non-empty")
    if np.any(~np.isfinite(q)):
        raise ValueError("q_nm1 contains non-finite values")

    # Enforce strictly increasing bin centers (required for robust edge construction).
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q_nm1 must be strictly increasing for reciprocal binning")

    # Construct bin edges from centers:
    # edges[0] = q0 - dq0/2 ; edges[i] = (q[i-1]+q[i])/2 ; edges[-1] = q[-1] + dqlast/2
    mid = 0.5 * (q[:-1] + q[1:])
    left = q[0] - 0.5 * (q[1] - q[0])
    right = q[-1] + 0.5 * (q[-1] - q[-2]) if q.size > 1 else q[-1] + 1e-6
    q_edges = np.concatenate(([left], mid, [right])).astype(np.float64)

    # Guard against non-positive left edge due to user q starting very near 0
    # (not mathematically illegal, but it makes binning more intuitive).
    if q_edges[0] < 0.0:
        q_edges[0] = 0.0

    acc = np.zeros_like(q, dtype=np.float64)
    cnt = np.zeros_like(q, dtype=np.int64)  # counts of q-vectors accumulated (across frames)
    n_frames = 0

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
                raise ValueError("DCD lacks unit cell lengths; pass box_nm=(Lx,Ly,Lz) in nm")
            b = box_fallback
        else:
            b = np.asarray(box_frame_nm, dtype=np.float64).reshape(3)

        if np.any(b <= 0.0):
            raise ValueError("box lengths must be positive")

        # One position per group (protein) in [0, L)
        centers = group_centers_nm(
            xyz_nm,
            groups,
            masses=masses,
            box_nm=b,
            center=center,
            unwrap=bool(unwrap),
            wrap=True,
        )  # (n_groups, 3)

        # Build allowed reciprocal vectors for this frame's box (NPT-safe)
        qvecs, qmag = _q_vectors_cubic_or_ortho(b, float(q_edges[-1]))  # include up to max edge
        if qvecs.size == 0:
            n_frames += 1
            continue

        # Keep only vectors within the bin edge range
        m = (qmag >= float(q_edges[0])) & (qmag < float(q_edges[-1]))
        if not np.any(m):
            n_frames += 1
            continue
        qvecs = qvecs[m]
        qmag = qmag[m]

        # rho(q) = sum_j exp(i q·r_j)
        phase = qvecs @ centers.T  # (nq, n_groups)
        rho_re = np.sum(np.cos(phase), axis=1)
        rho_im = np.sum(np.sin(phase), axis=1)
        sq_qvec = (rho_re * rho_re + rho_im * rho_im) / float(n_groups)  # (nq,)

        # Bin by |q| to the provided q centers
        bin_idx = np.searchsorted(q_edges, qmag, side="right") - 1
        ok = (bin_idx >= 0) & (bin_idx < acc.size)
        np.add.at(acc, bin_idx[ok], sq_qvec[ok])
        np.add.at(cnt, bin_idx[ok], 1)

        n_frames += 1

    if n_frames <= 0:
        raise ValueError("no frames selected for S(q) computation")

    # Average over all q-vectors accumulated (across all frames) per bin.
    # This is equivalent to averaging per-frame then across frames only if each frame
    # has the same set of q-vectors; in NPT the set can change slightly, and this
    # weighted average is typically what you want.
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = acc / np.maximum(cnt, 1)

    return sq, cnt, int(n_frames)


def structure_factor_from_dcd_reciprocal(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[Sequence[int]]] = "protein",
    center: str = "cog",
    unwrap: bool = True,
    q_nm1: Optional[np.ndarray] = None,
    q_min_nm1: Optional[float] = None,
    q_max_nm1: float = 20.0,
    n_q: int = 200,
    stride: int = 1,
    chunk: int = 500,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    box_nm: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """
    Isotropically averaged S(q) from solute-group centers using the *reciprocal-space*
    estimator appropriate for periodic boundary conditions.

    Callable like structure_factor_from_dcd().

    Uncertainties
    -------------
    - If multiple DCDs are provided: stderr across DCDs (treating each as a replicate).
    - If only one DCD: uncertainties are returned as 0.

    Notes
    -----
    - q_nm1 is interpreted as desired *bin centers* (must be strictly increasing).
    - If q_nm1 is None: a linear grid is constructed like structure_factor_from_dcd().
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

    if isinstance(selection, str):
        groups_global = StructureSelector(selection).atom_lists(tmpl)
    else:
        groups_global = [[int(i) for i in g] for g in selection]

    groups_global = [g for g in groups_global if g]
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

    # Build the requested q grid exactly like structure_factor_from_dcd()
    if q_nm1 is None:
        b0 = _peek_first_box_nm(
            dcd_list[0],
            tmpl_model,
            atom_indices,
            int(stride),
            box_nm=box_nm,
        )
        q0 = 2.0 * math.pi / float(np.min(b0))
        qmin = float(q0 if q_min_nm1 is None else q_min_nm1)
        qmax = float(q_max_nm1)
        if int(n_q) < 2:
            raise ValueError("n_q must be >= 2")
        if qmax <= qmin:
            raise ValueError("q_max_nm1 must be > q_min_nm1")
        q = np.linspace(qmin, qmax, int(n_q), dtype=np.float64)
        q_info = {"q_box_min_nm1": float(q0)}
    else:
        q = np.asarray(q_nm1, dtype=np.float64).reshape(-1)
        if q.size < 1:
            raise ValueError("q_nm1 must be non-empty")
        q_info = {}

    # Replicate-by-replicate (DCD-by-DCD)
    sq_blocks: list[np.ndarray] = []
    cnt_blocks: list[np.ndarray] = []
    frames_per_block: list[int] = []

    for dcd in dcd_list:
        sq, cnt, n_frames = _sq_single_dcd_recip(
            dcd,
            tmpl_model,
            atom_indices=atom_indices,
            groups=groups,
            masses=masses_sel,
            center=center_mode,
            unwrap=bool(unwrap),
            q_nm1=q,
            stride=int(stride),
            chunk=int(chunk),
            frame_start=int(frame_start),
            frame_stop=frame_stop,
            box_nm=box_nm,
        )
        sq_blocks.append(sq)
        cnt_blocks.append(cnt)
        frames_per_block.append(int(n_frames))

    sq_arr = np.stack(sq_blocks, axis=0)
    sq_mean = np.mean(sq_arr, axis=0)

    n_blocks = int(sq_arr.shape[0])
    if n_blocks < 2:
        sq_err = np.zeros_like(sq_mean)
    else:
        sq_err = np.std(sq_arr, axis=0, ddof=1) / math.sqrt(float(n_blocks))

    # Diagnostics: average number of q-vectors per bin (across replicates).
    cnt_mean = np.mean(np.stack(cnt_blocks, axis=0).astype(np.float64), axis=0)

    out: dict[str, Any] = {
        "q_nm1": q,
        "q": q,  # alias
        "sq": sq_mean,
        "sq_err": sq_err,
        "sq_stderr": sq_err,  # alias
        "n_blocks": n_blocks,
        "frames_per_block": np.asarray(frames_per_block, dtype=np.int64),
        "selection": selection,
        "center": center_mode,
        "unwrap": bool(unwrap),
        "stride": int(stride),
        "qvecs_per_bin_mean": cnt_mean,
    }
    out.update(q_info)
    return out

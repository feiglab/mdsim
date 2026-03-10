from __future__ import annotations

import base64
import io
import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .analysis import _as_file_list, _box_lengths_nm, _selection_to_groups, atom_masses
from .molecule_data import PDBReader, iter_dcd

FileLike = Union[str, Path, io.BytesIO, io.StringIO]


def _centers_of_groups_whole_chain(
    xyz_nm: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    masses: Optional[np.ndarray],
    center: str,
) -> np.ndarray:
    """
    Per-group centers from coordinates exactly as stored in the current frame.

    No intra-group PBC unwrapping is applied. This assumes each selected
    group/chain is already whole within the frame.
    """
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    mode = str(center).strip().lower()
    if mode not in {"cog", "com"}:
        raise ValueError("center must be 'cog' or 'com'")

    out = np.empty((len(groups), 3), dtype=np.float64)

    if mode == "cog" or masses is None:
        for i, g in enumerate(groups):
            ii = np.asarray(g, dtype=np.int64)
            out[i] = np.mean(xyz[ii, :], axis=0)
        return out

    m = np.asarray(masses, dtype=np.float64)
    for i, g in enumerate(groups):
        ii = np.asarray(g, dtype=np.int64)
        w = m[ii]
        tot = float(np.sum(w))
        if tot <= 0.0:
            out[i] = np.mean(xyz[ii, :], axis=0)
        else:
            out[i] = np.sum(xyz[ii, :] * w[:, None], axis=0) / tot
    return out


def _time_unwrap_centers_step(
    prev_unwrapped_nm: np.ndarray,
    curr_wrapped_nm: np.ndarray,
    box_nm: np.ndarray,
) -> np.ndarray:
    """
    Unwrap one frame of wrapped group centers against previous unwrapped centers
    using minimum-image displacements.
    """
    prev = np.asarray(prev_unwrapped_nm, dtype=np.float64)
    curr = np.asarray(curr_wrapped_nm, dtype=np.float64)
    b = np.asarray(box_nm, dtype=np.float64).reshape(1, 3)

    d = curr - prev
    d -= np.rint(d / b) * b
    return prev + d


def _weighted_center(
    xyz_nm: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray(xyz_nm, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    tot = float(np.sum(w))
    if tot <= 0.0:
        return np.mean(x, axis=0)
    return np.sum(x * w[:, None], axis=0) / tot


def _trimmed_center_from_groups(
    group_centers_nm: np.ndarray,
    group_weights: np.ndarray,
    *,
    keep_fraction: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a weighted center after excluding the farthest groups.

    Parameters
    ----------
    group_centers_nm : (n_groups, 3)
        Per-group centers, typically already time-unwrapped.
    group_weights : (n_groups,)
        Weights for the center calculation.
    keep_fraction : float
        Fraction of groups to keep, based on distance from a provisional center.

    Returns
    -------
    center_nm : (3,)
        Trimmed weighted center.
    keep_mask : (n_groups,) bool
        True for groups retained in the trimmed center.
    """
    x = np.asarray(group_centers_nm, dtype=np.float64)
    w = np.asarray(group_weights, dtype=np.float64).reshape(-1)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("group_centers_nm must have shape (n_groups, 3)")
    if w.shape[0] != x.shape[0]:
        raise ValueError("group_weights length must match number of groups")
    if not (0.0 < float(keep_fraction) <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1]")

    n = int(x.shape[0])
    n_keep = max(1, int(np.ceil(float(keep_fraction) * float(n))))

    c0 = _weighted_center(x, w)
    d2 = np.sum((x - c0[None, :]) ** 2, axis=1)

    order = np.argsort(d2)
    keep_idx = order[:n_keep]

    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_idx] = True

    c_trim = _weighted_center(x[keep_mask], w[keep_mask])
    return c_trim, keep_mask


def _wrap_centered_box(xyz_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    """
    Wrap coordinates into [-L/2, L/2) along each axis.
    """
    x = np.asarray(xyz_nm, dtype=np.float64)
    b = np.asarray(box_nm, dtype=np.float64).reshape(1, 3)
    return x - np.rint(x / b) * b


def _choose_grid_shape(
    box_nm: np.ndarray,
    spacing_nm: float,
) -> tuple[int, int, int]:
    b = np.asarray(box_nm, dtype=np.float64).reshape(3)
    if spacing_nm <= 0.0:
        raise ValueError("spacing_nm must be > 0")
    nx, ny, nz = [max(1, int(math.ceil(float(L) / float(spacing_nm)))) for L in b]
    return nx, ny, nz


def _grid_geometry_centered(
    box_nm: np.ndarray,
    dims: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return origin and spacing for a centered image grid.

    The resulting grid is centered on the origin and voxel coordinates correspond
    to voxel centers.
    """
    b = np.asarray(box_nm, dtype=np.float64).reshape(3)
    nx, ny, nz = [int(v) for v in dims]

    spacing = np.array(
        [
            float(b[0]) / float(nx),
            float(b[1]) / float(ny),
            float(b[2]) / float(nz),
        ],
        dtype=np.float64,
    )

    origin = np.array(
        [
            -0.5 * float(b[0]) + 0.5 * spacing[0],
            -0.5 * float(b[1]) + 0.5 * spacing[1],
            -0.5 * float(b[2]) + 0.5 * spacing[2],
        ],
        dtype=np.float64,
    )
    return origin, spacing


def _gaussian_density_grid_periodic(
    xyz_centered_nm: np.ndarray,
    weights: np.ndarray,
    box_nm: np.ndarray,
    dims: tuple[int, int, int],
    *,
    sigma_nm: float,
    truncate_sigma: float = 3.0,
) -> np.ndarray:
    """
    Build a Gaussian-smoothed periodic density grid on a centered box.

    Parameters
    ----------
    xyz_centered_nm : (n_atoms, 3)
        Coordinates in a centered periodic box, typically wrapped into [-L/2, L/2).
    weights : (n_atoms,)
        Per-atom weights; masses if mass density is desired, or ones for number density.
    box_nm : (3,)
        Orthorhombic box lengths in nm.
    dims : (nx, ny, nz)
        Grid shape.
    sigma_nm : float
        Gaussian sigma in nm.
    truncate_sigma : float
        Truncate support at this many sigmas.

    Returns
    -------
    rho : (nx, ny, nz) float32
        Density in weight / nm^3 units.
    """
    if sigma_nm <= 0.0:
        raise ValueError("sigma_nm must be > 0")
    if truncate_sigma <= 0.0:
        raise ValueError("truncate_sigma must be > 0")

    x = np.asarray(xyz_centered_nm, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    b = np.asarray(box_nm, dtype=np.float64).reshape(3)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("xyz_centered_nm must have shape (n_atoms, 3)")
    if w.shape[0] != x.shape[0]:
        raise ValueError("weights must have length n_atoms")

    nx, ny, nz = [int(v) for v in dims]
    origin, spacing = _grid_geometry_centered(b, dims)
    dx, dy, dz = [float(v) for v in spacing]

    rho = np.zeros((nx, ny, nz), dtype=np.float64)

    rx = max(1, int(math.ceil(truncate_sigma * sigma_nm / dx)))
    ry = max(1, int(math.ceil(truncate_sigma * sigma_nm / dy)))
    rz = max(1, int(math.ceil(truncate_sigma * sigma_nm / dz)))

    pref = 1.0 / ((2.0 * math.pi) ** 1.5 * sigma_nm**3)

    for p, wt in zip(x, w):
        fx = (p[0] - origin[0]) / dx
        fy = (p[1] - origin[1]) / dy
        fz = (p[2] - origin[2]) / dz

        ic = int(np.rint(fx))
        jc = int(np.rint(fy))
        kc = int(np.rint(fz))

        for di in range(-rx, rx + 1):
            i = (ic + di) % nx
            gx = origin[0] + i * dx
            ddx = gx - p[0]
            ddx -= np.rint(ddx / b[0]) * b[0]
            ddx2 = ddx * ddx

            for dj in range(-ry, ry + 1):
                j = (jc + dj) % ny
                gy = origin[1] + j * dy
                ddy = gy - p[1]
                ddy -= np.rint(ddy / b[1]) * b[1]
                ddy2 = ddy * ddy

                for dk in range(-rz, rz + 1):
                    k = (kc + dk) % nz
                    gz = origin[2] + k * dz
                    ddz = gz - p[2]
                    ddz -= np.rint(ddz / b[2]) * b[2]
                    r2 = ddx2 + ddy2 + ddz * ddz
                    rho[i, j, k] += wt * pref * math.exp(-0.5 * r2 / (sigma_nm * sigma_nm))

    return rho.astype(np.float32, copy=False)


def _write_vti_scalar(
    path: Union[str, Path],
    data: np.ndarray,
    *,
    origin_nm: Sequence[float],
    spacing_nm: Sequence[float],
    array_name: str = "density",
) -> None:
    """
    Write a single scalar 3D image as VTK XML ImageData (.vti).

    Notes
    -----
    - data must have shape (nx, ny, nz)
    - stored as Float32 point data
    - flattened in Fortran order for VTK image conventions
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("data must have shape (nx, ny, nz)")

    nx, ny, nz = [int(v) for v in arr.shape]
    ox, oy, oz = [float(v) for v in origin_nm]
    sx, sy, sz = [float(v) for v in spacing_nm]

    flat = np.ravel(arr, order="F")
    payload = flat.tobytes(order="C")
    header = struct.pack("<Q", len(payload))
    encoded = base64.b64encode(header + payload).decode("ascii")

    whole_extent = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"
    origin_str = f"{ox:.8f} {oy:.8f} {oz:.8f}"
    spacing_str = f"{sx:.8f} {sy:.8f} {sz:.8f}"

    lines = [
        '<?xml version="1.0"?>',
        (
            '<VTKFile type="ImageData" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">'
        ),
        (
            f'  <ImageData WholeExtent="{whole_extent}" '
            f'Origin="{origin_str}" Spacing="{spacing_str}">'
        ),
        f'    <Piece Extent="{whole_extent}">',
        f'      <PointData Scalars="{array_name}">',
        f'        <DataArray type="Float32" Name="{array_name}" format="binary">',
        encoded,
        "        </DataArray>",
        "      </PointData>",
        "      <CellData/>",
        "    </Piece>",
        "  </ImageData>",
        "</VTKFile>",
    ]
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def _write_pvd(
    path: Union[str, Path],
    files: Sequence[Union[str, Path]],
    *,
    dt: Optional[float] = None,
) -> None:
    """
    Write a ParaView .pvd collection file for the generated .vti frames.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for i, fn in enumerate(files):
        t = float(i) if dt is None else float(i) * float(dt)
        lines.append(f'    <DataSet timestep="{t:.8f}" group="" part="0" file="{Path(fn).name}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def density_grids_to_vti_from_dcd(
    pdb_file: FileLike,
    dcd_files: Union[FileLike, Sequence[FileLike]],
    *,
    selection: Union[str, Sequence[str], Sequence[Sequence[int]]] = "protein",
    center_mode: str = "com",  # "cog" | "com"
    trimmed_center_keep_fraction: float = 0.75,
    spacing_nm: float = 0.5,
    sigma_nm: float = 0.75,
    truncate_sigma: float = 3.0,
    box_nm: Optional[Sequence[float]] = None,
    stride: int = 1,
    chunk: int = 100,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
    out_dir: Union[str, Path] = "density_vti",
    prefix: str = "dens",
    array_name: str = "density",
    mass_weighted_density: bool = True,
    write_pvd: bool = True,
    pvd_dt: Optional[float] = None,
) -> dict[str, Any]:
    """
    Generate one Gaussian-smoothed density grid per frame and save as .vti.

    Parameters
    ----------
    pdb_file, dcd_files
        Same meaning/pattern as rg_from_dcd().
    selection
        Same semantics as rg_from_dcd():
          - "protein"     -> one group per chain, all atoms in each chain
          - "protein.CA"  -> one group per chain, only CA atoms
          - list[str]     -> one selector per output group
          - list[list[int]] -> explicit atom index groups in template indexing
    center_mode
        "cog" or "com" for per-group centers and global trimmed center.
    trimmed_center_keep_fraction
        Fraction of groups to keep when recomputing the condensate center.
        Example: 0.75 excludes the farthest 25% of groups from the provisional center.
    spacing_nm
        Target grid spacing in nm.
    sigma_nm
        Gaussian smoothing width in nm.
    truncate_sigma
        Deposit Gaussian support out to this many sigma.
    box_nm
        Fallback orthorhombic box lengths (nm) if absent from DCD.
    stride, chunk, frame_start, frame_stop
        Same style as other trajectory analysis functions in the package.
    out_dir, prefix
        Output directory and file prefix.
    array_name
        Scalar array name stored in the .vti file.
    mass_weighted_density
        If True, deposit masses for mass density. If False, deposit ones for number density.
    write_pvd
        If True, also write a ParaView .pvd collection file.
    pvd_dt
        Optional timestep spacing for the .pvd metadata.

    Returns
    -------
    dict
        Summary of generated files and grid settings.

    Notes
    -----
    Centering strategy:
      1) Compute one center per selected group from coordinates as stored in each frame.
      2) Time-unwrap those group centers across frames using minimum-image jumps.
      3) Compute a provisional global center from all groups.
      4) Exclude the farthest (1 - keep_fraction) groups.
      5) Recompute the global center from the retained groups.
      6) Recenter all selected atoms using that trimmed center.
      7) Wrap into a centered periodic box and deposit Gaussian density.

    This assumes each selected chain/group is already whole within each frame,
    and only whole-group jumps across PBC occur between frames.
    """
    mode = str(center_mode).strip().lower()
    if mode not in {"cog", "com"}:
        raise ValueError("center_mode must be 'cog' or 'com'")
    if not (0.0 < float(trimmed_center_keep_fraction) <= 1.0):
        raise ValueError("trimmed_center_keep_fraction must be in (0, 1]")
    if float(spacing_nm) <= 0.0:
        raise ValueError("spacing_nm must be > 0")
    if float(sigma_nm) <= 0.0:
        raise ValueError("sigma_nm must be > 0")
    if float(truncate_sigma) <= 0.0:
        raise ValueError("truncate_sigma must be > 0")
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

    # Flatten selected atoms once, then remap group indices into selected-atom order
    atom_set: set[int] = set()
    for g in groups_full:
        atom_set.update(int(i) for i in g.tolist())
    atom_indices_full = sorted(atom_set)

    idx_map = {old: new for new, old in enumerate(atom_indices_full)}
    groups_sel = [
        np.asarray([idx_map[int(i)] for i in g.tolist()], dtype=np.int64) for g in groups_full
    ]

    masses_all = atom_masses(tmpl_model)
    masses_sel = np.asarray(masses_all[atom_indices_full], dtype=np.float64)

    # Per-group weights for center calculation
    if mode == "com":
        group_weights = np.array(
            [float(np.sum(masses_sel[g])) for g in groups_sel],
            dtype=np.float64,
        )
    else:
        group_weights = np.array(
            [float(len(g)) for g in groups_sel],
            dtype=np.float64,
        )

    # Per-atom weights for density deposition
    if mass_weighted_density:
        deposit_weights = masses_sel
    else:
        deposit_weights = np.ones_like(masses_sel, dtype=np.float64)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []
    frames_used = 0

    prev_group_centers_unwrapped: Optional[np.ndarray] = None
    last_dims: Optional[tuple[int, int, int]] = None
    last_origin: Optional[np.ndarray] = None
    last_spacing: Optional[np.ndarray] = None
    last_keep_mask: Optional[np.ndarray] = None

    box_fallback = None if box_nm is None else _box_lengths_nm(box_nm)

    stop_all = False
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
                stop_all = True
                break

            if box_frame_nm is None:
                if box_fallback is None:
                    raise ValueError("DCD lacks box; pass box_nm=(Lx,Ly,Lz) in nm")
                b = box_fallback
            else:
                b = _box_lengths_nm(box_frame_nm)

            xyz = np.asarray(xyz_sel_nm, dtype=np.float64)

            # Per-group centers in the current wrapped frame
            group_centers_wrapped = _centers_of_groups_whole_chain(
                xyz,
                groups_sel,
                masses=masses_sel if mode == "com" else None,
                center=mode,
            )

            # Time-unwrap whole-group centers from frame to frame
            if prev_group_centers_unwrapped is None:
                group_centers_unwrapped = np.asarray(group_centers_wrapped, dtype=np.float64)
            else:
                group_centers_unwrapped = _time_unwrap_centers_step(
                    prev_group_centers_unwrapped,
                    group_centers_wrapped,
                    b,
                )

            prev_group_centers_unwrapped = group_centers_unwrapped

            # Trimmed center: exclude farthest groups from provisional center
            condensate_center_unwrapped, keep_mask = _trimmed_center_from_groups(
                group_centers_unwrapped,
                group_weights,
                keep_fraction=float(trimmed_center_keep_fraction),
            )
            last_keep_mask = keep_mask

            # Recenter all selected atoms, then wrap into centered periodic box
            xyz_centered = xyz - condensate_center_unwrapped.reshape(1, 3)
            xyz_centered = _wrap_centered_box(xyz_centered, b)

            dims = _choose_grid_shape(b, float(spacing_nm))
            origin, spacing = _grid_geometry_centered(b, dims)

            rho = _gaussian_density_grid_periodic(
                xyz_centered,
                deposit_weights,
                b,
                dims,
                sigma_nm=float(sigma_nm),
                truncate_sigma=float(truncate_sigma),
            )

            fn = out_path / f"{prefix}_{frames_used:05d}.vti"
            _write_vti_scalar(
                fn,
                rho,
                origin_nm=origin,
                spacing_nm=spacing,
                array_name=array_name,
            )
            files_written.append(str(fn))

            frames_used += 1
            last_dims = dims
            last_origin = origin
            last_spacing = spacing

        if stop_all:
            break

    if frames_used == 0:
        raise ValueError("no frames selected")

    pvd_file = None
    if write_pvd:
        pvd_file = out_path / f"{prefix}.pvd"
        _write_pvd(pvd_file, files_written, dt=pvd_dt)

    return {
        "files": files_written,
        "pvd_file": None if pvd_file is None else str(pvd_file),
        "frames_used": int(frames_used),
        "selection": selection,
        "center_mode": mode,
        "trimmed_center_keep_fraction": float(trimmed_center_keep_fraction),
        "spacing_nm": float(spacing_nm),
        "sigma_nm": float(sigma_nm),
        "truncate_sigma": float(truncate_sigma),
        "mass_weighted_density": bool(mass_weighted_density),
        "dims": None if last_dims is None else tuple(int(v) for v in last_dims),
        "origin_nm": None if last_origin is None else np.asarray(last_origin, dtype=np.float64),
        "grid_spacing_nm": (
            None if last_spacing is None else np.asarray(last_spacing, dtype=np.float64)
        ),
        "n_groups": int(len(groups_sel)),
        "n_groups_kept_last_frame": None if last_keep_mask is None else int(np.sum(last_keep_mask)),
    }

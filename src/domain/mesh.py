from __future__ import annotations

from pathlib import Path

import dolfinx.mesh as dmesh
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI


def _normalize_xdmf_path(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    suf = p.suffix.lower()
    if suf in {".h5", ".hdf5", ".hdf"}:
        xdmf = p.with_suffix(".xdmf")
        if xdmf.exists():
            return xdmf
        raise ValueError(
            f"Got HDF5 file {p.name!r}. Please pass the corresponding .xdmf file "
            f"(expected {xdmf.name!r} next to it)."
        )
    return p


def read_mesh_xdmf(
    comm: MPI.Comm,
    path: str | Path,
    *,
    mesh_name: str = "Grid",
    ghost_mode: dmesh.GhostMode = dmesh.GhostMode.shared_facet,
) -> dmesh.Mesh:
    p = _normalize_xdmf_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with XDMFFile(comm, str(p), "r") as xdmf:
        try:
            mesh = xdmf.read_mesh(name=mesh_name, ghost_mode=ghost_mode)
        except TypeError:
            try:
                mesh = xdmf.read_mesh(name=mesh_name)
            except TypeError:
                mesh = xdmf.read_mesh()

    tdim = mesh.topology.dim
    if tdim >= 1:
        mesh.topology.create_connectivity(tdim - 1, tdim)
    return mesh


def read_meshtags_xdmf(
    mesh: dmesh.Mesh,
    path: str | Path,
    *,
    name: str,
    dim: int | None = None,
) -> dmesh.MeshTags:
    p = _normalize_xdmf_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with XDMFFile(mesh.comm, str(p), "r") as xdmf:
        try:
            tags = xdmf.read_meshtags(mesh, name=name)
        except TypeError:
            try:
                tags = xdmf.read_meshtags(mesh, name)
            except TypeError:
                tags = xdmf.read_meshtags(mesh, name, mesh.geometry)  # type: ignore[misc]

    try:
        tags.name = str(name)
    except Exception:
        pass

    if dim is not None and int(tags.dim) != int(dim):
        raise ValueError(f"MeshTags {name!r} has dim={tags.dim}, expected dim={dim}.")
    return tags


def merge_meshtags(
    mesh: dmesh.Mesh,
    dim: int,
    old: dmesh.MeshTags,
    new_indices: np.ndarray,
    new_values: np.ndarray,
    *,
    override: bool,
) -> dmesh.MeshTags:
    oi = np.asarray(old.indices, dtype=np.int32)
    ov = np.asarray(old.values, dtype=np.int32)
    ni = np.asarray(new_indices, dtype=np.int32).ravel()
    nv = np.asarray(new_values, dtype=np.int32).ravel()

    if ni.size == 0:
        return old

    if override:
        idx_all = np.concatenate([oi, ni])
        val_all = np.concatenate([ov, nv])
    else:
        idx_all = np.concatenate([ni, oi])
        val_all = np.concatenate([nv, ov])

    order = np.argsort(idx_all, kind="mergesort")
    idx_s = idx_all[order]
    val_s = val_all[order]

    uniq_idx, first, counts = np.unique(idx_s, return_index=True, return_counts=True)
    last_pos = first + counts - 1
    uniq_val = val_s[last_pos]

    return dmesh.meshtags(mesh, dim, uniq_idx, uniq_val)


def load_boundary_facets_from_xdmf(
    mesh: dmesh.Mesh,
    path: str | Path,
    *,
    tags_name: str,
    marker: int,
) -> np.ndarray:
    fdim = mesh.topology.dim - 1
    tags = read_meshtags_xdmf(mesh, path, name=tags_name, dim=fdim)

    values = np.asarray(tags.values, dtype=np.int32)
    indices = np.asarray(tags.indices, dtype=np.int32)
    return indices[values == int(marker)]

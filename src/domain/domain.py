from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np
from mpi4py import MPI
from networks_fenicsx import NetworkMesh

from .graph import build_graph_from_polydata, compute_radius_by_tag
from .io import read_vtk_legacy_polydata_ascii
from .mesh import (
    load_boundary_facets_from_xdmf,
    merge_meshtags,
    read_mesh_xdmf,
    read_meshtags_xdmf,
)
from src.system import collect, deep_close_destroy


@dataclass(slots=True)
class Domain3D:
    mesh: dmesh.Mesh
    boundaries: dmesh.MeshTags | None = None
    outlet_marker: int | None = None

    _cache: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        tdim = self.mesh.topology.dim
        if tdim >= 1:
            self.mesh.topology.create_connectivity(tdim - 1, tdim)

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm

    def get_functionspace(self, element: Any) -> Any:
        key = ("fs", element)
        V = self._cache.get(key)
        if V is None:
            V = fem.functionspace(self.mesh, element)
            self._cache[key] = V
        return V

    def clear_cache(self) -> None:
        if self._cache:
            try:
                deep_close_destroy(self._cache, max_depth=3)
            except Exception:
                pass
            self._cache.clear()
            collect()

    def release(self) -> None:
        try:
            self.clear_cache()
        except Exception:
            pass
        self.boundaries = None
        self.outlet_marker = None
        self.mesh = None  # type: ignore[assignment]
        collect()

    def __del__(self) -> None:
        try:
            self.clear_cache()
        except Exception:
            pass

    def add_boundary_facets(
        self,
        facets: np.ndarray,
        *,
        marker: int,
        override: bool = True,
        name: str = "boundaries",
    ) -> None:
        """
        Tag boundary facets with `marker` in `self.boundaries`.

        Parallel-safe behavior:
          - It's OK if `facets` is empty on this rank.
          - It's an error only if `facets` is empty on ALL ranks.
        """
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        facets = np.asarray(facets, dtype=np.int32).ravel()

        # Allow empty locally; only error if empty globally.
        n_global = self.comm.allreduce(int(facets.size), op=MPI.SUM)
        if n_global == 0:
            raise ValueError("No facets were provided to add_boundary_facets() on any rank.")

        # Ensure MeshTags exists on every rank (possibly empty locally).
        if self.boundaries is None:
            empty = np.zeros((0,), dtype=np.int32)
            self.boundaries = dmesh.meshtags(mesh, fdim, empty, empty)
            try:
                self.boundaries.name = str(name)
            except Exception:
                pass
        else:
            if int(self.boundaries.dim) != int(fdim):
                raise ValueError(
                    f"Domain3D.boundaries has dim={self.boundaries.dim}, expected {fdim}."
                )

        # Only merge on ranks that actually own facets for this marker.
        if facets.size:
            facets = np.unique(facets)
            values = np.full((facets.size,), int(marker), dtype=np.int32)
            self.boundaries = merge_meshtags(
                mesh,
                fdim,
                self.boundaries,
                facets,
                values,
                override=override,
            )
            try:
                if self.boundaries is not None and not getattr(self.boundaries, "name", ""):
                    self.boundaries.name = str(name)
            except Exception:
                pass

        # Record outlet marker conventionally (matches your current behavior).
        self.outlet_marker = int(marker)

    @classmethod
    def from_xdmf(
        cls,
        comm: MPI.Comm,
        path: str | Path,
        *,
        mesh_name: str = "Grid",
        ghost_mode: dmesh.GhostMode = dmesh.GhostMode.shared_facet,
        boundaries_name: str | None = None,
        boundaries_path: str | Path | None = None,
        outlet_marker: int | None = None,
    ) -> "Domain3D":
        mesh = read_mesh_xdmf(comm, path, mesh_name=mesh_name, ghost_mode=ghost_mode)
        dom = cls(mesh=mesh)

        if boundaries_name is not None:
            bpath = path if boundaries_path is None else boundaries_path
            fdim = mesh.topology.dim - 1
            dom.boundaries = read_meshtags_xdmf(mesh, bpath, name=boundaries_name, dim=fdim)
            try:
                if dom.boundaries is not None and not getattr(dom.boundaries, "name", ""):
                    dom.boundaries.name = str(boundaries_name)
            except Exception:
                pass
            if outlet_marker is not None:
                dom.outlet_marker = int(outlet_marker)

        return dom

    def mark_outlet_from_xdmf(
        self,
        path: str | Path,
        *,
        tags_name: str,
        marker: int,
        replace_boundaries: bool = True,
        override: bool = True,
    ) -> np.ndarray:
        """
        Set/merge facet tags from XDMF and return the local facet indices for `marker`.

        Parallel-safe behavior:
          - Returns empty array on ranks with no owned facets for that marker.
          - Errors only if the marker does not exist on ANY rank.
        """
        marker_i = int(marker)

        if replace_boundaries:
            fdim = self.mesh.topology.dim - 1
            tags = read_meshtags_xdmf(self.mesh, path, name=tags_name, dim=fdim)
            try:
                tags.name = str(tags_name)
            except Exception:
                pass

            self.boundaries = tags
            self.outlet_marker = marker_i

            facets = np.asarray(tags.indices, dtype=np.int32)[
                np.asarray(tags.values, dtype=np.int32) == marker_i
            ]
            n_global = self.comm.allreduce(int(facets.size), op=MPI.SUM)
            if n_global == 0:
                raise ValueError(
                    f"MeshTags {tags_name!r} from {path!r} contains no facets with marker={marker_i}."
                )
            return facets

        facets = load_boundary_facets_from_xdmf(
            self.mesh,
            path,
            tags_name=tags_name,
            marker=marker_i,
        )

        # This handles the global-empty check and keeps tags consistent across ranks.
        self.add_boundary_facets(facets, marker=marker_i, override=override)
        return facets

    @classmethod
    def from_box(
        cls,
        comm: MPI.Comm,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
        target_h: float,
        cell_type: dmesh.CellType = dmesh.CellType.tetrahedron,
    ) -> "Domain3D":
        extent = max_corner - min_corner
        n = [max(2, int(np.ceil(extent[i] / target_h))) for i in range(3)]
        mesh = dmesh.create_box(
            comm,
            [min_corner.tolist(), max_corner.tolist()],
            n,
            cell_type=cell_type,
        )
        return cls(mesh=mesh)


@dataclass(slots=True)
class Domain1D:
    mesh: dmesh.Mesh
    boundaries: dmesh.MeshTags
    inlet_marker: int
    outlet_marker: int
    subdomains: dmesh.MeshTags | None = None
    radius_by_tag: Mapping[int, float] | np.ndarray | None = None

    _cache: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mesh.topology.create_connectivity(0, 1)
        self.mesh.topology.create_connectivity(1, 0)

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm

    def get_functionspace(self, element: Any) -> Any:
        key = ("fs", element)
        V = self._cache.get(key)
        if V is None:
            V = fem.functionspace(self.mesh, element)
            self._cache[key] = V
        return V

    def clear_cache(self) -> None:
        if self._cache:
            try:
                deep_close_destroy(self._cache, max_depth=3)
            except Exception:
                pass
            self._cache.clear()
            collect()

    def release(self) -> None:
        try:
            self.clear_cache()
        except Exception:
            pass
        self.subdomains = None
        self.radius_by_tag = None
        self.boundaries = None  # type: ignore[assignment]
        self.mesh = None  # type: ignore[assignment]
        collect()

    def __del__(self) -> None:
        try:
            self.clear_cache()
        except Exception:
            pass

    def boundary_vertices(self, marker: int) -> np.ndarray:
        values = self.boundaries.values
        indices = self.boundaries.indices
        return indices[values == int(marker)].astype(np.int32, copy=False)

    @property
    def inlet_vertices(self) -> np.ndarray:
        return self.boundary_vertices(self.inlet_marker)

    @property
    def outlet_vertices(self) -> np.ndarray:
        return self.boundary_vertices(self.outlet_marker)

    @classmethod
    def from_network(
        cls,
        graph: Any,
        points_per_edge: int,
        comm: MPI.Comm,
        graph_rank: int = 0,
        inlet_marker: int | None = None,
        outlet_marker: int | None = None,
        color_strategy: Any | None = None,
    ) -> "Domain1D":
        network = NetworkMesh(
            graph,
            N=int(points_per_edge),
            comm=comm,
            graph_rank=int(graph_rank),
            color_strategy=color_strategy,
        )

        inlet = int(network.out_marker) if inlet_marker is None else int(inlet_marker)
        outlet = int(network.in_marker) if outlet_marker is None else int(outlet_marker)

        return cls(
            mesh=network.mesh,
            boundaries=network.boundaries,
            subdomains=getattr(network, "subdomains", None),
            inlet_marker=inlet,
            outlet_marker=outlet,
        )

    @classmethod
    def from_vtk_polydata(
        cls,
        comm: MPI.Comm,
        path: str | Path,
        *,
        points_per_edge: int = 1,
        graph_rank: int = 0,
        color_strategy: Any | None = None,
        radius_name: str = "Radius",
        default_radius: float = 1.0,
        reverse_edges: bool = False,
        inlet_marker: int | None = None,
        outlet_marker: int | None = None,
        strict_if_grouped: bool = True,
    ) -> "Domain1D":
        p = Path(path).expanduser().resolve()

        graph: Any | None = None
        radius_by_tag: np.ndarray | None = None

        if comm.rank == int(graph_rank):
            pts, polylines, pdat, cdat = read_vtk_legacy_polydata_ascii(p)
            npts = int(pts.shape[0])

            pr = pdat.get(radius_name)
            if pr is not None:
                pr = np.asarray(pr, dtype=np.float64).reshape((-1,))
                if pr.size != npts:
                    pr = None

            cr = cdat.get(radius_name)
            if cr is not None:
                cr = np.asarray(cr, dtype=np.float64).reshape((-1,))

            graph = build_graph_from_polydata(
                pts,
                polylines,
                point_radius=pr,
                cell_radius=cr,
                default_radius=float(default_radius),
                reverse_edges=bool(reverse_edges),
            )

            radius_by_tag = compute_radius_by_tag(
                graph,
                color_strategy=color_strategy,
                default_radius=float(default_radius),
                strict_if_grouped=bool(strict_if_grouped),
            )

        radius_by_tag = comm.bcast(radius_by_tag, root=int(graph_rank))

        network = NetworkMesh(
            graph,
            N=int(points_per_edge),
            comm=comm,
            graph_rank=int(graph_rank),
            color_strategy=color_strategy,
        )

        inlet = int(network.out_marker) if inlet_marker is None else int(inlet_marker)
        outlet = int(network.in_marker) if outlet_marker is None else int(outlet_marker)

        return cls(
            mesh=network.mesh,
            boundaries=network.boundaries,
            subdomains=getattr(network, "subdomains", None),
            inlet_marker=inlet,
            outlet_marker=outlet,
            radius_by_tag=radius_by_tag,
        )

# multihepatic/src/domain/domain.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import dolfinx.mesh as dmesh
import dolfinx.fem as fem
import numpy as np
from mpi4py import MPI
from networks_fenicsx import NetworkMesh
import networkx as nx

# XDMF/HDF5 mesh I/O helpers (see domain/mesh.py)
from src.domain.mesh import read_mesh_xdmf, read_meshtags_xdmf, load_boundary_facets_from_xdmf

from src.system import deep_close_destroy, collect


def _axis_to_int(axis: int | str) -> int:
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
        return axis
    a = axis.lower().strip()
    if a == "x":
        return 0
    if a == "y":
        return 1
    if a == "z":
        return 2
    raise ValueError(f"axis must be 0/1/2 or 'x'/'y'/'z'; got {axis!r}")


def _apply_affine_to_points(affine_4x4: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 affine to an (N,3) point array, returning (N,3).

    This is used to optionally transform mesh coordinates into the NIfTI "world"
    coordinate frame before mapping into voxel indices.
    """
    A = np.asarray(affine_4x4, dtype=np.float64)
    if A.shape != (4, 4):
        raise ValueError(f"Expected affine shape (4,4), got {A.shape}.")

    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {P.shape}.")

    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    Ph = np.concatenate([P, ones], axis=1)  # (N,4)
    Q = (A @ Ph.T).T
    return Q[:, :3].astype(np.float64, copy=False)


def _merge_meshtags(
        mesh: dmesh.Mesh,
        dim: int,
        old: dmesh.MeshTags,
        new_indices: np.ndarray,
        new_values: np.ndarray,
        *,
        override: bool,
) -> dmesh.MeshTags:
    """Merge tags on a given entity dim. If override=True, new_values win on overlaps."""
    oi = np.asarray(old.indices, dtype=np.int32)
    ov = np.asarray(old.values, dtype=np.int32)
    ni = np.asarray(new_indices, dtype=np.int32).ravel()
    nv = np.asarray(new_values, dtype=np.int32).ravel()

    if ni.size == 0:
        return old

    # Concatenate so that "last occurrence wins" per index group.
    if override:
        idx_all = np.concatenate([oi, ni])
        val_all = np.concatenate([ov, nv])
    else:
        idx_all = np.concatenate([ni, oi])
        val_all = np.concatenate([nv, ov])

    # Stable sort by entity index
    order = np.argsort(idx_all, kind="mergesort")
    idx_s = idx_all[order]
    val_s = val_all[order]

    # Take the last value per unique index (so "later in concat" wins)
    uniq_idx, first, counts = np.unique(idx_s, return_index=True, return_counts=True)
    last_pos = first + counts - 1
    uniq_val = val_s[last_pos]

    return dmesh.meshtags(mesh, dim, uniq_idx, uniq_val)


def _is_section_header(line: str) -> bool:
    """Return True if a line likely starts a new VTK legacy section."""
    s = line.strip()
    if not s:
        return False
    # Common legacy POLYDATA section headers
    return s.startswith(
        (
            "POINTS",
            "LINES",
            "POLYGONS",
            "VERTICES",
            "TRIANGLE_STRIPS",
            "POINT_DATA",
            "CELL_DATA",
            "FIELD",
            "SCALARS",
            "LOOKUP_TABLE",
        )
    )


def _read_n_tokens(lines: list[str], i0: int, n: int) -> tuple[list[str], int]:
    """Read at least n whitespace-separated tokens starting from lines[i0]."""
    toks: list[str] = []
    i = i0
    while len(toks) < n and i < len(lines):
        s = lines[i].strip()
        if s:
            toks.extend(s.split())
        i += 1
    return toks, i


def _read_vtk_legacy_polydata_ascii(path: str | Path) -> tuple[
    np.ndarray, list[np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """
    Minimal reader for *ASCII legacy* VTK POLYDATA files containing:
      - POINTS
      - LINES (either classic legacy format or the OFFSETS/CONNECTIVITY style)
      - POINT_DATA and/or CELL_DATA with either FIELD or SCALARS arrays

    Returns:
      points: (npoints, 3) float64
      lines:  list of 1D int arrays (each array is a polyline's point indices)
      point_data: dict[name] -> ndarray
      cell_data:  dict[name] -> ndarray
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    lines_txt = p.read_text().splitlines()
    nlines_txt = len(lines_txt)

    points: np.ndarray | None = None
    polylines: list[np.ndarray] = []
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    i = 0
    while i < nlines_txt:
        line = lines_txt[i].strip()
        if not line:
            i += 1
            continue

        # ---- POINTS ----
        if line.startswith("POINTS"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POINTS line: {line!r}")
            npts = int(parts[1])
            # dtype token in parts[2] is ignored (float/double)
            i += 1
            toks, i = _read_n_tokens(lines_txt, i, 3 * npts)
            if len(toks) < 3 * npts:
                raise ValueError(f"POINTS section ended early (need {3*npts} floats).")
            arr = np.asarray(toks[: 3 * npts], dtype=np.float64)
            points = arr.reshape((npts, 3))
            continue

        # ---- LINES ----
        if line.startswith("LINES"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed LINES line: {line!r}")
            n_cells = int(parts[1])
            total_ints = int(parts[2])
            i += 1
            # Skip blank lines
            while i < nlines_txt and not lines_txt[i].strip():
                i += 1
            if i >= nlines_txt:
                raise ValueError("LINES section ended unexpectedly.")

            nxt = lines_txt[i].strip()
            # VTK 5.1 legacy "OFFSETS ... / CONNECTIVITY ..." style
            if nxt.startswith("OFFSETS"):
                # OFFSETS <dtype>
                i += 1
                offsets_toks: list[str] = []
                while i < nlines_txt:
                    s = lines_txt[i].strip()
                    if not s:
                        i += 1
                        continue
                    if s.startswith("CONNECTIVITY"):
                        break
                    offsets_toks.extend(s.split())
                    i += 1
                if i >= nlines_txt or not lines_txt[i].strip().startswith("CONNECTIVITY"):
                    raise ValueError("LINES section missing CONNECTIVITY header.")
                # CONNECTIVITY <dtype>
                i += 1
                conn_toks: list[str] = []
                while i < nlines_txt:
                    s = lines_txt[i].strip()
                    if not s:
                        i += 1
                        continue
                    if _is_section_header(s) and not s[0].isdigit() and not s[0] == "-":
                        # Next section
                        break
                    # Heuristic: stop when we hit a known section header (POINT_DATA/CELL_DATA/etc.)
                    if s.startswith(("POINT_DATA", "CELL_DATA", "POLYGONS", "VERTICES", "TRIANGLE_STRIPS", "LINES", "POINTS")):
                        break
                    conn_toks.extend(s.split())
                    i += 1

                offsets = np.asarray(offsets_toks, dtype=np.int64)
                conn = np.asarray(conn_toks, dtype=np.int64)

                # OFFSETS is typically length n_cells+1; be forgiving if last is omitted
                if offsets.size == n_cells:
                    offsets = np.concatenate([offsets, [conn.size]])
                if offsets.size < 2:
                    raise ValueError("OFFSETS array too short.")

                # Trust offsets rather than the header count if they disagree
                n_cells_actual = int(offsets.size - 1)
                if n_cells_actual != n_cells:
                    n_cells = n_cells_actual

                if offsets[0] != 0:
                    offsets = offsets - offsets[0]
                if offsets[-1] > conn.size:
                    raise ValueError("OFFSETS indicates more connectivity entries than present.")

                for c in range(n_cells):
                    a = int(offsets[c])
                    b = int(offsets[c + 1])
                    pts_idx = conn[a:b]
                    if pts_idx.size >= 2:
                        polylines.append(pts_idx.astype(np.int64, copy=False))
                continue

            # Classic legacy LINES format:
            #   total_ints integers follow, grouped as:
            #     k i0 i1 ... i{k-1}
            # repeated n_cells times.
            toks, i = _read_n_tokens(lines_txt, i, total_ints)
            if len(toks) < total_ints:
                raise ValueError(f"LINES section ended early (need {total_ints} ints).")
            ints = list(map(int, toks[:total_ints]))
            pos = 0
            for _ in range(n_cells):
                if pos >= len(ints):
                    break
                k = int(ints[pos]); pos += 1
                pts_idx = np.asarray(ints[pos:pos + k], dtype=np.int64)
                pos += k
                if pts_idx.size >= 2:
                    polylines.append(pts_idx)
            continue

        # ---- POINT_DATA / CELL_DATA ----
        if line.startswith("POINT_DATA") or line.startswith("CELL_DATA"):
            is_point = line.startswith("POINT_DATA")
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed {parts[0]} line: {line!r}")
            n = int(parts[1])
            i += 1
            # Skip blanks
            while i < nlines_txt and not lines_txt[i].strip():
                i += 1
            if i >= nlines_txt:
                break

            def store(name: str, arr: np.ndarray) -> None:
                (point_data if is_point else cell_data)[name] = arr

            # FIELD style
            if lines_txt[i].strip().startswith("FIELD"):
                parts2 = lines_txt[i].split()
                if len(parts2) < 3:
                    raise ValueError(f"Malformed FIELD line: {lines_txt[i]!r}")
                n_arrays = int(parts2[2])
                i += 1
                for _ in range(n_arrays):
                    header = lines_txt[i].split()
                    if len(header) < 4:
                        raise ValueError(f"Malformed FIELD array header: {lines_txt[i]!r}")
                    name = header[0]
                    ncomp = int(header[1])
                    ntuple = int(header[2])
                    # dtype header[3] ignored (float/double/int)
                    i += 1
                    nvals = ncomp * ntuple
                    toks_vals, i = _read_n_tokens(lines_txt, i, nvals)
                    if len(toks_vals) < nvals:
                        raise ValueError(f"FIELD array {name!r} ended early.")
                    vals = np.asarray(toks_vals[:nvals], dtype=np.float64)
                    if ncomp == 1:
                        store(name, vals.reshape((ntuple,)))
                    else:
                        store(name, vals.reshape((ntuple, ncomp)))
                continue

            # SCALARS style
            if lines_txt[i].strip().startswith("SCALARS"):
                # SCALARS name type [numComp]
                scal = lines_txt[i].split()
                if len(scal) < 3:
                    raise ValueError(f"Malformed SCALARS line: {lines_txt[i]!r}")
                name = scal[1]
                ncomp = int(scal[3]) if len(scal) >= 4 else 1
                i += 1
                # Optional LOOKUP_TABLE line
                if i < nlines_txt and lines_txt[i].strip().startswith("LOOKUP_TABLE"):
                    i += 1
                nvals = n * ncomp
                toks_vals, i = _read_n_tokens(lines_txt, i, nvals)
                if len(toks_vals) < nvals:
                    raise ValueError(f"SCALARS array {name!r} ended early.")
                vals = np.asarray(toks_vals[:nvals], dtype=np.float64)
                if ncomp == 1:
                    store(name, vals.reshape((n,)))
                else:
                    store(name, vals.reshape((n, ncomp)))
                continue

            # If we reach here, we don't support the next data layout; skip.
            continue

        i += 1

    if points is None:
        raise ValueError(f"{path!r} does not contain a POINTS section.")
    if len(polylines) == 0:
        raise ValueError(f"{path!r} does not contain any LINES connectivity.")
    return points, polylines, point_data, cell_data


def _build_graph_from_polydata(
        points: np.ndarray,
        polylines: list[np.ndarray],
        *,
        point_radius: np.ndarray | None,
        cell_radius: np.ndarray | None,
        default_radius: float,
        reverse_edges: bool,
) -> nx.DiGraph:
    """
    Build a DiGraph where nodes are VTK point indices and edges follow polyline connectivity.

    Radius assignment:
      - If point_radius is provided (len == npoints): per-segment radius = mean(endpoint radii)
      - Else if cell_radius is provided (len == npolyline-cells): radius = cell_radius[cell_id]
      - Else: radius = default_radius
    """
    npts = int(points.shape[0])
    G = nx.DiGraph()
    for v in range(npts):
        # networks_fenicsx expects 'pos' and contiguous integer node IDs.
        G.add_node(int(v), pos=points[v].astype(np.float64, copy=False).tolist())

    pr = None
    if point_radius is not None:
        pr = np.asarray(point_radius, dtype=np.float64).reshape((npts,))

    cr = None
    if cell_radius is not None:
        cr = np.asarray(cell_radius, dtype=np.float64).reshape((-1,))

    for cell_id, conn in enumerate(polylines):
        if conn.size < 2:
            continue
        seq = conn[::-1] if reverse_edges else conn
        # Walk consecutive point pairs
        for a, b in zip(seq[:-1], seq[1:]):
            u = int(a); v = int(b)
            if u == v:
                continue
            if pr is not None:
                r = 0.5 * (float(pr[u]) + float(pr[v]))
            elif cr is not None and cell_id < cr.size:
                r = float(cr[cell_id])
            else:
                r = float(default_radius)

            if G.has_edge(u, v):
                # Keep something deterministic if duplicates exist: average them.
                old = float(G.edges[u, v].get("radius", r))
                G.edges[u, v]["radius"] = 0.5 * (old + r)
            else:
                G.add_edge(u, v, radius=float(r))
    return G


def _color_edges_like_networks_fenicsx(
        graph: nx.DiGraph,
        strategy: str | Callable[..., Any] | None,
) -> dict[tuple[int, int], int]:
    """
    Replicates networks_fenicsx.mesh.color_graph(...):
      - strategy is None  -> unique color per directed edge in graph.edges order
      - strategy not None -> greedy coloring on the line graph of graph.to_undirected()
    """
    if strategy is not None:
        undirected_edge_graph = nx.line_graph(graph.to_undirected())
        edge_coloring = nx.coloring.greedy_color(undirected_edge_graph, strategy=strategy)
    else:
        edge_coloring = {edge: i for i, edge in enumerate(graph.edges)}
    return edge_coloring


def _radius_by_tag_from_graph(
        graph: nx.DiGraph,
        *,
        color_strategy: str | Callable[..., Any] | None,
        default_radius: float,
        strict_if_grouped: bool = True,
        rtol: float = 0.0,
        atol: float = 0.0,
) -> np.ndarray:
    """
    Build a lookup array radius_by_tag where tag == subdomain marker produced by NetworkMesh.

    IMPORTANT:
      - If color_strategy is None, each edge gets its own tag, so radii can vary per edge.
      - If color_strategy groups multiple edges into the same tag, then a single radius must
        represent the whole group. With strict_if_grouped=True we error if radii differ.
    """
    edge_coloring = _color_edges_like_networks_fenicsx(graph, color_strategy)
    if len(edge_coloring) == 0:
        return np.zeros((0,), dtype=np.float64)

    num_tags = int(max(edge_coloring.values())) + 1
    buckets: list[list[float]] = [[] for _ in range(num_tags)]

    for (u, v) in graph.edges:
        key = (int(u), int(v))
        tag = edge_coloring.get(key, None)
        if tag is None:
            # Be robust if the coloring dict ended up with reversed undirected keys
            tag = edge_coloring.get((int(v), int(u)), None)
        if tag is None:
            raise RuntimeError(f"Could not find a color/tag for edge {key}.")
        r = float(graph.edges[u, v].get("radius", default_radius))
        buckets[int(tag)].append(r)

    out = np.full((num_tags,), float(default_radius), dtype=np.float64)
    for tag, vals in enumerate(buckets):
        if len(vals) == 0:
            continue
        r0 = float(vals[0])
        if strict_if_grouped and len(vals) > 1:
            # If grouping is enabled, require a single consistent radius per tag.
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if not np.isclose(vmin, vmax, rtol=float(rtol), atol=float(atol)):
                raise ValueError(
                    "color_strategy grouped multiple edges into the same subdomain tag, "
                    "but their radii differ. Use color_strategy=None (unique tag per edge), "
                    "or pass an explicit cell_radius to PressureProblem instead.\n"
                    f"tag={tag}, radius_range=[{vmin}, {vmax}]"
                )
        out[tag] = r0
    return out


@dataclass(slots=True)
class Domain3D:
    mesh: dmesh.Mesh

    # Facet (tdim-1) tags describing boundary subdomains for the tissue mesh.
    # If set, PressureProblem will apply Robin terms only on ds(outlet_marker).
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
        """
        Cache FunctionSpace objects on this Domain.

        Creating many FunctionSpaces over time can exhaust MPI communicator IDs
        (especially with MPICH) because IndexMaps build neighborhood communicators.
        Reusing spaces avoids that churn in long-running processes.
        """
        key = ("fs", element)
        V = self._cache.get(key, None)
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
        """
        Explicitly drop references to heavy objects so MPI communicators can be freed.
        After calling release(), the Domain is no longer usable.
        """
        try:
            self.clear_cache()
        except Exception:
            pass
        self.boundaries = None
        self.outlet_marker = None
        # Drop mesh last; this is what actually owns the duplicated MPI communicator.
        self.mesh = None  # type: ignore[assignment]
        collect()

    def __del__(self) -> None:
        # Best-effort cleanup; avoid raising during interpreter shutdown.
        try:
            self.clear_cache()
        except Exception:
            pass

    def axis_bounds(self, axis: int | str) -> tuple[float, float]:
        """Global (MPI) min/max coordinate along axis."""
        a = _axis_to_int(axis)
        x = self.mesh.geometry.x  # shape: (num_points_local, gdim)
        if x.size == 0:
            local_min = float("inf")
            local_max = float("-inf")
        else:
            local_min = float(np.min(x[:, a]))
            local_max = float(np.max(x[:, a]))
        gmin = self.comm.allreduce(local_min, op=MPI.MIN)
        gmax = self.comm.allreduce(local_max, op=MPI.MAX)
        return gmin, gmax

    def add_boundary_facets(
            self,
            facets: np.ndarray,
            *,
            marker: int,
            override: bool = True,
    ) -> None:
        """Tag the given boundary facets with 'marker' in self.boundaries.

        If self.boundaries already exists, merge tags. By default, new tags override
        existing tags for the same facet indices.
        """
        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        facets = np.asarray(facets, dtype=np.int32).ravel()
        if facets.size == 0:
            raise ValueError("No facets were provided to add_boundary_facets().")

        # Ensure unique/sorted indices for stable behavior
        facets = np.unique(facets)
        values = np.full((facets.size,), int(marker), dtype=np.int32)

        if self.boundaries is None:
            self.boundaries = dmesh.meshtags(self.mesh, fdim, facets, values)
        else:
            if self.boundaries.dim != fdim:
                raise ValueError(
                    f"Domain3D.boundaries has dim={self.boundaries.dim}, expected {fdim} for facet tags."
                )
            self.boundaries = _merge_meshtags(
                self.mesh, fdim, self.boundaries, facets, values, override=override
            )

        # Give facet tags a stable name for XDMF round-tripping/ParaView
        try:
            if self.boundaries is not None and not getattr(self.boundaries, "name", ""):
                self.boundaries.name = "boundaries"
        except Exception:
            pass
        # Record which tag corresponds to the Robin/outflow part of the boundary
        self.outlet_marker = int(marker)

    def mark_outlet_axis_plane(
            self,
            axis: int | str,
            *,
            value: float | None = None,
            side: str | None = None,
            tol: float | None = None,
            marker: int = 1,
            override: bool = True,
    ) -> np.ndarray:
        """Convenience: define the sink/outlet boundary as an axis-aligned plane.

        Examples:
          - x = xmax:  mark_outlet_axis_plane("x", side="max")
          - y = 0.0:   mark_outlet_axis_plane("y", value=0.0)
          - z = zmin:  mark_outlet_axis_plane(2, side="min")

        This tags the located boundary facets with 'marker' and stores them in
        Domain3D.boundaries; Domain3D.outlet_marker is set to 'marker'.
        """
        a = _axis_to_int(axis)
        if value is None:
            if side is None:
                raise ValueError("Provide either value=... or side='min'/'max'.")
            s = side.lower().strip()
            amin, amax = self.axis_bounds(a)
            if s == "min":
                value = amin
            elif s == "max":
                value = amax
            else:
                raise ValueError("side must be 'min' or 'max'.")

        # NOTE: np.isclose default rtol=1e-5 can lead to major over-selection if
        # coordinates have a large absolute offset. We always set rtol=0 so only
        # atol matters, and choose a robust default atol if not provided.
        tol_was_auto = tol is None
        if tol is None:
            amin, amax = self.axis_bounds(a)
            extent = abs(amax - amin)
            tol = max(
                1e-8 * max(1.0, extent),
                1e-12 * max(1.0, abs(float(value))),
                )

        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        # Build connectivities used by locate_entities_boundary and later measures
        # Connectivity isn't strictly required for locate_entities_boundary, but
        # creating it improves robustness across versions and later postprocessing.
        self.mesh.topology.create_connectivity(fdim, 0)
        self.mesh.topology.create_connectivity(0, fdim)

        def _locate(with_tol: float) -> np.ndarray:
            def plane_marker(x: np.ndarray) -> np.ndarray:
                # x has shape (gdim, num_points)
                return np.isclose(x[a], float(value), atol=float(with_tol), rtol=0.0)

            return dmesh.locate_entities_boundary(self.mesh, fdim, plane_marker)

        facets = _locate(float(tol))

        # If tol was auto-picked, and nothing was found LOCALLY, relax a few times.
        # This helps when the boundary plane isn't represented exactly in floating point.
        if facets.size == 0 and tol_was_auto:
            for factor in (10.0, 100.0, 1000.0, 10000.0):
                facets = _locate(float(tol) * factor)
                if facets.size:
                    tol = float(tol) * factor
                    break

        # In parallel, it's normal that some ranks own zero facets on this plane.
        # Only error if NO rank found any facets.
        n_global = self.comm.allreduce(int(facets.size), op=MPI.SUM)
        if n_global == 0:
            raise ValueError(
                f"mark_outlet_axis_plane found no boundary facets on any rank for axis={axis!r}, "
                f"value={value}, tol={tol}. (Try increasing tol.)"
            )

        # Ensure boundaries MeshTags exists on ALL ranks (can be empty locally),
        # so downstream code consistently uses ds(subdomain_data=...) everywhere.
        if self.boundaries is None:
            empty = np.zeros((0,), dtype=np.int32)
            self.boundaries = dmesh.meshtags(self.mesh, fdim, empty, empty)
            try:
                self.boundaries.name = "boundaries"
            except Exception:
                pass

        # Add local facets if we have any on this rank
        if facets.size:
            self.add_boundary_facets(facets, marker=marker, override=override)
        else:
            # Still record the outlet marker even on ranks with no local facets
            self.outlet_marker = int(marker)
        return facets

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
        """Construct a Domain3D by reading a mesh from an XDMF (HDF5-backed) file."""
        mesh = read_mesh_xdmf(comm, path, mesh_name=mesh_name, ghost_mode=ghost_mode)
        dom = cls(mesh=mesh)

        if boundaries_name is not None:
            bpath = path if boundaries_path is None else boundaries_path
            fdim = mesh.topology.dim - 1
            dom.boundaries = read_meshtags_xdmf(mesh, bpath, name=boundaries_name, dim=fdim)
            # Ensure a stable tag name for round-tripping
            try:
                if dom.boundaries is not None and not getattr(dom.boundaries, "name", ""):
                    dom.boundaries.name = str(boundaries_name)
            except Exception:
                pass
            if outlet_marker is not None:
                dom.outlet_marker = int(outlet_marker)

        return dom

    @classmethod
    def from_meshfile(
            cls,
            comm: MPI.Comm,
            path: str | Path,
            **kwargs: Any,
    ) -> "Domain3D":
        """Alias for from_xdmf(...)."""
        return cls.from_xdmf(comm, path, **kwargs)

    def set_boundaries_from_xdmf(
            self,
            path: str | Path,
            *,
            name: str,
            outlet_marker: int | None = None,
            replace: bool = True,
            override: bool = True,
    ) -> None:
        """Load facet MeshTags from an XDMF file into this Domain3D."""
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        tags = read_meshtags_xdmf(mesh, path, name=name, dim=fdim)
        try:
            tags.name = str(name)
        except Exception:
            pass

        if replace or self.boundaries is None:
            self.boundaries = tags
        else:
            # Merge facet tags; new tags can override existing ones if override=True
            self.boundaries = _merge_meshtags(
                mesh,
                fdim,
                self.boundaries,
                np.asarray(tags.indices, dtype=np.int32),
                np.asarray(tags.values, dtype=np.int32),
                override=override,
            )

        if outlet_marker is not None:
            self.outlet_marker = int(outlet_marker)

    def mark_outlet_from_xdmf(
            self,
            path: str | Path,
            *,
            tags_name: str,
            marker: int,
            replace_boundaries: bool = True,
            override: bool = True,
    ) -> np.ndarray:
        """Mark the outlet/sink boundary using facet tags loaded from XDMF."""
        marker_i = int(marker)

        if replace_boundaries:
            self.set_boundaries_from_xdmf(path, name=tags_name, outlet_marker=marker_i, replace=True)
            assert self.boundaries is not None
            facets = np.asarray(self.boundaries.indices, dtype=np.int32)[
                np.asarray(self.boundaries.values, dtype=np.int32) == marker_i
                ]
        else:
            facets = load_boundary_facets_from_xdmf(self.mesh, path, tags_name=tags_name, marker=marker_i)
            if facets.size == 0:
                raise ValueError(
                    f"No facets found with marker={marker_i} in MeshTags {tags_name!r} from {path!r}."
                )
            self.add_boundary_facets(facets, marker=marker_i, override=override)

        if facets.size == 0:
            raise ValueError(
                f"MeshTags {tags_name!r} from {path!r} contains no facets with marker={marker_i}."
            )

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
        mesh = dmesh.create_box(comm, [min_corner.tolist(), max_corner.tolist()], n, cell_type=cell_type)
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
        V = self._cache.get(key, None)
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
        """
        Explicitly drop references to heavy objects so MPI communicators can be freed.
        After calling release(), the Domain is no longer usable.
        """
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
        return indices[values == marker].astype(np.int32, copy=False)

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
            N=points_per_edge,
            comm=comm,
            graph_rank=graph_rank,
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
        """
        Construct a 1D network domain from an ASCII legacy VTK POLYDATA file.
        """
        p = Path(path).expanduser().resolve()

        graph: nx.DiGraph | None = None
        radius_by_tag: np.ndarray | None = None

        if comm.rank == graph_rank:
            pts, polylines, pdat, cdat = _read_vtk_legacy_polydata_ascii(p)
            npts = int(pts.shape[0])

            pr = pdat.get(radius_name, None)
            if pr is not None:
                pr = np.asarray(pr, dtype=np.float64).reshape((-1,))
                if pr.size != npts:
                    # Not a point-radius array, ignore as point-radius
                    pr = None

            cr = cdat.get(radius_name, None)
            if cr is not None:
                cr = np.asarray(cr, dtype=np.float64).reshape((-1,))

            graph = _build_graph_from_polydata(
                pts,
                polylines,
                point_radius=pr,
                cell_radius=cr,
                default_radius=float(default_radius),
                reverse_edges=bool(reverse_edges),
            )

            radius_by_tag = _radius_by_tag_from_graph(
                graph,
                color_strategy=color_strategy,
                default_radius=float(default_radius),
                strict_if_grouped=bool(strict_if_grouped),
            )

        radius_by_tag = comm.bcast(radius_by_tag, root=graph_rank)

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

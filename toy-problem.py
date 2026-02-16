from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Mapping, Tuple

import dolfinx.mesh as dmesh
import numpy as np
from mpi4py import MPI
import ufl

from src import (
    Parameters,
    AssemblyOptions,
    SolverOptions,
    OutputNames,
    OutputOptions,
    write_solution,
)
from src.domain import Domain1D, Domain3D
from src.domain.io import read_vtk_legacy_polydata_ascii
from src.domain.graph import build_graph_from_polydata, compute_radius_by_tag
from src.problem import PressureVelocityProblem
from src.system import (
    make_rank_logger,
    print_environment,
    setup_mpi_debug,
    setup_faulthandler,
    barrier as mpi_barrier,
)

def _max_radius(radius_by_tag: Mapping[int, float] | np.ndarray | None, fallback: float) -> float:
    if radius_by_tag is None:
        return float(fallback)
    if isinstance(radius_by_tag, np.ndarray):
        arr = radius_by_tag
    else:
        arr = np.fromiter(radius_by_tag.values(), dtype=np.float64)
    if arr.size == 0:
        return float(fallback)
    return float(np.max(arr))


def _make_coordinate_element(cell: ufl.Cell, gdim: int) -> object:
    try:
        import basix
        import basix.ufl

        try:
            return basix.ufl.element("Lagrange", basix.CellType.interval, 1, shape=(gdim,))
        except TypeError:
            return basix.ufl.element("Lagrange", basix.CellType.interval, 1)
    except Exception:
        pass

    VectorElement = getattr(ufl, "VectorElement", None)
    if VectorElement is None:
        try:
            from ufl.finiteelement import VectorElement as VectorElement  # type: ignore[assignment]
        except Exception:
            try:
                from ufl.element import VectorElement as VectorElement  # type: ignore[assignment]
            except Exception:
                VectorElement = None
    if VectorElement is not None:
        try:
            return VectorElement("Lagrange", cell, 1, dim=gdim)
        except TypeError:
            return VectorElement("Lagrange", cell, 1)

    FiniteElement = getattr(ufl, "FiniteElement", None)
    if FiniteElement is None:
        try:
            from ufl.finiteelement import FiniteElement as FiniteElement  # type: ignore[assignment]
        except Exception:
            try:
                from ufl.element import FiniteElement as FiniteElement  # type: ignore[assignment]
            except Exception:
                FiniteElement = None
    if FiniteElement is not None:
        for kwargs in ({"value_shape": (gdim,)}, {"shape": (gdim,)}):
            try:
                return FiniteElement("Lagrange", cell, 1, **kwargs)
            except TypeError:
                continue
        return FiniteElement("Lagrange", cell, 1)

    raise RuntimeError("Could not build a coordinate element for the interval mesh.")


def _create_mesh_compat(
    comm: MPI.Comm,
    cells: np.ndarray,
    coords: np.ndarray,
    domain: ufl.Mesh,
) -> dmesh.Mesh:
    try:
        return dmesh.create_mesh(comm, cells, coords, domain)
    except TypeError:
        return dmesh.create_mesh(comm, cells, domain, coords)
    except ValueError as exc:
        try:
            return dmesh.create_mesh(comm, cells, domain, coords)
        except Exception:
            raise exc


def _graph_to_arrays(graph: object) -> Tuple[np.ndarray, np.ndarray, dict[int, int]]:
    nodes = list(graph.nodes)  # type: ignore[attr-defined]
    node_index = {int(n): i for i, n in enumerate(nodes)}
    coords = np.zeros((len(nodes), 3), dtype=np.float64)
    for n in nodes:
        pos = graph.nodes[n]["pos"]  # type: ignore[index]
        coords[node_index[int(n)], :] = np.asarray(pos, dtype=np.float64)
    edges = [(node_index[int(u)], node_index[int(v)]) for u, v in graph.edges]  # type: ignore[attr-defined]
    cells = np.asarray(edges, dtype=np.int64)
    return coords, cells, node_index


def _refine_edges(
    coords: np.ndarray,
    cells: np.ndarray,
    points_per_edge: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_per_edge <= 1 or cells.size == 0:
        cell_tags = np.arange(cells.shape[0], dtype=np.int32)
        return coords, cells, cell_tags

    n_points = int(points_per_edge)
    if n_points < 2:
        n_points = 2

    new_coords = [coords[i].astype(np.float64, copy=False) for i in range(coords.shape[0])]
    new_cells: list[Tuple[int, int]] = []
    new_tags: list[int] = []

    for edge_idx, (u, v) in enumerate(cells):
        u = int(u)
        v = int(v)
        p0 = coords[u]
        p1 = coords[v]
        if n_points == 2:
            seq = [u, v]
        else:
            seq = [u]
            for k in range(1, n_points - 1):
                t = float(k) / float(n_points - 1)
                p = (1.0 - t) * p0 + t * p1
                new_coords.append(np.asarray(p, dtype=np.float64))
                seq.append(len(new_coords) - 1)
            seq.append(v)

        for a, b in zip(seq[:-1], seq[1:]):
            new_cells.append((int(a), int(b)))
            new_tags.append(int(edge_idx))

    coords_out = np.vstack(new_coords)
    cells_out = np.asarray(new_cells, dtype=np.int64)
    tags_out = np.asarray(new_tags, dtype=np.int32)
    return coords_out, cells_out, tags_out


def _infer_sources_sinks_from_graph(
    graph: object,
    node_index: dict[int, int],
    coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sources = [n for n in graph.nodes if graph.in_degree(n) == 0 and graph.out_degree(n) > 0]  # type: ignore[attr-defined]
    sinks = [n for n in graph.nodes if graph.out_degree(n) == 0 and graph.in_degree(n) > 0]  # type: ignore[attr-defined]

    if len(sources) == 0 or len(sinks) == 0:
        deg = dict(graph.degree())  # type: ignore[attr-defined]
        terminals = [n for n, d in deg.items() if int(d) == 1]
        if len(terminals) >= 2:
            xs = np.array([coords[node_index[int(n)], 0] for n in terminals], dtype=np.float64)
            inlet = terminals[int(np.argmin(xs))]
            outlet = terminals[int(np.argmax(xs))]
            sources = [inlet]
            sinks = [outlet]
        else:
            raise ValueError("Unable to infer inlet/outlet vertices from graph connectivity.")

    sink_set = {int(s) for s in sinks}
    sources = [int(s) for s in sources if int(s) not in sink_set]
    return (
        np.array([node_index[int(s)] for s in sources], dtype=np.int32),
        np.array([node_index[int(s)] for s in sinks], dtype=np.int32),
    )


def _build_domain1d_from_graph_direct(
    comm: MPI.Comm,
    coords: np.ndarray,
    cells: np.ndarray,
    *,
    sources: np.ndarray,
    sinks: np.ndarray,
    cell_tags: np.ndarray,
    radius_by_tag: np.ndarray,
) -> Domain1D:
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords with shape (N,3); got {coords.shape}.")
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError(f"Expected cells with shape (N,2); got {cells.shape}.")

    cell = ufl.Cell("interval")
    coord_el = _make_coordinate_element(cell, gdim=3)
    domain = ufl.Mesh(coord_el)
    mesh = _create_mesh_compat(comm, cells, coords, domain)

    inlet_marker = 1
    outlet_marker = 2
    boundary_indices = np.concatenate([sources, sinks]).astype(np.int32, copy=False)
    boundary_values = np.concatenate(
        [
            np.full((sources.size,), inlet_marker, dtype=np.int32),
            np.full((sinks.size,), outlet_marker, dtype=np.int32),
        ]
    )
    boundaries = dmesh.meshtags(mesh, 0, boundary_indices, boundary_values)

    cell_indices = np.arange(cells.shape[0], dtype=np.int32)
    if cell_tags.shape[0] != cell_indices.shape[0]:
        raise ValueError("cell_tags length does not match number of cells.")
    subdomains = dmesh.meshtags(mesh, 1, cell_indices, cell_tags.astype(np.int32, copy=False))

    return Domain1D(
        mesh=mesh,
        boundaries=boundaries,
        subdomains=subdomains,
        inlet_marker=inlet_marker,
        outlet_marker=outlet_marker,
        radius_by_tag=radius_by_tag,
    )

def solve_coupled_test_graph(
        outdir: Path,
        *,
        params: Parameters = Parameters(),
        N_per_edge: int = 12,
        tissue_h: float = 0.002,
        degree_3d: int = 1,
        degree_1d: int = 1,
        circle_quad_degree: int = 20,
        output_format: str = "xdmf",
):
    comm = MPI.COMM_WORLD
    setup_mpi_debug(comm)

    rprint = make_rank_logger(comm)

    def barrier(tag: str) -> None:
        mpi_barrier(comm, tag, rprint)

    setup_faulthandler(rprint=rprint)
    print_environment(comm, rprint)

    barrier("start")

    try:
        rprint(f"outdir={outdir}")
        t0 = time.time()
        outdir.mkdir(parents=True, exist_ok=True)
        rprint(f"outdir.mkdir done in {time.time() - t0:.3f}s")
        barrier("after mkdir")

        network_path = Path(__file__).resolve().parent / "network.vtk"
        if comm.rank == 0 and not network_path.exists():
            raise FileNotFoundError(str(network_path))

        rprint(f"Reading VTK network: {network_path}")
        t0 = time.time()
        vtk_default_radius = 1.0
        radius_name = "Radius"
        graph_rank = 0

        graph = None
        radius_by_tag = None
        bounds = None
        coords = None
        cells = None
        sources = None
        sinks = None

        if comm.rank == graph_rank:
            pts, polylines, pdat, cdat = read_vtk_legacy_polydata_ascii(network_path)
            npts = int(pts.shape[0])

            pr = pdat.get(radius_name, None)
            if pr is not None:
                pr = np.asarray(pr, dtype=np.float64).reshape((-1,))
                if pr.size != npts:
                    pr = None

            cr = cdat.get(radius_name, None)
            if cr is not None:
                cr = np.asarray(cr, dtype=np.float64).reshape((-1,))

            graph = build_graph_from_polydata(
                pts,
                polylines,
                point_radius=pr,
                cell_radius=cr,
                default_radius=float(vtk_default_radius),
                reverse_edges=False,
            )

            radius_by_tag = compute_radius_by_tag(
                graph,
                color_strategy=None,
                default_radius=float(vtk_default_radius),
                strict_if_grouped=True,
            )

            coords, cells, node_index = _graph_to_arrays(graph)
            coords, cells, cell_tags = _refine_edges(coords, cells, N_per_edge)
            sources, sinks = _infer_sources_sinks_from_graph(graph, node_index, coords)
            bounds = (np.min(coords, axis=0), np.max(coords, axis=0))
        else:
            cell_tags = None

        radius_by_tag = comm.bcast(radius_by_tag, root=graph_rank)
        bounds = comm.bcast(bounds, root=graph_rank)
        coords = comm.bcast(coords, root=graph_rank)
        cells = comm.bcast(cells, root=graph_rank)
        cell_tags = comm.bcast(cell_tags, root=graph_rank)
        sources = comm.bcast(sources, root=graph_rank)
        sinks = comm.bcast(sinks, root=graph_rank)
        if bounds is None:
            raise RuntimeError("Failed to compute network bounds from VTK on graph_rank.")
        if coords is None or cells is None or cell_tags is None or sources is None or sinks is None:
            raise RuntimeError("Failed to broadcast graph mesh data from graph_rank.")

        network = _build_domain1d_from_graph_direct(
            comm,
            coords,
            cells,
            sources=sources,
            sinks=sinks,
            cell_tags=cell_tags,
            radius_by_tag=radius_by_tag,
        )
        rprint(f"Network mesh created from graph in {time.time() - t0:.3f}s")
        barrier("after NetworkMesh")

        lmbda = network.mesh
        rprint(f"Network mesh: tdim={lmbda.topology.dim}, gdim={lmbda.geometry.dim}, comm.size={lmbda.comm.size}")
        for dim in [0, lmbda.topology.dim]:
            im = lmbda.topology.index_map(dim)
            rprint(f"index_map(dim={dim}): size_local={im.size_local}, num_ghosts={im.num_ghosts}")

        bnd = network.boundaries
        rprint(
            f"boundaries: indices.shape={bnd.indices.shape}, values.shape={bnd.values.shape}, "
            f"values_unique={np.unique(bnd.values) if bnd.values.size else 'EMPTY'}"
        )
        if network.subdomains is None:
            rprint("subdomains: None  (!!! this would break radius build)")
        else:
            sd = network.subdomains
            rprint(
                f"subdomains: indices.shape={sd.indices.shape}, values.shape={sd.values.shape}, "
                f"values_unique={np.unique(sd.values) if sd.values.size else 'EMPTY'}"
            )

        # Match the original marker print exactly:
        inlet_marker = network.inlet_marker  # this is NetworkMesh.out_marker
        outlet_marker = network.outlet_marker  # this is NetworkMesh.in_marker
        rprint(
            f"Markers: in_marker={outlet_marker}, out_marker={inlet_marker} "
            f"=> inlet_marker={inlet_marker}, outlet_marker={outlet_marker}"
        )
        rprint(f"inlet_vertices(local view)={network.inlet_vertices.tolist()}")
        rprint(f"outlet_vertices(local view)={network.outlet_vertices.tolist()}")
        barrier("after markers/vertices")

        if radius_by_tag is None:
            rprint("radius_by_tag: None (this would break radius build)")
            radius_arr = np.zeros((0,), dtype=np.float64)
        elif isinstance(radius_by_tag, np.ndarray):
            radius_arr = radius_by_tag
        else:
            radius_arr = np.fromiter(radius_by_tag.values(), dtype=np.float64)

        if radius_arr.size:
            rprint(
                "radius_by_tag stats: "
                f"n={radius_arr.size}, min={float(np.min(radius_arr))}, max={float(np.max(radius_arr))}"
            )
        else:
            rprint("radius_by_tag stats: EMPTY (using fallback radius)")

        max_r = _max_radius(radius_by_tag, fallback=vtk_default_radius)

        net_min = np.asarray(bounds[0], dtype=np.float64)
        net_max = np.asarray(bounds[1], dtype=np.float64)
        if not (np.all(np.isfinite(net_min)) and np.all(np.isfinite(net_max))):
            raise RuntimeError("Failed to compute finite network bounds for tissue box.")

        extent = net_max - net_min
        max_extent = float(np.max(extent))
        pad = max(2.0 * max_r, 0.05 * max_extent, 2.0 * float(tissue_h))
        mn = net_min - pad
        mx = net_max + pad

        extent_box = mx - mn
        n_est = np.maximum(2, np.ceil(extent_box / float(tissue_h))).astype(int)
        max_n_env = int(os.environ.get("TISSUE_MAX_N", "64"))
        max_cells_env = int(os.environ.get("TISSUE_MAX_CELLS", "500000"))
        effective_h = float(tissue_h)
        cell_factor = 6

        n_cells_est = int(np.prod(n_est) * cell_factor)
        for _ in range(4):
            scale_n = 1.0
            scale_c = 1.0
            if max_n_env > 0 and int(np.max(n_est)) > max_n_env:
                scale_n = float(np.max(n_est)) / float(max_n_env)
            if max_cells_env > 0 and n_cells_est > max_cells_env:
                scale_c = (float(n_cells_est) / float(max_cells_env)) ** (1.0 / 3.0)
            scale = max(scale_n, scale_c)
            if scale <= 1.0:
                break
            effective_h *= scale
            n_est = np.maximum(2, np.ceil(extent_box / effective_h)).astype(int)
            n_cells_est = int(np.prod(n_est) * cell_factor)

        rprint(
            "Building tissue mesh: "
            f"mn={mn.tolist()} mx={mx.tolist()} pad={pad:.6g} h={effective_h} "
            f"n={n_est.tolist()} est_cells~{n_cells_est}"
        )
        t0 = time.time()
        tissue = Domain3D.from_box(
            comm,
            mn,
            mx,
            target_h=effective_h,
            cell_type=dmesh.CellType.tetrahedron,
        )
        tissue.mark_outlet_axis_plane("x", side="max", marker=7)
        omega = tissue.mesh
        rprint(f"Tissue mesh created in {time.time() - t0:.3f}s (tdim={omega.topology.dim}, gdim={omega.geometry.dim})")
        im3 = omega.topology.index_map(omega.topology.dim)
        rprint(f"omega cells: local={im3.size_local}, ghosts={im3.num_ghosts}")
        barrier("after tissue mesh")

        sd = network.subdomains
        if sd is None:
            raise RuntimeError("network.subdomains is None; cannot build DG0 cell radii")
        if radius_by_tag is None:
            raise RuntimeError("radius_by_tag is None; cannot build DG0 cell radii")

        if sd.values.size:
            max_tag_local = int(np.max(sd.values))
        else:
            max_tag_local = -1
        max_tag_global = comm.allreduce(max_tag_local, op=MPI.MAX)
        rprint(
            f"subdomain max_tag_local={max_tag_local}, max_tag_global={max_tag_global}, "
            f"radius_by_tag_max_index={radius_arr.size - 1}"
        )
        if max_tag_global >= radius_arr.size:
            rprint("!!! ERROR: subdomain tag exceeds radius_by_tag size; would crash indexing.")
        barrier("before solve")

        assembly = AssemblyOptions(
            degree_3d=degree_3d,
            degree_1d=degree_1d,
            circle_quadrature_degree=circle_quad_degree,
        )

        solver = SolverOptions(
            petsc_options_prefix="la_test_graph",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
        )

        with PressureVelocityProblem(
                tissue,
                network,
                params=params,
                assembly=assembly,
                solver=solver,
                radius_by_tag=radius_by_tag,
                default_radius=float(max_r),
                log=rprint,
                barrier=barrier,
        ) as prob:
            sol = prob.solve()

        # Match original field names for output
        sol.tissue_pressure.name = "p_t"
        sol.network_pressure.name = "P"
        if sol.tissue_velocity is not None:
            sol.tissue_velocity.name = "v_tissue"

        # Match original field names for output
        sol.tissue_pressure.name = "p_t"
        sol.network_pressure.name = "P"

        if comm.rank == 0:
            print("LAM test solved!", flush=True)

        if os.environ.get("SKIP_IO", "0") == "1":
            rprint("SKIP_IO=1 -> skipping all output writing.")
            return sol

        fmt = output_format.lower()
        rprint(f"Writing output, format={fmt}")

        names = OutputNames(
            tissue_pressure="p_t",
            network="network",
            network_vtx="P",
            tissue_velocity="v_tissue",
        )
        write_solution(
            outdir,
            tissue,
            network,
            sol,
            options=OutputOptions(format=fmt, time=0.0, write_meshtags=True, names=names),
        )

        if fmt == "vtx":
            rprint("Wrote VTX omega p_t.bp")
            rprint("Wrote VTX lmbda P.bp")
        elif fmt == "vtk":
            rprint("Wrote VTK omega p_t.pvd")
            rprint("Wrote VTK lmbda network.pvd")
        else:
            rprint("Wrote XDMF omega p_t.xdmf")
            rprint("Wrote XDMF lmbda network.xdmf")

        barrier("after IO")

        if comm.rank == 0:
            print(f"Results written to: {outdir} (format={fmt})", flush=True)

        return sol

    except Exception as e:
        # Mirror the original “abort to avoid deadlock” behavior
        from src.system import abort_on_exception
        abort_on_exception(comm, rprint, e)
        raise  # unreachable after Abort, but keeps linters happy


def main():
    results_root = Path(__file__).resolve().parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = results_root / timestamp

    fmt = os.environ.get("DOLFINX_OUTPUT_FORMAT", "vtk")
    solve_coupled_test_graph(outdir=outdir, output_format=fmt)


if __name__ == "__main__":
    main()

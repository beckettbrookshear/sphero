#!/usr/bin/env python3
"""
Simple 3D airflow visualization demo.

Usage:
  python simulate_flow.py --mesh path/to/model.stl

This script loads a 3D mesh (STL/OBJ), creates a voxel grid around it,
builds a heuristic velocity field that diverts flow around the object,
integrates streamlines from a plane upstream, and shows the result with PyVista.

This is a visual demo (not a full CFD solver).
"""
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator


def build_grid(mesh, padding=0.2, nx=80, ny=80, nz=80):
    bounds = mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    pad = max(dx, dy, dz) * padding
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    zmin -= pad
    zmax += pad
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return (x, y, z, X, Y, Z, points)


def compute_distance_field(mesh_points, grid_points):
    tree = cKDTree(mesh_points)
    dists, _ = tree.query(grid_points)
    return dists


def build_velocity_field(x, y, z, points, mesh, U=1.0, d0=None, kappa=1.5):
    # compute distance to mesh
    dists = compute_distance_field(mesh.points, points)
    nx = len(x); ny = len(y); nz = len(z)
    D = dists.reshape((ny, nx, nz))  # note meshgrid used indexing='xy'
    if d0 is None:
        d0 = max(x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]) * 0.05

    f = np.exp(-(D / d0) ** 2)

    # gradient of the scalar 'f' to push flow around the object
    gx, gy, gz = np.gradient(f, x, y, z, edge_order=2)
    grad = np.stack((gx, gy, gz), axis=-1)

    # lateral component proportional to gradient magnitude
    lateral = -kappa * grad * f[..., np.newaxis]

    # main flow in +X direction reduced near the object
    vx = (U * (1 - f))

    V = np.empty((ny, nx, nz, 3), dtype=float)
    V[..., 0] = vx
    V[..., 1] = lateral[..., 1]
    V[..., 2] = lateral[..., 2]

    return V, D


def integrate_streamlines(V, x, y, z, seeds, dt=0.5, nsteps=500, min_dist=1e-3):
    nx, ny, nz = len(x), len(y), len(z)
    interp_vx = RegularGridInterpolator((y, x, z), V[..., 0], bounds_error=False, fill_value=None)
    interp_vy = RegularGridInterpolator((y, x, z), V[..., 1], bounds_error=False, fill_value=None)
    interp_vz = RegularGridInterpolator((y, x, z), V[..., 2], bounds_error=False, fill_value=None)

    streamlines = []
    for s in seeds:
        pts = [s.copy()]
        p = s.copy()
        for _ in range(nsteps):
            try:
                valx = interp_vx((p[1], p[0], p[2]))
                valy = interp_vy((p[1], p[0], p[2]))
                valz = interp_vz((p[1], p[0], p[2]))
            except Exception:
                break
            if valx is None or np.isnan(valx):
                break
            v = np.array([float(valx), float(valy), float(valz)])
            if np.linalg.norm(v) < 1e-6:
                break
            p = p + v * dt
            pts.append(p.copy())
            if not (x[0] <= p[0] <= x[-1] and y[0] <= p[1] <= y[-1] and z[0] <= p[2] <= z[-1]):
                break
        if len(pts) > 1:
            streamlines.append(np.array(pts))
    return streamlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Path to STL/OBJ mesh file")
    parser.add_argument("--U", type=float, default=1.0, help="Freestream speed")
    parser.add_argument("--res", type=int, default=80, help="Grid resolution (max for each axis)")
    args = parser.parse_args()

    mesh = pv.read(args.mesh)
    nx = ny = nz = args.res
    x, y, z, X, Y, Z, points = build_grid(mesh, nx=nx, ny=ny, nz=nz)

    V, D = build_velocity_field(x, y, z, points, mesh, U=args.U)

    # create seeds on a plane upstream of the object
    xmin = x[0]
    ys = np.linspace(y[0], y[-1], 12)
    zs = np.linspace(z[0], z[-1], 12)
    seeds = []
    for yy in ys:
        for zz in zs:
            seeds.append(np.array([xmin + (x[-1]-x[0]) * 0.02, yy, zz]))

    streamlines = integrate_streamlines(V, x, y, z, seeds, dt=0.5, nsteps=800)

    # Build PyVista objects for visualization
    pl = pv.Plotter()
    pl.add_mesh(mesh, color="lightgray", opacity=1.0)

    # add streamlines
    for pts in streamlines:
        spline = pv.Spline(pts, len(pts))
        pl.add_mesh(spline, color="red", line_width=2)

    pl.add_axes()
    pl.show(cpos="iso")


if __name__ == "__main__":
    main()

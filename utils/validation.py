"""Validation shared by STL upload and voxel preprocessing."""

import numpy as np


def validate_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError("This STL has no triangles. Upload a valid triangle mesh.")
    if not np.isfinite(vertices).all():
        raise ValueError("This STL contains invalid coordinates. Re-export it from your CAD tool.")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("This STL contains invalid triangle references. Re-export the mesh.")
    with np.errstate(over="ignore", invalid="ignore"):
        corners = vertices[triangles]
        areas = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
    if not np.isfinite(areas).all() or not np.any(areas > 0):
        raise ValueError("This STL has no usable surface or its coordinates are too large. Check its geometry and scale.")


def validate_voxels(voxels, dimension):
    voxels = np.asarray(voxels)
    if voxels.shape != (dimension,) * 3 or not np.isfinite(voxels).all():
        raise ValueError("Voxelization produced invalid data. Check the mesh geometry and scale.")
    if not np.isin(voxels, [0, 1]).all() or not np.any(voxels):
        raise ValueError("Voxelization produced no usable occupied cells. Try another mesh or increase point density.")

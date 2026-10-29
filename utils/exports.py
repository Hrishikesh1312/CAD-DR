"""Portable exports using the exact voxel-grid transform."""
import io
import json
import struct

import numpy as np
from skimage.measure import marching_cubes


def numpy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def voxel_surface(voxels, transform):
    if not np.any(voxels):
        raise ValueError("No occupied voxels to export. Lower the reconstruction threshold.")
    vertices, faces, _, _ = marching_cubes(np.pad(voxels, 1), level=0.5)
    vertices = (vertices - 1) * np.asarray(transform['spacing']) + np.asarray(transform['origin'])
    return vertices, faces


def binary_stl(voxels, transform):
    vertices, faces = voxel_surface(voxels, transform)
    triangles = vertices[faces]
    # Marching cubes' default ordering points inward for occupied cells.
    triangles = triangles[:, [0, 2, 1]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    np.divide(normals, norms[:, None], out=normals, where=norms[:, None] > 0)
    records = np.zeros(len(triangles), dtype=[('normal', '<f4', (3,)),
                                            ('vertices', '<f4', (3, 3)),
                                            ('attribute', '<u2')])
    records['normal'] = normals
    records['vertices'] = triangles
    return b'CAD-DR reconstruction'.ljust(80, b'\0') + struct.pack('<I', len(triangles)) + records.tobytes()


def latent_archive(latent, transform, metadata):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, latent=latent, origin=transform['origin'],
                        spacing=transform['spacing'], metadata=np.asarray(json.dumps(metadata)))
    return buffer.getvalue()

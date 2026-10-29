import unittest
from types import SimpleNamespace

import numpy as np

from utils.validation import validate_mesh, validate_voxels


class ValidationTests(unittest.TestCase):
    def mesh(self, vertices, triangles=((0, 1, 2),)):
        return SimpleNamespace(vertices=np.asarray(vertices), triangles=np.asarray(triangles))

    def test_valid_surface_including_planar_mesh(self):
        validate_mesh(self.mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

    def test_invalid_meshes(self):
        meshes = [
            self.mesh([], []),
            self.mesh([[0, 0, 0]] * 3),
            self.mesh([[0, 0, 0], [1, 0, 0], [np.nan, 1, 0]]),
            self.mesh([[0, 0, 0]] * 3, [[0, 1, 3]]),
        ]
        for mesh in meshes:
            with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                validate_mesh(mesh)

    def test_valid_voxels(self):
        voxels = np.zeros((4, 4, 4))
        voxels[1, 1, 1] = 1
        validate_voxels(voxels, 4)

    def test_invalid_voxels(self):
        for voxels in [np.zeros((4, 4, 4)), np.ones((2, 2, 2)),
                       np.full((4, 4, 4), np.nan), np.full((4, 4, 4), 0.5)]:
            with self.subTest(shape=voxels.shape), self.assertRaises(ValueError):
                validate_voxels(voxels, 4)


if __name__ == '__main__':
    unittest.main()

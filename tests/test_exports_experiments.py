import io
import json
import struct
import unittest

import numpy as np
import open3d as o3d

from utils.conversion_utils import ConversionUtils
from utils.exports import binary_stl, latent_archive, voxel_surface
from utils.experiments import threshold_sweep


class ExportTests(unittest.TestCase):
    def test_grid_transform_matches_source_bounds(self):
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector([[10, 20, 30], [18, 28, 38]])
        voxels, transform = ConversionUtils.convert_pointcloud_to_binvox(points, 4, return_transform=True)
        np.testing.assert_allclose(transform['origin'], [11, 21, 31])
        np.testing.assert_allclose(transform['spacing'], [2, 2, 2])
        vertices, _ = voxel_surface(np.ones((4, 4, 4), dtype=np.uint8), transform)
        np.testing.assert_allclose(vertices.min(axis=0), [10, 20, 30])
        np.testing.assert_allclose(vertices.max(axis=0), [18, 28, 38])
        self.assertEqual(voxels.sum(), 2)

    def test_binary_stl_has_outward_normals_and_valid_records(self):
        data = binary_stl(np.ones((2,2,2), dtype=np.uint8), {'origin':[0,0,0], 'spacing':[1,1,1]})
        count = struct.unpack('<I', data[80:84])[0]
        self.assertEqual(len(data), 84 + count * 50)
        records = np.frombuffer(data[84:], dtype=[('normal','<f4',(3,)), ('vertices','<f4',(3,3)), ('attribute','<u2')])
        triangles = records['vertices']
        signed_volume = np.einsum('ij,ij->i', triangles[:,0], np.cross(triangles[:,1], triangles[:,2])).sum() / 6
        self.assertGreater(signed_volume, 0)
        np.testing.assert_allclose(np.linalg.norm(records['normal'], axis=1), 1, atol=1e-6)

    def test_empty_export_and_archive_round_trip(self):
        transform = {'origin':[1,2,3], 'spacing':[.5,.5,.5]}
        with self.assertRaises(ValueError):
            binary_stl(np.zeros((2,2,2)), transform)
        latent = np.arange(8, dtype=np.float32).reshape(1,1,2,2,2)
        with np.load(io.BytesIO(latent_archive(latent, transform, {'format_version':1})), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['latent'], latent)
            np.testing.assert_array_equal(archive['origin'], [1,2,3])
            self.assertEqual(json.loads(archive['metadata'].item())['format_version'], 1)


class ExperimentTests(unittest.TestCase):
    def test_known_optimum_and_strict_threshold(self):
        rows = threshold_sweep(np.array([0,1,1]), np.array([.2,.6,.8]), [.2,.6,1])
        self.assertEqual(rows[0]['IoU'], 1)
        self.assertEqual(rows[1]['Recall'], .5)
        self.assertEqual(rows[2]['F1'], 0)

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            threshold_sweep(np.ones(2), np.array([np.nan, .1]))


if __name__ == '__main__':
    unittest.main()

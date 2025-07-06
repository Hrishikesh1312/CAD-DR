import os
import numpy as np
import pandas as pd
from pathlib import Path
import open3d as o3d
from pyntcloud import PyntCloud

class ConversionUtils:

    @staticmethod
    def list_files_in_directory(directory_path):
        return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    @staticmethod
    def stl_to_ply(path, number_of_points, output_dir):
        imported_file_name = Path(path).stem
        mesh = o3d.io.read_triangle_mesh(path)
        pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
        output_path = os.path.join(output_dir, f"{imported_file_name}.ply")
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)

    @staticmethod
    def convert_to_binvox(path, dim):
        point_cloud = np.loadtxt(path, skiprows=12)[:, 0:3]
        df = pd.DataFrame(point_cloud, columns=['x','y','z'])
        cloud = PyntCloud(df)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=dim, n_y=dim, n_z=dim)
        voxelgrid = cloud.structures[voxelgrid_id]
        return voxelgrid.get_feature_vector(mode="binary")
    
    @staticmethod
    def convert_pointcloud_to_binvox(point_cloud, dim=64):
        import pandas as pd
        from pyntcloud import PyntCloud

        points = np.asarray(point_cloud.points)
        df = pd.DataFrame(data=points, columns=['x', 'y', 'z'])
        cloud = PyntCloud(df)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=dim, n_y=dim, n_z=dim)
        voxelgrid = cloud.structures[voxelgrid_id]
        return voxelgrid.get_feature_vector(mode="binary")


# Author: Hrishikesh Naramparambath
# Class for open3d and matplotlib visualizations of original and reconstructed models

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


class Visualization:

    def matplotlib_visualize_original(original_sample):
        voxel_data = original_sample
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = np.indices(voxel_data.shape)
        x1, y1, z1 = x[voxel_data == 1], y[voxel_data == 1], z[voxel_data == 1]
        ax.scatter(x1, y1, z1, c='b', marker='s')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def matplotlib_visualize_reconstructed(reconstructed_sample):
        voxel_data = reconstructed_sample
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = np.indices(voxel_data.shape)
        x1, y1, z1 = x[voxel_data == 1], y[voxel_data == 1], z[voxel_data == 1]
        ax.scatter(x1, y1, z1, c='b', marker='s')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def open3d_visualize_original(original_sample):
        voxel_data = original_sample
        x, y, z = np.where(voxel_data == 1)
        points = np.column_stack((x, y, z))
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(point_cloud)
        view_control = visualizer.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, 1, 0])
        visualizer.run()
        visualizer.destroy_window()

    def open3d_visualize_reconstructed(reconstructed_sample):
        voxel_data = reconstructed_sample
        x, y, z = np.where(voxel_data == 1)
        points = np.column_stack((x, y, z))
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(point_cloud)
        view_control = visualizer.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, 1, 0])
        visualizer.run()
        visualizer.destroy_window()

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import open3d as o3d
from skimage import measure
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

class Visualization:

    @staticmethod
    def matplotlib_visualize(voxel_data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = np.indices(voxel_data.shape)
        x1, y1, z1 = x[voxel_data == 1], y[voxel_data == 1], z[voxel_data == 1]
        ax.scatter(x1, y1, z1, c='b', marker='s')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    @staticmethod
    def open3d_visualize(voxel_data):
        x, y, z = np.where(voxel_data == 1)
        points = np.column_stack((x, y, z))
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)
        vis.run()
        vis.destroy_window()

    @staticmethod
    def plotly_visualize_stl(file_path):
        mesh = o3d.io.read_triangle_mesh(file_path)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        fig = go.Figure(data=[go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color='lightblue',
            opacity=0.50
        )])

        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig)

    @staticmethod
    def plotly_visualize_voxel(voxel_data):
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(voxel_data, level=0.5)
        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='cyan',
            opacity=0.7
        )])
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig)

    @staticmethod
    def plotly_visualize_mesh_from_voxel(voxel_data):
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(voxel_data, level=0.5)
        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=0.6
        )])
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig)

    @staticmethod
    def plotly_visualize_latent_space(latent, threshold=0.35):
        import matplotlib.pyplot as plt
        shape = latent.shape[1:5]
        colors = plt.cm.get_cmap('tab20', shape[0])
        fig = go.Figure()
        for i in range(shape[0]):
            binary = latent[0, i, :, :, :]
            x, y, z = np.where(binary >= threshold)
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=3, color=colors(i), opacity=0.8),
                name=f'Channel {i+1}'
            ))
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig)


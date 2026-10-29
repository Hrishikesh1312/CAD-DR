import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import open3d as o3d
import streamlit as st
from utils.theme import get_palette

class Visualization:

    @staticmethod
    def render_chart(fig, key=None):
        palette = get_palette(st.session_state.get("appearance", "Dark"))
        fig.update_layout(
            height=410,
            paper_bgcolor=palette["surface"],
            font=dict(family="Arial, sans-serif", color=palette["muted"], size=12),
            scene=dict(
                aspectmode="data",
                bgcolor=palette["surface"],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)),
            ),
            uirevision="cad-view",
            legend=dict(orientation="h", y=-0.05),
        )
        st.plotly_chart(fig, key=key, theme=None, use_container_width=True,
                        config={"displaylogo": False, "scrollZoom": False})

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
    def plotly_visualize_stl(file_path, key=None):
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
            color=get_palette(st.session_state.get("appearance", "Dark"))["mesh"],
            opacity=0.50
        )])

        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        Visualization.render_chart(fig, key=key)

    @staticmethod
    def plotly_visualize_voxel(voxel_data, key=None):
        from skimage import measure
        if not np.any(voxel_data):
            st.info("No occupied voxels at this threshold. Lower the threshold to see a surface.")
            return
        # An empty border also closes surfaces for fully occupied grids.
        padded = np.pad(voxel_data, 1, mode="constant")
        verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
        verts -= 1
        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=get_palette(st.session_state.get("appearance", "Dark"))["voxel"],
            opacity=0.7
        )])
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        Visualization.render_chart(fig, key=key)

    @staticmethod
    def plotly_visualize_mesh_from_voxel(voxel_data, key=None):
        from skimage import measure
        if not np.any(voxel_data):
            st.info("No occupied voxels at this threshold. Lower the threshold to see a surface.")
            return
        # An empty border also closes surfaces for fully occupied grids.
        padded = np.pad(voxel_data, 1, mode="constant")
        verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
        verts -= 1
        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=get_palette(st.session_state.get("appearance", "Dark"))["mesh"],
            opacity=0.6
        )])
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        Visualization.render_chart(fig, key=key)

    @staticmethod
    def plotly_visualize_latent_space(latent, threshold=0.35, key=None):
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
                marker=dict(size=3, color="rgb({},{},{})".format(*(int(c * 255) for c in colors(i)[:3])), opacity=0.8),
                name=f'Channel {i+1}'
            ))
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                        margin=dict(l=0, r=0, b=0, t=0))
        Visualization.render_chart(fig, key=key)

    @staticmethod
    def plotly_compare_voxels(original, reconstructed):
        masks = [
            (original.astype(bool) & reconstructed.astype(bool), "Shared", "#66B99A"),
            (original.astype(bool) & ~reconstructed.astype(bool), "Missing", "#F2AD64"),
            (~original.astype(bool) & reconstructed.astype(bool), "Extra", "#86ACFF"),
        ]
        fig = go.Figure()
        for mask, name, color in masks:
            points = np.argwhere(mask)
            # Keep the browser responsive; metrics still use every voxel.
            if len(points) > 15000:
                points = points[np.linspace(0, len(points) - 1, 15000, dtype=int)]
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers', name=name,
                marker=dict(size=2, color=color, opacity=0.65),
            ))
        fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                     zaxis=dict(visible=False)), margin=dict(l=0,r=0,b=0,t=0))
        Visualization.render_chart(fig, key="voxel_overlay")

    @staticmethod
    def plotly_cross_section(original, reconstructed, axis, index):
        truth = np.take(original, index, axis=axis).astype(bool)
        prediction = np.take(reconstructed, index, axis=axis).astype(bool)
        classes = truth.astype(int) + prediction.astype(int) * 2
        palette = get_palette(st.session_state.get("appearance", "Dark"))
        fig = go.Figure(go.Heatmap(
            z=classes.T, zmin=0, zmax=3,
            colorscale=[[0, palette['surface']], [.1666, palette['surface']],
                        [.1667, '#F2AD64'], [.4999, '#F2AD64'],
                        [.5, '#86ACFF'], [.8332, '#86ACFF'],
                        [.8333, '#66B99A'], [1, '#66B99A']],
            colorbar=dict(tickvals=[0,1,2,3], ticktext=['Empty','Missing','Extra','Shared']),
            hovertemplate='Cell (%{x}, %{y})<extra></extra>',
        ))
        fig.update_layout(yaxis=dict(scaleanchor='x'), margin=dict(l=20,r=20,b=20,t=20))
        Visualization.render_chart(fig, key="cross_section")

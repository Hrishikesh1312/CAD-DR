import streamlit as st
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go
from keras.models import load_model
from pyntcloud import PyntCloud
from skimage import measure
import matplotlib.pyplot as plt

autoencoder = load_model("saved-models/autoencoder.keras")
encoder = load_model("saved-models/encoder.keras")

def visualize_stl_plotly(file_path):
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

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, b=0, t=0))

    st.plotly_chart(fig)

def convert_to_binvox(point_cloud, dim=64):
    points = np.asarray(point_cloud.points)
    df = pd.DataFrame(data=points, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=dim, n_y=dim, n_z=dim)
    voxelgrid = cloud.structures[voxelgrid_id]
    binvox_array = voxelgrid.get_feature_vector(mode="binary")
    return binvox_array

def visualize_voxel_plotly(voxel_data):
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

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, b=0, t=0))

    st.plotly_chart(fig)

def visualize_latent_space_plotly(latent_representation, threshold=0.35):
    latent_shape = latent_representation.shape[1:5]
    colors = plt.cm.get_cmap('tab20', latent_shape[0])

    fig = go.Figure()

    for i in range(latent_shape[0]):
        binary_array = latent_representation[0, i, :, :, :]
        x, y, z = np.where(binary_array >= threshold)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=colors(i), opacity=0.8),
            name=f'Channel {i + 1}'
        ))

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, b=0, t=0))

    st.plotly_chart(fig)

def visualize_triangulated_reconstruction(reconstructed_sample):
    verts, faces, _, _ = measure.marching_cubes(reconstructed_sample, level=0.5)

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

    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)),
                      margin=dict(l=0, r=0, b=0, t=0))

    st.plotly_chart(fig)

st.title("CAD-DR")
st.subheader(
    "Deep convolutional autoencoder for dimensionality reduction of 3D CAD models.")
stl_file = st.file_uploader("Upload an STL file", type=['stl'])

autoencoder_done = False
encoder_done = False
latent_representation = None

if stl_file:
    st.subheader("Original 3D CAD Model")
    st.write("Triangulated STL Model visualization")

    temp_file_path = "uploaded_model.stl"
    with open(temp_file_path, "wb") as f:
        f.write(stl_file.getbuffer())

    visualize_stl_plotly(temp_file_path)

    point_cloud = o3d.io.read_triangle_mesh(
        temp_file_path).sample_points_uniformly(number_of_points=20000)
    binvox_array = convert_to_binvox(point_cloud)

    st.write("Voxelized STL Model Visualization")
    visualize_voxel_plotly(binvox_array)

    if st.button("Run Autoencoder and Encoder"):
        if not autoencoder_done:
            with st.spinner('Running autoencoder...'):
                progress_bar = st.progress(0)
                binvox_array_reshaped = binvox_array.reshape(1, 64, 64, 64, 1)

                for i in range(1, 101):
                    if i % 20 == 0:
                        progress_bar.progress(i / 100)
                    reconstructed_data = autoencoder.predict(binvox_array_reshaped)
                progress_bar.empty()
                autoencoder_done = True
            st.subheader("Reconstructed 3D CAD Model")
            st.write("Voxelized Reconstructed Model Visualization")
            reconstructed_sample = reconstructed_data[0].reshape(64, 64, 64)
            threshold = 0.35
            reconstructed_sample = (reconstructed_sample > threshold).astype(int)
            visualize_voxel_plotly(reconstructed_sample)

            st.write("Triangulated Reconstructed Model Visualization")
            visualize_triangulated_reconstruction(reconstructed_sample)

        if not encoder_done:
            with st.spinner('Running encoder...'):
                progress_bar = st.progress(0)

                for i in range(1, 101):
                    if i % 20 == 0:
                        progress_bar.progress(i / 100)
                    latent_representation = encoder.predict(
                        binvox_array_reshaped)
                progress_bar.empty()
                encoder_done = True
        st.subheader("Latent Space")
        st.write("Visualization - 16 channels of 16x16x16 data")
        visualize_latent_space_plotly(latent_representation)

        latent_space_file_path = 'latent_space.npy'
        np.save(latent_space_file_path, latent_representation)

        st.success(
            f"Latent space saved as a NumPy object: {latent_space_file_path}")
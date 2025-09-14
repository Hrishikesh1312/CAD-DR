import streamlit as st
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from keras.models import load_model
from config import POINT_CLOUD_DENSITY, VOXEL_DIM
from utils.conversion_utils import ConversionUtils
from utils.visualization import Visualization
import os

autoencoder = load_model("data/saved-models/autoencoder.keras")
encoder = load_model("data/saved-models/encoder.keras")

st.title("CAD-DR")
st.subheader(
    "Deep Convolutional Autoencoder for Dimensionality Reduction of 3D CAD Models")

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

    Visualization.plotly_visualize_stl(temp_file_path)

    mesh = o3d.io.read_triangle_mesh(temp_file_path)
    point_cloud = mesh.sample_points_uniformly(
        number_of_points=POINT_CLOUD_DENSITY)
    binvox_array = ConversionUtils.convert_pointcloud_to_binvox(
        point_cloud, VOXEL_DIM)

    st.write("Voxelized STL Model Visualization")
    Visualization.plotly_visualize_voxel(binvox_array)

    if st.button("Run Autoencoder and Encoder"):
        with st.spinner('Running Autoencoder...'):
            binvox_array_reshaped = binvox_array.reshape(1, 64, 64, 64, 1)
            progress_bar = st.progress(0)

            for i in range(1, 101):
                if i % 20 == 0:
                    progress_bar.progress(i / 100)
                reconstructed_data = autoencoder.predict(binvox_array_reshaped)

            progress_bar.empty()
            reconstructed_sample = reconstructed_data[0].reshape(64, 64, 64)
            threshold = 0.35
            reconstructed_sample = (
                reconstructed_sample > threshold).astype(int)

        st.subheader("Reconstructed 3D CAD Model")
        st.write("Voxelized Reconstructed Model")
        Visualization.plotly_visualize_voxel(reconstructed_sample)

        st.write("Triangulated Reconstructed Model")
        Visualization.plotly_visualize_mesh_from_voxel(reconstructed_sample)

        with st.spinner("Running Encoder..."):
            progress_bar = st.progress(0)
            for i in range(1, 101):
                if i % 20 == 0:
                    progress_bar.progress(i / 100)
                latent_representation = encoder.predict(binvox_array_reshaped)
            progress_bar.empty()

        st.subheader("Latent Space")
        st.write("Visualization - 16 channels of 16x16x16 data")
        Visualization.plotly_visualize_latent_space(latent_representation)

        latent_space_path = "latent_space.npy"
        np.save(latent_space_path, latent_representation)
        st.success(f"Latent space saved as NumPy array: {latent_space_path}")

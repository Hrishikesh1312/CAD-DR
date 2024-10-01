import streamlit as st
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import pandas as pd
from keras.models import load_model
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage, stats, ndimage
from skimage import measure

plt.style.use('dark_background')

autoencoder = load_model("saved-models/autoencoder.keras")
encoder = load_model("saved-models/encoder.keras")

def visualize_latent_space(latent_representation, threshold=0.35):
    latent_shape = latent_representation.shape[1:5]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap('tab20', latent_shape[0])

    for i in range(latent_shape[0]):
        binary_array = latent_representation[0, i, :, :, :]
        x, y, z = np.where(binary_array >= threshold)
        ax.scatter(x, y, z, c=[colors(i)], marker='o',
                   s=20, label=f'Channel {i + 1}')

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    st.pyplot(fig)

def stl_to_point_cloud(file_path, num_points=20000):
    mesh = o3d.io.read_triangle_mesh(file_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    return point_cloud

def convert_to_binvox(point_cloud, dim=64):
    points = np.asarray(point_cloud.points)
    df = pd.DataFrame(data=points, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=dim, n_y=dim, n_z=dim)
    voxelgrid = cloud.structures[voxelgrid_id]
    binvox_array = voxelgrid.get_feature_vector(mode="binary")
    return binvox_array

def visualize_point_cloud(point_cloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    vis.run()
    vis.destroy_window()

def visualize_voxel(voxel_data):
    verts, faces, _, _ = measure.marching_cubes(voxel_data, level=0.5)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces, color='cyan', alpha=0.7, edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    st.pyplot(fig)

st.title("3D Model Autoencoder Visualization")

stl_file = st.file_uploader("Upload an STL file", type=['stl'])

if stl_file:
    st.write("Visualizing the STL model...")
    temp_file_path = "uploaded_model.stl"
    with open(temp_file_path, "wb") as f:
        f.write(stl_file.getbuffer())
    mesh = o3d.io.read_triangle_mesh(temp_file_path)
    point_cloud = stl_to_point_cloud(temp_file_path)

    if st.button("View Point Cloud"):
        visualize_point_cloud(point_cloud)

    st.write("Converting to binary voxel grid...")
    binvox_array = convert_to_binvox(point_cloud)

    st.write("Visualizing the voxelized data...")
    visualize_voxel(binvox_array)

    st.write("Running the autoencoder for reconstruction...")
    binvox_array_reshaped = binvox_array.reshape(1, 64, 64, 64, 1)
    reconstructed_data = autoencoder.predict(binvox_array_reshaped)

    st.write("Visualizing reconstructed model...")
    reconstructed_sample = reconstructed_data[0].reshape(64, 64, 64)
    threshold = 0.35
    reconstructed_sample = (reconstructed_sample > threshold).astype(int)
    visualize_voxel(reconstructed_sample)

    st.write("Extracting latent space representation...")
    latent_representation = encoder.predict(binvox_array_reshaped)

    st.write("Visualizing latent space with 16 channels...")
    visualize_latent_space(latent_representation)
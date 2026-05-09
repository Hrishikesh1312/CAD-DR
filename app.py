import os
import time
import tempfile
import streamlit as st
import numpy as np
import torch
import open3d as o3d
import plotly.graph_objects as go
from config import POINT_CLOUD_DENSITY, VOXEL_DIM
from model.autoencoder import Autoencoder
from utils.conversion_utils import ConversionUtils
from utils.visualization import Visualization


# ---------------------------------------------------------
# Reconstruction Metrics
# ---------------------------------------------------------

def compute_reconstruction_metrics(original, reconstructed):

    original = original.astype(np.uint8).flatten()
    reconstructed = reconstructed.astype(np.uint8).flatten()

    intersection = np.sum(
        (original == 1) & (reconstructed == 1)
    )

    union = np.sum(
        (original == 1) | (reconstructed == 1)
    )

    true_positive = intersection

    true_negative = np.sum(
        (original == 0) & (reconstructed == 0)
    )

    false_positive = np.sum(
        (original == 0) & (reconstructed == 1)
    )

    false_negative = np.sum(
        (original == 1) & (reconstructed == 0)
    )

    voxel_accuracy = (
        (true_positive + true_negative)
        / len(original)
    )

    iou = (
        intersection / union
        if union > 0 else 0
    )

    precision = (
        true_positive /
        (true_positive + false_positive)
        if (true_positive + false_positive) > 0 else 0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if (true_positive + false_negative) > 0 else 0
    )

    f1_score = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "Voxel Accuracy": voxel_accuracy,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

# ---------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CAD-DR",
    page_icon="data:image/png;base64,iVBORw0KGgo=",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------

if "latent_representation" not in st.session_state:
    st.session_state.latent_representation = None

if "reconstructed_sample" not in st.session_state:
    st.session_state.reconstructed_sample = None

if "inference_complete" not in st.session_state:
    st.session_state.inference_complete = False

# ---------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model = Autoencoder().to(device)
    model.load_state_dict(
        torch.load(
            "data/saved-models/autoencoder.pt",
            map_location=device
        )
    )
    model.eval()
    return model


model = load_model()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Configuration")

threshold = st.sidebar.slider(
    "Reconstruction Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.01,
    help=(
        "Threshold used to binarize the reconstructed voxel grid. "
        "Values above the threshold are converted to occupied voxels (1), "
        "while values below are treated as empty space (0). "
        "Lower thresholds produce denser reconstructions."
    )
)

point_density = st.sidebar.number_input(
    "Point Cloud Density",
    min_value=1000,
    max_value=100000,
    value=POINT_CLOUD_DENSITY,
    step=1000
)

show_statistics = st.sidebar.checkbox(
    "Show Model Statistics",
    value=True
)

save_outputs = st.sidebar.checkbox(
    "Save Latent Representation",
    value=True
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("CAD-DR")

st.subheader(
    "Deep Convolutional Autoencoder for Dimensionality Reduction of 3D CAD Models"
)

st.markdown("---")

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an STL file",
    type=["stl"]
)

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

if uploaded_file:

    # -----------------------------------------------------
    # Save Uploaded File
    # -----------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".stl"
    ) as tmp_file:

        tmp_file.write(uploaded_file.getbuffer())
        temp_file_path = tmp_file.name

    # -----------------------------------------------------
    # File Information
    # -----------------------------------------------------

    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.subheader("Uploaded File Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Filename", uploaded_file.name)

    with col2:
        st.metric("File Size (MB)", f"{file_size_mb:.2f}")

    with col3:
        st.metric("Execution Device", str(device).upper())

    st.markdown("---")

    # -----------------------------------------------------
    # Original STL Visualization
    # -----------------------------------------------------

    st.subheader("Original 3D CAD Model")
    st.write("Triangulated STL model visualization")

    Visualization.plotly_visualize_stl(temp_file_path)

    # -----------------------------------------------------
    # Mesh Loading
    # -----------------------------------------------------

    with st.spinner("Loading STL mesh..."):

        mesh = o3d.io.read_triangle_mesh(temp_file_path)

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        point_cloud = mesh.sample_points_uniformly(
            number_of_points=point_density
        )

        binvox_array = ConversionUtils.convert_pointcloud_to_binvox(
            point_cloud,
            VOXEL_DIM
        )

    # -----------------------------------------------------
    # Statistics Section
    # -----------------------------------------------------

    if show_statistics:

        st.subheader("Model Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Vertices", len(vertices))

        with col2:
            st.metric("Triangles", len(triangles))

        with col3:
            st.metric("Voxel Dimension", VOXEL_DIM)

        with col4:
            st.metric("Point Cloud Size", point_density)

    # -----------------------------------------------------
    # Voxel Visualization
    # -----------------------------------------------------

    st.subheader("Voxelized STL Model")
    st.write("Voxelized representation of the input CAD model")

    Visualization.plotly_visualize_voxel(binvox_array)

    st.markdown("---")

    # -----------------------------------------------------
    # Inference Button
    # -----------------------------------------------------

    if st.button("Run Autoencoder and Encoder", use_container_width=True):

        total_start_time = time.time()

        # -------------------------------------------------
        # Autoencoder Inference
        # -------------------------------------------------

        st.subheader("Autoencoder Inference")

        with st.spinner("Running autoencoder inference..."):

            inference_start = time.time()

            binvox_tensor = (
                torch.FloatTensor(binvox_array)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            progress_bar = st.progress(0)

            with torch.no_grad():

                for i in range(1, 101):

                    if i % 10 == 0:
                        progress_bar.progress(i)

                    reconstructed_data = model(binvox_tensor)

            progress_bar.empty()

            inference_end = time.time()

            reconstructed_sample = (
                reconstructed_data
                .squeeze()
                .cpu()
                .numpy()
            )

            reconstructed_sample = (
                reconstructed_sample > threshold
            ).astype(int)

            st.session_state.reconstructed_sample = reconstructed_sample

        # -------------------------------------------------
        # Reconstruction Results
        # -------------------------------------------------

        st.subheader("Reconstructed 3D CAD Model")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Voxelized reconstructed model")
            Visualization.plotly_visualize_voxel(
                reconstructed_sample
            )

        with col2:
            st.write("Triangulated reconstructed model")
            Visualization.plotly_visualize_mesh_from_voxel(
                reconstructed_sample
            )

        # -------------------------------------------------
        # Reconstruction Metrics
        # -------------------------------------------------

        metrics = compute_reconstruction_metrics(
            binvox_array,
            reconstructed_sample
        )

        st.subheader("Reconstruction Accuracy Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Voxel Accuracy",
                f"{metrics['Voxel Accuracy']:.4f}"
            )

        with col2:
            st.metric(
                "IoU",
                f"{metrics['IoU']:.4f}"
            )

        with col3:
            st.metric(
                "Precision",
                f"{metrics['Precision']:.4f}"
            )

        with col4:
            st.metric(
                "Recall",
                f"{metrics['Recall']:.4f}"
            )

        with col5:
            st.metric(
                "F1 Score",
                f"{metrics['F1 Score']:.4f}"
            )

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        st.subheader("Latent Space Encoding")

        with st.spinner("Generating latent representation..."):

            encoder_start = time.time()

            progress_bar = st.progress(0)

            with torch.no_grad():

                for i in range(1, 101):

                    if i % 10 == 0:
                        progress_bar.progress(i)

                    latent_representation = model.encode(
                        binvox_tensor
                    )

            progress_bar.empty()

            encoder_end = time.time()

            latent_representation = (
                latent_representation
                .cpu()
                .numpy()
            )

            st.session_state.latent_representation = (
                latent_representation
            )

        # -------------------------------------------------
        # Latent Space Visualization
        # -------------------------------------------------

        st.subheader("Latent Space")

        st.write(
            "Visualization of latent tensor channels "
            "(16 channels of 16x16x16 feature maps)"
        )

        Visualization.plotly_visualize_latent_space(
            latent_representation
        )

        # -------------------------------------------------
        # Performance Metrics
        # -------------------------------------------------

        st.subheader("Execution Metrics")

        total_end_time = time.time()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Autoencoder Time",
                f"{inference_end - inference_start:.2f} sec"
            )

        with col2:
            st.metric(
                "Encoder Time",
                f"{encoder_end - encoder_start:.2f} sec"
            )

        with col3:
            st.metric(
                "Total Runtime",
                f"{total_end_time - total_start_time:.2f} sec"
            )

        # -------------------------------------------------
        # Save Outputs
        # -------------------------------------------------

        if save_outputs:

            os.makedirs("outputs", exist_ok=True)

            latent_space_path = os.path.join(
                "outputs",
                "latent_space.npy"
            )

            reconstruction_path = os.path.join(
                "outputs",
                "reconstructed_voxel.npy"
            )

            np.save(
                latent_space_path,
                latent_representation
            )

            np.save(
                reconstruction_path,
                reconstructed_sample
            )

            st.success(
                "Latent representation and reconstructed "
                "voxel grid saved successfully."
            )

            with open(latent_space_path, "rb") as f:
                st.download_button(
                    label="Download Latent Representation",
                    data=f,
                    file_name="latent_space.npy",
                    mime="application/octet-stream"
                )

            with open(reconstruction_path, "rb") as f:
                st.download_button(
                    label="Download Reconstructed Voxel Grid",
                    data=f,
                    file_name="reconstructed_voxel.npy",
                    mime="application/octet-stream"
                )

        st.session_state.inference_complete = True

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "CAD-DR | Deep Learning Based 3D CAD Model "
    "Dimensionality Reduction System"
)

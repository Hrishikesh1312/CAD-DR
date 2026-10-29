import os

# Enable CPU fallback for unsupported MPS operations before importing PyTorch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hashlib
import pandas as pd
import plotly.graph_objects as go
import time
import tempfile
import logging
from pathlib import Path
import streamlit as st
import numpy as np
import torch
import open3d as o3d
from config import POINT_CLOUD_DENSITY, VOXEL_DIM
from model.convolutional_autoencoder import ConvolutionalAutoencoder
from utils.conversion_utils import ConversionUtils
from utils.visualization import Visualization
from utils.validation import validate_mesh, validate_voxels
from utils.theme import theme_variables
from utils.exports import numpy_bytes, binary_stl, latent_archive
from utils.experiments import threshold_sweep

logger = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "data/saved-models/autoencoder.pt"


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
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------

if "inference_result" not in st.session_state:
    st.session_state.inference_result = None

# ---------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------

compute_device = ""

if torch.cuda.is_available():
    compute_device = "cuda"
elif torch.backends.mps.is_available():
    compute_device = "mps"
else:
    compute_device = "cpu"

device = torch.device(compute_device)

# ---------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model = ConvolutionalAutoencoder().to(device)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()
    return model


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model weights are missing. Restore data/saved-models/autoencoder.pt and restart the app.")
    st.stop()
except Exception:
    logger.exception("Unable to load autoencoder weights")
    st.error("The model could not be loaded. Check that its weights match the autoencoder and the selected device is available.")
    st.stop()

@st.cache_data(show_spinner=False)
def voxelize_upload(file_bytes, point_density, voxel_dim):
    with tempfile.NamedTemporaryFile(suffix=".stl") as source:
        source.write(file_bytes)
        source.flush()
        mesh = o3d.io.read_triangle_mesh(source.name)
        validate_mesh(mesh)
        point_cloud = mesh.sample_points_uniformly(number_of_points=point_density)
        voxels, transform = ConversionUtils.convert_pointcloud_to_binvox(
            point_cloud, voxel_dim, return_transform=True
        )
        validate_voxels(voxels, voxel_dim)
        return voxels, transform


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("CAD-DR")
st.sidebar.caption("Reconstruction workspace")
theme = st.sidebar.radio("Appearance", ["Light", "Dark"], index=1, horizontal=True, key="appearance")
st.markdown(theme_variables(theme), unsafe_allow_html=True)
st.sidebar.markdown("### Reconstruction")

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

st.sidebar.markdown("### Input sampling")
point_density = st.sidebar.number_input(
    "Point Cloud Density",
    min_value=1000,
    max_value=100000,
    value=POINT_CLOUD_DENSITY,
    step=1000
)

st.sidebar.markdown("### Display & export")
show_statistics = st.sidebar.checkbox(
    "Show Model Statistics",
    value=True
)

save_outputs = st.sidebar.checkbox(
    "Show download options",
    value=True
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown("""
<style>
.stApp {
    --background-color: var(--cad-background);
    --secondary-background-color: var(--cad-sidebar);
    --text-color: var(--cad-text);
    --primary-color: var(--cad-accent);
    background: var(--cad-background); color: var(--cad-text);
}
[data-testid="stHeader"] { background: var(--cad-background); }
[data-testid="stSidebar"] { background: var(--cad-sidebar); }
/* Both the in-sidebar collapse button and the floating reopen control. */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    color: var(--cad-text) !important;
    background: var(--cad-surface) !important;
    border: 1px solid var(--cad-border) !important;
    border-radius: 7px;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover {
    color: var(--cad-accent) !important;
    border-color: var(--cad-accent) !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg { color: inherit; fill: currentColor; }
.stApp a { color: var(--cad-accent); }
[data-baseweb="radio"]:has(input:checked) > div:first-child,
[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
    background-color: var(--cad-accent);
    border-color: var(--cad-accent);
}
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--cad-accent);
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--cad-accent);
}

.stApp h1, .stApp h2, .stApp h3, .stApp label,
[data-testid="stMetricValue"], [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"], [data-testid="stRadio"] {
    color: var(--cad-text);
}
[data-testid="stCaptionContainer"] { color: var(--cad-muted); }
[data-baseweb="input"], [data-baseweb="input"] input {
    background: var(--cad-surface); color: var(--cad-text);
    border-color: var(--cad-border);
}
[data-testid="stNumberInput"] button,
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
    background: var(--cad-surface); color: var(--cad-text); border-color: var(--cad-border);
}
[data-testid="stBaseButton-primary"] {
    background: var(--cad-accent); border-color: var(--cad-accent); color: var(--cad-button-text);
}
[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] { color: var(--cad-button-text); }
[data-testid="stBaseButton-primary"]:hover { filter: brightness(.92); }
[data-baseweb="tab-list"] { background: transparent; }
button[data-baseweb="tab"] { color: var(--cad-muted); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--cad-accent); }
[data-baseweb="tab-highlight"] { background: var(--cad-accent); }
[data-testid="stFileUploaderDropzone"] * { color: var(--cad-text); }
[data-testid="stAlert"] { background: var(--cad-sidebar); color: var(--cad-text); }
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color: var(--cad-text); }

.block-container { max-width: 1380px; padding-top: 2.75rem; padding-bottom: 3rem; }
h1 { letter-spacing: -.045em; font-weight: 750; padding-bottom: .35rem; }
h2, h3 { letter-spacing: -.025em; }
h3 { font-size: 1.2rem; font-weight: 650; }
.cad-eyebrow { color: var(--cad-accent); font-size: .72rem; font-weight: 700;
    letter-spacing: .18em; text-transform: uppercase; margin-bottom: .5rem; }
.cad-intro { color: var(--cad-muted); max-width: 680px; font-size: 1.03rem;
    line-height: 1.7; margin-bottom: 1.8rem; }
[data-testid="stSidebar"] { border-right: 1px solid var(--cad-border); }
[data-testid="stSidebar"] h1 { font-size: 1.4rem; letter-spacing: -.03em; }
[data-testid="stSidebar"] h3 { font-size: .75rem; letter-spacing: .09em;
    text-transform: uppercase; color: var(--cad-muted); margin-top: 1.2rem; }
[data-testid="stMetric"] { background: var(--cad-surface); border: 1px solid var(--cad-border);
    border-radius: 10px; padding: 1.05rem 1.2rem; min-height: 108px;
    box-shadow: 0 2px 5px rgba(23,43,77,.025); }
[data-testid="stMetricLabel"] { color: var(--cad-muted); font-size: .8rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 650;
    font-variant-numeric: tabular-nums; letter-spacing: -.025em; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: var(--cad-border) !important;
    border-radius: 10px !important; }
[data-testid="stFileUploader"] { background: var(--cad-surface); border: 1px solid var(--cad-border);
    border-radius: 10px; padding: 1rem; }
[data-testid="stFileUploaderDropzone"] { background: var(--cad-background);
    border: 1px dashed var(--cad-border); border-radius: 7px; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--cad-border);
    border-radius: 10px; overflow: hidden; background: var(--cad-surface); }
button[data-baseweb="tab"] { font-size: .9rem; font-weight: 600;
    padding: .7rem 1rem; }
[data-testid="stBaseButton-primary"] { min-height: 3rem; border-radius: 8px;
    font-weight: 600; box-shadow: 0 3px 8px rgba(37,99,235,.15); }
[data-testid="stBaseButton-primary"]:hover { box-shadow: 0 4px 12px rgba(37,99,235,.24); }
button:focus-visible { outline: 3px solid var(--cad-focus) !important; outline-offset: 3px; }
hr { border-color: var(--cad-border); margin: 1.75rem 0; }
@media (max-width: 640px) {
    .block-container { padding: 1.25rem 1rem 2rem; }
    [data-testid="stMetric"] { padding: .85rem; min-height: auto; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    button[data-baseweb="tab"] { padding: .6rem .5rem; font-size: .8rem; }
}
</style>
<div class="cad-eyebrow">CAD geometry · Reconstruction studio</div>
""", unsafe_allow_html=True)
st.title("CAD-DR")
st.markdown(
    '<div class="cad-intro">Explore your CAD geometry, reconstruct its shape, '
    'and inspect its compressed representation in one workspace.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Start with an STL model",
    type=["stl"],
    help="Upload a triangulated CAD mesh. Your original file is kept unchanged."
)

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------

input_key = (
    hashlib.sha256(uploaded_file.getvalue()).hexdigest(), point_density, VOXEL_DIM
) if uploaded_file is not None else None
if st.session_state.get("input_key") != input_key:
    st.session_state.input_key = input_key
    st.session_state.inference_result = None

if uploaded_file is None:
    st.markdown("### From CAD model to reconstruction")
    for column, title, description in zip(
        st.columns(3),
        ["01 · Upload", "02 · Reconstruct", "03 · Explore"],
        ["Choose an STL and inspect its geometry.",
         "Encode your shape and reconstruct its voxel surface.",
         "Adjust the threshold, compare results, and download your data."],
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(description)
    st.info("Upload an STL above to begin. Reconstruction controls are in the sidebar.")

if uploaded_file is not None:
    if uploaded_file.size == 0:
        st.error("This STL file is empty. Upload a file containing a triangle mesh.")
        st.stop()

    # -----------------------------------------------------
    # Save Uploaded File
    # -----------------------------------------------------

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".stl"
        ) as tmp_file:

            temp_file_path = tmp_file.name
            tmp_file.write(uploaded_file.getbuffer())

        # -----------------------------------------------------
        # File Information
        # -----------------------------------------------------

        file_size_mb = uploaded_file.size / (1024 * 1024)

        st.subheader("Model overview")

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

        mesh = o3d.io.read_triangle_mesh(temp_file_path)
        validate_mesh(mesh)

        # -----------------------------------------------------
        # Mesh Loading
        # -----------------------------------------------------

        with st.spinner("Loading STL mesh..."):

            # Decimate mesh for visualization only — does not affect voxelization
            MAX_TRIANGLES = 50000
            if len(mesh.triangles) > MAX_TRIANGLES:
                mesh = mesh.simplify_quadric_decimation(
                    target_number_of_triangles=MAX_TRIANGLES
                )
                st.info(f"Mesh decimated to {MAX_TRIANGLES} triangles for visualization.")

            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

            binvox_array, voxel_transform = voxelize_upload(uploaded_file.getvalue(), point_density, VOXEL_DIM)

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

        st.subheader("Input preview")
        st.caption("Drag to rotate · scroll to zoom · double-click to reset the view")
        original_column, voxel_column = st.columns(2)
        with original_column:
            with st.container(border=True):
                st.markdown("**Original surface**")
                Visualization.plotly_visualize_stl(temp_file_path, key="original_surface")
        with voxel_column:
            with st.container(border=True):
                st.markdown(f"**Voxelized input · {VOXEL_DIM}³**")
                Visualization.plotly_visualize_voxel(binvox_array, key="input_voxels")

        st.markdown("---")

        # -----------------------------------------------------
        # Inference Button
        # -----------------------------------------------------

        st.caption("The encoder compresses the voxel grid into latent features; the decoder reconstructs the shape.")
        if st.button("Encode & reconstruct", type="primary", use_container_width=True):
            st.session_state.inference_result = None
            try:
                with st.status("Encoding shape…", expanded=False) as inference_status:
                    total_start = time.perf_counter()
                    binvox_tensor = torch.as_tensor(
                        binvox_array, dtype=torch.float32, device=device
                    ).unsqueeze(0).unsqueeze(0)

                    with torch.inference_mode():
                        encoder_start = time.perf_counter()
                        latent_tensor = model.encode(binvox_tensor)
                        # Copy to CPU before stopping timers to include device execution.
                        latent_representation = latent_tensor.cpu().numpy()
                        encoder_time = time.perf_counter() - encoder_start
                        inference_status.update(label="Decoding reconstruction…")
                        decoder_start = time.perf_counter()
                        probabilities = model.decode(latent_tensor).squeeze().cpu().numpy()
                        decoder_time = time.perf_counter() - decoder_start

                    if not np.isfinite(probabilities).all() or not np.isfinite(latent_representation).all():
                        raise ValueError("The model produced non-finite output.")
                    if probabilities.shape != binvox_array.shape:
                        raise ValueError("The reconstruction shape does not match the input.")
                    st.session_state.inference_result = {
                        "probabilities": probabilities,
                        "latent": latent_representation,
                        "input_voxels": binvox_array.copy(),
                        "transform": voxel_transform,
                        "encoder_time": encoder_time,
                        "decoder_time": decoder_time,
                        "total_time": time.perf_counter() - total_start,
                    }
                    inference_status.update(label="Encoding and reconstruction complete.", state="complete")

            except Exception:
                logger.exception("Autoencoder inference failed")
                st.error("Reconstruction failed. Try again with another STL, or restart the app if the device is out of memory.")

        result = st.session_state.inference_result
        if result is None:
            st.caption("Ready to encode and reconstruct. Changing point density resets existing results.")
        if result is not None:
            reconstructed_sample = (result["probabilities"] > threshold).astype(np.uint8)
            latent_representation = result["latent"]

            # -------------------------------------------------
            # Reconstruction Results
            # -------------------------------------------------

            st.success("Encoding and reconstruction complete. Adjust the threshold in the sidebar to explore the result.")
            comparison_tab, quality_tab, experiments_tab, latent_tab, export_tab = st.tabs(
                ["Shape comparison", "Quality & timing", "Experiments", "Latent space", "Downloads"]
            )
            with comparison_tab:
                st.subheader("Reconstructed 3D CAD Model")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Voxelized input**")
                    Visualization.plotly_visualize_voxel(
                        result["input_voxels"], key="comparison_input"
                    )

                with col2:
                    st.markdown(f"**Reconstruction · threshold {threshold:.2f}**")
                    Visualization.plotly_visualize_mesh_from_voxel(
                        reconstructed_sample
                    )

                with st.expander("Inspect missing and extra geometry"):
                    st.caption("Green: shared · orange: missing · blue: extra. Overlay previews show up to 15,000 cells per group.")
                    view = st.radio("Inspection view", ["3D overlay", "Cross-section"], horizontal=True)
                    if view == "3D overlay":
                        Visualization.plotly_compare_voxels(result["input_voxels"], reconstructed_sample)
                    else:
                        axis_name = st.selectbox("Slice axis", ["X", "Y", "Z"])
                        axis = ["X", "Y", "Z"].index(axis_name)
                        index = st.slider("Slice index", 0, VOXEL_DIM - 1, VOXEL_DIM // 2)
                        Visualization.plotly_cross_section(result["input_voxels"], reconstructed_sample, axis, index)

            # -------------------------------------------------
            # Reconstruction Metrics
            # -------------------------------------------------

            metrics = compute_reconstruction_metrics(
                result["input_voxels"],
                reconstructed_sample
            )

            with quality_tab:
                st.subheader("Reconstruction quality")
                st.caption("IoU and F1 measure occupied-cell agreement. Accuracy also counts empty space.")

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

            with experiments_tab:
                st.subheader("Threshold experiment")
                st.caption("Compare thresholds against this input voxel grid. This measures reconstruction fidelity for one model, not generalization to unseen shapes.")
                if "threshold_sweep" not in result:
                    result["threshold_sweep"] = threshold_sweep(result["input_voxels"], result["probabilities"])
                sweep = pd.DataFrame(result["threshold_sweep"])
                best = sweep.loc[sweep["IoU"].idxmax()]
                c1, c2, c3 = st.columns(3)
                c1.metric("Best tested threshold (IoU)", f"{best['Threshold']:.2f}")
                c2.metric("Best tested IoU", f"{best['IoU']:.4f}")
                c3.metric("F1 at that threshold", f"{best['F1']:.4f}")
                chart = go.Figure()
                for metric, color in [("IoU", "#79B8FF"), ("F1", "#66B99A"),
                                      ("Precision", "#D19BFF"), ("Recall", "#F2AD64")]:
                    chart.add_trace(go.Scatter(x=sweep["Threshold"], y=sweep[metric],
                                              name=metric, line=dict(color=color)))
                chart.add_vline(x=threshold, line_dash="dot", annotation_text="Current threshold")
                chart.update_layout(xaxis_title="Reconstruction threshold", yaxis_title="Score",
                                    yaxis_range=[0, 1], margin=dict(l=20,r=20,t=30,b=20))
                Visualization.render_chart(chart, key="threshold_experiment")
                st.caption("101 thresholds tested from 0.00 to 1.00. Ties select the lowest threshold. Use the sidebar slider to try a value.")
                st.download_button("Download threshold experiment CSV", sweep.to_csv(index=False).encode(),
                                   file_name=f"{Path(uploaded_file.name).stem}_thresholds.csv", mime="text/csv")

            # -------------------------------------------------
            # Latent Space Visualization
            # -------------------------------------------------

            with latent_tab:
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

            with quality_tab:
                st.subheader("Execution Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Decoder Time",
                        f"{result['decoder_time']:.2f} sec"
                    )

                with col2:
                    st.metric(
                        "Encoder Time",
                        f"{result['encoder_time']:.2f} sec"
                    )

                with col3:
                    st.metric(
                        "Total Inference Time",
                        f"{result['total_time']:.2f} sec"
                    )

            # -------------------------------------------------
            # Save Outputs
            # -------------------------------------------------

            with export_tab:
                st.subheader("Export your results")
                st.caption("STL exports retain the source coordinate system. STL files do not declare units; use the same units as the original model.")
                latent_data = numpy_bytes(latent_representation)
                voxel_data = numpy_bytes(reconstructed_sample)
                packed_size = np.packbits(result["input_voxels"].astype(bool)).nbytes
                columns = st.columns(3)
                columns[0].metric("Original STL", f"{uploaded_file.size / 1024:.1f} KiB")
                columns[1].metric("Latent array (.npy)", f"{len(latent_data) / 1024:.1f} KiB")
                reduction = 100 * (1 - len(latent_data) / uploaded_file.size)
                columns[2].metric("Size reduction vs STL", f"{reduction:.1f}%")
                st.caption(f"Packed binary input baseline: {packed_size / 1024:.1f} KiB (without metadata). Negative reduction means the latent file is larger. Model weights are not included in these sizes.")
                if not save_outputs:
                    st.info("Enable ‘Show download options’ in the sidebar to export results.")
                else:
                    stem = Path(uploaded_file.name).stem
                    st.download_button("Download latent array (.npy)", latent_data,
                                       file_name=f"{stem}_latent.npy", mime="application/octet-stream")
                    st.download_button("Download reconstructed voxels (.npy)", voxel_data,
                                       file_name=f"{stem}_voxels.npy", mime="application/octet-stream")
                    archive = latent_archive(latent_representation, result["transform"], {
                        "format_version": 1, "source_name": uploaded_file.name,
                        "source_sha256": input_key[0], "voxel_dimension": VOXEL_DIM,
                        "point_density": point_density, "threshold": threshold,
                        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                        "coordinate_convention": "origin is voxel-center index (0,0,0)",
                    })
                    st.download_button("Download latent archive with metadata (.npz)", archive,
                                       file_name=f"{stem}_latent.npz", mime="application/octet-stream")
                    st.caption(f"Compressed latent archive including metadata: {len(archive) / 1024:.1f} KiB.")
                    if np.any(reconstructed_sample):
                        st.download_button("Download reconstructed STL", binary_stl(reconstructed_sample, result["transform"]),
                                           file_name=f"{stem}_reconstructed.stl", mime="model/stl")
                    else:
                        st.info("Lower the threshold to create occupied voxels before exporting an STL.")

    except ValueError as exc:
        st.session_state.inference_result = None
        st.error(str(exc))
    except Exception:
        st.session_state.inference_result = None
        logger.exception("STL processing failed")
        st.error("This STL could not be processed. Check that it contains a valid triangle mesh and try exporting it again.")
    finally:
        if temp_file_path is not None:
            try:
                os.remove(temp_file_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception("Unable to remove temporary STL")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "CAD-DR | Deep Learning Based 3D CAD Model "
    "Dimensionality Reduction System"
)

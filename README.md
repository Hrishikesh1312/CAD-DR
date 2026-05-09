# CAD-DR
**Deep Convolutional Autoencoder for Dimensionality Reduction of 3D CAD Models**

CAD-DR is a deep learning-based system for dimensionality reduction of 3D CAD models using a 3D convolutional autoencoder. The system supports full STL to voxel transformation, encoding, reconstruction, and interactive visualization using a Streamlit-based GUI.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Performance](#performance)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

The goal of CAD-DR is to compress 3D models into low-dimensional latent representations while preserving key geometric features. This allows for efficient storage, retrieval, and downstream tasks like shape classification and reconstruction.

**Key Features:**
- 3D STL → Point Cloud → Voxel conversion pipeline
- Fully convolutional encoder-decoder architecture with SELU activations
- Streamlit GUI with real-time visualization (STL, voxel, latent space)
- Latent space export and reconstruction preview
- PyTorch-based implementation with GPU acceleration support
- Pre-trained models included for immediate use

**Compression Performance:**
The developed pipeline can reduce an STL of 25MB size to less than 300kB - a staggering 98% reduction in storage size while preserving geometric details.

---

## Architecture

### Autoencoder (PyTorch Implementation)

**Input Format:** Binary voxel grids (1×64×64×64)

**Encoder:**
- Conv3D(1→32, kernel_size=3, padding=1) → SELU → AvgPool3D(2)
- Conv3D(32→16, kernel_size=3, padding=1) → SELU → AvgPool3D(2)
- **Bottleneck (Latent Space):** 16×16×16×16

**Decoder:**
- Conv3D(16→16, kernel_size=3, padding=1) → SELU → Upsample(2, mode='nearest')
- Conv3D(16→32, kernel_size=3, padding=1) → SELU → Upsample(2, mode='nearest')
- Conv3D(32→1, kernel_size=3, padding=1) → Sigmoid

### Training Configuration

- **Framework:** PyTorch (Torch 2.x compatible)
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy (BCELoss)
- **Metrics:** Binary Accuracy
- **Batch Size:** 10
- **Epochs:** 50 (with early stopping)
- **Early Stopping:** Patience=5, Min Delta=0.0001
- **Device:** Automatic GPU acceleration if CUDA is available

---

## Dataset

- **Source:** ABC Dataset (STL format)
- **Total Models:** 1000 STL files
- **Point Cloud Density:** 20,000 points per model
- **Voxel Resolution:** 64×64×64 (binary voxelization)
- **Train/Test Split:** 80% training (800 models) / 20% testing (200 models)

---

## Project Structure

```
CAD-DR/
├── app.py                 # Streamlit GUI application
├── train.py               # Training script
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
├── README.md
│
├── model/
│   └── autoencoder.py     # PyTorch Autoencoder model definition
│
├── utils/
│   ├── conversion_utils.py    # STL↔PLY↔Voxel conversion utilities
│   └── visualization.py       # 3D visualization functions
│
└── data/
    ├── abc-dataset-stl/       # Input: Original STL files
    ├── abc-dataset-ply/       # Intermediate: Point clouds in PLY format
    ├── sample-stl/            # Sample STL files for testing
    ├── checkpoints/           # Training checkpoints
    └── saved-models/          # Pre-trained models
        ├── autoencoder.pt     # Full autoencoder model
        └── encoder.pt         # Encoder-only model
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Hrishikesh1312/CAD-DR.git
cd CAD-DR/
```

### 2. Environment setup

```bash
conda create -n cad-dr python=3.11
conda activate cad-dr
pip install -r requirements.txt
```

### 3. Verify Installation

The repository includes pre-trained PyTorch models (`autoencoder.pt` and `encoder.pt`), which are sufficient for running the Streamlit application without requiring training.

---

## Usage

### Running the Streamlit Application

To launch the interactive GUI:

```bash
streamlit run app.py
```

**Features:**
- **STL Upload:** Upload and visualize 3D CAD models in STL format with interactive 3D rendering
- **Voxel Visualization:** See the voxelized representation of your models (64³ resolution)
- **Autoencoding:** Run the model to compress and reconstruct 3D shapes
- **Latent Space Encoding:** Extract low-dimensional (16×16×16×16) representations of 3D models
- **Real-time Visualization:** Plotly-based interactive 3D plots for exploration

### Training the Autoencoder

To train the model from scratch on a custom dataset:

#### Step 1: Prepare the Dataset

1. Download the ABC Dataset from https://archive.nyu.edu/handle/2451/43778
2. Extract the 7z archive and select STL files
3. Copy 1000 STL files to `data/abc-dataset-stl/`

#### Step 2: Run Training

```bash
python train.py
```

**Training Pipeline:**
The script automatically handles all preprocessing stages:

1. **STL → PLY Conversion:** Converts STL files to point cloud format (20,000 points per model)
   - Samples uniform points from triangle meshes
   - Skips if PLY files already exist
   
2. **PLY → Voxel Conversion:** Voxelizes point clouds to 64³ binary grids
   
3. **Data Splitting:** Automatically splits into 80% training / 20% testing
   
4. **Model Training:** 
   - Trains with progress bars and real-time loss/accuracy metrics
   - Implements early stopping with checkpoint saving
   - Displays detailed epoch-by-epoch statistics
   
5. **Model Saving:** Saves trained model to `data/saved-models/autoencoder.pt`

**Output:**
- Trained autoencoder weights
- Training logs and metrics
- Checkpoint files for resuming training

---

## Technical Details

### System Requirements

- **Tested On:** Ubuntu 26.04 (originally developed on Ubuntu 22.04)
- **Hardware (Development):** AMD Ryzen 7 4800H (8 cores), 16GB RAM, Nvidia GTX1660Ti (6GB VRAM)
- **Training Time:** ~20 minutes for 1000 samples
- **Framework:** PyTorch 2.x compatible

### Device Support

- **GPU:** Automatically uses CUDA if available for faster training and inference
- **CPU:** Falls back to CPU if CUDA is not available

---

## Performance

- **Compression Ratio:** 98% reduction (25MB → 300KB)
- **Latent Space Size:** 16×16×16×16 (65,536 values)
- **Training Time:** ~15 minutes on GTX1660Ti
- **Binary Accuracy:** Tracked during training with early stopping mechanism

---

## License

This project is open source and available under the MIT License.

For questions, issues, or contributions, please feel free to open an issue or submit a pull request.

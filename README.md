# CAD-DR
**Deep Convolutional Autoencoder for Dimensionality Reduction of 3D CAD Models**

CAD-DR is a deep learning-based system for dimensionality reduction of 3D CAD models using a 3D convolutional autoencoder. The system supports full STL to voxel transformation, encoding, reconstruction, and interactive visualization using a Streamlit-based GUI.

---

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the GUI](#running-the-gui)
  - [Training the Autoencoder](#training-the-autoencoder)
- [Metrics](#metrics)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

The goal of CAD-DR is to compress 3D models into low-dimensional latent representations while preserving key geometric features. This allows for efficient storage, retrieval, and downstream tasks like shape classification and reconstruction.

Key features:
- 3D STL → Point Cloud → Voxel conversion pipeline
- Fully convolutional encoder-decoder architecture
- Streamlit GUI with real-time visualization (STL, voxel, latent space)
- Latent space export and reconstruction preview

The developed pipeline can reduce an STL of 25MB size to less than 300kB - a staggering 98% reduction in storage size, without having to compromise on details.

The hardware used for development is a system having AMD Ryzen 7 4800H with 8 cores, 16GB RAM and Nvidia GTX1660Ti GPU with 6GB VRAM. Training took approximately 20 minutes.
The pipeline has been developed and tested on Ubuntu 22.04 and Fedora 42. It should run without any hassles on any Linux system. For Windows, slight modifications may have to be carried out in parts of the code.
The code had been initially developed using TensorFlow 2.14, but has been updated to work on newer versions, with the latest supported version being 2.19

---

## Architecture

### Autoencoder

- **Input**: Binary voxel grids (64×64×64×1)
- **Encoder**:
  - Conv3D(32) → AvgPool3D
  - Conv3D(16) → AvgPool3D
- **Latent space**: 16×16×16×16
- **Decoder**:
  - Conv3D(16) → UpSample3D
  - Conv3D(32) → UpSample3D
  - Conv3D(1, sigmoid)

---

## Dataset

- **Source**: ABC Dataset (STL format)
- **Preprocessing**:
  - 1000 STL files
  - 20,000 points per point cloud
  - Binary voxelization (64³ resolution)
- **Split**:
  - Training: 800 models
  - Testing: 200 models

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

### 3. Training

This repository includes saved models - both the autoencoder in its entirety as well as the encoder separately, which is enough for running the Streamlit application. However, you can also train the model from scratch on data of your choice.

You would have to manually download the ABC dataset from https://archive.nyu.edu/handle/2451/43778 in case you wish to replicate the training process followed by me. The dataset is available as .obj, .stl, .step formats and so on. In our case, we require the .stl format.
Extract the downloaded 7z archive, and move 1000 files - the STL files itself - to data/abc-dataset-stl folder. Then run:

```bash
python train.py
```

This would generate the .keras saved models in data/saved-models

### 4. Streamlit application

To run the Streamlit application, simply run the following command:

```bash
streamlit run app.py
```

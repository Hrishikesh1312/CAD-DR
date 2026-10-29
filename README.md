# CAD-DR

**Deep convolutional autoencoder for dimensionality reduction and reconstruction of 3D CAD models**

CAD-DR converts an STL mesh into a `64 × 64 × 64` binary voxel grid, compresses it with a 3D convolutional encoder, and reconstructs it with a decoder. Its Streamlit workspace provides interactive previews, reconstruction diagnostics, threshold experiments, and exports for downstream analysis.

## Features

- STL validation, point-cloud sampling, and voxelization
- Cached preprocessing and single-pass encoder/decoder inference
- NVIDIA CUDA, Apple Metal (MPS), and CPU support
- Professional light and charcoal-dark themes; dark mode is the default
- Interactive original, voxel, reconstruction, latent-space, overlay, and cross-section views
- Live reconstruction-threshold adjustment without rerunning the model
- Voxel accuracy, IoU, precision, recall, F1, and execution timing
- Threshold sweep across 101 values with downloadable results
- Downloads for reconstructed STL, voxel arrays, latent arrays, and metadata archives
- Compression statistics based on actual generated files
- Reproducible command-line comparison of compatible model checkpoints
- Clear handling of empty, malformed, or degenerate meshes and model failures

## How it works

```text
STL mesh
   ↓ validate and uniformly sample points
64³ binary voxel grid
   ↓ encoder
16 × 16 × 16 × 16 latent tensor
   ↓ decoder
64³ occupancy probabilities
   ↓ reconstruction threshold
binary reconstructed shape
```

The encoder learns local 3D features and reduces the spatial dimensions twice. The decoder expands those latent features back to the original voxel resolution. Its sigmoid output assigns an occupancy probability between 0 and 1 to every voxel.

The reconstruction threshold converts those probabilities into a binary shape. Lower values usually create denser reconstructions; higher values create sparser ones. Changing the threshold in the application updates the preview, metrics, and downloads without running inference again.

## Architecture

Input shape: `1 × 64 × 64 × 64`

### Encoder

1. `Conv3D(1 → 32, kernel=3, padding=1)` → SELU → `AvgPool3D(2)`
2. `Conv3D(32 → 16, kernel=3, padding=1)` → SELU → `AvgPool3D(2)`
3. Latent tensor: `16 × 16 × 16 × 16` (65,536 values)

### Decoder

1. `Conv3D(16 → 16, kernel=3, padding=1)` → SELU → nearest-neighbor upsampling
2. `Conv3D(16 → 32, kernel=3, padding=1)` → SELU → nearest-neighbor upsampling
3. `Conv3D(32 → 1, kernel=3, padding=1)` → sigmoid

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/Hrishikesh1312/CAD-DR.git
cd CAD-DR

conda create -n cad-dr python=3.11
conda activate cad-dr
pip install -r requirements.txt
```

The repository includes pretrained autoencoder and encoder weights under `data/saved-models/`, so the application can run without retraining.

## Run the application

```bash
streamlit run app.py
```

The Streamlit file watcher is disabled to avoid PyTorch module-inspection errors. Restart the Streamlit process after changing source files.

### Typical workflow

1. Upload a triangulated STL mesh.
2. Choose point-cloud density, reconstruction threshold, and theme in the sidebar.
3. Inspect the original surface and voxelized input.
4. Select **Encode & reconstruct**. The encoder creates the latent representation, then the decoder produces reconstruction probabilities.
5. Explore the result tabs:
   - **Shape comparison:** input and reconstructed shapes, a shared/missing/extra overlay, and cross-sections.
   - **Quality & timing:** voxel metrics plus encoder, decoder, and total inference time.
   - **Experiments:** metrics over 101 thresholds and a downloadable CSV.
   - **Latent space:** interactive channel visualization.
   - **Downloads:** reconstruction and latent exports with measured file sizes.

Inference results persist across ordinary UI reruns. Changing the uploaded file, point-cloud density, or voxel dimension invalidates stale results.

## Exports

The Downloads tab provides:

- Latent tensor as `.npy`
- Reconstructed binary voxel grid as `.npy`
- Compressed `.npz` latent archive containing grid origin and spacing, source/model hashes, threshold, point density, voxel dimension, and format metadata
- Reconstructed binary STL generated with marching cubes

The STL export uses the voxel grid's recorded origin and spacing, retaining the source coordinate system and scale within the limits of sampling and voxel resolution. STL files do not declare physical units, so interpret the export using the same units as the source model.

Compression figures compare the generated latent file with the uploaded STL and show a packed binary voxel baseline. They exclude decoder weights, which are required to reconstruct a latent representation. Importing a latent archive back into the application is not currently supported.

## Reconstruction experiments

The Experiments tab evaluates the current model output at thresholds from `0.00` to `1.00` in increments of `0.01`. It plots IoU, F1, precision, and recall and reports the threshold with the highest IoU.

This is a per-model-input diagnostic. A threshold that performs well for one uploaded shape is not evidence that it generalizes to unseen data.

### Compare checkpoints

Use `evaluate.py` to compare checkpoints that implement the current `ConvolutionalAutoencoder` architecture:

```bash
python evaluate.py \
  --models data/saved-models/autoencoder.pt \
  --inputs data/sample-stl/chair.stl data/sample-stl/car.stl \
  --output-dir data/experiments/baseline \
  --device cpu
```

Pass multiple paths after `--models` to compare training runs. The command samples every input once so each checkpoint receives identical voxel data, then writes:

- `per_input.csv`: every input, model, threshold, BCE, and reconstruction metric
- `threshold_summary.csv`: cohort averages for every model and threshold
- `best_thresholds.csv`: the highest-mean-IoU threshold for each checkpoint
- `manifest.json`: paths, hashes, seeds, configuration, and library versions

Use a validation cohort to select a threshold and a separate untouched test cohort for final reporting. The bundled sample STL files are useful for smoke tests but should not be assumed to be held out from training. Sampling is repeatable within the recorded environment, though results may vary across Open3D versions or compute devices. The output directory must not already exist.

Additional options:

```bash
python evaluate.py --help
```

## Training

The training configuration defaults to:

- ABC Dataset STL inputs
- 20,000 sampled points per mesh
- `64³` voxel grids
- 80/20 train-validation split
- Adam optimizer and binary cross-entropy loss
- Batch size 10
- Up to 50 epochs
- Early stopping with patience 5 and minimum delta `0.0001`

Place training STL files in `data/abc-dataset-stl/`, then run:

```bash
python train.py
```

The script converts STL files to PLY point clouds, voxelizes the dataset, trains the autoencoder, saves checkpoints, and writes the final weights to `data/saved-models/autoencoder.pt` and `data/saved-models/encoder.pt`.

The training script uses the first 80% of the generated array for training and the remaining 20% for validation. Arrange or shuffle source files deliberately if you require a randomized or stratified split.

## Validation metrics

- **Voxel accuracy:** fraction of all occupied and empty voxels classified correctly. Empty space can dominate this score.
- **IoU:** intersection of occupied input and reconstructed voxels divided by their union.
- **Precision:** fraction of reconstructed occupied voxels also occupied in the input.
- **Recall:** fraction of occupied input voxels recovered by the reconstruction.
- **F1:** harmonic mean of precision and recall.

IoU and F1 are generally more informative than voxel accuracy for sparse voxel grids.

## Project structure

```text
CAD-DR/
├── app.py                              # Streamlit application
├── train.py                            # Training pipeline
├── evaluate.py                         # Checkpoint comparison CLI
├── config.py                           # Model and preprocessing defaults
├── requirements.txt
├── model/
│   └── convolutional_autoencoder.py    # Encoder-decoder architecture
├── utils/
│   ├── conversion_utils.py             # Point-cloud and voxel conversion
│   ├── experiments.py                  # Threshold sweep metrics
│   ├── exports.py                      # STL, NumPy, and metadata exports
│   ├── theme.py                        # Light/dark theme palettes
│   ├── validation.py                   # Mesh and voxel validation
│   └── visualization.py                # Plotly and 3D comparison views
├── tests/
│   ├── test_exports_experiments.py
│   └── test_validation.py
└── data/
    ├── sample-stl/                     # Example meshes
    ├── checkpoints/                    # Training checkpoints
    └── saved-models/                   # Pretrained weights
```

## Tests

Run the unit tests with the project environment:

```bash
python -m unittest discover -s tests -v
```

The tests cover mesh and voxel validation, coordinate-preserving STL generation, latent archive round-trips, and threshold experiment behavior.

## Notes and limitations

- The reconstruction is voxel-based and therefore approximate. Thin walls, small holes, and fine features may be lost at `64³` resolution.
- Point-cloud density affects voxel coverage and preprocessing time.
- The latent tensor is a learned feature representation, not a directly editable CAD model.
- Checkpoint comparison currently supports weights compatible with the included architecture.
- Report measured compression and quality for the actual dataset and export format rather than assuming a universal compression ratio.

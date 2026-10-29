"""Compare compatible autoencoder checkpoints on the same sampled STL inputs."""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn.functional as F

from config import POINT_CLOUD_DENSITY, VOXEL_DIM
from model.convolutional_autoencoder import ConvolutionalAutoencoder
from utils.conversion_utils import ConversionUtils
from utils.experiments import threshold_sweep
from utils.validation import validate_mesh, validate_voxels


def evaluate(models, inputs, output_dir, point_density=POINT_CLOUD_DENSITY, seed=42, device='cpu'):
    """Each mesh is sampled once; every checkpoint receives identical voxel inputs."""
    rows = []
    manifest = {'seed': seed, 'point_density': point_density, 'voxel_dimension': VOXEL_DIM,
                'device': device, 'torch_version': torch.__version__,
                'numpy_version': np.__version__, 'open3d_version': o3d.__version__,
                'models': [], 'inputs': []}
    loaded = []
    for path in models:
        path = Path(path).resolve()
        model = ConvolutionalAutoencoder().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        loaded.append((str(path), model))
        manifest['models'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    # Cache voxel inputs on disk to bound memory and avoid repeated random sampling.
    with tempfile.TemporaryDirectory(prefix='cad-dr-eval-') as cache:
        for index, path in enumerate(inputs):
            path = Path(path).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sample_seed = (seed + int(digest[:8], 16)) % (2**31)
            o3d.utility.random.seed(sample_seed)
            mesh = o3d.io.read_triangle_mesh(str(path))
            validate_mesh(mesh)
            voxels = ConversionUtils.convert_pointcloud_to_binvox(
                mesh.sample_points_uniformly(number_of_points=point_density), VOXEL_DIM)
            validate_voxels(voxels, VOXEL_DIM)
            np.save(Path(cache) / f'{index}.npy', voxels.astype(np.float32))
            manifest['inputs'].append({'path': str(path), 'sha256': digest, 'sampling_seed': sample_seed})
        with torch.inference_mode():
            for model_path, model in loaded:
                for index, source in enumerate(manifest['inputs']):
                    voxels = np.load(Path(cache) / f'{index}.npy', allow_pickle=False)
                    tensor = torch.as_tensor(voxels, device=device)[None, None]
                    output = model(tensor)
                    probabilities = output.squeeze().cpu().numpy()
                    bce = F.binary_cross_entropy(output, tensor).item()
                    for row in threshold_sweep(voxels, probabilities):
                        rows.append({'Model': model_path, 'Input': source['path'], 'BCE': bce, **row})
    results = pd.DataFrame(rows)
    summary = results.groupby(['Model', 'Threshold'], as_index=False)[['IoU', 'F1', 'Precision', 'Recall', 'BCE']].mean()
    best = summary.loc[summary.groupby('Model')['IoU'].idxmax()].reset_index(drop=True)
    output_dir = Path(output_dir)
    # Avoid overwriting an existing experiment.
    output_dir.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / 'per_input.csv', index=False)
    summary.to_csv(output_dir / 'threshold_summary.csv', index=False)
    best.to_csv(output_dir / 'best_thresholds.csv', index=False)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', type=Path, required=True)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True,
                        help='STL files from your validation cohort; do not use training inputs.')
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory for this experiment.')
    parser.add_argument('--point-density', type=int, default=POINT_CLOUD_DENSITY)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.point_density < 1:
        parser.error('--point-density must be positive')
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory')
    best = evaluate(args.models, args.inputs, args.output_dir, args.point_density, args.seed, args.device)
    print(best.to_string(index=False))
    print('Thresholds are selected on this cohort. Report final quality on a separate untouched test set.')


if __name__ == '__main__':
    main()

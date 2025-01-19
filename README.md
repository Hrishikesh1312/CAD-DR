# CAD-DR: CAD Dimensionality Reduction

## Introduction
A 3D convolutional autoencoder for the purpose of dimensionality reduction of CAD models. The autoencoder is based on the EfficientNet architecture. The development took place on a system having the following specifications:

- GPU: Nvidia GTX 1660 Ti (6GB)
- CPU: Ryzen 7 4800H
- 16GB RAM
- Ubuntu 22.04

With changes to the original architecture's parameters, a better accuracy and lower loss has been achieved.

## Details
- The models was trained on 800 models from the ABC dataset
- The train dataset comprises of 200 models from the same
- Input format: STL
- The STL files are first converted to point cloud (.ply), and then to 3D binary voxel arrays

## Requirements
The model was created using TensorFlow 2.14. Usage of GPU is suggested, and the TensorFlow installation for the same can be done using pip: 
```bash
pip install tensorflow[and-cuda]
```  
For training the model on a native Windows environment (i.e., without the use of WSL2), the TensorFlow installation can be done as follows:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
```
In either case, the suggested Python version is 3.10

### Other required libraries too can be installed via pip:
| Library | Version |
|---------|---------|
| numpy | 1.26.1 |
| pandas | 2.1.1 |
| pyntcloud | 0.3.1 |
| matplotlib | 3.8.0 |
| open3d | 0.17.0 |

The autoencoder.ipynb file contains code for visualizing in open3d as well as matplotlib.  
The matplotlib visualization will work on all systems while the open3d visualization is geared towards a higher quality, interactive visualization.  

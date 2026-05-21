# medical-img-proc

WIP - plans and ToDos:
 - Cone beam simulator
 - TBD
## Install dependencies
1. As a prequisite, you need to install Cuda Driver with Toolkit libraries. Then you need to add Cuda libraries to the path.
```bash
 export PATH=/usr/local/cuda/bin:$PATH
 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
 ```

2. To install dependencies, you can use pip or uv environment manager. To install dependencies using pip, run the following command:
```bash
uv pip install -r requirements.txt
```

# Downloading sample data (e.g.: Lung HR-CT)
1. Downloading sample <b>Chest HR CT</b> data e.g. from the following source from cancer imaging archive:
https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/ [1] <br>
For downloading from he Cancer Imaging Archive, you also need data retriever tool:
https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide
2. you can download CT collection with the following command e.g.: <br>
`/opt/nbia-data-retriever/bin/nbia-data-retriever --cli Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia -d ./data/`

[1] Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2020.NNC2-0461

# Projection and Backprojection

## Overview
This project implements fast, GPU-accelerated X-ray projection and backprojection for 3D medical CT volumes using CUDA. These operations are fundamental to computed tomography (CT) simulation and reconstruction.

- **Projection** simulates the acquisition of X-ray images (sinograms) from a 3D volume by integrating attenuation along rays from a moving source to a detector. This is used to generate synthetic X-ray or CT projection data from volumetric scans.
- **Backprojection** reconstructs a 3D volume from a set of 2D projections (sinograms) by "smearing" the measured values back along the original ray paths. This is a key step in filtered backprojection (FBP) CT reconstruction.

## CUDA Implementation
Both operations are implemented as custom CUDA kernels for high performance. The code supports multiple modes:
- **Mode 0:** Nearest-neighbour interpolation
- **Mode 1:** Bilinear interpolation
- **Mode 2:** Hardware-accelerated trilinear interpolation using CUDA 3D textures (recommended for best quality and speed)

The main Python interfaces are:
- `Projector` (in `utils/CudaWrapper.py`):
    - Method: `project(volume)`
    - Input: 3D numpy array (volume)
    - Output: 3D numpy array (sinogram: [nproj, nv, nu])
- `Backprojector` (in `utils/CudaWrapper.py`):
    - Method: `backproject(sinogram)`
    - Input: 3D numpy array (sinogram)
    - Output: 3D numpy array (reconstructed volume)

## Example Usage
See `examples/x_ray_simulator_example.py` for a full pipeline:
1. Load and preprocess a CT volume
2. Simulate X-ray projections using `Projector`
3. (Optionally) Filter the projections (e.g., with a Hamming filter)
4. Reconstruct the volume using `Backprojector`
5. Visualize the result

## Applications
- Simulating X-ray/CT acquisition for algorithm development
- Testing and benchmarking CT reconstruction methods
- Educational demonstrations of tomographic principles

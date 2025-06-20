#  Perspective Crop Layers (PCL) – This repository makes code modular and working for higher environment dependecies and docker,HPC integration explained

This is repositroy is from the original [Perspective Crop Layers (PCL)](https://github.com/yu-frank/PerspectiveCropLayers) repository introduces a modular, scalable, and maintainable design for handling multiple human pose estimation datasets, including Human3.6M and MPI-INF-3DHP.

> **PCL: Perspective Crop Layers for Monocular 3D Human Pose Estimation**  
> [arXiv 2011.13607](https://arxiv.org/abs/2011.13607)

---

## Python 3.9 Compatibility and Dependency Fixes

The original repository was written for **Python 3.6** and uses outdated libraries (e.g. `torch._six`). To ensure reproducibility and compatibility with modern systems:

- Migrated codebase to **Python 3.9**
- Rebuilt all requirements in `req_new.txt`
- Updated deprecated/removed modules

### Fix for `torch._six` Removal

One key fix applied is the replacement of deprecated imports in `margipose/data/__init__.py`:

```diff
- from torch._six import string_classes, int_classes
+ string_classes = (str,)
+ int_classes = (int,)
```
## Perspective Crop Layers – Modular Refactor

###  Key Improvements

Changes successfuly tested in humanPose-demo1.ipynb file.

#### 1. `config_data.py`: Unified Dataset Configuration

Introduced a new module `config_data.py` to handle all dataset-specific configurations, making it easy to plug in additional datasets or update parameters without modifying the main pipeline.
These files codeis modularized :- train.py, runner.py, runner_utils.py, model.py

##### Features:
- Canonical skeleton conversion
- Access to mean/std for both 2D and 3D normalization
- Dynamic switching between STN (rectangular crop) and PCL (virtual camera crop) statistics
- Extensible for additional datasets
- Device-aware normalization using `get_mean_std_normalized(device)`

##### Example:
```python
from config_data import get_dataset_config

dataset_config = get_dataset_config("H36m")
mean_3d = dataset_config.get_joint_mean()
std_2d, mean_2d = dataset_config.get_2d_mean_std(slant=True, stn=False)
canonical_pose = dataset_config.to_canonical(pose_2d)
```
## Role of config_data.py
The file config_data.py introduces object-oriented modularity by encapsulating dataset specific logic into polymorphic classes:
 - H36mConfig and MPI3DHPConfig both inherit from DatasetConfig.
 - Each config handles its own mean, std, and canonicalization.
 - The factory get_dataset_config(name) dynamically returns the correct class
   
This design enables:
 - Consistent access to normalization and canonical joint order.
 - Cleaner high-level logic (e.g., preprocessing) without if dataset == ... clutter.
 - Easy support for new datasets with minimal changes


## ⚙ Installation (Python 3.9 with Conda)

To set up the environment:

```bash
# 1. Create and activate a new Conda environment
conda create -n pcl_env python=3.9 -y
conda activate pcl_env

# 2. Install dependencies
pip install -r req_new.txt

# 3. (Optional) Enable Jupyter kernel for this environment
pip install ipykernel jupyterlab
python -m ipykernel install --user --name=pcl --display-name "Python (PCL)"

```
### SLURM HPC Environment Setup
This repository supports use on HPC systems with SLURM. To prepare your environment:

Similarly as explained create venv and install req_new.txt in your HPC system. To run training, use the provided SLURM batch script(proposed script) `run_code.sbatch`. Replace the dataset path in the script to match your cluster dataset directory path.

## Using Docker and Makefile
The repository includes a Dockerfile and Makefile for easy reproducibility, dependency isolation, and testing. I have written a unit test located in `tests/test_runner.py` which validates forward pass of `LinearModel` using mock 2D input to see everything works correctly. Below are the make commands to create docker image and run the container.

```bash
# 1. make docker  # Build the Docker image 'pcl-train'
# 2. make run-docker  # Run an interactive container using that image
# 3. make test-docker   # Run tests inside Docker
# 4. make clean         # Remove intermediate Docker artifacts
```
It should look like this ![Docker Container](imgs/Screenshot%20from%202025-06-19%2023-39-42.png)

### Singularity Support for SLURM HPC
Since Docker requires root privileges and cannot run directly on most SLURM clusters, this repo also supports Singularity.
Steps to use Docker image with Singularity on HPC:
1. Initialize your Docker VM on HPC system. For ELwetritsch system 
   
```bash
# 1. rz-docker init     # Only needed once
# 2. rz-docker check    # Check if VM is ready
# 3. rz-docker login    # Enter your personal Docker VM
```
2. Build and save Docker image.
3. Transfer the image to the cluster using scp or rsync commands.
4. Convert the Docker image to a Singularity image: Example below
```bash
singularity build pcl-train.simg docker-archive:///work/your_username/pcl-train.tar

```
5. Run the converted image with Singularity `singularity run pcl-train.simg`
6. Submit via SLURM using Singularity. Example script
```bash
#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -J PCLTrain
#SBATCH --output=pcl_train_%j.out

module load singularity
singularity exec --nv /work/your_username/pcl-train.simg conda run -n pcl_env python train.py

```
   

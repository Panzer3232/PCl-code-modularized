#  Perspective Crop Layers (PCL) – This repository makes code modular and working for higher environment dependecies

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

#### 1. `config_data.py`: Unified Dataset Configuration

Introduced a new module `config_data.py` to handle all dataset-specific configurations, making it easy to plug in additional datasets or update parameters without modifying the main pipeline.

##### Features:
- Canonical skeleton conversion
- Access to mean/std for both 2D and 3D normalization
- Dynamic switching between STN (rectangular crop) and PCL (virtual camera crop) statistics
- Extensible for additional datasets

##### Example:
```python
from config_data import get_dataset_config

dataset_config = get_dataset_config("H36m")
mean_3d = dataset_config.get_joint_mean()
std_2d, mean_2d = dataset_config.get_2d_mean_std(slant=True, stn=False)
canonical_pose = dataset_config.to_canonical(pose_2d)
```
## ⚙ Installation (Python 3.9 with Conda)

To set up the environment:

```bash
# 1. Create and activate a new Conda environment
conda create -n pcl python=3.9 -y
conda activate pcl

# 2. Install dependencies
pip install -r req_new.txt

# 3. (Optional) Enable Jupyter kernel for this environment
pip install ipykernel jupyterlab
python -m ipykernel install --user --name=pcl --display-name "Python (PCL)"

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


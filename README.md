#  Perspective Crop Layers (PCL) – Refactored and Improved for Python 3.9+

This repository improves and modernizes the official implementation of the **CVPR 2020 paper**:

> **PCL: Perspective Crop Layers for Monocular 3D Human Pose Estimation**  
> [arXiv 2011.13607](https://arxiv.org/abs/2011.13607)

---

## 🚀 Python 3.9 Compatibility and Dependency Fixes

The original repository was written for **Python 3.6** and uses outdated libraries (e.g. `torch._six`). To ensure reproducibility and compatibility with modern systems:

- Migrated codebase to **Python 3.9**
- Rebuilt all requirements in `new_req.txt`
- Updated deprecated/removed modules

### Fix for `torch._six` Removal

One key fix applied is the replacement of deprecated imports in `margipose/data/__init__.py`:

```diff
- from torch._six import string_classes, int_classes
+ string_classes = (str,)
+ int_classes = (int,)

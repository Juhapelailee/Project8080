"""
Visualizes a T1-weighted MRI scan (.nii or .nii.gz)
in axial, coronal, and sagittal planes.

Usage from the command line:
    python visualize_mri.py path/to/sub-12_ses-wave1_T1w.nii.gz

Or edit the DEFAULT_PATH variable below and run without arguments.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# This is used if no command line argument is provided. Change it to your own file path if needed.
DEFAULT_PATH = "data/raw/sub-12/ses-wave1/anat/sub-12_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz"


def load_scan(path: str) -> np.ndarray:
    """Loads a NIfTI file and returns its data as a numpy array."""
    img = nib.load(path)
    data = img.get_fdata()
    print(f"Loaded: {path}")
    print(f"  Image size (voxels): {data.shape}")
    print(f"  Voxel size (mm):      {img.header.get_zooms()}")
    return data


def show_three_views(data: np.ndarray, title: str = "") -> None:
    """Displays central slices in axial, coronal, and sagittal planes."""
    x_mid, y_mid, z_mid = (s // 2 for s in data.shape)

    coronal = data[x_mid, :, :]
    axial = data[:, y_mid, :]
    sagittal = data[:, :, z_mid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # np.rot90 rotates the slice so it displays anatomically right-side up
    axes[0].imshow(np.rot90(sagittal), cmap="gray")
    axes[0].set_title("Sagittal")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(coronal), cmap="gray")
    axes[1].set_title("Coronal")
    axes[1].axis("off")

    axes[2].imshow(np.rot90(axial), cmap="gray")
    axes[2].set_title("Axial")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("mri_three_views.png", dpi=150, bbox_inches="tight")
    print("Image saved: mri_three_views.png")
    plt.show()


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    if not Path(path).exists():
        print(f"Error: file not found at path '{path}'")
        print("Provide the correct path as an argument: python visualize_mri.py path/to/file.nii.gz")
        sys.exit(1)

    data = load_scan(path)
    show_three_views(data, title=Path(path).name)


if __name__ == "__main__":
    main()

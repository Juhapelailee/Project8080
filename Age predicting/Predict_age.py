"""THis script trains a 3D CNN to predict a subject's age from their T1w MRI volume."""

#IMPORTS
from datetime import datetime
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader

#Requirements from other folders
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_ROOT = DATA_DIR / "raw"
PARTICIPANTS_TSV = DATA_DIR / "participants.tsv"
AGE_COLUMN = "AgeMRI_W1"

# Downsampled from the native (256, 256, 160) so a batch fits in memory on a
# laptop: 5 stride-2 pools in BrainAgeCNN need dims divisible by 32 anyway.
TARGET_SHAPE = (128, 128, 96)

BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def find_t1w_paths(data_root: Path = DATA_ROOT) -> dict[str, Path]:
    """Finds each subject's T1w scan under data/raw.

    A subject with multiple runs (e.g. sub-752 has run-1 and run-2) keeps
    only its lowest-numbered run, so the result has one scan per subject.

    Returns a dict: {"sub-1003": Path("data/raw/sub-1003/.../..._T1w.nii.gz"), ...}
    """
    paths: dict[str, Path] = {}
    for path in sorted(data_root.glob("sub-*/ses-wave1/anat/*_T1w.nii.gz")):
        subject_id = path.name.split("_")[0]
        paths.setdefault(subject_id, path)
    return paths


def load_ages(tsv_path: Path = PARTICIPANTS_TSV, column: str = AGE_COLUMN) -> dict[str, float]:
    """Reads participants.tsv and returns {subject_id: age}, skipping n/a ages.

    Returns a dict: e.g. {"sub-1003": 54.0, "sub-1007": 73.0, ...} (Subject id and age in years)
    """
    with open(tsv_path, newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return {
            row["participant_id"]: float(row[column])
            for row in rows
            if row[column] not in ("n/a", "")
        }


def split_subjects(
    subject_ids: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Splits subject IDs into train/val/test lists (remaining fraction goes to test)."""
    ids = sorted(subject_ids)
    random.Random(seed).shuffle(ids)

    n_train = int(len(ids) * train_frac)
    n_val = int(len(ids) * val_frac)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


class MRIAgeDataset(Dataset):
    """Loads T1w MRI volumes paired with their Wave 1 MRI-visit age."""

    def __init__(self, scan_paths: dict[str, Path], ages: dict[str, float], subject_ids: list[str]):
        self.subject_ids = subject_ids
        self.scan_paths = scan_paths
        self.ages = ages

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject_id = self.subject_ids[index]

        volume = nib.load(self.scan_paths[subject_id]).get_fdata(dtype=np.float32)
        # Z-score per volume: raw voxel intensities aren't comparable across
        # subjects/scanners, so scale each volume to its own mean 0, std 1.
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)

        # Adds the channel dimension torch conv3d layers expect: (1, D, H, W)
        tensor = torch.from_numpy(volume).unsqueeze(0)

        # interpolate needs a batch dim too: (1, 1, D, H, W) in, then drop it again
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=TARGET_SHAPE, mode="trilinear", align_corners=False
        ).squeeze(0)

        age = torch.tensor(self.ages[subject_id], dtype=torch.float32)
        return tensor, age


class BrainAgeCNN(nn.Module):
    """Lightweight 3D CNN for age regression from a T1w volume (SFCN-style).

    Each block halves every spatial dimension, so 5 blocks need input dims
    divisible by 32 (see TARGET_SHAPE). Global average pooling before the
    final linear layer keeps the parameter count small and avoids overfitting
    on a few hundred training subjects like we have here. The final output is a single scalar (predicted age). 
    Groupnorm is used instead of BatchNorm3d because with a batch size of 4 the per-batch statistics BatchNorm relies on are too noisy and destabilize training;
    GroupNorm normalizes within each sample instead, so it's unaffected by batch size. 
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        block_channels = [32, 64, 128, 256, 256]

        blocks = []
        prev_channels = in_channels
        for out_channels in block_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(prev_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(2),
                )
            )
            prev_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.4)
        self.regressor = nn.Linear(block_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.regressor(x).squeeze(1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    age_mean: float,
    age_std: float,
) -> float:
    """Runs one training epoch and returns the mean MSE loss (on normalized ages)."""
    model.train()
    total_loss = 0.0
    for volumes, ages in loader:
        volumes, ages = volumes.to(device), ages.to(device)
        normalized_ages = (ages - age_mean) / age_std

        optimizer.zero_grad()
        predictions = model(volumes)
        loss = F.mse_loss(predictions, normalized_ages)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * volumes.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, age_mean: float, age_std: float
) -> float:
    """Returns the mean absolute error in years."""
    model.eval()
    total_abs_error = 0.0
    for volumes, ages in loader:
        volumes, ages = volumes.to(device), ages.to(device)
        predictions = model(volumes) * age_std + age_mean
        total_abs_error += (predictions - ages).abs().sum().item()
    return total_abs_error / len(loader.dataset)


if __name__ == "__main__":

    scan_paths = find_t1w_paths()
    ages = load_ages()
    usable_ids = sorted(set(scan_paths) & set(ages))
    print(f"Found {len(scan_paths)} scans, {len(ages)} subjects with {AGE_COLUMN}.")
    print(f"Usable subjects (scan + age): {len(usable_ids)}")

    train_ids, val_ids, test_ids = split_subjects(usable_ids)
    print(f"Train/val/test sizes: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    train_set = MRIAgeDataset(scan_paths, ages, train_ids)
    val_set = MRIAgeDataset(scan_paths, ages, val_ids)
    test_set = MRIAgeDataset(scan_paths, ages, test_ids)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Normalize ages using train-set statistics only, to avoid leaking val/test info.
    train_ages = torch.tensor([ages[s] for s in train_ids])
    age_mean, age_std = train_ages.mean().item(), train_ages.std().item()
    print(f"Train age mean/std: {age_mean:.1f} / {age_std:.1f} years")

    # Use the fastest device available: CUDA GPU (most machines/servers),
    # Apple GPU (mps, e.g. this Mac), otherwise fall back to plain CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = BrainAgeCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Sanity checks before committing to a full training run: confirm the
    # data pipeline produces sane shapes/values, and that a forward pass
    # through the untrained model works before wasting time if something's off.
    sample_volumes, sample_ages = next(iter(train_loader))
    print(f"Sample batch shape: {tuple(sample_volumes.shape)}, dtype: {sample_volumes.dtype}")
    print(
        f"Sample batch value range: [{sample_volumes.min():.2f}, {sample_volumes.max():.2f}], "
        f"mean: {sample_volumes.mean():.2f}"
    )
    print(f"Sample batch ages: {sample_ages.tolist()}")

    with torch.no_grad():
        sample_predictions = model(sample_volumes.to(device)) * age_std + age_mean
    print(f"Trying untrained model predictions: {age_mean:.1f}): {sample_predictions.tolist()}")
    print("Starting training...\n")

    # Timestamping the filename avoids overwriting a previous checkpoint if the script is re-run.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = CHECKPOINT_DIR / f"brain_age_cnn_{datetime.now():%Y%m%d_%H%M%S}.pt"
    best_val_mae = float("inf")
    best_epoch = 0

    #TRAINING
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, age_mean, age_std)
        val_mae = evaluate(model, val_loader, device, age_mean, age_std)
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} - train loss: {train_loss:.4f} - val MAE: {val_mae:.2f} years")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Best val MAE: {best_val_mae:.2f} years at epoch {best_epoch} (checkpoint saved to {CHECKPOINT_PATH})")

    # Now the best checkpoint is evaluated on the test
    # testing set never influenced training or the epoch/hyperparameter choices.

    #TESTING   
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    test_mae = evaluate(model, test_loader, device, age_mean, age_std)
    print(f"Test MAE (best checkpoint, epoch {best_epoch}): {test_mae:.2f} years")

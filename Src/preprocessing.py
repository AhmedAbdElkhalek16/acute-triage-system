"""
preprocessing.py
----------------
Unified preprocessing pipeline for medical images.
Handles both X-Ray (PNG/DICOM) and CT (DICOM series).
"""

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

IMAGE_SIZE = 512          # All images resized to 512×512
CT_WINDOW  = (-1000, 400) # Hounsfield unit range for CT windowing

# ImageNet stats — used because EfficientNet/DenseNet were pretrained on it
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
#  DICOM Loaders
# ─────────────────────────────────────────────

def load_dicom_xray(path: str) -> np.ndarray:
    """
    Load a chest X-Ray DICOM file → uint8 numpy array (H, W, 3).
    Applies VOI LUT (window/level) if present, then normalizes to 0–255.
    """
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # Some scanners store inverted images (MONOCHROME1)
    if hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img

    # Normalize to 0–255
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    # Convert grayscale → RGB (models expect 3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def load_dicom_ct(path: str,
                  wl: int = -600,
                  ww: int = 1400) -> np.ndarray:
    """
    Load a CT DICOM slice and apply Hounsfield windowing.
    Default window: lung window (WL=-600, WW=1400).

    Common windows:
        Brain:  WL=40,   WW=80
        Lung:   WL=-600, WW=1400
        Soft:   WL=50,   WW=400
    """
    ds  = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    # Convert to Hounsfield Units
    slope     = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    img       = img * slope + intercept

    # Apply window
    low  = wl - ww / 2
    high = wl + ww / 2
    img  = np.clip(img, low, high)
    img  = (img - low) / (high - low)
    img  = (img * 255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def load_image(path: str, modality: str = 'xray') -> np.ndarray:
    """
    Universal loader — auto-detects PNG vs DICOM.
    modality: 'xray' | 'ct'
    """
    path = str(path)

    if path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # DICOM
    if modality == 'ct':
        return load_dicom_ct(path)
    return load_dicom_xray(path)


# ─────────────────────────────────────────────
#  Augmentation Pipelines
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """
    Training augmentations — conservative, clinically valid.
    We do NOT flip CT vertically (anatomically wrong) but do allow
    slight rotations to simulate patient positioning variation.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.05,
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4
        ),
        A.GaussNoise(p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),   # Enhances local contrast
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """Validation/Test — no augmentation, only resize + normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
#  Dataset Classes
# ─────────────────────────────────────────────

class XRayDataset(Dataset):
    """
    Dataset for Chest X-Ray classification.

    Expects a DataFrame with columns:
        image_path  : str  — path to PNG or DICOM
        label       : int  — 0=normal, 1=pneumothorax, 2=pneumonia
    """

    LABEL_MAP = {
        0: 'Normal',
        1: 'Pneumothorax',
        2: 'Pneumonia',
    }

    def __init__(self, df, transforms=None, modality='xray'):
        self.df         = df.reset_index(drop=True)
        self.transforms = transforms
        self.modality   = modality

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = load_image(row['image_path'], modality=self.modality)
        label = int(row['label'])

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, label


class CTDataset(Dataset):
    """
    Dataset for CT scan classification.

    Expects a DataFrame with columns:
        image_path  : str
        label       : int  — 0=normal, 1=hemorrhage, 2=embolism
        window      : str  — 'brain' | 'lung' | 'soft'  (optional)
    """

    WINDOW_PRESETS = {
        'brain': (-40,  80),   # WL, WW
        'lung':  (-600, 1400),
        'soft':  (50,   400),
    }

    LABEL_MAP = {
        0: 'Normal',
        1: 'Intracranial Hemorrhage',
        2: 'Pulmonary Embolism',
    }

    def __init__(self, df, transforms=None, window='brain'):
        self.df         = df.reset_index(drop=True)
        self.transforms = transforms
        wl, ww          = self.WINDOW_PRESETS.get(window, (-40, 80))
        self.wl         = wl
        self.ww         = ww

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = load_dicom_ct(row['image_path'], wl=self.wl, ww=self.ww)
        label = int(row['label'])

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, label


# ─────────────────────────────────────────────
#  Quick sanity check (run this file directly)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd

    print("Testing transforms on a dummy image...")

    dummy = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    t = train_tf(image=dummy)['image']
    v = val_tf(image=dummy)['image']

    print(f"  Train output shape : {t.shape}")   # torch.Size([3, 512, 512])
    print(f"  Val   output shape : {v.shape}")
    print(f"  dtype              : {t.dtype}")
    print("Preprocessing OK!")
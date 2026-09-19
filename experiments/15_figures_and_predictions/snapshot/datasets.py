# -*- coding: utf-8 -*-
"""
DvXray dual-view dataset loader (datasets.py)
- Returns ((imgA, imgB), target)
- Two modes, training / validation (synchronised augmentation)
- Build the DataLoader through build_loaders(args)
"""

import os
import random
from typing import List, Tuple
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from models.modules.augmentations import RandAugment, TrivialAugmentWide


# ===========================
# Basic constants
# ===========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ===========================
# Helper functions
# ===========================
def _read_class_names(path: str, num_classes: int) -> List[str]:
    """Read the class-name file; fall back to cls_0, cls_1, ..."""
    if path and os.path.isfile(path):
        names = [ln.strip() for ln in open(path, "r", encoding="utf-8") if ln.strip()]
        if len(names) >= num_classes:
            return names[:num_classes]
    return [f"cls_{i}" for i in range(num_classes)]

def _parse_labels(tokens: List[str], num_classes: int) -> torch.Tensor:
    """Parse the multi-hot labels; space or comma separated"""
    if len(tokens) == 1 and ("," in tokens[0]):
        vals = [int(x) for x in tokens[0].replace(",", " ").split()]
    else:
        vals = [int(x) for x in tokens]
    if len(vals) != num_classes:
        raise ValueError(f"wrong label dimension: expected {num_classes}, got {len(vals)}; content={vals[:10]}")
    return torch.tensor(vals, dtype=torch.float32)

def _parse_line(line: str, num_classes: int) -> Tuple[str, str, torch.Tensor]:
    """
    Line format:
        imgA_path imgB_path label1 label2 ... labelK
    or
        imgA_path imgB_path 1,0,0,1,...,0
    """
    toks = line.strip().split()
    if len(toks) < 3:
        raise ValueError(f"malformed annotation line: {line}")
    a, b = toks[0], toks[1]
    y = _parse_labels(toks[2:], num_classes)
    return a, b, y

# ===========================
# [core change] synchronised augmentation class
# ===========================
class SynchronizedTransform:
    """
    Wrapper that applies exactly the same random parameters to both views.
    """
    def __init__(self, transform: nn.Module):
        self.transform = transform

    def __call__(self, imgA: Image.Image, imgB: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # save and fix the RNG state so that both transform calls draw the same randomness
        seed = random.randint(0, 2**32)
        random.seed(seed)
        torch.manual_seed(seed)
        imgA = self.transform(imgA)
        
        random.seed(seed)
        torch.manual_seed(seed)
        imgB = self.transform(imgB)
        
        return imgA, imgB


# ===========================
# Dataset class (final revision)
# ===========================
class DualViewTxtDataset(Dataset):
    def __init__(
        self,
        list_file: str,
        input_size: int,
        num_classes: int,
        train: bool,
        class_names: List[str],
        aug_mode: str = 'standard',
        rand_aug_n: int = 2,
        rand_aug_m: int = 9,
        view_mode: str = 'paired',
    ):
        super().__init__()
        self.samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    self.samples.append(ln.strip())

        self.num_classes = num_classes
        self.input_size = input_size
        self.train = train
        self.view_mode = str(view_mode).lower()
        if self.view_mode not in {'paired', 'a_only', 'b_only', 'mismatched'}:
            raise ValueError(
                "view_mode must be paired, a_only, b_only, or mismatched; "
                f"got {view_mode!r}"
            )
        # changed to:
        self.use_conditional_aug = aug_mode.startswith('conditional')

        # --- [fix] identify the "weak" classes ---
        hard_class_names = {"Scissors", "Lighter", "Razor_blade","Knife"}
        self.hard_class_indices = {i for i, name in enumerate(class_names) if name in hard_class_names}
        self.class_names = class_names
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}

        if self.train:
            # only log "enabled" when the master switch is on
            if self.use_conditional_aug:
                print(f"[targeted augmentation] enabled (ON), strengthened training is applied to weak-class indices {self.hard_class_indices}.")
            else:
                print("[targeted augmentation] disabled (OFF), every sample uses standard augmentation.")

        # --- [fix] replace the placeholder with the real augmentation pipeline ---
        if self.train:
            # base augmentation (crop and flip)
            base_transforms = [
                T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3./4., 4./3.)),
                T.RandomHorizontalFlip(),
            ]
            if aug_mode == 'none':
                # [new] mode 0: no augmentation (resize only)
                self.transforms = SynchronizedTransform(T.Resize((input_size, input_size), antialias=True)) 

            elif aug_mode == 'standard':
                # mode 1: standard (mild colour jitter only)
                self.transforms = SynchronizedTransform(T.Compose(base_transforms + [T.ColorJitter(0.1, 0.1, 0.1, 0.05)]))
            
            elif aug_mode == 'conditional':
                # mode 2: conditional (two pipelines)
                self.normal_transforms = SynchronizedTransform(T.Compose(base_transforms + [T.ColorJitter(0.1, 0.1, 0.1, 0.05)]))
                self.strong_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]))

            elif aug_mode == 'conditional_4':
                self.aug_mode = 'conditional_4'
                self.normal_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                ]))
                self.strong_transforms = SynchronizedTransform(T.Compose([
                    T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),   # gentler than conditional_2
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                ]))
                # reuse the existing weak-class set (adjust hard_class_names at the top of __init__)
                self.hard_class_indices = set(getattr(self, "hard_class_indices", set()))

            elif aug_mode == 'rand_aug':
                # mode 3: RandAugment
                self.transforms = SynchronizedTransform(T.Compose(base_transforms + [RandAugment(n=rand_aug_n, m=rand_aug_m)]))
            
            elif aug_mode == 'trivial_aug':
                # mode 4: TrivialAugment
                self.transforms = SynchronizedTransform(T.Compose(base_transforms + [TrivialAugmentWide()]))
            
            else:
                raise ValueError(f"Unknown aug_mode: {aug_mode}")
        
        else: # validation split
            self.val_transforms = SynchronizedTransform(T.Resize((input_size, input_size), antialias=True))


        # the final tensor conversion and normalisation are shared by all modes
        self.to_tensor_and_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.samples)

    # --- [fix] the old, redundant augmentation helpers were removed safely ---

    def __getitem__(self, idx: int):
        a_path, b_path, y = _parse_line(self.samples[idx], self.num_classes)
        if self.view_mode == 'mismatched':
            if len(self.samples) < 2:
                raise ValueError("mismatched view evaluation requires at least two samples")
            mismatch_idx = (idx + len(self.samples) // 2) % len(self.samples)
            _, b_path, _ = _parse_line(
                self.samples[mismatch_idx], self.num_classes
            )
        if not os.path.isfile(a_path) or not os.path.isfile(b_path):
            raise FileNotFoundError(f"missing image: {a_path} or {b_path}")

        imgA = Image.open(a_path).convert("RGB")
        imgB = Image.open(b_path).convert("RGB")

        # Controlled view ablations. Copies are made before synchronized
        # augmentation so a_only/b_only isolate view content, not RNG noise.
        if self.view_mode == 'a_only':
            imgB = imgA.copy()
        elif self.view_mode == 'b_only':
            imgA = imgB.copy()
        
        if self.train:
            if getattr(self, "aug_mode", None)  == 'conditional_4' or self.use_conditional_aug:
                # conditional augmentation: route samples by weak-class membership
                contains_hard_class = any((y[i] == 1) for i in self.hard_class_indices)
                if contains_hard_class and self.strong_transforms is not None:
                    imgA, imgB = self.strong_transforms(imgA, imgB)
                else:
                    # legacy conditional uses normal_transforms; standard/none use self.transforms
                    if hasattr(self, "normal_transforms") and self.normal_transforms is not None:
                        imgA, imgB = self.normal_transforms(imgA, imgB)
                    else:
                        imgA, imgB = self.transforms(imgA, imgB)

            else:
                # other training modes: standard / none / rand_aug / trivial_aug
                imgA, imgB = self.transforms(imgA, imgB)

        else:
            # validation always uses val_transforms
            imgA, imgB = self.val_transforms(imgA, imgB)

        # final toTensor and normalisation
        ta = self.to_tensor_and_norm(imgA)
        tb = self.to_tensor_and_norm(imgB)
        
        return (ta, tb), y

   
# ===========================
# DataLoader construction
# ===========================
def build_loaders(args):
    """
    Called from main_finetune.py
    args must contain:
        train_list / val_list / classes_file / input_size / batch_size / num_classes
    Returns:
        train_loader, val_loader, class_names
    """
    num_workers = getattr(args, "num_workers", 8)
    class_names = _read_class_names(
        getattr(args, "classes_file", ""), args.num_classes
    )
    # --- [core fix] read every new augmentation argument from args ---
    train_set = DualViewTxtDataset(
        args.train_list, args.input_size, args.num_classes, train=True, class_names=class_names,
        aug_mode=args.aug_mode,
        rand_aug_n=args.rand_aug_n,
        rand_aug_m=args.rand_aug_m,
        view_mode=getattr(args, "view_mode", "paired"),
    )
    # -----------------------------------------------

    val_set = DualViewTxtDataset(
        args.val_list, args.input_size, args.num_classes, train=False, class_names=class_names,
        view_mode=getattr(args, "view_mode", "paired"),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, class_names

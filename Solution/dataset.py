import os
import pathlib

import einops
import numpy
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGE_WIDTH = 860
IMAGE_HEIGHT = 520
SCREENSHOTS_DIR = 'screenshots'
MASKS_LOOT_DIR = 'masks_loot'
MASKS_ENEMIES_DIR = 'masks_enemies'


class LethalDataset(Dataset):
    def __init__(self, root_dir, percent_from=0, percent_to=100, seed=31337, augmentations=False):
        self.root_dir = pathlib.Path(root_dir)
        self.image_files = sorted(os.listdir(self.root_dir / 'screenshots'))
        random.seed = seed
        if 'test' not in str(self.root_dir):
            random.shuffle(self.image_files)
        assert 'test' not in str(self.root_dir) or (augmentations is False and percent_from == 0 and percent_to == 100)
        self.augmentations = augmentations
        self.image_files = self.image_files[
                           len(self.image_files) * percent_from // 100:
                           len(self.image_files) * percent_to // 100]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        def transform(img):
            return einops.rearrange(torch.asarray(numpy.array(img), dtype=torch.float), 'h w c -> c h w') / 255.0

        def mask_transform(mask_img):
            return torch.asarray(numpy.array(mask_img), dtype=torch.float) / 255.0

        def augmentation(img, masks):
            if random.random() < 0.25:
                img = torch.flip(img, dims=(-1,))
                masks = [torch.flip(m, dims=(-1,)) for m in masks]
            if random.random() < 0.35:
                img = torch.clamp(img * torch.normal(1.0, 0.03, (1,)), min=0.0, max=1.0)
            return img, masks

        name = self.image_files[idx]
        with Image.open(self.root_dir / SCREENSHOTS_DIR / name) as img:
            if 'test' in str(self.root_dir):
                return name, transform(img)
            with Image.open(self.root_dir / MASKS_ENEMIES_DIR / name) as mask_en:
                with Image.open(self.root_dir / MASKS_LOOT_DIR / name) as mask_lt:
                    img, mask_en, mask_lt = transform(img), mask_transform(mask_en), mask_transform(mask_lt)
                    if self.augmentations:
                        img, (mask_en, mask_lt) = augmentation(img, (mask_en, mask_lt))
                    return name, img, mask_en, mask_lt

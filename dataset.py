import os
import tifffile as tiff
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class ChangeDetectionDataset(Dataset):

    def __init__(self, root_dir):

        self.pre_dir = os.path.join(root_dir, "pre-event")
        self.post_dir = os.path.join(root_dir, "post-event")
        self.mask_dir = os.path.join(root_dir, "target")

        self.files = os.listdir(self.pre_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        pre_path = os.path.join(self.pre_dir, file_name)
        post_path = os.path.join(self.post_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)

        # Read images
        pre = tiff.imread(pre_path)
        post = tiff.imread(post_path)
        mask = tiff.imread(mask_path)

        # Label remapping
        # Background (0) -> 0
        # Intact (1) -> 0
        # Damaged (2) -> 1
        # Destroyed (3) -> 1
        mask = np.where(mask >= 2, 1, 0).astype(np.uint8)

        # Resize
        pre = cv2.resize(pre, (256, 256))
        post = cv2.resize(post, (256, 256))

        # IMPORTANT: preserve mask labels
        mask = cv2.resize(
            mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        # Ensure binary mask
        mask = (mask > 0).astype(np.float32)

        # Normalize EO image
        pre = pre.astype(np.float32) / 255.0

        # Normalize SAR image
        post = post.astype(np.float32)
        post = (post - post.min()) / (post.max() - post.min() + 1e-8)

        # Add SAR channel dimension
        if len(post.shape) == 2:
            post = np.expand_dims(post, axis=-1)

        # Combine EO + SAR
        image = np.concatenate([pre, post], axis=-1)

        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1).float()

        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask
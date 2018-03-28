import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset


class StyleTransferDataset(Dataset):
    def __init__(self, root, style_img, transforms=None, limit=None):
        self.style_img = Image.open(style_img).convert('RGB')
        self.transforms = transforms
        self.paths = glob(os.path.join(root, '*'))
        if limit:
            self.paths = self.paths[:limit]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        content_image = Image.open(self.paths[idx]).convert('RGB')
        style_image = self.style_img

        if self.transforms:
            # Currently random transforms do not get applied in the same way to both images
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)

        return content_image, style_image

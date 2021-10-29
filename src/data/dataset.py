import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class TrashDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file, sep=",")
        self.annotations["filename"] = (
            self.annotations["INPUT:image"].dropna().apply(lambda x: x.split("/")[-1])
        )

        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return self.annotations.shape[0]

    @staticmethod
    def region_to_mask(region, image):
        _, height, width = image.shape
        polygon = []
        for poly in json.loads("[" + region.replace("\\", "") + "]"):
            for pair in poly["points"]:
                polygon.append((pair["top"] * height, pair["left"] * width))
        img = Image.new("L", image.shape[1:], 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img).T
        # mask = mask.reshape((3, image.shape[1], image))
        return mask

    def __getitem__(self, idx):
        img_path = self.img_dir / self.annotations["filename"][idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        mask = self.region_to_mask(
            self.annotations["OUTPUT:path"][idx],
            image,
        )
        return image, mask

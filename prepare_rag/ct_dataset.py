import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CTDataset(Dataset):
    def __init__(self, annotation_file, root_dir):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.data = []
        for key in self.annotations.keys():
            self.data += self.annotations[key]
        self.nii_to_tensor = self.nii_img_to_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx]['image_path'][0])
        image = self.nii_to_tensor(img_path)
        sample = {'image': image, 'meta': self.data[idx]}
        return sample

    def nii_img_to_tensor(self, path):
        try:
            img_data = np.load(path)['arr_0']
        except:
            img_data = np.load(path)["data"]
        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = img_data * 1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data + 400) / 600)).astype(np.float32)

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480, 480, 240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (
            pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor


def create_dataloader(annotation_file, data_root, num_workers, batch_size=4):
    dataset = CTDataset(annotation_file, data_root)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    return dataloader

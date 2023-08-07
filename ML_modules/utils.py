import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class PointnetGPD_Dataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, transform = None) -> None:
        self.root_dir = root_dir
        self.annotations = pd.read_csv(f'{root_dir}/{csv_file}')
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx, 0]
        data_path = os.path.join(self.root_dir, item)
        label = self.annotations.iloc[idx, 1]
        data_pts = np.load(f'{data_path}')

        if self.transform is not None:
            data_pts = self.transform(data_pts)
        data_pts = torch.permute(data_pts, (1, 0)) # size: (500, 3) --> (3, 500)
        label = torch.FloatTensor([label])

        return (data_pts, label)

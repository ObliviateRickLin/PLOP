import os
import torch.utils.data as torch_data
from typing import Dict

class TrajectoryDataset(torch_data.Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

    def load_image_data(self, idx: int) -> Dict:
        raise NotImplementedError()

    def load_bev_data(self, idx: int) -> Dict:
        raise NotImplementedError()

    def load_trajectory_data(self, idx: int) -> Dict:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Dict:
        image_data = self.load_image_data(idx)
        bev_data = self.load_bev_data(idx)
        trajectory_data = self.load_trajectory_data(idx)
        return {'image_data': image_data, 'bev_data': bev_data, 'trajectory_data': trajectory_data}

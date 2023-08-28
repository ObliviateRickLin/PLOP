import os
import torch.utils.data as torch_data

from typing import Dict
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper

from ..interface import TrajectoryDataset
from .utils import preprocess_camera_image, generate_bev


class NuScenesTrajectoryDataset(TrajectoryDataset):
    """
    NuScenesTrajectoryDataset is a subclass of TrajectoryDataset, designed to work with the nuScenes dataset.
    """

    def __init__(self, data_dir: str, version: str = 'v1.0-mini'):
        """
        Initialize the dataset.

        Parameters:
            data_dir (str): The directory where the nuScenes data is stored.
            version (str): The version of the nuScenes dataset to use.
        """
        super().__init__(data_dir)
        self.nusc = NuScenes(version=version, dataroot=data_dir, verbose=True)
        self.helper = PredictHelper(self.nusc)
        self.sample_tokens = get_prediction_challenge_split(split, dataroot=data_dir, version=version)

    def load_image_data(self, idx: int) -> Dict:
        """
        Load image data for a given index.

        Parameters:
            idx (int): The index of the sample to load.

        Returns:
            dict: A dictionary containing the preprocessed image data.
        """
        instance_token, sample_token = self.sample_tokens[idx].split('_')
        sample = self.nusc.get('sample', sample_token)
        cam_front_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_front_image_path = os.path.join(self.data_dir, cam_front_data['filename'])
        input_batch = preprocess_camera_image(cam_front_image_path)
        return {'image_data': input_batch}

    def load_bev_data(self, idx: int) -> Dict:
        """
        Load Bird's Eye View (BEV) data for a given index.

        Parameters:
            idx (int): The index of the sample to load.

        Returns:
            dict: A dictionary containing the BEV data.
        """
        instance_token, sample_token = self.sample_tokens[idx].split('_')
        bev_output = generate_bev(self.nusc, sample_token)
        return {'bev_data': bev_output}

    def load_trajectory_data(self, idx: int) -> Dict:
        """
        Load trajectory data for a given index.

        Parameters:
            idx (int): The index of the sample to load.

        Returns:
            dict: A dictionary containing the ego and agent trajectories.
        """
        instance_token, sample_token = self.sample_tokens[idx].split('_')
        ego_trajectory = get_ego_trajectory(self.nusc, sample_token)
        agent_trajectory = get_agent_trajectory(self.helper, instance_token, sample_token)
        return {'ego_trajectory': ego_trajectory, 'agent_trajectory': agent_trajectory}

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.

        Parameters:
            idx (int): The index of the sample to get.

        Returns:
            dict: A dictionary containing image, BEV, and trajectory data.
        """
        image_data_dict = self.load_image_data(idx)
        bev_data_dict = self.load_bev_data(idx)
        trajectory_data_dict = self.load_trajectory_data(idx)
        
        # Combine the three dictionaries into one
        combined_dict = {**image_data_dict, **bev_data_dict, **trajectory_data_dict}
        
        return combined_dict

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of sample tokens in the specific split.
        """
        return len(self.sample_tokens)






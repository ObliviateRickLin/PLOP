# 文件路径：datasets/utils.py

from PIL import Image
from torchvision import transforms
from nuscenes.prediction.helper import PredictHelper
from nuscenes.nuscenes import NuScenes
from typing import Dict, List


def get_ego_trajectory(nusc: NuScenes, sample_token: str, seconds: float = 2.0, sample_rate: float = 2.0) -> np.ndarray:
    ego_trajectory = []
    num_samples = int(sample_rate * seconds)
    current_sample_token = sample_token

    for _ in range(num_samples):
        sample = nusc.get('sample', current_sample_token)
        ego_pose_token = nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token']
        ego_pose = nusc.get('ego_pose', ego_pose_token)
        # 只取x, y坐标
        ego_trajectory.append([ego_pose['translation'][0], ego_pose['translation'][1]])

        if not sample['next']:
            break
        current_sample_token = sample['next']
    
    # 转换为NumPy数组
    ego_trajectory_np = np.array(ego_trajectory)
    return ego_trajectory_np

# Function to get single agent trajectory
def get_agent_trajectory(helper: PredictHelper, instance_token: str, sample_token: str, seconds: float = 2.0):
    return helper.get_past_for_agent(instance_token, sample_token, seconds=seconds, in_agent_frame=False)

def preprocess_camera_image(image_path: str, target_size=(320, 640)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


import numpy as np
from nuscenes.nuscenes import NuScenes

# BEV参数
BEV_WIDTH = 121
BEV_HEIGHT = 21
BEV_DEPTH = 5

from pyquaternion import Quaternion
import numpy as np

# Convert lidar points from lidar coordinate system to ego vehicle coordinate system.
def lidar_to_ego(lidar_points, calibrated_sensor_token):
    calibrated_sensor = nusc.get('calibrated_sensor', calibrated_sensor_token)
    rotation_matrix = Quaternion(calibrated_sensor['rotation']).rotation_matrix
    translation = np.array(calibrated_sensor['translation'])

    # Rotate and translate lidar points
    lidar_points = np.dot(lidar_points, rotation_matrix.T) + translation
    return lidar_points

# Convert a point in global coordinates to ego vehicle coordinate system.
def global_to_ego(global_point, ego_pose):
    """
    Convert a point in global coordinates to ego vehicle coordinate system.
    """
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    # Convert global point to ego coordinates
    local_point = ego_rotation.inverse.rotate(global_point - ego_translation)
    return local_point[:2]  # Return only x and y


def extract_lidar_info(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_filepath = os.path.join(nusc_data_path, lidar_data['filename'])
    lidar_points = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 5)[:, :3]  # x, y, z

    # Convert lidar points from lidar coordinate system to ego vehicle coordinate system
    lidar_points = lidar_to_ego(lidar_points, lidar_data['calibrated_sensor_token'])

    # Filter out points based on z-axis
    lidar_points = lidar_points[lidar_points[:, 2] > 0.3]

    # Project to BEV
    # 在Lidar信息部分
    for point in lidar_points:
        x, y = point[:2]
        if -60.5 <= x <= 60.5 and -10.5 <= y <= 10.5:
            i = int((x + 60.5) // 1)
            j = int((y + 10.5) // 1)
            bev[j, i, 4] += 1  # Increment Lidar point count
            bev[j, i, 0] += x  # Sum up x for mean calculation later
            bev[j, i, 1] += y  # Sum up y for mean calculation later

    # After processing all Lidar points, compute mean x and y
    bev[:, :, 0] /= (bev[:, :, 4] + 1e-6)  # Mean x
    bev[:, :, 1] /= (bev[:, :, 4] + 1e-6)  # Mean y


    return bev

def extract_vehicle_info(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
    vehicles = [ann for ann in annotations if ann['category_name'] in ['vehicle.car', 'vehicle.truck', 'vehicle.bus']]

    # 在车辆信息部分
    for vehicle in vehicles:
      global_coords = np.array(vehicle['translation'])  # x, y, z
      x, y = global_to_ego(global_coords, ego_pose)
      
      if -60.5 <= x <= 60.5 and -10.5 <= y <= 10.5:
          i = int((x + 60.5) // 1)
          j = int((y + 10.5) // 1)
          bev[j, i, 0] = x  # x position in local coordinates
          bev[j, i, 1] = y  # y position in local coordinates
          
          # Add vehicle state (assuming dynamic for now)
          bev[j, i, 2] = 2  # 2 might represent dynamic state #(remain added)
          
          # Add vehicle class
          if vehicle['category_name'] == 'vehicle.car':
              bev[j, i, 3] = 1
          elif vehicle['category_name'] == 'vehicle.truck':
              bev[j, i, 3] = 2
          elif vehicle['category_name'] == 'vehicle.bus':
              bev[j, i, 3] = 3


    return bev



def generate_bev(nusc, sample_token):
    bev = extract_lidar_info(nusc, sample_token)
    bev = extract_vehicle_info(nusc, sample_token)
    return bev

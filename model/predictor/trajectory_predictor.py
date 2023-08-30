import torch
import torch.nn as nn

# Base Trajectory Predictor
class BaseTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim, K):
        super(BaseTrajectoryPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.output_layer = nn.Linear(64, K * 51)  # K * 51 for Gaussian Mixture Model

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x

# Ego Vehicle Trajectory Predictor
class EgoTrajectoryPredictor(BaseTrajectoryPredictor):
    def __init__(self, input_dim, K):
        super(EgoTrajectoryPredictor, self).__init__(input_dim, K)

# Neighbor Vehicles Trajectory Predictor
class NeighborTrajectoryPredictor(BaseTrajectoryPredictor):
    def __init__(self, input_dim, K):
        super(NeighborTrajectoryPredictor, self).__init__(input_dim, K)



from encoder.image_encoder import VGG16Encoder
from encoder.bev_encoder import BirdviewEncoder
from predictor.trajectory_predictor import EgoPastLSTM
from encoder.trajectory_encoder import EgoPastEncoder, NbrsPastEncoder  # Assuming this will be implemented
from predictor.auxiliary_segmentation import UNetDecoder  # Assuming this will be implemented
from utils.mixture_density_network import MixtureDensityNetwork  # Assuming this will be implemented
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = VGG16Encoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4, x5)
        return x, x6

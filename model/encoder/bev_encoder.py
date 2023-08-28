import torch.nn as nn
import torch.nn.functional as F

class BirdviewEncoder(nn.Module):
    def __init__(self):
        super(BirdviewEncoder, self).__init__()
        # Conv3D layer
        self.conv3d = nn.Conv3d(in_channels=5, out_channels=20, kernel_size=(20, 9, 9))
        # Conv2D layers
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # MaxPooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # AveragePooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=(6, 1))
        # Fully Connected layer
        self.fc = nn.Linear(in_features=512, out_features=512)

    def forward(self, x):
        x = self.conv3d(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

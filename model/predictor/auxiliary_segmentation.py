import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Define the UNetDecoder class based on the provided code
class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec_conv1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec_conv1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(64, 4, kernel_size=1) #four here need to be further adjusted based on the further requirements

    def forward(self, x1, x2, x3, x4, x5):
        x = self.upconv1(x5)
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.dec_conv1_1(x))
        x = F.relu(self.dec_conv1_2(x))
        
        x = self.upconv2(x)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.dec_conv2_1(x))
        x = F.relu(self.dec_conv2_2(x))
        
        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.dec_conv3_1(x))
        x = F.relu(self.dec_conv3_2(x))
        
        x = self.upconv4(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.dec_conv4_1(x))
        x = F.relu(self.dec_conv4_2(x))
        
        x = self.final_conv(x)
        
        return x

import torch, torch.nn as nn
import torch.nn.functional as F

class SimpleSegDecoder(nn.Module):
    def __init__(self, embed_dim=768, out_channels=2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, 256, 2, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.head = nn.Conv2d(64, out_channels, 1)
    def forward(self, feat_map):
        x = F.relu(self.up1(feat_map))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        x = self.head(x)
        return x

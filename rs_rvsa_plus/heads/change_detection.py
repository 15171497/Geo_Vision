
import torch, torch.nn as nn
import torch.nn.functional as F

class SiameseChangeDecoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, 256, 2, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.head = nn.Conv2d(64, 1, 1)
    def _ups(self, f):
        x = F.relu(self.up1(f))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        return x
    def forward(self, fA, fB):
        a = self._ups(fA)
        b = self._ups(fB)
        d = torch.abs(a-b)
        out = self.head(d)
        return out

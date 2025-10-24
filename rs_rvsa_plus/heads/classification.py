
import torch, torch.nn as nn

class ClsHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, feats):
        if feats.ndim==3: feats = feats[:,0]
        return self.fc(feats)

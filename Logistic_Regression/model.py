import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

class LogReg(nn.Module):
    def __init__(self):
        super(Loggression, self).__init__()
        self.classifier = nn.Linear(900, 1)
    def forward(self, x):
        x = x.view(-1, x.size(-1))
        logits = self.classifier(x).view(-1, 4)
        return logits

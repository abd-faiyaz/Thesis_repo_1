import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteCNN(nn.Module):
    """
    Shallow 1-D CNN
    """
    def __init__(self, embed_dim: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)

        self.conv3 = nn.Conv1d(32, 32, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = x.mean(dim=2)
        return self.fc(x)
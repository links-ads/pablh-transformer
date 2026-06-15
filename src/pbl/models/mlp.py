import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim_in=207, dim_out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, dim_out)
        )

    def forward(self, x):
        return self.net(x)

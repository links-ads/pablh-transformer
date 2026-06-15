from torch import nn
from torch import rand

class ResMlp(nn.Module):
    def __init__(self, dim_in=207, dim_out=1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(dim_in, 128),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.Linear(128, dim_out),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim, dim_out=None, path_dropout_rate=0., dropout_rate=0.2):
        if dim_out is None:
            dim_out = dim
        super().__init__()
        if dropout_rate > 0.0:
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Dropout(dropout_rate),
                nn.Linear(dim, dim_out),
                nn.LayerNorm(dim_out),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim_out),
                nn.LayerNorm(dim_out),
                nn.GELU(),
            )
        self.path_dropout = path_dropout_rate

    def forward(self, x):
        if self.training and self.path_dropout > 0.0:
            if rand(1).item() < self.path_dropout:
                return x  # skip block
        return x + self.block(x)

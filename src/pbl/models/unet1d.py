import torch
from torch import nn
from torch import rand

class Unet1d(nn.Module):
    def __init__(self, dim_in=207, input_features=None, dim_out=1):
        if input_features is not None:
            dim_in = input_features
        super().__init__()

        self.enc1 = EncoderBlock(dim_in, 64, downsample=False)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)

        self.bottleneck = ResidualBlock1d(256, 512)

        self.dec3 = DecoderBlock(512, 256, use_skip=True)
        self.dec2 = DecoderBlock(256, 128, use_skip=True)
        self.dec1 = DecoderBlock(128, 64, use_skip=True, upsample=False)

        self.final = nn.Conv1d(64, dim_out, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)       # (B, C, L)

        e1 = self.enc1(x)           # (B, 64,  L)
        e2 = self.enc2(e1)          # (B, 128, L/2)
        e3 = self.enc3(e2)          # (B, 256, L/4)

        b = self.bottleneck(e3)     # (B, 512, L/4)

        d3 = self.dec3(b,  e3)      # (B, 256, L/4)
        d2 = self.dec2(d3, e2)      # (B, 128, L/2)
        d1 = self.dec1(d2, e1)      # (B, 64,  L)

        out = self.final(d1)        # (B, 1, L)
        return out.transpose(1, 2)  # (B, L, 1)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, dropout_rate=0.3):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            ResidualBlock1d(in_channels, out_channels, dropout_rate=dropout_rate),
            ResidualBlock1d(out_channels, out_channels, dropout_rate=dropout_rate),
        )
        
        self.downsample = None
        if downsample:
            self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv_block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3, use_skip=True, upsample=True):
        super().__init__()
        
        self.use_skip = use_skip
        self.upsample_layer = None
        
        if upsample:
            self.upsample_layer = nn.ConvTranspose1d(
                in_channels, out_channels, 
                kernel_size=2, stride=2
            )
        else:
            self.upsample_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        conv_in_channels = out_channels * 2 if use_skip else out_channels
        
        self.conv_block = nn.Sequential(
            ResidualBlock1d(conv_in_channels, out_channels, dropout_rate=dropout_rate),
            ResidualBlock1d(out_channels, out_channels, dropout_rate=dropout_rate),
        )
    
    def forward(self, x, skip_connection=None):
        x = self.upsample_layer(x)
        
        if skip_connection is not None:
            if x.size(2) != skip_connection.size(2):
                diff = skip_connection.size(2) - x.size(2)
                if diff > 0:
                    x = torch.nn.functional.pad(x, (0, diff))
                else:
                    x = x[:, :, :skip_connection.size(2)]
            
            x = torch.cat([x, skip_connection], dim=1)
        
        x = self.conv_block(x)
        return x


class ResidualBlock1d(nn.Module):
    def __init__(self, channels, channels_out=None, path_dropout_rate=0.3, dropout_rate=0.3, kernel_size=3):
        if channels_out is None:
            channels_out = channels
        super().__init__()
        
        padding = kernel_size // 2
        
        if dropout_rate > 0.0:
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv1d(channels, channels_out, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels_out),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels_out, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels_out),
                nn.ReLU(),
            )
        
        self.path_dropout = path_dropout_rate
        
        self.projection = None
        if channels != channels_out:
            self.projection = nn.Conv1d(channels, channels_out, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.projection is not None:
            residual = self.projection(residual)
        
        if self.training and self.path_dropout > 0.0:
            if rand(1).item() < self.path_dropout:
                return residual
            
        return residual + self.block(x)
 
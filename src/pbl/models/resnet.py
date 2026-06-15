from torch import rand, cat, nn
import torch

    
class ResidualBlock(nn.Module):
    def __init__(self, dim, dim_out=None, path_dropout_rate=0.5, dropout_rate=0.5):
        if dim_out is None:
            dim_out = dim
        super().__init__()
        if dropout_rate > 0.0:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.GroupNorm(32, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(dim, dim_out, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.GroupNorm(32, dim_out),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.GroupNorm(32, dim),
                nn.ReLU(),
                nn.Conv2d(dim, dim_out, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.GroupNorm(32, dim_out),
                nn.ReLU(),
            )
        self.path_dropout = path_dropout_rate
        if dim != dim_out:
            self.skip_conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        if self.training and self.path_dropout > 0.0:
            if rand(1).item() < self.path_dropout:
                return x  # skip block
        x = self.block(x) + self.skip_conv(x)
        return x


class Resnet(nn.Module):
    def __init__(self, dim_in=3, dim_out=1):
        super().__init__()
        self.stem = nn.Sequential(
            # 1x1 conv
            nn.Conv2d(dim_in, 128, kernel_size=1),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.down_block1 = nn.Sequential(
            #downsample
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.down_block2 = nn.Sequential(
            #downsample
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        # self.down_block3 = nn.Sequential(
        #     #downsample
        #     nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),    
        #     ResidualBlock(1024),
        #     ResidualBlock(1024),
        #     ResidualBlock(1024),
        #     ResidualBlock(1024),
        # )
        # self.up_block3 = nn.Sequential(
        #     # upsample bilinear
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(1024, 512, kernel_size=1, padding_mode='reflect'),
        # )
        self.up_block2 = nn.Sequential(
            # upsample bilinear
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, kernel_size=1, padding_mode='reflect'),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
        )
        self.up_block1 = nn.Sequential(
            # upsample bilinear
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512+256, 384, kernel_size=1, padding_mode='reflect'),
            nn.GroupNorm(32, 384),
            nn.ReLU(),
        )
        # self.out = nn.Conv2d(384+128, dim_out, kernel_size=1, padding_mode='reflect')
        self.out_block = nn.Sequential(
            nn.Conv2d(384+128, 64, kernel_size=1, padding_mode='reflect'),
            nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, padding_mode='reflect')
        )
        


    def forward(self, x, nan_mask=None):
        # set to zero the nan values
        # if nan_mask is not None:
        #     # if nan mask is int type, convert to bool
        #     if nan_mask.dtype == torch.uint8:
        #         nan_mask = nan_mask.bool()
        #     x = x.masked_fill(nan_mask, 0.0)
        #set all nans of x to zero
        # compute non nan mean per channel
        # mean_per_channel = torch.nanmean(x, dim=[0,2,3])
        # # for each channel, set nans to mean of that channel
        # for c in range(x.shape[1]):
        #     x[:,c,:,:] = torch.nan_to_num(x[:,c,:,:], nan=mean_per_channel[c].item())
        # x = torch.where(
        #     torch.isnan(x),
        #     mean_per_channel[None, :, None, None],
        #     x
        # )
        # x = torch.nan_to_num(x, nan=0.0)
        # x = nan_impute(x)
        out_stem = self.stem(x)
        out_down1 = self.down_block1(out_stem)
        x = self.down_block2(out_down1)
        # x = self.down_block3(out_down2)
        # x = self.up_block3(x)
        # x = cat([x, out_down2], dim=1)
        x = self.up_block2(x)
        x = cat([x, out_down1], dim=1)
        x = self.up_block1(x)
        x = cat([x, out_stem], dim=1)
        x = self.out_block(x)
        return x

    
from matplotlib import patches
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pbl.models.tiling import overlapping_predictions_torch


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., path_dropout = 0.):
        super().__init__()
        self.path_dropout = path_dropout
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            if self.training:
                if self.path_dropout > 0.:
                    if torch.rand(1).item() < self.path_dropout:
                        continue
            x = attn(x) + x
            x = ff(x) + x
            x = self.norm(x)
        return x

class ViT(nn.Module):
    def __init__(
        self, 
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, depth, 
        heads, 
        mlp_dim, 
        channels = 3, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0.0,
        masking_ratio = 0.0,
        path_dropout = 0.,
    ):
        super().__init__()
        self.image_size = image_size
        self.masking_ratio = masking_ratio
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head, 
            mlp_dim=mlp_dim, 
            dropout=dropout, 
            path_dropout=path_dropout
        )
        # self.mlp_head = nn.Linear(dim, num_classes)
        
        self.missing_token = nn.Parameter(torch.randn(1, 1, dim))
        

    def forward(self, img):
        device = img.device

        nan_mask = torch.isnan(img)
        # set nan patches to zero
        img = torch.where(img.isnan(), torch.tensor(0.0, device=img.device, dtype=img.dtype), img)
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)
        tokens += self.pos_embedding.to(device, dtype=tokens.dtype)
        # collapse channel dimension of mask 
        nan_mask = nan_mask.all(dim=1)
        nan_mask = nan_mask.all(dim=0)
        nan_indices = (nan_mask).flatten().nonzero().squeeze(-1)
        rand_indices = torch.rand(num_patches, device = device).argsort(dim = -1)
        if self.training:
            # masking_ratio = self.masking_ratio
            # masking ratio is randomly sampled between 0 and self.masking_ratio
            masking_ratio = torch.rand(1).item() * self.masking_ratio
            nan_indices_ratio = nan_indices.shape[0] / num_patches
            # masking ration is the max between the sampled masking ratio and the ratio needed to mask all nan indices
            masking_ratio = max(masking_ratio, nan_indices_ratio)
        else:
            masking_ratio = 0.0
        num_masked = int(masking_ratio * num_patches)
        # remove nan indices from rand_indices
        rand_indices = rand_indices[~torch.isin(rand_indices, nan_indices)]
        masked_indices, unmasked_indices = rand_indices[:num_masked], rand_indices[num_masked:]
        # concat nan_indices to masked_indices
        masked_indices = torch.cat([masked_indices, nan_indices], dim=0)
        # convert masked_indices and unmasked_indices to batch size
        masked_indices = masked_indices.unsqueeze(0).repeat(batch, 1)
        unmasked_indices = unmasked_indices.unsqueeze(0).repeat(batch, 1)
        num_masked = masked_indices.shape[1]
        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]
        encoded_tokens = self.transformer(tokens)
        # put back in the missing tokens at the masked indices
        decoder_tokens = torch.zeros(batch, num_patches, encoded_tokens.shape[-1], device=device)
        decoder_tokens[batch_range, unmasked_indices] = encoded_tokens        
        missing_tokens = repeat(self.missing_token, '1 1 d -> b n d', b = batch, n = num_masked)
        decoder_tokens[batch_range, masked_indices] = missing_tokens
        return decoder_tokens
    
    
class MAE(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        channels,
        encoder_depth,
        encoder_heads,
        encoder_dim_head,
        encoder_dropout,
        encoder_path_dropout,
        encoder_emb_dropout,
        num_classes,
        
        masking_ratio = 0.5,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        decoder_path_dropout = 0.,
        decoder_dropout = 0.
    ):
        super().__init__()
        
        encoder_mlp_dim = dim * 4
        
        patch_size = 1  # temporarily fixed to 1x1 patches
        self.image_size = image_size
        num_patches = image_size * image_size
        patch_dim = channels * patch_size * patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        
        self.ir_encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=encoder_depth,
            heads=encoder_heads,
            mlp_dim=encoder_mlp_dim,
            channels=20+19,
            dim_head=encoder_dim_head,
            dropout=encoder_dropout,
            emb_dropout=encoder_emb_dropout,
            masking_ratio=masking_ratio,
            path_dropout=encoder_path_dropout,
        )
        
        self.mw_encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=encoder_depth,
            heads=encoder_heads,
            mlp_dim=encoder_mlp_dim,
            channels=168+19,
            dim_head=encoder_dim_head,
            dropout=encoder_dropout,
            emb_dropout=encoder_emb_dropout,
            masking_ratio=masking_ratio,
            path_dropout=encoder_path_dropout,
        )
        
        encoder_dim = dim 
        
        # pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        decoder_dim = dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim = decoder_dim, 
            depth = decoder_depth, 
            heads = decoder_heads, 
            dim_head = decoder_dim_head, 
            mlp_dim = decoder_dim * 4, 
            dropout = decoder_dropout,
            path_dropout = decoder_path_dropout,
        )
        # self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, 1)

    def forward_(self, img):
        
        # get first 20 channels, those are ir
        ir = img[:, :20, :, :]
        mw = img[:, 20:168+20, :, :]
        aux = img[:, 168+20:, :, :]
        
        ir = torch.cat([ir, aux], dim=1)
        mw = torch.cat([mw, aux], dim=1)
        
        # encode both modalities
        ir_encoded = self.ir_encoder(ir)
        mw_encoded = self.mw_encoder(mw)
        
        # naive approach: sum the encoded tokens
        encoded_tokens = ir_encoded + mw_encoded

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = encoded_tokens + self.pos_embedding
        
        
        # # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)
        # # concat the masked tokens to the decoder tokens and attend with decoder
        # decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        # decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        # decoder_tokens[batch_range, masked_indices] = mask_tokens
        
        
        
        decoded_tokens = self.decoder(decoder_tokens)
        # splice out the mask tokens and project to pixel values
        pred_pixel_values = self.to_pixels(decoded_tokens)
        # reshape to image
        pred_pixel_values = rearrange(pred_pixel_values, 'b (h w) c -> b c h w', h=img.shape[2], w=img.shape[3])
        return pred_pixel_values
    
    def forward(self, img):
        if self.training:
            return self.forward_(img)
        else:
            if img.shape[0] != 1:
                raise ValueError("batch size must be 1 in inference mode")
            out = overlapping_predictions_torch(
                input = img,
                prediction_fn=self.forward_,
                size=self.image_size,
                stride=max(1, self.image_size // 2),
                batch_size=32,
            )
            return out
    
class Transformer2d(nn.Module):
    def __init__(self, dim_in=207, dim_out=1, image_size=16):
        super().__init__()
        self.mae = MAE(
            channels=dim_in,
            image_size=image_size,
            num_classes=dim_out,
            dim=384,
            encoder_depth=2,
            encoder_heads=16,
            encoder_dim_head=24,
            encoder_dropout=0.,
            encoder_emb_dropout=0.,
            encoder_path_dropout=0.,
            masking_ratio=1,
            decoder_depth=2,
            decoder_heads=16,
            decoder_dim_head=24,
            decoder_dropout=0.,
            decoder_path_dropout=0.0,
        )

    def forward(self, img):
        return self.mae(img)
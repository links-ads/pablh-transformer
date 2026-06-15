from torch import nn

class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_features=207,
        dim_out=1,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_features, d_model)

        # Create multiple encoder blocks
        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, dim_out)

    def forward(self, x):
        # x: (B, 30, 207)
        x = self.input_proj(x)  # (B, 30, d_model)

        # Pass through each encoder block
        for encoder in self.encoder_blocks:
            x = encoder(x)

        out = self.output_proj(x)  # (B, 30, dim_out)
        return out

 
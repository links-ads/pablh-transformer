import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import optim
from pbl.models.unet1d import Unet1d
from pbl.models.transformer import TransformerPredictor
import numpy as np

model_dict_1d = {
    "unet1d": Unet1d,
    "transformer1d": TransformerPredictor,
}


class Module1d(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        model_cls = model_dict_1d[config.name]
        if config.name == "transformer1d":
            self.model = model_cls(input_features=207, dim_out=1)
        else:
            self.model = model_cls(dim_in=207, dim_out=1)
        if config.loss == "mse":
            self.loss = F.l1_loss
        else:
            self.loss = F.l1_loss
        self.target_list = []
        self.predicted_list = []

    def forward(self, x):
        return self.model(x)

    def _target_col(self, target):
        # target: (B, L, 4) — use first column (BLH)
        return target[:, :, 0]

    def training_step(self, batch, batch_idx):
        input = batch["input"].float()
        target = self._target_col(batch["target"])
        pred = self.model(input).squeeze(-1)  # (B, L)
        loss = self.loss(pred, target)
        self.log("loss_train", loss, prog_bar=True, batch_size=input.size(0))
        self.log("loss_train_avg", loss, on_step=False, on_epoch=True, batch_size=input.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch["input"].float()
        target = self._target_col(batch["target"])
        scale_factor = batch["scale_factor"]
        pred = self.model(input).squeeze(-1)  # (B, L)
        loss = self.loss(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=input.size(0))
        sf = scale_factor.unsqueeze(-1)  # (B, 1)
        pred_phys = pred * sf
        target_phys = target * sf
        mae = F.l1_loss(pred_phys, target_phys)
        self.log("val_mae", mae, on_step=False, on_epoch=True, batch_size=input.size(0))
        bias = (pred_phys - target_phys).mean()
        self.log("val_bias", bias, on_step=False, on_epoch=True, batch_size=input.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        input = batch["input"].float()
        target = self._target_col(batch["target"])
        scale_factor = batch["scale_factor"]
        eval_mask = batch.get("eval_mask", None)
        pred = self.model(input).squeeze(-1)  # (B, L)
        sf = scale_factor.unsqueeze(-1)  # (B, 1)
        pred_phys = pred * sf
        target_phys = target * sf
        # filter samples by eval_mask (per-sample bool)
        if eval_mask is not None:
            mask = eval_mask.bool()
            pred_phys = pred_phys[mask]
            target_phys = target_phys[mask]
        if pred_phys.numel() == 0:
            return torch.tensor(0.0)
        pred_flat = pred_phys.reshape(-1)
        target_flat = target_phys.reshape(-1)
        n = pred_flat.size(0)
        mae = F.l1_loss(pred_flat, target_flat)
        rmse = torch.sqrt(F.mse_loss(pred_flat, target_flat))
        mape = torch.mean(torch.abs((target_flat - pred_flat) / (target_flat.abs() + 1e-8)))
        bias = (pred_flat - target_flat).mean()
        self.log("test_mae", mae, on_step=False, on_epoch=True, batch_size=n)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, batch_size=n)
        self.log("test_mape", mape, on_step=False, on_epoch=True, batch_size=n)
        self.log("test_bias", bias, on_step=False, on_epoch=True, batch_size=n)
        self.predicted_list.append(pred_flat.detach())
        self.target_list.append(target_flat.detach())
        return torch.tensor(0.0)

    def on_test_epoch_end(self):
        if self.predicted_list:
            all_pred = torch.cat(self.predicted_list)
            all_target = torch.cat(self.target_list)
            pearson = torch.corrcoef(torch.stack([all_pred, all_target]))[0, 1]
            self.log("test_pearson", pearson)
            self.predicted_list.clear()
            self.target_list.clear()

    def configure_optimizers(self):
        lr = self.config.min_lr
        if self.config.lr_schedule == 'lin':
            return {"optimizer": optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)}
        elif self.config.lr_schedule == 'rop':
            optimizer = optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.config.rop_rate, patience=self.config.rop_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mae", "frequency": 1},
            }
        elif self.config.lr_schedule == 'cos':
            optimizer = optim.AdamW(self.parameters(), lr=self.config.max_lr, betas=(0.9, 0.999), weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, eta_min=self.config.min_lr,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.lr_schedule == 'exp':
            optimizer = optim.AdamW(self.parameters(), lr=self.config.max_lr, betas=(0.9, 0.999), weight_decay=0.01)
            return {"optimizer": optimizer}
        else:
            raise ValueError(f"Invalid lr schedule: {self.config.lr_schedule}")

    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        if self.config.lr_schedule == 'exp':
            min_exp = np.log2(self.config.min_lr)
            max_exp = np.log2(self.config.max_lr)
            exp = min_exp + (max_exp - min_exp) * (1 - self.current_epoch / self.config.max_epochs)
            self.trainer.optimizers[0].param_groups[0]['lr'] = 2 ** exp

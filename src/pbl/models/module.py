import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import optim
from pbl.models.transformer import TransformerPredictor as Transformer
from pbl.models.mlp import MLP
from pbl.models.resmlp import ResMlp
from pbl.models.transformer2d import Transformer2d
from pbl.models.resnet import Resnet
from pbl.models.unet1d import Unet1d
import numpy as np

model_dict = {
    "transformer": Transformer,
    "mlp": MLP,
    "resmlp": ResMlp,
    "resnet": Resnet,
    "unet1d": Unet1d,
    "mae_dual": Transformer2d,
}

class Module(pl.LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.lr = config.min_lr
        self.val_step_outputs = [] # needed for val_epoch_end to work
        self.batch_fetched = False
        self.batch = None
        self.pred = None
        self.config = config
        self.batch_size = config.batch_size
        self.test_plot_count = 0
        self.val_in_cpu = False
        self.dim_out = 1
        image_size = getattr(config, 'train_size', 8)
        models_without_image_size = {'resnet', 'unet1d'}
        if config.name in models_without_image_size:
            self.model = model_dict[config.name](dim_in=207, dim_out=1)
        else:
            self.model = model_dict[config.name](dim_in=207, dim_out=1, image_size=image_size)
        self.threads = []
        if config.loss == "mse":
            self.loss = F.l1_loss
        elif config.loss == "mse_scaled":
            self.loss = MseScaledLoss()
        self.mae = F.l1_loss
        self.rmse = Rmse()
        self.mape = Mape()
        self.target_list = []
        self.predicted_list = []
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx):
        input = batch["input"]
        target = batch["target"]
        input_mask = batch.get("input_mask", None)
        target_mask = batch.get("target_mask", None).bool()
        # check if all target_mask are False
        if not torch.any(target_mask):
            # return zero loss
            loss = torch.tensor(0.0, device=input.device, requires_grad=True)
            self.log("loss_train", loss, prog_bar=True, batch_size=input.size(0))
            self.log("loss_train_avg", loss, on_step=False, on_epoch=True, batch_size=input.size(0))
            return None
        pred = self.model(input)
        pred = pred[:, 0][target_mask]
        target = target[target_mask]
        loss = self.loss(pred, target)
        self.log("loss_train", loss, prog_bar=True, batch_size=input.size(0))
        self.log("loss_train_avg", loss, on_step=False, on_epoch=True, batch_size=input.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        input = batch["input"]
        target = batch["target"]
        input_mask = batch.get("input_mask", None)
        target_mask = batch.get("target_mask", None)
        pred = self.model(input)
        pred_ = pred[:, 0][target_mask]
        target_ = target[target_mask]
        loss = self.loss(pred_, target_)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=input.size(0))
        scale_factor = batch["scale_factor"]
        pred_ = pred_ * scale_factor
        target_ = target_ * scale_factor
        mae = self.mae(pred_, target_)
        self.log("val_mae", mae, on_step=False, on_epoch=True, batch_size=input.size(0))
        mape = torch.mean(torch.abs((target_ - pred_) / (target_ + 1e-8)))
        self.log("val_mape", mape, on_step=False, on_epoch=True, batch_size=input.size(0))
        pearson = torch.corrcoef(torch.stack([pred_, target_]))[0, 1]
        self.log("val_pearson", pearson, on_step=False, on_epoch=True, batch_size=input.size(0))
        bias = torch.mean(pred_ - target_)
        self.log("val_bias", bias, on_step=False, on_epoch=True, batch_size=input.size(0))
        # compute valid mae, which is mae computed only on points where input and target are both valid
        # used only to compare with other models that do not use input mask
        # collapse input_mask channel dimension by any
        input_mask_ = input_mask.bool()
        input_mask_ = input_mask_.all(dim=1).squeeze(1)
        pred = pred * scale_factor
        target = target * scale_factor
        valid_mae = F.l1_loss(pred.squeeze(1), target, reduction='none')
        valid_mae = valid_mae * input_mask_
        valid_mae = valid_mae * target_mask
        # total mask is the number of valid points
        total_mask = input_mask_.float() * target_mask.float()
        if total_mask.sum() == 0:
            valid_mae = torch.tensor(0.0, device=valid_mae.device)
        else:
            valid_mae = valid_mae.sum() / total_mask.sum()
        self.log("val_mae_valid", valid_mae, on_step=False, on_epoch=True, batch_size=input.size(0))
        return loss

    def test_step(self, batch: dict, batch_idx):
        input = batch["input"]
        target = batch["target"]
        target_mask = batch.get("target_mask", None)
        eval_spatial_mask = batch.get("eval_spatial_mask", None)
        if eval_spatial_mask is not None:
            target_mask = target_mask & eval_spatial_mask
        pred = self(input)
        scale_factor = batch["scale_factor"]
        pred = pred * scale_factor
        target = target * scale_factor
        input_mask = batch.get("input_mask", None)
        input_mask_ = input_mask.bool().all(dim=1).squeeze(1)
        combined_mask = target_mask & input_mask_
        pred_ = pred[:, 0][combined_mask]
        target_ = target[combined_mask]
        if pred_.numel() > 1:
            n = pred_.size(0)
            mae = self.mae(pred_, target_)
            rmse = torch.sqrt(F.mse_loss(pred_, target_, reduction='none').mean())
            mape = torch.mean(torch.abs((target_ - pred_) / (target_ + 1e-8)))
            bias = torch.mean(pred_ - target_)
            self.log("test_mae", mae, on_step=False, on_epoch=True, batch_size=n)
            self.log("test_rmse", rmse, on_step=False, on_epoch=True, batch_size=n)
            self.log("test_mape", mape, on_step=False, on_epoch=True, batch_size=n)
            self.log("test_bias", bias, on_step=False, on_epoch=True, batch_size=n)
            self.predicted_list.append(pred_.detach())
            self.target_list.append(target_.detach())
        # test mae valid and rmse
        valid_mae = F.l1_loss(pred.squeeze(1), target, reduction='none')
        valid_mae = valid_mae * input_mask_
        valid_mae = valid_mae * target_mask
        total_mask = input_mask_.float() * target_mask.float()
        if total_mask.sum() == 0:
            valid_mae = torch.tensor(0.0, device=valid_mae.device)
        else:
            valid_mae = valid_mae.sum() / total_mask.sum()
        self.log("test_mae_valid", valid_mae, on_step=False, on_epoch=True, batch_size=input.size(0))
        valid_rmse = F.mse_loss(pred.squeeze(1), target, reduction='none')
        valid_rmse = valid_rmse * input_mask_
        valid_rmse = valid_rmse * target_mask
        if total_mask.sum() == 0:
            valid_rmse = torch.tensor(0.0, device=valid_rmse.device)
        else:
            valid_rmse = torch.sqrt(valid_rmse.sum() / total_mask.sum())
        self.log("test_rmse_valid", valid_rmse, on_step=False, on_epoch=True, batch_size=input.size(0))
        return torch.tensor(0.0)
    
    def on_test_epoch_end(self):
        if self.predicted_list:
            all_pred = torch.cat(self.predicted_list)
            all_target = torch.cat(self.target_list)
            pearson = torch.corrcoef(torch.stack([all_pred, all_target]))[0, 1]
            self.log("test_pearson", pearson)
            self.predicted_list.clear()
            self.target_list.clear()
        # # create scatter plot of target vs predicted
        # import matplotlib.pyplot as plt
        # import numpy as np
        # target = np.concatenate(self.target_list, axis=0)[:,0]
        # predicted = np.concatenate(self.predicted_list, axis=0)
        # plt.figure(figsize=(10, 10))
        # plt.scatter(target, predicted, s=1)
        # plt.xlabel("Target")
        # plt.ylabel("Predicted")
        # plt.title("Target vs Predicted")
        # plt.xlim(0, 4000)
        # plt.ylim(0, 4000)
        # plt.plot([0, 4000], [0, 4000], color='red', linestyle='--')
        # plt.savefig(f"target_vs_predicted_{self.test_plot_count}.png")   
        
        # import matplotlib.pyplot as plt
        # import numpy as np
        # from matplotlib.colors import LinearSegmentedColormap
        # target = np.concatenate(self.target_list, axis=0)[:,0]
        # predicted = np.concatenate(self.predicted_list, axis=0)[:,0]
        # colors = ["#b3d7ff", "#294d7f", "#287593", "#759387", "#bfa96d", "#b7925e", "#b37953", "#a64c4c", "#9e214b", "#460f26", "#000000",]
        # # Create the custom colormap
        # cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        # # Compute point density
        # xy = np.vstack([target, predicted])
        # z = np.histogram2d(target, predicted, bins=100)[0]
        # # Map each point to its density
        # ix = np.clip(((target - target.min()) / (target.max() - target.min()) * 99).astype(int), 0, 99)
        # iy = np.clip(((predicted - predicted.min()) / (predicted.max() - predicted.min()) * 99).astype(int), 0, 99)
        # density = z[ix, iy]

        # plt.figure(figsize=(9.5, 7.5))
        # plt.scatter(target, predicted, c=density, cmap=cmap, s=1)
        # plt.xlabel("Target")
        # plt.ylabel("Predicted")
        # plt.xlim(0, 4000)
        # plt.ylim(0, 4000)
        # plt.plot([0, 4000], [0, 4000], color='red', linestyle='--')
        # plt.colorbar(label='Density')
        # plt.savefig(f"target_vs_predicted2_{self.test_plot_count}.png", dpi=450)
        
    def configure_optimizers(self):
        lr = self.config.min_lr
        if self.config.lr_schedule == 'lin':
            return {
                "optimizer": optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
            }
        elif self.config.lr_schedule == 'exp':
            optimizer = optim.AdamW(self.parameters(), lr=self.config.max_lr, betas=(0.9, 0.999), weight_decay=0.01)
            return {
                "optimizer": optimizer,
            }

        elif self.config.lr_schedule == 'oc':
            raise ValueError("One cycle lr schedule not implemented")
        elif self.config.lr_schedule == 'rop':
            # reduce on plateau
            optimizer = optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.config.rop_rate,
                patience=self.config.rop_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mae",
                    "frequency": 1,
                },
            }
        elif self.config.lr_schedule == 'cos':
            optimizer = optim.AdamW(self.parameters(), lr=self.config.max_lr, betas=(0.9, 0.999), weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,
                eta_min=self.config.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        else:
            raise ValueError("Invalid lr schedule")
        
    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        if self.config.lr_schedule == 'exp':
            # adjust lr
            min_lr = self.config.min_lr
            max_lr = self.config.max_lr
            max_epochs = self.config.max_epochs
            # same but for the exponent 
            min_exp = np.log2(min_lr)
            max_exp = np.log2(max_lr)
            exp = min_exp + (max_exp - min_exp) * (1 - self.current_epoch / max_epochs)
            lr = 2**exp
            self.trainer.optimizers[0].param_groups[0]['lr'] = lr

class Module0d(pl.LightningModule):
    def __init__(
        self,
        config: dict,
    ):
        super().__init__()
        self.lr = config.min_lr
        self.config = config
        self.batch_size = config.batch_size
        self.test_plot_count = 0
        self.model = model_dict[config.name](dim_in=207, dim_out=1)
        if config.loss == "mse":
            self.loss = F.mse_loss
        elif config.loss == "mse_scaled":
            self.loss = MseScaledLoss()
        self.mae = F.l1_loss
        self.rmse = Rmse()
        self.target_list = []
        self.predicted_list = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx):
        input = batch["input"]
        target = batch["target"]
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        pred = self.model(input)
        loss = self.loss(pred, target)
        self.log("loss_train", loss, prog_bar=True, batch_size=input.size(0))
        self.log("loss_train_avg", loss, on_step=False, on_epoch=True, batch_size=input.size(0))
        self.log("lr_step", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input = batch["input"]
        target = batch["target"]
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        pred = self.model(input)
        loss = self.loss(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=input.size(0))
        scale_factor = batch["scale_factor"]
        pred = pred * scale_factor[0]
        target = target * scale_factor[0]
        mae = self.mae(pred, target)
        self.log("val_mae", mae, on_step=False, on_epoch=True, batch_size=input.size(0))
        return loss

    def test_step(self, batch: dict, batch_idx):
        input = batch["input"]
        target = batch["target"]
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        pred = self.model(input)
        scale_factor = batch["scale_factor"]
        pred = pred * scale_factor[0]
        target = target * scale_factor[0]
        eval_mask = batch.get("eval_mask", None)
        if eval_mask is not None:
            pred = pred[eval_mask]
            target = target[eval_mask]
        if pred.numel() == 0:
            return torch.tensor(0.0)
        n = pred.size(0)
        mae = self.mae(pred, target)
        self.log("test_mae", mae, on_step=False, on_epoch=True, batch_size=n)
        rmse = torch.sqrt(F.mse_loss(pred, target, reduction='none').mean())
        self.log("test_rmse", rmse, on_step=False, on_epoch=True, batch_size=n)
        mape = torch.mean(torch.abs((target - pred) / (target + 1e-8)))
        self.log("test_mape", mape, on_step=False, on_epoch=True, batch_size=n)
        bias = torch.mean(pred - target)
        self.log("test_bias", bias, on_step=False, on_epoch=True, batch_size=n)
        self.predicted_list.append(pred.reshape(-1))
        self.target_list.append(target.reshape(-1))
        return torch.tensor(0.0)

    def on_test_epoch_end(self):
        all_pred = torch.cat(self.predicted_list)
        all_target = torch.cat(self.target_list)
        pearson = torch.corrcoef(torch.stack([all_pred, all_target]))[0, 1]
        self.log("test_pearson", pearson)
        self.predicted_list.clear()
        self.target_list.clear()

    def configure_optimizers(self):
        lr = self.config.min_lr
        if self.config.lr_schedule == 'lin':
            return {
                "optimizer": optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
            }
        elif self.config.lr_schedule == 'exp':
            optimizer = optim.AdamW(self.parameters(), lr=self.config.max_lr, betas=(0.9, 0.999), weight_decay=0.01)
            return {"optimizer": optimizer}
        elif self.config.lr_schedule == 'rop':
            optimizer = optim.AdamW(self.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.rop_rate,
                patience=self.config.rop_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mae", "frequency": 1},
            }
        elif self.config.lr_schedule == 'cos':
            optimizer = optim.AdamW(self.parameters(), lr=self.config.max_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,
                eta_min=0,
                T_mult=1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "interval": "step",
            }
        else:
            raise ValueError("Invalid lr schedule")

    def on_train_epoch_start(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        if self.config.lr_schedule == 'exp':
            min_lr = self.config.min_lr
            max_lr = self.config.max_lr
            max_epochs = self.config.max_epochs
            min_exp = np.log2(min_lr)
            max_exp = np.log2(max_lr)
            exp = min_exp + (max_exp - min_exp) * (1 - self.current_epoch / max_epochs)
            lr = 2**exp
            self.trainer.optimizers[0].param_groups[0]['lr'] = lr


def get_module(config):
    from pbl.models.module1d import Module1d
    dims = getattr(config, 'dims', 2)
    if dims == 0:
        return Module0d
    if dims == 1:
        return Module1d
    return Module


class MseScaledLoss(torch.nn.Module):
    def forward(self, pred, target):
        t = target[:, 0]
        p = pred[:, 0]        
        mse = F.mse_loss(p, t, reduction='none')
        max = target[:, 2]
        min = target[:, 1]
        dist = max - min
        dist = dist + 1e-3  # to avoid division by zero
        mse = mse/dist
        mse = mse.mean() / 4000
        return mse
    
class MseLoss(torch.nn.Module):
    def forward(self, pred, target):
        t = target[:, 0]
        p = pred[:, 0]        
        mse = F.mse_loss(p, t)
        return mse
    
class Mae(torch.nn.Module):
    def forward(self, pred, target):
        t = target[:, 0]
        p = pred[:, 0]
        return F.l1_loss(p, t)
    
class Rmse(torch.nn.Module):
    def forward(self, pred, target):
        t = target
        p = pred
        return torch.sqrt(F.mse_loss(p, t))
    
class Mape(torch.nn.Module):
    def forward(self, pred, target):
        t = target[:, 0]
        p = pred[:, 0]
        return torch.mean(torch.abs((t - p) / t))
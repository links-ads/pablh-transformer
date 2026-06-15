import os
import warnings
from datetime import datetime

import torch

warnings.filterwarnings("ignore", message=".*The 'val_dataloader' does not have many workers which may be a bottleneck.*")
warnings.filterwarnings("ignore", message=".*The 'test_dataloader' does not have many workers which may be a bottleneck.*")
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import hydra
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig
from pbl.data.datamodule import DataModule
from pbl.models.module import Module, get_module
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from rich import print as richprint
from lightning.pytorch import seed_everything

torch.set_float32_matmul_precision('medium')

seed_everything(9)


def train(cfg):
    run_id = cfg.run_id
    wandb_dir = os.path.join('logs', run_id)
    logger = WandbLogger(name=run_id, project='pbl', save_dir=wandb_dir)
    datamodule = DataModule(cfg, verbose=True, train_size=cfg.model.train_size)
    model = get_module(cfg.model)(cfg.model)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        mode='min',
        filename='best',
        verbose=True,
        dirpath=f'logs/{run_id}/checkpoints',
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.model.es_patience,
        mode='min',
        verbose=True,
    )
    precision = "bf16-mixed" if torch.cuda.is_available() else "32-true"
    trainer = Trainer(
        precision=precision,
        max_epochs=cfg.model.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        accumulate_grad_batches=cfg.model.accumulate_batches,
    )
    trainer.fit(model, datamodule=datamodule)
    checkpoint_path = f'logs/{run_id}/checkpoints/best.ckpt'
    model = get_module(cfg.model).load_from_checkpoint(checkpoint_path, config=cfg.model)
    trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)


def test(cfg):
    run_id = cfg.run_id
    run_id_to_test = str(cfg.command.ver_number)
    wandb_dir = os.path.join('logs', run_id_to_test)
    logger = WandbLogger(name=run_id, project='pbl', save_dir=wandb_dir)
    checkpoint_path = f"logs/{run_id_to_test}/checkpoints/best.ckpt"
    model = get_module(cfg.model).load_from_checkpoint(checkpoint_path, config=cfg.model)
    trainer = Trainer(logger=logger)
    datamodule = DataModule(cfg, verbose=True, train_size=cfg.model.train_size)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)


def lr_find(cfg):
    run_id = cfg.run_id
    logger = TensorBoardLogger('logs', name=run_id, version='lr_find')
    datamodule = DataModule(cfg, verbose=True, train_size=cfg.model.train_size)
    model = Module(cfg.model)
    trainer = Trainer(
        logger=logger,
        default_root_dir=f'logs/{run_id}',
        num_sanity_val_steps=0,
    )
    tuner = Tuner(trainer=trainer)
    lr_finder = tuner.lr_find(
        model,
        datamodule=datamodule,
        min_lr=cfg.command.min_lr,
        max_lr=cfg.command.max_lr,
        early_stop_threshold=cfg.command.es_threshold,
        num_training=cfg.command.num_training,
    )
    fig = lr_finder.plot(suggest=True)
    fig.savefig(f'logs/{run_id}/lr_finder.png')
    print(lr_finder.suggestion())


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    richprint(cfg)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    OmegaConf.set_struct(cfg, False)
    cfg['run_id'] = run_id
    if cfg.command.name == 'train':
        train(cfg)
    if cfg.command.name == 'test':
        test(cfg)
    if cfg.command.name == 'lr_find':
        lr_find(cfg)


if __name__ == "__main__":
    main()

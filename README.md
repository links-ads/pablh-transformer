# PABLH Estimation via Dual-Encoder Transformer

Code for the paper **"PABLH Estimation from Satellite Radiances via a Dual-Encoder Transformer"**.

The model estimates Planetary Atmospheric Boundary Layer Height (PABLH) from MetOp satellite radiances (IASI infrared + AMSU-A/MHS microwave) using a dual-encoder Vision Transformer architecture with dynamic token masking for robustness to cloud-induced sensor gaps.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/pablh-transformer.git
cd pablh-transformer
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset

The model is trained on the dataset introduced in the companion paper (Larosa et al., under review), available on HuggingFace at [links-ads/metop-era5-pablh](https://huggingface.co/datasets/links-ads/metop-era5-pablh).

Download it and place the `data/` folder (or symlink it) at:

```
data/planetary-boundary-layer-height/metop_era5_pblh_dataset/
```

Or pass the path explicitly via `dataset.path=...` at runtime (see below).

---

## Training

The default configuration in `conf/` corresponds to the best model reported in the paper. To train from scratch:

```bash
python -m pbl dataset.path=/path/to/metop_era5_pblh_dataset
```

This will train the Transformer-2D model with the hyperparameters from `conf/model/transformer2d.yaml`. Checkpoints and logs are saved under `logs/<run_id>/`.

To override any parameter:

```bash
python -m pbl dataset.path=/path/to/metop_era5_pblh_dataset model.batch_size=4 model.max_lr=1e-4
```

Training uses [Weights & Biases](https://wandb.ai) for logging. Set `WANDB_MODE=disabled` to disable it:

```bash
WANDB_MODE=disabled python -m pbl dataset.path=/path/to/metop_era5_pblh_dataset
```

---

## Inference

To run inference with a trained checkpoint:

```bash
python -m pbl command=test command.ver_number=<run_id> dataset.path=/path/to/metop_era5_pblh_dataset
```

where `<run_id>` is the timestamp string of the training run (e.g. `20260101_120000`), which corresponds to the folder `logs/<run_id>/checkpoints/best.ckpt`.

---

## Configuration

All configuration is managed via [Hydra](https://hydra.cc). The config files are in `conf/`:

| File | Description |
|------|-------------|
| `conf/model/transformer2d.yaml` | Model and training hyperparameters (best configuration) |
| `conf/dataset/iasi2d.yaml` | Dataset settings |
| `conf/command/train.yaml` | Training command |
| `conf/command/test.yaml` | Inference command |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{innocenti2026pablh,
  author    = {Lorenzo Innocenti and Luca Catalano and Claudio Rossi and
               Salvatore Larosa and Domenico Cimini and Paolo Garza},
  title     = {{PABLH} Estimation from Satellite Radiances via a Dual-Encoder Transformer},
  year      = {2026},
}
```

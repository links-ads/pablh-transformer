"""
XGBoost baseline on IASI1d data.
Each FOV position is treated as an independent sample (207 features → 1 BLH value).
After training, metrics are computed on: all, EU-only, clear-sky, cloudy-sky subsets.
"""
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
import time
import os

CACHE_DIR = "outputs/cache"
SCALE_FACTOR = 4000.0

EU_LON_MIN, EU_LON_MAX = -24.542225, 29.486706
EU_LAT_MIN, EU_LAT_MAX = 50.37499, 71.154709

PARAMS = {
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1,
    "colsample_bytree": 1,
    "eta": 0.3,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "device": "cuda",
}
NUM_BOOST_ROUND = 100
EARLY_STOPPING_ROUNDS = 10


def load_split(split):
    path = os.path.join(CACHE_DIR, f"iasi1d_{split}.npz")
    d = np.load(path, allow_pickle=True)
    N = d["samples"].shape[0]
    X = d["samples"].reshape(-1, 207).astype(np.float32)        # (N*30, 207)
    y = d["labels"].reshape(-1, 4)[:, 0].astype(np.float32)     # (N*30,)
    lat = np.vstack(d["raw_lat"]).reshape(-1).astype(np.float32)         # (N*30,)
    lon = np.vstack(d["raw_lon"]).reshape(-1).astype(np.float32)         # (N*30,)
    cf2 = np.vstack(d["raw_cloudflag2"]).reshape(-1).astype(np.float32)  # (N*30,)
    return X, y, lat, lon, cf2


def build_masks(lat, lon, cf2):
    eu = (
        (EU_LON_MIN <= lon) & (lon <= EU_LON_MAX) &
        (EU_LAT_MIN <= lat) & (lat <= EU_LAT_MAX)
    )
    clear = cf2 == 0
    cloudy = cf2 > 0
    return {"all": np.ones(len(lat), dtype=bool), "eu": eu, "clear": clear, "cloudy": cloudy}


def metrics(y_true, y_pred, sf=SCALE_FACTOR):
    y_true = y_true * sf
    y_pred = y_pred * sf
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    r, _ = pearsonr(y_true, y_pred)
    bias = (y_pred - y_true).mean()
    return mae, rmse, r, bias


def print_metrics(split, subset, y_true, y_pred, n):
    mae, rmse, r, bias = metrics(y_true, y_pred)
    print(f"  [{split} | {subset:6s}] n={n:>8d}  MAE={mae:.1f}m  RMSE={rmse:.1f}m  "
          f"Pearson={r:.4f}  Bias={bias:+.1f}m")


def main():
    print("Loading data...")
    X_train, y_train, lat_tr, lon_tr, cf2_tr = load_split("train")
    X_val,   y_val,   lat_va, lon_va, cf2_va = load_split("val")
    X_test,  y_test,  lat_te, lon_te, cf2_te = load_split("test")
    print(f"  train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    print("\nTraining...")
    t0 = time.time()
    evals_result = {}
    model = xgb.train(
        PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=10,
    )
    print(f"Done in {time.time() - t0:.1f}s  (best iter: {model.best_iteration})")

    os.makedirs("outputs/models", exist_ok=True)
    model.save_model("outputs/models/xgboost_iasi1d.json")
    print("Model saved to outputs/models/xgboost_iasi1d.json\n")

    print("Results:")
    for split, dmat, y_true, lat, lon, cf2 in [
        ("val",  dval,  y_val,  lat_va, lon_va, cf2_va),
        ("test", dtest, y_test, lat_te, lon_te, cf2_te),
    ]:
        pred = model.predict(dmat)
        masks = build_masks(lat, lon, cf2)
        for name, mask in masks.items():
            if mask.sum() == 0:
                continue
            print_metrics(split, name, y_true[mask], pred[mask], mask.sum())
        print()


if __name__ == "__main__":
    main()

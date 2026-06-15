"""
Random Forest baseline on IASI1d data.
Each FOV position is treated as an independent sample (207 features → 1 BLH value).
After training, metrics are computed on: all, EU-only, clear-sky, cloudy-sky subsets.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import time
import os
import pickle

CACHE_DIR = "outputs/cache"
SCALE_FACTOR = 4000.0

EU_LON_MIN, EU_LON_MAX = -24.542225, 29.486706
EU_LAT_MIN, EU_LAT_MAX = 50.37499, 71.154709

PARAMS = dict(
    n_estimators=100,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features="sqrt",
    bootstrap=True,
    n_jobs=-1,
    random_state=9,
    verbose=1,
)


def load_split(split):
    path = os.path.join(CACHE_DIR, f"iasi1d_{split}.npz")
    d = np.load(path, allow_pickle=True)
    X = d["samples"].reshape(-1, 207).astype(np.float32)
    y = d["labels"].reshape(-1, 4)[:, 0].astype(np.float32)
    lat = np.vstack(d["raw_lat"]).reshape(-1).astype(np.float32)
    lon = np.vstack(d["raw_lon"]).reshape(-1).astype(np.float32)
    cf2 = np.vstack(d["raw_cloudflag2"]).reshape(-1).astype(np.float32)
    return X, y, lat, lon, cf2


def build_masks(lat, lon, cf2):
    eu = (
        (EU_LON_MIN <= lon) & (lon <= EU_LON_MAX) &
        (EU_LAT_MIN <= lat) & (lat <= EU_LAT_MAX)
    )
    return {
        "all":    np.ones(len(lat), dtype=bool),
        "eu":     eu,
        "clear":  cf2 == 0,
        "cloudy": cf2 > 0,
    }


def metrics(y_true, y_pred, sf=SCALE_FACTOR):
    y_true = y_true * sf
    y_pred = y_pred * sf
    mae  = np.abs(y_true - y_pred).mean()
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

    model = RandomForestRegressor(**PARAMS)

    print("\nTraining...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Done in {time.time() - t0:.1f}s")

    os.makedirs("outputs/models", exist_ok=True)
    with open("outputs/models/rf_iasi1d_small.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to outputs/models/rf_iasi1d_small.pkl\n")

    print("Results:")
    for split, X, y_true, lat, lon, cf2 in [
        ("val",  X_val,  y_val,  lat_va, lon_va, cf2_va),
        ("test", X_test, y_test, lat_te, lon_te, cf2_te),
    ]:
        pred = model.predict(X)
        masks = build_masks(lat, lon, cf2)
        for name, mask in masks.items():
            if mask.sum() == 0:
                continue
            print_metrics(split, name, y_true[mask], pred[mask], mask.sum())
        print()


if __name__ == "__main__":
    main()

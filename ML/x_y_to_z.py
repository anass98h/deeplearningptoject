"""
Train a model that predicts joint-wise Z from X,Y.

Input directory  :  ML/data/frames          (all .csv files)
Saved artefacts  :  xy_to_z_best.keras, X_scaler.pkl, y_scaler.pkl, best_params.pkl
Python ≥3.10, TensorFlow ≥2.16, SciKeras ≥0.12
"""

from __future__ import annotations
from pathlib import Path
import os, random, warnings
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------#
# 0  .  Reproducibility knobs -- set before importing TF
# ---------------------------------------------------------------------------#
os.environ.update(
    PYTHONHASHSEED="42",
    TF_DETERMINISTIC_OPS="1",
    TF_CUDNN_DETERMINISM="1",
)
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------#
# 1  .  Hyper-parameters you are most likely to tweak
# ---------------------------------------------------------------------------#
DIR_DATA    = Path("ML/data/kinect_good_preprocessed")        # where your *.csv live
USE_SCALER  = True
PATIENCE    = 10
MAX_EPOCHS  = 200

param_grid = {                              # grid / random search space
    "model__units":         [256],
    "model__n_hidden":      [12],
    "model__dropout":       [0.05],
    "model__learning_rate": [3e-4],
    "batch_size":           [128],
}

# ---------------------------------------------------------------------------#
# 2  .  Load every CSV → build (X, y)
# ---------------------------------------------------------------------------#
def split_xyz_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ([all _x, _y], [all _z]) columns in consistent joint order."""
    xyz_cols = [c for c in df.columns if c not in ("FrameNo",)]
    joints   = sorted({c.rsplit("_", 1)[0] for c in xyz_cols})
    xy_cols  = [f"{j}_{a}" for j in joints for a in ("x", "y")]
    z_cols   = [f"{j}_z"   for j in joints]
    missing  = [c for c in xy_cols + z_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing in CSV: {missing[:6]} …")
    return xy_cols, z_cols

X_chunks, y_chunks = [], []

for csv_path in sorted(DIR_DATA.glob("*.csv")):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    xy_cols, z_cols = split_xyz_cols(df)

    X_chunks.append(df[xy_cols].to_numpy(dtype=np.float32))
    y_chunks.append(df[z_cols].to_numpy(dtype=np.float32))
    print(f"✅ {csv_path.name:<30}  frames: {len(df)}")

if not X_chunks:
    raise RuntimeError("No CSV files found in ML/data/frames")

X = np.concatenate(X_chunks, axis=0)
y = np.concatenate(y_chunks, axis=0)
print(f"\n🎯 Final shapes  X:{X.shape}  y:{y.shape}")

# ---------------------------------------------------------------------------#
# 3  .  Train / test split  + optional scaling
# ---------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if USE_SCALER:
    X_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y)
    X, y     = X_scaler.transform(X), y_scaler.transform(y)
else:
    X_scaler = y_scaler = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# ---------------------------------------------------------------------------#
# 4  .  Build model factory
# ---------------------------------------------------------------------------#
import tensorflow as tf, keras
tf.random.set_seed(42)
warnings.filterwarnings("ignore", category=UserWarning)

def build_model(
    meta,
    units=128,
    n_hidden=4,
    dropout=0.10,
    learning_rate=1e-3,
    l2=1e-4,
):
    """Return a depth-regression MLP.
    `meta` is auto-injected by SciKeras and includes the data shapes."""
    input_dim  = meta["n_features_in_"]   # 26  (2 × joints)
    target_dim = meta["n_outputs_"]       # 13  (z for each joint)

    reg = keras.regularizers.L2(l2)
    inputs = keras.layers.Input(shape=(input_dim,))     # ← tuple!
    x = inputs
    for _ in range(n_hidden):
        x = keras.layers.Dense(units, kernel_regularizer=reg)(x)
        x = keras.layers.PReLU()(x)
        x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(target_dim)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model

# ---------------------------------------------------------------------------#
# 5  .  Wrap in SciKeras + Grid/Random search
# ---------------------------------------------------------------------------#
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, mean_squared_error
neg_mse = make_scorer(mean_squared_error, greater_is_better=False)

reg = KerasRegressor(
    model=build_model,
    epochs=MAX_EPOCHS,
    verbose=0,
    random_state=42,
    shuffle=True,
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)
reduce_lr  = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=PATIENCE//2, min_lr=1e-6
)

grid = GridSearchCV(
    estimator=reg,
    param_grid=param_grid,
    scoring=neg_mse,
    cv=5,
    refit=True,
    n_jobs=1,           # <= keep memory usage sane
    verbose=10,
)

print(f"\n⏳ GridSearchCV  configs: {len(ParameterGrid(param_grid))}  × 5-fold CV\n")
grid_result = grid.fit(
    X_train, y_train,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
)

# ---------------------------------------------------------------------------#
# 6  .  Report & evaluate
# ---------------------------------------------------------------------------#
print("\n🏆 Best hyper-parameters:")
for k, v in grid_result.best_params_.items():
    print(f"   • {k:18s}: {v}")
print("Best CV MSE :", -grid_result.best_score_)

best_model = grid_result.best_estimator_.model_

test_mse, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print("\n📊 Scaled Test MSE :", test_mse)
print("📊 Scaled Test MAE :", test_mae)

from sklearn.metrics import mean_absolute_error, r2_score
y_pred_scaled = best_model.predict(X_test)
y_test_orig   = y_scaler.inverse_transform(y_test) if y_scaler else y_test
y_pred_orig   = y_scaler.inverse_transform(y_pred_scaled) if y_scaler else y_pred_scaled

print("\n📊 Original-scale Test MAE :", mean_absolute_error(y_test_orig, y_pred_orig))
print("📊 Original-scale R²       :", r2_score(y_test_orig, y_pred_orig))

# ---------------------------------------------------------------------------#
# 7  .  Save everything
# ---------------------------------------------------------------------------#
from joblib import dump
best_model.save("xy_to_z_best.keras")
dump(grid_result.best_params_, "best_params.pkl")
if X_scaler is not None:
    dump(X_scaler, "X_scaler.pkl")
    dump(y_scaler, "y_scaler.pkl")
else:
    Path("NO_SCALER_USED.txt").touch()

print("\n💾  Model stored in xy_to_z_best.keras\n")

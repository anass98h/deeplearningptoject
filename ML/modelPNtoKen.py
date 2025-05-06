from pathlib import Path
import os
import random
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility settings
# --------------------------------------------------------------------------
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

# constants & helper function
DIR_POSE   = Path("ML/data/output_poses")
DIR_KINECT = Path("ML/data/kinect_good")

def xy_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that end with '_x' or '_y'."""
    return [c for c in df.columns if c.endswith(("_x", "_y"))]

# stack all matching Pose/Kinect pairs
X_chunks, y_chunks = [], []

for pose_path in sorted(DIR_POSE.glob("*.csv")):
    key      = pose_path.stem
    kin_path = DIR_KINECT / f"{key}_kinect.csv"
    if not kin_path.exists():
        print(f"⚠️  {key}: missing sister file – skipped.")
        continue

    # read & clean
    df_pose = pd.read_csv(pose_path)
    df_pose.columns = df_pose.columns.str.strip()
    df_kin  = pd.read_csv(kin_path)
    df_kin.columns  = df_kin.columns.str.strip()

    # align on FrameNo
    shared_frames = np.intersect1d(df_pose["FrameNo"], df_kin["FrameNo"])
    if shared_frames.size == 0:
        print(f"⚠️  {key}: no overlapping frames – skipped.")
        continue

    df_pose = df_pose[df_pose["FrameNo"].isin(shared_frames)].sort_values("FrameNo")
    df_kin  = df_kin [df_kin ["FrameNo"].isin(shared_frames)].sort_values("FrameNo")
    if not np.array_equal(df_pose["FrameNo"].values, df_kin["FrameNo"].values):
        print(f"⚠️  {key}: frame mismatch after alignment – skipped.")
        continue

    # collect xy columns
    pose_xy_cols = xy_columns(df_pose)
    missing = [c for c in pose_xy_cols if c not in df_kin.columns]
    if missing:
        print(f"⚠️  {key}: Kinect file missing {len(missing)} XY columns – skipped.")
        continue

    X_chunks.append(df_pose[pose_xy_cols].to_numpy(dtype=float))
    y_chunks.append(df_kin [pose_xy_cols].to_numpy(dtype=float))
    print(f"✅  {key}: kept {len(df_pose)} frames.")

if not X_chunks:
    raise RuntimeError("No valid file pairs were found – nothing to train on.")

features = np.vstack(X_chunks)
targets  = np.vstack(y_chunks)

print("\n🎯  Finished:")
print("    features :", features.shape)
print("    targets  :", targets.shape)

# --------------------------------------------------------------------------
# Reproducibility & imports
# --------------------------------------------------------------------------
import os
os.environ['PYTHONHASHSEED']       = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISM'] = '1'

import random
random.seed(42)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import keras
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from joblib import dump
from scikeras.wrappers import KerasRegressor

# 1. Load data
X = features.copy()
y = targets.copy()

# 2. Configuration
USE_SCALER = True
PATIENCE   = 10
MAX_EPOCHS = 200

param_grid = {
    "model__units":        [128],
    "model__n_hidden":     [10],
    "batch_size":          [128],
    "model__learning_rate":[0.001],
}

# 3. Optional scaling
if USE_SCALER:
    X_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y)
    X = X_scaler.transform(X)
    y = y_scaler.transform(y)
else:
    X_scaler = y_scaler = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# 4. Model factory & wrapper
def build_model(units=128, n_hidden=2, learning_rate=0.001):
    model = keras.Sequential([keras.layers.Input(shape=(26,))])
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(units))
        model.add(keras.layers.PReLU())
    model.add(keras.layers.Dense(26))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

reg = KerasRegressor(
    model=build_model,
    epochs=MAX_EPOCHS,
    verbose=0,
    random_state=42,
    shuffle=True
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True
)

neg_mse = make_scorer(mean_squared_error, greater_is_better=False)

grid = GridSearchCV(
    estimator=reg,
    param_grid=param_grid,
    scoring=neg_mse,
    cv=10,
    n_jobs=-1,
    refit=True,
    verbose=10
)

# 5. Run grid search
print(f"⏳  Running GridSearchCV with {len(ParameterGrid(param_grid))} configs × {grid.cv}-fold CV\n")
grid_result = grid.fit(
    X_train, y_train,
    validation_split=0.1,
    callbacks=[early_stop]
)

# 6. Report results
print("\n🏆  Best hyper-parameters:")
for k, v in grid_result.best_params_.items():
    print(f"   • {k:12s}: {v}")
print("Best CV MSE :", -grid_result.best_score_)

best_model = grid_result.best_estimator_.model_

# 7. Evaluate on test set
test_mse, test_mae = best_model.evaluate(X_test, y_test, verbose=0)

y_pred_scaled = best_model.predict(X_test)
y_test_orig   = y_scaler.inverse_transform(y_test)
y_pred_orig   = y_scaler.inverse_transform(y_pred_scaled)

mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)

print("\n📊 Scaled Test MSE :", test_mse)
print("📊 Scaled Test MAE :", test_mae)
print("\n📊 Original-scale Test MSE :", mse_orig)
print("📊 Original-scale Test MAE :", mae_orig)

# 8. Save artifacts
best_model.save("xy_to_xy_best.keras")
dump(grid_result.best_params_, "best_params.pkl")
if X_scaler is not None:
    dump(X_scaler, "X_scaler.pkl")
    dump(y_scaler, "y_scaler.pkl")
else:
    open("NO_SCALER_USED.txt", "w").close()

print("\n💾  Model saved to xy_to_xy_best.keras")
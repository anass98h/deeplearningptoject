#!/usr/bin/env python3
"""
train_exercise_segmenter.py
--------------------------
Given pairs of PoseNet‑extracted full‑length CSVs and their manually
trimmed counterparts (containing only the exercise segment), this script
builds a sequence model that predicts – for every frame – whether the
exercise is being performed. From the per‑frame probabilities we later
derive start/end frame indices automatically.

Highlights
==========
* Robust loader that aligns full and trimmed sequences on `FrameNo`.
* Generates binary labels (1 = in‑exercise, 0 = outside).
* Optional StandardScaler on XY features.
* Pads sequences to the longest length and uses masking so the model
  ignores padded timesteps.
* Bidirectional LSTM ➜ TimeDistributed Dense with sigmoid output.
* EarlyStopping + model checkpointing; saves best model as `.keras`.
* Provides an `infer_segment()` helper that turns raw probabilities into
  integer `(start_frame, end_frame)` using a configurable threshold and
  minimum segment length.
* Command‑line flags for most hyper‑parameters so you can sweep with
  grid/search tools later.
"""

# -------------------------------------------------------------------- #
# 0. Imports & CLI                                                     #
# -------------------------------------------------------------------- #
import argparse, warnings, os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------------------------------------------- #
# 1. Command‑line arguments                                            #
# -------------------------------------------------------------------- #
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dir_full",   type=Path, default="ML/data/output_poses",      help="Directory with full‑length PoseNet CSVs")
parser.add_argument("--dir_trim",   type=Path, default="ML/data/output_poses_preprocessed",  help="Directory with trimmed CSVs (exercise only)")
parser.add_argument("--use_scaler", action="store_true", help="Apply StandardScaler to XY features")
parser.add_argument("--units",      type=int,   default=64,   help="LSTM units per direction")
parser.add_argument("--n_layers",   type=int,   default=2,    help="Number of stacked Bi‑LSTM layers")
parser.add_argument("--batch",      type=int,   default=32,   help="Batch size")
parser.add_argument("--epochs",     type=int,   default=100,  help="Max epochs")
parser.add_argument("--patience",   type=int,   default=10,   help="EarlyStopping patience")
parser.add_argument("--min_seg",    type=int,   default=15,   help="Minimum accepted segment length at inference (frames)")
parser.add_argument("--prob_th",    type=float, default=0.5,  help="Probability threshold for positive class at inference")
parser.add_argument("--out",        type=Path,  default="exercise_segmenter.keras", help="Path to save best model")
args = parser.parse_args()

# -------------------------------------------------------------------- #
# 2. Helpers                                                           #
# -------------------------------------------------------------------- #

def xy_columns(df: pd.DataFrame) -> list[str]:
    """Return columns ending with '_x' or '_y' in original order."""
    return [c for c in df.columns if c.endswith(("_x", "_y"))]


def load_pair(full_csv: Path, trim_csv: Path):
    """Load a (full, trimmed) pair and return numpy (frames, feats), labels."""
    df_full = pd.read_csv(full_csv).rename(columns=str.strip)
    df_trim = pd.read_csv(trim_csv).rename(columns=str.strip)

    # Determine the [start, end] based on FrameNo present in trimmed
    start_f, end_f = df_trim["FrameNo"].min(), df_trim["FrameNo"].max()

    # Keep only XY columns + FrameNo
    xy_cols = xy_columns(df_full)
    feats   = df_full[xy_cols].to_numpy(dtype=float)
    frames  = df_full["FrameNo"].to_numpy()

    labels  = ((frames >= start_f) & (frames <= end_f)).astype(np.float32)
    return feats, labels


def pad_sequences(seqs, pad_value=0.0):
    """Pad a list of 2‑D arrays (T_i, F) to shape (N, T_max, F)."""
    lengths = [s.shape[0] for s in seqs]
    T_max   = max(lengths)
    F       = seqs[0].shape[1]
    batch   = np.full((len(seqs), T_max, F), pad_value, dtype=np.float32)
    masks   = np.zeros((len(seqs), T_max), dtype=bool)
    for i, seq in enumerate(seqs):
        batch[i, :seq.shape[0], :] = seq
        masks[i, :seq.shape[0]] = True
    return batch, masks, lengths


def infer_segment(probabilities: np.ndarray, threshold: float = 0.5, min_len: int = 15):
    """Convert per‑frame probabilities ➜ (start, end) indices or (None, None)."""
    pos = probabilities >= threshold
    # Find longest contiguous positive run ≥ min_len
    best_start = best_end = None
    current_start = None
    for i, p in enumerate(pos):
        if p and current_start is None:
            current_start = i
        elif not p and current_start is not None:
            span_len = i - current_start
            if span_len >= min_len:
                best_start, best_end = current_start, i - 1
                break
            current_start = None
    # Edge case: run goes till the last frame
    if current_start is not None and (len(pos) - current_start) >= min_len:
        best_start, best_end = current_start, len(pos) - 1
    return best_start, best_end

# -------------------------------------------------------------------- #
# 3. Gather training data                                             #
# -------------------------------------------------------------------- #
print("📂  Scanning for CSV pairs…")
pairs = []
for full_csv in sorted(args.dir_full.glob("*.csv")):
    key = full_csv.stem
    trim_csv = args.dir_trim / f"{key}_exercise.csv"
    if not trim_csv.exists():
        print(f"⚠️  {key}: missing trimmed file – skipped.")
        continue
    pairs.append((full_csv, trim_csv))

if not pairs:
    raise RuntimeError("No valid (full, trimmed) CSV pairs found.")

feat_seqs, label_seqs = [], []
for full_csv, trim_csv in pairs:
    X_i, y_i = load_pair(full_csv, trim_csv)
    feat_seqs.append(X_i)
    label_seqs.append(y_i[:, None])  # make (T,1) so shapes align
    print(f"✅  {full_csv.stem}: frames={X_i.shape[0]}")

# -------------------------------------------------------------------- #
# 4. Optional scaling                                                 #
# -------------------------------------------------------------------- #
if args.use_scaler:
    scaler = StandardScaler().fit(np.vstack(feat_seqs))
    feat_seqs = [scaler.transform(f) for f in feat_seqs]
else:
    scaler = None

# Pad & mask
X, masks, lengths = pad_sequences(feat_seqs)
y, _, _           = pad_sequences(label_seqs, pad_value=0.0)

num_samples, T_max, n_features = X.shape
print(f"📊  Dataset shape: {X.shape}, Labels: {y.shape}")

# -------------------------------------------------------------------- #
# 5. Train / test split                                               #
# -------------------------------------------------------------------- #
train_idx, test_idx = train_test_split(np.arange(num_samples), test_size=0.2, random_state=42)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
mask_train, mask_test = masks[train_idx], masks[test_idx]

# -------------------------------------------------------------------- #
# 6. Build model                                                      #
# -------------------------------------------------------------------- #
inputs   = keras.Input(shape=(T_max, n_features))
x        = keras.layers.Masking(mask_value=0.0)(inputs)
for _ in range(args.n_layers):
    x = keras.layers.Bidirectional(keras.layers.LSTM(args.units, return_sequences=True))(x)
outputs  = keras.layers.TimeDistributed(keras.layers.Dense(1, activation="sigmoid"))(x)
model    = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
model.summary()

# Sample weights so padded positions don't contribute to loss
sample_weights = mask_train.astype(np.float32)
val_weights    = mask_test.astype(np.float32)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True),
]

history = model.fit(
    X_train, y_train,
    epochs=args.epochs,
    batch_size=args.batch,
    validation_data=(X_test, y_test, val_weights),
    sample_weight=sample_weights,
    callbacks=callbacks,
    verbose=2,
)

# -------------------------------------------------------------------- #
# 7. Evaluation                                                       #
# -------------------------------------------------------------------- #
print("\n🧪  Evaluating on held‑out videos…")
loss, acc = model.evaluate(X_test, y_test, sample_weight=val_weights, verbose=0)
print(f"   loss={loss:.4f}, binary_accuracy={acc:.4f}")

# -------------------------------------------------------------------- #
# 8. Save artefacts                                                   #
# -------------------------------------------------------------------- #
if scaler is not None:
    import joblib
    joblib.dump(scaler, args.out.with_suffix("_scaler.pkl"))
    print("🔖  Scaler saved.")

print(f"💾  Best model saved to {args.out}")

# -------------------------------------------------------------------- #
# 9. Inference demo (optional)                                        #
# -------------------------------------------------------------------- #
if __name__ == "__main__":
    # quick demo on first test sample
    idx = 0
    sample = X_test[idx:idx+1]
    probs  = model.predict(sample)[0, :lengths[test_idx[idx]], 0]
    s, e   = infer_segment(probs, threshold=args.prob_th, min_len=args.min_seg)
    print(f"\n📺  Predicted exercise frames: start={s}, end={e}")

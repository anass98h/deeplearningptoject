# ╔════════════════════════════════════════════════════════════════════╗
#  Pose-boundary detection — clean, self-contained training script
# ╚════════════════════════════════════════════════════════════════════╝
"""
Pipeline
--------
1.  Load every full-pose CSV and its trimmed counterpart.
2.  Build smoothed + δ-features and binary frame labels.
3.  Split *entire videos* into train/test with GroupShuffleSplit.
4.  Standard-scale features (fit on train only).
5.  Tune a small MLP (scikeras + GridSearchCV + early-stopping).
6.  Save best model (.keras) and scaler (joblib).
7.  Report frame-level test metrics.
8.  Compute per-video boundary error on test set.
"""
# ──────────────────────────────────────────────────────────────────────
# 0. Imports
# ──────────────────────────────────────────────────────────────────────
import warnings, os
from pathlib import Path
from collections import defaultdict

import numpy   as np
import pandas  as pd

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (precision_score, recall_score, f1_score,
                                      classification_report, make_scorer)
from scikeras.wrappers        import KerasClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import joblib


# ──────────────────────────────────────────────────────────────────────
# 1. Data loading  ➜  X_processed, y_frames, video_ids, offsets
# ──────────────────────────────────────────────────────────────────────
full_dir    = Path("ML/data/output_poses")
trimmed_dir = Path("ML/data/output_poses_preprocessed")

X_parts, y_parts, video_ids = [], [], []
offsets, cursor = {}, 0                       # keep slice of each video


def smooth_sequence(seq: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing along time axis."""
    pad = window // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([padded[i:i + window].mean(axis=0)
                     for i in range(len(seq))])


for full_path in sorted(full_dir.glob("*.csv")):
    vid = full_path.stem
    trim_path = trimmed_dir / f"{vid}.csv"
    if not trim_path.exists():
        print(f"⚠️  {vid}: trimmed file missing — skipped."); continue

    df_full = pd.read_csv(full_path).sort_values("FrameNo")
    df_trim = pd.read_csv(trim_path).sort_values("FrameNo")
    if df_full.empty or df_trim.empty:
        print(f"⚠️  {vid}: empty CSV — skipped."); continue

    # frame labels: 1 inside trimmed range, else 0
    start_fno, end_fno = df_trim["FrameNo"].iloc[[0, -1]]
    labels = df_full["FrameNo"].between(start_fno, end_fno).astype(int).values

    pose_cols = [c for c in df_full.columns if c != "FrameNo"]
    raw_pose  = df_full[pose_cols].values.astype(float)

    smoothed  = smooth_sequence(raw_pose, window=5)
    deltas    = np.diff(smoothed, axis=0, prepend=smoothed[[0]])
    features  = np.hstack([smoothed, deltas])

    X_parts.append(features)
    y_parts.append(labels)
    video_ids.extend([vid] * len(labels))
    offsets[vid] = (cursor, cursor + len(features))
    cursor += len(features)

    print(f"✅ {vid}: {labels.sum()} exercise frames of {len(labels)} total.")

if not X_parts:
    raise RuntimeError("No valid video pairs found.")

X_processed = np.vstack(X_parts)          # (N_frames, D_feat)
y_frames    = np.concatenate(y_parts)     # (N_frames,)
video_ids   = np.array(video_ids)         # (N_frames,)


# ──────────────────────────────────────────────────────────────────────
# 2. Train/test split (grouped by video)
# ──────────────────────────────────────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X_processed, y_frames, groups=video_ids))

train_videos = np.unique(video_ids[train_idx])
test_videos  = np.unique(video_ids[test_idx])
print(f"\nTrain videos: {len(train_videos)}   |   Test videos: {len(test_videos)}")

X_train, y_train = X_processed[train_idx], y_frames[train_idx]
X_test,  y_test  = X_processed[test_idx],  y_frames[test_idx]

scaler = StandardScaler().fit(X_train)     # no leakage
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


# ──────────────────────────────────────────────────────────────────────
# 3. Model definition & hyper-parameter search
# ──────────────────────────────────────────────────────────────────────
def create_model(hidden_units=64, hidden_layers=1,
                 dropout_rate=0.0, learning_rate=1e-3):
    m = keras.Sequential([keras.layers.Input(shape=(X_train.shape[1],))])
    for _ in range(hidden_layers):
        m.add(keras.layers.Dense(hidden_units, activation="relu"))
        if dropout_rate:
            m.add(keras.layers.Dropout(dropout_rate))
    m.add(keras.layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


early = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

clf = KerasClassifier(model=create_model, verbose=0, callbacks=[early])

param_grid = {
    "model__hidden_units":  [256],
    "model__hidden_layers": [12],
    "model__dropout_rate":  [0.30],
    "model__learning_rate": [1e-4],
    "epochs":     [50],
    "batch_size": [64],
}

grid = GridSearchCV(clf, param_grid,
                    scoring=make_scorer(f1_score),
                    cv=3, n_jobs=-1, verbose=10)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_.model_
print("\nBest hyper-parameters →", grid.best_params_)


# ──────────────────────────────────────────────────────────────────────
# 4. Persist model + scaler
# ──────────────────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
best_model.save("models/boundary_model.keras")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Model saved to   models/boundary_model.keras")
print("✅ Scaler saved to  models/scaler.pkl")


# ──────────────────────────────────────────────────────────────────────
# 5. Frame-level metrics on test set
# ──────────────────────────────────────────────────────────────────────
y_pred_prob = best_model.predict(X_test).ravel()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("\n──────── Frame-level test metrics ────────")
print("Precision :", precision_score(y_test, y_pred))
print("Recall    :", recall_score(y_test,  y_pred))
print("F1-score  :", f1_score(y_test,     y_pred))
print("\n", classification_report(y_test, y_pred, digits=3))


# ──────────────────────────────────────────────────────────────────────
# 6. Boundary-error evaluation (only unseen videos)
# ──────────────────────────────────────────────────────────────────────
print("\n──────── Per-video boundary error (test set) ────────")

delta_start, delta_end = [], []

for vid in test_videos:
    s, e = offsets[vid]
    gt   = y_frames[s:e]

    if gt.sum() == 0:                       # no positives: skip
        warnings.warn(f"{vid}: all-negative — skipped.")
        continue

    true_start = int(np.argmax(gt == 1))
    true_end   = int(len(gt) - 1 - np.argmax(gt[::-1] == 1))

    X_vid     = scaler.transform(X_processed[s:e])
    pred_prob = best_model.predict(X_vid).ravel()
    pred_lbl  = (pred_prob >= 0.5).astype(int)

    # Remove 1-frame glitches with simple majority filter
    for i in range(1, len(pred_lbl) - 1):
        if pred_lbl[i-1] == pred_lbl[i+1] != pred_lbl[i]:
            pred_lbl[i] = pred_lbl[i-1]

    # Longest contiguous 1-segment
    segments, in_seg, s0 = [], False, 0
    for i, lab in enumerate(pred_lbl):
        if lab and not in_seg:
            in_seg, s0 = True, i
        if (not lab and in_seg):
            segments.append((s0, i-1)); in_seg = False
    if in_seg:
        segments.append((s0, len(pred_lbl)-1))

    if not segments:
        print(f"{vid}: ❌ no segment predicted"); continue

    seg_len = [e2 - s2 + 1 for s2, e2 in segments]
    pred_start, pred_end = segments[int(np.argmax(seg_len))]

    d_s, d_e = pred_start - true_start, pred_end - true_end
    delta_start.append(d_s); delta_end.append(d_e)

    print(f"{vid}:  GT[{true_start:>4}, {true_end:>4}]  |  "
          f"Pred[{pred_start:>4}, {pred_end:>4}]  "
          f"→ Δstart {d_s:+4d}  Δend {d_e:+4d}")

# Aggregate boundary error
if delta_start:
    ds, de = np.array(delta_start), np.array(delta_end)
    print("\n──────── Aggregate boundary error ────────")
    print(f"Δstart  mean {ds.mean():+6.2f} ± {ds.std():.2f}   "
          f"median |Δ| {np.median(np.abs(ds)):.1f} frames")
    print(f"Δend    mean {de.mean():+6.2f} ± {de.std():.2f}   "
          f"median |Δ| {np.median(np.abs(de)):.1f} frames")
else:
    print("No boundary stats (model predicted no positives).")

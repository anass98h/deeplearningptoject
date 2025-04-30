# ======================================================================
# 0.  IMPORTS
# ======================================================================
import os, re, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection    import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing      import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics            import make_scorer, f1_score

import tensorflow as tf
from keras import layers, models, callbacks

from scikeras.wrappers import KerasClassifier
from joblib import dump
# ----------------------------------------------------------------------

# ======================================================================
# 1.  LOAD & SPLIT
# ======================================================================
def load_squat_binary_matched(uncut_dir, cut_dir):
    uncut   = {f for f in os.listdir(uncut_dir) if f.endswith("_kinect.csv")}
    cut     = {f for f in os.listdir(cut_dir)   if f.endswith("_kinect.csv")}
    matched = sorted(uncut & cut, key=lambda f: re.match(r'^([A-Z])(\d+)', f).groups())

    X_parts, y_parts, groups = [], [], []
    for fn in matched:
        df_full = pd.read_csv(os.path.join(uncut_dir, fn))
        df_cut  = pd.read_csv(os.path.join(cut_dir,   fn))
        cut_set = set(df_cut["FrameNo"])
        X_parts.append(df_full.drop(columns=["FrameNo"]))
        y_parts.append(df_full["FrameNo"].isin(cut_set).astype(int))
        groups.extend([fn] * len(df_full))

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    return X, y, np.array(groups)

UNCUT = "ML/data/kinect_good"
CUT   = "ML/data/kinect_good_preprocessed"

X, y, groups = load_squat_binary_matched(UNCUT, CUT)
print(f"Total frames: {len(y)}, sequences: {len(np.unique(groups))}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups[train_idx]
X_test,  y_test,  groups_test  = X.iloc[test_idx],  y.iloc[test_idx],  groups[test_idx]

# strip accidental whitespace in column names
X_train.columns = X_train.columns.str.strip()
X_test.columns  = X_test.columns.str.strip()


# ======================================================================
# 2.  DATA-AUGMENTATION (mirror + rotate)
# ======================================================================
JOINT_PAIRS = [
    ("left_shoulder","right_shoulder"),
    ("left_elbow",   "right_elbow"),
    ("left_hand",    "right_hand"),
    ("left_hip",     "right_hip"),
    ("left_knee",    "right_knee"),
    ("left_foot",    "right_foot"),
]

def mirror_df(df):
    m = df.copy()
    for c in m.columns:
        if c.endswith("_x"):
            m[c] = -m[c]
    for L, R in JOINT_PAIRS:
        for axis in ("x","y","z"):
            lcol, rcol = f"{L}_{axis}", f"{R}_{axis}"
            if lcol in m and rcol in m:
                m[lcol], m[rcol] = m[rcol].copy(), m[lcol].copy()
    return m

def rotate_df(df, angle):
    r = df.copy()
    c, s = np.cos(angle), np.sin(angle)
    for col in r.columns:
        if col.endswith("_x"):
            base = col[:-2]
            xcol, zcol = f"{base}_x", f"{base}_z"
            if zcol in r:
                x, z = df[xcol].values, df[zcol].values
                r[xcol] = c*x - s*z
                r[zcol] = s*x + c*z
    return r

aug_X = [X_train]
aug_y = [y_train]
aug_g = [groups_train]

aug_X.append(mirror_df(X_train));       aug_y.append(y_train.copy()); aug_g.append(groups_train.copy())
for ang in (np.deg2rad(15), np.deg2rad(-15)):
    aug_X.append(rotate_df(X_train, ang)); aug_y.append(y_train.copy()); aug_g.append(groups_train.copy())

X_train_aug       = pd.concat(aug_X, ignore_index=True)
y_train_aug       = pd.concat(aug_y, ignore_index=True)
groups_train_aug  = np.concatenate(aug_g)

print("Train frames before aug:", len(X_train))
print("Train frames after  aug:", len(X_train_aug))


# ======================================================================
# 3.  SLIDING-WINDOWS
# ======================================================================
WINDOW = 11
HALF   = WINDOW // 2

def build_windows(X_df, y_arr, g_arr):
    X_np = X_df.values
    X_win, y_win, centre_global_idx = [], [], []
    for i in range(HALF, len(X_np) - HALF):
        if np.all(g_arr[i-HALF : i+HALF+1] == g_arr[i]):
            X_win.append(X_np[i-HALF : i+HALF+1].ravel())
            y_win.append(y_arr[i])
            centre_global_idx.append(i)
    return np.stack(X_win), np.array(y_win), np.array(centre_global_idx)

X_win,      y_win,      centre_idx_train = build_windows(
    X_train_aug, y_train_aug.values, groups_train_aug
)
X_test_win, y_test_win, centre_idx_test  = build_windows(
    X_test, y_test.values, groups_test
)

print(f"Windowed data — train: {X_win.shape}, test: {X_test_win.shape}")


# ======================================================================
# 4.  SCALING & CLASS-WEIGHTS
# ======================================================================
scaler  = StandardScaler().fit(X_win)
X_tr    = scaler.transform(X_win)
X_te    = scaler.transform(X_test_win)

classes = np.unique(y_win)
cw      = compute_class_weight(class_weight="balanced",
                               classes=classes,
                               y=y_win)
class_weight = dict(zip(classes, cw))


# ======================================================================
# 5.  MODEL FACTORY (parametrized for grid search)
# ======================================================================
def build_model(n_layers=2, units=64, learning_rate=0.001):
    m = models.Sequential()
    m.add(layers.Input(shape=(X_tr.shape[1],)))
    for _ in range(n_layers):
        m.add(layers.Dense(units, activation="relu"))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return m


# ======================================================================
# 6.  GRID SEARCH CV
# ======================================================================
param_grid = {
    "model__n_layers":          [12],              # medium vs. slightly deeper net
    "model__units":             [128],           # moderate vs. higher capacity
    "optimizer__learning_rate": [1e-4],        # standard Adam LR vs. a smaller step
    "batch_size":               [128],           # small vs. medium batches
    "epochs":                   [50],           # enough to converge vs. extra training
}

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

reg = KerasClassifier(
    model=build_model,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0
)

f1_scorer = make_scorer(f1_score)

grid = GridSearchCV(
    estimator=reg,
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=5,
    n_jobs=-1,
    refit=True,
    verbose=3
)

print(f"\n⏳ Running GridSearchCV over {np.prod([len(v) for v in param_grid.values()])} configs × {grid.cv}-fold CV\n")
grid_result = grid.fit(X_tr, y_win)

print("\n🏆 Best hyper-parameters:")
for k, v in grid_result.best_params_.items():
    print(f"   • {k:20s}: {v}")
print("Best CV F1-score :", grid_result.best_score_)

best_model = grid_result.best_estimator_.model_


# ======================================================================
# 7.  EVALUATION (use best_model)
# ======================================================================
loss, acc, prec, rec = best_model.evaluate(X_te, y_test_win, verbose=0)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
print(f"\nWindowed-test → loss {loss:.4f}  acc {acc:.4f}  precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")


# ======================================================================
# 8.  BOUNDARY-ERROR EVALUATION
# ======================================================================
print("\n───────── Per-video boundary error ─────────")
delta_start_list, delta_end_list = [], []
per_video_results = {}

unique_test_vids = np.unique(groups_test)

for vid in unique_test_vids:
    frame_mask  = (groups_test == vid)
    y_vid       = y_test.values[frame_mask]
    X_vid_df    = X_test.loc[frame_mask]
    n_frames = len(y_vid)
    if y_vid.sum() == 0 or n_frames < WINDOW:
        warnings.warn(f"{vid}: skipped (no positives or too short).")
        continue

    win_feats, centres = [], []
    for idx in range(HALF, n_frames - HALF):
        win_feats.append(X_vid_df.iloc[idx-HALF : idx+HALF+1].values.ravel())
        centres.append(idx)

    X_vid_win = scaler.transform(np.stack(win_feats))
    pred_prob = best_model.predict(X_vid_win, verbose=0).ravel()
    pred_lbl  = (pred_prob >= 0.5).astype(int)

    frame_pred = np.zeros(n_frames, dtype=int)
    for c_idx, lbl in zip(centres, pred_lbl):
        frame_pred[c_idx] = lbl
    for i in range(1, n_frames-1):
        if frame_pred[i-1] == frame_pred[i+1] != frame_pred[i]:
            frame_pred[i] = frame_pred[i-1]

    true_start = int(np.argmax(y_vid == 1))
    true_end   = int(n_frames - 1 - np.argmax(y_vid[::-1] == 1))

    segments, in_seg = [], False
    for i, l in enumerate(frame_pred):
        if l == 1 and not in_seg:
            in_seg, seg_start = True, i
        if (l == 0 and in_seg) or (in_seg and i == n_frames-1):
            seg_end = i-1 if l == 0 else i
            segments.append((seg_start, seg_end))
            in_seg = False

    if not segments:
        print(f"{vid}: ❌  no segment predicted")
        continue

    seg_lens = [e - s + 1 for s, e in segments]
    idx_long = int(np.argmax(seg_lens))
    pred_start, pred_end = segments[idx_long]

    d_start = pred_start - true_start
    d_end   = pred_end   - true_end
    delta_start_list.append(d_start)
    delta_end_list.append(d_end)
    per_video_results[vid] = {
        "true":  (true_start, true_end),
        "pred":  (pred_start, pred_end),
        "delta": (d_start, d_end)
    }

    print(f"{vid}:  GT [{true_start:>4}, {true_end:>4}]  |  "
          f"Pred [{pred_start:>4}, {pred_end:>4}]  "
          f"→  Δstart {d_start:+4d}  Δend {d_end:+4d}")

if delta_start_list:
    ds = np.array(delta_start_list); de = np.array(delta_end_list)
    print("\n───────── Aggregate boundary error ─────────")
    print(f"Δstart  mean {ds.mean():+6.2f} ± {ds.std():.2f}   "
          f"median |Δ| {np.median(np.abs(ds)):.1f} frames")
    print(f"Δend    mean {de.mean():+6.2f} ± {de.std():.2f}   "
          f"median |Δ| {np.median(np.abs(de)):.1f} frames")
else:
    print("No boundary statistics (no positive predictions).")


# ======================================================================
# 9.  SAVE ARTIFACTS
# ======================================================================
# Save the best model in Keras v3 format
best_model.save("kinect_cutting_model.keras")

# Save the input scaler
dump(scaler, "kinect_cutting_scaler.pkl")

print("\n💾  Model saved to kinect_cutting_model.keras")
#!/usr/bin/env python3
"""
hybrid_cnn_rnn_numframes_grid.py – grid-searches over NUM_FRAMES, too
"""
# ---------------------------------------------------------------------
import os, time, itertools, numpy as np, pandas as pd, cv2, tensorflow as tf
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU only
DEBUG    = True
N_FOLDS  = 10        # change to 10 later
# ---------------- 1. CONFIG ------------------------------------------
video_folder   = Path("ML/data/kinect_good_vs_bad")
pose_folder    = Path("ML/data/kinect_good_vs_bad_with_c")
label_csv_path = Path("ML/merged_video_scores.csv")

FRAME_OPTIONS = [15]          # <<< values you want to try
MAX_FRAMES    = max(FRAME_OPTIONS)

IMG_H, IMG_W  = 64, 64

GRID = {
    "num_frames":     FRAME_OPTIONS,
    "optimizer":      ["rmsprop"],
    "learning_rate":  [1e-3],
    "batch_size":     [16],
    "epochs":         [32],
}
# ---------------- 2. LOAD DATA (up to MAX_FRAMES once) ---------------
labels_df = pd.read_csv(label_csv_path)
if DEBUG: print(f"[INFO] {len(labels_df)} label rows")

Ximgs, Xpose, Y = [], [], []
for i, row in labels_df.iterrows():
    vid = video_folder / row["filename"]
    if vid.suffix == "": vid = vid.with_suffix(".avi")
    pose_csv = pose_folder / (vid.stem + ".csv")
    if not (vid.is_file() and pose_csv.is_file()):
        if DEBUG: print(f"[WARN] missing {vid.name}"); continue

    # video
    cap, frames = cv2.VideoCapture(str(vid)), []
    while len(frames) < MAX_FRAMES:
        r, fr = cap.read()
        if not r: break
        frames.append(cv2.resize(fr,(IMG_W,IMG_H),
                                 interpolation=cv2.INTER_AREA).astype("float32")/255.)
    cap.release()
    if len(frames) < MAX_FRAMES:
        frames += [np.zeros((IMG_H,IMG_W,3),"float32")]*(MAX_FRAMES-len(frames))
    Ximgs.append(np.stack(frames[:MAX_FRAMES]))

    # pose
    df = pd.read_csv(pose_csv)
    if "FrameNo" in df.columns: df = df.drop(columns=["FrameNo"])
    pose = df.iloc[:MAX_FRAMES].to_numpy("float32")
    if pose.shape[0] < MAX_FRAMES:
        pose = np.vstack([pose,
                          np.zeros((MAX_FRAMES-pose.shape[0], pose.shape[1]),"float32")])
    Xpose.append(pose)
    Y.append([row["goodness"], row["confidence_score"]])

    if DEBUG and (i+1)%50==0:
        print(f"[DEBUG] processed {i+1}/{len(labels_df)}")

Ximgs, Xpose, Y = map(np.asarray, (Ximgs, Xpose, Y))
print(f"[SHAPE] imgs {Ximgs.shape}, pose {Xpose.shape}, Y {Y.shape}")
# ---------------- 3. SPLIT + KFOLDS ----------------------------------
from sklearn.model_selection import train_test_split, KFold
Xi_tr, Xi_te, Xp_tr, Xp_te, y_tr, y_te = train_test_split(
    Ximgs, Xpose, Y, test_size=0.2, random_state=42, shuffle=True)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
pose_dim = Xpose.shape[2]
print(f"[SPLIT] train {Xi_tr.shape[0]} • test {Xi_te.shape[0]}  folds={N_FOLDS}")
# ---------------- 4. MODEL FACTORY -----------------------------------
from tensorflow.keras.layers import (Input, TimeDistributed, Conv2D, MaxPooling2D,
                                     Flatten, LSTM, Dense, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def mae_good(y_true, y_pred):  return tf.reduce_mean(tf.abs(y_true[:,0]-y_pred[:,0]))
def mae_conf(y_true,y_pred):   return tf.reduce_mean(tf.abs(y_true[:,1]-y_pred[:,1]))

def build_model(num_frames, optimizer="adam", learning_rate=1e-3):
    vin = Input((num_frames, IMG_H, IMG_W, 3))
    x   = TimeDistributed(Conv2D(16,(3,3),activation="relu"))(vin)
    x   = TimeDistributed(MaxPooling2D(2))(x)
    x   = TimeDistributed(Flatten())(x)
    x   = LSTM(32)(x)

    pin = Input((num_frames, pose_dim))
    y   = LSTM(32)(pin)

    z   = Concatenate()([x, y])
    z   = Dense(32, activation="relu")(z)
    z   = Dense(16, activation="relu")(z)
    out = Dense(2, activation="linear")(z)

    opt = (tf.keras.optimizers.Adam(learning_rate)
           if optimizer=="adam" else tf.keras.optimizers.RMSprop(learning_rate))
    model = Model([vin,pin], out)
    model.compile(opt, loss="mae",
                  metrics=[mae_good, mae_conf, "mae", "mse"])
    return model
# ---------------- 5. GRID SEARCH -------------------------------------
def evaluate(params):
    nf   = params["num_frames"]
    maes = []
    for f,(tr_idx,val_idx) in enumerate(kf.split(Xi_tr),1):
        model = build_model(nf, params["optimizer"], params["learning_rate"])
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        hist = model.fit(
            [Xi_tr[tr_idx,:nf], Xp_tr[tr_idx,:nf]], y_tr[tr_idx],
            validation_data=([Xi_tr[val_idx,:nf], Xp_tr[val_idx,:nf]], y_tr[val_idx]),
            epochs=params["epochs"], batch_size=params["batch_size"],
            callbacks=[es], verbose=0)
        
        mae = hist.history["val_mae"][-1]
        val_good = hist.history["val_mae_good"][-1]       # <- new
        val_conf = hist.history["val_mae_conf"][-1]       # <- new
        print(f"   fold {f}/{N_FOLDS}  nf={nf}"
            f"  MAE_good={val_good:.3f}"
            f"  MAE_conf={val_conf:.3f}"
            f"  mae={mae:.3f}")
        maes.append(mae)
        tf.keras.backend.clear_session()
    return np.mean(maes)

print("\n[INFO] >>> GRID over "
      f"{np.prod([len(v) for v in GRID.values()])} combos <<<")
best_p, best_mae = None, np.inf
for combo in itertools.product(*GRID.values()):
    p = dict(zip(GRID.keys(), combo))
    print(f"\n[TEST] {p}")
    t0=time.time(); m = evaluate(p); dt=time.time()-t0
    print(f"[MEAN] MAE={m:.4f}  ({dt:.1f}s)")
    if m < best_mae: best_mae, best_p = m, p.copy()

print(f"\n[BEST] {best_p}  → MAE={best_mae:.4f}")
# ---------------- 6. FINAL TRAIN -------------------------------------
nf = best_p["num_frames"]
final = build_model(nf, best_p["optimizer"], best_p["learning_rate"])
final.summary()
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
final.fit([Xi_tr[:,:nf], Xp_tr[:,:nf]], y_tr,
          validation_split=0.1,
          epochs=best_p["epochs"], batch_size=best_p["batch_size"],
          callbacks=[es], verbose=1)
# ---------------- 7. EVAL --------------------------------------------
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE
pred_tr = final.predict([Xi_tr[:,:nf], Xp_tr[:,:nf]], verbose=0)
pred_te = final.predict([Xi_te[:,:nf], Xp_te[:,:nf]], verbose=0)
print(f"[TRAIN] MAE_total={MAE(y_tr,pred_tr):.4f}")
print(f"[TEST ] MAE_total={MAE(y_te,pred_te):.4f}")
# ---------------- 8. SAVE --------------------------------------------
final.save("trained_hybrid_model.keras")
print("[SAVE] model saved")

#!/usr/bin/env python3
"""
predict_exercise_fixed.py – infer goodness + confidence from ONE video

Simply edit FILE_NAME below, then run:
    python predict_exercise_fixed.py
"""
# --------------------------------------------------------------------
import os, sys, cv2, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
import time  # <-- Added to measure inference time

# --------------- 1. EDIT HERE ---------------------------------------
FILE_NAME   = "A30"          # <<< put your video filename here
MODEL_PATH  = Path("trained_hybrid_model.keras")
VIDEO_FOLDER = Path("ML/data/kinect_good_vs_bad")
POSE_FOLDER  = Path("ML/data/kinect_good_vs_bad_with_c")
NUM_FRAMES   = 15                      # must match the trained model
IMG_H, IMG_W = 64, 64                 # must match the trained model
# --------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # CPU-only (remove if GPU ok)

# --- custom metrics so the model loads -------------------------------------
def mae_good(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))
def mae_conf(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

# --- helper to load & preprocess a single sample ---------------------------
def load_sample(file_name:str):
    stem = Path(file_name).stem
    vid_path  = VIDEO_FOLDER / (stem + ".avi")
    pose_path = POSE_FOLDER  / (stem + ".csv")

    if not vid_path.is_file() or not pose_path.is_file():
        raise FileNotFoundError(f"Missing pair: {vid_path}, {pose_path}")

    # --- video frames -------------------------------------------------------
    cap = cv2.VideoCapture(str(vid_path))
    frames = []
    while len(frames) < NUM_FRAMES:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, (IMG_W, IMG_H),
                       interpolation=cv2.INTER_AREA).astype("float32")/255.
        frames.append(f)
    cap.release()
    if len(frames) < NUM_FRAMES:
        frames += [np.zeros((IMG_H, IMG_W, 3), "float32")] * (NUM_FRAMES - len(frames))
    frames = np.stack(frames[:NUM_FRAMES])                     # (n,h,w,3)

    # --- pose CSV ----------------------------------------------------------
    df = pd.read_csv(pose_path)
    if "FrameNo" in df.columns:
        df = df.drop(columns=["FrameNo"])
    pose = df.iloc[:NUM_FRAMES].to_numpy("float32")
    if pose.shape[0] < NUM_FRAMES:
        pad = np.zeros((NUM_FRAMES - pose.shape[0], pose.shape[1]), "float32")
        pose = np.vstack([pose, pad])
    pose = pose[:NUM_FRAMES]                                   # (n,feat)

    # add batch dimension
    return frames[np.newaxis, ...], pose[np.newaxis, ...]

# ---------------- MAIN ------------------------------------------------------
if __name__ == "__main__":
    try:
        X_vid, X_pose = load_sample(FILE_NAME)
    except FileNotFoundError as e:
        sys.exit(f"[ERROR] {e}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"mae_good": mae_good, "mae_conf": mae_conf}
    )

    start_time = time.time()
    pred = model.predict([X_vid, X_pose], verbose=0)[0]
    duration = time.time() - start_time

    goodness_pred, conf_pred = pred

    print(f"\nPrediction for '{FILE_NAME}':")
    print(f"  goodness         = {goodness_pred:.3f}")
    print(f"  confidence_score = {conf_pred:.3f}")
    print(f"  prediction_time  = {duration:.4f} seconds")

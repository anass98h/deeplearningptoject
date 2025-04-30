#!/usr/bin/env python3
"""
predict_squat_cut.py

Load a trained .keras model + scaler, preprocess one Kinect CSV,
apply sliding-window inference, and output per-frame probabilities
and labels, plus the detected start/end frames.
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ── USER-CONFIGurable paths ───────────────────────────────────────
MODEL_PATH   = "kinect_cutting_model.keras"      # your trained model
SCALER_PATH  = "kinect_cutting_scaler.pkl"       # your trained StandardScaler
INPUT_CSV    = "ML/data/kinect_good/A1_kinect.csv"  # input file with FrameNo + 39 features
OUTPUT_CSV   = "predictions_kinect_split.csv"                 # where to write FrameNo, prob, label
# ────────────────────────────────────────────────────────────────

# window size must match training
WINDOW = 11
HALF   = WINDOW // 2

def build_windows(X: np.ndarray):
    """
    Given array of shape (N, 39), return:
      • windows: ndarray of shape (N-WINDOW+1, WINDOW*39)
      • centers: array of center indices, length N-WINDOW+1
    """
    N, F = X.shape
    ws, centers = [], []
    for i in range(HALF, N - HALF):
        ws.append(X[i-HALF : i+HALF+1].ravel())
        centers.append(i)
    return np.stack(ws), np.array(centers)

def deglitch(labels: np.ndarray):
    """Remove isolated label flips: if label[i-1] == label[i+1] != label[i]."""
    lbl = labels.copy()
    for i in range(1, len(lbl)-1):
        if lbl[i-1] == lbl[i+1] != lbl[i]:
            lbl[i] = lbl[i-1]
    return lbl

def find_longest_segment(labels: np.ndarray):
    """
    Return (start_idx, end_idx) of the longest contiguous run of 1s.
    Returns None if no 1's.
    """
    segments, in_seg = [], False
    for i, l in enumerate(labels):
        if l == 1 and not in_seg:
            in_seg, seg_start = True, i
        if (l == 0 and in_seg) or (in_seg and i == len(labels)-1):
            seg_end = i-1 if l == 0 else i
            segments.append((seg_start, seg_end))
            in_seg = False
    if not segments:
        return None
    lengths = [e - s + 1 for s, e in segments]
    idx = int(np.argmax(lengths))
    return segments[idx]

def main():
    # 1) Load model & scaler
    model  = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2) Read CSV
    df = pd.read_csv(INPUT_CSV).sort_values("FrameNo")
    frames = df["FrameNo"].to_numpy(dtype=int)

    # feature columns = all except FrameNo
    feat_cols = [c for c in df.columns if c != "FrameNo"]
    X_raw = df[feat_cols].to_numpy(dtype=float)

    # 3) Build sliding windows & centers
    X_win, centers = build_windows(X_raw)

    # 4) Scale & predict
    Xs   = scaler.transform(X_win)
    prob = model.predict(Xs, verbose=0).ravel()
    lbl  = (prob >= 0.5).astype(int)

    # 5) Map back to per-frame
    N = len(frames)
    frame_prob  = np.zeros(N, dtype=float)
    frame_label = np.zeros(N, dtype=int)
    for c, p, l in zip(centers, prob, lbl):
        frame_prob[c]  = p
        frame_label[c] = l

    # 6) De-glitch
    smooth_label = deglitch(frame_label)

    # 7) Find longest segment
    seg = find_longest_segment(smooth_label)
    if seg is None:
        print("⚠️  No squat segment detected.")
    else:
        start_idx, end_idx = seg
        start_frame = frames[start_idx]
        end_frame   = frames[end_idx]
        print(f"✅ Squat segment: frames {start_frame} → {end_frame}")

    # 8) Write predictions
    out_df = pd.DataFrame({
        "FrameNo":       frames,
        "probability":   frame_prob,
        "raw_label":     frame_label,
        "smoothed_label": smooth_label
    })
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾  Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

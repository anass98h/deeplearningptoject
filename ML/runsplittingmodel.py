#!/usr/bin/env python
"""
use_model_simple.py

Load a .keras model + scaler, preprocess one CSV, and write out predictions.
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ── USER-CONFIGurable paths ───────────────────────────────────────
MODEL_PATH   = "models/boundary_model.keras"   # your .keras model file
SCALER_PATH  = "models/scaler.pkl"             # your saved StandardScaler
INPUT_CSV    = "ML/data/output_poses/A83.csv"      # CSV with FrameNo + pose features
OUTPUT_CSV   = "ML/data/predictions/predictions.csv"          # where to write probs+labels
# ────────────────────────────────────────────────────────────────
def smooth_sequence(seq, window=5):
    pad = window // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(seq)
    for i in range(len(seq)):
        out[i] = padded[i : i + window].mean(axis=0)
    return out

def load_and_preprocess(path):
    df = pd.read_csv(path).sort_values("FrameNo")
    frames = df["FrameNo"].to_numpy()
    feat_cols = [c for c in df.columns if c != "FrameNo"]
    raw = df[feat_cols].to_numpy(dtype=float)
    sm = smooth_sequence(raw, window=5)
    deltas = np.diff(sm, axis=0, prepend=sm[[0]])
    X = np.hstack([sm, deltas])
    return X, frames

def find_longest_segment(labels):
    """
    Given a 1D array of 0/1 labels, find the (start_idx, end_idx)
    of the longest contiguous run of 1's. Returns None if no 1's.
    """
    segments = []
    in_seg = False
    for i, l in enumerate(labels):
        if l == 1 and not in_seg:
            in_seg = True
            seg_start = i
        if l == 0 and in_seg:
            segments.append((seg_start, i-1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(labels)-1))
    if not segments:
        return None
    # pick longest
    lengths = [e - s + 1 for s,e in segments]
    idx = int(np.argmax(lengths))
    return segments[idx]

def main():
    # 1) load artifacts
    model  = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2) preprocess
    X, frames = load_and_preprocess(INPUT_CSV)

    # 3) scale + predict
    Xs    = scaler.transform(X)
    prob  = model.predict(Xs).ravel()
    label = (prob >= 0.5).astype(int)

    # 4) de-glitch single-frame flips
    for i in range(1, len(label) - 1):
        if label[i-1] == label[i+1] != label[i]:
            label[i] = label[i-1]

    # 5) find longest exercise segment
    seg = find_longest_segment(label)
    if seg is None:
        print("⚠️  No exercise segment detected.")
        return
    start_idx, end_idx = seg
    start_frame = int(frames[start_idx])
    end_frame   = int(frames[end_idx])

    # 6) output
    print(f"✅ Exercise segment from frame {start_frame} to {end_frame}")

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from pathlib import Path
import numpy as np
import pandas as pd

full_dir    = Path(r"ML/data/output_poses")
trimmed_dir = Path(r"ML/data/output_poses_preprocessed")

X_processed_flat, y_frames, video_ids = [], [], []
offsets = {}            # key → (start_idx, end_idx) in X_processed_flat

def smooth_sequence(seq, window=5):
    pad = window // 2
    seq_padded = np.pad(seq, ((pad, pad), (0, 0)), mode='edge')
    out = np.empty_like(seq)
    for i in range(len(seq)):
        out[i] = seq_padded[i:i+window].mean(axis=0)
    return out

full_files = sorted(full_dir.glob("*.csv"))
cursor = 0
for full_path in full_files:
    key = full_path.stem
    trim_path = trimmed_dir / f"{key}.csv"
    if not trim_path.exists():
        print(f"⚠️  {key}: trimmed file missing – skipped.")
        continue

    df_full = pd.read_csv(full_path).sort_values("FrameNo")
    df_trim = pd.read_csv(trim_path).sort_values("FrameNo")
    if df_full.empty or df_trim.empty:
        print(f"⚠️  {key}: empty CSV – skipped.")
        continue

    # Label frames
    start_fno, end_fno = df_trim["FrameNo"].iloc[[0, -1]]
    labels = df_full["FrameNo"].between(start_fno, end_fno).astype(int).to_numpy()

    pose_cols = [c for c in df_full.columns if c != "FrameNo"]
    raw_pose  = df_full[pose_cols].to_numpy(dtype=float)

    # Smoothing + delta features
    smoothed = smooth_sequence(raw_pose, window=5)
    deltas   = np.diff(smoothed, axis=0, prepend=smoothed[[0]])
    features = np.hstack([smoothed, deltas])

    X_processed_flat.append(features)
    y_frames.append(labels)
    video_ids.extend([key]*len(labels))

    offsets[key] = (cursor, cursor + len(features))  # record slice for later
    cursor += len(features)

    print(f"✅ {key}: {labels.sum()} exercise frames of {len(labels)} total.")

if not X_processed_flat:
    raise RuntimeError("No valid video pairs found.")

X_processed = np.vstack(X_processed_flat)   # (N_frames, D_feat)
y_frames    = np.concatenate(y_frames)      # (N_frames,)

# Now X_processed and y_frames are aligned.
# Standardize features
scaler = StandardScaler()
# Split train/test by video before fitting scaler, to avoid leaking test info in scaler.
train_idx, test_idx = train_test_split(np.arange(len(video_ids)), test_size=0.2, shuffle=True, 
                                       stratify=video_ids)  # stratify by video id so each split has some frames from each video; 
                                                             # alternatively, split by unique videos for a stricter separation.
X_train = X_processed[train_idx]
X_test = X_processed[test_idx]
y_train = y_frames[train_idx]
y_test = y_frames[test_idx]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build Keras Model (Fully Connected Binary Classifier)
def create_model(hidden_units=64, hidden_layers=1, dropout_rate=0.0, learning_rate=1e-3):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    # Add hidden layers
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(hidden_units, activation='relu'))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    # Output layer for binary classification
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Wrap the model with scikeras KerasClassifier for use in GridSearchCV
clf = KerasClassifier(model=create_model, verbose=0)  # we will set parameters via param_grid

# 4. Hyperparameter Tuning with GridSearchCV
param_grid = {
    "model__hidden_units": [64],        # number of neurons in each hidden layer
    "model__hidden_layers": [5],          # number of hidden layers
    "model__dropout_rate": [0.3],       # dropout to test (0 for none, 0.3 as an example)
    "model__learning_rate": [1e-4],    # a couple of learning rates to try
    "epochs": [20],   # you can increase this; using 20 for speed in example
    "batch_size": [32] 
}
# We use F1-score for selecting the best model. We'll define a scorer for binary F1.
from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score)

grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=f1_scorer, cv=3, n_jobs=-1, verbose=10)
grid.fit(X_train, y_train)

print("Best hyperparameters:", grid.best_params_)
best_model = grid.best_estimator_.model_  # Keras model from best estimator

# 5. Evaluate on Test Data
y_pred_prob = best_model.predict(X_test).reshape(-1)  # predict probabilities
y_pred = (y_pred_prob >= 0.5).astype(int)             # threshold at 0.5 for binary predictions

print("Test Precision:", precision_score(y_test, y_pred))
print("Test Recall:", recall_score(y_test, y_pred))
print("Test F1-score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# 6. Post-processing to get start and end frame indices for each video in test set
# Here we show how to derive start/end for one video sequence. In practice, loop over test videos.
unique_test_videos = set([video_ids[i] for i in test_idx])
for vid in unique_test_videos:
    start, end = offsets[vid]          # indices in X_processed
    features_scaled = scaler.transform(X_processed[start:end])
    pred_prob  = best_model.predict(features_scaled).ravel()
    pred_labels = (pred_prob >= 0.5).astype(int)
    segments = []
    in_segment = False
    seg_start = 0
    for i, lbl in enumerate(pred_labels):
        if lbl == 1 and not in_segment:
            in_segment = True
            seg_start = i
        if lbl == 0 and in_segment:
            # segment ended at i-1
            segments.append((seg_start, i-1))
            in_segment = False
    if in_segment:
        segments.append((seg_start, len(pred_labels)-1))

    if len(segments) == 0:
        print(f"No exercise detected in video {vid}")
    else:
        # Choose the longest segment as the exercise
        seg_lengths = [e - s + 1 for (s, e) in segments]
        best_idx = np.argmax(seg_lengths)
        start_frame, end_frame = segments[best_idx]
        print(f"Video {vid}: Predicted exercise segment from frame {start_frame} to {end_frame}")
        # If needed, you can compare with true start/end from the labels for this video.

#!/usr/bin/env python3
"""
exercise_segment_detector.py
────────────────────────────
Detect the start and end frame of an exercise segment in PoseNet CSV
data.  Two selectable training objectives:

    loss_mode = "frame"     → weighted-BCE per-frame classification
    loss_mode = "boundary"  → MSE regression of normalised start/end

Author: you
"""

# ------------------------------------------------------------------ #
# 0. CONFIGURE HERE                                                  #
# ------------------------------------------------------------------ #
loss_mode = "boundary"        # "frame"   or   "boundary"

# universal hyper-parameters
HIDDEN_UNITS  = 128
N_LAYERS      = 12
DROPOUT       = 0
LR            = 1e-4
EPOCHS        = 50
BATCH_SIZE    = 256
ALPHA         = 2.0        # weight strength near boundaries (frame mode)

# ------------------------------------------------------------------ #
# 1. Imports & seeds                                                 #
# ------------------------------------------------------------------ #
import warnings, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42); tf.random.set_seed(42)

# ------------------------------------------------------------------ #
# 2. Load & preprocess CSVs                                          #
# ------------------------------------------------------------------ #
DIR_FULL    = Path("ML/data/output_poses")
DIR_TRIMMED = Path("ML/data/output_poses_preprocessed")

def smooth(seq, w=5):
    pad = w//2
    seqp = np.pad(seq, ((pad,pad),(0,0)), mode='edge')
    out  = np.empty_like(seq)
    for i in range(len(seq)):
        out[i] = seqp[i:i+w].mean(0)
    return out

X_vids, y_vids, lens, keys = [], [], [], []
for p_full in sorted(DIR_FULL.glob("*.csv")):
    key = p_full.stem
    p_trim = DIR_TRIMMED / f"{key}.csv"
    if not p_trim.exists():
        print(f"⚠️ {key}: missing trim — skipped"); continue

    df_f = pd.read_csv(p_full).sort_values("FrameNo")
    df_t = pd.read_csv(p_trim).sort_values("FrameNo")
    if df_f.empty or df_t.empty:
        print(f"⚠️ {key}: empty — skipped"); continue

    s,e = df_t["FrameNo"].iloc[[0,-1]]
    y   = df_f["FrameNo"].between(s,e).astype(int).to_numpy()
    pose_cols = [c for c in df_f.columns if c!="FrameNo"]
    raw = df_f[pose_cols].to_numpy(float)
    sm  = smooth(raw,5)
    dlt = np.diff(sm,axis=0,prepend=sm[[0]])
    feats = np.hstack([sm,dlt])

    X_vids.append(feats); y_vids.append(y)
    lens.append(len(y));  keys.append(key)
    print(f"✅ {key}: {y.sum():3d}/{len(y):3d} exercise frames")

assert X_vids, "No valid videos found."

T_max   = max(lens)
F_dim   = X_vids[0].shape[1]
N       = len(X_vids)

X = np.zeros((N,T_max,F_dim), np.float32)
y = np.zeros((N,T_max)      , np.int8)
mask = np.zeros((N,T_max)   , bool)
for i,(f,l,L) in enumerate(zip(X_vids,y_vids,lens)):
    X[i,:L] = f;  y[i,:L] = l;  mask[i,:L] = True

# ------------------------------------------------------------------ #
# 3. Split & scale                                                   #
# ------------------------------------------------------------------ #
gss = GroupShuffleSplit(1,test_size=0.2,random_state=42)
train_idx, test_idx = next(gss.split(X, groups=np.arange(N)))
X_tr,X_te = X[train_idx],X[test_idx]
y_tr,y_te = y[train_idx],y[test_idx]
mask_tr,mask_te = mask[train_idx],mask[test_idx]

scaler = StandardScaler()
scaler.fit(X_tr[mask_tr])
X_tr = scaler.transform(X_tr.reshape(-1,F_dim)).reshape(X_tr.shape)
X_te = scaler.transform(X_te.reshape(-1,F_dim)).reshape(X_te.shape)

# ------------------------------------------------------------------ #
# 4. Extra targets for "boundary" mode                               #
# ------------------------------------------------------------------ #
def true_pos_norm(lbl):
    n,T = lbl.shape
    out = np.zeros((n,2),np.float32)
    for i in range(n):
        idx = np.where(lbl[i]==1)[0]
        if idx.size:
            out[i,0]=idx[0]/(T-1)
            out[i,1]=idx[-1]/(T-1)
    return out

if loss_mode=="boundary":
    y_tr_pos = true_pos_norm(y_tr)
    y_te_pos = true_pos_norm(y_te)

# ------------------------------------------------------------------ #
# 5. Custom weighted-BCE                                             #
# ------------------------------------------------------------------ #
def weighted_bce(y_true,y_pred):
    eps = K.epsilon()
    y_true = tf.squeeze(y_true,-1); y_pred=tf.squeeze(y_pred,-1)
    y_pred = K.clip(y_pred,eps,1-eps)
    bce = -(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))      # (B,T)

    T = tf.shape(y_true)[1]
    idx = tf.expand_dims(tf.range(T,dtype=tf.int32),0)            # (1,T)

    start = tf.cast(tf.argmax(y_true,1),tf.int32)                 # (B,)
    end   = T-1-tf.cast(tf.argmax(tf.reverse(y_true,[1]),1),tf.int32)
    start,end = tf.expand_dims(start,1),tf.expand_dims(end,1)
    dist = tf.minimum(tf.abs(idx-start), tf.abs(idx-end))
    dist = tf.cast(dist,tf.float32)
    w = 1.0 + ALPHA/(dist+1.0)

    loss_seq = tf.reduce_sum(bce*w,1)/tf.reduce_sum(w,1)
    return tf.reduce_mean(loss_seq)

# ------------------------------------------------------------------ #
# 6. Build model                                                     #
# ------------------------------------------------------------------ #
inp = keras.Input(shape=(T_max,F_dim))
x = layers.Masking()(inp)
for _ in range(N_LAYERS):
    x = layers.Dense(HIDDEN_UNITS,activation='relu')(x)
    x = layers.Dropout(DROPOUT)(x)

if loss_mode=="frame":
    out = layers.TimeDistributed(layers.Dense(1,activation='sigmoid'))(x)
    model = keras.Model(inp,out)
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss=weighted_bce,
                  metrics=['accuracy'])
    y_train = y_tr[...,None]   # (B,T,1)

else:  # boundary
    emb = layers.GlobalAveragePooling1D()(x)
    pos = layers.Dense(2,activation='sigmoid')(emb)
    model = keras.Model(inp,pos)
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss='mse',
                  metrics=['mae'])
    y_train = y_tr_pos

print(model.summary())

# ------------------------------------------------------------------ #
# 7. Train                                                           #
# ------------------------------------------------------------------ #
model.fit(X_tr, y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_split=0.1,
          verbose=2)

# ------------------------------------------------------------------ #
# 8. Helper to convert per-frame probs → segment                     #
# ------------------------------------------------------------------ #
def get_segment_from_probs(prob):
    lbl = (prob>=0.5).astype(int)
    # majority filter len-3
    for i in range(1,len(lbl)-1):
        if lbl[i-1]==lbl[i+1]!=lbl[i]:
            lbl[i]=lbl[i-1]
    if 1 not in lbl: return None,None,lbl
    s = int(np.argmax(lbl)); e = int(len(lbl)-1-np.argmax(lbl[::-1]))
    return s,e,lbl

# ------------------------------------------------------------------ #
# 9. Evaluate                                                        #
# ------------------------------------------------------------------ #
start_err,end_err,iou_list=[],[],[]
key_test = np.array(keys)[test_idx]

for i,vid in enumerate(key_test):
    L = lens[keys.index(vid)]

    if loss_mode=="frame":
        prob = model.predict(X_te[i:i+1])[0,:L,0]
        ps,pe,_ = get_segment_from_probs(prob)
    else:
        pos = model.predict(X_te[i:i+1])[0]
        ps,pe = int(round(pos[0]*(L-1))), int(round(pos[1]*(L-1)))
        if ps>pe: ps,pe = pe,ps

    true_lbl = y_te[i,:L]
    if 1 not in true_lbl:
        continue
    ts = int(np.argmax(true_lbl)); te=int(L-1-np.argmax(true_lbl[::-1]))

    if ps is None or pe is None:
        print(f"{vid:>6}:  GT[{ts:3d},{te:3d}] | Pred[None]  (miss)")
        continue

    start_err.append(abs(ps-ts)); end_err.append(abs(pe-te))
    inter = max(0,min(pe,te)-max(ps,ts)+1)
    union = (pe-ps+1)+(te-ts+1)-inter
    iou_list.append(inter/union)
    print(f"{vid:>6}:  GT[{ts:3d},{te:3d}] | Pred[{ps:3d},{pe:3d}]  "
          f"Δs {ps-ts:+3d}  Δe {pe-te:+3d}")

if start_err:
    print("\n──── Summary ────")
    print(f"Mean |Δstart| : {np.mean(start_err):.1f}")
    print(f"Mean |Δend|   : {np.mean(end_err):.1f}")
    print(f"Mean IoU      : {np.mean(iou_list):.3f}")
else:
    print("\nNo segments detected on test set.")

# ------------------------------------------------------------------ #
# 10. Save artefacts                                                 #
# ------------------------------------------------------------------ #
model.save(f"segment_model_{loss_mode}.keras")
joblib.dump(scaler,"feature_scaler.pkl")
print("💾  model and scaler saved.")

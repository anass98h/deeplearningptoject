import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1) Load the data (as in your original script)
# ----------------------------
def load_data():
    data_dir = 'ML/data/kinect_good_preprocessed'
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    
    seqs_X = []
    seqs_Z = []
    
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        # Drop frame columns if present
        df = df.drop(columns=['FrameNo','frame'], errors='ignore')
        df.columns = df.columns.str.strip()
        
        feature_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
        target_cols  = [col for col in df.columns if col.endswith('_z')]
        
        X_values = df[feature_cols].to_numpy()
        Z_values = df[target_cols].to_numpy()
        
        seqs_X.append(X_values)
        seqs_Z.append(Z_values)
    
    return seqs_X, seqs_Z

# ----------------------------
# 2) Utility: Create sliding windows
# ----------------------------
def create_windows_from_sequence(X_seq, Z_seq, window_size):
    X_windows = []
    Y_windows = []
    num_frames = X_seq.shape[0]
    if num_frames < window_size:
        return np.array(X_windows), np.array(Y_windows)
    for start in range(0, num_frames - window_size + 1):
        end = start + window_size
        X_windows.append(X_seq[start:end])
        Y_windows.append(Z_seq[end - 1])  # target from the last frame
    return np.array(X_windows), np.array(Y_windows)

# ----------------------------
# 3) Build your model (example: Dense)
# ----------------------------
def build_model(window_size, num_features, output_dim, num_layers, num_units, learning_rate):
    model = keras.Sequential()
    model.add(keras.Input(shape=(window_size, num_features)))
    model.add(keras.layers.Flatten())
    
    for _ in range(num_layers):
        model.add(keras.layers.Dense(num_units, activation='relu', kernel_initializer='he_uniform'))
    
    # Final layer for regression (linear activation)
    model.add(keras.layers.Dense(output_dim, activation='linear'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse', 
                  metrics=['mae'])
    return model

# ----------------------------
# 4) Metric Computation
# ----------------------------
def compute_regression_metrics(y_true, y_pred):
    """
    y_true, y_pred: shape (N, D)
    Returns dict with RSS, MAE, MSE, Cross-Entropy, KL-Div
    """
    eps = 1e-9  # for safe log
    
    # Residual Sum of Squares
    residuals = y_true - y_pred
    rss = np.sum(residuals**2)
    
    # MAE
    mae = np.mean(np.abs(residuals))
    
    # MSE
    mse = np.mean(residuals**2)
    
    # Convert any negative or zero predictions to a small positive to avoid log(0)
    # and create "pseudo" distributions for cross-entropy/KL
    y_true_safe = np.maximum(y_true, eps)
    y_pred_safe = np.maximum(y_pred, eps)
    
    # Normalize each row so it sums to 1 (makes them look like distributions)
    y_true_dist = y_true_safe / np.sum(y_true_safe, axis=1, keepdims=True)
    y_pred_dist = y_pred_safe / np.sum(y_pred_safe, axis=1, keepdims=True)
    
    # Cross-Entropy H = - Σ p(x) log q(x)
    cross_entropy = -np.sum(y_true_dist * np.log(y_pred_dist + eps), axis=1).mean()
    
    # KL-Divergence = Σ p(x) log [ p(x)/q(x) ]
    kl_div = np.sum(y_true_dist * np.log((y_true_dist + eps)/(y_pred_dist + eps)), axis=1).mean()
    
    return {
        'RSS': rss,
        'MAE': mae,
        'MSE': mse,
        'CrossEntropy': cross_entropy,
        'KL': kl_div
    }

# ----------------------------
# 5) Main script to train with a single set of hyperparams + 10-Fold CV
# ----------------------------
def main():
    # YOUR CHOSEN PARAMETERS (example)
    architecture = 'gru'
    window_size = 10
    num_layers = 6
    num_units = 128
    learning_rate = 0.001
    
    print("Loading data ...")
    seqs_X, seqs_Z = load_data()
    num_sequences = len(seqs_X)
    num_features = seqs_X[0].shape[1]
    output_dim   = seqs_Z[0].shape[1]
    
    print("Data loaded. Building 10-fold cross validation ...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # We'll collect metrics across folds
    all_metrics = []
    
    print(f"Training a single model with parameters:")
    print(f"  Architecture: {architecture}")
    print(f"  Window Size : {window_size}")
    print(f"  Layers      : {num_layers}")
    print(f"  Units       : {num_units}")
    print(f"  LR          : {learning_rate}")
    
    # For each fold
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(num_sequences)), start=1):
        # Create training data from train_idx sequences
        X_train_list, y_train_list = [], []
        for i in train_idx:
            X_seq = seqs_X[i]
            y_seq = seqs_Z[i]
            X_wins, y_wins = create_windows_from_sequence(X_seq, y_seq, window_size)
            if X_wins.size == 0:
                continue
            X_train_list.append(X_wins)
            y_train_list.append(y_wins)
        
        # Create testing data from test_idx sequences
        X_test_list, y_test_list = [], []
        for i in test_idx:
            X_seq = seqs_X[i]
            y_seq = seqs_Z[i]
            X_wins, y_wins = create_windows_from_sequence(X_seq, y_seq, window_size)
            if X_wins.size == 0:
                continue
            X_test_list.append(X_wins)
            y_test_list.append(y_wins)
        
        if len(X_train_list) == 0 or len(X_test_list) == 0:
            print(f"Fold {fold_idx}: Not enough data to train/test. Skipping.")
            continue
        
        X_train_all = np.vstack(X_train_list)
        y_train_all = np.vstack(y_train_list)
        X_test_all  = np.vstack(X_test_list)
        y_test_all  = np.vstack(y_test_list)
        
        # Scale the training data
        X_train_2d = X_train_all.reshape((-1, num_features))
        scaler = StandardScaler().fit(X_train_2d)
        X_train_scaled = scaler.transform(X_train_2d).reshape(X_train_all.shape)
        
        # Scale the testing data
        X_test_2d = X_test_all.reshape((-1, num_features))
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test_all.shape)
        
        # Build the model
        model = build_model(window_size, num_features, output_dim, num_layers, num_units, learning_rate)
        
        print(f"\n========== Fold {fold_idx} ==========")
        model.summary()  # Print model info nicely
        
        # Train (you can adjust callbacks, epochs, etc.)
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
        model.fit(X_train_scaled, y_train_all, 
                  epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping])
        
        # Create a directory for the model if it doesn't exist
        save_dir = 'saved_model'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the Keras model using .keras format
        model.save(os.path.join(save_dir, 'kinect_depth_model.keras'))
        print("Model saved successfully.")
        # Make predictions to calculate custom metrics
        y_pred = model.predict(X_test_scaled)
        
        # Compute metrics
        metrics = compute_regression_metrics(y_test_all, y_pred)
        
        # Print results
        print(f"Fold {fold_idx} metrics:")
        for mkey, mval in metrics.items():
            print(f"  {mkey} = {mval:.6f}")
        
        all_metrics.append(metrics)
    
    # 6) Print average metrics across folds
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        print("\n========== AVERAGE METRICS ACROSS FOLDS ==========")
        for mkey, mval in avg_metrics.items():
            print(f"  {mkey} = {mval:.6f}")
    else:
        print("No valid folds were processed. Please check your data or parameters.")

if __name__ == "__main__":
    main()

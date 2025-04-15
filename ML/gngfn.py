import os
# Suppress most TensorFlow info messages (set '2' to show warnings and errors only)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time, csv, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import multiprocessing as mp

# ----------------------------
# Global Data Loading (runs only in main)
# ----------------------------
def load_data():
    data_dir = 'ML/data/kinect_good_preprocessed'  # replace with your actual path
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    
    seqs_X = []
    seqs_Z = []
    
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {file}", flush=True)
        except Exception as e:
            print(f"Error loading {file}: {e}", flush=True)
            continue
    
        # Drop frame number columns if present
        if 'FrameNo' in df.columns or 'frame' in df.columns:
            df = df.drop(columns=['FrameNo'], errors='ignore')
            df = df.drop(columns=['frame'], errors='ignore')
    
        # Strip any whitespace from column names
        df.columns = df.columns.str.strip()
    
        # Separate feature and target columns based on their suffix
        feature_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
        target_cols  = [col for col in df.columns if col.endswith('_z')]
    
        X_values = df[feature_cols].to_numpy()
        Z_values = df[target_cols].to_numpy()
    
        seqs_X.append(X_values)
        seqs_Z.append(Z_values)
    
    num_sequences = len(seqs_X)
    if num_sequences == 0:
        raise ValueError("No sequences loaded. Please check your data directory.")
    print(f"Loaded {num_sequences} sequences. Example sequence shape: {seqs_X[0].shape}", flush=True)
    return seqs_X, seqs_Z

# ----------------------------
# Utility Function: Create Sliding Windows
# ----------------------------
def create_windows_from_sequence(X_seq, Z_seq, window_size):
    """
    Given a sequence (frames x features) and corresponding targets,
    return all sliding windows of length 'window_size'. The target for each 
    window is taken from the last frame.
    """
    X_windows = []
    Y_windows = []
    num_frames = X_seq.shape[0]
    if num_frames < window_size:
        return np.array(X_windows), np.array(Y_windows)
    for start in range(0, num_frames - window_size + 1):
        end = start + window_size
        X_windows.append(X_seq[start:end])
        Y_windows.append(Z_seq[end - 1])
    return np.array(X_windows), np.array(Y_windows)

# ----------------------------
# Model Builder
# ----------------------------
def build_model(architecture, window_size, num_layers, num_units, learning_rate):
    """
    Build and compile a Keras model based on the hyperparameters.
    Input shape is (window_size, num_features).
    Output dimension is set to output_dim.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    
    model = keras.Sequential()
    input_shape = (window_size, num_features)
    
    if architecture == 'dense':
        model.add(keras.Input(shape=input_shape))
        model.add(keras.layers.Flatten())
        for _ in range(num_layers):
            model.add(keras.layers.Dense(num_units, activation='relu', kernel_initializer='he_uniform'))
        model.add(keras.layers.Dense(output_dim, activation='linear'))
    elif architecture == 'lstm':
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            if i == 0:
                model.add(keras.layers.LSTM(num_units, return_sequences=return_seq, input_shape=input_shape))
            else:
                model.add(keras.layers.LSTM(num_units, return_sequences=return_seq))
        model.add(keras.layers.Dense(output_dim, activation='linear'))
    elif architecture == 'gru':
        for i in range(num_layers):
            return_seq = (i < num_layers - 1)
            if i == 0:
                model.add(keras.layers.GRU(num_units, return_sequences=return_seq, input_shape=input_shape))
            else:
                model.add(keras.layers.GRU(num_units, return_sequences=return_seq))
        model.add(keras.layers.Dense(output_dim, activation='linear'))
    elif architecture == 'cnn':
        for i in range(num_layers):
            if i == 0:
                model.add(keras.layers.Conv1D(filters=num_units, kernel_size=3, activation='relu',
                                              padding='same', input_shape=input_shape))
            else:
                model.add(keras.layers.Conv1D(filters=num_units, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(output_dim, activation='linear'))
    elif architecture == 'cnn_lstm':
        conv_layers = max(0, num_layers - 1)
        for i in range(conv_layers):
            if i == 0:
                model.add(keras.layers.Conv1D(filters=num_units, kernel_size=3, activation='relu',
                                              padding='same', input_shape=input_shape))
            else:
                model.add(keras.layers.Conv1D(filters=num_units, kernel_size=3, activation='relu', padding='same'))
        model.add(keras.layers.LSTM(num_units))
        model.add(keras.layers.Dense(output_dim, activation='linear'))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse', metrics=['mae'])
    return model

# ----------------------------
# Worker Initialization
# ----------------------------
def init_worker(seqs_X, seqs_Z, n_features, out_dim):
    global global_seqs_X, global_seqs_Z, num_features, output_dim
    global_seqs_X = seqs_X
    global_seqs_Z = seqs_Z
    num_features = n_features
    output_dim = out_dim

# ----------------------------
# Experiment Function (Called in Workers)
# ----------------------------
def run_experiment(architecture, window_size, num_layers, num_units, learning_rate):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []
    t0 = time.time()
    
    seq_indices = np.arange(len(global_seqs_X))
    for train_idx, test_idx in kf.split(seq_indices):
        X_train_list, y_train_list = [], []
        for i in train_idx:
            X_seq = global_seqs_X[i]
            y_seq = global_seqs_Z[i]
            X_wins, y_wins = create_windows_from_sequence(X_seq, y_seq, window_size)
            if X_wins.size == 0:
                continue
            X_train_list.append(X_wins)
            y_train_list.append(y_wins)
        if len(X_train_list) == 0:
            continue
        X_train_all = np.vstack(X_train_list)
        y_train_all = np.vstack(y_train_list)
        
        ns, ws, nf = X_train_all.shape
        X_train_2d = X_train_all.reshape((-1, nf))
        scaler = StandardScaler().fit(X_train_2d)
        X_train_scaled = scaler.transform(X_train_2d).reshape((ns, ws, nf))
        
        X_test_list, y_test_list = [], []
        for i in test_idx:
            X_seq = global_seqs_X[i]
            y_seq = global_seqs_Z[i]
            X_wins, y_wins = create_windows_from_sequence(X_seq, y_seq, window_size)
            if X_wins.size == 0:
                continue
            ns_test, ws_test, nf = X_wins.shape
            X_test_2d = X_wins.reshape((-1, nf))
            X_test_scaled = scaler.transform(X_test_2d).reshape((ns_test, ws_test, nf))
            X_test_list.append(X_test_scaled)
            y_test_list.append(y_wins)
        if len(X_test_list) == 0:
            continue
        X_test_all = np.vstack(X_test_list)
        y_test_all = np.vstack(y_test_list)
        
        model = build_model(architecture, window_size, num_layers, num_units, learning_rate)
        model.summary()
        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
        model.fit(X_train_scaled, y_train_all, epochs=10, batch_size=32, verbose=0, callbacks=[early_stopping])
        save_dir = 'saved_model'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the Keras model using .keras format
        model.save(os.path.join(save_dir, 'kinect_depth_model.keras'))
        print("Model saved successfully.")
        loss, mae = model.evaluate(X_test_all, y_test_all, verbose=0)
        mse_scores.append(loss)
        mae_scores.append(mae)
        keras.backend.clear_session()
        del model
        gc.collect()
    
    avg_mse = float(np.mean(mse_scores)) if mse_scores else float('nan')
    avg_mae = float(np.mean(mae_scores)) if mae_scores else float('nan')
    total_time = round(time.time() - t0, 2)
    
    result = [architecture, window_size, num_layers, num_units, learning_rate]
    for i in range(10):
        if i < len(mse_scores):
            result.append(mse_scores[i])
        else:
            result.append("")
    result.extend([avg_mse, avg_mae, total_time])
    gc.collect()
    return result

# ----------------------------
# Wrapper for Experiment (for logging and sequential dispatch)
# ----------------------------
def experiment_wrapper(params):
    arch, win, layers, units, lr = params
    print(f"Starting experiment: Architecture={arch}, Window={win}, Layers={layers}, Units={units}, LR={lr}", flush=True)
    return run_experiment(arch, win, layers, units, lr)

# ----------------------------
# Main: Grid Search with Resume Capability and Controlled Task Submission
# ----------------------------
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    global_seqs_X, global_seqs_Z = load_data()
    num_sequences = len(global_seqs_X)
    num_features = global_seqs_X[0].shape[1]
    output_dim = global_seqs_Z[0].shape[1]
    
    architectures = ['dense', 'gru', 'cnn']
    window_sizes = [5, 10, 15]
    num_layers_list = [5, 6, 8, 10, 12]
    num_units_list = [16, 32, 64, 128]
    learning_rates = [0.01, 0.005, 0.001, 0.0005]

    csv_filename = "experiment_results.csv"
    header = ["Architecture", "Window", "Layers", "Units", "LearningRate"] + \
             [f"Fold{i+1}_MSE" for i in range(10)] + ["Avg_MSE", "Avg_MAE", "Train_Time_sec"]
    done_experiments = set()
    
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    exp_id = (row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip())
                    done_experiments.add(exp_id)
        csv_file = open(csv_filename, 'a', newline='')
        writer = csv.writer(csv_file)
    else:
        csv_file = open(csv_filename, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(header)
        csv_file.flush()

    experiments_to_run = []
    for arch in architectures:
        for win in window_sizes:
            for layers in num_layers_list:
                for units in num_units_list:
                    for lr in learning_rates:
                        exp_id = (str(arch), str(win), str(layers), str(units), str(lr))
                        if exp_id in done_experiments:
                            print(f"Skipping experiment: Architecture={arch}, Window={win}, Layers={layers}, Units={units}, LR={lr}", flush=True)
                            continue
                        experiments_to_run.append((arch, win, layers, units, lr))
    
    print(f"Experiments remaining to run: {len(experiments_to_run)}", flush=True)
    
    num_workers = 1  # Adjust based on your available resources.
    pool = mp.Pool(processes=num_workers, initializer=init_worker, 
                   initargs=(global_seqs_X, global_seqs_Z, num_features, output_dim))
    
    # Use imap_unordered so that tasks are submitted sequentially to the pool
    for result_row in tqdm(pool.imap_unordered(experiment_wrapper, experiments_to_run),
                             total=len(experiments_to_run),
                             desc="Experiments"):
        writer.writerow(result_row)
        csv_file.flush()
        print(f"Completed: Architecture={result_row[0]}, Window={result_row[1]}, Layers={result_row[2]}, Units={result_row[3]}, LR={result_row[4]} | Train Time: {result_row[-1]} sec", flush=True)
        gc.collect()
        
    pool.close()
    pool.join()
    csv_file.close()
    print(f"Grid search completed. Results saved to {csv_filename}", flush=True)

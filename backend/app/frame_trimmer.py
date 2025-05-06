import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_distance(df, joint1_base, joint2_base):
    """
    Calculates the Euclidean distance between two joints.

    Args:
        df (pd.DataFrame): DataFrame containing joint coordinates.
        joint1_base (str): Base name of the first joint (e.g., "left_shoulder").
        joint2_base (str): Base name of the second joint (e.g., "right_shoulder").

    Returns:
        np.array: Array containing the distance between the two joints.
    """
    x1 = df[f"{joint1_base}_x"].values
    y1 = df[f"{joint1_base}_y"].values
    z1 = df[f"{joint1_base}_z"].values if f"{joint1_base}_z" in df else np.zeros_like(x1)  # Handle 2D case
    x2 = df[f"{joint2_base}_x"].values
    y2 = df[f"{joint2_base}_y"].values
    z2 = df[f"{joint2_base}_z"].values if f"{joint2_base}_z" in df else np.zeros_like(x2)  # Handle 2D case

    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def trim_frames(input_csv, output_csv=None, model_dir="models"):
    """
    Trim a sequence using the distance-feature enhanced model to identify the 
    relevant segment of motion data.
    
    Args:
        input_csv: Path to the input CSV file (Kinect 3D format)
        output_csv: Path to save the output CSV file (trimmed motion data)
        model_dir: Directory containing the model and scaler files
        
    Returns:
        Path to the saved CSV file and tuple with (start_index, end_index)
    """
    try:
        # If output_csv is not provided, generate one based on input_csv
        if output_csv is None:
            input_path = Path(input_csv)
            output_dir = input_path.parent / "trimmed_output"
            output_dir.mkdir(exist_ok=True)
            output_csv = output_dir / f"{input_path.stem}_trimmed.csv"
        
        # Set model and scaler paths
        model_path = Path(model_dir) / "kinect_cutting_model_with_distances.keras"
        scaler_path = Path(model_dir) / "kinect_cutting_scaler_with_distances.pkl"
        
        # Check if files exist
        if not model_path.exists():
            logger.warning(f"Cutting model not found: {model_path}")
            # Return the original data if model is not available
            df = pd.read_csv(input_csv)
            df.to_csv(output_csv, index=False)
            return output_csv, (0, len(df) - 1)
            
        if not scaler_path.exists():
            logger.warning(f"Scaler not found: {scaler_path}")
            # Return the original data if scaler is not available
            df = pd.read_csv(input_csv)
            df.to_csv(output_csv, index=False)
            return output_csv, (0, len(df) - 1)
        
        logger.info(f"Trimming frames in {input_csv}")
        
        # Load the CSV file
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {input_csv}: {len(df)} frames")
        
        # Make a copy of the dataframe for feature calculation
        df_features = df.drop(columns=["FrameNo"]).copy()
        df_features.columns = df_features.columns.str.strip()
        
        # Calculate the distance features
        df_features['torso_left_knee_distance'] = calculate_distance(df_features, 'right_shoulder', 'left_knee')
        df_features['left_hip_right_knee_distance'] = calculate_distance(df_features, 'left_hip', 'right_knee')
        
        # Load model and scaler
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = load(scaler_path)
        
        # Create windows
        n_frames = len(df_features)
        WINDOW = 11
        HALF = WINDOW // 2
        
        # If sequence is too short, return original
        if n_frames < WINDOW:
            logger.warning(f"Sequence too short for trimming ({n_frames} frames, need at least {WINDOW})")
            df.to_csv(output_csv, index=False)
            return output_csv, (0, n_frames - 1)
        
        win_feats, centres = [], []
        for idx in range(HALF, n_frames - HALF):
            win_feats.append(df_features.iloc[idx-HALF : idx+HALF+1].values.ravel())
            centres.append(idx)
        
        # Preprocess and predict
        X_win = scaler.transform(np.stack(win_feats))
        pred_prob = model.predict(X_win, verbose=0).ravel()
        pred_lbl = (pred_prob >= 0.5).astype(int)
        
        # Initialize frame predictions
        frame_pred = np.zeros(n_frames, dtype=int)
        
        # Fill in predictions at window centers
        for c_idx, lbl in zip(centres, pred_lbl):
            frame_pred[c_idx] = lbl
        
        # Apply smoothing to remove isolated predictions
        for i in range(1, n_frames-1):
            if frame_pred[i-1] == frame_pred[i+1] != frame_pred[i]:
                frame_pred[i] = frame_pred[i-1]
        
        # Find segments
        segments, in_seg = [], False
        for i, l in enumerate(frame_pred):
            if l == 1 and not in_seg:
                in_seg, seg_start = True, i
            if (l == 0 and in_seg) or (in_seg and i == n_frames-1):
                seg_end = i-1 if l == 0 else i
                segments.append((seg_start, seg_end))
                in_seg = False
        
        # If no segments found, return original sequence
        if not segments:
            logger.warning("No valid segments found, returning original sequence")
            df.to_csv(output_csv, index=False)
            return output_csv, (0, n_frames - 1)
        
        # Select the longest segment
        seg_lens = [e - s + 1 for s, e in segments]
        idx_long = int(np.argmax(seg_lens))
        start_idx, end_idx = segments[idx_long]
        
        # Log results
        logger.info(f"Cutting result: [{start_idx}, {end_idx}]")
        cut_percentage = (end_idx-start_idx+1)/len(df)
        logger.info(f"Cut length: {end_idx-start_idx+1} frames ({cut_percentage:.2f} of original)")
        
        # Get the trimmed sequence
        trimmed_df = df.iloc[start_idx:end_idx+1].copy()
        
        # Save the trimmed sequence
        trimmed_df.to_csv(output_csv, index=False)
        logger.info(f"Trimmed sequence saved to {output_csv}")
        
        return output_csv, (start_idx, end_idx)
        
    except Exception as e:
        logger.error(f"Error trimming frames: {e}")
        # If an error occurs, return the original data
        try:
            df = pd.read_csv(input_csv)
            df.to_csv(output_csv, index=False)
            return output_csv, (0, len(df) - 1)
        except:
            raise
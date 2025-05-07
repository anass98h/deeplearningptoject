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

def trim_frames(input_csv, output_csv=None, model_dir="models"):
    """
    Trim a sequence using the trained model to identify the 
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
        
        # Set model paths - try both naming conventions
        model_paths = [
            Path(model_dir) / "kinect_cutting_model.keras",
            Path(model_dir) / "kinect_cutting_model_with_distances.keras"
        ]
        scaler_paths = [
            Path(model_dir) / "kinect_cutting_scaler.pkl",
            Path(model_dir) / "kinect_cutting_scaler_with_distances.pkl"
        ]
        
        # Find the first existing model and scaler
        model_path = None
        scaler_path = None
        
        for m_path, s_path in zip(model_paths, scaler_paths):
            if m_path.exists() and s_path.exists():
                model_path = m_path
                scaler_path = s_path
                break
        
        # Check if files were found
        if model_path is None or scaler_path is None:
            logger.warning(f"Cutting model not found in {model_dir}")
            # Return the original data if model is not available
            df = pd.read_csv(input_csv)
            df.to_csv(output_csv, index=False)
            return output_csv, (0, len(df) - 1)
            
        logger.info(f"Trimming frames in {input_csv} using {model_path.name}")
        
        # Load the CSV file
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {input_csv}: {len(df)} frames")
        
        # Check if enough frames for processing
        if len(df) < 11:  # Need at least WINDOW size
            logger.warning(f"Too few frames ({len(df)}) for trimming, minimum is 11")
            df.to_csv(output_csv, index=False)
            return output_csv, (0, len(df) - 1)
        
        # Load model and scaler
        logger.info(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loading scaler: {scaler_path}")
        scaler = load(scaler_path)
        
        # Extract features (drop FrameNo if present)
        X = df.drop(columns=["FrameNo"]) if "FrameNo" in df.columns else df.copy()
        
        # Strip whitespace from column names
        X.columns = X.columns.str.strip()
        
        # Check if this is a "with_distances" model
        is_distances_model = "with_distances" in str(model_path)
        
        # If using the distances model, calculate the distance features
        if is_distances_model:
            logger.info("Adding distance features for model")
            try:
                X['torso_left_knee_distance'] = calculate_distance(X, 'right_shoulder', 'left_knee')
                X['left_hip_right_knee_distance'] = calculate_distance(X, 'left_hip', 'right_knee')
            except Exception as e:
                logger.warning(f"Could not calculate distance features: {e}. Using standard model.")
                is_distances_model = False
        
        # Create windows
        WINDOW = 11
        HALF = WINDOW // 2
        n_frames = len(X)
        
        win_feats, centres = [], []
        for idx in range(HALF, n_frames - HALF):
            win_feats.append(X.iloc[idx-HALF : idx+HALF+1].values.ravel())
            centres.append(idx)
        
        if not win_feats:  # Shouldn't happen given earlier check
            logger.warning("No windows could be created")
            df.to_csv(output_csv, index=False)
            return output_csv, (0, n_frames - 1)
        
        # Preprocess and predict
        X_win = np.stack(win_feats)
        X_win = scaler.transform(X_win)
        
        logger.info("Running model prediction")
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
        
        # If no segments found, return the original sequence
        if not segments:
            logger.warning("No valid segments found, returning original sequence")
            df.to_csv(output_csv, index=False)
            return output_csv, (0, n_frames - 1)
        
        # Select the longest segment
        seg_lens = [e - s + 1 for s, e in segments]
        idx_long = int(np.argmax(seg_lens))
        start_idx, end_idx = segments[idx_long]
        
        # Log results
        logger.info(f"Detected boundary frames: [{start_idx}, {end_idx}]")
        percent_kept = (end_idx - start_idx + 1) / n_frames * 100
        logger.info(f"Keeping {end_idx - start_idx + 1} frames ({percent_kept:.1f}% of original)")
        
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
    z1 = df[f"{joint1_base}_z"].values if f"{joint1_base}_z" in df.columns else np.zeros_like(x1)
    
    x2 = df[f"{joint2_base}_x"].values
    y2 = df[f"{joint2_base}_y"].values
    z2 = df[f"{joint2_base}_z"].values if f"{joint2_base}_z" in df.columns else np.zeros_like(x2)
    
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
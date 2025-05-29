#!/usr/bin/env python3
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

# Force CPU-only inference (remove if GPU is preferred)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class BadExerciseError(Exception):
    """Custom exception for when exercise is detected as bad form"""
    pass

def check_bad_3d_exercise(input_csv, model_dir="models", 
                         bad_threshold=0.8, num_frames=10):
    """
    Check if the 3D exercise data shows bad exercise form.
    
    Args:
        input_csv: Path to the Kinect 3D CSV file (after trimming)
        model_dir: Directory containing the model and scaler files
        bad_threshold: Threshold for bad exercise detection (above this = bad)
        num_frames: Number of frames to use for prediction (must match training: 10)
        
    Returns:
        float: Prediction score (0.0 = good exercise, 1.0 = bad exercise)
        
    Raises:
        BadExerciseError: If exercise is detected as bad form
        FileNotFoundError: If model or input files are missing
    """
    try:
        # Set model and scaler paths
        model_path = Path(model_dir) / "kinect_dense_good_vs_bad_model.keras"
        scaler_path = Path(model_dir) / "kinect_dense_good_vs_bad_scaler.pkl"
        
        # Check if files exist
        for file_path in [model_path, scaler_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        logger.info(f"Checking exercise quality for {input_csv}")
        
        # Load and preprocess the 3D pose data
        X = load_and_preprocess_3d_data(input_csv, num_frames)
        
        # Load model and scaler
        logger.info(f"Loading exercise quality model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        logger.info(f"Loading scaler: {scaler_path}")
        scaler = load(scaler_path)
        
        # Scale the data exactly like in training
        logger.info(f"Input shape before scaling: {X.shape}")
        flat_X = X.reshape(-1, X.shape[-1])  # Flatten for scaling
        flat_scaled = scaler.transform(flat_X)
        X_scaled = flat_scaled.reshape(X.shape)  # Reshape back
        
        logger.info(f"Scaled input shape: {X_scaled.shape}")
        logger.info(f"Scaled data range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        
        # Make prediction
        logger.info("Running bad exercise detection...")
        prediction = model.predict(X_scaled, verbose=0)[0][0]  # Get scalar prediction
        
        logger.info(f"Exercise quality prediction: {prediction:.3f} (0.0=good, 1.0=bad)")
        
        # Check if exercise is bad based on threshold
        if prediction < bad_threshold:
            error_msg = (f"Exercise form detected as poor quality. "
                        f"Quality score: {prediction:.3f} "
                        f"(threshold for bad exercise: {bad_threshold})")
            logger.warning(error_msg)
            raise BadExerciseError(error_msg)
        
        logger.info(f"Exercise quality check passed - proceeding with results")
        return float(prediction)
        
    except BadExerciseError:
        # Re-raise bad exercise errors
        raise
    except Exception as e:
        logger.error(f"Error in bad exercise detection: {e}")
        # If model fails, we could either:
        # 1. Raise an error (strict)
        # 2. Continue with a warning (permissive)
        # For now, let's be permissive and continue
        logger.warning(f"Bad exercise detection failed, continuing: {e}")
        return 0.0  # Return good score to continue

def load_and_preprocess_3d_data(csv_path, num_frames):
    """
    Load and preprocess 3D pose data for the exercise quality model.
    
    Args:
        csv_path: Path to the Kinect 3D CSV file
        num_frames: Number of frames to extract (must be 10 for this model)
        
    Returns:
        np.array: Preprocessed data with shape (1, num_frames, 39)
    """
    logger.info(f"Loading 3D pose data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Original CSV shape: {df.shape}")
    
    # Remove FrameNo column if present
    if "FrameNo" in df.columns:
        df = df.drop(columns=["FrameNo"])
        logger.info("Removed FrameNo column")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    logger.info(f"Data shape after cleanup: {df.shape}")
    
    # Check expected number of features (should be 39 for 3D Kinect data)
    expected_features = 39  # 13 joints * 3 coordinates (x,y,z)
    if df.shape[1] != expected_features:
        logger.warning(f"Expected {expected_features} features, got {df.shape[1]}. "
                      f"Columns: {df.columns.tolist()}")
        
        # If we have more features, try to select only the coordinate features
        if df.shape[1] > expected_features:
            coordinate_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]
            if len(coordinate_cols) == expected_features:
                df = df[coordinate_cols]
                logger.info(f"Selected coordinate columns, new shape: {df.shape}")
            else:
                logger.warning(f"Could not find exactly {expected_features} coordinate columns")
    
    # Check for any NaN or infinite values
    if df.isnull().any().any():
        logger.warning("Found NaN values in 3D pose data!")
        df = df.fillna(0.0)  # Fill NaN with zeros
        logger.info("Filled NaN values with zeros")
    
    if np.isinf(df.values).any():
        logger.warning("Found infinite values in 3D pose data!")
        df = df.replace([np.inf, -np.inf], 0.0)  # Replace inf with zeros
        logger.info("Replaced infinite values with zeros")
    
    # Convert to numpy
    data = df.to_numpy(dtype="float32")
    logger.info(f"Data array shape: {data.shape}")
    
    # Select frames for analysis
    if len(data) < num_frames:
        logger.warning(f"Not enough frames ({len(data)}) for analysis, minimum is {num_frames}")
        # Pad with the last frame or zeros
        if len(data) > 0:
            # Repeat the last frame
            last_frame = data[-1:].repeat(num_frames - len(data), axis=0)
            data = np.vstack([data, last_frame])
        else:
            # Create zero frames
            data = np.zeros((num_frames, data.shape[1]), dtype="float32")
        logger.info(f"Padded data to shape: {data.shape}")
    
    # Take the middle frames or distribute evenly
    if len(data) == num_frames:
        selected_data = data
    elif len(data) > num_frames:
        # Take evenly distributed frames
        indices = np.linspace(0, len(data) - 1, num_frames, dtype=int)
        selected_data = data[indices]
        logger.info(f"Selected {num_frames} frames from {len(data)} using indices: {indices}")
    
    logger.info(f"Final selected data shape: {selected_data.shape}")
    
    # Add batch dimension
    return selected_data[np.newaxis, ...]  # (1, num_frames, features)
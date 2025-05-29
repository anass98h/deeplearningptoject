#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
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

def score_exercise(input_csv, model_dir="models"):
    """
    Score the exercise quality using the trained CNN-LSTM model.
    
    Args:
        input_csv: Path to the Kinect 3D CSV file (after trimming)
        model_dir: Directory containing the model and scaler files
        
    Returns:
        float: Exercise score (0.0 to 4.0, where higher is better)
        
    Raises:
        FileNotFoundError: If model or input files are missing
    """
    try:
        # Configuration (must match training parameters)
        FRAMES = 48  # fixed frames per sequence
        
        # Set model and scaler paths
        model_path = Path(model_dir) / "scorer_model.keras"
        scaler_path = Path(model_dir) / "scorer_scaler.pkl"
        
        # Check if files exist
        for file_path in [model_path, scaler_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        logger.info(f"Scoring exercise for {input_csv}")
        
        # Load model and scaler
        logger.info(f"Loading scoring model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        logger.info(f"Loading scaler: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load and preprocess the sequence
        logger.info("Loading and preprocessing sequence data")
        seq = load_sequence(input_csv)
        
        # Sample/pad to exact number of frames
        sampled_seq = sample_sequence(seq, n_frames=FRAMES)
        logger.info(f"Sampled sequence shape: {sampled_seq.shape}")
        
        # Scale the sequence (frame-wise scaling)
        scaled_seq = scaler.transform(sampled_seq)
        logger.info(f"Scaled sequence shape: {scaled_seq.shape}")
        logger.info(f"Scaled data range: [{scaled_seq.min():.3f}, {scaled_seq.max():.3f}]")
        
        # Add batch dimension and predict
        x = np.expand_dims(scaled_seq, axis=0)  # (1, FRAMES, features)
        logger.info(f"Input shape for model: {x.shape}")
        
        logger.info("Running exercise scoring prediction...")
        y_pred = model.predict(x, verbose=0)[0, 0]
        
        # Ensure score is within expected range
        score = float(np.clip(y_pred, 0.0, 4.0))
        
        logger.info(f"Exercise score: {score:.3f} (0.0=perfect, 4.0=worst)")
        return score
        
    except Exception as e:
        logger.error(f"Error in exercise scoring: {e}")
        raise

def load_sequence(csv_path):
    """
    Load sequence data from CSV file (drop FrameNo column).
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        np.array: Sequence data as numpy array
    """
    logger.info(f"Loading sequence from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Original CSV shape: {df.shape}")
    
    # Remove FrameNo column if present
    if "FrameNo" in df.columns:
        df = df.drop(columns=["FrameNo"])
        logger.info("Removed FrameNo column")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    logger.info(f"Final data shape: {df.shape}")
    
    # Check for any NaN or infinite values
    if df.isnull().any().any():
        logger.warning("Found NaN values in sequence data!")
        df = df.fillna(0.0)  # Fill NaN with zeros
        logger.info("Filled NaN values with zeros")
    
    if np.isinf(df.values).any():
        logger.warning("Found infinite values in sequence data!")
        df = df.replace([np.inf, -np.inf], 0.0)  # Replace inf with zeros
        logger.info("Replaced infinite values with zeros")
    
    # Convert to numpy
    seq = df.to_numpy(dtype="float32")
    logger.info(f"Sequence shape: {seq.shape}")
    
    return seq

def sample_sequence(seq, n_frames=48):
    """
    Uniformly sample or pad `seq` to exactly `n_frames`.
    If longer, picks evenly spaced frames; if shorter, pads with zeros.
    
    Args:
        seq: Input sequence as numpy array (T, D)
        n_frames: Target number of frames
        
    Returns:
        np.array: Sampled/padded sequence (n_frames, D)
    """
    T, D = seq.shape
    logger.info(f"Sampling sequence from {T} frames to {n_frames} frames")
    
    if T >= n_frames:
        # Sample evenly spaced frames
        idx = np.linspace(0, T - 1, n_frames).round().astype(int)
        sampled = seq[idx]
        logger.info(f"Sampled {n_frames} frames using indices: {idx[:5]}...{idx[-5:]}")
    else:
        # Pad with zeros
        pad = np.zeros((n_frames - T, D), dtype=seq.dtype)
        sampled = np.vstack([seq, pad])
        logger.info(f"Padded {n_frames - T} frames with zeros")
    
    return sampled

def get_score_interpretation(score):
    """
    Provide human-readable interpretation of the exercise score.
    
    Args:
        score: Exercise score (0.0 to 4.0, where 0.0 is perfect and 4.0 is worst)
        
    Returns:
        str: Human-readable interpretation
    """
    if score <= 0.5:
        return "Excellent form"
    elif score <= 1.0:
        return "Very good form" 
    elif score <= 1.5:
        return "Good form"
    elif score <= 2.0:
        return "Fair form"
    elif score <= 2.5:
        return "Poor form"
    elif score <= 3.0:
        return "Very poor form"
    else:
        return "Extremely poor form"

def score_exercise_with_interpretation(input_csv, model_dir="models"):
    """
    Score exercise and return both numerical score and interpretation.
    
    Args:
        input_csv: Path to the Kinect 3D CSV file
        model_dir: Directory containing the model and scaler files
        
    Returns:
        tuple: (score, interpretation) where score is float and interpretation is str
    """
    score = score_exercise(input_csv, model_dir)
    interpretation = get_score_interpretation(score)
    return score, interpretation
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_to_kinect2d(input_csv, output_csv=None, model_dir="models"):
    """
    Convert pose data from PoseNet/MoveNet format to Kinect 2D format using a trained model.
    
    Args:
        input_csv: Path to the input CSV file (MoveNet format)
        output_csv: Path to save the output CSV file (Kinect 2D format)
        model_dir: Directory containing the model and scaler files
        
    Returns:
        Path to the saved CSV file
    """
    try:
        # If output_csv is not provided, generate one based on input_csv
        if output_csv is None:
            input_path = Path(input_csv)
            output_dir = input_path.parent / "kinect2d_output"
            output_dir.mkdir(exist_ok=True)
            output_csv = output_dir / f"{input_path.stem}_kinect2d.csv"
        
        # Set model paths
        model_path = Path(model_dir) / "xy_to_xy_best_2.keras"
        x_scaler_path = Path(model_dir) / "X_scaler_best.pkl"
        y_scaler_path = Path(model_dir) / "y_scaler_best.pkl"
        
        # Check if files exist
        for file_path in [model_path, x_scaler_path, y_scaler_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        logger.info(f"Converting {input_csv} to Kinect 2D format using model from {model_dir}")
        
        # Load input data
        df = pd.read_csv(input_csv)
        if "FrameNo" not in df.columns:
            raise ValueError("Input CSV must contain a 'FrameNo' column")
        
        frame_no = df["FrameNo"].values
        
        # Extract feature columns
        feature_cols = [c for c in df.columns if c != "FrameNo"]
        X = df[feature_cols].to_numpy(dtype=float)
        
        # Load scalers and model
        X_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        model = keras.models.load_model(model_path)
        
        # Scale, predict, and inverse-scale
        X_scaled = X_scaler.transform(X)
        y_pred_scaled = model.predict(X_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        
        # Build and save output
        df_out = pd.DataFrame(y_pred, columns=feature_cols)
        df_out.insert(0, "FrameNo", frame_no)
        df_out.to_csv(output_csv, index=False)
        
        logger.info(f"Successfully converted to Kinect 2D format, saved to {output_csv}")
        
        return output_csv
        
    except Exception as e:
        logger.error(f"Error converting to Kinect 2D format: {e}")
        raise
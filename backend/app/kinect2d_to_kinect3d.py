import pandas as pd
import numpy as np
import tensorflow as tf
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_depth_predictions(input_csv, output_csv=None, model_path=None, model_dir="models"):
    """
    Add z-coordinate predictions to a Kinect 2D format CSV file to create Kinect 3D data.
    
    Args:
        input_csv: Path to the input CSV file (Kinect 2D format)
        output_csv: Path to save the output CSV file (Kinect 3D format)
        model_path: Path to the Keras model for z-coordinate prediction
        model_dir: Directory containing the model (if model_path is not provided)
        
    Returns:
        Path to the saved CSV file
    """
    try:
        # If output_csv is not provided, generate one based on input_csv
        if output_csv is None:
            input_path = Path(input_csv)
            output_dir = input_path.parent / "kinect3d_output"
            output_dir.mkdir(exist_ok=True)
            output_csv = output_dir / f"{input_path.stem}_kinect3d.csv"
        
        # Set model path if not provided
        if model_path is None:
            model_path = Path(model_dir) / "kinect_depth_model.keras"
        
        # Check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Adding depth predictions to {input_csv} using model {model_path}")
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully")
        
        # Read the CSV
        df = pd.read_csv(input_csv)
        
        # Clean column names - strip whitespace
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Create a new DataFrame to store original data and predictions
        result_df = df.copy()
        
        # Lists to store predicted z values for each body part
        z_predictions = {
            'head_z': [],
            'left_shoulder_z': [],
            'left_elbow_z': [],
            'right_shoulder_z': [],
            'right_elbow_z': [],
            'left_hand_z': [],
            'right_hand_z': [],
            'left_hip_z': [],
            'right_hip_z': [],
            'left_knee_z': [],
            'right_knee_z': [],
            'left_foot_z': [],
            'right_foot_z': []
        }
        
        # Process each row
        logger.info(f"Processing {len(df)} rows...")
        
        # Function to prepare a single row for prediction
        def prepare_input_for_model(row):
            # Extract the features needed for prediction (all columns except FrameNo)
            features = row.drop('FrameNo', axis=0, errors='ignore').values
            
            # Reshape to match the expected input shape (None, 10, 26)
            features_expanded = np.tile(features, (10, 1))  # Duplicate row 10 times
            
            # Add batch dimension to get shape (1, 10, 26)
            return np.expand_dims(features_expanded, axis=0)
        
        # Process rows in batches to show progress
        batch_size = 10
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            batch = df.iloc[i:end_idx]
            
            for idx, row in batch.iterrows():
                # Prepare input for model
                input_data = prepare_input_for_model(row)
                
                # Get predictions from model
                predictions = model.predict(input_data, verbose=0)[0]
                
                # If predictions have multiple time steps, take the last one
                if len(predictions.shape) > 1:
                    predictions = predictions[-1]  # Take the last timestep prediction
                
                # Assuming the model outputs all z values in order
                for j, key in enumerate(z_predictions.keys()):
                    z_predictions[key].append(predictions[j])
            
            logger.info(f"Processed {end_idx}/{len(df)} rows")
        
        # Add predicted z values to the result DataFrame
        for z_column, values in z_predictions.items():
            result_df[z_column] = values
        
        # Desired column order (with consistent naming - no spaces)
        columns_order = ['FrameNo', 
                        'head_x', 'head_y', 'head_z',
                        'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                        'left_elbow_x', 'left_elbow_y', 'left_elbow_z',
                        'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
                        'right_elbow_x', 'right_elbow_y', 'right_elbow_z',
                        'left_hand_x', 'left_hand_y', 'left_hand_z',
                        'right_hand_x', 'right_hand_y', 'right_hand_z',
                        'left_hip_x', 'left_hip_y', 'left_hip_z',
                        'right_hip_x', 'right_hip_y', 'right_hip_z',
                        'left_knee_x', 'left_knee_y', 'left_knee_z',
                        'right_knee_x', 'right_knee_y', 'right_knee_z',
                        'left_foot_x', 'left_foot_y', 'left_foot_z',
                        'right_foot_x', 'right_foot_y', 'right_foot_z']
        
        # Filter to only include columns that actually exist in our DataFrame
        available_columns = [col for col in columns_order if col in result_df.columns]
        
        # Check for missing columns
        missing_columns = [col for col in columns_order if col not in result_df.columns]
        if missing_columns:
            logger.warning(f"These columns were expected but not found: {missing_columns}")
        
        # Check for extra columns
        extra_columns = [col for col in result_df.columns if col not in columns_order]
        if extra_columns:
            logger.warning(f"These columns were found but not in the ordering: {extra_columns}")
            available_columns.extend(extra_columns)
        
        # Reorder columns
        result_df = result_df[available_columns]
        
        # Save the result to the output CSV file
        result_df.to_csv(output_csv, index=False)
        logger.info(f"Depth predictions added successfully, results saved to {output_csv}")
        
        return output_csv
        
    except Exception as e:
        logger.error(f"Error adding depth predictions: {e}")
        raise
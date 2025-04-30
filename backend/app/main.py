from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel
import logging
from app.model_loader import ModelLoader, ModelInfo
import time
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi.responses import PlainTextResponse
from pathlib import Path
import asyncio
import random

# Define the directory to save Json files
POSENET_DATA = Path("posenet_data")
POSENET_DATA.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

class WeakLink(str, Enum):
    FORWARD_HEAD = "ForwardHead"
    LEFT_ARM_FALL_FORWARD = "LeftArmFallForward"
    RIGHT_ARM_FALL_FORWARD = "RightArmFallForward"
    LEFT_SHOULDER_ELEVATION = "LeftShoulderElevation"
    RIGHT_SHOULDER_ELEVATION = "RightShoulderElevation"
    EXCESSIVE_FORWARD_LEAN = "ExcessiveForwardLean"
    LEFT_ASYMMETRICAL_WEIGHT_SHIFT = "LeftAsymmetricalWeightShift"
    RIGHT_ASYMMETRICAL_WEIGHT_SHIFT = "RightAsymmetricalWeightShift"
    LEFT_KNEE_MOVES_INWARD = "LeftKneeMovesInward"
    RIGHT_KNEE_MOVES_INWARD = "RightKneeMovesInward"
    LEFT_KNEE_MOVES_OUTWARD = "LeftKneeMovesOutward"
    RIGHT_KNEE_MOVES_OUTWARD = "RightKneeMovesOutward"
    LEFT_HEEL_RISES = "LeftHeelRises"
    RIGHT_HEEL_RISES = "RightHeelRises"

class WeakestLinkResponse(BaseModel):
    model_name: str
    weakest_link: WeakLink

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

# Response models
class PredictionResponse(BaseModel):
    model_name: str
    category: str
    score: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    available_models: List[str]

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

class ComparisonMetrics(BaseModel):
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error

class KeypointMetrics(BaseModel):
    keypoint: str
    mae: float
    mse: float

class DepthPredictionResponse(BaseModel):
    model_name: str
    z_values: Dict[str, List[float]]
    processing_time_ms: float
    framework: str
    sequence_length: int
    has_ground_truth: bool
    overall_metrics: Optional[ComparisonMetrics] = None
    keypoint_metrics: Optional[List[KeypointMetrics]] = None
    ground_truth: Optional[Dict[str, List[float]]] = None

def categorize_score(score: float) -> str:
    """Categorize a score based on defined thresholds."""
    if score < 40:
        return "Bad"
    elif score < 70:
        return "Good"
    elif score < 90:
        return "Great"
    else:
        return "Excellent"

app = FastAPI(title="Model Serving API")
model_loader = ModelLoader()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    """Load all models during startup."""
    try:
        model_loader.load_models()
    except Exception as e:
        logging.error(f"Failed to load models during startup: {str(e)}")

@app.post("/predict/{model_name}", 
         response_model=PredictionResponse,
         responses={
             404: {"model": ErrorResponse},
             500: {"model": ErrorResponse}
         })
async def predict(
    model_name: str,
    file: UploadFile = File(...)
):
    # First check if model exists
    model = model_loader.get_model(model_name)
    model_info = model_loader.get_model_info(model_name)
    
    if not model or not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    try:
        # Read and process the file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Make prediction
        prediction = model.predict(df)
        score = float(np.clip(prediction[0] , 0, 1))
        category = categorize_score(score * 100)
        
        return PredictionResponse(
            model_name=model_name,
            category=category,
            score=score
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"error_type": "prediction_error"}
        )

@app.get("/models", 
         response_model=List[ModelInfo],
         responses={
             500: {"model": ErrorResponse}
         })
async def list_regression_models():
    """List all regression models."""
    try:
        # Only return regression models
        return model_loader.list_models(model_type="regression")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list regression models: {str(e)}",
            headers={"error_type": "model_list_error"}
        )


@app.post("/refresh-models", 
         response_model=dict,
         responses={
             500: {"model": ErrorResponse}
         })
async def refresh_models(background_tasks: BackgroundTasks):
    """Refresh models from the models directory."""
    try:
        background_tasks.add_task(model_loader.load_models)
        return {"message": "Model refresh started"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model refresh: {str(e)}",
            headers={"error_type": "refresh_error"}
        )

@app.get("/health",
         response_model=HealthResponse,
         responses={
             500: {"model": ErrorResponse}
         })
async def health():
    """Health check endpoint."""
    try:
        # Check if both directories exist
        if not model_loader.regression_dir.exists() or not model_loader.classifier_dir.exists():
            return HealthResponse(
                status="unhealthy",
                models_loaded=0,
                available_models=[],
            )
        
        # Check if we have any models loaded
        models = model_loader.list_models()  # This gets all models, both regression and classifier
        if not models:
            return HealthResponse(
                status="degraded",
                models_loaded=0,
                available_models=[],
            )
        
        # Everything is fine
        return HealthResponse(
            status="healthy",
            models_loaded=len(model_loader.models),
            available_models=[info.name for info in models]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}",
            headers={"error_type": "health_check_error"}
        )

@app.get("/categorizing-models", 
         response_model=List[ModelInfo],
         responses={
             500: {"model": ErrorResponse}
         })
async def list_categorizing_models():
    """List all models used for weakest link categorization."""
    try:
        # Only return classifier models
        return model_loader.list_models(model_type="classifier")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list categorizing models: {str(e)}",
            headers={"error_type": "model_list_error"}
        )

class WeakestLinkResponse(BaseModel):
    model_name: str
    weakest_link: WeakLink
    processing_time_ms: float  # Added field for processing time in milliseconds

@app.post("/classify-weakest-link/{model_name}", 
         response_model=WeakestLinkResponse,
         responses={
             404: {"model": ErrorResponse},
             500: {"model": ErrorResponse},
             422: {"model": ErrorResponse}
         })
async def classify_weakest_link(
    model_name: str,
    file: UploadFile = File(...)
):
    # Get model and verify it's a classifier
    model_info = model_loader.get_model_info(model_name)
    if not model_info or model_info.model_type != "classifier":
        raise HTTPException(
            status_code=404,
            detail=f"Classifier model '{model_name}' not found"
        )
    
    model = model_loader.get_model(model_name)
    
    try:
        
        
        # Read and process the file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        # Start timing
        start_time = time.time()
        # Make prediction
        predicted_class = model.predict(df)[0]
        
        # End timing and calculate duration in milliseconds
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        return WeakestLinkResponse(
            model_name=model_name,
            weakest_link=predicted_class,
            processing_time_ms=processing_time_ms 
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"error_type": "prediction_error"}
        )

@app.post("/upload-posenet-data", 
          response_model=dict,
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def upload_posenet_data(file: UploadFile = File(...)):
    """
    Endpoint to upload PoseNet JSON data and save it to the server.
    """
    try:
        # Validate file type
        if not file.filename.endswith(".json"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only JSON files are allowed.",
                headers={"error_type": "file_format_error"}
            )
        
        # Save the file to the posenet_data directory
        file_path = POSENET_DATA / file.filename
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        return {"message": f"File '{file.filename}' uploaded successfully."}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}",
            headers={"error_type": "upload_error"}
        )
    

    
@app.get("/depth-models", 
         response_model=List[ModelInfo],
         responses={
             500: {"model": ErrorResponse}
         })
async def list_depth_models():
    """List all depth estimation models."""
    try:
        # Only return depth estimation models
        return model_loader.list_models(model_type="depth_estimator")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list depth estimation models: {str(e)}",
            headers={"error_type": "model_list_error"}
        )
    
@app.post("/predict-depth/{model_name}", 
         response_model=DepthPredictionResponse,
         responses={
             404: {"model": ErrorResponse},
             500: {"model": ErrorResponse},
             400: {"model": ErrorResponse}
         })
async def predict_depth(
    model_name: str,
    file: UploadFile = File(...),
    include_ground_truth: bool = True
):
    """
    Endpoint to predict z-coordinates (depth) for skeletal keypoints.
    
    Takes a CSV file that can optionally include ground truth z-values.
    If z-values are present, they will be used for comparison metrics.
    Works with any number of rows in the input CSV.
    """
    # First check if model exists
    model = model_loader.get_model(model_name)
    model_info = model_loader.get_model_info(model_name)
    
    if not model or not model_info:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    # Verify it's a depth estimator model
    if model_info.model_type != "depth_estimator":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not a depth estimator model"
        )
    
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents), sep=None, engine='python')
        
        # Determine if the data contains z values
        all_columns = df.columns.tolist()
        x_y_columns = [col for col in all_columns if not col.endswith('_z')]
        z_columns = [col for col in all_columns if col.endswith('_z')]
        
        has_ground_truth = len(z_columns) > 0
        ground_truth_data = None
        
        # If we have z columns, save them for comparison
        if has_ground_truth:
            ground_truth_data = df[z_columns].copy()
        
        # Get the actual sequence length
        actual_sequence_length = len(df)
        
        start_time = time.time()
        
        # Get only the required columns for prediction (x and y coordinates)
        input_data = df[x_y_columns].copy()
        
        required_sequence_length = 10
        
        # Different handling based on model framework
        framework = model_info.framework
        
        if framework == "keras":
            # For Keras models, we need to convert to numpy array and ensure proper shape
            input_array = input_data.values
            
            # Special handling for single row input - duplicate to 10 rows
            if actual_sequence_length == 1:
                # Duplicate the single row to get 10 identical rows
                single_row = input_array[0]
                input_array = np.tile(single_row, (required_sequence_length, 1))
                logging.info(f"Duplicated single row to {required_sequence_length} rows")
            # If we have 2-9 rows, also duplicate to 10
            elif 1 < actual_sequence_length < required_sequence_length:
                # Calculate how many times we need to repeat the data
                repetitions = int(np.ceil(required_sequence_length / actual_sequence_length))
                # Create repeated array and trim to exactly 10 rows
                repeated_array = np.tile(input_array, (repetitions, 1))
                input_array = repeated_array[:required_sequence_length]
                logging.info(f"Extended {actual_sequence_length} rows to {required_sequence_length} rows")
            # If we have more than 10 rows, use just the first 10
            elif actual_sequence_length > required_sequence_length:
                input_array = input_array[:required_sequence_length]
                logging.info(f"Using first {required_sequence_length} rows from {actual_sequence_length} total")
                if has_ground_truth:
                    ground_truth_data = ground_truth_data.iloc[:required_sequence_length]
            
            # Always ensure we have batch dimension
            input_array = np.expand_dims(input_array, axis=0)  # Now (1, sequence_length, features)
            
            # Make prediction with Keras model
            z_predictions = model.predict(input_array)
            
            # If model returns a 3D tensor, get the first batch
            if len(z_predictions.shape) == 3:
                z_predictions = z_predictions[0]  # Now (sequence_length, num_keypoints)
        else:
            # For sklearn models
            z_predictions = model.predict(input_data)
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Organize predictions by keypoint
        keypoints = ['head', 'left_shoulder', 'left_elbow', 'right_shoulder', 
                    'right_elbow', 'left_hand', 'right_hand', 'left_hip', 
                    'right_hip', 'left_knee', 'right_knee', 'left_foot', 'right_foot']
        
        z_values = {}
        # If original data had fewer rows, only return that many predictions
        output_rows = min(actual_sequence_length, z_predictions.shape[0])
        
        for i, keypoint in enumerate(keypoints):
            # Only return as many rows as the original input had
            if actual_sequence_length < required_sequence_length:
                z_values[f"{keypoint}_z"] = z_predictions[:output_rows, i].tolist()
            else:
                z_values[f"{keypoint}_z"] = z_predictions[:, i].tolist()
        
        # Compare predictions with ground truth if available
        overall_metrics = None
        keypoint_metrics = None
        ground_truth_dict = None
        
        if has_ground_truth:
            # Prepare ground truth dictionary
            ground_truth_dict = {}
            for col in z_columns:
                ground_truth_dict[col] = ground_truth_data[col].tolist()
            
            # Calculate overall metrics (across all keypoints)
            all_predictions = []
            all_ground_truth = []
            
            # Gather metrics for each keypoint and calculate overall metrics
            keypoint_metrics = []
            
            for i, keypoint in enumerate(keypoints):
                z_col = f"{keypoint}_z"
                if z_col in z_columns:
                    predictions = np.array(z_values[z_col])
                    truth = np.array(ground_truth_dict[z_col])
                    
                    # Make sure predictions and truth have the same length
                    min_length = min(len(predictions), len(truth))
                    if min_length > 0:  # Only calculate if we have valid data
                        predictions = predictions[:min_length]
                        truth = truth[:min_length]
                        
                        # Add to the overall lists
                        all_predictions.extend(predictions)
                        all_ground_truth.extend(truth)
                        
                        # Calculate metrics for this keypoint - no correlation
                        mae = float(mean_absolute_error(truth, predictions))
                        mse = float(mean_squared_error(truth, predictions))
                        
                        keypoint_metrics.append(KeypointMetrics(
                            keypoint=keypoint,
                            mae=mae,
                            mse=mse
                        ))
            
            # Calculate overall metrics - no correlation
            if all_predictions and all_ground_truth:
                # Convert to arrays and ensure they have the same length
                all_predictions = np.array(all_predictions)
                all_ground_truth = np.array(all_ground_truth)
                
                # Ensure they have the same shape
                all_predictions = all_predictions.reshape(-1)
                all_ground_truth = all_ground_truth.reshape(-1)
                
                # Calculate metrics
                overall_mae = float(mean_absolute_error(all_ground_truth, all_predictions))
                overall_mse = float(mean_squared_error(all_ground_truth, all_predictions))
                
                overall_metrics = ComparisonMetrics(
                    mae=overall_mae,
                    mse=overall_mse
                )
        
        # Return the original number of rows in sequence_length
        return DepthPredictionResponse(
            model_name=model_name,
            z_values=z_values,
            processing_time_ms=processing_time_ms,
            framework=framework,
            sequence_length=actual_sequence_length,  # Return the original sequence length
            has_ground_truth=has_ground_truth,
            overall_metrics=overall_metrics,
            keypoint_metrics=keypoint_metrics,
            ground_truth=ground_truth_dict if include_ground_truth else None
        )
    
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        logging.error(error_detail)
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"error_type": "prediction_error"}
        )
    

@app.get("/kinect-data", 
         responses={
             404: {"model": ErrorResponse},
             500: {"model": ErrorResponse}
         })
async def get_kinect_data():
    """
    Simple endpoint to serve the A1_kinect.csv data for posenet visualization.
    Returns the file content directly from disk. The file should be placed in the 'data' directory.
    """
    try:
        # Path to the CSV file
        csv_path = Path("./A1_kinect.csv")
        
        # Check if file exists
        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="A1_kinect.csv file not found. Please place the file in the 'data' directory.",
                headers={"error_type": "file_not_found"}
            )
        
        # Simply read and return the file contents
        with open(csv_path, "r") as f:
            csv_content = f.read()
        
        # Return the raw file content
        return {
            "content": csv_content
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving CSV file: {str(e)}",
            headers={"error_type": "file_read_error"}
        )
    
@app.post("/trim-frames", response_class=PlainTextResponse)
async def trim_frames(file: UploadFile = File(...)):
    try:

        # Path to the CSV file
        csv_path = Path("./tests/output.csv")
        
        # Check if file exists
        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="output.csv file not found. Please place the file in the root directory.",
                headers={"error_type": "file_not_found"}
            )
        
        logging.info(f"Returning pre-trimmed data from {csv_path}")
        
        # Simply read and return the file contents
        with open(csv_path, "r") as f:
            csv_content = f.read()
        
        return csv_content
    
    except Exception as e:
        logging.error(f"Error serving trimmed CSV file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trim frames: {str(e)}",
            headers={"error_type": "processing_error"}
        )
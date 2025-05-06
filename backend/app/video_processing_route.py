from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import tempfile
import shutil
from pathlib import Path
import uuid
import time

# Import the processing functions
from movenet_extraction import process_video_file
from posenet_to_kinect2d import convert_to_kinect2d
from kinect2d_to_kinect3d import add_depth_predictions
from frame_trimmer import trim_frames

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directories
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
KINECT2D_DIR = Path("./outputs/kinect2d")
KINECT2D_DIR.mkdir(exist_ok=True)
KINECT3D_DIR = Path("./outputs/kinect3d")
KINECT3D_DIR.mkdir(exist_ok=True)
TRIMMED_DIR = Path("./outputs/trimmed")
TRIMMED_DIR.mkdir(exist_ok=True)

# Model directory
MODEL_DIR = Path("./models")

# Track processing jobs
jobs = {}

@app.post("/process-video")
async def process_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Endpoint to process a video file through the complete pipeline:
    1. Extract poses with MoveNet
    2. Convert to Kinect 2D format
    3. Add depth predictions for Kinect 3D format
    4. Trim frames to identify the relevant segment
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        #job_id = "test"  # Shortened UUID for easier tracking
        
        # Create a job directory for this specific upload
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Log the received file
        logger.info(f"Received video file: {file.filename} (job ID: {job_id})")
        
        # Create a file to store the uploaded video
        video_path = job_dir / file.filename
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Initialize job data
        jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "uploaded",
            "message": "File uploaded, processing will begin shortly",
            "created_at": time.time(),
            "updated_at": time.time(),
            "files": {
                "video": str(video_path),
                "original": None,
                "kinect2d": None,
                "kinect3d": None,
                "trimmed": None
            }
        }
        
        # Define the background task function
        def process_video_task(job_id, file_path):
            try:
                # Step 1: Process video with MoveNet to extract pose data
                update_job_status(job_id, "processing", "Extracting poses with MoveNet")

                movenet_output_filename = f"{Path(file_path).stem}_movenet.csv"
                movenet_output_path = job_dir / movenet_output_filename  # Construct the full path
                process_video_file(file_path, output_dir=str(movenet_output_path)) # Pass the full path as a string

                logger.info(f"MoveNet processing complete, results saved to {movenet_output_path}")
                jobs[job_id]["files"]["original"] = str(movenet_output_path)
                
                # Check if model directory exists for subsequent steps
                if not MODEL_DIR.exists():
                    logger.warning(f"Model directory {MODEL_DIR} not found, skipping conversion steps")
                    update_job_status(job_id, "completed", "Processing completed (without models)")
                    return
                
                # Step 2: Convert MoveNet output to Kinect 2D format
                try:
                    update_job_status(job_id, "processing", "Converting to Kinect 2D format")
                    
                    kinect2d_output_path = job_dir / f"{Path(file_path).stem}_kinect2d.csv"
                    print("Kinect Path" , kinect2d_output_path)
                    convert_to_kinect2d(
                        input_csv=movenet_output_path,
                        output_csv=kinect2d_output_path,
                        model_dir=MODEL_DIR
                    )
                    
                    logger.info(f"Kinect 2D conversion complete, results saved to {kinect2d_output_path}")
                    jobs[job_id]["files"]["kinect2d"] = str(kinect2d_output_path)
                    
                    # Step 3: Add depth predictions to get Kinect 3D format
                    try:
                        update_job_status(job_id, "processing", "Adding depth predictions for Kinect 3D format")
                        
                        kinect3d_output_path = job_dir / f"{Path(file_path).stem}_kinect3d.csv"
                        add_depth_predictions(
                            input_csv=kinect2d_output_path,
                            output_csv=kinect3d_output_path,
                            model_dir=MODEL_DIR
                        )
                        
                        logger.info(f"Kinect 3D conversion complete, results saved to {kinect3d_output_path}")
                        jobs[job_id]["files"]["kinect3d"] = str(kinect3d_output_path)
                        
                        # Step 4: Trim frames to identify the relevant segment
                        # This step uses the Kinect 3D data specifically
                        try:
                            update_job_status(job_id, "processing", "Trimming frames to identify relevant segment")
                            
                            trimmed_output_path = job_dir / f"{Path(file_path).stem}_trimmed.csv"
                            trim_result, trim_indices = trim_frames(
                                input_csv=kinect3d_output_path,  # Specifically use Kinect 3D data
                                output_csv=trimmed_output_path,
                                model_dir=MODEL_DIR
                            )
                            
                            logger.info(f"Frame trimming complete, results saved to {trimmed_output_path}")
                            logger.info(f"Trim indices: [{trim_indices[0]}, {trim_indices[1]}]")
                            
                            jobs[job_id]["files"]["trimmed"] = str(trimmed_output_path)
                            update_job_status(job_id, "completed", "Processing completed successfully")
                            
                        except Exception as e:
                            logger.error(f"Error in frame trimming: {str(e)}")
                            # If trimming fails, use the Kinect 3D data
                            jobs[job_id]["files"]["trimmed"] = jobs[job_id]["files"]["kinect3d"]
                            update_job_status(job_id, "completed", "Processing completed with errors in trimming")
                    except Exception as e:
                        logger.error(f"Error in Kinect 3D conversion: {str(e)}")
                        # If 3D conversion fails, we cannot do trimming which requires 3D data
                        jobs[job_id]["files"]["kinect3d"] = jobs[job_id]["files"]["kinect2d"]
                        jobs[job_id]["files"]["trimmed"] = jobs[job_id]["files"]["kinect2d"]
                        update_job_status(job_id, "completed", "Processing completed with errors in 3D conversion")
                except Exception as e:
                    logger.error(f"Error in Kinect 2D conversion: {str(e)}")
                    # If 2D conversion fails, we cannot proceed to 3D or trimming
                    jobs[job_id]["files"]["kinect2d"] = jobs[job_id]["files"]["original"]
                    jobs[job_id]["files"]["kinect3d"] = jobs[job_id]["files"]["original"]
                    jobs[job_id]["files"]["trimmed"] = jobs[job_id]["files"]["original"]
                    update_job_status(job_id, "completed", "Processing completed with errors in 2D conversion")
                
            except Exception as e:
                logger.error(f"Error in background processing: {str(e)}")
                update_job_status(job_id, "failed", f"Processing failed: {str(e)}")
        
        # Add the task to the background tasks
        background_tasks.add_task(process_video_task, job_id, str(video_path))
        
        # Return the job ID for tracking
        return {"job_id": job_id, "message": "Video upload received, processing started in background"}
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process video: {str(e)}",
            headers={"error_type": "processing_error"}
        )

def update_job_status(job_id, status, message):
    """Update the status of a job"""
    if job_id in jobs:
        jobs[job_id]["status"] = status
        jobs[job_id]["message"] = message
        jobs[job_id]["updated_at"] = time.time()

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a job
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
            headers={"error_type": "job_not_found"}
        )
    
    return jobs[job_id]

@app.get("/video-data/{job_id}/original", response_class=PlainTextResponse)
async def get_original_data(job_id: str):
    """
    Get the original MoveNet data for a specific job
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
            headers={"error_type": "job_not_found"}
        )
    
    job = jobs[job_id]
    
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job not finished: {job_id}",
            headers={"error_type": "job_not_finished"}
        )
    
    if not job["files"]["original"] or not os.path.exists(job["files"]["original"]):
        raise HTTPException(
            status_code=404,
            detail=f"Original data not found for job: {job_id}",
            headers={"error_type": "data_not_found"}
        )
    
    with open(job["files"]["original"], "r") as f:
        content = f.read()
    
    return content

@app.get("/video-data/{job_id}/kinect2d", response_class=PlainTextResponse)
async def get_kinect2d_data(job_id: str):
    """
    Get the Kinect 2D data for a specific job
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
            headers={"error_type": "job_not_found"}
        )
    
    job = jobs[job_id]
    
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job not finished: {job_id}",
            headers={"error_type": "job_not_finished"}
        )
    
    if not job["files"]["kinect2d"] or not os.path.exists(job["files"]["kinect2d"]):
        # Fall back to original data
        if not job["files"]["original"] or not os.path.exists(job["files"]["original"]):
            raise HTTPException(
                status_code=404,
                detail=f"Kinect 2D data not found for job: {job_id}",
                headers={"error_type": "data_not_found"}
            )
        
        with open(job["files"]["original"], "r") as f:
            content = f.read()
    else:
        with open(job["files"]["kinect2d"], "r") as f:
            content = f.read()
    
    return content

@app.get("/video-data/{job_id}/kinect3d", response_class=PlainTextResponse)
async def get_kinect3d_data(job_id: str):
    """
    Get the Kinect 3D data for a specific job
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
            headers={"error_type": "job_not_found"}
        )
    
    job = jobs[job_id]
    
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job not finished: {job_id}",
            headers={"error_type": "job_not_finished"}
        )
    
    if not job["files"]["kinect3d"] or not os.path.exists(job["files"]["kinect3d"]):
        # Fall back to kinect2d data
        return await get_kinect2d_data(job_id)
    
    with open(job["files"]["kinect3d"], "r") as f:
        content = f.read()
    
    return content

@app.get("/video-data/{job_id}/trimmed", response_class=PlainTextResponse)
async def get_trimmed_data(job_id: str):
    """
    Get the trimmed data for a specific job
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
            headers={"error_type": "job_not_found"}
        )
    
    job = jobs[job_id]
    
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job not finished: {job_id}",
            headers={"error_type": "job_not_finished"}
        )
    
    if not job["files"]["trimmed"] or not os.path.exists(job["files"]["trimmed"]):
        # Fall back to kinect3d data
        return await get_kinect3d_data(job_id)
    
    with open(job["files"]["trimmed"], "r") as f:
        content = f.read()
    
    return content
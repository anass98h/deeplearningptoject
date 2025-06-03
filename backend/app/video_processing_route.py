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
from ugly_2d_detector import check_ugly_2d, UglyDetectionError
from posenet_to_kinect2d import convert_to_kinect2d
from kinect2d_to_kinect3d import add_depth_predictions
from frame_trimmer import trim_frames
from bad_3d_detector import check_bad_3d_exercise, BadExerciseError
from exercise_scorer import score_exercise_with_interpretation

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
    2. Check for ugly/poor quality video (NEW)
    3. Convert to Kinect 2D format
    4. Add depth predictions for Kinect 3D format
    5. Trim frames to identify the relevant segment
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
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
            },
            "quality_scores": {
                "ugly_2d_goodness": None,
                "ugly_2d_confidence": None,
                "bad_3d_exercise_score": None,
                "final_exercise_score": None,
                "score_interpretation": None,
                "advanced_analysis": None
            }
        }
        
        # Define the background task function
        def process_video_task(job_id, file_path):
            try:
                # Step 1: Process video with MoveNet to extract pose data
                update_job_status(job_id, "processing", "Extracting poses with MoveNet")

                movenet_output_filename = f"{Path(file_path).stem}_movenet.csv"
                movenet_output_path = job_dir / movenet_output_filename
                process_video_file(file_path, output_dir=str(movenet_output_path))

                logger.info(f"MoveNet processing complete, results saved to {movenet_output_path}")
                jobs[job_id]["files"]["original"] = str(movenet_output_path)
                
                # Step 2: Check for ugly/poor quality video (NEW STEP)
                try:
                    update_job_status(job_id, "processing", "Checking video quality")
                    
                    result = check_ugly_2d(
                        video_path=file_path,
                        pose_csv_path=movenet_output_path,
                        model_dir=MODEL_DIR
                    )
                    
                    # Handle different return formats (with or without advanced analysis)
                    if len(result) == 3:
                        goodness_score, confidence_score, advanced_analysis = result
                    else:
                        goodness_score, confidence_score = result
                        advanced_analysis = {}
                    
                    # Store quality scores
                    jobs[job_id]["quality_scores"]["ugly_2d_goodness"] = float(goodness_score)
                    jobs[job_id]["quality_scores"]["ugly_2d_confidence"] = float(confidence_score)
                    jobs[job_id]["quality_scores"]["advanced_analysis"] = advanced_analysis
                    
                    logger.info(f"Video quality check passed - Goodness: {goodness_score:.3f}, Confidence: {confidence_score:.3f}")
                    
                    # Log advanced analysis if available
                    if advanced_analysis and advanced_analysis.get("analysis_type") == "advanced_quality":
                        overall = advanced_analysis.get("overall_assessment", {})
                        if overall:
                            logger.info(f"Advanced analysis - Quality: {overall.get('assessment', 'unknown')}, "
                                      f"Score: {overall.get('quality_score', 0):.2f}")
                            if overall.get("issues_detected"):
                                logger.info(f"Issues detected: {', '.join(overall['issues_detected'])}")
                    
                    # After ugly detection, we might have a new CSV with confidence scores
                    # We need to create a version without confidence for the rest of the pipeline
                    confidence_csv_path = movenet_output_path.parent / f"{movenet_output_path.stem}_with_confidence.csv"
                    if confidence_csv_path.exists():
                        # Create a version without confidence scores for the pipeline
                        logger.info("Removing confidence scores for pipeline compatibility")
                        df_with_conf = pd.read_csv(confidence_csv_path)
                        
                        # Keep only x,y coordinates and FrameNo, drop confidence columns
                        columns_to_keep = ['FrameNo']
                        for col in df_with_conf.columns:
                            if col.endswith('_x') or col.endswith('_y'):
                                columns_to_keep.append(col)
                        
                        df_without_conf = df_with_conf[columns_to_keep]
                        df_without_conf.to_csv(movenet_output_path, index=False)
                        logger.info(f"Saved pipeline-compatible pose data to {movenet_output_path}")
                    
                except UglyDetectionError as e:
                    # Video is too ugly to process - but capture analysis data for diagnostics
                    error_message = f"Video quality check failed: {str(e)}"
                    logger.error(error_message)
                    
                    # Extract analysis data from the exception if available
                    if hasattr(e, 'advanced_analysis'):
                        jobs[job_id]["quality_scores"]["advanced_analysis"] = e.advanced_analysis
                    if hasattr(e, 'goodness_score'):
                        jobs[job_id]["quality_scores"]["ugly_2d_goodness"] = e.goodness_score
                    if hasattr(e, 'confidence_score'):
                        jobs[job_id]["quality_scores"]["ugly_2d_confidence"] = e.confidence_score
                    
                    # Log detailed diagnostics if available
                    if hasattr(e, 'advanced_analysis') and e.advanced_analysis:
                        advanced = e.advanced_analysis
                        if advanced.get("analysis_type") == "advanced_quality":
                            overall = advanced.get("overall_assessment", {})
                            if overall:
                                logger.error(f"Video failed with quality assessment: {overall.get('assessment', 'unknown')}")
                                if overall.get("issues_detected"):
                                    logger.error(f"Specific issues: {', '.join(overall['issues_detected'])}")
                                if overall.get("recommendation"):
                                    logger.error(f"Recommendation: {overall['recommendation']}")
                    
                    update_job_status(job_id, "failed", error_message)
                    return
                except Exception as e:
                    # If ugly detection fails for other reasons, log warning but continue
                    logger.warning(f"Ugly detection failed, continuing with pipeline: {e}")
                
                # Check if model directory exists for subsequent steps
                if not MODEL_DIR.exists():
                    logger.warning(f"Model directory {MODEL_DIR} not found, skipping conversion steps")
                    update_job_status(job_id, "completed", "Processing completed (without models)")
                    return
                
                # Step 3: Convert MoveNet output to Kinect 2D format
                try:
                    update_job_status(job_id, "processing", "Converting to Kinect 2D format")
                    
                    kinect2d_output_path = job_dir / f"{Path(file_path).stem}_kinect2d.csv"
                    convert_to_kinect2d(
                        input_csv=movenet_output_path,
                        output_csv=kinect2d_output_path,
                        model_dir=MODEL_DIR
                    )
                    
                    logger.info(f"Kinect 2D conversion complete, results saved to {kinect2d_output_path}")
                    jobs[job_id]["files"]["kinect2d"] = str(kinect2d_output_path)
                    
                    # Step 4: Add depth predictions to get Kinect 3D format
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
                        
                        # Step 5: Trim frames to identify the relevant segment
                        try:
                            update_job_status(job_id, "processing", "Trimming frames to identify relevant segment")
                            
                            trimmed_output_path = job_dir / f"{Path(file_path).stem}_trimmed.csv"
                            trim_result, trim_indices = trim_frames(
                                input_csv=kinect3d_output_path,
                                output_csv=trimmed_output_path,
                                model_dir=MODEL_DIR
                            )
                            
                            logger.info(f"Frame trimming complete, results saved to {trimmed_output_path}")
                            logger.info(f"Trim indices: [{trim_indices[0]}, {trim_indices[1]}]")
                            
                            jobs[job_id]["files"]["trimmed"] = str(trimmed_output_path)
                            
                            # Step 6: Check for bad exercise form (NEW STEP)
                            try:
                                update_job_status(job_id, "processing", "Analyzing exercise form quality")
                                
                                exercise_score = check_bad_3d_exercise(
                                    input_csv=trimmed_output_path,
                                    model_dir=MODEL_DIR
                                )
                                
                                # Store exercise quality score
                                jobs[job_id]["quality_scores"]["bad_3d_exercise_score"] = float(exercise_score)
                                
                                logger.info(f"Exercise form check passed - Quality score: {exercise_score:.3f}")
                                
                                # Step 7: Score the exercise (NEW STEP)
                                try:
                                    update_job_status(job_id, "processing", "Calculating final exercise score")
                                    
                                    final_score, score_interpretation = score_exercise_with_interpretation(
                                        input_csv=trimmed_output_path,
                                        model_dir=MODEL_DIR
                                    )
                                    
                                    # Store final scoring results
                                    jobs[job_id]["quality_scores"]["final_exercise_score"] = float(final_score)
                                    jobs[job_id]["quality_scores"]["score_interpretation"] = score_interpretation
                                    
                                    logger.info(f"Exercise scoring complete - Score: {final_score:.3f} ({score_interpretation})")
                                    update_job_status(job_id, "completed", 
                                                    f"Processing completed successfully - Score: {final_score:.1f}/5.0 ({score_interpretation})")
                                    
                                except Exception as e:
                                    # If scoring fails, log warning but still mark as completed
                                    logger.warning(f"Exercise scoring failed, but exercise passed quality checks: {e}")
                                    update_job_status(job_id, "completed", "Processing completed successfully (without scoring)")
                                
                            except BadExerciseError as e:
                                # Exercise form is bad - stop pipeline and return error
                                error_message = f"Exercise form check failed: {str(e)}"
                                logger.error(error_message)
                                update_job_status(job_id, "failed", error_message)
                                return
                            except Exception as e:
                                # If bad exercise detection fails for other reasons, log warning but continue
                                logger.warning(f"Bad exercise detection failed, continuing: {e}")
                                
                                # Still try to score the exercise
                                try:
                                    update_job_status(job_id, "processing", "Calculating final exercise score")
                                    
                                    final_score, score_interpretation = score_exercise_with_interpretation(
                                        input_csv=trimmed_output_path,
                                        model_dir=MODEL_DIR
                                    )
                                    
                                    # Store final scoring results
                                    jobs[job_id]["quality_scores"]["final_exercise_score"] = float(final_score)
                                    jobs[job_id]["quality_scores"]["score_interpretation"] = score_interpretation
                                    
                                    logger.info(f"Exercise scoring complete - Score: {final_score:.3f} ({score_interpretation})")
                                    update_job_status(job_id, "completed", 
                                                    f"Processing completed successfully - Score: {final_score:.1f}/5.0 ({score_interpretation})")
                                    
                                except Exception as e2:
                                    logger.warning(f"Exercise scoring also failed: {e2}")
                                    update_job_status(job_id, "completed", "Processing completed successfully")
                            
                        except Exception as e:
                            logger.error(f"Error in frame trimming: {str(e)}")
                            jobs[job_id]["files"]["trimmed"] = jobs[job_id]["files"]["kinect3d"]
                            update_job_status(job_id, "completed", "Processing completed with errors in trimming")
                    except Exception as e:
                        logger.error(f"Error in Kinect 3D conversion: {str(e)}")
                        jobs[job_id]["files"]["kinect3d"] = jobs[job_id]["files"]["kinect2d"]
                        jobs[job_id]["files"]["trimmed"] = jobs[job_id]["files"]["kinect2d"]
                        update_job_status(job_id, "completed", "Processing completed with errors in 3D conversion")
                except Exception as e:
                    logger.error(f"Error in Kinect 2D conversion: {str(e)}")
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

@app.get("/video-data/{job_id}/untrimmed", response_class=PlainTextResponse)
async def get_untrimmed_data(job_id: str):
    """
    Get the untrimmed (full) 3D data for a specific job
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
    
    # Return the full kinect3d data (untrimmed)
    return await get_kinect3d_data(job_id)

@app.get("/video-data/{job_id}/final")
async def get_final_results(job_id: str):
    """
    Get the complete final results including all scores and both trimmed + untrimmed 3D skeleton data
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
    
    # Get the trimmed 3D skeleton data
    trimmed_data = None
    if job["files"]["trimmed"] and os.path.exists(job["files"]["trimmed"]):
        with open(job["files"]["trimmed"], "r") as f:
            trimmed_data = f.read()
    elif job["files"]["kinect3d"] and os.path.exists(job["files"]["kinect3d"]):
        # Fallback to kinect3d data if trimmed not available
        with open(job["files"]["kinect3d"], "r") as f:
            trimmed_data = f.read()
    
    # Get the untrimmed (full) 3D skeleton data
    untrimmed_data = None
    if job["files"]["kinect3d"] and os.path.exists(job["files"]["kinect3d"]):
        with open(job["files"]["kinect3d"], "r") as f:
            untrimmed_data = f.read()
    elif job["files"]["kinect2d"] and os.path.exists(job["files"]["kinect2d"]):
        # Fallback to kinect2d data if kinect3d not available
        with open(job["files"]["kinect2d"], "r") as f:
            untrimmed_data = f.read()
    
    # Prepare the complete response
    response = {
        "job_id": job_id,
        "filename": job["filename"],
        "status": job["status"],
        "message": job["message"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "quality_scores": job["quality_scores"],
        "skeleton_data": {
            "trimmed": trimmed_data,
            "untrimmed": untrimmed_data
        },
        "data_formats": {
            "trimmed": "kinect_3d_trimmed_exercise_segment",
            "untrimmed": "kinect_3d_full_sequence"
        }
    }
    
    return response
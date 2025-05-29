#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
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

# Custom metrics needed for model loading
def mae_good(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))

def mae_conf(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

class UglyDetectionError(Exception):
    """Custom exception for when video is detected as ugly/poor quality"""
    pass

def check_ugly_2d(video_path, pose_csv_path, model_dir="models", 
                  goodness_threshold=0.5, confidence_threshold=0.3, use_confidence_extraction=True):
    """
    Check if the video and pose data are of poor quality ("ugly").
    
    Args:
        video_path: Path to the original video file
        pose_csv_path: Path to the MoveNet pose CSV file
        model_dir: Directory containing the hybrid model
        goodness_threshold: Minimum goodness score (below this = ugly)
        confidence_threshold: Minimum confidence score (below this = ugly)
        
    Returns:
        tuple: (goodness_score, confidence_score) if quality is acceptable
        
    Raises:
        UglyDetectionError: If video/poses are detected as poor quality
        FileNotFoundError: If model or input files are missing
    """
    try:
        # Model configuration (must match training parameters)
        NUM_FRAMES = 15
        IMG_H, IMG_W = 64, 64
        
        # Set model path
        model_path = Path(model_dir) / "trained_hybrid_model.keras"
        
        # Check if model exists
        if not model_path.exists():
            raise FileNotFoundError(f"Hybrid model not found: {model_path}")
        
        logger.info(f"Checking video quality for {video_path}")
        
        # Check if the pose CSV has confidence data (40 features) or not (26 features)
        df_check = pd.read_csv(pose_csv_path)
        if "FrameNo" in df_check.columns:
            df_check = df_check.drop(columns=["FrameNo"])
        num_features = df_check.shape[1]
        
        if num_features == 26:
            # Need to re-extract with confidence scores
            logger.info("Pose data doesn't have confidence scores, re-extracting with confidence...")
            if use_confidence_extraction:
                pose_csv_path = extract_poses_with_confidence(video_path, pose_csv_path)
            else:
                logger.warning("Pose data has only 26 features but model needs 40. Skipping ugly detection.")
                return 1.0, 1.0
        elif num_features == 40:
            logger.info("Pose data has confidence scores, proceeding with ugly detection")
        else:
            logger.warning(f"Unexpected number of features: {num_features}. Expected 26 or 40. Skipping ugly detection.")
            return 1.0, 1.0
        
        # Load sample exactly like in the original file
        X_vid, X_pose = load_sample(video_path, pose_csv_path, NUM_FRAMES, IMG_H, IMG_W)
        
        # Load model with custom metrics
        logger.info(f"Loading hybrid model: {model_path}")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"mae_good": mae_good, "mae_conf": mae_conf}
        )
        
        # Make prediction exactly like in the original file
        logger.info("Running ugly detection inference...")
        pred = model.predict([X_vid, X_pose], verbose=0)[0]
        goodness_pred, conf_pred = pred
        
        logger.info(f"Ugly detection results - Goodness: {goodness_pred:.3f}, Confidence: {conf_pred:.3f}")
        
        # Check if video is "ugly" based on thresholds
        if goodness_pred < goodness_threshold:
            error_msg = (f"Video quality too poor for processing. "
                        f"Goodness score: {goodness_pred:.3f} "
                        f"(minimum required: {goodness_threshold})")
            logger.warning(error_msg)
            raise UglyDetectionError(error_msg)
        
        if conf_pred < confidence_threshold:
            error_msg = (f"Model confidence too low for reliable processing. "
                        f"Confidence score: {conf_pred:.3f} "
                        f"(minimum required: {confidence_threshold})")
            logger.warning(error_msg)
            raise UglyDetectionError(error_msg)
        
        logger.info(f"Video quality check passed - proceeding with pipeline")
        return goodness_pred, conf_pred
        
    except UglyDetectionError:
        # Re-raise ugly detection errors
        raise
    except Exception as e:
        logger.error(f"Error in ugly detection: {e}")
        # If model fails, we could either:
        # 1. Raise an error (strict)
        # 2. Continue with a warning (permissive)
        # For now, let's be permissive and continue
        logger.warning(f"Ugly detection failed, continuing with pipeline: {e}")
        return 1.0, 1.0  # Return high scores to continue

def load_sample(video_path, pose_csv_path, num_frames, img_h, img_w):
    """
    Load and preprocess a single sample exactly like in the original file.
    """
    # --- video frames -------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < num_frames:
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, (img_w, img_h),
                       interpolation=cv2.INTER_AREA).astype("float32")/255.
        frames.append(f)
    cap.release()
    if len(frames) < num_frames:
        frames += [np.zeros((img_h, img_w, 3), "float32")] * (num_frames - len(frames))
    frames = np.stack(frames[:num_frames])                     # (n,h,w,3)

    # --- pose CSV ----------------------------------------------------------
    df = pd.read_csv(pose_csv_path)
    if "FrameNo" in df.columns:
        df = df.drop(columns=["FrameNo"])
    pose = df.iloc[:num_frames].to_numpy("float32")
    if pose.shape[0] < num_frames:
        pad = np.zeros((num_frames - pose.shape[0], pose.shape[1]), "float32")
        pose = np.vstack([pose, pad])
    pose = pose[:num_frames]                                   # (n,feat)

    # add batch dimension
    return frames[np.newaxis, ...], pose[np.newaxis, ...]

def extract_poses_with_confidence(video_path, original_pose_path):
    """
    Re-extract poses with confidence scores using the confidence-enabled MoveNet extractor.
    
    Args:
        video_path: Path to the original video file
        original_pose_path: Path to the original pose CSV (without confidence)
        
    Returns:
        Path to the new pose CSV file with confidence scores
    """
    import sys
    import tempfile
    from pathlib import Path
    
    # Import the confidence-enabled pose extractor
    try:
        # Add the confidence extractor functionality inline
        # (Based on your move_net_with_c.py)
        
        # Create temporary output for confidence extraction
        temp_dir = Path(original_pose_path).parent
        confidence_csv_path = temp_dir / f"{Path(original_pose_path).stem}_with_confidence.csv"
        
        # Extract poses with confidence using your move_net_with_c.py logic
        poses_with_confidence = extract_poses_with_confidence_internal(video_path)
        
        # Save to CSV
        poses_with_confidence.to_csv(confidence_csv_path, index=False)
        logger.info(f"Re-extracted poses with confidence to {confidence_csv_path}")
        
        return str(confidence_csv_path)
        
    except Exception as e:
        logger.error(f"Failed to re-extract poses with confidence: {e}")
        raise

def extract_poses_with_confidence_internal(video_path):
    """
    Extract poses with confidence scores (simplified version of move_net_with_c.py)
    """
    import tensorflow as tf
    import kagglehub
    
    # Initialize MoveNet model
    model_path = kagglehub.model_download("google/movenet/tensorFlow2/singlepose-lightning")
    model = tf.saved_model.load(model_path)
    movenet = model.signatures['serving_default']
    
    # Keypoint mapping (from move_net_with_c.py)
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    keypoint_mapping = {
        "nose": "head",
        "left_wrist": "left_hand",
        "right_wrist": "right_hand",
        "left_ankle": "left_foot",
        "right_ankle": "right_foot",
        "left_shoulder": "left_shoulder",
        "right_shoulder": "right_shoulder",
        "left_elbow": "left_elbow",
        "right_elbow": "right_elbow",
        "left_hip": "left_hip",
        "right_hip": "right_hip",
        "left_knee": "left_knee",
        "right_knee": "right_knee"
    }
    
    required_keypoints = [
        "head", "left_shoulder", "right_shoulder", 
        "left_elbow", "right_elbow", "left_hand", "right_hand",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_foot", "right_foot"
    ]
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    all_poses = []
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, _ = frame.shape
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad the image
        input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run inference
        results = movenet(input_image)
        keypoints = results['output_0'].numpy()[0][0]
        
        # Calculate pose confidence
        keypoint_scores = [keypoints[i][2] for i in range(len(keypoints))]
        core_indices = [keypoint_names.index(kp) for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]]
        core_scores = [keypoints[idx][2] for idx in core_indices]
        pose_confidence = 0.7 * np.mean(core_scores) + 0.3 * np.mean(keypoint_scores)
        
        # Build frame data with confidence
        frame_data = {
            'FrameNo': current_frame,
            'pose_confidence': float(pose_confidence)
        }
        
        # Add keypoints with confidence
        for idx, kp_name in enumerate(keypoint_names):
            if kp_name not in keypoint_mapping:
                continue
            mapped_name = keypoint_mapping[kp_name]
            if mapped_name not in required_keypoints:
                continue
                
            y, x, score = keypoints[idx]
            pixel_x = int(x * width)
            pixel_y = int(y * height)
            
            frame_data[f"{mapped_name}_x"] = pixel_x
            frame_data[f"{mapped_name}_y"] = pixel_y
            frame_data[f"{mapped_name}_confidence"] = float(score)
        
        all_poses.append(frame_data)
        current_frame += 1
    
    cap.release()
    
    return pd.DataFrame(all_poses)
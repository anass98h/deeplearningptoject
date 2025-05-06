import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import kagglehub
import logging
import os
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PoseExtractor:
    def __init__(self, model_type='lightning'):
        self.model_type = model_type
        
        # Download and load model from Kaggle Hub
        logger.info(f"Loading MoveNet {model_type} model from Kaggle Hub...")
        try:
            if model_type == 'lightning':
                model_path = kagglehub.model_download("google/movenet/tensorFlow2/singlepose-lightning")
            elif model_type == 'thunder':
                model_path = kagglehub.model_download("google/movenet/tensorFlow2/singlepose-thunder")
            else:
                raise ValueError(f"Unknown model type: {model_type}. Choose 'lightning' or 'thunder'.")
            
            # Load the model
            self.model = tf.saved_model.load(model_path)
            self.movenet = self.model.signatures['serving_default']
            logger.info("MoveNet model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading MoveNet model: {e}")
            raise
        
        # Define keypoint names for MoveNet
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Define mapping from MoveNet to required output format
        self.keypoint_mapping = {
            "nose": "head",
            "left_wrist": "left_hand",
            "right_wrist": "right_hand",
            "left_ankle": "left_foot",
            "right_ankle": "right_foot",
            # The rest keep the same names
            "left_shoulder": "left_shoulder",
            "right_shoulder": "right_shoulder",
            "left_elbow": "left_elbow",
            "right_elbow": "right_elbow",
            "left_hip": "left_hip",
            "right_hip": "right_hip",
            "left_knee": "left_knee",
            "right_knee": "right_knee"
        }
        
        # Required output keypoints
        self.required_keypoints = [
            "head", "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", "left_hand", "right_hand",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_foot", "right_foot"
        ]

    def process_frame(self, frame):
        height, width, _ = frame.shape
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad the image to keep aspect ratio
        input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 
                                              192, 192)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        results = self.movenet(input_image)
        keypoints = results['output_0'].numpy()[0][0]
        
        keypoints_data = {}
        
        for idx, kp_name in enumerate(self.keypoint_names):
            if kp_name not in self.keypoint_mapping:
                continue
                
            # Get the mapped name
            mapped_name = self.keypoint_mapping[kp_name]
            
            # Skip if not in required output list
            if mapped_name not in self.required_keypoints:
                continue
                
            y, x, score = keypoints[idx]
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(x * width)
            pixel_y = int(y * height)
            
            # Add only x and y to the output data (no z)
            keypoints_data[f"{mapped_name}_x"] = pixel_x
            keypoints_data[f"{mapped_name}_y"] = pixel_y
        
        return keypoints_data

def extract_poses_from_video(video_path, output_path, sample_rate=1, model_type='lightning'):
    """
    Process a video file and extract pose data using MoveNet.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the output CSV file
        sample_rate: Process every Nth frame (1 = process all frames)
        model_type: MoveNet model type ('lightning' or 'thunder')
        
    Returns:
        Path to the saved CSV file
    """
    logger.info(f"Processing video: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"Could not open video file: {video_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {fps}fps, {frame_count} frames, {width}x{height} resolution")
    
    # Initialize the pose extractor
    pose_extractor = PoseExtractor(model_type=model_type)
    
    # Process frames
    all_poses = []
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame at the specified sample rate
        if current_frame % sample_rate == 0:
            try:
                keypoints_data = pose_extractor.process_frame(frame)
                
                frame_data = {
                    'FrameNo': current_frame,
                }
                frame_data.update(keypoints_data)
                
                all_poses.append(frame_data)
                
                if current_frame % 10 == 0:
                    logger.info(f"Processed frame {current_frame}/{frame_count}")
            except Exception as e:
                logger.error(f"Error processing frame {current_frame}: {e}")
        
        current_frame += 1
    
    # Release video capture
    cap.release()
    
    # Convert to DataFrame and save to CSV
    if all_poses:
        poses_df = pd.DataFrame(all_poses)
        poses_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(all_poses)} poses to {output_path}")
        return output_path
    else:
        error_msg = "No poses extracted from video"
        logger.error(error_msg)
        raise ValueError(error_msg)

def process_video_file(video_file_path, output_dir=None):
    """
    Process a video file and return the path to the extracted pose data.
    
    Args:
        video_file_path: Path to the input video file
        output_dir: Directory to save the output CSV file (optional)
        
    Returns:
        Path to the saved CSV file
    """
    print("Video file path:", video_file_path)
    print("Output directory:", output_dir)
    if output_dir is None:
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
    
    # Create output filename from input filename
    video_name = os.path.basename(video_file_path)
    base_name, _ = os.path.splitext(video_name)
    output_path = output_dir # / f"{base_name}_poses.csv"
    print("Output path:", output_path)
    # Process the video
    try:
        result_path = extract_poses_from_video(
            video_path=video_file_path,
            output_path=output_path
        )
        return result_path
    except Exception as e:
        logger.error(f"Error processing video file: {e}")
        raise
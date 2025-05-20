import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import time
import tensorflow as tf
import kagglehub


class PoseExtractor:
    def __init__(self, model_type='lightning'):
        """
        Initialize the pose extractor with MoveNet model from Kaggle Hub
        
        Args:
            model_type: 'lightning' or 'thunder'
        """
        self.model_type = model_type
        
        # Download and load model from Kaggle Hub
        print(f"Loading MoveNet {model_type} model from Kaggle Hub...")
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
            print("MoveNet model loaded successfully!")
        except Exception as e:
            print(f"Error loading MoveNet model: {e}")
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

    # Remove the manual confidence calculation method as we'll use MoveNet's built-in score

    def process_frame(self, frame):
        """
        Process a single frame to extract pose keypoints
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            Dictionary with keypoint data in the required format
        """
        height, width, _ = frame.shape
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and pad the image to keep aspect ratio
        input_image = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 
                                              192, 192)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run inference
        results = self.movenet(input_image)
        keypoints = results['output_0'].numpy()[0][0]
        
        # Calculate pose confidence score
        # Note: MoveNet in TensorFlow.js provides a direct pose confidence score,
        # but in the Python implementation, we need to calculate it from keypoint scores
        # This follows the concept mentioned in the Analytics India Mag article where
        # "overall confidence in the estimated person's pose" determines which poses to show
        keypoint_scores = [keypoints[i][2] for i in range(len(keypoints))]
        
        # Get confidence of core joints (shoulders and hips)
        # These are crucial for determining if a valid pose is detected
        core_indices = [
            self.keypoint_names.index(kp) for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            if kp in self.keypoint_names
        ]
        core_scores = [keypoints[idx][2] for idx in core_indices]
        
        # Calculate pose confidence using weighted average approach:
        # - 70% weight on core body keypoints (shoulders, hips)
        # - 30% weight on average of all keypoints
        pose_confidence = 0.7 * np.mean(core_scores) + 0.3 * np.mean(keypoint_scores) if core_scores else np.mean(keypoint_scores)
        
        # Create a dictionary with keypoint data in the required format
        keypoints_data = {
            'pose_confidence': float(pose_confidence)  # Add overall pose confidence score
        }
        
        # Map MoveNet keypoints to required format
        for idx, kp_name in enumerate(self.keypoint_names):
            # Skip keypoints that are not in our mapping
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
            
            # Add x, y coordinates and confidence score to the output data
            keypoints_data[f"{mapped_name}_x"] = pixel_x
            keypoints_data[f"{mapped_name}_y"] = pixel_y
            keypoints_data[f"{mapped_name}_confidence"] = float(score)  # Add confidence score
        
        return keypoints_data


def process_video(video_path, output_dir, sample_rate=1, model_type='lightning'):
    """
    Extract poses from video using MoveNet
    
    Args:
        video_path: Path to the input AVI file
        output_dir: Directory to save the outputs
        sample_rate: Process every Nth frame (default: 1 - process every frame)
        model_type: Type of MoveNet model to use ('lightning' or 'thunder')
        
    Returns:
        DataFrame with pose keypoints for each processed frame
    """
    # Extract the base filename (without extension) from the path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path}")
    print(f"  - FPS: {fps}")
    print(f"  - Frame count: {frame_count}")
    print(f"  - Resolution: {width}x{height}")
    
    # Initialize the pose extractor
    pose_extractor = PoseExtractor(model_type=model_type)
    
    # Prepare the output data structure
    all_poses = []
    
    # Process frames
    current_frame = 0
    processing_times = []
    
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame based on sample_rate
            if current_frame % sample_rate == 0:
                # Measure processing time
                start_time = time.time()
                
                # Extract pose from the frame
                keypoints_data = pose_extractor.process_frame(frame)
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                
                # Add frame number
                frame_data = {
                    'FrameNo': current_frame,
                }
                frame_data.update(keypoints_data)
                
                all_poses.append(frame_data)
            
            current_frame += 1
            pbar.update(1)
    
    # Release the video
    cap.release()
    
    # Convert the list of dictionaries to a DataFrame
    poses_df = pd.DataFrame(all_poses)
    
    # Save the poses to a CSV file using the base filename
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    poses_df.to_csv(csv_path, index=False)
    print(f"Saved poses to {csv_path}")
    
    # Calculate and print performance metrics
    if processing_times:
        avg_time = np.mean(processing_times)
        print(f"Average processing time per frame: {avg_time:.4f} seconds ({1/avg_time:.2f} FPS)")
    
    return poses_df


def batch_process_videos(video_dir, output_dir="output_poses", extension=".avi", 
                         sample_rate=1, model_type='lightning'):
    """
    Process all videos in a directory
    
    Args:
        video_dir: Directory containing AVI files
        output_dir: Directory for outputs (simple flat structure)
        extension: File extension to look for (default: .avi)
        sample_rate: Process every Nth frame
        model_type: Type of MoveNet model to use ('lightning' or 'thunder')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(extension)]
    print(f"Found {len(video_files)} {extension} files")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        
        print(f"\nProcessing {video_file}...")
        try:
            process_video(video_path, output_dir, sample_rate, model_type)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")


# Example usage
if __name__ == "__main__":
    # Process a single video
    # process_video("path/to/your/video.avi", "output_poses", sample_rate=1, model_type='lightning')
    
    # Process all videos in a directory
    batch_process_videos("./videos_of_good", "videos_of_good_with_c", sample_rate=1, model_type='lightning')
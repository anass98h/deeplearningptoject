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


def process_video(video_path, output_dir, sample_rate=1, model_type='lightning'):

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    

    os.makedirs(output_dir, exist_ok=True)
    

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
    

    pose_extractor = PoseExtractor(model_type=model_type)
    

    all_poses = []
    
 
    current_frame = 0
    processing_times = []
    
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            

            if current_frame % sample_rate == 0:
 
                start_time = time.time()
                

                keypoints_data = pose_extractor.process_frame(frame)
                

                process_time = time.time() - start_time
                processing_times.append(process_time)
                

                frame_data = {
                    'FrameNo': current_frame,
                }
                frame_data.update(keypoints_data)
                
                all_poses.append(frame_data)
            
            current_frame += 1
            pbar.update(1)
    

    cap.release()
    

    poses_df = pd.DataFrame(all_poses)
    

    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    poses_df.to_csv(csv_path, index=False)
    print(f"Saved poses to {csv_path}")
    

    if processing_times:
        avg_time = np.mean(processing_times)
        print(f"Average processing time per frame: {avg_time:.4f} seconds ({1/avg_time:.2f} FPS)")
    
    return poses_df


def batch_process_videos(video_dir, output_dir="output_poses", extension=".avi", 
                         sample_rate=1, model_type='lightning'):

    os.makedirs(output_dir, exist_ok=True)
    

    video_files = [f for f in os.listdir(video_dir) if f.endswith(extension)]
    print(f"Found {len(video_files)} {extension} files")
    

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        
        print(f"\nProcessing {video_file}...")
        try:
            process_video(video_path, output_dir, sample_rate, model_type)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")



if __name__ == "__main__":
    # Process a single video, this is what are are going to use for the pipeline
    # process_video("path/to/your/video.avi", "output_poses", sample_rate=1, model_type='lightning')
    
    # Process all videos in a directory
    batch_process_videos("./all_videos", "output_poses", sample_rate=1, model_type='lightning')
    
    print("Please modify the paths and uncomment the appropriate function calls to process your videos.")

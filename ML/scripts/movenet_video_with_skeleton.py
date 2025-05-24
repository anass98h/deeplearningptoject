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
            
            # Log model details
            print(f"Using MoveNet {model_type} with input size: {256 if model_type == 'thunder' else 192}x{256 if model_type == 'thunder' else 192}")
            
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
        
        # Define skeleton connections for visualization
        self.skeleton_connections = [
            ("head", "left_shoulder"), ("head", "right_shoulder"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_foot"),
            ("right_hip", "right_knee"), ("right_knee", "right_foot")
        ]

    def process_frame(self, frame):
        height, width, _ = frame.shape
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Determine input size based on model type
        input_size = 256 if self.model_type == 'thunder' else 192
        
        # Calculate resize ratio and padding
        if width > height:
            # Landscape orientation
            resize_ratio = input_size / width
            target_height = int(height * resize_ratio)
            pad_top = (input_size - target_height) // 2
            pad_bottom = input_size - target_height - pad_top
            pad_left = 0
            pad_right = 0
        else:
            # Portrait or square orientation
            resize_ratio = input_size / height
            target_width = int(width * resize_ratio)
            pad_left = (input_size - target_width) // 2
            pad_right = input_size - target_width - pad_left
            pad_top = 0
            pad_bottom = 0
        
        # Resize with padding
        input_image = tf.image.resize_with_pad(
            tf.expand_dims(frame_rgb, axis=0), 
            input_size, 
            input_size
        )
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run inference
        try:
            results = self.movenet(input_image)
            keypoints = results['output_0'].numpy()[0][0]
        except Exception as e:
            print(f"Error during model inference: {e}")
            # Return empty data in case of error
            empty_data = {'pose_confidence': 0.0}
            empty_dict = {}
            for kp in self.required_keypoints:
                empty_data[f"{kp}_x"] = 0
                empty_data[f"{kp}_y"] = 0
                empty_data[f"{kp}_confidence"] = 0.0
                empty_dict[kp] = (0, 0, 0.0)
            return empty_data, empty_dict
        
        # Calculate pose confidence score
        keypoint_scores = [keypoints[i][2] for i in range(len(keypoints))]
        
        # Get confidence of core joints (shoulders and hips)
        core_indices = [
            self.keypoint_names.index(kp) for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            if kp in self.keypoint_names
        ]
        core_scores = [keypoints[idx][2] for idx in core_indices]
        
        # Calculate weighted pose confidence
        pose_confidence = (0.7 * np.mean(core_scores) + 0.3 * np.mean(keypoint_scores) if core_scores else np.mean(keypoint_scores))
        
        # Create output dictionaries
        keypoints_data = {'pose_confidence': float(pose_confidence)}
        keypoints_dict = {}
        
        # Map keypoints to original image coordinates
        for idx, kp_name in enumerate(self.keypoint_names):
            if kp_name not in self.keypoint_mapping:
                continue
                
            mapped_name = self.keypoint_mapping[kp_name]
            if mapped_name not in self.required_keypoints:
                continue
                
            y, x, score = keypoints[idx]
            
            # Convert normalized coordinates to original image coordinates
            if width > height:
                # Landscape: remove padding and scale back
                pixel_y = (y * input_size - pad_top) / (input_size - pad_top - pad_bottom) * height
                pixel_x = x * width
            else:
                # Portrait: remove padding and scale back
                pixel_x = (x * input_size - pad_left) / (input_size - pad_left - pad_right) * width
                pixel_y = y * height
            
            # Ensure coordinates are within image bounds
            pixel_x = max(0, min(width - 1, int(pixel_x)))
            pixel_y = max(0, min(height - 1, int(pixel_y)))
            
            # Add to output
            keypoints_data[f"{mapped_name}_x"] = pixel_x
            keypoints_data[f"{mapped_name}_y"] = pixel_y
            keypoints_data[f"{mapped_name}_confidence"] = float(score)
            keypoints_dict[mapped_name] = (pixel_x, pixel_y, score)
        
        return keypoints_data, keypoints_dict
    
    def _fix_landmarks_scale(self, keypoints_dict, height, width):

        fixed_keypoints = keypoints_dict.copy()
        
        # Check if we have necessary landmarks
        if 'head' not in keypoints_dict or 'left_hip' not in keypoints_dict or 'right_hip' not in keypoints_dict:
            return fixed_keypoints
        
        # Get the head position
        head_x, head_y, head_score = keypoints_dict['head']
        
        # Get hip positions
        left_hip_x, left_hip_y, left_hip_score = keypoints_dict['left_hip']
        right_hip_x, right_hip_y, right_hip_score = keypoints_dict['right_hip']
        
        # Calculate hip center
        hip_center_x = (left_hip_x + right_hip_x) // 2
        hip_center_y = (left_hip_y + right_hip_y) // 2
        
        # Calculate torso height (head to hip center)
        torso_height = hip_center_y - head_y
        
        # In a standard human body, the torso is approximately 35-40% of total height
        # So we can estimate the full body height
        if torso_height > 0:
            estimated_body_height = torso_height / 0.4  # Assuming torso is 40% of body
            
            # Check if we have foot positions for better estimation
            if 'left_foot' in keypoints_dict and 'right_foot' in keypoints_dict:
                left_foot_y = keypoints_dict['left_foot'][1]
                right_foot_y = keypoints_dict['right_foot'][1]
                foot_y = max(left_foot_y, right_foot_y)
                actual_body_height = foot_y - head_y
                
                # Calculate scale factor (how much we need to scale the detected keypoints)
                scale_factor = actual_body_height / estimated_body_height if estimated_body_height > 0 else 1.0
                
                # Only apply scaling if the difference is significant
                if 0.7 < scale_factor < 1.4:
                    # Scale factor is reasonable, adjust slightly if needed
                    adjustment_factor = min(max(scale_factor, 0.8), 1.2)
                else:
                    # Scale factor seems off, use a more conservative adjustment
                    adjustment_factor = 1.0
            else:
                # Without feet, use a standard adjustment
                adjustment_factor = 1.0
                
            # Calculate actual body height with adjustment
            actual_body_height = int(estimated_body_height * adjustment_factor)
            
            # For debugging
            # print(f"Torso height: {torso_height}px, Estimated full height: {estimated_body_height}px, Actual: {actual_body_height}px")
            # print(f"Adjustment factor: {adjustment_factor}")
            
            # Store the body height for visualization
            return fixed_keypoints, actual_body_height
        
        # If we can't calculate torso height, return the original keypoints
        return fixed_keypoints, 0

    def draw_skeleton(self, frame, keypoints_dict, pose_confidence, min_confidence=0.2, thickness=3, point_radius=6):

        skeleton_frame = frame.copy()
        height, width, _ = frame.shape
        
        # Only draw the skeleton if the overall confidence is decent
        if pose_confidence > 0.3:
            # Fix scale issues and get estimated body height
            fixed_keypoints, body_height = self._fix_landmarks_scale(keypoints_dict, height, width)
            
            # Draw the skeleton connections
            for start_point, end_point in self.skeleton_connections:
                if start_point in fixed_keypoints and end_point in fixed_keypoints:
                    start_pos = fixed_keypoints[start_point]
                    end_pos = fixed_keypoints[end_point]
                    
                    # Check if both points have decent confidence
                    if start_pos[2] > min_confidence and end_pos[2] > min_confidence:
                        # Color the lines based on confidence (green for high confidence, yellow for medium, red for low)
                        avg_confidence = (start_pos[2] + end_pos[2]) / 2
                        if avg_confidence > 0.7:
                            line_color = (0, 255, 0)  # Green for high confidence
                        elif avg_confidence > 0.4:
                            line_color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            line_color = (0, 0, 255)  # Red for low confidence
                        
                        # Calculate dynamic thickness based on estimated body height
                        dynamic_thickness = thickness
                        if body_height > 0:
                            # Scale thickness based on body height
                            # Higher values for taller bodies, but within reasonable range
                            dynamic_thickness = max(2, min(6, int(thickness * body_height / 500)))
                        
                        # Draw the line
                        cv2.line(skeleton_frame, 
                                (int(start_pos[0]), int(start_pos[1])),
                                (int(end_pos[0]), int(end_pos[1])),
                                line_color, dynamic_thickness)
            
            # Draw the keypoints
            for keypoint, (x, y, score) in fixed_keypoints.items():
                if score > min_confidence:
                    # Color the points based on confidence
                    if score > 0.7:
                        point_color = (0, 0, 255)  # Red for high confidence
                    elif score > 0.4:
                        point_color = (0, 165, 255)  # Orange for medium confidence
                    else:
                        point_color = (0, 255, 255)  # Yellow for low confidence
                    
                    # Calculate dynamic radius based on estimated body height
                    dynamic_radius = point_radius
                    if body_height > 0:
                        # Scale point size based on body height
                        dynamic_radius = max(3, min(10, int(point_radius * body_height / 500)))
                    
                    # Draw the keypoint
                    cv2.circle(skeleton_frame, (int(x), int(y)), dynamic_radius, point_color, -1)
            
            # Add pose confidence to the frame
            confidence_text = f"Pose Confidence: {pose_confidence:.2f}"
            confidence_color = (0, 255, 0) if pose_confidence > 0.5 else (0, 165, 255) if pose_confidence > 0.3 else (0, 0, 255)
            cv2.putText(skeleton_frame, confidence_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
            
            # Add body height measurement for debugging
            if body_height > 0:
                height_text = f"Height: {body_height}px"
                cv2.putText(skeleton_frame, height_text, (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return skeleton_frame


def process_video(video_path, output_dir, sample_rate=1, model_type='lightning', 
                  create_skeleton_video=True, show_preview=False):

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
    
    # Set up the video writer for skeleton overlay if requested
    skeleton_video_writer = None
    if create_skeleton_video:
        skeleton_video_path = os.path.join(output_dir, f"{base_name}_skeleton.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec (H.264)
        skeleton_video_writer = cv2.VideoWriter(
            skeleton_video_path, fourcc, fps, (width, height)
        )
        if not skeleton_video_writer.isOpened():
            print(f"Warning: Could not open video writer for {skeleton_video_path}")
            create_skeleton_video = False
    
    # Process frames
    current_frame = 0
    processing_times = []
    
    # Store the last processed keypoints for frames we skip
    last_keypoints_dict = None
    last_pose_confidence = 0
    
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
                keypoints_data, keypoints_dict = pose_extractor.process_frame(frame)
                pose_confidence = keypoints_data['pose_confidence']
                
                # Store for interpolation in skipped frames
                last_keypoints_dict = keypoints_dict
                last_pose_confidence = pose_confidence
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                
                # Add frame number
                frame_data = {
                    'FrameNo': current_frame,
                }
                frame_data.update(keypoints_data)
                
                all_poses.append(frame_data)
                
                # Create and save frame with skeleton overlay if requested
                if create_skeleton_video:
                    skeleton_frame = pose_extractor.draw_skeleton(frame, keypoints_dict, pose_confidence)
                    skeleton_video_writer.write(skeleton_frame)
                    
                    # Show preview if requested
                    if show_preview:
                        # Resize for display if the frame is too large
                        if width > 1280 or height > 720:
                            display_width = min(1280, int(width * 720 / height))
                            display_height = min(720, int(height * 1280 / width))
                            display_frame = cv2.resize(skeleton_frame, (display_width, display_height))
                        else:
                            display_frame = skeleton_frame
                            
                        cv2.imshow('Frame with Skeleton', display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            
            # For frames we skip based on sample_rate, still add them to the video
            # with the skeleton from the previous processed frame
            elif create_skeleton_video and current_frame > 0 and last_keypoints_dict is not None:
                if not sample_rate == 1:  # Only needed if we're skipping frames
                    skeleton_frame = pose_extractor.draw_skeleton(frame, last_keypoints_dict, last_pose_confidence)
                    skeleton_video_writer.write(skeleton_frame)
                    
                    if show_preview:
                        # Resize for display if the frame is too large
                        if width > 1280 or height > 720:
                            display_width = min(1280, int(width * 720 / height))
                            display_height = min(720, int(height * 1280 / width))
                            display_frame = cv2.resize(skeleton_frame, (display_width, display_height))
                        else:
                            display_frame = skeleton_frame
                            
                        cv2.imshow('Frame with Skeleton', display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            
            current_frame += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    if create_skeleton_video and skeleton_video_writer is not None:
        skeleton_video_writer.release()
    
    if show_preview:
        cv2.destroyAllWindows()
    
    # Convert the list of dictionaries to a DataFrame
    poses_df = pd.DataFrame(all_poses)
    
    # Save the poses to a CSV file using the base filename
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    poses_df.to_csv(csv_path, index=False)
    print(f"Saved poses to {csv_path}")
    
    if create_skeleton_video:
        print(f"Saved skeleton video to {skeleton_video_path}")
    
    # Calculate and print performance metrics
    if processing_times:
        avg_time = np.mean(processing_times)
        print(f"Average processing time per frame: {avg_time:.4f} seconds ({1/avg_time:.2f} FPS)")
    
    return poses_df


def batch_process_videos(video_dir, output_dir="output_poses", extension=".avi", 
                         sample_rate=1, model_type='lightning', 
                         create_skeleton_videos=True, show_preview=False):

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files (case insensitive extension matching)
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(extension.lower())]
    print(f"Found {len(video_files)} {extension} files")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        
        print(f"\nProcessing {video_file}...")
        try:
            process_video(
                video_path, 
                output_dir, 
                sample_rate, 
                model_type,
                create_skeleton_video=create_skeleton_videos,
                show_preview=show_preview
            )
        except Exception as e:
            print(f"Error processing {video_file}: {e}")



if __name__ == "__main__":
    # Process a single video with skeleton visualization
    process_video(
        "ML/data/kinect_good_vs_bad/G08.avi", 
        "output_poses_with_skeleton", 
        sample_rate=1, 
        model_type='thunder',
        create_skeleton_video=True,
        show_preview=True  # Set to True to see real-time preview
    )
    
    # Process all videos in a directory with skeleton visualization
    # batch_process_videos(
    #     "./kinect_good_vs_bad", 
    #     "kinect_good_vs_bad_with_c", 
    #     sample_rate=1, 
    #     model_type='lightning',
    #     create_skeleton_videos=True,
    #     show_preview=False  # Set to True to see real-time preview
    # )
    
    print("Processing complete. Skeleton videos are saved in the output directory.")
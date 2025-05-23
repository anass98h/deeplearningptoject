#!/usr/bin/env python3
"""
batch_video_processor.py – Process multiple videos and save results to CSV

Simply edit the configuration variables below, then run:
    python batch_video_processor.py
"""
# --------------------------------------------------------------------
import os, sys, cv2, numpy as np, pandas as pd, tensorflow as tf
import time
import glob
from tqdm import tqdm  # For progress bar
import imageio.v3 as iio  # For more robust video reading

# Set up path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "ML/data/")

# --------------- CONFIGURATION (EDIT HERE) ---------------------------------------
# Paths
VIDEO_FOLDER = os.path.join(DATA_BASE_DIR, "kinect_good_vs_bad")
POSE_FOLDER  = os.path.join(DATA_BASE_DIR, "kinect_good_vs_bad_with_c")
OUTPUT_CSV   = os.path.join(DATA_BASE_DIR, "video_uggly_p.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "trained_hybrid_model.keras")

# Model parameters
NUM_FRAMES   = 15                      # must match the trained model
IMG_H, IMG_W = 64, 64                 # must match the trained model

# Classification threshold
GOODNESS_THRESHOLD = 3.0              # Include if goodness >= this value
# --------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # CPU-only (remove if GPU ok)

# --- custom metrics so the model loads -------------------------------------
def mae_good(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))
def mae_conf(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

# --- helper to load & preprocess a single sample ---------------------------
def load_sample(file_name):
    stem = os.path.splitext(os.path.basename(file_name))[0]
    vid_path  = os.path.join(VIDEO_FOLDER, f"{stem}.avi")
    pose_path = os.path.join(POSE_FOLDER, f"{stem}.csv")

    if not os.path.isfile(vid_path) or not os.path.isfile(pose_path):
        raise FileNotFoundError(f"Missing pair: {vid_path}, {pose_path}")

    # --- video frames using imageio instead of OpenCV ----------------------
    try:
        # Read frames using imageio
        frames = []
        for i, frame in enumerate(iio.imiter(vid_path)):
            if i >= NUM_FRAMES:
                break
            # Resize and normalize frame
            frame = cv2.resize(frame, (IMG_W, IMG_H),
                           interpolation=cv2.INTER_AREA).astype("float32")/255.
            frames.append(frame)
    except Exception as e:
        raise IOError(f"Error reading video file with imageio: {vid_path}, error: {str(e)}")
    
    # Pad with zeros if we didn't get enough frames
    if len(frames) == 0:
        raise IOError(f"No frames could be read from {vid_path}")
        
    if len(frames) < NUM_FRAMES:
        print(f"Warning: Only {len(frames)}/{NUM_FRAMES} frames read from {vid_path}, padding with zeros")
        frames += [np.zeros((IMG_H, IMG_W, 3), "float32")] * (NUM_FRAMES - len(frames))
    frames = np.stack(frames[:NUM_FRAMES])                     # (n,h,w,3)

    # --- pose CSV ----------------------------------------------------------
    df = pd.read_csv(pose_path)
    if "FrameNo" in df.columns:
        df = df.drop(columns=["FrameNo"])
    pose = df.iloc[:NUM_FRAMES].to_numpy("float32")
    if pose.shape[0] < NUM_FRAMES:
        pad = np.zeros((NUM_FRAMES - pose.shape[0], pose.shape[1]), "float32")
        pose = np.vstack([pose, pad])
    pose = pose[:NUM_FRAMES]                                   # (n,feat)

    # add batch dimension
    return frames[np.newaxis, ...], pose[np.newaxis, ...]

# --- function to process a single video and return results ------------------
def process_video(file_name, model):
    try:
        X_vid, X_pose = load_sample(file_name)
        
        start_time = time.time()
        pred = model.predict([X_vid, X_pose], verbose=0)[0]
        duration = time.time() - start_time
        
        goodness_pred, conf_pred = pred
        
        # Determine include/exclude status based on goodness threshold
        in_ex = "include" if goodness_pred >= GOODNESS_THRESHOLD else "exclude"
        
        return {
            'filename': os.path.splitext(os.path.basename(file_name))[0],  # Just the name without extension
            'goodness': goodness_pred,
            'confidence_score': conf_pred,
            'in_ex': in_ex,
            'processing_time': duration  # Keep for logging but won't be in final CSV
        }
    except Exception as e:
        print(f"[ERROR] Processing {file_name}: {e}")
        return {
            'filename': os.path.splitext(os.path.basename(file_name))[0],
            'goodness': None,
            'confidence_score': None,
            'in_ex': "error",
            'processing_time': None,
            'error': str(e)
        }

# ---------------- MAIN ------------------------------------------------------
if __name__ == "__main__":
    # Print the paths for debugging
    print(f"DATA_BASE_DIR: {DATA_BASE_DIR}")
    print(f"VIDEO_FOLDER: {VIDEO_FOLDER}")
    print(f"POSE_FOLDER: {POSE_FOLDER}")
    print(f"OUTPUT_CSV: {OUTPUT_CSV}")
    print(f"MODEL_PATH: {MODEL_PATH}")

    # Check if folders exist
    if not os.path.isdir(VIDEO_FOLDER):
        sys.exit(f"[ERROR] Video folder not found: {VIDEO_FOLDER}")
    if not os.path.isdir(POSE_FOLDER):
        sys.exit(f"[ERROR] Pose folder not found: {POSE_FOLDER}")
    
    # Load the model
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"mae_good": mae_good, "mae_conf": mae_conf}
        )
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load model: {e}")
    
    # Get all video files
    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.avi"))
    if not video_files:
        sys.exit(f"[ERROR] No video files found in {VIDEO_FOLDER}")
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Goodness threshold for inclusion: {GOODNESS_THRESHOLD}")
    
    # Process each video and collect results
    results = []
    skipped_files = []
    
    for i, file_path in enumerate(tqdm(video_files, desc="Processing videos")):
        file_name = os.path.basename(file_path)
        print(f"\nProcessing file {i+1}/{len(video_files)}: {file_name}")
        
        try:
            result = process_video(file_path, model)
            results.append(result)
            
            # Print result for current file
            if result['goodness'] is not None:
                print(f"  goodness         = {result['goodness']:.3f}")
                print(f"  confidence_score = {result['confidence_score']:.3f}")
                print(f"  classification   = {result['in_ex']}")
                print(f"  prediction_time  = {result['processing_time']:.4f} seconds")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
                skipped_files.append(file_name)
        except Exception as e:
            print(f"  Critical error processing {file_name}: {str(e)}")
            skipped_files.append(file_name)
            # Create an error result entry
            results.append({
                'filename': os.path.splitext(file_name)[0],
                'goodness': None,
                'confidence_score': None,
                'in_ex': "error",
                'processing_time': None,
                'error': str(e)
            })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)
    
    # Convert results to DataFrame and save to CSV (only the requested columns)
    results_df = pd.DataFrame(results)
    
    # Select and save only the required columns
    output_columns = ['filename', 'goodness', 'confidence_score', 'in_ex']
    results_df[output_columns].to_csv(OUTPUT_CSV, index=False)
    
    # Print summary statistics
    print(f"\nResults saved to {OUTPUT_CSV}")
    print(f"Total files processed: {len(results)}")
    print(f"Successfully processed: {len(results) - len(skipped_files)}")
    print(f"Skipped files: {len(skipped_files)}")
    
    # After processing, print summary of skipped files
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files due to errors:")
        for f in skipped_files[:10]:  # Show first 10 skipped files
            print(f"  - {f}")
        if len(skipped_files) > 10:
            print(f"  ...and {len(skipped_files) - 10} more")
    
    # Calculate statistics for successful predictions
    successful = results_df.dropna(subset=['goodness'])
    if not successful.empty:
        include_count = (successful['in_ex'] == 'include').sum()
        exclude_count = (successful['in_ex'] == 'exclude').sum()
        
        print(f"Successfully processed: {len(successful)}/{len(results)}")
        print(f"Included videos: {include_count} ({include_count/len(successful)*100:.1f}%)")
        print(f"Excluded videos: {exclude_count} ({exclude_count/len(successful)*100:.1f}%)")
        print(f"Average goodness: {successful['goodness'].mean():.3f}")
        print(f"Average confidence: {successful['confidence_score'].mean():.3f}")
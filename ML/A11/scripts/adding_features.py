import os
import pandas as pd
import numpy as np
from math import sqrt, acos, pi
from tqdm import tqdm

# Define directories
input_dir = '../../data/uncut/output_poses'
normalized_dir = '../../data/uncut/output_poses_normalized'
output_dir = '../../data/uncut/output_poses_normalized_features'
cut_data_dir = '../../data/kinect_good_preprocessed'
uncut_data_dir = '../../data/uncut/kinect_good_preprocessed_not_cut'

# Create directories if they don't exist
os.makedirs(normalized_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Normalizing factors
x_factor = 1920
y_factor = 1080

# Function to collect joint pairs from user input
def collect_joint_pairs(first_file_path):
    # Load the first file to extract joint names
    try:
        first_df = pd.read_csv(first_file_path)
        columns = first_df.columns[1:]  # Skip first column (assumed to be FrameNo)
        
        # Extract joint names from column names (remove the _x or _y suffix)
        joints = set()
        for col in columns:
            if col.endswith('_x') or col.endswith('_y'):
                joint_name = col[:-2]  # Remove the last 2 characters (_x or _y)
                joints.add(joint_name)
        
        joints = sorted(list(joints))
        
        print("\nAvailable joints in the dataset:")
        # Print joints in multiple columns for better readability
        for i, joint in enumerate(joints):
            print(f"{i+1:2d}. {joint:<20}", end="")
            if (i+1) % 3 == 0:
                print()  # New line every 3 joints
        if len(joints) % 3 != 0:
            print()  # Ensure a new line at the end
            
    except Exception as e:
        print(f"Error reading the first file: {e}")
        joints = []
    
    joint_pairs = []
    
    print("\nPlease enter the joint pairs you want to calculate distances for.")
    print("Format: jointA,jointB (e.g., left_shoulder,right_shoulder)")
    print("Enter 'f' to finish.")
    
    while True:
        user_input = input("\nEnter joint pair or 'f' to finish: ")
        
        if user_input.lower() == 'f':
            break
        
        if ',' in user_input:
            joint_pairs.append(user_input)
        else:
            print("Invalid format. Please use 'jointA,jointB' format.")
    
    print("\nSelected joint pairs:")
    for i, pair in enumerate(joint_pairs):
        print(f"{i+1}. {pair}")
    
    return joint_pairs

# Function to calculate distances for multiple joint pairs
def calculate_distances(df, joint_pairs):
    """
    Calculate distances for multiple joint pairs
    
    Args:
        df (DataFrame): DataFrame containing joint coordinates
        joint_pairs (list): List of joint pairs to calculate distances for, each pair is "jointA,jointB"
        
    Returns:
        dict: Dictionary of distance arrays for each joint pair
    """
    results = {}
    
    for pair in joint_pairs:
        joint_a, joint_b = pair.split(',')
        distance_col_name = f"{joint_a}_{joint_b}"
        
        joint_a_x = f"{joint_a}_x"
        joint_a_y = f"{joint_a}_y"
        joint_b_x = f"{joint_b}_x"
        joint_b_y = f"{joint_b}_y"
        
        # Check if all required columns exist
        required_cols = [joint_a_x, joint_a_y, joint_b_x, joint_b_y]
        if not all(col in df.columns for col in required_cols):
            print(f"Missing columns for joint pair: {pair}")
            continue
        
        # Calculate Euclidean distance
        distances = []
        for i in range(len(df)):
            xa = df[joint_a_x].iloc[i]
            ya = df[joint_a_y].iloc[i]
            xb = df[joint_b_x].iloc[i]
            yb = df[joint_b_y].iloc[i]
            
            # Handle NaN values
            if pd.isna(xa) or pd.isna(ya) or pd.isna(xb) or pd.isna(yb):
                distances.append(np.nan)
            else:
                distance = sqrt((xa - xb)**2 + (ya - yb)**2)
                # Round to 6 decimal places
                distance = round(distance, 6)
                distances.append(distance)
        
        results[distance_col_name] = distances
    
    return results

# Function to calculate angles for multiple joint pairs
def calculate_angles(df, joint_pairs):
    """
    Calculate cosine of angles between vectors for multiple joint pairs
    
    Args:
        df (DataFrame): DataFrame containing joint coordinates
        joint_pairs (list): List of joint pairs to calculate angles for, each pair is "jointA,jointB"
        
    Returns:
        dict: Dictionary of cosine angle values for each joint pair
    """
    results = {}
    
    for pair in joint_pairs:
        joint_a, joint_b = pair.split(',')
        angle_col_name = f"{joint_a}_{joint_b}_angle"
        
        joint_a_x = f"{joint_a}_x"
        joint_a_y = f"{joint_a}_y"
        joint_b_x = f"{joint_b}_x"
        joint_b_y = f"{joint_b}_y"
        
        # Check if all required columns exist
        required_cols = [joint_a_x, joint_a_y, joint_b_x, joint_b_y]
        if not all(col in df.columns for col in required_cols):
            print(f"Missing columns for joint pair: {pair}")
            continue
        
        # Calculate angle between vectors (cosine value)
        cosine_values = []
        for i in range(len(df)):
            xa1 = df[joint_a_x].iloc[i]  # x-coordinate of joint A
            ya1 = df[joint_a_y].iloc[i]  # y-coordinate of joint A
            xb1 = df[joint_b_x].iloc[i]  # x-coordinate of joint B
            yb1 = df[joint_b_y].iloc[i]  # y-coordinate of joint B
            
            # Handle NaN values
            if pd.isna(xa1) or pd.isna(ya1) or pd.isna(xb1) or pd.isna(yb1):
                cosine_values.append(np.nan)
            else:
                # Calculate dot product and magnitudes
                dot_product = (xa1 * xb1) + (ya1 * yb1)
                magnitude_a = sqrt(xa1**2 + ya1**2)
                magnitude_b = sqrt(xb1**2 + yb1**2)
                
                if magnitude_a == 0 or magnitude_b == 0:
                    cosine_values.append(np.nan)
                else:
                    # Apply the formula: cos_theta = (xa1*xb1 + ya1*yb1) / (sqrt(xa1^2 + ya1^2) * sqrt(xb1^2 + yb1^2))
                    cos_theta = dot_product / (magnitude_a * magnitude_b)
                    
                    # Handle numerical precision issues
                    cos_theta = min(1.0, max(-1.0, cos_theta))
                    
                    # Store the cosine value directly (without converting to degrees)
                    cosine_values.append(round(cos_theta, 6))
        
        results[angle_col_name] = cosine_values
    
    return results

# Function to normalize coordinates in a dataframe
def normalize_coordinates(df):
    normalized_df = df.copy()
    
    # Normalize all coordinate columns
    columns = df.columns[1:]  # Skip first column (assumed to be FrameNo)
    
    for i in range(0, len(columns), 2):
        if i + 1 < len(columns):  # Ensure we have both x and y
            x_col = columns[i]
            y_col = columns[i+1]
            
            # Check if these are actually coordinates (column names often contain 'x' and 'y')
            if ('x' in x_col.lower() and 'y' in y_col.lower()) or \
               ('X' in x_col and 'Y' in y_col):
                
                # Normalize x by dividing by x_factor and round to 6 decimal places
                normalized_df[x_col] = (df[x_col] / x_factor).round(6)
                
                # Normalize y by dividing by y_factor and round to 6 decimal places
                normalized_df[y_col] = (df[y_col] / y_factor).round(6)
    
    return normalized_df

# MAIN FUNCTION 0: Normalize dataset (initial process)
def normalize_dataset():
    print("\n=== NORMALIZING DATASET (INITIAL PROCESS) ===")
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return False
    
    print(f"Found {len(csv_files)} files to normalize")
    print("Starting normalization process...\n")
    
    # Use tqdm to show progress bar
    for file_name in tqdm(csv_files, desc="Normalizing files"):
        # Read the CSV file
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path)
        
        # Normalize coordinates
        normalized_df = normalize_coordinates(df)
        
        # Save normalized data
        output_path = os.path.join(normalized_dir, file_name)
        normalized_df.to_csv(output_path, index=False)
    
    print("\nAll files normalized!")
    print(f"Normalized files are saved in: {normalized_dir}")
    return True

# MAIN FUNCTION 1: Add distance features
def add_distance_features():
    print("\n=== ADDING DISTANCE FEATURES ===")
    
    # Check if normalized directory exists and has files
    if not os.path.exists(normalized_dir):
        print(f"Normalized directory {normalized_dir} does not exist.")
        print("Please run the normalization process first.")
        return
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(normalized_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {normalized_dir}")
        return
    
    print("Starting the joint distance calculation process...\n")
    
    # Get the path to the first file for joint extraction
    first_file = csv_files[0]
    first_file_path = os.path.join(normalized_dir, first_file)
    
    # Step 1: Collect joint pairs to calculate distances
    joint_pairs = collect_joint_pairs(first_file_path)
    
    if not joint_pairs:
        print("No joint pairs selected. Exiting.")
        return
    
    # Process first file separately with confirmation
    first_df = pd.read_csv(first_file_path)
    
    # Calculate distances for all joint pairs (using already normalized data)
    distance_results = calculate_distances(first_df, joint_pairs)
    
    # Create a copy of the dataframe for features
    feature_df = first_df.copy()
    
    # Add distance columns to the dataframe
    for col_name, distances in distance_results.items():
        feature_df[col_name] = distances
    
    # Save the first processed file
    output_path = os.path.join(output_dir, first_file)
    feature_df.to_csv(output_path, index=False)
    
    print(f"\nFirst file has been processed: {first_file}")
    print(f"Please check the file at: {output_path}")
    print("It contains normalized coordinates and calculated distances.")
    
    # Wait for user confirmation to continue with the rest of the files
    continue_processing = input("\nIs the data correct? Continue processing the remaining files? (y/n): ")
    
    if continue_processing.lower() != 'y':
        print("Processing stopped after the first file.")
        return
    
    # Process the rest of the files
    print("\nProcessing remaining files...")
    
    # Use tqdm to show progress bar
    for file_name in tqdm(csv_files[1:], desc="Processing files"):
        # Read the normalized CSV file
        file_path = os.path.join(normalized_dir, file_name)
        df = pd.read_csv(file_path)
        
        # Calculate distances for all joint pairs
        distance_results = calculate_distances(df, joint_pairs)
        
        # Create a copy of the dataframe for features
        feature_df = df.copy()
        
        # Add distance columns to the dataframe
        for col_name, distances in distance_results.items():
            feature_df[col_name] = distances
        
        # Save processed data
        output_path = os.path.join(output_dir, file_name)
        feature_df.to_csv(output_path, index=False)
    
    print("\nAll files processed!")
    print(f"Output files are saved in: {output_dir}")

# MAIN FUNCTION 2: Add angle features
def add_angle_features():
    print("\n=== ADDING ANGLE FEATURES ===")
    
    # Check if normalized directory exists and has files
    if not os.path.exists(normalized_dir):
        print(f"Normalized directory {normalized_dir} does not exist.")
        print("Please run the normalization process first.")
        return
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(normalized_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {normalized_dir}")
        return
    
    print("Starting the joint angle calculation process...\n")
    
    # Get the path to the first file for joint extraction
    first_file = csv_files[0]
    first_file_path = os.path.join(normalized_dir, first_file)
    
    # Step 1: Collect joint pairs to calculate angles
    print("Please specify joint pairs to calculate angles between vectors.")
    print("The angle calculation uses the formula:")
    print("cos_theta = (xa1*xb1 + ya1*yb1) / (sqrt(xa1^2 + ya1^2) * sqrt(xb1^2 + yb1^2))")
    print("The raw cosine values will be stored (range from -1 to 1)")
    
    joint_pairs = collect_joint_pairs(first_file_path)
    
    if not joint_pairs:
        print("No joint pairs selected. Exiting.")
        return
    
    # Process first file separately with confirmation
    first_df = pd.read_csv(first_file_path)
    
    # Calculate angles for all joint pairs (using already normalized data)
    angle_results = calculate_angles(first_df, joint_pairs)
    
    # Create a copy of the dataframe for features
    feature_df = first_df.copy()
    
    # Add angle columns to the dataframe
    for col_name, angles in angle_results.items():
        feature_df[col_name] = angles
    
    # Save the first processed file
    output_path = os.path.join(output_dir, first_file)
    feature_df.to_csv(output_path, index=False)
    
    print(f"\nFirst file has been processed: {first_file}")
    print(f"Please check the file at: {output_path}")
    print("It contains normalized coordinates and calculated cosine angle values.")
    
    # Wait for user confirmation to continue with the rest of the files
    continue_processing = input("\nIs the data correct? Continue processing the remaining files? (y/n): ")
    
    if continue_processing.lower() != 'y':
        print("Processing stopped after the first file.")
        return
    
    # Process the rest of the files
    print("\nProcessing remaining files...")
    
    # Use tqdm to show progress bar
    for file_name in tqdm(csv_files[1:], desc="Processing files"):
        # Read the normalized CSV file
        file_path = os.path.join(normalized_dir, file_name)
        df = pd.read_csv(file_path)
        
        # Calculate angles for all joint pairs
        angle_results = calculate_angles(df, joint_pairs)
        
        # Create a copy of the dataframe for features
        feature_df = df.copy()
        
        # Add angle columns to the dataframe
        for col_name, angles in angle_results.items():
            feature_df[col_name] = angles
        
        # Save processed data
        output_path = os.path.join(output_dir, file_name)
        feature_df.to_csv(output_path, index=False)
    
    print("\nAll files processed!")
    print(f"Output files are saved in: {output_dir}")

# MAIN FUNCTION 3: Add classification labels
def add_classification_labels():
    print("\n=== ADDING CLASSIFICATION LABELS ===")
    
    # Check if normalized directory exists and has files
    if not os.path.exists(normalized_dir):
        print(f"Normalized directory {normalized_dir} does not exist.")
        print("Please run the normalization process first.")
        return
    
    # Get list of CSV files from the uncut data directory (using normalized files)
    uncut_csv_files = [f for f in os.listdir(normalized_dir) if f.endswith('.csv')]
    
    if not uncut_csv_files:
        print(f"No CSV files found in {normalized_dir}")
        return
    
    # Get list of CSV files from the cut data directory for matching
    cut_csv_files = [f for f in os.listdir(cut_data_dir) if f.endswith('.csv')]
    
    if not cut_csv_files:
        print(f"No CSV files found in {cut_data_dir}")
        return
    
    print(f"Found {len(uncut_csv_files)} uncut files and {len(cut_csv_files)} cut files")
    print("Starting classification label process...\n")
    
    # Process each uncut file
    for uncut_file in tqdm(uncut_csv_files, desc="Processing files"):
        # Find matching cut file
        if uncut_file in cut_csv_files:
            # Read normalized uncut data
            uncut_df = pd.read_csv(os.path.join(normalized_dir, uncut_file))
            
            # Read cut data
            cut_df = pd.read_csv(os.path.join(cut_data_dir, uncut_file))
            
            # Initialize start_stop column with default value 0 (non-labeled frames)
            uncut_df['start_stop'] = 0
            
            # Get first frame number in cut data
            if not cut_df.empty:
                start_frame = cut_df['FrameNo'].iloc[0]
                end_frame = cut_df['FrameNo'].iloc[-1]
                
                # Mark start and end frames in uncut data
                # Find the row in uncut_df with matching FrameNo for start
                start_index = uncut_df[uncut_df['FrameNo'] == start_frame].index
                if not start_index.empty:
                    uncut_df.loc[start_index, 'start_stop'] = 1
                
                # Find the row in uncut_df with matching FrameNo for end
                end_index = uncut_df[uncut_df['FrameNo'] == end_frame].index
                if not end_index.empty:
                    uncut_df.loc[end_index, 'start_stop'] = 1
            
            # Save the updated uncut file to output directory
            output_path = os.path.join(output_dir, uncut_file)
            uncut_df.to_csv(output_path, index=False)
            print(f"Processed {uncut_file}")
        else:
            print(f"No matching cut file found for {uncut_file}")
    
    print("\nAll files processed with classification labels!")
    print(f"Output files are saved in: {output_dir}")

# Main menu function
def main():
    print("\n=== POSENET FEATURE PROCESSING ===")
    print("Checking for normalized dataset...")
    
    # Check if normalized dataset exists
    normalized_files = []
    if os.path.exists(normalized_dir):
        normalized_files = [f for f in os.listdir(normalized_dir) if f.endswith('.csv')]
    
    # If normalized dataset doesn't exist, run normalization first
    if not normalized_files:
        print("Normalized dataset not found. Running normalization process first...")
        if not normalize_dataset():
            print("Normalization failed. Exiting.")
            return
    else:
        print(f"Found {len(normalized_files)} normalized files in {normalized_dir}")
    
    while True:
        print("\n=== MAIN MENU ===")
        print("0. Normalize dataset (already done)")
        print("1. Add distance features")
        print("2. Add angle features")
        print("3. Add classification labels")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ")
        
        if choice == '0':
            normalize_dataset()
        elif choice == '1':
            add_distance_features()
        elif choice == '2':
            add_angle_features()
        elif choice == '3':
            add_classification_labels()
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
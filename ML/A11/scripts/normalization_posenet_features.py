import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm

# Define input and output directories
input_dir = '../../data/uncut/output_poses'
output_dir = '../../data/uncut/output_poses_normalized_features'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Normalizing factors
x_factor = 1920
y_factor = 1080

# Get list of CSV files
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

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

# Function to calculate distance between two joints
def calculate_distance(df, joint_a, joint_b):
    joint_a_x = f"{joint_a}_x"
    joint_a_y = f"{joint_a}_y"
    joint_b_x = f"{joint_b}_x"
    joint_b_y = f"{joint_b}_y"
    
    # Check if all required columns exist
    required_cols = [joint_a_x, joint_a_y, joint_b_x, joint_b_y]
    for col in required_cols:
        if col not in df.columns:
            print(f"Column {col} not found in dataframe")
            return None
    
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
    
    return distances

# Function to process a single file
def process_file(file_name, joint_pairs, ask_confirmation=False):
    print(f"Processing file: {file_name}")
    
    # Read the CSV file
    file_path = os.path.join(input_dir, file_name)
    df = pd.read_csv(file_path)
    
    # Skip first column and print header to verify if ask_confirmation is True
    if ask_confirmation:
        columns = df.columns[1:]
        print("Columns (skipping first column):", columns.tolist())
        
        proceed = input("Continue processing this file? (y/n): ")
        if proceed.lower() != 'y':
            print(f"Skipping {file_name}")
            return False
    
    # Create a copy of the dataframe for normalized values
    normalized_df = df.copy()
    
    # First normalize all coordinate columns
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
    
    # Calculate distances for each joint pair and add as new columns
    for pair in joint_pairs:
        joint_a, joint_b = pair.split(',')
        distance_col_name = f"{joint_a}_{joint_b}"
        
        # Calculate distance using the normalized values
        distances = calculate_distance(normalized_df, joint_a, joint_b)
        
        if distances is not None:
            normalized_df[distance_col_name] = distances
        else:
            print(f"Could not calculate distance for {joint_a} and {joint_b}")
    
    # Save processed data to output directory with the same filename
    output_path = os.path.join(output_dir, file_name)
    normalized_df.to_csv(output_path, index=False)
    
    return True

# Main process
if csv_files:
    print("Starting the joint distance calculation process...\n")
    
    # Get the path to the first file for joint extraction
    first_file = csv_files[0]
    first_file_path = os.path.join(input_dir, first_file)
    
    # Step 1: Collect joint pairs to calculate distances
    joint_pairs = collect_joint_pairs(first_file_path)
    
    if not joint_pairs:
        print("No joint pairs selected. Exiting.")
    else:
        # Process first file separately with confirmation
        success = process_file(first_file, joint_pairs, ask_confirmation=True)
        
        if success:
            temp_path = os.path.join(output_dir, first_file)
            print(f"\nFirst file has been processed: {first_file}")
            print(f"Please check the file at: {temp_path}")
            print("It contains normalized coordinates and calculated distances.")
            
            # Wait for user confirmation to continue with the rest of the files
            continue_processing = input("\nIs the data correct? Continue processing the remaining files? (y/n): ")
            
            if continue_processing.lower() == 'y':
                # Process the rest of the files without asking confirmation for each file
                print("\nProcessing remaining files...")
                
                # Use tqdm to show progress bar
                for file_name in tqdm(csv_files[1:], desc="Processing files"):
                    process_file(file_name, joint_pairs, ask_confirmation=False)
                
                print("\nAll files processed!")
                print(f"Output files are saved in: {output_dir}")
            else:
                print("Processing stopped after the first file.")
        else:
            print("Processing stopped. First file was skipped.")
else:
    print(f"No CSV files found in {input_dir}")
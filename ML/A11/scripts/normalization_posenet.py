import os
import pandas as pd
import numpy as np

# Define input and output directories
input_dir = '../../data/uncut/output_poses'
output_dir = '../../data/uncut/output_poses_normalized'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Normalizing factors
x_factor = 1920
y_factor = 1080

# Get list of CSV files
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Function to process a single file
def process_file(file_name, ask_confirmation=False):
    print(f"Processing file: {file_name}")
    
    # Read the CSV file
    file_path = os.path.join(input_dir, file_name)
    df = pd.read_csv(file_path)
    
    # Skip first column and print header to verify
    columns = df.columns[1:]
    print("Columns (skipping first column):", columns.tolist())
    
    # Wait for confirmation before continuing if ask_confirmation is True
    if ask_confirmation:
        proceed = input("Continue normalizing this file? (y/n): ")
        if proceed.lower() != 'y':
            print(f"Skipping {file_name}")
            return False
    
    # Create a copy of the dataframe for normalized values
    normalized_df = df.copy()
    
    # Process each pair of columns (x,y)
    for i in range(0, len(columns), 2):
        if i + 1 < len(columns):  # Ensure we have both x and y
            x_col = columns[i]
            y_col = columns[i+1]
            
            # Check if these are actually coordinates (column names often contain 'x' and 'y')
            if ('x' in x_col.lower() and 'y' in y_col.lower()) or \
               ('X' in x_col and 'Y' in y_col):
                print(f"Normalizing columns: {x_col} and {y_col}")
                
                # Normalize x by dividing by x_factor and round to 6 decimal places
                normalized_df[x_col] = (df[x_col] / x_factor).round(6)
                
                # Normalize y by dividing by y_factor and round to 6 decimal places
                normalized_df[y_col] = (df[y_col] / y_factor).round(6)
    
    # Save normalized data to output directory with the same filename
    output_path = os.path.join(output_dir, file_name)
    normalized_df.to_csv(output_path, index=False)
    print(f"Normalized file saved to: {output_path}")
    print("-" * 50)
    return True

# Process first file separately with confirmation
if csv_files:
    first_file = csv_files[0]
    success = process_file(first_file, ask_confirmation=True)
    
    if success:
        print(f"\nFirst file has been processed: {first_file}")
        print(f"Please check the normalized file at: {os.path.join(output_dir, first_file)}")
        print("Verify that the normalization is correct before continuing.")
        
        # Wait for user confirmation to continue with the rest of the files
        continue_processing = input("\nContinue processing the remaining files? (y/n): ")
        
        if continue_processing.lower() == 'y':
            # Process the rest of the files without asking confirmation for each file
            for file_name in csv_files[1:]:
                process_file(file_name, ask_confirmation=False)
            print("All files processed!")
        else:
            print("Processing stopped after the first file.")
    else:
        print("Processing stopped. First file was skipped.")
else:
    print(f"No CSV files found in {input_dir}")
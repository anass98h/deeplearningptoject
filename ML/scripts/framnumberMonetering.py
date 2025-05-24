import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the folder containing the CSV files
folder_path = 'ML/data/all_kinect_trimmed/'  # Replace with your actual folder path

# Path to the file that contains the list of filenames (first column only)
file_list_csv = 'ML/data/dataset_for_scoring.csv'  # Replace with your actual CSV path

# Read the list of filenames from the first column and append ".csv"
file_list = pd.read_csv(file_list_csv).iloc[:, 0].astype(str) + '_kinect.csv'

# Store filename and row count
file_row_counts = []

# Process only files listed in the CSV
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            row_count = len(df)
            file_row_counts.append((filename, row_count))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    else:
        print(f"File not found: {filename}")

# Sort files by row count descending
file_row_counts.sort(key=lambda x: x[1], reverse=True)

# Print the results
print("\nFilename and Row Count (Sorted by Row Count Descending):")
for fname, count in file_row_counts:
    print(f"{fname}: {count} rows")

# Plot the distribution
row_counts = [count for _, count in file_row_counts]
plt.hist(row_counts, bins=10, edgecolor='black')
plt.title('Distribution of Row Counts in Listed CSV Files')
plt.xlabel('Number of Rows')
plt.ylabel('Number of Files')
plt.grid(True)
plt.show()

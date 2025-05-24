import pandas as pd

# Load the first CSV (semicolon-delimited)
df1 = pd.read_csv('ML/data/video_uggly_p.csv', delimiter=';')

# Load the second CSV (comma-delimited)
df2 = pd.read_csv('ML/data/good_bad_results.csv')
import pandas as pd

# Load the CSVs
df1 = pd.read_csv('ML/data/video_uggly_p.csv', delimiter=';')
df2 = pd.read_csv('ML/data/good_bad_results.csv')
df3 = pd.read_csv('ML/data/scores.csv')

# Filter df1 and df2 based on criteria
df1_filtered = df1[df1['in_ex'].str.lower() == 'include']
df2_filtered = df2[df2['class'].str.lower() == 'good']

# Merge df1 and df2 on filename
merged = pd.merge(df1_filtered, df2_filtered, on='filename')

# Clean df3 by removing "_kinect" from Var1 to match filenames
df3['filename'] = df3['Var1'].str.replace('_kinect', '', regex=False)

# Merge with df3 on filename
final_merged = pd.merge(merged[['filename']], df3[['filename', 'Var2']], on='filename')

# Write to output
final_merged.to_csv('dataset_for_scoring.csv', index=False)

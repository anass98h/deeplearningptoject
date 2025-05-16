import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "ML/data/")

# Default configuration 
DATA_DIR_CONFIG = {
    # 'input_dir': 'uncut/output_poses',
    # 'cut_data_dir':'kinect_good_preprocessed',
    # 'uncut_data_dir':'kinect_good_preprocessed_not_cut',
    'unprocessed': {
        'bad_good':'kinect_good_preprocessed'
        # 'bad_good':'test'

    },
    'processed': {
        'augmentation': {
                'mirror': 'augmentation/miroor',
                'rotate': 'augmentation/rotate',
                'scale': 'augmentation/scale',
                'cascade': 'augmentation/cascade',
                # 'shear': 'augmentation/translate',
        },
    }
}

aug_config = {
    "mirror": {
        "x": {"apply": True},
        "y": {"apply": False},
        "z": {"apply": False}
    },

    "rotate": {
        "x": {"apply": True, "angle_deg": np.pi / 7},
        "y": {"apply": False, "angle_deg": np.pi / 7},
        "z": {"apply": False, "angle_deg": np.pi / 7}
    },

    "scale": {
        "apply": True,
        # (e.g., 1.5 stretches, 0.8 compresses)
        "factors": [1.5, 1.5, 1.5]  # [fx, fy, fz]  
    },

    # "shear": {
    #     "x": {"apply": True, "shear_factor": 0.1},  # x ← x + shear_factor*y
    #     "y": {"apply": False, "shear_factor": 0.1}  # y ← y + shear_factor*x
    # },

}

current_file = ''

def _get_nested_config(config_path: str) -> str:
    """
    Retrieves a value from the nested DATA_DIR_CONFIG dictionary using dot notation.
    
    Args:
        config_path: Path in the config using dot notation (e.g., 'processed.augmentation.mirror')
    
    Returns:
        The directory path string from the config
        
    Raises:
        KeyError: If the path doesn't exist in the configuration
    """
    keys = config_path.split('.')
    current = DATA_DIR_CONFIG
    
    # Track the path for better error messages
    current_path = ""
    
    for i, key in enumerate(keys):
        current_path = f"{current_path}.{key}" if current_path else key
        
        if key not in current:
            raise KeyError(f"Config key '{key}' not found in path '{config_path}' at position {current_path}")
        
        current = current[key]
        
        # If we've reached a string value but we're not at the end of the keys,
        # then the path is too deep
        if isinstance(current, str) and i < len(keys) - 1:
            raise KeyError(f"Config path '{config_path}' goes too deep. '{current_path}' is already a leaf node.")
    
    # After processing all keys, current should be a string path
    if not isinstance(current, str):
        raise ValueError(f"Config path '{config_path}' points to a directory structure, not a specific path")
    
    return current

def _get_full_path(config_path: str) -> str:
    dir_path = _get_nested_config(config_path)
    full_dir_path = os.path.join(DATA_BASE_DIR, dir_path)
    return full_dir_path

def load_data(config_path: str, filename: str) -> pd.DataFrame:
    """
    Load data from a file in the directory specified by the config path.
    
    Args:
        config_path: Path in the config using dot notation (e.g., 'processed.augmentation.mirror')
        filename: Name of the file to load
    
    Returns:
        DataFrame containing the loaded data
    
    Raises:
        KeyError: If the config path doesn't exist
        FileNotFoundError: If the directory or file doesn't exist
    """
    try:
        # Get directory path from config
        full_dir_path = _get_full_path(config_path)
        
        # Check if directory exists
        if not os.path.exists(full_dir_path):
            raise FileNotFoundError(f"Directory not found for config path '{config_path}': {full_dir_path}")
        
        # Build the complete file path
        file_path = os.path.join(full_dir_path, filename)
        print(f"Loading data from: {file_path}")
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        # Check file extension to determine how to read it
        _, ext = os.path.splitext(filename)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path)
        elif ext.lower() in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Supported formats: .csv, .xls, .xlsx")
    
    except KeyError as e:
        # Re-raise with more context
        raise KeyError(f"Error accessing config: {str(e)}")
    except Exception as e:
        # Re-raise other exceptions with context
        raise type(e)(f"Error loading data from '{config_path}/{filename}': {str(e)}")



def save_data(data: pd.DataFrame, config_path: str, filename: str, create_dir: bool = True) -> str:
    """
    Save DataFrame to the specified output location defined in DATA_DIR_CONFIG.
    
    Args:
        data: DataFrame to save
        config_path: Path in the config using dot notation (e.g., 'processed.augmentation.mirror')
        filename: Name of the file to save
        create_dir: Whether to create the directory if it doesn't exist
        
    Returns:
        Path where the data was saved
        
    Example:
        # Save to base directory
        save_data(df, 'processed', 'normalized_data.csv')
        
        # Save to nested directory
        save_data(df, 'processed.augmentation.mirror', 'mirrored_data.csv')
    """
    # print("saving: ", data)
    try:
        # Get directory path from config
        dir_path = _get_nested_config(config_path)
        full_dir_path = os.path.join(DATA_BASE_DIR, dir_path)
        
        # Create directory if it doesn't exist
        if create_dir and not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path, exist_ok=True)
        
        # Full path for saving the file
        full_path = os.path.join(full_dir_path, filename)
        
        # Save the DataFrame based on the file extension
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.csv':
            data.to_csv(full_path, index=False)
        elif ext.lower() in ['.xls', '.xlsx']:
            data.to_excel(full_path, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Supported formats: .csv, .xls, .xlsx")
        
        return full_path
    
    except KeyError as e:
        # Re-raise with more context
        raise KeyError(f"Error accessing config: {str(e)}")
    except Exception as e:
        # Re-raise other exceptions with context
        raise type(e)(f"Error saving data to '{config_path}/{filename}': {str(e)}")
    
def get_joint_names(columns):
    joints = set()
    for col in columns:
        # Split by underscore and remove the last part (x, y, or z)
        if '_' in col:
            joint_name = '_'.join(col.split('_')[:-1])
            # Exclude single character names like 'x', 'y', 'z'
            # and ensure we strip any whitespace
            if len(joint_name.strip()) > 1:  # More than one character
                joints.add(joint_name.strip())
    joints = list(joints)
    # print(f"get_joint_names_ Joints: {joints}")
    return joints
def mirror_to_new_df(df: pd.DataFrame, axis: str = 'x') -> pd.DataFrame:
    """
    Generate a new DataFrame containing mirrored joint columns.
    Each original joint coordinate (x, y, z) is multiplied by -1 on the chosen axis.
    Column names stay the same as the original.
    
    Parameters:
        df    : Input DataFrame with joint columns.
        axis  : Axis to mirror across ('x', 'y', or 'z').
    Returns:
        DataFrame of same length with mirrored data but original column names
    """
    joints = get_joint_names(df.columns)
    
    # Create a copy of the DataFrame - preserving ALL columns exactly
    df_mirror = df.copy()
    
    # Process each joint while keeping column names the same
    for j in joints:
        for coord in ('x', 'y', 'z'):
            # Handle special case for head_x column
            col = f" {j}_{coord}" if coord == 'x' and j == 'head' else f"{j}_{coord}"
            
            if coord == axis:
                # Apply mirroring only to the specified axis - keeping the SAME column name
                df_mirror[col] = np.round(df[col] * -1, 6)
    
    return df_mirror

def add_mirror_augmentation(df: pd.DataFrame, file_prefix: str = None) -> pd.DataFrame:
    """Apply mirroring augmentation and save to files with appropriate naming"""
    # Check if the axis is valid
    if 'mirror' not in aug_config:
        raise ValueError("Invalid augmentation configuration: 'mirror' key not found.")
    
    # Iterate through the axes and apply mirroring
    for axis in aug_config['mirror']:
        if aug_config['mirror'][axis]['apply']:
            df_mirror = mirror_to_new_df(df, axis)
            save_data(df_mirror, 'processed.augmentation.mirror', f'{file_prefix}_mirror_{axis}.csv')
    
    return df_mirror

def rotate_to_new_df(
    df: pd.DataFrame,
    axis: str = 'y',
    angle_rad: float = np.pi / 7
) -> pd.DataFrame:
    """
    Generate a new DataFrame containing rotated joint columns.
    Rotates each joint position around the specified axis by angle_rad.
    Column names stay the same as the original.
    
    Parameters:
        df        : Input DataFrame with joint columns.
        axis      : Axis to rotate around ('x', 'y', or 'z').
        angle_rad : Rotation angle in radians.
    Returns:
        DataFrame of same length with rotated data but original column names
    """
    # Build rotation matrix
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]])
    elif axis == 'y':
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[ c, -s, 0],
                      [ s,  c, 0],
                      [ 0,  0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    joints = get_joint_names(df.columns)
    
    # Create a copy of the DataFrame with ALL columns preserved
    df_rot = df.copy()
    
    # Apply rotation per joint while keeping column names the same
    for j in joints:
        x_col = f" {j}_x" if j == 'head' else f"{j}_x"
        y_col = f"{j}_y"
        z_col = f"{j}_z"
        
        x = df[x_col].values
        y = df[y_col].values
        z = df[z_col].values
        
        # Calculate rotated coordinates
        new_x = np.round(R[0,0]*x + R[0,1]*y + R[0,2]*z, 6)
        new_y = np.round(R[1,0]*x + R[1,1]*y + R[1,2]*z, 6)
        new_z = np.round(R[2,0]*x + R[2,1]*y + R[2,2]*z, 6)
        
        # Update the DataFrame in place - keeping the SAME column names
        df_rot[x_col] = new_x
        df_rot[y_col] = new_y
        df_rot[z_col] = new_z
    
    return df_rot

def add_rotation_augmentation(df: pd.DataFrame, file_prefix: str = None) -> pd.DataFrame:
    """Apply rotation augmentation and save to files with appropriate naming"""
    # Check if the axis is valid
    if 'rotate' not in aug_config:
        raise ValueError("Invalid augmentation configuration: 'rotate' key not found.")
    
    # Iterate through the axes and apply rotation
    for axis in aug_config['rotate']:
        if aug_config['rotate'][axis]['apply']:
            angle = aug_config['rotate'][axis]['angle_deg']
            df_rotate = rotate_to_new_df(df, axis, angle)
            angle_deg = int(angle * 180 / np.pi)  # Convert to degrees for filename
            save_data(df_rotate, 'processed.augmentation.rotate', f'{file_prefix}_rotate_{axis}_{angle_deg}deg.csv')
    
    return df_rotate

def scale_to_new_df(
    df: pd.DataFrame,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0
) -> pd.DataFrame:
    """
    Generate a new DataFrame containing scaled joint coordinates.
    Column names stay the same as the original.
    
    Parameters:
        df      : Input DataFrame with joint columns.
        scale_x : Scaling factor for the X-axis
        scale_y : Scaling factor for the Y-axis
        scale_z : Scaling factor for the Z-axis
    Returns:
        DataFrame of same length with scaled data but original column names
    """
    joints = get_joint_names(df.columns)
    
    # Create a copy of the DataFrame with ALL columns preserved
    df_scaled = df.copy()
    
    # Apply scaling per joint while keeping column names the same
    for j in joints:
        x_col = f" {j}_x" if j == 'head' else f"{j}_x"
        y_col = f"{j}_y"
        z_col = f"{j}_z"
        
        # Apply scaling while keeping the SAME column names
        df_scaled[x_col] = np.round(df[x_col] * scale_x, 6)
        df_scaled[y_col] = np.round(df[y_col] * scale_y, 6)
        df_scaled[z_col] = np.round(df[z_col] * scale_z, 6)
    
    return df_scaled

def add_scale_augmentation(df: pd.DataFrame, file_prefix: str = None) -> pd.DataFrame:
    """Apply scaling augmentation and save to files with appropriate naming"""
    # Check if the axis is valid
    if 'scale' not in aug_config:
        raise ValueError("Invalid augmentation configuration: 'scale' key not found.")
    
    factors = aug_config['scale']['factors']
    # Apply scaling if enabled
    if aug_config['scale']['apply']:
        df_scale = scale_to_new_df(df, factors[0], factors[1], factors[2])
        # Include the scale factors in the filename
        scale_str = f"scale_{factors[0]}_{factors[1]}_{factors[2]}"
        save_data(df_scale, 'processed.augmentation.scale', f'{file_prefix}_{scale_str}.csv')
    
    return df_scale


def apply_cascade_augmentation(df: pd.DataFrame, file_prefix: str = None) -> None:
    """
    Apply cascade augmentation by combining multiple augmentation techniques sequentially.
    This creates data with combinations of different augmentation types.
    All DataFrames maintain the original column naming scheme.
    
    Parameters:
        df          : Input DataFrame with joint columns
        file_prefix : Prefix for output filenames
    
    Returns:
        None - Files are saved directly to the appropriate directories
    """
    print(f"Applying cascade augmentation for {file_prefix}")
    
    # Define which augmentations are enabled
    mirror_enabled = any(aug_config['mirror'][axis]['apply'] for axis in aug_config['mirror'])
    rotate_enabled = any(aug_config['rotate'][axis]['apply'] for axis in aug_config['rotate'])
    scale_enabled = aug_config['scale']['apply']
    
    # --- Mirror + Rotate cascade ---
    if mirror_enabled and rotate_enabled:
        for m_axis in aug_config['mirror']:
            if aug_config['mirror'][m_axis]['apply']:
                # First apply mirroring
                df_mirrored = mirror_to_new_df(df, m_axis)
                
                # Then apply rotation to the mirrored data
                for r_axis in aug_config['rotate']:
                    if aug_config['rotate'][r_axis]['apply']:
                        angle = aug_config['rotate'][r_axis]['angle_deg']
                        angle_deg = int(angle * 180 / np.pi)
                        df_mirror_rotate = rotate_to_new_df(df_mirrored, r_axis, angle)
                        save_data(df_mirror_rotate, 'processed.augmentation.cascade', 
                                 f'{file_prefix}_mirror_{m_axis}_rotate_{r_axis}_{angle_deg}deg.csv')
    
    # --- Rotate + Scale cascade ---
    if rotate_enabled and scale_enabled:
        factors = aug_config['scale']['factors']
        scale_str = f"scale_{factors[0]}_{factors[1]}_{factors[2]}"
        
        for r_axis in aug_config['rotate']:
            if aug_config['rotate'][r_axis]['apply']:
                # First apply rotation
                angle = aug_config['rotate'][r_axis]['angle_deg']
                angle_deg = int(angle * 180 / np.pi)
                df_rotated = rotate_to_new_df(df, r_axis, angle)
                
                # Then scale the rotated data
                df_rotate_scale = scale_to_new_df(df_rotated, factors[0], factors[1], factors[2])
                save_data(df_rotate_scale, 'processed.augmentation.cascade', 
                         f'{file_prefix}_rotate_{r_axis}_{angle_deg}deg_{scale_str}.csv')
    
    # --- Mirror + Scale cascade ---
    if mirror_enabled and scale_enabled:
        factors = aug_config['scale']['factors']
        scale_str = f"scale_{factors[0]}_{factors[1]}_{factors[2]}"
        
        for m_axis in aug_config['mirror']:
            if aug_config['mirror'][m_axis]['apply']:
                # First apply mirroring
                df_mirrored = mirror_to_new_df(df, m_axis)
                
                # Then scale the mirrored data
                df_mirror_scale = scale_to_new_df(df_mirrored, factors[0], factors[1], factors[2])
                save_data(df_mirror_scale, 'processed.augmentation.cascade', 
                         f'{file_prefix}_mirror_{m_axis}_{scale_str}.csv')
    
    # --- Mirror + Rotate + Scale cascade (full cascade) ---
    if mirror_enabled and rotate_enabled and scale_enabled:
        factors = aug_config['scale']['factors']
        scale_str = f"scale_{factors[0]}_{factors[1]}_{factors[2]}"
        
        for m_axis in aug_config['mirror']:
            if aug_config['mirror'][m_axis]['apply']:
                # First apply mirroring
                df_mirrored = mirror_to_new_df(df, m_axis)
                
                for r_axis in aug_config['rotate']:
                    if aug_config['rotate'][r_axis]['apply']:
                        # Then apply rotation to mirrored data
                        angle = aug_config['rotate'][r_axis]['angle_deg']
                        angle_deg = int(angle * 180 / np.pi)
                        df_mirror_rotate = rotate_to_new_df(df_mirrored, r_axis, angle)
                        
                        # Finally apply scaling to the mirrored + rotated data
                        df_full_cascade = scale_to_new_df(df_mirror_rotate, factors[0], factors[1], factors[2])
                        save_data(df_full_cascade, 'processed.augmentation.cascade', 
                                 f'{file_prefix}_mirror_{m_axis}_rotate_{r_axis}_{angle_deg}deg_{scale_str}.csv')

# df = pd.read_csv("centralized_squat.csv")
def main():
    print("\n=== AUGMENTATION PROCESSING ===")
    processing_dir = _get_full_path('unprocessed.bad_good')
    
    files = []
    if os.path.exists(processing_dir):
            files = [f for f in os.listdir(processing_dir) if f.endswith('.csv')]

    print(f"Files in {processing_dir}: {len(files)} files found")
    
    total_files = len(files)
    for i, file in enumerate(files):
        file_progress = (i / total_files) * 100
        print(f"\n[{file_progress:.1f}%] Processing file {i+1}/{total_files}: {file}")
        
        df = load_data('unprocessed.bad_good', file)
        current_file = file.split('.')[0]
        
        # Track augmentation steps
        # add_mirror_augmentation(df, current_file)
        # add_rotation_augmentation(df, current_file)
        # add_scale_augmentation(df, current_file)
        apply_cascade_augmentation(df, current_file)

        print(f"✓ Completed processing file: {file}")

    print("\n=== AUGMENTATION PROCESSING COMPLETE ===")
    print(f"Processed {total_files} files with {len(aug_config)} types of augmentations")
# original_df = load_data('unprocessed.bad_good', 'A1.csv')
# print("Original DataFrame:")
# print(original_df)

if __name__ == "__main__":
    main()
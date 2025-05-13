import numpy as np
import cv2
from data_loader import load_data, save_data

# Configuration for augmentation
aug_config = {
    'mirror': True,                # Enable/disable mirroring
    'rotate': True,                # Enable/disable rotation
    'rotation_angles': [90, 180, 270],  # Rotation angles in degrees
    'stretch_compress': True,      # Enable/disable stretching/compressing
    'stretch_factors': [0.8, 1.2], # Factors for stretching/compressing
}

def mirror_image(image):
    """
    Mirror an image horizontally
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Mirrored image
    """
    return cv2.flip(image, 1)  # 1 means horizontal flip

def rotate_image(image, angle):
    """
    Rotate an image by a specified angle
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def stretch_compress_image(image, factor):
    """
    Stretch or compress an image by a specified factor
    
    Args:
        image: Input image as numpy array
        factor: Stretch factor (>1 for stretch, <1 for compress)
        
    Returns:
        Stretched/compressed image
    """
    height, width = image.shape[:2]
    new_width = int(width * factor)
    # Resize while maintaining aspect ratio
    resized = cv2.resize(image, (new_width, height))
    
    # If the new image is smaller, pad it to the original size
    # If the new image is larger, crop it to the original size
    result = np.zeros_like(image)
    if factor < 1:  # Compressed (smaller width)
        start_x = (width - new_width) // 2
        result[:, start_x:start_x+new_width] = resized
    else:  # Stretched (larger width)
        start_x = (new_width - width) // 2
        result = resized[:, start_x:start_x+width]
    
    return result

def apply_augmentation(images, labels):
    """
    Apply configured augmentations to the dataset
    
    Args:
        images: List of input images
        labels: List of corresponding labels
        
    Returns:
        Tuple of augmented images and labels
    """
    augmented_images = []
    augmented_labels = []
    
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        
        # Add original image
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Apply mirroring if enabled
        if aug_config['mirror']:
            mirrored = mirror_image(image)
            augmented_images.append(mirrored)
            augmented_labels.append(label)
        
        # Apply rotation if enabled
        if aug_config['rotate']:
            for angle in aug_config['rotation_angles']:
                rotated = rotate_image(image, angle)
                augmented_images.append(rotated)
                augmented_labels.append(label)
        
        # Apply stretching/compressing if enabled
        if aug_config['stretch_compress']:
            for factor in aug_config['stretch_factors']:
                transformed = stretch_compress_image(image, factor)
                augmented_images.append(transformed)
                augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)

def main():
    """
    Main function to load data, apply augmentation, and save the augmented dataset
    """
    print("Loading original dataset...")
    images, labels = load_data()
    
    print(f"Original dataset size: {len(images)} images")
    
    print("Applying augmentation...")
    augmented_images, augmented_labels = apply_augmentation(images, labels)
    
    print(f"Augmented dataset size: {len(augmented_images)} images")
    
    print("Saving augmented dataset...")
    save_data(augmented_images, augmented_labels, "augmented_dataset")
    
    print("Data augmentation completed successfully!")

if __name__ == "__main__":
    main()

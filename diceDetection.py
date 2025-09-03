import matplotlib.pyplot as plt
import skimage
from skimage.color import *
from skimage import io
from skimage.feature import blob_doh
from sklearn.cluster import DBSCAN
import numpy as np

def load_and_preprocess_image(image_path):
    """Load and preprocess the image for dice detection."""
    image = io.imread(image_path)  # Load image
    image_hsv = rgb2hsv(image)  # Convert from RGB to HSV
    image_saturation = image_hsv[:, :, 1]  # Extract saturation channel
    return image, image_saturation

def binarize_and_clean_image(image_saturation):
    """Binarize and clean the image using morphological operations."""
    binary_image = image_saturation > 0.5  # Binarization
    
    # Apply morphological operations
    binary_image = skimage.morphology.opening(binary_image, skimage.morphology.disk(3))
    binary_image = skimage.morphology.erosion(binary_image, skimage.morphology.disk(1))
    
    return binary_image

def detect_dots(binary_image):
    """Detect dots in the binarized image."""
    detected_dots = blob_doh(binary_image, min_sigma=5, max_sigma=15, threshold=.05)
    return detected_dots

def calculate_dice_size(dot_coordinates, cluster_labels, cluster_id):
    """Calculate the size of a dice based on its dots distribution."""
    dots_in_dice = dot_coordinates[cluster_labels == cluster_id]
    
    if len(dots_in_dice) < 2:
        return dots_in_dice[0].size * 23  # Default size for single dot or very small clusters
    
    # Calculate bounding box of dots
    min_x, min_y = np.min(dots_in_dice, axis=0)
    max_x, max_y = np.max(dots_in_dice, axis=0)
    
    # Calculate width and height
    width = max_x - min_x
    height = max_y - min_y
    
    # Use the larger dimension plus some padding
    dice_size = max(width, height) * 1.5
    
    # Apply size constraints
    dice_size = max(30, min(100, dice_size))  # Limit between 30 and 100 pixels
    
    return dice_size

def cluster_dots(detected_dots):
    """Cluster dots into dice using DBSCAN."""
    if len(detected_dots) == 0:
        return None, None, None, None
    
    dot_coordinates = (detected_dots.T[0:2]).T  # Extract x and y coordinates
    
    # Apply DBSCAN clustering
    dot_clusters = DBSCAN(eps=35, min_samples=1).fit(dot_coordinates)
    num_dice = max(dot_clusters.labels_) + 1
    
    dice_values = []
    dice_centers = []
    dice_sizes = []
    
    for i in range(num_dice):
        dots_in_dice = dot_coordinates[dot_clusters.labels_ == i]
        dice_values.append(len(dots_in_dice))
        dice_center = np.mean(dots_in_dice, axis=0)
        dice_centers.append(dice_center)
        
        # Calculate dice size based on dot distribution
        dice_size = calculate_dice_size(dot_coordinates, dot_clusters.labels_, i)
        dice_sizes.append(dice_size)
    
    return np.array(dice_centers), dice_values, dot_clusters, dice_sizes

def visualize_results(image, detected_dots, dice_centers, dice_values, dice_sizes, image_name):
    """Visualize the detection results with dice size information."""
    figure, axes = plt.subplots(figsize=(12, 10))
    
    # Draw detected dots
    for i in range(len(detected_dots)):
        outer_circle = plt.Circle((detected_dots[i, 1], detected_dots[i, 0]), 
                                 detected_dots[i, 2], fill=True, color='red', alpha=0.7)
        inner_circle = plt.Circle((detected_dots[i, 1], detected_dots[i, 0]), 
                                 detected_dots[i, 2] / 2, fill=True, color='black', alpha=0.9)
        axes.add_artist(outer_circle)
        axes.add_artist(inner_circle)
    
    # Draw dice boundaries with calculated sizes
    if dice_centers is not None:
        for i in range(len(dice_centers)):
            dice_size = dice_sizes[i]
            dice_boundary = plt.Circle((dice_centers[i, 1], dice_centers[i, 0]), 
                                     dice_size, fill=False, color='blue', linewidth=2, linestyle='--')
            axes.add_artist(dice_boundary)
        
        # Add text with dice values and sizes
        text_y_position = 30
        for i in range(len(dice_centers)):
            text = f"Dice {i+1}: Value={dice_values[i]}, Size={dice_sizes[i]:.1f}px"
            plt.text(20, text_y_position, text, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            text_y_position += 40
    
    axes.set_aspect(1)
    plt.axis('off')
    axes.imshow(image)
    plt.title(f'Dice Detection with Size Calculation - {image_name}')
    plt.show()

def process_dice_image(image_path):
    """Main function to process a single dice image."""
    print(f"Processing {image_path}")
    
    # Load and preprocess image
    image, image_saturation = load_and_preprocess_image(image_path)
    
    # Binarize and clean image
    binary_image = binarize_and_clean_image(image_saturation)
    
    # Detect dots
    detected_dots = detect_dots(binary_image)
    
    # Cluster dots into dice and calculate sizes
    dice_centers, dice_values, dot_clusters, dice_sizes = cluster_dots(detected_dots)
    
    # Visualize results
    visualize_results(image, detected_dots, dice_centers, dice_values, dice_sizes, image_path)
    
    return dice_values, dice_sizes

def main():
    """Main function to process all dice images."""
    NUM_SAMPLES = 13
    
    for i in range(1, NUM_SAMPLES + 1):
        image_path = f'dices{i}.jpg'
        try:
            dice_values, dice_sizes = process_dice_image(image_path)
            print(f"Dice values for {image_path}: {dice_values}")
            print(f"Dice sizes for {image_path}: {[f'{size:.1f}px' for size in dice_sizes]}")
            if dice_sizes:
                print(f"Average dice size: {np.mean(dice_sizes):.1f}px\n")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    main()
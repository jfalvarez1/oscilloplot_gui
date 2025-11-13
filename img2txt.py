import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def image_to_xy_coordinates(image_path, num_points=1500, edge_threshold1=50, edge_threshold2=150):
    """
    Convert an image to X,Y coordinate arrays for oscilloscope display.
    
    Parameters:
    - image_path: Path to input image
    - num_points: Target number of points (will affect smoothness and refresh rate)
    - edge_threshold1: Lower threshold for Canny edge detection
    - edge_threshold2: Upper threshold for Canny edge detection
    
    Returns:
    - x_coords: Array of x coordinates
    - y_coords: Array of y coordinates
    """
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)
    
    # Find contours (edge traces)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Combine all contours into single path
    all_points = []
    for contour in contours:
        # Only include contours with sufficient points
        if len(contour) > 10:
            points = contour.reshape(-1, 2)
            all_points.extend(points)
    
    if len(all_points) == 0:
        raise ValueError("No edges detected. Try adjusting edge thresholds.")
    
    all_points = np.array(all_points)
    
    # Resample to target number of points
    if len(all_points) > num_points:
        # Downsample
        indices = np.linspace(0, len(all_points) - 1, num_points, dtype=int)
        points = all_points[indices]
    else:
        points = all_points
    
    # Extract x and y coordinates
    x_coords = points[:, 0].astype(float)
    y_coords = points[:, 1].astype(float)
    
    # Normalize to -1 to 1 range (for oscilloscope)
    x_norm = 2 * (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) - 1
    y_norm = 2 * (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) - 1
    
    # Flip y-axis (image coordinates are top-down, scope is bottom-up)
    y_norm = -y_norm
    
    return x_norm, y_norm, edges


def trace_single_contour(image_path, contour_index=0, num_points=1500, 
                         edge_threshold1=50, edge_threshold2=150):
    """
    Trace a single continuous contour (better for clean images with one main object).
    """
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        raise ValueError("No contours found")
    
    # Sort by contour size (perimeter)
    contours = sorted(contours, key=cv2.contourPerimeter, reverse=True)
    
    # Use specified contour (default: largest)
    if contour_index >= len(contours):
        contour_index = 0
    
    contour = contours[contour_index].reshape(-1, 2)
    
    # Resample to target points
    if len(contour) > num_points:
        indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
        points = contour[indices]
    else:
        points = contour
    
    x_coords = points[:, 0].astype(float)
    y_coords = points[:, 1].astype(float)
    
    # Normalize
    x_norm = 2 * (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) - 1
    y_norm = 2 * (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) - 1
    y_norm = -y_norm
    
    return x_norm, y_norm, edges


def save_matlab_arrays(x_coords, y_coords, output_file='coordinates.txt'):
    """
    Save normalized coordinates as text file with x=[] and y=[] format.
    """
    
    with open(output_file, 'w') as f:
        # Write x array
        f.write("x_fun=[")
        f.write(','.join(f'{val:.6f}' for val in x_coords))
        f.write("];\n\n")
        
        # Write y array
        f.write("y_fun=[")
        f.write(','.join(f'{val:.6f}' for val in y_coords))
        f.write("];\n")
    
    print(f"Coordinates saved to {output_file}")


def visualize_trace(x_coords, y_coords, edges=None):
    """
    Visualize the traced coordinates and original edges.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot edges if available
    if edges is not None:
        axes[0].imshow(edges, cmap='gray')
        axes[0].set_title('Detected Edges')
        axes[0].axis('off')
    
    # Plot trace
    axes[1].plot(x_coords, y_coords, 'b-', linewidth=0.5)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Oscilloscope Trace ({len(x_coords)} points)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path> [num_points]")
        print("Example: python script.py image.png 2000")
        sys.exit(1)
    
    image_file = sys.argv[1]
    num_points = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    print(f"Processing image: {image_file}")
    print(f"Target points: {num_points}")
    
    try:
        # Option A: Trace all edges
        x, y, edges = image_to_xy_coordinates(
            image_file, 
            num_points=num_points,  # Use command line argument
            edge_threshold1=50,  # Lower = more edges detected
            edge_threshold2=150   # Higher = fewer edges
        )
        
        # Option B: Trace single largest contour (uncomment to use)
        # x, y, edges = trace_single_contour(
        #     image_file,
        #     num_points=1000,
        #     contour_index=0  # 0 = largest contour
        # )
        
        print(f"Generated {len(x)} coordinate points")
        print(f"Refresh rate at Fs*100: {100000/len(x):.1f} Hz")
        
        # Visualize
        visualize_trace(x, y, edges)
        
        # Save as text file with normalized coordinates
        save_matlab_arrays(x, y, 'coordinates.txt')
        print(f"Copy the arrays from coordinates.txt into your MATLAB script")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure image file path is correct")
        print("2. Try adjusting edge_threshold1 and edge_threshold2")
        print("3. Use trace_single_contour() for cleaner images")

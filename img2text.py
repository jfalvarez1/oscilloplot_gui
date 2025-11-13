import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import messagebox

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


class InteractiveEditor:
    """
    Interactive editor for post-processing extracted coordinates.
    - Left click: Erase nearby points
    - Right click: Add point
    - 's' key: Save to file
    - 'r' key: Reset to original
    """

    def __init__(self, x_coords, y_coords, edges=None, output_file='coordinates.txt'):
        self.x_original = x_coords.copy()
        self.y_original = y_coords.copy()
        self.x_coords = x_coords.copy()
        self.y_coords = y_coords.copy()
        self.edges = edges
        self.output_file = output_file
        self.erase_radius = 0.1  # Radius for erasing points

        # Create figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot edges
        if edges is not None:
            self.axes[0].imshow(edges, cmap='gray')
            self.axes[0].set_title('Detected Edges (Reference)')
            self.axes[0].axis('off')
        else:
            self.axes[0].text(0.5, 0.5, 'No edge image available',
                             ha='center', va='center', transform=self.axes[0].transAxes)
            self.axes[0].axis('off')

        # Plot editable trace
        self.line, = self.axes[1].plot(self.x_coords, self.y_coords, 'b.',
                                       markersize=2, picker=5)
        self.axes[1].set_aspect('equal')
        self.update_title()
        self.axes[1].set_xlabel('X')
        self.axes[1].set_ylabel('Y')
        self.axes[1].grid(True, alpha=0.3)

        # Add instruction text
        instruction_text = (
            'LEFT CLICK: Erase points | RIGHT CLICK: Add point\n'
            'SCROLL: Adjust erase radius | S: Save | R: Reset | Q: Quit'
        )
        self.fig.text(0.5, 0.02, instruction_text, ha='center',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Add control buttons
        self.add_buttons()

        plt.tight_layout(rect=[0, 0.06, 1, 1])

    def update_title(self):
        """Update the title with current point count"""
        self.axes[1].set_title(
            f'Editable Trace ({len(self.x_coords)} points) | '
            f'Erase radius: {self.erase_radius:.3f}'
        )

    def add_buttons(self):
        """Add control buttons to the figure"""
        from matplotlib.widgets import Button

        # Save button
        ax_save = plt.axes([0.35, 0.92, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.save_coordinates)

        # Reset button
        ax_reset = plt.axes([0.45, 0.92, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_coordinates)

        # Quit button
        ax_quit = plt.axes([0.55, 0.92, 0.08, 0.04])
        self.btn_quit = Button(ax_quit, 'Quit')
        self.btn_quit.on_clicked(self.quit_editor)

    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.axes[1]:
            return

        if event.button == 1:  # Left click - erase
            self.erase_points(event.xdata, event.ydata)
        elif event.button == 3:  # Right click - add
            self.add_point(event.xdata, event.ydata)

    def on_scroll(self, event):
        """Handle scroll events to adjust erase radius"""
        if event.button == 'up':
            self.erase_radius *= 1.2
        elif event.button == 'down':
            self.erase_radius /= 1.2

        self.erase_radius = max(0.01, min(0.5, self.erase_radius))
        self.update_title()
        self.fig.canvas.draw()

    def on_key(self, event):
        """Handle key press events"""
        if event.key == 's':
            self.save_coordinates(None)
        elif event.key == 'r':
            self.reset_coordinates(None)
        elif event.key == 'q':
            self.quit_editor(None)

    def erase_points(self, x, y):
        """Erase points within radius of click"""
        if x is None or y is None:
            return

        # Calculate distances
        distances = np.sqrt((self.x_coords - x)**2 + (self.y_coords - y)**2)

        # Keep points outside erase radius
        mask = distances > self.erase_radius

        points_removed = np.sum(~mask)
        if points_removed > 0:
            self.x_coords = self.x_coords[mask]
            self.y_coords = self.y_coords[mask]
            self.update_plot()
            print(f"Erased {points_removed} points")

    def add_point(self, x, y):
        """Add a new point at click location"""
        if x is None or y is None:
            return

        # Find nearest point to determine insertion index
        if len(self.x_coords) > 0:
            distances = np.sqrt((self.x_coords - x)**2 + (self.y_coords - y)**2)
            nearest_idx = np.argmin(distances)

            # Insert near the closest point
            self.x_coords = np.insert(self.x_coords, nearest_idx, x)
            self.y_coords = np.insert(self.y_coords, nearest_idx, y)
        else:
            # First point
            self.x_coords = np.array([x])
            self.y_coords = np.array([y])

        self.update_plot()
        print(f"Added point at ({x:.3f}, {y:.3f})")

    def update_plot(self):
        """Refresh the plot with current coordinates"""
        self.line.set_data(self.x_coords, self.y_coords)
        self.update_title()
        self.fig.canvas.draw()

    def save_coordinates(self, event):
        """Save current coordinates to file"""
        if len(self.x_coords) == 0:
            print("Error: No points to save!")
            return

        save_matlab_arrays(self.x_coords, self.y_coords, self.output_file)
        print(f"✓ Saved {len(self.x_coords)} points to {self.output_file}")

    def reset_coordinates(self, event):
        """Reset to original coordinates"""
        self.x_coords = self.x_original.copy()
        self.y_coords = self.y_original.copy()
        self.update_plot()
        print("Reset to original coordinates")

    def quit_editor(self, event):
        """Close the editor"""
        plt.close(self.fig)

    def show(self):
        """Display the interactive editor"""
        plt.show()


def edit_coordinates_interactive(x_coords, y_coords, edges=None, output_file='coordinates.txt'):
    """
    Launch interactive editor for post-processing coordinates.

    Usage:
    - Left click: Erase nearby points
    - Right click: Add a point
    - Scroll wheel: Adjust erase radius
    - 's' key or Save button: Save to file
    - 'r' key or Reset button: Reset to original
    - 'q' key or Quit button: Close editor
    """
    editor = InteractiveEditor(x_coords, y_coords, edges, output_file)
    editor.show()
    return editor.x_coords, editor.y_coords


# Example usage
if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python img2text.py <image_path> [num_points] [--edit]")
        print("\nOptions:")
        print("  <image_path>   Path to the input image")
        print("  [num_points]   Target number of points (default: 1000)")
        print("  --edit         Launch interactive editor for post-processing")
        print("\nExamples:")
        print("  python img2text.py image.png 2000")
        print("  python img2text.py image.png 2000 --edit")
        print("\nInteractive Editor Controls:")
        print("  - LEFT CLICK: Erase nearby points")
        print("  - RIGHT CLICK: Add a point")
        print("  - SCROLL WHEEL: Adjust erase radius")
        print("  - S key: Save to coordinates.txt")
        print("  - R key: Reset to original")
        print("  - Q key: Quit editor")
        sys.exit(1)

    image_file = sys.argv[1]

    # Check for --edit flag
    use_editor = '--edit' in sys.argv

    # Get num_points (filter out --edit from args)
    args_without_flags = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
    num_points = int(args_without_flags[0]) if len(args_without_flags) > 0 else 1000

    print(f"Processing image: {image_file}")
    print(f"Target points: {num_points}")
    if use_editor:
        print("Interactive editor mode enabled")

    try:
        # Option A: Trace all edges (default)
        x, y, edges = image_to_xy_coordinates(
            image_file,
            num_points=num_points,
            edge_threshold1=50,  # Lower = more edges detected
            edge_threshold2=150   # Higher = fewer edges
        )

        # Option B: Trace single largest contour (uncomment to use instead)
        # x, y, edges = trace_single_contour(
        #     image_file,
        #     num_points=num_points,
        #     contour_index=0  # 0 = largest contour
        # )

        print(f"Generated {len(x)} coordinate points")
        print(f"Refresh rate at 100kHz: {100000/len(x):.1f} Hz")

        if use_editor:
            # Launch interactive editor
            print("\n=== Launching Interactive Editor ===")
            print("Controls:")
            print("  LEFT CLICK: Erase points | RIGHT CLICK: Add point")
            print("  SCROLL: Adjust erase radius | S: Save | R: Reset | Q: Quit")
            print("\nEdit the trace, then press 'S' to save or use the Save button")

            x_edited, y_edited = edit_coordinates_interactive(x, y, edges, 'coordinates.txt')
            print(f"\nFinal point count: {len(x_edited)}")
        else:
            # Just visualize and save
            visualize_trace(x, y, edges)
            save_matlab_arrays(x, y, 'coordinates.txt')
            print(f"\n✓ Saved to coordinates.txt")
            print("Load this file in the oscilloscope GUI using 'Load Text File (.txt)'")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure image file path is correct")
        print("2. Try adjusting edge_threshold1 and edge_threshold2")
        print("3. Use trace_single_contour() for cleaner images")
        print("4. Install required packages: pip install opencv-python numpy scipy matplotlib")

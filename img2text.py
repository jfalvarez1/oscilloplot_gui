import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import messagebox


def sharpen_image(img):
    """Apply sharpening filter to enhance edges"""
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def skeletonize_edges(edges):
    """Reduce edges to single pixel thickness"""
    # Convert to binary
    binary = (edges > 0).astype(np.uint8)
    # Apply skeletonization to get single-pixel thickness
    skeleton = skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)


def vectorize_contours(contours, num_points, min_contour_length=50):
    """
    Vectorize contours by following paths naturally instead of random sampling.
    Filters out single dots and small artifacts.

    Parameters:
    - contours: List of contours from cv2.findContours
    - num_points: Target number of points to resample
    - min_contour_length: Minimum perimeter length to keep (filters out dots/noise)

    Returns coordinates that follow the natural path of contours.
    """
    all_points = []

    for contour in contours:
        # Calculate contour perimeter (length)
        perimeter = cv2.arcLength(contour, closed=True)

        # Filter out small contours (single dots, small artifacts)
        # Keep only contours with sufficient length (actual lines/outlines)
        if perimeter > min_contour_length and len(contour) > 10:
            # Reshape contour to get points
            points = contour.reshape(-1, 2)
            # Add points in order (following the path)
            all_points.extend(points)

    if len(all_points) == 0:
        return np.array([]), np.array([])

    all_points = np.array(all_points)

    # Resample to target number of points while maintaining path order
    if len(all_points) > num_points:
        # Use linear interpolation to resample evenly along the path
        indices = np.linspace(0, len(all_points) - 1, num_points, dtype=int)
        points = all_points[indices]
    else:
        points = all_points

    return points[:, 0], points[:, 1]


def image_to_xy_coordinates_outlines(image_path, num_points=1500, threshold_value=127, invert=False):
    """
    Convert an image to X,Y coordinate arrays by tracing OUTLINES only.
    Optimized for tracing outlines of humans and objects (not filled shapes).
    Filters out single dots and small artifacts.

    Parameters:
    - image_path: Path to input image
    - num_points: Target number of points
    - threshold_value: Binary threshold (0-255). Lower = more white becomes outline
    - invert: If True, trace dark objects on light background

    Returns:
    - x_coords: Array of x coordinates
    - y_coords: Array of y coordinates
    - binary_img: Binary threshold image for reference
    """

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpen the image to enhance edges
    sharpened = sharpen_image(gray)

    # Apply slight blur to reduce noise (after sharpening)
    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)

    # Apply binary threshold to separate object from background
    if invert:
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Opening: erosion followed by dilation - removes small dots and noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing: dilation followed by erosion - closes small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Skeletonize to get single-pixel thickness outlines
    skeleton = skeletonize_edges(binary)

    # Find contours (EXTERNAL only - just outlines, no interior details)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("No outlines detected. Try adjusting threshold value.")

    # Sort contours by perimeter (largest first)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

    # Calculate minimum contour length based on image size to filter out artifacts
    img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    min_contour_length = max(50, img_diagonal * 0.02)  # At least 2% of image diagonal

    # Vectorize contours (follow paths naturally, filter out small artifacts)
    x_coords, y_coords = vectorize_contours(contours, num_points, min_contour_length)

    if len(x_coords) == 0:
        raise ValueError("No valid contours found. Try adjusting threshold or lowering filtering.")

    # Normalize to -1 to 1 range
    x_norm = 2 * (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) - 1
    y_norm = 2 * (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) - 1

    # Flip y-axis
    y_norm = -y_norm

    return x_norm, y_norm, skeleton


def image_to_xy_coordinates(image_path, num_points=1500, edge_threshold1=50, edge_threshold2=150):
    """
    Convert an image to X,Y coordinate arrays for oscilloscope display using EDGE DETECTION.
    Use this for line drawings or when you want interior details.
    Filters out single dots and small artifacts.

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

    # Sharpen the image to enhance edges
    sharpened = sharpen_image(gray)

    # Apply Gaussian blur to reduce noise (after sharpening)
    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)

    # Detect edges using Canny
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)

    # Apply morphological operations to remove noise and small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Opening: erosion followed by dilation - removes small dots and noise
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing: dilation followed by erosion - closes small holes
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Skeletonize to get single-pixel thickness edges
    skeleton = skeletonize_edges(edges)

    # Find contours (edge traces)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("No edges detected. Try adjusting edge thresholds.")

    # Sort contours by perimeter (largest first) for better path ordering
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

    # Calculate minimum contour length based on image size to filter out artifacts
    img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    min_contour_length = max(50, img_diagonal * 0.02)  # At least 2% of image diagonal

    # Vectorize contours (follow paths naturally, filter out small artifacts)
    x_coords, y_coords = vectorize_contours(contours, num_points, min_contour_length)

    if len(x_coords) == 0:
        raise ValueError("No valid contours found. Try adjusting edge thresholds or lowering filtering.")

    # Normalize to -1 to 1 range (for oscilloscope)
    x_norm = 2 * (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) - 1
    y_norm = 2 * (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) - 1

    # Flip y-axis (image coordinates are top-down, scope is bottom-up)
    y_norm = -y_norm

    return x_norm, y_norm, skeleton


def trace_single_contour(image_path, contour_index=0, num_points=1500,
                         edge_threshold1=50, edge_threshold2=150):
    """
    Trace a single continuous contour (better for clean images with one main object).
    Now with sharpening and single-pixel thickness.
    """

    # Read and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpen the image
    sharpened = sharpen_image(gray)

    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
    edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)

    # Skeletonize to single-pixel thickness
    skeleton = skeletonize_edges(edges)

    # Find contours
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("No contours found")

    # Sort by contour size (perimeter)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

    # Use specified contour (default: largest)
    if contour_index >= len(contours):
        contour_index = 0

    contour = contours[contour_index].reshape(-1, 2)

    # Resample to target points while maintaining path order
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

    return x_norm, y_norm, skeleton


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
    - Sliders: Adjust detection sensitivity (mode-dependent)
    """

    def __init__(self, x_coords, y_coords, edges=None, output_file='coordinates.txt',
                 image_path=None, num_points=1000, mode='outline',
                 edge_threshold1=50, edge_threshold2=150, threshold_value=127, invert=False):
        self.x_original = x_coords.copy()
        self.y_original = y_coords.copy()
        self.x_coords = x_coords.copy()
        self.y_coords = y_coords.copy()
        self.edges = edges
        self.output_file = output_file
        self.erase_radius = 0.1  # Radius for erasing points
        self.add_radius = 0.05  # Radius for adding points (smaller than erase)
        self.active_radius_mode = 'erase'  # 'erase' or 'add'

        # Store parameters for re-running detection
        self.image_path = image_path
        self.num_points = num_points
        self.mode = mode  # 'outline' or 'edge'

        # Edge detection parameters
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2

        # Outline detection parameters
        self.threshold_value = threshold_value
        self.invert = invert

        # Create figure with more space for sliders
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot edges
        if edges is not None:
            self.edge_img = self.axes[0].imshow(edges, cmap='gray')
            self.axes[0].set_title('Detected Edges (Reference)')
            self.axes[0].axis('off')
        else:
            self.edge_img = None
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

        # Add preview circle for radius visualization
        self.preview_circle = plt.Circle((0, 0), 0.1, fill=False, color='red',
                                        linestyle='--', visible=False)
        self.axes[1].add_patch(self.preview_circle)

        # Add instruction text
        if self.image_path is not None:
            if self.mode == 'outline':
                instruction_text = (
                    'LEFT DRAG: Erase | RIGHT DRAG: Add | SCROLL: Adjust radius | TAB: Toggle radius mode\n'
                    'SLIDER: Threshold | CHECKBOX: Invert | APPLY: Update detection | S: Save | R: Reset | Q: Quit'
                )
            else:
                instruction_text = (
                    'LEFT DRAG: Erase | RIGHT DRAG: Add | SCROLL: Adjust radius | TAB: Toggle radius mode\n'
                    'SLIDERS: Edge sensitivity | APPLY: Update detection | S: Save | R: Reset | Q: Quit'
                )
        else:
            instruction_text = (
                'LEFT DRAG: Erase | RIGHT DRAG: Add | SCROLL: Adjust radius | TAB: Toggle radius mode\n'
                'S: Save | R: Reset | Q: Quit'
            )
        self.fig.text(0.5, 0.02, instruction_text, ha='center',
                     fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Mouse button state tracking for continuous erase/add
        self.mouse_pressed = False
        self.mouse_button = None

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Add control buttons and sliders
        self.add_buttons()
        if self.image_path is not None:
            self.add_sliders()

        plt.tight_layout(rect=[0, 0.06, 1, 1])

    def update_title(self):
        """Update the title with current point count"""
        mode_indicator = "ERASE" if self.active_radius_mode == 'erase' else "ADD"
        active_radius = self.erase_radius if self.active_radius_mode == 'erase' else self.add_radius
        self.axes[1].set_title(
            f'Editable Trace ({len(self.x_coords)} points) | '
            f'Mode: {mode_indicator} | Radius: {active_radius:.3f} | '
            f'[TAB to switch]'
        )

    def add_buttons(self):
        """Add control buttons to the figure"""
        from matplotlib.widgets import Button

        # Apply button (for detection updates)
        ax_apply = plt.axes([0.28, 0.92, 0.08, 0.04])
        self.btn_apply = Button(ax_apply, 'Apply')
        self.btn_apply.on_clicked(lambda e: self.update_detection(None))

        # Save button
        ax_save = plt.axes([0.38, 0.92, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.save_coordinates)

        # Reset button
        ax_reset = plt.axes([0.48, 0.92, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_coordinates)

        # Quit button
        ax_quit = plt.axes([0.58, 0.92, 0.08, 0.04])
        self.btn_quit = Button(ax_quit, 'Quit')
        self.btn_quit.on_clicked(self.quit_editor)

    def add_sliders(self):
        """Add sliders for detection threshold adjustment (mode-dependent)"""
        from matplotlib.widgets import Slider, CheckButtons

        if self.mode == 'outline':
            # Slider for binary threshold
            ax_slider1 = plt.axes([0.15, 0.12, 0.25, 0.02])
            self.slider_threshold = Slider(
                ax_slider1, 'Threshold',
                valmin=0, valmax=255, valinit=self.threshold_value,
                valstep=1, color='lightgreen'
            )
            # Note: No on_changed callback - use Apply button instead

            # Checkbox for invert
            ax_check = plt.axes([0.15, 0.08, 0.15, 0.03])
            self.check_invert = CheckButtons(ax_check, ['Invert (dark→light)'], [self.invert])
            # Note: No on_clicked callback - use Apply button instead

        elif self.mode == 'edge':
            # Slider for lower threshold (edge_threshold1)
            ax_slider1 = plt.axes([0.15, 0.12, 0.25, 0.02])
            self.slider1 = Slider(
                ax_slider1, 'Low Threshold',
                valmin=1, valmax=200, valinit=self.edge_threshold1,
                valstep=1, color='lightblue'
            )
            # Note: No on_changed callback - use Apply button instead

            # Slider for upper threshold (edge_threshold2)
            ax_slider2 = plt.axes([0.15, 0.09, 0.25, 0.02])
            self.slider2 = Slider(
                ax_slider2, 'High Threshold',
                valmin=1, valmax=300, valinit=self.edge_threshold2,
                valstep=1, color='lightcoral'
            )
            # Note: No on_changed callback - use Apply button instead

    def update_detection(self, val):
        """Re-run detection with new parameter values (mode-dependent)"""
        if self.image_path is None:
            return

        try:
            if self.mode == 'outline':
                # Get current threshold value
                self.threshold_value = int(self.slider_threshold.val)

                # Get checkbox state (returns list of active checkboxes)
                self.invert = self.check_invert.get_status()[0] if hasattr(self, 'check_invert') else self.invert

                print(f"Updating outline detection: Threshold={self.threshold_value}, Invert={self.invert}")

                # Re-run outline detection
                x_new, y_new, binary_img = image_to_xy_coordinates_outlines(
                    self.image_path,
                    num_points=self.num_points,
                    threshold_value=self.threshold_value,
                    invert=self.invert
                )

                reference_img = binary_img
                title = 'Binary Threshold (Reference)'

            elif self.mode == 'edge':
                # Get current slider values
                self.edge_threshold1 = int(self.slider1.val)
                self.edge_threshold2 = int(self.slider2.val)

                # Ensure threshold1 < threshold2
                if self.edge_threshold1 >= self.edge_threshold2:
                    return

                print(f"Updating edge detection: Low={self.edge_threshold1}, High={self.edge_threshold2}")

                # Re-run edge detection
                x_new, y_new, edges_img = image_to_xy_coordinates(
                    self.image_path,
                    num_points=self.num_points,
                    edge_threshold1=self.edge_threshold1,
                    edge_threshold2=self.edge_threshold2
                )

                reference_img = edges_img
                title = 'Detected Edges (Reference)'

            # Update reference image display
            if self.edge_img is not None:
                self.edge_img.set_data(reference_img)
            else:
                self.edge_img = self.axes[0].imshow(reference_img, cmap='gray')
                self.axes[0].set_title(title)
                self.axes[0].axis('off')

            # Update coordinates
            self.x_original = x_new.copy()
            self.y_original = y_new.copy()
            self.x_coords = x_new.copy()
            self.y_coords = y_new.copy()

            # Update trace plot
            self.line.set_data(self.x_coords, self.y_coords)
            self.update_title()

            # Redraw
            self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating detection: {e}")

    def on_press(self, event):
        """Handle mouse button press events"""
        if event.inaxes != self.axes[1]:
            return

        self.mouse_pressed = True
        self.mouse_button = event.button

        # Perform initial action
        if event.button == 1:  # Left button - erase
            self.erase_points(event.xdata, event.ydata)
        elif event.button == 3:  # Right button - add
            self.add_point(event.xdata, event.ydata)

    def on_release(self, event):
        """Handle mouse button release events"""
        self.mouse_pressed = False
        self.mouse_button = None
        self.preview_circle.set_visible(False)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        """Handle mouse motion events"""
        if event.inaxes != self.axes[1]:
            self.preview_circle.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        # Show preview circle at cursor
        if event.xdata is not None and event.ydata is not None:
            current_radius = self.erase_radius if self.active_radius_mode == 'erase' else self.add_radius
            color = 'red' if self.active_radius_mode == 'erase' else 'green'
            self.preview_circle.set_center((event.xdata, event.ydata))
            self.preview_circle.set_radius(current_radius)
            self.preview_circle.set_color(color)
            self.preview_circle.set_visible(True)

            # If mouse button is held, continuously erase/add
            if self.mouse_pressed:
                if self.mouse_button == 1:  # Left button - continuous erase
                    self.erase_points(event.xdata, event.ydata)
                elif self.mouse_button == 3:  # Right button - continuous add
                    self.add_point(event.xdata, event.ydata)

            self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        """Handle scroll events to adjust radius (based on active mode)"""
        if self.active_radius_mode == 'erase':
            if event.button == 'up':
                self.erase_radius *= 1.2
            elif event.button == 'down':
                self.erase_radius /= 1.2
            self.erase_radius = max(0.01, min(0.5, self.erase_radius))
        else:  # add mode
            if event.button == 'up':
                self.add_radius *= 1.2
            elif event.button == 'down':
                self.add_radius /= 1.2
            self.add_radius = max(0.005, min(0.3, self.add_radius))

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
        elif event.key == 'tab':
            # Toggle between erase and add radius modes
            self.active_radius_mode = 'add' if self.active_radius_mode == 'erase' else 'erase'
            self.update_title()
            self.fig.canvas.draw()
            print(f"Switched to {self.active_radius_mode.upper()} mode")

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
        """Add a new point at click location if not too close to existing points"""
        if x is None or y is None:
            return

        # Check if there's already a point very close (within add_radius)
        if len(self.x_coords) > 0:
            distances = np.sqrt((self.x_coords - x)**2 + (self.y_coords - y)**2)
            min_distance = np.min(distances)

            # Don't add if too close to existing point (prevents clustering during drag)
            if min_distance < self.add_radius * 0.5:
                return

            nearest_idx = np.argmin(distances)

            # Insert near the closest point
            self.x_coords = np.insert(self.x_coords, nearest_idx, x)
            self.y_coords = np.insert(self.y_coords, nearest_idx, y)
        else:
            # First point
            self.x_coords = np.array([x])
            self.y_coords = np.array([y])

        self.update_plot()

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


def edit_coordinates_interactive(x_coords, y_coords, edges=None, output_file='coordinates.txt',
                                image_path=None, num_points=1000, mode='outline',
                                edge_threshold1=50, edge_threshold2=150,
                                threshold_value=127, invert=False):
    """
    Launch interactive editor for post-processing coordinates.

    Usage:
    - Left click: Erase nearby points
    - Right click: Add a point
    - Scroll wheel: Adjust erase radius
    - Sliders: Adjust detection sensitivity (updates live, mode-dependent)
    - 's' key or Save button: Save to file
    - 'r' key or Reset button: Reset to original
    - 'q' key or Quit button: Close editor
    """
    editor = InteractiveEditor(x_coords, y_coords, edges, output_file,
                              image_path, num_points, mode,
                              edge_threshold1, edge_threshold2,
                              threshold_value, invert)
    editor.show()
    return editor.x_coords, editor.y_coords


# Example usage
if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python img2text.py <image_path> [num_points] [--edit] [--mode=MODE]")
        print("\nOptions:")
        print("  <image_path>   Path to the input image")
        print("  [num_points]   Target number of points (default: 1000)")
        print("  --edit         Launch interactive editor for post-processing")
        print("  --mode=MODE    Detection mode: 'outline' (default) or 'edge'")
        print("                 outline: Trace object outlines (best for filled shapes)")
        print("                 edge: Detect all edges (best for line drawings)")
        print("\nExamples:")
        print("  python img2text.py image.png 2000")
        print("  python img2text.py image.png 2000 --edit")
        print("  python img2text.py image.png 2000 --edit --mode=outline")
        print("  python img2text.py image.png 2000 --mode=edge")
        print("\nInteractive Editor Controls:")
        print("  - LEFT CLICK: Erase nearby points")
        print("  - RIGHT CLICK: Add a point")
        print("  - SCROLL WHEEL: Adjust erase radius")
        print("  - SLIDERS: Adjust detection sensitivity (live updates)")
        print("  - S key: Save to coordinates.txt")
        print("  - R key: Reset to original")
        print("  - Q key: Quit editor")
        sys.exit(1)

    image_file = sys.argv[1]

    # Check for flags
    use_editor = '--edit' in sys.argv

    # Get mode
    mode = 'outline'  # Default to outline mode
    for arg in sys.argv:
        if arg.startswith('--mode='):
            mode = arg.split('=')[1]
            if mode not in ['outline', 'edge']:
                print(f"Invalid mode: {mode}. Using 'outline' mode.")
                mode = 'outline'

    # Get num_points (filter out flags from args)
    args_without_flags = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
    num_points = int(args_without_flags[0]) if len(args_without_flags) > 0 else 1000

    print(f"Processing image: {image_file}")
    print(f"Target points: {num_points}")
    print(f"Mode: {mode}")
    if use_editor:
        print("Interactive editor mode enabled")

    try:
        # Run detection based on mode
        if mode == 'outline':
            print("Using OUTLINE mode (traces object boundaries, ignores interior)")
            x, y, reference_img = image_to_xy_coordinates_outlines(
                image_file,
                num_points=num_points,
                threshold_value=127,
                invert=False
            )
        elif mode == 'edge':
            print("Using EDGE mode (detects all edges including interior details)")
            x, y, reference_img = image_to_xy_coordinates(
                image_file,
                num_points=num_points,
                edge_threshold1=50,
                edge_threshold2=150
            )

        print(f"Generated {len(x)} coordinate points")
        print(f"Refresh rate at 100kHz: {100000/len(x):.1f} Hz")

        if use_editor:
            # Launch interactive editor
            print("\n=== Launching Interactive Editor ===")
            print("Controls:")
            print("  LEFT CLICK: Erase points | RIGHT CLICK: Add point")
            print("  SCROLL: Adjust erase radius")
            if mode == 'outline':
                print("  SLIDER: Adjust threshold (0-255)")
                print("  CHECKBOX: Toggle invert (dark objects vs light objects)")
            else:
                print("  SLIDERS: Adjust Low/High edge thresholds")
            print("  S: Save | R: Reset | Q: Quit")
            print("\nEdit the trace, then press 'S' to save or use the Save button")

            x_edited, y_edited = edit_coordinates_interactive(
                x, y, reference_img, 'coordinates.txt',
                image_path=image_file,
                num_points=num_points,
                mode=mode,
                edge_threshold1=50,
                edge_threshold2=150,
                threshold_value=127,
                invert=False
            )
            print(f"\nFinal point count: {len(x_edited)}")
        else:
            # Just visualize and save
            visualize_trace(x, y, reference_img)
            save_matlab_arrays(x, y, 'coordinates.txt')
            print(f"\n✓ Saved to coordinates.txt")
            print("Load this file in the oscilloscope GUI using 'Load Text File (.txt)'")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure image file path is correct")
        print("2. Try adjusting edge_threshold1 and edge_threshold2")
        print("3. Use trace_single_contour() for cleaner images")
        print("4. Install required packages: pip install opencv-python numpy scipy matplotlib scikit-image")

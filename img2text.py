import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
import sys


def sharpen_image(img, strength=1.0):
    """
    Apply adaptive sharpening filter to enhance edges.

    Parameters:
    - img: Grayscale image
    - strength: Sharpening strength (0.5 to 2.0, default 1.0)
    """
    # Unsharp mask method for better quality
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.0 + strength, gaussian, -strength, 0)
    return sharpened


def vectorize_image(image_path, threshold=127, invert=False, epsilon_factor=0.001,
                   min_line_length=20, sharpen_strength=1.5):
    """
    Vectorize an image to clean line segments.

    Process:
    1. Sharpen image to enhance edges
    2. Apply binary threshold
    3. Find contours (object outlines)
    4. Simplify contours to vector lines using Douglas-Peucker algorithm
    5. Filter out small artifacts

    Parameters:
    - image_path: Path to input image
    - threshold: Binary threshold value (0-255)
    - invert: If True, detect dark objects on light background
    - epsilon_factor: Line simplification factor (0.0001-0.01). Lower = more detail
    - min_line_length: Minimum line length to keep (filters artifacts)
    - sharpen_strength: Sharpening strength (0.5-2.0)

    Returns:
    - line_segments: List of simplified contours (vectorized lines)
    - preview_img: Binary image for preview
    """

    # Read and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Sharpen image to enhance edges
    sharpened = sharpen_image(gray, strength=sharpen_strength)

    # Step 2: Apply binary threshold
    if invert:
        _, binary = cv2.threshold(sharpened, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary = cv2.threshold(sharpened, threshold, 255, cv2.THRESH_BINARY)

    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 3: Find contours (only external outlines)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No outlines detected. Try adjusting threshold.")

    # Step 4: Vectorize contours using polygon approximation (Douglas-Peucker)
    img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    epsilon = epsilon_factor * img_diagonal  # Adaptive simplification tolerance

    vectorized_contours = []
    for contour in contours:
        # Simplify contour to vector lines
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

        # Filter by perimeter (line length)
        perimeter = cv2.arcLength(approx, closed=True)

        if perimeter > min_line_length and len(approx) > 2:
            vectorized_contours.append(approx)

    if len(vectorized_contours) == 0:
        raise ValueError("No valid lines found. Try lowering min_line_length or epsilon_factor.")

    # Sort by size (largest first)
    vectorized_contours = sorted(vectorized_contours,
                                 key=lambda c: cv2.arcLength(c, True),
                                 reverse=True)

    return vectorized_contours, binary


def contours_to_coordinates(contours, num_points):
    """
    Convert vectorized contours to coordinate arrays for oscilloscope.

    Parameters:
    - contours: List of simplified contours
    - num_points: Target number of points to resample

    Returns:
    - x_coords, y_coords: Normalized coordinate arrays
    """
    all_points = []

    # Extract all points from vectorized contours
    for contour in contours:
        points = contour.reshape(-1, 2)
        all_points.extend(points)

    if len(all_points) == 0:
        return np.array([]), np.array([])

    all_points = np.array(all_points)

    # Resample to target number of points
    if len(all_points) > num_points:
        indices = np.linspace(0, len(all_points) - 1, num_points, dtype=int)
        points = all_points[indices]
    else:
        points = all_points

    x_coords = points[:, 0].astype(float)
    y_coords = points[:, 1].astype(float)

    # Normalize to -1 to 1 range
    if x_coords.max() > x_coords.min():
        x_norm = 2 * (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) - 1
    else:
        x_norm = np.zeros_like(x_coords)

    if y_coords.max() > y_coords.min():
        y_norm = 2 * (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) - 1
    else:
        y_norm = np.zeros_like(y_coords)

    # Flip y-axis (image coordinates are top-down)
    y_norm = -y_norm

    return x_norm, y_norm


def save_coordinates(x_coords, y_coords, output_file='coordinates.txt'):
    """Save coordinates in oscilloscope format."""
    with open(output_file, 'w') as f:
        f.write("x_fun=[")
        f.write(','.join(f'{val:.6f}' for val in x_coords))
        f.write("];\n\n")

        f.write("y_fun=[")
        f.write(','.join(f'{val:.6f}' for val in y_coords))
        f.write("];\n")

    print(f"✓ Saved {len(x_coords)} points to {output_file}")


class InteractiveVectorEditor:
    """
    Interactive editor for vectorized image processing.

    Features:
    - Adjustable threshold slider
    - Line simplification control (epsilon)
    - Minimum line length filter
    - Invert checkbox
    - Interactive point editing (erase/add)
    - Apply button to update detection
    """

    def __init__(self, image_path, num_points=1500, threshold=127, invert=False,
                 epsilon_factor=0.001, min_line_length=20, output_file='coordinates.txt'):

        self.image_path = image_path
        self.num_points = num_points
        self.threshold = threshold
        self.invert = invert
        self.epsilon_factor = epsilon_factor
        self.min_line_length = min_line_length
        self.output_file = output_file
        self.sharpen_strength = 1.5

        # Editing parameters
        self.erase_radius = 0.1
        self.add_radius = 0.05
        self.active_radius_mode = 'erase'
        self.mouse_pressed = False
        self.mouse_button = None

        # Initial processing
        self.update_vectorization()

        # Create GUI
        self.setup_gui()

    def update_vectorization(self):
        """Run vectorization with current parameters."""
        try:
            print(f"Vectorizing: threshold={self.threshold}, epsilon={self.epsilon_factor:.4f}, "
                  f"min_length={self.min_line_length}, invert={self.invert}")

            contours, binary = vectorize_image(
                self.image_path,
                threshold=self.threshold,
                invert=self.invert,
                epsilon_factor=self.epsilon_factor,
                min_line_length=self.min_line_length,
                sharpen_strength=self.sharpen_strength
            )

            self.binary_img = binary
            self.contours = contours

            # Convert to coordinates
            x, y = contours_to_coordinates(contours, self.num_points)

            if len(x) == 0:
                raise ValueError("No points generated")

            self.x_original = x.copy()
            self.y_original = y.copy()
            self.x_coords = x.copy()
            self.y_coords = y.copy()

            print(f"✓ Generated {len(self.x_coords)} points from {len(contours)} vectorized outlines")

            return True

        except Exception as e:
            print(f"Error during vectorization: {e}")
            return False

    def setup_gui(self):
        """Create the interactive GUI."""
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7))

        # Left: Binary preview
        self.preview_img = self.axes[0].imshow(self.binary_img, cmap='gray')
        self.axes[0].set_title('Vectorized Preview (Binary Threshold)')
        self.axes[0].axis('off')

        # Right: Editable trace
        self.line, = self.axes[1].plot(self.x_coords, self.y_coords, 'b.',
                                       markersize=2, picker=5)
        self.axes[1].set_aspect('equal')
        self.axes[1].set_xlabel('X')
        self.axes[1].set_ylabel('Y')
        self.axes[1].grid(True, alpha=0.3)

        # Preview circle for editing
        self.preview_circle = plt.Circle((0, 0), 0.1, fill=False, color='red',
                                        linestyle='--', visible=False)
        self.axes[1].add_patch(self.preview_circle)

        self.update_title()

        # Add controls
        self.add_sliders()
        self.add_buttons()

        # Instructions
        instruction_text = (
            'SLIDERS: Threshold | Simplification | Min Length | Sharpening | '
            'CHECKBOX: Invert | APPLY: Update\n'
            'LEFT DRAG: Erase | RIGHT DRAG: Add | SCROLL: Radius | TAB: Toggle mode | '
            'S: Save | R: Reset | Q: Quit'
        )
        self.fig.text(0.5, 0.02, instruction_text, ha='center', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    def add_sliders(self):
        """Add control sliders."""
        # Threshold slider
        ax_thresh = plt.axes([0.15, 0.18, 0.25, 0.02])
        self.slider_thresh = Slider(ax_thresh, 'Threshold', 0, 255,
                                    valinit=self.threshold, valstep=1, color='lightgreen')

        # Epsilon (simplification) slider - logarithmic scale
        ax_epsilon = plt.axes([0.15, 0.15, 0.25, 0.02])
        self.slider_epsilon = Slider(ax_epsilon, 'Simplification', 0.0001, 0.01,
                                     valinit=self.epsilon_factor, color='lightblue')

        # Min line length slider
        ax_minlen = plt.axes([0.15, 0.12, 0.25, 0.02])
        self.slider_minlen = Slider(ax_minlen, 'Min Length', 5, 200,
                                    valinit=self.min_line_length, valstep=1, color='lightcoral')

        # Sharpening strength slider
        ax_sharpen = plt.axes([0.15, 0.09, 0.25, 0.02])
        self.slider_sharpen = Slider(ax_sharpen, 'Sharpening', 0.5, 2.5,
                                     valinit=self.sharpen_strength, color='lightyellow')

        # Invert checkbox
        ax_invert = plt.axes([0.15, 0.06, 0.15, 0.03])
        self.check_invert = CheckButtons(ax_invert, ['Invert (dark→light)'], [self.invert])

    def add_buttons(self):
        """Add control buttons."""
        # Apply button
        ax_apply = plt.axes([0.25, 0.93, 0.08, 0.04])
        self.btn_apply = Button(ax_apply, 'Apply')
        self.btn_apply.on_clicked(self.on_apply)

        # Save button
        ax_save = plt.axes([0.35, 0.93, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.on_save)

        # Reset button
        ax_reset = plt.axes([0.45, 0.93, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.on_reset)

        # Quit button
        ax_quit = plt.axes([0.55, 0.93, 0.08, 0.04])
        self.btn_quit = Button(ax_quit, 'Quit')
        self.btn_quit.on_clicked(self.on_quit)

    def update_title(self):
        """Update plot title with current info."""
        mode = "ERASE" if self.active_radius_mode == 'erase' else "ADD"
        radius = self.erase_radius if self.active_radius_mode == 'erase' else self.add_radius
        self.axes[1].set_title(
            f'Vectorized Trace ({len(self.x_coords)} points) | '
            f'Mode: {mode} | Radius: {radius:.3f}'
        )

    def on_apply(self, event):
        """Apply button: Update vectorization with new parameters."""
        self.threshold = int(self.slider_thresh.val)
        self.epsilon_factor = float(self.slider_epsilon.val)
        self.min_line_length = int(self.slider_minlen.val)
        self.sharpen_strength = float(self.slider_sharpen.val)
        self.invert = self.check_invert.get_status()[0]

        if self.update_vectorization():
            # Update displays
            self.preview_img.set_data(self.binary_img)
            self.line.set_data(self.x_coords, self.y_coords)
            self.update_title()
            self.fig.canvas.draw_idle()

    def on_save(self, event):
        """Save coordinates to file."""
        if len(self.x_coords) == 0:
            print("Error: No points to save!")
            return
        save_coordinates(self.x_coords, self.y_coords, self.output_file)

    def on_reset(self, event):
        """Reset to original coordinates."""
        self.x_coords = self.x_original.copy()
        self.y_coords = self.y_original.copy()
        self.line.set_data(self.x_coords, self.y_coords)
        self.update_title()
        self.fig.canvas.draw_idle()
        print("Reset to original")

    def on_quit(self, event):
        """Close editor."""
        plt.close(self.fig)

    def on_press(self, event):
        """Handle mouse press."""
        if event.inaxes != self.axes[1]:
            return
        self.mouse_pressed = True
        self.mouse_button = event.button

        if event.button == 1:  # Left - erase
            self.erase_points(event.xdata, event.ydata)
        elif event.button == 3:  # Right - add
            self.add_point(event.xdata, event.ydata)

    def on_release(self, event):
        """Handle mouse release."""
        self.mouse_pressed = False
        self.mouse_button = None
        self.preview_circle.set_visible(False)
        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        """Handle mouse motion."""
        if event.inaxes != self.axes[1]:
            self.preview_circle.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        if event.xdata is not None and event.ydata is not None:
            radius = self.erase_radius if self.active_radius_mode == 'erase' else self.add_radius
            color = 'red' if self.active_radius_mode == 'erase' else 'green'

            self.preview_circle.set_center((event.xdata, event.ydata))
            self.preview_circle.set_radius(radius)
            self.preview_circle.set_color(color)
            self.preview_circle.set_visible(True)

            if self.mouse_pressed:
                if self.mouse_button == 1:
                    self.erase_points(event.xdata, event.ydata)
                elif self.mouse_button == 3:
                    self.add_point(event.xdata, event.ydata)

            self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        """Adjust radius with scroll wheel."""
        if self.active_radius_mode == 'erase':
            self.erase_radius *= 1.2 if event.button == 'up' else 1/1.2
            self.erase_radius = max(0.01, min(0.5, self.erase_radius))
        else:
            self.add_radius *= 1.2 if event.button == 'up' else 1/1.2
            self.add_radius = max(0.005, min(0.3, self.add_radius))

        self.update_title()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Handle key presses."""
        if event.key == 's':
            self.on_save(None)
        elif event.key == 'r':
            self.on_reset(None)
        elif event.key == 'q':
            self.on_quit(None)
        elif event.key == 'tab':
            self.active_radius_mode = 'add' if self.active_radius_mode == 'erase' else 'erase'
            self.update_title()
            self.fig.canvas.draw_idle()
            print(f"Switched to {self.active_radius_mode.upper()} mode")

    def erase_points(self, x, y):
        """Erase points within radius."""
        if x is None or y is None or len(self.x_coords) == 0:
            return

        distances = np.sqrt((self.x_coords - x)**2 + (self.y_coords - y)**2)
        mask = distances > self.erase_radius

        removed = np.sum(~mask)
        if removed > 0:
            self.x_coords = self.x_coords[mask]
            self.y_coords = self.y_coords[mask]
            self.line.set_data(self.x_coords, self.y_coords)
            self.update_title()
            self.fig.canvas.draw_idle()

    def add_point(self, x, y):
        """Add a point at location."""
        if x is None or y is None:
            return

        if len(self.x_coords) > 0:
            distances = np.sqrt((self.x_coords - x)**2 + (self.y_coords - y)**2)
            if np.min(distances) < self.add_radius * 0.5:
                return

            nearest_idx = np.argmin(distances)
            self.x_coords = np.insert(self.x_coords, nearest_idx, x)
            self.y_coords = np.insert(self.y_coords, nearest_idx, y)
        else:
            self.x_coords = np.array([x])
            self.y_coords = np.array([y])

        self.line.set_data(self.x_coords, self.y_coords)
        self.update_title()
        self.fig.canvas.draw_idle()

    def show(self):
        """Display the editor."""
        plt.show()


def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("=" * 70)
        print("VECTORIZED IMAGE TO OSCILLOSCOPE CONVERTER")
        print("=" * 70)
        print("\nUsage: python img2text.py <image_path> [num_points] [--edit]")
        print("\nArguments:")
        print("  <image_path>  Path to input image")
        print("  [num_points]  Target number of points (default: 1500)")
        print("  --edit        Launch interactive editor")
        print("\nExamples:")
        print("  python img2text.py photo.jpg 2000 --edit")
        print("  python img2text.py drawing.png 1000")
        print("\nFeatures:")
        print("  • Sharpens image to enhance edges")
        print("  • Vectorizes using polygon approximation (clean lines)")
        print("  • Adjustable threshold for line detection")
        print("  • Line simplification control (less detail = cleaner)")
        print("  • Filters out single dots and small artifacts")
        print("  • Interactive editing with Apply button")
        print("\nInteractive Editor Controls:")
        print("  Sliders:")
        print("    - Threshold: Adjust line detection sensitivity")
        print("    - Simplification: Control line detail (lower = more detail)")
        print("    - Min Length: Filter out short lines/artifacts")
        print("    - Sharpening: Enhance edge detection")
        print("  Checkbox:")
        print("    - Invert: Detect dark objects on light background")
        print("  Editing:")
        print("    - LEFT DRAG: Erase points")
        print("    - RIGHT DRAG: Add points")
        print("    - SCROLL: Adjust radius")
        print("    - TAB: Toggle erase/add mode")
        print("  Keys:")
        print("    - APPLY button: Update detection with new settings")
        print("    - S: Save to coordinates.txt")
        print("    - R: Reset to original")
        print("    - Q: Quit")
        print("=" * 70)
        sys.exit(1)

    # Parse arguments
    image_file = sys.argv[1]
    use_editor = '--edit' in sys.argv

    args_without_flags = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
    num_points = int(args_without_flags[0]) if len(args_without_flags) > 0 else 1500

    print(f"\n{'='*70}")
    print(f"Processing: {image_file}")
    print(f"Target points: {num_points}")
    print(f"Mode: {'Interactive Editor' if use_editor else 'Auto-process'}")
    print(f"{'='*70}\n")

    try:
        if use_editor:
            # Launch interactive editor
            editor = InteractiveVectorEditor(
                image_file,
                num_points=num_points,
                threshold=127,
                invert=False,
                epsilon_factor=0.001,
                min_line_length=20
            )
            editor.show()
        else:
            # Auto-process and save
            print("Vectorizing image...")
            contours, binary = vectorize_image(image_file, threshold=127, invert=False)

            print(f"✓ Found {len(contours)} vectorized outlines")

            x, y = contours_to_coordinates(contours, num_points)
            print(f"✓ Generated {len(x)} coordinate points")
            print(f"✓ Refresh rate at 100kHz: {100000/len(x):.1f} Hz")

            # Visualize
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(binary, cmap='gray')
            axes[0].set_title('Vectorized Binary')
            axes[0].axis('off')

            axes[1].plot(x, y, 'b-', linewidth=0.5)
            axes[1].set_aspect('equal')
            axes[1].set_title(f'Oscilloscope Preview ({len(x)} points)')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Save
            save_coordinates(x, y, 'coordinates.txt')
            print("\n✓ Done! Load 'coordinates.txt' in the oscilloscope GUI")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check image path is correct")
        print("  2. Try --edit mode to adjust parameters interactively")
        print("  3. Install: pip install opencv-python numpy matplotlib")
        sys.exit(1)


if __name__ == "__main__":
    main()

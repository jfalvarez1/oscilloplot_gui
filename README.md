# Oscilloplot - Oscilloscope XY Audio Generator

A powerful Python GUI application for generating XY audio signals that can be displayed on a real analog oscilloscope. Create mesmerizing visual patterns, Lissajous figures, spirals, and custom drawings that come to life on your oscilloscope's phosphor screen!

![JUANTRONIX Screenshot](screenshots/main_gui.png)

## Features

- **Real-time XY Oscilloscope Preview** - Classic CRT green phosphor display with density-based rendering
- **Audio Output** - Generate stereo audio signals (Left = X, Right = Y) for real oscilloscope display
- **Multiple Pattern Generators**:
  - Sum of Harmonics (Lissajous figures)
  - Archimedean Spirals
  - Random Harmonics
  - Custom Test Patterns
  - Freehand Drawing Canvas
  - Sound Pad Sequencer (16-step)
- **Image to Coordinates Converter** - Convert photos and drawings to oscilloscope-ready coordinates
- **Real-time Effects**:
  - Rotation
  - Fade In/Out
  - Shrink
  - Wavy Modulation
- **Export Options** - Save audio as WAV files
- **Optimized Version** - 10-100x faster rendering with matplotlib blitting

## Screenshots

| Main Application | Sound Pad | Image Converter |
|-----------------|-----------|-----------------|
| ![Main GUI](screenshots/main_gui.png) | ![Sound Pad](screenshots/sound_pad.png) | ![Image Converter](screenshots/img2txt.png) |

> **Note**: To add screenshots, create a `screenshots/` folder and add PNG images of the application.

## Installation

### Prerequisites

- Python 3.8 or higher
- A stereo audio output device
- (Optional) An analog oscilloscope in XY mode

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy>=1.20.0 sounddevice>=0.4.0 matplotlib>=3.3.0 soundfile>=0.10.0
```

For the image converter, you'll also need:

```bash
pip install opencv-python
```

## Usage

### Main Application

#### Standard Version
```bash
python oscilloscope_gui.py
```

#### Optimized Version (Recommended)
```bash
python oscilloscope_gui_optimized.py
```

The optimized version provides dramatically better performance:
- 50-90% lower CPU usage
- 10-100x faster rendering
- Smoother animations at 50 FPS

### Quick Start

1. **Launch the application** - Run `oscilloscope_gui.py` or `oscilloscope_gui_optimized.py`
2. **Load or generate a pattern**:
   - Click "Generate Test Pattern" for a quick demo
   - Click "Sum of Harmonics" to create Lissajous figures
   - Click "Draw Pattern" to draw your own shape
3. **Adjust parameters** using the control panel on the left
4. **Click Play** to hear the audio and see the live preview
5. **Connect to oscilloscope** - Route stereo audio output to your scope's X and Y inputs

---

## Scripts Reference

### 1. `oscilloscope_gui.py` - Main Application

The primary GUI application for generating and playing XY oscilloscope patterns.

```bash
python oscilloscope_gui.py
```

**Features:**
- Load patterns from MATLAB (.m), text (.txt), or NumPy (.npz) files
- Generate patterns using built-in generators
- Real-time live preview with adjustable FPS
- Audio playback with stereo output (X = Left, Y = Right)
- Apply effects: rotation, fade, shrink, wavy modulation
- Export to WAV file

**Controls:**
| Section | Description |
|---------|-------------|
| **Data Source** | Load files or generate patterns |
| **Audio Parameters** | Sample rate, playback multiplier, duration |
| **Effects** | Rotation, fade, shrink, wavy modulation |
| **Live Preview** | Enable/disable, adjust window size and FPS |
| **Transport** | Play, Stop, Export WAV |

---

### 2. `oscilloscope_gui_optimized.py` - Optimized Version

Same features as the main application but with significant performance optimizations.

```bash
python oscilloscope_gui_optimized.py
```

**Key Optimizations:**
- **Matplotlib Blitting** - Only redraws changed elements (10-100x faster)
- **Circular NumPy Buffers** - Pre-allocated arrays eliminate memory allocations
- **Smart Decimation** - Adaptive downsampling for smooth rendering
- **Cached Background** - Background saved and restored for each frame

**Performance Comparison:**

| Metric | Original | Optimized |
|--------|----------|-----------|
| CPU Usage | 40-80% | 5-15% |
| Frame Time | 50-200ms | 2-8ms |
| Memory Allocs/sec | ~500 | ~10 |

**When to Use:**
- For general use (recommended)
- When using large window sizes (>5,000 samples)
- When CPU usage is a concern
- For longer sessions or presentations

---

### 3. `img2txt.py` - Image to Coordinates Converter

Convert photos and drawings into XY coordinates for oscilloscope display.

```bash
# Basic usage
python img2txt.py <image_path> [num_points]

# With interactive editor
python img2txt.py <image_path> [num_points] --edit
```

**Examples:**
```bash
# Convert a photo with 2000 points and open editor
python img2txt.py photo.jpg 2000 --edit

# Quick convert a line drawing
python img2txt.py drawing.png 1000

# Auto-process with default settings
python img2txt.py logo.png
```

**Processing Methods:**
| Method | Description | Best For |
|--------|-------------|----------|
| **Simple** | Basic sharpening | Quick processing |
| **Bilateral** | Noise reduction + edge preservation | Photos (recommended) |
| **DoG** | Difference of Gaussians | Artistic line drawings |

**Thresholding Modes:**
- **Binary** - Simple threshold (adjustable 0-255)
- **Adaptive** - Auto-adjusts for varying lighting (best for photos)

**Interactive Editor Controls:**
| Control | Action |
|---------|--------|
| Left Drag | Erase points |
| Right Drag | Add points |
| Scroll | Adjust radius |
| TAB | Toggle erase/add mode |
| S | Save coordinates |
| R | Reset to original |
| Q | Quit |

**Output:**
Creates a `coordinates.txt` file that can be loaded in the main application.

---

### 4. `check_version.py` - Feature Verification Tool

A utility script to verify that all Sound Pad features are present in the GUI.

```bash
python check_version.py
```

**What it checks:**
- Position Offset Variable
- Position Offset Slider Control
- Time Slots Grid (16 buttons)
- Clear Selected Step Button
- Grid Settings Scrollable Panel

---

## File Formats

### Loading Patterns

**MATLAB Format (.m)**
```matlab
x_fun = [0.0, 0.1, 0.2, ...];
y_fun = [0.5, 0.6, 0.7, ...];
```

**Text Format (.txt)**
```
x_fun=[0.0,0.1,0.2,...];
y_fun=[0.5,0.6,0.7,...];
```

**NumPy Format (.npz)**
```python
import numpy as np
np.savez('pattern.npz', x=x_coords, y=y_coords)
```

### Export Format

**WAV Audio** - Stereo 16-bit PCM
- Left channel: X coordinates
- Right channel: Y coordinates

---

## Pattern Generators

### Sum of Harmonics
Create Lissajous figures by combining sine waves:
- X = A₁·sin(f₁·t + φ₁) + A₂·sin(f₂·t + φ₂) + ...
- Y = B₁·sin(g₁·t + ψ₁) + B₂·sin(g₂·t + ψ₂) + ...

### Archimedean Spiral
Generate spiral patterns:
- r = a + b·θ
- X = r·cos(θ)
- Y = r·sin(θ)

### Sound Pad Sequencer
16-step sequencer with:
- Grid-based pattern editing
- Position offset control
- Step selection and clearing
- Pattern preview

### Random Harmonics
Generate random Lissajous-like patterns with:
- Random frequency ratios
- Random phase offsets
- Random amplitudes

---

## Connecting to an Oscilloscope

1. **Set oscilloscope to XY mode** (also called X-Y, dual channel, or Lissajous mode)
2. **Connect audio output**:
   - Left channel → X input (Channel 1)
   - Right channel → Y input (Channel 2)
3. **Adjust oscilloscope settings**:
   - Set time/div to external or X-Y mode
   - Adjust gain on both channels equally
   - Set coupling to AC or DC depending on your pattern

**Recommended Audio Settings:**
- Sample Rate: 100-200 kHz (1000 Hz × 100-200 multiplier)
- Duration: 1-2 seconds for smooth looping

---

## Additional Documentation

- **[OPTIMIZATION_README.md](OPTIMIZATION_README.md)** - Detailed optimization techniques
- **[QUICK_START_OPTIMIZED.md](QUICK_START_OPTIMIZED.md)** - Quick start for optimized version
- **[DENSITY_RENDERING.md](DENSITY_RENDERING.md)** - How density-based rendering works

---

## Troubleshooting

### No audio output
- Check that your audio device is selected in system settings
- Verify the sample rate is supported by your sound card
- Try lowering the playback multiplier

### Laggy preview
- Use the optimized version (`oscilloscope_gui_optimized.py`)
- Reduce window size in Live Preview settings
- Lower the FPS setting

### Image converter errors
- Ensure OpenCV is installed: `pip install opencv-python`
- Try different processing methods (Simple, Bilateral, DoG)
- Use the `--edit` flag to adjust parameters interactively

### Pattern looks wrong on oscilloscope
- Check X/Y channel connections aren't swapped
- Ensure both channels have the same gain setting
- Verify oscilloscope is in XY mode, not time-base mode

---

## License

This project is open source. Feel free to use, modify, and distribute.

---

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## Credits

**JUANTRONIX** - Inspired by classic Tektronix oscilloscopes from the 90s.

Built with:
- Python
- NumPy
- Matplotlib
- sounddevice
- Tkinter
- OpenCV (for image processing)

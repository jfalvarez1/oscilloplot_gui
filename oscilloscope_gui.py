#!/usr/bin/env python3
"""
Oscilloscope XY Audio Generator with Real-Time GUI
Features live visualization, parameter controls, and various effects
"""

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")  # Dark mode
ctk.set_default_color_theme("blue")  # Color theme


class OscilloscopeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Oscilloscope XY Audio Generator")
        self.root.geometry("1200x800")
        
        # Audio state
        self.is_playing = False
        self.audio_thread = None
        self.stop_flag = threading.Event()
        self.update_queue = queue.Queue()

        # Current data - Default sine wave (X = t, Y = sin(6πt))
        t = np.linspace(-1, 1, 500)  # Time from -1 to 1 for nice display
        self.x_data = t
        self.y_data = np.sin(6 * np.pi * t)  # sin(6πt) - 3 complete cycles
        self.current_audio = None
        self.current_fs = 0
        
        # Live preview state
        self.preview_position = 0
        self.preview_window_size = 5000  # Number of samples to show at once (increased for large fades)
        self.preview_active = True  # Enable by default
        self.preview_buffer = []  # Rolling buffer for streaming display
        self.last_preview_update = 0  # Track last update position

        # Effect change debouncing
        self.effect_change_timer = None
        self.is_regenerating = False  # Flag to prevent simultaneous regenerations


        # Create GUI
        self.create_widgets()
        self.update_display()

        # Initialize wavy labels with default values
        self.update_wavy_labels_only()

        # Start update loops
        self.root.after(50, self.check_updates)
        fps = self.fps_var.get()
        delay_ms = int(1000 / fps)
        self.root.after(delay_ms, self.update_live_preview)  # Preview update based on FPS slider
    
    def create_widgets(self):
        """Create all GUI widgets"""

        # Main container
        main_container = ctk.CTkFrame(self.root)
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)

        # Left panel - Controls (with scrollbar)
        control_container = ctk.CTkFrame(main_container, width=300)
        control_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_container.grid_propagate(False)

        # Create scrollable frame for controls
        control_scrollable = ctk.CTkScrollableFrame(control_container, width=280)
        control_scrollable.pack(fill=tk.BOTH, expand=True)
        control_frame = control_scrollable

        # Right panel - Display with JUANTRONIX branding
        display_container = ctk.CTkFrame(main_container)
        display_container.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_container.columnconfigure(0, weight=1)
        display_container.rowconfigure(1, weight=1)  # Row 1 for display (row 0 for branding)

        # JUANTRONIX branding in oscilloscope style
        branding_frame = ctk.CTkFrame(display_container, fg_color="transparent")
        branding_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        juantronix_label = ctk.CTkLabel(
            branding_frame,
            text="JUANTRONIX",
            font=("Courier New", 28, "bold"),
            text_color="#00FF00"  # Classic oscilloscope green
        )
        juantronix_label.pack()

        model_label = ctk.CTkLabel(
            branding_frame,
            text="XY AUDIO OSCILLOSCOPE",
            font=("Courier New", 10),
            text_color="#808080"  # Gray
        )
        model_label.pack()

        # Display frame
        display_frame = ctk.CTkFrame(display_container)
        display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Create controls
        self.create_control_panel(control_frame)
        
        # Create display
        self.create_display(display_frame)
    
    def create_control_panel(self, parent):
        """Create control panel with parameters and buttons"""
        
        row = 0
        
        # === FILE CONTROLS ===
        file_frame = ctk.CTkFrame(parent)
        ctk.CTkLabel(file_frame, text="Data Source", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        file_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ctk.CTkButton(file_frame, text="Load MATLAB File (.m)", 
                  command=self.load_matlab_file).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Load Text File (.txt)", 
                  command=self.load_txt_file).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Load NumPy File (.npz)",
                  command=self.load_numpy_file).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Generate Test Pattern",
                  command=self.generate_test_pattern).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Draw Pattern",
                  command=self.open_drawing_canvas).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Sum of Harmonics",
                  command=self.open_harmonic_sum).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Archimedean Spiral",
                  command=self.open_archimedean_spiral).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Sound Pad",
                  command=self.open_sound_pad).pack(fill=tk.X, pady=2)
        ctk.CTkButton(file_frame, text="Random Harmonics",
                  command=self.generate_random_harmonics).pack(fill=tk.X, pady=2)

        # Data info
        self.data_info_label = ctk.CTkLabel(file_frame, text="Points: 3", 
                                         font=('Arial', 9, 'italic'))
        self.data_info_label.pack(pady=5)
        
        # === AUDIO PARAMETERS ===
        audio_frame = ctk.CTkFrame(parent)
        audio_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Title
        ctk.CTkLabel(audio_frame, text="Audio Parameters", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=(5,0))

        # Content frame for grid layout
        audio_content = ctk.CTkFrame(audio_frame, fg_color="transparent")
        audio_content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sample Rate
        ctk.CTkLabel(audio_content, text="Base Sample Rate (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.sample_rate_var = tk.IntVar(value=1000)
        self.sample_rate_spin = tk.Spinbox(audio_content, from_=100, to=10000, 
                                           textvariable=self.sample_rate_var, width=10)
        self.sample_rate_spin.grid(row=0, column=1, pady=2)
        
        # Playback Multiplier (Frequency)
        ctk.CTkLabel(audio_content, text="Playback Multiplier:").grid(row=1, column=0, sticky=tk.W)
        self.freq_mult_var = tk.IntVar(value=100)
        self.freq_mult_spin = tk.Spinbox(audio_content, from_=10, to=500, 
                                         textvariable=self.freq_mult_var, width=10)
        self.freq_mult_spin.grid(row=1, column=1, pady=2)
        
        ctk.CTkLabel(audio_content, text="→ Actual Rate:").grid(row=2, column=0, sticky=tk.W)
        self.actual_rate_label = ctk.CTkLabel(audio_content, text="100 kHz", 
                                          font=('Arial', 9, 'bold'))
        self.actual_rate_label.grid(row=2, column=1, pady=2)
        
        # Duration
        ctk.CTkLabel(audio_content, text="Duration (seconds):").grid(row=3, column=0, sticky=tk.W)
        self.duration_var = tk.IntVar(value=15)
        self.duration_spin = tk.Spinbox(audio_content, from_=5, to=120, 
                                        textvariable=self.duration_var, width=10)
        self.duration_spin.grid(row=3, column=1, pady=2)
        
        # N Repeat
        ctk.CTkLabel(audio_content, text="Pattern Repeats:").grid(row=4, column=0, sticky=tk.W)
        self.n_repeat_var = tk.IntVar(value=200)  # Increased default for full rotations
        self.n_repeat_spin = tk.Spinbox(audio_content, from_=1, to=2000, 
                                        textvariable=self.n_repeat_var, width=10)
        self.n_repeat_spin.grid(row=4, column=1, pady=2)
        
        # Rotation info label
        self.rotation_info_label = ctk.CTkLabel(audio_content, text="",
                                            font=('Arial', 8, 'italic'),
                                            text_color='blue')
        self.rotation_info_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(2,5))

        # Update rate label on change
        self.freq_mult_var.trace('w', self.update_rate_label)
        self.sample_rate_var.trace('w', self.update_rate_label)
        
        # === EFFECTS ===
        effects_frame = ctk.CTkFrame(parent)
        ctk.CTkLabel(effects_frame, text="Effects", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        effects_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Enable Reflections
        self.reflections_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(effects_frame, text="Enable Mirror Reflections", 
                       variable=self.reflections_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=5)
        
        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)
        
        # Y-Axis Fade Sequence
        self.y_fade_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(effects_frame, text="Y-Axis Fade Sequence", 
                       variable=self.y_fade_var,
                       command=self.effect_changed).pack(anchor=tk.W)
        
        ctk.CTkLabel(effects_frame, text="Y Fade Steps:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.y_fade_steps = tk.IntVar(value=10)
        self.y_fade_steps.trace('w', lambda *args: self.effect_changed())
        self.y_fade_steps_spin = tk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.y_fade_steps,
                   command=self.effect_changed)
        self.y_fade_steps_spin.pack(anchor=tk.W, padx=20)

        ctk.CTkLabel(effects_frame, text="Y Fade Speed (repeats/step):",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.y_fade_speed = tk.IntVar(value=1)
        self.y_fade_speed.trace('w', lambda *args: self.effect_changed())
        self.y_fade_speed_spin = tk.Spinbox(effects_frame, from_=1, to=20, width=8,
                   textvariable=self.y_fade_speed,
                   command=self.effect_changed)
        self.y_fade_speed_spin.pack(anchor=tk.W, padx=20)
        
        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)
        
        # X-Axis Fade Sequence
        self.x_fade_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(effects_frame, text="X-Axis Fade Sequence", 
                       variable=self.x_fade_var,
                       command=self.effect_changed).pack(anchor=tk.W)
        
        ctk.CTkLabel(effects_frame, text="X Fade Steps:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.x_fade_steps = tk.IntVar(value=10)
        self.x_fade_steps.trace('w', lambda *args: self.effect_changed())
        self.x_fade_steps_spin = tk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.x_fade_steps,
                   command=self.effect_changed)
        self.x_fade_steps_spin.pack(anchor=tk.W, padx=20)

        ctk.CTkLabel(effects_frame, text="X Fade Speed (repeats/step):",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.x_fade_speed = tk.IntVar(value=1)
        self.x_fade_speed.trace('w', lambda *args: self.effect_changed())
        self.x_fade_speed_spin = tk.Spinbox(effects_frame, from_=1, to=20, width=8,
                   textvariable=self.x_fade_speed,
                   command=self.effect_changed)
        self.x_fade_speed_spin.pack(anchor=tk.W, padx=20)

        # Alternate X/Y Fade option
        self.alternate_xy_fade_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(effects_frame, text="Alternate X/Y Fade (X first, then Y, repeat)",
                       variable=self.alternate_xy_fade_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=(5,0))

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Shrink/Unshrink (scale both X and Y together)
        self.shrink_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(effects_frame, text="Shrink/Unshrink (Scale X & Y together)",
                       variable=self.shrink_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        ctk.CTkLabel(effects_frame, text="Shrink Steps:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.shrink_steps = tk.IntVar(value=10)
        self.shrink_steps.trace('w', lambda *args: self.effect_changed())
        self.shrink_steps_spin = tk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.shrink_steps,
                   command=self.effect_changed)
        self.shrink_steps_spin.pack(anchor=tk.W, padx=20)

        ctk.CTkLabel(effects_frame, text="Shrink Speed (repeats/step):",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.shrink_speed = tk.IntVar(value=1)
        self.shrink_speed.trace('w', lambda *args: self.effect_changed())
        self.shrink_speed_spin = tk.Spinbox(effects_frame, from_=1, to=20, width=8,
                   textvariable=self.shrink_speed,
                   command=self.effect_changed)
        self.shrink_speed_spin.pack(anchor=tk.W, padx=20)

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Noise Effects
        noise_frame = ctk.CTkFrame(effects_frame)
        noise_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(noise_frame, text="Noise", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        noise_frame.pack(fill=tk.X, pady=5)

        self.x_noise_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(noise_frame, text="Add X-Channel Noise",
                       variable=self.x_noise_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        ctk.CTkLabel(noise_frame, text="X Noise Amplitude:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.x_noise_amp = tk.DoubleVar(value=0.05)
        ctk.CTkSlider(noise_frame, from_=0.001, to=0.3,
                 variable=self.x_noise_amp,
                 command=lambda v: self.effect_changed()).pack(fill=tk.X, padx=20)

        self.y_noise_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(noise_frame, text="Add Y-Channel Noise",
                       variable=self.y_noise_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=(10,0))

        ctk.CTkLabel(noise_frame, text="Y Noise Amplitude:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.y_noise_amp = tk.DoubleVar(value=0.05)
        ctk.CTkSlider(noise_frame, from_=0.001, to=0.3,
                 variable=self.y_noise_amp,
                 command=lambda v: self.effect_changed()).pack(fill=tk.X, padx=20)

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Wavy Effects
        wavy_frame = ctk.CTkFrame(effects_frame)
        wavy_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(wavy_frame, text="Wavy Effect", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        wavy_frame.pack(fill=tk.X, pady=5)

        self.x_wavy_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(wavy_frame, text="Add X-Channel Wavy",
                       variable=self.x_wavy_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        # X Amplitude with value label
        x_amp_frame = ctk.CTkFrame(wavy_frame)
        x_amp_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(x_amp_frame, text="X Amplitude (K):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.x_wavy_amp_label = ctk.CTkLabel(x_amp_frame, text="0.200",
                 font=('Arial', 8, 'bold'))
        self.x_wavy_amp_label.pack(side=tk.RIGHT)

        self.x_wavy_amp = tk.DoubleVar(value=0.2)
        ctk.CTkSlider(wavy_frame, from_=0.0, to=1.0,
                 variable=self.x_wavy_amp,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        # X Frequency with value label and entry
        x_freq_frame = ctk.CTkFrame(wavy_frame)
        x_freq_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(x_freq_frame, text="X Angular Frequency (ω):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.x_wavy_freq = tk.DoubleVar(value=10.0)
        x_freq_entry = ctk.CTkEntry(x_freq_frame, textvariable=self.x_wavy_freq, width=12)
        x_freq_entry.pack(side=tk.RIGHT, padx=5)
        self.x_wavy_freq_label = ctk.CTkLabel(x_freq_frame, text="10.0",
                 font=('Arial', 8, 'bold'))
        self.x_wavy_freq_label.pack(side=tk.RIGHT)

        ctk.CTkSlider(wavy_frame, from_=1.0, to=1000000.0,
                 variable=self.x_wavy_freq,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        self.y_wavy_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(wavy_frame, text="Add Y-Channel Wavy",
                       variable=self.y_wavy_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=(10,0))

        # Y Amplitude with value label
        y_amp_frame = ctk.CTkFrame(wavy_frame)
        y_amp_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(y_amp_frame, text="Y Amplitude (K):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.y_wavy_amp_label = ctk.CTkLabel(y_amp_frame, text="0.200",
                 font=('Arial', 8, 'bold'))
        self.y_wavy_amp_label.pack(side=tk.RIGHT)

        self.y_wavy_amp = tk.DoubleVar(value=0.2)
        ctk.CTkSlider(wavy_frame, from_=0.0, to=1.0,
                 variable=self.y_wavy_amp,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        # Y Frequency with value label and entry
        y_freq_frame = ctk.CTkFrame(wavy_frame)
        y_freq_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(y_freq_frame, text="Y Angular Frequency (ω):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.y_wavy_freq = tk.DoubleVar(value=10.0)
        y_freq_entry = ctk.CTkEntry(y_freq_frame, textvariable=self.y_wavy_freq, width=12)
        y_freq_entry.pack(side=tk.RIGHT, padx=5)
        self.y_wavy_freq_label = ctk.CTkLabel(y_freq_frame, text="10.0",
                 font=('Arial', 8, 'bold'))
        self.y_wavy_freq_label.pack(side=tk.RIGHT)

        ctk.CTkSlider(wavy_frame, from_=1.0, to=1000000.0,
                 variable=self.y_wavy_freq,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Rotation
        rotation_frame = ctk.CTkFrame(effects_frame)
        rotation_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(rotation_frame, text="Rotation", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        rotation_frame.pack(fill=tk.X, pady=5)
        
        self.rotation_mode_var = tk.StringVar(value="Off")
        ctk.CTkRadioButton(rotation_frame, text="Off", variable=self.rotation_mode_var, 
                       value="Off", command=self.rotation_mode_changed).pack(anchor=tk.W)
        ctk.CTkRadioButton(rotation_frame, text="Static Angle", variable=self.rotation_mode_var, 
                       value="Static", command=self.rotation_mode_changed).pack(anchor=tk.W)
        ctk.CTkRadioButton(rotation_frame, text="Rotate Clockwise (CW)", variable=self.rotation_mode_var, 
                       value="CW", command=self.rotation_mode_changed).pack(anchor=tk.W)
        ctk.CTkRadioButton(rotation_frame, text="Rotate Counter-Clockwise (CCW)", variable=self.rotation_mode_var, 
                       value="CCW", command=self.rotation_mode_changed).pack(anchor=tk.W)
        
        ctk.CTkLabel(rotation_frame, text="Static Angle (degrees):").pack(anchor=tk.W, pady=(5,0))
        self.rotation_angle = tk.DoubleVar(value=0.0)
        ctk.CTkSlider(rotation_frame, from_=-180, to=180,
                 variable=self.rotation_angle,
                 command=lambda v: self.rotation_mode_changed()).pack(fill=tk.X)
        
        ctk.CTkLabel(rotation_frame, text="Rotation Speed (degrees/cycle):").pack(anchor=tk.W, pady=(5,0))
        self.rotation_speed = tk.DoubleVar(value=5.0)
        ctk.CTkSlider(rotation_frame, from_=0.5, to=45,
                 variable=self.rotation_speed,
                 command=lambda v: self.rotation_mode_changed()).pack(fill=tk.X)

        ctk.CTkLabel(rotation_frame, text="Tip: 360° ÷ speed = steps per rotation\nMore Pattern Repeats = more rotations",
                 font=('Arial', 7, 'italic'), text_color='gray').pack(anchor=tk.W, pady=(5,0))

        # Update rotation info when values change (set up after all variables are created)
        self.n_repeat_var.trace('w', self.update_rotation_info)
        self.rotation_speed.trace('w', self.update_rotation_info)

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Tremolo Effect (Amplitude Modulation)
        tremolo_frame = ctk.CTkFrame(effects_frame)
        tremolo_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(tremolo_frame, text="Tremolo (Amplitude Modulation)", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        tremolo_frame.pack(fill=tk.X, pady=5)

        self.tremolo_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(tremolo_frame, text="Enable Tremolo",
                       variable=self.tremolo_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        # Depth control
        depth_frame = ctk.CTkFrame(tremolo_frame)
        depth_frame.pack(fill=tk.X, padx=20, pady=(5,0))
        ctk.CTkLabel(depth_frame, text="Depth (%):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.tremolo_depth_label = ctk.CTkLabel(depth_frame, text="50",
                 font=('Arial', 8, 'bold'))
        self.tremolo_depth_label.pack(side=tk.RIGHT)

        self.tremolo_depth = tk.DoubleVar(value=50.0)
        ctk.CTkSlider(tremolo_frame, from_=0.0, to=100.0,
                 variable=self.tremolo_depth,
                 command=lambda v: self.tremolo_depth_label.config(text=f"{self.tremolo_depth.get():.0f}")).pack(fill=tk.X, padx=20)

        # Rate control
        rate_frame = ctk.CTkFrame(tremolo_frame)
        rate_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(rate_frame, text="Rate (Hz):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.tremolo_rate_label = ctk.CTkLabel(rate_frame, text="2.0",
                 font=('Arial', 8, 'bold'))
        self.tremolo_rate_label.pack(side=tk.RIGHT)

        self.tremolo_rate = tk.DoubleVar(value=2.0)
        ctk.CTkSlider(tremolo_frame, from_=0.1, to=20.0,
                 variable=self.tremolo_rate,
                 command=lambda v: self.tremolo_rate_label.config(text=f"{self.tremolo_rate.get():.1f}")).pack(fill=tk.X, padx=20)

        # Waveform type
        waveform_row = ctk.CTkFrame(tremolo_frame)
        waveform_row.pack(fill=tk.X, padx=20, pady=5)
        ctk.CTkLabel(waveform_row, text="Waveform:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 10))
        self.tremolo_wave_var = tk.StringVar(value="sine")
        ctk.CTkRadioButton(waveform_row, text="Sine", variable=self.tremolo_wave_var, value="sine").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(waveform_row, text="Triangle", variable=self.tremolo_wave_var, value="triangle").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(waveform_row, text="Square", variable=self.tremolo_wave_var, value="square").pack(side=tk.LEFT, padx=5)

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Ring Modulation Effect
        ring_frame = ctk.CTkFrame(effects_frame)
        ring_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(ring_frame, text="Ring Modulation", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        ring_frame.pack(fill=tk.X, pady=5)

        self.ring_mod_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(ring_frame, text="Enable Ring Modulation",
                       variable=self.ring_mod_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        # Carrier Frequency control
        carrier_frame = ctk.CTkFrame(ring_frame)
        carrier_frame.pack(fill=tk.X, padx=20, pady=(5,0))
        ctk.CTkLabel(carrier_frame, text="Carrier Freq (Hz):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.ring_carrier_label = ctk.CTkLabel(carrier_frame, text="200",
                 font=('Arial', 8, 'bold'))
        self.ring_carrier_label.pack(side=tk.RIGHT)

        self.ring_carrier_freq = tk.DoubleVar(value=200.0)
        ctk.CTkSlider(ring_frame, from_=10.0, to=2000.0,
                 variable=self.ring_carrier_freq,
                 command=lambda v: self.ring_carrier_label.config(text=f"{self.ring_carrier_freq.get():.0f}")).pack(fill=tk.X, padx=20)

        # Depth control
        ring_depth_frame = ctk.CTkFrame(ring_frame)
        ring_depth_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(ring_depth_frame, text="Mix (%):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.ring_mix_label = ctk.CTkLabel(ring_depth_frame, text="50",
                 font=('Arial', 8, 'bold'))
        self.ring_mix_label.pack(side=tk.RIGHT)

        self.ring_mix = tk.DoubleVar(value=50.0)
        ctk.CTkSlider(ring_frame, from_=0.0, to=100.0,
                 variable=self.ring_mix,
                 command=lambda v: self.ring_mix_label.config(text=f"{self.ring_mix.get():.0f}")).pack(fill=tk.X, padx=20)

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Echo/Delay Effect
        echo_frame = ctk.CTkFrame(effects_frame)
        echo_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(echo_frame, text="Echo/Delay", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        echo_frame.pack(fill=tk.X, pady=5)

        self.echo_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(echo_frame, text="Enable Echo",
                       variable=self.echo_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        # Number of echoes
        echoes_frame = ctk.CTkFrame(echo_frame)
        echoes_frame.pack(fill=tk.X, padx=20, pady=(5,0))
        ctk.CTkLabel(echoes_frame, text="Number of Echoes:",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.echo_count_label = ctk.CTkLabel(echoes_frame, text="3",
                 font=('Arial', 8, 'bold'))
        self.echo_count_label.pack(side=tk.RIGHT)

        self.echo_count = tk.IntVar(value=3)
        ctk.CTkSlider(echo_frame, from_=1, to=10,
                 variable=self.echo_count,
                 command=lambda v: self.echo_count_label.config(text=f"{self.echo_count.get():.0f}")).pack(fill=tk.X, padx=20)

        # Decay control
        decay_frame = ctk.CTkFrame(echo_frame)
        decay_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(decay_frame, text="Decay Factor:",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.echo_decay_label = ctk.CTkLabel(decay_frame, text="0.70",
                 font=('Arial', 8, 'bold'))
        self.echo_decay_label.pack(side=tk.RIGHT)

        self.echo_decay = tk.DoubleVar(value=0.7)
        ctk.CTkSlider(echo_frame, from_=0.1, to=0.95,
                 variable=self.echo_decay,
                 command=lambda v: self.echo_decay_label.config(text=f"{self.echo_decay.get():.2f}")).pack(fill=tk.X, padx=20)

        # Delay time (as percentage of pattern length)
        delay_frame = ctk.CTkFrame(echo_frame)
        delay_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(delay_frame, text="Delay Time (%):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.echo_delay_label = ctk.CTkLabel(delay_frame, text="10",
                 font=('Arial', 8, 'bold'))
        self.echo_delay_label.pack(side=tk.RIGHT)

        self.echo_delay = tk.DoubleVar(value=10.0)
        ctk.CTkSlider(echo_frame, from_=1.0, to=50.0,
                 variable=self.echo_delay,
                 command=lambda v: self.echo_delay_label.config(text=f"{self.echo_delay.get():.0f}")).pack(fill=tk.X, padx=20)

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Kaleidoscope Effect
        kaleido_frame = ctk.CTkFrame(effects_frame)
        kaleido_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(kaleido_frame, text="Kaleidoscope", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        kaleido_frame.pack(fill=tk.X, pady=5)

        self.kaleido_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(kaleido_frame, text="Enable Kaleidoscope",
                       variable=self.kaleido_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        # Number of symmetry sections
        sections_frame = ctk.CTkFrame(kaleido_frame)
        sections_frame.pack(fill=tk.X, padx=20, pady=(5,0))
        ctk.CTkLabel(sections_frame, text="Symmetry Sections:",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.kaleido_sections_label = ctk.CTkLabel(sections_frame, text="6",
                 font=('Arial', 8, 'bold'))
        self.kaleido_sections_label.pack(side=tk.RIGHT)

        self.kaleido_sections = tk.IntVar(value=6)
        ctk.CTkSlider(kaleido_frame, from_=2, to=12,
                 variable=self.kaleido_sections,
                 command=lambda v: self.kaleido_sections_label.config(text=f"{self.kaleido_sections.get():.0f}")).pack(fill=tk.X, padx=20)

        # Mirror option
        self.kaleido_mirror_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(kaleido_frame, text="Mirror Reflections",
                       variable=self.kaleido_mirror_var,
                       command=self.effect_changed).pack(anchor=tk.W, padx=20)

        ctk.CTkLabel(kaleido_frame, text="Creates symmetrical copies around center",
                 font=('Arial', 7, 'italic'), text_color='gray').pack(anchor=tk.W, padx=20, pady=(5,0))

        ctk.CTkFrame(effects_frame, height=2).pack(fill=tk.X, pady=5)

        # Distortion/Clipping Effect
        distortion_frame = ctk.CTkFrame(effects_frame)
        distortion_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(distortion_frame, text="Distortion", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        distortion_frame.pack(fill=tk.X, pady=5)

        self.distortion_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(distortion_frame, text="Enable Distortion",
                       variable=self.distortion_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        # Distortion type
        dist_type_row = ctk.CTkFrame(distortion_frame)
        dist_type_row.pack(fill=tk.X, padx=20, pady=5)
        ctk.CTkLabel(dist_type_row, text="Type:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 10))
        self.distortion_type_var = tk.StringVar(value="soft")
        ctk.CTkRadioButton(dist_type_row, text="Soft Clip", variable=self.distortion_type_var, value="soft").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(dist_type_row, text="Hard Clip", variable=self.distortion_type_var, value="hard").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(dist_type_row, text="Fold", variable=self.distortion_type_var, value="fold").pack(side=tk.LEFT, padx=5)

        # Threshold control
        threshold_frame = ctk.CTkFrame(distortion_frame)
        threshold_frame.pack(fill=tk.X, padx=20)
        ctk.CTkLabel(threshold_frame, text="Threshold:",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.distortion_threshold_label = ctk.CTkLabel(threshold_frame, text="0.50",
                 font=('Arial', 8, 'bold'))
        self.distortion_threshold_label.pack(side=tk.RIGHT)

        self.distortion_threshold = tk.DoubleVar(value=0.5)
        ctk.CTkSlider(distortion_frame, from_=0.1, to=2.0,
                 variable=self.distortion_threshold,
                 command=lambda v: self.distortion_threshold_label.config(text=f"{self.distortion_threshold.get():.2f}")).pack(fill=tk.X, padx=20)

        # === ACTION BUTTONS ===
        button_frame = ctk.CTkFrame(parent)
        button_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.apply_btn = ctk.CTkButton(button_frame, text="Apply & Generate",
                                    command=self.apply_parameters)
        self.apply_btn.pack(fill=tk.X, pady=2)

        self.play_btn = ctk.CTkButton(button_frame, text="▶ Play Audio",
                                   command=self.toggle_playback)
        self.play_btn.pack(fill=tk.X, pady=2)

        ctk.CTkButton(button_frame, text="Reset Effects",
                  command=self.reset_effects).pack(fill=tk.X, pady=2)

        ctk.CTkButton(button_frame, text="Save to WAV",
                  command=self.save_to_wav).pack(fill=tk.X, pady=2)
        
        # === STATUS ===
        status_frame = ctk.CTkFrame(parent)
        ctk.CTkLabel(status_frame, text="Status", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        status_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        self.status_label = ctk.CTkLabel(status_frame, text="Ready", 
                                      wraplength=200, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X)
        
        # Live Preview Toggle
        ctk.CTkFrame(status_frame, height=2).pack(fill=tk.X, pady=5)

        ctk.CTkLabel(status_frame, text="Display Mode:",
                 font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(5,2))

        self.live_preview_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(status_frame, text="Live Preview",
                       variable=self.live_preview_var,
                       command=self.toggle_live_preview).pack(anchor=tk.W)

        ctk.CTkLabel(status_frame, text="Shows real-time output during playback",
                 font=('Arial', 7, 'italic'),
                 text_color='gray').pack(anchor=tk.W, padx=(20,0))

        # Preview window size control
        preview_control_frame = ctk.CTkFrame(status_frame)
        preview_control_frame.pack(fill=tk.X, pady=(5,0))
        ctk.CTkLabel(preview_control_frame, text="Window Size:",
                 font=('Arial', 8)).pack(side=tk.LEFT, padx=(15,5))
        self.preview_size_var = tk.IntVar(value=5000)
        self.preview_size_spin = tk.Spinbox(preview_control_frame, from_=100, to=50000,
                                       width=8, textvariable=self.preview_size_var,
                                       command=self.update_preview_size)
        self.preview_size_spin.pack(side=tk.LEFT)
        ctk.CTkLabel(preview_control_frame, text="samples",
                 font=('Arial', 8)).pack(side=tk.LEFT, padx=(5,0))

        # Frame refresh rate control
        fps_control_frame = ctk.CTkFrame(status_frame)
        fps_control_frame.pack(fill=tk.X, pady=(5,0))
        ctk.CTkLabel(fps_control_frame, text="Frame Rate:",
                 font=('Arial', 8)).pack(side=tk.LEFT, padx=(15,5))
        self.fps_var = tk.IntVar(value=25)
        self.fps_slider = ctk.CTkSlider(fps_control_frame, from_=5, to=60,
                                    variable=self.fps_var,
                                    command=self.update_fps)
        self.fps_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.fps_label = ctk.CTkLabel(fps_control_frame, text="25 FPS",
                                   font=('Arial', 8), width=7)
        self.fps_label.pack(side=tk.LEFT, padx=(0,5))

        # Bind Enter key to all spinbox fields to trigger Apply & Generate
        self.bind_enter_to_apply()
    
    def bind_enter_to_apply(self):
        """Bind Enter key to all input fields to trigger Apply & Generate"""
        # List of all spinbox widgets
        spinboxes = [
            self.sample_rate_spin,
            self.freq_mult_spin,
            self.duration_spin,
            self.n_repeat_spin,
            self.y_fade_steps_spin,
            self.y_fade_speed_spin,
            self.x_fade_steps_spin,
            self.x_fade_speed_spin,
            self.shrink_steps_spin,
            self.shrink_speed_spin,
            self.preview_size_spin,
        ]

        # Bind Return/Enter key to trigger apply_parameters
        for spinbox in spinboxes:
            spinbox.bind('<Return>', lambda e: self.apply_parameters())
            spinbox.bind('<KP_Enter>', lambda e: self.apply_parameters())  # Keypad Enter

    def create_display(self, parent):
        """Create matplotlib display"""

        # Configure matplotlib to handle large paths
        import matplotlib as mpl
        mpl.rcParams['agg.path.chunksize'] = 10000

        # Create figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')

        self.ax = self.fig.add_subplot(111, facecolor='#000000')
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, color='#00ff00', alpha=0.3, linestyle='--')
        self.ax.set_xlabel('X (Left Channel)', color='#00ff00')
        self.ax.set_ylabel('Y (Right Channel)', color='#00ff00')
        self.ax.tick_params(colors='#00ff00')

        # Initial plot - use both lines and points for realistic oscilloscope effect
        # Very light connecting lines (almost invisible)
        self.line, = self.ax.plot([], [], color='#00ff00', linewidth=0.5, alpha=0.15)
        # Bright points for the actual beam positions
        self.points = self.ax.scatter([], [], color='#00ff00', s=1.5, alpha=0.9)

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_rate_label(self, *args):
        """Update the actual playback rate label"""
        fs = self.sample_rate_var.get()
        mult = self.freq_mult_var.get()
        actual = fs * mult
        
        if actual >= 1000:
            self.actual_rate_label.config(text=f"{actual/1000:.0f} kHz")
        else:
            self.actual_rate_label.config(text=f"{actual} Hz")
    
    def normalize_data(self, data):
        """Normalize data to [-1, 1]"""
        data = np.asarray(data, dtype=np.float32)
        data_min = data.min()
        data_max = data.max()
        
        if data_max == data_min:
            return np.zeros_like(data)
        
        return 2.0 * (data - data_min) / (data_max - data_min) - 1.0
    
    def effect_changed(self):
        """Handle effect changes - debounced to prevent freeze from rapid changes"""
        # Always update display immediately for visual feedback
        self.update_display()

        # Cancel any pending regeneration timer
        if self.effect_change_timer is not None:
            self.root.after_cancel(self.effect_change_timer)
            self.effect_change_timer = None

        # Only schedule regeneration if audio exists
        if hasattr(self, 'current_audio') and self.current_audio is not None:
            # Schedule regeneration after 600ms delay
            # This allows multiple rapid changes to be batched together
            self.effect_change_timer = self.root.after(600, self.delayed_regenerate)

    def delayed_regenerate(self):
        """Execute the actual regeneration after debounce delay"""
        self.effect_change_timer = None
        if not self.is_regenerating:
            self.apply_parameters()
    
    def update_rotation_info(self, *args):
        """Update rotation info label showing number of full rotations"""
        try:
            if hasattr(self, 'rotation_mode_var'):
                mode = self.rotation_mode_var.get()
                if mode in ["CW", "CCW"]:
                    n_repeat = self.n_repeat_var.get()
                    speed = self.rotation_speed.get()
                    num_rotations = (n_repeat * speed) / 360
                    
                    direction = "↻ CW" if mode == "CW" else "↺ CCW"
                    self.rotation_info_label.config(
                        text=f"{direction}: {num_rotations:.1f} full rotations"
                    )
                else:
                    self.rotation_info_label.config(text="")
        except:
            pass
    
    def rotation_mode_changed(self):
        """Handle rotation mode changes - debounced to prevent freeze"""
        self.update_rotation_info()
        self.update_display()

        # Cancel any pending regeneration timer
        if self.effect_change_timer is not None:
            self.root.after_cancel(self.effect_change_timer)
            self.effect_change_timer = None

        # Only schedule regeneration if audio exists
        if hasattr(self, 'current_audio') and self.current_audio is not None:
            # Schedule regeneration after 600ms delay
            self.effect_change_timer = self.root.after(600, self.delayed_regenerate)

    def update_wavy_labels_only(self):
        """Update wavy effect value labels without triggering effect_changed"""
        # Update amplitude labels (3 decimal places)
        x_amp = self.x_wavy_amp.get()
        y_amp = self.y_wavy_amp.get()
        self.x_wavy_amp_label.config(text=f"{x_amp:.3f}")
        self.y_wavy_amp_label.config(text=f"{y_amp:.3f}")

        # Update frequency labels (formatted for readability)
        x_freq = self.x_wavy_freq.get()
        y_freq = self.y_wavy_freq.get()

        # Format frequency based on magnitude
        def format_freq(freq):
            if freq >= 1000:
                return f"{freq:.0f}"
            elif freq >= 100:
                return f"{freq:.1f}"
            else:
                return f"{freq:.2f}"

        self.x_wavy_freq_label.config(text=format_freq(x_freq))
        self.y_wavy_freq_label.config(text=format_freq(y_freq))

    def update_wavy_labels(self):
        """Update wavy effect value labels and trigger effect preview"""
        self.update_wavy_labels_only()
        # Trigger effect changed for preview update
        self.effect_changed()

    def reset_effects(self):
        """Reset all effects to their default values"""
        # Turn off all effects
        self.reflections_var.set(False)
        self.y_fade_var.set(False)
        self.x_fade_var.set(False)
        self.alternate_xy_fade_var.set(False)
        self.shrink_var.set(False)
        self.x_noise_var.set(False)
        self.y_noise_var.set(False)
        self.x_wavy_var.set(False)
        self.y_wavy_var.set(False)
        self.rotation_mode_var.set("Off")
        self.tremolo_var.set(False)
        self.ring_mod_var.set(False)
        self.echo_var.set(False)
        self.kaleido_var.set(False)
        self.distortion_var.set(False)

        # Reset values to defaults
        self.y_fade_steps.set(10)
        self.y_fade_speed.set(1)
        self.x_fade_steps.set(10)
        self.x_fade_speed.set(1)
        self.shrink_steps.set(10)
        self.shrink_speed.set(1)
        self.x_noise_amp.set(0.05)
        self.y_noise_amp.set(0.05)
        self.x_wavy_amp.set(0.2)
        self.y_wavy_amp.set(0.2)
        self.x_wavy_freq.set(10.0)
        self.y_wavy_freq.set(10.0)
        self.rotation_angle.set(0.0)
        self.rotation_speed.set(5.0)
        self.tremolo_depth.set(50.0)
        self.tremolo_rate.set(2.0)
        self.ring_carrier_freq.set(200.0)
        self.ring_mix.set(50.0)
        self.echo_count.set(3)
        self.echo_decay.set(0.7)
        self.echo_delay.set(10.0)
        self.kaleido_sections.set(6)
        self.kaleido_mirror_var.set(True)
        self.distortion_threshold.set(0.5)
        self.distortion_type_var.set("soft")

        # Update wavy labels to reflect reset values
        self.update_wavy_labels_only()

        # Update display
        self.update_display()

        # If audio exists, regenerate with reset effects
        if hasattr(self, 'current_audio') and self.current_audio is not None:
            self.apply_parameters()
    
    def apply_effects(self, x, y):
        """Apply selected effects to the data - FOR DISPLAY PREVIEW ONLY - ALL EFFECTS BLEND"""
        x_norm = x.copy()
        y_norm = y.copy()

        # Determine which effects are enabled (with defensive checks for initialization)
        has_y_fade = self.y_fade_var.get() if hasattr(self, 'y_fade_var') else False
        has_x_fade = self.x_fade_var.get() if hasattr(self, 'x_fade_var') else False
        has_shrink = self.shrink_var.get() if hasattr(self, 'shrink_var') else False
        rotation_mode = self.rotation_mode_var.get() if hasattr(self, 'rotation_mode_var') else "Off"

        # Build effect factor arrays
        y_fade_factors = None
        x_fade_factors = None
        shrink_factors = None

        # Build Y-Fade factors
        if has_y_fade:
            n_fade_y = self.y_fade_steps.get()
            y_fade_speed = self.y_fade_speed.get()
            fade_down_y = np.linspace(1, 0, n_fade_y, dtype=np.float32)
            fade_negative_down_y = np.linspace(0, -1, n_fade_y, dtype=np.float32)[1:]
            fade_negative_up_y = np.linspace(-1, 0, n_fade_y, dtype=np.float32)[1:]
            fade_up_y = np.linspace(0, 1, n_fade_y, dtype=np.float32)[1:]
            one_cycle_y = np.concatenate([fade_down_y, fade_negative_down_y, fade_negative_up_y, fade_up_y])
            y_fade_factors = np.repeat(one_cycle_y, y_fade_speed)

        # Build X-Fade factors
        if has_x_fade:
            n_fade_x = self.x_fade_steps.get()
            x_fade_speed = self.x_fade_speed.get()
            fade_down_x = np.linspace(1, 0, n_fade_x, dtype=np.float32)
            fade_negative_down_x = np.linspace(0, -1, n_fade_x, dtype=np.float32)[1:]
            fade_negative_up_x = np.linspace(-1, 0, n_fade_x, dtype=np.float32)[1:]
            fade_up_x = np.linspace(0, 1, n_fade_x, dtype=np.float32)[1:]
            one_cycle_x = np.concatenate([fade_down_x, fade_negative_down_x, fade_negative_up_x, fade_up_x])
            x_fade_factors = np.repeat(one_cycle_x, x_fade_speed)

        # Build Shrink factors
        if has_shrink:
            n_shrink = self.shrink_steps.get()
            shrink_speed = self.shrink_speed.get()
            shrink_down = np.linspace(1, 0, n_shrink, dtype=np.float32)
            shrink_up = np.linspace(0, 1, n_shrink, dtype=np.float32)[1:]
            one_cycle_shrink = np.concatenate([shrink_down, shrink_up])
            shrink_factors = np.repeat(one_cycle_shrink, shrink_speed)

        # Determine if any effects are enabled
        any_effect_enabled = has_y_fade or has_x_fade or has_shrink

        if not any_effect_enabled:
            # No effects - return as is
            pass
        else:
            # Calculate number of preview cycles based on complexity
            max_length = 0
            if y_fade_factors is not None:
                max_length = max(max_length, len(y_fade_factors))
            if x_fade_factors is not None:
                max_length = max(max_length, len(x_fade_factors))
            if shrink_factors is not None:
                max_length = max(max_length, len(shrink_factors))

            # Adapt preview cycles based on total complexity
            if max_length > 500:
                num_preview_cycles = 1
            elif max_length > 200:
                num_preview_cycles = 2
            else:
                num_preview_cycles = 3

            # Check for alternate X/Y fade mode
            use_alternate = (self.alternate_xy_fade_var.get() and
                           has_x_fade and has_y_fade and
                           not has_shrink)

            x_frames = []
            y_frames = []

            if use_alternate:
                # ALTERNATE MODE: X fade first, then Y fade, repeat
                for cycle_idx in range(num_preview_cycles):
                    # First: Do X fade sequence
                    for x_fade_idx in range(len(x_fade_factors)):
                        x_factor = x_fade_factors[x_fade_idx]
                        x_frames.append(x_norm * x_factor)
                        y_frames.append(y_norm)

                    # Then: Do Y fade sequence
                    for y_fade_idx in range(len(y_fade_factors)):
                        y_factor = y_fade_factors[y_fade_idx]
                        x_frames.append(x_norm)
                        y_frames.append(y_norm * y_factor)

                x = np.concatenate(x_frames)
                y = np.concatenate(y_frames)

            else:
                # BLENDED MODE: All effects apply simultaneously
                # Determine total frames for preview
                total_frames = max_length * num_preview_cycles

                for frame_idx in range(total_frames):
                    # Get current effect factors (cycling through each effect's sequence)
                    y_factor = 1.0
                    x_factor = 1.0
                    scale_factor = 1.0

                    if y_fade_factors is not None:
                        y_factor = y_fade_factors[frame_idx % len(y_fade_factors)]

                    if x_fade_factors is not None:
                        x_factor = x_fade_factors[frame_idx % len(x_fade_factors)]

                    if shrink_factors is not None:
                        scale_factor = shrink_factors[frame_idx % len(shrink_factors)]

                    # Apply ALL effects to this frame
                    x_current = x_norm * x_factor * scale_factor
                    y_current = y_norm * y_factor * scale_factor

                    x_frames.append(x_current)
                    y_frames.append(y_current)

                x = np.concatenate(x_frames)
                y = np.concatenate(y_frames)

        # Mirror Reflections
        if hasattr(self, 'reflections_var') and self.reflections_var.get():
            x, y = self.apply_reflections(x, y)

        # Rotation - static angle only for display
        if rotation_mode == "Static":
            angle_rad = np.radians(self.rotation_angle.get())
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a

            # Renormalize after rotation to prevent clipping
            x = self.normalize_data(x_rot)
            y = self.normalize_data(y_rot)

        # Apply wavy effect if enabled
        if (hasattr(self, 'x_wavy_var') and self.x_wavy_var.get()) or (hasattr(self, 'y_wavy_var') and self.y_wavy_var.get()):
            # Create time array based on position (0 to 2π)
            t = np.linspace(0, 2*np.pi, len(x))

            if hasattr(self, 'x_wavy_var') and self.x_wavy_var.get():
                K_x = self.x_wavy_amp.get()
                w_x = self.x_wavy_freq.get()
                x = x + K_x * np.sin(w_x * t)

            if hasattr(self, 'y_wavy_var') and self.y_wavy_var.get():
                K_y = self.y_wavy_amp.get()
                w_y = self.y_wavy_freq.get()
                y = y + K_y * np.sin(w_y * t)

        # Ring Modulation Effect
        if hasattr(self, 'ring_mod_var') and self.ring_mod_var.get():
            t = np.linspace(0, 2*np.pi, len(x))
            carrier_freq = self.ring_carrier_freq.get()
            mix = self.ring_mix.get() / 100.0

            # Generate carrier wave
            carrier = np.sin(carrier_freq * t)

            # Ring modulation: multiply signal by carrier
            x_mod = x * carrier
            y_mod = y * carrier

            # Mix dry and modulated signals
            x = (1 - mix) * x + mix * x_mod
            y = (1 - mix) * y + mix * y_mod

        # Distortion Effect
        if hasattr(self, 'distortion_var') and self.distortion_var.get():
            threshold = self.distortion_threshold.get()
            dist_type = self.distortion_type_var.get()

            if dist_type == "soft":
                # Soft clipping (tanh-like curve)
                x = np.tanh(x / threshold) * threshold
                y = np.tanh(y / threshold) * threshold
            elif dist_type == "hard":
                # Hard clipping
                x = np.clip(x, -threshold, threshold)
                y = np.clip(y, -threshold, threshold)
            else:  # fold
                # Wave folding
                x = np.where(np.abs(x) > threshold,
                           threshold - (np.abs(x) - threshold), x)
                y = np.where(np.abs(y) > threshold,
                           threshold - (np.abs(y) - threshold), y)

        # Echo/Delay Effect
        if hasattr(self, 'echo_var') and self.echo_var.get():
            num_echoes = self.echo_count.get()
            decay = self.echo_decay.get()
            delay_pct = self.echo_delay.get() / 100.0

            delay_samples = int(len(x) * delay_pct)

            # Create arrays to hold echo sum
            x_with_echo = np.copy(x)
            y_with_echo = np.copy(y)

            # Add delayed copies with decay
            for i in range(1, num_echoes + 1):
                offset = i * delay_samples
                amplitude = decay ** i

                # Pad with zeros and add delayed signal
                if offset < len(x):
                    x_delayed = np.concatenate([np.zeros(offset), x[:-offset]]) * amplitude
                    y_delayed = np.concatenate([np.zeros(offset), y[:-offset]]) * amplitude
                    x_with_echo += x_delayed
                    y_with_echo += y_delayed

            x = x_with_echo
            y = y_with_echo

        # Kaleidoscope Effect
        if hasattr(self, 'kaleido_var') and self.kaleido_var.get():
            sections = self.kaleido_sections.get()
            mirror = self.kaleido_mirror_var.get()

            # Create rotated copies of the pattern
            all_x = []
            all_y = []

            for i in range(sections):
                angle = (2 * np.pi * i) / sections

                # Rotate the pattern
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a

                all_x.append(x_rot)
                all_y.append(y_rot)

                # Add mirrored copy if enabled
                if mirror:
                    x_mir = x * cos_a + y * sin_a
                    y_mir = -x * sin_a + y * cos_a
                    all_x.append(x_mir)
                    all_y.append(y_mir)

            # Concatenate all sections
            x = np.concatenate(all_x)
            y = np.concatenate(all_y)

        return x, y
    
    def toggle_live_preview(self):
        """Toggle live preview mode"""
        self.preview_active = self.live_preview_var.get()
        if self.preview_active:
            self.preview_position = 0
            self.preview_buffer = []
            self.last_preview_update = 0
        else:
            # Return to static preview when disabled
            self.update_display()

    def update_preview_size(self):
        """Update preview window size"""
        self.preview_window_size = self.preview_size_var.get()

    def update_fps(self, *args):
        """Update frame refresh rate"""
        fps = int(self.fps_var.get())
        self.fps_label.config(text=f"{fps} FPS")

    def update_live_preview(self):
        """Update display to show current audio output position - MIMICS TAS 465 ANALOG SCOPE"""
        try:
            if self.preview_active and self.is_playing and self.current_audio is not None:
                try:
                    import time
                    if hasattr(self, 'playback_start_time'):
                        # Hide scatter plot during live preview (prevents static persistence)
                        # Only show the animated line trace, matching real oscilloscope behavior
                        if self.points.get_offsets().shape[0] > 0:
                            self.points.set_offsets(np.empty((0, 2)))

                        # Make line bright during live preview (since scatter is hidden)
                        self.line.set_alpha(0.9)
                        self.line.set_linewidth(1.0)

                        # Calculate current playback position
                        elapsed = time.time() - self.playback_start_time
                        sample_position = int(elapsed * self.current_fs)

                        # Wrap around if we exceed audio length (looping)
                        total_samples = len(self.current_audio)
                        if sample_position >= total_samples:
                            sample_position = sample_position % total_samples
                            # Update start time for smoother looping
                            self.playback_start_time = time.time() - (sample_position / self.current_fs)

                        # TAS 465 BEHAVIOR: Show only current instantaneous pattern, not historical trail
                        # Calculate base pattern length (original data before effects/repetitions)
                        base_pattern_len = len(self.x_data)

                        # Determine how many times pattern is repeated in audio
                        # (from generate_audio: pattern is repeated n_repeat times with effects)
                        pattern_repeat_count = max(1, total_samples // base_pattern_len)
                        samples_per_repetition = total_samples // pattern_repeat_count

                        # Find which repetition we're currently in
                        current_repetition = sample_position // samples_per_repetition
                        repetition_start = current_repetition * samples_per_repetition

                        # Extract ONLY the current pattern state (one complete trace)
                        # Show 3-5 repetitions of this for visibility, like scope persistence
                        display_reps = min(5, max(3, 1000 // base_pattern_len))
                        extract_len = min(samples_per_repetition * display_reps,
                                         total_samples - repetition_start)

                        if extract_len > 10:
                            current_segment = self.current_audio[repetition_start:repetition_start + extract_len]
                            x_preview = current_segment[:, 0]
                            y_preview = current_segment[:, 1]

                            # Update plot - only the line, no scatter (prevents persistence)
                            self.line.set_data(x_preview, y_preview)

                            # Keep consistent axis limits for smooth viewing
                            self.ax.set_xlim(-1.2, 1.2)
                            self.ax.set_ylim(-1.2, 1.2)

                            self.canvas.draw_idle()
                except Exception as e:
                    pass  # Silently ignore preview errors

            # Schedule next update based on FPS slider
            if self.root.winfo_exists():
                fps = self.fps_var.get()
                delay_ms = int(1000 / fps)
                self.root.after(delay_ms, self.update_live_preview)
        except Exception:
            pass  # Window destroyed, stop scheduling
    
    def calculate_density_colors(self, x_data, y_data, bins=200):
        """Calculate colors based on point density - brighter where beam repeats more"""
        # Create 2D histogram to count how many times each region is visited
        hist, x_edges, y_edges = np.histogram2d(
            x_data, y_data,
            bins=bins,
            range=[[-1.5, 1.5], [-1.5, 1.5]]
        )

        # Find which bin each point belongs to
        x_indices = np.digitize(x_data, x_edges) - 1
        y_indices = np.digitize(y_data, y_edges) - 1

        # Clamp indices to valid range
        x_indices = np.clip(x_indices, 0, bins - 1)
        y_indices = np.clip(y_indices, 0, bins - 1)

        # Get density value for each point
        densities = hist[x_indices, y_indices]

        # Normalize densities to [0, 1] with gamma correction for better visibility
        if densities.max() > 0:
            normalized = densities / densities.max()
            # Apply gamma to reduce contrast (higher gamma = less dramatic differences)
            gamma = 0.8  # Reduced from 0.5 for subtler effect
            normalized = np.power(normalized, gamma)
        else:
            normalized = np.ones_like(densities)

        # Convert to colors (RGBA) - green with subtle varying alpha and brightness
        colors = np.zeros((len(normalized), 4))
        colors[:, 1] = 0.5 + 0.5 * normalized  # Green channel - subtle variation (0.5 to 1.0)
        colors[:, 3] = 0.7 + 0.3 * normalized  # Alpha - subtle variation (0.7 to 1.0)

        return colors

    def update_display(self):
        """Update the oscilloscope display with density-based brightness"""
        # Normalize data
        x_norm = self.normalize_data(self.x_data)
        y_norm = self.normalize_data(self.y_data)

        # Apply effects
        x_display, y_display = self.apply_effects(x_norm, y_norm)

        # Check if CW/CCW rotation is enabled - if so, hide scatter plot to prevent overlay
        rotation_mode = self.rotation_mode_var.get() if hasattr(self, 'rotation_mode_var') else "Off"
        if rotation_mode in ["CW", "CCW"]:
            # Hide scatter plot when rotation animation is active
            # (live preview will show the rotating pattern)
            self.line.set_data([], [])
            if self.points.get_offsets().shape[0] > 0:
                self.points.set_offsets(np.empty((0, 2)))
            self.canvas.draw_idle()
            return

        # Calculate density-based colors BEFORE tiling (prevents artificial persistence)
        # This ensures density reflects natural beam speed, not repetition artifacts
        colors_single = self.calculate_density_colors(x_display, y_display)

        # Repeat pattern for visibility
        display_repeats = min(20, max(1, 100 // len(x_norm)))
        x_display = np.tile(x_display, display_repeats)
        y_display = np.tile(y_display, display_repeats)
        # Tile the colors too so each repetition has same density-based brightness
        colors = np.tile(colors_single, (display_repeats, 1))

        # Downsample if too many points for rendering (prevents matplotlib overflow)
        max_display_points = 50000
        if len(x_display) > max_display_points:
            # Downsample by taking every nth point
            step = len(x_display) // max_display_points
            x_display = x_display[::step]
            y_display = y_display[::step]
            colors = colors[::step]

        # Update plot - both line and points for realistic oscilloscope effect
        self.line.set_data(x_display, y_display)
        # Restore line to faint for static display (bright scatter provides visibility)
        self.line.set_alpha(0.15)
        self.line.set_linewidth(0.5)

        # Clear old scatter plot and create new one (prevents ghosting/persistence)
        self.points.remove()
        self.points = self.ax.scatter(x_display, y_display, c=colors, s=1.5)

        # Update limits if needed
        margin = 0.1
        x_range = [x_display.min() - margin, x_display.max() + margin]
        y_range = [y_display.min() - margin, y_display.max() + margin]

        max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        center_x = (x_range[0] + x_range[1]) / 2
        center_y = (y_range[0] + y_range[1]) / 2

        self.ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
        self.ax.set_ylim(center_y - max_range/2, center_y + max_range/2)

        self.canvas.draw_idle()
    
    def generate_audio(self):
        """Generate audio with current parameters and effects - ALL EFFECTS BLEND TOGETHER"""

        self.status_label.config(text="Generating audio...")
        self.root.update()

        # Normalize base pattern
        x_norm = self.normalize_data(self.x_data)
        y_norm = self.normalize_data(self.y_data)

        # Get parameters
        n_repeat = self.n_repeat_var.get()
        rotation_mode = self.rotation_mode_var.get()

        # ===================================================================
        # NEW APPROACH: Build effect factor arrays, then apply all together
        # ===================================================================

        # Step 1: Determine which effects are enabled
        has_y_fade = self.y_fade_var.get()
        has_x_fade = self.x_fade_var.get()
        has_shrink = self.shrink_var.get()
        has_rotation = rotation_mode in ["CW", "CCW"]

        # Step 2: Build effect factor arrays
        y_fade_factors = None
        x_fade_factors = None
        shrink_factors = None
        rotation_angles = None

        # Build Y-Fade factors
        if has_y_fade:
            n_fade_y = self.y_fade_steps.get()
            y_fade_speed = self.y_fade_speed.get()
            # Create one complete fade cycle: 1 → 0 → -1 → 0 → 1
            fade_down_y = np.linspace(1, 0, n_fade_y, dtype=np.float32)
            fade_negative_down_y = np.linspace(0, -1, n_fade_y, dtype=np.float32)[1:]
            fade_negative_up_y = np.linspace(-1, 0, n_fade_y, dtype=np.float32)[1:]
            fade_up_y = np.linspace(0, 1, n_fade_y, dtype=np.float32)[1:]
            one_cycle_y = np.concatenate([fade_down_y, fade_negative_down_y, fade_negative_up_y, fade_up_y])
            # Apply speed by repeating each step
            y_fade_factors = np.repeat(one_cycle_y, y_fade_speed)

        # Build X-Fade factors
        if has_x_fade:
            n_fade_x = self.x_fade_steps.get()
            x_fade_speed = self.x_fade_speed.get()
            # Create one complete fade cycle: 1 → 0 → -1 → 0 → 1
            fade_down_x = np.linspace(1, 0, n_fade_x, dtype=np.float32)
            fade_negative_down_x = np.linspace(0, -1, n_fade_x, dtype=np.float32)[1:]
            fade_negative_up_x = np.linspace(-1, 0, n_fade_x, dtype=np.float32)[1:]
            fade_up_x = np.linspace(0, 1, n_fade_x, dtype=np.float32)[1:]
            one_cycle_x = np.concatenate([fade_down_x, fade_negative_down_x, fade_negative_up_x, fade_up_x])
            # Apply speed by repeating each step
            x_fade_factors = np.repeat(one_cycle_x, x_fade_speed)

        # Build Shrink factors
        if has_shrink:
            n_shrink = self.shrink_steps.get()
            shrink_speed = self.shrink_speed.get()
            # Create one complete shrink cycle: 1 → 0 → 1
            shrink_down = np.linspace(1, 0, n_shrink, dtype=np.float32)
            shrink_up = np.linspace(0, 1, n_shrink, dtype=np.float32)[1:]
            one_cycle_shrink = np.concatenate([shrink_down, shrink_up])
            # Apply speed by repeating each step
            shrink_factors = np.repeat(one_cycle_shrink, shrink_speed)

        # Build Rotation angles
        if has_rotation:
            speed = self.rotation_speed.get()
            direction = -1 if rotation_mode == "CW" else 1
            # Create angle sequence for n_repeat steps
            rotation_angles = np.array([direction * speed * i for i in range(n_repeat)], dtype=np.float32)

        # Step 3: Determine total number of frames
        # If ANY effect is enabled, use n_repeat as base
        # Otherwise just repeat the pattern n_repeat times

        any_effect_enabled = has_y_fade or has_x_fade or has_shrink or has_rotation

        if not any_effect_enabled:
            # No effects - simple tile
            x_repeated = np.tile(x_norm, n_repeat)
            y_repeated = np.tile(y_norm, n_repeat)

        else:
            # Effects enabled - apply all simultaneously frame-by-frame
            x_frames = []
            y_frames = []

            # Check for alternate X/Y fade mode
            use_alternate = (self.alternate_xy_fade_var.get() and
                           has_x_fade and has_y_fade and
                           not has_shrink and not has_rotation)

            if use_alternate:
                # ALTERNATE MODE: X fade first, then Y fade, repeat
                # This is a special case where X and Y don't blend
                for repeat_idx in range(n_repeat):
                    # First: Do X fade sequence
                    for x_fade_idx in range(len(x_fade_factors)):
                        x_factor = x_fade_factors[x_fade_idx]
                        x_frames.append(x_norm * x_factor)
                        y_frames.append(y_norm)

                    # Then: Do Y fade sequence
                    for y_fade_idx in range(len(y_fade_factors)):
                        y_factor = y_fade_factors[y_fade_idx]
                        x_frames.append(x_norm)
                        y_frames.append(y_norm * y_factor)

                x_repeated = np.concatenate(x_frames)
                y_repeated = np.concatenate(y_frames)

            else:
                # BLENDED MODE: All effects apply simultaneously to each frame
                for frame_idx in range(n_repeat):
                    # Get current effect factors (cycling through each effect's sequence)
                    y_factor = 1.0
                    x_factor = 1.0
                    scale_factor = 1.0
                    rotation_angle = 0.0

                    if y_fade_factors is not None:
                        y_factor = y_fade_factors[frame_idx % len(y_fade_factors)]

                    if x_fade_factors is not None:
                        x_factor = x_fade_factors[frame_idx % len(x_fade_factors)]

                    if shrink_factors is not None:
                        scale_factor = shrink_factors[frame_idx % len(shrink_factors)]

                    if rotation_angles is not None:
                        rotation_angle = rotation_angles[frame_idx % len(rotation_angles)]

                    # Apply ALL effects to this frame
                    x_current = x_norm * x_factor * scale_factor
                    y_current = y_norm * y_factor * scale_factor

                    # Apply rotation if present
                    if rotation_angle != 0.0:
                        angle_rad = np.radians(rotation_angle)
                        cos_a = np.cos(angle_rad)
                        sin_a = np.sin(angle_rad)

                        x_rot = x_current * cos_a - y_current * sin_a
                        y_rot = x_current * sin_a + y_current * cos_a

                        x_frames.append(x_rot)
                        y_frames.append(y_rot)
                    else:
                        x_frames.append(x_current)
                        y_frames.append(y_current)

                x_repeated = np.concatenate(x_frames)
                y_repeated = np.concatenate(y_frames)

                # Renormalize entire sequence to prevent clipping
                x_repeated = self.normalize_data(x_repeated)
                y_repeated = self.normalize_data(y_repeated)

        # Apply Mirror Reflections if enabled
        if hasattr(self, 'reflections_var') and self.reflections_var.get():
            x_repeated, y_repeated = self.apply_reflections(x_repeated, y_repeated)

        # Handle Static rotation (applied after all other effects)
        if rotation_mode == "Static":
            angle_rad = np.radians(self.rotation_angle.get())
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            x_rot = x_repeated * cos_a - y_repeated * sin_a
            y_rot = x_repeated * sin_a + y_repeated * cos_a

            # Renormalize to prevent clipping
            x_repeated = self.normalize_data(x_rot)
            y_repeated = self.normalize_data(y_rot)

        # Calculate playback rate and target length
        fs = self.sample_rate_var.get()
        mult = self.freq_mult_var.get()
        duration = self.duration_var.get()
        actual_fs = fs * mult
        target_length = int(actual_fs * duration)

        # Tile to fill duration
        seq_len = len(x_repeated)
        num_tiles = int(np.ceil(target_length / seq_len))

        x_full = np.tile(x_repeated, num_tiles)[:target_length]
        y_full = np.tile(y_repeated, num_tiles)[:target_length]

        # Apply noise if enabled (independent on each channel)
        if self.x_noise_var.get():
            x_noise_amp = self.x_noise_amp.get()
            x_noise = np.random.uniform(-x_noise_amp, x_noise_amp, len(x_full))
            x_full = x_full + x_noise

        if self.y_noise_var.get():
            y_noise_amp = self.y_noise_amp.get()
            y_noise = np.random.uniform(-y_noise_amp, y_noise_amp, len(y_full))
            y_full = y_full + y_noise

        # Apply wavy effect if enabled
        if self.x_wavy_var.get() or self.y_wavy_var.get():
            # Create time array based on actual sample positions
            t = np.arange(len(x_full)) / actual_fs

            if self.x_wavy_var.get():
                K_x = self.x_wavy_amp.get()
                w_x = self.x_wavy_freq.get()
                x_full = x_full + K_x * np.sin(w_x * 2 * np.pi * t)

            if self.y_wavy_var.get():
                K_y = self.y_wavy_amp.get()
                w_y = self.y_wavy_freq.get()
                y_full = y_full + K_y * np.sin(w_y * 2 * np.pi * t)

        # Apply tremolo effect if enabled (amplitude modulation)
        if hasattr(self, 'tremolo_var') and self.tremolo_var.get():
            # Create time array if not already created
            if not (self.x_wavy_var.get() or self.y_wavy_var.get()):
                t = np.arange(len(x_full)) / actual_fs

            depth = self.tremolo_depth.get() / 100.0  # Convert percentage to 0-1
            rate = self.tremolo_rate.get()
            wave_type = self.tremolo_wave_var.get()

            # Generate modulation waveform
            if wave_type == "sine":
                mod = np.sin(2 * np.pi * rate * t)
            elif wave_type == "triangle":
                mod = 2 * np.abs(2 * (rate * t - np.floor(rate * t + 0.5))) - 1
            else:  # square
                mod = np.sign(np.sin(2 * np.pi * rate * t))

            # Scale modulation: (1 - depth) + depth * modulation
            # This keeps amplitude between (1-depth) and 1
            modulation = (1 - depth) + depth * (mod + 1) / 2

            # Apply to both channels
            x_full = x_full * modulation
            y_full = y_full * modulation

        # Apply ring modulation effect if enabled
        if self.ring_mod_var.get():
            # Create time array if not already created
            if not (self.x_wavy_var.get() or self.y_wavy_var.get() or self.tremolo_var.get()):
                t = np.arange(len(x_full)) / actual_fs

            carrier_freq = self.ring_carrier_freq.get()
            mix = self.ring_mix.get() / 100.0

            # Generate carrier wave
            carrier = np.sin(2 * np.pi * carrier_freq * t)

            # Ring modulation: multiply signal by carrier
            x_mod = x_full * carrier
            y_mod = y_full * carrier

            # Mix dry and modulated signals
            x_full = (1 - mix) * x_full + mix * x_mod
            y_full = (1 - mix) * y_full + mix * y_mod

        # Apply distortion effect if enabled
        if hasattr(self, 'distortion_var') and self.distortion_var.get():
            threshold = self.distortion_threshold.get()
            dist_type = self.distortion_type_var.get()

            if dist_type == "soft":
                # Soft clipping (tanh-like curve)
                x_full = np.tanh(x_full / threshold) * threshold
                y_full = np.tanh(y_full / threshold) * threshold
            elif dist_type == "hard":
                # Hard clipping
                x_full = np.clip(x_full, -threshold, threshold)
                y_full = np.clip(y_full, -threshold, threshold)
            else:  # fold
                # Wave folding
                x_full = np.where(np.abs(x_full) > threshold,
                           threshold - (np.abs(x_full) - threshold), x_full)
                y_full = np.where(np.abs(y_full) > threshold,
                           threshold - (np.abs(y_full) - threshold), y_full)

        # Apply echo/delay effect if enabled
        if hasattr(self, 'echo_var') and self.echo_var.get():
            num_echoes = self.echo_count.get()
            decay = self.echo_decay.get()
            delay_pct = self.echo_delay.get() / 100.0

            delay_samples = int(len(x_full) * delay_pct)

            # Create arrays to hold echo sum
            x_with_echo = np.copy(x_full)
            y_with_echo = np.copy(y_full)

            # Add delayed copies with decay
            for i in range(1, num_echoes + 1):
                offset = i * delay_samples
                amplitude = decay ** i

                # Pad with zeros and add delayed signal
                if offset < len(x_full):
                    x_delayed = np.concatenate([np.zeros(offset), x_full[:-offset]]) * amplitude
                    y_delayed = np.concatenate([np.zeros(offset), y_full[:-offset]]) * amplitude
                    x_with_echo += x_delayed
                    y_with_echo += y_delayed

            x_full = x_with_echo
            y_full = y_with_echo

        # Apply kaleidoscope effect if enabled
        if hasattr(self, 'kaleido_var') and self.kaleido_var.get():
            sections = self.kaleido_sections.get()
            mirror = self.kaleido_mirror_var.get()

            # Create rotated copies of the pattern
            all_x = []
            all_y = []

            for i in range(sections):
                angle = (2 * np.pi * i) / sections

                # Rotate the pattern
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                x_rot = x_full * cos_a - y_full * sin_a
                y_rot = x_full * sin_a + y_full * cos_a

                all_x.append(x_rot)
                all_y.append(y_rot)

                # Add mirrored copy if enabled
                if mirror:
                    x_mir = x_full * cos_a + y_full * sin_a
                    y_mir = -x_full * sin_a + y_full * cos_a
                    all_x.append(x_mir)
                    all_y.append(y_mir)

            # Concatenate all sections
            x_full = np.concatenate(all_x)
            y_full = np.concatenate(all_y)

        # Create stereo
        stereo = np.column_stack([x_full, y_full]).astype(np.float32)

        self.current_audio = stereo
        self.current_fs = actual_fs

        # Update status with rotation info if applicable
        if has_rotation and rotation_angles is not None:
            num_full_rotations = (len(rotation_angles) * self.rotation_speed.get()) / 360
            self.status_label.config(text=f"Ready - {len(stereo)} samples @ {actual_fs/1000:.0f}kHz ({num_full_rotations:.1f} rotations)")
        else:
            self.status_label.config(text=f"Ready - {len(stereo)} samples @ {actual_fs/1000:.0f}kHz")

        return stereo, actual_fs
    
    def apply_reflections(self, x, y):
        """
        Apply mirror reflections as in original MATLAB code
        This creates the 4-stage reflection pattern
        """
        # Get current length
        base_len = len(x)
        
        # Stage 1: Already have the base pattern (or faded pattern)
        # Just keep x and y as-is for first stage
        
        # Stage 2: Negative Y reflection (mirror about x-axis)
        # Flip y to negative
        x_stage2 = x.copy()
        y_stage2 = -y.copy()
        
        # Stage 3: Negative X reflection (mirror about y-axis)
        # Flip x to negative, keep y at last value
        x_stage3 = -x.copy()
        y_stage3 = np.full(base_len, y[-1]) if len(y) > 0 else y.copy()
        
        # Stage 4: Positive X reflection (back to positive x)
        # Return x to positive, keep y at last value
        x_stage4 = x.copy()
        y_stage4 = np.full(base_len, y[-1]) if len(y) > 0 else y.copy()
        
        # Concatenate all stages
        x_reflected = np.concatenate([x, x_stage2, x_stage3, x_stage4])
        y_reflected = np.concatenate([y, y_stage2, y_stage3, y_stage4])
        
        return x_reflected, y_reflected

    def create_fade_sequence(self, x_norm, y_norm, n_fade, enable_reflections=False):
        """
        DEPRECATED - kept for compatibility but not used in main flow
        Use individual fade checkboxes instead
        """
        if not enable_reflections:
            # Simple fade - just fade y while keeping x constant
            fade_factors = np.linspace(1, 0, n_fade, dtype=np.float32)
            x_seq = np.tile(x_norm, n_fade)
            y_seq = np.concatenate([y_norm * fade_factors[i] for i in range(n_fade)])
            return x_seq, y_seq
        
        # With reflections - recreate MATLAB behavior exactly
        fade_factors = np.linspace(1, 0, n_fade, dtype=np.float32)
        
        # Stage 1: Y fades from 1 to 0, X stays constant
        x_seq = np.tile(x_norm, n_fade)
        y_seq = np.concatenate([y_norm * fade_factors[i] for i in range(n_fade)])
        
        # Stage 2: Y goes from 0 to -1 (negative reflection about x-axis)
        neg_fade = -fade_factors[1:]  # Skip first (0) to avoid duplicate
        for factor in neg_fade:
            y_seq = np.concatenate([y_seq, y_norm * factor])
            x_seq = np.concatenate([x_seq, x_norm])
        
        # Stage 3: X goes from 1 to -1 (negative reflection about y-axis)
        # Y stays at last value
        for factor in neg_fade:
            x_seq = np.concatenate([x_seq, x_norm * factor])
            y_seq = np.concatenate([y_seq, y_norm])
        
        # Stage 4: X goes from -1 back toward 1 (positive reflection)
        # Y stays at last value
        pos_fade = fade_factors[1:]  # Skip first to avoid duplicate
        for factor in pos_fade:
            x_seq = np.concatenate([x_seq, x_norm * factor])
            y_seq = np.concatenate([y_seq, y_norm])
        
        return x_seq, y_seq
    
    def apply_parameters(self):
        """Apply parameters, generate audio, and auto-play"""
        # Prevent concurrent regenerations
        if self.is_regenerating:
            return

        try:
            self.is_regenerating = True

            # Stop current playback if playing and clear the buffer
            was_playing = self.is_playing
            if self.is_playing:
                self.stop_playback()

            # Generate new audio
            self.generate_audio()

            # Auto-play the generated audio if it was playing before
            if was_playing or not hasattr(self, '_first_generate'):
                self.start_playback()
                self._first_generate = True

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate audio:\n{str(e)}")
        finally:
            self.is_regenerating = False
    
    def toggle_playback(self):
        """Toggle between play and pause"""
        if self.is_playing:
            # Pause/Stop
            self.stop_playback()
        else:
            # Play/Resume
            self.start_playback()
    
    def start_playback(self):
        """Start playing audio"""
        if self.current_audio is None:
            messagebox.showwarning("No Audio", "Please click 'Apply & Generate' first to create audio.")
            return

        self.is_playing = True
        self.play_btn.config(text="⏸ Pause")

        # Update status based on live preview mode
        if self.preview_active:
            self.status_label.config(text="Playing (Live Preview)")
        else:
            self.status_label.config(text="Playing...")

        # Track playback start time for live preview
        import time
        self.playback_start_time = time.time()
        self.preview_position = 0
        self.preview_buffer = []
        self.last_preview_update = 0

        self.stop_flag.clear()
        self.audio_thread = threading.Thread(target=self.play_audio_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def stop_playback(self):
        """Stop playing audio"""
        self.stop_flag.set()
        sd.stop()
        self.is_playing = False
        self.play_btn.config(text="▶ Play Audio")
        self.status_label.config(text="Paused")

        # Always restore static display (including scatter plot) when stopping
        self.update_display()
    
    def play_audio_thread(self):
        """Thread function for playing audio in continuous loop"""
        try:
            while not self.stop_flag.is_set():
                # Play audio with looping
                sd.play(self.current_audio, self.current_fs, loop=True)
                # Wait for playback to be stopped
                sd.wait()
                # Check if we should continue looping
                if self.stop_flag.is_set():
                    break
        except Exception as e:
            self.update_queue.put(("error", str(e)))
    
    def check_updates(self):
        """Check for updates from audio thread"""
        try:
            try:
                while True:
                    msg, data = self.update_queue.get_nowait()

                    if msg == "playback_complete":
                        self.is_playing = False
                        self.play_btn.config(text="▶ Play Audio")
                        self.status_label.config(text="Playback complete")
                    elif msg == "error":
                        self.is_playing = False
                        self.play_btn.config(text="▶ Play Audio")
                        self.status_label.config(text=f"Error: {data}")
            except queue.Empty:
                pass

            # Schedule next update
            if self.root.winfo_exists():
                self.root.after(50, self.check_updates)
        except Exception:
            pass  # Window destroyed, stop scheduling
    
    def load_txt_file(self):
        """Load coordinates from text file with x_fun=[] and y_fun=[] format"""
        filename = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            x, y = self.extract_txt_arrays(filename)
            self.x_data = x
            self.y_data = y
            self.data_info_label.config(text=f"Points: {len(x)}")
            self.update_display()
            self.status_label.config(text=f"Loaded {len(x)} points from text file")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load text file:\n{str(e)}")
    
    def extract_txt_arrays(self, filename):
        """
        Extract x_fun and y_fun arrays from text file
        Supports formats:
        - x_fun=[1 2 3 4];
        - x_fun=[ 1 2 3 4 ];
        - x_fun=[1,2,3,4];
        - x_fun=[ 1, 2, 3, 4 ];
        """
        import re
        
        with open(filename, 'r') as f:
            content = f.read()
        
        def extract_array(var_name):
            # Pattern to match: var_name = [ ... ];
            # Handles spaces, commas, semicolons
            pattern = rf'{var_name}\s*=\s*\[(.*?)\];'
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                raise ValueError(f"Could not find '{var_name}=[...];' in file")
            
            array_str = match.group(1)
            
            # Remove comments if any
            array_str = re.sub(r'%.*?$', '', array_str, flags=re.MULTILINE)
            
            # Replace multiple spaces/newlines with single space
            array_str = re.sub(r'\s+', ' ', array_str)
            
            # Remove commas (treat them as spaces)
            array_str = array_str.replace(',', ' ')
            
            # Split and convert to float
            values = [float(x.strip()) for x in array_str.split() if x.strip()]
            
            if len(values) == 0:
                raise ValueError(f"No values found for '{var_name}'")
            
            return np.array(values, dtype=np.float32)
        
        x_fun = extract_array('x_fun')
        y_fun = extract_array('y_fun')
        
        # Verify arrays are same length
        if len(x_fun) != len(y_fun):
            raise ValueError(f"Array length mismatch: x_fun has {len(x_fun)} points, y_fun has {len(y_fun)} points")
        
        return x_fun, y_fun
    
    def load_matlab_file(self):
        """Load coordinates from MATLAB file"""
        filename = filedialog.askopenfilename(
            title="Select MATLAB File",
            filetypes=[("MATLAB Files", "*.m"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            x, y = self.extract_matlab_arrays(filename)
            self.x_data = x
            self.y_data = y
            self.data_info_label.config(text=f"Points: {len(x)}")
            self.update_display()
            self.status_label.config(text=f"Loaded {len(x)} points from MATLAB file")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load MATLAB file:\n{str(e)}")
    
    def extract_matlab_arrays(self, filename):
        """Extract x_fun and y_fun from MATLAB file"""
        import re
        
        with open(filename, 'r') as f:
            content = f.read()
        
        def extract_array(var_name):
            pattern = rf'{var_name}\s*=\s*\[(.*?)\];'
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                raise ValueError(f"Could not find '{var_name}'")
            array_str = match.group(1)
            array_str = re.sub(r'%.*?$', '', array_str, flags=re.MULTILINE)
            values = [float(x.strip()) for x in array_str.split(',') if x.strip()]
            return np.array(values, dtype=np.float32)
        
        return extract_array('x_fun'), extract_array('y_fun')
    
    def load_numpy_file(self):
        """Load coordinates from NumPy file"""
        filename = filedialog.askopenfilename(
            title="Select NumPy File",
            filetypes=[("NumPy Files", "*.npz"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            data = np.load(filename)
            self.x_data = data['x']
            self.y_data = data['y']
            self.data_info_label.config(text=f"Points: {len(self.x_data)}")
            self.update_display()
            self.status_label.config(text=f"Loaded {len(self.x_data)} points from NumPy file")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NumPy file:\n{str(e)}")
    
    def generate_test_pattern(self):
        """Generate a test pattern with parametric equations"""
        patterns = {
            # Basic shapes
            "Circle": lambda t: (np.cos(t), np.sin(t)),
            "Ellipse": lambda t: (1.5*np.cos(t), np.sin(t)),

            # Lissajous curves with different phase shifts
            "Lissajous 3:2": lambda t: (np.sin(3*t), np.sin(2*t)),
            "Lissajous 3:2 (π/2 phase)": lambda t: (np.sin(3*t), np.sin(2*t + np.pi/2)),
            "Lissajous 5:4": lambda t: (np.sin(5*t), np.sin(4*t)),
            "Lissajous 5:4 (π/4 phase)": lambda t: (np.sin(5*t), np.sin(4*t + np.pi/4)),
            "Lissajous 7:5 (π/3 phase)": lambda t: (np.sin(7*t), np.sin(5*t + np.pi/3)),

            # Stars and flowers
            "Star (5-point)": lambda t: (np.cos(t) * (1 + 0.5*np.sin(5*t)),
                                         np.sin(t) * (1 + 0.5*np.sin(5*t))),
            "Flower (6-petal)": lambda t: (np.cos(t) * (1 + 0.3*np.cos(6*t)),
                                          np.sin(t) * (1 + 0.3*np.cos(6*t))),
            "Rose Curve (4-petal)": lambda t: (np.cos(4*t) * np.cos(t),
                                               np.cos(4*t) * np.sin(t)),

            # Spirals
            "Spiral (Archimedean)": lambda t: (t/10*np.cos(t), t/10*np.sin(t)),
            "Spiral (Logarithmic)": lambda t: (np.exp(t/10)*np.cos(t), np.exp(t/10)*np.sin(t)),

            # Parametric 3D-Style Curves
            # Note: True 3D requires X,Y,Z. These are 2D curves that create 3D-like effects
            "Helix (3D Spiral)": lambda t: (np.cos(t)*0.7, np.sin(t)*0.7 + (t-np.pi)/(2*np.pi)*0.5),
            "Trefoil Knot": lambda t: ((np.sin(t) + 2*np.sin(2*t)) * 0.3,
                                       (np.cos(t) - 2*np.cos(2*t)) * 0.3),
            "Torus Knot": lambda t: ((2 + np.cos(3*t))*np.cos(2*t)*0.3,
                                     (2 + np.cos(3*t))*np.sin(2*t)*0.3),

            # Complex parametric curves
            "Hypotrochoid": lambda t: ((3)*np.cos(t) + (1)*np.cos((3)/(1)*t),
                                       (3)*np.sin(t) - (1)*np.sin((3)/(1)*t)),
            "Epitrochoid": lambda t: ((5)*np.cos(t) - (2)*np.cos((5)/(2)*t),
                                      (5)*np.sin(t) - (2)*np.sin((5)/(2)*t)),
            "Butterfly Curve": lambda t: (np.sin(t)*(np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5),
                                         np.cos(t)*(np.exp(np.cos(t)) - 2*np.cos(4*t) - np.sin(t/12)**5)),

            # Figure-8 and infinity
            "Figure-8": lambda t: (np.sin(t), np.sin(2*t)),
            "Infinity (∞)": lambda t: (np.cos(t), np.sin(2*t)),

            # Cardioid and epicycloid
            "Cardioid": lambda t: ((1-np.cos(t))*np.cos(t), (1-np.cos(t))*np.sin(t)),
            "Deltoid": lambda t: (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) - np.sin(2*t)),
        }

        # Create scrollable dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Test Pattern")
        dialog.geometry("380x550")

        ctk.CTkLabel(dialog, text="Choose a parametric pattern:",
                 font=('Arial', 10, 'bold')).pack(pady=10)

        # Create frame for scrollable area
        scroll_container = ctk.CTkFrame(dialog)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Create canvas with scrollbar - set explicit height to ensure scrollbar shows
        canvas = tk.Canvas(scroll_container, highlightthickness=0, height=400)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ctk.CTkFrame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        selected = tk.StringVar(value="Circle")

        # Group patterns by category
        categories = {
            "Basic Shapes": ["Circle", "Ellipse"],
            "Lissajous Curves": [k for k in patterns.keys() if "Lissajous" in k],
            "Stars & Flowers": ["Star (5-point)", "Flower (6-petal)", "Rose Curve (4-petal)"],
            "Spirals": ["Spiral (Archimedean)", "Spiral (Logarithmic)"],
            "Knots & 3D-Style": ["Helix (3D Spiral)", "Trefoil Knot", "Torus Knot"],
            "Complex Curves": ["Hypotrochoid", "Epitrochoid", "Butterfly Curve",
                              "Cardioid", "Deltoid"],
            "Special": ["Figure-8", "Infinity (∞)"]
        }

        for category, pattern_names in categories.items():
            ctk.CTkLabel(scrollable_frame, text=category,
                     font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
            for name in pattern_names:
                ctk.CTkRadioButton(scrollable_frame, text=name, variable=selected,
                               value=name).pack(anchor=tk.W, padx=30)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)

        # Button frame at bottom (always visible)
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def apply():
            pattern_name = selected.get()

            # Different time ranges for different patterns
            if "Spiral" in pattern_name:
                t = np.linspace(0, 6*np.pi, 800)
            elif "Butterfly" in pattern_name:
                t = np.linspace(0, 12*np.pi, 1000)
            elif "Lissajous" in pattern_name or "Figure" in pattern_name or "Infinity" in pattern_name:
                t = np.linspace(0, 2*np.pi, 500)
            else:
                t = np.linspace(0, 2*np.pi, 600)

            x, y = patterns[pattern_name](t)
            self.x_data = x
            self.y_data = y
            self.data_info_label.config(text=f"Points: {len(x)}")
            self.update_display()
            self.status_label.config(text=f"Generated {pattern_name} pattern")
            dialog.destroy()

        ctk.CTkButton(button_frame, text="Generate Pattern", command=apply).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

    def open_drawing_canvas(self):
        """Open a canvas for drawing custom patterns"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Draw Pattern")
        dialog.geometry("550x650")

        # Instructions
        instruction_frame = ctk.CTkFrame(dialog)
        instruction_frame.pack(pady=10)
        ctk.CTkLabel(instruction_frame, text="Draw your pattern below:",
                 font=('Arial', 10, 'bold')).pack()
        ctk.CTkLabel(instruction_frame, text="Click and drag to draw • The path will be traced in order",
                 font=('Arial', 8), text_color='gray').pack()

        # Drawing canvas (square aspect ratio to match oscilloscope display)
        canvas_frame = ctk.CTkFrame(dialog)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(canvas_frame, width=500, height=500, bg='white',
                          highlightthickness=1, highlightbackground='gray')
        canvas.pack()

        # Drawing state
        drawing_data = {
            'is_drawing': False,
            'points': [],  # Store (x, y) tuples
            'canvas_objects': []  # Store line IDs for clearing
        }

        def start_drawing(event):
            """Start drawing when mouse button is pressed"""
            drawing_data['is_drawing'] = True
            # Append to existing points instead of replacing them
            drawing_data['points'].append((event.x, event.y))

        def draw(event):
            """Draw as mouse moves"""
            if drawing_data['is_drawing']:
                x, y = event.x, event.y
                prev_x, prev_y = drawing_data['points'][-1]

                # Draw line from previous point to current point
                line_id = canvas.create_line(prev_x, prev_y, x, y,
                                             fill='blue', width=2, capstyle=tk.ROUND)
                drawing_data['canvas_objects'].append(line_id)
                drawing_data['points'].append((x, y))

        def stop_drawing(event):
            """Stop drawing when mouse button is released"""
            drawing_data['is_drawing'] = False

        def clear_canvas():
            """Clear the canvas"""
            # Delete all drawn objects
            for obj_id in drawing_data['canvas_objects']:
                canvas.delete(obj_id)
            drawing_data['canvas_objects'] = []
            drawing_data['points'] = []

        def apply_drawing():
            """Convert drawn path to oscilloscope data"""
            if len(drawing_data['points']) < 2:
                messagebox.showwarning("No Drawing", "Please draw a pattern first!")
                return

            # Get canvas dimensions (square aspect ratio)
            canvas_size = 500

            # Convert canvas coordinates to normalized coordinates (-1 to 1)
            points = np.array(drawing_data['points'])
            x_canvas = points[:, 0]
            y_canvas = points[:, 1]

            # Center and normalize (canvas is square, so same scaling for both axes)
            # X: left=0 -> -1, center=250 -> 0, right=500 -> 1
            x_norm = (x_canvas - canvas_size/2) / (canvas_size/2)

            # Y: top=0 -> 1, center=250 -> 0, bottom=500 -> -1 (flip Y axis)
            y_norm = -(y_canvas - canvas_size/2) / (canvas_size/2)

            # Set as current data
            self.x_data = x_norm
            self.y_data = y_norm
            self.data_info_label.config(text=f"Points: {len(x_norm)}")
            self.update_display()
            self.status_label.config(text=f"Loaded drawn pattern ({len(x_norm)} points)")
            dialog.destroy()

        # Bind mouse events
        canvas.bind("<Button-1>", start_drawing)
        canvas.bind("<B1-Motion>", draw)
        canvas.bind("<ButtonRelease-1>", stop_drawing)

        # Button frame
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ctk.CTkButton(button_frame, text="Clear", command=clear_canvas).pack(
            side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Apply Drawing", command=apply_drawing).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=5)

    def open_harmonic_sum(self):
        """Open dialog to create pattern from sum of harmonics"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Sum of Harmonics")
        dialog.geometry("700x600")

        # Instructions
        instruction_frame = ctk.CTkFrame(dialog)
        instruction_frame.pack(pady=10, padx=10, fill=tk.X)
        ctk.CTkLabel(instruction_frame, text="Create pattern from sum of sinusoidal terms",
                 font=('Arial', 10, 'bold')).pack()
        ctk.CTkLabel(instruction_frame, text="X(t) = Σ A_n·sin(ω_n·t + φ_n)  or  A_n·cos(ω_n·t + φ_n)",
                 font=('Arial', 8), text_color='gray').pack()

        # Main container with two columns
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Storage for terms
        # Each term: {'type': 'sin'/'cos', 'amp': float, 'freq': float, 'phase': float,
        #             'phase_sweep': bool, 'phase_start': float, 'phase_end': float, 'sweep_steps': int}
        x_terms = []
        y_terms = []

        # X Channel (Left)
        x_frame = ctk.CTkFrame(main_frame)
        x_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(x_frame, text="X Channel (Left)", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        x_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # X terms display
        x_canvas = tk.Canvas(x_frame, height=300, highlightthickness=1, highlightbackground='gray')
        x_scrollbar = ttk.Scrollbar(x_frame, orient="vertical", command=x_canvas.yview)
        x_scrollable = ctk.CTkFrame(x_canvas)

        x_scrollable.bind("<Configure>", lambda e: x_canvas.configure(scrollregion=x_canvas.bbox("all")))
        x_canvas.create_window((0, 0), window=x_scrollable, anchor="nw")
        x_canvas.configure(yscrollcommand=x_scrollbar.set)

        x_canvas.pack(side="left", fill="both", expand=True)
        x_scrollbar.pack(side="right", fill="y")

        # Y Channel (Right)
        y_frame = ctk.CTkFrame(main_frame)
        y_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(y_frame, text="Y Channel (Right)", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        y_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Y terms display
        y_canvas = tk.Canvas(y_frame, height=300, highlightthickness=1, highlightbackground='gray')
        y_scrollbar = ttk.Scrollbar(y_frame, orient="vertical", command=y_canvas.yview)
        y_scrollable = ctk.CTkFrame(y_canvas)

        y_scrollable.bind("<Configure>", lambda e: y_canvas.configure(scrollregion=y_canvas.bbox("all")))
        y_canvas.create_window((0, 0), window=y_scrollable, anchor="nw")
        y_canvas.configure(yscrollcommand=y_scrollbar.set)

        y_canvas.pack(side="left", fill="both", expand=True)
        y_scrollbar.pack(side="right", fill="y")

        def update_preview():
            """Update oscilloscope display with current harmonic sum"""
            # Time array (0 to 2π with high resolution)
            num_points = 1000
            t = np.linspace(0, 2*np.pi, num_points)

            # Calculate X channel (use phase_start for preview)
            if x_terms:
                x_data = np.zeros(num_points)
                for term in x_terms:
                    phase = term.get('phase_start', 0)
                    if term['type'] == 'sin':
                        x_data += term['amp'] * np.sin(term['freq'] * t + phase)
                    else:  # cos
                        x_data += term['amp'] * np.cos(term['freq'] * t + phase)
            else:
                x_data = t * 0  # Zero array

            # Calculate Y channel (use phase_start for preview)
            if y_terms:
                y_data = np.zeros(num_points)
                for term in y_terms:
                    phase = term.get('phase_start', 0)
                    if term['type'] == 'sin':
                        y_data += term['amp'] * np.sin(term['freq'] * t + phase)
                    else:  # cos
                        y_data += term['amp'] * np.cos(term['freq'] * t + phase)
            else:
                y_data = t * 0  # Zero array

            # Update display
            self.x_data = x_data
            self.y_data = y_data
            self.data_info_label.config(text=f"Points: {len(x_data)}")
            self.update_display()
            self.status_label.config(text=f"Preview: N={len(x_terms)} terms (X), M={len(y_terms)} terms (Y)")

        def refresh_x_display():
            """Refresh X channel term display"""
            for widget in x_scrollable.winfo_children():
                widget.destroy()

            if not x_terms:
                ctk.CTkLabel(x_scrollable, text="No terms added yet",
                         font=('Arial', 8, 'italic'), text_color='gray').pack(pady=10)
            else:
                for i, term in enumerate(x_terms):
                    term_frame = ctk.CTkFrame(x_scrollable, relief='solid', borderwidth=1)
                    term_frame.pack(fill=tk.X, pady=2, padx=2)

                    # Term label
                    term_text = f"{term['amp']:.2f}·{term['type']}({term['freq']:.1f}·t"
                    if term.get('phase_sweep', False):
                        term_text += f", φ sweep: {term['phase_start']:.2f}→{term['phase_end']:.2f}"
                    else:
                        if term.get('phase_start', 0) != 0:
                            term_text += f" + {term['phase_start']:.2f}"
                    term_text += ")"
                    ctk.CTkLabel(term_frame, text=f"Term {i+1}: {term_text}",
                             font=('Arial', 8)).pack(side=tk.LEFT, padx=5, pady=2)

                    # Remove button
                    def remove_x_term(idx=i):
                        x_terms.pop(idx)
                        refresh_x_display()
                        update_preview()

                    ctk.CTkButton(term_frame, text="✕", width=3,
                              command=remove_x_term).pack(side=tk.RIGHT, padx=2, pady=2)

        def refresh_y_display():
            """Refresh Y channel term display"""
            for widget in y_scrollable.winfo_children():
                widget.destroy()

            if not y_terms:
                ctk.CTkLabel(y_scrollable, text="No terms added yet",
                         font=('Arial', 8, 'italic'), text_color='gray').pack(pady=10)
            else:
                for i, term in enumerate(y_terms):
                    term_frame = ctk.CTkFrame(y_scrollable, relief='solid', borderwidth=1)
                    term_frame.pack(fill=tk.X, pady=2, padx=2)

                    # Term label
                    term_text = f"{term['amp']:.2f}·{term['type']}({term['freq']:.1f}·t"
                    if term.get('phase_sweep', False):
                        term_text += f", φ sweep: {term['phase_start']:.2f}→{term['phase_end']:.2f}"
                    else:
                        if term.get('phase_start', 0) != 0:
                            term_text += f" + {term['phase_start']:.2f}"
                    term_text += ")"
                    ctk.CTkLabel(term_frame, text=f"Term {i+1}: {term_text}",
                             font=('Arial', 8)).pack(side=tk.LEFT, padx=5, pady=2)

                    # Remove button
                    def remove_y_term(idx=i):
                        y_terms.pop(idx)
                        refresh_y_display()
                        update_preview()

                    ctk.CTkButton(term_frame, text="✕", width=3,
                              command=remove_y_term).pack(side=tk.RIGHT, padx=2, pady=2)

        # Add term controls for X
        x_add_frame = ctk.CTkFrame(x_frame)
        x_add_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(x_add_frame, text="Add Term", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        x_add_frame.pack(fill=tk.X, pady=(10, 0))

        # Row 1: Type selection
        x_type_row = ctk.CTkFrame(x_add_frame)
        x_type_row.pack(fill=tk.X, pady=2)
        x_type_var = tk.StringVar(value="sin")
        ctk.CTkRadioButton(x_type_row, text="sin", variable=x_type_var, value="sin").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(x_type_row, text="cos", variable=x_type_var, value="cos").pack(side=tk.LEFT, padx=5)

        # Row 2: Parameters
        x_param_row = ctk.CTkFrame(x_add_frame)
        x_param_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(x_param_row, text="A:").pack(side=tk.LEFT, padx=(5, 2))
        x_amp_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(x_param_row, textvariable=x_amp_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(x_param_row, text="ω:").pack(side=tk.LEFT, padx=(5, 2))
        x_freq_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(x_param_row, textvariable=x_freq_var, width=8).pack(side=tk.LEFT, padx=2)

        # Row 3: Phase sweep options
        x_phase_sweep_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(x_add_frame, text="Phase Sweep",
                       variable=x_phase_sweep_var).pack(anchor=tk.W, pady=2)

        x_phase_row = ctk.CTkFrame(x_add_frame)
        x_phase_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(x_phase_row, text="φ start:").pack(side=tk.LEFT, padx=(5, 2))
        x_phase_start_var = tk.DoubleVar(value=0.0)
        ctk.CTkEntry(x_phase_row, textvariable=x_phase_start_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(x_phase_row, text="φ end:").pack(side=tk.LEFT, padx=(5, 2))
        x_phase_end_var = tk.DoubleVar(value=6.28)  # 2π
        ctk.CTkEntry(x_phase_row, textvariable=x_phase_end_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(x_phase_row, text="steps:").pack(side=tk.LEFT, padx=(5, 2))
        x_sweep_steps_var = tk.IntVar(value=20)
        ctk.CTkEntry(x_phase_row, textvariable=x_sweep_steps_var, width=6).pack(side=tk.LEFT, padx=2)

        # Frequency sweep options
        x_freq_sweep_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(x_add_frame, text="Frequency Sweep",
                       variable=x_freq_sweep_var).pack(anchor=tk.W, pady=2)

        x_freq_sweep_row = ctk.CTkFrame(x_add_frame)
        x_freq_sweep_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(x_freq_sweep_row, text="ω start:").pack(side=tk.LEFT, padx=(5, 2))
        x_freq_start_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(x_freq_sweep_row, textvariable=x_freq_start_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(x_freq_sweep_row, text="ω end:").pack(side=tk.LEFT, padx=(5, 2))
        x_freq_end_var = tk.DoubleVar(value=5.0)
        ctk.CTkEntry(x_freq_sweep_row, textvariable=x_freq_end_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(x_freq_sweep_row, text="steps:").pack(side=tk.LEFT, padx=(5, 2))
        x_freq_sweep_steps_var = tk.IntVar(value=20)
        ctk.CTkEntry(x_freq_sweep_row, textvariable=x_freq_sweep_steps_var, width=6).pack(side=tk.LEFT, padx=2)

        # Row 4: Add button
        x_button_row = ctk.CTkFrame(x_add_frame)
        x_button_row.pack(fill=tk.X, pady=(5, 0))

        def add_x_term():
            x_terms.append({
                'type': x_type_var.get(),
                'amp': x_amp_var.get(),
                'freq': x_freq_var.get(),
                'phase_sweep': x_phase_sweep_var.get(),
                'phase_start': x_phase_start_var.get(),
                'phase_end': x_phase_end_var.get(),
                'sweep_steps': x_sweep_steps_var.get(),
                'freq_sweep': x_freq_sweep_var.get(),
                'freq_start': x_freq_start_var.get(),
                'freq_end': x_freq_end_var.get(),
                'freq_sweep_steps': x_freq_sweep_steps_var.get()
            })
            refresh_x_display()
            update_preview()

        ctk.CTkButton(x_button_row, text="Add Term to X Channel", command=add_x_term).pack(fill=tk.X, padx=5)

        # Add term controls for Y
        y_add_frame = ctk.CTkFrame(y_frame)
        y_add_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(y_add_frame, text="Add Term", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        y_add_frame.pack(fill=tk.X, pady=(10, 0))

        # Row 1: Type selection
        y_type_row = ctk.CTkFrame(y_add_frame)
        y_type_row.pack(fill=tk.X, pady=2)
        y_type_var = tk.StringVar(value="sin")
        ctk.CTkRadioButton(y_type_row, text="sin", variable=y_type_var, value="sin").pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(y_type_row, text="cos", variable=y_type_var, value="cos").pack(side=tk.LEFT, padx=5)

        # Row 2: Parameters
        y_param_row = ctk.CTkFrame(y_add_frame)
        y_param_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(y_param_row, text="A:").pack(side=tk.LEFT, padx=(5, 2))
        y_amp_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(y_param_row, textvariable=y_amp_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(y_param_row, text="ω:").pack(side=tk.LEFT, padx=(5, 2))
        y_freq_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(y_param_row, textvariable=y_freq_var, width=8).pack(side=tk.LEFT, padx=2)

        # Row 3: Phase sweep options
        y_phase_sweep_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(y_add_frame, text="Phase Sweep",
                       variable=y_phase_sweep_var).pack(anchor=tk.W, pady=2)

        y_phase_row = ctk.CTkFrame(y_add_frame)
        y_phase_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(y_phase_row, text="φ start:").pack(side=tk.LEFT, padx=(5, 2))
        y_phase_start_var = tk.DoubleVar(value=0.0)
        ctk.CTkEntry(y_phase_row, textvariable=y_phase_start_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(y_phase_row, text="φ end:").pack(side=tk.LEFT, padx=(5, 2))
        y_phase_end_var = tk.DoubleVar(value=6.28)  # 2π
        ctk.CTkEntry(y_phase_row, textvariable=y_phase_end_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(y_phase_row, text="steps:").pack(side=tk.LEFT, padx=(5, 2))
        y_sweep_steps_var = tk.IntVar(value=20)
        ctk.CTkEntry(y_phase_row, textvariable=y_sweep_steps_var, width=6).pack(side=tk.LEFT, padx=2)

        # Frequency sweep options
        y_freq_sweep_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(y_add_frame, text="Frequency Sweep",
                       variable=y_freq_sweep_var).pack(anchor=tk.W, pady=2)

        y_freq_sweep_row = ctk.CTkFrame(y_add_frame)
        y_freq_sweep_row.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(y_freq_sweep_row, text="ω start:").pack(side=tk.LEFT, padx=(5, 2))
        y_freq_start_var = tk.DoubleVar(value=1.0)
        ctk.CTkEntry(y_freq_sweep_row, textvariable=y_freq_start_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(y_freq_sweep_row, text="ω end:").pack(side=tk.LEFT, padx=(5, 2))
        y_freq_end_var = tk.DoubleVar(value=5.0)
        ctk.CTkEntry(y_freq_sweep_row, textvariable=y_freq_end_var, width=8).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(y_freq_sweep_row, text="steps:").pack(side=tk.LEFT, padx=(5, 2))
        y_freq_sweep_steps_var = tk.IntVar(value=20)
        ctk.CTkEntry(y_freq_sweep_row, textvariable=y_freq_sweep_steps_var, width=6).pack(side=tk.LEFT, padx=2)

        # Row 4: Add button
        y_button_row = ctk.CTkFrame(y_add_frame)
        y_button_row.pack(fill=tk.X, pady=(5, 0))

        def add_y_term():
            y_terms.append({
                'type': y_type_var.get(),
                'amp': y_amp_var.get(),
                'freq': y_freq_var.get(),
                'phase_sweep': y_phase_sweep_var.get(),
                'phase_start': y_phase_start_var.get(),
                'phase_end': y_phase_end_var.get(),
                'sweep_steps': y_sweep_steps_var.get(),
                'freq_sweep': y_freq_sweep_var.get(),
                'freq_start': y_freq_start_var.get(),
                'freq_end': y_freq_end_var.get(),
                'freq_sweep_steps': y_freq_sweep_steps_var.get()
            })
            refresh_y_display()
            update_preview()

        ctk.CTkButton(y_button_row, text="Add Term to Y Channel", command=add_y_term).pack(fill=tk.X, padx=5)

        # Initialize displays
        refresh_x_display()
        refresh_y_display()

        # Generate button
        def apply_harmonic_sum():
            """Generate pattern from harmonic sums"""
            if not x_terms and not y_terms:
                messagebox.showwarning("No Terms", "Please add at least one term to X or Y channel!")
                return

            # Time array (0 to 2π with high resolution)
            num_points = 1000
            t = np.linspace(0, 2*np.pi, num_points)

            # Check if any term has phase or frequency sweep enabled
            has_phase_sweep = any(term.get('phase_sweep', False) for term in x_terms + y_terms)
            has_freq_sweep = any(term.get('freq_sweep', False) for term in x_terms + y_terms)

            if has_phase_sweep or has_freq_sweep:
                # Find maximum sweep steps among all terms
                max_steps = 1
                for term in x_terms + y_terms:
                    if term.get('phase_sweep', False):
                        max_steps = max(max_steps, term.get('sweep_steps', 1))
                    if term.get('freq_sweep', False):
                        max_steps = max(max_steps, term.get('freq_sweep_steps', 1))

                # Generate frames with phase sweeping
                x_frames = []
                y_frames = []

                for step_idx in range(max_steps):
                    # Calculate X channel for this frame
                    if x_terms:
                        x_frame = np.zeros(num_points)
                        for term in x_terms:
                            # Handle phase sweep
                            if term.get('phase_sweep', False):
                                # Interpolate phase from start to end
                                steps = term['sweep_steps']
                                phase_range = term['phase_end'] - term['phase_start']
                                phase = term['phase_start'] + (phase_range * step_idx / max(steps - 1, 1))
                            else:
                                phase = term.get('phase_start', 0)

                            # Handle frequency sweep
                            if term.get('freq_sweep', False):
                                # Interpolate frequency from start to end
                                steps = term.get('freq_sweep_steps', 20)
                                freq_range = term['freq_end'] - term['freq_start']
                                freq = term['freq_start'] + (freq_range * step_idx / max(steps - 1, 1))
                            else:
                                freq = term.get('freq', 1.0)

                            if term['type'] == 'sin':
                                x_frame += term['amp'] * np.sin(freq * t + phase)
                            else:  # cos
                                x_frame += term['amp'] * np.cos(freq * t + phase)
                        x_frames.append(x_frame)
                    else:
                        x_frames.append(np.zeros(num_points))

                    # Calculate Y channel for this frame
                    if y_terms:
                        y_frame = np.zeros(num_points)
                        for term in y_terms:
                            # Handle phase sweep
                            if term.get('phase_sweep', False):
                                # Interpolate phase from start to end
                                steps = term['sweep_steps']
                                phase_range = term['phase_end'] - term['phase_start']
                                phase = term['phase_start'] + (phase_range * step_idx / max(steps - 1, 1))
                            else:
                                phase = term.get('phase_start', 0)

                            # Handle frequency sweep
                            if term.get('freq_sweep', False):
                                # Interpolate frequency from start to end
                                steps = term.get('freq_sweep_steps', 20)
                                freq_range = term['freq_end'] - term['freq_start']
                                freq = term['freq_start'] + (freq_range * step_idx / max(steps - 1, 1))
                            else:
                                freq = term.get('freq', 1.0)

                            if term['type'] == 'sin':
                                y_frame += term['amp'] * np.sin(freq * t + phase)
                            else:  # cos
                                y_frame += term['amp'] * np.cos(freq * t + phase)
                        y_frames.append(y_frame)
                    else:
                        y_frames.append(np.zeros(num_points))

                # Concatenate all frames
                x_data = np.concatenate(x_frames)
                y_data = np.concatenate(y_frames)

                # Build status message
                sweep_types = []
                if has_phase_sweep:
                    sweep_types.append("phase")
                if has_freq_sweep:
                    sweep_types.append("frequency")
                sweep_str = " and ".join(sweep_types)
                status_msg = f"Generated harmonic sum with {sweep_str} sweep ({max_steps} frames)"
            else:
                # No phase sweep - generate single frame
                # Calculate X channel
                if x_terms:
                    x_data = np.zeros(num_points)
                    for term in x_terms:
                        phase = term.get('phase_start', 0)
                        if term['type'] == 'sin':
                            x_data += term['amp'] * np.sin(term['freq'] * t + phase)
                        else:  # cos
                            x_data += term['amp'] * np.cos(term['freq'] * t + phase)
                else:
                    x_data = np.zeros(num_points)

                # Calculate Y channel
                if y_terms:
                    y_data = np.zeros(num_points)
                    for term in y_terms:
                        phase = term.get('phase_start', 0)
                        if term['type'] == 'sin':
                            y_data += term['amp'] * np.sin(term['freq'] * t + phase)
                        else:  # cos
                            y_data += term['amp'] * np.cos(term['freq'] * t + phase)
                else:
                    y_data = np.zeros(num_points)

                status_msg = f"Generated harmonic sum (N={len(x_terms)}, M={len(y_terms)})"

            # Set as current data
            self.x_data = x_data
            self.y_data = y_data
            self.data_info_label.config(text=f"Points: {len(x_data)}")
            self.update_display()
            self.status_label.config(text=status_msg)

            # Apply parameters and generate audio
            self.apply_parameters()

        # Button frame
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ctk.CTkButton(button_frame, text="Apply & Generate", command=apply_harmonic_sum).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=5)

    def open_archimedean_spiral(self):
        """Open dialog to create Archimedean spiral pattern"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Archimedean Spiral")
        dialog.geometry("600x600")

        # Instructions
        instruction_frame = ctk.CTkFrame(dialog)
        instruction_frame.pack(pady=10, padx=10, fill=tk.X)
        ctk.CTkLabel(instruction_frame, text="Create Archimedean Spiral Pattern",
                 font=('Arial', 10, 'bold')).pack()
        ctk.CTkLabel(instruction_frame, text="X(t) = a·t·sin(b·t)    Y(t) = a·t·cos(b·t)",
                 font=('Arial', 8), text_color='gray').pack()

        # Main container with two columns
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # X Channel (Left)
        x_frame = ctk.CTkFrame(main_frame)
        x_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(x_frame, text="X Channel (Left)", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        x_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # X parameters
        x_params_frame = ctk.CTkFrame(x_frame)
        x_params_frame.pack(fill=tk.X, pady=10)

        ctk.CTkLabel(x_params_frame, text="Parameter 'a' (amplitude):").grid(row=0, column=0, sticky=tk.W, pady=5)
        x_a_var = tk.DoubleVar(value=1.0)
        x_a_scale = ctk.CTkSlider(x_params_frame, from_=0.1, to=5.0, variable=x_a_var)
        x_a_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        x_a_entry = ctk.CTkEntry(x_params_frame, textvariable=x_a_var, width=10)
        x_a_entry.grid(row=0, column=2, padx=5)

        ctk.CTkLabel(x_params_frame, text="Parameter 'b' (frequency):").grid(row=1, column=0, sticky=tk.W, pady=5)
        x_b_var = tk.DoubleVar(value=1.0)
        x_b_scale = ctk.CTkSlider(x_params_frame, from_=0.1, to=10.0, variable=x_b_var)
        x_b_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        x_b_entry = ctk.CTkEntry(x_params_frame, textvariable=x_b_var, width=10)
        x_b_entry.grid(row=1, column=2, padx=5)

        x_params_frame.columnconfigure(1, weight=1)

        # X frequency sweep option
        x_sweep_enabled_var = tk.BooleanVar(value=False)
        x_sweep_check = ctk.CTkCheckBox(x_frame, text="Enable Frequency Sweep",
                                        variable=x_sweep_enabled_var)
        x_sweep_check.pack(anchor=tk.W, pady=5)

        x_sweep_frame = ctk.CTkFrame(x_frame)
        x_sweep_frame.pack(fill=tk.X, pady=5)

        ctk.CTkLabel(x_sweep_frame, text="Start Freq:").grid(row=0, column=0, sticky=tk.W, padx=5)
        x_sweep_start_var = tk.DoubleVar(value=0.1)
        ctk.CTkEntry(x_sweep_frame, textvariable=x_sweep_start_var, width=8).grid(row=0, column=1, padx=5)

        ctk.CTkLabel(x_sweep_frame, text="End Freq:").grid(row=0, column=2, sticky=tk.W, padx=5)
        x_sweep_end_var = tk.DoubleVar(value=10.0)
        ctk.CTkEntry(x_sweep_frame, textvariable=x_sweep_end_var, width=8).grid(row=0, column=3, padx=5)

        # X formula display
        x_formula_label = ctk.CTkLabel(x_frame, text="", font=('Arial', 9, 'italic'), text_color='blue')
        x_formula_label.pack(pady=10)

        # Y Channel (Right)
        y_frame = ctk.CTkFrame(main_frame)
        y_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(y_frame, text="Y Channel (Right)", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        y_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Y parameters
        y_params_frame = ctk.CTkFrame(y_frame)
        y_params_frame.pack(fill=tk.X, pady=10)

        ctk.CTkLabel(y_params_frame, text="Parameter 'a' (amplitude):").grid(row=0, column=0, sticky=tk.W, pady=5)
        y_a_var = tk.DoubleVar(value=1.0)
        y_a_scale = ctk.CTkSlider(y_params_frame, from_=0.1, to=5.0, variable=y_a_var)
        y_a_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        y_a_entry = ctk.CTkEntry(y_params_frame, textvariable=y_a_var, width=10)
        y_a_entry.grid(row=0, column=2, padx=5)

        ctk.CTkLabel(y_params_frame, text="Parameter 'b' (frequency):").grid(row=1, column=0, sticky=tk.W, pady=5)
        y_b_var = tk.DoubleVar(value=1.0)
        y_b_scale = ctk.CTkSlider(y_params_frame, from_=0.1, to=10.0, variable=y_b_var)
        y_b_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        y_b_entry = ctk.CTkEntry(y_params_frame, textvariable=y_b_var, width=10)
        y_b_entry.grid(row=1, column=2, padx=5)

        y_params_frame.columnconfigure(1, weight=1)

        # Y frequency sweep option
        y_sweep_enabled_var = tk.BooleanVar(value=False)
        y_sweep_check = ctk.CTkCheckBox(y_frame, text="Enable Frequency Sweep",
                                        variable=y_sweep_enabled_var)
        y_sweep_check.pack(anchor=tk.W, pady=5)

        y_sweep_frame = ctk.CTkFrame(y_frame)
        y_sweep_frame.pack(fill=tk.X, pady=5)

        ctk.CTkLabel(y_sweep_frame, text="Start Freq:").grid(row=0, column=0, sticky=tk.W, padx=5)
        y_sweep_start_var = tk.DoubleVar(value=0.1)
        ctk.CTkEntry(y_sweep_frame, textvariable=y_sweep_start_var, width=8).grid(row=0, column=1, padx=5)

        ctk.CTkLabel(y_sweep_frame, text="End Freq:").grid(row=0, column=2, sticky=tk.W, padx=5)
        y_sweep_end_var = tk.DoubleVar(value=10.0)
        ctk.CTkEntry(y_sweep_frame, textvariable=y_sweep_end_var, width=8).grid(row=0, column=3, padx=5)

        # Y formula display
        y_formula_label = ctk.CTkLabel(y_frame, text="", font=('Arial', 9, 'italic'), text_color='blue')
        y_formula_label.pack(pady=10)

        # Chaser Effect Options
        chaser_frame = ctk.CTkFrame(dialog)
        chaser_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(chaser_frame, text="Chaser Effect (Optional)", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        chaser_frame.pack(fill=tk.X, padx=10, pady=10)

        # Enable chaser checkbox
        chaser_enabled_var = tk.BooleanVar(value=False)
        chaser_check = ctk.CTkCheckBox(chaser_frame, text="Enable Chaser Effect",
                                       variable=chaser_enabled_var)
        chaser_check.pack(anchor=tk.W, pady=5)

        # Chaser parameters frame
        chaser_params_frame = ctk.CTkFrame(chaser_frame)
        chaser_params_frame.pack(fill=tk.X, pady=5)

        # Trail length parameter (number of points)
        ctk.CTkLabel(chaser_params_frame, text="Trail Length (points):").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        trail_length_var = tk.IntVar(value=100)
        trail_scale = ctk.CTkSlider(chaser_params_frame, from_=5, to=500, variable=trail_length_var)
        trail_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        trail_entry = ctk.CTkEntry(chaser_params_frame, textvariable=trail_length_var, width=10)
        trail_entry.grid(row=0, column=2, padx=5)

        # Point repetitions parameter (controls speed)
        ctk.CTkLabel(chaser_params_frame, text="Point Repetitions (speed):").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        point_reps_var = tk.IntVar(value=10)
        reps_scale = ctk.CTkSlider(chaser_params_frame, from_=1, to=100, variable=point_reps_var)
        reps_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        reps_entry = ctk.CTkEntry(chaser_params_frame, textvariable=point_reps_var, width=10)
        reps_entry.grid(row=1, column=2, padx=5)

        chaser_params_frame.columnconfigure(1, weight=1)

        # Info label
        chaser_info = ctk.CTkLabel(chaser_frame,
                               text="Higher repetitions = slower movement. Each point repeated N times.",
                               font=('Arial', 8, 'italic'), text_color='gray')
        chaser_info.pack(pady=5)

        def update_formula_labels():
            """Update the formula display labels"""
            x_a = x_a_var.get()
            x_b = x_b_var.get()
            y_a = y_a_var.get()
            y_b = y_b_var.get()
            x_formula_label.config(text=f"x(t) = {x_a:.2f}·t·sin({x_b:.2f}·t)")
            y_formula_label.config(text=f"y(t) = {y_a:.2f}·t·cos({y_b:.2f}·t)")

        def update_preview(*args):
            """Update oscilloscope display with current spiral parameters"""
            # Time array (0 to 4π with high resolution for nice spiral)
            num_points = 2000
            t = np.linspace(0, 4*np.pi, num_points)
            t_max = t[-1]

            # Calculate X channel: x = a*t*sin(b*t) with optional frequency sweep
            x_a = x_a_var.get()
            if x_sweep_enabled_var.get():
                # Frequency sweep: compute proper phase from integrated frequency
                # f(t) = f_start + (f_end - f_start) * t / t_max
                # phase(t) = integral of f(t) = f_start*t + (f_end - f_start)*t^2/(2*t_max)
                f_start = x_sweep_start_var.get()
                f_end = x_sweep_end_var.get()
                phase = f_start * t + (f_end - f_start) * t**2 / (2 * t_max)
                x_data = x_a * t * np.sin(phase)
                x_freq_label = f"sweep {f_start:.2f}->{f_end:.2f}"
            else:
                x_b = x_b_var.get()
                x_data = x_a * t * np.sin(x_b * t)
                x_freq_label = f"{x_b:.2f}"

            # Calculate Y channel: y = a*t*cos(b*t) with optional frequency sweep
            y_a = y_a_var.get()
            if y_sweep_enabled_var.get():
                # Frequency sweep: compute proper phase from integrated frequency
                f_start = y_sweep_start_var.get()
                f_end = y_sweep_end_var.get()
                phase = f_start * t + (f_end - f_start) * t**2 / (2 * t_max)
                y_data = y_a * t * np.cos(phase)
                y_freq_label = f"sweep {f_start:.2f}->{f_end:.2f}"
            else:
                y_b = y_b_var.get()
                y_data = y_a * t * np.cos(y_b * t)
                y_freq_label = f"{y_b:.2f}"

            # Update display
            self.x_data = x_data
            self.y_data = y_data
            self.data_info_label.config(text=f"Points: {len(x_data)}")
            self.update_display()
            self.status_label.config(text=f"Preview: Archimedean Spiral (X: a={x_a:.2f}, b={x_freq_label}; Y: a={y_a:.2f}, b={y_freq_label})")
            update_formula_labels()

        # Bind parameter changes to update preview
        x_a_var.trace('w', update_preview)
        x_b_var.trace('w', update_preview)
        y_a_var.trace('w', update_preview)
        y_b_var.trace('w', update_preview)
        x_sweep_enabled_var.trace('w', update_preview)
        x_sweep_start_var.trace('w', update_preview)
        x_sweep_end_var.trace('w', update_preview)
        y_sweep_enabled_var.trace('w', update_preview)
        y_sweep_start_var.trace('w', update_preview)
        y_sweep_end_var.trace('w', update_preview)

        # Initial preview
        update_preview()

        def apply_spiral():
            """Generate final spiral pattern and apply audio"""
            # Time array (0 to 4π with high resolution)
            num_points = 2000
            t = np.linspace(0, 4*np.pi, num_points)
            t_max = t[-1]

            # Calculate X channel with optional frequency sweep
            x_a = x_a_var.get()
            if x_sweep_enabled_var.get():
                # Frequency sweep: compute proper phase from integrated frequency
                f_start = x_sweep_start_var.get()
                f_end = x_sweep_end_var.get()
                phase = f_start * t + (f_end - f_start) * t**2 / (2 * t_max)
                x_full = x_a * t * np.sin(phase)
                x_freq_info = f"sweep {f_start:.2f}->{f_end:.2f}"
            else:
                x_b = x_b_var.get()
                x_full = x_a * t * np.sin(x_b * t)
                x_freq_info = f"{x_b:.2f}"

            # Calculate Y channel with optional frequency sweep
            y_a = y_a_var.get()
            if y_sweep_enabled_var.get():
                # Frequency sweep: compute proper phase from integrated frequency
                f_start = y_sweep_start_var.get()
                f_end = y_sweep_end_var.get()
                phase = f_start * t + (f_end - f_start) * t**2 / (2 * t_max)
                y_full = y_a * t * np.cos(phase)
                y_freq_info = f"sweep {f_start:.2f}->{f_end:.2f}"
            else:
                y_b = y_b_var.get()
                y_full = y_a * t * np.cos(y_b * t)
                y_freq_info = f"{y_b:.2f}"

            # Check if chaser effect is enabled
            if chaser_enabled_var.get():
                # Chaser effect: Create smooth sliding window that moves along the spiral
                # Each position shows a continuous segment of the path
                trail_length = int(trail_length_var.get())
                point_reps = int(point_reps_var.get())

                # Ensure trail length is valid
                if trail_length < 1:
                    trail_length = 1
                if trail_length > num_points:
                    trail_length = num_points
                if point_reps < 1:
                    point_reps = 1

                # Create sliding windows that move along the spiral
                # Each window is a continuous segment of trail_length points
                x_segments = []
                y_segments = []

                # Calculate step size for smooth movement (advance 1 point at a time)
                max_start = num_points - trail_length

                for start_idx in range(0, max_start + 1):
                    end_idx = start_idx + trail_length

                    # Extract continuous segment
                    x_seg = x_full[start_idx:end_idx]
                    y_seg = y_full[start_idx:end_idx]

                    # Repeat this segment position point_reps times (controls speed)
                    for _ in range(point_reps):
                        x_segments.append(x_seg)
                        y_segments.append(y_seg)

                # Concatenate all segments to create smooth chase animation
                x_data = np.concatenate(x_segments)
                y_data = np.concatenate(y_segments)

                num_positions = max_start + 1
                status_msg = f"Generated Archimedean Spiral with Chaser (X: a={x_a:.2f}, b={x_freq_info}; Y: a={y_a:.2f}, b={y_freq_info}; trail={trail_length}, reps={point_reps})"
            else:
                # No chaser effect: use full spiral
                x_data = x_full
                y_data = y_full
                status_msg = f"Generated Archimedean Spiral (X: a={x_a:.2f}, b={x_freq_info}; Y: a={y_a:.2f}, b={y_freq_info})"

            # Set as current data
            self.x_data = x_data
            self.y_data = y_data
            self.data_info_label.config(text=f"Points: {len(x_data)}")
            self.update_display()
            self.status_label.config(text=status_msg)

            # Apply parameters and generate audio
            self.apply_parameters()

        # Button frame
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ctk.CTkButton(button_frame, text="Apply & Generate", command=apply_spiral).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=5)

    def open_sound_pad(self):
        """Open sound pad interface for live performance and sequencing"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Sound Pad")
        dialog.geometry("900x750")

        # Sound pad state
        pad_notes = {}  # Maps (row, col) to frequency
        # 3 independent sequence timelines for layered drum patterns
        # Changed to dictionaries mapping time_slot -> (pad_idx, duration)
        track1_sequence = {}  # Track 1: Drum/Kick line
        track2_sequence = {}  # Track 2: Snare line
        track3_sequence = {}  # Track 3: Beat/HiHat line
        current_track = tk.IntVar(value=1)  # Which track is being recorded
        is_playing_live = False
        current_instrument = tk.StringVar(value="sine")
        current_warp = tk.DoubleVar(value=0.0)

        # Time slot / step sequencer settings
        tempo_bpm = tk.IntVar(value=120)  # BPM for playback
        num_steps = tk.IntVar(value=16)  # Number of time slots per pattern
        current_step = tk.IntVar(value=0)  # Currently selected time slot for recording
        position_offset = tk.DoubleVar(value=0.0)  # Fine-tune position (0.0 to 0.99)
        note_duration = tk.DoubleVar(value=0.25)  # Duration of each note in seconds

        # Note frequencies (chromatic scale starting from C3)
        base_notes = [130.81, 138.59, 146.83, 155.56,  # C3, C#3, D3, D#3
                      164.81, 174.61, 185.00, 196.00,  # E3, F3, F#3, G3
                      207.65, 220.00, 233.08, 246.94,  # G#3, A3, A#3, B3
                      261.63, 277.18, 293.66, 311.13]  # C4, C#4, D4, D#4

        # Main container
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Pad grid
        left_panel = ctk.CTkFrame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ctk.CTkLabel(left_panel, text="Sound Pad Grid", font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        # Create 4x4 grid of pads
        pad_grid = ctk.CTkFrame(left_panel)
        pad_grid.pack(fill=tk.BOTH, expand=True)

        pad_buttons = {}

        def generate_pad_sound(note_freq, instrument, warp, duration=0.5):
            """Generate sound for a pad"""
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))

            # Drums ignore note_freq and warp
            if instrument in ["kick", "bass", "snare", "hihat", "tom"]:
                if instrument == "kick":
                    # Kick drum - pitch sweep down with fast decay
                    freq_sweep = np.linspace(150, 40, len(t))
                    phase = np.cumsum(freq_sweep) / sample_rate
                    signal = np.sin(2 * np.pi * phase)
                    envelope = np.exp(-8 * t)
                    signal = signal * envelope
                elif instrument == "bass":
                    # Bass drum - lower and slower than kick, deep boom
                    freq_sweep = np.linspace(80, 30, len(t))
                    phase = np.cumsum(freq_sweep) / sample_rate
                    signal = np.sin(2 * np.pi * phase)
                    envelope = np.exp(-5 * t)  # Slower decay than kick
                    signal = signal * envelope
                elif instrument == "snare":
                    # Snare - mix of noise and tone
                    tone_freq = 200
                    tone = np.sin(2 * np.pi * tone_freq * t)
                    noise = np.random.uniform(-1, 1, len(t))
                    signal = 0.3 * tone + 0.7 * noise
                    envelope = np.exp(-15 * t)
                    signal = signal * envelope
                elif instrument == "hihat":
                    # Hi-hat - filtered noise with very fast decay
                    noise = np.random.uniform(-1, 1, len(t))
                    # Simple high-pass filter effect
                    signal = noise - np.concatenate([[0], noise[:-1]])
                    envelope = np.exp(-25 * t)
                    signal = signal * envelope
                elif instrument == "tom":
                    # Tom drum - pitch sweep with medium decay
                    freq_sweep = np.linspace(note_freq * 1.5, note_freq * 0.7, len(t))
                    phase = np.cumsum(freq_sweep) / sample_rate
                    signal = np.sin(2 * np.pi * phase)
                    envelope = np.exp(-6 * t)
                    signal = signal * envelope
            else:
                # Apply warp effect by modulating frequency (for non-drum instruments)
                if abs(warp) > 0.01:
                    warp_freq = 5 + abs(warp) * 20  # Warp modulation frequency
                    warp_depth = abs(warp) * note_freq * 0.3  # Warp depth
                    freq_mod = note_freq + warp_depth * np.sin(2 * np.pi * warp_freq * t)
                    # Calculate instantaneous phase for warped sound
                    phase = np.cumsum(freq_mod) / sample_rate
                else:
                    # No warp - use linear phase
                    phase = note_freq * t

                if instrument == "sine":
                    # Pure sine wave
                    signal = np.sin(2 * np.pi * phase)
                elif instrument == "square":
                    # Square wave - use modulo approach for proper warping
                    phase_wrapped = phase % 1.0
                    signal = np.where(phase_wrapped < 0.5, 1.0, -1.0)
                elif instrument == "saw":
                    # Sawtooth wave
                    phase_wrapped = phase % 1.0
                    signal = 2 * phase_wrapped - 1
                elif instrument == "triangle":
                    # Triangle wave
                    phase_wrapped = phase % 1.0
                    signal = np.where(phase_wrapped < 0.5,
                                    4 * phase_wrapped - 1,
                                    3 - 4 * phase_wrapped)
                elif instrument == "piano":
                    # Piano-like sound (sine with harmonics and envelope)
                    signal = (np.sin(2 * np.pi * phase) +
                             0.5 * np.sin(2 * np.pi * phase * 2) +
                             0.25 * np.sin(2 * np.pi * phase * 3))
                    envelope = np.exp(-3 * t)
                    signal = signal * envelope
                elif instrument == "organ":
                    # Organ-like sound (multiple harmonics)
                    signal = (np.sin(2 * np.pi * phase) +
                             0.8 * np.sin(2 * np.pi * phase * 2) +
                             0.6 * np.sin(2 * np.pi * phase * 3) +
                             0.4 * np.sin(2 * np.pi * phase * 4))
                elif instrument == "bell":
                    # Bell-like sound (inharmonic partials)
                    signal = (np.sin(2 * np.pi * phase) +
                             0.7 * np.sin(2 * np.pi * phase * 2.76) +
                             0.5 * np.sin(2 * np.pi * phase * 5.40))
                    envelope = np.exp(-4 * t)
                    signal = signal * envelope
                else:
                    signal = np.sin(2 * np.pi * phase)

            # Normalize
            signal = signal / (np.max(np.abs(signal)) + 1e-10)

            # Create Lissajous pattern (X and Y from same signal with phase shift)
            x = signal
            # Generate Y with phase shift
            if instrument in ["kick", "bass", "snare", "hihat", "tom"]:
                # For drums, create phase-shifted version of same signal
                y = np.concatenate([signal[len(signal)//4:], signal[:len(signal)//4]])
            elif abs(warp) > 0.01:
                y = np.sin(2 * np.pi * phase + np.pi/2)
            else:
                y = np.sin(2 * np.pi * note_freq * t + np.pi/2)

            return x, y

        def play_pad(row, col):
            """Play sound when pad is pressed"""
            pad_idx = row * 4 + col
            note_freq = base_notes[pad_idx]

            # Generate pattern
            x, y = generate_pad_sound(note_freq, current_instrument.get(),
                                     current_warp.get(), duration=0.5)

            # Update display
            self.x_data = x
            self.y_data = y
            self.data_info_label.config(text=f"Points: {len(x)}")
            self.update_display()
            self.status_label.config(text=f"Pad {pad_idx+1} - {note_freq:.1f} Hz")

            # Play sound once (not looping) using sounddevice directly
            stereo_audio = np.column_stack([x, y]).astype(np.float32)
            sd.play(stereo_audio, 44100, blocking=False)

            # Visual feedback
            button = pad_buttons[(row, col)]
            original_color = button['background'] if 'background' in button.keys() else 'SystemButtonFace'
            button.config(bg='#4CAF50')
            dialog.after(500, lambda: button.config(bg=original_color))

        # Create pad buttons
        for row in range(4):
            for col in range(4):
                pad_idx = row * 4 + col
                note_freq = base_notes[pad_idx]
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                note_name = note_names[pad_idx % 12] + str(3 + pad_idx // 12)

                btn = tk.Button(pad_grid, text=f"{note_name}\n{note_freq:.0f}Hz",
                               font=('Arial', 10, 'bold'),
                               command=lambda r=row, c=col: play_pad(r, c),
                               relief=tk.RAISED,
                               bd=3,
                               bg='#E0E0E0',
                               activebackground='#4CAF50')
                btn.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                pad_buttons[(row, col)] = btn
                pad_notes[(row, col)] = note_freq

        # Make grid cells expand
        for i in range(4):
            pad_grid.rowconfigure(i, weight=1)
            pad_grid.columnconfigure(i, weight=1)

        # Right panel: Controls
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Instrument selection
        instrument_frame = ctk.CTkFrame(right_panel)
        instrument_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(instrument_frame, text="Instrument", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        instrument_frame.pack(fill=tk.X, pady=(0, 10))

        instruments = [("Sine Wave", "sine"),
                      ("Square Wave", "square"),
                      ("Sawtooth", "saw"),
                      ("Triangle", "triangle"),
                      ("Piano", "piano"),
                      ("Organ", "organ"),
                      ("Bell", "bell"),
                      ("Kick Drum", "kick"),
                      ("Bass Drum", "bass"),
                      ("Snare Drum", "snare"),
                      ("Hi-Hat", "hihat"),
                      ("Tom Drum", "tom")]

        for name, value in instruments:
            ctk.CTkRadioButton(instrument_frame, text=name,
                          variable=current_instrument,
                          value=value).pack(anchor=tk.W, pady=2)

        # Warp control
        warp_frame = ctk.CTkFrame(right_panel)
        warp_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(warp_frame, text="Warp Effect", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        warp_frame.pack(fill=tk.X, pady=(0, 10))

        warp_label_frame = ctk.CTkFrame(warp_frame)
        warp_label_frame.pack(fill=tk.X)
        ctk.CTkLabel(warp_label_frame, text="Warp Amount:").pack(side=tk.LEFT)
        warp_value_label = ctk.CTkLabel(warp_label_frame, text="0.0", font=('Arial', 9, 'bold'))
        warp_value_label.pack(side=tk.RIGHT)

        warp_scale = ctk.CTkSlider(warp_frame, from_=-1.0, to=1.0,
                              variable=current_warp,
                              command=lambda v: warp_value_label.config(text=f"{current_warp.get():.2f}"))
        warp_scale.pack(fill=tk.X, pady=5)

        ctk.CTkLabel(warp_frame, text="Adds frequency modulation",
                 font=('Arial', 8, 'italic'), text_color='gray').pack()

        # Sequencer - 3 Track System
        sequencer_frame = ctk.CTkFrame(right_panel)
        sequencer_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(sequencer_frame, text="3-Track Sequencer", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        sequencer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ctk.CTkLabel(sequencer_frame, text="Layer drum patterns across 3 tracks:",
                 font=('Arial', 9, 'bold')).pack(anchor=tk.W)

        # Track selector
        track_select_frame = ctk.CTkFrame(sequencer_frame)
        track_select_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(track_select_frame, text="Record to:", font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=5)

        def on_track_change(*args):
            """Update step grid when track changes"""
            update_step_grid()

        current_track.trace_add('write', on_track_change)

        ctk.CTkRadioButton(track_select_frame, text="Track 1 (Drum)",
                       variable=current_track, value=1).pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(track_select_frame, text="Track 2 (Snare)",
                       variable=current_track, value=2).pack(side=tk.LEFT, padx=5)
        ctk.CTkRadioButton(track_select_frame, text="Track 3 (Beat)",
                       variable=current_track, value=3).pack(side=tk.LEFT, padx=5)

        # Tempo and timing controls
        tempo_frame = ctk.CTkFrame(sequencer_frame)
        tempo_frame.pack(fill=tk.X, pady=5)

        ctk.CTkLabel(tempo_frame, text="BPM:", font=('Arial', 8, 'bold')).grid(row=0, column=0, padx=5, sticky='w')
        tempo_spinbox = tk.Spinbox(tempo_frame, from_=60, to=200, textvariable=tempo_bpm, width=6)
        tempo_spinbox.grid(row=0, column=1, padx=5)

        ctk.CTkLabel(tempo_frame, text="Steps:", font=('Arial', 8, 'bold')).grid(row=0, column=2, padx=5, sticky='w')
        steps_spinbox = tk.Spinbox(tempo_frame, from_=8, to=32, textvariable=num_steps, width=6,
                                    command=lambda: (update_sequence_displays(), update_step_grid()))
        steps_spinbox.grid(row=0, column=3, padx=5)

        ctk.CTkLabel(tempo_frame, text="Note Duration (s):", font=('Arial', 8, 'bold')).grid(row=0, column=4, padx=5, sticky='w')
        duration_spinbox = tk.Spinbox(tempo_frame, from_=0.1, to=2.0, increment=0.05,
                                      textvariable=note_duration, width=6, format="%.2f")
        duration_spinbox.grid(row=0, column=5, padx=5)

        # Position offset control (second row)
        ctk.CTkLabel(tempo_frame, text="Position Offset:", font=('Arial', 8, 'bold')).grid(row=1, column=0, padx=5, sticky='w', pady=(5,0))
        offset_label = ctk.CTkLabel(tempo_frame, text="0.00", font=('Arial', 8))
        offset_label.grid(row=1, column=1, padx=5, sticky='w', pady=(5,0))

        def update_offset_label(val):
            offset_label.config(text=f"{position_offset.get():.2f}")

        offset_scale = ctk.CTkSlider(tempo_frame, from_=0.0, to=0.99,
                               variable=position_offset, command=update_offset_label)
        offset_scale.grid(row=1, column=2, columnspan=3, padx=5, sticky='ew', pady=(5,0))

        ctk.CTkLabel(tempo_frame, text="(Fine-tune note position)", font=('Arial', 7, 'italic'),
                 text_color='gray').grid(row=1, column=5, padx=5, sticky='w', pady=(5,0))

        # Visual step grid for selecting time slots
        step_grid_frame = ctk.CTkFrame(sequencer_frame)
        step_grid_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(step_grid_frame, text="Time Slots (Click to select)", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        step_grid_frame.pack(fill=tk.X, pady=5)

        step_buttons = {}

        def update_step_grid():
            """Update the visual step grid to show current state"""
            for step_num, btn in step_buttons.items():
                # Check if this step has a note in the current track
                track_num = current_track.get()
                track_seq = None
                if track_num == 1:
                    track_seq = track1_sequence
                elif track_num == 2:
                    track_seq = track2_sequence
                elif track_num == 3:
                    track_seq = track3_sequence

                # Check if any position starting with this step exists (e.g., 4.0, 4.25, 4.5)
                has_note = False
                if track_seq:
                    for pos in track_seq.keys():
                        if int(pos) == step_num:
                            has_note = True
                            break

                # Update button appearance
                if step_num == current_step.get():
                    # Selected step
                    btn.config(relief=tk.SUNKEN, bg='#2196F3', fg='white')
                elif has_note:
                    # Step has a note (at any fractional position)
                    btn.config(relief=tk.RAISED, bg='#4CAF50', fg='white')
                else:
                    # Empty step
                    btn.config(relief=tk.RAISED, bg='#E0E0E0', fg='black')

        def select_step(step_num):
            """Select a time slot for recording"""
            current_step.set(step_num)
            update_step_grid()

        # Create step buttons grid
        steps_grid_container = ctk.CTkFrame(step_grid_frame)
        steps_grid_container.pack(fill=tk.X)

        for i in range(16):  # Default 16 steps, will update dynamically
            btn = tk.Button(steps_grid_container, text=str(i+1), width=3, height=1,
                          font=('Arial', 8), command=lambda step=i: select_step(step))
            btn.grid(row=i // 8, column=i % 8, padx=2, pady=2)
            step_buttons[i] = btn

        update_step_grid()

        # 3 Sequence displays
        tracks_display_frame = ctk.CTkFrame(sequencer_frame)
        tracks_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Track 1
        track1_label = ctk.CTkLabel(tracks_display_frame, text="Track 1 (Drum):",
                                font=('Courier', 8, 'bold'), text_color='#D32F2F')
        track1_label.grid(row=0, column=0, sticky='w', pady=(0, 2))
        track1_text = tk.Text(tracks_display_frame, height=3, width=40, font=('Courier', 7))
        track1_text.grid(row=1, column=0, sticky='ew', pady=(0, 5))
        track1_text.insert('1.0', "Empty")
        track1_text.config(state=tk.DISABLED)

        # Track 2
        track2_label = ctk.CTkLabel(tracks_display_frame, text="Track 2 (Snare):",
                                font=('Courier', 8, 'bold'), text_color='#1976D2')
        track2_label.grid(row=2, column=0, sticky='w', pady=(0, 2))
        track2_text = tk.Text(tracks_display_frame, height=3, width=40, font=('Courier', 7))
        track2_text.grid(row=3, column=0, sticky='ew', pady=(0, 5))
        track2_text.insert('1.0', "Empty")
        track2_text.config(state=tk.DISABLED)

        # Track 3
        track3_label = ctk.CTkLabel(tracks_display_frame, text="Track 3 (Beat):",
                                font=('Courier', 8, 'bold'), text_color='#388E3C')
        track3_label.grid(row=4, column=0, sticky='w', pady=(0, 2))
        track3_text = tk.Text(tracks_display_frame, height=3, width=40, font=('Courier', 7))
        track3_text.grid(row=5, column=0, sticky='ew', pady=(0, 5))
        track3_text.insert('1.0', "Empty")
        track3_text.config(state=tk.DISABLED)

        tracks_display_frame.columnconfigure(0, weight=1)

        def update_sequence_displays():
            """Update all 3 track displays with time slot visualization"""
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

            # Update Track 1
            track1_text.config(state=tk.NORMAL)
            track1_text.delete('1.0', tk.END)
            if not track1_sequence:
                track1_text.insert('1.0', "Empty")
            else:
                # Show all notes with their positions sorted by time
                sorted_positions = sorted(track1_sequence.keys())
                notes_list = []
                for pos in sorted_positions:
                    pad_idx, _ = track1_sequence[pos]
                    note = f"{note_names[pad_idx % 12]}{3 + pad_idx // 12}"
                    # Show position with offset if fractional
                    if pos % 1 == 0:  # Integer position
                        notes_list.append(f"[{int(pos)+1}:{note}]")
                    else:  # Fractional position
                        notes_list.append(f"[{pos+1:.2f}:{note}]")
                track1_text.insert('1.0', " ".join(notes_list))
            track1_text.config(state=tk.DISABLED)

            # Update Track 2
            track2_text.config(state=tk.NORMAL)
            track2_text.delete('1.0', tk.END)
            if not track2_sequence:
                track2_text.insert('1.0', "Empty")
            else:
                sorted_positions = sorted(track2_sequence.keys())
                notes_list = []
                for pos in sorted_positions:
                    pad_idx, _ = track2_sequence[pos]
                    note = f"{note_names[pad_idx % 12]}{3 + pad_idx // 12}"
                    if pos % 1 == 0:
                        notes_list.append(f"[{int(pos)+1}:{note}]")
                    else:
                        notes_list.append(f"[{pos+1:.2f}:{note}]")
                track2_text.insert('1.0', " ".join(notes_list))
            track2_text.config(state=tk.DISABLED)

            # Update Track 3
            track3_text.config(state=tk.NORMAL)
            track3_text.delete('1.0', tk.END)
            if not track3_sequence:
                track3_text.insert('1.0', "Empty")
            else:
                sorted_positions = sorted(track3_sequence.keys())
                notes_list = []
                for pos in sorted_positions:
                    pad_idx, _ = track3_sequence[pos]
                    note = f"{note_names[pad_idx % 12]}{3 + pad_idx // 12}"
                    if pos % 1 == 0:
                        notes_list.append(f"[{int(pos)+1}:{note}]")
                    else:
                        notes_list.append(f"[{pos+1:.2f}:{note}]")
                track3_text.insert('1.0', " ".join(notes_list))
            track3_text.config(state=tk.DISABLED)

        def add_to_sequence(row, col):
            """Add pad to current track sequence at the selected time slot with position offset"""
            pad_idx = row * 4 + col
            duration = note_duration.get()
            step = current_step.get()
            offset = position_offset.get()

            # Calculate fractional position (e.g., step 4 + offset 0.5 = position 4.5)
            position = step + offset

            track_num = current_track.get()
            if track_num == 1:
                track1_sequence[position] = (pad_idx, duration)
            elif track_num == 2:
                track2_sequence[position] = (pad_idx, duration)
            elif track_num == 3:
                track3_sequence[position] = (pad_idx, duration)

            update_sequence_displays()
            update_step_grid()

        def clear_current_step():
            """Remove note from current step in selected track"""
            step = current_step.get()
            offset = position_offset.get()
            position = step + offset

            track_num = current_track.get()
            track_seq = None
            if track_num == 1:
                track_seq = track1_sequence
            elif track_num == 2:
                track_seq = track2_sequence
            elif track_num == 3:
                track_seq = track3_sequence

            # Remove the note at this position if it exists
            if track_seq is not None and position in track_seq:
                del track_seq[position]
                update_sequence_displays()
                update_step_grid()

        def clear_current_track():
            """Clear the currently selected track"""
            track_num = current_track.get()
            if track_num == 1:
                track1_sequence.clear()
            elif track_num == 2:
                track2_sequence.clear()
            elif track_num == 3:
                track3_sequence.clear()
            update_sequence_displays()

        def clear_all_tracks():
            """Clear all 3 tracks"""
            track1_sequence.clear()
            track2_sequence.clear()
            track3_sequence.clear()
            update_sequence_displays()

        def merge_and_apply():
            """Merge all 3 tracks together and apply to main oscilloscope with time-based positioning"""
            if not (track1_sequence or track2_sequence or track3_sequence):
                messagebox.showwarning("Empty Tracks", "All tracks are empty. Add notes first.")
                return

            sample_rate = 44100
            bpm = tempo_bpm.get()
            steps = num_steps.get()

            # Calculate duration of one step based on BPM
            # At 120 BPM: 60/120 = 0.5 seconds per beat
            # For 16 steps per bar (4 beats): 0.5/4 = 0.125 seconds per step
            beats_per_bar = 4
            seconds_per_beat = 60.0 / bpm
            seconds_per_step = (seconds_per_beat * beats_per_bar) / steps

            # Total duration based on number of steps
            total_duration = seconds_per_step * steps
            total_samples = int(sample_rate * total_duration)

            # Initialize mixed output buffers
            x_mixed = np.zeros(total_samples, dtype=np.float32)
            y_mixed = np.zeros(total_samples, dtype=np.float32)

            # Process each track
            for track_seq in [track1_sequence, track2_sequence, track3_sequence]:
                if not track_seq:
                    continue

                for position, (pad_idx, duration) in track_seq.items():
                    note_freq = base_notes[pad_idx]
                    x, y = generate_pad_sound(note_freq, current_instrument.get(),
                                            current_warp.get(), duration=duration)

                    # Calculate where to place this note based on position (supports fractional positions)
                    start_time = position * seconds_per_step
                    start_sample = int(start_time * sample_rate)
                    end_sample = min(start_sample + len(x), total_samples)
                    segment_len = end_sample - start_sample

                    # Mix into output buffers
                    if start_sample < total_samples:
                        x_mixed[start_sample:end_sample] += x[:segment_len]
                        y_mixed[start_sample:end_sample] += y[:segment_len]

            # Normalize mixed output
            max_val = max(np.max(np.abs(x_mixed)), np.max(np.abs(y_mixed)))
            if max_val > 0:
                x_mixed = x_mixed / max_val
                y_mixed = y_mixed / max_val

            # Update main oscilloscope
            self.x_data = x_mixed
            self.y_data = y_mixed
            self.data_info_label.config(text=f"Points: {len(x_mixed)}")
            self.update_display()

            total_notes = len(track1_sequence) + len(track2_sequence) + len(track3_sequence)
            self.status_label.config(text=f"Merged 3 tracks ({total_notes} notes, {total_duration:.1f}s, {bpm} BPM)")

            # Apply to generate audio for playback
            self.apply_parameters()

        # Sequencer controls
        seq_controls = ctk.CTkFrame(sequencer_frame)
        seq_controls.pack(fill=tk.X, pady=5)

        recording_var = tk.BooleanVar(value=False)
        record_check = ctk.CTkCheckBox(seq_controls, text="Record pads to selected track",
                                      variable=recording_var)
        record_check.pack(anchor=tk.W, pady=2)

        # Clear step button
        ctk.CTkButton(seq_controls, text="Clear Selected Step", command=clear_current_step).pack(
            anchor=tk.W, pady=2, fill=tk.X)

        # Override play_pad to add to sequence when recording
        original_play_pad = play_pad
        def play_pad_with_record(row, col):
            original_play_pad(row, col)
            if recording_var.get():
                add_to_sequence(row, col)

        # Update all pad buttons to use new function
        for row in range(4):
            for col in range(4):
                pad_buttons[(row, col)].config(command=lambda r=row, c=col: play_pad_with_record(r, c))

        seq_btn_frame = ctk.CTkFrame(sequencer_frame)
        seq_btn_frame.pack(fill=tk.X, pady=5)

        ctk.CTkButton(seq_btn_frame, text="Merge & Apply", command=merge_and_apply).pack(side=tk.TOP, pady=2, fill=tk.X)

        clear_btn_frame = ctk.CTkFrame(sequencer_frame)
        clear_btn_frame.pack(fill=tk.X)
        ctk.CTkButton(clear_btn_frame, text="Clear Current Track", command=clear_current_track).pack(
            side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ctk.CTkButton(clear_btn_frame, text="Clear All", command=clear_all_tracks).pack(
            side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Grid Settings Panel with Scrollbar
        grid_settings_frame = ctk.CTkFrame(right_panel)
        grid_settings_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(grid_settings_frame, text="Sound Pad Grid Settings", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        grid_settings_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 10))

        # Create a canvas and scrollbar for grid settings
        settings_canvas = tk.Canvas(grid_settings_frame, highlightthickness=0, height=120)
        settings_scrollbar = ttk.Scrollbar(grid_settings_frame, orient="vertical", command=settings_canvas.yview)

        # Create a frame inside the canvas for settings content
        settings_content = ctk.CTkFrame(settings_canvas)

        # Bind the frame to update scroll region when it changes size
        settings_content.bind("<Configure>",
                            lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))

        settings_canvas.create_window((0, 0), window=settings_content, anchor='nw')
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)

        settings_canvas.pack(side="left", fill="both", expand=True)
        settings_scrollbar.pack(side="right", fill="y")

        # Grid settings controls inside scrollable area
        grid_rows_var = tk.IntVar(value=4)
        grid_cols_var = tk.IntVar(value=4)
        pad_spacing_x = tk.IntVar(value=5)
        pad_spacing_y = tk.IntVar(value=5)

        ctk.CTkLabel(settings_content, text="Grid Dimensions:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(5, 2))

        ctk.CTkLabel(settings_content, text="Rows:").grid(row=1, column=0, sticky='w', padx=5)
        tk.Spinbox(settings_content, from_=2, to=8, textvariable=grid_rows_var, width=8).grid(
            row=1, column=1, sticky='w', padx=5, pady=2)

        ctk.CTkLabel(settings_content, text="Columns:").grid(row=2, column=0, sticky='w', padx=5)
        tk.Spinbox(settings_content, from_=2, to=8, textvariable=grid_cols_var, width=8).grid(
            row=2, column=1, sticky='w', padx=5, pady=2)

        ctk.CTkLabel(settings_content, text="Pad Spacing:", font=('Arial', 9, 'bold')).grid(
            row=3, column=0, columnspan=2, sticky='w', pady=(10, 2))

        ctk.CTkLabel(settings_content, text="Horizontal (px):").grid(row=4, column=0, sticky='w', padx=5)
        tk.Spinbox(settings_content, from_=0, to=20, textvariable=pad_spacing_x, width=8).grid(
            row=4, column=1, sticky='w', padx=5, pady=2)

        ctk.CTkLabel(settings_content, text="Vertical (px):").grid(row=5, column=0, sticky='w', padx=5)
        tk.Spinbox(settings_content, from_=0, to=20, textvariable=pad_spacing_y, width=8).grid(
            row=5, column=1, sticky='w', padx=5, pady=2)

        ctk.CTkLabel(settings_content, text="Appearance:", font=('Arial', 9, 'bold')).grid(
            row=6, column=0, columnspan=2, sticky='w', pady=(10, 2))

        ctk.CTkLabel(settings_content, text="Border Width:").grid(row=7, column=0, sticky='w', padx=5)
        border_width_var = tk.IntVar(value=3)
        tk.Spinbox(settings_content, from_=1, to=10, textvariable=border_width_var, width=8).grid(
            row=7, column=1, sticky='w', padx=5, pady=2)

        ctk.CTkLabel(settings_content, text="Font Size:").grid(row=8, column=0, sticky='w', padx=5)
        font_size_var = tk.IntVar(value=10)
        tk.Spinbox(settings_content, from_=6, to=16, textvariable=font_size_var, width=8).grid(
            row=8, column=1, sticky='w', padx=5, pady=2)

        def apply_grid_settings():
            """Apply grid settings (for future implementation)"""
            messagebox.showinfo("Grid Settings",
                              f"Grid: {grid_rows_var.get()}x{grid_cols_var.get()}\n"
                              f"Spacing: {pad_spacing_x.get()}x{pad_spacing_y.get()} px\n"
                              f"Border: {border_width_var.get()} px\n"
                              f"Font: {font_size_var.get()} pt\n\n"
                              "Note: Grid reconstruction not yet implemented.\n"
                              "Close and reopen Sound Pad to apply changes.")

        ctk.CTkButton(settings_content, text="Apply Settings", command=apply_grid_settings).grid(
            row=9, column=0, columnspan=2, pady=(10, 5), sticky='ew', padx=5)

        ctk.CTkLabel(settings_content, text="(Settings shown for demonstration)",
                 font=('Arial', 7, 'italic'), text_color='gray').grid(
            row=10, column=0, columnspan=2, pady=(0, 5))

        # Bottom buttons
        bottom_frame = ctk.CTkFrame(right_panel)
        bottom_frame.pack(fill=tk.X)

        ctk.CTkButton(bottom_frame, text="Close", command=dialog.destroy).pack(fill=tk.X)

    def generate_random_harmonics(self):
        """Open dialog for random harmonics generation with two modes"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Random Harmonics Generator")
        dialog.geometry("500x650")

        # Stop audio when dialog is closed
        def on_dialog_close():
            if self.is_playing:
                self.stop_playback()
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        # Instructions
        instruction_frame = ctk.CTkFrame(dialog)
        instruction_frame.pack(pady=10, padx=10, fill=tk.X)
        ctk.CTkLabel(instruction_frame, text="Generate Random Harmonic Patterns",
                 font=('Arial', 12, 'bold')).pack()
        ctk.CTkLabel(instruction_frame, text="Choose generation mode and click Generate",
                 font=('Arial', 9), text_color='gray').pack()

        # Create a frame to hold the canvas and scrollbar
        scroll_container = ctk.CTkFrame(dialog)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create canvas for scrolling
        canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)

        # Create frame inside canvas for mode selection
        scrollable_frame = ctk.CTkFrame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Unbind mousewheel when dialog is destroyed
        def cleanup_mousewheel():
            canvas.unbind_all("<MouseWheel>")
            if self.is_playing:
                self.stop_playback()
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", cleanup_mousewheel)

        # Mode selection (now inside scrollable_frame)
        mode_frame = ctk.CTkFrame(scrollable_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(mode_frame, text="Generation Mode", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        mode_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        mode_var = tk.StringVar(value="random")

        # Mode 1: Fully Random
        mode1_frame = ctk.CTkFrame(mode_frame)
        mode1_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(mode1_frame, text="Fully Random",
                       variable=mode_var, value="random").pack(anchor=tk.W)
        ctk.CTkLabel(mode1_frame, text="• X and Y channels completely independent",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode1_frame, text="• 1-10 terms per channel",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode1_frame, text="• Frequencies: 0-1000",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode1_frame, text="• Phase shifts: 0-1000",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)

        ctk.CTkFrame(mode_frame, height=2).pack(fill=tk.X, pady=10)

        # Mode 2: Mirrored (sin/cos swap)
        mode2_frame = ctk.CTkFrame(mode_frame)
        mode2_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(mode2_frame, text="Mirrored (Sin/Cos Swap)",
                       variable=mode_var, value="mirrored").pack(anchor=tk.W)
        ctk.CTkLabel(mode2_frame, text="• X and Y channels use same parameters",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode2_frame, text="• sin terms on X become cos on Y",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode2_frame, text="• cos terms on X become sin on Y",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode2_frame, text="• Creates symmetrical Lissajous patterns",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)

        ctk.CTkFrame(mode_frame, height=2).pack(fill=tk.X, pady=10)

        # Mode 3: Mirrored with Frequency Sweep
        mode3_frame = ctk.CTkFrame(mode_frame)
        mode3_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(mode3_frame, text="Mirrored + Frequency Sweep",
                       variable=mode_var, value="freq_sweep").pack(anchor=tk.W)
        ctk.CTkLabel(mode3_frame, text="• Like mirrored mode (sin/cos swap)",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode3_frame, text="• Y channel frequencies sweep continuously",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode3_frame, text="• Creates morphing animated patterns",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)

        ctk.CTkFrame(mode_frame, height=2).pack(fill=tk.X, pady=10)

        # Mode 4: Mirrored with Phase Sweep
        mode4_frame = ctk.CTkFrame(mode_frame)
        mode4_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(mode4_frame, text="Mirrored + Phase Sweep",
                       variable=mode_var, value="phase_sweep").pack(anchor=tk.W)
        ctk.CTkLabel(mode4_frame, text="• Like mirrored mode (sin/cos swap)",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode4_frame, text="• Y channel phases sweep continuously",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode4_frame, text="• Creates rotating Lissajous patterns",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)

        ctk.CTkFrame(mode_frame, height=2).pack(fill=tk.X, pady=10)

        # Mode 5: Mirrored with Frequency and Phase Sweep
        mode5_frame = ctk.CTkFrame(mode_frame)
        mode5_frame.pack(fill=tk.X, pady=5)
        ctk.CTkRadioButton(mode5_frame, text="Mirrored + Frequency & Phase Sweep",
                       variable=mode_var, value="freq_phase_sweep").pack(anchor=tk.W)
        ctk.CTkLabel(mode5_frame, text="• Like mirrored mode (sin/cos swap)",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode5_frame, text="• Y channel frequencies AND phases sweep continuously",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)
        ctk.CTkLabel(mode5_frame, text="• Creates complex morphing animations",
                 font=('Arial', 8), text_color='gray').pack(anchor=tk.W, padx=20)

        # Terms control section (outside scrollable area, fixed at bottom)
        terms_frame = ctk.CTkFrame(dialog)
        terms_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(terms_frame, text="Number of Terms", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        terms_frame.pack(fill=tk.X, padx=10, pady=10)

        random_terms_var = tk.BooleanVar(value=True)
        manual_terms_var = tk.IntVar(value=3)

        def toggle_terms_mode():
            """Enable/disable manual terms entry based on checkbox"""
            if random_terms_var.get():
                terms_spinbox.config(state="disabled")
            else:
                terms_spinbox.config(state="normal")

        terms_checkbox = ctk.CTkCheckBox(
            terms_frame,
            text="Random number of terms (1-10)",
            variable=random_terms_var,
            command=toggle_terms_mode
        )
        terms_checkbox.pack(anchor=tk.W, pady=5)

        manual_frame = ctk.CTkFrame(terms_frame)
        manual_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(manual_frame, text="Manual term count:").pack(side=tk.LEFT, padx=5)
        terms_spinbox = tk.Spinbox(
            manual_frame,
            from_=0,
            to=20,
            textvariable=manual_terms_var,
            width=10,
            state="disabled"
        )
        terms_spinbox.pack(side=tk.LEFT, padx=5)
        ctk.CTkLabel(manual_frame, text="(0 = single sin/cos pair)",
                 font=('Arial', 8), text_color='gray').pack(side=tk.LEFT, padx=5)

        # Preview area (inside scrollable_frame)
        preview_frame = ctk.CTkFrame(scrollable_frame)
        preview_frame.pack(fill=tk.X, pady=5)
        ctk.CTkLabel(preview_frame, text="Last Generated", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        preview_text = tk.Text(preview_frame, height=8, width=60, font=('Courier', 8), state=tk.DISABLED)
        preview_text.pack(fill=tk.BOTH, expand=True)

        def generate_pattern():
            """Generate pattern based on selected mode"""
            mode = mode_var.get()

            # Calculate based on actual playback parameters to prevent overcalculation
            if mode in ["freq_sweep", "phase_sweep", "freq_phase_sweep"]:
                # Sweep modes: use minimal points since pattern gets repeated/tiled
                # 100 points total ÷ 10 frames = 10 points per frame (efficient)
                num_points = 100
            else:
                # Normal modes: 1000 points for smooth single pattern
                num_points = 1000

            t = np.linspace(0, 2*np.pi, num_points)

            # Determine number of terms
            if random_terms_var.get():
                # Random mode: 1-10 terms
                num_terms_x = np.random.randint(1, 11)
                num_terms_y = np.random.randint(1, 11)
                num_terms = np.random.randint(1, 11)
            else:
                # Manual mode: use specified count (0-20)
                manual_count = manual_terms_var.get()
                num_terms_x = manual_count
                num_terms_y = manual_count
                num_terms = manual_count

            # Special case: 0 terms = single sin/cos pair
            if not random_terms_var.get() and manual_terms_var.get() == 0:
                # Simple case: X = sin, Y = cos
                frequency = np.random.uniform(1, 100)
                phase = np.random.uniform(0, 2*np.pi)

                x_data = np.sin(frequency * t + phase)
                y_data = np.cos(frequency * t + phase)

                x_description = [f"1.00·sin({frequency:.1f}t + {phase:.1f})"]
                y_description = [f"1.00·cos({frequency:.1f}t + {phase:.1f})"]

                status_msg = f"Random harmonics: Single sin/cos pair"

            elif mode == "random":
                # Mode 1: Fully Random

                # Generate X channel
                x_data = np.zeros(num_points)
                x_description = []

                for i in range(num_terms_x):
                    wave_type = np.random.choice(['sin', 'cos'])
                    amplitude = np.random.uniform(0.3, 1.0)
                    frequency = np.random.uniform(0, 1000)
                    phase = np.random.uniform(0, 1000)

                    if wave_type == 'sin':
                        x_data += amplitude * np.sin(frequency * t + phase)
                        x_description.append(f"{amplitude:.2f}·sin({frequency:.1f}t + {phase:.1f})")
                    else:
                        x_data += amplitude * np.cos(frequency * t + phase)
                        x_description.append(f"{amplitude:.2f}·cos({frequency:.1f}t + {phase:.1f})")

                # Generate Y channel
                y_data = np.zeros(num_points)
                y_description = []

                for i in range(num_terms_y):
                    wave_type = np.random.choice(['sin', 'cos'])
                    amplitude = np.random.uniform(0.3, 1.0)
                    frequency = np.random.uniform(0, 1000)
                    phase = np.random.uniform(0, 1000)

                    if wave_type == 'sin':
                        y_data += amplitude * np.sin(frequency * t + phase)
                        y_description.append(f"{amplitude:.2f}·sin({frequency:.1f}t + {phase:.1f})")
                    else:
                        y_data += amplitude * np.cos(frequency * t + phase)
                        y_description.append(f"{amplitude:.2f}·cos({frequency:.1f}t + {phase:.1f})")

                status_msg = f"Random harmonics (fully random): X({num_terms_x} terms), Y({num_terms_y} terms)"

            elif mode == "mirrored":
                # Mode 2: Mirrored (sin/cos swap)
                # num_terms already set above based on random/manual choice

                # Generate shared parameters
                terms = []
                for i in range(max(1, num_terms)):
                    wave_type = np.random.choice(['sin', 'cos'])
                    amplitude = np.random.uniform(0.3, 1.0)
                    frequency = np.random.uniform(0, 1000)
                    phase = np.random.uniform(0, 1000)
                    terms.append((wave_type, amplitude, frequency, phase))

                # Generate X channel
                x_data = np.zeros(num_points)
                x_description = []

                for wave_type, amplitude, frequency, phase in terms:
                    if wave_type == 'sin':
                        x_data += amplitude * np.sin(frequency * t + phase)
                        x_description.append(f"{amplitude:.2f}·sin({frequency:.1f}t + {phase:.1f})")
                    else:
                        x_data += amplitude * np.cos(frequency * t + phase)
                        x_description.append(f"{amplitude:.2f}·cos({frequency:.1f}t + {phase:.1f})")

                # Generate Y channel (swap sin/cos)
                y_data = np.zeros(num_points)
                y_description = []

                for wave_type, amplitude, frequency, phase in terms:
                    if wave_type == 'sin':
                        # sin on X becomes cos on Y
                        y_data += amplitude * np.cos(frequency * t + phase)
                        y_description.append(f"{amplitude:.2f}·cos({frequency:.1f}t + {phase:.1f})")
                    else:
                        # cos on X becomes sin on Y
                        y_data += amplitude * np.sin(frequency * t + phase)
                        y_description.append(f"{amplitude:.2f}·sin({frequency:.1f}t + {phase:.1f})")

                status_msg = f"Random harmonics (mirrored): {num_terms} terms"

            elif mode == "freq_sweep":
                # Mode 3: Mirrored with Frequency Sweep
                # num_terms already set above based on random/manual choice
                num_sweep_steps = 10  # Reduced from 20 to prevent preview flooding

                # Generate shared parameters
                terms = []
                for i in range(max(1, num_terms)):
                    wave_type = np.random.choice(['sin', 'cos'])
                    amplitude = np.random.uniform(0.01, 1.0)
                    frequency_start = np.random.uniform(0, 1000)
                    frequency_end = np.random.uniform(0, 1000)
                    phase = np.random.uniform(0, 1000)
                    terms.append((wave_type, amplitude, frequency_start, frequency_end, phase))

                # Generate frames
                all_x_frames = []
                all_y_frames = []

                for step in range(num_sweep_steps):
                    # Interpolate frequencies for this frame
                    interp_factor = step / (num_sweep_steps - 1) if num_sweep_steps > 1 else 0

                    # Generate X channel (constant frequency)
                    x_frame = np.zeros(num_points)
                    for wave_type, amplitude, freq_start, freq_end, phase in terms:
                        frequency = freq_start  # X uses start frequency
                        if wave_type == 'sin':
                            x_frame += amplitude * np.sin(frequency * t + phase)
                        else:
                            x_frame += amplitude * np.cos(frequency * t + phase)

                    # Generate Y channel (swept frequency)
                    y_frame = np.zeros(num_points)
                    for wave_type, amplitude, freq_start, freq_end, phase in terms:
                        frequency = freq_start + (freq_end - freq_start) * interp_factor
                        if wave_type == 'sin':
                            # sin on X becomes cos on Y
                            y_frame += amplitude * np.cos(frequency * t + phase)
                        else:
                            # cos on X becomes sin on Y
                            y_frame += amplitude * np.sin(frequency * t + phase)

                    all_x_frames.append(x_frame)
                    all_y_frames.append(y_frame)

                # Concatenate all frames
                x_data = np.concatenate(all_x_frames)
                y_data = np.concatenate(all_y_frames)

                # Build descriptions
                x_description = []
                y_description = []
                for wave_type, amplitude, freq_start, freq_end, phase in terms:
                    if wave_type == 'sin':
                        x_description.append(f"{amplitude:.2f}·sin({freq_start:.1f}t + {phase:.1f})")
                        y_description.append(f"{amplitude:.2f}·cos({freq_start:.1f}→{freq_end:.1f} t + {phase:.1f})")
                    else:
                        x_description.append(f"{amplitude:.2f}·cos({freq_start:.1f}t + {phase:.1f})")
                        y_description.append(f"{amplitude:.2f}·sin({freq_start:.1f}→{freq_end:.1f} t + {phase:.1f})")

                status_msg = f"Random harmonics (freq sweep): {num_terms} terms, {num_sweep_steps} frames"

            elif mode == "phase_sweep":
                # Mode 4: Mirrored with Phase Sweep
                # num_terms already set above based on random/manual choice
                num_sweep_steps = 10  # Reduced from 20 to prevent preview flooding

                # Generate shared parameters
                terms = []
                for i in range(max(1, num_terms)):
                    wave_type = np.random.choice(['sin', 'cos'])
                    amplitude = np.random.uniform(0.01, 1.0)
                    frequency = np.random.uniform(0, 1000)
                    phase_start = np.random.uniform(0, 1000)
                    phase_end = np.random.uniform(0, 1000)
                    terms.append((wave_type, amplitude, frequency, phase_start, phase_end))

                # Generate frames
                all_x_frames = []
                all_y_frames = []

                for step in range(num_sweep_steps):
                    # Interpolate phases for this frame
                    interp_factor = step / (num_sweep_steps - 1) if num_sweep_steps > 1 else 0

                    # Generate X channel (constant phase)
                    x_frame = np.zeros(num_points)
                    for wave_type, amplitude, frequency, phase_start, phase_end in terms:
                        phase = phase_start  # X uses start phase
                        if wave_type == 'sin':
                            x_frame += amplitude * np.sin(frequency * t + phase)
                        else:
                            x_frame += amplitude * np.cos(frequency * t + phase)

                    # Generate Y channel (swept phase)
                    y_frame = np.zeros(num_points)
                    for wave_type, amplitude, frequency, phase_start, phase_end in terms:
                        phase = phase_start + (phase_end - phase_start) * interp_factor
                        if wave_type == 'sin':
                            # sin on X becomes cos on Y
                            y_frame += amplitude * np.cos(frequency * t + phase)
                        else:
                            # cos on X becomes sin on Y
                            y_frame += amplitude * np.sin(frequency * t + phase)

                    all_x_frames.append(x_frame)
                    all_y_frames.append(y_frame)

                # Concatenate all frames
                x_data = np.concatenate(all_x_frames)
                y_data = np.concatenate(all_y_frames)

                # Build descriptions
                x_description = []
                y_description = []
                for wave_type, amplitude, frequency, phase_start, phase_end in terms:
                    if wave_type == 'sin':
                        x_description.append(f"{amplitude:.2f}·sin({frequency:.1f}t + {phase_start:.1f})")
                        y_description.append(f"{amplitude:.2f}·cos({frequency:.1f}t + {phase_start:.1f}→{phase_end:.1f})")
                    else:
                        x_description.append(f"{amplitude:.2f}·cos({frequency:.1f}t + {phase_start:.1f})")
                        y_description.append(f"{amplitude:.2f}·sin({frequency:.1f}t + {phase_start:.1f}→{phase_end:.1f})")

                status_msg = f"Random harmonics (phase sweep): {num_terms} terms, {num_sweep_steps} frames"

            elif mode == "freq_phase_sweep":
                # Mode 5: Mirrored with both Frequency and Phase Sweep
                # num_terms already set above based on random/manual choice
                num_sweep_steps = 10  # Reduced from 20 to prevent preview flooding

                # Generate shared parameters
                terms = []
                for i in range(max(1, num_terms)):
                    wave_type = np.random.choice(['sin', 'cos'])
                    amplitude = np.random.uniform(0.01, 1.0)
                    frequency_start = np.random.uniform(0, 1000)
                    frequency_end = np.random.uniform(0, 1000)
                    phase_start = np.random.uniform(0, 1000)
                    phase_end = np.random.uniform(0, 1000)
                    terms.append((wave_type, amplitude, frequency_start, frequency_end, phase_start, phase_end))

                # Generate frames
                all_x_frames = []
                all_y_frames = []

                for step in range(num_sweep_steps):
                    # Interpolate both frequencies and phases for this frame
                    interp_factor = step / (num_sweep_steps - 1) if num_sweep_steps > 1 else 0

                    # Generate X channel (constant frequency and phase)
                    x_frame = np.zeros(num_points)
                    for wave_type, amplitude, freq_start, freq_end, phase_start, phase_end in terms:
                        frequency = freq_start  # X uses start frequency
                        phase = phase_start  # X uses start phase
                        if wave_type == 'sin':
                            x_frame += amplitude * np.sin(frequency * t + phase)
                        else:
                            x_frame += amplitude * np.cos(frequency * t + phase)

                    # Generate Y channel (swept frequency and phase)
                    y_frame = np.zeros(num_points)
                    for wave_type, amplitude, freq_start, freq_end, phase_start, phase_end in terms:
                        frequency = freq_start + (freq_end - freq_start) * interp_factor
                        phase = phase_start + (phase_end - phase_start) * interp_factor
                        if wave_type == 'sin':
                            # sin on X becomes cos on Y
                            y_frame += amplitude * np.cos(frequency * t + phase)
                        else:
                            # cos on X becomes sin on Y
                            y_frame += amplitude * np.sin(frequency * t + phase)

                    all_x_frames.append(x_frame)
                    all_y_frames.append(y_frame)

                # Concatenate all frames
                x_data = np.concatenate(all_x_frames)
                y_data = np.concatenate(all_y_frames)

                # Build descriptions
                x_description = []
                y_description = []
                for wave_type, amplitude, freq_start, freq_end, phase_start, phase_end in terms:
                    if wave_type == 'sin':
                        x_description.append(f"{amplitude:.2f}·sin({freq_start:.1f}t + {phase_start:.1f})")
                        y_description.append(f"{amplitude:.2f}·cos({freq_start:.1f}→{freq_end:.1f} t + {phase_start:.1f}→{phase_end:.1f})")
                    else:
                        x_description.append(f"{amplitude:.2f}·cos({freq_start:.1f}t + {phase_start:.1f})")
                        y_description.append(f"{amplitude:.2f}·sin({freq_start:.1f}→{freq_end:.1f} t + {phase_start:.1f}→{phase_end:.1f})")

                status_msg = f"Random harmonics (freq+phase sweep): {num_terms} terms, {num_sweep_steps} frames"

            # Normalize
            x_data = x_data / (np.max(np.abs(x_data)) + 1e-10)
            y_data = y_data / (np.max(np.abs(y_data)) + 1e-10)

            # Set as current data
            self.x_data = x_data
            self.y_data = y_data
            self.data_info_label.config(text=f"Points: {len(x_data)}")
            self.update_display()
            self.status_label.config(text=status_msg)

            # Update preview
            preview_text.config(state=tk.NORMAL)
            preview_text.delete('1.0', tk.END)
            preview_text.insert(tk.END, f"X channel:\n")
            for desc in x_description:
                preview_text.insert(tk.END, f"  + {desc}\n")
            preview_text.insert(tk.END, f"\nY channel:\n")
            for desc in y_description:
                preview_text.insert(tk.END, f"  + {desc}\n")
            preview_text.config(state=tk.DISABLED)

            # Print to console
            print("\n=== Random Harmonics Generated ===")
            print(f"Mode: {mode}")
            print(f"X channel:")
            for desc in x_description:
                print(f"  + {desc}")
            print(f"\nY channel:")
            for desc in y_description:
                print(f"  + {desc}")
            print("="*35 + "\n")

            # Apply parameters and generate audio
            self.apply_parameters()

        # Buttons (inside scrollable_frame)
        button_frame = ctk.CTkFrame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        ctk.CTkButton(button_frame, text="Generate", command=generate_pattern).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ctk.CTkButton(button_frame, text="Close", command=cleanup_mousewheel).pack(
            side=tk.LEFT, padx=5)

    def save_to_wav(self):
        """Save current audio to WAV file"""
        if self.current_audio is None:
            messagebox.showwarning("Warning", "No audio generated yet. Click 'Apply & Generate' first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save WAV File",
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            import soundfile as sf
            sf.write(filename, self.current_audio, self.current_fs)
            messagebox.showinfo("Success", f"Saved to {filename}")
        except ImportError:
            messagebox.showerror("Error", "soundfile library not installed.\nInstall with: pip install soundfile")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")


def main():
    root = ctk.CTk()
    root.title("JUANTRONIX - Oscilloscope XY Audio Generator")
    app = OscilloscopeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

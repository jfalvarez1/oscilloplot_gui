#!/usr/bin/env python3
"""
Oscilloscope XY Audio Generator with Real-Time GUI
Features live visualization, parameter controls, and various effects
"""

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue


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
        
        # Current data
        self.x_data = np.array([1, 2, 3])
        self.y_data = np.array([4, 4, 3])
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

        # Initialize wavy and lightning labels with default values
        self.update_wavy_labels_only()
        self.update_lightning_labels_only()

        # Start update loops
        self.root.after(50, self.check_updates)
        self.root.after(20, self.update_live_preview)  # 50 FPS preview update
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)
        
        # Left panel - Controls (with scrollbar)
        control_container = ttk.Frame(main_container, width=300)
        control_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_container.grid_propagate(False)

        # Create canvas and scrollbar for controls
        control_canvas = tk.Canvas(control_container, highlightthickness=0, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.LabelFrame(control_canvas, text="Controls", padding="10")

        control_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create window in canvas
        canvas_frame = control_canvas.create_window((0, 0), window=control_frame, anchor=tk.NW)

        # Configure scrolling
        def configure_scroll_region(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def configure_canvas_width(event):
            # Update canvas window width to match canvas width (minus scrollbar)
            canvas_width = control_canvas.winfo_width()
            control_canvas.itemconfig(canvas_frame, width=canvas_width)

        control_frame.bind("<Configure>", configure_scroll_region)
        control_canvas.bind("<Configure>", configure_canvas_width)

        # Mouse wheel scrolling
        def on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        control_canvas.bind_all("<MouseWheel>", on_mousewheel)  # Windows
        control_canvas.bind_all("<Button-4>", lambda e: control_canvas.yview_scroll(-1, "units"))  # Linux up
        control_canvas.bind_all("<Button-5>", lambda e: control_canvas.yview_scroll(1, "units"))   # Linux down
        
        # Right panel - Display
        display_frame = ttk.LabelFrame(main_container, text="Oscilloscope Display", padding="10")
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
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
        file_frame = ttk.LabelFrame(parent, text="Data Source", padding="5")
        file_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        ttk.Button(file_frame, text="Load MATLAB File (.m)", 
                  command=self.load_matlab_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Text File (.txt)", 
                  command=self.load_txt_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load NumPy File (.npz)",
                  command=self.load_numpy_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Generate Test Pattern",
                  command=self.generate_test_pattern).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Draw Pattern",
                  command=self.open_drawing_canvas).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Sum of Harmonics",
                  command=self.open_harmonic_sum).pack(fill=tk.X, pady=2)

        # Data info
        self.data_info_label = ttk.Label(file_frame, text="Points: 3", 
                                         font=('Arial', 9, 'italic'))
        self.data_info_label.pack(pady=5)
        
        # === AUDIO PARAMETERS ===
        audio_frame = ttk.LabelFrame(parent, text="Audio Parameters", padding="5")
        audio_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Sample Rate
        ttk.Label(audio_frame, text="Base Sample Rate (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.sample_rate_var = tk.IntVar(value=1000)
        self.sample_rate_spin = ttk.Spinbox(audio_frame, from_=100, to=10000, 
                                           textvariable=self.sample_rate_var, width=10)
        self.sample_rate_spin.grid(row=0, column=1, pady=2)
        
        # Playback Multiplier (Frequency)
        ttk.Label(audio_frame, text="Playback Multiplier:").grid(row=1, column=0, sticky=tk.W)
        self.freq_mult_var = tk.IntVar(value=100)
        self.freq_mult_spin = ttk.Spinbox(audio_frame, from_=10, to=500, 
                                         textvariable=self.freq_mult_var, width=10)
        self.freq_mult_spin.grid(row=1, column=1, pady=2)
        
        ttk.Label(audio_frame, text="→ Actual Rate:").grid(row=2, column=0, sticky=tk.W)
        self.actual_rate_label = ttk.Label(audio_frame, text="100 kHz", 
                                          font=('Arial', 9, 'bold'))
        self.actual_rate_label.grid(row=2, column=1, pady=2)
        
        # Duration
        ttk.Label(audio_frame, text="Duration (seconds):").grid(row=3, column=0, sticky=tk.W)
        self.duration_var = tk.IntVar(value=15)
        self.duration_spin = ttk.Spinbox(audio_frame, from_=5, to=120, 
                                        textvariable=self.duration_var, width=10)
        self.duration_spin.grid(row=3, column=1, pady=2)
        
        # N Repeat
        ttk.Label(audio_frame, text="Pattern Repeats:").grid(row=4, column=0, sticky=tk.W)
        self.n_repeat_var = tk.IntVar(value=200)  # Increased default for full rotations
        self.n_repeat_spin = ttk.Spinbox(audio_frame, from_=1, to=2000, 
                                        textvariable=self.n_repeat_var, width=10)
        self.n_repeat_spin.grid(row=4, column=1, pady=2)
        
        # Rotation info label
        self.rotation_info_label = ttk.Label(audio_frame, text="",
                                            font=('Arial', 8, 'italic'),
                                            foreground='blue')
        self.rotation_info_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(2,5))

        # Update rate label on change
        self.freq_mult_var.trace('w', self.update_rate_label)
        self.sample_rate_var.trace('w', self.update_rate_label)
        
        # === EFFECTS ===
        effects_frame = ttk.LabelFrame(parent, text="Effects", padding="5")
        effects_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Enable Reflections
        self.reflections_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(effects_frame, text="Enable Mirror Reflections", 
                       variable=self.reflections_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=5)
        
        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Y-Axis Fade Sequence
        self.y_fade_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(effects_frame, text="Y-Axis Fade Sequence", 
                       variable=self.y_fade_var,
                       command=self.effect_changed).pack(anchor=tk.W)
        
        ttk.Label(effects_frame, text="Y Fade Steps:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.y_fade_steps = tk.IntVar(value=10)
        self.y_fade_steps.trace('w', lambda *args: self.effect_changed())
        self.y_fade_steps_spin = ttk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.y_fade_steps,
                   command=self.effect_changed)
        self.y_fade_steps_spin.pack(anchor=tk.W, padx=20)

        ttk.Label(effects_frame, text="Y Fade Speed (repeats/step):",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.y_fade_speed = tk.IntVar(value=1)
        self.y_fade_speed.trace('w', lambda *args: self.effect_changed())
        self.y_fade_speed_spin = ttk.Spinbox(effects_frame, from_=1, to=20, width=8,
                   textvariable=self.y_fade_speed,
                   command=self.effect_changed)
        self.y_fade_speed_spin.pack(anchor=tk.W, padx=20)
        
        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # X-Axis Fade Sequence
        self.x_fade_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(effects_frame, text="X-Axis Fade Sequence", 
                       variable=self.x_fade_var,
                       command=self.effect_changed).pack(anchor=tk.W)
        
        ttk.Label(effects_frame, text="X Fade Steps:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.x_fade_steps = tk.IntVar(value=10)
        self.x_fade_steps.trace('w', lambda *args: self.effect_changed())
        self.x_fade_steps_spin = ttk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.x_fade_steps,
                   command=self.effect_changed)
        self.x_fade_steps_spin.pack(anchor=tk.W, padx=20)

        ttk.Label(effects_frame, text="X Fade Speed (repeats/step):",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.x_fade_speed = tk.IntVar(value=1)
        self.x_fade_speed.trace('w', lambda *args: self.effect_changed())
        self.x_fade_speed_spin = ttk.Spinbox(effects_frame, from_=1, to=20, width=8,
                   textvariable=self.x_fade_speed,
                   command=self.effect_changed)
        self.x_fade_speed_spin.pack(anchor=tk.W, padx=20)

        # Alternate X/Y Fade option
        self.alternate_xy_fade_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(effects_frame, text="Alternate X/Y Fade (X first, then Y, repeat)",
                       variable=self.alternate_xy_fade_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=(5,0))

        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # Shrink/Unshrink (scale both X and Y together)
        self.shrink_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(effects_frame, text="Shrink/Unshrink (Scale X & Y together)",
                       variable=self.shrink_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        ttk.Label(effects_frame, text="Shrink Steps:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.shrink_steps = tk.IntVar(value=10)
        self.shrink_steps.trace('w', lambda *args: self.effect_changed())
        self.shrink_steps_spin = ttk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.shrink_steps,
                   command=self.effect_changed)
        self.shrink_steps_spin.pack(anchor=tk.W, padx=20)

        ttk.Label(effects_frame, text="Shrink Speed (repeats/step):",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.shrink_speed = tk.IntVar(value=1)
        self.shrink_speed.trace('w', lambda *args: self.effect_changed())
        self.shrink_speed_spin = ttk.Spinbox(effects_frame, from_=1, to=20, width=8,
                   textvariable=self.shrink_speed,
                   command=self.effect_changed)
        self.shrink_speed_spin.pack(anchor=tk.W, padx=20)

        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # Noise Effects
        noise_frame = ttk.LabelFrame(effects_frame, text="Noise", padding="5")
        noise_frame.pack(fill=tk.X, pady=5)

        self.x_noise_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(noise_frame, text="Add X-Channel Noise",
                       variable=self.x_noise_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        ttk.Label(noise_frame, text="X Noise Amplitude:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.x_noise_amp = tk.DoubleVar(value=0.05)
        ttk.Scale(noise_frame, from_=0.001, to=0.3, orient=tk.HORIZONTAL,
                 variable=self.x_noise_amp,
                 command=lambda v: self.effect_changed()).pack(fill=tk.X, padx=20)

        self.y_noise_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(noise_frame, text="Add Y-Channel Noise",
                       variable=self.y_noise_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=(10,0))

        ttk.Label(noise_frame, text="Y Noise Amplitude:",
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.y_noise_amp = tk.DoubleVar(value=0.05)
        ttk.Scale(noise_frame, from_=0.001, to=0.3, orient=tk.HORIZONTAL,
                 variable=self.y_noise_amp,
                 command=lambda v: self.effect_changed()).pack(fill=tk.X, padx=20)

        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # Wavy Effects
        wavy_frame = ttk.LabelFrame(effects_frame, text="Wavy Effect", padding="5")
        wavy_frame.pack(fill=tk.X, pady=5)

        self.x_wavy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(wavy_frame, text="Add X-Channel Wavy",
                       variable=self.x_wavy_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        # X Amplitude with value label
        x_amp_frame = ttk.Frame(wavy_frame)
        x_amp_frame.pack(fill=tk.X, padx=20)
        ttk.Label(x_amp_frame, text="X Amplitude (K):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.x_wavy_amp_label = ttk.Label(x_amp_frame, text="0.200",
                 font=('Arial', 8, 'bold'))
        self.x_wavy_amp_label.pack(side=tk.RIGHT)

        self.x_wavy_amp = tk.DoubleVar(value=0.2)
        ttk.Scale(wavy_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                 variable=self.x_wavy_amp,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        # X Frequency with value label
        x_freq_frame = ttk.Frame(wavy_frame)
        x_freq_frame.pack(fill=tk.X, padx=20)
        ttk.Label(x_freq_frame, text="X Angular Frequency (ω):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.x_wavy_freq_label = ttk.Label(x_freq_frame, text="10.0",
                 font=('Arial', 8, 'bold'))
        self.x_wavy_freq_label.pack(side=tk.RIGHT)

        self.x_wavy_freq = tk.DoubleVar(value=10.0)
        ttk.Scale(wavy_frame, from_=1.0, to=1000000.0, orient=tk.HORIZONTAL,
                 variable=self.x_wavy_freq,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        self.y_wavy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(wavy_frame, text="Add Y-Channel Wavy",
                       variable=self.y_wavy_var,
                       command=self.effect_changed).pack(anchor=tk.W, pady=(10,0))

        # Y Amplitude with value label
        y_amp_frame = ttk.Frame(wavy_frame)
        y_amp_frame.pack(fill=tk.X, padx=20)
        ttk.Label(y_amp_frame, text="Y Amplitude (K):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.y_wavy_amp_label = ttk.Label(y_amp_frame, text="0.200",
                 font=('Arial', 8, 'bold'))
        self.y_wavy_amp_label.pack(side=tk.RIGHT)

        self.y_wavy_amp = tk.DoubleVar(value=0.2)
        ttk.Scale(wavy_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                 variable=self.y_wavy_amp,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        # Y Frequency with value label
        y_freq_frame = ttk.Frame(wavy_frame)
        y_freq_frame.pack(fill=tk.X, padx=20)
        ttk.Label(y_freq_frame, text="Y Angular Frequency (ω):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.y_wavy_freq_label = ttk.Label(y_freq_frame, text="10.0",
                 font=('Arial', 8, 'bold'))
        self.y_wavy_freq_label.pack(side=tk.RIGHT)

        self.y_wavy_freq = tk.DoubleVar(value=10.0)
        ttk.Scale(wavy_frame, from_=1.0, to=1000000.0, orient=tk.HORIZONTAL,
                 variable=self.y_wavy_freq,
                 command=lambda v: self.update_wavy_labels()).pack(fill=tk.X, padx=20)

        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # Rotation
        rotation_frame = ttk.LabelFrame(effects_frame, text="Rotation", padding="5")
        rotation_frame.pack(fill=tk.X, pady=5)
        
        self.rotation_mode_var = tk.StringVar(value="Off")
        ttk.Radiobutton(rotation_frame, text="Off", variable=self.rotation_mode_var, 
                       value="Off", command=self.rotation_mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(rotation_frame, text="Static Angle", variable=self.rotation_mode_var, 
                       value="Static", command=self.rotation_mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(rotation_frame, text="Rotate Clockwise (CW)", variable=self.rotation_mode_var, 
                       value="CW", command=self.rotation_mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(rotation_frame, text="Rotate Counter-Clockwise (CCW)", variable=self.rotation_mode_var, 
                       value="CCW", command=self.rotation_mode_changed).pack(anchor=tk.W)
        
        ttk.Label(rotation_frame, text="Static Angle (degrees):").pack(anchor=tk.W, pady=(5,0))
        self.rotation_angle = tk.DoubleVar(value=0.0)
        ttk.Scale(rotation_frame, from_=-180, to=180, orient=tk.HORIZONTAL,
                 variable=self.rotation_angle,
                 command=lambda v: self.rotation_mode_changed()).pack(fill=tk.X)
        
        ttk.Label(rotation_frame, text="Rotation Speed (degrees/cycle):").pack(anchor=tk.W, pady=(5,0))
        self.rotation_speed = tk.DoubleVar(value=5.0)
        ttk.Scale(rotation_frame, from_=0.5, to=45, orient=tk.HORIZONTAL,
                 variable=self.rotation_speed,
                 command=lambda v: self.rotation_mode_changed()).pack(fill=tk.X)

        ttk.Label(rotation_frame, text="Tip: 360° ÷ speed = steps per rotation\nMore Pattern Repeats = more rotations",
                 font=('Arial', 7, 'italic'), foreground='gray').pack(anchor=tk.W, pady=(5,0))

        # Update rotation info when values change (set up after all variables are created)
        self.n_repeat_var.trace('w', self.update_rotation_info)
        self.rotation_speed.trace('w', self.update_rotation_info)

        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # Lightning Bolts Effect
        lightning_frame = ttk.LabelFrame(effects_frame, text="Lightning Bolts", padding="5")
        lightning_frame.pack(fill=tk.X, pady=5)

        self.lightning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lightning_frame, text="Add Lightning Bolts",
                       variable=self.lightning_var,
                       command=self.effect_changed).pack(anchor=tk.W)

        self.lightning_mode_var = tk.StringVar(value="Static")
        ttk.Radiobutton(lightning_frame, text="Static", variable=self.lightning_mode_var,
                       value="Static", command=self.effect_changed).pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(lightning_frame, text="Dynamic (Animated)", variable=self.lightning_mode_var,
                       value="Dynamic", command=self.effect_changed).pack(anchor=tk.W, padx=20)

        # Number of bolts with value label
        bolt_count_frame = ttk.Frame(lightning_frame)
        bolt_count_frame.pack(fill=tk.X, padx=20, pady=(5,0))
        ttk.Label(bolt_count_frame, text="Number of Bolts:",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.lightning_count_label = ttk.Label(bolt_count_frame, text="4",
                 font=('Arial', 8, 'bold'))
        self.lightning_count_label.pack(side=tk.RIGHT)

        self.lightning_count = tk.IntVar(value=4)
        ttk.Scale(lightning_frame, from_=1, to=12, orient=tk.HORIZONTAL,
                 variable=self.lightning_count,
                 command=lambda v: self.update_lightning_labels()).pack(fill=tk.X, padx=20)

        # Jaggedness with value label
        jagged_frame = ttk.Frame(lightning_frame)
        jagged_frame.pack(fill=tk.X, padx=20)
        ttk.Label(jagged_frame, text="Jaggedness (Detail):",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.lightning_jaggedness_label = ttk.Label(jagged_frame, text="3",
                 font=('Arial', 8, 'bold'))
        self.lightning_jaggedness_label.pack(side=tk.RIGHT)

        self.lightning_jaggedness = tk.IntVar(value=3)
        ttk.Scale(lightning_frame, from_=2, to=6, orient=tk.HORIZONTAL,
                 variable=self.lightning_jaggedness,
                 command=lambda v: self.update_lightning_labels()).pack(fill=tk.X, padx=20)

        # Bolt length with value label
        length_frame = ttk.Frame(lightning_frame)
        length_frame.pack(fill=tk.X, padx=20)
        ttk.Label(length_frame, text="Bolt Length:",
                 font=('Arial', 8)).pack(side=tk.LEFT)
        self.lightning_length_label = ttk.Label(length_frame, text="0.30",
                 font=('Arial', 8, 'bold'))
        self.lightning_length_label.pack(side=tk.RIGHT)

        self.lightning_length = tk.DoubleVar(value=0.3)
        ttk.Scale(lightning_frame, from_=0.1, to=0.8, orient=tk.HORIZONTAL,
                 variable=self.lightning_length,
                 command=lambda v: self.update_lightning_labels()).pack(fill=tk.X, padx=20)
        
        # === ACTION BUTTONS ===
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1

        self.apply_btn = ttk.Button(button_frame, text="Apply & Generate",
                                    command=self.apply_parameters,
                                    style='Accent.TButton')
        self.apply_btn.pack(fill=tk.X, pady=2)

        self.play_btn = ttk.Button(button_frame, text="▶ Play Audio",
                                   command=self.toggle_playback)
        self.play_btn.pack(fill=tk.X, pady=2)

        ttk.Button(button_frame, text="Reset Effects",
                  command=self.reset_effects).pack(fill=tk.X, pady=2)

        ttk.Button(button_frame, text="Save to WAV",
                  command=self.save_to_wav).pack(fill=tk.X, pady=2)
        
        # === STATUS ===
        status_frame = ttk.LabelFrame(parent, text="Status", padding="5")
        status_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        self.status_label = ttk.Label(status_frame, text="Ready", 
                                      wraplength=200, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X)
        
        # Live Preview Toggle
        ttk.Separator(status_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        ttk.Label(status_frame, text="Display Mode:",
                 font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(5,2))

        self.live_preview_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(status_frame, text="Live Preview",
                       variable=self.live_preview_var,
                       command=self.toggle_live_preview).pack(anchor=tk.W)

        ttk.Label(status_frame, text="Shows real-time output during playback",
                 font=('Arial', 7, 'italic'),
                 foreground='gray').pack(anchor=tk.W, padx=(20,0))

        # Preview window size control
        preview_control_frame = ttk.Frame(status_frame)
        preview_control_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Label(preview_control_frame, text="Window Size:",
                 font=('Arial', 8)).pack(side=tk.LEFT, padx=(15,5))
        self.preview_size_var = tk.IntVar(value=5000)
        self.preview_size_spin = ttk.Spinbox(preview_control_frame, from_=100, to=50000,
                                       width=8, textvariable=self.preview_size_var,
                                       command=self.update_preview_size)
        self.preview_size_spin.pack(side=tk.LEFT)
        ttk.Label(preview_control_frame, text="samples",
                 font=('Arial', 8)).pack(side=tk.LEFT, padx=(5,0))

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

        # Initial plot
        self.line, = self.ax.plot([], [], color='#00ff00', linewidth=1.5, alpha=0.8)

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

    def update_lightning_labels_only(self):
        """Update lightning effect value labels without triggering effect_changed"""
        count = self.lightning_count.get()
        jaggedness = self.lightning_jaggedness.get()
        length = self.lightning_length.get()

        self.lightning_count_label.config(text=f"{count}")
        self.lightning_jaggedness_label.config(text=f"{jaggedness}")
        self.lightning_length_label.config(text=f"{length:.2f}")

    def update_lightning_labels(self):
        """Update lightning effect value labels and trigger effect preview"""
        self.update_lightning_labels_only()
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
        self.lightning_var.set(False)

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
        self.lightning_mode_var.set("Static")
        self.lightning_count.set(4)
        self.lightning_jaggedness.set(3)
        self.lightning_length.set(0.3)

        # Update wavy and lightning labels to reflect reset values
        self.update_wavy_labels_only()
        self.update_lightning_labels_only()

        # Update display
        self.update_display()

        # If audio exists, regenerate with reset effects
        if hasattr(self, 'current_audio') and self.current_audio is not None:
            self.apply_parameters()
    
    def apply_effects(self, x, y):
        """Apply selected effects to the data - FOR DISPLAY PREVIEW ONLY - ALL EFFECTS BLEND"""
        x_norm = x.copy()
        y_norm = y.copy()

        # Determine which effects are enabled
        has_y_fade = self.y_fade_var.get()
        has_x_fade = self.x_fade_var.get()
        has_shrink = self.shrink_var.get()
        rotation_mode = self.rotation_mode_var.get()

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
        if self.reflections_var.get():
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
        if self.x_wavy_var.get() or self.y_wavy_var.get():
            # Create time array based on position (0 to 2π)
            t = np.linspace(0, 2*np.pi, len(x))

            if self.x_wavy_var.get():
                K_x = self.x_wavy_amp.get()
                w_x = self.x_wavy_freq.get()
                x = x + K_x * np.sin(w_x * t)

            if self.y_wavy_var.get():
                K_y = self.y_wavy_amp.get()
                w_y = self.y_wavy_freq.get()
                y = y + K_y * np.sin(w_y * t)

        # Apply lightning bolts if enabled
        if self.lightning_var.get():
            num_bolts = self.lightning_count.get()
            jaggedness = self.lightning_jaggedness.get()
            length = self.lightning_length.get()
            is_dynamic = self.lightning_mode_var.get() == "Dynamic"

            # For static preview, use frame 0; for dynamic, use time-based index
            import time
            frame_idx = int(time.time() * 5) if is_dynamic else 0

            # Use current x, y (after effects) for bounding box calculation
            x_lightning, y_lightning = self.generate_lightning_bolts(
                x, y, num_bolts, jaggedness, length,
                dynamic=is_dynamic, frame_idx=frame_idx
            )

            # Don't normalize lightning separately - keep it in same coordinate space
            if len(x_lightning) > 0:
                # Prepend lightning (so it's drawn first as background)
                x = np.concatenate([x_lightning, x])
                y = np.concatenate([y_lightning, y])

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
    
    def update_live_preview(self):
        """Update display to show current audio output position as a continuous stream"""
        try:
            if self.preview_active and self.is_playing and self.current_audio is not None:
                try:
                    import time
                    if hasattr(self, 'playback_start_time'):
                        # Calculate current playback position
                        elapsed = time.time() - self.playback_start_time
                        sample_position = int(elapsed * self.current_fs)

                        # Wrap around if we exceed audio length (looping)
                        total_samples = len(self.current_audio)
                        if sample_position >= total_samples:
                            sample_position = sample_position % total_samples
                            # Reset buffer on loop
                            self.preview_buffer = []
                            self.last_preview_update = sample_position
                            # Update start time for smoother looping
                            self.playback_start_time = time.time() - (sample_position / self.current_fs)

                        # Add new samples to rolling buffer since last update
                        if sample_position > self.last_preview_update:
                            new_samples = self.current_audio[self.last_preview_update:sample_position]
                            if len(self.preview_buffer) == 0:
                                self.preview_buffer = new_samples.tolist()
                            else:
                                self.preview_buffer.extend(new_samples.tolist())
                            self.last_preview_update = sample_position

                            # Keep buffer at desired window size (remove old samples)
                            if len(self.preview_buffer) > self.preview_window_size:
                                self.preview_buffer = self.preview_buffer[-self.preview_window_size:]

                        # Display the rolling buffer
                        if len(self.preview_buffer) >= 10:
                            buffer_array = np.array(self.preview_buffer)
                            x_preview = buffer_array[:, 0]
                            y_preview = buffer_array[:, 1]

                            # Update plot
                            self.line.set_data(x_preview, y_preview)

                            # Keep consistent axis limits for smooth viewing
                            self.ax.set_xlim(-1.2, 1.2)
                            self.ax.set_ylim(-1.2, 1.2)

                            self.canvas.draw_idle()
                except Exception as e:
                    pass  # Silently ignore preview errors

            # Schedule next update (50 FPS)
            if self.root.winfo_exists():
                self.root.after(20, self.update_live_preview)
        except Exception:
            pass  # Window destroyed, stop scheduling
    
    def update_display(self):
        """Update the oscilloscope display"""
        # Normalize data
        x_norm = self.normalize_data(self.x_data)
        y_norm = self.normalize_data(self.y_data)

        # Apply effects
        x_display, y_display = self.apply_effects(x_norm, y_norm)

        # Repeat pattern for visibility
        display_repeats = min(20, max(1, 100 // len(x_norm)))
        x_display = np.tile(x_display, display_repeats)
        y_display = np.tile(y_display, display_repeats)

        # Downsample if too many points for rendering (prevents matplotlib overflow)
        max_display_points = 50000
        if len(x_display) > max_display_points:
            # Downsample by taking every nth point
            step = len(x_display) // max_display_points
            x_display = x_display[::step]
            y_display = y_display[::step]

        # Update plot
        self.line.set_data(x_display, y_display)
        
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
        if self.reflections_var.get():
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

        # Add lightning bolts if enabled (before tiling)
        if self.lightning_var.get():
            num_bolts = self.lightning_count.get()
            jaggedness = self.lightning_jaggedness.get()
            length = self.lightning_length.get()
            is_dynamic = self.lightning_mode_var.get() == "Dynamic"

            if is_dynamic:
                # For dynamic mode, generate multiple variations
                num_variations = 5
                lightning_variations_x = []
                lightning_variations_y = []

                for variation_idx in range(num_variations):
                    x_lightning, y_lightning = self.generate_lightning_bolts(
                        x_repeated, y_repeated, num_bolts, jaggedness, length,
                        dynamic=True, frame_idx=variation_idx
                    )
                    if len(x_lightning) > 0:
                        # Keep lightning in same coordinate space as pattern
                        lightning_variations_x.append(x_lightning)
                        lightning_variations_y.append(y_lightning)

                # Combine pattern with lightning for each variation
                if lightning_variations_x:
                    # Prepend lightning to pattern (so pattern draws on top)
                    combined_x = []
                    combined_y = []
                    for i in range(num_variations):
                        combined_x.append(lightning_variations_x[i])
                        combined_x.append(x_repeated)
                        combined_y.append(lightning_variations_y[i])
                        combined_y.append(y_repeated)

                    x_repeated = np.concatenate(combined_x)
                    y_repeated = np.concatenate(combined_y)
            else:
                # For static mode, generate once and prepend
                x_lightning, y_lightning = self.generate_lightning_bolts(
                    x_repeated, y_repeated, num_bolts, jaggedness, length,
                    dynamic=False, frame_idx=0
                )

                if len(x_lightning) > 0:
                    # Keep lightning in same coordinate space as pattern
                    # Prepend lightning to pattern (so pattern draws on top)
                    x_repeated = np.concatenate([x_lightning, x_repeated])
                    y_repeated = np.concatenate([y_lightning, y_repeated])

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

    def generate_lightning_bolt(self, x_start, y_start, x_end, y_end, jaggedness, seed=None):
        """
        Generate a single lightning bolt using recursive midpoint displacement
        """
        if seed is not None:
            np.random.seed(seed)

        # Create initial line from start to end
        points = [(x_start, y_start), (x_end, y_end)]

        # Recursively subdivide the line with random displacement
        for iteration in range(jaggedness):
            new_points = [points[0]]
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]

                # Calculate midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                # Calculate displacement perpendicular to the line
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)

                # Perpendicular vector (normalized)
                if length > 0:
                    perp_x = -dy / length
                    perp_y = dx / length
                else:
                    perp_x, perp_y = 0, 0

                # Random displacement (decreases with each iteration)
                displacement = (np.random.random() - 0.5) * length * 0.5 / (2 ** iteration)

                # Apply displacement
                mid_x += perp_x * displacement
                mid_y += perp_y * displacement

                new_points.append((mid_x, mid_y))
                new_points.append(points[i + 1])

            points = new_points

        # Convert to arrays
        x_bolt = np.array([p[0] for p in points])
        y_bolt = np.array([p[1] for p in points])

        return x_bolt, y_bolt

    def generate_lightning_bolts(self, x_data, y_data, num_bolts, jaggedness, length, dynamic=False, frame_idx=0):
        """
        Generate multiple lightning bolts emanating from the outer edges of the image
        """
        # Find bounding box of current data
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

        # Expand bounding box slightly
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin = 0.05

        x_min -= x_range * margin
        x_max += x_range * margin
        y_min -= y_range * margin
        y_max += y_range * margin

        all_x_bolts = []
        all_y_bolts = []

        # Generate bolts from different edge positions
        for i in range(num_bolts):
            # Seed for deterministic static bolts or random for dynamic
            if dynamic:
                # Add frame index to seed for animation
                seed = i * 1000 + frame_idx
            else:
                seed = i + 42  # Fixed seed for static mode

            # Set seed before any random operations
            if seed is not None:
                np.random.seed(seed)

            # Choose which edge (top, bottom, left, right)
            edge = i % 4
            position = (i / num_bolts) + (frame_idx * 0.1 if dynamic else 0)

            # Random offset for endpoint (now uses seeded random)
            rand_offset = (np.random.random() - 0.5) * 2

            if edge == 0:  # Top edge
                x_start = x_min + (x_max - x_min) * (position % 1)
                y_start = y_max
                x_end = x_start + rand_offset * length * x_range
                y_end = y_max + length * y_range
            elif edge == 1:  # Bottom edge
                x_start = x_min + (x_max - x_min) * (position % 1)
                y_start = y_min
                x_end = x_start + rand_offset * length * x_range
                y_end = y_min - length * y_range
            elif edge == 2:  # Left edge
                x_start = x_min
                y_start = y_min + (y_max - y_min) * (position % 1)
                x_end = x_min - length * x_range
                y_end = y_start + rand_offset * length * y_range
            else:  # Right edge
                x_start = x_max
                y_start = y_min + (y_max - y_min) * (position % 1)
                x_end = x_max + length * x_range
                y_end = y_start + rand_offset * length * y_range

            # Generate the bolt (seed already set above)
            x_bolt, y_bolt = self.generate_lightning_bolt(
                x_start, y_start, x_end, y_end, jaggedness, seed
            )

            all_x_bolts.append(x_bolt)
            all_y_bolts.append(y_bolt)

        # Combine all bolts
        if all_x_bolts:
            x_lightning = np.concatenate(all_x_bolts)
            y_lightning = np.concatenate(all_y_bolts)
        else:
            x_lightning = np.array([])
            y_lightning = np.array([])

        return x_lightning, y_lightning

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
        
        # Reset preview to static view if live preview is off
        if not self.preview_active:
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

        ttk.Label(dialog, text="Choose a parametric pattern:",
                 font=('Arial', 10, 'bold')).pack(pady=10)

        # Create frame for scrollable area
        scroll_container = ttk.Frame(dialog)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Create canvas with scrollbar
        canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

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
            "Complex Curves": ["Hypotrochoid", "Epitrochoid", "Butterfly Curve",
                              "Cardioid", "Deltoid"],
            "Special": ["Figure-8", "Infinity (∞)"]
        }

        for category, pattern_names in categories.items():
            ttk.Label(scrollable_frame, text=category,
                     font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=10, pady=(10, 5))
            for name in pattern_names:
                ttk.Radiobutton(scrollable_frame, text=name, variable=selected,
                               value=name).pack(anchor=tk.W, padx=30)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)

        # Button frame at bottom (always visible)
        button_frame = ttk.Frame(dialog)
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

        ttk.Button(button_frame, text="Generate Pattern", command=apply).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

    def open_drawing_canvas(self):
        """Open a canvas for drawing custom patterns"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Draw Pattern")
        dialog.geometry("550x650")

        # Instructions
        instruction_frame = ttk.Frame(dialog)
        instruction_frame.pack(pady=10)
        ttk.Label(instruction_frame, text="Draw your pattern below:",
                 font=('Arial', 10, 'bold')).pack()
        ttk.Label(instruction_frame, text="Click and drag to draw • The path will be traced in order",
                 font=('Arial', 8), foreground='gray').pack()

        # Drawing canvas (square aspect ratio to match oscilloscope display)
        canvas_frame = ttk.Frame(dialog)
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
            drawing_data['points'] = [(event.x, event.y)]

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
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Clear", command=clear_canvas).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply Drawing", command=apply_drawing).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.LEFT, padx=5)

    def open_harmonic_sum(self):
        """Open dialog to create pattern from sum of harmonics"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Sum of Harmonics")
        dialog.geometry("700x600")

        # Instructions
        instruction_frame = ttk.Frame(dialog)
        instruction_frame.pack(pady=10, padx=10, fill=tk.X)
        ttk.Label(instruction_frame, text="Create pattern from sum of sinusoidal terms",
                 font=('Arial', 10, 'bold')).pack()
        ttk.Label(instruction_frame, text="X(t) = Σ A_n·sin(ω_n·t + φ_n)  or  A_n·cos(ω_n·t + φ_n)",
                 font=('Arial', 8), foreground='gray').pack()

        # Main container with two columns
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Storage for terms
        x_terms = []  # List of dicts: {'type': 'sin'/'cos', 'amp': float, 'freq': float, 'phase': float}
        y_terms = []

        # X Channel (Left)
        x_frame = ttk.LabelFrame(main_frame, text="X Channel (Left)", padding="10")
        x_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # X terms display
        x_canvas = tk.Canvas(x_frame, height=300, highlightthickness=1, highlightbackground='gray')
        x_scrollbar = ttk.Scrollbar(x_frame, orient="vertical", command=x_canvas.yview)
        x_scrollable = ttk.Frame(x_canvas)

        x_scrollable.bind("<Configure>", lambda e: x_canvas.configure(scrollregion=x_canvas.bbox("all")))
        x_canvas.create_window((0, 0), window=x_scrollable, anchor="nw")
        x_canvas.configure(yscrollcommand=x_scrollbar.set)

        x_canvas.pack(side="left", fill="both", expand=True)
        x_scrollbar.pack(side="right", fill="y")

        # Y Channel (Right)
        y_frame = ttk.LabelFrame(main_frame, text="Y Channel (Right)", padding="10")
        y_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Y terms display
        y_canvas = tk.Canvas(y_frame, height=300, highlightthickness=1, highlightbackground='gray')
        y_scrollbar = ttk.Scrollbar(y_frame, orient="vertical", command=y_canvas.yview)
        y_scrollable = ttk.Frame(y_canvas)

        y_scrollable.bind("<Configure>", lambda e: y_canvas.configure(scrollregion=y_canvas.bbox("all")))
        y_canvas.create_window((0, 0), window=y_scrollable, anchor="nw")
        y_canvas.configure(yscrollcommand=y_scrollbar.set)

        y_canvas.pack(side="left", fill="both", expand=True)
        y_scrollbar.pack(side="right", fill="y")

        def refresh_x_display():
            """Refresh X channel term display"""
            for widget in x_scrollable.winfo_children():
                widget.destroy()

            if not x_terms:
                ttk.Label(x_scrollable, text="No terms added yet",
                         font=('Arial', 8, 'italic'), foreground='gray').pack(pady=10)
            else:
                for i, term in enumerate(x_terms):
                    term_frame = ttk.Frame(x_scrollable, relief='solid', borderwidth=1)
                    term_frame.pack(fill=tk.X, pady=2, padx=2)

                    # Term label
                    term_text = f"{term['amp']:.2f}·{term['type']}({term['freq']:.1f}·t"
                    if term['phase'] != 0:
                        term_text += f" + {term['phase']:.2f}"
                    term_text += ")"
                    ttk.Label(term_frame, text=f"Term {i+1}: {term_text}",
                             font=('Arial', 8)).pack(side=tk.LEFT, padx=5, pady=2)

                    # Remove button
                    def remove_x_term(idx=i):
                        x_terms.pop(idx)
                        refresh_x_display()

                    ttk.Button(term_frame, text="✕", width=3,
                              command=remove_x_term).pack(side=tk.RIGHT, padx=2, pady=2)

        def refresh_y_display():
            """Refresh Y channel term display"""
            for widget in y_scrollable.winfo_children():
                widget.destroy()

            if not y_terms:
                ttk.Label(y_scrollable, text="No terms added yet",
                         font=('Arial', 8, 'italic'), foreground='gray').pack(pady=10)
            else:
                for i, term in enumerate(y_terms):
                    term_frame = ttk.Frame(y_scrollable, relief='solid', borderwidth=1)
                    term_frame.pack(fill=tk.X, pady=2, padx=2)

                    # Term label
                    term_text = f"{term['amp']:.2f}·{term['type']}({term['freq']:.1f}·t"
                    if term['phase'] != 0:
                        term_text += f" + {term['phase']:.2f}"
                    term_text += ")"
                    ttk.Label(term_frame, text=f"Term {i+1}: {term_text}",
                             font=('Arial', 8)).pack(side=tk.LEFT, padx=5, pady=2)

                    # Remove button
                    def remove_y_term(idx=i):
                        y_terms.pop(idx)
                        refresh_y_display()

                    ttk.Button(term_frame, text="✕", width=3,
                              command=remove_y_term).pack(side=tk.RIGHT, padx=2, pady=2)

        # Add term controls for X
        x_add_frame = ttk.LabelFrame(x_frame, text="Add Term", padding="5")
        x_add_frame.pack(fill=tk.X, pady=(10, 0))

        x_type_var = tk.StringVar(value="sin")
        ttk.Radiobutton(x_add_frame, text="sin", variable=x_type_var, value="sin").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(x_add_frame, text="cos", variable=x_type_var, value="cos").pack(side=tk.LEFT, padx=2)

        ttk.Label(x_add_frame, text="A:").pack(side=tk.LEFT, padx=(10, 2))
        x_amp_var = tk.DoubleVar(value=1.0)
        ttk.Entry(x_add_frame, textvariable=x_amp_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(x_add_frame, text="ω:").pack(side=tk.LEFT, padx=(5, 2))
        x_freq_var = tk.DoubleVar(value=1.0)
        ttk.Entry(x_add_frame, textvariable=x_freq_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(x_add_frame, text="φ:").pack(side=tk.LEFT, padx=(5, 2))
        x_phase_var = tk.DoubleVar(value=0.0)
        ttk.Entry(x_add_frame, textvariable=x_phase_var, width=6).pack(side=tk.LEFT, padx=2)

        def add_x_term():
            x_terms.append({
                'type': x_type_var.get(),
                'amp': x_amp_var.get(),
                'freq': x_freq_var.get(),
                'phase': x_phase_var.get()
            })
            refresh_x_display()

        ttk.Button(x_add_frame, text="Add", command=add_x_term).pack(side=tk.LEFT, padx=5)

        # Add term controls for Y
        y_add_frame = ttk.LabelFrame(y_frame, text="Add Term", padding="5")
        y_add_frame.pack(fill=tk.X, pady=(10, 0))

        y_type_var = tk.StringVar(value="sin")
        ttk.Radiobutton(y_add_frame, text="sin", variable=y_type_var, value="sin").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(y_add_frame, text="cos", variable=y_type_var, value="cos").pack(side=tk.LEFT, padx=2)

        ttk.Label(y_add_frame, text="A:").pack(side=tk.LEFT, padx=(10, 2))
        y_amp_var = tk.DoubleVar(value=1.0)
        ttk.Entry(y_add_frame, textvariable=y_amp_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(y_add_frame, text="ω:").pack(side=tk.LEFT, padx=(5, 2))
        y_freq_var = tk.DoubleVar(value=1.0)
        ttk.Entry(y_add_frame, textvariable=y_freq_var, width=6).pack(side=tk.LEFT, padx=2)

        ttk.Label(y_add_frame, text="φ:").pack(side=tk.LEFT, padx=(5, 2))
        y_phase_var = tk.DoubleVar(value=0.0)
        ttk.Entry(y_add_frame, textvariable=y_phase_var, width=6).pack(side=tk.LEFT, padx=2)

        def add_y_term():
            y_terms.append({
                'type': y_type_var.get(),
                'amp': y_amp_var.get(),
                'freq': y_freq_var.get(),
                'phase': y_phase_var.get()
            })
            refresh_y_display()

        ttk.Button(y_add_frame, text="Add", command=add_y_term).pack(side=tk.LEFT, padx=5)

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

            # Calculate X channel
            if x_terms:
                x_data = np.zeros(num_points)
                for term in x_terms:
                    if term['type'] == 'sin':
                        x_data += term['amp'] * np.sin(term['freq'] * t + term['phase'])
                    else:  # cos
                        x_data += term['amp'] * np.cos(term['freq'] * t + term['phase'])
            else:
                x_data = np.zeros(num_points)

            # Calculate Y channel
            if y_terms:
                y_data = np.zeros(num_points)
                for term in y_terms:
                    if term['type'] == 'sin':
                        y_data += term['amp'] * np.sin(term['freq'] * t + term['phase'])
                    else:  # cos
                        y_data += term['amp'] * np.cos(term['freq'] * t + term['phase'])
            else:
                y_data = np.zeros(num_points)

            # Set as current data
            self.x_data = x_data
            self.y_data = y_data
            self.data_info_label.config(text=f"Points: {len(x_data)}")
            self.update_display()
            self.status_label.config(text=f"Generated harmonic sum (N={len(x_terms)}, M={len(y_terms)})")
            dialog.destroy()

        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Generate Pattern", command=apply_harmonic_sum).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
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
    root = tk.Tk()
    app = OscilloscopeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

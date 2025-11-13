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
        self.preview_window_size = 500  # Number of samples to show at once
        self.preview_active = False
        
        # Create GUI
        self.create_widgets()
        self.update_display()
        
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
        ttk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.y_fade_steps,
                   command=self.effect_changed).pack(anchor=tk.W, padx=20)
        
        ttk.Separator(effects_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # X-Axis Fade Sequence
        self.x_fade_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(effects_frame, text="X-Axis Fade Sequence", 
                       variable=self.x_fade_var,
                       command=self.effect_changed).pack(anchor=tk.W)
        
        ttk.Label(effects_frame, text="X Fade Steps:", 
                 font=('Arial', 8)).pack(anchor=tk.W, padx=20)
        self.x_fade_steps = tk.IntVar(value=10)
        ttk.Spinbox(effects_frame, from_=2, to=50, width=8,
                   textvariable=self.x_fade_steps,
                   command=self.effect_changed).pack(anchor=tk.W, padx=20)
        
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
                 command=lambda v: self.update_display()).pack(fill=tk.X)
        
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
        self.live_preview_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(status_frame, text="Live Preview (matches audio output)", 
                       variable=self.live_preview_var,
                       command=self.toggle_live_preview).pack(anchor=tk.W)
    
    def create_display(self, parent):
        """Create matplotlib display"""
        
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
        """Handle effect changes - update display and regenerate if playing"""
        self.update_display()
        if self.is_playing:
            # Regenerate and restart playback
            try:
                self.stop_playback()
                self.generate_audio()
                self.start_playback()
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
    
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
        """Handle rotation mode changes - regenerate and play if needed"""
        self.update_rotation_info()
        self.update_display()
        if hasattr(self, 'auto_regenerate') and self.is_playing:
            self.apply_parameters()
    
    def apply_effects(self, x, y):
        """Apply selected effects to the data - FOR DISPLAY PREVIEW ONLY"""
        x = x.copy()
        y = y.copy()
        
        # Y-Axis Fade Sequence
        if self.y_fade_var.get():
            n_fade = self.y_fade_steps.get()
            # Fade from 1 to 0, then back from 0 to 1
            fade_down = np.linspace(1, 0, n_fade, dtype=np.float32)
            fade_up = np.linspace(0, 1, n_fade, dtype=np.float32)[1:]  # Skip 0 to avoid duplicate
            fade_factors = np.concatenate([fade_down, fade_up])
            
            # Tile x for both fade down and fade up
            total_fades = len(fade_factors)
            x = np.tile(x, total_fades)
            y = np.concatenate([y * fade_factors[i] for i in range(total_fades)])
        
        # X-Axis Fade Sequence
        if self.x_fade_var.get():
            n_fade = self.x_fade_steps.get()
            fade_factors = np.linspace(1, 0, n_fade, dtype=np.float32)
            y = np.tile(y, n_fade)
            x = np.concatenate([x * fade_factors[i] for i in range(n_fade)])
        
        # Mirror Reflections
        if self.reflections_var.get():
            x, y = self.apply_reflections(x, y)
        
        # Rotation - static angle only for display
        if self.rotation_mode_var.get() == "Static":
            angle_rad = np.radians(self.rotation_angle.get())
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            # Renormalize after rotation to prevent clipping
            x = self.normalize_data(x_rot)
            y = self.normalize_data(y_rot)
        
        return x, y
    
    def toggle_live_preview(self):
        """Toggle live preview mode"""
        self.preview_active = self.live_preview_var.get()
        if self.preview_active:
            self.preview_position = 0
    
    def update_live_preview(self):
        """Update display to show current audio output position"""
        if self.preview_active and self.is_playing and self.current_audio is not None:
            try:
                # Calculate approximate playback position
                import time
                if hasattr(self, 'playback_start_time'):
                    elapsed = time.time() - self.playback_start_time
                    sample_position = int(elapsed * self.current_fs)
                    
                    # Wrap around if we exceed audio length
                    if sample_position >= len(self.current_audio):
                        sample_position = sample_position % len(self.current_audio)
                    
                    # Extract window of samples to display (moving window)
                    window_half = self.preview_window_size // 2
                    window_start = max(0, sample_position - window_half)
                    window_end = min(len(self.current_audio), sample_position + window_half)
                    
                    # Make sure we have enough samples
                    if window_end - window_start < 50:
                        window_end = min(len(self.current_audio), window_start + self.preview_window_size)
                    
                    if window_end > window_start:
                        # Get the windowed data
                        x_preview = self.current_audio[window_start:window_end, 0]
                        y_preview = self.current_audio[window_start:window_end, 1]
                        
                        # Update plot
                        self.line.set_data(x_preview, y_preview)
                        
                        # Auto-scale to current window
                        if len(x_preview) > 0:
                            margin = 0.1
                            x_range = [x_preview.min() - margin, x_preview.max() + margin]
                            y_range = [y_preview.min() - margin, y_preview.max() + margin]
                            
                            max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
                            center_x = (x_range[0] + x_range[1]) / 2
                            center_y = (y_range[0] + y_range[1]) / 2
                            
                            self.ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
                            self.ax.set_ylim(center_y - max_range/2, center_y + max_range/2)
                        
                        self.canvas.draw_idle()
            except Exception as e:
                pass  # Silently ignore preview errors
        
        # Schedule next update
        self.root.after(20, self.update_live_preview)
    
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
        """Generate audio with current parameters and effects"""
        
        self.status_label.config(text="Generating audio...")
        self.root.update()
        
        # Normalize
        x_norm = self.normalize_data(self.x_data)
        y_norm = self.normalize_data(self.y_data)
        
        # Start with base pattern
        x_base = x_norm.copy()
        y_base = y_norm.copy()
        
        # Apply Y-Axis Fade Sequence if enabled
        if self.y_fade_var.get():
            n_fade = self.y_fade_steps.get()
            # Fade from 1 to 0, then back from 0 to 1
            fade_down = np.linspace(1, 0, n_fade, dtype=np.float32)
            fade_up = np.linspace(0, 1, n_fade, dtype=np.float32)[1:]  # Skip 0 to avoid duplicate
            fade_factors = np.concatenate([fade_down, fade_up])
            
            # Tile x for both fade down and fade up
            total_fades = len(fade_factors)
            x_base = np.tile(x_base, total_fades)
            y_base = np.concatenate([y_base * fade_factors[i] for i in range(total_fades)])
        
        # Apply X-Axis Fade Sequence if enabled
        if self.x_fade_var.get():
            n_fade = self.x_fade_steps.get()
            fade_factors = np.linspace(1, 0, n_fade, dtype=np.float32)
            y_base = np.tile(y_base, n_fade)
            x_base = np.concatenate([x_base * fade_factors[i] for i in range(n_fade)])
        
        # Apply Mirror Reflections if enabled
        if self.reflections_var.get():
            x_base, y_base = self.apply_reflections(x_base, y_base)
        
        # Get parameters
        fs = self.sample_rate_var.get()
        n_repeat = self.n_repeat_var.get()
        mult = self.freq_mult_var.get()
        duration = self.duration_var.get()
        
        # Handle rotation modes
        rotation_mode = self.rotation_mode_var.get()
        
        if rotation_mode == "Static":
            # Static rotation - apply once
            angle_rad = np.radians(self.rotation_angle.get())
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            x_rot = x_base * cos_a - y_base * sin_a
            y_rot = x_base * sin_a + y_base * cos_a
            
            # Renormalize to prevent clipping
            x_base = self.normalize_data(x_rot)
            y_base = self.normalize_data(y_rot)
            
            # Repeat the rotated pattern
            x_repeated = np.tile(x_base, n_repeat)
            y_repeated = np.tile(y_base, n_repeat)
            
        elif rotation_mode in ["CW", "CCW"]:
            # Dynamic rotation - create rotating sequence
            speed = self.rotation_speed.get()
            direction = -1 if rotation_mode == "CW" else 1  # CW is negative angle
            
            # Calculate how many repetitions needed for a full 360° rotation
            steps_per_rotation = int(360 / speed)
            
            # Use n_repeat to determine how many full rotations to do
            # Each full rotation takes steps_per_rotation repetitions
            total_steps = n_repeat
            
            # Create rotating sequence
            x_repeated = []
            y_repeated = []
            
            for i in range(total_steps):
                angle = direction * speed * i
                angle_rad = np.radians(angle)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                
                x_rot = x_base * cos_a - y_base * sin_a
                y_rot = x_base * sin_a + y_base * cos_a
                
                x_repeated.append(x_rot)
                y_repeated.append(y_rot)
            
            x_repeated = np.concatenate(x_repeated)
            y_repeated = np.concatenate(y_repeated)
            
            # Renormalize entire sequence to prevent clipping
            x_repeated = self.normalize_data(x_repeated)
            y_repeated = self.normalize_data(y_repeated)
            
            # Calculate info for user
            num_full_rotations = (total_steps * speed) / 360
            self.status_label.config(text=f"Generating {num_full_rotations:.1f} rotations...")
            
        else:
            # No rotation
            x_repeated = np.tile(x_base, n_repeat)
            y_repeated = np.tile(y_base, n_repeat)
        
        # Calculate playback rate and target length
        actual_fs = fs * mult
        target_length = int(actual_fs * duration)
        
        # Tile to fill duration
        seq_len = len(x_repeated)
        num_tiles = int(np.ceil(target_length / seq_len))
        
        x_full = np.tile(x_repeated, num_tiles)[:target_length]
        y_full = np.tile(y_repeated, num_tiles)[:target_length]
        
        # Create stereo
        stereo = np.column_stack([x_full, y_full]).astype(np.float32)
        
        self.current_audio = stereo
        self.current_fs = actual_fs
        
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
        try:
            # Stop current playback if playing
            if self.is_playing:
                self.stop_playback()
            
            # Generate new audio
            self.generate_audio()
            
            # Auto-play the generated audio
            self.start_playback()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate audio:\n{str(e)}")
    
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
        self.status_label.config(text="Playing...")
        
        # Track playback start time for live preview
        import time
        self.playback_start_time = time.time()
        self.preview_position = 0
        
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
        """Thread function for playing audio"""
        try:
            sd.play(self.current_audio, self.current_fs, blocking=True)
            if not self.stop_flag.is_set():
                self.update_queue.put(("playback_complete", None))
        except Exception as e:
            self.update_queue.put(("error", str(e)))
    
    def check_updates(self):
        """Check for updates from audio thread"""
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
        
        self.root.after(50, self.check_updates)
    
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
        """Generate a test pattern"""
        patterns = {
            "Circle": lambda t: (np.cos(t), np.sin(t)),
            "Lissajous 3:2": lambda t: (np.sin(3*t), np.sin(2*t)),
            "Star": lambda t: (np.cos(t) * (1 + 0.5*np.sin(5*t)), 
                              np.sin(t) * (1 + 0.5*np.sin(5*t))),
            "Spiral": lambda t: (t/10*np.cos(t), t/10*np.sin(t)),
        }
        
        # Simple dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Test Pattern")
        dialog.geometry("250x200")
        
        ttk.Label(dialog, text="Choose a pattern:").pack(pady=10)
        
        selected = tk.StringVar(value="Circle")
        
        for name in patterns.keys():
            ttk.Radiobutton(dialog, text=name, variable=selected, 
                           value=name).pack(anchor=tk.W, padx=20)
        
        def apply():
            t = np.linspace(0, 4*np.pi, 500)
            x, y = patterns[selected.get()](t)
            self.x_data = x
            self.y_data = y
            self.data_info_label.config(text=f"Points: {len(x)}")
            self.update_display()
            self.status_label.config(text=f"Generated {selected.get()} pattern")
            dialog.destroy()
        
        ttk.Button(dialog, text="Generate", command=apply).pack(pady=20)
    
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

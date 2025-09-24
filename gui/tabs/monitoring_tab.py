"""Monitoring tab for training progress."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.style as mplstyle
from collections import deque


class MonitoringTab:
    """Tab for monitoring training progress."""

    def __init__(self, parent, start_callback: Callable, stop_callback: Callable, pause_callback: Callable):
        """Initialize monitoring tab."""
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.pause_callback = pause_callback
        self.frame = ttk.Frame(parent)

        # Initialize data storage for plots
        self.loss_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.steps_history = deque(maxlen=1000)

        # Theme colors for charts
        self.chart_themes = {
            'light': {
                'bg': '#ffffff',
                'fg': '#000000',
                'grid': '#e0e0e0',
                'line_color': '#0078d7',
                'line2_color': '#28a745',
                'alpha': 0.7
            },
            'dark': {
                'bg': '#2b2b2b',
                'fg': '#e0e0e0',
                'grid': '#404040',
                'line_color': '#00bfff',
                'line2_color': '#00ff7f',
                'alpha': 0.8
            },
            'synthwave': {
                'bg': '#241b2f',
                'fg': '#00ffff',
                'grid': '#372844',
                'line_color': '#ff00ff',
                'line2_color': '#00ffff',
                'alpha': 0.9
            }
        }
        self.current_theme = 'light'

        self._create_widgets()

    def _create_widgets(self):
        """Create tab widgets."""
        # Main container with scrollbar
        canvas = tk.Canvas(self.frame)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel for scrolling - only when mouse is over this canvas
        def _on_mousewheel(event):
            # Check if the mouse is over this canvas
            try:
                widget = self.frame.winfo_containing(event.x_root, event.y_root)
                if widget:
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except:
                pass

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        # Bind when mouse enters and unbind when it leaves
        scrollable_frame.bind("<Enter>", _bind_mousewheel)
        scrollable_frame.bind("<Leave>", _unbind_mousewheel)

        # Control panel
        control_frame = ttk.Frame(scrollable_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 5))  # No padding on top, 5 on bottom

        self.start_button = ttk.Button(control_frame, text="Start Training", command=self.start_callback)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_callback, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = ttk.Button(control_frame, text="Pause", command=self.pause_callback, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, length=300, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=20)

        self.progress_label = ttk.Label(control_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Metrics display
        metrics_frame = ttk.LabelFrame(scrollable_frame, text="Training Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create matplotlib figure with theming
        self.figure = Figure(figsize=(12, 5), facecolor='#f0f0f0')
        self.figure.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12, wspace=0.25)

        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)

        # Apply initial theme
        self._apply_chart_theme()

        self.canvas = FigureCanvasTkAgg(self.figure, metrics_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add chart controls
        controls_frame = ttk.Frame(metrics_frame)
        controls_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(controls_frame, text="Clear Charts", command=self.clear_charts).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Charts", command=self.export_charts).pack(side=tk.LEFT, padx=5)

        self.auto_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Auto Scale", variable=self.auto_scale_var).pack(side=tk.LEFT, padx=10)

        # Log display
        log_frame = ttk.LabelFrame(scrollable_frame, text="Training Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scroll.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def update_progress(self, value: float):
        """Update progress bar."""
        self.progress['value'] = value * 100
        self.progress_label.config(text=f"{int(value * 100)}%")

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics display."""
        # Add data to history
        step = metrics.get('step', len(self.steps_history))
        loss = metrics.get('loss', 0)
        reward = metrics.get('reward', 0)

        self.steps_history.append(step)
        self.loss_history.append(loss)
        self.reward_history.append(reward)

        # Update plots
        self._update_plots()

    def _apply_chart_theme(self, theme_name: str = None):
        """Apply theme to matplotlib charts."""
        if theme_name:
            self.current_theme = theme_name

        theme = self.chart_themes[self.current_theme]

        # Set figure background
        self.figure.patch.set_facecolor(theme['bg'])

        # Configure axes for both plots
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor(theme['bg'])
            ax.tick_params(colors=theme['fg'])
            ax.xaxis.label.set_color(theme['fg'])
            ax.yaxis.label.set_color(theme['fg'])
            ax.title.set_color(theme['fg'])
            ax.spines['bottom'].set_color(theme['fg'])
            ax.spines['top'].set_color(theme['fg'])
            ax.spines['right'].set_color(theme['fg'])
            ax.spines['left'].set_color(theme['fg'])
            ax.grid(True, alpha=0.3, color=theme['grid'], linestyle='-', linewidth=0.5)

        # Set titles with proper styling
        self.ax1.set_title("Training Loss", fontsize=12, fontweight='bold', color=theme['fg'])
        self.ax1.set_xlabel("Step", fontsize=10, color=theme['fg'])
        self.ax1.set_ylabel("Loss", fontsize=10, color=theme['fg'])

        self.ax2.set_title("Average Reward", fontsize=12, fontweight='bold', color=theme['fg'])
        self.ax2.set_xlabel("Step", fontsize=10, color=theme['fg'])
        self.ax2.set_ylabel("Reward", fontsize=10, color=theme['fg'])

        # Only update plots if we have data and not called from _update_plots
        if self.steps_history and not getattr(self, '_updating_plots', False):
            self._update_plots()

    def _update_plots(self):
        """Update the plot data."""
        if not self.steps_history:
            return

        # Set flag to prevent recursion
        self._updating_plots = True

        theme = self.chart_themes[self.current_theme]

        # Clear and redraw loss plot
        self.ax1.clear()
        self.ax1.set_facecolor(theme['bg'])
        self.ax1.plot(self.steps_history, self.loss_history,
                     color=theme['line_color'], linewidth=2,
                     alpha=theme['alpha'], label='Loss')
        self.ax1.fill_between(self.steps_history, self.loss_history,
                             alpha=0.2, color=theme['line_color'])

        # Reapply labels and styling for ax1
        self.ax1.set_title("Training Loss", fontsize=12, fontweight='bold', color=theme['fg'])
        self.ax1.set_xlabel("Step", fontsize=10, color=theme['fg'])
        self.ax1.set_ylabel("Loss", fontsize=10, color=theme['fg'])
        self.ax1.grid(True, alpha=0.3, color=theme['grid'], linestyle='-', linewidth=0.5)
        self.ax1.tick_params(colors=theme['fg'])

        # Clear and redraw reward plot
        self.ax2.clear()
        self.ax2.set_facecolor(theme['bg'])
        self.ax2.plot(self.steps_history, self.reward_history,
                     color=theme['line2_color'], linewidth=2,
                     alpha=theme['alpha'], label='Reward')
        self.ax2.fill_between(self.steps_history, self.reward_history,
                             alpha=0.2, color=theme['line2_color'])

        # Reapply labels and styling for ax2
        self.ax2.set_title("Average Reward", fontsize=12, fontweight='bold', color=theme['fg'])
        self.ax2.set_xlabel("Step", fontsize=10, color=theme['fg'])
        self.ax2.set_ylabel("Reward", fontsize=10, color=theme['fg'])
        self.ax2.grid(True, alpha=0.3, color=theme['grid'], linestyle='-', linewidth=0.5)
        self.ax2.tick_params(colors=theme['fg'])

        # Auto scale if enabled
        if self.auto_scale_var.get():
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()

        # Redraw canvas
        self.canvas.draw()

        # Reset flag
        self._updating_plots = False

    def clear_charts(self):
        """Clear all chart data."""
        self.steps_history.clear()
        self.loss_history.clear()
        self.reward_history.clear()

        theme = self.chart_themes[self.current_theme]

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        # Reapply basic styling without recursion
        self.ax1.set_facecolor(theme['bg'])
        self.ax1.set_title("Training Loss", fontsize=12, fontweight='bold', color=theme['fg'])
        self.ax1.set_xlabel("Step", fontsize=10, color=theme['fg'])
        self.ax1.set_ylabel("Loss", fontsize=10, color=theme['fg'])
        self.ax1.grid(True, alpha=0.3, color=theme['grid'], linestyle='-', linewidth=0.5)
        self.ax1.tick_params(colors=theme['fg'])

        self.ax2.set_facecolor(theme['bg'])
        self.ax2.set_title("Average Reward", fontsize=12, fontweight='bold', color=theme['fg'])
        self.ax2.set_xlabel("Step", fontsize=10, color=theme['fg'])
        self.ax2.set_ylabel("Reward", fontsize=10, color=theme['fg'])
        self.ax2.grid(True, alpha=0.3, color=theme['grid'], linestyle='-', linewidth=0.5)
        self.ax2.tick_params(colors=theme['fg'])

        self.canvas.draw()

    def export_charts(self):
        """Export charts to image file."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="Export Charts",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("PDF Files", "*.pdf"), ("SVG Files", "*.svg")]
        )
        if filename:
            self.figure.savefig(filename, dpi=150, bbox_inches='tight',
                              facecolor=self.figure.get_facecolor())

    def set_theme(self, theme_name: str):
        """Set the chart theme."""
        self._apply_chart_theme(theme_name)

    def add_log(self, log_entry: Dict[str, Any]):
        """Add log entry to display."""
        message = log_entry.get('message', '')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def set_training_state(self, is_training: bool):
        """Update button states based on training state."""
        if is_training:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)

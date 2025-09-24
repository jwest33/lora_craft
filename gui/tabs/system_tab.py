"""System configuration tab."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any


class SystemTab:
    """Tab for system configuration."""

    def __init__(self, parent, system_config, on_config_change: Callable):
        """Initialize system tab."""
        self.on_config_change = on_config_change
        self.system_config = system_config
        self.frame = ttk.Frame(parent)
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

        # System Info
        info_frame = ttk.LabelFrame(scrollable_frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 5))  # No padding on top, 5 on bottom

        info_text = tk.Text(info_frame, height=10, width=60)
        info_text.pack(padx=5, pady=5)
        info_text.insert(tk.END, self.system_config.get_system_summary())
        info_text.config(state=tk.DISABLED)

        # Optimization Settings
        opt_frame = ttk.LabelFrame(scrollable_frame, text="Optimization Settings", padding=10)
        opt_frame.pack(fill=tk.X, padx=10, pady=5)

        self.use_flash_attention = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Use Flash Attention", variable=self.use_flash_attention).pack(anchor=tk.W, padx=5, pady=2)

        self.gradient_checkpointing = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Gradient Checkpointing", variable=self.gradient_checkpointing).pack(anchor=tk.W, padx=5, pady=2)

        self.mixed_precision = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Mixed Precision (FP16)", variable=self.mixed_precision).pack(anchor=tk.W, padx=5, pady=2)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def get_config(self) -> Dict[str, Any]:
        """Get configuration from tab."""
        return {
            "use_flash_attention": self.use_flash_attention.get(),
            "gradient_checkpointing": self.gradient_checkpointing.get(),
            "mixed_precision": self.mixed_precision.get(),
        }

    def load_config(self, config: Dict[str, Any]):
        """Load configuration into tab."""
        if "use_flash_attention" in config:
            self.use_flash_attention.set(config["use_flash_attention"])
        if "gradient_checkpointing" in config:
            self.gradient_checkpointing.set(config["gradient_checkpointing"])
        if "mixed_precision" in config:
            self.mixed_precision.set(config["mixed_precision"])

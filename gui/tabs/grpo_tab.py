"""GRPO settings tab."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any


class GRPOTab:
    """Tab for GRPO-specific settings."""

    def __init__(self, parent, on_config_change: Callable):
        """Initialize GRPO tab."""
        self.on_config_change = on_config_change
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

        # Generation Parameters
        gen_frame = ttk.LabelFrame(scrollable_frame, text="Generation Parameters", padding=10)
        gen_frame.pack(fill=tk.X, padx=10, pady=(0, 5))  # No padding on top, 5 on bottom

        ttk.Label(gen_frame, text="Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.temperature = tk.DoubleVar(value=0.7)
        ttk.Scale(gen_frame, from_=0.1, to=2.0, variable=self.temperature, orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(gen_frame, textvariable=self.temperature).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(gen_frame, text="Top-p:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.top_p = tk.DoubleVar(value=0.95)
        ttk.Scale(gen_frame, from_=0.1, to=1.0, variable=self.top_p, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(gen_frame, text="Generations per Prompt:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.num_generations = tk.IntVar(value=4)
        ttk.Spinbox(gen_frame, from_=1, to=10, textvariable=self.num_generations, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # GRPO Parameters
        grpo_frame = ttk.LabelFrame(scrollable_frame, text="GRPO Parameters", padding=10)
        grpo_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(grpo_frame, text="KL Penalty:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.kl_penalty = tk.DoubleVar(value=0.01)
        ttk.Entry(grpo_frame, textvariable=self.kl_penalty, width=15).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(grpo_frame, text="Clip Range:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.clip_range = tk.DoubleVar(value=0.2)
        ttk.Entry(grpo_frame, textvariable=self.clip_range, width=15).grid(row=1, column=1, padx=5, pady=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def get_config(self) -> Dict[str, Any]:
        """Get configuration from tab."""
        return {
            "temperature": self.temperature.get(),
            "top_p": self.top_p.get(),
            "num_generations": self.num_generations.get(),
            "kl_penalty": self.kl_penalty.get(),
            "clip_range": self.clip_range.get(),
        }

    def load_config(self, config: Dict[str, Any]):
        """Load configuration into tab."""
        if "temperature" in config:
            self.temperature.set(config["temperature"])
        if "top_p" in config:
            self.top_p.set(config["top_p"])
        if "num_generations" in config:
            self.num_generations.set(config["num_generations"])
        if "kl_penalty" in config:
            self.kl_penalty.set(config["kl_penalty"])
        if "clip_range" in config:
            self.clip_range.set(config["clip_range"])

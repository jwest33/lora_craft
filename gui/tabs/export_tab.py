"""Export tab for model export functionality."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Dict, Any


class ExportTab:
    """Tab for model export."""

    def __init__(self, parent, export_callback: Callable):
        """Initialize export tab."""
        self.export_callback = export_callback
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

        # Export Format
        format_frame = ttk.LabelFrame(scrollable_frame, text="Export Format", padding=10)
        format_frame.pack(fill=tk.X, padx=10, pady=(0, 5))  # No padding on top, 5 on bottom

        ttk.Label(format_frame, text="Format:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_format = tk.StringVar(value="safetensors")
        formats = ["SafeTensors (16-bit)", "SafeTensors (4-bit)", "GGUF (Q4_K_M)", "GGUF (Q5_K_M)", "GGUF (Q8_0)", "HuggingFace"]
        ttk.Combobox(format_frame, textvariable=self.export_format, values=formats, width=30).grid(row=0, column=1, padx=5, pady=5)

        # Output Settings
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir = tk.StringVar(value="./exports")
        ttk.Entry(output_frame, textvariable=self.output_dir, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(output_frame, text="Model Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_name = tk.StringVar(value="my_model")
        ttk.Entry(output_frame, textvariable=self.model_name, width=40).grid(row=1, column=1, padx=5, pady=5)

        # HuggingFace Hub
        hub_frame = ttk.LabelFrame(scrollable_frame, text="HuggingFace Hub (Optional)", padding=10)
        hub_frame.pack(fill=tk.X, padx=10, pady=5)

        self.push_to_hub = tk.BooleanVar(value=False)
        ttk.Checkbutton(hub_frame, text="Push to HuggingFace Hub", variable=self.push_to_hub).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Label(hub_frame, text="Hub Model ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.hub_model_id = tk.StringVar()
        ttk.Entry(hub_frame, textvariable=self.hub_model_id, width=40, state=tk.DISABLED).grid(row=1, column=1, padx=5, pady=5)

        # Export button
        export_button = ttk.Button(scrollable_frame, text="Export Model", command=self._export_model)
        export_button.pack(pady=20)

        # Status
        self.status_label = ttk.Label(scrollable_frame, text="Ready to export")
        self.status_label.pack(pady=5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def _export_model(self):
        """Trigger model export."""
        config = {
            "format": self.export_format.get(),
            "output_dir": self.output_dir.get(),
            "model_name": self.model_name.get(),
            "push_to_hub": self.push_to_hub.get(),
            "hub_model_id": self.hub_model_id.get() if self.push_to_hub.get() else None,
        }
        
        self.status_label.config(text="Exporting...")
        self.export_callback()
        messagebox.showinfo("Export", "Model export started. Check the training tab for progress.")

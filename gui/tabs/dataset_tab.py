"""Dataset and prompts configuration tab."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Dict, Any, Optional
import json
import threading
import os
from pathlib import Path
from core.dataset_handler import DatasetHandler, DatasetConfig
from core.prompt_templates import PromptTemplate, TemplateConfig, TemplateLibrary


class DatasetTab:
    """Tab for dataset and prompt configuration."""

    def __init__(self, parent, on_config_change: Callable):
        """Initialize dataset tab.

        Args:
            parent: Parent widget
            on_config_change: Callback for configuration changes
        """
        self.on_config_change = on_config_change
        self.frame = ttk.Frame(parent)
        self.dataset_handler = None
        self.dataset = None
        self.template_library = TemplateLibrary()
        self.current_template = self.template_library.get("grpo")  # Get default GRPO template
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

        # Dataset Source Section
        source_frame = ttk.LabelFrame(scrollable_frame, text="Dataset Source", padding=10)
        source_frame.pack(fill=tk.X, padx=10, pady=(0, 5))  # No padding on top, 5 on bottom

        # Configure grid weights for better layout
        source_frame.columnconfigure(1, weight=1)  # Make column 1 expand

        # Row 0: Source type selection
        ttk.Label(source_frame, text="Source Type:").grid(row=0, column=0, sticky=tk.W, padx=(5, 10), pady=(5, 10))
        self.source_type = tk.StringVar(value="HuggingFace Hub")  # Fixed to match the list
        source_types = ["HuggingFace Hub", "Local File", "API Endpoint", "Direct Input"]
        source_menu = ttk.Combobox(source_frame, textvariable=self.source_type, values=source_types, width=25)
        source_menu.grid(row=0, column=1, sticky=tk.W, padx=(0, 10), pady=(5, 10))
        source_menu.bind("<<ComboboxSelected>>", self._on_source_type_change)

        # Row 1: Dataset selection
        ttk.Label(source_frame, text="Dataset:").grid(row=1, column=0, sticky=tk.W, padx=(5, 10), pady=(0, 10))

        # Dataset entry frame to group dataset input and buttons
        dataset_frame = ttk.Frame(source_frame)
        dataset_frame.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        dataset_frame.columnconfigure(0, weight=1)  # Make entry expand

        self.dataset_path = tk.StringVar(value="tatsu-lab/alpaca")

        # Default datasets for quick selection
        default_datasets = [
            "tatsu-lab/alpaca",
            "openai/gsm8k",
            "open-r1/DAPO-Math-17k-Processed",
            "nvidia/OpenMathReasoning"
        ]

        # Combobox that allows both dropdown selection and custom text input
        self.dataset_entry = ttk.Combobox(dataset_frame, textvariable=self.dataset_path,
                                         values=default_datasets, width=50)
        self.dataset_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.dataset_entry.bind("<<ComboboxSelected>>", self._on_dataset_selected)

        # Browse button - only visible for Local File
        self.browse_button = ttk.Button(dataset_frame, text="Browse", command=self._browse_dataset)
        self.browse_button.grid(row=0, column=1, padx=5)
        # Initially hide browse button since default is HuggingFace Hub
        self.browse_button.grid_remove()

        # Load button
        self.load_button = ttk.Button(dataset_frame, text="Load Dataset", command=self._load_dataset)
        self.load_button.grid(row=0, column=2, padx=(5, 0))

        # Row 2: Split and Options
        options_frame = ttk.Frame(source_frame)
        options_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 5))

        # Split selection
        ttk.Label(options_frame, text="Split:").grid(row=0, column=0, sticky=tk.W, padx=(5, 10))
        self.split = tk.StringVar(value="train")
        split_entry = ttk.Entry(options_frame, textvariable=self.split, width=15)
        split_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))

        # Force redownload checkbox with better styling
        self.force_redownload = tk.BooleanVar(value=False)
        force_cb = ttk.Checkbutton(options_frame, text="Force redownload from source",
                                   variable=self.force_redownload,
                                   style="TCheckbutton")
        force_cb.grid(row=0, column=2, sticky=tk.W, padx=(10, 5))

        # Field Mapping Section
        mapping_frame = ttk.LabelFrame(scrollable_frame, text="Field Mapping", padding=10)
        mapping_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(mapping_frame, text="Instruction Field:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.instruction_field = tk.StringVar(value="instruction")
        ttk.Entry(mapping_frame, textvariable=self.instruction_field, width=30).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(mapping_frame, text="Response Field:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.response_field = tk.StringVar(value="response")
        ttk.Entry(mapping_frame, textvariable=self.response_field, width=30).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(mapping_frame, text="System Field (optional):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.system_field = tk.StringVar()
        ttk.Entry(mapping_frame, textvariable=self.system_field, width=30).grid(row=2, column=1, padx=5, pady=5)

        # Data Preview Section
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Preview table
        columns = ("Index", "Instruction", "Response")
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=8)

        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=200 if col == "Index" else 400)

        preview_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=preview_scroll.set)

        self.preview_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Statistics
        self.stats_label = ttk.Label(preview_frame, text="No dataset loaded")
        self.stats_label.grid(row=1, column=0, pady=5)

        # Progress bar for loading
        self.progress_bar = ttk.Progressbar(preview_frame, mode='indeterminate', length=400)
        # Progress bar is not packed initially, will be shown during loading

        # GRPO Prompt Template Section
        template_frame = ttk.LabelFrame(scrollable_frame, text="GRPO Prompt Template (Unsloth-Compatible)", padding=10)
        template_frame.pack(fill=tk.X, padx=10, pady=5)

        # Model type selection (for model-specific GRPO templates)
        model_frame = ttk.Frame(template_frame)
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

        ttk.Label(model_frame, text="Model Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_type = tk.StringVar(value="Auto-Detect")
        model_types = ["Auto-Detect", "Qwen", "Llama", "Mistral", "ChatML"]
        self.model_menu = ttk.Combobox(model_frame, textvariable=self.model_type, values=model_types, width=20, state="readonly")
        self.model_menu.pack(side=tk.LEFT, padx=5)
        self.model_menu.bind("<<ComboboxSelected>>", self._on_model_type_changed)

        # Load GRPO template button
        self.load_grpo_button = ttk.Button(model_frame, text="Load GRPO Template", command=self._load_grpo_template)
        self.load_grpo_button.pack(side=tk.LEFT, padx=10)

        # Template info label
        self.template_info = ttk.Label(model_frame, text="Using default GRPO template", foreground="green")
        self.template_info.pack(side=tk.LEFT, padx=10)

        # GRPO Marker Configuration
        marker_frame = ttk.LabelFrame(template_frame, text="GRPO Markers", padding=5)
        marker_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)

        # Reasoning markers
        ttk.Label(marker_frame, text="Reasoning Start:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.reasoning_start = tk.StringVar(value="<start_working_out>")
        ttk.Entry(marker_frame, textvariable=self.reasoning_start, width=25).grid(row=0, column=1, padx=5, pady=3)

        ttk.Label(marker_frame, text="Reasoning End:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        self.reasoning_end = tk.StringVar(value="<end_working_out>")
        ttk.Entry(marker_frame, textvariable=self.reasoning_end, width=25).grid(row=0, column=3, padx=5, pady=3)

        # Solution markers
        ttk.Label(marker_frame, text="Solution Start:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.solution_start = tk.StringVar(value="<SOLUTION>")
        ttk.Entry(marker_frame, textvariable=self.solution_start, width=25).grid(row=1, column=1, padx=5, pady=3)

        ttk.Label(marker_frame, text="Solution End:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=3)
        self.solution_end = tk.StringVar(value="</SOLUTION>")
        ttk.Entry(marker_frame, textvariable=self.solution_end, width=25).grid(row=1, column=3, padx=5, pady=3)

        # System Prompt Configuration
        system_frame = ttk.LabelFrame(template_frame, text="System Prompt", padding=5)
        system_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)

        self.system_prompt = tk.Text(system_frame, height=4, width=70, wrap=tk.WORD)
        self.system_prompt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Default GRPO system prompt
        default_system = (
            "You are given a problem.\n"
            "Think about the problem and provide your working out.\n"
            "Place it between <start_working_out> and <end_working_out>.\n"
            "Then, provide your solution between <SOLUTION></SOLUTION>"
        )
        self.system_prompt.insert(tk.END, default_system)

        # System prompt scrollbar
        system_scroll = ttk.Scrollbar(system_frame, command=self.system_prompt.yview)
        system_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.system_prompt.config(yscrollcommand=system_scroll.set)

        # Template Preview Section
        preview_frame = ttk.LabelFrame(template_frame, text="Template Preview (Jinja2 Format)", padding=5)
        preview_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)

        self.template_preview = tk.Text(preview_frame, height=10, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.template_preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Preview scrollbar
        preview_scroll = ttk.Scrollbar(preview_frame, command=self.template_preview.yview)
        preview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.template_preview.config(yscrollcommand=preview_scroll.set)

        # Update and Test buttons
        button_frame = ttk.Frame(template_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Update Template", command=self._update_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Test with Sample", command=self._test_template).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self._reset_to_default).pack(side=tk.LEFT, padx=5)

        # Initialize with default GRPO template
        self._load_grpo_template()

        # Remove old template definitions since we're using the new system
        self.templates = {}
        # Custom templates storage (kept for backward compatibility)
        self.custom_templates = {}

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _on_model_type_changed(self, event=None):
        """Handle model type change."""
        model_type = self.model_type.get()
        if model_type != "Auto-Detect":
            self._load_grpo_template()

    def _load_grpo_template(self):
        """Load GRPO template for selected model type."""
        model_type = self.model_type.get()

        # Map UI names to template model types
        model_map = {
            "Auto-Detect": "qwen",  # Default to Qwen for auto-detect since it's the example model
            "Qwen": "qwen",
            "Llama": "llama",
            "Mistral": "mistral",
            "ChatML": "chatml"
        }

        template_model = model_map.get(model_type, "qwen")

        # Create GRPO template for the model
        self.current_template = PromptTemplate.from_model_type(template_model, grpo_mode=True)

        # Update UI with template configuration
        config = self.current_template.config
        self.reasoning_start.set(config.reasoning_start_marker)
        self.reasoning_end.set(config.reasoning_end_marker)
        self.solution_start.set(config.solution_start_marker)
        self.solution_end.set(config.solution_end_marker)

        # Update system prompt if available
        if config.system_prompt:
            self.system_prompt.delete("1.0", tk.END)
            self.system_prompt.insert(tk.END, config.system_prompt)

        # Update template preview
        self._update_template_preview()

        # Update info label
        self.template_info.config(
            text=f"Loaded GRPO template for {model_type}",
            foreground="green"
        )

    def _update_template_preview(self):
        """Update the template preview display."""
        if not self.current_template:
            return

        # Get the Jinja2 template
        template_str = self.current_template.config.chat_template or "No chat template defined"

        # Enable text widget for update
        self.template_preview.config(state=tk.NORMAL)
        self.template_preview.delete("1.0", tk.END)

        # Format for display (add some line breaks for readability)
        formatted = template_str.replace("{% ", "\n{% ").replace(" %}", " %}\n")
        self.template_preview.insert(tk.END, formatted)

        # Disable again
        self.template_preview.config(state=tk.DISABLED)

    def _update_template(self):
        """Update the current template with new settings."""
        if not self.current_template:
            messagebox.showerror("Error", "No template loaded")
            return

        # Update template configuration
        config = self.current_template.config
        config.reasoning_start_marker = self.reasoning_start.get()
        config.reasoning_end_marker = self.reasoning_end.get()
        config.solution_start_marker = self.solution_start.get()
        config.solution_end_marker = self.solution_end.get()
        config.system_prompt = self.system_prompt.get("1.0", tk.END).strip()

        # Recreate template with updated config
        self.current_template = PromptTemplate(config)

        # Update preview
        self._update_template_preview()

        messagebox.showinfo("Success", "Template updated successfully!")

    def _test_template(self):
        """Test the template with a sample message."""
        if not self.current_template:
            messagebox.showerror("Error", "No template loaded")
            return

        # Create test messages
        test_messages = [
            {"role": "user", "content": "What is the square root of 144?"}
        ]

        # Apply template
        try:
            # Without generation prompt
            result = self.current_template.apply_chat_template(
                test_messages,
                add_generation_prompt=False,
                eos_token="<|im_end|>"  # Default for testing
            )

            # With generation prompt (for inference)
            result_with_prompt = self.current_template.apply_chat_template(
                test_messages,
                add_generation_prompt=True,
                eos_token="<|im_end|>",
                reasoning_start=self.reasoning_start.get()
            )

            # Show results
            test_window = tk.Toplevel(self.frame)
            test_window.title("Template Test Results")
            test_window.geometry("800x600")

            # Create notebook for tabs
            notebook = ttk.Notebook(test_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Tab 1: Without generation prompt
            frame1 = ttk.Frame(notebook)
            notebook.add(frame1, text="Training Format")

            text1 = tk.Text(frame1, wrap=tk.WORD)
            text1.pack(fill=tk.BOTH, expand=True)
            text1.insert(tk.END, result)

            # Tab 2: With generation prompt
            frame2 = ttk.Frame(notebook)
            notebook.add(frame2, text="Inference Format")

            text2 = tk.Text(frame2, wrap=tk.WORD)
            text2.pack(fill=tk.BOTH, expand=True)
            text2.insert(tk.END, result_with_prompt)

            # Close button
            ttk.Button(test_window, text="Close", command=test_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Template Error", f"Failed to apply template:\n{str(e)}")

    def _reset_to_default(self):
        """Reset to default GRPO template."""
        self.model_type.set("Auto-Detect")
        self._load_grpo_template()

    def _on_source_type_change(self, event=None):
        """Handle source type change."""
        source = self.source_type.get()

        if "HuggingFace" in source:
            self.dataset_entry.config(state="normal")
            # Hide browse button for HuggingFace
            self.browse_button.grid_remove()
            # Keep current value if it's already a HuggingFace dataset
            if not self.dataset_path.get() or "api.example.com" in self.dataset_path.get():
                self.dataset_path.set("tatsu-lab/alpaca")
        elif "Local" in source:
            self.dataset_entry.config(state="normal")
            # Show browse button for Local File
            self.browse_button.grid()
        elif "API" in source:
            self.dataset_entry.config(state="normal")
            # Hide browse button for API
            self.browse_button.grid_remove()
            self.dataset_path.set("https://api.example.com/dataset")
        else:  # Direct Input
            self.dataset_entry.config(state="disabled")
            # Hide browse button for Direct Input
            self.browse_button.grid_remove()

        self.on_config_change("dataset_source", source.lower().replace(" ", "_"))

    def _on_dataset_selected(self, event=None):
        """Handle dataset selection from dropdown."""
        selected = self.dataset_path.get()

        # Auto-configure field mappings based on known datasets
        dataset_configs = {
            "tatsu-lab/alpaca": {
                "instruction_field": "instruction",
                "response_field": "output",
                "system_field": "",
                "split": "train"
            },
            "openai/gsm8k": {
                "instruction_field": "question",
                "response_field": "answer",
                "system_field": "",
                "split": "train"
            },
            "open-r1/DAPO-Math-17k-Processed": {
                "instruction_field": "prompt",
                "response_field": "chosen",
                "system_field": "",
                "split": "train"
            },
            "nvidia/OpenMathReasoning": {
                "instruction_field": "problem",
                "response_field": "solution",
                "system_field": "",
                "split": "train"
            }
        }

        # Apply configuration if it's a known dataset
        if selected in dataset_configs:
            config = dataset_configs[selected]
            self.instruction_field.set(config["instruction_field"])
            self.response_field.set(config["response_field"])
            self.system_field.set(config["system_field"])
            self.split.set(config["split"])

            # Update status
            messagebox.showinfo("Dataset Selected",
                              f"Selected: {selected}\n"
                              f"Field mappings have been auto-configured.")

        self.on_config_change("dataset_path", selected)

    def _browse_dataset(self):
        """Browse for local dataset file."""
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("All Supported", "*.json;*.jsonl;*.csv;*.parquet"),
                ("JSON Files", "*.json"),
                ("JSONL Files", "*.jsonl"),
                ("CSV Files", "*.csv"),
                ("Parquet Files", "*.parquet"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self.dataset_path.set(file_path)
            self.on_config_change("dataset_path", file_path)

    def _load_dataset(self):
        """Load and preview dataset."""
        dataset_name = self.dataset_path.get()
        source_type = self.source_type.get()

        if not dataset_name:
            messagebox.showerror("Error", "Please enter a dataset name or path")
            return

        # Check if dataset is already downloaded
        cache_dir = Path("datasets_cache") / dataset_name.replace("/", "_")
        dataset_info_file = cache_dir / "dataset_info.json"

        use_cache = False
        if cache_dir.exists() and dataset_info_file.exists() and not self.force_redownload.get():
            # Ask user if they want to use cached version
            use_cache = messagebox.askyesno(
                "Dataset Cache",
                f"Dataset '{dataset_name}' is already downloaded.\n"
                "Do you want to use the cached version?"
            )
        elif self.force_redownload.get() and cache_dir.exists():
            # User forced redownload, clear the cache
            import shutil
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                print(f"Warning: Could not clear cache: {e}")

        # Disable buttons during loading
        self.load_button.config(state=tk.DISABLED, text="Loading...")

        # Update status based on cache status
        if use_cache:
            self.stats_label.config(text="Loading dataset from cache...")
        else:
            self.stats_label.config(text="Downloading dataset... This may take a few minutes for first time.")

        # Show progress bar
        self.progress_bar.grid(row=2, column=0, pady=5)
        self.progress_bar.start(10)  # Start animation

        # Clear previous preview
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)

        # Load dataset in background thread
        def load_dataset_thread():
            try:
                # Determine source type
                if "HuggingFace" in source_type:
                    source = "huggingface"
                elif "Local" in source_type:
                    source = "local"
                elif "API" in source_type:
                    source = "api"
                else:
                    source = "direct"

                # Configure dataset
                config = DatasetConfig(
                    source_type=source,
                    source_path=dataset_name,
                    split=self.split.get() or "train",
                    instruction_field=self.instruction_field.get() or "instruction",
                    response_field=self.response_field.get() or "response",
                    system_field=self.system_field.get() or None,
                    max_samples=None,  # Don't limit during initial load
                    streaming=False,  # Ensure we download the full dataset
                    shuffle=False  # Don't shuffle for preview
                )

                # Create handler
                self.dataset_handler = DatasetHandler(config)

                # Check if we should use cache
                if use_cache and cache_dir.exists():
                    # Load from cache
                    cached_data_file = cache_dir / "data.jsonl"
                    if cached_data_file.exists():
                        # Create a new config for loading cached data
                        cache_config = DatasetConfig(
                            source_type="local",
                            source_path=str(cached_data_file),
                            split="train",  # Cached data doesn't have splits
                            instruction_field=self.instruction_field.get() or "instruction",
                            response_field=self.response_field.get() or "response",
                            system_field=self.system_field.get() or None,
                            max_samples=None,
                            streaming=False,
                            shuffle=False
                        )
                        # Create a new handler for cached data
                        cache_handler = DatasetHandler(cache_config)
                        self.dataset = cache_handler.load()
                        # Copy the dataset back to the main handler
                        self.dataset_handler.dataset = self.dataset
                        self.dataset_handler.statistics = cache_handler.statistics
                    else:
                        raise FileNotFoundError("Cached data file not found")
                else:
                    # Load fresh from source
                    self.dataset = self.dataset_handler.load()

                    # Ensure dataset is loaded
                    if self.dataset is None or len(self.dataset) == 0:
                        raise ValueError("Dataset is empty or failed to load")

                    # Save to cache if from HuggingFace
                    if source == "huggingface" and self.dataset:
                        def save_cache():
                            try:
                                cache_dir.mkdir(parents=True, exist_ok=True)

                                # Save dataset (only first 1000 samples for cache to save space)
                                cached_data_file = cache_dir / "data.jsonl"

                                # Ensure the dataset handler is properly initialized
                                if self.dataset_handler and self.dataset_handler.dataset:
                                    if len(self.dataset) > 1000:
                                        # Create a new temporary handler for export
                                        cache_dataset = self.dataset.select(range(1000))
                                        temp_handler = DatasetHandler(config)
                                        temp_handler.dataset = cache_dataset
                                        temp_handler.export(str(cached_data_file), format='jsonl')
                                    else:
                                        self.dataset_handler.export(str(cached_data_file), format='jsonl')

                                    # Save metadata
                                    with open(dataset_info_file, 'w') as f:
                                        json.dump({
                                            'dataset_name': dataset_name,
                                            'split': config.split,
                                            'total_samples': len(self.dataset),
                                            'cached_samples': min(1000, len(self.dataset)),
                                            'fields': self.dataset.column_names if hasattr(self.dataset, 'column_names') else []
                                        }, f, indent=2)
                                else:
                                    print(f"Warning: Dataset handler not properly initialized for caching")
                            except Exception as e:
                                # Log cache save error but don't fail the load
                                print(f"Warning: Failed to cache dataset: {e}")

                        # Save cache after updating UI
                        save_cache()

                # Get preview samples (limit to 10 for UI display)
                preview_samples = self.dataset_handler.get_preview(num_samples=10)

                # Get statistics (ensure they're calculated)
                if not self.dataset_handler.statistics:
                    self.dataset_handler.statistics = self.dataset_handler._calculate_statistics(self.dataset)
                stats = self.dataset_handler.statistics

                # Update UI in main thread
                self.frame.after(0, self._update_preview_ui, preview_samples, stats)

            except Exception as e:
                self.frame.after(0, self._handle_load_error, str(e))

        # Start loading thread
        thread = threading.Thread(target=load_dataset_thread, daemon=True)
        thread.start()

    def _update_preview_ui(self, samples, stats):
        """Update UI with loaded dataset."""
        # Stop and hide progress bar
        self.progress_bar.stop()
        self.progress_bar.grid_remove()

        # Clear preview
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)

        # Add samples to preview
        for i, sample in enumerate(samples, 1):
            instruction = sample.get(self.instruction_field.get(), '')
            response = sample.get(self.response_field.get(), '')

            # Truncate long text for display
            if len(instruction) > 100:
                instruction = instruction[:97] + "..."
            if len(response) > 100:
                response = response[:97] + "..."

            self.preview_tree.insert("", tk.END, values=(str(i), instruction, response))

        # Update statistics
        if stats:
            stats_text = (
                f"Total samples: {stats.total_samples} | "
                f"Avg instruction length: {stats.avg_instruction_length:.0f} | "
                f"Avg response length: {stats.avg_response_length:.0f}"
            )
            self.stats_label.config(text=stats_text)
        else:
            self.stats_label.config(text="Dataset loaded")

        # Re-enable button
        self.load_button.config(state=tk.NORMAL, text="Load Dataset")

        messagebox.showinfo("Success",
                          f"Dataset loaded successfully!\n"
                          f"Total samples: {stats.total_samples if stats else 'Unknown'}")

    def _handle_load_error(self, error_msg):
        """Handle dataset loading error."""
        # Stop and hide progress bar
        self.progress_bar.stop()
        self.progress_bar.grid_remove()

        self.stats_label.config(text="Failed to load dataset")
        self.load_button.config(state=tk.NORMAL, text="Load Dataset")
        messagebox.showerror("Error", f"Failed to load dataset:\n{error_msg}")

    def import_dataset(self):
        """Import dataset from file."""
        self._browse_dataset()
        if self.dataset_path.get():
            self._load_dataset()

    def get_config(self) -> Dict[str, Any]:
        """Get configuration from tab."""
        config = {
            "dataset_source": self.source_type.get(),
            "dataset_path": self.dataset_path.get(),
            "dataset_split": self.split.get(),
            "instruction_field": self.instruction_field.get(),
            "response_field": self.response_field.get(),
            "system_field": self.system_field.get() or None,
            "model_type": self.model_type.get(),
            "system_prompt": self.system_prompt.get("1.0", tk.END).strip(),
            "reasoning_start": self.reasoning_start.get(),
            "reasoning_end": self.reasoning_end.get(),
            "solution_start": self.solution_start.get(),
            "solution_end": self.solution_end.get(),
        }

        # Save current template configuration if available
        if self.current_template:
            config["template_config"] = self.current_template.to_dict()

        return config

    def load_config(self, config: Dict[str, Any]):
        """Load configuration into tab."""
        if "dataset_source" in config:
            source = config["dataset_source"]
            # Handle old config values
            if source.lower() == "huggingface":
                source = "HuggingFace Hub"
            self.source_type.set(source)
        if "dataset_path" in config:
            self.dataset_path.set(config["dataset_path"])
        if "dataset_split" in config:
            self.split.set(config["dataset_split"])
        if "instruction_field" in config:
            self.instruction_field.set(config["instruction_field"])
        if "response_field" in config:
            self.response_field.set(config["response_field"])
        if "system_field" in config and config["system_field"]:
            self.system_field.set(config["system_field"])

        # Load model type
        if "model_type" in config:
            self.model_type.set(config["model_type"])

        # Load GRPO markers
        if "reasoning_start" in config:
            self.reasoning_start.set(config["reasoning_start"])
        if "reasoning_end" in config:
            self.reasoning_end.set(config["reasoning_end"])
        if "solution_start" in config:
            self.solution_start.set(config["solution_start"])
        if "solution_end" in config:
            self.solution_end.set(config["solution_end"])

        # Load system prompt
        if "system_prompt" in config:
            self.system_prompt.delete("1.0", tk.END)
            self.system_prompt.insert(tk.END, config["system_prompt"])

        # Load template configuration if available
        if "template_config" in config:
            template_dict = config["template_config"]
            template_config = TemplateConfig(**template_dict)
            self.current_template = PromptTemplate(template_config)
            self._update_template_preview()
        else:
            # Load default GRPO template
            self._load_grpo_template()

        # Handle legacy template configurations for backward compatibility
        if "template_name" in config and "template_text" in config:
            # Old config format - convert to GRPO
            self.template_info.config(
                text="Legacy config detected - converted to GRPO",
                foreground="orange"
            )

    def get_prompt_template(self) -> Optional[PromptTemplate]:
        """Get the current prompt template for training.

        Returns:
            PromptTemplate instance or None
        """
        if not self.current_template:
            # Create default if not initialized
            self._load_grpo_template()
        return self.current_template

    def format_dataset_with_template(self, dataset):
        """Format dataset using the current GRPO template.

        Args:
            dataset: Dataset to format

        Returns:
            Formatted dataset ready for GRPO training
        """
        if not self.current_template:
            raise ValueError("No template loaded")

        template = self.current_template

        def format_sample(sample):
            """Format a single sample for GRPO."""
            # Extract fields
            instruction = sample.get(self.instruction_field.get(), "")
            response = sample.get(self.response_field.get(), "")
            system = sample.get(self.system_field.get(), "") if self.system_field.get() else ""

            # Create message format
            messages = []
            if system or template.config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": system or template.config.system_prompt
                })
            messages.append({"role": "user", "content": instruction})

            # For training, include the full response
            if response:
                # Format response with GRPO markers
                formatted_response = (
                    f"{template.config.reasoning_start_marker}\n"
                    f"{response}\n"
                    f"{template.config.reasoning_end_marker}\n"
                    f"{template.config.solution_start_marker}\n"
                    f"[Solution will be here]\n"
                    f"{template.config.solution_end_marker}"
                )
                messages.append({"role": "assistant", "content": formatted_response})

            # Apply template
            formatted_text = template.apply_chat_template(
                messages,
                add_generation_prompt=False,
                eos_token=template.config.eos_token or ""
            )

            return {"text": formatted_text, "messages": messages}

        # Apply formatting to dataset
        if hasattr(dataset, 'map'):
            return dataset.map(format_sample)
        else:
            # Handle list datasets
            return [format_sample(sample) for sample in dataset]

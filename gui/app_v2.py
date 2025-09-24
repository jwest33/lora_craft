# -*- coding: utf-8 -*-
"""Improved GRPO Fine-Tuner GUI Application with Enhanced UX"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from core import (
    SystemConfig,
    GRPOTrainer,
    GRPOConfig,
    DatasetHandler,
    PromptTemplate,
    CustomRewardBuilder
)
from utils.logging_config import LogManager, get_logger
from utils.validators import validate_training_config

logger = get_logger(__name__)


class SimplifiedGRPOApp:
    """Simplified and intuitive GRPO Fine-Tuner application."""

    def __init__(self, root: tk.Tk):
        """Initialize the application."""
        self.root = root
        self.root.title("GRPO Model Trainer")
        self.root.geometry("1400x900")

        # Application state
        self.config = {}
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.message_queue = queue.Queue()
        self.current_step = 0
        self.advanced_mode = tk.BooleanVar(value=False)

        # System configuration
        self.system_config = SystemConfig()

        # Setup UI
        self._setup_styles()
        self._create_ui()
        self._setup_logging()

        # Load defaults
        self._load_smart_defaults()

        # Start message processing
        self._process_messages()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_styles(self):
        """Setup clean, modern styling."""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors with consistent theme
        self.colors = {
            'primary': '#2196F3',
            'primary_dark': '#1976D2',
            'secondary': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336',
            'bg': '#ffffff',           # Main background - white
            'bg_secondary': '#f8f9fa',  # Secondary areas - very light gray
            'bg_sidebar': '#e8eaf0',    # Sidebar - light blue-gray
            'text': '#212121',
            'text_secondary': '#616161',
            'border': '#d0d0d0'
        }

        # Configure styles
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'))
        style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Subheading.TLabel', font=('Segoe UI', 11))
        style.configure('Help.TLabel', font=('Segoe UI', 9), foreground=self.colors['text_secondary'])

        # Primary button style
        style.configure('Primary.TButton',
                       font=('Segoe UI', 10, 'bold'),
                       relief='flat',
                       borderwidth=0)
        style.map('Primary.TButton',
                 background=[('active', self.colors['primary_dark']),
                           ('!active', self.colors['primary'])],
                 foreground=[('active', 'white'),
                           ('!active', 'white')])

        # Secondary button style
        style.configure('Secondary.TButton',
                       font=('Segoe UI', 10),
                       relief='flat',
                       borderwidth=1)

        # Page/Card style - white background
        style.configure('Page.TFrame',
                       background=self.colors['bg'],
                       relief='flat',
                       borderwidth=0)

        # Sidebar style - light blue-gray background
        style.configure('Sidebar.TFrame',
                       background=self.colors['bg_sidebar'],
                       relief='flat',
                       borderwidth=0)

        # Main container style
        style.configure('Main.TFrame',
                       background=self.colors['bg_secondary'],
                       relief='flat',
                       borderwidth=0)

    def _create_ui(self):
        """Create the main UI layout."""
        # Set root background
        self.root.configure(bg=self.colors['bg_secondary'])

        # Main container
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left sidebar for workflow
        self._create_workflow_sidebar(main_container)

        # Create main content area
        self._create_main_content(main_container)

        # Create bottom action bar
        self._create_action_bar()

    def _create_workflow_sidebar(self, parent):
        """Create workflow sidebar with step indicators."""
        # Use tk.Frame instead of ttk.Frame for consistent background
        sidebar = tk.Frame(parent, width=250, bg=self.colors['bg_sidebar'])
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # Title with explicit background
        title_label = tk.Label(
            sidebar,
            text="Training Workflow",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['bg_sidebar'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(20, 30), padx=20)

        # Create a container frame for workflow steps
        self.steps_container = tk.Frame(sidebar, bg=self.colors['bg_sidebar'])
        self.steps_container.pack(fill=tk.X, padx=0, pady=0)

        # Workflow steps
        self.workflow_steps = [
            {'name': 'Select Model', 'icon': '1', 'status': 'active'},
            {'name': 'Choose Dataset', 'icon': '2', 'status': 'pending'},
            {'name': 'Configure Training', 'icon': '3', 'status': 'pending'},
            {'name': 'Train Model', 'icon': '4', 'status': 'pending'},
            {'name': 'Export Results', 'icon': '5', 'status': 'pending'}
        ]

        self.step_frames = []
        for i, step in enumerate(self.workflow_steps):
            step_frame = self._create_step_indicator(self.steps_container, step, i)
            step_frame.pack(fill=tk.X, padx=20, pady=5)
            self.step_frames.append(step_frame)

        # Spacer
        tk.Frame(sidebar, bg=self.colors['bg_sidebar']).pack(fill=tk.BOTH, expand=True)

        # Advanced mode toggle
        advanced_frame = tk.Frame(sidebar, bg=self.colors['bg_sidebar'])
        advanced_frame.pack(fill=tk.X, padx=20, pady=20)

        tk.Checkbutton(
            advanced_frame,
            text="Advanced Mode",
            variable=self.advanced_mode,
            command=self._toggle_advanced_mode,
            bg=self.colors['bg_sidebar'],
            fg=self.colors['text'],
            activebackground=self.colors['bg_sidebar'],
            selectcolor=self.colors['bg_sidebar']
        ).pack(side=tk.LEFT)

        # Quick actions
        quick_actions_label = tk.Label(
            sidebar,
            text="Quick Actions",
            font=('Segoe UI', 11),
            bg=self.colors['bg_sidebar'],
            fg=self.colors['text']
        )
        quick_actions_label.pack(pady=(20, 10), padx=20)

        ttk.Button(
            sidebar,
            text="Load Configuration",
            command=self._load_config
        ).pack(fill=tk.X, padx=20, pady=2)

        ttk.Button(
            sidebar,
            text="Save Configuration",
            command=self._save_config
        ).pack(fill=tk.X, padx=20, pady=2)

        ttk.Button(
            sidebar,
            text="View Documentation",
            command=self._show_documentation
        ).pack(fill=tk.X, padx=20, pady=2)

    def _create_step_indicator(self, parent, step, index):
        """Create a step indicator widget."""
        # Use tk.Frame for consistent background
        frame = tk.Frame(parent, bg=self.colors['bg_sidebar'])

        # Determine colors based on status
        if step['status'] == 'completed':
            bg_color = self.colors['secondary']
            fg_color = 'white'
        elif step['status'] == 'active':
            bg_color = self.colors['primary']
            fg_color = 'white'
        else:
            bg_color = '#e0e0e0'
            fg_color = '#666666'

        # Create circular indicator
        indicator = tk.Label(
            frame,
            text=step['icon'],
            width=3,
            height=1,
            bg=bg_color,
            fg=fg_color,
            font=('Segoe UI', 11, 'bold'),
            relief='solid',
            bd=1
        )
        indicator.pack(side=tk.LEFT, padx=(0, 10))

        # Step name - use tk.Label for consistent display
        name_label = tk.Label(
            frame,
            text=step['name'],
            font=('Segoe UI', 10, 'bold' if step['status'] == 'active' else 'normal'),
            bg=self.colors['bg_sidebar'],
            fg=self.colors['text'] if step['status'] == 'active' else self.colors['text_secondary']
        )
        name_label.pack(side=tk.LEFT)

        # Make clickable
        for widget in [frame, indicator, name_label]:
            widget.bind("<Button-1>", lambda e, idx=index: self._navigate_to_step(idx))
            widget.bind("<Enter>", lambda e: frame.configure(cursor="hand2"))
            widget.bind("<Leave>", lambda e: frame.configure(cursor=""))

        return frame

    def _create_main_content(self, parent):
        """Create main content area with stacked pages."""
        self.content_frame = tk.Frame(parent, bg=self.colors['bg'])
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create stacked pages for each workflow step
        self.pages = {}

        # Page 1: Model Selection
        self.pages['model'] = self._create_model_page(self.content_frame)

        # Page 2: Dataset Selection
        self.pages['dataset'] = self._create_dataset_page(self.content_frame)

        # Page 3: Training Configuration
        self.pages['config'] = self._create_config_page(self.content_frame)

        # Page 4: Training Progress
        self.pages['training'] = self._create_training_page(self.content_frame)

        # Page 5: Export
        self.pages['export'] = self._create_export_page(self.content_frame)

        # Show first page
        self._show_page('model')

    def _create_model_page(self, parent):
        """Create model selection page."""
        page = tk.Frame(parent, bg=self.colors['bg'])

        # Header
        header_frame = ttk.Frame(page)
        header_frame.pack(fill=tk.X, padx=30, pady=(30, 20))

        ttk.Label(header_frame, text="Select Your Model", style='Title.TLabel').pack(side=tk.LEFT)

        # Help text
        ttk.Label(
            page,
            text="Choose a model that fits your hardware and requirements. Models are grouped by family.",
            style='Help.TLabel'
        ).pack(anchor=tk.W, padx=30, pady=(0, 20))

        # Model cards container
        cards_container = ttk.Frame(page)
        cards_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Create model families
        model_families = {
            'Qwen3 Series': {
                'description': 'Versatile models for general tasks',
                'models': [
                    {'name': 'Qwen3-0.6B', 'vram': '1.2GB', 'use_case': 'Testing & Edge devices'},
                    {'name': 'Qwen3-1.7B', 'vram': '3.4GB', 'use_case': 'Balanced performance'},
                    {'name': 'Qwen3-4B', 'vram': '8GB', 'use_case': 'Advanced reasoning'},
                    {'name': 'Qwen3-8B', 'vram': '16GB', 'use_case': 'Professional tasks'}
                ]
            },
            'LLaMA 3.2 Series': {
                'description': 'Meta\'s latest instruction-tuned models',
                'models': [
                    {'name': 'Llama-3.2-1B', 'vram': '2GB', 'use_case': 'Fast inference'},
                    {'name': 'Llama-3.2-3B', 'vram': '6GB', 'use_case': 'Quality conversations'}
                ]
            },
            'Phi-4 Series': {
                'description': 'Microsoft\'s reasoning-focused models',
                'models': [
                    {'name': 'phi-4-reasoning', 'vram': '30GB', 'use_case': 'Complex reasoning'}
                ]
            }
        }

        # Create scrollable frame for model cards
        canvas = tk.Canvas(cards_container, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(cards_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Variable to store selected model
        self.selected_model = tk.StringVar()

        # Get available GPU memory
        gpu_memory = 0
        if self.system_config.gpu_info:
            gpu_memory = self.system_config.gpu_info[0].memory_total

        for family_name, family_data in model_families.items():
            # Family header
            family_frame = ttk.LabelFrame(scrollable_frame, text=family_name, padding=15)
            family_frame.pack(fill=tk.X, padx=10, pady=10)

            ttk.Label(
                family_frame,
                text=family_data['description'],
                style='Help.TLabel'
            ).pack(anchor=tk.W, pady=(0, 10))

            # Model cards
            models_frame = ttk.Frame(family_frame)
            models_frame.pack(fill=tk.X)

            for i, model in enumerate(family_data['models']):
                model_card = self._create_model_card(
                    models_frame,
                    model,
                    gpu_memory,
                    family_name
                )
                model_card.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky='ew')
                models_frame.columnconfigure(i % 2, weight=1)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        return page

    def _create_model_card(self, parent, model_info, gpu_memory, family):
        """Create a model card widget."""
        # Determine if recommended
        vram_required = float(model_info['vram'].replace('GB', ''))
        is_recommended = gpu_memory > 0 and vram_required <= gpu_memory / 1024

        card = ttk.Frame(parent, relief='ridge', borderwidth=1)
        card.configure(cursor="hand2")

        # Model name with radio button
        header_frame = ttk.Frame(card)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        radio = ttk.Radiobutton(
            header_frame,
            text=model_info['name'],
            variable=self.selected_model,
            value=f"unsloth/{model_info['name']}",
            command=lambda: self._on_model_selected(model_info['name'])
        )
        radio.pack(side=tk.LEFT)

        if is_recommended:
            recommended_label = tk.Label(
                header_frame,
                text="RECOMMENDED",
                bg=self.colors['secondary'],
                fg='white',
                font=('Segoe UI', 8, 'bold'),
                padx=5
            )
            recommended_label.pack(side=tk.RIGHT)

        # VRAM requirement
        vram_frame = ttk.Frame(card)
        vram_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(vram_frame, text="VRAM: ", style='Help.TLabel').pack(side=tk.LEFT)
        ttk.Label(vram_frame, text=model_info['vram'], font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT)

        # Use case
        ttk.Label(
            card,
            text=model_info['use_case'],
            style='Help.TLabel'
        ).pack(anchor=tk.W, padx=10, pady=(2, 10))

        # Make entire card clickable
        def select_model(event=None):
            self.selected_model.set(f"unsloth/{model_info['name']}")
            self._on_model_selected(model_info['name'])

        card.bind("<Button-1>", select_model)

        return card

    def _create_dataset_page(self, parent):
        """Create dataset selection page."""
        page = tk.Frame(parent, bg=self.colors['bg'])

        # Header
        ttk.Label(page, text="Choose Your Dataset", style='Title.TLabel').pack(
            anchor=tk.W, padx=30, pady=(30, 10)
        )

        ttk.Label(
            page,
            text="Select from popular datasets or upload your own data",
            style='Help.TLabel'
        ).pack(anchor=tk.W, padx=30, pady=(0, 20))

        # Dataset source selection
        source_frame = ttk.Frame(page)
        source_frame.pack(fill=tk.X, padx=30, pady=10)

        self.dataset_source = tk.StringVar(value="library")

        # Option 1: Dataset Library
        library_option = ttk.Radiobutton(
            source_frame,
            text="Choose from Dataset Library",
            variable=self.dataset_source,
            value="library",
            command=self._update_dataset_view
        )
        library_option.pack(anchor=tk.W, pady=5)

        # Option 2: Upload File
        upload_option = ttk.Radiobutton(
            source_frame,
            text="Upload Local File",
            variable=self.dataset_source,
            value="upload",
            command=self._update_dataset_view
        )
        upload_option.pack(anchor=tk.W, pady=5)

        # Option 3: HuggingFace Hub
        hub_option = ttk.Radiobutton(
            source_frame,
            text="Load from HuggingFace Hub",
            variable=self.dataset_source,
            value="hub",
            command=self._update_dataset_view
        )
        hub_option.pack(anchor=tk.W, pady=5)

        # Dataset content area
        self.dataset_content_frame = ttk.Frame(page)
        self.dataset_content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Create dataset library view (default)
        self._create_dataset_library_view()

        return page

    def _create_dataset_library_view(self):
        """Create dataset library view."""
        # Clear content frame
        for widget in self.dataset_content_frame.winfo_children():
            widget.destroy()

        # Popular datasets
        datasets = [
            {
                'name': 'Alpaca Dataset',
                'id': 'tatsu-lab/alpaca',
                'description': 'General instruction-following dataset with 52K examples',
                'size': '52K samples',
                'type': 'Instruction'
            },
            {
                'name': 'GSM8K Math',
                'id': 'openai/gsm8k',
                'description': 'Grade school math problems with step-by-step solutions',
                'size': '8.5K samples',
                'type': 'Mathematics'
            },
            {
                'name': 'OpenMath Reasoning',
                'id': 'nvidia/OpenMathReasoning',
                'description': 'Advanced mathematical reasoning with detailed explanations',
                'size': '100K samples',
                'type': 'Mathematics'
            },
            {
                'name': 'Code Alpaca',
                'id': 'sahil2801/CodeAlpaca-20k',
                'description': 'Code generation and explanation dataset',
                'size': '20K samples',
                'type': 'Code'
            }
        ]

        # Create scrollable grid
        canvas = tk.Canvas(self.dataset_content_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.dataset_content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.selected_dataset = tk.StringVar()

        for i, dataset in enumerate(datasets):
            card = self._create_dataset_card(scrollable_frame, dataset)
            card.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky='ew')
            scrollable_frame.columnconfigure(i % 2, weight=1)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_dataset_card(self, parent, dataset_info):
        """Create a dataset card widget."""
        card = ttk.LabelFrame(parent, text=dataset_info['name'], padding=15)

        # Type badge
        type_frame = ttk.Frame(card)
        type_frame.pack(fill=tk.X, pady=(0, 5))

        type_label = tk.Label(
            type_frame,
            text=dataset_info['type'],
            bg=self.colors['primary'],
            fg='white',
            font=('Segoe UI', 8),
            padx=8,
            pady=2
        )
        type_label.pack(side=tk.LEFT)

        ttk.Label(
            type_frame,
            text=dataset_info['size'],
            style='Help.TLabel'
        ).pack(side=tk.RIGHT)

        # Description
        ttk.Label(
            card,
            text=dataset_info['description'],
            wraplength=250,
            style='Help.TLabel'
        ).pack(anchor=tk.W, pady=5)

        # Select button
        ttk.Button(
            card,
            text="Select This Dataset",
            style='Primary.TButton',
            command=lambda: self._select_dataset(dataset_info)
        ).pack(fill=tk.X, pady=(10, 0))

        return card

    def _create_config_page(self, parent):
        """Create training configuration page."""
        page = tk.Frame(parent, bg=self.colors['bg'])

        # Header
        ttk.Label(page, text="Configure Training", style='Title.TLabel').pack(
            anchor=tk.W, padx=30, pady=(30, 10)
        )

        # Preset selection
        preset_frame = ttk.Frame(page)
        preset_frame.pack(fill=tk.X, padx=30, pady=20)

        ttk.Label(preset_frame, text="Training Preset:", style='Subheading.TLabel').pack(side=tk.LEFT, padx=(0, 10))

        self.training_preset = tk.StringVar(value="balanced")
        presets = [
            ("Quick Test", "quick"),
            ("Balanced", "balanced"),
            ("High Quality", "quality")
        ]

        for text, value in presets:
            ttk.Radiobutton(
                preset_frame,
                text=text,
                variable=self.training_preset,
                value=value,
                command=self._apply_preset
            ).pack(side=tk.LEFT, padx=10)

        # Basic settings (always visible)
        basic_frame = ttk.LabelFrame(page, text="Basic Settings", padding=20)
        basic_frame.pack(fill=tk.X, padx=30, pady=10)

        # Epochs
        epochs_frame = ttk.Frame(basic_frame)
        epochs_frame.pack(fill=tk.X, pady=5)
        ttk.Label(epochs_frame, text="Training Epochs:", width=20).pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=3)
        epochs_spin = ttk.Spinbox(
            epochs_frame, from_=1, to=10, textvariable=self.epochs_var, width=10
        )
        epochs_spin.pack(side=tk.LEFT, padx=10)
        ttk.Label(
            epochs_frame,
            text="Number of complete passes through the dataset",
            style='Help.TLabel'
        ).pack(side=tk.LEFT)

        # Batch size
        batch_frame = ttk.Frame(basic_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(batch_frame, text="Batch Size:", width=20).pack(side=tk.LEFT)
        self.batch_var = tk.IntVar(value=4)
        batch_spin = ttk.Spinbox(
            batch_frame, from_=1, to=32, textvariable=self.batch_var, width=10
        )
        batch_spin.pack(side=tk.LEFT, padx=10)
        ttk.Label(
            batch_frame,
            text="Samples per training step (auto-adjusted for GPU)",
            style='Help.TLabel'
        ).pack(side=tk.LEFT)

        # Learning rate
        lr_frame = ttk.Frame(basic_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Learning Rate:", width=20).pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value="2e-4")
        lr_entry = ttk.Entry(lr_frame, textvariable=self.lr_var, width=10)
        lr_entry.pack(side=tk.LEFT, padx=10)
        ttk.Label(lr_frame, text="Training speed (default: 2e-4)", style='Help.TLabel').pack(side=tk.LEFT)

        # Advanced settings (hidden by default)
        self.advanced_frame = ttk.LabelFrame(page, text="Advanced Settings", padding=20)

        if self.advanced_mode.get():
            self.advanced_frame.pack(fill=tk.X, padx=30, pady=10)

        # LoRA settings in advanced
        lora_frame = ttk.Frame(self.advanced_frame)
        lora_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lora_frame, text="LoRA Rank:", width=20).pack(side=tk.LEFT)
        self.lora_rank_var = tk.IntVar(value=16)
        ttk.Spinbox(lora_frame, from_=4, to=128, textvariable=self.lora_rank_var, width=10).pack(side=tk.LEFT, padx=10)

        return page

    def _create_training_page(self, parent):
        """Create training progress page."""
        page = tk.Frame(parent, bg=self.colors['bg'])

        # Header
        ttk.Label(page, text="Training Progress", style='Title.TLabel').pack(
            anchor=tk.W, padx=30, pady=(30, 10)
        )

        # Status card
        status_card = ttk.LabelFrame(page, text="Status", padding=20)
        status_card.pack(fill=tk.X, padx=30, pady=10)

        self.training_status_label = ttk.Label(status_card, text="Ready to start training", style='Subheading.TLabel')
        self.training_status_label.pack(anchor=tk.W)

        # Progress bar
        progress_frame = ttk.Frame(page)
        progress_frame.pack(fill=tk.X, padx=30, pady=20)

        ttk.Label(progress_frame, text="Overall Progress:").pack(anchor=tk.W, pady=(0, 5))
        self.main_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.main_progress.pack(fill=tk.X)
        self.progress_label = ttk.Label(progress_frame, text="0%", style='Help.TLabel')
        self.progress_label.pack(anchor=tk.W, pady=(5, 0))

        # Metrics display
        metrics_frame = ttk.LabelFrame(page, text="Training Metrics", padding=20)
        metrics_frame.pack(fill=tk.X, padx=30, pady=10)

        # Create metric displays
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)

        self.metric_labels = {}
        metrics = [
            ('Epoch', '0/0'),
            ('Loss', '-.---'),
            ('Learning Rate', '-.--e-4'),
            ('Time Elapsed', '00:00:00'),
            ('Samples/sec', '-'),
            ('GPU Memory', '- GB')
        ]

        for i, (name, default) in enumerate(metrics):
            frame = ttk.Frame(metrics_grid)
            frame.grid(row=i // 3, column=i % 3, padx=20, pady=10, sticky='w')
            ttk.Label(frame, text=f"{name}:", style='Help.TLabel').pack(side=tk.LEFT)
            label = ttk.Label(frame, text=default, font=('Segoe UI', 10, 'bold'))
            label.pack(side=tk.LEFT, padx=(5, 0))
            self.metric_labels[name.lower().replace(' ', '_')] = label

        # Control buttons
        control_frame = ttk.Frame(page)
        control_frame.pack(pady=30)

        self.train_button = ttk.Button(
            control_frame,
            text="Start Training",
            style='Primary.TButton',
            command=self._start_training,
            width=20
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Training",
            command=self._stop_training,
            state=tk.DISABLED,
            width=20
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        return page

    def _create_export_page(self, parent):
        """Create export page."""
        page = tk.Frame(parent, bg=self.colors['bg'])

        # Header
        ttk.Label(page, text="Export Your Model", style='Title.TLabel').pack(
            anchor=tk.W, padx=30, pady=(30, 10)
        )

        ttk.Label(
            page,
            text="Choose how to export your trained model",
            style='Help.TLabel'
        ).pack(anchor=tk.W, padx=30, pady=(0, 20))

        # Export options
        export_options = [
            {
                'name': 'GGUF (Ollama/LM Studio)',
                'format': 'gguf',
                'description': 'Quantized format for local deployment with Ollama or LM Studio',
                'icon': '[GGUF]'
            },
            {
                'name': 'SafeTensors',
                'format': 'safetensors',
                'description': 'Modern format for HuggingFace models, faster and safer',
                'icon': '[Safe]'
            },
            {
                'name': 'PyTorch',
                'format': 'pytorch',
                'description': 'Standard PyTorch checkpoint for further training',
                'icon': '[PyTorch]'
            },
            {
                'name': 'HuggingFace Hub',
                'format': 'hub',
                'description': 'Upload directly to HuggingFace model hub',
                'icon': '[HF]'
            }
        ]

        self.export_format = tk.StringVar()

        for option in export_options:
            card = ttk.LabelFrame(page, padding=20)
            card.pack(fill=tk.X, padx=30, pady=10)

            # Header with radio button
            header_frame = ttk.Frame(card)
            header_frame.pack(fill=tk.X)

            ttk.Radiobutton(
                header_frame,
                text=f"{option['icon']} {option['name']}",
                variable=self.export_format,
                value=option['format'],
                style='Subheading.TLabel'
            ).pack(side=tk.LEFT)

            # Description
            ttk.Label(
                card,
                text=option['description'],
                style='Help.TLabel'
            ).pack(anchor=tk.W, pady=(5, 0), padx=(25, 0))

        # Export button
        ttk.Button(
            page,
            text="Export Model",
            style='Primary.TButton',
            command=self._export_model,
            width=30
        ).pack(pady=30)

        return page

    def _create_action_bar(self):
        """Create bottom action bar."""
        action_bar = ttk.Frame(self.root)
        action_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

        # Navigation buttons
        self.prev_button = ttk.Button(
            action_bar,
            text="< Previous",
            command=self._previous_step
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(
            action_bar,
            text="Next >",
            style='Primary.TButton',
            command=self._next_step
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(
            action_bar,
            text="Welcome! Select a model to begin.",
            style='Help.TLabel'
        )
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Quick start button
        ttk.Button(
            action_bar,
            text="Quick Start Wizard",
            style='Primary.TButton',
            command=self._show_quick_start
        ).pack(side=tk.RIGHT, padx=5)

    def _show_page(self, page_name):
        """Show a specific page and hide others."""
        for name, page in self.pages.items():
            if name == page_name:
                page.pack(fill=tk.BOTH, expand=True)
            else:
                page.pack_forget()

    def _navigate_to_step(self, step_index):
        """Navigate to a specific workflow step."""
        if step_index < self.current_step:
            # Allow going back
            self.current_step = step_index
        elif step_index == self.current_step + 1:
            # Allow going to next step if current is complete
            if self._validate_current_step():
                self.current_step = step_index
        else:
            # Can't skip steps
            messagebox.showinfo("Navigation", "Please complete the current step first.")
            return

        # Update UI
        self._update_workflow_display()

        # Show corresponding page
        page_map = ['model', 'dataset', 'config', 'training', 'export']
        self._show_page(page_map[self.current_step])

    def _next_step(self):
        """Go to next step in workflow."""
        if self.current_step < len(self.workflow_steps) - 1:
            if self._validate_current_step():
                self.current_step += 1
                self._update_workflow_display()

                page_map = ['model', 'dataset', 'config', 'training', 'export']
                self._show_page(page_map[self.current_step])

    def _previous_step(self):
        """Go to previous step in workflow."""
        if self.current_step > 0:
            self.current_step -= 1
            self._update_workflow_display()

            page_map = ['model', 'dataset', 'config', 'training', 'export']
            self._show_page(page_map[self.current_step])

    def _update_workflow_display(self):
        """Update workflow sidebar to reflect current step."""
        # Update step statuses
        for i, step in enumerate(self.workflow_steps):
            if i < self.current_step:
                step['status'] = 'completed'
            elif i == self.current_step:
                step['status'] = 'active'
            else:
                step['status'] = 'pending'

        # Remove old step frames from the container
        for frame in self.step_frames:
            frame.destroy()

        # Clear the list
        self.step_frames = []

        # Recreate all step indicators in the container
        for i, step in enumerate(self.workflow_steps):
            step_frame = self._create_step_indicator(self.steps_container, step, i)
            step_frame.pack(fill=tk.X, padx=20, pady=5)
            self.step_frames.append(step_frame)

        # Update navigation buttons
        self.prev_button.config(state=tk.NORMAL if self.current_step > 0 else tk.DISABLED)
        self.next_button.config(
            state=tk.NORMAL if self.current_step < len(self.workflow_steps) - 1 else tk.DISABLED
        )

    def _validate_current_step(self):
        """Validate current step before proceeding."""
        if self.current_step == 0:  # Model selection
            if not self.selected_model.get():
                messagebox.showwarning("Validation", "Please select a model before proceeding.")
                return False
        elif self.current_step == 1:  # Dataset selection
            if not hasattr(self, 'selected_dataset_info'):
                messagebox.showwarning("Validation", "Please select a dataset before proceeding.")
                return False
        elif self.current_step == 2:  # Configuration
            # Basic validation of settings
            try:
                epochs = self.epochs_var.get()
                batch = self.batch_var.get()
                lr = float(self.lr_var.get())
                if epochs < 1 or batch < 1 or lr <= 0:
                    raise ValueError()
            except:
                messagebox.showwarning("Validation", "Please ensure all training parameters are valid.")
                return False

        return True

    def _toggle_advanced_mode(self):
        """Toggle advanced mode on/off."""
        if self.advanced_mode.get():
            if hasattr(self, 'advanced_frame'):
                self.advanced_frame.pack(fill=tk.X, padx=30, pady=10, after=self.pages['config'].winfo_children()[3])
            self.status_label.config(text="Advanced mode enabled - additional settings available")
        else:
            if hasattr(self, 'advanced_frame'):
                self.advanced_frame.pack_forget()
            self.status_label.config(text="Advanced mode disabled - using simplified settings")

    def _show_quick_start(self):
        """Show quick start wizard."""
        from .quick_start_wizard import show_quick_start_wizard

        def on_wizard_complete(wizard_config):
            """Handle wizard completion."""
            # Apply wizard configuration
            self.config.update(wizard_config)

            # Update UI based on wizard results
            if wizard_config.get('model_name'):
                self.selected_model.set(wizard_config['model_name'])

            if wizard_config.get('dataset_path'):
                self.selected_dataset_info = {
                    'id': wizard_config['dataset_path'],
                    'name': wizard_config.get('dataset_display', wizard_config['dataset_path'])
                }

            if wizard_config.get('num_epochs'):
                self.epochs_var.set(wizard_config['num_epochs'])
            if wizard_config.get('batch_size'):
                self.batch_var.set(wizard_config['batch_size'])
            if wizard_config.get('learning_rate'):
                self.lr_var.set(str(wizard_config['learning_rate']))

            # If quick start mode, jump to training
            if wizard_config.get('quick_start'):
                # Validate steps 0-2 are complete
                self.current_step = 3
                self._update_workflow_display()
                self._show_page('training')
                self.status_label.config(text="Configuration loaded from Quick Start Wizard. Ready to train!")

                # Optionally auto-start training
                if messagebox.askyesno("Start Training", "Configuration complete! Start training now?"):
                    self._start_training()

        # Show wizard
        show_quick_start_wizard(self.root, self.system_config, on_wizard_complete)

    def _on_model_selected(self, model_name):
        """Handle model selection."""
        self.config['model_name'] = f"unsloth/{model_name}"
        self.status_label.config(text=f"Selected model: {model_name}")

    def _select_dataset(self, dataset_info):
        """Handle dataset selection."""
        self.selected_dataset_info = dataset_info
        self.selected_dataset.set(dataset_info['id'])
        self.config['dataset_path'] = dataset_info['id']
        self.status_label.config(text=f"Selected dataset: {dataset_info['name']}")

        # Auto-advance to next step
        self._next_step()

    def _update_dataset_view(self):
        """Update dataset view based on source selection."""
        source = self.dataset_source.get()

        for widget in self.dataset_content_frame.winfo_children():
            widget.destroy()

        if source == "library":
            self._create_dataset_library_view()
        elif source == "upload":
            self._create_upload_view()
        elif source == "hub":
            self._create_hub_view()

    def _create_upload_view(self):
        """Create file upload view."""
        upload_frame = ttk.Frame(self.dataset_content_frame)
        upload_frame.pack(fill=tk.BOTH, expand=True)

        # Instructions
        ttk.Label(
            upload_frame,
            text="Upload your dataset file (JSON, CSV, or Parquet format)",
            style='Subheading.TLabel'
        ).pack(pady=20)

        # File selection
        self.file_path_var = tk.StringVar()

        file_frame = ttk.Frame(upload_frame)
        file_frame.pack(pady=20)

        ttk.Entry(
            file_frame,
            textvariable=self.file_path_var,
            width=50,
            state='readonly'
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            file_frame,
            text="Browse Files",
            command=self._browse_file
        ).pack(side=tk.LEFT)

        # File info
        self.file_info_label = ttk.Label(upload_frame, text="", style='Help.TLabel')
        self.file_info_label.pack(pady=10)

    def _create_hub_view(self):
        """Create HuggingFace Hub view."""
        hub_frame = ttk.Frame(self.dataset_content_frame)
        hub_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            hub_frame,
            text="Enter HuggingFace dataset ID (e.g., 'tatsu-lab/alpaca')",
            style='Subheading.TLabel'
        ).pack(pady=20)

        entry_frame = ttk.Frame(hub_frame)
        entry_frame.pack(pady=20)

        self.hub_dataset_var = tk.StringVar()
        ttk.Entry(
            entry_frame,
            textvariable=self.hub_dataset_var,
            width=40
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            entry_frame,
            text="Load Dataset",
            style='Primary.TButton',
            command=self._load_hub_dataset
        ).pack(side=tk.LEFT)

    def _browse_file(self):
        """Browse for dataset file."""
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[
                ("All Supported", "*.json;*.jsonl;*.csv;*.parquet"),
                ("JSON files", "*.json;*.jsonl"),
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet")
            ]
        )

        if filename:
            self.file_path_var.set(filename)
            # Show file info
            file_size = Path(filename).stat().st_size / 1024 / 1024  # MB
            self.file_info_label.config(
                text=f"File: {Path(filename).name} ({file_size:.1f} MB)"
            )

            # Set config
            self.config['dataset_path'] = filename
            self.config['dataset_source'] = 'local'

            # Auto-advance
            self._next_step()

    def _load_hub_dataset(self):
        """Load dataset from HuggingFace Hub."""
        dataset_id = self.hub_dataset_var.get()
        if dataset_id:
            self.config['dataset_path'] = dataset_id
            self.config['dataset_source'] = 'huggingface'
            self.status_label.config(text=f"Loading dataset: {dataset_id}")
            self._next_step()

    def _apply_preset(self):
        """Apply training preset."""
        preset = self.training_preset.get()

        if preset == "quick":
            self.epochs_var.set(1)
            self.batch_var.set(8)
            self.lr_var.set("5e-4")
            self.status_label.config(text="Quick test preset applied - fast training for testing")
        elif preset == "balanced":
            self.epochs_var.set(3)
            self.batch_var.set(4)
            self.lr_var.set("2e-4")
            self.status_label.config(text="Balanced preset applied - good balance of speed and quality")
        elif preset == "quality":
            self.epochs_var.set(5)
            self.batch_var.set(2)
            self.lr_var.set("1e-4")
            self.status_label.config(text="High quality preset applied - slower but better results")

    def _load_smart_defaults(self):
        """Load smart defaults based on system capabilities."""
        # Set default model based on GPU
        if self.system_config.gpu_info:
            gpu_memory = self.system_config.gpu_info[0].memory_total / 1024  # GB

            if gpu_memory >= 16:
                self.selected_model.set("unsloth/Qwen3-8B")
            elif gpu_memory >= 8:
                self.selected_model.set("unsloth/Qwen3-4B")
            elif gpu_memory >= 4:
                self.selected_model.set("unsloth/Qwen3-1.7B")
            else:
                self.selected_model.set("unsloth/Qwen3-0.6B")

            # Adjust batch size based on GPU memory
            if gpu_memory >= 16:
                self.batch_var.set(8)
            elif gpu_memory >= 8:
                self.batch_var.set(4)
            else:
                self.batch_var.set(2)

    def _start_training(self):
        """Start training process."""
        # Gather config
        self.config.update({
            'num_epochs': self.epochs_var.get(),
            'batch_size': self.batch_var.get(),
            'learning_rate': float(self.lr_var.get()),
            'lora_rank': self.lora_rank_var.get() if self.advanced_mode.get() else 16
        })

        # Update UI
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.training_status_label.config(text="Training in progress...")

        # Start training thread (simplified for now)
        self.training_thread = threading.Thread(target=self._run_training)
        self.training_thread.daemon = True
        self.training_thread.start()

    def _run_training(self):
        """Run training (simplified version)."""
        import time
        import random

        epochs = self.config.get('num_epochs', 3)
        steps_per_epoch = 100

        for epoch in range(epochs):
            if not self.is_training:
                break

            for step in range(steps_per_epoch):
                if not self.is_training:
                    break

                # Update progress
                progress = (epoch * steps_per_epoch + step) / (epochs * steps_per_epoch)
                self.message_queue.put(('progress', progress))

                # Simulate metrics
                loss = 2.0 - (1.5 * progress) + random.uniform(-0.1, 0.1)
                self.message_queue.put(('metrics', {
                    'epoch': f"{epoch+1}/{epochs}",
                    'loss': f"{loss:.3f}",
                    'learning_rate': self.config.get('learning_rate', '2e-4')
                }))

                time.sleep(0.01)

        if self.is_training:
            self.message_queue.put(('complete', "Training completed successfully!"))

        self.is_training = False

    def _stop_training(self):
        """Stop training process."""
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop training?"):
            self.is_training = False
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.training_status_label.config(text="Training stopped")

    def _export_model(self):
        """Export trained model."""
        format_selected = self.export_format.get()
        if not format_selected:
            messagebox.showwarning("Export", "Please select an export format.")
            return

        # Get export path
        if format_selected != 'hub':
            file_path = filedialog.asksaveasfilename(
                title="Save Model As",
                defaultextension=".bin" if format_selected == 'pytorch' else "",
                filetypes=[("All Files", "*.*")]
            )

            if file_path:
                messagebox.showinfo("Export", f"Model will be exported to {file_path}")
        else:
            messagebox.showinfo("Export", "HuggingFace Hub upload coming soon!")

    def _setup_logging(self):
        """Setup logging for the application."""
        log_manager = LogManager()

        def log_callback(log_entry):
            # Process log entries if needed
            pass

        log_manager.setup_gui_handler(log_callback)

    def _process_messages(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()

                if msg_type == 'progress':
                    self.main_progress['value'] = msg_data * 100
                    self.progress_label.config(text=f"{int(msg_data * 100)}%")

                elif msg_type == 'metrics':
                    # Update metric labels
                    for key, value in msg_data.items():
                        label_key = key.lower().replace(' ', '_')
                        if label_key in self.metric_labels:
                            self.metric_labels[label_key].config(text=str(value))

                elif msg_type == 'complete':
                    messagebox.showinfo("Training Complete", msg_data)
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.training_status_label.config(text="Training completed!")
                    # Auto-advance to export
                    self.current_step = 4
                    self._update_workflow_display()
                    self._show_page('export')

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self._process_messages)

    def _load_config(self):
        """Load configuration from file."""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.config = json.load(f)
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def _save_config(self):
        """Save configuration to file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def _show_documentation(self):
        """Show documentation."""
        import webbrowser
        webbrowser.open("https://github.com/your-repo/grpo-gui/wiki")

    def _on_closing(self):
        """Handle application closing."""
        if self.is_training:
            if not messagebox.askyesno("Exit", "Training is in progress. Exit anyway?"):
                return

        self.root.quit()


def main():
    """Main entry point for improved GUI."""
    root = tk.Tk()
    app = SimplifiedGRPOApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
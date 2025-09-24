# -*- coding: utf-8 -*-
"""Quick Start Wizard for easy model training setup"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import json


class QuickStartWizard:
    """A simplified wizard for first-time users to quickly start training."""

    def __init__(self, parent, system_config, on_complete_callback):
        """Initialize the Quick Start Wizard.

        Args:
            parent: Parent tkinter window
            system_config: System configuration object
            on_complete_callback: Callback when wizard completes with config
        """
        self.parent = parent
        self.system_config = system_config
        self.on_complete_callback = on_complete_callback
        self.config = {}

        # Create wizard window
        self.window = tk.Toplevel(parent)
        self.window.title("Quick Start Wizard")
        self.window.geometry("800x600")
        self.window.transient(parent)
        self.window.grab_set()

        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

        # Set colors
        self.colors = {
            'primary': '#2196F3',
            'success': '#4CAF50',
            'bg': '#ffffff',
            'text': '#212121',
            'text_secondary': '#757575'
        }

        self.window.configure(bg=self.colors['bg'])

        # Create UI
        self._create_wizard_ui()

    def _create_wizard_ui(self):
        """Create the wizard interface."""
        # Main container
        main_frame = ttk.Frame(self.window, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Welcome header
        welcome_frame = ttk.Frame(main_frame)
        welcome_frame.pack(fill=tk.X, pady=(0, 30))

        title_label = ttk.Label(
            welcome_frame,
            text="Welcome to GRPO Model Trainer!",
            font=('Segoe UI', 20, 'bold')
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            welcome_frame,
            text="Let's get you started with training your first model in 3 simple steps",
            font=('Segoe UI', 11),
            foreground=self.colors['text_secondary']
        )
        subtitle_label.pack(pady=(10, 0))

        # Step indicator
        self.step_frame = ttk.Frame(main_frame)
        self.step_frame.pack(fill=tk.X, pady=20)

        self.steps = ['Choose Goal', 'Select Size', 'Pick Dataset']
        self.current_step = 0

        self._update_step_indicator()

        # Content area (will change based on step)
        self.content_frame = ttk.Frame(main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Navigation buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(20, 0))

        self.back_button = ttk.Button(
            nav_frame,
            text="< Back",
            command=self._previous_step,
            state=tk.DISABLED
        )
        self.back_button.pack(side=tk.LEFT)

        self.next_button = ttk.Button(
            nav_frame,
            text="Next >",
            command=self._next_step
        )
        self.next_button.pack(side=tk.RIGHT)

        self.skip_button = ttk.Button(
            nav_frame,
            text="Skip Wizard",
            command=self._skip_wizard
        )
        self.skip_button.pack(side=tk.RIGHT, padx=(0, 10))

        # Load first step
        self._show_step1_goal()

    def _update_step_indicator(self):
        """Update the step indicator display."""
        # Clear existing indicators
        for widget in self.step_frame.winfo_children():
            widget.destroy()

        for i, step_name in enumerate(self.steps):
            # Step container
            step_container = ttk.Frame(self.step_frame)
            step_container.pack(side=tk.LEFT, expand=True, fill=tk.X)

            # Circle indicator
            if i < self.current_step:
                # Completed
                bg = self.colors['success']
                text = "OK"
                fg = 'white'
            elif i == self.current_step:
                # Current
                bg = self.colors['primary']
                text = str(i + 1)
                fg = 'white'
            else:
                # Future
                bg = '#e0e0e0'
                text = str(i + 1)
                fg = self.colors['text_secondary']

            circle = tk.Label(
                step_container,
                text=text,
                width=3,
                height=1,
                bg=bg,
                fg=fg,
                font=('Segoe UI', 10, 'bold')
            )
            circle.pack()

            # Step name
            name_label = ttk.Label(
                step_container,
                text=step_name,
                font=('Segoe UI', 9, 'bold' if i == self.current_step else 'normal')
            )
            name_label.pack()

            # Connector line (except for last step)
            if i < len(self.steps) - 1:
                line = ttk.Frame(self.step_frame, height=2, style='TSeparator')
                line.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def _show_step1_goal(self):
        """Show step 1: Choose training goal."""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Question
        question_label = ttk.Label(
            self.content_frame,
            text="What would you like to train your model for?",
            font=('Segoe UI', 14, 'bold')
        )
        question_label.pack(pady=(0, 30))

        # Options
        self.goal_var = tk.StringVar(value="general")

        goals = [
            {
                'value': 'general',
                'title': '[Chat] General Chat & Assistance',
                'description': 'Train a helpful assistant for general conversation and tasks',
                'dataset': 'tatsu-lab/alpaca'
            },
            {
                'value': 'math',
                'title': '[Math] Mathematics & Problem Solving',
                'description': 'Optimize for mathematical reasoning and calculations',
                'dataset': 'openai/gsm8k'
            },
            {
                'value': 'code',
                'title': '[Code] Code Generation & Programming',
                'description': 'Focus on writing and explaining code',
                'dataset': 'sahil2801/CodeAlpaca-20k'
            },
            {
                'value': 'custom',
                'title': '[Custom] Custom / Advanced',
                'description': 'I want full control over the configuration',
                'dataset': None
            }
        ]

        for goal in goals:
            frame = ttk.Frame(self.content_frame, relief='ridge', borderwidth=1)
            frame.pack(fill=tk.X, pady=5, padx=50)

            radio = ttk.Radiobutton(
                frame,
                text=goal['title'],
                variable=self.goal_var,
                value=goal['value'],
                command=lambda g=goal: self._select_goal(g)
            )
            radio.pack(anchor=tk.W, padx=20, pady=(10, 5))

            desc_label = ttk.Label(
                frame,
                text=goal['description'],
                font=('Segoe UI', 9),
                foreground=self.colors['text_secondary']
            )
            desc_label.pack(anchor=tk.W, padx=(45, 20), pady=(0, 10))

            # Store goal data
            frame.goal_data = goal

    def _show_step2_size(self):
        """Show step 2: Select model size."""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Question
        question_label = ttk.Label(
            self.content_frame,
            text="How much GPU memory do you have?",
            font=('Segoe UI', 14, 'bold')
        )
        question_label.pack(pady=(0, 10))

        # Auto-detected info
        if self.system_config.gpu_info:
            gpu = self.system_config.gpu_info[0]
            gpu_memory_gb = gpu.memory_total / 1024

            info_label = ttk.Label(
                self.content_frame,
                text=f"Detected: {gpu.name} with {gpu_memory_gb:.1f}GB VRAM",
                font=('Segoe UI', 10),
                foreground=self.colors['success']
            )
            info_label.pack(pady=(0, 20))
        else:
            gpu_memory_gb = 0

        # Model size options
        self.size_var = tk.StringVar()

        sizes = [
            {
                'value': 'tiny',
                'title': 'Tiny (0.6B)',
                'vram': '1-2 GB',
                'model': 'unsloth/Qwen3-0.6B',
                'recommended_vram': 2
            },
            {
                'value': 'small',
                'title': 'Small (1.7B)',
                'vram': '3-4 GB',
                'model': 'unsloth/Qwen3-1.7B',
                'recommended_vram': 4
            },
            {
                'value': 'medium',
                'title': 'Medium (4B)',
                'vram': '6-8 GB',
                'model': 'unsloth/Qwen3-4B',
                'recommended_vram': 8
            },
            {
                'value': 'large',
                'title': 'Large (8B)',
                'vram': '12-16 GB',
                'model': 'unsloth/Qwen3-8B',
                'recommended_vram': 16
            }
        ]

        # Auto-select recommended size
        recommended_set = False
        for size in sizes:
            if gpu_memory_gb > 0 and gpu_memory_gb >= size['recommended_vram'] and not recommended_set:
                self.size_var.set(size['value'])
                recommended_set = True

        if not recommended_set:
            self.size_var.set('tiny')

        for size in sizes:
            frame = ttk.Frame(self.content_frame, relief='ridge', borderwidth=1)
            frame.pack(fill=tk.X, pady=5, padx=50)

            # Check if recommended
            is_recommended = (gpu_memory_gb > 0 and
                            gpu_memory_gb >= size['recommended_vram'] and
                            gpu_memory_gb < size.get('max_vram', float('inf')))

            header_frame = ttk.Frame(frame)
            header_frame.pack(fill=tk.X, padx=20, pady=(10, 5))

            radio = ttk.Radiobutton(
                header_frame,
                text=size['title'],
                variable=self.size_var,
                value=size['value'],
                command=lambda s=size: self._select_size(s)
            )
            radio.pack(side=tk.LEFT)

            if is_recommended:
                rec_label = tk.Label(
                    header_frame,
                    text="RECOMMENDED",
                    bg=self.colors['success'],
                    fg='white',
                    font=('Segoe UI', 8, 'bold'),
                    padx=5
                )
                rec_label.pack(side=tk.LEFT, padx=(10, 0))

            vram_label = ttk.Label(
                frame,
                text=f"Requires {size['vram']} VRAM",
                font=('Segoe UI', 9),
                foreground=self.colors['text_secondary']
            )
            vram_label.pack(anchor=tk.W, padx=(45, 20), pady=(0, 10))

            # Store size data
            frame.size_data = size

    def _show_step3_confirm(self):
        """Show step 3: Confirmation and dataset selection."""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Title
        title_label = ttk.Label(
            self.content_frame,
            text="Ready to Start Training!",
            font=('Segoe UI', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))

        # Summary frame
        summary_frame = ttk.LabelFrame(
            self.content_frame,
            text="Your Configuration",
            padding=20
        )
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=10)

        # Show selected configuration
        config_items = [
            ("Training Goal", self.config.get('goal_display', 'General')),
            ("Model", self.config.get('model_display', 'Qwen3-1.7B')),
            ("Dataset", self.config.get('dataset_display', 'Alpaca')),
            ("Training Mode", "Quick Start (Optimized settings)"),
            ("Estimated Time", self._estimate_training_time())
        ]

        for label, value in config_items:
            row_frame = ttk.Frame(summary_frame)
            row_frame.pack(fill=tk.X, pady=5)

            ttk.Label(
                row_frame,
                text=f"{label}:",
                font=('Segoe UI', 10),
                width=20
            ).pack(side=tk.LEFT)

            ttk.Label(
                row_frame,
                text=value,
                font=('Segoe UI', 10, 'bold')
            ).pack(side=tk.LEFT)

        # Tips
        tips_frame = ttk.Frame(self.content_frame)
        tips_frame.pack(fill=tk.X, padx=50, pady=(20, 0))

        ttk.Label(
            tips_frame,
            text="Tips:",
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor=tk.W)

        tips = [
            "Training will start immediately after clicking 'Start Training'",
            "You can monitor progress in the Training tab",
            "The model will be automatically saved when training completes"
        ]

        for tip in tips:
            ttk.Label(
                tips_frame,
                text=f"* {tip}",
                font=('Segoe UI', 9),
                foreground=self.colors['text_secondary']
            ).pack(anchor=tk.W, padx=(20, 0), pady=2)

        # Change next button text
        self.next_button.config(text="Start Training >", style='Primary.TButton')

    def _select_goal(self, goal_data):
        """Handle goal selection."""
        self.config['goal'] = goal_data['value']
        # Remove prefix in brackets if present
        title = goal_data['title']
        if '] ' in title:
            self.config['goal_display'] = title.split('] ', 1)[1]
        else:
            self.config['goal_display'] = title
        if goal_data['dataset']:
            self.config['dataset_path'] = goal_data['dataset']
            self.config['dataset_display'] = goal_data['dataset'].split('/')[-1].title()

    def _select_size(self, size_data):
        """Handle size selection."""
        self.config['model_name'] = size_data['model']
        self.config['model_display'] = size_data['title']

        # Set appropriate training parameters based on size
        if size_data['value'] == 'tiny':
            self.config['batch_size'] = 8
            self.config['num_epochs'] = 3
        elif size_data['value'] == 'small':
            self.config['batch_size'] = 4
            self.config['num_epochs'] = 3
        elif size_data['value'] == 'medium':
            self.config['batch_size'] = 2
            self.config['num_epochs'] = 2
        else:  # large
            self.config['batch_size'] = 1
            self.config['num_epochs'] = 2

        self.config['learning_rate'] = 2e-4
        self.config['lora_rank'] = 16

    def _estimate_training_time(self):
        """Estimate training time based on configuration."""
        # Simple estimation based on model size and epochs
        model = self.config.get('model_name', '')
        epochs = self.config.get('num_epochs', 3)

        if '0.6B' in model:
            base_time = 10
        elif '1.7B' in model:
            base_time = 20
        elif '4B' in model:
            base_time = 40
        elif '8B' in model:
            base_time = 60
        else:
            base_time = 30

        total_minutes = base_time * epochs

        if total_minutes < 60:
            return f"~{total_minutes} minutes"
        else:
            hours = total_minutes / 60
            return f"~{hours:.1f} hours"

    def _next_step(self):
        """Move to next step in wizard."""
        if self.current_step == 0:
            # Validate step 1
            if not self.goal_var.get():
                messagebox.showwarning("Selection Required", "Please select a training goal.")
                return

            if self.goal_var.get() == 'custom':
                # Skip wizard for custom configuration
                self.window.destroy()
                return

            self.current_step = 1
            self._show_step2_size()

        elif self.current_step == 1:
            # Validate step 2
            if not self.size_var.get():
                messagebox.showwarning("Selection Required", "Please select a model size.")
                return

            self.current_step = 2
            self._show_step3_confirm()

        elif self.current_step == 2:
            # Complete wizard
            self._complete_wizard()

        # Update UI
        self._update_step_indicator()
        self.back_button.config(state=tk.NORMAL if self.current_step > 0 else tk.DISABLED)

    def _previous_step(self):
        """Go back to previous step."""
        if self.current_step > 0:
            self.current_step -= 1

            if self.current_step == 0:
                self._show_step1_goal()
                self.back_button.config(state=tk.DISABLED)
            elif self.current_step == 1:
                self._show_step2_size()
                self.next_button.config(text="Next >")

            self._update_step_indicator()

    def _skip_wizard(self):
        """Skip the wizard and use defaults."""
        response = messagebox.askyesno(
            "Skip Wizard",
            "Skip the wizard and use default settings?"
        )

        if response:
            # Set minimal defaults
            self.config = {
                'model_name': 'unsloth/Qwen3-1.7B',
                'dataset_path': 'tatsu-lab/alpaca',
                'num_epochs': 3,
                'batch_size': 4,
                'learning_rate': 2e-4,
                'wizard_skipped': True
            }
            self.window.destroy()
            if self.on_complete_callback:
                self.on_complete_callback(self.config)

    def _complete_wizard(self):
        """Complete the wizard and apply configuration."""
        # Add wizard completion flag
        self.config['wizard_completed'] = True
        self.config['quick_start'] = True

        # Close wizard
        self.window.destroy()

        # Call completion callback
        if self.on_complete_callback:
            self.on_complete_callback(self.config)


def show_quick_start_wizard(parent, system_config, on_complete):
    """Show the Quick Start Wizard.

    Args:
        parent: Parent tkinter window
        system_config: System configuration object
        on_complete: Callback function when wizard completes

    Returns:
        QuickStartWizard instance
    """
    return QuickStartWizard(parent, system_config, on_complete)
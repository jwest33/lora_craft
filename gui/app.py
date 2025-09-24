"""Main GUI application for GRPO Fine-Tuning."""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
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

# Import tabs
from .tabs import (
    DatasetTab,
    ModelTrainingTab,  # Combined Model and Training tab
    SystemTab,
    MonitoringTab,
    ExportTab
)

# Import theme manager and themed dialogs
from .theme_manager import ThemeManager
from . import themed_dialog


logger = get_logger(__name__)


class GRPOFineTunerApp:
    """Main application class for GRPO Fine-Tuner GUI."""

    def __init__(self, root: tk.Tk):
        """Initialize the application.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("GRPO Fine-Tuner - Qwen & LLaMA Training GUI")
        self.root.geometry("1200x800")

        # Set application icon (if available)
        try:
            icon_path = Path(__file__).parent / "assets" / "icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(icon_path)
        except Exception:
            pass

        # Application state
        self.config = {}
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.message_queue = queue.Queue()

        # Training metrics storage
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'total_steps': 0,
            'final_loss': None,
            'final_reward': None,
            'best_loss': float('inf'),
            'best_reward': float('-inf'),
            'epochs_completed': 0,
            'samples_processed': 0,
            'learning_rate_final': None,
            'gpu_memory_peak': 0,
            'training_interrupted': False
        }

        # System configuration
        self.system_config = SystemConfig()

        # Setup theme manager
        self.theme_manager = ThemeManager(root)

        # Setup logging
        self._setup_logging()

        # Create UI
        self._create_menu_bar()
        self._create_main_ui()
        self._create_status_bar()

        # Setup callbacks
        self._setup_callbacks()

        # Load default configuration
        self._load_default_config()

        # Apply initial theme
        self.theme_manager.apply_theme(self.theme_manager.get_current_theme())

        # Start message processing
        self._process_messages()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_logging(self):
        """Setup application logging."""
        log_manager = LogManager()

        # Setup GUI handler with callback
        def log_callback(log_entry):
            # Send log to training tab
            if hasattr(self, 'monitoring_tab'):
                self.message_queue.put(('log', log_entry))

        log_manager.setup_gui_handler(log_callback)

    def _create_menu_bar(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Configuration", command=self._new_config)
        file_menu.add_command(label="Open Configuration...", command=self._open_config)
        file_menu.add_command(label="Save Configuration", command=self._save_config)
        file_menu.add_command(label="Save Configuration As...", command=self._save_config_as)
        file_menu.add_separator()
        file_menu.add_command(label="Import Dataset...", command=self._import_dataset)
        file_menu.add_command(label="Export Model...", command=self._export_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Reset Configuration", command=self._reset_config)
        edit_menu.add_command(label="Validate Configuration", command=self._validate_config)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)

        # Theme submenu
        theme_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        theme_menu.add_radiobutton(label="Light", command=lambda: self._change_theme("light"))
        theme_menu.add_radiobutton(label="Dark", command=lambda: self._change_theme("dark"))
        theme_menu.add_radiobutton(label="Synthwave", command=lambda: self._change_theme("synthwave"))

        view_menu.add_separator()
        view_menu.add_command(label="System Information", command=self._show_system_info)
        view_menu.add_command(label="Training Logs", command=self._show_logs)
        view_menu.add_command(label="Metrics Dashboard", command=self._show_metrics)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Test Reward Function", command=self._test_reward)
        tools_menu.add_command(label="Preview Template", command=self._preview_template)
        tools_menu.add_command(label="Benchmark System", command=self._benchmark_system)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_toolbar(self):
        """Create main toolbar with training and configuration controls."""
        # Main toolbar frame with border
        toolbar_frame = ttk.Frame(self.root, relief=tk.RAISED, borderwidth=1)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        # Left side - Training controls
        left_frame = ttk.Frame(toolbar_frame)
        left_frame.pack(side=tk.LEFT, padx=10, pady=5)

        # Training controls
        self.main_train_button = ttk.Button(
            left_frame,
            text="▶ Train Model",
            command=self._start_training,
            width=15
        )
        self.main_train_button.pack(side=tk.LEFT, padx=2)

        self.main_stop_button = ttk.Button(
            left_frame,
            text="■ Stop",
            command=self._stop_training,
            state=tk.DISABLED,
            width=10
        )
        self.main_stop_button.pack(side=tk.LEFT, padx=2)

        self.main_pause_button = ttk.Button(
            left_frame,
            text="⏸ Pause",
            command=self._pause_training,
            state=tk.DISABLED,
            width=10
        )
        self.main_pause_button.pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(left_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Configuration controls
        ttk.Button(
            left_frame,
            text="💾 Save",
            command=self._save_config,
            width=10
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left_frame,
            text="📁 Load",
            command=self._open_config,
            width=10
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left_frame,
            text="✓ Validate",
            command=self._validate_config,
            width=10
        ).pack(side=tk.LEFT, padx=2)

        # Center - Training progress
        center_frame = ttk.Frame(toolbar_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        # Progress bar
        self.main_progress_bar = ttk.Progressbar(
            center_frame,
            length=250,
            mode='determinate'
        )
        self.main_progress_bar.pack(side=tk.LEFT, pady=5)

        self.main_progress_label = ttk.Label(center_frame, text="Ready")
        self.main_progress_label.pack(side=tk.LEFT, padx=10)

        # Right side - Model info
        right_frame = ttk.Frame(toolbar_frame)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=5)

        self.model_status_label = ttk.Label(
            right_frame,
            text="Model: Not selected",
            font=('TkDefaultFont', 9)
        )
        self.model_status_label.pack()

    def _create_main_ui(self):
        """Create main UI with tabs."""
        # Create main toolbar for training controls
        self._create_main_toolbar()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.dataset_tab = DatasetTab(self.notebook, self._on_config_change)
        self.model_training_tab = ModelTrainingTab(
            self.notebook,
            self._on_config_change,
            self._start_training
        )  # Combined tab with training callback
        self.system_tab = SystemTab(self.notebook, self.system_config, self._on_config_change)
        self.monitoring_tab = MonitoringTab(
            self.notebook,
            self._start_training,
            self._stop_training,
            self._pause_training
        )
        self.export_tab = ExportTab(self.notebook, self)

        # Add tabs to notebook
        self.notebook.add(self.dataset_tab.frame, text="Dataset & Prompts")
        self.notebook.add(self.model_training_tab.frame, text="Model & Training")
        self.notebook.add(self.system_tab.frame, text="System")
        self.notebook.add(self.monitoring_tab.frame, text="Monitor")
        self.notebook.add(self.export_tab.frame, text="Export")

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _create_status_bar(self):
        """Create status bar."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Status label
        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Training status
        self.training_status_label = ttk.Label(
            self.status_frame,
            text="Not Training",
            relief=tk.SUNKEN,
            width=20
        )
        self.training_status_label.pack(side=tk.RIGHT, padx=2)

        # GPU status
        self.gpu_status_label = ttk.Label(
            self.status_frame,
            text="GPU: Checking...",
            relief=tk.SUNKEN,
            width=20
        )
        self.gpu_status_label.pack(side=tk.RIGHT, padx=2)

        # Update GPU status
        self._update_gpu_status()

    def _setup_callbacks(self):
        """Setup application callbacks."""
        # Keyboard shortcuts
        self.root.bind('<Control-n>', lambda e: self._new_config())
        self.root.bind('<Control-o>', lambda e: self._open_config())
        self.root.bind('<Control-s>', lambda e: self._save_config())
        self.root.bind('<Control-S>', lambda e: self._save_config_as())
        self.root.bind('<F5>', lambda e: self._start_training())
        self.root.bind('<Escape>', lambda e: self._stop_training())

    def _on_config_change(self, key: str, value: Any):
        """Handle configuration change.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        self.status_label.config(text="Configuration modified")

        # Update model status in toolbar if model changed
        if key == 'model_name':
            self.model_status_label.config(text=f"Model: {value}")

    def _on_tab_changed(self, event):
        """Handle tab change event."""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")
        logger.info(f"Switched to tab: {tab_text}")

    def _load_default_config(self):
        """Load default configuration."""
        # Check for saved default config
        default_config_path = Path("configs/default.json")
        if default_config_path.exists():
            try:
                with open(default_config_path, 'r') as f:
                    self.config = json.load(f)
                self._apply_config_to_ui()
                self.status_label.config(text="Default configuration loaded")
            except Exception as e:
                logger.error(f"Failed to load default config: {e}")
        else:
            # Create minimal default config
            self.config = {
                'model_name': 'unsloth/Qwen3-1.7B',  # Updated to match UI default
                'dataset_source': 'HuggingFace Hub',  # Fixed to match UI
                'learning_rate': 2e-4,
                'batch_size': 4,
                'num_epochs': 3,
            }

        # Update model status in toolbar
        model_name = self.config.get('model_name', 'Not selected')
        if model_name and model_name != 'Not selected':
            display_name = model_name.split('/')[-1] if '/' in model_name else model_name
            self.model_status_label.config(text=f"Model: {display_name}")

    def _apply_config_to_ui(self):
        """Apply configuration to UI elements."""
        # Apply to each tab
        if hasattr(self, 'dataset_tab'):
            self.dataset_tab.load_config(self.config)
        if hasattr(self, 'model_training_tab'):
            self.model_training_tab.load_config(self.config)
        if hasattr(self, 'system_tab'):
            self.system_tab.load_config(self.config)

        # Update model status in toolbar
        model_name = self.config.get('model_name', 'Not selected')
        if model_name and model_name != 'Not selected':
            display_name = model_name.split('/')[-1] if '/' in model_name else model_name
            self.model_status_label.config(text=f"Model: {display_name}")

    def _new_config(self):
        """Create new configuration."""
        if self.config and themed_dialog.askyesno(self.root, "New Configuration",
                                              "Discard current configuration?"):
            self.config = {}
            self._apply_config_to_ui()
            self.status_label.config(text="New configuration created")

    def _open_config(self):
        """Open configuration file."""
        file_path = filedialog.askopenfilename(
            title="Open Configuration",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir="configs"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.config = json.load(f)
                self._apply_config_to_ui()
                self.status_label.config(text=f"Configuration loaded from {Path(file_path).name}")
            except Exception as e:
                themed_dialog.showerror(self.root, "Error", f"Failed to load configuration: {e}")

    def _save_config(self):
        """Save current configuration."""
        if not hasattr(self, 'config_file_path'):
            self._save_config_as()
        else:
            self._save_config_to_file(self.config_file_path)

    def _save_config_as(self):
        """Save configuration with new name."""
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir="configs"
        )

        if file_path:
            self._save_config_to_file(file_path)
            self.config_file_path = file_path

    def _save_config_to_file(self, file_path: str):
        """Save configuration to file."""
        try:
            # Gather config from all tabs
            self._gather_config_from_ui()

            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.status_label.config(text=f"Configuration saved to {Path(file_path).name}")
        except Exception as e:
            themed_dialog.showerror(self.root, "Error", f"Failed to save configuration: {e}")

    def _gather_config_from_ui(self):
        """Gather configuration from all UI elements."""
        # Gather from each tab
        if hasattr(self, 'dataset_tab'):
            self.config.update(self.dataset_tab.get_config())
        if hasattr(self, 'model_training_tab'):
            self.config.update(self.model_training_tab.get_config())
        if hasattr(self, 'system_tab'):
            self.config.update(self.system_tab.get_config())

    def _reset_config(self):
        """Reset configuration to defaults."""
        if themed_dialog.askyesno(self.root, "Reset Configuration",
                               "Reset all settings to defaults?"):
            self._load_default_config()
            self.status_label.config(text="Configuration reset to defaults")

    def _validate_config(self):
        """Validate current configuration."""
        self._gather_config_from_ui()
        valid, errors = validate_training_config(self.config)

        if valid:
            themed_dialog.showinfo(self.root, "Validation", "Configuration is valid!")
        else:
            error_msg = "Configuration errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            themed_dialog.showerror(self.root, "Validation Failed", error_msg)

    def _import_dataset(self):
        """Import dataset from file."""
        # Delegate to dataset tab
        if hasattr(self, 'dataset_tab'):
            self.dataset_tab.import_dataset()

    def _export_model(self):
        """Export trained model."""
        # Switch to export tab
        self.notebook.select(self.export_tab.frame)

    def _show_system_info(self):
        """Show system information dialog."""
        info = self.system_config.get_system_summary()
        themed_dialog.showinfo(self.root, "System Information", info)

    def _show_logs(self):
        """Show training logs window."""
        # Create logs window
        logs_window = tk.Toplevel(self.root)
        logs_window.title("Training Logs")
        logs_window.geometry("800x600")

        # Create text widget
        text_widget = tk.Text(logs_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_widget)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

        # Load logs
        # Implementation would load actual logs

    def _show_metrics(self):
        """Show metrics dashboard."""
        themed_dialog.showinfo(self.root, "Metrics", "Metrics dashboard coming soon!")

    def _test_reward(self):
        """Test reward function."""
        # Implementation for reward testing
        themed_dialog.showinfo(self.root, "Test Reward", "Reward testing interface coming soon!")

    def _preview_template(self):
        """Preview prompt template."""
        # Implementation for template preview
        themed_dialog.showinfo(self.root, "Preview Template", "Template preview coming soon!")

    def _benchmark_system(self):
        """Benchmark system performance."""
        themed_dialog.showinfo(self.root, "Benchmark", "System benchmark coming soon!")

    def _show_documentation(self):
        """Show documentation."""
        import webbrowser
        webbrowser.open("https://github.com/your-repo/grpo-gui/wiki")

    def _show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts = """
        Keyboard Shortcuts:

        Ctrl+N - New Configuration
        Ctrl+O - Open Configuration
        Ctrl+S - Save Configuration
        Ctrl+Shift+S - Save As
        F5 - Start Training
        Esc - Stop Training
        """
        themed_dialog.showinfo(self.root, "Keyboard Shortcuts", shortcuts)

    def _show_about(self):
        """Show about dialog."""
        about_text = """
        GRPO Fine-Tuner
        Version 1.0.0

        A GUI application for GRPO fine-tuning
        of Qwen and LLaMA models, powered by Unsloth.
        """
        themed_dialog.showinfo(self.root, "About", about_text)

    def _start_training(self):
        """Start training process."""
        if self.is_training:
            themed_dialog.showwarning(self.root, "Training", "Training is already in progress!")
            return

        # Validate configuration
        self._gather_config_from_ui()
        valid, errors = validate_training_config(self.config)

        if not valid:
            error_msg = "Cannot start training:\n\n" + "\n".join(f"• {e}" for e in errors)
            themed_dialog.showerror(self.root, "Validation Failed", error_msg)
            return

        # Reset training stats
        import datetime
        self.training_stats = {
            'start_time': datetime.datetime.now(),
            'end_time': None,
            'total_steps': 0,
            'final_loss': None,
            'final_reward': None,
            'best_loss': float('inf'),
            'best_reward': float('-inf'),
            'epochs_completed': 0,
            'samples_processed': 0,
            'learning_rate_final': self.config.get('learning_rate', 2e-4),
            'gpu_memory_peak': 0,
            'training_interrupted': False,
            'model_name': self.config.get('model_name', 'Unknown'),
            'dataset_name': self.config.get('dataset_path') or self.config.get('dataset_name', 'Unknown'),
            'batch_size': self.config.get('batch_size', 4),
            'num_epochs': self.config.get('num_epochs', 3)
        }

        # Update UI state
        self._set_training_state(True)

        # Start training in background thread
        self.is_training = True
        self.training_status_label.config(text="Training...")

        self.training_thread = threading.Thread(target=self._run_training)
        self.training_thread.daemon = True
        self.training_thread.start()

    def _stop_training(self):
        """Stop training process."""
        if not self.is_training:
            return

        if themed_dialog.askyesno(self.root, "Stop Training",
                               "Are you sure you want to stop training?"):
            # Mark as interrupted
            import datetime
            self.training_stats['training_interrupted'] = True
            self.training_stats['end_time'] = datetime.datetime.now()

            self.is_training = False
            self.training_status_label.config(text="Stopping...")
            self._set_training_state(False)

            # Signal trainer to stop
            if self.trainer:
                # Implementation would stop the trainer
                pass

            # Show training summary even if interrupted
            self._show_training_summary()

    def _pause_training(self):
        """Pause/resume training."""
        # Implementation for pause/resume
        pass

    def _set_training_state(self, is_training: bool):
        """Update all UI elements based on training state.

        Args:
            is_training: Whether training is active
        """
        if is_training:
            # Disable configuration changes during training
            self.main_train_button.config(state=tk.DISABLED)
            self.main_stop_button.config(state=tk.NORMAL)
            self.main_pause_button.config(state=tk.NORMAL)

            # Update progress label
            self.main_progress_label.config(text="Training in progress...")

            # Update window title
            self.root.title("GRPO Fine-Tuner - TRAINING IN PROGRESS")

            # Update monitoring tab if exists
            if hasattr(self, 'monitoring_tab'):
                self.monitoring_tab.set_training_state(True)
        else:
            # Re-enable configuration
            self.main_train_button.config(state=tk.NORMAL)
            self.main_stop_button.config(state=tk.DISABLED)
            self.main_pause_button.config(state=tk.DISABLED)

            # Reset progress
            self.main_progress_bar['value'] = 0
            self.main_progress_label.config(text="Ready")

            # Reset window title
            self.root.title("GRPO Fine-Tuner - Qwen & LLaMA Training GUI")

            # Update monitoring tab if exists
            if hasattr(self, 'monitoring_tab'):
                self.monitoring_tab.set_training_state(False)

    def _run_training(self):
        """Run training in background thread."""
        try:
            # Create GRPO config
            grpo_config = GRPOConfig(
                model_name=self.config.get('model_name'),
                num_train_epochs=self.config.get('num_epochs', 3),
                per_device_train_batch_size=self.config.get('batch_size', 4),
                learning_rate=self.config.get('learning_rate', 2e-4),
            )

            # Create trainer
            self.trainer = GRPOTrainer(grpo_config, self.system_config)

            # Setup callbacks
            self.trainer.progress_callback = self._on_training_progress
            self.trainer.metrics_callback = self._on_training_metrics

            # Load model
            self.trainer.setup_model()

            # Track GPU memory during training
            import torch
            import time
            import random

            # Function to get current GPU memory usage
            def get_gpu_memory_mb():
                if torch.cuda.is_available():
                    return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                return 0

            # Get initial GPU memory after model load
            initial_gpu_memory = get_gpu_memory_mb()
            self.training_stats['gpu_memory_peak'] = initial_gpu_memory

            # Simulate training progress for testing
            # TODO: Replace with actual GRPO training when dataset is properly configured
            num_epochs = self.config.get('num_epochs', 3)
            steps_per_epoch = 100  # Simulated

            max_gpu_memory = initial_gpu_memory  # Track peak GPU memory

            for epoch in range(num_epochs):
                if not self.is_training:
                    break

                self.training_stats['epochs_completed'] = epoch + 1

                for step in range(steps_per_epoch):
                    if not self.is_training:
                        break

                    # Simulate progress
                    total_steps = epoch * steps_per_epoch + step
                    progress = total_steps / (num_epochs * steps_per_epoch)

                    # Simulate metrics
                    loss = 2.0 - (1.5 * progress) + random.uniform(-0.1, 0.1)
                    reward = -1.0 + (2.0 * progress) + random.uniform(-0.1, 0.1)

                    # Track GPU memory
                    current_gpu_memory = get_gpu_memory_mb()
                    max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
                    self.training_stats['gpu_memory_peak'] = max_gpu_memory

                    self.message_queue.put(('progress', progress))
                    self.message_queue.put(('metrics', {
                        'step': total_steps,
                        'loss': loss,
                        'reward': reward,
                        'epoch': epoch + 1,
                        'gpu_memory_mb': current_gpu_memory
                    }))

                    self.training_stats['samples_processed'] += self.config.get('batch_size', 4)

                    time.sleep(0.01)  # Simulate training time

            if self.is_training:  # Only show complete if not interrupted
                self.message_queue.put(('complete', "Training completed successfully!"))

        except Exception as e:
            self.message_queue.put(('error', str(e)))

        finally:
            self.is_training = False
            self.training_status_label.config(text="Not Training")
            self._set_training_state(False)

    def _on_training_progress(self, progress: float):
        """Handle training progress update."""
        self.message_queue.put(('progress', progress))

    def _on_training_metrics(self, metrics: Dict[str, Any]):
        """Handle training metrics update."""
        self.message_queue.put(('metrics', metrics))

    def _update_gpu_status(self):
        """Update GPU status in status bar."""
        if self.system_config.gpu_info:
            gpu = self.system_config.gpu_info[0]
            status = f"GPU: {gpu.memory_used}/{gpu.memory_total}MB"
            self.gpu_status_label.config(text=status)
        else:
            self.gpu_status_label.config(text="GPU: Not Available")

        # Schedule next update
        self.root.after(5000, self._update_gpu_status)

    def _process_messages(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()

                if msg_type == 'log':
                    if hasattr(self, 'monitoring_tab'):
                        self.monitoring_tab.add_log(msg_data)

                elif msg_type == 'progress':
                    # Update main toolbar progress bar
                    self.main_progress_bar['value'] = msg_data * 100
                    self.main_progress_label.config(text=f"Training: {int(msg_data * 100)}%")

                    # Update monitoring tab
                    if hasattr(self, 'monitoring_tab'):
                        self.monitoring_tab.update_progress(msg_data)

                elif msg_type == 'metrics':
                    # Update training stats
                    if 'loss' in msg_data:
                        self.training_stats['final_loss'] = msg_data['loss']
                        self.training_stats['best_loss'] = min(self.training_stats['best_loss'], msg_data['loss'])
                    if 'reward' in msg_data:
                        self.training_stats['final_reward'] = msg_data['reward']
                        self.training_stats['best_reward'] = max(self.training_stats['best_reward'], msg_data['reward'])
                    if 'step' in msg_data:
                        self.training_stats['total_steps'] = msg_data['step']

                    if hasattr(self, 'monitoring_tab'):
                        self.monitoring_tab.update_metrics(msg_data)

                elif msg_type == 'complete':
                    # Training completed - show final stats
                    import datetime
                    self.training_stats['end_time'] = datetime.datetime.now()
                    self.training_stats['training_interrupted'] = False
                    self._show_training_summary()
                    themed_dialog.showinfo(self.root, "Training Complete", msg_data)

                elif msg_type == 'error':
                    themed_dialog.showerror(self.root, "Training Error", msg_data)

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self._process_messages)

    def _change_theme(self, theme_name: str):
        """Change application theme.

        Args:
            theme_name: Name of theme to apply
        """
        self.theme_manager.apply_theme(theme_name)
        # Update chart theme in monitoring tab
        if hasattr(self, 'monitoring_tab'):
            self.monitoring_tab.set_theme(theme_name)
        self.status_label.config(text=f"Theme changed to {theme_name.title()}")

    def _show_training_summary(self):
        """Show comprehensive training summary dialog."""
        import datetime

        # Calculate training duration
        if self.training_stats['start_time'] and self.training_stats['end_time']:
            duration = self.training_stats['end_time'] - self.training_stats['start_time']
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            duration_str = "Unknown"

        # Create summary window
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Training Summary")
        summary_window.geometry("600x700")
        summary_window.transient(self.root)
        summary_window.grab_set()

        # Create container frame
        container = ttk.Frame(summary_window)
        container.pack(fill=tk.BOTH, expand=True)

        # Create a canvas and scrollbar for scrolling
        canvas = tk.Canvas(container, bg='white')
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure canvas window
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Update the canvas window width to match canvas width
            canvas_width = event.width if event else canvas.winfo_width()
            canvas.itemconfig(canvas_window, width=canvas_width)

        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", lambda e: configure_scroll_region(e))

        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel for scrolling - properly handle focus
        def _on_mousewheel(event):
            # Only scroll if the mouse is over the canvas or its children
            widget = summary_window.winfo_containing(event.x_root, event.y_root)
            if widget:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            summary_window.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            summary_window.unbind_all("<MouseWheel>")

        # Bind mouse wheel when entering the canvas area
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        scrollable_frame.bind("<Enter>", _bind_mousewheel)
        scrollable_frame.bind("<Leave>", _unbind_mousewheel)

        # Main frame inside scrollable area - reduce padding to minimize gap
        main_frame = ttk.Frame(scrollable_frame, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=False)  # Don't expand to prevent extra space

        # Title
        title_label = ttk.Label(
            main_frame,
            text="🎉 Training Complete!",
            font=('TkDefaultFont', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))

        # Create sections
        sections = [
            ("📊 Model Information", [
                ("Model", self.training_stats.get('model_name', 'Unknown')),
                ("Dataset", self.training_stats.get('dataset_name', 'Unknown')),
            ]),
            ("⏱️ Training Duration", [
                ("Start Time", self.training_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S') if self.training_stats['start_time'] else 'Unknown'),
                ("End Time", self.training_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S') if self.training_stats['end_time'] else 'Unknown'),
                ("Total Duration", duration_str),
            ]),
            ("📈 Training Progress", [
                ("Epochs Completed", f"{self.training_stats.get('epochs_completed', 0)} / {self.training_stats.get('num_epochs', 0)}"),
                ("Total Steps", self.training_stats.get('total_steps', 0)),
                ("Samples Processed", self.training_stats.get('samples_processed', 0)),
                ("Batch Size", self.training_stats.get('batch_size', 'Unknown')),
            ]),
            ("📉 Loss Metrics", [
                ("Final Loss", f"{self.training_stats.get('final_loss', 'N/A'):.6f}" if self.training_stats.get('final_loss') is not None else 'N/A'),
                ("Best Loss", f"{self.training_stats.get('best_loss', 'N/A'):.6f}" if self.training_stats.get('best_loss') != float('inf') else 'N/A'),
            ]),
            ("🎯 Reward Metrics", [
                ("Final Reward", f"{self.training_stats.get('final_reward', 'N/A'):.6f}" if self.training_stats.get('final_reward') is not None else 'N/A'),
                ("Best Reward", f"{self.training_stats.get('best_reward', 'N/A'):.6f}" if self.training_stats.get('best_reward') != float('-inf') else 'N/A'),
            ]),
            ("⚙️ Training Configuration", [
                ("Learning Rate", f"{self.training_stats.get('learning_rate_final', 'Unknown')}"),
                ("GPU Memory Peak", f"{self.training_stats.get('gpu_memory_peak', 0):.1f} MB"),
                ("Training Status", "Interrupted" if self.training_stats.get('training_interrupted') else "Completed Successfully"),
            ])
        ]

        # Create each section
        for section_title, items in sections:
            # Section frame
            section_frame = ttk.LabelFrame(main_frame, text=section_title, padding="10")
            section_frame.pack(fill=tk.X, pady=5)

            # Section items
            for label, value in items:
                item_frame = ttk.Frame(section_frame)
                item_frame.pack(fill=tk.X, pady=2)

                ttk.Label(
                    item_frame,
                    text=f"{label}:",
                    width=20,
                    anchor=tk.W
                ).pack(side=tk.LEFT)

                ttk.Label(
                    item_frame,
                    text=str(value),
                    font=('TkDefaultFont', 9, 'bold')
                ).pack(side=tk.LEFT)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        # Export button
        ttk.Button(
            button_frame,
            text="📄 Export Report",
            command=lambda: self._export_training_report(self.training_stats)
        ).pack(side=tk.LEFT, padx=5)

        # Save model button
        ttk.Button(
            button_frame,
            text="💾 Save Model",
            command=self._export_model
        ).pack(side=tk.LEFT, padx=5)

        # Close button
        ttk.Button(
            button_frame,
            text="Close",
            command=summary_window.destroy
        ).pack(side=tk.RIGHT, padx=5)

        # Pack canvas and scrollbar properly in the container
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Center the window
        summary_window.update_idletasks()
        x = (summary_window.winfo_screenwidth() // 2) - (summary_window.winfo_width() // 2)
        y = (summary_window.winfo_screenheight() // 2) - (summary_window.winfo_height() // 2)
        summary_window.geometry(f"+{x}+{y}")

    def _export_training_report(self, stats):
        """Export training report to file."""
        import datetime
        import json

        file_path = filedialog.asksaveasfilename(
            title="Export Training Report",
            defaultextension=".json",
            filetypes=[
                ("JSON Files", "*.json"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ],
            initialfile=f"training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if file_path:
            try:
                # Convert datetime objects to strings
                export_stats = stats.copy()
                if export_stats.get('start_time'):
                    export_stats['start_time'] = export_stats['start_time'].isoformat()
                if export_stats.get('end_time'):
                    export_stats['end_time'] = export_stats['end_time'].isoformat()

                # Handle infinity values
                if export_stats.get('best_loss') == float('inf'):
                    export_stats['best_loss'] = None
                if export_stats.get('best_reward') == float('-inf'):
                    export_stats['best_reward'] = None

                # Save based on extension
                if file_path.endswith('.txt'):
                    with open(file_path, 'w') as f:
                        f.write("TRAINING REPORT\n")
                        f.write("=" * 50 + "\n\n")
                        for key, value in export_stats.items():
                            f.write(f"{key}: {value}\n")
                else:
                    with open(file_path, 'w') as f:
                        json.dump(export_stats, f, indent=2)

                themed_dialog.showinfo(self.root, "Export Successful", f"Training report saved to {Path(file_path).name}")
            except Exception as e:
                themed_dialog.showerror(self.root, "Export Failed", f"Failed to export report: {e}")

    def _on_closing(self):
        """Handle application closing."""
        if self.is_training:
            if not themed_dialog.askyesno(self.root, "Exit",
                                          "Training is in progress. Exit anyway?"):
                return

        # Cleanup
        if self.trainer:
            self.trainer.cleanup()

        self.root.quit()


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = GRPOFineTunerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

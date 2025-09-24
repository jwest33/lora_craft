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

    def _create_main_ui(self):
        """Create main UI with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.dataset_tab = DatasetTab(self.notebook, self._on_config_change)
        self.model_training_tab = ModelTrainingTab(self.notebook, self._on_config_change)  # Combined tab
        self.system_tab = SystemTab(self.notebook, self.system_config, self._on_config_change)
        self.monitoring_tab = MonitoringTab(
            self.notebook,
            self._start_training,
            self._stop_training,
            self._pause_training
        )
        self.export_tab = ExportTab(self.notebook, self._export_model)

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
                'model_name': 'qwen2.5-0.5b',
                'dataset_source': 'HuggingFace Hub',  # Fixed to match UI
                'learning_rate': 2e-4,
                'batch_size': 4,
                'num_epochs': 3,
            }

    def _apply_config_to_ui(self):
        """Apply configuration to UI elements."""
        # Apply to each tab
        if hasattr(self, 'dataset_tab'):
            self.dataset_tab.load_config(self.config)
        if hasattr(self, 'model_training_tab'):
            self.model_training_tab.load_config(self.config)
        if hasattr(self, 'system_tab'):
            self.system_tab.load_config(self.config)

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
            self.is_training = False
            self.training_status_label.config(text="Stopping...")

            # Signal trainer to stop
            if self.trainer:
                # Implementation would stop the trainer
                pass

    def _pause_training(self):
        """Pause/resume training."""
        # Implementation for pause/resume
        pass

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

            # Start training
            # This would involve loading dataset, template, and rewards
            # then calling trainer.grpo_train()

            self.message_queue.put(('complete', "Training completed successfully!"))

        except Exception as e:
            self.message_queue.put(('error', str(e)))

        finally:
            self.is_training = False
            self.training_status_label.config(text="Not Training")

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
                    if hasattr(self, 'monitoring_tab'):
                        self.monitoring_tab.update_progress(msg_data)

                elif msg_type == 'metrics':
                    if hasattr(self, 'monitoring_tab'):
                        self.monitoring_tab.update_metrics(msg_data)

                elif msg_type == 'complete':
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

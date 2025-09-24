# -*- coding: utf-8 -*-
"""User-friendly error handling and help system"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import logging

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Provides user-friendly error messages and solutions."""

    # Common errors and their user-friendly explanations
    ERROR_MAPPINGS = {
        'CUDA out of memory': {
            'title': 'GPU Memory Error',
            'message': 'Your GPU ran out of memory during training.',
            'solution': 'Try reducing the batch size or using a smaller model.',
            'actions': ['reduce_batch_size', 'select_smaller_model']
        },
        'No module named': {
            'title': 'Missing Dependency',
            'message': 'A required package is not installed.',
            'solution': 'Install missing dependencies using: pip install -r requirements.txt',
            'actions': ['show_install_command']
        },
        'Connection error': {
            'title': 'Network Error',
            'message': 'Unable to download the dataset or model.',
            'solution': 'Check your internet connection and try again.',
            'actions': ['retry', 'use_local_file']
        },
        'Invalid configuration': {
            'title': 'Configuration Error',
            'message': 'Your training configuration has invalid settings.',
            'solution': 'Please review and correct the highlighted settings.',
            'actions': ['show_validation', 'reset_to_defaults']
        },
        'Dataset not found': {
            'title': 'Dataset Error',
            'message': 'The specified dataset could not be found.',
            'solution': 'Verify the dataset name or try a different dataset.',
            'actions': ['browse_datasets', 'upload_file']
        }
    }

    @classmethod
    def handle_error(cls, parent, error, context=None):
        """Handle an error with user-friendly messaging.

        Args:
            parent: Parent tkinter widget
            error: The exception that occurred
            context: Additional context about what was happening
        """
        error_str = str(error)
        error_type = type(error).__name__

        # Log the full error for debugging
        logger.error(f"Error in {context or 'unknown context'}: {error_str}", exc_info=True)

        # Find matching error pattern
        error_info = None
        for pattern, info in cls.ERROR_MAPPINGS.items():
            if pattern.lower() in error_str.lower():
                error_info = info
                break

        if error_info:
            cls._show_friendly_error(parent, error_info, error_str)
        else:
            cls._show_generic_error(parent, error_type, error_str, context)

    @classmethod
    def _show_friendly_error(cls, parent, error_info, original_error):
        """Show a user-friendly error dialog."""
        dialog = FriendlyErrorDialog(parent, error_info, original_error)

    @classmethod
    def _show_generic_error(cls, parent, error_type, error_str, context):
        """Show a generic error dialog for unrecognized errors."""
        message = f"An error occurred"
        if context:
            message += f" while {context}"
        message += ".\n\n"

        if len(error_str) > 200:
            message += error_str[:200] + "..."
        else:
            message += error_str

        message += "\n\nPlease check your settings and try again."

        messagebox.showerror(f"Error: {error_type}", message)


class FriendlyErrorDialog:
    """A user-friendly error dialog with solutions."""

    def __init__(self, parent, error_info, original_error):
        """Create a friendly error dialog.

        Args:
            parent: Parent tkinter widget
            error_info: Dictionary with error information
            original_error: Original error string
        """
        self.parent = parent
        self.error_info = error_info
        self.original_error = original_error

        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(error_info['title'])
        self.dialog.geometry("600x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the window
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

        self._create_ui()

    def _create_ui(self):
        """Create the dialog UI."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Error icon and title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        # Error icon (using text for simplicity)
        icon_label = ttk.Label(
            title_frame,
            text="(!)",
            font=('Segoe UI', 24)
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 15))

        # Error title
        title_label = ttk.Label(
            title_frame,
            text=self.error_info['title'],
            font=('Segoe UI', 14, 'bold')
        )
        title_label.pack(side=tk.LEFT)

        # Error message
        message_label = ttk.Label(
            main_frame,
            text=self.error_info['message'],
            wraplength=550,
            font=('Segoe UI', 10)
        )
        message_label.pack(anchor=tk.W, pady=(0, 15))

        # Solution section
        solution_frame = ttk.LabelFrame(main_frame, text="Suggested Solution", padding="15")
        solution_frame.pack(fill=tk.X, pady=(0, 15))

        solution_label = ttk.Label(
            solution_frame,
            text=self.error_info['solution'],
            wraplength=520,
            font=('Segoe UI', 10)
        )
        solution_label.pack(anchor=tk.W)

        # Action buttons
        if self.error_info.get('actions'):
            action_frame = ttk.Frame(main_frame)
            action_frame.pack(fill=tk.X, pady=(0, 15))

            for action in self.error_info['actions']:
                if action == 'reduce_batch_size':
                    ttk.Button(
                        action_frame,
                        text="Reduce Batch Size",
                        command=self._reduce_batch_size
                    ).pack(side=tk.LEFT, padx=5)
                elif action == 'select_smaller_model':
                    ttk.Button(
                        action_frame,
                        text="Select Smaller Model",
                        command=self._select_smaller_model
                    ).pack(side=tk.LEFT, padx=5)
                elif action == 'retry':
                    ttk.Button(
                        action_frame,
                        text="Retry",
                        command=self._retry
                    ).pack(side=tk.LEFT, padx=5)

        # Details section (collapsible)
        self.details_visible = tk.BooleanVar(value=False)
        details_button = ttk.Button(
            main_frame,
            text="Show Details ▼",
            command=self._toggle_details
        )
        details_button.pack(anchor=tk.W, pady=(0, 10))

        # Details text (hidden by default)
        self.details_frame = ttk.Frame(main_frame)

        details_text = tk.Text(
            self.details_frame,
            height=8,
            width=70,
            wrap=tk.WORD,
            font=('Courier', 9)
        )
        details_text.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(details_text, command=details_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        details_text.config(yscrollcommand=scrollbar.set)

        # Insert error details
        details_text.insert('1.0', self.original_error)
        details_text.config(state=tk.DISABLED)

        # Close button
        close_button = ttk.Button(
            main_frame,
            text="Close",
            command=self.dialog.destroy
        )
        close_button.pack(pady=(10, 0))

    def _toggle_details(self):
        """Toggle visibility of error details."""
        if self.details_visible.get():
            self.details_frame.pack_forget()
            self.details_visible.set(False)
        else:
            self.details_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            self.details_visible.set(True)

    def _reduce_batch_size(self):
        """Action to reduce batch size."""
        # This would communicate with the main app to reduce batch size
        messagebox.showinfo(
            "Batch Size",
            "Batch size will be reduced. Please try training again."
        )
        self.dialog.destroy()

    def _select_smaller_model(self):
        """Action to select a smaller model."""
        # This would communicate with the main app to show model selection
        messagebox.showinfo(
            "Model Selection",
            "Please select a smaller model from the Model Selection page."
        )
        self.dialog.destroy()

    def _retry(self):
        """Action to retry the operation."""
        self.dialog.destroy()


class HelpSystem:
    """Provides contextual help throughout the application."""

    HELP_TOPICS = {
        'model_selection': {
            'title': 'Choosing the Right Model',
            'content': """
Model size affects both quality and resource requirements:

* **Tiny (0.6B)**: Best for testing and edge devices
  - Fastest training and inference
  - Lowest memory requirements (1-2GB VRAM)
  - Good for simple tasks

* **Small (1.7B)**: Balanced performance
  - Good quality for most tasks
  - Moderate memory requirements (3-4GB VRAM)
  - Recommended for beginners

* **Medium (4B)**: Enhanced capabilities
  - Better reasoning and understanding
  - Higher memory requirements (6-8GB VRAM)
  - Good for complex tasks

* **Large (8B)**: Professional quality
  - Best performance and accuracy
  - Highest memory requirements (12-16GB VRAM)
  - Suitable for production use
"""
        },
        'dataset_selection': {
            'title': 'Understanding Datasets',
            'content': """
Datasets determine what your model learns:

* **Alpaca**: General instruction-following
  - 52K examples of questions and answers
  - Good starting point for chat models

* **GSM8K**: Mathematics focus
  - Grade school math problems
  - Improves reasoning abilities

* **Code Alpaca**: Programming focus
  - Code generation and explanation
  - Best for coding assistants

* **Custom Dataset**: Your own data
  - Upload JSON, CSV, or Parquet files
  - Must have instruction and response columns
"""
        },
        'training_parameters': {
            'title': 'Training Parameters Explained',
            'content': """
Key parameters that affect training:

* **Epochs**: Number of complete passes through dataset
  - More epochs = better learning (but risk overfitting)
  - Typical: 1-3 for fine-tuning

* **Batch Size**: Samples processed together
  - Larger = faster but more memory
  - Smaller = slower but more stable
  - Auto-adjusted based on GPU memory

* **Learning Rate**: How fast the model learns
  - Higher = faster but less stable
  - Lower = slower but more precise
  - Default: 2e-4 is good for most cases

* **LoRA Rank**: Complexity of fine-tuning
  - Higher = more parameters to train
  - Lower = faster but less flexible
  - Default: 16 is balanced
"""
        }
    }

    @classmethod
    def show_help(cls, parent, topic):
        """Show help dialog for a specific topic.

        Args:
            parent: Parent tkinter widget
            topic: Help topic key
        """
        if topic in cls.HELP_TOPICS:
            help_info = cls.HELP_TOPICS[topic]
            HelpDialog(parent, help_info['title'], help_info['content'])
        else:
            messagebox.showinfo("Help", "Help topic not found.")


class HelpDialog:
    """A help dialog window."""

    def __init__(self, parent, title, content):
        """Create a help dialog.

        Args:
            parent: Parent tkinter widget
            title: Dialog title
            content: Help content (supports markdown-like formatting)
        """
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Help: {title}")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)

        # Center the window
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

        # Create UI
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text=title,
            font=('Segoe UI', 14, 'bold')
        )
        title_label.pack(anchor=tk.W, pady=(0, 15))

        # Content
        text_widget = tk.Text(
            main_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            relief=tk.FLAT,
            bg='#f5f5f5'
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_widget, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

        # Parse and insert formatted content
        self._format_content(text_widget, content)

        text_widget.config(state=tk.DISABLED)

        # Close button
        ttk.Button(
            main_frame,
            text="Close",
            command=self.dialog.destroy
        ).pack(pady=(15, 0))

    def _format_content(self, text_widget, content):
        """Format and insert content with basic markdown support."""
        lines = content.strip().split('\n')

        for line in lines:
            if line.startswith('* **') and '**:' in line:
                # Bullet point with bold title
                parts = line.split('**:', 1)
                title = parts[0].replace('* **', '').strip()
                desc = parts[1].strip() if len(parts) > 1 else ''

                text_widget.insert(tk.END, '* ')
                text_widget.insert(tk.END, title, 'bold')
                text_widget.insert(tk.END, f': {desc}\n')

            elif line.startswith('  -'):
                # Sub-bullet
                text_widget.insert(tk.END, f'    {line[2:]}\n', 'indent')

            else:
                # Normal line
                text_widget.insert(tk.END, f'{line}\n')

        # Configure tags
        text_widget.tag_configure('bold', font=('Segoe UI', 10, 'bold'))
        text_widget.tag_configure('indent', lmargin1=40, lmargin2=40)
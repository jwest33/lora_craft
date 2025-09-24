"""Custom themed dialogs for the application."""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple, List
import os


class ThemedDialog:
    """Base class for themed dialogs."""

    def __init__(self, parent, title: str, message: str):
        """Initialize themed dialog.

        Args:
            parent: Parent window
            title: Dialog title
            message: Dialog message
        """
        self.result = None
        self.parent = parent

        # Create toplevel window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.resizable(False, False)

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center dialog on parent window
        self._center_on_parent()

        # Create content
        self._create_content(message)

        # Bind escape key
        self.dialog.bind('<Escape>', lambda e: self.cancel())

        # Focus on first button
        self.dialog.after(100, self._set_focus)

    def _center_on_parent(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()

        # Get parent position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Get dialog size
        dialog_width = self.dialog.winfo_reqwidth()
        dialog_height = self.dialog.winfo_reqheight()

        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        # Ensure dialog is on screen
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        x = min(max(0, x), screen_width - dialog_width)
        y = min(max(0, y), screen_height - dialog_height)

        self.dialog.geometry(f"+{x}+{y}")

    def _create_content(self, message: str):
        """Create dialog content. To be overridden by subclasses."""
        raise NotImplementedError

    def _set_focus(self):
        """Set focus to appropriate widget."""
        pass

    def cancel(self):
        """Cancel and close dialog."""
        self.result = None
        self.dialog.destroy()

    def show(self):
        """Show dialog and return result."""
        self.dialog.wait_window()
        return self.result


class MessageBox(ThemedDialog):
    """Themed message box dialog."""

    ICONS = {
        'info': 'ℹ',
        'warning': '⚠',
        'error': '✕',
        'question': '?'
    }

    COLORS = {
        'info': '#17a2b8',
        'warning': '#ffc107',
        'error': '#dc3545',
        'question': '#6c757d'
    }

    def __init__(self, parent, title: str, message: str,
                 msg_type: str = 'info', buttons: List[str] = None):
        """Initialize message box.

        Args:
            parent: Parent window
            title: Dialog title
            message: Dialog message
            msg_type: Type of message (info, warning, error, question)
            buttons: List of button labels (default: ['OK'])
        """
        self.msg_type = msg_type
        self.buttons = buttons or ['OK']
        super().__init__(parent, title, message)

    def _create_content(self, message: str):
        """Create message box content."""
        # Main frame with padding
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Icon and message frame
        msg_frame = ttk.Frame(main_frame)
        msg_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Icon (if not using emoji)
        # For now, using text icon
        icon_label = ttk.Label(
            msg_frame,
            text=self.ICONS.get(self.msg_type, ''),
            font=('TkDefaultFont', 24, 'bold')
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 15))

        # Apply color to icon
        if self.msg_type in self.COLORS:
            icon_label.configure(foreground=self.COLORS[self.msg_type])

        # Message
        msg_label = ttk.Label(
            msg_frame,
            text=message,
            wraplength=400,
            justify=tk.LEFT
        )
        msg_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Create buttons
        self.button_widgets = []
        for i, label in enumerate(self.buttons):
            btn = ttk.Button(
                button_frame,
                text=label,
                command=lambda l=label: self._button_click(l),
                width=10
            )
            btn.pack(side=tk.RIGHT, padx=(5, 0) if i > 0 else 0)
            self.button_widgets.append(btn)

        # Set first button as default
        if self.button_widgets:
            self.button_widgets[0].configure(style='Accent.TButton')
            self.dialog.bind('<Return>', lambda e: self._button_click(self.buttons[0]))

    def _set_focus(self):
        """Set focus to first button."""
        if self.button_widgets:
            self.button_widgets[0].focus_set()

    def _button_click(self, label: str):
        """Handle button click."""
        self.result = label
        self.dialog.destroy()


class FileDialog(ThemedDialog):
    """Themed file selection dialog."""

    def __init__(self, parent, title: str = "Select File",
                 mode: str = "open", filetypes: List[Tuple[str, str]] = None,
                 initialdir: str = None):
        """Initialize file dialog.

        Args:
            parent: Parent window
            title: Dialog title
            mode: Dialog mode (open, save, directory)
            filetypes: List of (description, pattern) tuples
            initialdir: Initial directory
        """
        self.mode = mode
        self.filetypes = filetypes or [("All Files", "*.*")]
        self.initialdir = initialdir or os.getcwd()
        super().__init__(parent, title, "")

    def _create_content(self, message: str):
        """Create file dialog content."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Path entry frame
        path_frame = ttk.Frame(main_frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(path_frame, text="Path:").pack(side=tk.LEFT, padx=(0, 10))

        self.path_var = tk.StringVar(value=self.initialdir)
        self.path_entry = ttk.Entry(path_frame, textvariable=self.path_var, width=50)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_btn = ttk.Button(path_frame, text="Browse", command=self._browse)
        browse_btn.pack(side=tk.LEFT)

        # File list frame
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # File listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            height=15,
            width=60
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.configure(command=self.file_listbox.yview)

        # Bind double-click
        self.file_listbox.bind('<Double-Button-1>', lambda e: self._select_file())

        # File type filter
        if len(self.filetypes) > 1:
            filter_frame = ttk.Frame(main_frame)
            filter_frame.pack(fill=tk.X, pady=(0, 10))

            ttk.Label(filter_frame, text="File Type:").pack(side=tk.LEFT, padx=(0, 10))

            self.filter_var = tk.StringVar(value=self.filetypes[0][0])
            filter_combo = ttk.Combobox(
                filter_frame,
                textvariable=self.filter_var,
                values=[ft[0] for ft in self.filetypes],
                state='readonly',
                width=30
            )
            filter_combo.pack(side=tk.LEFT)
            filter_combo.bind('<<ComboboxSelected>>', lambda e: self._update_file_list())

        # Filename entry for save mode
        if self.mode == 'save':
            name_frame = ttk.Frame(main_frame)
            name_frame.pack(fill=tk.X, pady=(0, 10))

            ttk.Label(name_frame, text="File Name:").pack(side=tk.LEFT, padx=(0, 10))

            self.name_var = tk.StringVar()
            name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=40)
            name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Select/Save button
        action_text = "Save" if self.mode == 'save' else "Select"
        self.action_btn = ttk.Button(
            button_frame,
            text=action_text,
            command=self._select_file,
            style='Accent.TButton'
        )
        self.action_btn.pack(side=tk.RIGHT)

        # Populate file list
        self._update_file_list()

    def _browse(self):
        """Browse for directory using standard dialog."""
        # Fall back to standard tkinter dialog for now
        from tkinter import filedialog

        if self.mode == 'directory':
            path = filedialog.askdirectory(
                parent=self.dialog,
                initialdir=self.path_var.get()
            )
        elif self.mode == 'save':
            path = filedialog.asksaveasfilename(
                parent=self.dialog,
                initialdir=self.path_var.get(),
                filetypes=self.filetypes
            )
        else:
            path = filedialog.askopenfilename(
                parent=self.dialog,
                initialdir=self.path_var.get(),
                filetypes=self.filetypes
            )

        if path:
            self.result = path
            self.dialog.destroy()

    def _update_file_list(self):
        """Update the file list based on current path and filter."""
        # This is a simplified implementation
        # In a real implementation, would list files from the directory
        pass

    def _select_file(self):
        """Select the current file."""
        # Simplified - just use the browse functionality for now
        self._browse()


class ProgressDialog(ThemedDialog):
    """Themed progress dialog."""

    def __init__(self, parent, title: str, message: str,
                 maximum: int = 100, show_cancel: bool = True):
        """Initialize progress dialog.

        Args:
            parent: Parent window
            title: Dialog title
            message: Progress message
            maximum: Maximum progress value
            show_cancel: Show cancel button
        """
        self.maximum = maximum
        self.show_cancel = show_cancel
        self.cancelled = False
        super().__init__(parent, title, message)

    def _create_content(self, message: str):
        """Create progress dialog content."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Message label
        self.msg_label = ttk.Label(main_frame, text=message, wraplength=400)
        self.msg_label.pack(fill=tk.X, pady=(0, 15))

        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            orient='horizontal',
            length=400,
            mode='determinate',
            maximum=self.maximum
        )
        self.progress.pack(fill=tk.X, pady=(0, 10))

        # Progress text
        self.progress_text = ttk.Label(main_frame, text="0%")
        self.progress_text.pack(fill=tk.X, pady=(0, 15))

        # Cancel button
        if self.show_cancel:
            cancel_btn = ttk.Button(
                main_frame,
                text="Cancel",
                command=self._cancel
            )
            cancel_btn.pack()

    def update_progress(self, value: int, message: str = None):
        """Update progress bar and message.

        Args:
            value: Progress value
            message: Optional new message
        """
        self.progress['value'] = value
        percentage = int((value / self.maximum) * 100)
        self.progress_text.configure(text=f"{percentage}%")

        if message:
            self.msg_label.configure(text=message)

        self.dialog.update()

    def _cancel(self):
        """Handle cancel button."""
        self.cancelled = True
        self.cancel()


class InputDialog(ThemedDialog):
    """Themed input dialog."""

    def __init__(self, parent, title: str, prompt: str,
                 initial_value: str = "", multiline: bool = False):
        """Initialize input dialog.

        Args:
            parent: Parent window
            title: Dialog title
            prompt: Input prompt
            initial_value: Initial input value
            multiline: Use multiline text input
        """
        self.initial_value = initial_value
        self.multiline = multiline
        super().__init__(parent, title, prompt)

    def _create_content(self, prompt: str):
        """Create input dialog content."""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Prompt label
        prompt_label = ttk.Label(main_frame, text=prompt, wraplength=400)
        prompt_label.pack(fill=tk.X, pady=(0, 10))

        # Input field
        if self.multiline:
            # Text widget for multiline
            text_frame = ttk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

            scrollbar = ttk.Scrollbar(text_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            self.input_widget = tk.Text(
                text_frame,
                height=10,
                width=50,
                yscrollcommand=scrollbar.set
            )
            self.input_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.configure(command=self.input_widget.yview)

            if self.initial_value:
                self.input_widget.insert('1.0', self.initial_value)
        else:
            # Entry widget for single line
            self.input_var = tk.StringVar(value=self.initial_value)
            self.input_widget = ttk.Entry(
                main_frame,
                textvariable=self.input_var,
                width=50
            )
            self.input_widget.pack(fill=tk.X, pady=(0, 15))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # OK button
        ok_btn = ttk.Button(
            button_frame,
            text="OK",
            command=self._ok,
            style='Accent.TButton'
        )
        ok_btn.pack(side=tk.RIGHT)

        # Bind enter key for single line
        if not self.multiline:
            self.dialog.bind('<Return>', lambda e: self._ok())

    def _set_focus(self):
        """Set focus to input widget."""
        self.input_widget.focus_set()
        if not self.multiline and self.initial_value:
            self.input_widget.selection_range(0, tk.END)

    def _ok(self):
        """Handle OK button."""
        if self.multiline:
            self.result = self.input_widget.get('1.0', 'end-1c')
        else:
            self.result = self.input_var.get()
        self.dialog.destroy()


# Convenience functions that mimic tkinter.messagebox API
def showinfo(parent, title: str, message: str) -> str:
    """Show info message box."""
    dialog = MessageBox(parent, title, message, 'info')
    return dialog.show()


def showwarning(parent, title: str, message: str) -> str:
    """Show warning message box."""
    dialog = MessageBox(parent, title, message, 'warning')
    return dialog.show()


def showerror(parent, title: str, message: str) -> str:
    """Show error message box."""
    dialog = MessageBox(parent, title, message, 'error')
    return dialog.show()


def askyesno(parent, title: str, message: str) -> bool:
    """Show yes/no question dialog."""
    dialog = MessageBox(parent, title, message, 'question', ['Yes', 'No'])
    result = dialog.show()
    return result == 'Yes'


def askyesnocancel(parent, title: str, message: str) -> Optional[bool]:
    """Show yes/no/cancel question dialog."""
    dialog = MessageBox(parent, title, message, 'question', ['Yes', 'No', 'Cancel'])
    result = dialog.show()
    if result == 'Yes':
        return True
    elif result == 'No':
        return False
    else:
        return None


def askstring(parent, title: str, prompt: str, initial_value: str = "") -> Optional[str]:
    """Show string input dialog."""
    dialog = InputDialog(parent, title, prompt, initial_value, multiline=False)
    return dialog.show()


def asktext(parent, title: str, prompt: str, initial_value: str = "") -> Optional[str]:
    """Show multiline text input dialog."""
    dialog = InputDialog(parent, title, prompt, initial_value, multiline=True)
    return dialog.show()
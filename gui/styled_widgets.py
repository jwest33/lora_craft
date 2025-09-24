"""Professional styled widgets for the application."""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any, Callable
from .style_constants import FONTS, SPACING, PADDING, COLORS, DIMENSIONS


class StyledFrame(ttk.Frame):
    """Enhanced frame with professional styling."""

    def __init__(self, parent, title: Optional[str] = None, padding: str = 'frame', **kwargs):
        """Initialize styled frame.

        Args:
            parent: Parent widget
            title: Optional title for the frame
            padding: Padding size key from PADDING dict
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        if title:
            # Create a titled section with professional styling
            self._create_titled_section(title, padding)
        else:
            # Just apply padding
            pad_value = PADDING.get(padding, PADDING['frame'])
            self.configure(padding=pad_value)

    def _create_titled_section(self, title: str, padding: str):
        """Create a titled section with header."""
        # Header frame
        header_frame = ttk.Frame(self, style='SectionHeader.TFrame')
        header_frame.pack(fill=tk.X, padx=0, pady=(0, SPACING['sm']))

        # Title label with larger font
        title_label = ttk.Label(
            header_frame,
            text=title,
            font=FONTS['h2'],
            style='SectionHeader.TLabel'
        )
        title_label.pack(side=tk.LEFT, padx=SPACING['md'], pady=SPACING['sm'])

        # Separator
        ttk.Separator(self, orient='horizontal').pack(fill=tk.X, padx=SPACING['sm'])

        # Content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=PADDING[padding], pady=SPACING['md'])


class StyledLabelFrame(ttk.LabelFrame):
    """Enhanced label frame with professional styling."""

    def __init__(self, parent, text: str, padding: str = 'section', **kwargs):
        """Initialize styled label frame.

        Args:
            parent: Parent widget
            text: Frame title
            padding: Padding size key
            **kwargs: Additional options
        """
        super().__init__(parent, text=text, **kwargs)

        # Apply padding
        pad_value = PADDING.get(padding, PADDING['section'])
        self.configure(padding=pad_value)

        # Use h3 font for label
        self.configure(style='Styled.TLabelframe')


class StyledButton(ttk.Button):
    """Enhanced button with hover effects and styling."""

    def __init__(self, parent, text: str, command: Optional[Callable] = None,
                 style_type: str = 'default', **kwargs):
        """Initialize styled button.

        Args:
            parent: Parent widget
            text: Button text
            command: Click callback
            style_type: Button style ('default', 'primary', 'accent', 'danger')
            **kwargs: Additional options
        """
        style_map = {
            'default': 'Styled.TButton',
            'primary': 'Primary.TButton',
            'accent': 'Accent.TButton',
            'danger': 'Danger.TButton',
        }

        super().__init__(
            parent,
            text=text,
            command=command,
            style=style_map.get(style_type, 'Styled.TButton'),
            cursor='hand2',
            **kwargs
        )


class StyledEntry(ttk.Entry):
    """Enhanced entry with better visual feedback."""

    def __init__(self, parent, textvariable=None, placeholder: str = "", **kwargs):
        """Initialize styled entry.

        Args:
            parent: Parent widget
            textvariable: Text variable
            placeholder: Placeholder text
            **kwargs: Additional options
        """
        super().__init__(parent, textvariable=textvariable, style='Styled.TEntry', **kwargs)

        self.placeholder = placeholder
        self.placeholder_shown = False

        if placeholder and not (textvariable and textvariable.get()):
            self._show_placeholder()

        # Bind focus events
        self.bind('<FocusIn>', self._on_focus_in)
        self.bind('<FocusOut>', self._on_focus_out)

    def _show_placeholder(self):
        """Show placeholder text."""
        if not self.get():
            self.insert(0, self.placeholder)
            self.configure(foreground='gray')
            self.placeholder_shown = True

    def _hide_placeholder(self):
        """Hide placeholder text."""
        if self.placeholder_shown:
            self.delete(0, tk.END)
            self.configure(foreground='')
            self.placeholder_shown = False

    def _on_focus_in(self, event):
        """Handle focus in event."""
        self._hide_placeholder()

    def _on_focus_out(self, event):
        """Handle focus out event."""
        if not self.get():
            self._show_placeholder()


class StyledLabel(ttk.Label):
    """Enhanced label with typography options."""

    def __init__(self, parent, text: str, style_type: str = 'body', **kwargs):
        """Initialize styled label.

        Args:
            parent: Parent widget
            text: Label text
            style_type: Typography style from FONTS
            **kwargs: Additional options
        """
        font = FONTS.get(style_type, FONTS['body'])
        super().__init__(parent, text=text, font=font, **kwargs)


class SectionHeader(ttk.Frame):
    """Section header with title and optional controls."""

    def __init__(self, parent, title: str, subtitle: str = ""):
        """Initialize section header.

        Args:
            parent: Parent widget
            title: Section title
            subtitle: Optional subtitle
        """
        super().__init__(parent, style='SectionHeader.TFrame')

        # Title
        self.title_label = StyledLabel(self, title, style_type='h2')
        self.title_label.pack(side=tk.TOP, anchor=tk.W, padx=SPACING['md'], pady=(SPACING['sm'], 0))

        # Subtitle
        if subtitle:
            self.subtitle_label = StyledLabel(self, subtitle, style_type='small_italic')
            self.subtitle_label.configure(foreground='gray')
            self.subtitle_label.pack(side=tk.TOP, anchor=tk.W, padx=SPACING['md'], pady=(0, SPACING['sm']))

        # Separator
        ttk.Separator(self, orient='horizontal').pack(fill=tk.X, pady=(SPACING['xs'], 0))


class FormRow(ttk.Frame):
    """A form row with label and input field."""

    def __init__(self, parent, label: str, widget_type: str = 'entry',
                 description: str = "", **widget_kwargs):
        """Initialize form row.

        Args:
            parent: Parent widget
            label: Field label
            widget_type: Type of input widget ('entry', 'combo', 'spinbox', 'text')
            description: Optional field description
            **widget_kwargs: Arguments for the widget
        """
        super().__init__(parent)

        # Create grid layout
        self.columnconfigure(1, weight=1)

        # Label
        self.label = StyledLabel(self, label, style_type='label_bold')
        self.label.grid(row=0, column=0, sticky=tk.W, padx=(0, SPACING['lg']), pady=SPACING['sm'])

        # Input widget
        if widget_type == 'entry':
            self.widget = StyledEntry(self, **widget_kwargs)
        elif widget_type == 'combo':
            self.widget = ttk.Combobox(self, **widget_kwargs)
        elif widget_type == 'spinbox':
            self.widget = ttk.Spinbox(self, **widget_kwargs)
        elif widget_type == 'text':
            text_frame = ttk.Frame(self)
            text_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=SPACING['sm'])
            self.widget = tk.Text(text_frame, height=3, **widget_kwargs)
            self.widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = ttk.Scrollbar(text_frame, command=self.widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.widget.configure(yscrollcommand=scrollbar.set)
        else:
            self.widget = ttk.Entry(self, **widget_kwargs)

        if widget_type != 'text':
            self.widget.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=SPACING['sm'])

        # Description
        if description:
            self.desc = StyledLabel(self, description, style_type='small_italic')
            self.desc.configure(foreground='gray')
            self.desc.grid(row=1, column=1, sticky=tk.W, pady=(0, SPACING['sm']))


class ButtonGroup(ttk.Frame):
    """Group of buttons with consistent spacing."""

    def __init__(self, parent, buttons: list, align: str = 'right'):
        """Initialize button group.

        Args:
            parent: Parent widget
            buttons: List of (text, command, style_type) tuples
            align: Alignment ('left', 'right', 'center')
        """
        super().__init__(parent)

        # Determine packing side
        if align == 'right':
            side = tk.RIGHT
            padx = (SPACING['sm'], 0)
        elif align == 'left':
            side = tk.LEFT
            padx = (0, SPACING['sm'])
        else:
            side = tk.LEFT
            padx = SPACING['xs']

        # Create buttons
        self.buttons = []
        for i, (text, command, style_type) in enumerate(buttons):
            btn = StyledButton(self, text, command, style_type)
            btn.pack(side=side, padx=padx if i > 0 else 0)
            self.buttons.append(btn)


class Card(ttk.Frame):
    """Card-style container with shadow effect."""

    def __init__(self, parent, title: str = "", padding: str = 'section'):
        """Initialize card.

        Args:
            parent: Parent widget
            title: Optional card title
            padding: Padding size
        """
        super().__init__(parent, style='Card.TFrame', relief='flat', borderwidth=1)

        # Apply padding
        pad_value = PADDING.get(padding, PADDING['section'])

        # Title
        if title:
            title_frame = ttk.Frame(self, style='CardHeader.TFrame')
            title_frame.pack(fill=tk.X)

            title_label = StyledLabel(title_frame, title, style_type='h3')
            title_label.pack(side=tk.LEFT, padx=pad_value, pady=(pad_value, SPACING['sm']))

            ttk.Separator(self, orient='horizontal').pack(fill=tk.X)

        # Content area
        self.content = ttk.Frame(self)
        self.content.pack(fill=tk.BOTH, expand=True, padx=pad_value, pady=pad_value)


class StatusBar(ttk.Frame):
    """Professional status bar with sections."""

    def __init__(self, parent):
        """Initialize status bar."""
        super().__init__(parent, style='StatusBar.TFrame', relief='sunken', borderwidth=1)

        # Left section - main status
        self.status_label = StyledLabel(self, "Ready", style_type='status')
        self.status_label.pack(side=tk.LEFT, padx=SPACING['md'], pady=SPACING['xs'])

        # Right section - additional info
        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT)

        # Separator
        ttk.Separator(self.right_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=SPACING['sm'])

        # Additional status labels
        self.info_label = StyledLabel(self.right_frame, "", style_type='status')
        self.info_label.pack(side=tk.LEFT, padx=SPACING['md'])

    def set_status(self, text: str, info: str = ""):
        """Set status bar text."""
        self.status_label.configure(text=text)
        if info:
            self.info_label.configure(text=info)
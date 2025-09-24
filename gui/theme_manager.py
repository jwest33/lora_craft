"""Theme manager for GUI application."""

import tkinter as tk
from tkinter import ttk, font
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
try:
    from .style_constants import FONTS, SPACING, PADDING, COLORS, DIMENSIONS, get_style_config
except ImportError:
    # Fallback if style_constants not available
    FONTS = {'body': ('TkDefaultFont', 10, 'normal')}
    SPACING = {'md': 12, 'sm': 8, 'lg': 16}
    PADDING = {'frame': 20}
    def get_style_config(theme): return {'colors': {}}

logger = logging.getLogger(__name__)


class ThemeManager:
    """Manages application themes and styling."""

    # Built-in theme presets
    THEMES = {
        "light": {
            "name": "Light",
            "window": {
                "bg": "#f0f0f0"
            },
            "frame": {
                "bg": "#f0f0f0"
            },
            "labelframe": {
                "bg": "#f0f0f0",
                "fg": "#000000"
            },
            "label": {
                "bg": "#f0f0f0",
                "fg": "#000000"
            },
            "button": {
                "bg": "#e1e1e1",
                "fg": "#000000",
                "activebackground": "#d0d0d0",
                "activeforeground": "#000000",
                "highlightbackground": "#f0f0f0"
            },
            "entry": {
                "bg": "#ffffff",
                "fg": "#000000",
                "insertbackground": "#000000",
                "selectbackground": "#0078d7",
                "selectforeground": "#ffffff"
            },
            "text": {
                "bg": "#ffffff",
                "fg": "#000000",
                "insertbackground": "#000000",
                "selectbackground": "#0078d7",
                "selectforeground": "#ffffff"
            },
            "combobox": {
                "fieldbackground": "#ffffff",
                "background": "#ffffff",
                "foreground": "#000000",
                "selectbackground": "#0078d7",
                "selectforeground": "#ffffff",
                "arrowcolor": "#000000"
            },
            "spinbox": {
                "fieldbackground": "#ffffff",
                "background": "#ffffff",
                "foreground": "#000000",
                "buttonbackground": "#e1e1e1",
                "arrowcolor": "#000000"
            },
            "checkbutton": {
                "bg": "#f0f0f0",
                "fg": "#000000",
                "selectcolor": "#ffffff",
                "activebackground": "#e0e0e0",
                "activeforeground": "#000000",
                "indicatorcolor": "#000000"
            },
            "menu": {
                "bg": "#f0f0f0",
                "fg": "#000000",
                "activebackground": "#0078d7",
                "activeforeground": "#ffffff",
                "selectcolor": "#000000"
            },
            "treeview": {
                "fieldbackground": "#ffffff",
                "background": "#ffffff",
                "foreground": "#000000",
                "selectbackground": "#0078d7",
                "selectforeground": "#ffffff"
            },
            "scrollbar": {
                "background": "#e1e1e1",
                "troughcolor": "#f0f0f0",
                "activebackground": "#c0c0c0",
                "arrowcolor": "#606060"
            },
            "notebook": {
                "background": "#f0f0f0",
                "tabbackground": "#d8d8d8",  # Lighter inactive tab
                "tabforeground": "#666666",  # Grayed out text for inactive
                "selectedtabbackground": "#f0f0f0",  # Same as background
                "selectedtabforeground": "#000000"  # Bold black for active
            },
            "progressbar": {
                "background": "#e1e1e1",
                "troughcolor": "#f0f0f0",
                "barcolor": "#0078d7"
            },
            "canvas": {
                "bg": "#f0f0f0",
                "highlightbackground": "#f0f0f0"
            }
        },
        "dark": {
            "name": "Dark",
            "window": {
                "bg": "#1e1e1e"
            },
            "frame": {
                "bg": "#1e1e1e"
            },
            "labelframe": {
                "bg": "#1e1e1e",
                "fg": "#cccccc"
            },
            "label": {
                "bg": "#1e1e1e",
                "fg": "#cccccc"
            },
            "button": {
                "bg": "#2d2d30",
                "fg": "#cccccc",
                "activebackground": "#3e3e42",
                "activeforeground": "#ffffff",
                "highlightbackground": "#1e1e1e"
            },
            "entry": {
                "bg": "#2d2d30",
                "fg": "#cccccc",
                "insertbackground": "#cccccc",
                "selectbackground": "#007acc",
                "selectforeground": "#ffffff"
            },
            "text": {
                "bg": "#2d2d30",
                "fg": "#cccccc",
                "insertbackground": "#cccccc",
                "selectbackground": "#007acc",
                "selectforeground": "#ffffff"
            },
            "combobox": {
                "fieldbackground": "#2d2d30",
                "background": "#2d2d30",
                "foreground": "#cccccc",
                "selectbackground": "#007acc",
                "selectforeground": "#ffffff",
                "arrowcolor": "#cccccc"
            },
            "spinbox": {
                "fieldbackground": "#2d2d30",
                "background": "#2d2d30",
                "foreground": "#cccccc",
                "buttonbackground": "#3e3e42",
                "arrowcolor": "#cccccc"
            },
            "checkbutton": {
                "bg": "#1e1e1e",
                "fg": "#cccccc",
                "selectcolor": "#2d2d30",
                "activebackground": "#3e3e42",
                "activeforeground": "#ffffff",
                "indicatorcolor": "#cccccc"
            },
            "menu": {
                "bg": "#2d2d30",
                "fg": "#cccccc",
                "activebackground": "#007acc",
                "activeforeground": "#ffffff",
                "selectcolor": "#cccccc"
            },
            "treeview": {
                "fieldbackground": "#2d2d30",
                "background": "#2d2d30",
                "foreground": "#cccccc",
                "selectbackground": "#007acc",
                "selectforeground": "#ffffff"
            },
            "scrollbar": {
                "background": "#3e3e42",
                "troughcolor": "#2d2d30",
                "activebackground": "#007acc",
                "arrowcolor": "#cccccc"
            },
            "notebook": {
                "background": "#1e1e1e",
                "tabbackground": "#252526",  # Darker inactive tab
                "tabforeground": "#808080",  # Grayed out text for inactive
                "selectedtabbackground": "#1e1e1e",  # Same as background
                "selectedtabforeground": "#ffffff"  # Bright white for active
            },
            "progressbar": {
                "background": "#3e3e42",
                "troughcolor": "#2d2d30",
                "barcolor": "#007acc"
            },
            "canvas": {
                "bg": "#1e1e1e",
                "highlightbackground": "#1e1e1e"
            }
        },
        "synthwave": {
            "name": "Synthwave",
            "window": {
                "bg": "#241b2f"
            },
            "frame": {
                "bg": "#241b2f"
            },
            "labelframe": {
                "bg": "#241b2f",
                "fg": "#f92aad"
            },
            "label": {
                "bg": "#241b2f",
                "fg": "#f92aad"
            },
            "button": {
                "bg": "#262335",
                "fg": "#00ffff",
                "activebackground": "#ff00ff",
                "activeforeground": "#ffffff",
                "highlightbackground": "#241b2f",
                "highlightcolor": "#ff00ff",
                "highlightthickness": 0
            },
            "entry": {
                "bg": "#262335",
                "fg": "#00ffff",
                "insertbackground": "#00ffff",
                "selectbackground": "#ff00ff",
                "selectforeground": "#ffffff",
                "highlightcolor": "#ff00ff",
                "highlightbackground": "#241b2f",
                "highlightthickness": 1
            },
            "text": {
                "bg": "#262335",
                "fg": "#00ffff",
                "insertbackground": "#00ffff",
                "selectbackground": "#ff00ff",
                "selectforeground": "#ffffff",
                "highlightcolor": "#ff00ff",
                "highlightbackground": "#241b2f",
                "highlightthickness": 1
            },
            "combobox": {
                "fieldbackground": "#262335",
                "background": "#262335",
                "foreground": "#00ffff",
                "selectbackground": "#ff00ff",
                "selectforeground": "#ffffff",
                "arrowcolor": "#ff00ff"
            },
            "spinbox": {
                "fieldbackground": "#262335",
                "background": "#262335",
                "foreground": "#00ffff",
                "buttonbackground": "#372844",
                "arrowcolor": "#00ffff"
            },
            "checkbutton": {
                "bg": "#241b2f",
                "fg": "#00ffff",
                "selectcolor": "#372844",
                "activebackground": "#ff00ff",
                "activeforeground": "#ffffff",
                "indicatorcolor": "#00ffff"
            },
            "menu": {
                "bg": "#262335",
                "fg": "#00ffff",
                "activebackground": "#ff00ff",
                "activeforeground": "#ffffff",
                "selectcolor": "#00ffff"
            },
            "treeview": {
                "fieldbackground": "#262335",
                "background": "#262335",
                "foreground": "#00ffff",
                "selectbackground": "#ff00ff",
                "selectforeground": "#ffffff"
            },
            "scrollbar": {
                "background": "#ff00ff",
                "troughcolor": "#262335",
                "activebackground": "#00ffff",
                "arrowcolor": "#262335"
            },
            "notebook": {
                "background": "#241b2f",
                "tabbackground": "#1a1420",  # Darker inactive tab
                "tabforeground": "#72718f",  # Muted color for inactive
                "selectedtabbackground": "#241b2f",  # Same as background
                "selectedtabforeground": "#00ffff"  # Bright cyan for active
            },
            "progressbar": {
                "background": "#262335",
                "troughcolor": "#241b2f",
                "barcolor": "#ff00ff"
            },
            "canvas": {
                "bg": "#241b2f",
                "highlightbackground": "#241b2f",
                "highlightthickness": 0
            }
        }
    }

    def __init__(self, root: tk.Tk):
        """Initialize theme manager.

        Args:
            root: Root tkinter window
        """
        self.root = root
        self.current_theme = "light"
        self.custom_font = None
        self.style = ttk.Style()

        # Load custom font
        self._load_custom_font()

        # Load user preferences
        self._load_preferences()

    def _load_custom_font(self):
        """Load custom font from assets."""
        try:
            font_path = Path(__file__).parent.parent / "assets" / "Rationale-Regular.ttf"
            if font_path.exists():
                # Register the font with tkinter
                self.custom_font = font.Font(family="Rationale", size=10)
                logger.info(f"Loaded custom font from {font_path}")
            else:
                logger.warning(f"Custom font not found at {font_path}")
                # Use fallback font
                self.custom_font = font.Font(family="Segoe UI", size=10)
        except Exception as e:
            logger.error(f"Failed to load custom font: {e}")
            self.custom_font = font.Font(family="Segoe UI", size=10)

    def _load_preferences(self):
        """Load user preferences including last selected theme."""
        prefs_path = Path("configs/user_preferences.json")
        if prefs_path.exists():
            try:
                with open(prefs_path, 'r') as f:
                    prefs = json.load(f)
                    self.current_theme = prefs.get("theme", "light")
                    logger.info(f"Loaded theme preference: {self.current_theme}")
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")

    def _save_preferences(self):
        """Save user preferences."""
        prefs_path = Path("configs/user_preferences.json")
        prefs_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing preferences or create new
        prefs = {}
        if prefs_path.exists():
            try:
                with open(prefs_path, 'r') as f:
                    prefs = json.load(f)
            except:
                pass

        # Update theme preference
        prefs["theme"] = self.current_theme

        # Save preferences
        try:
            with open(prefs_path, 'w') as f:
                json.dump(prefs, f, indent=2)
            logger.info(f"Saved theme preference: {self.current_theme}")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

    def apply_theme(self, theme_name: str):
        """Apply a theme to the application.

        Args:
            theme_name: Name of theme to apply (light, dark, synthwave)
        """
        if theme_name not in self.THEMES:
            logger.error(f"Unknown theme: {theme_name}")
            return

        self.current_theme = theme_name
        theme = self.THEMES[theme_name]

        # Configure root window
        self.root.configure(bg=theme["window"]["bg"])

        # Configure ttk styles
        self._configure_ttk_styles(theme, theme_name)

        # Apply theme to all widgets
        self._apply_theme_to_widgets(self.root, theme)

        # Force update of all widgets
        self.root.update_idletasks()

        # Save preference
        self._save_preferences()

        logger.info(f"Applied theme: {theme_name}")

    def _configure_ttk_styles(self, theme: Dict[str, Any], theme_name: str = None):
        """Configure ttk widget styles.

        Args:
            theme: Theme configuration dictionary
            theme_name: Name of the theme being applied
        """
        # Set the theme base
        self.style.theme_use('clam')  # Use clam theme as base for better customization

        # Get professional style configuration
        style_config = get_style_config(theme_name or self.current_theme)
        colors = style_config['colors']

        # Configure notebook (tabs) with enhanced styling
        self.style.configure("TNotebook",
                           background=theme["notebook"]["background"],
                           borderwidth=0,
                           tabposition='n',
                           tabmargins=[2, 5, 2, 0])  # Margins around tabs

        # Configure inactive tabs (smaller)
        self.style.configure("TNotebook.Tab",
                           background=theme["notebook"]["tabbackground"],
                           foreground=theme["notebook"]["tabforeground"],
                           padding=[12, 6],  # Smaller padding for inactive tabs
                           borderwidth=0,
                           focuscolor='none')

        # Configure active/selected tabs (larger)
        self.style.map("TNotebook.Tab",
                      padding=[("selected", [20, 10])],  # Larger padding when selected
                      background=[("selected", theme["notebook"]["background"])],  # Same as notebook background
                      foreground=[("selected", theme["notebook"]["selectedtabforeground"])],
                      borderwidth=[("selected", 0)],
                      relief=[("selected", "flat")])

        # Apply professional styles
        self._configure_button_styles(theme, colors)
        self._configure_entry_styles(theme, colors)
        self._configure_frame_styles(theme, colors)
        self._configure_label_styles(theme, colors)

        # Configure frames
        self.style.configure("TFrame", background=theme["frame"]["bg"])

        # Configure label frames
        if theme_name == "synthwave":
            # Special styling for synthwave to remove white borders
            self.style.configure("TLabelframe",
                               background=theme["labelframe"]["bg"],
                               foreground=theme["labelframe"]["fg"],
                               bordercolor="#372844",
                               lightcolor="#372844",
                               darkcolor="#372844",
                               borderwidth=1,
                               relief="solid")
        else:
            self.style.configure("TLabelframe",
                               background=theme["labelframe"]["bg"],
                               foreground=theme["labelframe"]["fg"],
                               bordercolor=theme["labelframe"]["fg"],
                               lightcolor=theme["labelframe"]["bg"],
                               darkcolor=theme["labelframe"]["fg"],
                               borderwidth=1,
                               relief="solid")
        self.style.configure("TLabelframe.Label",
                           background=theme["labelframe"]["bg"],
                           foreground=theme["labelframe"]["fg"],
                           font=self.custom_font)

        # Configure labels
        self.style.configure("TLabel",
                           background=theme["label"]["bg"],
                           foreground=theme["label"]["fg"],
                           font=self.custom_font)

        # Configure buttons
        self.style.configure("TButton",
                           background=theme["button"]["bg"],
                           foreground=theme["button"]["fg"],
                           borderwidth=1,
                           focuscolor='none',
                           font=self.custom_font)
        self.style.map("TButton",
                      background=[("active", theme["button"]["activebackground"])],
                      foreground=[("active", theme["button"]["activeforeground"])])

        # Configure entry widgets
        self.style.configure("TEntry",
                           fieldbackground=theme["entry"]["bg"],
                           background=theme["entry"]["bg"],
                           foreground=theme["entry"]["fg"],
                           insertcolor=theme["entry"]["insertbackground"],
                           bordercolor=theme["entry"]["fg"],
                           lightcolor=theme["entry"]["bg"],
                           darkcolor=theme["entry"]["bg"],
                           font=self.custom_font)
        self.style.map("TEntry",
                      fieldbackground=[("focus", theme["entry"]["bg"])],
                      bordercolor=[("focus", theme["entry"]["selectbackground"])])

        # Configure combobox
        self.style.configure("TCombobox",
                           fieldbackground=theme["combobox"]["fieldbackground"],
                           background=theme["combobox"]["background"],
                           foreground=theme["combobox"]["foreground"],
                           selectbackground=theme["combobox"]["selectbackground"],
                           selectforeground=theme["combobox"]["selectforeground"],
                           arrowcolor=theme["combobox"]["arrowcolor"],
                           bordercolor=theme["combobox"]["foreground"],
                           lightcolor=theme["combobox"]["fieldbackground"],
                           darkcolor=theme["combobox"]["fieldbackground"],
                           font=self.custom_font)
        self.style.map("TCombobox",
                      fieldbackground=[("readonly", theme["combobox"]["fieldbackground"])],
                      selectbackground=[("readonly", theme["combobox"]["selectbackground"])],
                      selectforeground=[("readonly", theme["combobox"]["selectforeground"])])

        # Configure the dropdown listbox (popdown)
        self.root.option_add('*TCombobox*Listbox.background', theme["combobox"]["fieldbackground"])
        self.root.option_add('*TCombobox*Listbox.foreground', theme["combobox"]["foreground"])
        self.root.option_add('*TCombobox*Listbox.selectBackground', theme["combobox"]["selectbackground"])
        self.root.option_add('*TCombobox*Listbox.selectForeground', theme["combobox"]["selectforeground"])

        # Configure checkbuttons
        self.style.configure("TCheckbutton",
                           background=theme["checkbutton"]["bg"],
                           foreground=theme["checkbutton"]["fg"],
                           indicatorcolor=theme["checkbutton"]["indicatorcolor"],
                           focuscolor='none',
                           borderwidth=0,
                           font=self.custom_font)
        self.style.map("TCheckbutton",
                      background=[('active', theme["checkbutton"]["activebackground"])],
                      foreground=[('active', theme["checkbutton"]["activeforeground"])],
                      indicatorcolor=[('selected', theme["checkbutton"]["activebackground"]),
                                    ('pressed', theme["checkbutton"]["activebackground"])])

        # Configure spinbox
        self.style.configure("TSpinbox",
                           fieldbackground=theme["spinbox"]["fieldbackground"],
                           background=theme["spinbox"]["background"],
                           foreground=theme["spinbox"]["foreground"],
                           buttonbackground=theme["spinbox"]["buttonbackground"],
                           arrowcolor=theme["spinbox"]["arrowcolor"],
                           bordercolor=theme["spinbox"]["foreground"],
                           lightcolor=theme["spinbox"]["fieldbackground"],
                           darkcolor=theme["spinbox"]["fieldbackground"],
                           font=self.custom_font)

        # Configure treeview
        self.style.configure("Treeview",
                           background=theme["treeview"]["background"],
                           fieldbackground=theme["treeview"]["fieldbackground"],
                           foreground=theme["treeview"]["foreground"],
                           bordercolor=theme["treeview"]["foreground"],
                           lightcolor=theme["treeview"]["background"],
                           darkcolor=theme["treeview"]["foreground"],
                           font=self.custom_font)
        self.style.configure("Treeview.Heading",
                           background=theme["button"]["bg"],
                           foreground=theme["button"]["fg"],
                           bordercolor=theme["button"]["fg"],
                           lightcolor=theme["button"]["bg"],
                           darkcolor=theme["button"]["fg"],
                           font=self.custom_font)
        self.style.map("Treeview",
                      background=[("selected", theme["treeview"]["selectbackground"])],
                      foreground=[("selected", theme["treeview"]["selectforeground"])],
                      fieldbackground=[("!disabled", theme["treeview"]["fieldbackground"])])
        self.style.map("Treeview.Heading",
                      background=[("active", theme["button"]["activebackground"])],
                      foreground=[("active", theme["button"]["activeforeground"])])

        # Configure scrollbar - special handling for synthwave
        if theme_name == "synthwave":
            self.style.configure("TScrollbar",
                               background=theme["scrollbar"]["background"],
                               troughcolor=theme["scrollbar"]["troughcolor"],
                               arrowcolor=theme["scrollbar"]["arrowcolor"],
                               bordercolor="#372844",
                               lightcolor="#372844",
                               darkcolor="#372844",
                               borderwidth=0,
                               gripcount=0)
        else:
            self.style.configure("TScrollbar",
                               background=theme["scrollbar"]["background"],
                               troughcolor=theme["scrollbar"]["troughcolor"],
                               arrowcolor=theme["scrollbar"]["arrowcolor"],
                               borderwidth=0)
        self.style.map("TScrollbar",
                      background=[("active", theme["scrollbar"]["activebackground"])])

        # Configure progressbar
        self.style.configure("TProgressbar",
                           background=theme["progressbar"]["barcolor"],
                           troughcolor=theme["progressbar"]["troughcolor"],
                           borderwidth=0)

    def _apply_theme_to_widgets(self, widget, theme: Dict[str, Any], visited=None):
        """Recursively apply theme to all widgets.

        Args:
            widget: Widget to apply theme to
            theme: Theme configuration dictionary
            visited: Set of visited widgets to avoid loops
        """
        if visited is None:
            visited = set()

        # Avoid processing the same widget twice
        if widget in visited:
            return
        visited.add(widget)

        # Apply theme based on widget type
        widget_class = widget.winfo_class()

        if widget_class == "Tk" or widget_class == "Toplevel":
            widget.configure(bg=theme["window"]["bg"])
        elif widget_class == "Frame":
            if not isinstance(widget, ttk.Frame):
                widget.configure(bg=theme["frame"]["bg"])
        elif widget_class == "Label":
            if not isinstance(widget, ttk.Label):
                widget.configure(
                    bg=theme["label"]["bg"],
                    fg=theme["label"]["fg"],
                    font=self.custom_font
                )
        elif widget_class == "Button":
            if not isinstance(widget, ttk.Button):
                widget.configure(
                    bg=theme["button"]["bg"],
                    fg=theme["button"]["fg"],
                    activebackground=theme["button"]["activebackground"],
                    activeforeground=theme["button"]["activeforeground"],
                    font=self.custom_font
                )
        elif widget_class == "Entry":
            if not isinstance(widget, ttk.Entry):
                widget.configure(
                    bg=theme["entry"]["bg"],
                    fg=theme["entry"]["fg"],
                    insertbackground=theme["entry"]["insertbackground"],
                    selectbackground=theme["entry"]["selectbackground"],
                    selectforeground=theme["entry"]["selectforeground"],
                    font=self.custom_font
                )
        elif widget_class == "Text":
            widget.configure(
                bg=theme["text"]["bg"],
                fg=theme["text"]["fg"],
                insertbackground=theme["text"]["insertbackground"],
                selectbackground=theme["text"]["selectbackground"],
                selectforeground=theme["text"]["selectforeground"],
                font=self.custom_font
            )
        elif widget_class == "Canvas":
            widget.configure(
                bg=theme["canvas"]["bg"],
                highlightbackground=theme["canvas"]["highlightbackground"]
            )
        elif widget_class == "Menu":
            widget.configure(
                bg=theme["menu"]["bg"],
                fg=theme["menu"]["fg"],
                activebackground=theme["menu"]["activebackground"],
                activeforeground=theme["menu"]["activeforeground"],
                font=self.custom_font
            )
            # Apply to menu items
            for i in range(widget.index("end") + 1):
                try:
                    if widget.type(i) == "cascade":
                        submenu = widget.nametowidget(widget.entryconfig(i, "menu")[4])
                        self._apply_theme_to_widgets(submenu, theme, visited)
                except:
                    pass

        # Recursively apply to children
        for child in widget.winfo_children():
            self._apply_theme_to_widgets(child, theme, visited)

    def get_current_theme(self) -> str:
        """Get the current theme name.

        Returns:
            Current theme name
        """
        return self.current_theme

    def get_theme_names(self) -> list:
        """Get list of available theme names.

        Returns:
            List of theme names
        """
        return list(self.THEMES.keys())

    def get_theme_display_names(self) -> Dict[str, str]:
        """Get mapping of theme keys to display names.

        Returns:
            Dictionary mapping theme keys to display names
        """
        return {key: theme["name"] for key, theme in self.THEMES.items()}

    def _configure_button_styles(self, theme: Dict[str, Any], colors: Dict[str, Any]):
        """Configure professional button styles."""
        # Default button style
        self.style.configure("Styled.TButton",
                           background=colors.get('bg_secondary', theme["button"]["bg"]),
                           foreground=colors.get('text', theme["button"]["fg"]),
                           borderwidth=1,
                           relief='flat',
                           padding=(SPACING['md'], SPACING['sm']),
                           font=FONTS.get('button', self.custom_font))
        self.style.map("Styled.TButton",
                      background=[('active', colors.get('bg_tertiary', theme["button"]["activebackground"]))],
                      foreground=[('active', colors.get('text', theme["button"]["activeforeground"]))])

        # Primary button style
        self.style.configure("Primary.TButton",
                           background=colors.get('primary', '#0066cc'),
                           foreground=colors.get('primary_text', '#ffffff'),
                           borderwidth=0,
                           relief='flat',
                           padding=(SPACING['lg'], SPACING['sm']),
                           font=FONTS.get('button', self.custom_font))
        self.style.map("Primary.TButton",
                      background=[('active', colors.get('primary_hover', '#0052a3'))])

        # Accent button style
        self.style.configure("Accent.TButton",
                           background=colors.get('accent', '#00a86b'),
                           foreground=colors.get('accent_text', '#ffffff'),
                           borderwidth=0,
                           relief='flat',
                           padding=(SPACING['lg'], SPACING['sm']),
                           font=FONTS.get('button', self.custom_font))
        self.style.map("Accent.TButton",
                      background=[('active', colors.get('accent_hover', '#008755'))])

    def _configure_entry_styles(self, theme: Dict[str, Any], colors: Dict[str, Any]):
        """Configure professional entry styles."""
        self.style.configure("Styled.TEntry",
                           fieldbackground=colors.get('bg', theme["entry"]["bg"]),
                           background=colors.get('bg', theme["entry"]["bg"]),
                           foreground=colors.get('text', theme["entry"]["fg"]),
                           bordercolor=colors.get('border', theme["entry"]["fg"]),
                           insertcolor=colors.get('text', theme["entry"]["insertbackground"]),
                           padding=SPACING['sm'],
                           font=FONTS.get('input', self.custom_font))
        self.style.map("Styled.TEntry",
                      bordercolor=[('focus', colors.get('primary', theme["entry"]["selectbackground"]))])

    def _configure_frame_styles(self, theme: Dict[str, Any], colors: Dict[str, Any]):
        """Configure professional frame styles."""
        # Card style frame
        self.style.configure("Card.TFrame",
                           background=colors.get('bg', theme["frame"]["bg"]),
                           bordercolor=colors.get('border', '#dee2e6'),
                           relief='flat',
                           borderwidth=1)

        # Section header frame
        self.style.configure("SectionHeader.TFrame",
                           background=colors.get('section_bg', theme["frame"]["bg"]),
                           relief='flat',
                           borderwidth=0)

        # Status bar frame
        self.style.configure("StatusBar.TFrame",
                           background=colors.get('bg_secondary', theme["frame"]["bg"]),
                           relief='sunken',
                           borderwidth=1)

    def _configure_label_styles(self, theme: Dict[str, Any], colors: Dict[str, Any]):
        """Configure professional label styles."""
        # Section header label
        self.style.configure("SectionHeader.TLabel",
                           background=colors.get('section_bg', theme["label"]["bg"]),
                           foreground=colors.get('section_text', theme["label"]["fg"]),
                           font=FONTS.get('h2', self.custom_font))

        # Different text styles
        for style_name, font_key in [
            ('H1', 'h1'), ('H2', 'h2'), ('H3', 'h3'),
            ('Body', 'body'), ('BodyBold', 'body_bold'),
            ('Small', 'small'), ('SmallItalic', 'small_italic')
        ]:
            self.style.configure(f"{style_name}.TLabel",
                               background=theme["label"]["bg"],
                               foreground=colors.get('text', theme["label"]["fg"]),
                               font=FONTS.get(font_key, self.custom_font))
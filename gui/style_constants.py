"""Style constants and typography system for professional UI."""

# Typography System
FONTS = {
    # Headers
    'h1': ('Segoe UI', 18, 'bold'),
    'h2': ('Segoe UI', 14, 'bold'),
    'h3': ('Segoe UI', 12, 'bold'),

    # Body text
    'body': ('Segoe UI', 10, 'normal'),
    'body_bold': ('Segoe UI', 10, 'bold'),
    'small': ('Segoe UI', 9, 'normal'),
    'small_italic': ('Segoe UI', 9, 'italic'),

    # UI elements
    'button': ('Segoe UI', 10, 'normal'),
    'input': ('Segoe UI', 10, 'normal'),
    'label': ('Segoe UI', 10, 'normal'),
    'label_bold': ('Segoe UI', 10, 'bold'),

    # Special
    'mono': ('Consolas', 10, 'normal'),
    'status': ('Segoe UI', 9, 'normal'),
}

# Spacing System (pixels)
SPACING = {
    'none': 0,
    'xs': 4,
    'sm': 8,
    'md': 12,
    'lg': 16,
    'xl': 24,
    'xxl': 32,
    '3xl': 48,
}

# Padding for different elements
PADDING = {
    'frame': 20,
    'frame_compact': 12,
    'section': 16,
    'group': 12,
    'input': 8,
    'button': (10, 6),  # (x, y)
}

# Colors for professional look
COLORS = {
    'light': {
        # Primary colors
        'primary': '#0066cc',
        'primary_hover': '#0052a3',
        'primary_text': '#ffffff',

        # Accent colors
        'accent': '#00a86b',
        'accent_hover': '#008755',
        'accent_text': '#ffffff',

        # Neutral colors
        'bg': '#ffffff',
        'bg_secondary': '#f8f9fa',
        'bg_tertiary': '#e9ecef',
        'border': '#dee2e6',
        'border_strong': '#adb5bd',

        # Text colors
        'text': '#212529',
        'text_secondary': '#6c757d',
        'text_tertiary': '#adb5bd',
        'text_link': '#0066cc',

        # Status colors
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'info': '#17a2b8',

        # Section headers
        'section_bg': '#f0f4f8',
        'section_border': '#d0dae4',
        'section_text': '#2c3e50',
    },
    'dark': {
        # Primary colors
        'primary': '#4da3ff',
        'primary_hover': '#66b0ff',
        'primary_text': '#000000',

        # Accent colors
        'accent': '#00d084',
        'accent_hover': '#00b06f',
        'accent_text': '#000000',

        # Neutral colors
        'bg': '#1e1e1e',
        'bg_secondary': '#252526',
        'bg_tertiary': '#2d2d30',
        'border': '#3e3e42',
        'border_strong': '#555559',

        # Text colors
        'text': '#cccccc',
        'text_secondary': '#999999',
        'text_tertiary': '#666666',
        'text_link': '#4da3ff',

        # Status colors
        'success': '#4ec76f',
        'warning': '#ffcc00',
        'error': '#f14c4c',
        'info': '#3ba7cc',

        # Section headers
        'section_bg': '#1e1e1e',
        'section_border': '#3e3e42',
        'section_text': '#e0e0e0',
    },
    'synthwave': {
        # Primary colors
        'primary': '#ff00ff',
        'primary_hover': '#ff33ff',
        'primary_text': '#000000',

        # Accent colors
        'accent': '#00ffff',
        'accent_hover': '#33ffff',
        'accent_text': '#000000',

        # Neutral colors
        'bg': '#241b2f',
        'bg_secondary': '#262335',
        'bg_tertiary': '#2a2041',
        'border': '#372844',
        'border_strong': '#483a58',

        # Text colors
        'text': '#f92aad',
        'text_secondary': '#00ffff',
        'text_tertiary': '#72f1b8',
        'text_link': '#ff00ff',

        # Status colors
        'success': '#72f1b8',
        'warning': '#fdee00',
        'error': '#ff2a6d',
        'info': '#00ffff',

        # Section headers
        'section_bg': '#241b2f',
        'section_border': '#ff00ff',
        'section_text': '#00ffff',
    }
}

# Widget dimensions
DIMENSIONS = {
    'button_height': 32,
    'input_height': 28,
    'combo_width': 200,
    'entry_width': 200,
    'label_width': 120,
    'section_radius': 6,
}

# Professional styling classes
STYLES = {
    'frame': {
        'relief': 'flat',
        'borderwidth': 0,
    },
    'labelframe': {
        'relief': 'flat',
        'borderwidth': 1,
        'labelanchor': 'nw',
    },
    'button': {
        'relief': 'flat',
        'borderwidth': 1,
        'cursor': 'hand2',
    },
    'entry': {
        'relief': 'flat',
        'borderwidth': 1,
    },
}

def get_style_config(theme='light'):
    """Get complete style configuration for a theme.

    Args:
        theme: Theme name ('light', 'dark', 'synthwave')

    Returns:
        Dictionary with all style settings
    """
    return {
        'fonts': FONTS,
        'spacing': SPACING,
        'padding': PADDING,
        'colors': COLORS.get(theme, COLORS['light']),
        'dimensions': DIMENSIONS,
        'styles': STYLES,
    }
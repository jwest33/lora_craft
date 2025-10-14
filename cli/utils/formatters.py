"""Data formatting utilities."""

from typing import Any, Dict, List
from datetime import datetime


def format_timestamp(ts: Any) -> str:
    """Format timestamp to readable string."""
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return ts
    elif isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    return str(ts)


def format_status(status: str) -> str:
    """Format status with color codes."""
    status_colors = {
        'running': 'green',
        'completed': 'blue',
        'error': 'red',
        'stopped': 'yellow',
        'pending': 'cyan'
    }
    color = status_colors.get(status.lower(), 'white')
    return f"[{color}]{status}[/{color}]"


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + '...'


def format_model_name(name: str) -> str:
    """Format model name for display."""
    # Remove common prefixes
    prefixes = ['unsloth/', 'meta-llama/', 'mistralai/', 'microsoft/']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_list_for_display(items: List[Dict[str, Any]],
                            fields: List[str],
                            formatters: Dict[str, callable] = None) -> List[Dict[str, str]]:
    """
    Format list of items for table display.

    Args:
        items: List of items to format
        fields: Fields to include in output
        formatters: Optional dict of field -> formatter function

    Returns:
        Formatted list ready for display
    """
    formatters = formatters or {}
    result = []

    for item in items:
        formatted_item = {}
        for field in fields:
            value = item.get(field, 'N/A')

            # Apply custom formatter if available
            if field in formatters:
                value = formatters[field](value)

            formatted_item[field] = value

        result.append(formatted_item)

    return result

"""
Display utilities for terminal output.

Handles formatting, colors, tables, and progress display.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box
from typing import List, Dict, Any, Optional
import json
import yaml


console = Console()


def print_success(message: str):
    """Print success message in green."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """Print error message in red."""
    console.print(f"[red]✗[/red] {message}", style="red")


def print_warning(message: str):
    """Print warning message in yellow."""
    console.print(f"[yellow]⚠[/yellow] {message}", style="yellow")


def print_info(message: str):
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_table(data: List[Dict[str, Any]], title: Optional[str] = None):
    """
    Print data as a formatted table.

    Args:
        data: List of dictionaries to display
        title: Optional table title
    """
    if not data:
        print_info("No data to display")
        return

    # Create table
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")

    # Add columns from first row
    first_row = data[0]
    for key in first_row.keys():
        table.add_column(key.replace('_', ' ').title(), style="white")

    # Add rows
    for row in data:
        table.add_row(*[str(v) for v in row.values()])

    console.print(table)


def print_panel(content: str, title: Optional[str] = None, style: str = "cyan"):
    """
    Print content in a panel.

    Args:
        content: Content to display
        title: Optional panel title
        style: Panel border style
    """
    console.print(Panel(content, title=title, border_style=style))


def print_json(data: Any, pretty: bool = True):
    """Print JSON data."""
    if pretty:
        console.print_json(json.dumps(data, indent=2))
    else:
        print(json.dumps(data))


def print_yaml(data: Any):
    """Print YAML data."""
    yaml_str = yaml.dump(data, default_flow_style=False)
    console.print(yaml_str)


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


def create_progress() -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )


def confirm(message: str, default: bool = False) -> bool:
    """
    Ask for user confirmation.

    Args:
        message: Confirmation message
        default: Default value if user presses enter

    Returns:
        True if confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    response = console.input(f"{message} [{default_str}]: ").strip().lower()

    if not response:
        return default

    return response in ['y', 'yes']


def select_option(prompt: str, options: List[str]) -> Optional[str]:
    """
    Let user select from a list of options.

    Args:
        prompt: Prompt message
        options: List of options

    Returns:
        Selected option or None if cancelled
    """
    console.print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        console.print(f"  {i}. {option}")

    while True:
        try:
            choice = console.input("\nEnter choice (number or name): ").strip()

            # Try as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]

            # Try as name
            for option in options:
                if choice.lower() == option.lower():
                    return option

            print_error("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print_info("\nCancelled")
            return None


def print_metrics(metrics: Dict[str, Any], title: str = "Metrics"):
    """Print metrics in a formatted way."""
    table = Table(title=title, box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in metrics.items():
        formatted_key = key.replace('_', ' ').title()
        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        table.add_row(formatted_key, formatted_value)

    console.print(table)

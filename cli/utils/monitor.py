"""
Real-time training monitor with live dashboard.

Provides a terminal UI for monitoring training progress with metrics,
logs, and system resources.
"""

import time
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console
from cli.utils.formatters import format_status


class TrainingMonitor:
    """Real-time training monitor with live updates."""

    def __init__(self, client, session_id, update_interval=2):
        """
        Initialize training monitor.

        Args:
            client: APIClient instance
            session_id: Training session ID to monitor
            update_interval: Update interval in seconds
        """
        self.client = client
        self.session_id = session_id
        self.update_interval = update_interval
        self.console = Console()

        # State
        self.status = "unknown"
        self.metrics = {}
        self.logs = []
        self.last_log_count = 0

    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )

        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )

        return layout

    def generate_header(self) -> Panel:
        """Generate header panel."""
        header_text = Text()
        header_text.append(f"Training Monitor - Session: ", style="bold")
        header_text.append(self.session_id[:16] + "...", style="cyan")
        header_text.append(f" | Status: ", style="bold")
        header_text.append(self.status, style=self._get_status_style())

        return Panel(header_text, style="blue")

    def generate_metrics_panel(self) -> Panel:
        """Generate metrics panel."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green", width=20)

        if self.metrics:
            for key, value in self.metrics.items():
                formatted_key = key.replace('_', ' ').title()

                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)

                table.add_row(formatted_key, formatted_value)
        else:
            table.add_row("No metrics available", "")

        return Panel(table, title="[b]Metrics[/b]", border_style="cyan")

    def generate_logs_panel(self) -> Panel:
        """Generate logs panel."""
        log_text = Text()

        if self.logs:
            # Show last 15 logs
            recent_logs = self.logs[-15:]
            for log in recent_logs:
                log_text.append(log + "\n")
        else:
            log_text.append("No logs available", style="dim")

        return Panel(log_text, title="[b]Recent Logs[/b]", border_style="yellow")

    def generate_progress_panel(self) -> Panel:
        """Generate progress panel."""
        progress_text = Text()

        # Extract progress if available
        progress_pct = 0
        if isinstance(self.metrics, dict):
            progress_pct = self.metrics.get('progress', 0)

        progress_text.append(f"Progress: {progress_pct:.1f}%\n", style="bold")

        # Add a simple text-based progress bar
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        progress_text.append(f"[{bar}]", style="cyan")

        return Panel(progress_text, title="[b]Progress[/b]", border_style="green")

    def _get_status_style(self) -> str:
        """Get style for status display."""
        status_styles = {
            'running': 'green',
            'completed': 'blue',
            'error': 'red',
            'stopped': 'yellow',
            'pending': 'cyan'
        }
        return status_styles.get(self.status.lower(), 'white')

    def update_data(self):
        """Fetch latest data from API."""
        # Get status
        success, result = self.client.get_training_status(self.session_id)
        if success:
            status_data = result.get('status', {})
            self.status = status_data.get('status', 'unknown')

        # Get metrics
        success, result = self.client.get_training_metrics(self.session_id)
        if success:
            self.metrics = result

        # Get logs (incremental)
        success, result = self.client.get_training_logs(self.session_id, limit=100)
        if success:
            logs = result.get('logs', [])
            # Only keep new logs
            if len(logs) > self.last_log_count:
                self.logs.extend(logs[self.last_log_count:])
                self.last_log_count = len(logs)

                # Keep only last 100 logs in memory
                if len(self.logs) > 100:
                    self.logs = self.logs[-100:]

    def generate_dashboard(self) -> Layout:
        """Generate the complete dashboard."""
        layout = self.create_layout()

        layout["header"].update(self.generate_header())
        layout["left"].update(self.generate_metrics_panel())
        layout["right"].split(
            Layout(self.generate_progress_panel()),
            Layout(self.generate_logs_panel())
        )

        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" to exit", style="dim")
        layout["footer"].update(Panel(footer_text, style="dim"))

        return layout

    def run(self):
        """Run the live monitor."""
        with Live(self.generate_dashboard(), refresh_per_second=1) as live:
            try:
                while True:
                    self.update_data()
                    live.update(self.generate_dashboard())

                    # Check if training is complete
                    if self.status in ['completed', 'error', 'stopped']:
                        time.sleep(2)  # Show final state for 2 seconds
                        break

                    time.sleep(self.update_interval)

            except KeyboardInterrupt:
                pass

        # Show final status
        self.console.print(f"\n[bold]Final Status:[/bold] {format_status(self.status)}")

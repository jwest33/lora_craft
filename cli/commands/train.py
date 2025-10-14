"""Training management commands."""

import click
import json
import yaml
from pathlib import Path
from cli.utils.display import (
    print_error, print_success, print_info, print_table,
    print_panel, console, confirm
)
from cli.utils.formatters import format_timestamp, format_status


@click.group()
def train():
    """Training management commands."""
    pass


@train.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--watch', '-w', is_flag=True, help='Watch training progress in real-time')
@click.pass_obj
def start(ctx, config_file, watch):
    """Start a training session from a config file."""
    client = ctx.get_client()

    # Load config file
    config_path = Path(config_file)

    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
    except Exception as e:
        print_error(f"Failed to load config file: {e}")
        return

    # Handle nested config structure (frontend format)
    # If config has a 'config' key, extract the actual training config
    if isinstance(config_data, dict) and 'config' in config_data:
        config = config_data['config']
        config_name = config_data.get('name', config_path.stem)
        print_info(f"Loading config: {config_name}")
    else:
        # Flat config format (legacy)
        config = config_data

    # Validate config
    print_info("Validating configuration...")
    valid_success, valid_result = client.validate_config(config)

    if not valid_success or not valid_result.get('valid', False):
        print_error("Configuration validation failed:")
        errors = valid_result.get('errors', [])
        for error in errors:
            console.print(f"  [red]•[/red] {error}")
        return

    print_success("Configuration is valid")

    # Start training
    print_info("Starting training session...")

    success, result = client.start_training(config)

    if not success:
        print_error(f"Failed to start training: {result}")
        return

    session_id = result.get('session_id')
    print_success(f"Training started successfully")
    print_info(f"Session ID: {session_id}")

    if watch:
        # Import monitor command
        from cli.commands.train import monitor as monitor_cmd
        ctx2 = click.Context(monitor_cmd, obj=ctx)
        ctx2.invoke(monitor_cmd, session_id=session_id)


@train.command()
@click.argument('session_id')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation')
@click.pass_obj
def stop(ctx, session_id, force):
    """Stop a training session."""
    if not force:
        if not confirm(f"Stop training session {session_id}?"):
            print_info("Cancelled")
            return

    client = ctx.get_client()

    success, result = client.stop_training(session_id)

    if not success:
        print_error(f"Failed to stop training: {result}")
        return

    print_success(f"Training session {session_id} stopped")


@train.command()
@click.argument('session_id')
@click.pass_obj
def status(ctx, session_id):
    """Get training session status."""
    client = ctx.get_client()

    success, result = client.get_training_status(session_id)

    if not success:
        print_error(f"Failed to get status: {result}")
        return

    status_data = result.get('status', {})

    # Display status
    from rich.table import Table
    table = Table(title=f"Training Status: {session_id}", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", format_status(status_data.get('status', 'unknown')))
    table.add_row("Created", format_timestamp(status_data.get('created_at', 'Unknown')))
    table.add_row("Display Name", status_data.get('display_name', 'N/A'))
    table.add_row("Progress", f"{status_data.get('progress', 0):.1f}%")

    console.print(table)


@train.command()
@click.argument('session_id')
@click.option('--lines', '-n', type=int, default=50, help='Number of log lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.pass_obj
def logs(ctx, session_id, lines, follow):
    """View training logs."""
    client = ctx.get_client()

    if follow:
        # Follow mode - poll for new logs
        import time
        last_line_count = 0

        try:
            while True:
                success, result = client.get_training_logs(session_id, limit=lines)

                if success:
                    logs = result.get('logs', [])

                    # Show only new logs
                    if len(logs) > last_line_count:
                        for log in logs[last_line_count:]:
                            console.print(log)
                        last_line_count = len(logs)

                time.sleep(2)  # Poll every 2 seconds

        except KeyboardInterrupt:
            print_info("\nStopped following logs")
            return
    else:
        # One-time fetch
        success, result = client.get_training_logs(session_id, limit=lines)

        if not success:
            print_error(f"Failed to get logs: {result}")
            return

        logs = result.get('logs', [])

        if not logs:
            print_info("No logs available")
            return

        for log in logs:
            console.print(log)


@train.command('list')
@click.option('--status-filter', '-s', type=click.Choice(['all', 'running', 'completed', 'error', 'stopped']),
              default='all', help='Filter by status')
@click.pass_obj
def list_sessions(ctx, status_filter):
    """List all training sessions."""
    client = ctx.get_client()

    success, result = client.list_training_sessions()

    if not success:
        print_error(f"Failed to list sessions: {result}")
        return

    sessions = result if isinstance(result, list) else result.get('sessions', [])

    # Filter by status
    if status_filter != 'all':
        sessions = [s for s in sessions if s.get('status', '').lower() == status_filter]

    if not sessions:
        print_info("No training sessions found")
        return

    # Format for display
    display_data = []
    for session in sessions:
        display_data.append({
            'session_id': session.get('session_id', 'Unknown')[:12] + '...',
            'status': format_status(session.get('status', 'unknown')),
            'model': session.get('model', 'Unknown'),
            'created': format_timestamp(session.get('created_at', 'Unknown'))
        })

    print_table(display_data, title=f"Training Sessions ({status_filter.title()})")


@train.command()
@click.argument('session_id')
@click.pass_obj
def metrics(ctx, session_id):
    """View training metrics."""
    client = ctx.get_client()

    success, result = client.get_training_metrics(session_id)

    if not success:
        print_error(f"Failed to get metrics: {result}")
        return

    if not result:
        print_info("No metrics available yet")
        return

    # Display metrics
    from cli.utils.display import print_metrics
    print_metrics(result, title=f"Training Metrics: {session_id}")


@train.command()
@click.argument('session_id')
@click.option('--interval', '-i', type=int, default=2, help='Update interval in seconds')
@click.pass_obj
def monitor(ctx, session_id):
    """Monitor training progress in real-time."""
    from cli.utils.monitor import TrainingMonitor

    client = ctx.get_client()

    # Check if session exists
    success, result = client.get_training_status(session_id)

    if not success:
        print_error(f"Session not found: {session_id}")
        return

    print_info(f"Monitoring training session: {session_id}")
    print_info("Press Ctrl+C to exit")

    # Create and run monitor
    monitor = TrainingMonitor(client, session_id)

    try:
        monitor.run()
    except KeyboardInterrupt:
        print_info("\nStopped monitoring")


@train.command()
@click.argument('session_id')
@click.pass_obj
def history(ctx, session_id):
    """Get training history for a session."""
    client = ctx.get_client()

    success, result = client.get_training_history(session_id)

    if not success:
        print_error(f"Failed to get history: {result}")
        return

    # Display history
    console.print(f"\n[bold cyan]Training History: {session_id}[/bold cyan]")

    console.print(f"\n[yellow]Status:[/yellow] {format_status(result.get('status', 'unknown'))}")
    console.print(f"[yellow]Progress:[/yellow] {result.get('progress', 0):.1f}%")

    # Show recent logs
    logs = result.get('logs', [])
    if logs:
        console.print(f"\n[bold]Recent Logs (last {len(logs)}):[/bold]")
        for log in logs[-10:]:  # Last 10 logs
            console.print(f"  {log}")

    # Show metrics history
    metrics = result.get('metrics', [])
    if metrics:
        console.print(f"\n[bold]Metrics History:[/bold]")
        from cli.utils.display import print_metrics
        for i, metric in enumerate(metrics[-5:], 1):  # Last 5 metrics
            print_metrics(metric, title=f"Step {i}")

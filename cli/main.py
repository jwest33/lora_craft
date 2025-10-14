"""
LoRA Craft CLI - Main entry point

Command-line interface for managing GRPO training without the web UI.
"""

import click
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.client import APIClient
from cli.config import CLIConfig
from cli.utils.display import print_error, print_success, print_info, console


# Global context object
class Context:
    """CLI context for passing state between commands."""

    def __init__(self):
        self.config = CLIConfig()
        self.client = None

    def get_client(self) -> APIClient:
        """Get or create API client."""
        if self.client is None:
            self.client = APIClient(
                base_url=self.config.server_url,
                timeout=self.config.timeout
            )
        return self.client


pass_context = click.make_pass_decorator(Context, ensure=True)


@click.group()
@click.option('--server', '-s', help='Server URL (default: http://localhost:5000)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.version_option(version='0.1.0', prog_name='loracraft')
@click.pass_context
def cli(ctx, server, verbose):
    """
    LoRA Craft CLI - Train and manage LoRA models from the command line.

    This tool provides complete access to the LoRA Craft training pipeline
    without requiring the web UI.
    """
    ctx.obj = Context()

    if server:
        ctx.obj.config.server_url = server

    if verbose:
        ctx.obj.config.verbose = True

    # Check server health
    client = ctx.obj.get_client()
    success, result = client.health_check()

    if not success:
        print_error(f"Cannot connect to LoRA Craft server at {ctx.obj.config.server_url}")
        print_info("Make sure the server is running: python server.py")
        sys.exit(1)


# ==================== System Commands ====================

@cli.group()
def system():
    """System information and status commands."""
    pass


@system.command()
@pass_context
def status(ctx):
    """Show system status and resource usage."""
    client = ctx.get_client()

    # Get system status
    success, result = client.get_system_status()

    if not success:
        print_error(f"Failed to get system status: {result}")
        return

    # Display status
    from rich.table import Table
    table = Table(title="System Status", show_header=True, header_style="bold cyan")
    table.add_column("Resource", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Mode", result.get('mode', 'Unknown'))
    table.add_row("GPU", result.get('gpu', 'N/A'))
    table.add_row("VRAM", result.get('vram', 'N/A'))
    table.add_row("RAM", result.get('ram', 'N/A'))
    table.add_row("CPU", result.get('cpu', 'N/A'))

    console.print(table)


@system.command()
@pass_context
def info(ctx):
    """Show detailed system information."""
    client = ctx.get_client()

    success, result = client.get_system_info()

    if not success:
        print_error(f"Failed to get system info: {result}")
        return

    # Display info
    from cli.utils.display import print_panel
    import json

    info_text = json.dumps(result, indent=2)
    print_panel(info_text, title="System Information")


# ==================== Config Commands ====================

@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command('show')
@pass_context
def config_show(ctx):
    """Show current CLI configuration."""
    from cli.utils.display import print_yaml
    print_yaml(ctx.config.config)


@config.command('set')
@click.argument('key')
@click.argument('value')
@pass_context
def config_set(ctx, key, value):
    """Set a configuration value."""
    ctx.config.set(key, value)
    ctx.config.save()
    print_success(f"Set {key} = {value}")


@config.command('get')
@click.argument('key')
@pass_context
def config_get(ctx, key):
    """Get a configuration value."""
    value = ctx.config.get(key)
    if value is None:
        print_error(f"Configuration key '{key}' not found")
    else:
        console.print(f"{key}: {value}")


@config.command('view-training')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), default='json',
              help='Output format (default: json)')
@pass_context
def config_view_training(ctx, format):
    """View saved training configuration files."""
    from cli.utils.display import print_table, select_option, print_json, print_yaml
    from datetime import datetime

    client = ctx.get_client()

    # Fetch list of configs
    success, result = client.list_configs()

    if not success:
        print_error(f"Failed to fetch configuration list: {result}")
        return

    if not result or len(result) == 0:
        print_info("No training configurations found")
        return

    # Format configs for display
    config_data = []
    for cfg in result:
        modified_date = datetime.fromtimestamp(cfg['modified']).strftime('%Y-%m-%d %H:%M:%S')
        config_data.append({
            'Name': cfg['name'],
            'Filename': cfg['filename'],
            'Size': f"{cfg['size'] / 1024:.1f} KB",
            'Modified': modified_date
        })

    # Display table
    print_table(config_data, title="Available Training Configurations")

    # Let user select a config
    config_names = [cfg['filename'] for cfg in result]
    selected = select_option("\nSelect a configuration to view", config_names)

    if selected is None:
        return

    # Load selected config
    print_info(f"Loading configuration: {selected}")
    success, config_content = client.load_config(selected)

    if not success:
        print_error(f"Failed to load configuration: {config_content}")
        return

    # Display config
    console.print()
    if format == 'json':
        print_json(config_content)
    else:
        print_yaml(config_content)


# Import command modules
from cli.commands import dataset, train, model, export


# Register command groups
cli.add_command(dataset.dataset)
cli.add_command(train.train)
cli.add_command(model.model)
cli.add_command(export.export)


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()

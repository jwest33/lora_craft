"""Model export commands."""

import click
from cli.utils.display import (
    print_error, print_success, print_info, print_table,
    print_panel, console, confirm, create_progress
)
from cli.utils.formatters import format_timestamp, format_bytes


@click.group()
def export():
    """Model export commands."""
    pass


@export.command('formats')
@click.pass_obj
def list_formats(ctx):
    """List available export formats and quantization options."""
    client = ctx.get_client()

    success, result = client.get_export_formats()

    if not success:
        print_error(f"Failed to get export formats: {result}")
        return

    # Display formats
    console.print("\n[bold cyan]Available Export Formats:[/bold cyan]")

    formats = result.get('formats', {})
    for format_name, format_info in formats.items():
        console.print(f"\n[yellow]{format_name}:[/yellow]")
        console.print(f"  Description: {format_info.get('description', 'N/A')}")

        if 'quantization_levels' in format_info:
            console.print(f"  Quantization levels:")
            for level in format_info['quantization_levels']:
                console.print(f"    • {level}")


@export.command('list')
@click.argument('session_id')
@click.pass_obj
def list_exports(ctx, session_id):
    """List exports for a training session."""
    client = ctx.get_client()

    success, result = client.list_exports(session_id)

    if not success:
        print_error(f"Failed to list exports: {result}")
        return

    exports = result.get('exports', [])

    if not exports:
        print_info("No exports found for this session")
        return

    # Format for display
    display_data = []
    for exp in exports:
        display_data.append({
            'name': exp.get('name', 'Unknown'),
            'format': exp.get('format', 'Unknown'),
            'size': format_bytes(exp.get('size', 0)),
            'created': format_timestamp(exp.get('created_at', 'Unknown'))
        })

    print_table(display_data, title=f"Exports for {session_id}")


@export.command('create')
@click.argument('session_id')
@click.option('--format', '-f', type=click.Choice(['huggingface', 'gguf']),
              default='gguf', help='Export format')
@click.option('--quantization', '-q', type=click.Choice(['q4_k_m', 'q5_k_m', 'q8_0', 'f16']),
              default='q4_k_m', help='Quantization level (for GGUF)')
@click.option('--name', '-n', help='Export name (default: auto-generated)')
@click.pass_obj
def create_export(ctx, session_id, format, quantization, name):
    """Export a trained model to a specific format."""
    client = ctx.get_client()

    export_config = {
        'format': format,
        'quantization': quantization if format == 'gguf' else None,
        'name': name
    }

    print_info(f"Exporting model: {session_id}")
    print_info(f"Format: {format}")

    if format == 'gguf':
        print_info(f"Quantization: {quantization}")

    with create_progress() as progress:
        task = progress.add_task(f"Exporting to {format}...", total=None)

        success, result = client.export_model(session_id, export_config)

        progress.update(task, completed=True)

    if not success:
        error = result if isinstance(result, str) else result.get('error', 'Unknown error')
        print_error(f"Export failed: {error}")
        return

    print_success(f"Model exported successfully")

    export_path = result.get('path')
    if export_path:
        print_info(f"Export location: {export_path}")


@export.command('delete')
@click.argument('session_id')
@click.argument('export_name')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation')
@click.pass_obj
def delete_export(ctx, session_id, export_name, force):
    """Delete a specific export."""
    if not force:
        if not confirm(f"Delete export '{export_name}' from {session_id}?"):
            print_info("Cancelled")
            return

    # Note: The API endpoint needs format in the path, but we may not know it
    # For now, this is a simplified version
    print_error("This command requires the export format. Use 'export list' to see available exports.")
    print_info("Coming soon: Improved delete command")


@export.command('batch')
@click.argument('session_ids', nargs=-1, required=True)
@click.option('--format', '-f', type=click.Choice(['huggingface', 'gguf']),
              default='gguf', help='Export format')
@click.option('--quantization', '-q', type=click.Choice(['q4_k_m', 'q5_k_m', 'q8_0', 'f16']),
              default='q4_k_m', help='Quantization level')
@click.pass_obj
def batch_export(ctx, session_ids, format, quantization):
    """Export multiple models in batch."""
    client = ctx.get_client()

    if len(session_ids) == 0:
        print_error("No session IDs provided")
        return

    print_info(f"Batch exporting {len(session_ids)} models...")
    print_info(f"Format: {format}, Quantization: {quantization}")

    # Export each model
    failed = []

    with create_progress() as progress:
        task = progress.add_task("Batch export...", total=len(session_ids))

        for session_id in session_ids:
            console.print(f"\nExporting {session_id}...")

            export_config = {
                'format': format,
                'quantization': quantization if format == 'gguf' else None
            }

            success, result = client.export_model(session_id, export_config)

            if success:
                print_success(f"✓ {session_id}")
            else:
                error = result if isinstance(result, str) else result.get('error', 'Unknown')
                print_error(f"✗ {session_id}: {error}")
                failed.append(session_id)

            progress.advance(task)

    # Summary
    console.print(f"\n[bold]Batch Export Summary:[/bold]")
    console.print(f"  Total: {len(session_ids)}")
    console.print(f"  [green]Success: {len(session_ids) - len(failed)}[/green]")
    console.print(f"  [red]Failed: {len(failed)}[/red]")

    if failed:
        console.print(f"\n[red]Failed exports:[/red]")
        for session_id in failed:
            console.print(f"  • {session_id}")


@export.command('status')
@click.argument('session_id')
@click.pass_obj
def export_status(ctx, session_id):
    """Check export status for a session."""
    client = ctx.get_client()

    # Get list of exports
    success, result = client.list_exports(session_id)

    if not success:
        print_error(f"Failed to get export status: {result}")
        return

    exports = result.get('exports', [])

    if not exports:
        console.print(f"[yellow]No exports found for session {session_id}[/yellow]")
        console.print(f"\nCreate an export with:")
        console.print(f"  loracraft export create {session_id}")
        return

    # Show export status
    console.print(f"\n[bold cyan]Export Status for {session_id}:[/bold cyan]")
    console.print(f"\nTotal exports: {len(exports)}")

    for exp in exports:
        name = exp.get('name', 'Unknown')
        format_type = exp.get('format', 'Unknown')
        size = format_bytes(exp.get('size', 0))

        console.print(f"\n  [green]✓[/green] {name}")
        console.print(f"    Format: {format_type}")
        console.print(f"    Size: {size}")

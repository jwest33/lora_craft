"""Dataset management commands."""

import click
from cli.utils.display import (
    print_error, print_success, print_info, print_table,
    print_panel, console, confirm
)
from cli.utils.formatters import format_bytes, format_timestamp


@click.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command()
@click.option('--category', '-c', type=click.Choice(['all', 'math', 'code', 'general', 'qa']),
              default='all', help='Filter by category')
@click.option('--cached-only', is_flag=True, help='Show only cached datasets')
@click.pass_obj
def list(ctx, category, cached_only):
    """List available datasets."""
    client = ctx.get_client()

    success, result = client.list_datasets()

    if not success:
        print_error(f"Failed to list datasets: {result}")
        return

    datasets = result.get('datasets', [])

    # Filter by category
    if category != 'all':
        datasets = [d for d in datasets if d.get('category', '').lower() == category]

    # Filter cached only
    if cached_only:
        datasets = [d for d in datasets if d.get('is_cached', False)]

    if not datasets:
        print_info("No datasets found")
        return

    # Format for display
    display_data = []
    for ds in datasets:
        cache_status = "[green]✓[/green]" if ds.get('is_cached') else "[red]✗[/red]"

        display_data.append({
            'name': ds['name'],
            'path': ds['path'],
            'category': ds.get('category', 'general'),
            'size': ds.get('size', 'Unknown'),
            'cached': cache_status
        })

    print_table(display_data, title=f"Available Datasets ({category.title()})")


@dataset.command()
@click.argument('dataset_name')
@click.option('--config', '-c', help='Dataset configuration name (for multi-config datasets)')
@click.option('--force', '-f', is_flag=True, help='Force re-download even if cached')
@click.pass_obj
def download(ctx, dataset_name, config, force):
    """Download a dataset from HuggingFace."""
    client = ctx.get_client()

    print_info(f"Downloading dataset: {dataset_name}")

    if config:
        print_info(f"Configuration: {config}")

    from cli.utils.display import create_progress

    with create_progress() as progress:
        task = progress.add_task(f"Downloading {dataset_name}...", total=None)

        success, result = client.download_dataset(
            dataset_name=dataset_name,
            dataset_config=config,
            force_download=force
        )

        progress.update(task, completed=True)

    if not success:
        print_error(f"Download failed: {result}")
        return

    print_success(f"Dataset downloaded successfully")

    # Show dataset info
    if 'dataset_info' in result:
        info = result['dataset_info']
        print_info(f"Rows: {info.get('num_rows', 'Unknown')}")
        print_info(f"Columns: {', '.join(info.get('columns', []))}")


@dataset.command()
@click.argument('dataset_path')
@click.option('--config', '-c', help='Dataset configuration name')
@click.option('--samples', '-n', type=int, default=5, help='Number of samples to show')
@click.pass_obj
def preview(ctx, dataset_path, config, samples):
    """Preview dataset samples.

    DATASET_PATH should be the full path (e.g., 'tatsu-lab/alpaca').
    Use 'loracraft dataset list' to see available datasets and their paths.
    """
    client = ctx.get_client()

    success, result = client.sample_dataset(
        dataset_name=dataset_path,
        dataset_config=config,
        sample_size=samples
    )

    if not success:
        error_msg = str(result)
        print_error(f"Failed to preview dataset: {result}")

        # Provide helpful hint if it looks like user provided a name instead of path
        if "doesn't exist" in error_msg or "cannot be accessed" in error_msg:
            print_info("Tip: Use the full dataset path (e.g., 'tatsu-lab/alpaca'), not just the name.")
            print_info("Run 'loracraft dataset list' to see available datasets and their paths.")
        return

    samples_data = result.get('samples', [])

    if not samples_data:
        print_info("No samples available")
        return

    # Display samples
    console.print(f"\n[bold cyan]Dataset Preview: {dataset_path}[/bold cyan]")

    for i, sample in enumerate(samples_data, 1):
        console.print(f"\n[yellow]Sample {i}:[/yellow]")
        for key, value in sample.items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            console.print(f"  [cyan]{key}:[/cyan] {value_str}")


@dataset.command()
@click.argument('dataset_name')
@click.option('--config', '-c', help='Dataset configuration name')
@click.pass_obj
def fields(ctx, dataset_name, config):
    """Detect and show dataset fields."""
    client = ctx.get_client()

    success, result = client.detect_dataset_fields(
        dataset_name=dataset_name,
        dataset_config=config,
        is_local='uploads' in dataset_name
    )

    if not success:
        print_error(f"Failed to detect fields: {result}")
        return

    columns = result.get('columns', [])
    suggested = result.get('suggested_mappings', {})

    # Show columns
    console.print("\n[bold cyan]Dataset Columns:[/bold cyan]")
    for col in columns:
        console.print(f"  • {col}")

    # Show suggested mappings
    if suggested:
        console.print("\n[bold cyan]Suggested Field Mappings:[/bold cyan]")
        for field, column in suggested.items():
            console.print(f"  [green]{field}[/green] → [yellow]{column}[/yellow]")


@dataset.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.pass_obj
def upload(ctx, file_path):
    """Upload a custom dataset file."""
    client = ctx.get_client()

    print_info(f"Uploading dataset: {file_path}")

    from cli.utils.display import create_progress

    with create_progress() as progress:
        task = progress.add_task("Uploading...", total=None)

        success, result = client.upload_dataset(file_path)

        progress.update(task, completed=True)

    if not success:
        print_error(f"Upload failed: {result}")
        return

    print_success(f"Dataset uploaded successfully")
    print_info(f"Filename: {result.get('filename')}")

    # Show dataset info
    if 'dataset_info' in result:
        info = result['dataset_info']
        if 'num_rows' in info:
            print_info(f"Rows: {info['num_rows']}")
        if 'columns' in info:
            print_info(f"Columns: {', '.join(info['columns'])}")


@dataset.command('list-uploaded')
@click.pass_obj
def list_uploaded(ctx):
    """List uploaded datasets."""
    client = ctx.get_client()

    success, result = client.list_uploaded_datasets()

    if not success:
        print_error(f"Failed to list uploaded datasets: {result}")
        return

    datasets = result.get('datasets', [])

    if not datasets:
        print_info("No uploaded datasets found")
        return

    # Format for display
    display_data = []
    for ds in datasets:
        display_data.append({
            'filename': ds['filename'],
            'size': format_bytes(ds.get('size', 0)),
            'uploaded': format_timestamp(ds.get('uploaded_at', 'Unknown'))
        })

    print_table(display_data, title="Uploaded Datasets")


@dataset.command('clear-cache')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation')
@click.pass_obj
def clear_cache(ctx, force):
    """Clear dataset cache."""
    if not force:
        if not confirm("Are you sure you want to clear the dataset cache?"):
            print_info("Cancelled")
            return

    client = ctx.get_client()

    success, result = client.clear_dataset_cache()

    if not success:
        print_error(f"Failed to clear cache: {result}")
        return

    print_success("Dataset cache cleared successfully")


@dataset.command()
@click.argument('dataset_name')
@click.option('--config', '-c', help='Dataset configuration name')
@click.pass_obj
def status(ctx, dataset_name, config):
    """Check dataset cache status."""
    client = ctx.get_client()

    success, result = client.get_dataset_status(dataset_name, config)

    if not success:
        print_error(f"Failed to get status: {result}")
        return

    is_cached = result.get('is_cached', False)
    cache_info = result.get('cache_info')

    if is_cached:
        console.print(f"[green]✓[/green] Dataset is cached")

        if cache_info:
            console.print(f"\n[cyan]Cache Information:[/cyan]")
            console.print(f"  Rows: {cache_info.get('num_rows', 'Unknown')}")
            console.print(f"  Columns: {cache_info.get('num_columns', 'Unknown')}")
            console.print(f"  Size: {format_bytes(cache_info.get('size_bytes', 0))}")
            console.print(f"  Cached: {format_timestamp(cache_info.get('cached_at', 'Unknown'))}")
    else:
        console.print(f"[red]✗[/red] Dataset is not cached")
        console.print(f"\nUse 'loracraft dataset download {dataset_name}' to cache it")

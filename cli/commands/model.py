"""Model management and testing commands."""

import click
from cli.utils.display import (
    print_error, print_success, print_info, print_table,
    print_panel, console, confirm
)
from cli.utils.formatters import format_timestamp, format_model_name


@click.group()
def model():
    """Model management and testing commands."""
    pass


@model.command('list')
@click.option('--base', is_flag=True, help='List base models instead of trained')
@click.pass_obj
def list_models(ctx, base):
    """List trained models or base models."""
    client = ctx.get_client()

    if base:
        success, result = client.list_available_models()
        models = result.get('models', []) if success else []
        title = "Available Base Models"

        if not models:
            print_info("No base models available")
            return

        # Format for display
        display_data = []
        for model in models:
            display_data.append({
                'id': model.get('id', 'Unknown'),
                'name': format_model_name(model.get('name', 'Unknown')),
                'family': model.get('family', 'Unknown'),
                'size': model.get('size', 'Unknown')
            })

    else:
        success, result = client.list_trained_models()
        models = result.get('models', []) if success else []
        title = "Trained Models"

        if not models:
            print_info("No trained models found")
            return

        # Format for display
        display_data = []
        for model in models:
            session_id = model.get('session_id', 'Unknown')
            display_data.append({
                'session_id': session_id[:12] + '...',
                'model': format_model_name(model.get('model_name', 'Unknown')),
                'epochs': model.get('epochs', 0),
                'created': format_timestamp(model.get('created_at', 'Unknown'))
            })

    if not success:
        print_error(f"Failed to list models: {result}")
        return

    print_table(display_data, title=title)


@model.command()
@click.argument('session_id')
@click.pass_obj
def info(ctx, session_id):
    """Get detailed model information."""
    client = ctx.get_client()

    success, result = client.get_model_info(session_id)

    if not success:
        print_error(f"Failed to get model info: {result}")
        return

    # Display info
    from rich.table import Table
    table = Table(title=f"Model Information: {session_id}", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model Name", result.get('model_name', 'Unknown'))
    table.add_row("Status", result.get('status', 'Unknown'))
    table.add_row("Created", format_timestamp(result.get('created_at', 'Unknown')))

    if result.get('completed_at'):
        table.add_row("Completed", format_timestamp(result['completed_at']))

    table.add_row("Epochs Trained", str(result.get('epochs_trained', 0)))

    if result.get('best_reward') is not None:
        table.add_row("Best Reward", f"{result['best_reward']:.4f}")

    if result.get('checkpoint_path'):
        table.add_row("Checkpoint", result['checkpoint_path'])

    console.print(table)


@model.command()
@click.argument('session_id')
@click.option('--prompt', '-p', required=True, help='Test prompt')
@click.option('--temperature', '-t', type=float, default=0.7, help='Temperature (0.0-2.0)')
@click.option('--max-tokens', '-m', type=int, default=512, help='Maximum tokens to generate')
@click.option('--top-p', type=float, default=0.9, help='Top-p sampling (0.0-1.0)')
@click.pass_obj
def test(ctx, session_id, prompt, temperature, max_tokens, top_p):
    """Test a trained model with a prompt."""
    client = ctx.get_client()

    print_info(f"Testing model: {session_id}")
    print_info(f"Prompt: {prompt}")

    # Load model
    print_info("Loading model...")
    success, result = client.load_model(session_id, model_type='trained')

    if not success:
        print_error(f"Failed to load model: {result}")
        return

    # Generate response
    print_info("Generating response...")

    config = {
        'temperature': temperature,
        'max_length': max_tokens,
        'top_p': top_p
    }

    success, result = client.generate_response(session_id, prompt, config)

    if not success:
        print_error(f"Generation failed: {result}")
        return

    # Display response
    response = result.get('response', '')
    console.print("\n[bold cyan]Model Response:[/bold cyan]")
    print_panel(response, title="Response", style="green")


@model.command()
@click.argument('session_ids', nargs=-1, required=True)
@click.option('--prompt', '-p', required=True, help='Test prompt')
@click.option('--temperature', '-t', type=float, default=0.7, help='Temperature')
@click.option('--max-tokens', '-m', type=int, default=512, help='Maximum tokens')
@click.pass_obj
def compare(ctx, session_ids, prompt, temperature, max_tokens):
    """Compare multiple models on the same prompt."""
    client = ctx.get_client()

    if len(session_ids) < 2:
        print_error("Need at least 2 models to compare")
        return

    print_info(f"Comparing {len(session_ids)} models...")
    print_info(f"Prompt: {prompt}")

    config = {
        'temperature': temperature,
        'max_length': max_tokens
    }

    # Load all models first
    for session_id in session_ids:
        print_info(f"Loading {session_id}...")
        success, result = client.load_model(session_id, model_type='trained')
        if not success:
            print_error(f"Failed to load {session_id}: {result}")
            return

    # Compare
    print_info("Generating responses...")

    success, result = client.compare_models(list(session_ids), prompt, config)

    if not success:
        print_error(f"Comparison failed: {result}")
        return

    # Display results
    results = result.get('results', [])

    console.print("\n[bold cyan]Model Comparison Results:[/bold cyan]")

    for res in results:
        model_id = res.get('model_id', 'Unknown')
        response = res.get('response', '')
        success = res.get('success', False)

        if success:
            print_panel(response, title=f"Model: {model_id[:12]}...", style="green")
        else:
            error = res.get('error', 'Unknown error')
            print_panel(f"Error: {error}", title=f"Model: {model_id[:12]}...", style="red")


@model.command()
@click.argument('session_id')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation')
@click.pass_obj
def delete(ctx, session_id, force):
    """Delete a trained model and all associated data."""
    if not force:
        if not confirm(f"Delete model {session_id} and all its data?"):
            print_info("Cancelled")
            return

    client = ctx.get_client()

    print_info(f"Deleting model: {session_id}")

    success, result = client.delete_model(session_id)

    if not success:
        print_error(f"Failed to delete model: {result}")
        return

    print_success(f"Model {session_id} deleted successfully")

    if 'deleted' in result:
        deleted = result['deleted']
        console.print(f"\n[bold]Deleted items:[/bold]")
        for item in deleted:
            console.print(f"  • {item}")


@model.command('interactive')
@click.argument('session_id')
@click.option('--temperature', '-t', type=float, default=0.7, help='Temperature')
@click.option('--max-tokens', '-m', type=int, default=512, help='Maximum tokens')
@click.pass_obj
def interactive_test(ctx, session_id, temperature, max_tokens):
    """Interactive testing session with a model."""
    client = ctx.get_client()

    print_info(f"Loading model: {session_id}")

    # Load model
    success, result = client.load_model(session_id, model_type='trained')

    if not success:
        print_error(f"Failed to load model: {result}")
        return

    print_success("Model loaded successfully")
    print_info("Enter prompts to test (type 'exit' or 'quit' to stop)")

    config = {
        'temperature': temperature,
        'max_length': max_tokens
    }

    while True:
        try:
            prompt = console.input("\n[bold cyan]Prompt:[/bold cyan] ").strip()

            if prompt.lower() in ['exit', 'quit', 'q']:
                print_info("Exiting interactive mode")
                break

            if not prompt:
                continue

            # Generate response
            success, result = client.generate_response(session_id, prompt, config)

            if not success:
                print_error(f"Generation failed: {result}")
                continue

            response = result.get('response', '')
            console.print(f"\n[bold green]Response:[/bold green]")
            console.print(response)

        except KeyboardInterrupt:
            print_info("\nExiting interactive mode")
            break

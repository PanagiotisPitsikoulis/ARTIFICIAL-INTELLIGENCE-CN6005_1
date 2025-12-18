"""
Command Line Interface for Heart Disease Classification

This module provides a CLI using Click for training and evaluating the neural network.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║     Heart Disease Classification - Neural Network            ║
║     CN6005 - Artificial Intelligence Coursework 2025-26      ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def print_config_table(config):
    """Print configuration as a table."""
    table = Table(title="Training Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="green")

    table.add_row("Epochs", str(config.training.epochs))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Momentum", str(config.training.momentum))
    table.add_row("Batch Size", str(config.training.batch_size))
    table.add_row("Hidden Layers", str(config.model.hidden_sizes))
    table.add_row("Dropout Rate", str(config.model.dropout_rate))
    table.add_row("Weight Decay", str(config.training.weight_decay))
    table.add_row("Early Stopping", str(config.training.early_stopping_patience))
    table.add_row("Random Seed", str(config.training.seed))

    console.print(table)


def print_results_table(results):
    """Print test results as a table."""
    table = Table(title="Test Results", show_header=True, header_style="bold green")
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Accuracy", f"{results['accuracy']:.4f}")
    table.add_row("Precision", f"{results['precision']:.4f}")
    table.add_row("Recall", f"{results['recall']:.4f}")
    table.add_row("F1-Score", f"{results['f1_score']:.4f}")

    console.print(table)


@click.group()
@click.version_option(version="1.0.0", prog_name="heart-disease-classifier")
def cli():
    """Heart Disease Classification Neural Network CLI.

    Train and evaluate a neural network for classifying heart disease
    severity using the Cleveland Heart Disease dataset.
    """
    pass


@cli.command()
@click.option('--epochs', '-e', default=500, type=int, help='Number of training epochs')
@click.option('--lr', '-l', default=0.001, type=float, help='Learning rate')
@click.option('--momentum', '-m', default=0.9, type=float, help='SGD momentum')
@click.option('--batch-size', '-b', default=16, type=int, help='Training batch size')
@click.option('--hidden', '-h', multiple=True, type=int, default=[64, 32, 16],
              help='Hidden layer sizes (can be specified multiple times)')
@click.option('--dropout', '-d', default=0.3, type=float, help='Dropout rate')
@click.option('--weight-decay', '-w', default=1e-4, type=float, help='L2 regularization')
@click.option('--early-stopping', default=30, type=int, help='Early stopping patience')
@click.option('--seed', '-s', default=42, type=int, help='Random seed')
@click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']), default='sgd',
              help='Optimizer to use')
@click.option('--no-save', is_flag=True, help='Do not save model and visualizations')
@click.option('--quiet', '-q', is_flag=True, help='Suppress detailed output')
def train(epochs, lr, momentum, batch_size, hidden, dropout, weight_decay,
          early_stopping, seed, optimizer, no_save, quiet):
    """Train the neural network model.

    Example usage:

        python -m app.cli train --epochs 200 --lr 0.001

        python -m app.cli train -e 300 -l 0.01 -h 128 -h 64 -h 32
    """
    import torch
    import numpy as np
    from app.config import Config, DataConfig, ModelConfig, TrainingConfig
    from app.data import preprocess_pipeline
    from app.models import create_model
    from app.training import Trainer, analyze_generalization
    from app.visualization import generate_all_visualizations

    if not quiet:
        print_banner()

    # Convert hidden tuple to list
    hidden_sizes = list(hidden) if hidden else [64, 32, 16]

    # Create configuration
    config = Config(
        data=DataConfig(random_state=seed),
        model=ModelConfig(hidden_sizes=hidden_sizes, dropout_rate=dropout),
        training=TrainingConfig(
            epochs=epochs,
            learning_rate=lr,
            momentum=momentum,
            batch_size=batch_size,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping,
            seed=seed,
            optimizer=optimizer
        )
    )

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not quiet:
        print_config_table(config)

    # Step 1: Data Preprocessing
    console.print("\n[bold cyan]Step 1:[/bold cyan] Loading and preprocessing data...")

    data = preprocess_pipeline(
        data_path=config.data.data_file,
        batch_size=batch_size,
        random_state=seed
    )

    # Step 2: Create Model
    console.print("\n[bold cyan]Step 2:[/bold cyan] Creating neural network...")

    model = create_model(
        input_size=data['input_size'],
        hidden_sizes=hidden_sizes,
        num_classes=data['num_classes'],
        dropout_rate=dropout,
        verbose=not quiet
    )

    # Step 3: Training
    console.print("\n[bold cyan]Step 3:[/bold cyan] Training the model...")

    trainer = Trainer(model, config=config.training)

    history = trainer.train(
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=epochs,
        verbose=not quiet
    )

    # Step 4: Testing
    console.print("\n[bold cyan]Step 4:[/bold cyan] Testing the model...")

    test_results = trainer.test(data['test_loader'], verbose=not quiet)

    if not quiet:
        print_results_table(test_results)

    # Step 5: Generalization Analysis
    analysis = analyze_generalization(history, test_results)
    if not quiet:
        console.print(analysis)

    # Step 6: Save Results
    if not no_save:
        console.print("\n[bold cyan]Step 5:[/bold cyan] Saving results...")

        trainer.save_model(config.model_save_path)

        generate_all_visualizations(
            history=history,
            test_results=test_results,
            output_dir=config.outputs_dir,
            y_test=data['y_test']
        )

        # Save reports
        report_path = Path(config.outputs_dir) / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Heart Disease Classification - Test Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n")
            f.write(f"Precision: {test_results['precision']:.4f}\n")
            f.write(f"Recall: {test_results['recall']:.4f}\n")
            f.write(f"F1-Score: {test_results['f1_score']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(test_results['confusion_matrix']) + "\n\n")
            f.write("Classification Report:\n")
            f.write(test_results['classification_report'] + "\n")
            f.write(analysis)

        console.print(f"[green]Results saved to {config.outputs_dir}[/green]")

    # Summary
    console.print(Panel(
        f"[bold green]Training Complete![/bold green]\n\n"
        f"Test Accuracy: {test_results['accuracy']:.4f}\n"
        f"F1-Score: {test_results['f1_score']:.4f}\n"
        f"Epochs Trained: {len(history['train_loss'])}",
        title="Summary"
    ))


@cli.command()
@click.option('--model-path', '-m', default=None, help='Path to saved model')
@click.option('--data-path', '-d', default=None, help='Path to test data')
def evaluate(model_path, data_path):
    """Evaluate a trained model on test data.

    Example:

        python -m app.cli evaluate --model-path app/models/saved/model.pth
    """
    import torch
    from app.config import default_config
    from app.data import preprocess_pipeline
    from app.models import create_model
    from app.training import Trainer

    print_banner()

    model_path = model_path or default_config.model_save_path
    data_path = data_path or default_config.data.data_file

    console.print(f"Loading model from: {model_path}")

    # Load data
    data = preprocess_pipeline(data_path=data_path)

    # Create and load model
    model = create_model(
        input_size=data['input_size'],
        num_classes=data['num_classes'],
        verbose=False
    )

    trainer = Trainer(model)
    trainer.load_model(model_path)

    # Evaluate
    results = trainer.test(data['test_loader'])

    print_results_table(results)


@cli.command()
def info():
    """Display information about the dataset and model architecture."""
    from app.config import default_config
    from app.data import load_data, describe_data

    print_banner()

    console.print("[bold]Dataset Information[/bold]\n")

    df = load_data(config=default_config.data)
    describe_data(df)

    console.print("\n[bold]Default Model Architecture[/bold]")
    console.print(f"  Input features: {default_config.model.input_size}")
    console.print(f"  Hidden layers: {default_config.model.hidden_sizes}")
    console.print(f"  Output classes: {default_config.model.num_classes}")
    console.print(f"  Dropout rate: {default_config.model.dropout_rate}")


@cli.command()
@click.argument('output_dir', default='app/outputs')
def visualize(output_dir):
    """Generate visualizations from the last training run.

    OUTPUT_DIR: Directory containing training results (default: app/outputs)
    """
    import pickle
    from pathlib import Path

    console.print(f"Generating visualizations in: {output_dir}")
    console.print("[yellow]Note: This command requires a saved model and history.[/yellow]")


if __name__ == '__main__':
    cli()

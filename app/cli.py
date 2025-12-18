"""
Διεπαφη Γραμμης Εντολων για Ταξινομηση Καρδιακων Παθησεων

Αυτη η μοναδα παρεχει CLI χρησιμοποιωντας Click για εκπαιδευση και αξιολογηση του νευρωνικου δικτυου.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import sys
import os

# Προσθηκη γονικου φακελου στο path για εισαγωγες
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()


def print_banner():
    """Εκτυπωση banner εφαρμογης."""
    banner = """
======================================================================
     Ταξινομηση Καρδιακων Παθησεων - Νευρωνικο Δικτυο
     CN6005 - Τεχνητη Νοημοσυνη Εργασια 2025-26
======================================================================
    """
    console.print(Panel(banner, style="bold blue"))


def print_config_table(config):
    """Εκτυπωση ρυθμισεων ως πινακα."""
    table = Table(title="Ρυθμισεις Εκπαιδευσης", show_header=True, header_style="bold cyan")
    table.add_column("Παραμετρος", style="dim")
    table.add_column("Τιμη", style="green")

    table.add_row("Εποχες", str(config.training.epochs))
    table.add_row("Ρυθμος Μαθησης", str(config.training.learning_rate))
    table.add_row("Ορμη", str(config.training.momentum))
    table.add_row("Μεγεθος Batch", str(config.training.batch_size))
    table.add_row("Κρυφα Επιπεδα", str(config.model.hidden_sizes))
    table.add_row("Ρυθμος Dropout", str(config.model.dropout_rate))
    table.add_row("Weight Decay", str(config.training.weight_decay))
    table.add_row("Πρωιμη Διακοπη", str(config.training.early_stopping_patience))
    table.add_row("Seed Τυχαιοτητας", str(config.training.seed))

    console.print(table)


def print_results_table(results):
    """Εκτυπωση αποτελεσματων δοκιμης ως πινακα."""
    table = Table(title="Αποτελεσματα Δοκιμης", show_header=True, header_style="bold green")
    table.add_column("Μετρικη", style="dim")
    table.add_column("Τιμη", style="bold")

    table.add_row("Ακριβεια", f"{results['accuracy']:.4f}")
    table.add_row("Precision", f"{results['precision']:.4f}")
    table.add_row("Recall", f"{results['recall']:.4f}")
    table.add_row("F1-Score", f"{results['f1_score']:.4f}")

    console.print(table)


@click.group()
@click.version_option(version="1.0.0", prog_name="heart-disease-classifier")
def cli():
    """CLI Νευρωνικου Δικτυου Ταξινομησης Καρδιακων Παθησεων.

    Εκπαιδευση και αξιολογηση νευρωνικου δικτυου για ταξινομηση σοβαροτητας
    καρδιακων παθησεων χρησιμοποιωντας το συνολο δεδομενων Cleveland Heart Disease.
    """
    pass


@cli.command()
@click.option('--epochs', '-e', default=500, type=int, help='Αριθμος εποχων εκπαιδευσης')
@click.option('--lr', '-l', default=0.001, type=float, help='Ρυθμος μαθησης')
@click.option('--momentum', '-m', default=0.9, type=float, help='Ορμη SGD')
@click.option('--batch-size', '-b', default=16, type=int, help='Μεγεθος batch εκπαιδευσης')
@click.option('--hidden', '-h', multiple=True, type=int, default=[64, 32, 16],
              help='Μεγεθη κρυφων επιπεδων (μπορει να καθοριστει πολλες φορες)')
@click.option('--dropout', '-d', default=0.3, type=float, help='Ρυθμος dropout')
@click.option('--weight-decay', '-w', default=1e-4, type=float, help='L2 κανονικοποιηση')
@click.option('--early-stopping', default=30, type=int, help='Υπομονη πρωιμης διακοπης')
@click.option('--seed', '-s', default=42, type=int, help='Seed τυχαιοτητας')
@click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']), default='sgd',
              help='Optimizer προς χρηση')
@click.option('--no-save', is_flag=True, help='Να μην αποθηκευτει το μοντελο και οι οπτικοποιησεις')
@click.option('--quiet', '-q', is_flag=True, help='Καταστολη αναλυτικης εξοδου')
def train(epochs, lr, momentum, batch_size, hidden, dropout, weight_decay,
          early_stopping, seed, optimizer, no_save, quiet):
    """Εκπαιδευση του μοντελου νευρωνικου δικτυου.

    Παραδειγματα χρησης:

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

    # Μετατροπη tuple σε λιστα
    hidden_sizes = list(hidden) if hidden else [64, 32, 16]

    # Δημιουργια ρυθμισεων
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

    # Ρυθμιση seeds τυχαιοτητας
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not quiet:
        print_config_table(config)

    # Βημα 1: Προεπεξεργασια Δεδομενων
    console.print("\n[bold cyan]Βημα 1:[/bold cyan] Φορτωση και προεπεξεργασια δεδομενων...")

    data = preprocess_pipeline(
        data_path=config.data.data_file,
        batch_size=batch_size,
        random_state=seed
    )

    # Βημα 2: Δημιουργια Μοντελου
    console.print("\n[bold cyan]Βημα 2:[/bold cyan] Δημιουργια νευρωνικου δικτυου...")

    model = create_model(
        input_size=data['input_size'],
        hidden_sizes=hidden_sizes,
        num_classes=data['num_classes'],
        dropout_rate=dropout,
        verbose=not quiet
    )

    # Βημα 3: Εκπαιδευση
    console.print("\n[bold cyan]Βημα 3:[/bold cyan] Εκπαιδευση του μοντελου...")

    trainer = Trainer(model, config=config.training)

    history = trainer.train(
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=epochs,
        verbose=not quiet
    )

    # Βημα 4: Δοκιμη
    console.print("\n[bold cyan]Βημα 4:[/bold cyan] Δοκιμη του μοντελου...")

    test_results = trainer.test(data['test_loader'], verbose=not quiet)

    if not quiet:
        print_results_table(test_results)

    # Βημα 5: Αναλυση Γενικευσης
    analysis = analyze_generalization(history, test_results)
    if not quiet:
        console.print(analysis)

    # Βημα 6: Αποθηκευση Αποτελεσματων
    if not no_save:
        console.print("\n[bold cyan]Βημα 5:[/bold cyan] Αποθηκευση αποτελεσματων...")

        trainer.save_model(config.model_save_path)

        generate_all_visualizations(
            history=history,
            test_results=test_results,
            output_dir=config.outputs_dir,
            y_test=data['y_test']
        )

        # Αποθηκευση αναφορων
        report_path = Path(config.outputs_dir) / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Ταξινομηση Καρδιακων Παθησεων - Αποτελεσματα Δοκιμης\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Ακριβεια Δοκιμης: {test_results['accuracy']:.4f}\n")
            f.write(f"Precision: {test_results['precision']:.4f}\n")
            f.write(f"Recall: {test_results['recall']:.4f}\n")
            f.write(f"F1-Score: {test_results['f1_score']:.4f}\n\n")
            f.write("Πινακας Συγχυσης:\n")
            f.write(str(test_results['confusion_matrix']) + "\n\n")
            f.write("Αναφορα Ταξινομησης:\n")
            f.write(test_results['classification_report'] + "\n")
            f.write(analysis)

        console.print(f"[green]Αποτελεσματα αποθηκευτηκαν στο {config.outputs_dir}[/green]")

    # Συνοψη
    console.print(Panel(
        f"[bold green]Εκπαιδευση Ολοκληρωθηκε![/bold green]\n\n"
        f"Ακριβεια Δοκιμης: {test_results['accuracy']:.4f}\n"
        f"F1-Score: {test_results['f1_score']:.4f}\n"
        f"Εποχες Εκπαιδευσης: {len(history['train_loss'])}",
        title="Συνοψη"
    ))


@cli.command()
@click.option('--model-path', '-m', default=None, help='Διαδρομη προς αποθηκευμενο μοντελο')
@click.option('--data-path', '-d', default=None, help='Διαδρομη προς δεδομενα δοκιμης')
def evaluate(model_path, data_path):
    """Αξιολογηση εκπαιδευμενου μοντελου σε δεδομενα δοκιμης.

    Παραδειγμα:

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

    console.print(f"Φορτωση μοντελου απο: {model_path}")

    # Φορτωση δεδομενων
    data = preprocess_pipeline(data_path=data_path)

    # Δημιουργια και φορτωση μοντελου
    model = create_model(
        input_size=data['input_size'],
        num_classes=data['num_classes'],
        verbose=False
    )

    trainer = Trainer(model)
    trainer.load_model(model_path)

    # Αξιολογηση
    results = trainer.test(data['test_loader'])

    print_results_table(results)


@cli.command()
def info():
    """Εμφανιση πληροφοριων για το συνολο δεδομενων και την αρχιτεκτονικη μοντελου."""
    from app.config import default_config
    from app.data import load_data, describe_data

    print_banner()

    console.print("[bold]Πληροφοριες Συνολου Δεδομενων[/bold]\n")

    df = load_data(config=default_config.data)
    describe_data(df)

    console.print("\n[bold]Προεπιλεγμενη Αρχιτεκτονικη Μοντελου[/bold]")
    console.print(f"  Χαρακτηριστικα εισοδου: {default_config.model.input_size}")
    console.print(f"  Κρυφα επιπεδα: {default_config.model.hidden_sizes}")
    console.print(f"  Κλασεις εξοδου: {default_config.model.num_classes}")
    console.print(f"  Ρυθμος dropout: {default_config.model.dropout_rate}")


@cli.command()
@click.argument('output_dir', default='app/outputs')
def visualize(output_dir):
    """Δημιουργια οπτικοποιησεων απο την τελευταια εκπαιδευση.

    OUTPUT_DIR: Φακελος που περιεχει αποτελεσματα εκπαιδευσης (προεπιλογη: app/outputs)
    """
    console.print(f"Δημιουργια οπτικοποιησεων στο: {output_dir}")
    console.print("[yellow]Σημειωση: Αυτη η εντολη απαιτει αποθηκευμενο μοντελο και ιστορικο.[/yellow]")


if __name__ == '__main__':
    cli()

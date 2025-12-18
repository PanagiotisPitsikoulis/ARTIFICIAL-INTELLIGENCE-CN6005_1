"""
Διαδραστικη Διεπαφη Γραμμης Εντολων για Ταξινομηση Καρδιακων Παθησεων

Χρησιμοποιει InquirerPy για διαδραστικα μενου και επιλογες.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

# Προσθηκη γονικου φακελου στο path για εισαγωγες
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

console = Console()


def clear_screen():
    """Καθαρισμος οθονης."""
    console.clear()


def print_banner():
    """Εκτυπωση banner εφαρμογης."""
    banner = """
  Ταξινομηση Καρδιακων Παθησεων
  Νευρωνικο Δικτυο MLP

  CN6005 - Τεχνητη Νοημοσυνη
  Εργασια 2025-26
    """
    console.print(Panel(banner, style="bold blue", title="[white]Heart Disease Classifier[/white]"))


def print_config_table(config: Dict[str, Any]):
    """Εκτυπωση ρυθμισεων ως πινακα."""
    table = Table(title="Ρυθμισεις Εκπαιδευσης", show_header=True, header_style="bold cyan")
    table.add_column("Παραμετρος", style="dim")
    table.add_column("Τιμη", style="green")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)


def print_results_table(results: Dict[str, Any]):
    """Εκτυπωση αποτελεσματων δοκιμης ως πινακα."""
    table = Table(title="Αποτελεσματα Δοκιμης", show_header=True, header_style="bold green")
    table.add_column("Μετρικη", style="dim")
    table.add_column("Τιμη", style="bold")

    table.add_row("Ακριβεια", f"{results['accuracy']:.4f}")
    table.add_row("Precision", f"{results['precision']:.4f}")
    table.add_row("Recall", f"{results['recall']:.4f}")
    table.add_row("F1-Score", f"{results['f1_score']:.4f}")

    console.print(table)


def get_training_config() -> Dict[str, Any]:
    """Διαδραστικη διαμορφωση παραμετρων εκπαιδευσης."""
    console.print("\n[bold cyan]Διαμορφωση Παραμετρων Εκπαιδευσης[/bold cyan]\n")

    # Επιλογη προκαθορισμενης διαμορφωσης η προσαρμοσμενης
    config_type = inquirer.select(
        message="Επιλεξτε διαμορφωση:",
        choices=[
            Choice(value="default", name="Προεπιλεγμενες ρυθμισεις (συνισταται)"),
            Choice(value="quick", name="Γρηγορη εκπαιδευση (100 εποχες)"),
            Choice(value="thorough", name="Ενδελεχης εκπαιδευση (1000 εποχες)"),
            Choice(value="custom", name="Προσαρμοσμενες ρυθμισεις"),
        ],
        default="default",
    ).execute()

    if config_type == "default":
        return {
            "epochs": 500,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "batch_size": 16,
            "hidden_sizes": [64, 32, 16],
            "dropout": 0.3,
            "weight_decay": 1e-4,
            "early_stopping": 30,
            "optimizer": "sgd",
            "seed": 42
        }
    elif config_type == "quick":
        return {
            "epochs": 100,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "batch_size": 32,
            "hidden_sizes": [32, 16],
            "dropout": 0.2,
            "weight_decay": 1e-4,
            "early_stopping": 15,
            "optimizer": "adam",
            "seed": 42
        }
    elif config_type == "thorough":
        return {
            "epochs": 1000,
            "learning_rate": 0.0005,
            "momentum": 0.95,
            "batch_size": 8,
            "hidden_sizes": [128, 64, 32, 16],
            "dropout": 0.4,
            "weight_decay": 1e-5,
            "early_stopping": 50,
            "optimizer": "sgd",
            "seed": 42
        }
    else:
        # Προσαρμοσμενες ρυθμισεις
        return get_custom_config()


def get_custom_config() -> Dict[str, Any]:
    """Διαδραστικη εισαγωγη προσαρμοσμενων παραμετρων."""
    console.print("\n[dim]Εισαγετε τις προσαρμοσμενες παραμετρους:[/dim]\n")

    epochs = inquirer.number(
        message="Αριθμος εποχων:",
        default=500,
        min_allowed=10,
        max_allowed=5000
    ).execute()

    learning_rate = inquirer.select(
        message="Ρυθμος μαθησης:",
        choices=[
            Choice(value=0.1, name="0.1 (υψηλος)"),
            Choice(value=0.01, name="0.01 (μεσαιος)"),
            Choice(value=0.001, name="0.001 (χαμηλος - συνισταται)"),
            Choice(value=0.0001, name="0.0001 (πολυ χαμηλος)"),
        ],
        default=0.001,
    ).execute()

    batch_size = inquirer.select(
        message="Μεγεθος batch:",
        choices=[
            Choice(value=8, name="8 (μικρο)"),
            Choice(value=16, name="16 (μεσαιο - συνισταται)"),
            Choice(value=32, name="32 (μεγαλο)"),
            Choice(value=64, name="64 (πολυ μεγαλο)"),
        ],
        default=16,
    ).execute()

    architecture = inquirer.select(
        message="Αρχιτεκτονικη δικτυου:",
        choices=[
            Choice(value=[32, 16], name="Μικρο: 32 -> 16"),
            Choice(value=[64, 32], name="Μεσαιο: 64 -> 32"),
            Choice(value=[64, 32, 16], name="Προεπιλογη: 64 -> 32 -> 16 (συνισταται)"),
            Choice(value=[128, 64, 32], name="Μεγαλο: 128 -> 64 -> 32"),
            Choice(value=[128, 64, 32, 16], name="Βαθυ: 128 -> 64 -> 32 -> 16"),
        ],
        default=[64, 32, 16],
    ).execute()

    dropout = inquirer.select(
        message="Ρυθμος dropout:",
        choices=[
            Choice(value=0.1, name="0.1 (χαμηλος)"),
            Choice(value=0.2, name="0.2 (μετριος)"),
            Choice(value=0.3, name="0.3 (μεσαιος - συνισταται)"),
            Choice(value=0.4, name="0.4 (υψηλος)"),
            Choice(value=0.5, name="0.5 (πολυ υψηλος)"),
        ],
        default=0.3,
    ).execute()

    optimizer = inquirer.select(
        message="Optimizer:",
        choices=[
            Choice(value="sgd", name="SGD με momentum (συνισταται)"),
            Choice(value="adam", name="Adam"),
        ],
        default="sgd",
    ).execute()

    early_stopping = inquirer.number(
        message="Υπομονη πρωιμης διακοπης:",
        default=30,
        min_allowed=5,
        max_allowed=100
    ).execute()

    seed = inquirer.number(
        message="Seed τυχαιοτητας:",
        default=42,
        min_allowed=0,
        max_allowed=99999
    ).execute()

    return {
        "epochs": int(epochs),
        "learning_rate": learning_rate,
        "momentum": 0.9,
        "batch_size": batch_size,
        "hidden_sizes": architecture,
        "dropout": dropout,
        "weight_decay": 1e-4,
        "early_stopping": int(early_stopping),
        "optimizer": optimizer,
        "seed": int(seed)
    }


def run_training(config: Dict[str, Any]):
    """Εκτελεση εκπαιδευσης με τις δοθεισες ρυθμισεις."""
    import torch
    import numpy as np
    from app.config import Config, DataConfig, ModelConfig, TrainingConfig
    from app.data import preprocess_pipeline
    from app.models import create_model
    from app.training import Trainer, analyze_generalization
    from app.visualization import generate_all_visualizations

    # Εμφανιση ρυθμισεων
    print_config_table({
        "Εποχες": config["epochs"],
        "Ρυθμος Μαθησης": config["learning_rate"],
        "Μεγεθος Batch": config["batch_size"],
        "Κρυφα Επιπεδα": str(config["hidden_sizes"]),
        "Dropout": config["dropout"],
        "Optimizer": config["optimizer"].upper(),
        "Πρωιμη Διακοπη": config["early_stopping"],
        "Seed": config["seed"]
    })

    # Επιβεβαιωση
    confirm = inquirer.confirm(
        message="Να ξεκινησει η εκπαιδευση;",
        default=True
    ).execute()

    if not confirm:
        console.print("[yellow]Η εκπαιδευση ακυρωθηκε.[/yellow]")
        return

    # Δημιουργια config object
    app_config = Config(
        data=DataConfig(random_state=config["seed"]),
        model=ModelConfig(
            hidden_sizes=config["hidden_sizes"],
            dropout_rate=config["dropout"]
        ),
        training=TrainingConfig(
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            momentum=config["momentum"],
            batch_size=config["batch_size"],
            weight_decay=config["weight_decay"],
            early_stopping_patience=config["early_stopping"],
            seed=config["seed"],
            optimizer=config["optimizer"]
        )
    )

    # Ρυθμιση seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Βημα 1: Δεδομενα
    console.print("\n[bold cyan]Βημα 1/5:[/bold cyan] Φορτωση και προεπεξεργασια δεδομενων...")
    data = preprocess_pipeline(
        data_path=app_config.data.data_file,
        batch_size=config["batch_size"],
        random_state=config["seed"]
    )

    # Βημα 2: Μοντελο
    console.print("[bold cyan]Βημα 2/5:[/bold cyan] Δημιουργια νευρωνικου δικτυου...")
    model = create_model(
        input_size=data['input_size'],
        hidden_sizes=config["hidden_sizes"],
        num_classes=data['num_classes'],
        dropout_rate=config["dropout"],
        verbose=True
    )

    # Βημα 3: Εκπαιδευση
    console.print("[bold cyan]Βημα 3/5:[/bold cyan] Εκπαιδευση του μοντελου...\n")
    trainer = Trainer(model, config=app_config.training)
    history = trainer.train(
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=config["epochs"],
        verbose=True
    )

    # Βημα 4: Δοκιμη
    console.print("\n[bold cyan]Βημα 4/5:[/bold cyan] Δοκιμη του μοντελου...")
    test_results = trainer.test(data['test_loader'], verbose=True)
    print_results_table(test_results)

    # Αναλυση γενικευσης
    analysis = analyze_generalization(history, test_results)
    console.print(analysis)

    # Βημα 5: Αποθηκευση
    save_results = inquirer.confirm(
        message="Να αποθηκευτουν τα αποτελεσματα;",
        default=True
    ).execute()

    if save_results:
        console.print("[bold cyan]Βημα 5/5:[/bold cyan] Αποθηκευση αποτελεσματων...")

        trainer.save_model(app_config.model_save_path)

        generate_all_visualizations(
            history=history,
            test_results=test_results,
            output_dir=app_config.outputs_dir,
            y_test=data['y_test']
        )

        # Αποθηκευση αναφορας
        report_path = Path(app_config.outputs_dir) / 'classification_report.txt'
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

        console.print(f"[green]Αποτελεσματα αποθηκευτηκαν στο {app_config.outputs_dir}[/green]")

    # Συνοψη
    console.print(Panel(
        f"[bold green]Εκπαιδευση Ολοκληρωθηκε[/bold green]\n\n"
        f"Ακριβεια Δοκιμης: {test_results['accuracy']:.4f}\n"
        f"F1-Score: {test_results['f1_score']:.4f}\n"
        f"Εποχες Εκπαιδευσης: {len(history['train_loss'])}",
        title="Συνοψη"
    ))

    input("\nΠατηστε Enter για να συνεχισετε...")


def run_evaluation():
    """Αξιολογηση εκπαιδευμενου μοντελου."""
    import torch
    from app.config import default_config
    from app.data import preprocess_pipeline
    from app.models import create_model
    from app.training import Trainer

    console.print("\n[bold cyan]Αξιολογηση Μοντελου[/bold cyan]\n")

    model_path = default_config.model_save_path

    if not Path(model_path).exists():
        console.print("[red]Δεν βρεθηκε αποθηκευμενο μοντελο.[/red]")
        console.print(f"[dim]Αναμενομενη διαδρομη: {model_path}[/dim]")
        console.print("[yellow]Παρακαλω εκπαιδευστε πρωτα το μοντελο.[/yellow]")
        input("\nΠατηστε Enter για να συνεχισετε...")
        return

    console.print(f"Φορτωση μοντελου απο: {model_path}")

    # Φορτωση δεδομενων
    data = preprocess_pipeline(data_path=default_config.data.data_file)

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

    input("\nΠατηστε Enter για να συνεχισετε...")


def show_info():
    """Εμφανιση πληροφοριων για το συνολο δεδομενων."""
    from app.config import default_config
    from app.data import load_data, describe_data

    console.print("\n[bold cyan]Πληροφοριες Συστηματος[/bold cyan]\n")

    df = load_data(config=default_config.data)
    describe_data(df)

    console.print("\n[bold]Προεπιλεγμενη Αρχιτεκτονικη Μοντελου[/bold]")

    arch_table = Table(show_header=False, box=None)
    arch_table.add_column("", style="dim")
    arch_table.add_column("", style="green")

    arch_table.add_row("Χαρακτηριστικα εισοδου:", str(default_config.model.input_size))
    arch_table.add_row("Κρυφα επιπεδα:", str(default_config.model.hidden_sizes))
    arch_table.add_row("Κλασεις εξοδου:", str(default_config.model.num_classes))
    arch_table.add_row("Ρυθμος dropout:", str(default_config.model.dropout_rate))

    console.print(arch_table)

    # Περιγραφη κλασεων
    console.print("\n[bold]Περιγραφη Κλασεων[/bold]")

    class_table = Table(show_header=True, header_style="bold")
    class_table.add_column("Κλαση", style="cyan")
    class_table.add_column("Περιγραφη")

    class_table.add_row("0", "Καμια καρδιακη παθηση")
    class_table.add_row("1", "Ηπια καρδιακη παθηση")
    class_table.add_row("2", "Μετρια καρδιακη παθηση")
    class_table.add_row("3", "Σοβαρη καρδιακη παθηση")
    class_table.add_row("4", "Πολυ σοβαρη καρδιακη παθηση")

    console.print(class_table)

    input("\nΠατηστε Enter για να συνεχισετε...")


def show_visualizations():
    """Εμφανιση επιλογων οπτικοποιησεων."""
    from app.config import default_config

    output_dir = Path(default_config.outputs_dir)

    console.print("\n[bold cyan]Οπτικοποιησεις[/bold cyan]\n")

    if not output_dir.exists():
        console.print("[yellow]Δεν υπαρχουν διαθεσιμες οπτικοποιησεις.[/yellow]")
        console.print("[dim]Εκπαιδευστε πρωτα το μοντελο για να δημιουργηθουν.[/dim]")
        input("\nΠατηστε Enter για να συνεχισετε...")
        return

    # Λιστα διαθεσιμων αρχειων
    image_files = list(output_dir.glob("*.png"))

    if not image_files:
        console.print("[yellow]Δεν βρεθηκαν αρχεια οπτικοποιησεων.[/yellow]")
        input("\nΠατηστε Enter για να συνεχισετε...")
        return

    console.print(f"Βρεθηκαν {len(image_files)} αρχεια στο {output_dir}:\n")

    file_table = Table(show_header=True, header_style="bold")
    file_table.add_column("#", style="cyan")
    file_table.add_column("Αρχειο")
    file_table.add_column("Μεγεθος", style="dim")

    for i, f in enumerate(image_files, 1):
        size_kb = f.stat().st_size / 1024
        file_table.add_row(str(i), f.name, f"{size_kb:.1f} KB")

    console.print(file_table)

    console.print(f"\n[dim]Φακελος: {output_dir.absolute()}[/dim]")

    # Επιλογη ανοιγματος φακελου
    open_folder = inquirer.confirm(
        message="Να ανοιξει ο φακελος;",
        default=True
    ).execute()

    if open_folder:
        import subprocess
        import platform

        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(output_dir)])
        elif platform.system() == "Windows":
            subprocess.run(["explorer", str(output_dir)])
        else:  # Linux
            subprocess.run(["xdg-open", str(output_dir)])

    input("\nΠατηστε Enter για να συνεχισετε...")


def main_menu():
    """Κυριο μενου εφαρμογης."""
    while True:
        clear_screen()
        print_banner()

        action = inquirer.select(
            message="Επιλεξτε ενεργεια:",
            choices=[
                Choice(value="train", name="Εκπαιδευση μοντελου"),
                Choice(value="evaluate", name="Αξιολογηση μοντελου"),
                Choice(value="info", name="Πληροφοριες συστηματος"),
                Choice(value="visualize", name="Οπτικοποιησεις"),
                Separator(),
                Choice(value="exit", name="Εξοδος"),
            ],
            default="train",
            pointer=">>"
        ).execute()

        if action == "train":
            config = get_training_config()
            run_training(config)
        elif action == "evaluate":
            run_evaluation()
        elif action == "info":
            show_info()
        elif action == "visualize":
            show_visualizations()
        elif action == "exit":
            console.print("\n[bold blue]Ευχαριστουμε για τη χρηση![/bold blue]\n")
            break


def main():
    """Σημειο εισοδου CLI."""
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Διακοπη απο χρηστη.[/yellow]")
        sys.exit(0)


if __name__ == '__main__':
    main()

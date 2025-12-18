#!/usr/bin/env python3
"""
Κυριο Σημειο Εισοδου για Νευρωνικο Δικτυο Ταξινομησης Καρδιακων Παθησεων

CN6005 - Τεχνητη Νοημοσυνη Εργασια 2025-26

Χρηση:
    python -m app.main                      # Εκτελεση με προεπιλογες
    python -m app.main train --epochs 200   # Χρηση εντολων CLI
    python -m app.main --help               # Εμφανιση βοηθειας

Αυτο το script μπορει να εκτελεστει με δυο τροπους:
1. Αμεση εκτελεση: Εκτελει τη διαδικασια εκπαιδευσης με προεπιλεγμενες/CLI παραμετρους
2. Λειτουργια CLI: Χρησιμοποιει Click CLI για προχωρημενες επιλογες
"""

import sys
from pathlib import Path

# Εξασφαλιση οτι το app package ειναι εισαγωγιμο
app_dir = Path(__file__).resolve().parent
project_dir = app_dir.parent
sys.path.insert(0, str(project_dir))


def run_training_pipeline(epochs=500, lr=0.001, momentum=0.9, batch_size=16,
                          hidden_sizes=None, dropout=0.3, weight_decay=1e-4,
                          early_stopping=30, seed=42, save=True, verbose=True):
    """
    Εκτελεση της πληρους διαδικασιας εκπαιδευσης.

    Ορισματα:
        epochs: Αριθμος εποχων εκπαιδευσης
        lr: Ρυθμος μαθησης
        momentum: Ορμη SGD
        batch_size: Μεγεθος batch εκπαιδευσης
        hidden_sizes: Λιστα μεγεθων κρυφων επιπεδων
        dropout: Ρυθμος dropout
        weight_decay: L2 κανονικοποιηση
        early_stopping: Υπομονη πρωιμης διακοπης
        seed: Seed τυχαιοτητας
        save: Αν αποθηκευτει το μοντελο και τα αποτελεσματα
        verbose: Αν εκτυπωνεται η προοδος

    Επιστρεφει:
        Λεξικο με αποτελεσματα δοκιμης
    """
    import torch
    import numpy as np

    from app.config import Config, DataConfig, ModelConfig, TrainingConfig
    from app.data import preprocess_pipeline
    from app.models import create_model
    from app.training import Trainer, analyze_generalization
    from app.visualization import generate_all_visualizations

    if hidden_sizes is None:
        hidden_sizes = [64, 32, 16]

    # Ρυθμιση seeds τυχαιοτητας
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
            seed=seed
        )
    )

    if verbose:
        print("=" * 70)
        print("CN6005 - Τεχνητη Νοημοσυνη")
        print("Ταξινομηση Καρδιακων Παθησεων με Νευρωνικα Δικτυα")
        print("=" * 70)

    # Βημα 1: Προεπεξεργασια Δεδομενων
    if verbose:
        print("\n" + "=" * 70)
        print("ΒΗΜΑ 1: Φορτωση και Προεπεξεργασια Δεδομενων")
        print("=" * 70)

    data = preprocess_pipeline(
        data_path=config.data.data_file,
        batch_size=batch_size,
        random_state=seed
    )

    # Βημα 2: Δημιουργια Νευρωνικου Δικτυου
    if verbose:
        print("\n" + "=" * 70)
        print("ΒΗΜΑ 2: Δημιουργια Νευρωνικου Δικτυου")
        print("=" * 70)

    model = create_model(
        input_size=data['input_size'],
        hidden_sizes=hidden_sizes,
        num_classes=data['num_classes'],
        dropout_rate=dropout,
        verbose=verbose
    )

    # Βημα 3: Εκπαιδευση
    if verbose:
        print("\n" + "=" * 70)
        print("ΒΗΜΑ 3: Εκπαιδευση του Νευρωνικου Δικτυου")
        print("=" * 70)

    trainer = Trainer(model, config=config.training)

    history = trainer.train(
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=epochs,
        verbose=verbose
    )

    # Βημα 4: Δοκιμη
    if verbose:
        print("\n" + "=" * 70)
        print("ΒΗΜΑ 4: Δοκιμη του Μοντελου")
        print("=" * 70)

    test_results = trainer.test(data['test_loader'], verbose=verbose)

    # Βημα 5: Αναλυση Γενικευσης
    generalization_analysis = analyze_generalization(history, test_results)
    if verbose:
        print(generalization_analysis)

    # Βημα 6: Αποθηκευση Αποτελεσματων
    if save:
        if verbose:
            print("\n" + "=" * 70)
            print("ΒΗΜΑ 5: Αποθηκευση Μοντελου και Δημιουργια Οπτικοποιησεων")
            print("=" * 70)

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
            f.write(generalization_analysis)

        config_path = Path(config.outputs_dir) / 'training_config.txt'
        with open(config_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Ρυθμισεις Εκπαιδευσης\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Εποχες: {epochs}\n")
            f.write(f"Ρυθμος Μαθησης: {lr}\n")
            f.write(f"Ορμη: {momentum}\n")
            f.write(f"Μεγεθος Batch: {batch_size}\n")
            f.write(f"Κρυφα Επιπεδα: {hidden_sizes}\n")
            f.write(f"Ρυθμος Dropout: {dropout}\n")
            f.write(f"Weight Decay: {weight_decay}\n")
            f.write(f"Πρωιμη Διακοπη: {early_stopping}\n")
            f.write(f"Seed Τυχαιοτητας: {seed}\n")
            f.write(f"\nΠραγματικες εποχες εκπαιδευσης: {len(history['train_loss'])}\n")

        if verbose:
            print(f"Αποτελεσματα αποθηκευτηκαν στο {config.outputs_dir}")

    # Συνοψη
    if verbose:
        print("\n" + "=" * 70)
        print("ΣΥΝΟΨΗ")
        print("=" * 70)
        print(f"Τελικη Ακριβεια Δοκιμης: {test_results['accuracy']:.4f}")
        print(f"Τελικο F1-Score Δοκιμης: {test_results['f1_score']:.4f}")
        print(f"Εποχες εκπαιδευσης: {len(history['train_loss'])}")
        print("=" * 70)

    return test_results


def main():
    """Κυριο σημειο εισοδου - εκτελει CLI αν υπαρχουν ορισματα, αλλιως προεπιλεγμενη εκπαιδευση."""
    # Ελεγχος αν υπαρχουν ορισματα CLI
    if len(sys.argv) > 1:
        # Χρηση CLI
        from app.cli import cli
        cli()
    else:
        # Εκτελεση με προεπιλογες
        run_training_pipeline()


if __name__ == "__main__":
    main()

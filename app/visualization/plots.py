"""
Μοναδα Οπτικοποιησης για Ταξινομηση Καρδιακων Παθησεων

Αυτη η μοναδα δημιουργει καμπυλες εκπαιδευσης, πινακες συγχυσης και αλλες οπτικοποιησεις.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Optional
from pathlib import Path


def set_style():
    """Ρυθμιση στυλ γραφηματων για συνεπη οπτικα."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_training_curves(history: Dict,
                         save_path: Optional[str] = None,
                         show: bool = False) -> None:
    """
    Δημιουργια γραφηματων καμπυλων εκπαιδευσης και επικυρωσης.

    Ορισματα:
        history: Λεξικο με 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Διαδρομη αποθηκευσης εικονας (προαιρετικο)
        show: Αν εμφανιστει το γραφημα
    """
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Γραφημα απωλειας
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Απωλεια Εκπαιδευσης', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Απωλεια Επικυρωσης', linewidth=2)
    axes[0].set_xlabel('Εποχη')
    axes[0].set_ylabel('Απωλεια')
    axes[0].set_title('Απωλεια Εκπαιδευσης και Επικυρωσης')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Σημανση καλυτερης απωλειας επικυρωσης
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'Καλυτερη (εποχη {best_epoch})')
    axes[0].scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)

    # Γραφημα ακριβειας
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Ακριβεια Εκπαιδευσης', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Ακριβεια Επικυρωσης', linewidth=2)
    axes[1].set_xlabel('Εποχη')
    axes[1].set_ylabel('Ακριβεια')
    axes[1].set_title('Ακριβεια Εκπαιδευσης και Επικυρωσης')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Σημανση καλυτερης ακριβειας επικυρωσης
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    axes[1].axvline(x=best_acc_epoch, color='g', linestyle='--', alpha=0.7)
    axes[1].scatter([best_acc_epoch], [best_val_acc], color='g', s=100, zorder=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Καμπυλες εκπαιδευσης αποθηκευτηκαν στο {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(conf_matrix: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          show: bool = False) -> None:
    """
    Δημιουργια θερμικου χαρτη πινακα συγχυσης.

    Ορισματα:
        conf_matrix: Πινακας συγχυσης
        class_names: Λιστα ονοματων κλασεων
        save_path: Διαδρομη αποθηκευσης εικονας (προαιρετικο)
        show: Αν εμφανιστει το γραφημα
    """
    set_style()

    if class_names is None:
        class_names = [f'Κλαση {i}' for i in range(conf_matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Δημιουργια θερμικου χαρτη
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Πληθος'})

    ax.set_xlabel('Προβλεπομενη Ετικετα')
    ax.set_ylabel('Πραγματικη Ετικετα')

    # Προσθηκη ακριβειας στον τιτλο
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    ax.set_title(f'Πινακας Συγχυσης (Ακριβεια: {accuracy:.2%})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Πινακας συγχυσης αποθηκευτηκε στο {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(y: np.ndarray,
                            title: str = "Κατανομη Κλασεων",
                            save_path: Optional[str] = None,
                            show: bool = False) -> None:
    """
    Δημιουργια γραφηματος κατανομης κλασεων στο συνολο δεδομενων.

    Ορισματα:
        y: Πινακας ετικετων κλασεων
        title: Τιτλος γραφηματος
        save_path: Διαδρομη αποθηκευσης εικονας
        show: Αν εμφανιστει το γραφημα
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    unique, counts = np.unique(y, return_counts=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(unique)))

    bars = ax.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Κλαση (Σοβαροτητα Καρδιακης Νοσου)')
    ax.set_ylabel('Αριθμος Δειγματων')
    ax.set_title(title)
    ax.set_xticks(unique)

    # Προσθηκη ετικετων αριθμων στις μπαρες
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)

    # Προσθηκη ετικετων ποσοστου
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'({count/total:.1%})',
                   xy=(bar.get_x() + bar.get_width() / 2, height/2),
                   ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Γραφημα κατανομης κλασεων αποθηκευτηκε στο {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(test_results: Dict,
                            save_path: Optional[str] = None,
                            show: bool = False) -> None:
    """
    Δημιουργια γραφηματος συγκρισης διαφορετικων μετρικων.

    Ορισματα:
        test_results: Λεξικο με accuracy, precision, recall, f1_score
        save_path: Διαδρομη αποθηκευσης εικονας
        show: Αν εμφανιστει το γραφημα
    """
    set_style()

    metrics = ['Ακριβεια', 'Precision', 'Recall', 'F1-Score']
    values = [
        test_results['accuracy'],
        test_results['precision'],
        test_results['recall'],
        test_results['f1_score']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Βαθμολογια')
    ax.set_title('Μετρικες Επιδοσης Μοντελου')
    ax.set_ylim(0, 1)

    # Προσθηκη ετικετων τιμων στις μπαρες
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Προσθηκη οριζοντιας γραμμης στο 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Βαση (0.5)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Συγκριση μετρικων αποθηκευτηκε στο {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_all_visualizations(history: Dict,
                                test_results: Dict,
                                output_dir: str,
                                y_test: Optional[np.ndarray] = None) -> None:
    """
    Δημιουργια ολων των οπτικοποιησεων και αποθηκευση στον φακελο εξοδου.

    Ορισματα:
        history: Λεξικο ιστορικου εκπαιδευσης
        test_results: Λεξικο αποτελεσματων δοκιμης
        output_dir: Φακελος αποθηκευσης οπτικοποιησεων
        y_test: Ετικετες δοκιμης για κατανομη κλασεων
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Δημιουργια Οπτικοποιησεων ===")

    # Καμπυλες εκπαιδευσης
    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, 'training_curve.png'),
        show=False
    )

    # Πινακας συγχυσης
    class_names = ['Χωρις Νοσο (0)', 'Σοβαροτητα 1', 'Σοβαροτητα 2', 'Σοβαροτητα 3', 'Σοβαροτητα 4']
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        class_names=class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix.png'),
        show=False
    )

    # Συγκριση μετρικων
    plot_metrics_comparison(
        test_results,
        save_path=os.path.join(output_dir, 'metrics_comparison.png'),
        show=False
    )

    # Κατανομη κλασεων (αν παρεχεται y_test)
    if y_test is not None:
        plot_class_distribution(
            y_test,
            title="Κατανομη Κλασεων Συνολου Δοκιμης",
            save_path=os.path.join(output_dir, 'class_distribution.png'),
            show=False
        )

    print(f"Ολες οι οπτικοποιησεις αποθηκευτηκαν στο {output_dir}")


if __name__ == "__main__":
    print("Η μοναδα οπτικοποιησης φορτωθηκε επιτυχως")

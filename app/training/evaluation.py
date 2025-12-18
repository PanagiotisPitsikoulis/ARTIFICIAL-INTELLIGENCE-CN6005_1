"""
Μοναδα Αξιολογησης για Ταξινομηση Καρδιακων Παθησεων

Αυτη η μοναδα περιεχει συναρτησεις για αξιολογηση μοντελων και αναλυση γενικευσης.
"""

from typing import Dict
import numpy as np


def analyze_generalization(history: Dict, test_results: Dict) -> str:
    """
    Αναλυει την ικανοτητα γενικευσης του μοντελου.

    Ορισματα:
        history: Λεξικο ιστορικου εκπαιδευσης
        test_results: Λεξικο αποτελεσματων δοκιμης

    Επιστρεφει:
        Συμβολοσειρα αναλυσης
    """
    analysis = []
    analysis.append("\n" + "=" * 60)
    analysis.append("Αναλυση Γενικευσης")
    analysis.append("=" * 60)

    # Συγκριση επιδοσης εκπαιδευσης και επικυρωσης
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    test_acc = test_results['accuracy']

    gap = final_train_acc - final_val_acc
    analysis.append(f"\nΤελικη Ακριβεια Εκπαιδευσης: {final_train_acc:.4f}")
    analysis.append(f"Τελικη Ακριβεια Επικυρωσης: {final_val_acc:.4f}")
    analysis.append(f"Ακριβεια Δοκιμης: {test_acc:.4f}")
    analysis.append(f"\nΚενο Εκπαιδευσης-Επικυρωσης: {gap:.4f}")

    if gap > 0.1:
        analysis.append("\nΑξιολογηση: Ανιχνευτηκαν ενδειξεις ΥΠΕΡΠΡΟΣΑΡΜΟΓΗΣ.")
        analysis.append("Το μοντελο αποδιδει σημαντικα καλυτερα στα δεδομενα εκπαιδευσης απο τα δεδομενα επικυρωσης.")
        analysis.append("Συστασεις: Αυξηστε το dropout, προσθεστε κανονικοποιηση, ή μειωστε την πολυπλοκοτητα του μοντελου.")
    elif gap < 0.02:
        analysis.append("\nΑξιολογηση: ΚΑΛΗ ΓΕΝΙΚΕΥΣΗ.")
        analysis.append("Το μοντελο δειχνει συνεπη επιδοση σε συνολα εκπαιδευσης και επικυρωσης.")
    else:
        analysis.append("\nΑξιολογηση: ΑΠΟΔΕΚΤΗ ΓΕΝΙΚΕΥΣΗ.")
        analysis.append("Μικρο κενο μεταξυ επιδοσης εκπαιδευσης και επικυρωσης ειναι φυσιολογικο.")

    # Συγκριση επιδοσης επικυρωσης και δοκιμης
    val_test_gap = abs(final_val_acc - test_acc)
    analysis.append(f"\nΚενο Επικυρωσης-Δοκιμης: {val_test_gap:.4f}")

    if val_test_gap < 0.05:
        analysis.append("Το μοντελο γενικευει καλα σε αγνωστα δεδομενα δοκιμης.")
    else:
        analysis.append("Υπαρχει καποια διακυμανση στην επιδοση σε αγνωστα δεδομενα.")

    # Ελεγχος για υποπροσαρμογη
    if final_train_acc < 0.6:
        analysis.append("\nΠροειδοποιηση: Χαμηλη ακριβεια εκπαιδευσης υποδηλωνει ΥΠΟΠΡΟΣΑΡΜΟΓΗ.")
        analysis.append("Προτασεις: Αυξηστε τη χωρητικοτητα του μοντελου, εκπαιδευστε περισσοτερο, ή προσαρμοστε το ρυθμο μαθησης.")

    analysis.append("=" * 60)

    return "\n".join(analysis)


def evaluate_model(predictions: np.ndarray,
                   labels: np.ndarray,
                   class_names: list = None) -> Dict:
    """
    Υπολογισμος μετρικων αξιολογησης για προβλεψεις μοντελου.

    Ορισματα:
        predictions: Προβλεπομενες ετικετες κλασεων
        labels: Πραγματικες ετικετες κλασεων
        class_names: Προαιρετικη λιστα ονοματων κλασεων

    Επιστρεφει:
        Λεξικο με μετρικες αξιολογησης
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )

    if class_names is None:
        class_names = [f"Κλαση {i}" for i in range(len(np.unique(labels)))]

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    class_report = classification_report(
        labels, predictions,
        target_names=class_names,
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }


def compute_per_class_metrics(confusion_mat: np.ndarray) -> Dict:
    """
    Υπολογισμος μετρικων ανα κλαση απο τον πινακα συγχυσης.

    Ορισματα:
        confusion_mat: Πινακας συγχυσης

    Επιστρεφει:
        Λεξικο με μετρικες ανα κλαση
    """
    n_classes = confusion_mat.shape[0]
    metrics = {}

    for i in range(n_classes):
        tp = confusion_mat[i, i]
        fn = confusion_mat[i, :].sum() - tp
        fp = confusion_mat[:, i].sum() - tp
        tn = confusion_mat.sum() - tp - fn - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"κλαση_{i}"] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return metrics


if __name__ == "__main__":
    # Δοκιμη συναρτησεων αξιολογησης
    print("Η μοναδα αξιολογησης φορτωθηκε επιτυχως")

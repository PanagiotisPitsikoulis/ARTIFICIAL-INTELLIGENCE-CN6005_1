"""
Μοναδα Εκπαιδευσης για Νευρωνικο Δικτυο Ταξινομησης Καρδιακων Παθησεων

Αυτη η μοναδα διαχειριζεται την εκπαιδευση, επικυρωση και δοκιμη του νευρωνικου δικτυου.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import time
from typing import Dict, Optional, Tuple

from app.config import TrainingConfig, default_config


class EarlyStopping:
    """Πρωιμη διακοπη για αποφυγη υπερπροσαρμογης."""

    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        """
        Αρχικοποιηση πρωιμης διακοπης.

        Ορισματα:
            patience: Αριθμος εποχων αναμονης πριν τη διακοπη
            min_delta: Ελαχιστη αλλαγη για να θεωρηθει βελτιωση
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Επαναφορα κατασταση πρωιμης διακοπης."""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


class Trainer:
    """
    Κλαση εκπαιδευτη για εκπαιδευση και αξιολογηση νευρωνικου δικτυου.
    """

    def __init__(self, model: nn.Module,
                 config: Optional[TrainingConfig] = None,
                 device: Optional[str] = None):
        """
        Αρχικοποιηση του εκπαιδευτη.

        Ορισματα:
            model: Μοντελο PyTorch για εκπαιδευση
            config: Αντικειμενο TrainingConfig
            device: Συσκευη για χρηση ('cuda', 'mps', ή 'cpu')
        """
        self.model = model
        self.config = config or default_config.training

        # Προσδιορισμος συσκευης
        if device is not None:
            self.device = device
        elif self.config.device is not None:
            self.device = self.config.device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model.to(self.device)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        print(f"Χρηση συσκευης: {self.device}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Δημιουργια optimizer βασει ρυθμισεων."""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:  # Προεπιλογη SGD
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )

    def _create_scheduler(self, optimizer: optim.Optimizer):
        """Δημιουργια learning rate scheduler βασει ρυθμισεων."""
        if self.config.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience
            )
        elif self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
                gamma=self.config.lr_factor
            )
        return None

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: Optional[int] = None,
              learning_rate: Optional[float] = None,
              momentum: Optional[float] = None,
              weight_decay: Optional[float] = None,
              early_stopping_patience: Optional[int] = None,
              verbose: bool = True) -> Dict:
        """
        Εκπαιδευση του νευρωνικου δικτυου.

        Ορισματα:
            train_loader: DataLoader για δεδομενα εκπαιδευσης
            val_loader: DataLoader για δεδομενα επικυρωσης
            epochs: Αριθμος εποχων εκπαιδευσης (υπερισχυει ρυθμισεων)
            learning_rate: Ρυθμος μαθησης (υπερισχυει ρυθμισεων)
            momentum: Ορμη (υπερισχυει ρυθμισεων)
            weight_decay: L2 κανονικοποιηση (υπερισχυει ρυθμισεων)
            early_stopping_patience: Υπομονη για πρωιμη διακοπη (υπερισχυει ρυθμισεων)
            verbose: Αν εκτυπωνεται η προοδος

        Επιστρεφει:
            Λεξικο με ιστορικο εκπαιδευσης
        """
        # Υπερισχυση ρυθμισεων με παρεχομενες τιμες
        epochs = epochs or self.config.epochs
        if learning_rate is not None:
            self.config.learning_rate = learning_rate
        if momentum is not None:
            self.config.momentum = momentum
        if weight_decay is not None:
            self.config.weight_decay = weight_decay
        early_stopping_patience = early_stopping_patience or self.config.early_stopping_patience

        # Συναρτηση απωλειας
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = self._create_optimizer()

        # Learning rate scheduler
        scheduler = self._create_scheduler(optimizer)

        # Πρωιμη διακοπη
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )

        if verbose:
            print("\n" + "=" * 60)
            print("Ρυθμισεις Εκπαιδευσης")
            print("=" * 60)
            print(f"Εποχες: {epochs}")
            print(f"Ρυθμος Μαθησης: {self.config.learning_rate}")
            print(f"Ορμη: {self.config.momentum}")
            print(f"Weight Decay: {self.config.weight_decay}")
            print(f"Υπομονη Πρωιμης Διακοπης: {early_stopping_patience}")
            print(f"Optimizer: {self.config.optimizer.upper()}")
            print(f"Συναρτηση Απωλειας: CrossEntropyLoss")
            print("=" * 60)

        start_time = time.time()
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            # Φαση εκπαιδευσης
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Φαση επικυρωσης
            val_loss, val_acc = self.evaluate(val_loader, criterion)

            # Ενημερωση scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Καταγραφη ιστορικου
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Αποθηκευση καλυτερου μοντελου
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Εκτυπωση προοδου
            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                print(f"Εποχη [{epoch+1}/{epochs}] - "
                      f"Απωλεια Εκπ: {train_loss:.4f}, Ακριβεια Εκπ: {train_acc:.4f} - "
                      f"Απωλεια Επικ: {val_loss:.4f}, Ακριβεια Επικ: {val_acc:.4f}")

            # Πρωιμη διακοπη
            if self.config.early_stopping and early_stopping(val_loss):
                if verbose:
                    print(f"\nΠρωιμη διακοπη στην εποχη {epoch + 1}")
                break

        # Φορτωση καλυτερου μοντελου
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        total_time = time.time() - start_time

        if verbose:
            print(f"\nΕκπαιδευση ολοκληρωθηκε σε {total_time:.2f} δευτερολεπτα")
            print(f"Καλυτερη απωλεια επικυρωσης: {best_val_loss:.4f}")

        return self.history

    def evaluate(self, data_loader: DataLoader,
                 criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
        """
        Αξιολογηση του μοντελου σε συνολο δεδομενων.

        Ορισματα:
            data_loader: DataLoader για αξιολογηση
            criterion: Συναρτηση απωλειας (προαιρετικο)

        Επιστρεφει:
            Πλειαδα απο (απωλεια, ακριβεια)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def test(self, test_loader: DataLoader, verbose: bool = True) -> Dict:
        """
        Δοκιμη του μοντελου και παραγωγη αναλυτικων μετρικων.

        Ορισματα:
            test_loader: DataLoader για δεδομενα δοκιμης
            verbose: Αν εκτυπωνονται τα αποτελεσματα

        Επιστρεφει:
            Λεξικο με μετρικες δοκιμης
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Υπολογισμος μετρικων
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        class_report = classification_report(all_labels, all_predictions, zero_division=0)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }

        if verbose:
            print("\n" + "=" * 60)
            print("Αποτελεσματα Δοκιμης")
            print("=" * 60)
            print(f"Ακριβεια: {accuracy:.4f}")
            print(f"Precision (σταθμισμενο): {precision:.4f}")
            print(f"Recall (σταθμισμενο): {recall:.4f}")
            print(f"F1-Score (σταθμισμενο): {f1:.4f}")
            print("\nΠινακας Συγχυσης:")
            print(conf_matrix)
            print("\nΑναφορα Ταξινομησης:")
            print(class_report)
            print("=" * 60)

        return results

    def save_model(self, path: str):
        """Αποθηκευση του μοντελου σε αρχειο."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)
        print(f"Το μοντελο αποθηκευτηκε στο {path}")

    def load_model(self, path: str):
        """Φορτωση του μοντελου απο αρχειο."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Το μοντελο φορτωθηκε απο {path}")


if __name__ == "__main__":
    print("Η μοναδα εκπαιδευσης φορτωθηκε επιτυχως")

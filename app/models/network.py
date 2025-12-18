"""
Μοντελο Νευρωνικου Δικτυου για Ταξινομηση Καρδιακων Παθησεων

Αυτη η μοναδα ορισζει την αρχιτεκτονικη του νευρωνικου δικτυου χρησιμοποιωντας PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Optional

from app.config import ModelConfig, default_config


class HeartDiseaseClassifier(nn.Module):
    """
    Πολυεπιπεδο Perceptron για ταξινομηση καρδιακων παθησεων.

    Αρχιτεκτονικη:
        - Επιπεδο εισοδου: 13 χαρακτηριστικα
        - Κρυφα επιπεδα: Παραμετροποιησιμα (προεπιλογη: 64 -> 32 -> 16)
        - Επιπεδο εξοδου: 5 κλασεις (σοβαροτητα καρδιακης νοσου 0-4)
        - Ενεργοποιηση: ReLU για κρυφα επιπεδα
        - Dropout: Εφαρμοζεται μετα απο καθε κρυφο επιπεδο για κανονικοποιηση
        - Εξοδος: Log-softmax για χρηση με NLLLoss
    """

    def __init__(self, input_size: int = 13,
                 hidden_sizes: Optional[List[int]] = None,
                 num_classes: int = 5,
                 dropout_rate: float = 0.3,
                 config: Optional[ModelConfig] = None):
        """
        Αρχικοποιηση του νευρωνικου δικτυου.

        Ορισματα:
            input_size: Αριθμος χαρακτηριστικων εισοδου (προεπιλογη: 13)
            hidden_sizes: Λιστα μεγεθων κρυφων επιπεδων (προεπιλογη: [64, 32, 16])
            num_classes: Αριθμος κλασεων εξοδου (προεπιλογη: 5)
            dropout_rate: Πιθανοτητα dropout (προεπιλογη: 0.3)
            config: Αντικειμενο ModelConfig (υπερισχυει αλλων παραμετρων αν παρεχεται)
        """
        super(HeartDiseaseClassifier, self).__init__()

        # Χρηση config αν παρεχεται
        if config is not None:
            input_size = config.input_size
            hidden_sizes = config.hidden_sizes
            num_classes = config.num_classes
            dropout_rate = config.dropout_rate

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Δυναμικη κατασκευη επιπεδων
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            # Γραμμικο επιπεδο
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization για σταθερη εκπαιδευση
            layers.append(nn.BatchNorm1d(hidden_size))
            # ReLU ενεργοποιηση
            layers.append(nn.ReLU())
            # Dropout για κανονικοποιηση
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Επιπεδο εξοδου
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # Αρχικοποιηση βαρων
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Αρχικοποιηση βαρων χρησιμοποιωντας Xavier (Glorot) αρχικοποιηση.

        Αυτη η στρατηγικη βοηθα να διατηρηθει η διακυμανση των ενεργοποιησεων
        και των κλισεων σε ολα τα επιπεδα, οδηγωντας σε καλυτερη συγκλιση.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier ομοιομορφη αρχικοποιηση για βαρη
                init.xavier_uniform_(module.weight)
                # Αρχικοποιηση biases σε μικρες θετικες τιμες
                if module.bias is not None:
                    init.constant_(module.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass μεσω του δικτυου.

        Ορισματα:
            x: Tensor εισοδου σχηματος (batch_size, input_size)

        Επιστρεφει:
            Tensor εξοδου σχηματος (batch_size, num_classes)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Παραγωγη προβλεψεων (ετικετες κλασεων) για δεδομενα εισοδου.

        Ορισματα:
            x: Tensor εισοδου σχηματος (batch_size, input_size)

        Επιστρεφει:
            Προβλεπομενες ετικετες κλασεων σχηματος (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Παραγωγη πιθανοτητων προβλεψης για δεδομενα εισοδου.

        Ορισματα:
            x: Tensor εισοδου σχηματος (batch_size, input_size)

        Επιστρεφει:
            Tensor πιθανοτητων σχηματος (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities

    def get_architecture_summary(self) -> str:
        """
        Παραγωγη περιληψης της αρχιτεκτονικης του δικτυου.

        Επιστρεφει:
            Συμβολοσειρα που περιγραφει την αρχιτεκτονικη
        """
        summary = []
        summary.append("=" * 50)
        summary.append("Αρχιτεκτονικη Νευρωνικου Δικτυου")
        summary.append("=" * 50)
        summary.append(f"Επιπεδο Εισοδου: {self.input_size} χαρακτηριστικα")
        summary.append("")

        for i, hidden_size in enumerate(self.hidden_sizes):
            summary.append(f"Κρυφο Επιπεδο {i+1}:")
            summary.append(f"  - Νευρωνες: {hidden_size}")
            summary.append(f"  - Ενεργοποιηση: ReLU")
            summary.append(f"  - Batch Normalization: Ναι")
            summary.append(f"  - Dropout: {self.dropout_rate}")
            summary.append("")

        summary.append(f"Επιπεδο Εξοδου: {self.num_classes} κλασεις")
        summary.append(f"  - Ενεργοποιηση: Softmax (μεσω CrossEntropyLoss)")
        summary.append("")
        summary.append(f"Αρχικοποιηση Βαρων: Xavier Ομοιομορφη")
        summary.append(f"Αρχικοποιηση Bias: Σταθερη (0.01)")
        summary.append("=" * 50)

        return "\n".join(summary)

    def count_parameters(self) -> int:
        """
        Μετρηση συνολικου αριθμου εκπαιδευσιμων παραμετρων.

        Επιστρεφει:
            Συνολικος αριθμος εκπαιδευσιμων παραμετρων
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_size: int = 13,
                 hidden_sizes: Optional[List[int]] = None,
                 num_classes: int = 5,
                 dropout_rate: float = 0.3,
                 config: Optional[ModelConfig] = None,
                 verbose: bool = True) -> HeartDiseaseClassifier:
    """
    Συναρτηση factory για δημιουργια μοντελου HeartDiseaseClassifier.

    Ορισματα:
        input_size: Αριθμος χαρακτηριστικων εισοδου
        hidden_sizes: Λιστα μεγεθων κρυφων επιπεδων
        num_classes: Αριθμος κλασεων εξοδου
        dropout_rate: Πιθανοτητα dropout
        config: Αντικειμενο ModelConfig
        verbose: Αν εκτυπωθει η περιληψη αρχιτεκτονικης

    Επιστρεφει:
        Αρχικοποιημενο μοντελο HeartDiseaseClassifier
    """
    model = HeartDiseaseClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        config=config
    )

    if verbose:
        print(model.get_architecture_summary())
        print(f"\nΣυνολικες εκπαιδευσιμες παραμετροι: {model.count_parameters():,}")

    return model


if __name__ == "__main__":
    # Δοκιμη του μοντελου
    model = create_model()

    # Δοκιμη forward pass
    sample_input = torch.randn(5, 13)
    output = model(sample_input)

    print(f"\nΔοκιμη forward pass:")
    print(f"Σχημα εισοδου: {sample_input.shape}")
    print(f"Σχημα εξοδου: {output.shape}")

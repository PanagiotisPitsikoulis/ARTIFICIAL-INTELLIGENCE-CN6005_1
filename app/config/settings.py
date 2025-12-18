"""
Ρυθμισεις Παραμετρων για Ταξινομηση Καρδιακων Παθησεων

Αυτη η μοναδα περιεχει ολες τις παραμετροποιησιμες ρυθμισεις του project.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os


# Βασικες διαδρομες
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
OUTPUTS_DIR = BASE_DIR / "outputs"


@dataclass
class DataConfig:
    """Ρυθμισεις για φορτωση και προεπεξεργασια δεδομενων."""

    # Διαδρομη αρχειου δεδομενων
    data_file: str = str(RAW_DATA_DIR / "processed.cleveland.data")

    # Ονοματα στηλων για το συνολο δεδομενων
    column_names: List[str] = field(default_factory=lambda: [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ])

    # Αναλογιες διαχωρισμου δεδομενων
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15

    # Seed τυχαιοτητας για αναπαραγωγιμοτητα
    random_state: int = 42

    # Ενδειξη ελλειπουσας τιμης
    na_values: str = "?"


@dataclass
class ModelConfig:
    """Ρυθμισεις για αρχιτεκτονικη νευρωνικου δικτυου."""

    # Διαστασεις εισοδου και εξοδου
    input_size: int = 13
    num_classes: int = 5

    # Ρυθμισεις κρυφων επιπεδων
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32, 16])

    # Κανονικοποιηση
    dropout_rate: float = 0.3

    # Αρχικοποιηση βαρων
    init_method: str = "xavier_uniform"
    bias_init: float = 0.01


@dataclass
class TrainingConfig:
    """Ρυθμισεις για διαδικασια εκπαιδευσης."""

    # Παραμετροι εκπαιδευσης
    epochs: int = 500
    batch_size: int = 16

    # Παραμετροι optimizer
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer: str = "sgd"  # Επιλογες: "sgd", "adam"

    # Learning rate scheduler
    lr_scheduler: str = "plateau"  # Επιλογες: "plateau", "step", "none"
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Πρωιμη διακοπη
    early_stopping: bool = True
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.001

    # Συσκευη
    device: Optional[str] = None  # None = αυτοματη ανιχνευση

    # Αναπαραγωγιμοτητα
    seed: int = 42


@dataclass
class Config:
    """Κυρια κλαση ρυθμισεων που συνδυαζει ολες τις ρυθμισεις."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Διαδρομες εξοδου
    model_save_path: str = str(SAVED_MODELS_DIR / "heart_disease_model.pth")
    outputs_dir: str = str(OUTPUTS_DIR)

    # Καταγραφη
    verbose: bool = True

    def __post_init__(self):
        """Εξασφαλιση υπαρξης φακελων."""
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Δημιουργια Config απο λεξικο."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            model_save_path=config_dict.get("model_save_path", str(SAVED_MODELS_DIR / "heart_disease_model.pth")),
            outputs_dir=config_dict.get("outputs_dir", str(OUTPUTS_DIR)),
            verbose=config_dict.get("verbose", True)
        )

    def to_dict(self) -> dict:
        """Μετατροπη Config σε λεξικο."""
        return {
            "data": {
                "data_file": self.data.data_file,
                "train_size": self.data.train_size,
                "val_size": self.data.val_size,
                "test_size": self.data.test_size,
                "random_state": self.data.random_state,
            },
            "model": {
                "input_size": self.model.input_size,
                "num_classes": self.model.num_classes,
                "hidden_sizes": self.model.hidden_sizes,
                "dropout_rate": self.model.dropout_rate,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "momentum": self.training.momentum,
                "weight_decay": self.training.weight_decay,
                "early_stopping_patience": self.training.early_stopping_patience,
                "seed": self.training.seed,
            },
            "model_save_path": self.model_save_path,
            "outputs_dir": self.outputs_dir,
        }


# Προεπιλεγμενο αντικειμενο ρυθμισεων
default_config = Config()


if __name__ == "__main__":
    # Δοκιμη ρυθμισεων
    config = Config()
    print("Οι ρυθμισεις φορτωθηκαν επιτυχως!")
    print(f"Αρχειο δεδομενων: {config.data.data_file}")
    print(f"Κρυφα επιπεδα: {config.model.hidden_sizes}")
    print(f"Ρυθμος μαθησης: {config.training.learning_rate}")

"""
Κοινες ρυθμισεις για δοκιμες pytest.

Περιλαμβανει:
- Fixtures για κοινα δεδομενα δοκιμων
- Ρυθμισεις για αναπαραγωγιμοτητα
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Προσθηκη root διαδρομης στο path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Ρυθμιζει seed για αναπαραγωγιμοτητα σε καθε δοκιμη"""
    np.random.seed(42)
    torch.manual_seed(42)
    yield


@pytest.fixture
def sample_features():
    """Δημιουργει δειγματα χαρακτηριστικων για δοκιμες"""
    return np.random.randn(100, 13).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Δημιουργει δειγματα ετικετων για δοκιμες"""
    return np.random.randint(0, 5, 100)


@pytest.fixture
def sample_model():
    """Δημιουργει μοντελο για δοκιμες"""
    from app.models.network import HeartDiseaseClassifier
    return HeartDiseaseClassifier(
        input_size=13,
        hidden_sizes=[32, 16],
        num_classes=5,
        dropout_rate=0.3
    )


@pytest.fixture
def sample_batch():
    """Δημιουργει batch δεδομενων για δοκιμες"""
    x = torch.randn(16, 13)
    y = torch.randint(0, 5, (16,))
    return x, y

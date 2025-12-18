"""
Δοκιμες για τη βιβλιοθηκη lib.

Δοκιμαζει τις συναρτησεις:
- utils (set_seed, get_device, get_timestamp, format_time)
- metrics (calculate_metrics, calculate_confusion_matrix, calculate_per_class_metrics)
- io (save_results, load_results, ensure_directory)
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from app.lib.utils import (
    set_seed,
    get_device,
    get_timestamp,
    format_time,
    count_parameters
)
from app.lib.metrics import (
    calculate_metrics,
    calculate_confusion_matrix,
    calculate_per_class_metrics,
    get_classification_report,
    calculate_generalization_gap
)
from app.lib.io import (
    ensure_directory,
    save_results,
    load_results,
    save_text,
    load_text
)


class TestUtils:
    """Δοκιμες για utils.py"""

    def test_set_seed_reproducibility(self):
        """Ελεγχει οτι το seed παραγει αναπαραγωγιμα αποτελεσματα"""
        set_seed(42)
        random1 = np.random.rand(5)
        torch_rand1 = torch.rand(5)

        set_seed(42)
        random2 = np.random.rand(5)
        torch_rand2 = torch.rand(5)

        np.testing.assert_array_equal(random1, random2)
        torch.testing.assert_close(torch_rand1, torch_rand2)

    def test_get_device_returns_valid_device(self):
        """Ελεγχει οτι επιστρεφει εγκυρη συσκευη"""
        device = get_device()
        assert device in ['cuda', 'mps', 'cpu']

    def test_get_device_preferred(self):
        """Ελεγχει οτι η προτιμωμενη συσκευη επιστρεφεται"""
        device = get_device(preferred='cpu')
        assert device == 'cpu'

    def test_get_timestamp_format(self):
        """Ελεγχει τη μορφη του timestamp"""
        timestamp = get_timestamp()
        # Προεπιλεγμενη μορφη: YYYYMMDD_HHMMSS
        assert len(timestamp) == 15
        assert timestamp[8] == '_'

    def test_get_timestamp_custom_format(self):
        """Ελεγχει προσαρμοσμενη μορφη timestamp"""
        timestamp = get_timestamp("%Y-%m-%d")
        assert len(timestamp) == 10
        assert timestamp[4] == '-'

    def test_format_time_seconds(self):
        """Ελεγχει μορφοποιηση χρονου σε δευτερολεπτα"""
        result = format_time(45.5)
        assert "45.5" in result
        assert "δ" in result

    def test_format_time_minutes(self):
        """Ελεγχει μορφοποιηση χρονου σε λεπτα"""
        result = format_time(150)  # 2 λεπτα 30 δευτερολεπτα
        assert "2λ" in result
        assert "30δ" in result

    def test_format_time_hours(self):
        """Ελεγχει μορφοποιηση χρονου σε ωρες"""
        result = format_time(3660)  # 1 ωρα 1 λεπτο
        assert "1ω" in result
        assert "1λ" in result

    def test_count_parameters(self):
        """Ελεγχει τη μετρηση παραμετρων μοντελου"""
        model = torch.nn.Linear(10, 5)
        params = count_parameters(model)
        # 10*5 weights + 5 biases = 55
        assert params == 55


class TestMetrics:
    """Δοκιμες για metrics.py"""

    def setup_method(self):
        """Προετοιμασια δεδομενων δοκιμης"""
        self.y_true = np.array([0, 0, 1, 1, 2, 2])
        self.y_pred = np.array([0, 0, 1, 0, 2, 2])

    def test_calculate_metrics_keys(self):
        """Ελεγχει οτι επιστρεφονται ολες οι μετρικες"""
        metrics = calculate_metrics(self.y_true, self.y_pred)
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        for key in expected_keys:
            assert key in metrics

    def test_calculate_metrics_accuracy(self):
        """Ελεγχει τον υπολογισμο ακριβειας"""
        metrics = calculate_metrics(self.y_true, self.y_pred)
        # 5 σωστες / 6 συνολο = 0.833...
        assert abs(metrics['accuracy'] - 5/6) < 0.01

    def test_calculate_metrics_perfect(self):
        """Ελεγχει τελεια ταξινομηση"""
        y_perfect = np.array([0, 1, 2])
        metrics = calculate_metrics(y_perfect, y_perfect)
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0

    def test_calculate_confusion_matrix_shape(self):
        """Ελεγχει τις διαστασεις του πινακα συγχυσης"""
        cm = calculate_confusion_matrix(self.y_true, self.y_pred)
        assert cm.shape == (3, 3)

    def test_calculate_confusion_matrix_values(self):
        """Ελεγχει τις τιμες του πινακα συγχυσης"""
        cm = calculate_confusion_matrix(self.y_true, self.y_pred)
        # Διαγωνιος: σωστες προβλεψεις
        assert cm[0, 0] == 2  # κλαση 0
        assert cm[1, 1] == 1  # κλαση 1
        assert cm[2, 2] == 2  # κλαση 2

    def test_calculate_per_class_metrics(self):
        """Ελεγχει μετρικες ανα κλαση"""
        cm = calculate_confusion_matrix(self.y_true, self.y_pred)
        per_class = calculate_per_class_metrics(cm)

        assert len(per_class) == 3
        for i in range(3):
            key = f"κλαση_{i}"
            assert key in per_class
            assert 'precision' in per_class[key]
            assert 'recall' in per_class[key]
            assert 'f1_score' in per_class[key]

    def test_get_classification_report(self):
        """Ελεγχει τη δημιουργια αναφορας"""
        report = get_classification_report(self.y_true, self.y_pred)
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report

    def test_calculate_generalization_gap(self):
        """Ελεγχει τον υπολογισμο κενων γενικευσης"""
        gaps = calculate_generalization_gap(0.95, 0.90, 0.88)

        assert gaps['train_val_gap'] == 0.05
        assert gaps['train_test_gap'] == 0.07
        assert abs(gaps['val_test_gap'] - 0.02) < 0.001


class TestIO:
    """Δοκιμες για io.py"""

    def test_ensure_directory_creates(self):
        """Ελεγχει δημιουργια φακελου"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "test" / "nested"
            result = ensure_directory(new_dir)
            assert result.exists()
            assert result.is_dir()

    def test_ensure_directory_existing(self):
        """Ελεγχει με υπαρχοντα φακελο"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_directory(tmpdir)
            assert result.exists()

    def test_save_load_results(self):
        """Ελεγχει αποθηκευση και φορτωση αποτελεσματων"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"
            data = {
                'accuracy': 0.95,
                'epochs': [1, 2, 3],
                'nested': {'key': 'value'}
            }

            save_results(data, filepath)
            loaded = load_results(filepath)

            assert loaded['accuracy'] == 0.95
            assert loaded['epochs'] == [1, 2, 3]
            assert loaded['nested']['key'] == 'value'

    def test_save_results_numpy_conversion(self):
        """Ελεγχει μετατροπη numpy arrays"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "numpy_results.json"
            data = {
                'array': np.array([1, 2, 3]),
                'float': np.float32(0.5)
            }

            save_results(data, filepath)
            loaded = load_results(filepath)

            assert loaded['array'] == [1, 2, 3]
            assert loaded['float'] == 0.5

    def test_load_results_not_found(self):
        """Ελεγχει σφαλμα για ανυπαρκτο αρχειο"""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/path/file.json")

    def test_save_load_text(self):
        """Ελεγχει αποθηκευση και φορτωση κειμενου"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            text = "Δοκιμαστικο κειμενο\nΜε πολλες γραμμες"

            save_text(text, filepath)
            loaded = load_text(filepath)

            assert loaded == text

    def test_load_text_not_found(self):
        """Ελεγχει σφαλμα για ανυπαρκτο αρχειο κειμενου"""
        with pytest.raises(FileNotFoundError):
            load_text("/nonexistent/path/file.txt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Δοκιμες για την προεπεξεργασια δεδομενων.

Δοκιμαζει:
- Φορτωση δεδομενων
- Καθαρισμο δεδομενων
- Διαχωρισμο σε train/val/test
- Κανονικοποιηση
- Δημιουργια DataLoaders
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from app.data.preprocessing import (
    clean_data,
    prepare_features_targets,
    split_data,
    normalize_features,
    create_data_loaders
)
from app.config import DataConfig


class TestCleanData:
    """Δοκιμες για τον καθαρισμο δεδομενων"""

    def test_removes_nan_rows(self):
        """Ελεγχει αφαιρεση γραμμων με NaN"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [1, np.nan, 3, 4],
            'target': [0, 1, 0, 1]
        })

        cleaned = clean_data(df)

        assert len(cleaned) == 2
        assert not cleaned.isnull().any().any()

    def test_keeps_complete_rows(self):
        """Ελεγχει διατηρηση πληρων γραμμων"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        })

        cleaned = clean_data(df)

        assert len(cleaned) == 4


class TestPrepareFeatures:
    """Δοκιμες για προετοιμασια χαρακτηριστικων"""

    def test_separates_features_target(self):
        """Ελεγχει διαχωρισμο χαρακτηριστικων και στοχου"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 2]
        })

        X, y = prepare_features_targets(df)

        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_correct_dtypes(self):
        """Ελεγχει τους τυπους δεδομενων"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 2]
        })

        X, y = prepare_features_targets(df)

        assert X.dtype == np.float32
        assert y.dtype == np.int64


class TestSplitData:
    """Δοκιμες για διαχωρισμο δεδομενων"""

    def setup_method(self):
        """Προετοιμασια δεδομενων δοκιμης"""
        np.random.seed(42)
        self.X = np.random.randn(100, 13).astype(np.float32)
        self.y = np.random.randint(0, 5, 100)

    def test_split_sizes(self):
        """Ελεγχει τα μεγεθη των διαχωρισμενων συνολων"""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(self.X, self.y)

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(self.X)

    def test_no_overlap(self):
        """Ελεγχει οτι δεν υπαρχει επικαλυψη"""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(self.X, self.y)

        # Δημιουργια μοναδικων αναγνωριστικων
        train_set = set(map(tuple, X_train))
        val_set = set(map(tuple, X_val))
        test_set = set(map(tuple, X_test))

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_corresponding_labels(self):
        """Ελεγχει αντιστοιχια δεδομενων και ετικετων"""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(self.X, self.y)

        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)


class TestNormalizeFeatures:
    """Δοκιμες για κανονικοποιηση"""

    def setup_method(self):
        """Προετοιμασια δεδομενων"""
        np.random.seed(42)
        self.X_train = np.random.randn(70, 13).astype(np.float32) * 10 + 5
        self.X_val = np.random.randn(15, 13).astype(np.float32) * 10 + 5
        self.X_test = np.random.randn(15, 13).astype(np.float32) * 10 + 5

    def test_train_normalized(self):
        """Ελεγχει κανονικοποιηση εκπαιδευτικου συνολου"""
        X_train_scaled, _, _, _ = normalize_features(
            self.X_train, self.X_val, self.X_test
        )

        # Μεσος ορος κοντα στο 0, τυπικη αποκλιση κοντα στο 1
        assert abs(X_train_scaled.mean()) < 0.1
        assert abs(X_train_scaled.std() - 1.0) < 0.1

    def test_returns_scaler(self):
        """Ελεγχει οτι επιστρεφεται ο scaler"""
        _, _, _, scaler = normalize_features(
            self.X_train, self.X_val, self.X_test
        )

        assert hasattr(scaler, 'transform')
        assert hasattr(scaler, 'mean_')

    def test_consistent_transformation(self):
        """Ελεγχει συνεπεια μετασχηματισμου"""
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_features(
            self.X_train, self.X_val, self.X_test
        )

        # Εφαρμογη scaler ξανα στα δεδομενα επικυρωσης
        X_val_again = scaler.transform(self.X_val)

        np.testing.assert_array_almost_equal(X_val_scaled, X_val_again)


class TestDataLoaders:
    """Δοκιμες για DataLoaders"""

    def setup_method(self):
        """Προετοιμασια δεδομενων"""
        np.random.seed(42)
        self.X_train = np.random.randn(70, 13).astype(np.float32)
        self.y_train = np.random.randint(0, 5, 70)
        self.X_val = np.random.randn(15, 13).astype(np.float32)
        self.y_val = np.random.randint(0, 5, 15)
        self.X_test = np.random.randn(15, 13).astype(np.float32)
        self.y_test = np.random.randint(0, 5, 15)

    def test_creates_dataloaders(self):
        """Ελεγχει δημιουργια dataloaders"""
        train_loader, val_loader, test_loader = create_data_loaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            batch_size=16
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_size(self):
        """Ελεγχει το μεγεθος batch"""
        batch_size = 16
        train_loader, _, _ = create_data_loaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            batch_size=batch_size
        )

        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] <= batch_size

    def test_tensor_types(self):
        """Ελεγχει τους τυπους tensor"""
        train_loader, _, _ = create_data_loaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test
        )

        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.dtype == torch.float32
        assert batch_y.dtype == torch.int64

    def test_total_samples(self):
        """Ελεγχει συνολικα δειγματα μεσω dataloader"""
        train_loader, _, _ = create_data_loaders(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            batch_size=16
        )

        total = sum(batch_x.shape[0] for batch_x, _ in train_loader)
        assert total == len(self.X_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

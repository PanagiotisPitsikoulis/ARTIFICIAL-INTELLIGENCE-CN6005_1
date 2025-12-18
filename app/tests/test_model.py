"""
Δοκιμες για το νευρωνικο δικτυο.

Δοκιμαζει:
- Δημιουργια μοντελου HeartDiseaseClassifier
- Forward pass
- Προβλεψεις
- Αρχιτεκτονικη
"""

import pytest
import torch
import numpy as np

from app.models.network import HeartDiseaseClassifier, create_model


class TestHeartDiseaseClassifier:
    """Δοκιμες για την κλαση HeartDiseaseClassifier"""

    def setup_method(self):
        """Προετοιμασια μοντελου για δοκιμες"""
        self.input_size = 13
        self.num_classes = 5
        self.hidden_sizes = [32, 16]
        self.batch_size = 8

        self.model = HeartDiseaseClassifier(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            num_classes=self.num_classes,
            dropout_rate=0.3
        )

    def test_model_creation(self):
        """Ελεγχει τη δημιουργια μοντελου"""
        assert self.model is not None
        assert isinstance(self.model, torch.nn.Module)

    def test_model_attributes(self):
        """Ελεγχει τα χαρακτηριστικα του μοντελου"""
        assert self.model.input_size == self.input_size
        assert self.model.num_classes == self.num_classes
        assert self.model.hidden_sizes == self.hidden_sizes
        assert self.model.dropout_rate == 0.3

    def test_forward_pass_shape(self):
        """Ελεγχει τις διαστασεις εξοδου"""
        x = torch.randn(self.batch_size, self.input_size)
        output = self.model(x)

        assert output.shape == (self.batch_size, self.num_classes)

    def test_forward_pass_single_sample(self):
        """Ελεγχει forward pass με ενα δειγμα"""
        self.model.eval()  # BatchNorm απαιτει batch > 1 σε train mode
        x = torch.randn(1, self.input_size)
        output = self.model(x)

        assert output.shape == (1, self.num_classes)

    def test_predict_returns_labels(self):
        """Ελεγχει οτι predict επιστρεφει ετικετες"""
        x = torch.randn(self.batch_size, self.input_size)
        predictions = self.model.predict(x)

        assert predictions.shape == (self.batch_size,)
        assert predictions.dtype == torch.int64
        assert all(0 <= p < self.num_classes for p in predictions)

    def test_predict_proba_shape(self):
        """Ελεγχει τις διαστασεις πιθανοτητων"""
        x = torch.randn(self.batch_size, self.input_size)
        proba = self.model.predict_proba(x)

        assert proba.shape == (self.batch_size, self.num_classes)

    def test_predict_proba_sums_to_one(self):
        """Ελεγχει οτι οι πιθανοτητες αθροιζουν σε 1"""
        x = torch.randn(self.batch_size, self.input_size)
        proba = self.model.predict_proba(x)

        sums = proba.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(self.batch_size), atol=1e-5, rtol=1e-5)

    def test_predict_proba_non_negative(self):
        """Ελεγχει οτι οι πιθανοτητες ειναι μη αρνητικες"""
        x = torch.randn(self.batch_size, self.input_size)
        proba = self.model.predict_proba(x)

        assert (proba >= 0).all()

    def test_count_parameters(self):
        """Ελεγχει τη μετρηση παραμετρων"""
        params = self.model.count_parameters()
        assert params > 0
        assert isinstance(params, int)

    def test_architecture_summary(self):
        """Ελεγχει την περιληψη αρχιτεκτονικης"""
        summary = self.model.get_architecture_summary()

        assert isinstance(summary, str)
        assert "Εισοδου" in summary or "Input" in summary
        assert "Κρυφο" in summary or "Hidden" in summary
        assert "Εξοδου" in summary or "Output" in summary

    def test_different_hidden_sizes(self):
        """Ελεγχει διαφορετικα μεγεθη κρυφων επιπεδων"""
        hidden_configs = [
            [64],
            [128, 64],
            [256, 128, 64, 32]
        ]

        for hidden in hidden_configs:
            model = HeartDiseaseClassifier(
                input_size=13,
                hidden_sizes=hidden,
                num_classes=5
            )
            x = torch.randn(4, 13)
            output = model(x)
            assert output.shape == (4, 5)

    def test_dropout_effect(self):
        """Ελεγχει οτι το dropout λειτουργει"""
        x = torch.randn(100, self.input_size)

        self.model.train()
        outputs_train = [self.model(x) for _ in range(5)]

        # Σε train mode με dropout, τα αποτελεσματα πρεπει να διαφερουν
        all_same = all(torch.allclose(outputs_train[0], o) for o in outputs_train[1:])
        assert not all_same

        self.model.eval()
        outputs_eval = [self.model(x) for _ in range(5)]

        # Σε eval mode, τα αποτελεσματα πρεπει να ειναι ιδια
        all_same = all(torch.allclose(outputs_eval[0], o) for o in outputs_eval[1:])
        assert all_same


class TestCreateModel:
    """Δοκιμες για τη συναρτηση create_model"""

    def test_create_model_default(self):
        """Ελεγχει δημιουργια με προεπιλεγμενες ρυθμισεις"""
        model = create_model(verbose=False)

        assert isinstance(model, HeartDiseaseClassifier)
        assert model.input_size == 13
        assert model.num_classes == 5

    def test_create_model_custom_params(self):
        """Ελεγχει δημιουργια με προσαρμοσμενες παραμετρους"""
        model = create_model(
            input_size=20,
            hidden_sizes=[100, 50],
            num_classes=3,
            dropout_rate=0.5,
            verbose=False
        )

        assert model.input_size == 20
        assert model.hidden_sizes == [100, 50]
        assert model.num_classes == 3
        assert model.dropout_rate == 0.5


class TestModelGradients:
    """Δοκιμες για gradients"""

    def test_gradients_flow(self):
        """Ελεγχει οτι τα gradients ρεουν σωστα"""
        model = HeartDiseaseClassifier(input_size=13, num_classes=5)
        x = torch.randn(4, 13)
        y = torch.tensor([0, 1, 2, 3])

        criterion = torch.nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Ελεγχος οτι ολες οι παραμετροι εχουν gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Δεν υπαρχει gradient για {name}"

    def test_weight_update(self):
        """Ελεγχει οτι τα βαρη ενημερωνονται"""
        model = HeartDiseaseClassifier(input_size=13, num_classes=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Αποθηκευση αρχικων βαρων (μονο weights, οχι biases)
        initial_weights = []
        for name, p in model.named_parameters():
            if 'weight' in name:
                initial_weights.append((name, p.clone()))

        x = torch.randn(4, 13)
        y = torch.tensor([0, 1, 2, 3])

        criterion = torch.nn.CrossEntropyLoss()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Ελεγχος οτι τα βαρη αλλαξαν
        changes_found = False
        for name, initial in initial_weights:
            current = dict(model.named_parameters())[name]
            if not torch.allclose(initial, current):
                changes_found = True
                break
        assert changes_found, "Κανενα weight δεν αλλαξε μετα την ενημερωση"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

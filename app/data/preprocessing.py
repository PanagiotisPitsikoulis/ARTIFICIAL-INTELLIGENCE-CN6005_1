"""
Μοναδα Προεπεξεργασιας Δεδομενων για Ταξινομηση Καρδιακων Παθησεων

Αυτη η μοναδα διαχειριζεται τη φορτωση, τον καθαρισμο και την προετοιμασια
του συνολου δεδομενων Cleveland Heart Disease για εκπαιδευση νευρωνικου δικτυου.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional
from pathlib import Path

from app.config import DataConfig, default_config


def load_data(data_path: Optional[str] = None,
              config: Optional[DataConfig] = None) -> pd.DataFrame:
    """
    Φορτωνει το συνολο δεδομενων καρδιακων παθησεων απο αρχειο CSV.

    Ορισματα:
        data_path: Διαδρομη προς το αρχειο δεδομενων. Αν None, χρησιμοποιει config.
        config: Αντικειμενο DataConfig. Αν None, χρησιμοποιει προεπιλογη.

    Επιστρεφει:
        DataFrame με τα ακατεργαστα δεδομενα
    """
    if config is None:
        config = default_config.data

    if data_path is None:
        data_path = config.data_file

    df = pd.read_csv(
        data_path,
        names=config.column_names,
        na_values=config.na_values
    )
    print(f"Φορτωθηκαν {len(df)} δειγματα απο {data_path}")
    return df


def describe_data(df: pd.DataFrame) -> Dict:
    """
    Δημιουργει περιγραφικα στατιστικα για το συνολο δεδομενων.

    Ορισματα:
        df: DataFrame εισοδου

    Επιστρεφει:
        Λεξικο με στατιστικα δεδομενων
    """
    stats = {
        'total_samples': len(df),
        'features': len(df.columns) - 1,
        'missing_values': df.isnull().sum().to_dict(),
        'class_distribution': df['target'].value_counts().to_dict(),
        'numeric_stats': df.describe().to_dict()
    }

    print("\n=== Περιγραφη Συνολου Δεδομενων ===")
    print(f"Συνολικα δειγματα: {stats['total_samples']}")
    print(f"Αριθμος χαρακτηριστικων: {stats['features']}")
    print(f"\nΕλλειπουσες τιμες ανα στηλη:")
    for col, count in stats['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count}")
    print(f"\nΚατανομη κλασεων (στοχος):")
    for cls, count in sorted(stats['class_distribution'].items()):
        print(f"  Κλαση {cls}: {count} δειγματα ({count/len(df)*100:.1f}%)")

    return stats


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Καθαριζει το συνολο δεδομενων διαχειριζομενο τις ελλειπουσες τιμες.

    Στρατηγικη: Αφαιρεση γραμμων με ελλειπουσες τιμες (μονο 6 γραμμες επηρεαζονται)

    Ορισματα:
        df: DataFrame εισοδου με πιθανες ελλειπουσες τιμες

    Επιστρεφει:
        Καθαρισμενο DataFrame
    """
    initial_count = len(df)
    df_clean = df.dropna()
    removed_count = initial_count - len(df_clean)

    print(f"\n=== Καθαρισμος Δεδομενων ===")
    print(f"Αφαιρεθηκαν {removed_count} γραμμες με ελλειπουσες τιμες")
    print(f"Εναπομενοντα δειγματα: {len(df_clean)}")

    return df_clean


def prepare_features_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Διαχωριζει χαρακτηριστικα και μεταβλητη στοχο.

    Ορισματα:
        df: Καθαρισμενο DataFrame

    Επιστρεφει:
        Πλειαδα απο (πινακας χαρακτηριστικων, πινακας στοχων)
    """
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)

    return X, y


def split_data(X: np.ndarray, y: np.ndarray,
               config: Optional[DataConfig] = None) -> Tuple:
    """
    Διαχωριζει δεδομενα σε συνολα εκπαιδευσης, επικυρωσης και δοκιμης.

    Ορισματα:
        X: Πινακας χαρακτηριστικων
        y: Πινακας στοχων
        config: Αντικειμενο DataConfig. Αν None, χρησιμοποιει προεπιλογη.

    Επιστρεφει:
        Πλειαδα απο (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if config is None:
        config = default_config.data

    # Πρωτος διαχωρισμος: απομονωση συνολου δοκιμης
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    # Δευτερος διαχωρισμος: διαχωρισμος εκπαιδευσης και επικυρωσης
    val_ratio = config.val_size / (config.train_size + config.val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=config.random_state,
        stratify=y_temp
    )

    print(f"\n=== Διαχωρισμος Δεδομενων ===")
    print(f"Συνολο εκπαιδευσης: {len(X_train)} δειγματα ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Συνολο επικυρωσης: {len(X_val)} δειγματα ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Συνολο δοκιμης: {len(X_test)} δειγματα ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train: np.ndarray,
                       X_val: np.ndarray,
                       X_test: np.ndarray) -> Tuple:
    """
    Κανονικοποιει χαρακτηριστικα με StandardScaler (z-score κανονικοποιηση).

    Ο scaler προσαρμοζεται μονο στα δεδομενα εκπαιδευσης για αποφυγη διαρροης δεδομενων.

    Ορισματα:
        X_train: Χαρακτηριστικα εκπαιδευσης
        X_val: Χαρακτηριστικα επικυρωσης
        X_test: Χαρακτηριστικα δοκιμης

    Επιστρεφει:
        Πλειαδα απο (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n=== Κανονικοποιηση Χαρακτηριστικων ===")
    print(f"Εφαρμογη StandardScaler (z-score κανονικοποιηση)")
    print(f"Χαρακτηριστικα κλιμακωθηκαν σε mean=0, std=1 βασει δεδομενων εκπαιδευσης")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        batch_size: int = 16) -> Tuple:
    """
    Δημιουργει PyTorch DataLoaders για εκπαιδευση, επικυρωση και δοκιμη.

    Ορισματα:
        X_train, y_train: Δεδομενα εκπαιδευσης
        X_val, y_val: Δεδομενα επικυρωσης
        X_test, y_test: Δεδομενα δοκιμης
        batch_size: Μεγεθος batch για εκπαιδευση

    Επιστρεφει:
        Πλειαδα απο (train_loader, val_loader, test_loader)
    """
    # Μετατροπη σε PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Δημιουργια datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # Δημιουργια data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n=== Δημιουργια DataLoaders ===")
    print(f"Μεγεθος batch: {batch_size}")
    print(f"Batches εκπαιδευσης: {len(train_loader)}")
    print(f"Batches επικυρωσης: {len(val_loader)}")
    print(f"Batches δοκιμης: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def preprocess_pipeline(data_path: Optional[str] = None,
                        batch_size: int = 16,
                        random_state: int = 42) -> Dict:
    """
    Εκτελει την πληρη διαδικασια προεπεξεργασιας.

    Ορισματα:
        data_path: Διαδρομη προς το αρχειο δεδομενων
        batch_size: Μεγεθος batch για DataLoaders
        random_state: Seed τυχαιοτητας

    Επιστρεφει:
        Λεξικο με ολα τα επεξεργασμενα δεδομενα και μεταδεδομενα
    """
    from app.config import DataConfig

    config = DataConfig(random_state=random_state)

    if data_path is not None:
        config.data_file = data_path

    # Φορτωση και περιγραφη
    df = load_data(config=config)
    stats = describe_data(df)

    # Καθαρισμος
    df_clean = clean_data(df)

    # Προετοιμασια χαρακτηριστικων και στοχων
    X, y = prepare_features_targets(df_clean)

    # Διαχωρισμος
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, config)

    # Κανονικοποιηση
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_features(
        X_train, X_val, X_test
    )

    # Δημιουργια DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        batch_size=batch_size
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'stats': stats,
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'input_size': X_train.shape[1],
        'num_classes': len(np.unique(y))
    }


if __name__ == "__main__":
    # Δοκιμη της διαδικασιας προεπεξεργασιας
    from app.config import default_config
    result = preprocess_pipeline(data_path=default_config.data.data_file)
    print(f"\n=== Ολοκληρωση Διαδικασιας ===")
    print(f"Μεγεθος εισοδου: {result['input_size']}")
    print(f"Αριθμος κλασεων: {result['num_classes']}")

# CN6005 - Artificial Intelligence Coursework

Neural Network for Heart Disease Classification

## Project Overview

This project implements a neural network classifier for predicting heart disease using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The model classifies patients into 5 categories (0-4) based on 14 clinical attributes.

## Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **File:** `processed.cleveland.data`
- **Samples:** 303
- **Features:** 13 input features + 1 target variable
- **Classes:** 5 (0 = no disease, 1-4 = increasing severity)

### Features Description

| # | Attribute | Description |
|---|-----------|-------------|
| 1 | age | Age in years |
| 2 | sex | Sex (1 = male, 0 = female) |
| 3 | cp | Chest pain type (1-4) |
| 4 | trestbps | Resting blood pressure (mm Hg) |
| 5 | chol | Serum cholesterol (mg/dl) |
| 6 | fbs | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| 7 | restecg | Resting ECG results (0-2) |
| 8 | thalach | Maximum heart rate achieved |
| 9 | exang | Exercise induced angina (1 = yes, 0 = no) |
| 10 | oldpeak | ST depression induced by exercise |
| 11 | slope | Slope of peak exercise ST segment (1-3) |
| 12 | ca | Number of major vessels colored by fluoroscopy (0-3) |
| 13 | thal | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| 14 | target | Diagnosis (0-4) |

## Project Structure

```
ARTIFICIAL-INTELLIGENCE-CN6005_1/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── cli.py                  # Click CLI interface
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # Configuration classes
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Data loading & preprocessing
│   │   └── raw/
│   │       └── processed.cleveland.data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── network.py          # Neural network architecture
│   │   └── saved/
│   │       └── heart_disease_model.pth
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training logic
│   │   └── evaluation.py       # Evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py            # Visualization functions
│   └── outputs/
│       ├── training_curve.png
│       ├── confusion_matrix.png
│       ├── metrics_comparison.png
│       ├── class_distribution.png
│       └── classification_report.txt
├── docs/
│   └── instructions.md
├── course-resources/
├── venv/
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- click
- rich

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd ARTIFICIAL-INTELLIGENCE-CN6005_1
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start (Default Settings)

```bash
source venv/bin/activate
python -m app.main
```

### Using the CLI

The project includes a full CLI built with Click:

```bash
# Show help
python -m app.cli --help

# Train with default settings
python -m app.cli train

# Train with custom parameters
python -m app.cli train --epochs 200 --lr 0.01 --batch-size 32

# Train with custom hidden layers
python -m app.cli train -h 128 -h 64 -h 32 -h 16

# Use Adam optimizer instead of SGD
python -m app.cli train --optimizer adam

# Quiet mode (minimal output)
python -m app.cli train --quiet

# Show dataset info
python -m app.cli info

# Evaluate a saved model
python -m app.cli evaluate --model-path app/models/saved/model.pth
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--epochs` | `-e` | Number of training epochs | 500 |
| `--lr` | `-l` | Learning rate | 0.001 |
| `--momentum` | `-m` | SGD momentum | 0.9 |
| `--batch-size` | `-b` | Training batch size | 16 |
| `--hidden` | `-h` | Hidden layer sizes (multiple) | 64 32 16 |
| `--dropout` | `-d` | Dropout rate | 0.3 |
| `--weight-decay` | `-w` | L2 regularization | 1e-4 |
| `--early-stopping` | | Early stopping patience | 30 |
| `--seed` | `-s` | Random seed | 42 |
| `--optimizer` | `-o` | Optimizer (sgd/adam) | sgd |
| `--no-save` | | Don't save model/outputs | False |
| `--quiet` | `-q` | Suppress detailed output | False |

### Output Files

After running, check the `app/outputs/` directory for:
- `training_curve.png` - Loss and accuracy over epochs
- `confusion_matrix.png` - Model predictions vs actual labels
- `metrics_comparison.png` - Bar chart of evaluation metrics
- `class_distribution.png` - Test set class distribution
- `classification_report.txt` - Detailed metrics per class
- `training_config.txt` - Training hyperparameters used

## Neural Network Architecture

```
Input Layer (13 features)
    ↓
Hidden Layer 1 (64 neurons)
    → Batch Normalization
    → ReLU Activation
    → Dropout (0.3)
    ↓
Hidden Layer 2 (32 neurons)
    → Batch Normalization
    → ReLU Activation
    → Dropout (0.3)
    ↓
Hidden Layer 3 (16 neurons)
    → Batch Normalization
    → ReLU Activation
    → Dropout (0.3)
    ↓
Output Layer (5 classes)
    → Softmax (via CrossEntropyLoss)
```

### Training Features

- **Optimizer:** SGD with momentum (or Adam)
- **Loss Function:** CrossEntropyLoss
- **Learning Rate Scheduler:** ReduceLROnPlateau
- **Early Stopping:** Prevents overfitting
- **Weight Initialization:** Xavier Uniform
- **Regularization:** Dropout + L2 Weight Decay

## Programmatic Usage

```python
from app.main import run_training_pipeline

# Run with custom parameters
results = run_training_pipeline(
    epochs=200,
    lr=0.001,
    hidden_sizes=[128, 64, 32],
    dropout=0.4,
    seed=123
)

print(f"Test Accuracy: {results['accuracy']:.4f}")
```

## Author

CN6005 Artificial Intelligence Coursework 2025-26

# CN6005 - Artificial Intelligence Coursework

Neural Network for Heart Disease Classification

## Overview

Multi-layer Perceptron (MLP) classifier for predicting heart disease severity using the Cleveland Heart Disease dataset from UCI ML Repository.

- **Samples:** 303
- **Features:** 13 clinical attributes
- **Classes:** 5 (0 = no disease, 1-4 = increasing severity)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive CLI

```bash
python -m app.cli
```

Use arrow keys to navigate the menu and select options.

### Quick Training

```bash
python -m app.main
```

## Project Structure

```
app/
  cli.py              # Interactive CLI
  main.py             # Main entry point
  config/             # Configuration
  data/               # Data preprocessing
  models/             # Neural network
  training/           # Training & evaluation
  visualization/      # Plots
  lib/                # Utilities
  tests/              # Unit tests
  outputs/            # Generated results
docs/                 # Documentation
```

## Neural Network

```
Input (13) -> Hidden (64) -> Hidden (32) -> Hidden (16) -> Output (5)
```

Each hidden layer includes:
- Batch Normalization
- ReLU Activation
- Dropout (0.3)

## Output Files

After training, check `app/outputs/`:
- `training_curve.png` - Loss/accuracy curves
- `confusion_matrix.png` - Predictions vs actual
- `classification_report.txt` - Detailed metrics

## Requirements

- Python 3.10+
- PyTorch, pandas, numpy, scikit-learn
- matplotlib, seaborn, rich, InquirerPy

## Author

CN6005 Artificial Intelligence Coursework 2025-26

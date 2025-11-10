# CIFAR-10 CNN (TensorFlow/Keras)

An end-to-end, portfolio-ready deep learning project that trains a Convolutional Neural Network (CNN)
to classify images from the **CIFAR-10** dataset.

## ✨ Features
- Clean repo structure with modular code (`src/`)
- Config-driven training via `configs/config.yaml`
- Training, evaluation, and inference scripts
- Saved model + metrics, confusion matrix, and sample predictions in `artifacts/`
- Unit test and GitHub Actions CI workflow
- MIT License

## 🚀 Quickstart
```bash
# 1) Create virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train
python src/train.py --config configs/config.yaml

# 4) Evaluate (confusion matrix, classification report)
python src/evaluate.py --model artifacts/model/cnn_cifar10.h5

# 5) Predict on your own image
python src/infer.py --model artifacts/model/cnn_cifar10.h5 --image path/to/image.jpg
```

## 🗂️ Project Structure
```
cifar10-cnn/
├── artifacts/                # Saved model, plots, reports (auto-created)
├── configs/
│   └── config.yaml
├── notebooks/                # Optional notebooks (EDA, experiments)
├── src/
│   ├── data.py               # Dataset loading & preprocessing
│   ├── model.py              # CNN architecture
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation & confusion matrix
│   ├── infer.py              # Inference on custom image
│   └── utils.py              # Plotting, seeding, helpers
├── tests/
│   └── test_model_shape.py   # Simple unit test
├── scripts/
│   └── train.sh              # Example helper script
├── .github/workflows/
│   └── python-app.yml        # CI: install + tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 📊 Outputs
- `artifacts/model/cnn_cifar10.h5` — Trained model (Keras H5)
- `artifacts/plots/history.png` — Training/validation curves
- `artifacts/plots/confusion_matrix.png` — Confusion matrix
- `artifacts/reports/classification_report.txt` — Precision/recall/F1

## 🧪 Run Tests
```bash
pytest -q
```

## 🧰 Tech
- TensorFlow/Keras, NumPy, Matplotlib, scikit-learn, Pillow, PyYAML

## 📝 Notes
- The CIFAR-10 dataset (60,000 32×32 color images across 10 classes) is downloaded automatically by Keras on first run.
- Typical accuracy for this model is **70–80%** depending on epochs/regularization.
```


# 🧠 MNIST Digit Classifier

A simple CNN-based digit recognizer trained on the MNIST dataset.

## 📂 Files
- `train_model.py`: Model training with data augmentation
- `pred.py`: Predict custom hand-drawn digits (PNG)
- `model.keras`: Saved model
- `pixil-frame-0.png`: Sample input image

## 🚀 Usage

```bash
pip install -r requirements.txt
python train_model.py
python pred.py
```

> Place your 28x28 PNG digit as `pixil-frame-0.png`

## 🛠️ Requirements
- tensorflow
- opencv-python
- matplotlib
- numpy

## 📄 License
MIT

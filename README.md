# DeepFake Audio Detection using VGG16

This project implements a deep learning approach to detect tampered (spoofed) audio using Mel-spectrogram representations and the VGG16 convolutional neural network architecture. It is trained on the CMFD dataset and tested for generalization on a diverse in-the-wild dataset.

---

## 🔧 Requirements

Install the following dependencies before running the code:

```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn tensorflow
```

Ensure you're using:
- Python 3.7+
- TensorFlow 2.x
- A system with sufficient memory/GPU is recommended for model training.

---

## 📁 Dataset Sources

### Training Dataset (CMFD):
This dataset contains both English and Chinese audio files, categorized into tampered and untampered audio.

🔗 **[Download CMFD Dataset](<https://github.com/WuQinfang/CMFD>)**

### Generalization Test Dataset (In-the-Wild):
Includes real and artificially generated audio from multiple synthesis models like RawNet2, RawGAT-ST, and PC-DARTS.

🔗 **[Download In-the-Wild Dataset](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)**

---

## 📄 References

The model and experimental setup were inspired by the following papers:

- STATNet: https://ieeexplore.ieee.org/document/10007949
- VGG: https://arxiv.org/abs/1907.12908>
- TSSDNet: https://arxiv.org/abs/2106.06341

---

## 🚀 Project Overview

### Preprocessing
- Converts `.wav` files into Mel-spectrograms of size `(128, 128)` or resized to `(224, 224)` for compatibility with VGG16.
- Normalizes and pads audio as needed for uniform input.

### Model
- VGG16 CNN architecture (ImageNet pretrained, fine-tuned on spectrograms).
- Custom classification head with `Dense`, `Dropout`, and `Softmax` layers.

### Evaluation
- Metrics: Accuracy, AUC-ROC, Confusion Matrix.
- Tested on both in-domain (CMFD) and out-of-domain (In-the-Wild) data.

---

## 📊 Results Summary

| Dataset        | Accuracy | AUC-ROC |
|----------------|----------|---------|
| CMFD           | 62.75%   | 0.693   |
| In-the-Wild    | 52.3%    | 0.618   |

---

## 📈 Future Improvements

- Incorporating LSTM for temporal modeling
- Data augmentation to improve robustness
- Domain adaptation for better generalization

---

## 🧠 Author

- Your Name  
- [LinkedIn/GitHub/Website (Optional)]


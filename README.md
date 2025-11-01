# 🖼️ Image Classification on Raspberry Pi 3B

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-3B-red.svg)](https://www.raspberrypi.org/)

A complete end-to-end machine learning project that trains an image classification model, quantizes it for edge deployment, and runs it on Raspberry Pi 3B with a user-friendly GUI.

![Project Demo](https://via.placeholder.com/800x400/2c3e50/ecf0f1?text=Add+Your+Demo+Screenshot+Here)

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Development Environment](#development-environment-training)
  - [Raspberry Pi Setup](#raspberry-pi-setup-deployment)
- [Usage](#-usage)
  - [Training the Model](#1-training-the-model)
  - [Quantizing the Model](#2-quantizing-the-model)
  - [Deploying on Raspberry Pi](#3-deploying-on-raspberry-pi)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Customization](#-customization)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## ✨ Features

- **🎯 Complete ML Pipeline**: Training → Quantization → Deployment
- **📊 CIFAR-10 Dataset**: Fetched automatically via TensorFlow API
- **⚡ Model Quantization**: INT8 quantization for 4x size reduction and 3x speed improvement
- **🖥️ User-Friendly GUI**: Beautiful Tkinter interface for Raspberry Pi
- **📷 Camera Support**: Optional Pi Camera integration
- **🔧 Optimized for Edge**: Specifically designed for Raspberry Pi 3B constraints
- **📈 Training Visualization**: Automatic plotting of training metrics
- **🚀 Fast Inference**: 200-400ms inference time on Raspberry Pi 3B

## 📁 Project Structure

```
image-classification-raspberry-pi/
│
├── train_model.py                    # Model training script
├── quantize_model.py                 # Model quantization script
├── raspi_classifier_gui.py           # Raspberry Pi GUI application
├── requirements.txt                  # Python dependencies for training
├── requirements_pi.txt               # Dependencies for Raspberry Pi
├── README.md                         # This file
├── LICENSE                           # MIT License
│
├── .gitignore                        # Git ignore file
│
├── docs/                             # Documentation
│   ├── TRAINING_GUIDE.md            # Detailed training guide
│   ├── DEPLOYMENT_GUIDE.md          # Deployment instructions
│   └── TROUBLESHOOTING.md           # Common issues and solutions
│
├── examples/                         # Example images and outputs
│   ├── sample_images/               # Sample test images
│   └── screenshots/                 # GUI screenshots
│
└── models/                           # Model files (gitignored)
    ├── image_classifier.h5          # Trained model (not uploaded)
    ├── image_classifier_quantized.tflite  # Quantized model (not uploaded)
    └── class_names.txt              # Class labels
```

## 🔧 Prerequisites

### Development Environment (Training)
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- GPU optional (speeds up training significantly)
- ~2GB free disk space

### Raspberry Pi (Deployment)
- Raspberry Pi 3B, 3B+, or 4
- Raspbian OS (Buster or newer)
- 4GB+ microSD card
- Optional: Pi Camera Module v1 or v2
- Display, keyboard, and mouse

## 🚀 Installation

### Development Environment (Training)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/image-classification-raspberry-pi.git
cd image-classification-raspberry-pi
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Raspberry Pi Setup (Deployment)

1. **Update system**
```bash
sudo apt-get update
sudo apt-get upgrade
```

2. **Install system dependencies**
```bash
sudo apt-get install -y \
    python3-numpy \
    python3-pil \
    python3-pil.imagetk \
    python3-tk \
    libopenblas0 \
    libatlas-base-dev
```

3. **Install TFLite runtime**
```bash
pip3 install --user --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

4. **Clone the repository**
```bash
git clone https://github.com/yourusername/image-classification-raspberry-pi.git
cd image-classification-raspberry-pi
```

## 📖 Usage

### 1. Training the Model

Train a lightweight CNN on the CIFAR-10 dataset:

```bash
python train_model.py
```

**What it does:**
- Downloads CIFAR-10 dataset (60,000 images) via TensorFlow API
- Trains a CNN optimized for Raspberry Pi
- Applies data augmentation
- Saves trained model as `image_classifier.h5`
- Generates training visualization plots

**Training time:** ~20-30 minutes (CPU) / ~5-10 minutes (GPU)

**Output files:**
- `image_classifier.h5` - Trained model (~4MB)
- `class_names.txt` - List of class labels
- `training_history.png` - Training metrics visualization

### 2. Quantizing the Model

Convert and quantize the model for Raspberry Pi deployment:

```bash
python quantize_model.py
```

**What it does:**
- Loads the trained Keras model
- Converts to TensorFlow Lite format
- Applies INT8 quantization
- Tests accuracy and benchmarks speed
- Saves quantized model

**Output:**
- `image_classifier_quantized.tflite` - Quantized model (~300KB)

**Expected results:**
- 4x smaller model size
- 3x faster inference
- ~5% accuracy trade-off

### 3. Deploying on Raspberry Pi

Transfer the necessary files to your Raspberry Pi:

```bash
# From your development machine
scp image_classifier_quantized.tflite pi@raspberrypi.local:~/image-classification-raspberry-pi/
scp class_names.txt pi@raspberrypi.local:~/image-classification-raspberry-pi/
```

Run the GUI application on Raspberry Pi:

```bash
# On Raspberry Pi
cd ~/image-classification-raspberry-pi
python3 raspi_classifier_gui.py
```

**Using the GUI:**
1. Click **"Load Image"** to select an image file
2. Click **"Classify Image"** to run prediction
3. View top 3 predictions with confidence scores
4. Optional: Use **"Capture Photo"** with Pi Camera

## 🏗️ Model Architecture

The model is a lightweight CNN specifically designed for edge devices:

```
Input (32x32x3)
    ↓
Data Augmentation
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.2)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Flatten → Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(10, softmax)
```

**Key features:**
- Small input size (32x32) for fast inference
- Batch normalization for stable training
- Dropout for regularization
- ~200K parameters

## 📊 Performance

### Model Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | ~75-80% |
| Validation Accuracy | ~70-75% |
| Test Accuracy (Quantized) | ~65-70% |
| Model Size (Original) | ~4MB |
| Model Size (Quantized) | ~300KB |

### Inference Performance

| Device | Inference Time | FPS |
|--------|---------------|-----|
| Raspberry Pi 3B | 200-400ms | 2.5-5 FPS |
| Raspberry Pi 4 | 80-150ms | 6.5-12 FPS |
| Desktop CPU | 10-20ms | 50-100 FPS |
| Desktop GPU | 2-5ms | 200-500 FPS |

### CIFAR-10 Classes

The model classifies images into 10 categories:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

## 🎨 Customization

### Using a Custom Dataset

Replace CIFAR-10 with your own dataset by modifying `train_model.py`:

```python
def load_custom_dataset(api_url):
    """Fetch dataset from your API"""
    response = requests.get(api_url)
    data = response.json()
    
    images = []
    labels = []
    
    for item in data['items']:
        # Download image
        img_response = requests.get(item['image_url'])
        img = Image.open(BytesIO(img_response.content))
        img = img.resize((32, 32))
        
        images.append(np.array(img))
        labels.append(item['label'])
    
    return np.array(images), np.array(labels)

# Replace the dataset loading section
train_data, test_data = load_custom_dataset('https://your-api.com/dataset')
```

### Adjusting Model Architecture

Modify the `build_model()` function in `train_model.py`:

```python
# For faster inference (lower accuracy)
- layers.Conv2D(128, (3, 3), ...)
+ layers.Conv2D(64, (3, 3), ...)

# For higher accuracy (slower inference)
+ layers.Conv2D(256, (3, 3), ...)
```

### Changing Input Size

```python
# In train_model.py
IMG_SIZE = 28  # Smaller = faster but less accurate
# or
IMG_SIZE = 64  # Larger = more accurate but slower
```

## 🔧 Troubleshooting

### Common Issues

**1. NumPy compatibility error on Raspberry Pi**
```bash
# Downgrade to NumPy 1.x
pip3 install "numpy<2.0"
```

**2. GUI not displaying**
```bash
# Install Tkinter
sudo apt-get install python3-tk python3-pil.imagetk
```

**3. Slow inference (>1 second)**
- Ensure you're using the quantized `.tflite` model
- Close background applications
- Check CPU temperature: `vcgencmd measure_temp`

**4. "Model not found" error**
```bash
# Make sure files are in the same directory
ls -la image_classifier_quantized.tflite class_names.txt
```

**5. Low accuracy on real-world images**
- CIFAR-10 is low resolution (32x32)
- Consider using transfer learning with MobileNetV2
- Train on a dataset similar to your use case

### Performance Optimization

**Faster Inference:**
```bash
# Enable performance governor (Raspberry Pi)
sudo cpufreq-set -g performance
```

**Lower Memory Usage:**
```python
# In raspi_classifier_gui.py, reduce batch size
BATCH_SIZE = 1
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** - For the excellent TensorFlow Lite framework
- **CIFAR-10 Dataset** - Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **Raspberry Pi Foundation** - For making edge computing accessible
- **Open Source Community** - For inspiration and support

## 📞 Contact

**Your Name** - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/image-classification-raspberry-pi](https://github.com/yourusername/image-classification-raspberry-pi)

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star ⭐

---

## 📚 Additional Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Model Optimization Techniques](https://www.tensorflow.org/model_optimization)

---

**Built with ❤️ for edge AI enthusiasts**

# 1D CNN Android Malware Detection

A one-dimensional Convolutional Neural Network (CNN) implementation for detecting malware in Android APK files using raw byte sequences. This project is based on the architecture described in the CSPA 2018 paper.

## Overview

This project implements a lightweight malware detection system that analyzes the raw bytes of APK files to classify them as either benign or malicious. The model uses a shallow 1-D CNN architecture optimized for byte-level pattern recognition.

## Project Structure

```
1D CNN/
├── malware_checker.py                              # Main inference script
├── train_model.py                                  # Training script
├── trained_model.pth                               # Pre-trained model weights
├── One_Dimensional_CNN_Android_Malware_Detection.md # Detailed documentation
├── README.md                                       # This file
└── Samples/
    ├── benign/                                     # Benign APK samples
    └── malware/                                    # Malware APK samples
```

## Files Description

### `malware_checker.py`

The inference script that loads a trained model and scans APK files for malware.

**ByteCNN Architecture:**

- Embedding layer: 256 → 8 dimensions
- Conv1d layer 1: 8 → 32 filters (kernel=5)
- Batch normalization + ReLU
- Conv1d layer 2: 32 → 32 filters (kernel=5)
- Max pooling (kernel=5, stride=5)
- Conv1d layer 3: 32 → 32 filters (kernel=5)
- Batch normalization + ReLU
- Conv1d layer 4: 32 → 32 filters (kernel=5)
- Max pooling (kernel=5, stride=5)
- Global average pooling
- Fully connected: 32 → 2 (benign/malware)

### `train_model.py`

Training script that trains the ByteCNN model on labeled APK samples.

**Training Configuration:**

- Epochs: 50
- Batch size: 8
- Optimizer: Adam (learning rate: 0.001)
- Train/validation split: 80/20
- Input segment: Last 1024 bytes of each APK

### `trained_model.pth`

Pre-trained model weights saved in PyTorch format. Achieves 95% accuracy on test set.

## Performance

- **Best Validation Accuracy:** 87.5%
- **Test Set Performance:** 95% (38/40 correct)
  - Benign samples: 90% accuracy (18/20)
  - Malware samples: 100% accuracy (20/20)

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy

### Setup

1. Install dependencies using pip:

```bash
pip install torch numpy
```

Or use the provided virtual environment:

```bash
source /path/to/ml_env/bin/activate
```

## Usage

### Scanning APK Files

#### Using Pre-trained Model

Scan a folder of APK files:

```bash
python malware_checker.py --samples ./Samples/benign --model-path ./trained_model.pth
```

#### Command-Line Options

```
--samples PATH           Directory containing APK files to scan (default: 1D CNN/Samples)
--model-path PATH        Path to trained model weights (.pth file)
--bytes N                Number of bytes to read from each APK (default: 1024)
--beginning              Read from beginning instead of end (default: reads from end)
--threshold FLOAT        Malware probability threshold for flagging (default: 0.5)
```

#### Output Format

The script outputs CSV format with columns:

```
file,goodware_prob,malware_prob,label
```

Example output:

```
122EEF66DC0B313FA50C27C916C77AA39EA9F2322D4724A60F81909A365695AE.apk,0.4322,0.5678,MALWARE
E02FEF06A42DE9D19325E8F7ED4D6477945B575ED26926486372F73912CE452F.apk,0.5650,0.4350,BENIGN
```

### Training a New Model

Train the model from scratch using your labeled samples:

```bash
python train_model.py
```

**Ensure your sample directory structure is:**

```
Samples/
├── benign/      # Contains benign APK files
└── malware/     # Contains malware APK files
```

The training script will:

1. Load all APK files from benign/ and malware/ directories
2. Split into 80% training and 20% validation sets
3. Train for 50 epochs with Adam optimizer
4. Save the best model to `trained_model.pth`

## Example Workflow

1. **Organize your APK samples:**

```bash
mkdir -p Samples/benign Samples/malware
# Copy benign APKs to Samples/benign/
# Copy malware APKs to Samples/malware/
```

2. **Train the model:**

```bash
python train_model.py
```

3. **Test on benign samples:**

```bash
python malware_checker.py --samples ./Samples/benign --model-path ./trained_model.pth
```

4. **Test on malware samples:**

```bash
python malware_checker.py --samples ./Samples/malware --model-path ./trained_model.pth
```

## Model Architecture Details

The ByteCNN model follows a shallow architecture optimized for computational efficiency:

- **Input:** Raw APK bytes (1024 bytes by default)
- **Embedding:** Maps byte values (0-255) to 8-dimensional vectors
- **Feature Extraction:** Two blocks of [Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → ReLU → MaxPool]
- **Classification:** Global average pooling followed by fully connected layer

This architecture captures local byte patterns while remaining lightweight for deployment.

## References

Based on the paper:

- **Title:** One-Dimensional CNN for Raw Android Malware Detection
- **Conference:** CSPA 2018
- **Focus:** Binary classification of APKs using byte-level patterns

## Key Features

✓ Lightweight model (fast inference)  
✓ Works with raw APK bytes (no feature engineering required)  
✓ High accuracy on malware detection (100% on test malware)  
✓ Simple command-line interface  
✓ CSV output for easy analysis  
✓ Customizable byte segment selection  
✓ Adjustable classification threshold

## Notes

- The model uses CPU by default but will automatically use CUDA if available
- Input APKs shorter than 1024 bytes are zero-padded
- The threshold can be adjusted to balance false positives vs. false negatives
- For production use, consider training on larger datasets for improved generalization

## License

This project is for educational purposes.

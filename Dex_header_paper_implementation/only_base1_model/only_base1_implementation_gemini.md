# Base Model 1 (MLP(H)) - Detailed Implementation Plan

## Phase 1: Workspace & Environment Setup
* **Dependencies**: `torch`, `torchvision`, `numpy`, `scikit-learn`, `tqdm`, `zipfile`.
* **Architecture**: A modular pipeline separating dataset preprocessing, data loading, model definition, and training loops.

## Phase 2: APK Preprocessing & DEX Feature Extraction
*Since the dataset consists of 50,000 APKs on a remote machine, doing on-the-fly extraction during training will bottleneck the GPU. We will include a dedicated pre-processing script.*
1. **APK Unpacking**: A script (`preprocess_apks.py`) that iterates over the 50,000 APKs, unzips them in memory using Python's `zipfile`, and extracts `classes.dex`.
2. **DEX Header Parsing**: 
   * Verify the magic number (`dex\n\035\0`).
   * Parse the `DexHeader` bytes to extract structural sizes and offsets (string IDs, type IDs, proto IDs, field IDs, method IDs, class defs, file size, signature, etc.).
3. **Feature Normalization & Saving**:
   * Encode into hexadecimal equivalents and normalize (e.g., Min-Max scaling).
   * Save the extracted 1D tensors to a fast-read format (like an aggregate `.npy` file, `.pt` PyTorch tensor file, or HDF5 database) along with their labels. This ensures the PyTorch DataLoader only reads ready-to-use tensors, maximizing training speed.

## Phase 3: PyTorch Dataset & DataLoader
1. **Dataset Class (`DexDataset`)**:
   * Loads the pre-processed tensor structure (not the raw APKs).
2. **DataLoader**:
   * **Batch Size**: 16 (as per paper).
   * Shuffled for the training set, sequential for verification/validation.

## Phase 4: Model Architecture (Base Model 1)
* **Input Layer**: Dynamically sized to the 1D header feature tensor length.
* **Hidden Block 1**: `nn.Linear` $\rightarrow$ `nn.BatchNorm1d` $\rightarrow$ `nn.ReLU`.
* **Hidden Block 2**: `nn.Linear` $\rightarrow$ `nn.BatchNorm1d` $\rightarrow$ `nn.ReLU`.
* **Output Layer**: `nn.Linear` to 1 output unit $\rightarrow$ `nn.Sigmoid` (Binary Classification: Malware vs. Benign).

## Phase 5: Training Loop with Resiliency & Progression
1. **Hyperparameters**:
   * **Loss**: `nn.BCELoss`
   * **Optimizer**: SGD (Learning Rate: 0.005, Momentum: 0.9)
   * **LR Decay**: Multiplicative factor of 0.5 via matching PyTorch scheduler.
2. **Real-time Metrics**:
   * Wrap the DataLoader iterates with `tqdm` to show progress bars, running loss, and batch speed out to the console.
3. **Checkpointing & Resume (Power Outage Resiliency)**:
   * **Save**: At the end of every epoch (or periodically during large epochs), save a checkpoint dictionary containing: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, and `current_loss`.
   * **Load**: Upon running the script, explicitly check if a `latest_checkpoint.pth` exists. If so, load the weights, optimizer states, and epoch iteration to seamlessly resume exactly where it dropped off.

## Phase 6: Evaluation
* During validation, calculate Accuracy, F1-Score, and AUC using `scikit-learn.metrics`.

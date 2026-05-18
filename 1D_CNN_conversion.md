# Implementation Plan: Integrating 1D-CNN into VigiDroid

## Objective
Convert the trained PyTorch 1D-CNN model to ONNX format and integrate it into `ScanService.java` alongside the existing XGBoost inference pipeline. 

Based on `1dcnn/src/model/dataset.py` and `1dcnn/src/main.py`, the 1D-CNN operates on the **last 1024 bytes** of the `.apk` file (using `FROM_END=True`), represented as long integers (byte values `0-255`).

---

## Phase 1: PyTorch `.pth` to `.onnx` Conversion

1. **Create an Export Script (`1dcnn/export_onnx.py`)**:
   - Initialize the `ByteCNN(embed_dim=8, num_classes=2)` architecture.
   - Load weights from `1dcnn/trained_model.pth`.
   - Set the model to inference mode (`model.eval()`).
   - Create a dummy input tensor of shape `(1, 1024)` with `dtype=torch.long`.
   - Use `torch.onnx.export` to save the model as `1dcnn_model.onnx`.

2. **Migrate the Model to Android**:
   - Copy `1dcnn_model.onnx` to `vigidroid/app/src/main/assets/`.

---

## Phase 2: Modifying `ScanService.java`

We will modify `ScanService.java` to load and run *both* models sequentially when an APK is scanned.

1. **Initialization**:
   - Add new class fields: `OrtSession ortSessionCnn;`.
   - Update `initOnnxModel()` to extract and load `1dcnn_model.onnx` into `ortSessionCnn`, similar to how `mh1m_2500_rp_XGBoost.onnx` is currently handled.

2. **Data Preparation for 1D-CNN**:
   - Create a new helper method `extractLastBytes(File apkFile, int byteLength)`.
   - It will open the file in `RandomAccessFile` (or similar), jump backward to the last 1024 bytes (`length - 1024`), and read the byte array.
   - Convert the Java `byte[]` to a `long[]` because the PyTorch model expects `torch.long`. Ensure negative bytes (implied by Java's signed bytes) are mapped to `0-255` positively using `b & 0xFF`.

3. **Running CNN Inference**:
   - Create a new method `runCnnInference(long[] inputVector)`.
   - Wrap the `long[]` array into a `LongBuffer` and feed it into `OnnxTensor.createTensor(...)` with shape `[1, 1024]`.
   - Execute `ortSessionCnn.run()`.
   - The model outputs logits of shape `[1, 2]`. We will extract these values and standardise them (e.g., probability of Malware via Softmax). 

4. **Updating the Pipeline & Logging**:
   - In `onHandleWork()`, right beside the XGBoost pipeline, execute the CNN pipeline:
     ```java
     // CNN Pipeline
     long cnnParseStart = SystemClock.elapsedRealtimeNanos();
     long[] cnnInput = extractLastBytes(apk, 1024);
     long cnnParseEnd = SystemClock.elapsedRealtimeNanos();
     
     long cnnInferStart = SystemClock.elapsedRealtimeNanos();
     float cnnScore = runCnnInference(cnnInput);
     long cnnInferEnd = SystemClock.elapsedRealtimeNanos();
     ```
   - Update the logging string (`result`) to output both `xgb_score=...` and `cnn_score=...`, as well as parsing/inference times for both algorithms.

5. **Resource Cleanup**:
   - Ensure `ortSessionCnn.close()` is called inside `onDestroy()`.

---

**Please review this plan. Provide your confirmation, and I will begin the implementation by creating the ONNX conversion script and editing `ScanService.java`!**
package com.msh.vigidroid;

/** Shared ONNX output parsing for scalar malware probabilities. */
final class OnnxProbabilityReader {

  private OnnxProbabilityReader() {}

  static float readScalar(Object value) {
    if (value instanceof float[][]) {
      return ((float[][]) value)[0][0];
    }
    if (value instanceof float[]) {
      return ((float[]) value)[0];
    }
    if (value instanceof double[][]) {
      return (float) ((double[][]) value)[0][0];
    }
    if (value instanceof double[]) {
      return (float) ((double[]) value)[0];
    }
    throw new IllegalStateException("Unexpected ONNX output type: " + value.getClass().getName());
  }
}

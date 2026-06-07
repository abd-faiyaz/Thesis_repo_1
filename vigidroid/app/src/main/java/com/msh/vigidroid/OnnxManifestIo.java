package com.msh.vigidroid;

import org.json.JSONArray;
import org.json.JSONObject;

/** Resolve ONNX input/output tensor names from export_manifest.json. */
final class OnnxManifestIo {

  private OnnxManifestIo() {}

  static String inputName(JSONObject manifest, String fallback) throws Exception {
    JSONArray inputs = manifest.optJSONArray("inputs");
    if (inputs != null && inputs.length() > 0) {
      return inputs.getJSONObject(0).optString("name", fallback);
    }
    JSONObject onnxCheck = manifest.optJSONObject("onnx_runtime_check");
    if (onnxCheck != null) {
      JSONArray inputNames = onnxCheck.optJSONArray("input_names");
      if (inputNames != null && inputNames.length() > 0) {
        return inputNames.optString(0, fallback);
      }
      return onnxCheck.optString("input_name", fallback);
    }
    return fallback;
  }

  static String outputName(JSONObject manifest, String fallback) throws Exception {
    JSONArray outputs = manifest.optJSONArray("outputs");
    if (outputs != null && outputs.length() > 0) {
      return outputs.getJSONObject(0).optString("name", fallback);
    }
    JSONObject onnxCheck = manifest.optJSONObject("onnx_runtime_check");
    if (onnxCheck != null) {
      return onnxCheck.optString("output_name", fallback);
    }
    return fallback;
  }
}

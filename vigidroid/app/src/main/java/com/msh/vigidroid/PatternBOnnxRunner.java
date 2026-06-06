package com.msh.vigidroid;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.Closeable;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/** ONNX inference for Pattern B pattern_b_dual_branch (header + manifest BoW → fused malware probability). */
public final class PatternBOnnxRunner implements Closeable {

  private static final String TAG = "PatternBOnnxRunner";
  public static final String MODEL_ID = "pattern_b_dual_branch";
  public static final String DOMAIN = "dex_header_manifest_dual";
  private static final String MODEL_ASSET = "models/pattern_b_dual_branch/model.onnx";
  private static final String MANIFEST_ASSET = "models/pattern_b_dual_branch/export_manifest.json";
  private static final String CACHE_FILE = "pattern_b_dual_branch_model.onnx";

  private final OrtEnvironment environment;
  private final OrtSession session;
  private final String headerInputName;
  private final String bowInputName;

  private PatternBOnnxRunner(
      OrtEnvironment environment,
      OrtSession session,
      String headerInputName,
      String bowInputName) {
    this.environment = environment;
    this.session = session;
    this.headerInputName = headerInputName;
    this.bowInputName = bowInputName;
  }

  public static PatternBOnnxRunner create(Context context, OrtEnvironment sharedEnv) throws Exception {
    String manifestJson = ModelAssetHelper.readAssetText(context, MANIFEST_ASSET);
    JSONObject manifest = new JSONObject(manifestJson);
    JSONObject onnxCheck = manifest.optJSONObject("onnx_runtime_check");
    String headerName = "header";
    String bowName = "bow";
    if (onnxCheck != null) {
      JSONArray inputNames = onnxCheck.optJSONArray("input_names");
      if (inputNames != null && inputNames.length() >= 2) {
        headerName = inputNames.getString(0);
        bowName = inputNames.getString(1);
      }
    }
    java.io.File modelFile = ModelAssetHelper.copyAssetToCache(context, MODEL_ASSET, CACHE_FILE);
    OrtSession.SessionOptions options = new OrtSession.SessionOptions();
    OrtSession session = sharedEnv.createSession(modelFile.getAbsolutePath(), options);
    Log.i(TAG, "Loaded Pattern B ONNX from " + modelFile.getAbsolutePath());
    return new PatternBOnnxRunner(sharedEnv, session, headerName, bowName);
  }

  public float predict(float[] header, float[] bow) throws OrtException {
    if (header.length != DexHeaderFeatureExtractor.FEATURE_DIM) {
      throw new IllegalArgumentException(
          "Expected header dim " + DexHeaderFeatureExtractor.FEATURE_DIM + ", got " + header.length);
    }
    if (bow.length != ManifestBowExtractor.BOW_DIM) {
      throw new IllegalArgumentException(
          "Expected bow dim " + ManifestBowExtractor.BOW_DIM + ", got " + bow.length);
    }

    long[] headerShape = new long[] {1, header.length};
    long[] bowShape = new long[] {1, bow.length};
    try (OnnxTensor headerTensor =
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(header), headerShape);
        OnnxTensor bowTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(bow), bowShape)) {
      Map<String, OnnxTensor> inputs = new HashMap<>();
      inputs.put(headerInputName, headerTensor);
      inputs.put(bowInputName, bowTensor);
      try (OrtSession.Result result = session.run(inputs)) {
        return readProbability(result.get(0).getValue());
      }
    }
  }

  private static float readProbability(Object value) {
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

  @Override
  public void close() {
    if (session != null) {
      try {
        session.close();
      } catch (Exception ignored) {
      }
    }
  }
}

package com.msh.vigidroid;

import android.content.Context;
import android.util.Log;

import org.json.JSONObject;

import java.io.Closeable;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * ONNX inference for LinRegDroid (173-d permissions → clamped malware_probability).
 * Post-processing: clamp(linear_score, 0, 1) is baked into the export; decision uses
 * LinRegDroid1 threshold from thresholds.json (default 0.5).
 */
public final class LinRegDroidOnnxRunner implements Closeable {

  private static final String TAG = "LinRegDroidOnnxRunner";
  public static final String MODEL_ID = "linregdroid_permission";
  public static final String DOMAIN = "manifest_permissions";
  private static final String MODEL_ASSET = "models/linregdroid_permission/model.onnx";
  private static final String MANIFEST_ASSET = "models/linregdroid_permission/export_manifest.json";
  private static final String THRESHOLDS_ASSET = "models/linregdroid_permission/thresholds.json";
  private static final String CACHE_FILE = "linregdroid_permission_model.onnx";

  private final OrtEnvironment environment;
  private final OrtSession session;
  private final String inputName;
  private final ModelThresholds thresholds;

  private LinRegDroidOnnxRunner(
      OrtEnvironment environment,
      OrtSession session,
      String inputName,
      ModelThresholds thresholds) {
    this.environment = environment;
    this.session = session;
    this.inputName = inputName;
    this.thresholds = thresholds;
  }

  public static LinRegDroidOnnxRunner create(Context context, OrtEnvironment sharedEnv)
      throws Exception {
    String manifestJson = ModelAssetHelper.readAssetText(context, MANIFEST_ASSET);
    JSONObject manifest = new JSONObject(manifestJson);
    String inputName = OnnxManifestIo.inputName(manifest, "permissions");
    ModelThresholds thresholds = ModelThresholds.fromAsset(context, THRESHOLDS_ASSET);
    java.io.File modelFile = ModelAssetHelper.copyAssetToCache(context, MODEL_ASSET, CACHE_FILE);
    OrtSession session = sharedEnv.createSession(modelFile.getAbsolutePath(), new OrtSession.SessionOptions());
    Log.i(
        TAG,
        "Loaded LinRegDroid ONNX (variant="
            + thresholds.getDecisionVariant()
            + ", threshold="
            + thresholds.getMalwareThreshold()
            + ") from "
            + modelFile.getAbsolutePath());
    return new LinRegDroidOnnxRunner(sharedEnv, session, inputName, thresholds);
  }

  public ModelThresholds getThresholds() {
    return thresholds;
  }

  /** Returns clamped malware probability in [0, 1] from ONNX output malware_probability. */
  public float predict(float[] permissions) throws OrtException {
    if (permissions.length != LinRegPermissionExtractor.FEATURE_DIM) {
      throw new IllegalArgumentException(
          "Expected "
              + LinRegPermissionExtractor.FEATURE_DIM
              + " permissions, got "
              + permissions.length);
    }
    long[] shape = new long[] {1, permissions.length};
    try (OnnxTensor tensor =
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(permissions), shape)) {
      Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
      try (OrtSession.Result result = session.run(inputs)) {
        Object value = result.get(0).getValue();
        return readProbability(value);
      }
    }
  }

  /** LinRegDroid1: malware when clamped probability >= threshold. */
  public boolean isMalware(float malwareProbability) {
    return thresholds.isMalware(malwareProbability);
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

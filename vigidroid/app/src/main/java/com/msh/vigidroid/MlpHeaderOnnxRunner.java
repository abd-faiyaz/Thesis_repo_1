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

/** ONNX inference for BM1 mlp_header (104-d Dex header → malware probability). */
public final class MlpHeaderOnnxRunner implements Closeable {

    private static final String TAG = "MlpHeaderOnnxRunner";
    public static final String MODEL_ID = "mlp_header";
    public static final String DOMAIN = "dex_header_d3";
    private static final String MODEL_ASSET = "models/mlp_header/model.onnx";
    private static final String MANIFEST_ASSET = "models/mlp_header/export_manifest.json";
    private static final String CACHE_FILE = "mlp_header_model.onnx";

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final String inputName;

    private MlpHeaderOnnxRunner(OrtEnvironment environment, OrtSession session, String inputName) {
        this.environment = environment;
        this.session = session;
        this.inputName = inputName;
    }

    public static MlpHeaderOnnxRunner create(Context context, OrtEnvironment sharedEnv) throws Exception {
        String manifestJson = ModelAssetHelper.readAssetText(context, MANIFEST_ASSET);
        JSONObject manifest = new JSONObject(manifestJson);
        JSONObject onnxCheck = manifest.optJSONObject("onnx_runtime_check");
        String inputName = onnxCheck != null
                ? onnxCheck.optString("input_name", "features")
                : "features";
        java.io.File modelFile = ModelAssetHelper.copyAssetToCache(context, MODEL_ASSET, CACHE_FILE);
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = sharedEnv.createSession(modelFile.getAbsolutePath(), options);
        Log.i(TAG, "Loaded BM1 ONNX from " + modelFile.getAbsolutePath());
        return new MlpHeaderOnnxRunner(sharedEnv, session, inputName);
    }

    public float predict(float[] features) throws OrtException {
        if (features.length != DexHeaderFeatureExtractor.FEATURE_DIM) {
            throw new IllegalArgumentException(
                    "Expected " + DexHeaderFeatureExtractor.FEATURE_DIM + " features, got "
                            + features.length);
        }
        long[] shape = new long[]{1, features.length};
        FloatBuffer buffer = FloatBuffer.wrap(features);
        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, buffer, shape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
            try (OrtSession.Result result = session.run(inputs)) {
                Object value = result.get(0).getValue();
                return readProbability(value);
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

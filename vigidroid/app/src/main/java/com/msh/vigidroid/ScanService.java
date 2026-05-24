package com.msh.vigidroid;

import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.Debug;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.JobIntentService;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import org.json.JSONArray;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ScanService extends JobIntentService {

    private static final int JOB_ID = 2001;
    private static final String TAG = "ScanService";

    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private OrtSession ortSessionCnn;
    private List<String> featureColumns = new ArrayList<>();
    private Map<String, Integer> featureIndex = new HashMap<>();


    public static void enqueueWork(Context context, Intent intent) {
        enqueueWork(context, ScanService.class, JOB_ID, intent);
    }

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            loadFeatureColumns();
            initOnnxModel();
        } catch (Exception e) {
            Log.e(TAG, "Init error", e);
            sendLog("Initialization error: " + e.getMessage(), "Error");
        }
    }

    @Override
    protected void onHandleWork(@NonNull Intent intent) {

        sendLog("ScanService started", "Running");

        boolean manual = intent.getBooleanExtra("manual_trigger", false);
        String trigger = manual ? "manual" : "download";
        if (manual) sendLog("Triggered by button", null);
        else sendLog("Triggered by BroadcastReceiver", null);

        // Check storage access (MANAGE_EXTERNAL_STORAGE expected)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                sendLog("No MANAGE_EXTERNAL_STORAGE permission. Abort.", "Error");
                return;
            }
        }

        // Collect APK files
        File downloads = new File(Environment.getExternalStorageDirectory(), "Download");
        if (!downloads.exists()) {
            sendLog("Downloads folder not found!", "Error");
            return;
        }

        File[] files = downloads.listFiles((dir, name) -> name.toLowerCase().endsWith(".apk"));
        if (files == null || files.length == 0) {
            sendLog("No APKs found.", "Idle");
            return;
        }

        // Iterate & simulate parsing
        for (File apk : files) {
            sendLog("Processing: " + apk.getName(), "Parsing");
            String apkName = apk.getName();

            long wallStart = SystemClock.elapsedRealtimeNanos();
            long cpuStart = Debug.threadCpuTimeNanos();
            long memStart = Debug.getNativeHeapAllocatedSize();

            // parse apk
            long parseStart = SystemClock.elapsedRealtimeNanos();
            Set<String> extractedTokens = extractManifestFeatures(apk);
            long parseEnd = SystemClock.elapsedRealtimeNanos();

            // vectorize
            long vecStart = SystemClock.elapsedRealtimeNanos();
            float[] inputVector = vectorize(extractedTokens);
            long vecEnd = SystemClock.elapsedRealtimeNanos();

            // run inference
            long inferStart = SystemClock.elapsedRealtimeNanos();
            float score = runInference(inputVector);
            long inferEnd = SystemClock.elapsedRealtimeNanos();

            // 1D CNN inference pipeline
            long cnnParseStart = SystemClock.elapsedRealtimeNanos();
            long[] cnnInput = extractLastBytes(apk, 1024);
            long cnnParseEnd = SystemClock.elapsedRealtimeNanos();

            long cnnInferStart = SystemClock.elapsedRealtimeNanos();
            float cnnScore = runCnnInference(cnnInput);
            long cnnInferEnd = SystemClock.elapsedRealtimeNanos();

            long cpuEnd = Debug.threadCpuTimeNanos();
            long memEnd = Debug.getNativeHeapAllocatedSize();
            long wallEnd = SystemClock.elapsedRealtimeNanos();

            double parsingMs = (parseEnd - parseStart) / 1_000_000.0;
            double vectorMs = (vecEnd - vecStart) / 1_000_000.0;
            double inferenceMs = (inferEnd - inferStart) / 1_000_000.0;
            
            double cnnParsingMs = (cnnParseEnd - cnnParseStart) / 1_000_000.0;
            double cnnInferenceMs = (cnnInferEnd - cnnInferStart) / 1_000_000.0;
            
            double totalMs = (wallEnd - wallStart) / 1_000_000.0;
            long cpuMs = cpuEnd - cpuStart;
            long memDelta = memEnd - memStart;

            String result = String.format(Locale.US,
                    "APK=%s xgb_parse=%.2fms vec=%.2fms xgb_infer=%.2fms xgb_score=%.5f | cnn_parse=%.2fms cnn_infer=%.2fms cnn_score=%.5f | total=%.2fms cpuMs=%d memDelta=%d",
                    apkName, parsingMs, vectorMs, inferenceMs, score, cnnParsingMs, cnnInferenceMs, cnnScore, totalMs, cpuMs, memDelta);

            sendLog(result, "Idle");

            writeScanMetrics(
                    trigger,
                    apk,
                    parsingMs, vectorMs, inferenceMs, score,
                    cnnParsingMs, cnnInferenceMs, cnnScore,
                    totalMs, cpuMs / 1_000_000.0, memDelta,
                    memEnd - memStart
            );
        }

        sendLog("Scan completed.", "Idle");
    }

    private void sendLog(String log, String status) {
        Intent i = new Intent("SCAN_LOG");
        i.putExtra("log", log);
        if (status != null) i.putExtra("status", status);
        LocalBroadcastManager.getInstance(this).sendBroadcast(i);
    }

    private Set<String> extractManifestFeatures(File apkFile) {
        Set<String> features = new HashSet<>();

        try (ZipFile zip = new ZipFile(apkFile)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();

            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                String entryName = entry.getName();
                if (entryName.equalsIgnoreCase("AndroidManifest.xml")) {
//                    ZipEntry manifestEntry = zip.getEntry("AndroidManifest.xml");
//                    if (manifestEntry == null) return features;

                    InputStream is = zip.getInputStream(entry);
                    AxmlReader reader = new AxmlReader(is);
                    Set<String> rawFeatures = reader.parse();

                    for (String rawFeature : rawFeatures) {
                        if (rawFeature != null && rawFeature.startsWith("android.permission.")){
                            features.add(normalizePermission(rawFeature));
                        }
                        if (rawFeature != null && rawFeature.startsWith("android.intent.action.")){
                            features.add(normalizeIntent(rawFeature));
                        }
                    }
                }
                else if (entryName.endsWith(".dex")) {
                    InputStream is = zip.getInputStream(entry);
                    MinimalDexParser.parse(is, features::add);
                    is.close();
                }
            }
        } catch (Exception e) {
            sendLog("Manifest parse error: " + e.getMessage(), null);
        }

        return features;
    }

    private String normalizePermission(String p) {
        p = p.toLowerCase(Locale.US);
        if (p.startsWith("android.permission.")) {
            p = p.substring("android.permission.".length());
        }
        return "permissions::" + p.replace('.', '_');
    }

    private String normalizeIntent(String i) {
        i = i.toLowerCase(Locale.US);
        if (i.startsWith("android.intent.action.")) {
            i = i.substring("android.intent.action.".length());
        }
        return "intents::" + i.replace('.', '_');
    }

    // ------------------------------
    // Vectorize tokens -> float[] aligned with loaded featureColumns
    // ------------------------------
    private float[] vectorize(Collection<String> tokens) {
        int n = featureColumns.size();
        float[] vec = new float[n];
        for (String t : tokens) {
            Integer idx = featureIndex.get(t);
            if (idx != null) vec[idx] = 1.0f;
        }
        return vec;
    }

    // ------------------------------
    // ONNX Runtime: init model and run inference
    // ------------------------------
    private void initOnnxModel() throws Exception {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment();

            // Copy model from assets to a temp file, then create session from path
            File modelFile = new File(getCacheDir(), "mh1m_2500_rp_XGBoost.onnx");
            if (!modelFile.exists()) {
                try (InputStream is = getAssets().open("mh1m_2500_rp_XGBoost.onnx");
                     FileOutputStream fos = new FileOutputStream(modelFile)) {
                    byte[] buf = new byte[8192];
                    int r;
                    while ((r = is.read(buf)) != -1) {
                        fos.write(buf, 0, r);
                    }
                }
            }

            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            // You can set intra op threads, graph optimization level etc. if desired:
            // sessionOptions.setIntraOpNumThreads(2);
            ortSession = ortEnvironment.createSession(modelFile.getAbsolutePath(), sessionOptions);
            
            // Load the CNN model
            File cnnModelFile = new File(getCacheDir(), "bytecnn_basemodel_2020.onnx");
            if (!cnnModelFile.exists()) {
                try (InputStream is = getAssets().open("bytecnn_basemodel_2020.onnx");
                     FileOutputStream fos = new FileOutputStream(cnnModelFile)) {
                    byte[] buf = new byte[8192];
                    int r;
                    while ((r = is.read(buf)) != -1) {
                        fos.write(buf, 0, r);
                    }
                }
            }
            ortSessionCnn = ortEnvironment.createSession(cnnModelFile.getAbsolutePath(), sessionOptions);
            
            sendLog("Both ONNX models loaded", null);
        } catch (Exception ex) {
            Log.e(TAG, "ONNX init failed", ex);
            throw ex;
        }
    }

    private float runInference(float[] inputVector) {
        if (ortEnvironment == null || ortSession == null) {
            sendLog("ONNX model not initialized", "Error");
            return -1f;
        }

        try (OrtSession.Result result = runModel(inputVector)) {
            // Usually XGBoost ONNX outputs a tensor of shape [1, n_classes] or [1,1]
            // We assume binary classifier with single float score in first entry
            Object o = result.get(0).getValue();
            if (o instanceof float[][]) {
                float[][] out = (float[][]) o;
                return out[0][0];

            } else if (o instanceof float[]) {
                float[] out = (float[]) o;
                return out[0];

            } else if (o instanceof long[]) {
                long[] out = (long[]) o;
                return (float) out[0];   // label or score

            } else if (o instanceof long[][]) {
                long[][] out = (long[][]) o;
                return (float) out[0][0];

            } else {
                sendLog("Unexpected ONNX output type: " + o.getClass().getName(), null);
                return -1f;
            }
        } catch (Exception e) {
            Log.e(TAG, "Inference failed", e);
            sendLog("Inference error: " + e.getMessage(), "Error");
            return -1f;
        }
    }

    private OrtSession.Result runModel(float[] inputVector) throws OrtException {

        long[] shape = new long[]{1, inputVector.length};
        FloatBuffer fb = FloatBuffer.wrap(inputVector);

        String inputName = ortSession.getInputNames().iterator().next();

        try (OnnxTensor tensor = OnnxTensor.createTensor(ortEnvironment, fb, shape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
            return ortSession.run(inputs);
        }
    }

    // ------------------------------
    // 1D-CNN Inference and Byte Extraction
    // ------------------------------
    private long[] extractLastBytes(File apkFile, int byteLength) {
        long[] result = new long[byteLength];
        try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(apkFile, "r")) {
            long fileLength = raf.length();
            long startPos = Math.max(0, fileLength - byteLength);
            raf.seek(startPos);
            
            byte[] buffer = new byte[(int)(fileLength - startPos)];
            raf.readFully(buffer);
            
            // Zero-pad from left if file is smaller than byteLength
            int padLength = byteLength - buffer.length;
            for (int i = 0; i < padLength; i++) {
                result[i] = 0;
            }
            for (int i = 0; i < buffer.length; i++) {
                result[padLength + i] = buffer[i] & 0xFF; // Convert to unsigned (0-255)
            }
        } catch (Exception e) {
            sendLog("CNN extraction error: " + e.getMessage(), null);
        }
        return result;
    }

    private float runCnnInference(long[] inputVector) {
        if (ortEnvironment == null || ortSessionCnn == null) {
            sendLog("ONNX CNN model not initialized", "Error");
            return -1f;
        }
        try {
            long[] shape = new long[]{1, inputVector.length};
            java.nio.LongBuffer lb = java.nio.LongBuffer.wrap(inputVector);
            String inputName = ortSessionCnn.getInputNames().iterator().next();

            try (OnnxTensor tensor = OnnxTensor.createTensor(ortEnvironment, lb, shape)) {
                Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
                try (OrtSession.Result result = ortSessionCnn.run(inputs)) {
                    Object o = result.get(0).getValue();
                    if (o instanceof float[][]) {
                        float[][] out = (float[][]) o;
                        // Assuming out[0][0] is benign, out[0][1] is malware
                        double exp0 = Math.exp(out[0][0]);
                        double exp1 = Math.exp(out[0][1]);
                        return (float) (exp1 / (exp0 + exp1)); // Softmax probability for Malware
                    }
                    return -1f;
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "CNN Inference failed", e);
            sendLog("CNN Inference error: " + e.getMessage(), "Error");
            return -1f;
        }
    }

    private void writeScanMetrics(
            String trigger,
            File apk,
            double xgbParseMs, double xgbVecMs, double xgbInferMs, float xgbScore,
            double cnnParseMs, double cnnInferMs, float cnnScore,
            double wallMs, double cpuMs, long memDeltaBytes,
            long scanMemDelta
    ) {
        try {
            MetricsWriter.ScanMetrics scan = new MetricsWriter.ScanMetrics();
            scan.trigger = trigger;
            scan.apkName = apk.getName();
            scan.apkPath = apk.getAbsolutePath();
            scan.apkSizeBytes = apk.length();
            scan.wallMs = wallMs;
            scan.cpuMs = cpuMs;
            scan.memDeltaBytes = memDeltaBytes;

            scan.stages.add(new MetricsWriter.StageMetrics(
                    "manifest_xgb", xgbParseMs, xgbVecMs, xgbInferMs, xgbScore, scanMemDelta / 2));
            scan.stages.add(new MetricsWriter.StageMetrics(
                    "bytecnn", cnnParseMs, 0.0, cnnInferMs, cnnScore, scanMemDelta / 2));

            float ensemble = computeEnsembleScore(xgbScore, cnnScore);
            if (ensemble >= 0f) {
                scan.ensembleScore = ensemble;
                scan.ensembleDecision = ensemble >= 0.5f ? "malware" : "benign";
                if (xgbScore >= 0f && cnnScore >= 0f && Math.abs(xgbScore - cnnScore) > 0.4f) {
                    scan.ensembleDecision = "uncertain";
                }
            }

            File out = MetricsWriter.writeScan(this, scan);
            sendLog("Metrics JSON: " + out.getName(), null);
        } catch (Exception e) {
            Log.e(TAG, "Metrics write failed", e);
            sendLog("Metrics JSON error: " + e.getMessage(), null);
        }
    }

    private float computeEnsembleScore(float xgbScore, float cnnScore) {
        int n = 0;
        float sum = 0f;
        if (xgbScore >= 0f) {
            sum += xgbScore;
            n++;
        }
        if (cnnScore >= 0f) {
            sum += cnnScore;
            n++;
        }
        return n > 0 ? sum / n : -1f;
    }

    // ------------------------------
    // Load feature columns from gzipped JSON (features.json.gz) in assets
    // ------------------------------
    private void loadFeatureColumns() throws Exception {
        InputStream is = getAssets().open("mh1m_2500_rp_features.json.gzip");
        GZIPInputStream gzis = new GZIPInputStream(new BufferedInputStream(is));
        StringBuilder sb = new StringBuilder();
        byte[] buf = new byte[8192];
        int r;
        while ((r = gzis.read(buf)) != -1) {
            sb.append(new String(buf, 0, r));
        }
        gzis.close();

        JSONArray arr = new JSONArray(sb.toString());
        featureColumns.clear();
        featureIndex.clear();
        for (int i = 0; i < arr.length(); i++) {
            String f = arr.getString(i);
            featureColumns.add(f);
            featureIndex.put(f, i);
        }

        sendLog("Loaded " + featureColumns.size() + " feature columns", null);
    }

    @Override
    public void onDestroy() {
        if (ortSession != null) {
            try { ortSession.close(); } catch (Exception ignored) {}
        }
        if (ortSessionCnn != null) {
            try { ortSessionCnn.close(); } catch (Exception ignored) {}
        }
        if (ortEnvironment != null) {
            try { ortEnvironment.close(); } catch (Exception ignored) {}
        }
        super.onDestroy();
    }
}

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
    private static final int XGB_FEATURE_DIM = 2500;
    private static final float XGB_VALIDATION_ACCURACY = 0.9748f;
    private static final float CNN_VALIDATION_ACCURACY = 0.9607843f;

    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private OrtSession ortSessionCnn;
    private List<String> featureColumns = new ArrayList<>();
    private Map<String, Integer> featureIndex = new HashMap<>();

    private static final class XgbFeatureExtractionResult {
        final float[] aggregatedVector;
        final int dexFilesFound;
        final long structuralParsingTimeMs;
        final long parseTimeNanos;
        final long vectorizeTimeNanos;

        XgbFeatureExtractionResult(float[] aggregatedVector,
                                   int dexFilesFound,
                                   long structuralParsingTimeMs,
                                   long parseTimeNanos,
                                   long vectorizeTimeNanos) {
            this.aggregatedVector = aggregatedVector;
            this.dexFilesFound = dexFilesFound;
            this.structuralParsingTimeMs = structuralParsingTimeMs;
            this.parseTimeNanos = parseTimeNanos;
            this.vectorizeTimeNanos = vectorizeTimeNanos;
        }
    }


    public static void enqueueWork(Context context, Intent intent) {
        enqueueWork(context, ScanService.class, JOB_ID, intent);
    }

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            loadFeatureColumns();
        } catch (Exception e) {
            Log.w(TAG, "XGBoost feature list not loaded (CNN-only mode OK)", e);
            sendLog("XGBoost features skipped: " + e.getMessage(), null);
        }
        try {
            initOnnxModel();
        } catch (Exception e) {
            Log.e(TAG, "ONNX init error", e);
            sendLog("ONNX init error: " + e.getMessage(), "Error");
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

            // parse apk (XGBoost pipeline — skip if model not loaded)
            long parseStart = SystemClock.elapsedRealtimeNanos();
            float[] inputVector = new float[XGB_FEATURE_DIM];
            float score = -1f;
            long parseEnd = parseStart;
            long vecStart = parseStart;
            long vecEnd = parseStart;
            long inferStart = parseStart;
            long inferEnd = parseStart;
            int totalDexFilesFound = 0;
            long structuralParsingTimeMs = 0L;

            if (ortSession != null && !featureColumns.isEmpty()) {
                XgbFeatureExtractionResult extraction = extractAggregatedXgbFeatures(apk);
                inputVector = extraction.aggregatedVector;
                totalDexFilesFound = extraction.dexFilesFound;
                structuralParsingTimeMs = extraction.structuralParsingTimeMs;

                parseEnd = parseStart + extraction.parseTimeNanos;
                vecStart = parseEnd;
                vecEnd = vecStart + extraction.vectorizeTimeNanos;

                inferStart = SystemClock.elapsedRealtimeNanos();
                score = runInference(inputVector);
                inferEnd = SystemClock.elapsedRealtimeNanos();
            } else {
                sendLog("XGBoost pipeline skipped (model or features missing)", null);
            }

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
            long xgbMemDelta = memDelta / 2;
            long cnnMemDelta = memDelta - xgbMemDelta;

            float ensemble = computeEnsembleScore(score, cnnScore);
            String ensembleDecision = null;
            if (ensemble >= 0f) {
                ensembleDecision = ensemble >= 0.5f ? "malware" : "benign";
            }

            sendLog(String.format(
                    Locale.US,
                    "XGBoost: score=%.4f parse=%.2fms vec=%.2fms infer=%.2fms mem=%d bytes",
                    score, parsingMs, vectorMs, inferenceMs, xgbMemDelta
            ), null);
            sendLog(String.format(
                    Locale.US,
                    "1D-CNN: score=%.4f parse=%.2fms infer=%.2fms mem=%d bytes",
                    cnnScore, cnnParsingMs, cnnInferenceMs, cnnMemDelta
            ), null);

            File metricsFile = writeScanMetrics(
                    trigger,
                    apk,
                    parsingMs, vectorMs, inferenceMs, score,
                    cnnParsingMs, cnnInferenceMs, cnnScore,
                    totalMs, cpuMs / 1_000_000.0, memDelta,
                    xgbMemDelta, cnnMemDelta,
                    totalDexFilesFound,
                    structuralParsingTimeMs,
                    ensemble,
                    ensembleDecision
            );

            sendScanResult(
                    apkName, ensemble, ensembleDecision,
                    totalMs, memDelta / (1024.0 * 1024.0),
                    metricsFile != null ? metricsFile.getName() : null
            );
            sendLog("Scanned: " + apkName, "Idle");
        }

        sendLog("Scan completed.", "Idle");
    }

    private void sendLog(String log, String status) {
        Intent i = new Intent(MainActivity.ACTION_SCAN_LOG);
        i.putExtra("log", log);
        if (status != null) i.putExtra("status", status);
        LocalBroadcastManager.getInstance(this).sendBroadcast(i);
    }

    private void sendScanResult(
            String apkName,
            float ensembleScore,
            String ensembleDecision,
            double totalMs,
            double totalMemMb,
            String metricsFileName
    ) {
        Intent i = new Intent(MainActivity.ACTION_SCAN_RESULT);
        i.putExtra("apk_name", apkName);
        i.putExtra("ensemble_score", ensembleScore);
        i.putExtra("ensemble_decision", ensembleDecision);
        i.putExtra("total_ms", totalMs);
        i.putExtra("total_mem_mb", totalMemMb);
        if (metricsFileName != null) {
            i.putExtra("metrics_file", metricsFileName);
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(i);
    }

    private XgbFeatureExtractionResult extractAggregatedXgbFeatures(File apkFile) {
        float[] aggregatedVector = new float[XGB_FEATURE_DIM];
        long parseTimeNanos = 0L;
        long vectorizeTimeNanos = 0L;
        int dexFilesFound = 0;
        long structuralParsingStart = SystemClock.elapsedRealtimeNanos();

        try (ZipFile zip = new ZipFile(apkFile)) {
            Enumeration<? extends ZipEntry> entries = zip.entries();

            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                String entryName = entry.getName();
                if (entryName == null) {
                    continue;
                }

                if (entryName.equalsIgnoreCase("AndroidManifest.xml")) {
                    long manifestParseStart = SystemClock.elapsedRealtimeNanos();
                    Set<String> manifestFeatures = new HashSet<>();
                    try (InputStream is = zip.getInputStream(entry)) {
                        AxmlReader reader = new AxmlReader(is);
                        Set<String> rawFeatures = reader.parse();

                        for (String rawFeature : rawFeatures) {
                            if (rawFeature != null && rawFeature.startsWith("android.permission.")) {
                                manifestFeatures.add(normalizePermission(rawFeature));
                            }
                            if (rawFeature != null && rawFeature.startsWith("android.intent.action.")) {
                                manifestFeatures.add(normalizeIntent(rawFeature));
                            }
                        }
                    }
                    parseTimeNanos += (SystemClock.elapsedRealtimeNanos() - manifestParseStart);

                    long manifestVectorStart = SystemClock.elapsedRealtimeNanos();
                    float[] manifestVector = vectorize(manifestFeatures);
                    orPoolInto(aggregatedVector, manifestVector);
                    vectorizeTimeNanos += (SystemClock.elapsedRealtimeNanos() - manifestVectorStart);
                    continue;
                }

                String lowerName = entryName.toLowerCase(Locale.US);
                if (lowerName.startsWith("classes") && lowerName.endsWith(".dex")) {
                    dexFilesFound++;

                    long dexParseStart = SystemClock.elapsedRealtimeNanos();
                    Set<String> dexFeatures = new HashSet<>();
                    try (InputStream is = zip.getInputStream(entry)) {
                        MinimalDexParser.parse(is, dexFeatures::add);
                    }
                    parseTimeNanos += (SystemClock.elapsedRealtimeNanos() - dexParseStart);

                    long dexVectorStart = SystemClock.elapsedRealtimeNanos();
                    float[] dexVector = vectorize(dexFeatures);
                    orPoolInto(aggregatedVector, dexVector);
                    vectorizeTimeNanos += (SystemClock.elapsedRealtimeNanos() - dexVectorStart);
                }
            }
        } catch (Exception e) {
            sendLog("Manifest parse error: " + e.getMessage(), null);
        }

        long structuralParsingTimeMs =
                (SystemClock.elapsedRealtimeNanos() - structuralParsingStart) / 1_000_000L;
        return new XgbFeatureExtractionResult(
                aggregatedVector,
                dexFilesFound,
                structuralParsingTimeMs,
                parseTimeNanos,
                vectorizeTimeNanos
        );
    }

    private void orPoolInto(float[] master, float[] candidate) {
        int n = Math.min(master.length, candidate.length);
        for (int i = 0; i < n; i++) {
            if (candidate[i] > 0.0f) {
                master[i] = 1.0f;
            }
        }
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
        float[] vec = new float[XGB_FEATURE_DIM];
        for (String t : tokens) {
            Integer idx = featureIndex.get(t);
            if (idx != null && idx >= 0 && idx < XGB_FEATURE_DIM) vec[idx] = 1.0f;
        }
        return vec;
    }

    // ------------------------------
    // ONNX Runtime: init model and run inference
    // ------------------------------
    private void initOnnxModel() throws Exception {
        ortEnvironment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

        boolean xgbOk = false;
        boolean cnnOk = false;

        try {
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
            ortSession = ortEnvironment.createSession(modelFile.getAbsolutePath(), sessionOptions);
            xgbOk = true;
            sendLog("XGBoost ONNX loaded", null);
        } catch (Exception ex) {
            Log.w(TAG, "XGBoost ONNX not loaded", ex);
            ortSession = null;
            sendLog("XGBoost ONNX skipped: " + ex.getMessage(), null);
        }

        try {
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
            cnnOk = true;
            sendLog("ByteCNN ONNX loaded", null);
        } catch (Exception ex) {
            Log.e(TAG, "ByteCNN ONNX not loaded", ex);
            ortSessionCnn = null;
            sendLog("ByteCNN ONNX missing: " + ex.getMessage(), "Error");
        }

        if (!xgbOk && !cnnOk) {
            throw new IllegalStateException("No ONNX models loaded from assets");
        }
    }

    private float runInference(float[] inputVector) {
        if (ortEnvironment == null || ortSession == null) {
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

    private File writeScanMetrics(
            String trigger,
            File apk,
            double xgbParseMs, double xgbVecMs, double xgbInferMs, float xgbScore,
            double cnnParseMs, double cnnInferMs, float cnnScore,
            double wallMs, double cpuMs, long memDeltaBytes,
            long xgbMemDelta,
            long cnnMemDelta,
            int totalDexFilesFound,
            long structuralParsingTimeMs,
            float ensemble,
            String ensembleDecision
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
            scan.totalDexFilesFound = totalDexFilesFound;
            scan.structuralParsingTimeMs = structuralParsingTimeMs;

            scan.stages.add(new MetricsWriter.StageMetrics(
                    "manifest_xgb", xgbParseMs, xgbVecMs, xgbInferMs, xgbScore, xgbMemDelta));
            scan.stages.add(new MetricsWriter.StageMetrics(
                    "bytecnn", cnnParseMs, 0.0, cnnInferMs, cnnScore, cnnMemDelta));

            if (ensemble >= 0f) {
                scan.ensembleScore = ensemble;
                scan.ensembleDecision = ensembleDecision;
                scan.ensemblePolicy = "weighted_validation_accuracy";
            }

            return MetricsWriter.writeScan(this, scan);
        } catch (Exception e) {
            Log.e(TAG, "Metrics write failed", e);
            sendLog("Metrics JSON error: " + e.getMessage(), null);
            return null;
        }
    }

    private float computeEnsembleScore(float xgbScore, float cnnScore) {
        boolean hasXgb = xgbScore >= 0f;
        boolean hasCnn = cnnScore >= 0f;
        if (!hasXgb && !hasCnn) {
            return -1f;
        }
        if (hasXgb && !hasCnn) {
            return xgbScore;
        }
        if (!hasXgb) {
            return cnnScore;
        }

        float total = XGB_VALIDATION_ACCURACY + CNN_VALIDATION_ACCURACY;
        float wXgb = XGB_VALIDATION_ACCURACY / total;
        float wCnn = CNN_VALIDATION_ACCURACY / total;
        return (wXgb * xgbScore) + (wCnn * cnnScore);
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

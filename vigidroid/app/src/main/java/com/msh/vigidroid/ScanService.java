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
    private MlpHeaderOnnxRunner mlpHeaderRunner;
    private DexHeaderFeatureExtractor mlpHeaderExtractor;
    private PatternAOnnxRunner patternARunner;
    private DexHeaderFeatureExtractor patternAHeaderExtractor;
    private ManifestBowExtractor patternABowExtractor;
    private PatternBOnnxRunner patternBRunner;
    private DexHeaderFeatureExtractor patternBHeaderExtractor;
    private ManifestBowExtractor patternBBowExtractor;
    private LinRegDroidOnnxRunner linRegRunner;
    private LinRegPermissionExtractor linRegExtractor;
    private MldpPrunedOnnxRunner mldpRunner;
    private MldpPrunedPermissionExtractor mldpExtractor;
    private BroadcastMldpHybridOnnxRunner broadcastMldpRunner;
    private BroadcastMldpHybridExtractor broadcastMldpExtractor;
    private MldpDexHeaderExtractor mldpDexHeaderExtractor;
    private MldpDexHeaderModeAOnnxRunner mldpDexHeaderModeARunner;
    private MldpDexHeaderModeBOnnxRunner mldpDexHeaderModeBRunner;
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
        try {
            initMlpHeaderPipeline();
        } catch (Exception e) {
            Log.w(TAG, "BM1 mlp_header pipeline not loaded", e);
            sendLog("BM1 mlp_header skipped: " + e.getMessage(), null);
        }
        try {
            initPatternAPipeline();
        } catch (Exception e) {
            Log.w(TAG, "Pattern A pipeline not loaded", e);
            sendLog("Pattern A skipped: " + e.getMessage(), null);
        }
        try {
            initPatternBPipeline();
        } catch (Exception e) {
            Log.w(TAG, "Pattern B pipeline not loaded", e);
            sendLog("Pattern B skipped: " + e.getMessage(), null);
        }
        try {
            initLinRegPermissionPipeline();
        } catch (Exception e) {
            Log.w(TAG, "LinRegDroid permission pipeline not loaded", e);
            sendLog("LinRegDroid permission skipped: " + e.getMessage(), null);
        }
        try {
            initMldpPrunedPermissionPipeline();
        } catch (Exception e) {
            Log.w(TAG, "MLDP pruned permission pipeline not loaded", e);
            sendLog("MLDP pruned permission skipped: " + e.getMessage(), null);
        }
        try {
            initBroadcastMldpHybridPipeline();
        } catch (Exception e) {
            Log.w(TAG, "Broadcast + MLDP hybrid pipeline not loaded", e);
            sendLog("Broadcast + MLDP hybrid skipped: " + e.getMessage(), null);
        }
        try {
            initMldpDexHeaderCascadePipeline();
        } catch (Exception e) {
            Log.w(TAG, "MLDP + Dex header cascade pipeline not loaded", e);
            sendLog("MLDP + Dex cascade skipped: " + e.getMessage(), null);
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

            MldpDexHeaderScanSnapshot cascadeSnapshot = runMldpDexHeaderCascade(apk, apkName);

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

            // Broadcast + MLDP hybrid (manifest-only early gate) — separate stage
            double broadcastMldpParseMs = 0.0;
            double broadcastMldpVecMs = 0.0;
            double broadcastMldpInferMs = 0.0;
            float broadcastMldpScore = -1f;
            long broadcastMldpMemDelta = 0L;
            if (broadcastMldpRunner != null && broadcastMldpExtractor != null) {
                long broadcastMldpMemBefore = Debug.getNativeHeapAllocatedSize();
                try {
                    BroadcastMldpHybridExtractor.ExtractionResult extraction =
                            broadcastMldpExtractor.extract(apk);
                    broadcastMldpParseMs = extraction.parseMs();
                    broadcastMldpVecMs = extraction.vectorizeMs();
                    long broadcastMldpInferStart = SystemClock.elapsedRealtimeNanos();
                    broadcastMldpScore = broadcastMldpRunner.predict(extraction.vector);
                    broadcastMldpInferMs =
                            (SystemClock.elapsedRealtimeNanos() - broadcastMldpInferStart) / 1_000_000.0;
                } catch (Exception ex) {
                    Log.w(TAG, "Broadcast + MLDP hybrid failed for " + apkName, ex);
                    sendLog("Broadcast + MLDP hybrid error: " + ex.getMessage(), null);
                }
                broadcastMldpMemDelta = Debug.getNativeHeapAllocatedSize() - broadcastMldpMemBefore;
            }

            if (broadcastMldpScore >= 0f) {
                sendLog(String.format(
                        Locale.US,
                        "Broadcast+MLDP: score=%.4f parse=%.2fms vec=%.2fms infer=%.2fms mem=%d bytes",
                        broadcastMldpScore,
                        broadcastMldpParseMs,
                        broadcastMldpVecMs,
                        broadcastMldpInferMs,
                        broadcastMldpMemDelta
                ), null);
            }

            // BM1 mlp_header (Dex header only) — separate stage; does not affect CNN/XGB ensemble
            double mlpParseMs = 0.0;
            double mlpVecMs = 0.0;
            double mlpInferMs = 0.0;
            float mlpScore = -1f;
            long mlpMemDelta = 0L;
            if (mlpHeaderRunner != null && mlpHeaderExtractor != null) {
                long mlpMemBefore = Debug.getNativeHeapAllocatedSize();
                try {
                    DexHeaderFeatureExtractor.ExtractionResult mlpExtraction =
                            mlpHeaderExtractor.extract(apk);
                    mlpParseMs = mlpExtraction.extractNanos / 1_000_000.0;
                    mlpVecMs = mlpExtraction.normalizeNanos / 1_000_000.0;
                    long mlpInferStart = SystemClock.elapsedRealtimeNanos();
                    mlpScore = mlpHeaderRunner.predict(mlpExtraction.features);
                    mlpInferMs = (SystemClock.elapsedRealtimeNanos() - mlpInferStart) / 1_000_000.0;
                } catch (Exception ex) {
                    Log.w(TAG, "BM1 mlp_header failed for " + apkName, ex);
                    sendLog("BM1 mlp_header error: " + ex.getMessage(), null);
                }
                mlpMemDelta = Debug.getNativeHeapAllocatedSize() - mlpMemBefore;
            }

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
            if (mlpScore >= 0f) {
                sendLog(String.format(
                        Locale.US,
                        "BM1 mlp_header: score=%.4f parse=%.2fms norm=%.2fms infer=%.2fms mem=%d bytes",
                        mlpScore, mlpParseMs, mlpVecMs, mlpInferMs, mlpMemDelta
                ), null);
            }

            // Pattern A (Dex header + manifest BoW) — separate stage; does not affect CNN/XGB ensemble
            double patternAParseMs = 0.0;
            double patternABowMs = 0.0;
            double patternAInferMs = 0.0;
            float patternAScore = -1f;
            long patternAMemDelta = 0L;
            if (patternARunner != null && patternAHeaderExtractor != null && patternABowExtractor != null) {
                long patternAMemBefore = Debug.getNativeHeapAllocatedSize();
                try {
                    DexHeaderFeatureExtractor.ExtractionResult headerExtraction =
                            patternAHeaderExtractor.extract(apk);
                    ManifestBowExtractor.ExtractionResult bowExtraction =
                            patternABowExtractor.extract(apk);
                    patternAParseMs = headerExtraction.extractNanos / 1_000_000.0;
                    patternABowMs = (bowExtraction.extractNanos + bowExtraction.vectorizeNanos) / 1_000_000.0;
                    long patternAInferStart = SystemClock.elapsedRealtimeNanos();
                    patternAScore = patternARunner.predict(
                            headerExtraction.features, bowExtraction.bow);
                    patternAInferMs =
                            (SystemClock.elapsedRealtimeNanos() - patternAInferStart) / 1_000_000.0;
                } catch (Exception ex) {
                    Log.w(TAG, "Pattern A failed for " + apkName, ex);
                    sendLog("Pattern A error: " + ex.getMessage(), null);
                }
                patternAMemDelta = Debug.getNativeHeapAllocatedSize() - patternAMemBefore;
            }

            if (patternAScore >= 0f) {
                sendLog(String.format(
                        Locale.US,
                        "Pattern A: score=%.4f header=%.2fms bow=%.2fms infer=%.2fms mem=%d bytes",
                        patternAScore, patternAParseMs, patternABowMs, patternAInferMs, patternAMemDelta
                ), null);
            }

            // Pattern B (dual branch fused ONNX) — separate stage; does not affect CNN/XGB ensemble
            double patternBParseMs = 0.0;
            double patternBBowMs = 0.0;
            double patternBInferMs = 0.0;
            float patternBScore = -1f;
            long patternBMemDelta = 0L;
            if (patternBRunner != null && patternBHeaderExtractor != null && patternBBowExtractor != null) {
                long patternBMemBefore = Debug.getNativeHeapAllocatedSize();
                try {
                    DexHeaderFeatureExtractor.ExtractionResult headerExtraction =
                            patternBHeaderExtractor.extract(apk);
                    ManifestBowExtractor.ExtractionResult bowExtraction =
                            patternBBowExtractor.extract(apk);
                    patternBParseMs = headerExtraction.extractNanos / 1_000_000.0;
                    patternBBowMs = (bowExtraction.extractNanos + bowExtraction.vectorizeNanos) / 1_000_000.0;
                    long patternBInferStart = SystemClock.elapsedRealtimeNanos();
                    patternBScore = patternBRunner.predict(
                            headerExtraction.features, bowExtraction.bow);
                    patternBInferMs =
                            (SystemClock.elapsedRealtimeNanos() - patternBInferStart) / 1_000_000.0;
                } catch (Exception ex) {
                    Log.w(TAG, "Pattern B failed for " + apkName, ex);
                    sendLog("Pattern B error: " + ex.getMessage(), null);
                }
                patternBMemDelta = Debug.getNativeHeapAllocatedSize() - patternBMemBefore;
            }

            if (patternBScore >= 0f) {
                sendLog(String.format(
                        Locale.US,
                        "Pattern B: score=%.4f header=%.2fms bow=%.2fms infer=%.2fms mem=%d bytes",
                        patternBScore, patternBParseMs, patternBBowMs, patternBInferMs, patternBMemDelta
                ), null);
            }

            // LinRegDroid (manifest permissions only) — separate stage; does not affect CNN/XGB ensemble
            double linRegParseMs = 0.0;
            double linRegVecMs = 0.0;
            double linRegInferMs = 0.0;
            float linRegScore = -1f;
            long linRegMemDelta = 0L;
            if (linRegRunner != null && linRegExtractor != null) {
                long linRegMemBefore = Debug.getNativeHeapAllocatedSize();
                try {
                    LinRegPermissionExtractor.ExtractionResult extraction = linRegExtractor.extract(apk);
                    linRegParseMs = extraction.extractNanos / 1_000_000.0;
                    linRegVecMs = extraction.vectorizeNanos / 1_000_000.0;
                    long linRegInferStart = SystemClock.elapsedRealtimeNanos();
                    linRegScore = linRegRunner.predict(extraction.vector);
                    linRegInferMs =
                            (SystemClock.elapsedRealtimeNanos() - linRegInferStart) / 1_000_000.0;
                } catch (Exception ex) {
                    Log.w(TAG, "LinRegDroid permission failed for " + apkName, ex);
                    sendLog("LinRegDroid permission error: " + ex.getMessage(), null);
                }
                linRegMemDelta = Debug.getNativeHeapAllocatedSize() - linRegMemBefore;
            }

            if (linRegScore >= 0f) {
                sendLog(String.format(
                        Locale.US,
                        "LinRegDroid: score=%.4f parse=%.2fms vec=%.2fms infer=%.2fms mem=%d bytes",
                        linRegScore, linRegParseMs, linRegVecMs, linRegInferMs, linRegMemDelta
                ), null);
            }

            // MLDP pruned permissions — separate stage; does not affect CNN/XGB ensemble
            double mldpParseMs = 0.0;
            double mldpVecMs = 0.0;
            double mldpInferMs = 0.0;
            float mldpScore = -1f;
            long mldpMemDelta = 0L;
            if (mldpRunner != null && mldpExtractor != null) {
                long mldpMemBefore = Debug.getNativeHeapAllocatedSize();
                try {
                    MldpPrunedPermissionExtractor.ExtractionResult extraction = mldpExtractor.extract(apk);
                    mldpParseMs = extraction.extractNanos / 1_000_000.0;
                    mldpVecMs = extraction.vectorizeNanos / 1_000_000.0;
                    long mldpInferStart = SystemClock.elapsedRealtimeNanos();
                    mldpScore = mldpRunner.predict(extraction.vector);
                    mldpInferMs = (SystemClock.elapsedRealtimeNanos() - mldpInferStart) / 1_000_000.0;
                } catch (Exception ex) {
                    Log.w(TAG, "MLDP pruned permission failed for " + apkName, ex);
                    sendLog("MLDP pruned permission error: " + ex.getMessage(), null);
                }
                mldpMemDelta = Debug.getNativeHeapAllocatedSize() - mldpMemBefore;
            }

            if (mldpScore >= 0f) {
                sendLog(String.format(
                        Locale.US,
                        "MLDP pruned: score=%.4f parse=%.2fms vec=%.2fms infer=%.2fms mem=%d bytes",
                        mldpScore, mldpParseMs, mldpVecMs, mldpInferMs, mldpMemDelta
                ), null);
            }

            File metricsFile = writeScanMetrics(
                    trigger,
                    apk,
                    cascadeSnapshot,
                    parsingMs, vectorMs, inferenceMs, score,
                    cnnParsingMs, cnnInferenceMs, cnnScore,
                    broadcastMldpParseMs, broadcastMldpVecMs, broadcastMldpInferMs,
                    broadcastMldpScore, broadcastMldpMemDelta,
                    mlpParseMs, mlpVecMs, mlpInferMs, mlpScore, mlpMemDelta,
                    patternAParseMs, patternABowMs, patternAInferMs, patternAScore, patternAMemDelta,
                    patternBParseMs, patternBBowMs, patternBInferMs, patternBScore, patternBMemDelta,
                    linRegParseMs, linRegVecMs, linRegInferMs, linRegScore, linRegMemDelta,
                    mldpParseMs, mldpVecMs, mldpInferMs, mldpScore, mldpMemDelta,
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
        return PermissionNormalizer.normalize(p);
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

    private void initMlpHeaderPipeline() throws Exception {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment();
        }
        mlpHeaderExtractor = DexHeaderFeatureExtractor.fromAssets(this);
        mlpHeaderRunner = MlpHeaderOnnxRunner.create(this, ortEnvironment);
        sendLog("BM1 mlp_header ONNX loaded", null);
    }

    private void initPatternAPipeline() throws Exception {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment();
        }
        patternAHeaderExtractor =
                DexHeaderFeatureExtractor.fromAssets(this, DexHeaderFeatureExtractor.PATTERN_A_NORMALIZATION_ASSET);
        patternABowExtractor = ManifestBowExtractor.fromAssets(this, ManifestBowExtractor.PATTERN_A_VOCAB_ASSET);
        patternARunner = PatternAOnnxRunner.create(this, ortEnvironment);
        sendLog("Pattern A ONNX loaded", null);
    }

    private void initPatternBPipeline() throws Exception {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment();
        }
        patternBHeaderExtractor =
                DexHeaderFeatureExtractor.fromAssets(this, DexHeaderFeatureExtractor.PATTERN_B_NORMALIZATION_ASSET);
        patternBBowExtractor = ManifestBowExtractor.fromAssets(this, ManifestBowExtractor.PATTERN_B_VOCAB_ASSET);
        patternBRunner = PatternBOnnxRunner.create(this, ortEnvironment);
        sendLog("Pattern B ONNX loaded", null);
    }

    private void initLinRegPermissionPipeline() throws Exception {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment();
        }
        linRegExtractor = LinRegPermissionExtractor.fromAssets(this);
        linRegRunner = LinRegDroidOnnxRunner.create(this, ortEnvironment);
        sendLog(
                "LinRegDroid "
                        + ModelRegistry.LINREGDROID_PERMISSION.modelId
                        + " ONNX loaded (domain="
                        + ModelRegistry.LINREGDROID_PERMISSION.domain
                        + ")",
                null);
    }

    private void initMldpPrunedPermissionPipeline() throws Exception {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment();
        }
        mldpExtractor = MldpPrunedPermissionExtractor.fromAssets(this);
        mldpRunner = MldpPrunedOnnxRunner.create(this, ortEnvironment);
        sendLog(
                "MLDP "
                        + ModelRegistry.MLDP_PRUNED_PERMISSION.modelId
                        + " ONNX loaded (domain="
                        + ModelRegistry.MLDP_PRUNED_PERMISSION.domain
                        + ", type="
                        + mldpRunner.getModelType()
                        + ")",
                null);
    }

    private static final class MldpDexHeaderScanSnapshot {
        double modeAParseMs;
        double modeADexMs;
        double modeAVecMs;
        double modeAInferMs;
        float modeAScore = -1f;
        long modeAMemDelta;
        double modeBParseMs;
        double modeBDexMs;
        double modeBVecMs;
        double modeBInferMs;
        float modeBStage1Score = -1f;
        float modeBStage2Score = -1f;
        float modeBScore = -1f;
        boolean modeBEarlyExit;
        long modeBMemDelta;
    }

    private MldpDexHeaderScanSnapshot runMldpDexHeaderCascade(File apk, String apkName) {
        MldpDexHeaderScanSnapshot snapshot = new MldpDexHeaderScanSnapshot();
        if (mldpDexHeaderExtractor == null) {
            return snapshot;
        }

        if (mldpDexHeaderModeBRunner != null) {
            long memBefore = Debug.getNativeHeapAllocatedSize();
            try {
                MldpDexHeaderExtractor.PermissionBlockResult perm =
                        mldpDexHeaderExtractor.extractPermissionBlock(apk);
                snapshot.modeBParseMs = perm.parseMs();
                snapshot.modeBVecMs = 0.0;

                long inferStart = SystemClock.elapsedRealtimeNanos();
                float stage1Score = mldpDexHeaderModeBRunner.predictStage1(perm.xS);
                snapshot.modeBStage1Score = stage1Score;
                MldpDexHeaderCascadeThresholds thresholds = mldpDexHeaderModeBRunner.getThresholds();

                if (thresholds.isEarlyExitBenign(stage1Score)
                        || thresholds.isEarlyExitMalware(stage1Score)) {
                    snapshot.modeBEarlyExit = true;
                    snapshot.modeBDexMs = 0.0;
                    snapshot.modeBStage2Score = MldpDexHeaderModeBOnnxRunner.SKIPPED_STAGE2_SCORE;
                    snapshot.modeBScore = stage1Score;
                } else {
                    MldpDexHeaderExtractor.DexBlockResult dex =
                            mldpDexHeaderExtractor.extractDexBlock(apk);
                    snapshot.modeBDexMs = dex.dexMs();
                    float stage2Score = mldpDexHeaderModeBRunner.predictStage2(dex.h);
                    snapshot.modeBStage2Score = stage2Score;
                    snapshot.modeBScore = stage2Score;
                    snapshot.modeBEarlyExit = false;
                }
                snapshot.modeBInferMs =
                        (SystemClock.elapsedRealtimeNanos() - inferStart) / 1_000_000.0;
            } catch (Exception ex) {
                Log.w(TAG, "MLDP + Dex cascade Mode B failed for " + apkName, ex);
                sendLog("MLDP + Dex cascade Mode B error: " + ex.getMessage(), null);
            }
            snapshot.modeBMemDelta = Debug.getNativeHeapAllocatedSize() - memBefore;
            if (snapshot.modeBScore >= 0f) {
                sendLog(
                        String.format(
                                Locale.US,
                                "MLDP+Dex cascade Mode B: score=%.4f s1=%.4f s2=%.4f early_exit=%s "
                                        + "parse=%.2fms dex=%.2fms infer=%.2fms mem=%d bytes",
                                snapshot.modeBScore,
                                snapshot.modeBStage1Score,
                                snapshot.modeBStage2Score,
                                snapshot.modeBEarlyExit,
                                snapshot.modeBParseMs,
                                snapshot.modeBDexMs,
                                snapshot.modeBInferMs,
                                snapshot.modeBMemDelta),
                        null);
            }
        }

        if (mldpDexHeaderModeARunner != null) {
            long memBefore = Debug.getNativeHeapAllocatedSize();
            try {
                MldpDexHeaderExtractor.ExtractionResult extraction = mldpDexHeaderExtractor.extract(apk);
                snapshot.modeAParseMs = extraction.parseMs();
                snapshot.modeADexMs = extraction.dexMs();
                snapshot.modeAVecMs = extraction.vectorizeMs();
                long inferStart = SystemClock.elapsedRealtimeNanos();
                snapshot.modeAScore = mldpDexHeaderModeARunner.predict(extraction.x);
                snapshot.modeAInferMs =
                        (SystemClock.elapsedRealtimeNanos() - inferStart) / 1_000_000.0;
            } catch (Exception ex) {
                Log.w(TAG, "MLDP + Dex cascade Mode A failed for " + apkName, ex);
                sendLog("MLDP + Dex cascade Mode A error: " + ex.getMessage(), null);
            }
            snapshot.modeAMemDelta = Debug.getNativeHeapAllocatedSize() - memBefore;
            if (snapshot.modeAScore >= 0f) {
                sendLog(
                        String.format(
                                Locale.US,
                                "MLDP+Dex cascade Mode A: score=%.4f parse=%.2fms dex=%.2fms "
                                        + "vec=%.2fms infer=%.2fms mem=%d bytes",
                                snapshot.modeAScore,
                                snapshot.modeAParseMs,
                                snapshot.modeADexMs,
                                snapshot.modeAVecMs,
                                snapshot.modeAInferMs,
                                snapshot.modeAMemDelta),
                        null);
            }
        }

        return snapshot;
    }

    private void initMldpDexHeaderCascadePipeline() throws Exception {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment();
        }
        mldpDexHeaderExtractor = MldpDexHeaderExtractor.fromAssets(this);
        mldpDexHeaderModeARunner = MldpDexHeaderModeAOnnxRunner.create(this, ortEnvironment);
        mldpDexHeaderModeBRunner = MldpDexHeaderModeBOnnxRunner.create(this, ortEnvironment);
        sendLog(
                "MLDP+Dex cascade loaded (Mode A="
                        + ModelRegistry.MLDP_DEXHEADER_CASCADE_MODE_A.modelId
                        + ", Mode B="
                        + ModelRegistry.MLDP_DEXHEADER_CASCADE_MODE_B.modelId
                        + ", domain="
                        + MldpDexHeaderModeAOnnxRunner.DOMAIN
                        + ")",
                null);
    }

    private void initBroadcastMldpHybridPipeline() throws Exception {
        if (ortEnvironment == null) {
            ortEnvironment = OrtEnvironment.getEnvironment();
        }
        broadcastMldpExtractor = BroadcastMldpHybridExtractor.fromAssets(this);
        broadcastMldpRunner = BroadcastMldpHybridOnnxRunner.create(this, ortEnvironment);
        sendLog(
                "Broadcast+MLDP "
                        + ModelRegistry.BROADCAST_MLDP_HYBRID.modelId
                        + " ONNX loaded (domain="
                        + ModelRegistry.BROADCAST_MLDP_HYBRID.domain
                        + ", type="
                        + broadcastMldpRunner.getModelType()
                        + ")",
                null);
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
            MldpDexHeaderScanSnapshot cascadeSnapshot,
            double xgbParseMs, double xgbVecMs, double xgbInferMs, float xgbScore,
            double cnnParseMs, double cnnInferMs, float cnnScore,
            double broadcastMldpParseMs, double broadcastMldpVecMs, double broadcastMldpInferMs,
            float broadcastMldpScore, long broadcastMldpMemDelta,
            double mlpParseMs, double mlpVecMs, double mlpInferMs, float mlpScore, long mlpMemDelta,
            double patternAParseMs, double patternABowMs, double patternAInferMs, float patternAScore,
            long patternAMemDelta,
            double patternBParseMs, double patternBBowMs, double patternBInferMs, float patternBScore,
            long patternBMemDelta,
            double linRegParseMs, double linRegVecMs, double linRegInferMs, float linRegScore,
            long linRegMemDelta,
            double mldpParseMs, double mldpVecMs, double mldpInferMs, float mldpScore, long mldpMemDelta,
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

            int cascadeInsertAt = 0;
            if (cascadeSnapshot.modeBScore >= 0f) {
                scan.stages.add(
                        cascadeInsertAt++,
                        MetricsWriter.StageMetrics.cascade(
                                MldpDexHeaderModeBOnnxRunner.DOMAIN,
                                ModelRegistry.MLDP_DEXHEADER_CASCADE_MODE_B.modelId,
                                "B",
                                cascadeSnapshot.modeBParseMs,
                                cascadeSnapshot.modeBDexMs,
                                cascadeSnapshot.modeBVecMs,
                                cascadeSnapshot.modeBInferMs,
                                cascadeSnapshot.modeBStage1Score,
                                cascadeSnapshot.modeBStage2Score,
                                cascadeSnapshot.modeBEarlyExit,
                                cascadeSnapshot.modeBScore,
                                cascadeSnapshot.modeBMemDelta));
            }
            if (cascadeSnapshot.modeAScore >= 0f) {
                scan.stages.add(
                        cascadeInsertAt,
                        MetricsWriter.StageMetrics.cascade(
                                MldpDexHeaderModeAOnnxRunner.DOMAIN,
                                ModelRegistry.MLDP_DEXHEADER_CASCADE_MODE_A.modelId,
                                "A",
                                cascadeSnapshot.modeAParseMs,
                                cascadeSnapshot.modeADexMs,
                                cascadeSnapshot.modeAVecMs,
                                cascadeSnapshot.modeAInferMs,
                                -1f,
                                -1f,
                                false,
                                cascadeSnapshot.modeAScore,
                                cascadeSnapshot.modeAMemDelta));
            }

            scan.stages.add(new MetricsWriter.StageMetrics(
                    "manifest_xgb", xgbParseMs, xgbVecMs, xgbInferMs, xgbScore, xgbMemDelta));
            scan.stages.add(new MetricsWriter.StageMetrics(
                    "bytecnn", cnnParseMs, 0.0, cnnInferMs, cnnScore, cnnMemDelta));
            if (broadcastMldpScore >= 0f) {
                scan.stages.add(new MetricsWriter.StageMetrics(
                        ModelRegistry.BROADCAST_MLDP_HYBRID.domain,
                        ModelRegistry.BROADCAST_MLDP_HYBRID.modelId,
                        broadcastMldpParseMs,
                        broadcastMldpVecMs,
                        broadcastMldpInferMs,
                        broadcastMldpScore,
                        broadcastMldpMemDelta));
            }
            if (mlpScore >= 0f) {
                scan.stages.add(new MetricsWriter.StageMetrics(
                        MlpHeaderOnnxRunner.DOMAIN,
                        mlpParseMs, mlpVecMs, mlpInferMs, mlpScore, mlpMemDelta));
            }
            if (patternAScore >= 0f) {
                scan.stages.add(new MetricsWriter.StageMetrics(
                        PatternAOnnxRunner.DOMAIN,
                        patternAParseMs, patternABowMs, patternAInferMs, patternAScore, patternAMemDelta));
            }
            if (patternBScore >= 0f) {
                scan.stages.add(new MetricsWriter.StageMetrics(
                        PatternBOnnxRunner.DOMAIN,
                        patternBParseMs, patternBBowMs, patternBInferMs, patternBScore, patternBMemDelta));
            }
            if (linRegScore >= 0f) {
                scan.stages.add(new MetricsWriter.StageMetrics(
                        ModelRegistry.LINREGDROID_PERMISSION.domain,
                        linRegParseMs, linRegVecMs, linRegInferMs, linRegScore, linRegMemDelta));
            }
            if (mldpScore >= 0f) {
                scan.stages.add(new MetricsWriter.StageMetrics(
                        ModelRegistry.MLDP_PRUNED_PERMISSION.domain,
                        mldpParseMs, mldpVecMs, mldpInferMs, mldpScore, mldpMemDelta));
            }

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
        if (mlpHeaderRunner != null) {
            try { mlpHeaderRunner.close(); } catch (Exception ignored) {}
        }
        if (patternARunner != null) {
            try { patternARunner.close(); } catch (Exception ignored) {}
        }
        if (patternBRunner != null) {
            try { patternBRunner.close(); } catch (Exception ignored) {}
        }
        if (linRegRunner != null) {
            try { linRegRunner.close(); } catch (Exception ignored) {}
        }
        if (mldpRunner != null) {
            try { mldpRunner.close(); } catch (Exception ignored) {}
        }
        if (broadcastMldpRunner != null) {
            try { broadcastMldpRunner.close(); } catch (Exception ignored) {}
        }
        if (mldpDexHeaderModeARunner != null) {
            try { mldpDexHeaderModeARunner.close(); } catch (Exception ignored) {}
        }
        if (mldpDexHeaderModeBRunner != null) {
            try { mldpDexHeaderModeBRunner.close(); } catch (Exception ignored) {}
        }
        if (ortEnvironment != null) {
            try { ortEnvironment.close(); } catch (Exception ignored) {}
        }
        super.onDestroy();
    }
}

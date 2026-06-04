package com.msh.vigidroid;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

/**
 * Writes one JSON file per APK scan under getExternalFilesDir(null)/metrics/.
 * Schema: Shared_pipeline_Files/schemas/device_scan.schema.json
 */
public final class MetricsWriter {

    private static final String TAG = "MetricsWriter";
    public static final String METRICS_SUBDIR = "metrics";
    public static final String AGGREGATE_FILENAME = "all_scan_metrics.json";

    private MetricsWriter() {}

    public static File getMetricsDir(Context context) {
        File base = context.getExternalFilesDir(null);
        if (base == null) {
            base = context.getFilesDir();
        }
        File dir = new File(base, METRICS_SUBDIR);
        if (!dir.exists() && !dir.mkdirs()) {
            Log.w(TAG, "Could not create metrics dir: " + dir.getAbsolutePath());
        }
        return dir;
    }

    public static String shortHash(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 4; i++) {
                sb.append(String.format(Locale.US, "%02x", digest[i]));
            }
            return sb.toString();
        } catch (Exception e) {
            return Integer.toHexString(input.hashCode());
        }
    }

    public static final class StageMetrics {
        public final String domain;
        public final double parseMs;
        public final double vectorizeMs;
        public final double inferenceMs;
        public final float score;
        public final long memDeltaBytes;

        public StageMetrics(String domain, double parseMs, double vectorizeMs,
                            double inferenceMs, float score, long memDeltaBytes) {
            this.domain = domain;
            this.parseMs = parseMs;
            this.vectorizeMs = vectorizeMs;
            this.inferenceMs = inferenceMs;
            this.score = score;
            this.memDeltaBytes = memDeltaBytes;
        }
    }

    public static final class ScanMetrics {
        public String scanId = UUID.randomUUID().toString();
        public long timestampMs = System.currentTimeMillis();
        public String trigger = "download";
        public String apkName;
        public String apkPath;
        public long apkSizeBytes;
        public final List<StageMetrics> stages = new ArrayList<>();
        public double wallMs;
        public double cpuMs;
        public long memDeltaBytes;
        public int totalDexFilesFound;
        public long structuralParsingTimeMs;
        public Float ensembleScore;
        public String ensembleDecision;
        public String ensemblePolicy = "sequential_mean";
    }

    private static JSONObject buildScanObject(ScanMetrics scan) throws Exception {
        JSONObject obj = new JSONObject();
        obj.put("scan_id", scan.scanId);
        obj.put("timestamp_ms", scan.timestampMs);

        String hash = shortHash(scan.apkName + ":" + scan.apkSizeBytes);
        JSONObject device = new JSONObject();
        device.put("model", Build.MODEL != null ? Build.MODEL : "unknown");
        device.put("manufacturer", Build.MANUFACTURER != null ? Build.MANUFACTURER : "unknown");
        device.put("api", Build.VERSION.SDK_INT);
        obj.put("device", device);

        JSONObject apk = new JSONObject();
        apk.put("name", scan.apkName);
        if (scan.apkPath != null) {
            apk.put("path", scan.apkPath);
        }
        apk.put("size_bytes", scan.apkSizeBytes);
        apk.put("sha256_short", hash);
        obj.put("apk", apk);

        obj.put("trigger", scan.trigger);

        JSONArray stages = new JSONArray();
        for (StageMetrics s : scan.stages) {
            JSONObject stage = new JSONObject();
            stage.put("domain", s.domain);
            stage.put("parse_ms", s.parseMs);
            stage.put("vectorize_ms", s.vectorizeMs);
            stage.put("inference_ms", s.inferenceMs);
            stage.put("score", s.score);
            stage.put("mem_delta_bytes", s.memDeltaBytes);
            stages.put(stage);
        }
        obj.put("stages", stages);

        if (scan.ensembleScore != null) {
            JSONObject ensemble = new JSONObject();
            ensemble.put("score", scan.ensembleScore);
            ensemble.put("decision", scan.ensembleDecision != null ? scan.ensembleDecision : "uncertain");
            ensemble.put("policy", scan.ensemblePolicy);
            obj.put("ensemble", ensemble);
        }

        JSONObject totals = new JSONObject();
        totals.put("wall_ms", scan.wallMs);
        totals.put("cpu_ms", scan.cpuMs);
        totals.put("mem_delta_bytes", scan.memDeltaBytes);
        totals.put("total_dex_files_found", scan.totalDexFilesFound);
        totals.put("structural_parsing_time_ms", scan.structuralParsingTimeMs);
        totals.put("battery_pct_delta", JSONObject.NULL);
        obj.put("totals", totals);
        return obj;
    }

    public static File writeScan(Context context, ScanMetrics scan) throws Exception {
        File dir = getMetricsDir(context);
        File out = new File(dir, AGGREGATE_FILENAME);
        JSONObject root;
        JSONArray scans;

        if (out.exists()) {
            try (FileInputStream fis = new FileInputStream(out)) {
                byte[] bytes = new byte[(int) out.length()];
                int offset = 0;
                while (offset < bytes.length) {
                    int count = fis.read(bytes, offset, bytes.length - offset);
                    if (count < 0) break;
                    offset += count;
                }
                String text = offset > 0
                        ? new String(bytes, 0, offset, StandardCharsets.UTF_8)
                        : "";
                root = text.trim().isEmpty() ? new JSONObject() : new JSONObject(text);
            }
        } else {
            root = new JSONObject();
        }

        if (!root.has("device")) {
            JSONObject device = new JSONObject();
            device.put("model", Build.MODEL != null ? Build.MODEL : "unknown");
            device.put("manufacturer", Build.MANUFACTURER != null ? Build.MANUFACTURER : "unknown");
            device.put("api", Build.VERSION.SDK_INT);
            root.put("device", device);
        }

        scans = root.optJSONArray("scans");
        if (scans == null) {
            scans = new JSONArray();
            root.put("scans", scans);
        }
        scans.put(buildScanObject(scan));

        try (OutputStreamWriter w = new OutputStreamWriter(new FileOutputStream(out), StandardCharsets.UTF_8)) {
            w.write(root.toString(2));
            w.write('\n');
        }

        Log.i(TAG, "Wrote metrics: " + out.getAbsolutePath());
        return out;
    }
}

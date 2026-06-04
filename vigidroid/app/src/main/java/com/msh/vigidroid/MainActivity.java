package com.msh.vigidroid;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import android.content.BroadcastReceiver;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.button.MaterialButton;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    public static final String ACTION_SCAN_LOG = "SCAN_LOG";
    public static final String ACTION_SCAN_RESULT = "SCAN_RESULT";

    private static final int COL_APK_DP = 168;
    private static final int COL_VERDICT_DP = 132;
    private static final int COL_TIME_DP = 84;
    private static final int COL_MEM_DP = 96;

    private TextView txtStatus, txtLog, txtMetricsPath, txtEmptyResults;
    private TableLayout tableResults;
    private HorizontalScrollView scrollResults;
    private MaterialButton btnStartScan, btnOpenMetrics, btnViewFullLog;
    private View contentMetrics, contentLog;
    private ImageView iconMetricsExpand, iconLogExpand;

    private boolean tableHeaderAdded;
    private int resultRowCount;
    private File metricsDir;

    private final BroadcastReceiver scanReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (ACTION_SCAN_RESULT.equals(intent.getAction())) {
                appendResultRow(intent);
                return;
            }
            if (ACTION_SCAN_LOG.equals(intent.getAction())) {
                String log = intent.getStringExtra("log");
                if (log != null) {
                    appendLog(log);
                }
                String status = intent.getStringExtra("status");
                if (status != null) {
                    setStatus(status);
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bindViews();
        setupCollapsibles();
        setupMetricsPath();
        requestAllFileAccess();

        btnStartScan.setOnClickListener(v -> startScan());

        IntentFilter filter = new IntentFilter();
        filter.addAction(ACTION_SCAN_LOG);
        filter.addAction(ACTION_SCAN_RESULT);
        LocalBroadcastManager.getInstance(this).registerReceiver(scanReceiver, filter);
    }

    private void bindViews() {
        txtStatus = findViewById(R.id.txtStatus);
        txtLog = findViewById(R.id.txtLog);
        txtMetricsPath = findViewById(R.id.txtMetricsPath);
        txtEmptyResults = findViewById(R.id.txtEmptyResults);
        tableResults = findViewById(R.id.tableResults);
        scrollResults = findViewById(R.id.scrollResults);
        btnStartScan = findViewById(R.id.btnStartScan);
        btnOpenMetrics = findViewById(R.id.btnOpenMetrics);
        btnViewFullLog = findViewById(R.id.btnViewFullLog);
        contentMetrics = findViewById(R.id.contentMetrics);
        contentLog = findViewById(R.id.contentLog);
        iconMetricsExpand = findViewById(R.id.iconMetricsExpand);
        iconLogExpand = findViewById(R.id.iconLogExpand);
    }

    private void setupCollapsibles() {
        findViewById(R.id.headerMetrics).setOnClickListener(v ->
                toggleSection(contentMetrics, iconMetricsExpand));
        findViewById(R.id.headerLog).setOnClickListener(v ->
                toggleSection(contentLog, iconLogExpand));

        btnOpenMetrics.setOnClickListener(v -> downloadMetrics());
        btnViewFullLog.setOnClickListener(v -> showFullLogDialog());
    }

    private void setupMetricsPath() {
        metricsDir = MetricsWriter.getMetricsDir(this);
        String pkg = getPackageName();
        txtMetricsPath.setText(
                metricsDir.getAbsolutePath()
                        + "\n\nDevice File Explorer:\n"
                        + "sdcard/Android/data/" + pkg + "/files/metrics/");
    }

    private void startScan() {
        tableResults.removeAllViews();
        tableHeaderAdded = false;
        resultRowCount = 0;
        txtEmptyResults.setVisibility(View.VISIBLE);
        scrollResults.setVisibility(View.GONE);
        txtLog.setText("");
        appendLog("Manual scan requested");
        setStatus(getString(R.string.status_scanning));

        Intent i = new Intent(MainActivity.this, ScanService.class);
        i.putExtra("manual_trigger", true);
        ScanService.enqueueWork(MainActivity.this, i);
    }

    private void toggleSection(View content, ImageView icon) {
        boolean show = content.getVisibility() != View.VISIBLE;
        content.setVisibility(show ? View.VISIBLE : View.GONE);
        icon.setImageResource(show ? R.drawable.ic_expand_less : R.drawable.ic_expand_more);
    }

    private void setStatus(String status) {
        if ("Idle".equalsIgnoreCase(status) || "Running".equalsIgnoreCase(status)) {
            txtStatus.setText(getString(R.string.status_idle));
        } else if (status.toLowerCase(Locale.US).contains("scan")
                || "Parsing".equalsIgnoreCase(status)) {
            txtStatus.setText(getString(R.string.status_scanning));
        } else if ("Error".equalsIgnoreCase(status)) {
            txtStatus.setText(status);
            txtStatus.setTextColor(ContextCompat.getColor(this, R.color.verdict_malware));
        } else {
            txtStatus.setText(status);
            txtStatus.setTextColor(ContextCompat.getColor(this, R.color.text_primary));
        }
        if (!"Error".equalsIgnoreCase(status)) {
            txtStatus.setTextColor(ContextCompat.getColor(this, R.color.text_primary));
        }
    }

    private void appendLog(String line) {
        CharSequence current = txtLog.getText();
        if (getString(R.string.log_empty).contentEquals(current)) {
            txtLog.setText(line + "\n");
        } else {
            txtLog.append(line + "\n");
        }
    }

    private void downloadMetrics() {
        if (metricsDir == null) {
            Toast.makeText(this, R.string.metrics_missing, Toast.LENGTH_SHORT).show();
            return;
        }
        File source = new File(metricsDir, MetricsWriter.AGGREGATE_FILENAME);
        if (!source.exists()) {
            Toast.makeText(this, R.string.metrics_missing, Toast.LENGTH_SHORT).show();
            return;
        }

        File downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        File target = new File(downloads, MetricsWriter.AGGREGATE_FILENAME);
        try (FileInputStream fis = new FileInputStream(source);
             FileOutputStream fos = new FileOutputStream(target)) {
            byte[] buf = new byte[8192];
            int read;
            while ((read = fis.read(buf)) != -1) {
                fos.write(buf, 0, read);
            }
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null) {
                clipboard.setPrimaryClip(ClipData.newPlainText("metrics_path", target.getAbsolutePath()));
            }
            Toast.makeText(this, getString(R.string.metrics_downloaded, target.getAbsolutePath()), Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.metrics_download_failed, e.getMessage()), Toast.LENGTH_LONG).show();
        }
        toggleSection(contentMetrics, iconMetricsExpand);
    }

    private void showFullLogDialog() {
        TextView body = new TextView(this);
        body.setText(txtLog.getText());
        body.setPadding(dp(16), dp(12), dp(16), dp(12));
        body.setTextIsSelectable(true);
        body.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f);
        body.setTypeface(Typeface.MONOSPACE);
        body.setTextColor(ContextCompat.getColor(this, R.color.text_secondary));

        new AlertDialog.Builder(this)
                .setTitle(R.string.system_log_title)
                .setView(body)
                .setPositiveButton(android.R.string.ok, null)
                .show();
    }

    private void ensureTableHeader() {
        if (tableHeaderAdded) {
            return;
        }
        String[] headers = {
                getString(R.string.col_apk),
                getString(R.string.col_verdict),
                getString(R.string.col_time),
                getString(R.string.col_memory)
        };
        int[] widths = {COL_APK_DP, COL_VERDICT_DP, COL_TIME_DP, COL_MEM_DP};

        TableRow row = new TableRow(this);
        row.setBackgroundColor(ContextCompat.getColor(this, R.color.table_header_bg));
        for (int i = 0; i < headers.length; i++) {
            row.addView(headerCell(headers[i], widths[i]));
        }
        tableResults.addView(row);
        tableHeaderAdded = true;
    }

    private void appendResultRow(Intent intent) {
        ensureTableHeader();
        txtEmptyResults.setVisibility(View.GONE);
        scrollResults.setVisibility(View.VISIBLE);

        String apkName = intent.getStringExtra("apk_name");
        float ensemble = intent.getFloatExtra("ensemble_score", -1f);
        String decision = intent.getStringExtra("ensemble_decision");
        double totalMs = intent.getDoubleExtra("total_ms", 0);
        double totalMemMb = intent.getDoubleExtra("total_mem_mb", 0);
        String metricsFile = intent.getStringExtra("metrics_file");

        TableRow row = new TableRow(this);
        int bg = (resultRowCount % 2 == 0)
                ? R.color.table_row_odd
                : R.color.table_row_even;
        row.setBackgroundColor(ContextCompat.getColor(this, bg));
        resultRowCount++;

        row.addView(apkCell(apkName != null ? apkName : "?", COL_APK_DP));
        row.addView(verdictBadgeCell(decision, ensemble, COL_VERDICT_DP));
        row.addView(dataCell(String.format(Locale.US, "%.0f ms", totalMs), COL_TIME_DP, Gravity.END));
        row.addView(dataCell(String.format(Locale.US, "%.2f MB", totalMemMb), COL_MEM_DP, Gravity.END));

        tableResults.addView(row);

        if (metricsFile != null) {
            appendLog("Saved: " + metricsFile);
        }
        setStatus(getString(R.string.status_idle));
    }

    private TextView headerCell(String text, int widthDp) {
        TextView tv = dataCell(text, widthDp, Gravity.CENTER);
        tv.setTypeface(null, Typeface.BOLD);
        tv.setTextColor(ContextCompat.getColor(this, R.color.text_primary));
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f);
        return tv;
    }

    private TextView apkCell(String apkName, int widthDp) {
        TextView tv = dataCell(apkName, widthDp, Gravity.START | Gravity.CENTER_VERTICAL);
        tv.setSingleLine(true);
        tv.setEllipsize(android.text.TextUtils.TruncateAt.MIDDLE);
        tv.setTextColor(ContextCompat.getColor(this, R.color.text_primary));
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f);
        return tv;
    }

    private TextView dataCell(String text, int widthDp, int gravity) {
        TextView tv = new TextView(this);
        tv.setText(text);
        tv.setGravity(gravity);
        tv.setPadding(dp(10), dp(12), dp(10), dp(12));
        TableRow.LayoutParams lp = new TableRow.LayoutParams(dp(widthDp), ViewGroup.LayoutParams.WRAP_CONTENT);
        tv.setLayoutParams(lp);
        return tv;
    }

    private LinearLayout verdictBadgeCell(String decision, float ensemble, int widthDp) {
        LinearLayout wrap = new LinearLayout(this);
        wrap.setOrientation(LinearLayout.HORIZONTAL);
        wrap.setGravity(Gravity.CENTER);
        wrap.setPadding(dp(6), dp(8), dp(6), dp(8));
        TableRow.LayoutParams lp = new TableRow.LayoutParams(dp(widthDp), ViewGroup.LayoutParams.WRAP_CONTENT);
        wrap.setLayoutParams(lp);

        if (decision == null || ensemble < 0f) {
            TextView dash = new TextView(this);
            dash.setText("—");
            dash.setGravity(Gravity.CENTER);
            wrap.addView(dash);
            return wrap;
        }

        int bgRes;
        int textColorRes;
        int iconRes;
        String label;

        switch (decision) {
            case "malware":
                bgRes = R.drawable.bg_badge_malware;
                textColorRes = R.color.verdict_malware;
                iconRes = R.drawable.ic_verdict_malware;
                label = getString(R.string.verdict_malware);
                break;
            case "uncertain":
                bgRes = R.drawable.bg_badge_uncertain;
                textColorRes = R.color.verdict_uncertain;
                iconRes = R.drawable.ic_verdict_uncertain;
                label = getString(R.string.verdict_uncertain);
                break;
            default:
                bgRes = R.drawable.bg_badge_benign;
                textColorRes = R.color.verdict_benign;
                iconRes = R.drawable.ic_verdict_benign;
                label = getString(R.string.verdict_benign);
                break;
        }

        LinearLayout badge = new LinearLayout(this);
        badge.setOrientation(LinearLayout.HORIZONTAL);
        badge.setGravity(Gravity.CENTER_VERTICAL);
        badge.setBackgroundResource(bgRes);
        badge.setPadding(dp(10), dp(6), dp(12), dp(6));

        ImageView icon = new ImageView(this);
        icon.setImageResource(iconRes);
        LinearLayout.LayoutParams iconLp = new LinearLayout.LayoutParams(dp(16), dp(16));
        iconLp.setMarginEnd(dp(6));
        icon.setLayoutParams(iconLp);
        badge.addView(icon);

        TextView tv = new TextView(this);
        tv.setText(label);
        tv.setTextColor(ContextCompat.getColor(this, textColorRes));
        tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f);
        tv.setTypeface(null, Typeface.BOLD);
        badge.addView(tv);

        wrap.addView(badge);
        return wrap;
    }

    private int dp(int value) {
        return Math.round(TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, value, getResources().getDisplayMetrics()));
    }

    @Override
    protected void onDestroy() {
        LocalBroadcastManager.getInstance(this).unregisterReceiver(scanReceiver);
        super.onDestroy();
    }

    private void requestAllFileAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            }
        }
    }
}

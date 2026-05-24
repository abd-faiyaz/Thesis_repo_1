package com.msh.vigidroid;

import androidx.appcompat.app.AppCompatActivity;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;

public class MainActivity extends AppCompatActivity {

    private TextView txtStatus, txtLog;
    private Button btnStartScan;

    private final BroadcastReceiver logReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String log = intent.getStringExtra("log");
            if (log != null) {
                txtLog.append(log + "\n");
            }

            String status = intent.getStringExtra("status");
            if (status != null) {
                txtStatus.setText("Status: " + status);
            }

            String apkName = intent.getStringExtra("apk_name");
            if (apkName != null) {
                // optionally highlight current file
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        requestAllFileAccess();

        txtStatus = findViewById(R.id.txtStatus);
        txtLog = findViewById(R.id.txtLog);
        btnStartScan = findViewById(R.id.btnStartScan);

        btnStartScan.setOnClickListener(v -> {
            txtLog.append("Manual scan requested\n");
            txtStatus.setText("Status: Scanning");

            Intent i = new Intent(MainActivity.this, ScanService.class);
            i.putExtra("manual_trigger", true);
            ScanService.enqueueWork(MainActivity.this, i);
        });

        showMetricsPullHint();

        LocalBroadcastManager.getInstance(this)
                .registerReceiver(logReceiver, new IntentFilter("SCAN_LOG"));
    }

    @Override
    protected void onDestroy() {
        LocalBroadcastManager.getInstance(this).unregisterReceiver(logReceiver);
        super.onDestroy();
    }

    private void showMetricsPullHint() {
        File metricsDir = MetricsWriter.getMetricsDir(this);
        txtLog.append("Metrics dir: " + metricsDir.getAbsolutePath() + "\n");
        txtLog.append("Pull to PC: adb pull \"" + metricsDir.getAbsolutePath()
                + "/\" Shared_pipeline_Files/results/device/\n");
    }

    private void requestAllFileAccess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            } else {
                Toast.makeText(this, "Permission already granted", Toast.LENGTH_SHORT).show();
            }
        }
    }
}

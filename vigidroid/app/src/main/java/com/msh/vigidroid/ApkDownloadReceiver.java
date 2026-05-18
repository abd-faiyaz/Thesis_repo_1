package com.msh.vigidroid;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;

public class ApkDownloadReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {

        Uri data = intent.getData();
        if (data != null && data.toString().endsWith(".apk")) {

            Intent serviceIntent = new Intent(context, ScanService.class);
            serviceIntent.putExtra("manual_trigger", false);
            ScanService.enqueueWork(context, serviceIntent);
        }
    }
}
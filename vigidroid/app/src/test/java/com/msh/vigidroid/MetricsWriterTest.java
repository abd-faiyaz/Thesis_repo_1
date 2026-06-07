package com.msh.vigidroid;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public class MetricsWriterTest {

  @Test
  public void stageMetrics_modelIdOptional() {
    MetricsWriter.StageMetrics legacy =
        new MetricsWriter.StageMetrics("manifest_xgb", 1.0, 2.0, 3.0, 0.5f, 100L);
    assertNull(legacy.modelId);

    MetricsWriter.StageMetrics hybrid =
        new MetricsWriter.StageMetrics(
            ModelRegistry.BROADCAST_MLDP_HYBRID.domain,
            ModelRegistry.BROADCAST_MLDP_HYBRID.modelId,
            4.0,
            5.0,
            6.0,
            0.73f,
            200L);
    assertEquals("broadcast_mldp_hybrid", hybrid.modelId);
    assertEquals("manifest_mldp_perm_receiver_actions", hybrid.domain);
  }

  @Test
  public void cascadeStageMetrics_exposesModeBandDexFields() {
    MetricsWriter.StageMetrics modeB =
        MetricsWriter.StageMetrics.cascade(
            MldpDexHeaderModeBOnnxRunner.DOMAIN,
            ModelRegistry.MLDP_DEXHEADER_CASCADE_MODE_B.modelId,
            "B",
            1.0,
            0.0,
            0.5,
            2.0,
            0.91f,
            -1f,
            true,
            0.91f,
            42L);
    assertEquals("B", modeB.mode);
    assertEquals(0.0, modeB.dexMs, 1e-9);
    assertTrue(modeB.earlyExit);
    assertEquals(0.91f, modeB.stage1Score, 1e-6f);
  }
}

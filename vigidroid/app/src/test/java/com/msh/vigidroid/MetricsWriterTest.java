package com.msh.vigidroid;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

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
}

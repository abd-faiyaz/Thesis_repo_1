package com.msh.vigidroid;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ModelRegistryTest {

  @Test
  public void linregdroid_assetsMatchExportManifest() {
    assertEquals("linregdroid_permission", ModelRegistry.LINREGDROID_PERMISSION.modelId);
    assertEquals("manifest_permissions", ModelRegistry.LINREGDROID_PERMISSION.domain);
    assertEquals(LinRegDroidOnnxRunner.MODEL_ID, ModelRegistry.LINREGDROID_PERMISSION.modelId);
    assertEquals(LinRegDroidOnnxRunner.DOMAIN, ModelRegistry.LINREGDROID_PERMISSION.domain);
    assertTrue(ModelRegistry.LINREGDROID_PERMISSION.assetsPrefix.contains("linregdroid_permission"));
  }

  @Test
  public void mldp_assetsMatchExportManifest() {
    assertEquals("mldp_pruned_permission", ModelRegistry.MLDP_PRUNED_PERMISSION.modelId);
    assertEquals("manifest_permissions_mldp", ModelRegistry.MLDP_PRUNED_PERMISSION.domain);
    assertEquals(MldpPrunedOnnxRunner.MODEL_ID, ModelRegistry.MLDP_PRUNED_PERMISSION.modelId);
    assertEquals(MldpPrunedOnnxRunner.DOMAIN, ModelRegistry.MLDP_PRUNED_PERMISSION.domain);
    assertTrue(ModelRegistry.MLDP_PRUNED_PERMISSION.assetsPrefix.contains("mldp_pruned_permission"));
  }

  @Test
  public void broadcastMldpHybrid_assetsMatchExportManifest() {
    assertEquals("broadcast_mldp_hybrid", ModelRegistry.BROADCAST_MLDP_HYBRID.modelId);
    assertEquals("manifest_mldp_perm_receiver_actions", ModelRegistry.BROADCAST_MLDP_HYBRID.domain);
    assertEquals(
        BroadcastMldpHybridOnnxRunner.MODEL_ID, ModelRegistry.BROADCAST_MLDP_HYBRID.modelId);
    assertEquals(
        BroadcastMldpHybridOnnxRunner.DOMAIN, ModelRegistry.BROADCAST_MLDP_HYBRID.domain);
    assertTrue(
        ModelRegistry.BROADCAST_MLDP_HYBRID.assetsPrefix.contains("broadcast_mldp_hybrid"));
  }
}

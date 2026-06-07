package com.msh.vigidroid;

/**
 * Canonical {@code model_id} → assets path → scan {@code domain} mapping.
 * Asset subdirs under {@code app/src/main/assets/models/{model_id}/} must match
 * {@code model_id} in each bundle's {@code export_manifest.json} (fusion / multistep).
 */
public final class ModelRegistry {

  public static final class Entry {
    public final String modelId;
    public final String domain;
    public final String assetsPrefix;

    Entry(String modelId, String domain, String assetsPrefix) {
      this.modelId = modelId;
      this.domain = domain;
      this.assetsPrefix = assetsPrefix;
    }
  }

  public static final Entry LINREGDROID_PERMISSION =
      new Entry(
          LinRegDroidOnnxRunner.MODEL_ID,
          LinRegDroidOnnxRunner.DOMAIN,
          "models/linregdroid_permission/");

  public static final Entry MLDP_PRUNED_PERMISSION =
      new Entry(
          MldpPrunedOnnxRunner.MODEL_ID,
          MldpPrunedOnnxRunner.DOMAIN,
          "models/mldp_pruned_permission/");

  public static final Entry BROADCAST_MLDP_HYBRID =
      new Entry(
          BroadcastMldpHybridOnnxRunner.MODEL_ID,
          BroadcastMldpHybridOnnxRunner.DOMAIN,
          "models/broadcast_mldp_hybrid/");

  public static final Entry MLP_HEADER =
      new Entry(MlpHeaderOnnxRunner.MODEL_ID, MlpHeaderOnnxRunner.DOMAIN, "models/mlp_header/");

  public static final Entry PATTERN_A_COMBINED =
      new Entry(PatternAOnnxRunner.MODEL_ID, PatternAOnnxRunner.DOMAIN, "models/pattern_a_combined/");

  public static final Entry PATTERN_B_DUAL_BRANCH =
      new Entry(
          PatternBOnnxRunner.MODEL_ID, PatternBOnnxRunner.DOMAIN, "models/pattern_b_dual_branch/");

  /** MoMo + thesis ONNX bundles registered in ScanService stages[]. */
  public static final Entry[] REGISTERED_MODELS = {
    LINREGDROID_PERMISSION,
    MLDP_PRUNED_PERMISSION,
    BROADCAST_MLDP_HYBRID,
    MLP_HEADER,
    PATTERN_A_COMBINED,
    PATTERN_B_DUAL_BRANCH,
  };

  private ModelRegistry() {}
}

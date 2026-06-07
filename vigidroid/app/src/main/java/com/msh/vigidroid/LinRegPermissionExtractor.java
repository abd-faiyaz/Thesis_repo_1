package com.msh.vigidroid;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * LinRegDroid A1: manifest permissions only, normalized to permissions:: tokens,
 * binary float[1, 173] aligned with features/permission_vocab.json.
 * Matches Python linear/src/features/permission_vector.py.
 */
public final class LinRegPermissionExtractor {

  public static final int FEATURE_DIM = 173;
  public static final String VOCAB_ASSET =
      "models/linregdroid_permission/features/permission_vocab.json";

  private final Map<String, Integer> tokenToIndex;
  private final int vectorSize;

  public static final class ExtractionResult {
    public final float[] vector;
    public final int permissionCount;
    public final long extractNanos;
    public final long vectorizeNanos;

    ExtractionResult(float[] vector, int permissionCount, long extractNanos, long vectorizeNanos) {
      this.vector = vector;
      this.permissionCount = permissionCount;
      this.extractNanos = extractNanos;
      this.vectorizeNanos = vectorizeNanos;
    }
  }

  public LinRegPermissionExtractor(Map<String, Integer> tokenToIndex, int vectorSize) {
    this.tokenToIndex = tokenToIndex;
    this.vectorSize = vectorSize;
  }

  public static LinRegPermissionExtractor fromAssets(Context context) throws Exception {
    String json = ModelAssetHelper.readAssetText(context, VOCAB_ASSET);
    JSONObject root = new JSONObject(json);
    JSONArray permissions = root.getJSONArray("permissions");
    int vectorSize = root.optInt("M", permissions.length());
    if (vectorSize != FEATURE_DIM) {
      throw new IllegalStateException("Expected vocab M " + FEATURE_DIM + ", got " + vectorSize);
    }
    Map<String, Integer> tokenToIndex = PermissionNormalizer.buildTokenToIndex(permissions);
    return new LinRegPermissionExtractor(tokenToIndex, vectorSize);
  }

  public ExtractionResult extract(File apkFile) throws Exception {
    long t0 = System.nanoTime();
    List<String> tokens = PermissionNormalizer.readNormalizedPermissions(apkFile);
    long t1 = System.nanoTime();
    float[] vector = buildBinaryVector(tokens);
    long t2 = System.nanoTime();
    return new ExtractionResult(vector, tokens.size(), t1 - t0, t2 - t1);
  }

  float[] buildBinaryVector(List<String> normalizedTokens) {
    return PermissionNormalizer.buildBinaryVector(normalizedTokens, tokenToIndex, vectorSize);
  }
}

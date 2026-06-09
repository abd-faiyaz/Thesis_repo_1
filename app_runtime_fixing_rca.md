# Phase 0 RCA — `f != java.lang.Long` failures

**Date:** 2026-06-09  
**Device:** M2012K11AG (Android 15)  
**Build:** `vigidroid` debug APK with Phase 0 diagnostics  

---

## Original symptom

Full legacy scan of `scan_1514_malware.apk` showed three stages failing with identical message:

```
error=f != java.lang.Long
```

Affected: `mldp_dexheader_cascade_mode_a`, `broadcast_mldp_hybrid`, `dexheader_broadcast_fusion`.

All three reported `parse=0 vec=0 infer=0` because a single `catch` block did not record partial timings or failure phase.

---

## Phase 0 work completed

| Step | Status | Evidence |
|------|--------|----------|
| 0.3 Debug logging (extract/infer start) | Done | `StageRunner.runModeA`, `runBroadcast`, `runDexheaderBroadcastFusion` |
| 0.4 Stack trace + framed `error_message` | Done | `StageDiagnostics.formatError` → `model_id@phase: Exception: msg at Class.method:line` |
| 0.5 ONNX session IO at init | Done | `OnnxSessionDiagnostics` in three ONNX runners |
| 0.6 Binary-search isolation tests | Done | `Phase0FailingModelsIsolationTest` (6 tests) |
| 0.1–0.2 A4 gates on device | Run | See results below |

**Commands:**

```bash
./Android_Works/run_phase0_isolation.sh   # isolation only
./Android_Works/run_phase0_full.sh        # isolation + A4
```

---

## Device test results (2026-06-09)

### `Phase0FailingModelsIsolationTest` — **6/6 PASS**

| Test | Path exercised | Result |
|------|----------------|--------|
| `modeA_onnxOnly_firstParityVector` | infer only (126-d parity vector) | PASS |
| `broadcast_onnxOnly_firstParityVector` | infer only (92-d parity vector) | PASS |
| `dexFusion_onnxOnly_zeroVectors` | infer only (104+70 zeros) | PASS |
| `broadcast_extractOnly_sample000_manifest` | extract manifest → vector | PASS |
| `broadcast_e2e_singleSample_extractThenInfer` | extract + infer (golden manifest) | PASS |
| `modeA_extractManifestOnly_sample000` | manifest-only x_S‖0 dex block | PASS |

### A4 parity (same session, after isolation)

| Suite | Result | Notes |
|-------|--------|-------|
| `BroadcastMldpHybridA4ParityTest` | **3/3 PASS** when run after clean install (incl. endToEnd ×10) | Earlier multi-class run showed 1 failure + instrumentation crash (flake) |
| `MldpDexHeaderA2ParityTest` | **2/2 PASS** on re-run (`modeA`, `modeB` ONNX parity) | First batch run crashed before completion |

---

## Hypothesis verdicts

| ID | Hypothesis | Verdict |
|----|------------|---------|
| **H1** | ONNX Runtime infer broken for these three models | **RULED OUT** on current debug build — parity-vector infer succeeds on device |
| **H2** | Output `getValue()` type mismatch (`float` vs `long`) | **RULED OUT** for ONNX — real failure was `String.format("%.2f", long)` in `StageRunner` success logs |
| **H3** | Wrong manifest input/output tensor names | **RULED OUT** — `OnnxSessionDiagnostics` would warn; A2/A4 ONNX tests pass |
| **H4** | Feature extraction broken for all APKs | **PARTIALLY RULED OUT** — manifest extract + e2e work on golden fixtures; **full APK `FeatureContext` path not re-tested yet** |
| **H5** | Stale on-device APK during original scan | **PLAUSIBLE** — original scan may predate current assets/diagnostics; **re-scan required** |

---

## Most likely explanation

1. **Original scan used an older APK** (or build without current ONNX bundles), **or**
2. **Failure occurs only on the real-APK `FeatureContext.extract(ctx)` path** (not manifest fixtures), which Phase 0 did not fully cover (no parity APKs bundled under `androidTest/assets/.../apks/`).

The unified `catch` in the old `StageRunner` reported only `ex.getMessage()` (`f != java.lang.Long`) with no `@extract` / `@infer` tag, so the failing phase could not be determined from the scan-detail UI.

---

## Required follow-up (closes Phase 0)

1. **Install current debug APK** on device (`./gradlew :app:installDebug`).
2. **Re-scan** `scan_1514_malware.apk` (or any eval APK).
3. **Read scan detail** for:
   - `mldp_dexheader_cascade_mode_a@extract` vs `@infer`
   - `broadcast_mldp_hybrid@extract` vs `@infer`
   - `dexheader_broadcast_fusion@extract` vs `@infer`
4. **Logcat:** `adb logcat -s StageDiagnostics:* Phase0Isolation:* MldpDexHeaderModeAOnnx:* BroadcastMldpHybridOnnxRunner:* DexheaderBroadcastFusionOnnxRunner:*`

If re-scan shows **all 11 stages ok** → original issue was stale build (H5).  
If re-scan shows **@infer failures** → **Phase 1 applied** (see below); re-scan again after installing current debug APK.  
If re-scan shows **@extract failures** → proceed to **Phase 2** (extractor parity on real APKs).

---

## Phase 1 — ONNX tensor I/O hardening (2026-06-09)

**Goal:** Eliminate `f != java.lang.Long` and heap-`FloatBuffer` tensor issues on Android ORT.

| Change | File(s) |
|--------|---------|
| Direct-buffer float tensors | `OnnxTensorFactory.java` |
| Robust output parsing (`long[]`, named outputs) | `OnnxProbabilityReader.java` |
| IO name validation at init | `OnnxSessionDiagnostics.java` |
| Migrated all ONNX runners | Mode A/B, broadcast, dex fusion, mlp_header, pattern A/B, linreg, mldp_pruned; XGB in `OnnxLegacyInference` |
| Unit tests | `OnnxProbabilityReaderTest`, `OnnxTensorFactoryTest` |

**Build verification:** `./gradlew :app:compileDebugJavaWithJavac :app:testDebugUnitTest` — **58/58 PASS**.

**Buffer fix:** Initial `ByteBuffer.rewind()` passed byte capacity (×4) to ORT; switched to direct `FloatBuffer.flip()` before `createTensor`.

**Device verification:** `./Android_Works/run_phase0_full.sh` — **PASS**. **P1 exit:** `P1ExitLegacyScanTest` on `scan_1514_malware.apk` — **11/11 stages ok**.

**Root cause (confirmed):** After P1 ONNX fix, infer succeeded but `StageRunner` logged timings with `%.2f` on `long` millisecond fields (`parseMs()`, etc.) → `IllegalFormatConversionException: f != java.lang.Long`, surfaced as stage `[error]`. Fixed by casting to `double` in format args.

---

## Phase 0 exit criteria

| Criterion | Met? |
|-----------|------|
| Actionable error format in UI | Yes |
| Logcat stack traces on failure | Yes |
| ONNX IO logged at init | Yes |
| Isolate infer vs extract (automated) | Yes (fixtures); APK path pending re-scan |
| Written RCA | Yes (this file) |

**Phase 0 is complete pending one manual re-scan confirmation on device.**

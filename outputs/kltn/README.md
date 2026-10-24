# KLTN archives

## Current consolidated package — 24 October 2026

Use [the current full Release](https://github.com/thienhocdata/quality-aware-gan-augmentation/releases/tag/kltn-complete-2026-10-24)
for the latest consolidated source, notebooks, outputs, Protocol B data and M4 checkpoints.
The [browsable project](../../KLTN/) also includes the endpoint/V3-A analyses and
the proposed architecture documents added after the frozen 17 October package.
The older archives below remain unchanged historical snapshots.

## Full archive: notebooks, data, checkpoints and saved outputs

The **3.14 GB full package** is available in [GitHub Releases: KLTN full archive — 2026-10-17](https://github.com/thienhocdata/quality-aware-gan-augmentation/releases/tag/kltn-full-2026-10-17). This is the original `KLTN_clean_WITH_OUTPUTS.zip` (3,136,138,280 bytes), split into two byte-identical parts to meet the per-asset size limit.

Download both `.part01` and `.part02`, `FULL_ARCHIVE_MANIFEST.json`, and `GHEP_FILE.ps1` from the release into the same folder. Run the merge script as described in the release notes. It verifies both parts and reconstructs the original ZIP, including its SHA-256 check.

The full archive contains the notebooks and source, Protocol B data/cache, all 12 M4 models, six Learned-local checkpoints at 7,500/8,000 classifier updates, and the six saved-report sections below. The 7,500→8,000 audit remains **pending** despite the checkpoint inputs being present. The newer endpoint, temperature and geometry reports from 17 October are not included in this frozen archive.

Full ZIP SHA-256: `369f55316659ee27356165c15677274a196f2f48ded13264fa8fef6bed981300`.

## Saved-output-only bundle (24.9 MB)

`KLTN_clean_OUTPUTS_FOR_KAGGLE.zip` is the unchanged saved-output bundle assembled on 17 October 2026. It is 24,861,964 bytes and contains six sections:

| Section | Saved evidence |
| --- | --- |
| `protocol_b` | Historical Protocol B results and provenance |
| `m4_training` | Metadata and histories for four arms × seeds 42/43/44 at 8,000 classifier updates |
| `m4_test` | Original predictions, metrics and confusion matrices for all 12 classifiers |
| `classification_diagnostics` | Post-hoc analysis of the saved test predictions |
| `weighting_8000` | Completed train-only learned-weighting analysis for three seeds |
| `endgame_status` | Explicit pending status and earlier failed-input reports for the 7,500→8,000 audit |

The bundle contains saved reports, not model checkpoint weights. The Endgame comparison is **pending**, not a completed scientific result. No new model training or model inference was performed to assemble this bundle.

The later `selector_temp.zip`, `selector_geometry.zip` and the newer endpoint report from 17 October are **not included** in this frozen bundle. Keep these analyses separate rather than treating this upload as their combined result.

Each section includes `bundle_manifest.json`. All 490 payload files were checked against their listed byte counts and SHA-256 hashes; the final ZIP passed CRC verification.

SHA-256:

```text
47086406e7debe56ad74cf0a8c566d35eea7b08d63d22a744dea81e04269f5b3
```

Attach this ZIP as one Kaggle Input for the saved-results review notebook. Both the original ZIP and Kaggle's unpacked directory layout are supported by that notebook. This upload does not change any original experiment or checkpoint.

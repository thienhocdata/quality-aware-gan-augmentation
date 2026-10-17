# KLTN consolidated saved outputs

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

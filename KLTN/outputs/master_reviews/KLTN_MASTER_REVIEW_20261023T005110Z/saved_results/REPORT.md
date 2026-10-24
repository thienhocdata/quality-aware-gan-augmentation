# Consolidated saved-results review

CPU read-only review of saved outputs. No training, model construction, checkpoint modification or test inference.

## M4 saved-test results

| Arm | Macro-F1 mean ± sample SD (%) | OA mean ± sample SD (%) |
|---|---:|---:|
| original_gan | 81.663 ± 2.877 | 87.552 ± 0.444 |
| matched_ros | 81.485 ± 0.967 | 87.143 ± 0.422 |
| random_local | 82.226 ± 3.004 | 88.018 ± 1.199 |
| learned_local | 83.009 ± 3.015 | 88.251 ± 1.229 |

## Boundaries

- B has cross-split spatial patch reuse; this consolidation does not eliminate spatial leakage.
- The Indian Pines test was viewed previously. Further design is post-hoc motivated.
- Three trained seeds are the replication units; test pixels, anchors and noise draws are not independent training runs.
- The weighting report is a final frozen endpoint observation; it does not establish a cause for seed 43 test performance.
- No completed 7500-to-8000 forward report was found. Its six checkpoints exist but the temporal diagnostic remains pending.
- M3 standalone report not located. This does not invalidate or recreate the existing M4 artifacts.

"""CPU-only algebra examples for the PDF review, not GAN/TF tests or data analysis."""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(20261024)
u = rng.normal(size=(121, 4))
v = rng.normal(size=(4, 40))
t = u @ v / 2.0
radius = 0.25
eps = 1e-12
delta = radius * t / np.sqrt(radius**2 + np.square(t).sum() + eps)

checker = ((np.indices((11, 11)).sum(axis=0) % 2) * 2 - 1).reshape(121, 1)
spectrum = np.zeros((1, 40))
spectrum[0, 0] = 1.0
checker_residual = checker @ spectrum

gauge = 100.0
same_t = (gauge * u) @ (v / gauge) / 2.0
# For L(T)=sum(T), parameter gradients transform under the gauge while T stays the same.
grad_u = np.ones((121, 40)) @ v.T / 2.0
grad_v = u.T @ np.ones((121, 40)) / 2.0
scaled_grad_u = grad_u / gauge
scaled_grad_v = grad_v * gauge

report = {
    "kind": "synthetic_numpy_algebra_only",
    "tensorflow_executed": False,
    "training_executed": False,
    "real_data_loaded": False,
    "parameters": {
        "A0_output_head": (256 + 1) * 4840,
        "A1_output_heads": (256 + 1) * (484 + 160),
        "head_parameter_ratio_A0_over_A1": 4840 / 644,
    },
    "low_rank_and_envelope": {
        "rank_before_envelope": int(np.linalg.matrix_rank(t)),
        "rank_after_envelope": int(np.linalg.matrix_rank(delta)),
        "delta_norm": float(np.linalg.norm(delta)),
        "radius": radius,
        "strict_bound": bool(np.linalg.norm(delta) < radius),
    },
    "low_rank_not_spatial_smoothness": {
        "checkerboard_rank": int(np.linalg.matrix_rank(checker_residual)),
        "neighbor_sign_flips": True,
    },
    "factor_scaling_symmetry": {
        "scale_U": gauge,
        "scale_V": 1 / gauge,
        "max_abs_output_difference": float(np.max(np.abs(same_t - t))),
        "grad_U_norm_ratio": float(np.linalg.norm(scaled_grad_u) / np.linalg.norm(grad_u)),
        "grad_V_norm_ratio": float(np.linalg.norm(scaled_grad_v) / np.linalg.norm(grad_v)),
    },
    "envelope_gain_at_origin_with_epsilon_1e_minus_8": [
        {"radius": r, "gain": float(r / np.sqrt(r*r + 1e-8))}
        for r in (1.0, 1e-4, 1e-8)
    ],
}
assert report["low_rank_and_envelope"]["rank_before_envelope"] <= 4
assert report["low_rank_and_envelope"]["rank_after_envelope"] <= 4
assert report["low_rank_and_envelope"]["strict_bound"]
assert report["low_rank_not_spatial_smoothness"]["checkerboard_rank"] == 1
target = Path(__file__).resolve().parent / "algebra_checks.json"
target.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

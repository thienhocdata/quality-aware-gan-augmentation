"""V3-A fresh M4 extension: signed per-head distance bias, no test access.

The immutable frozen M4 module is imported after checking its SHA. Its C/D/G,
Dense initialization, route implementation, loss, schedule, shared Adam, and
native checkpoint helpers are reused. This file replaces only the local head,
adds observational runtime gates, and controls a 200-C-update prefix of the
same fixed 8000-C-update run. No TensorFlow import occurs at module import.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

M4_SHA256 = "6d33ca6a0d39600f5f44c0d7e63547b1237149d625471ac73315d982ffbd8fb5"
METHOD = "V3A_SIGNED_ANCHOR_DISTANCE_BIAS_V1"
EPSILON = 1e-8
TARGET = 8000
PILOT = 200


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_m4(path):
    require(file_sha(path) == M4_SHA256, "Frozen M4 source SHA mismatch")
    spec = importlib.util.spec_from_file_location("_v3a_locked_m4", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def standardize_neighbor_distances(distance, epsilon=EPSILON):
    """Actual V3-A normalization; safe to verify on CPU without TensorFlow."""
    import numpy as np
    distance = np.asarray(distance, dtype=np.float64)
    require(distance.ndim == 2 and distance.shape[1] == 3,
            "Distance standardization expects N by3")
    require(np.isfinite(distance).all() and np.all(distance >= 0),
            "Distances must be finite nonnegative Euclidean distances")
    require(float(epsilon) == EPSILON, "V3-A epsilon is fixed at1e-8")
    mean = distance.mean(axis=1, keepdims=True)
    std = distance.std(axis=1, keepdims=True, ddof=0)
    q64 = (distance - mean) / np.maximum(std, epsilon)
    tied = np.all(distance == distance[:, :1], axis=1)
    q64[tied] = 0.0
    require(np.isfinite(q64).all(), "Nonfinite standardized distances")
    return q64.astype(np.float32), std[:, 0]


def configure(m4, options, seed, folder, start):
    import numpy as np
    max_seconds = float(options.get("max_worker_seconds", 4 * 3600))
    require(max_seconds > 0, "max_worker_seconds must be positive")
    m4.np = np
    m4.START = start
    m4.FOLDER = folder
    m4.CONFIG = {
        "data_mode": "auto_real_B", "b_run": options["b_run"],
        "stage": "final", "arm": "learned_local", "seed": seed,
        "worker_sha256": M4_SHA256, "target_c_updates": TARGET,
        "checkpoint_every_c_updates": 500, "log_every_loops": 50,
        "attempt_dir": str(folder), "attempt_id": f"v3a_seed_{seed}",
        "execution_mode": "v3a", "max_worker_seconds": max_seconds,
        "checkpoint_safety_seconds": float(options.get("checkpoint_safety_seconds", 180)),
        "deadline_monotonic": start + max_seconds + 600,
        "resume_from": None,
    }
    m4.REPORT = {
        "status": "STARTED", "complete": False, "method": METHOD,
        "arm": "learned_local", "seed": seed, "stage": "final",
        "route_counts": {}, "fake_patch_hashes": [], "G_anchor_consumption": [],
        "random_weight_consumption": [], "test_opened": False,
        "validation_opened": False, "scope": "V3-A fresh train-only fixed-budget extension",
        "frozen_m4_sha256": M4_SHA256, "v3a_source_sha256": file_sha(__file__),
    }
    return np


def patch_head(m4):
    """Only numerical model change: zero-initialized signed distance scalar."""
    np, tf, keras = m4.np, m4.tf, m4.keras

    class DistanceBiasedLocalMix(keras.layers.Layer):
        def __init__(self, support, neighbors, **kwargs):
            super().__init__(**kwargs)
            n = len(support)
            flat64 = support.reshape(n, m4.FLAT).astype(np.float64)
            # Same Euclidean metric/data as M4 graph construction. Row-wise
            # recomputation avoids changing graph tie breaking or row ordering.
            distance = np.linalg.norm(flat64[:, None, :] - flat64[neighbors], axis=2)
            standardized, std = standardize_neighbor_distances(distance)
            self.distance_np = distance
            self.distance_std_np = std
            self.q_np = standardized
            self.supports = tf.constant(support.reshape(n, m4.FLAT), dtype=tf.float32)
            self.neighbors = tf.constant(neighbors, dtype=tf.int32)
            self.standardized = tf.constant(self.q_np, dtype=tf.float32)
            self.beta = self.add_weight(name="distance_beta", shape=(),
                                        initializer="zeros", dtype="float32", trainable=True)

        def weights_and_rows(self, logits, anchor_ids):
            ids = tf.reshape(tf.cast(anchor_ids, tf.int32), [-1])
            allowed = tf.gather(self.neighbors, ids)
            selected_logits = tf.gather(logits, allowed, axis=1, batch_dims=1)
            selected_distance = tf.gather(self.standardized, ids)
            weights = tf.nn.softmax(selected_logits - self.beta * selected_distance, axis=1)
            return weights, ids, allowed

        def call(self, inputs):
            weights, ids, allowed = self.weights_and_rows(*inputs)
            anchor = tf.gather(self.supports, ids)
            neighbors = tf.gather(self.supports, allowed)
            return (1.0 - m4.RHO) * anchor + m4.RHO * tf.einsum("bk,bkp->bp", weights, neighbors)

    def build_v3a_head(support, neighbors, rho=m4.RHO):
        require(rho == 0.2 and m4.K == 3, "V3-A keeps rho=.2 and k=3")
        n = len(support)
        require(support.shape[1:] == m4.SHAPE and neighbors.shape == (n, 3), "Wrong head geometry")
        features = keras.layers.Input(shape=(784,), name="generator_features")
        anchors = keras.layers.Input(shape=(1,), dtype="int32", name="head_local_anchor_id")
        logits = keras.layers.Dense(n, activation=None, name="support_logits")(features)
        mixed = DistanceBiasedLocalMix(support, neighbors, name="distance_biased_mix")([logits, anchors])
        output = keras.layers.Reshape(m4.SHAPE, name="synthetic_patch")(mixed)
        return keras.Model([features, anchors], output, name=f"head_v3a_{n}")

    m4.tao_head_v2 = build_v3a_head


def mixer(head):
    return head.get_layer("distance_biased_mix")


def patch_gradient_observer(m4):
    """Preserve M4's identity transform; record beta connectivity separately."""
    original = m4.make_gradient_audit
    np, tf = m4.np, m4.tf

    def make_audit(owners):
        audit, observe = original(owners)
        beta_refs = {}
        head_refs = {}
        for h in range(15):
            variables = owners[f"head_{h}"]["trainable"]
            betas = [v for v in variables if v.shape.rank == 0 and "distance_beta" in v.name]
            require(len(betas) == 1 and len(variables) == 3, f"Missing tracked beta in head {h}")
            beta_refs[betas[0].ref()] = h
            for variable in variables:
                if variable.ref() != betas[0].ref():
                    head_refs[variable.ref()] = h
        with tf.device("/CPU:0"):
            audit.update(
                beta_gradient_seen=tf.Variable(np.zeros(15, np.int64), trainable=False),
                beta_gradient_sq_total=tf.Variable(np.zeros(15, np.float64), trainable=False),
                beta_gradient_last=tf.Variable(np.zeros(15, np.float64), trainable=False),
                dense_gradient_sq_total=tf.Variable(np.zeros(15, np.float64), trainable=False),
            )

        def wrapped(grads_and_vars):
            pairs = list(grads_and_vars)
            base_pairs = observe(pairs)
            beta_gradient, beta_seen = {}, set()
            dense = [[] for _ in range(15)]
            for gradient, variable in pairs:
                if gradient is None:
                    continue
                if variable.ref() in beta_refs:
                    h = beta_refs[variable.ref()]
                    beta_gradient[h] = tf.reshape(tf.cast(gradient, tf.float64), [])
                    beta_seen.add(h)
                elif variable.ref() in head_refs:
                    dense[head_refs[variable.ref()]].append(tf.reduce_sum(tf.square(tf.cast(gradient, tf.float64))))
            values = tf.stack([beta_gradient.get(h, tf.constant(0.0, tf.float64)) for h in range(15)])
            seen = tf.constant([int(h in beta_seen) for h in range(15)], dtype=tf.int64)
            dense_values = tf.stack([tf.add_n(parts) if parts else tf.constant(0.0, tf.float64) for parts in dense])
            ops = [audit["beta_gradient_seen"].assign_add(seen),
                   audit["beta_gradient_sq_total"].assign_add(tf.square(values)),
                   audit["beta_gradient_last"].assign(values),
                   audit["dense_gradient_sq_total"].assign_add(dense_values)]
            with tf.control_dependencies(ops):
                return [(None if gradient is None else tf.identity(gradient), variable)
                        for gradient, variable in base_pairs]
        return audit, wrapped

    m4.make_gradient_audit = make_audit


def owner_snapshot(m4, model):
    """Bounded pilot evidence: include BN buffers and both Adam slots."""
    result = {}
    for name, owner in model["owners"].items():
        values = []
        for part in ("trainable", "nontrainable"):
            for variable in owner[part]:
                values.append(variable.numpy().copy())
                if part == "trainable":
                    values.extend(model["opt"].get_slot(variable, key).numpy().copy() for key in ("m", "v"))
        result[name] = values
    return result


def patch_route_gate(m4, evidence):
    original = m4.update_gate
    original_infer = m4.infer_fake
    np = m4.np

    def update_gate(model, label, fn, active, family):
        checking = label.startswith("loop0/")
        before = owner_snapshot(m4, model) if checking else None
        result = original(model, label, fn, active, family)
        owner_names = list(model["owners"])
        if family in ("GC", "GD"):
            evidence["G_gradient_sq_total"] += float(model["audit"]["norm_sq"].numpy()[owner_names.index("G")])
        if checking:
            after = owner_snapshot(m4, model)
            unchanged = []
            for owner in owner_names:
                if owner not in active:
                    require(len(before[owner]) == len(after[owner]) and all(
                        np.array_equal(a, b) for a, b in zip(before[owner], after[owner])),
                        f"Actual pilot route {label} changed frozen {owner} weights/BN/Adam slots")
                    unchanged.append(owner)
            evidence["actual_route_freeze_checks"].append({
                "route": label, "active": sorted(active), "unchanged_owners": unchanged,
                "weights_BN_and_Adam_slots_checked": True,
                "clock_after": int(model["opt"].iterations.numpy()),
            })
        return result
    m4.update_gate = update_gate

    def infer_fake(model, item, arm, spec):
        checking = item["loop"] == 0
        before = m4.state_hashes(model) if checking else None
        result = original_infer(model, item, arm, spec)
        if checking:
            require(before == m4.state_hashes(model), "Actual C fake-slot inference changed model/BN/Adam/dropout/audit state")
            evidence["actual_fake_inference_state_unchanged"] = True
        return result
    m4.infer_fake = infer_fake


def dense_initialization(m4, model):
    return m4.arrays_sha(variable.numpy() for head in model["heads"]
                         for variable in head.get_layer("support_logits").trainable_variables)


def beta_record(m4, model, updates):
    np = m4.np
    values = np.array([float(mixer(head).beta.numpy()) for head in model["heads"]], dtype=np.float64)
    require(np.isfinite(values).all(), "Nonfinite beta")
    return {"c_updates": int(updates), "optimizer_clock": int(model["opt"].iterations.numpy()),
            "class_ids_original": (model["order"][:15] + 1).tolist(),
            "signed_beta": values.tolist(),
            "beta_gradient_seen": model["audit"]["beta_gradient_seen"].numpy().tolist(),
            "beta_gradient_sq_total": model["audit"]["beta_gradient_sq_total"].numpy().tolist(),
            "dense_gradient_sq_total": model["audit"]["dense_gradient_sq_total"].numpy().tolist(),
            "beta_unchanged_is_not_an_automatic_failure": True}


def selector_probe(m4, model, contract, updates):
    """Fixed train-only endpoint observation; never a selection criterion.

    Seven anchors per head and eight normal noise vectors are private keyed
    inputs, common across all V3 seeds and the200/8000 endpoints. The same eight
    noises are used for every anchor in a class to expose anchor interactions.
    """
    np, tf = m4.np, m4.tf
    out = m4.FOLDER / "selector_probes"
    out.mkdir(exist_ok=True)
    target = out / f"probe_{updates:05d}.npz"
    csv_path = out / f"probe_{updates:05d}_by_class.csv"
    manifest_path = out / f"probe_{updates:05d}.json"
    before = m4.state_hashes(model)
    state_signature = m4.json_sha(before)
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(saved.get("complete") is True and saved["state_signature"] == state_signature
                and saved["contract_sha256"] == file_sha(m4.FOLDER / "contract.json")
                and saved["npz_sha256"] == file_sha(target)
                and saved["csv_sha256"] == file_sha(csv_path), "Existing selector probe is stale or corrupt")
        return saved
    arrays, rows, input_hashes = {}, [], []
    for h, head in enumerate(model["heads"]):
        layer = mixer(head)
        n = len(model["supports"][h])
        require(n >= 7, "Fixed probe requires at least7 training supports")
        private = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261019, 6303, h])))
        anchors = private.choice(n, size=7, replace=False).astype(np.int32)
        noise8 = private.normal(size=(8, 100)).astype(np.float32)
        noise = np.tile(noise8, (7, 1))
        ids = np.repeat(anchors, 8)[:, None]
        condition = np.zeros((56, 16), np.float32)
        condition[:, h] = 1.0
        features = model["G"]([noise, condition], training=False)
        logits = head.get_layer("support_logits")(features)
        weights, _, allowed = layer.weights_and_rows(logits, ids)
        raw = tf.gather(logits, allowed, axis=1, batch_dims=1)
        q = tf.gather(layer.standardized, tf.reshape(ids, [-1]))
        corrected = raw - layer.beta * q
        patch = head([features, ids], training=False).numpy().reshape(7, 8, m4.FLAT)
        w64 = weights.numpy().reshape(7, 8, 3).astype(np.float64)
        raw64 = raw.numpy().reshape(7, 8, 3).astype(np.float64)
        corrected64 = corrected.numpy().reshape(7, 8, 3).astype(np.float64)
        feature64 = features.numpy().reshape(7, 8, 784).astype(np.float64)
        require(np.isfinite(patch).all() and np.isfinite(raw64).all()
                and np.isfinite(corrected64).all() and np.isfinite(w64).all(), "Nonfinite selector probe")
        require(np.all(w64 >= 0) and np.allclose(w64.sum(axis=-1), 1.0, atol=2e-7), "Probe weights not simplex")
        entropy = -np.sum(np.where(w64 > 0, w64 * np.log(np.clip(w64, 1e-300, 1.0)), 0.0), axis=-1)
        unique = np.array([len(np.unique(w64[a].argmax(axis=1))) for a in range(7)], dtype=np.int32)
        patch64 = patch.astype(np.float64)
        patch_rms = np.sqrt(np.mean(np.square(patch64 - patch64.mean(axis=1, keepdims=True)), axis=(1, 2)))
        feature_rms = np.sqrt(np.mean(np.square(feature64 - feature64.mean(axis=1, keepdims=True)), axis=(1, 2)))
        sort_raw = np.sort(raw64, axis=-1)
        raw_top2_gap = sort_raw[..., -1] - sort_raw[..., -2]
        q7 = layer.q_np[anchors]
        beta = float(layer.beta.numpy())
        bias_range = abs(beta) * (q7.max(axis=1).astype(np.float64) - q7.min(axis=1))
        row = {"head": h, "class_id": int(model["order"][h] + 1), "c_updates": updates,
               "supports": n, "anchors": 7, "noise_per_anchor": 8, "beta_signed": beta,
               "entropy_mean": float(entropy.mean()), "effective_supports_mean": float(np.exp(entropy).mean()),
               "max_weight_mean": float(w64.max(axis=-1).mean()),
               "weight_noise_sd_mean": float(w64.std(axis=1).mean()),
               "anchors_argmax_changes_fraction": float((unique > 1).mean()),
               "patch_noise_rms_mean": float(patch_rms.mean()),
               "G_feature_noise_rms_mean": float(feature_rms.mean()),
               "selected_raw_logit_noise_sd_mean": float(raw64.std(axis=1).mean()),
               "selected_corrected_logit_noise_sd_mean": float(corrected64.std(axis=1).mean()),
               "raw_top2_logit_gap_mean": float(raw_top2_gap.mean()),
               "absolute_distance_bias_range_mean": float(bias_range.mean()),
               "bias_range_over_raw_top2_gap_mean": float((bias_range[:, None] / np.maximum(raw_top2_gap, 1e-12)).mean())}
        rows.append(row)
        prefix = f"head_{h:02d}_"
        arrays.update({prefix + "anchor_ids": anchors, prefix + "neighbors": model["neighbor_graphs"][h][anchors],
                       prefix + "noise8": noise8, prefix + "standardized_distance": q7,
                       prefix + "beta": np.array(beta, dtype=np.float32),
                       prefix + "G_features": features.numpy().reshape(7, 8, 784),
                       prefix + "all_raw_logits": logits.numpy().reshape(7, 8, n),
                       prefix + "selected_raw_logits": raw.numpy().reshape(7, 8, 3),
                       prefix + "corrected_logits": corrected.numpy().reshape(7, 8, 3),
                       prefix + "weights": weights.numpy().reshape(7, 8, 3),
                       prefix + "argmax_unique_across_noise": unique,
                       prefix + "entropy": entropy, prefix + "raw_top2_logit_gap": raw_top2_gap,
                       prefix + "absolute_distance_bias_range": bias_range,
                       prefix + "patch_noise_rms": patch_rms, prefix + "G_feature_noise_rms": feature_rms})
        input_hashes.append({"head": h, "anchor_ids_sha256": m4.sha_array(anchors),
                             "noise8_sha256": m4.sha_array(noise8),
                             "neighbors_sha256": m4.sha_array(model["neighbor_graphs"][h][anchors]),
                             "q_sha256": m4.sha_array(q7)})
    require(before == m4.state_hashes(model), "Train-only selector probe changed model/BN/Adam/dropout/audit state")
    temporary = target.with_name(target.name + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(target)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {"complete": True, "method": METHOD, "seed": m4.CONFIG["seed"], "c_updates": updates,
              "scope": "fixed_train_only_descriptive_probe_no_model_selection_no_C_or_D_scores",
              "heads": 15, "anchors_per_head": 7, "noises_per_anchor": 8,
              "noise_distribution": "normal_0_1_float32_PCG64_private_SeedSequence[20261019,6303,head]",
              "same_noises_across_anchors_seeds_and_endpoints": True,
              "inputs": input_hashes, "state_signature": state_signature,
              "contract_sha256": file_sha(m4.FOLDER / "contract.json"),
              "npz_sha256": file_sha(target), "csv_sha256": file_sha(csv_path),
              "all_model_BN_Adam_dropout_audit_state_unchanged": True,
              "extra_optimizer_updates": 0, "test_opened": False, "validation_opened": False,
              "raw_full_patches_saved": False,
              "patch_variability": "RMS_over_4840_values_of_patch_minus_8_noise_mean_at_fixed_anchor",
              "classification_benefit_or_cause_inferred": False}
    atomic_json(manifest_path, record)
    print("TRAIN-ONLY SELECTOR PROBE", m4.CONFIG["seed"], updates, "15heads x7anchors x8noise", flush=True)
    return record


def startup_forward_gate(m4, model, spec):
    np, tf = m4.np, m4.tf
    rows = []
    item = m4.frame(spec, 0)
    before = m4.state_hashes(model)
    for h, head in enumerate(model["heads"]):
        layer = mixer(head)
        require(float(layer.beta.numpy()) == 0.0, "Fresh beta is not zero")
        private = item["g"][h]
        ids = private["anchors"]
        features = model["G"]([private["noise"], private["condition"]], training=False)
        logits = head.get_layer("support_logits")(features)
        weights, anchors, allowed = layer.weights_and_rows(logits, ids)
        baseline_weights = tf.nn.softmax(tf.gather(logits, allowed, axis=1, batch_dims=1), axis=1)
        expected = (1.0 - m4.RHO) * tf.gather(layer.supports, anchors) + m4.RHO * tf.einsum(
            "bk,bkp->bp", baseline_weights, tf.gather(layer.supports, allowed))
        actual = tf.reshape(head([features, ids], training=False), [-1, m4.FLAT])
        require(np.array_equal(weights.numpy(), baseline_weights.numpy()), "beta0 weights differ from exact V2 formula")
        require(np.allclose(actual.numpy(), expected.numpy(), rtol=0, atol=2e-7), "beta0 synthetic patch differs from V2 formula")
        w = weights.numpy()
        require(np.isfinite(w).all() and np.all(w >= 0) and np.allclose(w.sum(axis=1), 1.0, atol=2e-7), "Weights not simplex")
        a = model["supports"][h][ids[:, 0]].reshape(len(ids), -1)
        neighbor = model["supports"][h][model["neighbor_graphs"][h][ids[:, 0]]].reshape(len(ids), 3, -1)
        movement = np.linalg.norm(actual.numpy().astype(np.float64) - a, axis=1)
        bound = m4.RHO * np.linalg.norm(neighbor.astype(np.float64) - a[:, None], axis=2).max(axis=1)
        require(np.all(movement <= bound + 2e-5 * (1 + bound)), "Startup convex movement bound failed")
        rows.append({"head": h, "class_id": int(model["order"][h] + 1),
                     "beta0_exact_weight_equivalence": True, "patch_max_abs_delta": float(np.max(np.abs(actual.numpy() - expected.numpy()))),
                     "samples": int(len(ids)), "geometry_pass": True,
                     "distance_std_min": float(layer.distance_std_np.min()),
                     "distance_std_median": float(np.median(layer.distance_std_np)),
                     "distance_std_below_epsilon": int((layer.distance_std_np < EPSILON).sum())})
    require(before == m4.state_hashes(model), "Startup inference changed model/BN/Adam/dropout state")
    return {"pass": True, "beta0_compared_to": "same_fresh_Dense_logits_exact_M4_V2_formula",
            "extra_optimizer_updates": 0, "head_checks": rows,
            "whole_state_unchanged": True, "not_a_classification_test": True}


def make_contract(m4, model, spec, initialization, options):
    baseline = m4.job_contract(model, spec)
    baseline.update({
        "method": METHOD, "architecture_version": 1,
        "source_m4_sha256": M4_SHA256, "sha_worker": file_sha(__file__),
        "training_origin": "fresh_M4_seeds_beta_zero_not_V2_warm_start",
        "beta_policy": "unconstrained_signed_scalar_per_minority_head",
        "beta_initializer": "zeros", "beta_count": 15,
        "distance_metric": "Euclidean_full_4840_float64_same_training_split",
        "distance_standardization": "(d-mean_of_3)/max(population_std_of_3,epsilon); exact_ties_zero",
        "distance_epsilon": EPSILON,
        "distance_sha256": [m4.sha_array(mixer(head).distance_np) for head in model["heads"]],
        "standardized_distance_sha256": [m4.sha_array(mixer(head).q_np) for head in model["heads"]],
        "initial_C_trainable_sha256": initialization["C_trainable_sha256"],
        "initial_D_trainable_sha256": initialization["D_trainable_sha256"],
        "initial_G_trainable_sha256": initialization["G_trainable_sha256"],
        "initial_head_Dense_sha256": initialization["head_Dense_only_sha256"],
        "test_access_in_worker": False, "validation_selection_in_worker": False,
        "comparison_scope": "posthoc_Protocol_B_known_test_spatial_overlap_not_causal_mechanism_proof",
    })
    return plain(baseline)


def compare_original_metadata(m4, contract, options, seed):
    paths = options.get("original_checkpoint_metas", {})
    path = paths.get(str(seed), paths.get(seed))
    if not path:
        return {"available": False, "historical_paired_identical_claimed": False,
                "note": "Historical initialization/consumed trace not verified"}
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = record.get("contract", record)
    require(expected.get("arm") == "learned_local" and expected.get("stage") == "final"
            and expected.get("seed") == seed and record.get("c_updates", 8000) == 8000,
            "Original reference is not final V2 learned-local seed8000")
    fields = ("protocol_B_id", "implementation_B", "stage", "seed", "budget_C_updates",
              "train_X_sha256", "train_y_sha256", "class_order",
              "graph_sha256", "rho", "k", "optimizer_lr", "beta1", "batch_size")
    differences = {key: {"V3": contract.get(key), "M4": expected.get(key)}
                   for key in fields if plain(contract.get(key)) != plain(expected.get(key))}
    require(not differences, "V3/M4 recipe/provenance mismatch: " + json.dumps(differences))
    reported = ("tf", "tf_keras", "numpy", "schedule_sha256")
    runtime_schedule_differences = {
        key: {"V3": contract.get(key), "M4": expected.get(key)}
        for key in reported if plain(contract.get(key)) != plain(expected.get(key))}
    return {"available": True, "original_meta_sha256": file_sha(path), "matched_fields": list(fields),
            "historical_runtime_schedule_differences": runtime_schedule_differences,
            "historical_exact_schedule_descriptor_sha_matches": contract["schedule_sha256"] == expected.get("schedule_sha256"),
            "historical_paired_identical_claimed": False,
            "note": "Matched recipe only. Runtime/NumPy-dependent descriptor differences are reported, not normalized away. Final M4 metadata does not establish initial weights or consumed dropout/noise trace."}


def shared_runtime_identity(m4, options, seed):
    """Pin the actual new V3 runtime across all three jobs, without installs."""
    runtime = m4.REPORT["runtime"]
    require(runtime["tensorflow"].split(".")[:2] == ["2", "20"]
            and runtime["tf_keras"].split(".")[:2] == ["2", "20"]
            and m4.np.__version__.split(".")[0] == "2",
            "V3-A requires installed compatible TensorFlow/tf-keras2.20.x and NumPy2.x; no automatic install")
    identity = {"tensorflow": runtime["tensorflow"], "tf_keras": runtime["tf_keras"],
                "numpy": m4.np.__version__, "mixed_precision_policy": "float32",
                "op_determinism_enabled": True,
                "gpu_device_name": runtime["gpu_details"].get("device_name"),
                "gpu_compute_capability": runtime["gpu_details"].get("compute_capability")}
    path = Path(options["base"]) / "runtime_identity.json"
    if path.is_file():
        require(json.loads(path.read_text(encoding="utf-8")) == plain(identity),
                "Actual runtime/GPU type changed across the new V3 runs; refuse mixed-runtime continuation")
    else:
        require(seed == 42, "Seed42 pilot must pin the common runtime before other seeds start")
        atomic_json(path, identity)
    return identity


def setup(options, seed, folder, start):
    m4 = import_m4(options["m4_worker_path"])
    np = configure(m4, options, seed, folder, start)
    X, y, source = m4.get_training_data()
    m4.record_provenance(X, y, source)
    m4.REPORT["runtime"] = plain(m4.load_runtime())
    m4.REPORT["shared_runtime_identity"] = shared_runtime_identity(m4, options, seed)
    patch_head(m4)
    patch_gradient_observer(m4)
    evidence = {"actual_route_freeze_checks": [], "G_gradient_sq_total": 0.0,
                "actual_fake_inference_state_unchanged": False}
    m4._v3a_evidence = evidence
    patch_route_gate(m4, evidence)
    with m4.tf.device("/GPU:0"):
        model = m4.build_models(X, y, "learned_local", seed)
    initialization = m4.initial_fingerprints(model)
    initialization["head_Dense_only_sha256"] = dense_initialization(m4, model)
    initialization["beta_initial_values"] = [float(mixer(head).beta.numpy()) for head in model["heads"]]
    require(initialization["beta_initial_values"] == [0.0] * 15, "Invalid initial beta")
    # This original M4 helper hashes all head trainables. Rename it to avoid
    # presenting a hash containing beta as the M4 Dense-only comparator.
    initialization["head_all_trainables_including_beta_sha256"] = initialization.pop("head_dense_trainable_sha256")
    spec = m4.make_schedule(model["y_sorted"], seed=seed, batch_size=m4.BATCH,
                            latent_dim=100, g_points=7, n_heads=15,
                            class_order=(model["order"] + 1).tolist())
    require(spec["samples"] == 5528 and spec["n_batches"] == 56 and spec["tail_batch_size"] == 28, "Final M4 schedule changed")
    contract = make_contract(m4, model, spec, initialization, options)
    historical = compare_original_metadata(m4, contract, options, seed)
    startup = startup_forward_gate(m4, model, spec)
    # All beta variables must belong to the head and own m/v slots before read.
    for h, head in enumerate(model["heads"]):
        beta = mixer(head).beta
        require(any(v.ref() == beta.ref() for v in model["owners"][f"head_{h}"]["trainable"]), "Unowned beta")
        for slot in ("m", "v"):
            require(model["opt"].get_slot(beta, slot) is not None, "Missing beta Adam slot")
    atomic_json(folder / "initialization.json", initialization)
    atomic_json(folder / "startup_forward_gate.json", startup)
    atomic_json(folder / "original_M4_comparison.json", historical)
    prior = folder / "contract.json"
    if prior.is_file():
        require(json.loads(prior.read_text(encoding="utf-8")) == contract, "Existing V3-A job contract differs")
    atomic_json(prior, contract)
    m4.REPORT.update(initialization=initialization, original_M4_comparison=historical,
                     startup_forward_gate=startup, method=METHOD)
    return m4, X, y, model, spec, contract, evidence


def save_checkpoint(m4, model, contract, next_loop, history, digest):
    """Use M4 atomic native checkpoint protocol; safely accept exact duplicate."""
    updates = 2 * next_loop
    folder = m4.FOLDER
    base = folder / "state"
    destination = base / f"ckpt_{updates:05d}"
    if destination.is_dir():
        rec = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
        require(rec["contract"] == contract and rec["c_updates"] == updates
                and rec["state_hashes"] == m4.state_hashes(model), "Existing checkpoint differs from live state")
        for name, expected in rec["weights_hashes"].items():
            require(file_sha(destination / name) == expected, "Existing checkpoint component corrupted")
        atomic_json(base / "latest.json", {"name": destination.name, "c_updates": updates,
                                          "meta_sha256": file_sha(destination / "meta.json")})
    else:
        # Include Python-side observed prefix evidence inside the atomically
        # committed metadata itself. A hard interruption after pointer commit
        # cannot leave the latest checkpoint detached from its route evidence.
        original_atomic_json = m4.atomic_json
        def metadata_with_evidence(path, value):
            if Path(path).name == "meta.json":
                value = {**value, "v3_prefix_evidence": {
                    "route_counts": dict(m4.REPORT["route_counts"]),
                    "evidence": plain(m4._v3a_evidence)}}
            original_atomic_json(path, value)
        m4.atomic_json = metadata_with_evidence
        try:
            m4.ckpt_save(model, contract, next_loop, history, digest)
        finally:
            m4.atomic_json = original_atomic_json
    telemetry = beta_record(m4, model, updates)
    atomic_json(folder / "beta_telemetry" / f"beta_{updates:05d}.json", telemetry)
    if updates == PILOT:
        preserved = folder / "pilot_checkpoint"
        temporary = folder / "pilot_checkpoint.incomplete"
        if preserved.is_dir():
            require(file_sha(preserved / "meta.json") == file_sha(destination / "meta.json"),
                    "Preserved pilot checkpoint differs")
            rec = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
            for name, expected in rec["weights_hashes"].items():
                require(file_sha(preserved / name) == expected, "Preserved pilot checkpoint component corrupted")
        else:
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(destination, temporary)
            temporary.rename(preserved)
        require(file_sha(preserved / "meta.json") == file_sha(destination / "meta.json"),
                "Preserved pilot metadata copy hash mismatch")
        rec = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
        for name, expected in rec["weights_hashes"].items():
            require(file_sha(preserved / name) == expected, "Preserved pilot component copy hash mismatch")
    atomic_json(folder / "prefix_evidence.json", {
        "c_updates": updates, "checkpoint_meta_sha256": file_sha(destination / "meta.json"),
        "route_counts": m4.REPORT["route_counts"], "evidence": m4._v3a_evidence})
    return telemetry


def make_pilot_gate(m4, model, evidence):
    np = m4.np
    require(int(model["opt"].iterations.numpy()) == 3400, "Pilot Adam clock !=3400")
    require(int(model["audit"]["calls"].numpy()) == 3400, "Pilot actual optimizer calls !=3400")
    require(len(evidence["actual_route_freeze_checks"]) == 34, "Missing first actual loop route freeze/slot checks")
    require(evidence["actual_fake_inference_state_unchanged"], "Missing actual fake-slot inference freeze evidence")
    expected = {"C": 200, "D": 200, "GC": 1500, "GD": 1500}
    require(m4.REPORT["route_counts"] == expected, "Pilot route count mismatch")
    seen = model["audit"]["beta_gradient_seen"].numpy()
    require(np.array_equal(seen, np.full(15, 200)), "Beta not connected on each GC/GD route")
    require(evidence["G_gradient_sq_total"] > 0, "No actual nonzero G gradient observed in pilot")
    dense = model["audit"]["dense_gradient_sq_total"].numpy()
    require(np.all(dense > 0), "No actual nonzero Dense gradient observed for some head")
    require(all(np.isfinite(variable.numpy()).all() for variable in m4.state_variables(model).values()), "Nonfinite pilot state")
    return {"pass": True, "method": METHOD, "seed": m4.CONFIG["seed"], "c_updates": PILOT,
            "contract_sha256": file_sha(m4.FOLDER / "contract.json"),
            "checkpoint_meta_sha256": file_sha(m4.FOLDER / "pilot_checkpoint" / "meta.json"),
            "selector_probe_manifest_sha256": file_sha(m4.FOLDER / "selector_probes" / "probe_00200.json"),
            "optimizer_clock": 3400, "actual_route_counts": expected,
            "beta_gradient_connected_updates_per_head": seen.tolist(),
            "actual_route_freeze_checks": evidence["actual_route_freeze_checks"],
            "actual_fake_inference_state_unchanged": True,
            "G_gradient_sq_total": evidence["G_gradient_sq_total"],
            "dense_gradient_sq_total_per_head": dense.tolist(),
            "beta_magnitude_change_required": False,
            "test_opened": False, "validation_opened": False,
            "pass_scope": "technical_wiring_initial_200_updates_only_not_classification_benefit",
            "beta": beta_record(m4, model, PILOT)}


def progress(m4, model, updates, status, complete=False):
    atomic_json(m4.FOLDER / "progress.json", {
        "status": status, "complete": complete, "method": METHOD,
        "seed": m4.CONFIG["seed"], "c_updates": updates, "target_c_updates": TARGET,
        "optimizer_clock": int(model["opt"].iterations.numpy()),
        "test_opened": False, "elapsed_seconds": round(time.monotonic() - m4.START, 3)})


def verify_pilot(m4, model, contract):
    gate_path = m4.FOLDER / "pilot_gate.json"
    require(gate_path.is_file(), "Missing pilot_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    require(gate.get("pass") is True and gate.get("c_updates") == PILOT
            and gate.get("seed") == m4.CONFIG["seed"]
            and gate.get("contract_sha256") == file_sha(m4.FOLDER / "contract.json")
            and gate.get("checkpoint_meta_sha256") == file_sha(m4.FOLDER / "pilot_checkpoint" / "meta.json")
            and gate.get("selector_probe_manifest_sha256") == file_sha(m4.FOLDER / "selector_probes" / "probe_00200.json"),
            "Pilot runtime gate failed or is stale")
    next_loop, _, _ = m4.ckpt_resume(model, contract)
    require(next_loop == 100, "Fresh pilot verifier requires exact checkpoint200, not later/final checkpoint")
    actual = beta_record(m4, model, PILOT)
    require(actual == gate["beta"], "Fresh restore beta or beta-gradient telemetry changed")
    require(int(model["audit"]["calls"].numpy()) == 3400, "Fresh restore pilot audit mismatch")
    # M4 read().assert_consumed and state_hashes compare the complete live
    # model/BN/Adam/dropout/audit inventory to the committed native checkpoint.
    record = {"pass": True, "method": METHOD, "c_updates": PILOT,
              "fresh_process_full_native_read_assert_consumed": True,
              "all_committed_state_hashes_exact": True, "beta_and_Adam_slots_restored": True,
              "optimizer_clock": 3400, "extra_optimizer_updates": 0,
              "test_opened": False, "classification_benefit_claimed": False,
              "contract_sha256": file_sha(m4.FOLDER / "contract.json"),
              "pilot_gate_sha256": file_sha(gate_path),
              "pilot_checkpoint_meta_sha256": file_sha(m4.FOLDER / "pilot_checkpoint" / "meta.json")}
    atomic_json(m4.FOLDER / "pilot_restore_verification.json", record)
    m4.REPORT.update(status="PILOT_RESTORE_VERIFIED", complete=False, c_updates=PILOT,
                     pilot_restore_verification=record)
    print("PILOT RESTORE VERIFIED: all native state, beta, Adam slots; no updates", flush=True)


def train(m4, X, model, spec, contract, evidence, stop_updates):
    require(stop_updates in (PILOT, TARGET), "stop-updates must be200 or8000")
    if stop_updates == TARGET and m4.CONFIG["seed"] == 42:
        verified = m4.FOLDER / "pilot_restore_verification.json"
        require(verified.is_file(), "Seed42 FULL requires successful fresh-process pilot restore gate")
        check = json.loads(verified.read_text(encoding="utf-8"))
        require(check.get("pass") is True and check["contract_sha256"] == file_sha(m4.FOLDER / "contract.json")
                and check["pilot_gate_sha256"] == file_sha(m4.FOLDER / "pilot_gate.json")
                and check["pilot_checkpoint_meta_sha256"] == file_sha(m4.FOLDER / "pilot_checkpoint" / "meta.json"),
                "Pilot verification is stale")
    next_loop, history, digest = m4.ckpt_resume(model, contract)
    require(0 <= next_loop <= stop_updates // 2, "Resume beyond requested stopping point")
    if next_loop > 0:
        pointer = json.loads((m4.FOLDER / "state" / "latest.json").read_text(encoding="utf-8"))
        checkpoint_meta = m4.FOLDER / "state" / pointer["name"] / "meta.json"
        require(file_sha(checkpoint_meta) == pointer["meta_sha256"], "Prefix checkpoint metadata hash mismatch")
        rec = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        prefix = rec.get("v3_prefix_evidence")
        require(prefix is not None, "Cannot resume V3 checkpoint without atomically bound actual route evidence")
        evidence.update(prefix["evidence"])
        m4.REPORT["route_counts"].update(prefix["route_counts"])
    if next_loop == TARGET // 2 and (m4.FOLDER / "completed.json").is_file():
        seal = json.loads((m4.FOLDER / "completed.json").read_text(encoding="utf-8"))
        require(seal.get("complete") is True and seal["model_sha256"] == file_sha(m4.FOLDER / "classifier.weights.h5")
                and seal["job_contract_sha256"] == file_sha(m4.FOLDER / "contract.json")
                and seal["final_beta_sha256"] == file_sha(m4.FOLDER / "final_beta.json")
                and seal["selector_probe_8000_manifest_sha256"] == file_sha(m4.FOLDER / "selector_probes" / "probe_08000.json")
                and seal["native_checkpoint_meta_sha256"] == pointer["meta_sha256"], "Completed artifact seal mismatch")
        m4.REPORT.update(status="COMPLETE_REUSED", complete=True, c_updates=TARGET)
        return
    limit = m4.START + float(m4.CONFIG["max_worker_seconds"])
    recent = []
    while next_loop < stop_updates // 2:
        if time.monotonic() + max(30.0, m4.CONFIG["checkpoint_safety_seconds"]) >= limit and next_loop:
            save_checkpoint(m4, model, contract, next_loop, history, digest)
            progress(m4, model, 2 * next_loop, "PAUSED_WALLCLOCK")
            m4.REPORT.update(status="PAUSED_WALLCLOCK", complete=False, c_updates=2 * next_loop)
            return
        losses, trace = m4.train_once(model, X, spec, next_loop)
        digest = hashlib.sha256(bytes.fromhex(digest) + bytes.fromhex(m4.json_sha(trace))).hexdigest()
        next_loop += 1
        recent.append(losses)
        if next_loop % 50 == 0 or next_loop == stop_updates // 2:
            keys = sorted({key for row in recent for key in row})
            row = {"c_updates": 2 * next_loop, "adam_clock": 34 * next_loop,
                   "seconds": round(time.monotonic() - m4.START, 3),
                   **{key: float(m4.np.mean([v[key] for v in recent if key in v])) for key in keys}}
            require(all(m4.np.isfinite(row[key]) for key in keys), "Nonfinite interval loss")
            history.append(row)
            atomic_json(m4.FOLDER / "history.json", history)
            print("V3A TRAIN", m4.CONFIG["seed"], row, flush=True)
            recent = []
        if 2 * next_loop == PILOT or 2 * next_loop % 500 == 0 or next_loop == stop_updates // 2:
            save_checkpoint(m4, model, contract, next_loop, history, digest)
            if next_loop == 100:
                selector_probe(m4, model, contract, PILOT)
                # Regenerate from the current actual prefix on every arrival
                # at200; bind it only after the native checkpoint is committed.
                atomic_json(m4.FOLDER / "pilot_gate.json", make_pilot_gate(m4, model, evidence))
            progress(m4, model, 2 * next_loop, "TRAINING")
    if stop_updates == PILOT:
        # Also repairs a hard interruption after native checkpoint commit but
        # before the immutable pilot copy/telemetry was persisted.
        save_checkpoint(m4, model, contract, next_loop, history, digest)
        selector_probe(m4, model, contract, PILOT)
        atomic_json(m4.FOLDER / "pilot_gate.json", make_pilot_gate(m4, model, evidence))
        progress(m4, model, PILOT, "PILOT_COMPLETE")
        m4.REPORT.update(status="PILOT_COMPLETE", complete=False, c_updates=PILOT)
        print("PILOT COMPLETE:200 C updates; awaiting fresh-process restore gate", flush=True)
        return
    require(int(model["opt"].iterations.numpy()) == 136000
            and int(model["audit"]["calls"].numpy()) == 136000, "Final8000 optimizer/audit counters wrong")
    require(m4.np.array_equal(model["audit"]["beta_gradient_seen"].numpy(), m4.np.full(15, 8000)), "Final beta route connectivity counts wrong")
    selector_probe(m4, model, contract, TARGET)
    model_path = m4.FOLDER / "classifier.weights.h5"
    model["C"].save_weights(str(model_path))
    telemetry = beta_record(m4, model, TARGET)
    atomic_json(m4.FOLDER / "final_beta.json", telemetry)
    pointer = json.loads((m4.FOLDER / "state" / "latest.json").read_text(encoding="utf-8"))
    require(pointer["c_updates"] == TARGET, "Final native checkpoint missing")
    completed = {
        "status": "COMPLETE", "complete": True, "method": METHOD,
        "arm": "v3a_distance_bias", "seed": m4.CONFIG["seed"], "stage": "final",
        "c_updates": TARGET, "adam_clock": 136000,
        "model_sha256": file_sha(model_path),
        "job_contract_sha256": file_sha(m4.FOLDER / "contract.json"),
        "native_checkpoint_meta_sha256": pointer["meta_sha256"],
        "source_train_X_sha256": m4.REPORT["input"]["X_sha256"],
        "source_train_y_sha256": m4.REPORT["input"]["y_sha256"],
        "class_order_original_ids": (model["order"] + 1).tolist(),
        "worker_sha256": file_sha(__file__), "frozen_m4_sha256": M4_SHA256,
        "final_beta_sha256": file_sha(m4.FOLDER / "final_beta.json"),
        "selector_probe_8000_manifest_sha256": file_sha(m4.FOLDER / "selector_probes" / "probe_08000.json"),
        "test_opened": False, "validation_opened": False,
        "historical_M4_paired_identical_claimed": False,
        "elapsed_this_process_seconds": round(time.monotonic() - m4.START, 3),
    }
    atomic_json(m4.FOLDER / "completed.json", completed)
    progress(m4, model, TARGET, "COMPLETE", True)
    m4.REPORT.update(status="COMPLETE", complete=True, c_updates=TARGET, training_job=completed)
    print("V3A JOB COMPLETE", m4.CONFIG["seed"], "8000 C updates; no test in training worker", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "verify-pilot"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--stop-updates", type=int, choices=(PILOT, TARGET), default=TARGET)
    args = parser.parse_args()
    options = json.loads(Path(args.config).read_text(encoding="utf-8"))
    folder = Path(options["base"]) / "jobs" / f"seed_{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    m4 = None
    start = time.monotonic()
    try:
        require(sys.platform.startswith("linux") and Path("/kaggle/working").is_dir(), "V3-A runtime is Kaggle Linux only")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1")
                and "," not in os.environ["CUDA_VISIBLE_DEVICES"], "Exactly one assigned visible GPU per worker is required")
        m4, X, _, model, spec, contract, evidence = setup(options, args.seed, folder, start)
        if args.mode == "verify-pilot":
            require(args.seed == 42, "Bounded fresh pilot verifier is seed42")
            verify_pilot(m4, model, contract)
        else:
            train(m4, X, model, spec, contract, evidence, args.stop_updates)
    except BaseException as error:
        record = {"status": "FAILED", "complete": False, "method": METHOD,
                  "seed": args.seed, "mode": args.mode, "error_type": type(error).__name__,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "test_opened": False, "elapsed_seconds": round(time.monotonic() - start, 3)}
        if m4 is not None:
            m4.REPORT.update(record)
        atomic_json(folder / "failure.json", record)
        traceback.print_exc()
        return 1
    finally:
        if m4 is not None:
            m4.REPORT["elapsed_seconds"] = round(time.monotonic() - start, 3)
            m4.REPORT["mode"] = args.mode
            atomic_json(folder / ("pilot_verifier_report.json" if args.mode == "verify-pilot" else "worker_report.json"), m4.REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Fail-closed raw-terrain contract. No NASA weights or guessed normalization.

This is a domain-shifted Moonstone subset, not NASA-native imagery. Importing this
module needs only numpy, so input and checkpoint-shape audits can run offline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

NASA_CODE_REVISION = "d54c67aad513cb9daca444afa425cfb278e4fbf8"
NASA_MODEL_REVISION = "b657bdfe6fc4c0445d918f77a2d7b919805f3bce"
NASA_CHECKPOINT_BYTES = 2425755440
NASA_CHECKPOINT_SHA256 = "5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b"
MAPPING = {"elevation": "dtm", "slope": "slope"}
STRICT_SPATIAL_POLICY = "strict_config_v1"
NATIVE_SPATIAL_POLICY = "moonstone_native256_v1"
SPATIAL_POLICIES = (STRICT_SPATIAL_POLICY, NATIVE_SPATIAL_POLICY)
NASA_CONFIG_SHA256 = "fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619"
MOONSTONE_NOMINAL_MPP = 1737400.0 * np.pi / (180 * 128)


def spatial_policy_receipt(spatial_policy=STRICT_SPATIAL_POLICY, config_sha256=None):
    """Declare an input protocol, not evidence of model/runtime compatibility.

    Native256 is an explicit downstream transfer protocol. It does not reproduce
    NASA's pretraining data transforms, correct physical scale, or relax shape,
    missingness, checkpoint, runtime or full-cohort execution checks.
    """
    if spatial_policy not in SPATIAL_POLICIES:
        raise ValueError("unknown spatial policy")
    receipt = {"policy_id": spatial_policy, "spatial_transform": "none",
               "interpolation": "none", "test_access": "FORBIDDEN"}
    if spatial_policy == NATIVE_SPATIAL_POLICY:
        if config_sha256 != NASA_CONFIG_SHA256:
            raise ValueError("native256 requires the audited fixed-revision NASA config SHA256")
        receipt.update(config_sha256=NASA_CONFIG_SHA256,
                       raw_shape=[256, 256], model_shape=[256, 256], patch_shape=[16, 16],
                       upstream_pre_resize=[512, 512],
                       upstream_dataset_pipeline="NOT_APPLIED_DOWNSTREAM_NATIVE_GRID_PROTOCOL",
                       normalization="UNCHANGED_PINNED_CONFIG",
                       nominal_north_south_mpp=MOONSTONE_NOMINAL_MPP,
                       east_west_spacing="nominal_north_south_mpp * cos(latitude)",
                       nasa_native_scale_equivalence=False,
                       positional_embedding_compatibility="REQUIRES_STRICT_CHECKPOINT_AND_REAL_PREFLIGHT")
    return receipt


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def spatial_pair(value, field):
    if isinstance(value, int):
        value = (value, value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be an integer or a pair")
    return tuple(_positive_int(item, field) for item in value)


def _one_stat(stats, key):
    values = np.asarray(stats.get(key), dtype=np.float64).reshape(-1)
    if values.size != 1 or not np.isfinite(values).all():
        raise ValueError(f"expected one finite {key} value for the one-channel modality")
    return float(values[0])


def validate_domain(domain, *, spatial_policy=STRICT_SPATIAL_POLICY):
    """Validate the relevant entry of the *actual* pinned NASA YAML domains."""
    if spatial_policy not in SPATIAL_POLICIES:
        raise ValueError("unknown spatial policy")
    if domain.get("num_channels") != 1 or domain.get("pretokenized", False):
        raise ValueError("only one-channel untokenized terrain modalities are allowed")
    size = spatial_pair(domain.get("input_size"), "input_size")
    patch = spatial_pair(domain.get("patch_size"), "patch_size")
    if any(n % p for n, p in zip(size, patch)):
        raise ValueError("input_size must be divisible by patch_size")
    pre_resize = domain.get("pre_resize")
    if spatial_policy == NATIVE_SPATIAL_POLICY:
        if (size != (256, 256) or patch != (16, 16)
                or spatial_pair(pre_resize, "pre_resize") != (512, 512)):
            raise ValueError("native256 requires audited input256/patch16/pre_resize512 geometry")
    elif pre_resize is not None and spatial_pair(pre_resize, "pre_resize") != size:
        raise ValueError("pre_resize differs from input_size; audit spatial preprocessing first")
    if "scaler" not in domain:
        raise ValueError("scaler must be explicit, including null for no normalization")
    scaler, stats = domain["scaler"], domain.get("stats", {})
    if scaler in ("std", "local_mean_std"):
        if _one_stat(stats, "std") <= 0:
            raise ValueError("std must be positive")
        if scaler == "std":
            _one_stat(stats, "mean")
    elif scaler == "minmax":
        if _one_stat(stats, "max") <= _one_stat(stats, "min"):
            raise ValueError("max must exceed min")
    elif scaler is not None:
        raise ValueError("unsupported NASA scaler")
    return size


def validate_provenance(provenance, *, spatial_policy=STRICT_SPATIAL_POLICY):
    if provenance.get("input_space") != "raw_physical":
        raise ValueError("provide raw physical values, not Moonstone z-scores")
    if provenance.get("units") != {"elevation": "m", "slope": "degree"}:
        raise ValueError("elevation metres and slope degrees must be explicitly audited")
    if provenance.get("spatial_transform") != "none":
        raise ValueError("this first interface does not silently resize, crop or reproject")
    if provenance.get("domain_shift") != "MOONSTONE_TERRAIN_NOT_NASA_NATIVE":
        raise ValueError("the Moonstone/NASA physical-scale shift must be declared")
    if spatial_policy not in SPATIAL_POLICIES:
        raise ValueError("unknown spatial policy")
    if provenance.get("spatial_policy", STRICT_SPATIAL_POLICY) != spatial_policy:
        raise ValueError("provenance must explicitly match the selected spatial policy")
    if spatial_policy == NATIVE_SPATIAL_POLICY:
        if provenance.get("ground_sampling") != {
                "grid": "equirectangular", "pixels_per_degree": 128,
                "lunar_radius_m": 1737400, "east_west_spacing": "latitude_dependent",
                "nasa_native_60m_equivalence": False}:
            raise ValueError("native256 requires explicit audited anisotropic Moonstone ground sampling")
    resolution = provenance.get("meters_per_pixel")
    if isinstance(resolution, bool) or not isinstance(resolution, (float, int)) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("a finite positive audited meters_per_pixel is required")
    if spatial_policy == NATIVE_SPATIAL_POLICY and not np.isclose(resolution, MOONSTONE_NOMINAL_MPP, rtol=0, atol=1e-9):
        raise ValueError("native256 meters_per_pixel must be audited nominal north-south spacing")
    for key in ("elevation_source", "slope_source", "units_evidence", "nodata_evidence"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"missing provenance: {key}")


def prepare_terrain(raw, valid, domains, provenance, *,
                    spatial_policy=STRICT_SPATIAL_POLICY, config_sha256=None):
    """Normalize already shape-matched raw terrain, never infer missingness from zero.

    This first tier rejects *any* missing pixels instead of passing fabricated
    observations to an encoder without pixel-level validity support. Never filter
    rows after seeing scores: freeze a common eligible cohort before either arm.
    No labels, geology names, optical bands, or invented geometry are accepted.
    """
    spatial_policy_receipt(spatial_policy, config_sha256)
    validate_provenance(provenance, spatial_policy=spatial_policy)
    if set(raw) != set(MAPPING) or set(valid) != set(MAPPING):
        raise ValueError("exactly elevation and slope, with explicit validity masks, are required")
    result, batch_size = {}, None
    for source, destination in MAPPING.items():
        if destination not in domains:
            raise ValueError(f"NASA config lacks {destination}")
        domain = domains[destination]
        size = validate_domain(domain, spatial_policy=spatial_policy)
        original = np.asarray(raw[source])
        if not np.issubdtype(original.dtype, np.floating):
            raise ValueError("raw terrain must use floating point physical units")
        values = np.array(original, dtype=np.float32, copy=True)
        mask = np.asarray(valid[source])
        if values.ndim != 4 or values.shape[1] != 1 or values.shape[0] == 0 or values.shape[2:] != size:
            raise ValueError(f"{source} requires nonempty (B, 1, {size[0]}, {size[1]}); do not silently resize")
        if mask.dtype != np.bool_ or mask.shape != values.shape:
            raise ValueError("validity masks must be boolean and match raw arrays")
        if not mask.all() or not np.isfinite(values).all():
            raise ValueError("missing/nonfinite terrain is blocked; do not fill with fake observations")
        if source == "slope" and (np.any(values < 0) or np.any(values > 90)):
            raise ValueError("slope must be degrees in [0, 90]")
        if batch_size is not None and values.shape[0] != batch_size:
            raise ValueError("terrain batch sizes differ")
        batch_size = values.shape[0]
        scaler, stats = domain["scaler"], domain.get("stats", {})
        if scaler in ("std", "local_mean_std"):
            mean = (_one_stat(stats, "mean") if scaler == "std"
                    else values.mean(axis=(-2, -1), keepdims=True))
            values = (values - mean) / _one_stat(stats, "std")
        elif scaler == "minmax":
            low, high = _one_stat(stats, "min"), _one_stat(stats, "max")
            values = (values - low) / (high - low)
        if not np.isfinite(values).all():
            raise ValueError("normalization produced nonfinite values")
        result[destination] = np.ascontiguousarray(values, dtype=np.float32)
    return result


def unwrap_state_dict(checkpoint):
    """Accept only unambiguous tensor-state envelopes; reject prefix collisions."""
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("checkpoint must be a nonempty dictionary")
    wrappers = [key for key in ("model", "state_dict") if isinstance(checkpoint.get(key), dict)]
    if len(wrappers) > 1:
        raise ValueError("ambiguous checkpoint envelope")
    state = checkpoint[wrappers[0]] if wrappers else checkpoint
    normalized = {}
    for name, value in state.items():
        if not isinstance(name, str) or not hasattr(value, "shape"):
            raise ValueError("state_dict must map string names to tensors")
        name = name.removeprefix("module.")
        if name in normalized:
            raise ValueError("checkpoint key collision after removing module prefix")
        normalized[name] = value
    if not normalized:
        raise ValueError("empty state_dict")
    return normalized


def inference_key(name):
    return name == "register_tokens" or name.startswith(("encoder.", "encoder_norm.",
                                                          "encoder_embeddings.dtm.", "encoder_embeddings.slope."))


def audit_inference_state(target, checkpoint):
    """Require 100% selected inference tensors, including buffers, by name/shape.

    Decoder and unselected-modality weights may be excluded, but all exclusions
    are reported. Additional encoder keys fail rather than hiding config drift.
    """
    source = unwrap_state_dict(checkpoint)
    needed = {key: value for key, value in target.items() if inference_key(key)}
    for prefix in ("encoder.", "encoder_norm.", "encoder_embeddings.dtm.", "encoder_embeddings.slope."):
        if not any(key.startswith(prefix) for key in needed):
            raise ValueError(f"target is not a complete terrain encoder: {prefix}")
    missing = sorted(set(needed) - set(source))
    mismatched = sorted(key for key in needed.keys() & source.keys()
                        if tuple(needed[key].shape) != tuple(source[key].shape))
    extra_inference = sorted(key for key in source if inference_key(key) and key not in needed)
    if missing or mismatched or extra_inference:
        raise ValueError(f"incomplete NASA inference load: missing={missing}, shape={mismatched}, extra={extra_inference}")
    excluded = sorted(set(source) - set(needed))
    allowed_excluded = ("decoder.", "decoder_", "decoder_embeddings.", "encoder_embeddings.")
    unknown = [key for key in excluded if key != "mask_token" and not key.startswith(allowed_excluded)]
    if unknown:
        raise ValueError(f"unrecognized checkpoint keys: {unknown}")
    count = sum(int(np.prod(value.shape)) for value in needed.values())
    return {"inference_key_count": len(needed), "inference_element_count": count,
            "loaded_inference_fraction": 1.0, "excluded_keys": excluded,
            "loaded_keys": sorted(needed)}

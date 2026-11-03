"""Moonstone metadata and HDF5-derived pilot interface.

Contract inspected at upstream commit cbd4720653eb05afd975d2a36280858c5c56a275,
not an assertion of real HDF5/mmap equivalence. Preserve the official case-sensitive age mapping,
stored split membership and downstream finite/nonzero presence semantics. This code never uses
per-sample geology/name/age as an input channel. Global normalization is an upstream convention,
not a train-only fitted transformation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "cbd4720653eb05afd975d2a36280858c5c56a275"
AGE_NAMES = ("Pre-Nectarian", "Nectarian", "Imbrian", "Eratosthenian", "Copernican")
AGE_PREFIXES = (("pN", 0), ("EI", 2), ("N", 1), ("I", 2), ("E", 3), ("C", 4))
GROUP_CHANNELS = {
    "surface": ("wac_morphology", "elevation", "slope", "roughness"),
    "thermal": ("diviner_tbol_midnight", "diviner_temp_night", "rock_abundance", "christiansen_feature"),
    "spectral_m3": ("m3_750", "m3_950", "m3_1000", "m3_1250", "m3_1580", "m3_2000", "m3_2817", "m3_2857"),
    "gravity": ("grail_freeair", "grail_bouguer", "grail_uncertainty"),
    "radar": ("minirf_cpr", "minirf_s1"),
    "hapke": ("wac_hapke_415nm", "wac_hapke_566nm", "wac_hapke_604nm", "wac_hapke_689nm"),
    "composition": ("clementine_uvvis_750nm", "lpgrs_tio2", "lpgrs_feo"),
}
CHANNELS = tuple(channel for group in GROUP_CHANNELS.values() for channel in group)


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def age_target(name):
    name = decode_name(name)
    return next((target for prefix, target in AGE_PREFIXES if name.startswith(prefix)), -1)


def select_split(stored, latitude, split, track="official_stored"):
    stored, latitude = np.asarray(stored), np.asarray(latitude, dtype=float)
    if split not in (0, 1, 2) or stored.shape != latitude.shape:
        raise ValueError("invalid split or metadata shape")
    if track == "official_stored":
        if not np.isin(stored, (0, 1, 2)).all():
            raise ValueError("unknown stored split code")
        return stored == split
    if track != "official_geographic":
        raise ValueError("unknown split track")
    absolute = np.abs(latitude)
    return (absolute <= 60 if split == 0 else
            ((absolute > 60) & (absolute <= 70) if split == 1 else (absolute > 70) & (absolute <= 80)))


def geology_targets(unit_ids, names, mapping):
    """Preserve upstream ID-1 labels, rejecting invalid assets instead of silently filtering them."""
    if sorted(mapping.values()) != list(range(1, 50)):
        raise ValueError("49-class mapping must contain each ID 1..49 exactly once")
    ids = np.asarray(unit_ids)
    if not np.issubdtype(ids.dtype, np.integer) or not np.isin(ids, list(mapping.values())).all():
        raise ValueError("BLOCKED_INVALID_GEOLOGY_ID: upstream ID-1 would be outside 0..48")
    if len(ids) != len(names) or any(mapping.get(decode_name(name)) != int(unit)
                                     for unit, name in zip(ids, names)):
        raise ValueError("geology ID/name map mismatch")
    return ids.astype(np.int64) - 1


def normalize_hdf5(raw, stats, channels=CHANNELS):
    """Generic step15 z-score + restore raw NoData; radar provenance still requires real pairing.

    Finite raw zeros are data, including where their normalized value is nonzero. Already logged
    radar is not log1p-transformed again. The specialized fix_minirf zero policy is not asserted.
    """
    if tuple(channels) != CHANNELS or raw.shape[0] != len(CHANNELS):
        raise ValueError("unexpected input channel order/count")
    values = np.array(raw, dtype=np.float32, copy=True)
    invalid = ~np.isfinite(values)
    values[invalid] = 0
    for index, channel in enumerate(channels):
        if channel not in stats:
            raise ValueError(f"missing normalization statistics: {channel}")
        entry = stats[channel]
        mean, std = ((float(entry["mean"]), float(entry["std"])) if isinstance(entry, dict)
                     else tuple(map(float, entry)))
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("invalid global normalization statistics")
        values[index] = (values[index] - mean) / std
    values[invalid] = 0
    has_channel = (np.isfinite(values) & (values != 0)).reshape(len(channels), -1).any(axis=1)
    has_group = []
    offset = 0
    for group in GROUP_CHANNELS.values():
        has_group.append(has_channel[offset:offset+len(group)].any())
        offset += len(group)
    return values, has_channel.astype(np.float32), np.asarray(has_group, dtype=np.float32)


class MoonstoneHDF5Pilot:
    """Worker-local read-only HDF5 handles; explicitly HDF5_DERIVED_PILOT until paired validation.

    Accepts an actual HDF5 file or a locally extracted pilot with original metadata IDs/splits.
    The target, ID and optional location metadata remain outside the pixel feature tensor.
    """
    protocol = "HDF5_DERIVED_PILOT"

    def __init__(self, path, stats, mapping, task="age", split=0, track="official_stored"):
        import h5py
        if task not in ("age", "geology"):
            raise ValueError("only Age and Geology are in this interface")
        self.path, self.stats = str(Path(path)), stats
        self._handle, self._pid = None, None
        with h5py.File(self.path, "r") as handle:
            stored_channels = handle.attrs["channel_names"]
            if isinstance(stored_channels, (str, bytes)):
                stored_channels = json.loads(decode_name(stored_channels))
            if tuple(map(decode_name, stored_channels)) != CHANNELS:
                raise ValueError("HDF5 channel order mismatch")
            if handle["patches"].shape[1:] != (28, 256, 256):
                raise ValueError("unexpected patch dimensions")
            meta = handle["metadata"]
            self.ids = np.asarray(meta["patch_id"])
            if len(np.unique(self.ids)) != len(self.ids):
                raise ValueError("duplicate patch_id")
            if len(handle["patches"]) != len(self.ids):
                raise ValueError("patch/metadata length mismatch")
            self.rows, self.cols = np.asarray(meta["row_idx"]), np.asarray(meta["col_idx"])
            if not ((self.rows >= 0) & (self.rows < 90) & (self.cols >= 0) & (self.cols < 180)).all():
                raise ValueError("tile coordinates outside official grid")
            self.splits = np.asarray(meta["split"])
            self.latitudes = np.asarray(meta["center_lat"])
            names = np.asarray(meta["geology/dominant_unit_name"])
            unit_ids = np.asarray(meta["geology/dominant_unit_id"])
            if task == "age":
                target = np.asarray([age_target(name) for name in names])
                valid = target >= 0
                geology_targets(unit_ids[valid], names[valid], mapping)
            else:
                target = geology_targets(unit_ids, names, mapping)
            selected = select_split(self.splits, self.latitudes, split, track) & (target >= 0)
            self.indices, self.targets = np.flatnonzero(selected), target[selected]
            self.unknown_age_count = int(sum(age_target(name) < 0 for name in names))
            self.class_counts = np.bincount(self.targets, minlength=5 if task == "age" else 49)
            self.task, self.split, self.track = task, split, track

    def __len__(self):
        return len(self.indices)

    def _file(self):
        import h5py
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle, self._pid = h5py.File(self.path, "r"), os.getpid()
        return self._handle

    def __getitem__(self, index):
        source = int(self.indices[index])
        raw = self._file()["patches"][source]
        pixels, has_channel, has_group = normalize_hdf5(raw, self.stats)
        return {"pixels": pixels, "has_channel": has_channel, "has_group": has_group,
                "target": np.int64(self.targets[index]), "sample_id": str(self.ids[source])}

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"], state["_pid"] = None, None
        return state

"""HDF5 storage for oscilloscope traces.

File layout
-----------
``traces``            int16 dataset, shape ``(n_traces, n_samples)``; raw ADC counts
``metadata/<key>``    one dataset per key, shape ``(n_traces, width)``
file attrs            ``vertical_gain``, ``vertical_offset``, ``note``

Volts are recovered on demand as ``raw * vertical_gain - vertical_offset``.
Datasets are resizable, so traces are appended one at a time and a missed
capture never leaves an empty row behind.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import h5py
import numpy as np

TRACES = "traces"
METADATA = "metadata"
NOTE = "traces are raw int16 ADC counts; volts = traces*vertical_gain - vertical_offset"

PathLike = Union[str, Path]


def _normalise_metadata(metadata: Optional[Mapping]) -> Dict[str, np.ndarray]:
    """Turn every metadata value into a 1-D numpy array (bytes -> uint8)."""
    out: Dict[str, np.ndarray] = {}
    for key, val in (metadata or {}).items():
        if not key or "/" in key:
            raise ValueError(f"invalid metadata key {key!r} (empty or contains '/')")
        if isinstance(val, (bytes, bytearray)):
            arr = np.frombuffer(bytes(val), dtype="u1")
        else:
            arr = np.atleast_1d(np.asarray(val))
        if arr.ndim != 1:
            raise ValueError(f"metadata {key!r} must be scalar or 1-D, got shape {arr.shape}")
        out[key] = arr
    return out


def _as_int16(raw_counts) -> np.ndarray:
    raw = np.asarray(raw_counts)
    if raw.ndim != 1:
        raise ValueError(f"trace must be 1-D, got shape {raw.shape}")
    if raw.dtype.kind not in "iu":
        raise TypeError(f"trace must hold integer ADC counts, got dtype {raw.dtype}")
    if raw.dtype != np.int16:
        info = np.iinfo(np.int16)
        if raw.size and (raw.min() < info.min or raw.max() > info.max):
            raise ValueError("trace values do not fit in int16")
        raw = raw.astype(np.int16)
    return raw


class TraceWriter:
    """Append raw traces (plus per-trace metadata) to an HDF5 file.

    Parameters
    ----------
    path : file to write.
    mode : ``"a"`` (default; create or append), ``"w"`` (overwrite), ``"x"`` (fail if exists).
    compression_opts : gzip level 0-9.
    """

    def __init__(self, path: PathLike, mode: str = "a", compression_opts: int = 4):
        if mode not in ("a", "w", "x"):
            raise ValueError("mode must be 'a', 'w' or 'x'")
        self.path = str(path)
        self._level = compression_opts
        self._f = h5py.File(self.path, mode)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._f:
            self._f.close()

    def __len__(self) -> int:
        return self._f[TRACES].shape[0] if TRACES in self._f else 0

    # -- writing -----------------------------------------------------------
    def append(self, raw_counts, gain: float, offset: float,
               metadata: Optional[Mapping] = None) -> int:
        """Append one trace. Returns its index.

        Everything is validated before the file is modified, so a failed
        append leaves the file untouched.
        """
        raw = _as_int16(raw_counts)
        meta = _normalise_metadata(metadata)
        f = self._f

        if TRACES not in f:
            self._create(raw, gain, offset, meta)
            return 0

        ds = f[TRACES]
        if raw.shape[0] != ds.shape[1]:
            raise ValueError(f"trace has {raw.shape[0]} samples, file expects {ds.shape[1]}")
        if not (np.isclose(gain, f.attrs["vertical_gain"], rtol=1e-6, atol=0)
                and np.isclose(offset, f.attrs["vertical_offset"], rtol=1e-6, atol=0)):
            raise ValueError(
                "vertical gain/offset differ from the values stored in this file "
                f"({gain}, {offset}) vs ({f.attrs['vertical_gain']}, {f.attrs['vertical_offset']}); "
                "start a new file when scope vertical settings change")
        existing = set(f[METADATA].keys()) if METADATA in f else set()
        if set(meta) != existing:
            raise ValueError(f"metadata keys {sorted(meta)} do not match file keys {sorted(existing)}")
        for key, arr in meta.items():
            if arr.shape[0] != f[f"{METADATA}/{key}"].shape[1]:
                raise ValueError(f"metadata {key!r} width changed")

        idx = ds.shape[0]
        ds.resize(idx + 1, axis=0)
        ds[idx] = raw
        for key, arr in meta.items():
            m = f[f"{METADATA}/{key}"]
            m.resize(idx + 1, axis=0)
            m[idx] = arr
        f.flush()
        return idx

    def _create(self, raw: np.ndarray, gain: float, offset: float, meta: Dict[str, np.ndarray]) -> None:
        f = self._f
        n = raw.shape[0]
        ds = f.create_dataset(TRACES, shape=(1, n), maxshape=(None, n), dtype="i2",
                              chunks=(1, n), compression="gzip",
                              compression_opts=self._level, shuffle=True)
        ds[0] = raw
        f.attrs["vertical_gain"] = float(gain)
        f.attrs["vertical_offset"] = float(offset)
        f.attrs["note"] = NOTE
        for key, arr in meta.items():
            w = arr.shape[0]
            m = f.create_dataset(f"{METADATA}/{key}", shape=(1, w), maxshape=(None, w),
                                 dtype=arr.dtype, chunks=(1, w),
                                 compression="gzip", compression_opts=self._level)
            m[0] = arr
        f.flush()


class TraceReader:
    """Read traces written by :class:`TraceWriter`.

    ``reader[i]`` / ``reader[2:10]`` return volts; use :meth:`raw` for ADC counts.
    """

    def __init__(self, path: PathLike):
        self.path = str(path)
        self._f = h5py.File(self.path, "r")
        if TRACES not in self._f:
            self._f.close()
            raise ValueError(f"{self.path} contains no '{TRACES}' dataset")

    def __enter__(self) -> "TraceReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._f:
            self._f.close()

    @property
    def n_traces(self) -> int:
        return self._f[TRACES].shape[0]

    @property
    def n_samples(self) -> int:
        return self._f[TRACES].shape[1]

    @property
    def vertical_gain(self) -> float:
        return float(self._f.attrs["vertical_gain"])

    @property
    def vertical_offset(self) -> float:
        return float(self._f.attrs["vertical_offset"])

    @property
    def metadata_keys(self):
        return sorted(self._f[METADATA].keys()) if METADATA in self._f else []

    def __len__(self) -> int:
        return self.n_traces

    def raw(self, index=slice(None)) -> np.ndarray:
        """Raw int16 ADC counts."""
        return self._f[TRACES][index]

    def volts(self, index=slice(None), dtype=np.float32) -> np.ndarray:
        """Traces converted to volts: ``raw * gain - offset``."""
        dt = np.dtype(dtype)
        raw = self.raw(index).astype(dt)
        return raw * dt.type(self.vertical_gain) - dt.type(self.vertical_offset)

    __getitem__ = volts

    def metadata(self, key: str, index=slice(None)) -> np.ndarray:
        if key not in self.metadata_keys:
            raise KeyError(f"no metadata {key!r}; available: {self.metadata_keys}")
        return self._f[f"{METADATA}/{key}"][index]

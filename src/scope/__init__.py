"""scope: oscilloscope capture + HDF5 trace storage."""
from .h5io import TraceReader, TraceWriter
from .scope import Channel, Scope, ScopeError, Waveform, parse_inspect, resolve_resource

__version__ = "0.1.0"

__all__ = [
    "TraceWriter", "TraceReader",
    "Scope", "Channel", "ScopeError", "Waveform", "parse_inspect", "resolve_resource",
]

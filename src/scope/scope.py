"""LeCroy-style oscilloscope control over VISA (tested against HDO6104 command set).

The :class:`Scope` class only needs an object with ``write``, ``query`` and
``query_binary_values`` (a pyvisa resource), so it can be unit-tested with a fake.
"""
from __future__ import annotations

import re
import time
from typing import Dict, Optional, Union

import numpy as np

from scope.domain.valueobjects import Waveform


class ScopeError(RuntimeError):
    pass


_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_inspect(reply: str) -> float:
    """Extract the number from an ``INSPECT?`` reply like ``"VERTICAL_GAIN" : 1.5e-04``."""
    if ":" not in reply:
        raise ValueError(f"Unexpected INSPECT? reply: {reply!r}")
    m = _NUMBER.search(reply.split(":", 1)[1])
    if not m:
        raise ValueError(f"Could not parse a number out of INSPECT? reply: {reply!r}")
    return float(m.group(0))


def resolve_resource(address: str) -> str:
    """Turn ``192.168.0.2`` (or a hostname) into ``TCPIP0::192.168.0.2::inst0::INSTR``.

    Anything that already looks like a VISA resource string (contains ``::``,
    e.g. USB or GPIB addresses) is returned unchanged.
    """
    address = address.strip()
    if not address:
        raise ValueError("scope address is empty")
    if "::" in address:
        return address
    return f"TCPIP0::{address}::inst0::INSTR"


COUPLINGS = ("A1M", "D1M", "D50", "GND")


class Channel:
    """One trace on the scope (``C1``..``C4`` analog, or e.g. ``Z1``/``F1`` math/zoom).

    Obtain via ``scope.channel("C2")`` or ``scope["C2"]``. :meth:`read` and
    :meth:`capture` switch the trace on automatically if it is off.
    """

    def __init__(self, scope: "Scope", name: str):
        self.scope = scope
        self.name = name.upper()

    def __repr__(self) -> str:
        return f"Channel({self.name!r})"

    def __str__(self) -> str:
        return self.name

    @property
    def is_analog(self) -> bool:
        return re.fullmatch(r"C\d+", self.name) is not None

    def _write(self, cmd: str) -> None:
        self.scope.inst.write(f"{self.name}:{cmd}")

    def enable(self, on: bool = True) -> "Channel":
        self._write(f"TRACE {'ON' if on else 'OFF'}")
        return self

    @property
    def is_enabled(self) -> bool:
        return re.search(r"\bON\b", self.scope.inst.query(f"{self.name}:TRACE?").upper()) is not None

    def ensure_enabled(self) -> "Channel":
        """Turn the trace on only if it is currently off."""
        if not self.is_enabled:
            self.enable()
        return self

    def configure(self, *, volts_per_div: Optional[float] = None,
                  offset: Optional[float] = None,
                  coupling: Optional[str] = None) -> "Channel":
        """Optional: set vertical scale (V/div), offset (V) and/or input coupling.

        Usually you set these on the scope itself; this is for scripted setups.

        Only analog channels (``C1``..) have these settings. Arguments left as
        ``None`` are not touched. Everything is validated before any command is sent.
        """
        if not self.is_analog:
            raise ScopeError(f"{self.name} is not an analog channel; cannot set vertical settings")
        if coupling is not None and coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {COUPLINGS}, got {coupling!r}")
        if volts_per_div is not None and volts_per_div <= 0:
            raise ValueError("volts_per_div must be positive")
        if volts_per_div is not None:
            self._write(f"VOLT_DIV {volts_per_div}")
        if offset is not None:
            self._write(f"OFFSET {offset}")
        if coupling is not None:
            self._write(f"COUPLING {coupling}")
        return self

    def read(self) -> Waveform:
        """Read waveform using the previously verified LeCroy settings."""

        self.ensure_enabled()

        inst = self.scope.inst

        # Download waveform using the working command
        raw = inst.query_binary_values(
            f"{self.name}:WAVEFORM? DAT1",
            datatype="h",
            is_big_endian=True,
            container=np.array,
        )

        # Retrieve vertical scaling
        gain = parse_inspect(
            inst.query(
                f'{self.name}:INSPECT? "VERTICAL_GAIN"'
            )
        )

        offset = parse_inspect(
            inst.query(
                f'{self.name}:INSPECT? "VERTICAL_OFFSET"'
            )
        )

        return Waveform(
            np.asarray(raw, dtype=np.int16),
            gain,
            offset
        )

    def capture(self, timeout: float = 10.0, poll_interval: float = 0.0) -> Waveform:
        """Arm a single acquisition, wait for the trigger and read this channel.

        Raises :class:`ScopeError` if the scope does not trigger within ``timeout`` s.
        """
        self.ensure_enabled()
        self.scope.arm_single()
        if not self.scope.wait_for_stop(timeout, poll_interval):
            raise ScopeError(f"no trigger within {timeout} s")
        return self.read()


class Scope:
    """Connection to the oscilloscope. Per-channel settings live on :class:`Channel`."""

    def __init__(self, instrument):
        self.inst = instrument
        self._channels: Dict[str, Channel] = {}

    @classmethod
    def open(cls, address: str, timeout_ms: int = 20000, resource_manager=None) -> "Scope":
        """Connect to a scope by IP/hostname (``"192.168.0.2"``) or a full VISA string."""
        if resource_manager is None:
            try:
                import pyvisa
            except ImportError as e:  # pragma: no cover
                raise ImportError("pyvisa is required: pip install 'scope[scope]'") from e
            resource_manager = pyvisa.ResourceManager()
        rm = resource_manager
        inst = rm.open_resource(resolve_resource(address))
        inst.timeout = timeout_ms
        scope = cls(inst)
        scope.setup_comm()
        return scope

    def close(self) -> None:
        self.inst.close()

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def idn(self) -> str:
        return self.inst.query("*IDN?").strip()

    def setup_comm(self) -> None:
        self.inst.chunk_size = 4 * 1024 * 1024

        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"

        self.inst.write("CHDR OFF")
        self.inst.write("CORD HI")
        self.inst.write("CFMT DEF9,WORD,BIN")

        self.inst.write("WFSU SP,0,NP,0,FP,0")

        self.inst.query("*OPC?")

    # -- channels ----------------------------------------------------------
    def channel(self, name: Union[str, Channel]) -> Channel:
        if isinstance(name, Channel):
            return name
        key = name.upper()
        if key not in self._channels:
            self._channels[key] = Channel(self, key)
        return self._channels[key]

    __getitem__ = channel

    # -- trigger -----------------------------------------------------------
    def set_edge_trigger(self, source: Union[str, Channel], level: float, slope: str = "POS") -> None:
        """Optional: edge trigger on ``source`` (a channel or e.g. ``"EX"``).

        Usually the trigger is set on the scope itself; this is for scripted setups.
        """
        if slope not in ("POS", "NEG"):
            raise ValueError("slope must be 'POS' or 'NEG'")
        src = str(source).upper()
        self.inst.write(f"TRIG_SELECT EDGE,SR,{src}")
        self.inst.write(f"{src}:TRIG_LEVEL {level}")
        self.inst.write(f"{src}:TRIG_SLOPE {slope}")

    # -- acquisition -------------------------------------------------------
    def arm_single(self) -> None:
        self.inst.write("TRMD SINGLE")

    def state(self) -> str:
        return self.inst.query("TRMD?").strip()

    def wait_for_stop(self, timeout: float, poll_interval: float = 0.0) -> bool:
        """Poll ``TRMD?`` until the scope reports STOP. True if it did within ``timeout`` s."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if "STOP" in self.state().upper():
                return True
            if poll_interval:
                time.sleep(poll_interval)
        return False

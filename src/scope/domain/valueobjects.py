from dataclasses import dataclass

import numpy as np


@dataclass
class Waveform:
    raw: np.ndarray   # int16 ADC counts
    gain: float
    offset: float

    @property
    def volts(self) -> np.ndarray:
        return (
            self.raw.astype(np.float32) * np.float32(self.gain)
            - np.float32(self.offset)
        )
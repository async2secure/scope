
import matplotlib.pyplot as plt
import numpy as np

from scope import Scope, TraceWriter, TraceReader

with Scope.open("192.168.0.174") as scope:
    ch1 = scope.channel("C1")
    waveform = ch1.read()

    with TraceWriter("experiment.h5", mode="w") as writer:
        idx = writer.append(
            raw_counts=waveform.raw,
            gain=waveform.gain,
            offset=waveform.offset,
            metadata={
                "trace_id": 0,
            }
        )

        print("Saved trace:", idx)


with TraceReader("experiment.h5") as reader:

    print("Number of traces:", reader.n_traces)
    print("Samples per trace:", reader.n_samples)

    voltage = reader[0]
    raw = reader.raw(0)

    print(raw.dtype)

    # Read metadata
    trace_id = reader.metadata("trace_id", 0)

    print("Trace ID:", trace_id)
    print("Maximum voltage:", np.max(voltage))
    print("Minimum voltage:", np.min(voltage))

    # Plot
    plt.figure(figsize=(12, 5))

    plt.plot(voltage, linewidth=0.5)

    plt.xlabel("Sample")
    plt.ylabel("Voltage (V)")
    plt.title("Trace 0")

    plt.grid()
    plt.tight_layout()
    plt.show()
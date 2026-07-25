"""Replay the first two-channel EEG stream in an XDF file as FP1/FP2 LSL."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyxdf
from pylsl import StreamInfo, StreamOutlet, local_clock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", type=Path, required=True)
    parser.add_argument("--stream-name", default="BSense EEG Demo")
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=25)
    return parser.parse_args()


def stream_type(stream: dict) -> str:
    return str(stream.get("info", {}).get("type", [""])[0]).strip().lower()


def main() -> int:
    args = parse_args()
    streams, _header = pyxdf.load_xdf(str(args.xdf))
    candidates = [
        stream
        for stream in streams
        if stream_type(stream) == "eeg"
        and np.asarray(stream["time_series"]).ndim == 2
        and np.asarray(stream["time_series"]).shape[1] == 2
    ]
    if not candidates:
        raise RuntimeError("XDF 中没有找到双通道 EEG 流。")
    eeg = candidates[0]
    values = np.asarray(eeg["time_series"], dtype=np.float32)
    timestamps = np.asarray(eeg["time_stamps"], dtype=np.float64)
    sfreq = float(eeg["info"]["nominal_srate"][0])
    info = StreamInfo(
        args.stream_name,
        "EEG",
        2,
        sfreq,
        "float32",
        "bsense-independent-xdf-replay",
    )
    channels = info.desc().append_child("channels")
    for label in ("FP1", "FP2"):
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("type", "EEG")
        channel.append_child_value("unit", "device_units")
    outlet = StreamOutlet(info, chunk_size=args.chunk_size, max_buffered=120)
    print(
        f"LSL EEG ready: name={args.stream_name}, sfreq={sfreq:g}, "
        f"samples={len(values)}",
        flush=True,
    )
    time.sleep(args.startup_delay)
    origin = local_clock()
    relative = timestamps - timestamps[0]
    for offset in range(0, len(values), args.chunk_size):
        stop = min(offset + args.chunk_size, len(values))
        outlet.push_chunk(
            values[offset:stop].tolist(),
            (origin + relative[offset:stop]).tolist(),
        )
        if args.speed > 0:
            time.sleep(
                (stop - offset) / max(sfreq, 1.0) / args.speed
            )
    print("LSL EEG replay complete", flush=True)
    time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

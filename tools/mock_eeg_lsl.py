#!/usr/bin/env python3
"""Publish a two-channel FP1/FP2 LSL EEG stream as a remote-source stand-in."""

import argparse
import math
import random
import time

from pylsl import StreamInfo, StreamOutlet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Mock Official EEG")
    parser.add_argument("--rate", type=float, default=250.0)
    args = parser.parse_args()

    info = StreamInfo(args.name, "EEG", 2, args.rate, "float32", "mock-official-eeg")
    channels = info.desc().append_child("channels")
    for label in ("FP1", "FP2"):
        channels.append_child("channel").append_child_value("label", label)
    outlet = StreamOutlet(info)
    print(f"Publishing LSL EEG: {args.name}, FP1/FP2 @ {args.rate:g} Hz", flush=True)

    phase = 0.0
    step = 2.0 * math.pi * 10.0 / args.rate
    period = 1.0 / args.rate
    while True:
        now = time.monotonic()
        sample = [
            20.0 * math.sin(phase) + random.gauss(0.0, 2.0),
            18.0 * math.sin(phase + 0.2) + random.gauss(0.0, 2.0),
        ]
        outlet.push_sample(sample)
        phase += step
        time.sleep(max(0.0, period - (time.monotonic() - now)))


if __name__ == "__main__":
    main()


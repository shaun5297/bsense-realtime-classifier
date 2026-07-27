#!/usr/bin/env python3
"""Emit bci_result_v1 records to stdout or the Unitree Unix Socket plugin."""

import argparse
import json
import socket
import sys
import time


MOTIONS = {
    "idle": {"x": 0.0, "y": 0.0, "z": 0.0},
    "left_mi": {"x": 0.0, "y": 0.0, "z": 0.25},
    "right_mi": {"x": 0.0, "y": 0.0, "z": -0.25},
    "forward": {"x": 0.15, "y": 0.0, "z": 0.0},
    "backward": {"x": -0.15, "y": 0.0, "z": 0.0},
    "strafe_left": {"x": 0.0, "y": 0.15, "z": 0.0},
    "strafe_right": {"x": 0.0, "y": -0.15, "z": 0.0},
    "turn_left": {"x": 0.0, "y": 0.0, "z": 0.25},
    "turn_right": {"x": 0.0, "y": 0.0, "z": -0.25},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=tuple(MOTIONS), default="idle")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--socket-path", default="", help="send directly to ROS 2 plugin")
    args = parser.parse_args()

    sock = None
    if args.socket_path:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.socket_path)

    try:
        while True:
            message = {
                "protocol": "bci_result_v1",
                "label": args.label,
                "accepted": True,
                "confidence": 1.0,
                "timestamp": time.time(),
                "motion": MOTIONS[args.label],
            }
            encoded = (json.dumps(message, ensure_ascii=False) + "\n").encode()
            if sock is not None:
                sock.sendall(encoded)
            else:
                print(encoded.decode(), end="", flush=True)
            time.sleep(args.interval)
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    finally:
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    sys.exit(main())


"""Run a bounded XDF -> LSL -> classifier end-to-end validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=18.0)
    return parser.parse_args()


def stop_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
    try:
        return process.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate(timeout=3.0)


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stream_name = f"BSense Independent E2E {stamp}"
    result_file = project / "logs" / f"e2e_results_{stamp}.jsonl"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    creation_flags = (
        subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    )
    publisher = subprocess.Popen(
        [
            sys.executable,
            str(project / "tools" / "replay_xdf_eeg_lsl.py"),
            "--xdf",
            str(args.xdf.resolve()),
            "--stream-name",
            stream_name,
            "--speed",
            "5",
            "--startup-delay",
            "3",
        ],
        cwd=project,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    time.sleep(0.7)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project / "src")
    classifier = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bsense_classifier.cli",
            "--model",
            str(args.model.resolve()),
            "--stream-name",
            stream_name,
            "--output",
            str(result_file),
            "--resolve-timeout",
            "10",
        ],
        cwd=project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    time.sleep(args.seconds)
    classifier_out, classifier_err = stop_process(classifier)
    publisher_out, publisher_err = stop_process(publisher)
    print(publisher_out, end="")
    print(publisher_err, end="", file=sys.stderr)
    print(classifier_out, end="")
    print(classifier_err, end="", file=sys.stderr)
    records = []
    if result_file.exists():
        records = [
            json.loads(line)
            for line in result_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    print(f"E2E_RESULT_FILE={result_file}")
    print(f"E2E_RESULT_COUNT={len(records)}")
    if records:
        print(json.dumps(records[-1], ensure_ascii=False))
    connected = '"kind": "connected"' in classifier_out
    return 0 if connected and records else 1


if __name__ == "__main__":
    raise SystemExit(main())

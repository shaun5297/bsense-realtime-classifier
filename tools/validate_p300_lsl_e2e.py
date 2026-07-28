"""Replay one real P300 trial through LSL and validate the online control chain."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bsense_classifier.p300_control import (  # noqa: E402
    P300ControlEngine,
    P300EngineConfig,
    P300TargetRuntime,
)
from bsense_classifier.p300_training import (  # noqa: E402
    COMMANDS,
    WINDOW_OFFSET_SECONDS,
    _find_stream,
    _interpolate_epoch,
    _load_xdf,
    _marker_rows,
    discover_recordings,
)
from bsense_classifier.robot_bridge import (  # noqa: E402
    RobotBridgeClient,
    RobotBridgeConfig,
    SafeRobotController,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--subject",
        default="",
        help="采集目录名；留空时使用第一份 M7 记录。",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="global_trial；0 表示该记录的第一个完整 trial。",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _select_trial(
    data_root: Path,
    subject: str,
    requested_trial: int,
) -> tuple[str, int, str, list[tuple[dict[str, Any], np.ndarray]]]:
    recordings = discover_recordings(data_root)
    if not recordings:
        raise ValueError("没有找到 M7 P300 XDF。")
    selected_subject = subject or recordings[0].parent.name
    matches = [path for path in recordings if path.parent.name == selected_subject]
    if not matches:
        raise ValueError(f"没有找到采集目录 {selected_subject!r}。")

    path = matches[0]
    streams = _load_xdf(path)
    eeg = _find_stream(streams, "eeg")
    markers = _find_stream(streams, "markers")
    source_times = np.asarray(eeg["time_stamps"], dtype=np.float64)
    source_values = np.asarray(eeg["time_series"], dtype=np.float64)
    flashes = [
        row for row in _marker_rows(markers) if row.get("event") == "p300_flash"
    ]
    trial_ids = sorted(
        {
            int(row.get("global_trial", row.get("trial", 0)))
            for row in flashes
        }
    )
    if not trial_ids:
        raise ValueError(f"{path.name} 没有 P300 Trial。")
    trial_id = requested_trial or trial_ids[0]
    selected: list[tuple[dict[str, Any], np.ndarray]] = []
    for row in flashes:
        row_trial = int(row.get("global_trial", row.get("trial", 0)))
        if row_trial != trial_id:
            continue
        command = str(row.get("flash_command", "")).strip().lower()
        if command not in COMMANDS:
            continue
        raw = _interpolate_epoch(
            source_times,
            source_values,
            float(row["_timestamp_xdf"]) + WINDOW_OFFSET_SECONDS,
        )
        if raw is not None:
            selected.append((row, raw))
    counts = {
        command: sum(
            str(row.get("flash_command", "")).strip().lower() == command
            for row, _raw in selected
        )
        for command in COMMANDS
    }
    if not selected or any(count == 0 for count in counts.values()):
        raise ValueError(
            f"{selected_subject} 的 global_trial={trial_id} 不是完整六指令 Trial。"
        )
    target = str(selected[0][0].get("target_command", "")).strip().lower()
    return selected_subject, trial_id, target, selected


def _wait_event(
    engine: P300ControlEngine,
    kind: str,
    *,
    timeout: float,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = engine.events.get(timeout=min(0.25, deadline - time.monotonic()))
        except queue.Empty:
            continue
        events.append(event)
        if event["kind"] == "error":
            raise RuntimeError(str(event.get("message", "P300 在线引擎错误")))
        if kind == "flash_result" and event["kind"] == "flash_skipped":
            raise RuntimeError(
                "P300 在线窗口被跳过："
                + str(event.get("reason", "unknown_reason"))
            )
        if event["kind"] == kind:
            return event
    raise TimeoutError(f"等待 P300 在线事件 {kind!r} 超时。")


def _create_outlet(stream_name: str, sfreq: float) -> StreamOutlet:
    info = StreamInfo(
        stream_name,
        "EEG",
        2,
        sfreq,
        "float32",
        f"bsense-p300-e2e-{uuid.uuid4()}",
    )
    channels = info.desc().append_child("channels")
    for label in ("FP1", "FP2"):
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("type", "EEG")
        channel.append_child_value("unit", "device_units")
    return StreamOutlet(info, chunk_size=25, max_buffered=120)


def main() -> int:
    args = parse_args()
    subject, trial_id, target_command, epochs = _select_trial(
        args.data_root,
        args.subject,
        args.trial,
    )
    runtime = P300TargetRuntime.load(str(args.model))
    stream_name = f"BSense P300 E2E {uuid.uuid4().hex[:8]}"
    outlet = _create_outlet(stream_name, runtime.sfreq)
    minimum_flashes = min(
        sum(
            str(row.get("flash_command", "")).strip().lower() == command
            for row, _raw in epochs
        )
        for command in COMMANDS
    )
    engine = P300ControlEngine(
        runtime,
        P300EngineConfig(
            stream_name=stream_name,
            resolve_timeout=10.0,
            minimum_flashes_per_command=minimum_flashes,
            confidence_threshold=0.0,
            margin_threshold=0.0,
            minimum_quality_ratio=0.0,
        ),
    )
    events: list[dict[str, Any]] = []
    engine.start()
    bridge: RobotBridgeClient | None = None
    controller: SafeRobotController | None = None
    try:
        connected = _wait_event(engine, "connected", timeout=12.0, events=events)
        outlet.wait_for_consumers(2.0)
        time.sleep(0.2)
        trigger_time = local_clock() + 0.5
        for index, (row, raw) in enumerate(epochs):
            command = str(row["flash_command"]).strip().lower()
            engine.submit_flash(
                trial_id=trial_id,
                sequence=int(row.get("sequence", 0)),
                position=int(row.get("flash_position", index % len(COMMANDS))),
                command=command,
                timestamp=trigger_time,
            )
            core_times = (
                trigger_time
                + WINDOW_OFFSET_SECONDS
                + np.arange(raw.shape[1], dtype=np.float64) / runtime.sfreq
            )
            padding_samples = max(5, int(round(0.05 * runtime.sfreq)))
            padded = np.pad(
                raw,
                ((0, 0), (padding_samples, padding_samples)),
                mode="edge",
            )
            sample_times = (
                core_times[0]
                - padding_samples / runtime.sfreq
                + np.arange(padded.shape[1], dtype=np.float64) / runtime.sfreq
            )
            outlet.push_chunk(
                padded.T.astype(np.float32).tolist(),
                sample_times.tolist(),
            )
            _wait_event(engine, "flash_result", timeout=5.0, events=events)
            trigger_time = float(sample_times[-1]) + 0.25

        engine.finish_trial(trial_id)
        decision_event = _wait_event(
            engine,
            "decision",
            timeout=5.0,
            events=events,
        )
        decision = decision_event["decision"]
        if not decision.accepted:
            raise RuntimeError(
                "端到端验证使用零阈值仍被拒绝："
                + ", ".join(decision.rejection_reasons)
            )

        bridge = RobotBridgeClient(
            RobotBridgeConfig(
                transport="stdout",
                require_obstacle_clear=True,
                motion_duration_seconds=0.2,
            )
        )
        controller = SafeRobotController(bridge)
        controller.arm()
        command_response = controller.execute(
            decision.command,
            confidence=decision.confidence,
            source="bsense_p300_lsl_e2e",
        )
        controller.disarm()

        flash_results = [
            event for event in events if event["kind"] == "flash_result"
        ]
        result = {
            "ok": True,
            "dry_run": True,
            "live_lsl": True,
            "chain": [
                "real_xdf_trial",
                "lsl_eeg_stream",
                "online_event_locked_windows",
                "m7_p300_model",
                "six_command_aggregation",
                "safety_controller",
                "stdout_robot_bridge",
            ],
            "subject_id": subject,
            "global_trial": trial_id,
            "target_command": target_command,
            "predicted_command": decision.command,
            "prediction_correct": decision.command == target_command,
            "flash_results": len(flash_results),
            "connected": {
                key: value for key, value in connected.items() if key != "kind"
            },
            "decision": asdict(decision),
            "bridge_response": command_response,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        return 0
    finally:
        engine.stop()
        if controller is not None and controller.armed:
            controller.disarm()
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    raise SystemExit(main())

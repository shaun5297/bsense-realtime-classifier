"""Console entry point for headless real-time inference."""

from __future__ import annotations

import argparse
import json
import queue
import signal
from dataclasses import asdict
from pathlib import Path

from .lsl_engine import EngineConfig, RealtimeEngine
from .model_runtime import ModelRuntime
from .task_guidance import guidance_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BSense LSL 实时推理（命令行）")
    parser.add_argument("--model", required=True, help="model.joblib 路径")
    parser.add_argument("--stream-name", default="", help="EEG LSL 流名称")
    parser.add_argument("--stream-type", default="EEG", help="EEG LSL 流类型")
    parser.add_argument(
        "--marker-stream-name",
        default="BSense Experiment Markers",
        help="Marker LSL 流名称",
    )
    parser.add_argument("--no-marker", action="store_true", help="不连接 Marker 流")
    parser.add_argument(
        "--no-auto-event",
        action="store_true",
        help="关闭 M1/M4A 自动滚动分析，只接受 Marker/手动触发",
    )
    parser.add_argument("--output", help="可选的 JSONL 结果文件")
    parser.add_argument("--threshold", type=float, help="覆盖模型默认置信度阈值")
    parser.add_argument("--smoothing", type=int, default=3, help="平滑窗口数")
    parser.add_argument("--confirmation", type=int, default=2, help="连续确认次数")
    parser.add_argument("--resolve-timeout", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = ModelRuntime.load(args.model)
    guidance = guidance_for(runtime.task)
    print(
        json.dumps(
            {
                "kind": "model",
                "model": str(Path(args.model).resolve()),
                "task": runtime.task,
                "event_locked": runtime.event_locked,
                "guidance": asdict(guidance),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    engine = RealtimeEngine(
        runtime,
        EngineConfig(
            stream_name=args.stream_name,
            stream_type=args.stream_type,
            marker_stream_name=args.marker_stream_name,
            resolve_timeout=args.resolve_timeout,
            confidence_threshold=args.threshold,
            smoothing_windows=max(1, args.smoothing),
            confirmation_windows=max(1, args.confirmation),
            use_markers=not args.no_marker,
            auto_analyze_event_tasks=not args.no_auto_event,
            output_path=Path(args.output) if args.output else None,
        ),
    )
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        engine.stop()

    signal.signal(signal.SIGINT, request_stop)
    engine.start()
    exit_code = 0
    while engine.running or not engine.events.empty():
        try:
            event = engine.events.get(timeout=0.5)
        except queue.Empty:
            if stop_requested:
                break
            continue
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if event.get("kind") == "error":
            exit_code = 1
    engine.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

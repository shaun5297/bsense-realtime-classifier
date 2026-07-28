"""Validate XDF -> P300 model -> command aggregation -> stdout robot bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bsense_classifier.p300_control import (  # noqa: E402
    P300TrialAggregator,
    P300TargetRuntime,
)
from bsense_classifier.p300_training import (  # noqa: E402
    build_dataset,
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
        help="可选的采集目录名；留空时使用第一份记录。",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="可选的 global_trial；0 表示该记录的第一个完整 trial。",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    discovered = discover_recordings(args.data_root)
    if not discovered:
        raise ValueError("没有找到 M7 P300 XDF。")
    subject = args.subject or discovered[0].parent.name
    dataset = build_dataset(
        args.data_root,
        selected_groups={subject},
        max_trials_per_recording=max(1, args.trial),
    )
    runtime = P300TargetRuntime.load(str(args.model))
    if runtime.runtime.feature_mode == "erp_raw":
        model_features = dataset.raw_erp_features
        dataset_names = dataset.raw_erp_feature_names
    else:
        model_features = dataset.features
        dataset_names = dataset.feature_names
    artifact_names = tuple(runtime.runtime.feature_names)
    if artifact_names and artifact_names != dataset_names:
        raise RuntimeError("模型特征定义与当前数据生成的特征不一致。")

    available_trials = sorted(
        {
            int(row["global_trial"])
            for row in dataset.metadata
            if str(row["subject_id"]) == subject
        }
    )
    if not available_trials:
        raise ValueError(f"没有找到采集目录 {subject!r}。")
    trial_id = args.trial or available_trials[0]
    indices = [
        index
        for index, row in enumerate(dataset.metadata)
        if str(row["subject_id"]) == subject
        and int(row["global_trial"]) == trial_id
    ]
    if not indices:
        raise ValueError(f"{subject} 没有 global_trial={trial_id}。")

    probabilities = np.asarray(
        runtime.runtime.model.predict_proba(model_features[indices]),
        dtype=np.float64,
    )[:, runtime.target_index]
    aggregator = P300TrialAggregator(
        trial_id=trial_id,
        minimum_flashes_per_command=1,
        confidence_threshold=0.0,
        margin_threshold=0.0,
        minimum_quality_ratio=0.0,
    )
    target_command = str(dataset.metadata[indices[0]]["target_command"])
    for index, probability in zip(indices, probabilities, strict=True):
        aggregator.add(
            str(dataset.metadata[index]["flash_command"]),
            float(probability),
            quality_ok=True,
        )
    decision = aggregator.finish()

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
        source="bsense_p300_dry_run",
    )
    controller.disarm()
    controller.close()

    result = {
        "ok": True,
        "dry_run": True,
        "chain": [
            "xdf",
            "erp_features",
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
        "confidence": decision.confidence,
        "margin": decision.margin,
        "flash_counts": decision.flash_counts,
        "scores": decision.scores,
        "bridge_response": command_response,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

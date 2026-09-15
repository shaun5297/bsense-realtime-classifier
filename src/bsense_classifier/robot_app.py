"""Dedicated P300 visual control console for a Unitree quadruped."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, replace
import json
import queue
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import BOTH, END, LEFT, RIGHT, X, filedialog, messagebox, ttk

from .p300_control import (
    P300_COMMANDS,
    P300_LABELS,
    P300ControlEngine,
    P300Decision,
    P300EngineConfig,
    P300TargetRuntime,
)
from .robot_bridge import (
    BridgeError,
    RobotBridgeClient,
    RobotBridgeConfig,
    RobotCommand,
    SafeRobotController,
    SafetyInterlockError,
)


BG = "#0B1220"
CARD = "#121D2F"
CARD_LIGHT = "#19273D"
TEXT = "#EEF5FF"
MUTED = "#94A8C3"
BLUE = "#3B82F6"
CYAN = "#22D3EE"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"
BORDER = "#29405F"
HIGHLIGHT_MS = 100
SOA_MS = 175
SEQUENCES_PER_TRIAL = 10
COMMAND_ICONS: dict[RobotCommand, str] = {
    "forward": "↑",
    "backward": "↓",
    "left": "←",
    "right": "→",
    "stop": "■",
    "idle": "●",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_model_path() -> str:
    path = _project_root() / "models" / "m7_p300" / "model.joblib"
    return str(path) if path.exists() else ""


def _default_bridge_config() -> str:
    return str(_project_root() / "config" / "unitree_bridge.json")


class RobotControlApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.sequences_per_trial = getattr(args, "sequences_per_trial", SEQUENCES_PER_TRIAL)
        if self.sequences_per_trial not in range(2, 11):
            raise ValueError("每次选择的闪烁轮数必须在 2–10 之间。")
        from .p300_control import P300EngineConfig as _EngineDefaults

        self.confidence_threshold = float(
            getattr(args, "confidence_threshold", _EngineDefaults.confidence_threshold)
        )
        self.margin_threshold = float(
            getattr(args, "margin_threshold", _EngineDefaults.margin_threshold)
        )
        self.min_quality_ratio = float(
            getattr(args, "min_quality_ratio", _EngineDefaults.minimum_quality_ratio)
        )
        for _name, _value in (
            ("置信度门槛", self.confidence_threshold),
            ("领先差门槛", self.margin_threshold),
            ("质量比例门槛", self.min_quality_ratio),
        ):
            if not 0.0 <= _value <= 1.0:
                raise ValueError(f"{_name}必须在 0–1 之间。")
        self.thresholds_relaxed = gate_is_relaxed(
            self.confidence_threshold, self.margin_threshold, self.min_quality_ratio
        )
        self.root = root
        self.root.title("BSense P300 脑控机器狗")
        self.root.geometry("1420x900")
        self.root.minsize(1180, 760)
        self.root.configure(bg=BG)

        self.runtime: P300TargetRuntime | None = None
        self.engine: P300ControlEngine | None = None
        self.robot: SafeRobotController | None = None
        self._ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._status_polling = False
        self._eeg_connected = False
        self._selection_running = False
        self._session_active = False
        self._trial_id = 0
        self._flash_plan: list[tuple[int, int, RobotCommand]] = []
        self._flash_index = 0
        self._previous_flash: RobotCommand | None = None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audit_path = _project_root() / "logs" / f"m7_control_{stamp}.jsonl"
        self.command_tiles: dict[RobotCommand, tk.Label] = {}
        self.score_labels: dict[RobotCommand, tk.StringVar] = {
            command: tk.StringVar(value="—") for command in P300_COMMANDS
        }

        self.model_path = tk.StringVar(value=args.model or _default_model_path())
        self.stream_name = tk.StringVar(value=args.stream_name)
        self.bridge_config_path = tk.StringVar(
            value=args.bridge_config or _default_bridge_config()
        )
        self.socket_path_override = args.socket_path.strip()
        self.decoder_status = tk.StringVar(value="等待加载 M7 模型")
        self.bridge_status = tk.StringVar(value="尚未连接")
        self.arm_status = tk.StringVar(value="未解锁")
        self.result_text = tk.StringVar(value="准备中")
        self.result_detail = tk.StringVar(
            value=(
                "模型未训练时可使用下方格子联调桥接，"
                "但不会伪装成脑电结果。"
            )
        )
        self.trial_status = tk.StringVar(value="Trial —")
        self.signal_status = tk.StringVar(value="EEG 未连接")

        self._configure_styles()
        self._build()
        self._load_bridge_config()
        if self.model_path.get():
            self._load_model()
        self.root.bind("<space>", lambda _event: self._emergency_stop())
        self.root.bind("<Escape>", lambda _event: self._emergency_stop())
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(80, self._poll_events)
        self.root.after(500, self._poll_bridge_status)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "TEntry",
            fieldbackground=CARD_LIGHT,
            foreground=TEXT,
            bordercolor=BORDER,
            insertcolor=TEXT,
            padding=(8, 7),
        )
        style.configure(
            "Primary.TButton",
            background=BLUE,
            foreground="white",
            borderwidth=0,
            padding=(14, 9),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#2563EB")])
        style.configure(
            "Secondary.TButton",
            background=CARD_LIGHT,
            foreground=TEXT,
            padding=(12, 8),
        )
        style.configure(
            "Danger.TButton",
            background=RED,
            foreground="white",
            borderwidth=0,
            padding=(20, 14),
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        style.map("Danger.TButton", background=[("active", "#DC2626")])

    @staticmethod
    def _label(
        parent: tk.Widget,
        text: str = "",
        *,
        variable: tk.StringVar | None = None,
        size: int = 10,
        weight: str = "normal",
        color: str = TEXT,
        bg: str = CARD,
        wrap: int = 0,
        anchor: str = "w",
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            textvariable=variable,
            font=("Microsoft YaHei UI", size, weight),
            fg=color,
            bg=bg,
            justify=LEFT,
            anchor=anchor,
            wraplength=wrap,
        )

    @staticmethod
    def _card(parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )

    def _build(self) -> None:
        header = tk.Frame(self.root, bg="#08101D", height=92)
        header.pack(fill=X)
        header.pack_propagate(False)
        title = tk.Frame(header, bg="#08101D")
        title.pack(fill=BOTH, padx=28, pady=16)
        self._label(
            title,
            "BSense P300 脑控机器狗",
            size=21,
            weight="bold",
            bg="#08101D",
        ).pack(anchor="w")
        self._label(
            title,
            "FP1 / FP2 · 六指令视觉选择 · 避障联锁 · 限时运动 · 独立急停",
            color=MUTED,
            bg="#08101D",
        ).pack(anchor="w", pady=(3, 0))
        self._label(
            title,
            "空格 / Esc 急停",
            size=11,
            weight="bold",
            color=RED,
            bg="#08101D",
        ).pack(side=RIGHT, anchor="e")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=BOTH, expand=True, padx=18, pady=18)
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=4)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(9, 0))

        self._build_status(left)
        self._build_grid(left)
        self._build_controls(right)
        self._build_scores(right)
        self._build_log(right)

    def _build_status(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=X, pady=(0, 12))
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill=X, padx=16, pady=13)
        items = (
            ("解码器", self.decoder_status),
            ("机器狗桥接", self.bridge_status),
            ("控制授权", self.arm_status),
            ("信号", self.signal_status),
        )
        for index, (title, variable) in enumerate(items):
            inner.grid_columnconfigure(index, weight=1)
            cell = tk.Frame(inner, bg=CARD_LIGHT, padx=10, pady=8)
            cell.grid(
                row=0,
                column=index,
                sticky="ew",
                padx=(0, 5) if index < len(items) - 1 else 0,
            )
            self._label(
                cell, title, size=8, color=MUTED, bg=CARD_LIGHT
            ).pack(fill=X)
            self._label(
                cell,
                variable=variable,
                size=9,
                weight="bold",
                bg=CARD_LIGHT,
                wrap=180,
            ).pack(fill=X, pady=(2, 0))

    def _build_grid(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=BOTH, expand=True)
        top = tk.Frame(card, bg=CARD)
        top.pack(fill=X, padx=18, pady=(15, 4))
        self._label(top, "注视想要执行的指令", size=15, weight="bold").pack(
            side=LEFT
        )
        self._label(
            top,
            variable=self.trial_status,
            color=CYAN,
            weight="bold",
        ).pack(side=RIGHT)
        self._label(
            card,
            "每轮约 10.5 秒。格子会逐个闪烁；"
            "保持头部不动，只注视目标格。",
            color=MUTED,
            wrap=760,
        ).pack(fill=X, padx=18, pady=(0, 12))
        grid = tk.Frame(card, bg=CARD)
        grid.pack(fill=BOTH, expand=True, padx=16, pady=(0, 16))
        for row in range(2):
            grid.grid_rowconfigure(row, weight=1)
        for column in range(3):
            grid.grid_columnconfigure(column, weight=1)
        for index, command in enumerate(P300_COMMANDS):
            tile = tk.Label(
                grid,
                text=f"{COMMAND_ICONS[command]}\n{P300_LABELS[command]}",
                font=("Microsoft YaHei UI", 25, "bold"),
                fg=TEXT,
                bg=CARD_LIGHT,
                activebackground=BLUE,
                activeforeground="white",
                relief="flat",
                bd=0,
                cursor="hand2",
                highlightbackground=BORDER,
                highlightthickness=2,
            )
            tile.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=7,
                pady=7,
            )
            tile.bind(
                "<Button-1>",
                lambda _event, selected=command: self._manual_command(selected),
            )
            self.command_tiles[command] = tile

    def _build_controls(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=X, pady=(0, 12))
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill=X, padx=16, pady=14)
        self._label(inner, "系统配置", size=13, weight="bold").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10)
        )
        inner.grid_columnconfigure(1, weight=1)
        rows = (
            ("M7 模型", self.model_path, self._browse_model),
            ("EEG 流", self.stream_name, None),
            ("桥接配置", self.bridge_config_path, self._browse_bridge_config),
        )
        for row, (title, variable, browse) in enumerate(rows, start=1):
            self._label(inner, title, color=MUTED).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=4
            )
            ttk.Entry(inner, textvariable=variable).grid(
                row=row, column=1, columnspan=2, sticky="ew", pady=4
            )
            if browse:
                ttk.Button(
                    inner,
                    text="选择",
                    command=browse,
                    style="Secondary.TButton",
                ).grid(row=row, column=3, padx=(7, 0), pady=4)

        row = tk.Frame(inner, bg=CARD)
        row.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(12, 5))
        ttk.Button(
            row,
            text="加载模型",
            command=self._load_model,
            style="Secondary.TButton",
        ).pack(side=LEFT)
        self.connect_button = ttk.Button(
            row,
            text="连接 EEG",
            command=self._connect_eeg,
            style="Secondary.TButton",
        )
        self.connect_button.pack(side=LEFT, padx=7)
        self.arm_button = ttk.Button(
            row,
            text="安全检查并解锁",
            command=self._arm,
            style="Primary.TButton",
        )
        self.arm_button.pack(side=LEFT)

        actions = tk.Frame(inner, bg=CARD)
        actions.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.start_button = ttk.Button(
            actions,
            text="开始连续脑控",
            command=self._start_session,
            style="Primary.TButton",
        )
        self.start_button.pack(side=LEFT)
        ttk.Button(
            actions,
            text="暂停脑控",
            command=self._pause_session,
            style="Secondary.TButton",
        ).pack(side=LEFT, padx=7)
        ttk.Button(
            actions,
            text="立即急停",
            command=self._emergency_stop,
            style="Danger.TButton",
        ).pack(side=RIGHT)

        self._label(
            inner,
            "点击六宫格属于桥接联调指令，不会被记录为脑控结果；"
            "真实移动前仍必须解锁。",
            size=9,
            color=AMBER,
            wrap=560,
        ).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 0))

    def _build_scores(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=X, pady=(0, 12))
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill=X, padx=16, pady=14)
        self._label(inner, "实时解码结果", size=13, weight="bold").pack(fill=X)
        self._label(
            inner,
            variable=self.result_text,
            size=26,
            weight="bold",
            color=CYAN,
            wrap=540,
        ).pack(fill=X, pady=(12, 2))
        self._label(
            inner,
            variable=self.result_detail,
            color=MUTED,
            wrap=540,
        ).pack(fill=X)
        score_grid = tk.Frame(inner, bg=CARD)
        score_grid.pack(fill=X, pady=(12, 0))
        for index, command in enumerate(P300_COMMANDS):
            row = tk.Frame(score_grid, bg=CARD)
            row.grid(row=index // 3, column=index % 3, sticky="ew", padx=4, pady=2)
            score_grid.grid_columnconfigure(index % 3, weight=1)
            self._label(row, P300_LABELS[command], size=9, color=MUTED).pack(
                side=LEFT
            )
            self._label(
                row,
                variable=self.score_labels[command],
                size=9,
                weight="bold",
                color=TEXT,
            ).pack(side=RIGHT)

    def _build_log(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=BOTH, expand=True)
        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill=BOTH, expand=True, padx=16, pady=14)
        self._label(inner, "运行记录", size=13, weight="bold").pack(
            fill=X, pady=(0, 8)
        )
        self.log = tk.Text(
            inner,
            height=10,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            fg=TEXT,
            bg="#0D1727",
            insertbackground=TEXT,
            relief="flat",
            padx=9,
            pady=8,
            state="disabled",
        )
        self.log.pack(fill=BOTH, expand=True)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 M7 P300 模型",
            filetypes=[("Joblib 模型", "*.joblib"), ("全部文件", "*.*")],
        )
        if path:
            self.model_path.set(path)
            self._load_model()

    def _browse_bridge_config(self) -> None:
        path = filedialog.askopenfilename(
            title="选择机器狗桥接配置",
            filetypes=[("JSON 配置", "*.json"), ("全部文件", "*.*")],
        )
        if path:
            self.bridge_config_path.set(path)
            self._load_bridge_config()

    def _load_bridge_config(self) -> None:
        previous_robot = self.robot
        try:
            config = RobotBridgeConfig.load(self.bridge_config_path.get())
            if self.socket_path_override:
                config = replace(config, socket_path=self.socket_path_override)
            self.robot = SafeRobotController(RobotBridgeClient(config))
        except Exception as exc:
            if previous_robot is not None:
                previous_robot.close()
            self.robot = None
            self.bridge_status.set("配置无效")
            self._append_log(f"桥接配置加载失败：{exc}", error=True)
            return
        if previous_robot is not None:
            previous_robot.close()
        self.arm_status.set("未解锁")
        self.bridge_status.set(f"{config.transport} · 待检查")
        self._append_log(
            f"桥接配置已加载：{config.transport}；"
            f"动作脉冲 {config.motion_duration_seconds:g} 秒"
        )

    def _load_model(self) -> None:
        if self.engine and self.engine.running:
            messagebox.showinfo("正在运行", "请先断开 EEG，再更换模型。")
            return
        path = self.model_path.get().strip()
        if not path:
            self.runtime = None
            self.decoder_status.set("M7 模型尚未训练")
            self._append_log("未提供 M7 模型，仅可进行桥接与急停联调。")
            return
        try:
            self.runtime = P300TargetRuntime.load(path)
        except Exception as exc:
            self.runtime = None
            self.decoder_status.set("模型无效")
            self._append_log(f"M7 模型加载失败：{exc}", error=True)
            messagebox.showerror("M7 模型无法加载", str(exc))
            return
        runtime = self.runtime.runtime
        self.decoder_status.set("模型已加载")
        self._append_log(
            f"M7 模型已加载：{runtime.model_name}；"
            f"窗口 {runtime.window_seconds:g} 秒，偏移 "
            f"{runtime.window_offset_seconds:+g} 秒"
        )

    def _connect_eeg(self) -> None:
        if self.runtime is None:
            messagebox.showinfo(
                "模型未就绪",
                "请先加载训练完成的 task=m7_p300 二分类模型。",
            )
            return
        if self.engine and self.engine.running:
            return
        self.engine = P300ControlEngine(
            self.runtime,
            P300EngineConfig(
                stream_name=self.stream_name.get().strip(),
                minimum_flashes_per_command=min(6, self.sequences_per_trial),
                confidence_threshold=self.confidence_threshold,
                margin_threshold=self.margin_threshold,
                minimum_quality_ratio=self.min_quality_ratio,
            ),
        )
        self.engine.start()
        self.decoder_status.set("正在连接 EEG")
        self.connect_button.configure(state="disabled")

    def _run_background(self, name: str, action: object) -> None:
        def worker() -> None:
            try:
                result = action()
            except Exception as exc:
                self._ui_events.put((f"{name}_error", exc))
            else:
                self._ui_events.put((f"{name}_ok", result))

        threading.Thread(target=worker, daemon=True).start()

    def _arm(self) -> None:
        if self.robot is None:
            self._load_bridge_config()
        if self.robot is None:
            return
        self.arm_status.set("安全检查中")
        self._run_background("arm", self.robot.arm)

    def _manual_command(self, command: RobotCommand) -> None:
        if self._session_active:
            return
        if self.robot is None:
            messagebox.showerror("桥接未就绪", "请先加载有效的机器狗桥接配置。")
            return
        self.result_text.set(f"联调：{P300_LABELS[command]}")
        self.result_detail.set("人工点击指令，不属于脑电解码结果。")
        self._write_audit(
            {
                "kind": "manual_command",
                "command": command,
                "brain_decoded": False,
            }
        )
        self._run_background(
            "command",
            lambda: self.robot.execute(command, source="bsense_manual_test"),
        )

    def _start_session(self) -> None:
        if not self.runtime or not self.engine or not self.engine.running:
            messagebox.showinfo("EEG 未就绪", "请先加载 M7 模型并连接 EEG。")
            return
        if not self._eeg_connected:
            messagebox.showinfo("EEG 未连接", "正在等待有效的 FP1/FP2 LSL 数据流。")
            return
        if not self.robot or not self.robot.armed:
            messagebox.showinfo(
                "尚未解锁",
                "请先完成桥接、避障状态和急停检查。",
            )
            return
        self._session_active = True
        self.start_button.configure(state="disabled")
        self._begin_trial()

    def _pause_session(self) -> None:
        self._session_active = False
        self._selection_running = False
        self.start_button.configure(state="normal")
        self.result_detail.set("脑控已暂停；控制器仍保持解锁，可继续或急停。")
        if self.robot:
            self._run_background(
                "command",
                lambda: self.robot.execute("idle", source="bsense_pause"),
            )

    def _build_flash_plan(self) -> list[tuple[int, int, RobotCommand]]:
        plan: list[tuple[int, int, RobotCommand]] = []
        previous = self._previous_flash
        for sequence in range(1, self.sequences_per_trial + 1):
            order = list(P300_COMMANDS)
            for _attempt in range(20):
                random.shuffle(order)
                if previous is None or order[0] != previous:
                    break
            plan.extend(
                (sequence, P300_COMMANDS.index(command), command)
                for command in order
            )
            previous = order[-1]
        self._previous_flash = previous
        return plan

    def _begin_trial(self) -> None:
        if not self._session_active or not self.engine or not self.engine.running:
            return
        self._trial_id += 1
        self._flash_plan = self._build_flash_plan()
        self._flash_index = 0
        self._selection_running = True
        self.trial_status.set(f"Trial {self._trial_id} · 0/{self.sequences_per_trial}")
        self.result_text.set("注视目标")
        self.result_detail.set("刺激开始，请减少眨眼和面部动作。")
        for variable in self.score_labels.values():
            variable.set("—")
        self.root.after(500, self._flash_next)

    def _flash_next(self) -> None:
        if (
            not self._selection_running
            or not self._session_active
            or not self.engine
            or not self.engine.running
        ):
            return
        if self._flash_index >= len(self._flash_plan):
            self._selection_running = False
            self.result_text.set("正在聚合")
            self.result_detail.set("等待最后一个 ERP 窗口完成。")
            self.engine.finish_trial(self._trial_id)
            return
        sequence, position, command = self._flash_plan[self._flash_index]
        self._flash_index += 1
        self.trial_status.set(
            f"Trial {self._trial_id} · {sequence}/{self.sequences_per_trial}"
        )
        tile = self.command_tiles[command]
        tile.configure(bg=CYAN, fg=BG, highlightbackground="white")
        tile.update_idletasks()
        self.engine.submit_flash(
            trial_id=self._trial_id,
            sequence=sequence,
            position=position,
            command=command,
        )
        self.root.after(
            HIGHLIGHT_MS,
            lambda selected=command: self._end_highlight(selected),
        )
        self.root.after(SOA_MS, self._flash_next)

    def _end_highlight(self, command: RobotCommand) -> None:
        self.command_tiles[command].configure(
            bg=CARD_LIGHT,
            fg=TEXT,
            highlightbackground=BORDER,
        )

    def _handle_decision(self, decision: P300Decision) -> None:
        label = P300_LABELS[decision.command]
        self._write_audit(
            {
                "kind": "p300_decision",
                **asdict(decision),
                "brain_decoded": True,
            }
        )
        for command, score in decision.scores.items():
            self.score_labels[command].set(f"{score * 100:.1f}%")
        if not decision.accepted:
            self.result_text.set(f"候选：{label}")
            reasons = "、".join(decision.rejection_reasons)
            self.result_detail.set(
                f"未下发：置信度 {decision.confidence * 100:.1f}%，"
                f"领先 {decision.margin * 100:.1f}%；{reasons}"
            )
            self._append_log(
                f"Trial {decision.trial_id} 未接纳：{label}；{reasons}"
            )
            if self._session_active:
                self.root.after(1200, self._begin_trial)
            return
        self.result_text.set(label)
        self.result_detail.set(
            f"已接纳：置信度 {decision.confidence * 100:.1f}%，"
            f"领先第二名 {decision.margin * 100:.1f}%"
        )
        self._append_log(
            f"Trial {decision.trial_id} 接纳：{label}，"
            f"置信度 {decision.confidence * 100:.1f}%"
        )
        if self.robot is None:
            return
        if decision.command == "stop":
            self._session_active = False
            self.start_button.configure(state="normal")
        self._run_background(
            "decision_command",
            lambda: self.robot.execute(
                decision.command,
                confidence=decision.confidence,
                source="bsense_p300",
            ),
        )

    def _emergency_stop(self) -> None:
        self._session_active = False
        self._selection_running = False
        self.start_button.configure(state="normal")
        self.arm_status.set("急停锁定")
        self.result_text.set("急停")
        self.result_detail.set(
            "移动输出已归零；重新控制前必须再次安全检查并解锁。"
        )
        if self.robot:
            self._run_background("emergency", self.robot.emergency_stop)

    def _poll_bridge_status(self) -> None:
        if self.robot and not self._status_polling:
            self._status_polling = True
            self._run_background("status", self.robot.bridge.status)
        self.root.after(1000, self._poll_bridge_status)

    def _poll_events(self) -> None:
        if self.engine:
            while True:
                try:
                    event = self.engine.events.get_nowait()
                except queue.Empty:
                    break
                self._handle_engine_event(event)
        while True:
            try:
                kind, value = self._ui_events.get_nowait()
            except queue.Empty:
                break
            self._handle_ui_event(kind, value)
        self.root.after(80, self._poll_events)

    def _handle_engine_event(self, event: dict[str, object]) -> None:
        kind = event.get("kind")
        if kind == "connected":
            self._eeg_connected = True
            self.decoder_status.set("P300 解码器在线")
            self.signal_status.set(
                f"FP1/FP2 · {float(event['input_sfreq']):g} Hz"
            )
            self._append_log(f"EEG 已连接：{event['stream_name']}")
            self._append_log(
                f"门控门槛：置信度 ≥ {self.confidence_threshold:g}、"
                f"领先差 ≥ {self.margin_threshold:g}、"
                f"质量比例 ≥ {self.min_quality_ratio:g}、"
                f"每指令 ≥ {min(6, self.sequences_per_trial)} 次有效闪烁"
            )
            if self.thresholds_relaxed:
                self._append_log(
                    "警告：当前门槛低于默认值，接受率会被放宽，"
                    "解码结果不足以作为真实运动依据。"
                )
            warning = event.get("channel_warning")
            if warning:
                self._append_log(str(warning))
        elif kind == "status":
            self._append_log(str(event.get("message", "")))
        elif kind == "flash_result":
            quality_ok = bool(event.get("quality_ok"))
            if not quality_ok:
                self.signal_status.set(f"质量异常：{event.get('signal_quality')}")
        elif kind == "decision":
            self._handle_decision(event["decision"])
        elif kind == "flash_skipped":
            self._append_log(
                f"跳过不完整窗口：Trial {event.get('trial_id')} "
                f"{event.get('command')}"
            )
        elif kind == "error":
            self._eeg_connected = False
            self.decoder_status.set("解码器错误")
            self.signal_status.set("EEG 连接失败")
            self._append_log(str(event.get("message", "")), error=True)
            self._emergency_stop()
        elif kind == "stopped":
            self._eeg_connected = False
            self.connect_button.configure(state="normal")

    def _handle_ui_event(self, kind: str, value: object) -> None:
        if kind == "status_ok":
            self._status_polling = False
            status = value
            if not status.connected:
                self.bridge_status.set("离线")
            elif status.obstacle_detected is True:
                self.bridge_status.set("障碍物联锁")
            elif status.obstacle_detected is False:
                distance = (
                    status.raw.get("minimum_obstacle_m")
                    if status.raw
                    else None
                )
                if isinstance(distance, (int, float)):
                    self.bridge_status.set(f"在线 · 最近 {distance:.2f} m")
                else:
                    self.bridge_status.set("在线 · 通道安全")
            else:
                self.bridge_status.set("在线 · 避障未知")
            if status.emergency_stopped:
                self.arm_status.set("桥接端急停")
        elif kind == "status_error":
            self._status_polling = False
            self.bridge_status.set("状态读取失败")
        elif kind == "arm_ok":
            self.arm_status.set("已解锁")
            self.bridge_status.set("在线 · 通道安全")
            self._append_log("安全检查通过，控制器已解锁。")
        elif kind == "arm_error":
            self.arm_status.set("解锁失败")
            self._append_log(f"解锁失败：{value}", error=True)
        elif kind in {"command_ok", "decision_command_ok"}:
            self._append_log("机器狗指令已发送。")
            if kind == "decision_command_ok" and self._session_active:
                self.root.after(1200, self._begin_trial)
        elif kind in {"command_error", "decision_command_error"}:
            self._append_log(f"机器狗指令被拦截：{value}", error=True)
            if isinstance(value, (SafetyInterlockError, BridgeError)):
                self.arm_status.set("安全联锁")
                self._session_active = False
                self.start_button.configure(state="normal")
        elif kind == "emergency_ok":
            self._append_log("急停指令已发送，控制器已锁定。")
        elif kind == "emergency_error":
            self._append_log(f"急停发送失败：{value}", error=True)

    def _append_log(self, message: str, *, error: bool = False) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert(END, f"[{stamp}] {message}\n", "error" if error else "")
        self.log.tag_configure("error", foreground=RED)
        self.log.see(END)
        self.log.configure(state="disabled")

    def _write_audit(self, record: dict[str, object]) -> None:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sequences_per_trial": self.sequences_per_trial,
            **record,
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _close(self) -> None:
        self._session_active = False
        self._selection_running = False
        if self.engine:
            self.engine.stop(timeout=1.0)
        if self.robot:
            try:
                self.robot.emergency_stop()
            except BridgeError:
                pass
            self.robot.close()
        self.root.destroy()


def gate_is_relaxed(confidence: float, margin: float, quality: float) -> bool:
    """True when any gate is looser than the shipped default, so it can be surfaced."""
    from .p300_control import P300EngineConfig

    defaults = P300EngineConfig()
    return (
        confidence < defaults.confidence_threshold
        or margin < defaults.margin_threshold
        or quality < defaults.minimum_quality_ratio
    )


def _unit_float(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f"门槛必须在 0–1 之间，收到 {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BSense P300 脑控机器狗控制台")
    parser.add_argument(
        "--sequences-per-trial", type=int, choices=range(2, 11), default=SEQUENCES_PER_TRIAL,
        help="每次选择的闪烁轮数（默认 10）；缩短前需完成参赛者个体校准和在线验证。",
    )
    parser.add_argument("--model", default="", help="m7_p300 model.joblib 路径")
    parser.add_argument(
        "--confidence-threshold", type=_unit_float, default=0.55,
        help="胜出指令平均目标概率门槛（默认 0.55）；调低只用于台架联调，会放宽接受率。",
    )
    parser.add_argument(
        "--margin-threshold", type=_unit_float, default=0.08,
        help="第一名与第二名的概率差门槛（默认 0.08）；调低只用于台架联调。",
    )
    parser.add_argument(
        "--min-quality-ratio", type=_unit_float, default=0.8,
        help="合格 EEG 窗口比例门槛（默认 0.8）。",
    )
    parser.add_argument("--stream-name", default="", help="EEG LSL 流名称")
    parser.add_argument(
        "--bridge-config",
        default="",
        help="Unitree 桥接 JSON 配置路径",
    )
    parser.add_argument(
        "--socket-path",
        default="",
        help="覆盖 unitree_ros2 配置中的 Unix Socket 路径",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    RobotControlApp(root, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

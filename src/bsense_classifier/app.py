"""Tk desktop application for the BSense real-time classifier."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import queue
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from .lsl_engine import EngineConfig, RealtimeEngine
from .model_runtime import ArtifactError, ModelRuntime
from .task_guidance import TaskGuidance, guidance_for


APP_BG = "#F3F6FA"
CARD_BG = "#FFFFFF"
NAVY = "#17324D"
BLUE = "#2878D0"
GREEN = "#14805E"
AMBER = "#B4690E"
RED = "#BC3B3B"
MUTED = "#617184"
BORDER = "#DCE4EC"
MIN_CONFIDENCE_THRESHOLD = 0.30
MAX_CONFIDENCE_THRESHOLD = 0.95


def parse_confidence_threshold(value: object) -> float:
    threshold = float(value)
    if (
        not math.isfinite(threshold)
        or threshold < MIN_CONFIDENCE_THRESHOLD
        or threshold > MAX_CONFIDENCE_THRESHOLD
    ):
        raise ValueError("confidence threshold is outside the supported range")
    return threshold


def _default_model_path() -> str:
    project = Path(__file__).resolve().parents[2]
    preferred = project / "models" / "m3a_artifact" / "model.joblib"
    return str(preferred) if preferred.exists() else ""


class ClassifierApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("BSense 实时脑电分类器")
        self.root.geometry("1240x840")
        self.root.minsize(1040, 720)
        self.root.configure(bg=APP_BG)
        self.runtime: ModelRuntime | None = None
        self.engine: RealtimeEngine | None = None
        self.probability_widgets: list[tk.Widget] = []

        self.model_path = tk.StringVar(value=args.model or _default_model_path())
        self.preset_task = tk.StringVar(value="m3a_artifact")
        self.stream_name = tk.StringVar(value=args.stream_name)
        self.marker_name = tk.StringVar(value=args.marker_stream_name)
        self.output_path = tk.StringVar(value=args.output)
        self._output_is_auto = not bool(args.output)
        self.threshold = tk.DoubleVar(value=0.6)
        self.auto_event_analysis = tk.BooleanVar(value=True)
        self.show_rejected_candidate = tk.BooleanVar(value=True)
        self.connection_text = tk.StringVar(value="尚未连接")
        self.channel_text = tk.StringVar(value="等待验证 FP1 / FP2")
        self.quality_text = tk.StringVar(value="尚无数据")
        self.modality_text = tk.StringVar(value="模型输入：仅 EEG")
        self.result_text = tk.StringVar(value="—")
        self.result_detail = tk.StringVar(value="加载模型后即可开始")
        self.confidence_text = tk.StringVar(value="—")
        self.task_title = tk.StringVar(value="尚未加载任务模型")
        self.task_goal = tk.StringVar(value="选择 .joblib 模型文件，界面会显示具体实验步骤。")
        self.model_meta = tk.StringVar(value="")

        self._configure_styles()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._poll_events)
        if self.model_path.get():
            self.root.after(150, self._load_model)
        if args.autostart:
            self.root.after(500, self._start)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure(
            "TEntry",
            padding=(7, 6),
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "TCombobox",
            padding=(7, 6),
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "Primary.TButton",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="white",
            background=BLUE,
            padding=(16, 9),
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1F68B6"), ("disabled", "#9EB8D2")],
        )
        style.configure(
            "Secondary.TButton",
            font=("Microsoft YaHei UI", 10),
            padding=(13, 8),
        )
        style.configure(
            "Danger.TButton",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="white",
            background=RED,
            padding=(16, 9),
            borderwidth=0,
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#9D3030"), ("disabled", "#D6A6A6")],
        )
        style.configure(
            "Confidence.Horizontal.TProgressbar",
            troughcolor="#E8EEF4",
            background=BLUE,
            bordercolor="#E8EEF4",
            lightcolor=BLUE,
            darkcolor=BLUE,
            thickness=14,
        )

    @staticmethod
    def _label(
        parent: tk.Widget,
        text: str = "",
        *,
        variable: tk.StringVar | None = None,
        size: int = 10,
        weight: str = "normal",
        color: str = NAVY,
        bg: str = CARD_BG,
        wrap: int = 0,
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            textvariable=variable,
            font=("Microsoft YaHei UI", size, weight),
            fg=color,
            bg=bg,
            anchor="w",
            justify=LEFT,
            wraplength=wrap,
        )

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=CARD_BG,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=NAVY, height=104)
        header.pack(fill=X)
        header.pack_propagate(False)
        title_wrap = tk.Frame(header, bg=NAVY)
        title_wrap.pack(fill=BOTH, padx=32, pady=19)
        self._label(
            title_wrap,
            "BSense 实时脑电分类器",
            size=21,
            weight="bold",
            color="white",
            bg=NAVY,
        ).pack(anchor="w")
        self._label(
            title_wrap,
            "加载已训练模型 · 接收 LSL · 验证 FP1/FP2 · 输出实时结果",
            size=10,
            color="#C9D7E5",
            bg=NAVY,
        ).pack(anchor="w", pady=(4, 0))

        content = tk.Frame(self.root, bg=APP_BG)
        content.pack(fill=BOTH, expand=True)
        canvas = tk.Canvas(
            content,
            bg=APP_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(
            content,
            orient="vertical",
            command=canvas.yview,
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")
        body = tk.Frame(canvas, bg=APP_BG)
        body_window = canvas.create_window(
            (0, 0),
            window=body,
            anchor="nw",
        )
        body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(body_window, width=event.width),
        )
        body.configure(padx=22, pady=18)
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=4)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=APP_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        right = tk.Frame(body, bg=APP_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(9, 0))

        self._build_setup(left)
        self._build_guidance(left)
        self._build_result(right)
        self._build_log(right)

    def _build_setup(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=X, pady=(0, 14))
        inner = tk.Frame(card, bg=CARD_BG)
        inner.pack(fill=X, padx=18, pady=15)
        self._label(inner, "连接设置", size=13, weight="bold").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
        )
        inner.grid_columnconfigure(1, weight=1)

        self._label(inner, "内置任务", color=MUTED).grid(
            row=1, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        preset = ttk.Combobox(
            inner,
            textvariable=self.preset_task,
            values=[
                "m1_mi",
                "m2_nback",
                "m3a_artifact",
                "m3b_fatigue",
                "m4a_intent",
                "m4b_target",
            ],
            state="readonly",
        )
        preset.grid(row=1, column=1, columnspan=3, sticky="ew", pady=5)
        preset.bind("<<ComboboxSelected>>", self._select_preset)

        self._label(inner, "模型文件", color=MUTED).grid(
            row=2, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        ttk.Entry(inner, textvariable=self.model_path).grid(
            row=2, column=1, sticky="ew", pady=5
        )
        ttk.Button(
            inner,
            text="浏览",
            command=self._browse_model,
            style="Secondary.TButton",
        ).grid(row=2, column=2, padx=(8, 0), pady=5)
        ttk.Button(
            inner,
            text="加载",
            command=self._load_model,
            style="Secondary.TButton",
        ).grid(row=2, column=3, padx=(8, 0), pady=5)

        self._label(inner, "EEG 流名称", color=MUTED).grid(
            row=3, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        ttk.Entry(inner, textvariable=self.stream_name).grid(
            row=3, column=1, columnspan=3, sticky="ew", pady=5
        )
        self._label(
            inner,
            "留空时自动选择第一个 type=EEG 的 LSL 流",
            color=MUTED,
            size=9,
        ).grid(row=4, column=1, columnspan=3, sticky="w")

        self._label(inner, "Marker 流", color=MUTED).grid(
            row=5, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        ttk.Entry(inner, textvariable=self.marker_name).grid(
            row=5, column=1, columnspan=3, sticky="ew", pady=5
        )

        self._label(inner, "结果记录", color=MUTED).grid(
            row=6, column=0, sticky="w", pady=5, padx=(0, 10)
        )
        ttk.Entry(inner, textvariable=self.output_path).grid(
            row=6, column=1, columnspan=2, sticky="ew", pady=5
        )
        ttk.Button(
            inner,
            text="选择",
            command=self._browse_output,
            style="Secondary.TButton",
        ).grid(row=6, column=3, padx=(8, 0), pady=5)

        options = tk.Frame(inner, bg=CARD_BG)
        options.grid(
            row=7,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(8, 2),
        )
        self._label(options, "接纳阈值", color=MUTED).pack(side=LEFT)
        threshold_spinbox = ttk.Spinbox(
            options,
            from_=MIN_CONFIDENCE_THRESHOLD,
            to=MAX_CONFIDENCE_THRESHOLD,
            increment=0.01,
            width=6,
            textvariable=self.threshold,
            format="%.2f",
        )
        threshold_spinbox.pack(side=LEFT, padx=(8, 18))
        self.auto_event_check = ttk.Checkbutton(
            options,
            text="M1/M4A 自动滚动分析",
            variable=self.auto_event_analysis,
        )
        self.auto_event_check.pack(side=LEFT)
        ttk.Checkbutton(
            options,
            text="低置信度时显示候选",
            variable=self.show_rejected_candidate,
        ).pack(side=LEFT, padx=(16, 0))

        controls = tk.Frame(inner, bg=CARD_BG)
        controls.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(13, 0))
        self.start_button = ttk.Button(
            controls,
            text="开始接收并分析",
            command=self._start,
            style="Primary.TButton",
        )
        self.start_button.pack(side=LEFT)
        self.stop_button = ttk.Button(
            controls,
            text="停止",
            command=self._stop,
            style="Danger.TButton",
            state="disabled",
        )
        self.stop_button.pack(side=LEFT, padx=8)
        self.trigger_button = ttk.Button(
            controls,
            text="立即对齐触发一次",
            command=self._manual_trigger,
            style="Secondary.TButton",
            state="disabled",
        )
        self.trigger_button.pack(side=RIGHT)

    def _build_guidance(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=BOTH, expand=True)
        self.guide_frame = tk.Frame(card, bg=CARD_BG)
        self.guide_frame.pack(fill=BOTH, expand=True, padx=18, pady=15)
        self._label(
            self.guide_frame,
            variable=self.task_title,
            size=14,
            weight="bold",
            wrap=620,
        ).pack(fill=X)
        self._label(
            self.guide_frame,
            variable=self.model_meta,
            size=9,
            color=BLUE,
            wrap=620,
        ).pack(fill=X, pady=(4, 8))
        self._label(
            self.guide_frame,
            variable=self.task_goal,
            size=10,
            color=MUTED,
            wrap=620,
        ).pack(fill=X)
        separator = tk.Frame(self.guide_frame, bg=BORDER, height=1)
        separator.pack(fill=X, pady=12)
        self.steps_frame = tk.Frame(self.guide_frame, bg=CARD_BG)
        self.steps_frame.pack(fill=X)
        self.trigger_note_label = self._label(
            self.guide_frame, color=BLUE, wrap=620
        )
        self.trigger_note_label.pack(fill=X, pady=(12, 4))
        self.limitation_label = self._label(
            self.guide_frame, color=AMBER, size=9, wrap=620
        )
        self.limitation_label.pack(fill=X)

    def _build_result(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=X, pady=(0, 14))
        inner = tk.Frame(card, bg=CARD_BG)
        inner.pack(fill=X, padx=18, pady=15)
        top = tk.Frame(inner, bg=CARD_BG)
        top.pack(fill=X)
        self._label(top, "实时结果", size=13, weight="bold").pack(side=LEFT)
        self.status_badge = self._label(
            top,
            variable=self.connection_text,
            size=9,
            color=MUTED,
        )
        self.status_badge.pack(side=RIGHT)
        self.result_label = self._label(
            inner,
            variable=self.result_text,
            size=29,
            weight="bold",
            color=NAVY,
            wrap=450,
        )
        self.result_label.pack(fill=X, pady=(19, 2))
        self._label(
            inner,
            variable=self.result_detail,
            color=MUTED,
            wrap=450,
        ).pack(fill=X)

        confidence_row = tk.Frame(inner, bg=CARD_BG)
        confidence_row.pack(fill=X, pady=(15, 4))
        self._label(confidence_row, "置信度", color=MUTED).pack(side=LEFT)
        self._label(
            confidence_row,
            variable=self.confidence_text,
            color=BLUE,
            weight="bold",
        ).pack(side=RIGHT)
        self.confidence_bar = ttk.Progressbar(
            inner,
            maximum=100,
            value=0,
            style="Confidence.Horizontal.TProgressbar",
        )
        self.confidence_bar.pack(fill=X)

        status_grid = tk.Frame(inner, bg=CARD_BG)
        status_grid.pack(fill=X, pady=(15, 0))
        status_grid.grid_columnconfigure(0, weight=1)
        status_grid.grid_columnconfigure(1, weight=1)
        status_grid.grid_columnconfigure(2, weight=1)
        self._status_cell(
            status_grid, 0, "通道", self.channel_text
        )
        self._status_cell(
            status_grid, 1, "输入模态", self.modality_text
        )
        self._status_cell(
            status_grid, 2, "信号质量", self.quality_text
        )
        self.probability_frame = tk.Frame(inner, bg=CARD_BG)
        self.probability_frame.pack(fill=X, pady=(12, 0))

    def _status_cell(
        self,
        parent: tk.Widget,
        column: int,
        title: str,
        variable: tk.StringVar,
    ) -> None:
        frame = tk.Frame(parent, bg="#F7F9FC", padx=10, pady=8)
        frame.grid(
            row=0,
            column=column,
            sticky="ew",
            padx=(0, 5) if column == 0 else ((5, 5) if column == 1 else (5, 0)),
        )
        self._label(
            frame, title, size=8, color=MUTED, bg="#F7F9FC"
        ).pack(fill=X)
        self._label(
            frame,
            variable=variable,
            size=9,
            weight="bold",
            bg="#F7F9FC",
            wrap=200,
        ).pack(fill=X, pady=(2, 0))

    def _build_log(self, parent: tk.Widget) -> None:
        card = self._card(parent)
        card.pack(fill=BOTH, expand=True)
        inner = tk.Frame(card, bg=CARD_BG)
        inner.pack(fill=BOTH, expand=True, padx=18, pady=15)
        self._label(inner, "运行记录", size=13, weight="bold").pack(
            fill=X, pady=(0, 9)
        )
        self.log = tk.Text(
            inner,
            height=10,
            wrap="word",
            font=("Microsoft YaHei UI", 9),
            fg=NAVY,
            bg="#F7F9FC",
            relief="flat",
            padx=10,
            pady=8,
            state="disabled",
        )
        self.log.pack(fill=BOTH, expand=True)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="选择分类模型",
            filetypes=[("Joblib 模型", "*.joblib"), ("全部文件", "*.*")],
        )
        if path:
            self.model_path.set(path)
            self._load_model()

    def _select_preset(self, _event: object | None = None) -> None:
        project = Path(__file__).resolve().parents[2]
        path = project / "models" / self.preset_task.get() / "model.joblib"
        if not path.exists():
            messagebox.showerror("模型缺失", f"没有找到内置模型：{path}")
            return
        self.model_path.set(str(path))
        self._load_model()

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存实时结果",
            defaultextension=".jsonl",
            filetypes=[("JSON Lines", "*.jsonl"), ("全部文件", "*.*")],
        )
        if path:
            self.output_path.set(path)
            self._output_is_auto = False

    def _load_model(self) -> None:
        if self.engine and self.engine.running:
            messagebox.showinfo("正在运行", "请先停止实时分析，再更换模型。")
            return
        try:
            self.runtime = ModelRuntime.load(self.model_path.get())
        except Exception as exc:
            self.runtime = None
            self._append_log(f"模型加载失败：{exc}", error=True)
            messagebox.showerror("模型无法加载", str(exc))
            return
        self.threshold.set(self.runtime.default_threshold)
        if self.runtime.task in {
            "m1_mi",
            "m2_nback",
            "m3a_artifact",
            "m3b_fatigue",
            "m4a_intent",
            "m4b_target",
        }:
            self.preset_task.set(self.runtime.task)
        if self._output_is_auto or not self.output_path.get():
            project = Path(__file__).resolve().parents[2]
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path.set(
                str(project / "logs" / f"{self.runtime.task}_{stamp}.jsonl")
            )
        guidance = guidance_for(self.runtime.task)
        self._show_guidance(guidance)
        self.task_title.set(guidance.title)
        self.task_goal.set(guidance.short_goal)
        kind = "分类" if self.runtime.target_kind == "classification" else "数值估计"
        self.model_meta.set(
            f"{self.runtime.task} · {kind} · {self.runtime.window_seconds:g} 秒窗口 "
            f"· 模型采样率 {self.runtime.sfreq:g} Hz · 输入仅 EEG"
        )
        self.result_text.set("准备就绪")
        self.result_detail.set("确认任务说明后，点击“开始接收并分析”。")
        self.trigger_button.configure(
            state="normal" if self.runtime.event_locked else "disabled"
        )
        supports_auto = self.runtime.task in {"m1_mi", "m4a_intent"}
        self.auto_event_analysis.set(supports_auto)
        self.auto_event_check.configure(
            state="normal" if supports_auto else "disabled"
        )
        self._append_log(
            f"已加载模型：{self.runtime.task}（{self.runtime.model_name}）；"
            "当前模型只使用 FP1/FP2 EEG"
        )

    def _show_guidance(self, guidance: TaskGuidance) -> None:
        for child in self.steps_frame.winfo_children():
            child.destroy()
        for index, step in enumerate(guidance.steps, start=1):
            row = tk.Frame(self.steps_frame, bg=CARD_BG)
            row.pack(fill=X, pady=3)
            badge = tk.Label(
                row,
                text=str(index),
                font=("Microsoft YaHei UI", 9, "bold"),
                fg="white",
                bg=BLUE,
                width=2,
                pady=2,
            )
            badge.pack(side=LEFT, anchor="n", padx=(0, 8))
            self._label(row, step, wrap=560).pack(fill=X)
        self.trigger_note_label.configure(text=f"触发方式：{guidance.trigger_note}")
        self.limitation_label.configure(text=f"实验提示：{guidance.limitation}")

    def _start(self) -> None:
        if self.runtime is None:
            self._load_model()
        if self.runtime is None:
            return
        if self.engine and self.engine.running:
            return
        output = Path(self.output_path.get()) if self.output_path.get() else None
        try:
            threshold = parse_confidence_threshold(self.threshold.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("阈值无效", "接纳阈值必须是 0.30–0.95 之间的数字。")
            return
        config = EngineConfig(
            stream_name=self.stream_name.get().strip(),
            marker_stream_name=self.marker_name.get().strip()
            or "BSense Experiment Markers",
            confidence_threshold=threshold,
            smoothing_windows=3,
            confirmation_windows=2,
            auto_analyze_event_tasks=self.auto_event_analysis.get(),
            output_path=output,
        )
        self.engine = RealtimeEngine(self.runtime, config)
        self.engine.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.trigger_button.configure(
            state="normal" if self.runtime.event_locked else "disabled"
        )
        self.connection_text.set("正在查找 LSL")
        self.status_badge.configure(fg=AMBER)
        self.result_detail.set("正在等待实时 EEG 数据……")

    def _stop(self) -> None:
        if self.engine:
            self.engine.stop()
        self._set_stopped()

    def _set_stopped(self) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.connection_text.set("已停止")
        self.status_badge.configure(fg=MUTED)
        self.trigger_button.configure(
            state="normal"
            if self.runtime is not None and self.runtime.event_locked
            else "disabled"
        )

    def _manual_trigger(self) -> None:
        if not self.engine or not self.engine.running:
            messagebox.showinfo("尚未运行", "请先点击“开始接收并分析”。")
            return
        self.engine.manual_trigger()

    def _poll_events(self) -> None:
        if self.engine:
            while True:
                try:
                    event = self.engine.events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(event)
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "status":
            self._append_log(str(event.get("message", "")))
            return
        if kind == "connected":
            self.connection_text.set(f"已连接 · {event['stream_name']}")
            self.status_badge.configure(fg=GREEN)
            status = {
                "verified": "FP1 / FP2 已验证",
                "reordered": "FP2 / FP1 已自动换序",
                "assumed": "按第1路 FP1、第2路 FP2",
            }.get(event["channel_status"], event["channel_status"])
            self.channel_text.set(status)
            self._append_log(
                f"EEG 已连接：{event['stream_name']}，输入 {event['input_sfreq']:g} Hz"
            )
            available = event.get("available_stream_types") or []
            available_text = "、".join(available) if available else "未列出"
            self.modality_text.set(f"模型仅用 EEG；可见：{available_text}")
            self._append_log(
                f"当前模型输入仅 EEG；此刻 LSL 可见流类型：{available_text}"
            )
            if event.get("auto_event_analysis"):
                self._append_log("已启用自动滚动分析，不必逐次手动触发。")
            if event.get("channel_warning"):
                self._append_log(event["channel_warning"])
            return
        if kind == "result":
            self._show_result(event["record"])
            return
        if kind == "error":
            message = str(event.get("message", "未知错误"))
            self._append_log(message, error=True)
            self.connection_text.set("连接或推理失败")
            self.status_badge.configure(fg=RED)
            self.result_text.set("无法分析")
            self.result_detail.set(message)
            return
        if kind == "stopped":
            self._set_stopped()

    def _show_result(self, record: dict) -> None:
        quality_ok = bool(record.get("quality_ok"))
        self.quality_text.set(
            "正常" if quality_ok else f"异常：{record.get('signal_quality')}"
        )
        if record["target_kind"] == "regression":
            value = float(record["value"])
            self.result_text.set(f"{value:.2f}")
            self.result_detail.set(str(record.get("target_name", "数值")))
            self.confidence_text.set("不适用")
            self.confidence_bar["value"] = 0
            self._clear_probabilities()
            self._append_log(
                f"结果：{record.get('target_name')} = {value:.2f}"
                + ("" if record["accepted"] else "（信号质量异常）")
            )
            return
        confidence = float(record.get("confidence") or 0.0)
        accepted = bool(record.get("accepted"))
        accepted_label = record.get("label", "unknown")
        candidate = record.get("candidate_label", "unknown")
        show_candidate = (
            not accepted
            and self.show_rejected_candidate.get()
            and record.get("quality_ok")
        )
        visible_label = candidate if show_candidate else accepted_label
        self.result_text.set(str(visible_label))
        self.result_label.configure(fg=GREEN if accepted else AMBER)
        threshold = float(record.get("confidence_threshold") or 0.0)
        reasons = set(record.get("rejection_reasons") or [])
        if accepted:
            detail = "结果已通过质量、稳定性和置信度检查"
        elif "below_confidence_threshold" in reasons:
            detail = (
                f"候选结果，置信度未达到 {threshold * 100:.1f}% 接纳门槛；"
                "正式状态仍记为 unknown"
            )
        elif "waiting_for_stability" in reasons:
            detail = "候选结果，正在等待连续窗口结果稳定"
        else:
            detail = f"暂不采用；候选结果：{candidate}"
        self.result_detail.set(detail)
        self.confidence_text.set(f"{confidence * 100:.1f}%")
        self.confidence_bar["value"] = confidence * 100
        self._show_probabilities(
            record.get("classes", []),
            record.get("probabilities", []),
        )
        self._append_log(
            f"结果：{visible_label}"
            + ("" if accepted else "（候选，未接纳）")
            + f"，置信度 {confidence * 100:.1f}%"
        )

    def _clear_probabilities(self) -> None:
        for child in self.probability_frame.winfo_children():
            child.destroy()

    def _show_probabilities(self, classes: list, values: list) -> None:
        self._clear_probabilities()
        if self.runtime is None:
            return
        for row_index, (state, value) in enumerate(zip(classes, values)):
            label = self.runtime.label_mapping.get(int(state), str(state))
            row = tk.Frame(self.probability_frame, bg=CARD_BG)
            row.pack(fill=X, pady=2)
            self._label(row, label, size=8, color=MUTED).pack(side=LEFT)
            self._label(
                row,
                f"{float(value) * 100:.1f}%",
                size=8,
                color=MUTED,
            ).pack(side=RIGHT)

    def _append_log(self, message: str, *, error: bool = False) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert(END, f"[{stamp}] {message}\n", "error" if error else "")
        self.log.tag_configure("error", foreground=RED)
        self.log.see(END)
        self.log.configure(state="disabled")

    def _close(self) -> None:
        if self.engine:
            self.engine.stop(timeout=1.5)
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BSense 实时脑电分类器桌面程序")
    parser.add_argument("--model", default="", help="BSense model.joblib 路径")
    parser.add_argument("--stream-name", default="", help="EEG LSL 流名称；留空按类型查找")
    parser.add_argument(
        "--marker-stream-name",
        default="BSense Experiment Markers",
        help="事件 Marker LSL 流名称",
    )
    parser.add_argument("--output", default="", help="JSONL 结果文件")
    parser.add_argument("--autostart", action="store_true", help="加载后自动开始")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    try:
        ClassifierApp(root, args)
        root.mainloop()
    except ArtifactError as exc:
        messagebox.showerror("模型错误", str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

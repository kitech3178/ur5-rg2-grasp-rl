"""Desktop GUI: choose the algorithm / episodes / reward values, train, watch the
wrist camera live, test the reward on single episodes, and plot / compare results.

    python scripts/gui.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from grasp_rl.analysis import make_report, moving_average
from grasp_rl.config import PROJECT_ROOT, load_config
from grasp_rl.gui import worker

GUI_RESULTS = PROJECT_ROOT / "results" / "gui"
ALGOS = [("Q-learning (표 기반)", "qlearning"), ("DQN (신경망)", "dqn")]

# (config key, label, tooltip) — numbers become QDoubleSpinBox, bools QCheckBox
REWARD_FIELDS = [
    ("reward.success", "성공 보상", "물체를 들어올리는 데 성공했을 때"),
    ("reward.grasp_fail", "잡기 실패", "grasp 행동 후 들어올리지 못했을 때"),
    ("reward.step", "스텝 비용", "매 스텝마다 더해지는 값 (음수 = 빨리 끝내도록 유도)"),
    ("reward.collision", "충돌", "테이블/물체와 충돌해 이동이 막혔을 때"),
    ("reward.ik_fail", "IK 실패", "역기구학 해가 없을 때"),
    ("reward.out_of_workspace", "작업영역 이탈", "TCP가 작업 영역 밖으로 나가려 할 때"),
    ("reward.terminate_on_collision", "충돌 시 종료", "충돌하면 에피소드를 끝냄"),
    ("reward.terminate_on_grasp_fail", "잡기 실패 시 종료", "잡기에 실패하면 에피소드를 끝냄"),
    ("reward.shaping.enabled", "셰이핑 사용", "퍼텐셜 기반 셰이핑 F = γΦ(s') − Φ(s)"),
    ("reward.shaping.scale", "셰이핑 배율", "셰이핑 항 전체에 곱하는 값"),
    ("reward.shaping.w_xy", "  w_xy (위치)", "Φ의 수평 거리 가중치 (per m)"),
    ("reward.shaping.w_z", "  w_z (높이)", "Φ의 높이 오차 가중치 (per m)"),
    ("reward.shaping.w_yaw", "  w_yaw (각도)", "Φ의 정렬 각도 오차 가중치 (per rad)"),
]
Q_FIELDS = [
    ("qlearning.alpha", "학습률 α", "Q 값 갱신 크기"),
    ("qlearning.gamma", "할인율 γ", ""),
    ("qlearning.epsilon_decay_episodes", "ε 감소 에피소드", "탐험률 ε가 최솟값까지 줄어드는 에피소드 수"),
]
DQN_FIELDS = [
    ("dqn.lr", "학습률 lr", "Adam 학습률"),
    ("dqn.gamma", "할인율 γ", ""),
    ("dqn.batch_size", "배치 크기", ""),
    ("dqn.epsilon_decay_episodes", "ε 감소 에피소드", "탐험률 ε가 최솟값까지 줄어드는 에피소드 수"),
]


def _get(cfg, key):
    for p in key.split("."):
        cfg = cfg[p]
    return cfg


def _to_pixmap(img, scale_to=None):
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    q = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pm = QPixmap.fromImage(q)
    if scale_to:
        pm = pm.scaled(scale_to[0], scale_to[1], Qt.KeepAspectRatio, Qt.FastTransformation)
    return pm


def depth_to_rgb(depth):
    d = np.asarray(depth, dtype=float)
    valid = d > 0
    out = np.zeros(d.shape + (3,), np.uint8)
    if valid.any():
        lo, hi = np.percentile(d[valid], [1, 99])
        t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        out[valid] = (colormaps["viridis_r"](t[valid])[:, :3] * 255).astype(np.uint8)
    return out


class ImageView(QLabel):
    def __init__(self, title, size=(320, 240)):
        super().__init__()
        self.title, self.img = title, None
        self.setMinimumSize(*size)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # never grow with the pixmap
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111; color:#aaa; border:1px solid #333")
        self.setText(title)

    def show_image(self, img):
        self.img = img
        if img is None:
            self.setText(f"{self.title}\n(표시할 수 없음 — mujoco 설치/OpenGL 확인)")
        else:
            self.setPixmap(_to_pixmap(img, (self.width() - 2, self.height() - 2)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.img is not None:
            self.setPixmap(_to_pixmap(self.img, (self.width() - 2, self.height() - 2)))


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, rows=2, cols=3):
        self.fig = Figure(figsize=(9, 5))
        super().__init__(self.fig)
        self.axes = self.fig.subplots(rows, cols).ravel()

    def relayout(self):
        """Recompute margins (slow, ~50 ms) — only on setup, resize and new labels."""
        self.fig.tight_layout()
        self.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fig.tight_layout()


# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UR5 + RG2 파지 강화학습")
        self.resize(1500, 950)
        self.defaults = load_config()
        self.fields = {}          # config key -> widget
        self.proc = self.q = self.stop_evt = self.live = None
        self.job_kind = None
        self.ep_rows, self.ev_rows = [], []
        self.last_plot = 0.0
        self.plot_cost = 0.0
        self.plot_dirty = False
        self.prev_results = []
        self.run_label = ""

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_settings())
        split.addWidget(self._build_right())
        split.setSizes([380, 1120])
        self.setCentralWidget(split)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(50)
        self._algo_changed()
        self._setup_plots()
        self._refresh_runs()

    # ------------------------------------------------------------ settings
    def _spin(self, key, value):
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
        elif isinstance(value, int):
            w = QSpinBox()
            w.setRange(1, 10_000_000)
            w.setValue(value)
        else:
            w = QDoubleSpinBox()
            w.setRange(-1000.0, 1000.0)
            w.setDecimals(5 if 0 < abs(value) < 0.001 else 4)
            w.setSingleStep(0.1 if abs(value) >= 0.5 else 0.01)
            w.setValue(float(value))
        self.fields[key] = w
        return w

    def _group(self, title, specs):
        box = QGroupBox(title)
        form = QFormLayout(box)
        for key, label, tip in specs:
            w = self._spin(key, _get(self.defaults, key))
            lab = QLabel(label)
            if tip:
                lab.setToolTip(tip)
                w.setToolTip(tip)
            form.addRow(lab, w)
        return box

    def _build_settings(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)

        exp = QGroupBox("실험 설정")
        f = QFormLayout(exp)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("비우면 자동 (알고리즘_시뮬레이터_시각)")
        self.backend_box = QComboBox()
        self.backend_box.addItems(["simple", "mujoco"])
        self.backend_box.setToolTip("simple: 내장 해석 물리(빠름) / mujoco: MuJoCo 물리 + 컬러 카메라")
        self.algo_box = QComboBox()
        for text, key in ALGOS:
            self.algo_box.addItem(text, key)
        self.algo_box.currentIndexChanged.connect(self._algo_changed)
        self.obs_box = QComboBox()
        self.obs_box.addItem("features (MLP, 빠름)", "features")
        self.obs_box.addItem("depth 이미지 (CNN, 매우 느림)", "depth")
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(10, 1_000_000)
        self.episodes_spin.setSingleStep(100)
        self.episodes_spin.valueChanged.connect(self._episodes_changed)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.eval_spin = QSpinBox()
        self.eval_spin.setRange(5, 100_000)
        self.eval_spin.setToolTip("이 에피소드마다 탐험 없이(greedy) 30회 평가")
        f.addRow("실험 이름", self.name_edit)
        f.addRow("시뮬레이터", self.backend_box)
        f.addRow("알고리즘", self.algo_box)
        self.obs_label = QLabel("DQN 입력")
        f.addRow(self.obs_label, self.obs_box)
        f.addRow("학습 에피소드 수", self.episodes_spin)
        f.addRow("평가 주기 (에피소드)", self.eval_spin)
        f.addRow("시드", self.seed_spin)
        lay.addWidget(exp)

        self.q_group = self._group("Q-learning 파라미터", Q_FIELDS)
        self.dqn_group = self._group("DQN 파라미터", DQN_FIELDS)
        lay.addWidget(self.q_group)
        lay.addWidget(self.dqn_group)

        rw = self._group("보상 (Reward)", REWARD_FIELDS)
        reset_btn = QPushButton("보상 기본값으로")
        reset_btn.clicked.connect(self._reset_reward)
        rw.layout().addRow(reset_btn)
        lay.addWidget(rw)

        live = QGroupBox("학습 중 라이브 화면")
        f = QFormLayout(live)
        self.hz_spin = QDoubleSpinBox()
        self.hz_spin.setRange(0, 30)
        self.hz_spin.setValue(5)
        self.hz_spin.setToolTip("초당 화면 갱신 횟수 (0 = 끄기, 학습이 가장 빠름)")
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 2000)
        self.delay_spin.setSuffix(" ms")
        self.delay_spin.setToolTip("매 스텝 대기 시간 — 로봇 움직임을 눈으로 따라가고 싶을 때 (학습이 느려짐)")
        self.hz_spin.valueChanged.connect(self._live_changed)
        self.delay_spin.valueChanged.connect(self._live_changed)
        f.addRow("화면 갱신 (Hz)", self.hz_spin)
        f.addRow("스텝 지연", self.delay_spin)
        hint = QLabel("학습 중에도 바로 반영됩니다.")
        hint.setStyleSheet("color:#777")
        f.addRow(hint)
        lay.addWidget(live)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶ 학습 시작")
        self.start_btn.setStyleSheet("font-weight:bold; padding:8px")
        self.start_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton("■ 중지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        lay.addLayout(btns)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.status = QLabel("대기 중")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        return scroll

    def _algo_changed(self):
        algo = self.algo_box.currentData()
        self.q_group.setVisible(algo == "qlearning")
        self.dqn_group.setVisible(algo == "dqn")
        self.obs_box.setVisible(algo == "dqn")
        self.obs_label.setVisible(algo == "dqn")
        self.episodes_spin.setValue(int(self.defaults[algo].episodes))
        self._episodes_changed()

    def _episodes_changed(self):
        n = self.episodes_spin.value()
        algo = self.algo_box.currentData()
        ratio = self.defaults[algo].epsilon_decay_episodes / self.defaults[algo].episodes
        self.fields[f"{algo}.epsilon_decay_episodes"].setValue(max(1, int(round(n * ratio))))
        self.eval_spin.setValue(max(10, n // 40))

    def _reset_reward(self):
        for key, _, _ in REWARD_FIELDS:
            w, v = self.fields[key], _get(self.defaults, key)
            w.setChecked(v) if isinstance(w, QCheckBox) else w.setValue(v)

    def _overrides(self, include_algo=True):
        ov = [f"backend={self.backend_box.currentText()}", f"eval.every={self.eval_spin.value()}",
              f"experiment.seed={self.seed_spin.value()}"]
        for key, w in self.fields.items():
            if not include_algo and not key.startswith("reward."):
                continue
            if isinstance(w, QCheckBox):
                ov.append(f"{key}={'true' if w.isChecked() else 'false'}")
            else:
                ov.append(f"{key}={w.value()}")
        if self.algo_box.currentData() == "dqn":
            ov.append(f"dqn.observation={self.obs_box.currentData()}")
        return ov

    # ------------------------------------------------------------ right side
    def _build_right(self):
        right = QSplitter(Qt.Vertical)

        def titled(text):
            lab = QLabel(text)
            lab.setStyleSheet("font-weight:bold")
            return lab

        # main view (whole system) large on the left, the two vision views small on the right
        cam = QWidget()
        h = QHBoxLayout(cam)
        h.setContentsMargins(4, 4, 4, 4)
        main_col = QVBoxLayout()
        self.scene_view = ImageView("전체 시스템 화면", (480, 360))
        main_col.addWidget(titled("전체 시스템 (UR5 · RG2 · 테이블 · 물체)"))
        main_col.addWidget(self.scene_view, 1)
        self.info = QLabel("")
        self.info.setFont(QFont("Monospace", 9))
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        main_col.addWidget(self.info)
        h.addLayout(main_col, 3)

        vision_col = QVBoxLayout()
        self.depth_view = ImageView("손목 depth 카메라", (200, 150))
        self.rgb_view = ImageView("손목 컬러 카메라", (200, 150))
        vision_col.addWidget(titled("비전: Depth (에이전트 입력)"))
        vision_col.addWidget(self.depth_view, 1)
        vision_col.addWidget(titled("비전: 손목 컬러 카메라"))
        vision_col.addWidget(self.rgb_view, 1)
        h.addLayout(vision_col, 1)
        right.addWidget(cam)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plot_tab(), "결과 그래프")
        self.tabs.addTab(self._build_preview_tab(), "보상 테스트 (한 에피소드씩 보기)")
        self.tabs.addTab(self._build_compare_tab(), "결과 비교")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "로그")
        right.addWidget(self.tabs)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        right.setSizes([600, 400])
        return right

    def _build_plot_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.summary_label = QLabel("학습을 시작하면 그래프가 실시간으로 그려집니다.")
        self.summary_label.setWordWrap(True)
        lay.addWidget(self.summary_label)
        self.canvas = PlotCanvas()
        self.canvas.setMinimumHeight(420)          # scroll instead of squashing the plots
        sc = QScrollArea()
        sc.setWidget(self.canvas)
        sc.setWidgetResizable(True)
        lay.addWidget(sc)
        return w

    def _build_preview_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.policy_box = QComboBox()
        self.policy_box.addItem("정답 제어기 (scripted)", "scripted")
        self.policy_box.addItem("무작위 (random)", "random")
        self.policy_box.addItem("학습된 모델", "model")
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("q_table.npz 또는 dqn.pt — 학습이 끝나면 자동으로 채워짐")
        browse = QPushButton("찾기…")
        browse.clicked.connect(self._browse_model)
        self.prev_eps = QSpinBox()
        self.prev_eps.setRange(1, 100)
        self.prev_eps.setValue(3)
        self.prev_delay = QSpinBox()
        self.prev_delay.setRange(0, 3000)
        self.prev_delay.setValue(300)
        self.prev_delay.setSuffix(" ms")
        self.prev_delay.setToolTip("실행 중에도 바로 반영됩니다.")
        self.prev_delay.valueChanged.connect(self._live_changed)
        self.prev_btn = QPushButton("▶ 실행")
        self.prev_btn.clicked.connect(self._start_preview)
        for x in (QLabel("정책"), self.policy_box, self.model_edit, browse, QLabel("에피소드"),
                  self.prev_eps, QLabel("스텝 지연"), self.prev_delay, self.prev_btn):
            row.addWidget(x)
        lay.addLayout(row)
        note = QLabel("왼쪽의 현재 보상 값으로 한 스텝씩 보상이 어떻게 계산되는지 확인합니다 "
                      "(학습은 하지 않음). 보상 값을 바꾸고 다시 실행해 비교해 보세요.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.step_table = QTableWidget(0, 6)
        self.step_table.setHorizontalHeaderLabels(["에피소드", "스텝", "행동", "보상", "보상 구성", "이벤트"])
        self.step_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.step_table.verticalHeader().setVisible(False)
        lay.addWidget(self.step_table)
        self.prev_summary = QLabel("")
        lay.addWidget(self.prev_summary)
        return w

    def _build_compare_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        left = QVBoxLayout()
        self.run_list = QListWidget()
        left.addWidget(QLabel("결과 폴더 (체크해서 비교)"))
        left.addWidget(self.run_list)
        b1 = QPushButton("목록 새로고침")
        b1.clicked.connect(self._refresh_runs)
        b2 = QPushButton("선택한 결과 비교 ▶")
        b2.clicked.connect(self._compare)
        left.addWidget(b1)
        left.addWidget(b2)
        lay.addLayout(left, 1)
        scroll = self.compare_scroll = QScrollArea()
        self.compare_img = QLabel("두 개 이상 체크하고 비교를 누르세요.\n"
                                  "예: 같은 알고리즘을 보상 값만 바꿔 학습한 결과들")
        self.compare_img.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self.compare_img)
        scroll.setWidgetResizable(True)
        right = QVBoxLayout()
        right.addWidget(scroll, 1)
        self.compare_table = QLabel("")
        self.compare_table.setTextFormat(Qt.RichText)
        self.compare_table.setTextInteractionFlags(Qt.TextSelectableByMouse)
        right.addWidget(self.compare_table)
        lay.addLayout(right, 3)
        return w

    # ------------------------------------------------------------ jobs
    def _busy(self, busy):
        self.start_btn.setEnabled(not busy)
        self.prev_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _live_values(self, kind):
        """[frame_hz, step_delay_s] shared with the running job."""
        if kind == "train":
            return [float(self.hz_spin.value()), self.delay_spin.value() / 1000.0]
        return [0.0, self.prev_delay.value() / 1000.0]

    def _live_changed(self):
        if self.live is not None and self.job_kind is not None:
            self.live[:] = self._live_values(self.job_kind)

    def _launch(self, target, job, kind):
        ctx = mp.get_context("spawn")
        self.q, self.stop_evt = ctx.Queue(), ctx.Event()
        self.live = ctx.Array("d", self._live_values(kind))
        self.proc = ctx.Process(target=target, args=(job, self.q, self.stop_evt, self.live), daemon=True)
        self.proc.start()
        self.job_kind = kind
        self._busy(True)

    def _start_training(self):
        algo = self.algo_box.currentData()
        backend = self.backend_box.currentText()
        label = self.name_edit.text().strip() or f"{algo}_{backend}_{time.strftime('%m%d-%H%M%S')}"
        out = GUI_RESULTS / label
        if out.exists() and any(out.iterdir()):
            if QMessageBox.question(self, "덮어쓰기", f"{out} 가 이미 있습니다. 덮어쓸까요?") != QMessageBox.Yes:
                return
        n = self.episodes_spin.value()
        job = {"overrides": self._overrides(), "algo": algo, "episodes": n, "seed": self.seed_spin.value(),
               "out_dir": str(out), "label": label}
        self.ep_rows, self.ev_rows = [], []
        self.n_episodes, self.run_label, self.t_start = n, label, time.time()
        self._setup_plots()
        self.progress.setRange(0, n)
        self.progress.setValue(0)
        self.summary_label.setText(f"학습 중: {label}")
        self._log(f"학습 시작: {label}  ({algo}, {backend}, {n} 에피소드)\n  설정: {' '.join(job['overrides'])}")
        self.tabs.setCurrentIndex(0)
        self._launch(worker.train_worker, job, "train")
        self.status.setText("학습 프로세스 시작 중…")

    def _start_preview(self):
        policy = self.policy_box.currentData()
        ov = self._overrides(include_algo=False)
        model = None
        if policy == "model":
            model = self.model_edit.text().strip()
            if not model or not Path(model).exists():
                QMessageBox.warning(self, "모델 없음", "학습된 모델 파일(q_table.npz / dqn.pt)을 선택하세요.")
                return
            policy = "qlearning" if model.endswith(".npz") else "dqn"
            run_cfg = Path(model).parent / "config.yaml"
            if policy == "dqn" and run_cfg.exists():        # network shape must match the saved model
                d = yaml.safe_load(run_cfg.read_text(encoding="utf-8"))["dqn"]
                ov += [f"dqn.observation={d['observation']}", f"dqn.hidden={d['hidden']}",
                       f"dqn.conv_channels={d['conv_channels']}"]
        job = {"overrides": ov, "policy": policy, "model": model, "episodes": self.prev_eps.value(),
               "seed": 100_000 + self.seed_spin.value()}
        self.step_table.setRowCount(0)
        self.prev_results = []
        self.prev_summary.setText("실행 중…")
        self._log(f"보상 테스트: {policy} {model or ''}")
        self._launch(worker.preview_worker, job, "preview")
        self.status.setText("보상 테스트 실행 중…")

    def _stop(self):
        if self.stop_evt is not None:
            self.stop_evt.set()
            self.status.setText("중지 요청 — 현재 에피소드를 마치고 저장합니다…")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "모델 선택", str(PROJECT_ROOT / "results"),
                                              "모델 (*.npz *.pt)")
        if path:
            self.model_edit.setText(path)
            self.policy_box.setCurrentIndex(2)

    # ------------------------------------------------------------ messages
    def _poll(self):
        if self.q is None:
            return
        frame = None
        for _ in range(500):
            try:
                kind, data = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "frame":
                frame = data
            elif kind == "episode":
                self.ep_rows.append(data)
                self.plot_dirty = True
            elif kind == "eval":
                self.ev_rows.append(data)
                self.plot_dirty = True
                self._log(f"  평가 @ep {data['episode']}: 성공률 {data['success_rate']:.2f}")
            elif kind == "step":
                self._add_step_row(data)
            elif kind == "preview_episode":
                self.prev_results.append(data)
                self._log(f"  에피소드 {data['episode']}: {data['outcome']}  리턴 {data['return']:.2f}  "
                          f"길이 {data['length']}")
            elif kind == "done":
                if frame is not None:
                    self._show_frame(frame)
                self._finish(data)
                return
            elif kind == "error":
                self._log(data)
                QMessageBox.critical(self, "오류", data[-2000:])
                self._finish(None, error=True)
                return
        if frame is not None:
            self._show_frame(frame)
        if self.job_kind == "train" and self.ep_rows:
            self._update_progress()
        # redraw at most ~1/s, and never spend more than ~10% of the time drawing
        if self.plot_dirty and time.time() - self.last_plot > max(1.0, 10 * self.plot_cost):
            self._draw_plots()
        if self.proc is not None and not self.proc.is_alive() and self.q is not None:
            try:
                kind, data = self.q.get(timeout=0.2)
            except queue.Empty:
                self._log("작업 프로세스가 예기치 않게 종료되었습니다.")
                self._finish(None, error=True)
            else:
                self.q.put((kind, data))      # handled on the next poll

    def _update_progress(self):
        r = self.ep_rows[-1]
        last = self.ep_rows[-100:]
        sr = np.mean([x["success"] for x in last])
        el = time.time() - self.t_start
        eta = el / r["episode"] * (self.n_episodes - r["episode"])
        self.progress.setValue(int(r["episode"]))
        ev = f"  |  최근 평가 성공률 {self.ev_rows[-1]['success_rate']:.2f}" if self.ev_rows else ""
        self.status.setText(f"에피소드 {r['episode']}/{self.n_episodes}  ε={r['epsilon']:.3f}\n"
                            f"최근 100 에피소드 성공률 {sr:.2f}{ev}\n경과 {el:.0f}s, 남은 시간 약 {eta:.0f}s")

    def _show_frame(self, f):
        self.depth_view.show_image(depth_to_rgb(f["depth"]))
        self.rgb_view.show_image(f["cam_rgb"])
        self.scene_view.show_image(f["scene_rgb"])

        def feat(x):
            if x is None:
                return "물체 안 보임"
            return (f"dx {x['dx'] * 100:+6.1f} cm  dy {x['dy'] * 100:+6.1f} cm  "
                    f"dz {x['dz'] * 100:+6.1f} cm  yaw {np.rad2deg(x['yaw']):+6.1f}°")
        terms = "  ".join(f"{k} {v:+.2f}" for k, v in f["terms"].items()) or "-"
        d = f["depth"]
        dv = d[d > 0]
        self.info.setText(
            f"에피소드 {f['episode']}  스텝 {f['step']:2d}  행동 {f['action']:<6}  "
            f"보상 {f['reward']:+7.2f}  누적 {f['ret']:+7.2f}   이벤트 {', '.join(f['events']) or '-'}\n"
            f"보상 구성: {terms}\n"
            f"카메라 인식 : {feat(f['perceived'])}\n"
            f"참값        : {feat(f['gt'])}\n"
            f"depth 범위 {dv.min() * 100 if dv.size else 0:.1f}–{dv.max() * 100 if dv.size else 0:.1f} cm, "
            f"유효 픽셀 {dv.size}/{d.size}   (밝을수록 가까움)")

    def _add_step_row(self, s):
        t = self.step_table
        i = t.rowCount()
        t.insertRow(i)
        vals = [s["episode"], s["step"], s["action"], f"{s['reward']:+.3f}",
                "  ".join(f"{k}={v:+.3f}" for k, v in s["terms"].items()), ", ".join(s["events"])]
        for j, v in enumerate(vals):
            item = QTableWidgetItem(str(v))
            if j == 3:
                item.setForeground(QColor("#1d8a3a" if s["reward"] > 0 else "#d03030" if s["reward"] < -0.5
                                          else "#202020"))
            t.setItem(i, j, item)
        t.scrollToBottom()

    def _finish(self, data, error=False):
        kind = self.job_kind
        if self.proc is not None:
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.terminate()
        self.proc = self.q = self.stop_evt = self.live = None
        self.job_kind = None
        self._busy(False)
        if error:
            self.status.setText("오류로 중단됨 (로그 탭 참고)")
            return
        if kind == "train":
            s, out = data["summary"], Path(data["out_dir"])
            fe = s["final_eval"]
            text = (f"<b>{s['label']}</b> 완료 — {s['episodes']} 에피소드, {s['wall_time_s']} s<br>"
                    f"최종 평가 성공률 <b>{fe['success_rate']:.2f}</b>, 평균 길이 {fe['mean_length']:.1f}, "
                    f"과제 리턴 {fe['mean_task_return']:.2f}, 파라미터 수 {s['parameters']:,}<br>"
                    f"저장 위치: {out}")
            self.summary_label.setText(text)
            self.status.setText("학습 완료")
            self._log(f"완료: {out}")
            self._draw_plots(final=True)
            model = out / ("q_table.npz" if s["algo"] == "qlearning" else "dqn.pt")
            if model.exists():
                self.model_edit.setText(str(model))
                self.policy_box.setCurrentIndex(2)
            self._refresh_runs()
        else:
            n = len(self.prev_results)
            ok = sum(r["outcome"] == "success" for r in self.prev_results)
            avg = np.mean([r["return"] for r in self.prev_results]) if n else 0.0
            self.prev_summary.setText(f"{n} 에피소드: 성공 {ok}회, 평균 리턴 {avg:.2f}  —  "
                                      + ", ".join(r["outcome"] for r in self.prev_results))
            self.status.setText("보상 테스트 완료")

    # ------------------------------------------------------------ plots
    SERIES = [(0, "success", "Training success rate", "#2a6fdb"),
              (2, "task_return", "Task return (without shaping)", "#3a9d5d"),
              (3, "length", "Episode length (steps)", "#8a5cc2"),
              (4, "epsilon", "Exploration ε", "#777777")]

    def _setup_plots(self):
        """Build the six panels once per run; later updates only swap the data,
        which keeps the window responsive during long trainings."""
        ax = self.canvas.axes
        self.lines = {}
        for a in ax:
            a.clear()
            a.grid(alpha=0.3)
            a.set_xlabel("episode", fontsize=8)
            a.tick_params(labelsize=8)
        for i, key, title, color in self.SERIES:
            raw, = ax[i].plot([], [], color=color, alpha=0.15, lw=0.8)
            ma, = ax[i].plot([], [], color=color, lw=2)
            ax[i].set_title(title, fontsize=10)
            self.lines[key] = (i, raw, ma)
        self.eval_line, = ax[1].plot([], [], "o-", color="#e4572e", lw=2, ms=4)
        ax[0].set_ylim(-0.03, 1.03)
        ax[1].set_ylim(-0.03, 1.03)
        ax[1].set_title("Greedy evaluation success rate", fontsize=10)
        ax[5].set_title("Outcomes (last 20% of episodes)", fontsize=10)
        self.outcome_names = None
        self.canvas.fig.suptitle(self.run_label, fontsize=11)
        self.canvas.relayout()

    def _draw_plots(self, final=False):
        t0 = time.perf_counter()
        self.last_plot, self.plot_dirty = time.time(), False
        if not self.ep_rows:
            return
        ax = self.canvas.axes
        ep = np.array([r["episode"] for r in self.ep_rows], dtype=float)
        w = max(1, min(100, len(ep) // 10 or 1))
        stride = max(1, len(ep) // 3000)              # faint raw curve: thinned
        for key, (i, raw, ma) in self.lines.items():
            y = np.array([r[key] for r in self.ep_rows], dtype=float)
            raw.set_data(ep[::stride], y[::stride])
            ma.set_data(ep, moving_average(y, w))
            ax[i].relim()
            ax[i].autoscale_view(scaley=key != "success")
        ax[0].set_title(f"Training success rate (moving avg {w})", fontsize=10)
        if self.ev_rows:
            self.eval_line.set_data([r["episode"] for r in self.ev_rows],
                                    [r["success_rate"] for r in self.ev_rows])
            ax[1].relim()
            ax[1].autoscale_view(scaley=False)
        tail = self.ep_rows[-max(1, len(self.ep_rows) // 5):]
        names, counts = np.unique([r["outcome"] for r in tail], return_counts=True)
        a5 = ax[5]
        a5.clear()
        a5.grid(alpha=0.3)
        a5.bar(names, counts / counts.sum(), color="#2a6fdb")
        a5.set_title("Outcomes (last 20% of episodes)", fontsize=10)
        a5.tick_params(axis="x", labelrotation=30, labelsize=7)
        a5.tick_params(axis="y", labelsize=8)
        self.canvas.fig.suptitle(self.run_label + ("" if final else "  (training…)"), fontsize=11)
        if final or list(names) != self.outcome_names:  # new bar labels may need more margin
            self.outcome_names = list(names)
            self.canvas.relayout()
        else:
            self.canvas.draw()
        self.plot_cost = time.perf_counter() - t0

    # ------------------------------------------------------------ compare
    def _refresh_runs(self):
        checked = {self.run_list.item(i).data(Qt.UserRole) for i in range(self.run_list.count())
                   if self.run_list.item(i).checkState() == Qt.Checked}
        self.run_list.clear()
        root = PROJECT_ROOT / "results"
        runs = sorted(root.rglob("summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for s in runs:
            d = s.parent
            try:
                info = json.loads(s.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            fe = info.get("final_eval", {}).get("success_rate", float("nan"))
            item = QListWidgetItem(f"{d.relative_to(root)}   [{info['algo']}, {info['episodes']} ep, "
                                   f"성공률 {fe:.2f}]")
            item.setData(Qt.UserRole, str(d))
            item.setToolTip(item.text())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if str(d) in checked else Qt.Unchecked)
            self.run_list.addItem(item)

    def _compare(self):
        dirs = [Path(self.run_list.item(i).data(Qt.UserRole)) for i in range(self.run_list.count())
                if self.run_list.item(i).checkState() == Qt.Checked]
        if not dirs:
            QMessageBox.information(self, "비교", "비교할 결과를 하나 이상 체크하세요.")
            return
        tmp = Path(tempfile.mkdtemp(prefix="grasp_compare_"))
        root = PROJECT_ROOT / "results"
        for i, d in enumerate(dirs):
            dst = tmp / f"run{i}"
            dst.mkdir()
            for name in ("episodes.csv", "eval.csv"):
                shutil.copy(d / name, dst / name)
            s = json.loads((d / "summary.json").read_text())
            s["label"] = str(d.relative_to(root)) if d.is_relative_to(root) else d.name
            (dst / "summary.json").write_text(json.dumps(s))
        try:
            png, md = make_report(tmp, title="Comparison")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "비교 실패", str(e))
            return
        width = max(600, self.compare_scroll.viewport().width() - 10)
        self.compare_img.setPixmap(QPixmap(str(png)).scaledToWidth(width, Qt.SmoothTransformation))
        rows = [ln for ln in Path(md).read_text().splitlines() if ln.startswith("|") and "---" not in ln]
        html = "<table border=1 cellspacing=0 cellpadding=3>" + "".join(
            "<tr>" + "".join(f"<td>{c.strip()}</td>" for c in r.strip("|").split("|")) + "</tr>" for r in rows) + "</table>"
        self.compare_table.setText(html)
        self._log(f"비교 리포트: {png}")

    # ------------------------------------------------------------ misc
    def _log(self, text):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def closeEvent(self, event):
        if self.proc is not None:
            self.stop_evt.set()
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    geo = app.primaryScreen().availableGeometry()
    win.resize(max(1300, int(geo.width() * 0.95)), max(900, int(geo.height() * 0.95)))
    win.show()
    return app.exec()

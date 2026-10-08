import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import (QAction, QColor, QIcon, QImage, QKeySequence, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter,
                               QVBoxLayout, QWidget)

APP_NAME = "PneumoScan"
APP_DIR = Path(__file__).resolve().parent
ICON_PATH = APP_DIR / "assets" / "pneumoscan.ico"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DISPLAY_MAX = 1400
THUMB_SIZE = 48

STYLE = """
QMainWindow { background: #0f1115; }
QWidget { color: #e6e8ec; font-family: "Segoe UI"; font-size: 10pt; }
QLabel { background: transparent; }
QToolTip { background: #1d212b; color: #e6e8ec; border: 1px solid #2c3240; padding: 4px; }

#header { background: #151821; border-bottom: 1px solid #252a35; }
#appTitle { font-size: 15pt; font-weight: 600; }
#muted, #appSubtitle { color: #8b93a1; }
#sectionTitle { color: #8b93a1; font-size: 8.5pt; font-weight: 600; letter-spacing: 1px; }

#statusChip { background: #1d212b; border: 1px solid #2c3240; border-radius: 11px; padding: 3px 10px; color: #8b93a1; }
#statusChip[state="ready"] { color: #4fd18b; border-color: rgba(47,191,113,0.45); }
#statusChip[state="error"] { color: #ff6b68; border-color: rgba(239,83,80,0.55); }

#panel { background: #151821; border: 1px solid #252a35; border-radius: 10px; }
#viewer { background: #0b0d11; border: 1px dashed #2c3240; border-radius: 10px; color: #5f6777; font-size: 11pt; }

QPushButton { background: #1d212b; border: 1px solid #2c3240; border-radius: 6px; padding: 7px 14px; }
QPushButton:hover { background: #252a36; border-color: #394052; }
QPushButton:pressed { background: #191c25; }
QPushButton:disabled { color: #59606e; border-color: #23272f; }
QPushButton#primary { background: #3d7bfd; border-color: #3d7bfd; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #5a8fff; border-color: #5a8fff; }
QPushButton#segLeft { border-top-right-radius: 0; border-bottom-right-radius: 0; padding: 6px 18px; }
QPushButton#segRight { border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; padding: 6px 18px; }
QPushButton#segLeft:checked, QPushButton#segRight:checked { background: rgba(61,123,253,0.22); border-color: #3d7bfd; color: #ffffff; }
QPushButton#link { background: transparent; border: none; color: #6f9dff; padding: 0; }
QPushButton#link:hover { color: #9dbcff; }

#verdict { background: #1a1d25; border: 1px solid #2c3240; border-radius: 10px; }
#verdict[state="normal"] { background: rgba(47,191,113,0.10); border-color: rgba(47,191,113,0.55); }
#verdict[state="pneumonia"] { background: rgba(239,83,80,0.10); border-color: rgba(239,83,80,0.60); }
#verdictTitle { font-size: 17pt; font-weight: 600; }
#verdictTitle[state="normal"] { color: #4fd18b; }
#verdictTitle[state="pneumonia"] { color: #ff6b68; }
#bigNumber { font-size: 24pt; font-weight: 600; }

QProgressBar { background: #1d212b; border: none; border-radius: 4px; max-height: 8px; }
QProgressBar::chunk { background: #3d7bfd; border-radius: 4px; }
QProgressBar[state="normal"]::chunk { background: #2fbf71; }
QProgressBar[state="pneumonia"]::chunk { background: #ef5350; }
QProgressBar#busy { max-height: 4px; }

QSlider::groove:horizontal { height: 4px; background: #2c3240; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3d7bfd; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }

QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { border-radius: 8px; margin: 1px 0; }
QListWidget::item:hover { background: #1b1f28; }
QListWidget::item:selected { background: rgba(61,123,253,0.18); }

#badge { border-radius: 4px; padding: 1px 7px; font-size: 8.5pt; font-weight: 600; }
#badge[state="normal"] { color: #4fd18b; background: rgba(47,191,113,0.14); }
#badge[state="pneumonia"] { color: #ff6b68; background: rgba(239,83,80,0.14); }

QSplitter::handle { background: transparent; width: 10px; }
QStatusBar { background: #151821; border-top: 1px solid #252a35; color: #8b93a1; }
QStatusBar::item { border: none; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: #2c3240; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


def repolish(widget, state):
    widget.setProperty("state", state)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def pil_to_pixmap(img):
    img = img.convert("RGB")
    qimg = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode.startswith("I") or img.mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr - arr.min()) / max(float(np.ptp(arr)), 1e-6) * 255
        img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGB")


def collect_images(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return [str(f) for f in files]


def make_icon_pixmap(size=256):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 256
    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0, QColor("#5a8fff"))
    grad.setColorAt(1, QColor("#2f5fd0"))
    p.setBrush(grad)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(8 * s, 8 * s, 240 * s, 240 * s), 56 * s, 56 * s)

    p.setBrush(QColor(255, 255, 255, 235))
    for side in (-1, 1):
        path = QPainterPath()
        cx = 128 + side * 14
        path.moveTo(cx * s, 92 * s)
        path.cubicTo((cx + side * 70) * s, 70 * s, (cx + side * 82) * s, 200 * s, (cx + side * 52) * s, 204 * s)
        path.cubicTo((cx + side * 22) * s, 208 * s, cx * s, 196 * s, cx * s, 176 * s)
        path.closeSubpath()
        p.drawPath(path)
    p.setPen(QPen(QColor(255, 255, 255, 235), 14 * s, Qt.SolidLine, Qt.RoundCap))
    p.drawLine(QPointF(128 * s, 46 * s), QPointF(128 * s, 120 * s))
    p.end()
    return pm


@dataclass
class ScanResult:
    path: str
    prob: float
    image: QPixmap
    heatmap: QPixmap
    resolution: tuple


class ModelLoader(QThread):
    loaded = Signal(object, str)
    failed = Signal(str)

    def run(self):
        try:
            import torch
            from inference import Predictor
            predictor = Predictor()
            device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            self.loaded.emit(predictor, device)
        except Exception as e:
            self.failed.emit(str(e))


class AnalyzeWorker(QThread):
    result = Signal(str, float, object, object, tuple)
    error = Signal(str, str)
    progress = Signal(int)

    def __init__(self, predictor, paths):
        super().__init__()
        self.predictor, self.paths = predictor, paths

    def run(self):
        from inference import overlay_heatmap
        for i, path in enumerate(self.paths):
            try:
                image = load_image(path)
                prob, cam = self.predictor.gradcam(image)
                preview = image.copy()
                preview.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
                self.result.emit(path, prob, preview, overlay_heatmap(preview, cam), image.size)
            except Exception as e:
                self.error.emit(path, str(e))
            self.progress.emit(i + 1)


class ImageView(QLabel):
    PLACEHOLDER = "Drop chest X-ray images or folders here\n\nor press Ctrl+O to open files"

    def __init__(self):
        super().__init__(self.PLACEHOLDER)
        self.setObjectName("viewer")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(360, 360)
        self._pixmap = None

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self):
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            self.setText(self.PLACEHOLDER)
            return
        target = self.size() - QSize(24, 24)
        self.setPixmap(self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ScanRow(QWidget):
    def __init__(self, result):
        super().__init__()
        self.prob = result.prob
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(10)

        thumb = QLabel()
        thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb.setPixmap(result.image.scaled(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatioByExpanding,
                                            Qt.SmoothTransformation).copy(0, 0, THUMB_SIZE, THUMB_SIZE))
        lay.addWidget(thumb)

        text = QVBoxLayout()
        text.setSpacing(3)
        name = QLabel(Path(result.path).name)
        name.setToolTip(result.path)
        name.setMinimumWidth(1)
        text.addWidget(name)
        meta = QHBoxLayout()
        meta.setSpacing(8)
        self.badge = QLabel(objectName="badge")
        self.score = QLabel(f"{result.prob * 100:.1f}%", objectName="muted")
        meta.addWidget(self.badge)
        meta.addWidget(self.score)
        meta.addStretch()
        text.addLayout(meta)
        lay.addLayout(text, 1)

    def apply_threshold(self, threshold):
        flagged = self.prob > threshold
        self.badge.setText("PNEUMONIA" if flagged else "NORMAL")
        repolish(self.badge, "pneumonia" if flagged else "normal")


def section(title):
    label = QLabel(title.upper())
    label.setObjectName("sectionTitle")
    return label


def panel():
    frame = QFrame()
    frame.setObjectName("panel")
    return frame


class MainWindow(QMainWindow):
    LEGEND = "Heatmap (Grad-CAM): red areas had the strongest influence on the prediction"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Chest X-ray Analysis")
        self.setWindowIcon(QIcon(make_icon_pixmap()))
        self.resize(1360, 820)
        self.setMinimumSize(1080, 640)
        self.setAcceptDrops(True)

        self.predictor = None
        self.default_threshold = 0.5
        self.threshold = 0.5
        self.results = {}
        self.rows = {}
        self.pending = []
        self.worker = None
        self.current = None

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_history())
        splitter.addWidget(self._build_viewer())
        splitter.addWidget(self._build_details())
        splitter.setSizes([300, 720, 340])
        splitter.setStretchFactor(1, 1)

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(14, 14, 14, 14)
        body_lay.addWidget(splitter)
        outer.addWidget(body, 1)
        self.setCentralWidget(root)

        self.busy = QProgressBar(objectName="busy")
        self.busy.setFixedWidth(160)
        self.busy.setTextVisible(False)
        self.busy.hide()
        self.statusBar().addPermanentWidget(self.busy)
        self.statusBar().showMessage("Loading model…")

        self._build_actions()
        self._update_details()
        self._set_controls_enabled()

        self.loader = ModelLoader()
        self.loader.loaded.connect(self._on_model_loaded)
        self.loader.failed.connect(self._on_model_failed)
        self.loader.start()


    def _build_header(self):
        header = QFrame(objectName="header")
        lay = QHBoxLayout(header)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(12)

        logo = QLabel()
        logo.setPixmap(make_icon_pixmap(64).scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(logo)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(QLabel(APP_NAME, objectName="appTitle"))
        titles.addWidget(QLabel("AI-assisted pneumonia screening for chest radiographs", objectName="appSubtitle"))
        lay.addLayout(titles)
        lay.addStretch()

        self.open_btn = QPushButton("Open Images", objectName="primary")
        self.open_btn.clicked.connect(self.open_files)
        self.folder_btn = QPushButton("Open Folder")
        self.folder_btn.clicked.connect(self.open_folder)
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_csv)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all)
        for b in (self.open_btn, self.folder_btn, self.export_btn, self.clear_btn):
            b.setCursor(Qt.PointingHandCursor)
            lay.addWidget(b)

        lay.addSpacing(8)
        self.status_chip = QLabel("●  Loading model", objectName="statusChip")
        lay.addWidget(self.status_chip)
        return header

    def _build_history(self):
        box = panel()
        box.setMinimumWidth(250)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 14, 12, 12)
        lay.addWidget(section("Studies"))
        self.summary = QLabel("No images analysed yet", objectName="muted")
        lay.addWidget(self.summary)
        lay.addSpacing(4)
        self.list = QListWidget()
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.currentItemChanged.connect(self._on_select)
        lay.addWidget(self.list, 1)
        return box

    def _build_viewer(self):
        box = panel()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        top = QHBoxLayout()
        self.file_label = QLabel("No image selected", objectName="muted")
        self.file_label.setMinimumWidth(1)
        top.addWidget(self.file_label, 1)
        self.orig_btn = QPushButton("Original", objectName="segLeft", checkable=True)
        self.heat_btn = QPushButton("Heatmap", objectName="segRight", checkable=True)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        for b in (self.orig_btn, self.heat_btn):
            b.setCursor(Qt.PointingHandCursor)
            self.view_group.addButton(b)
            top.addWidget(b)
        top.setSpacing(0)
        self.heat_btn.setChecked(True)
        self.view_group.buttonClicked.connect(lambda _: self._show_current_image())
        lay.addLayout(top)

        self.viewer = ImageView()
        lay.addWidget(self.viewer, 1)

        legend = QHBoxLayout()
        self.legend = QLabel(self.LEGEND, objectName="muted")
        self.legend.setMinimumWidth(1)
        legend.addWidget(self.legend, 1)
        self.save_heat_btn = QPushButton("Save Image…")
        self.save_heat_btn.clicked.connect(self.save_image)
        legend.addWidget(self.save_heat_btn)
        lay.addLayout(legend)
        return box

    def _build_details(self):
        box = panel()
        box.setMinimumWidth(310)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        lay.addWidget(section("Result"))
        self.verdict = QFrame(objectName="verdict")
        v = QVBoxLayout(self.verdict)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(2)
        self.verdict_title = QLabel("—", objectName="verdictTitle")
        self.verdict_sub = QLabel("Open an image to start", objectName="muted")
        self.verdict_sub.setWordWrap(True)
        v.addWidget(self.verdict_title)
        v.addWidget(self.verdict_sub)
        lay.addWidget(self.verdict)

        lay.addSpacing(6)
        lay.addWidget(section("Pneumonia probability"))
        self.prob_label = QLabel("—", objectName="bigNumber")
        lay.addWidget(self.prob_label)
        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0, 1000)
        self.prob_bar.setTextVisible(False)
        lay.addWidget(self.prob_bar)

        lay.addSpacing(10)
        th_head = QHBoxLayout()
        th_head.addWidget(section("Decision threshold"))
        th_head.addStretch()
        self.th_value = QLabel("50%")
        th_head.addWidget(self.th_value)
        lay.addLayout(th_head)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.valueChanged.connect(self._on_threshold)
        lay.addWidget(self.slider)
        th_foot = QHBoxLayout()
        th_foot.addWidget(QLabel("Higher sensitivity", objectName="muted"))
        th_foot.addStretch()
        th_foot.addWidget(QLabel("Fewer false positives", objectName="muted"))
        lay.addLayout(th_foot)
        self.reset_btn = QPushButton("Reset to calibrated value", objectName="link")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(lambda: self.slider.setValue(round(self.default_threshold * 100)))
        lay.addWidget(self.reset_btn, alignment=Qt.AlignLeft)

        lay.addSpacing(10)
        lay.addWidget(section("Image"))
        self.info_image = QLabel("—", objectName="muted")
        self.info_image.setWordWrap(True)
        lay.addWidget(self.info_image)

        lay.addSpacing(6)
        lay.addWidget(section("Model"))
        self.info_model = QLabel("Loading…", objectName="muted")
        self.info_model.setWordWrap(True)
        lay.addWidget(self.info_model)

        lay.addStretch()
        note = QLabel("For research use only. Not intended for clinical diagnosis.", objectName="muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        return box

    def _build_actions(self):
        for text, key, slot in (("Open Images", QKeySequence.Open, self.open_files),
                                ("Open Folder", QKeySequence("Ctrl+Shift+O"), self.open_folder),
                                ("Export Report", QKeySequence("Ctrl+E"), self.export_csv),
                                ("Toggle Heatmap", QKeySequence("H"), self._toggle_view)):
            action = QAction(text, self)
            action.setShortcut(key)
            action.triggered.connect(slot)
            self.addAction(action)


    def _on_model_loaded(self, predictor, device):
        self.predictor = predictor
        self.default_threshold = predictor.threshold
        self.slider.setValue(round(predictor.threshold * 100))
        self.status_chip.setText(f"●  Model ready · {'GPU' if device != 'CPU' else 'CPU'}")
        repolish(self.status_chip, "ready")
        self.info_model.setText(
            f"Custom CNN · 4 conv blocks · 224 × 224 input\n"
            f"Calibrated threshold: {predictor.threshold * 100:.1f}%\n"
            f"Device: {device}")
        self.statusBar().showMessage("Ready", 3000)
        self._start_next()

    def _on_model_failed(self, message):
        self.status_chip.setText("●  Model unavailable")
        repolish(self.status_chip, "error")
        self.info_model.setText("Failed to load the model.")
        self.statusBar().showMessage("Model failed to load")
        QMessageBox.critical(self, APP_NAME, f"Could not load the model:\n\n{message}")


    def open_files(self):
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(self, "Open Chest X-ray Images", "",
                                                f"Images ({exts});;All files (*)")
        self.enqueue(paths)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder with X-ray Images")
        if folder:
            self.enqueue([folder])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.enqueue([u.toLocalFile() for u in event.mimeData().urls()])

    def enqueue(self, paths):
        files = [p for p in collect_images(paths) if p not in self.results and p not in self.pending]
        if not files:
            if paths:
                self.statusBar().showMessage("No new supported images found", 4000)
            return
        self.pending.extend(files)
        if self.predictor is None:
            self.statusBar().showMessage(f"{len(self.pending)} image(s) queued — waiting for the model…")
        self._start_next()

    def _start_next(self):
        if self.worker or not self.pending or self.predictor is None:
            return
        batch, self.pending = self.pending, []
        self.worker = AnalyzeWorker(self.predictor, batch)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(lambda n, total=len(batch): self._on_progress(n, total))
        self.worker.finished.connect(self._on_worker_done)
        self.busy.setRange(0, len(batch))
        self.busy.setValue(0)
        self.busy.show()
        self._set_controls_enabled()
        self.worker.start()

    def _on_progress(self, done, total):
        self.busy.setValue(done)
        self.statusBar().showMessage(f"Analysing {done} / {total}…")

    def _on_worker_done(self):
        self.worker.deleteLater()
        self.worker = None
        self.busy.hide()
        self.statusBar().showMessage(f"Analysis complete · {len(self.results)} image(s)", 5000)
        self._set_controls_enabled()
        self._start_next()

    def _on_result(self, path, prob, preview, heatmap, resolution):
        result = ScanResult(path, prob, pil_to_pixmap(preview), pil_to_pixmap(heatmap), resolution)
        self.results[path] = result

        row = ScanRow(result)
        row.apply_threshold(self.threshold)
        item = QListWidgetItem()
        item.setData(Qt.UserRole, path)
        item.setSizeHint(row.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, row)
        self.rows[path] = row

        if self.list.currentItem() is None:
            self.list.setCurrentItem(item)
        self._update_summary()
        self._set_controls_enabled()

    def _on_error(self, path, message):
        self.statusBar().showMessage(f"Could not read {Path(path).name}: {message}", 6000)


    def _on_select(self, item, _previous=None):
        self.current = item.data(Qt.UserRole) if item else None
        self._show_current_image()
        self._update_details()
        self._set_controls_enabled()

    def _toggle_view(self):
        (self.orig_btn if self.heat_btn.isChecked() else self.heat_btn).setChecked(True)
        self._show_current_image()

    def _show_current_image(self):
        result = self.results.get(self.current)
        if result is None:
            self.viewer.set_image(None)
            self.file_label.setText("No image selected")
            return
        self.viewer.set_image(result.heatmap if self.heat_btn.isChecked() else result.image)
        path = Path(result.path)
        self.file_label.setText(f"{path.parent.name}  /  {path.name}")
        self.file_label.setToolTip(result.path)
        self.legend.setText(self.LEGEND if self.heat_btn.isChecked() else "")

    def _on_threshold(self, value):
        self.threshold = value / 100
        self.th_value.setText(f"{value}%")
        for row in self.rows.values():
            row.apply_threshold(self.threshold)
        self._update_details()
        self._update_summary()

    def _update_details(self):
        result = self.results.get(self.current)
        if result is None:
            self.verdict_title.setText("—")
            self.verdict_sub.setText("Open an image to start" if not self.results else "Select a study")
            self.prob_label.setText("—")
            self.prob_bar.setValue(0)
            self.info_image.setText("—")
            for w in (self.verdict, self.verdict_title, self.prob_bar):
                repolish(w, "")
            return

        flagged = result.prob > self.threshold
        state = "pneumonia" if flagged else "normal"
        self.verdict_title.setText("Pneumonia" if flagged else "Normal")
        self.verdict_sub.setText("Findings consistent with pneumonia" if flagged
                                 else "No radiographic signs of pneumonia detected")
        self.prob_label.setText(f"{result.prob * 100:.1f}%")
        self.prob_bar.setValue(round(result.prob * 1000))
        for w in (self.verdict, self.verdict_title, self.prob_bar):
            repolish(w, state)
        w, h = result.resolution
        self.info_image.setText(f"{Path(result.path).name}\n{w} × {h} px")

    def _update_summary(self):
        total = len(self.results)
        if not total:
            self.summary.setText("No images analysed yet")
            return
        flagged = sum(r.prob > self.threshold for r in self.results.values())
        self.summary.setText(f"{total} studies · {flagged} flagged · {total - flagged} normal")

    def _set_controls_enabled(self):
        busy = self.worker is not None
        has_results = bool(self.results)
        self.export_btn.setEnabled(has_results)
        self.clear_btn.setEnabled(has_results and not busy)
        self.save_heat_btn.setEnabled(self.current in self.results)


    def save_image(self):
        result = self.results.get(self.current)
        if result is None:
            return
        heat = self.heat_btn.isChecked()
        suffix = "_heatmap" if heat else ""
        default = str(Path(result.path).with_name(Path(result.path).stem + suffix + ".png"))
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", default, "PNG image (*.png)")
        if path:
            (result.heatmap if heat else result.image).save(path, "PNG")
            self.statusBar().showMessage(f"Saved {path}", 4000)

    def export_csv(self):
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Report", "pneumoscan_report.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "pneumonia_probability", "result", "threshold"])
            for r in self.results.values():
                writer.writerow([r.path, f"{r.prob:.4f}",
                                 "PNEUMONIA" if r.prob > self.threshold else "NORMAL",
                                 f"{self.threshold:.2f}"])
        self.statusBar().showMessage(f"Report saved to {path}", 5000)

    def clear_all(self):
        self.list.clear()
        self.results.clear()
        self.rows.clear()
        self.current = None
        self._show_current_image()
        self._update_details()
        self._update_summary()
        self._set_controls_enabled()


def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PneumoScan.Desktop")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setWindowIcon(QIcon(make_icon_pixmap()))

    if "--make-icon" in sys.argv:
        ICON_PATH.parent.mkdir(exist_ok=True)
        make_icon_pixmap(256).save(str(ICON_PATH), "ICO")
        return

    window = MainWindow()
    window.show()
    window.enqueue(sys.argv[1:])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

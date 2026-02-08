import sys
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE_DIR = os.path.join(APP_DIR, "models_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_CACHE_DIR

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QProgressBar, QLabel, QLineEdit, QTextEdit,
                             QMessageBox, QDoubleSpinBox, QSlider, QCheckBox, QGroupBox, QStyledItemDelegate)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QTextDocument

# REPOSITORY MAPPINGS
MODELS = {
    "WD-EVA02-Large-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "WD-ViT-Large-v3": "SmilingWolf/wd-vit-large-tagger-v3",
    "WD-swinv2": "SmilingWolf/wd-swinv2-tagger-v3"
}

# Explicit Rating values expected in SmilingWolf CSVs
VALID_RATINGS = {"general", "sensitive", "questionable", "explicit"}

class WordWrapDelegate(QStyledItemDelegate):
    """Ensures text wraps properly and forces the row to expand vertically."""
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignTop | Qt.AlignLeft
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        text = index.model().data(index, Qt.DisplayRole)
        doc = QTextDocument()
        doc.setHtml(text)
        doc.setTextWidth(option.rect.width())
        return QSize(doc.idealWidth(), doc.size().height())

class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Preview")
        self.resize(800, 800)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel("Select an image to preview")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #050505; color: #666; font-family: 'Consolas';")
        self.label.setMinimumSize(200, 200)
        
        self.layout.addWidget(self.label)
        self.current_pixmap = None

    def display_image(self, path):
        if os.path.exists(path):
            self.current_pixmap = QPixmap(path)
            self._update_label_pixmap()
            self.setWindowTitle(f"Preview: {os.path.basename(path)}")
        else:
            self.label.setText(f"File Not Found: {os.path.basename(path)}")
            self.current_pixmap = None

    def _update_label_pixmap(self):
        if self.current_pixmap:
            scaled = self.current_pixmap.scaled(
                self.label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        if self.current_pixmap:
            self._update_label_pixmap()
        super().resizeEvent(event)

class TaggerWorker(QThread):
    progress = Signal(int)
    finished = Signal(dict)
    log = Signal(str)
    diag = Signal(str)

    def __init__(self, directory, model_configs, include_rating=False):
        super().__init__()
        self.directory = directory
        self.model_configs = model_configs 
        self.include_rating = include_rating
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            sessions = []
            for cfg in self.model_configs:
                repo = cfg['repo']
                self.diag.emit(f"Loading Session: {repo}")
                try:
                    model_path = hf_hub_download(repo, "model.onnx")
                except:
                    model_path = hf_hub_download(repo, "tagger.onnx")
                
                tags_path = hf_hub_download(repo, "selected_tags.csv")
                
                providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
                try:
                    sess = ort.InferenceSession(model_path, providers=providers)
                except Exception:
                    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                
                df = pd.read_csv(tags_path)
                sessions.append({
                    'repo': repo,
                    'session': sess,
                    'tags_df': df,
                    'threshold': cfg['threshold'],
                    'input_name': sess.get_inputs()[0].name,
                    'shape': sess.get_inputs()[0].shape[1:3]
                })

            image_files = sorted([f for f in os.listdir(self.directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
            
            for i, filename in enumerate(image_files):
                if not self._is_running: break
                img_path = os.path.join(self.directory, filename)
                
                all_tags = set()
                best_rating_name = None
                max_rating_prob = -1.0

                try:
                    raw_img = Image.open(img_path).convert("RGB")
                    img_np = np.array(raw_img).astype(np.float32)

                    for s in sessions:
                        h, w = s['shape']
                        input_img = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_AREA)
                        input_img = np.expand_dims(input_img, axis=0)
                        
                        probs = s['session'].run(None, {s['input_name']: input_img})[0][0]
                        df = s['tags_df']

                        for idx, p in enumerate(probs):
                            tag_item = df.iloc[idx]
                            cat = tag_item['category']
                            name_raw = str(tag_item['name'])
                            name_norm = name_raw.lower().replace("_", " ")

                            # Categorize: 0=General, 1=Character
                            if p >= s['threshold'] and cat in [0, 1]:
                                all_tags.add(name_norm)
                            
                            # Broad Rating Detection (Handles Cat 2 or name-based matches)
                            is_rating_entry = (cat == 2) or ("rating" in name_norm) or (name_norm in VALID_RATINGS)
                            
                            if is_rating_entry:
                                clean_rating = name_norm.split(":")[-1].strip() if ":" in name_norm else name_norm
                                if clean_rating in VALID_RATINGS:
                                    if p > max_rating_prob:
                                        max_rating_prob = p
                                        best_rating_name = clean_rating
                                        # Specific diagnostic output for the user
                                        self.diag.emit(f"IMG: {filename} | Model: {s['repo']} | Rating: {best_rating_name} ({p:.4f})")

                    final_list = sorted(list(all_tags))
                    
                    if self.include_rating and best_rating_name:
                        # Ensure we don't double up if a model returned it as a general tag
                        final_list = [t for t in final_list if not t.startswith("rating:")]
                        final_list.append(f"rating:{best_rating_name}")

                    tag_str = ", ".join(final_list)
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(tag_str)
                except Exception as e:
                    self.diag.emit(f"Error processing {filename}: {str(e)}")
                    continue
                
                self.progress.emit(int((i + 1) / len(image_files) * 100))
            
            self.finished.emit({})
        except Exception as e:
            self.log.emit(f"Fatal Error: {str(e)}")

class SurgicalTagger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surgical Tagger v2.6 - Explicit Controls & Broad Rating Sync")
        self.resize(1500, 1000)
        self.dataset_path = ""
        self.worker = None
        self.preview_window = PreviewWindow()
        
        container = QWidget()
        self.setCentralWidget(container)
        self.layout = QVBoxLayout(container)
        
        self.setup_ui()
        
        self.main_split = QHBoxLayout()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Filename", "Tags"])
        self.table.setWordWrap(True)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setItemDelegateForColumn(1, WordWrapDelegate())
        
        # Keep click-to-preview but we also added the explicit button
        self.table.itemClicked.connect(self.handle_item_clicked)
        
        self.diag_log = QTextEdit()
        self.diag_log.setReadOnly(True)
        self.diag_log.setMaximumWidth(450)
        self.diag_log.setStyleSheet("background-color: #0d0d0d; color: #00ff41; font-family: 'Consolas'; font-size: 11px;")

        self.main_split.addWidget(self.table, 3)
        self.main_split.addWidget(self.diag_log, 1)
        self.layout.addLayout(self.main_split)
        
        self.progress = QProgressBar()
        self.status = QLabel("System Idle")
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.status)

    def setup_ui(self):
        # Ensemble Config
        ens_group = QGroupBox("Ensemble Control (Quadratic Precision)")
        ens_layout = QVBoxLayout(ens_group)
        
        self.model_rows = {}
        for name in MODELS.keys():
            row = QHBoxLayout()
            cb = QCheckBox(name); cb.setChecked(True)
            
            spin = QDoubleSpinBox()
            spin.setRange(0.0000, 1.0000)
            spin.setDecimals(4)
            spin.setSingleStep(0.005)
            spin.setValue(0.3500)
            spin.setFixedWidth(90)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(int(np.sqrt(0.35) * 1000))

            # Quadratic Linkage
            slider.valueChanged.connect(lambda v, s=spin: s.setValue((v/1000)**2))
            spin.valueChanged.connect(lambda v, sl=slider: sl.blockSignals(True) or sl.setValue(int(np.sqrt(v)*1000)) or sl.blockSignals(False))
            
            row.addWidget(cb)
            row.addWidget(spin)
            row.addWidget(slider)
            self.model_rows[name] = {"check": cb, "spin": spin, "slider": slider}
            ens_layout.addLayout(row)
        
        opt_row = QHBoxLayout()
        self.check_rating = QCheckBox("Force Rating Postfix (e.g., rating:general)")
        self.check_rating.setChecked(True)
        opt_row.addWidget(self.check_rating); opt_row.addStretch()
        ens_layout.addLayout(opt_row)
        self.layout.addWidget(ens_group)

        # Navigation & Restoration of the Preview Button
        nav = QHBoxLayout()
        btn_open = QPushButton("📂 Open Dataset")
        btn_open.clicked.connect(self.select_dir)
        
        # RESTORED: Explicit Show Preview Button
        btn_preview = QPushButton("👁 Show Preview")
        btn_preview.clicked.connect(self.manual_preview_trigger)
        
        self.btn_run = QPushButton("⚡ START ENSEMBLE")
        self.btn_run.setStyleSheet("background-color: #1b5e20; color: white; min-height: 40px; font-weight: bold;")
        self.btn_run.clicked.connect(self.toggle_process)
        
        nav.addWidget(btn_open)
        nav.addWidget(btn_preview)
        nav.addWidget(self.btn_run)
        self.layout.addLayout(nav)

        # Editing Toolbar
        edit_bar = QHBoxLayout()
        self.f_in = QLineEdit(); self.f_in.setPlaceholderText("Find Tag...")
        self.r_in = QLineEdit(); self.r_in.setPlaceholderText("Replace With...")
        btn_rep = QPushButton("Replace"); btn_rep.clicked.connect(self.do_replace)
        self.p_in = QLineEdit(); self.p_in.setPlaceholderText("Prepend Tag...")
        btn_pre = QPushButton("Prepend"); btn_pre.clicked.connect(self.do_prepend)
        btn_save = QPushButton("💾 SAVE ALL TXT")
        btn_save.setStyleSheet("background-color: #0d47a1; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_to_disk)
        
        edit_bar.addWidget(self.f_in); edit_bar.addWidget(self.r_in); edit_bar.addWidget(btn_rep)
        edit_bar.addWidget(self.p_in); edit_bar.addWidget(btn_pre); edit_bar.addWidget(btn_save)
        self.layout.addLayout(edit_bar)

    def select_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if path:
            self.dataset_path = path
            self.load_table()

    def load_table(self):
        if not self.dataset_path: return
        self.table.setRowCount(0)
        files = sorted([f for f in os.listdir(self.dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        for f in files:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(f))
            tp = os.path.join(self.dataset_path, os.path.splitext(f)[0] + ".txt")
            content = ""
            if os.path.exists(tp):
                try:
                    with open(tp, "r", encoding="utf-8") as tf: content = tf.read()
                except: pass
            it = QTableWidgetItem(content)
            it.setTextAlignment(Qt.AlignTop | Qt.AlignLeft)
            self.table.setItem(row, 1, it)
        self.table.resizeRowsToContents()
        self.status.setText(f"Loaded {len(files)} images from {self.dataset_path}")

    def manual_preview_trigger(self):
        """Logic for the explicit Preview button."""
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.information(self, "Info", "Please select a row in the table first.")
            return
        filename = self.table.item(current_row, 0).text()
        self.preview_window.display_image(os.path.join(self.dataset_path, filename))
        if not self.preview_window.isVisible():
            self.preview_window.show()

    def handle_item_clicked(self, item):
        """Maintain the convenience of clicking the filename to preview."""
        if item.column() == 0:
            self.preview_window.display_image(os.path.join(self.dataset_path, item.text()))
            if not self.preview_window.isVisible():
                self.preview_window.show()

    def toggle_process(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_run.setText("⚡ START ENSEMBLE")
        else:
            if not self.dataset_path:
                QMessageBox.warning(self, "Error", "No dataset directory opened.")
                return
            self.diag_log.clear()
            configs = [{"repo": MODELS[n], "threshold": w["spin"].value()} for n, w in self.model_rows.items() if w["check"].isChecked()]
            if not configs:
                QMessageBox.warning(self, "Error", "No models selected for ensemble.")
                return
            self.worker = TaggerWorker(self.dataset_path, configs, self.check_rating.isChecked())
            self.worker.progress.connect(self.progress.setValue)
            self.worker.diag.connect(lambda msg: self.diag_log.append(msg))
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.start()
            self.btn_run.setText("⏹ STOP")

    def on_worker_finished(self):
        self.load_table()
        self.btn_run.setText("⚡ START ENSEMBLE")
        self.status.setText("Batch processing complete.")

    def save_to_disk(self):
        if not self.dataset_path: return
        for r in range(self.table.rowCount()):
            fn = self.table.item(r, 0).text()
            it = self.table.item(r, 1)
            txt = it.text() if it else ""
            with open(os.path.join(self.dataset_path, os.path.splitext(fn)[0] + ".txt"), "w", encoding="utf-8") as f:
                f.write(txt)
        self.status.setText("All changes committed to disk.")

    def do_replace(self):
        f, r = self.f_in.text().strip(), self.r_in.text().strip()
        if not f: return
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 1)
            if it:
                tags = [t.strip() for t in it.text().split(",") if t.strip()]
                new_tags = [r if t == f else t for t in tags]
                it.setText(", ".join(new_tags))
        self.table.resizeRowsToContents()

    def do_prepend(self):
        p = self.p_in.text().strip()
        if not p: return
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 1)
            if it:
                tags = [t.strip() for t in it.text().split(",") if t.strip()]
                if p in tags: tags.remove(p)
                it.setText(", ".join([p] + tags))
        self.table.resizeRowsToContents()

    def closeEvent(self, event):
        self.preview_window.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SurgicalTagger(); w.show()
    sys.exit(app.exec())







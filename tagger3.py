import sys
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE_DIR = os.path.join(APP_DIR, "models_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_CACHE_DIR

import cv2
import numpy as np
import pandas as pd
import json
from huggingface_hub import hf_hub_download, list_repo_tree
import onnxruntime as ort
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QProgressBar, QLabel, QLineEdit, QTextEdit,
                             QMessageBox, QDoubleSpinBox, QSlider, QCheckBox, QGroupBox, 
                             QStyledItemDelegate, QScrollArea, QInputDialog)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QTextDocument

# REPOSITORY MAPPINGS
# Pre-defined known good models
MODELS = {
    "WD-EVA02-Large-v3": {"repo": "SmilingWolf/wd-eva02-large-tagger-v3", "subfolder": None},
    "WD-ViT-Large-v3": {"repo": "SmilingWolf/wd-vit-large-tagger-v3", "subfolder": None},
    "WD-SwinV2-v3": {"repo": "SmilingWolf/wd-swinv2-tagger-v3", "subfolder": None},
    "WD-ConvNext-v3": {"repo": "SmilingWolf/wd-convnext-tagger-v3", "subfolder": None},
    "CL-Auto-Latest": {"repo": "cella110n/cl_tagger", "subfolder": "AUTO"}
}

class WordWrapDelegate(QStyledItemDelegate):
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
            scaled = self.current_pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        if self.current_pixmap: self._update_label_pixmap()
        super().resizeEvent(event)

class TaggerWorker(QThread):
    progress = Signal(int)
    finished = Signal(dict)
    log = Signal(str)
    diag = Signal(str)

    def __init__(self, directory, model_configs, filters):
        super().__init__()
        self.directory = directory
        self.model_configs = model_configs
        self.filters = filters 
        self._is_running = True

    def get_latest_cl_subfolder(self, repo_id):
        try:
            self.diag.emit(f"Scanning {repo_id} for versions...")
            items = list_repo_tree(repo_id)
            folders = []
            for item in items:
                item_path = getattr(item, 'path', str(item))
                is_dir = getattr(item, 'type', None) == 'directory' or 'RepoFolder' in str(type(item))
                if is_dir and item_path.startswith("cl_"):
                    folders.append(item_path)
            if not folders: return None
            latest = sorted(folders)[-1]
            return latest
        except Exception as e:
            self.diag.emit(f"Discovery Error: {str(e)}")
            return None

    def run(self):
        try:
            sessions = []
            for cfg in self.model_configs:
                repo = cfg['repo'] # Can be repo_id OR local path to .onnx
                subfolder = cfg.get('subfolder')
                is_local = os.path.isfile(repo)
                
                # --- MODEL LOADING ---
                if is_local:
                    self.diag.emit(f"Loading Local Model: {os.path.basename(repo)}")
                    model_path = repo
                    # For local custom models, we expect the tags file to be passed in 'subfolder' 
                    # (we repurpose this field for custom tag path)
                    tags_path = subfolder 
                else:
                    # Hugging Face Loading
                    if subfolder == "AUTO" and "cl_tagger" in repo:
                        subfolder = self.get_latest_cl_subfolder(repo)
                        if not subfolder: continue
                    
                    self.diag.emit(f"Downloading: {repo} (Sub: {subfolder})")
                    model_path = hf_hub_download(repo, "model.onnx", subfolder=subfolder)
                    
                    # Determine tag file for HF models
                    if "cl_tagger" in repo:
                        tags_path = hf_hub_download(repo, "tag_mapping.json", subfolder=subfolder)
                    else:
                        tags_path = hf_hub_download(repo, "selected_tags.csv")

                # --- TAG PARSING ---
                tag_names = []
                tag_categories = []
                
                try:
                    if tags_path.endswith('.csv'):
                        df = pd.read_csv(tags_path)
                        tag_names = df["name"].tolist()
                        tag_categories = df["category"].tolist()
                    elif tags_path.endswith('.json'):
                        with open(tags_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Handle various JSON structures
                        if isinstance(data, list):
                            # Simple list ["tag1", "tag2"]
                            tag_names = data
                            tag_categories = [0] * len(data) # Default to general
                        elif isinstance(data, dict):
                            # Cella Format: {"0": {"tag": "name", "category": 0}}
                            # OR JoyTag/Simple Format: {"0": "name", "1": "name"}
                            
                            keys = sorted(data.keys(), key=lambda x: int(x))
                            first_val = data[keys[0]]
                            
                            if isinstance(first_val, dict) and "tag" in first_val:
                                # Cella style
                                tag_names = [data[k]['tag'] for k in keys]
                                tag_categories = [data[k].get('category', 0) for k in keys]
                            else:
                                # Simple ID-to-Label
                                tag_names = [data[k] for k in keys]
                                tag_categories = [0] * len(keys)
                    else:
                        # Fallback for simple txt lists (line by line)
                        with open(tags_path, 'r', encoding='utf-8') as f:
                            lines = [l.strip() for l in f if l.strip()]
                        tag_names = lines
                        tag_categories = [0] * len(lines)
                except Exception as e:
                    self.diag.emit(f"Tag Parsing Error: {str(e)}")
                    continue

                # --- ONNX SESSION ---
                sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                input_meta = sess.get_inputs()[0]
                shape = input_meta.shape
                
                def get_dim(val):
                    try: return int(val)
                    except: return 448 

                # Layout Detection
                is_nchw = False
                if len(shape) >= 4:
                    if shape[1] == 3 or str(shape[1]).lower() == 'channels': 
                        is_nchw = True
                        h, w = get_dim(shape[2]), get_dim(shape[3])
                    else:
                        h, w = get_dim(shape[1]), get_dim(shape[2])
                else:
                    h, w = 448, 448

                self.diag.emit(f"Initialized: {w}x{h} (NCHW: {is_nchw})")

                sessions.append({
                    "sess": sess,
                    "input_name": input_meta.name,
                    "dim": (w, h),
                    "is_nchw": is_nchw,
                    "tags": tag_names,
                    "cats": tag_categories,
                    "thresh": cfg['thresh'],
                    "is_cl": "cl_tagger" in repo or is_local # Assume local models might need normalization too
                })

            files = [f for f in os.listdir(self.directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            results = {}

            for i, filename in enumerate(files):
                if not self._is_running: break
                path = os.path.join(self.directory, filename)
                raw_img = cv2.imread(path)
                if raw_img is None: continue
                img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                
                all_tags = set()
                for s in sessions:
                    img_proc = cv2.resize(img, s["dim"], interpolation=cv2.INTER_AREA).astype(np.float32)
                    
                    # CL and many newer ViT models expect 0-1 normalization
                    # WD14 usually expects 0-255 (BGR->RGB), but we can standardize on 0-1 for most ViTs
                    # If model yields garbage, this is usually the culprit. 
                    # SmilingWolf V3 models actually work well with float inputs in ONNX Runtime.
                    if s["is_cl"]:
                         img_proc /= 255.0
                    
                    if s["is_nchw"]:
                        img_proc = np.transpose(img_proc, (2, 0, 1))

                    img_batch = np.expand_dims(img_proc, axis=0)
                    probs = s["sess"].run(None, {s["input_name"]: img_batch})[0][0]
                    
                    for idx, score in enumerate(probs):
                        if score >= s["thresh"]:
                            if idx < len(s["tags"]):
                                tag = s["tags"][idx]
                                cat_val = s["cats"][idx]
                                cat = str(cat_val).lower()
                                
                                keep = False
                                # Flexible filtering
                                if cat in ["0", "general", "none"] and self.filters['general']: keep = True
                                elif cat in ["4", "character"] and self.filters['character']: keep = True
                                elif cat in ["1", "copyright", "series"] and self.filters['copyright']: keep = True
                                elif cat in ["9", "rating"] and self.filters['rating']: keep = True
                                elif cat in ["5", "meta"] and self.filters['meta']: keep = True
                                elif cat == "quality" and self.filters['quality']: keep = True
                                elif cat == "model" and self.filters['model']: keep = True
                                
                                # Fallback: if category is unknown (e.g. from custom JSON), keep it if general is checked
                                if cat not in ["0", "4", "1", "9", "5", "general", "character", "copyright", "rating", "meta", "quality", "model"]:
                                    if self.filters['general']: keep = True

                                if keep:
                                    all_tags.add(tag.replace("_", " "))

                results[filename] = ", ".join(sorted(list(all_tags)))
                self.progress.emit(int((i + 1) / len(files) * 100))

            self.finished.emit(results)
        except Exception as e:
            self.diag.emit(f"Worker Error: {str(e)}")
            self.finished.emit({})

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Booru Tagger v2026.3 (Universal + High Precision)")
        self.resize(1300, 900)
        self.dataset_path = ""
        self.custom_models = [] # Stores (name, onnx_path, tags_path)
        self.preview_window = PreviewWindow()
        self.init_ui()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        # SIDEBAR
        sidebar = QWidget(); sidebar.setFixedWidth(400)
        s_layout = QVBoxLayout(sidebar)
        
        dir_grp = QGroupBox("Directory")
        dl = QVBoxLayout(dir_grp)
        self.path_display = QLabel("No path selected")
        self.path_display.setWordWrap(True)
        btn_sel = QPushButton("Select Folder")
        btn_sel.clicked.connect(self.select_folder)
        dl.addWidget(self.path_display); dl.addWidget(btn_sel)
        s_layout.addWidget(dir_grp)

        # Model List
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.m_layout = QVBoxLayout(container)
        self.model_refs = {}
        
        # Add predefined models
        for name, data in MODELS.items():
            self.add_model_row(name, data['repo'], data['subfolder'])
            
        scroll.setWidget(container)
        s_layout.addWidget(scroll)

        # Custom Model Button
        btn_custom = QPushButton("+ Load Custom ONNX Model")
        btn_custom.clicked.connect(self.load_custom_model)
        s_layout.addWidget(btn_custom)

        cat_grp = QGroupBox("Categories")
        cl = QVBoxLayout(cat_grp)
        self.f_checks = {
            "general": QCheckBox("General Attributes"),
            "character": QCheckBox("Characters"),
            "copyright": QCheckBox("Copyright/Series"),
            "rating": QCheckBox("Content Ratings"),
            "meta": QCheckBox("Meta"),
            "quality": QCheckBox("Quality (CL/Custom)"),
            "model": QCheckBox("Model Specs")
        }
        for c in self.f_checks.values():
            c.setChecked(True)
            cl.addWidget(c)
        s_layout.addWidget(cat_grp)

        self.btn_go = QPushButton("RUN BATCH")
        self.btn_go.setFixedHeight(50)
        self.btn_go.setStyleSheet("background: #2e7d32; font-weight: bold; color: white;")
        self.btn_go.clicked.connect(self.start_tagging)
        s_layout.addWidget(self.btn_go)

        self.pbar = QProgressBar()
        s_layout.addWidget(self.pbar)
        
        self.logs = QTextEdit(); self.logs.setReadOnly(True); self.logs.setMaximumHeight(150)
        self.logs.setStyleSheet("background: #111; color: #0f0; font-family: 'Consolas'; font-size: 10px;")
        s_layout.addWidget(self.logs)

        # CONTENT AREA
        content = QWidget()
        clayout = QVBoxLayout(content)
        
        tools_grp = QGroupBox("Batch Editing Tools")
        tl = QVBoxLayout(tools_grp)
        r1 = QHBoxLayout()
        self.f_in = QLineEdit(); self.f_in.setPlaceholderText("Find tag...")
        self.r_in = QLineEdit(); self.r_in.setPlaceholderText("Replace with...")
        btn_rep = QPushButton("Replace All"); btn_rep.clicked.connect(self.do_replace)
        r1.addWidget(self.f_in); r1.addWidget(self.r_in); r1.addWidget(btn_rep)
        
        r2 = QHBoxLayout()
        self.p_in = QLineEdit(); self.p_in.setPlaceholderText("Prepend tag...")
        btn_pre = QPushButton("Prepend"); btn_pre.clicked.connect(self.do_prepend)
        r2.addWidget(self.p_in); r2.addWidget(btn_pre)
        tl.addLayout(r1); tl.addLayout(r2)
        clayout.addWidget(tools_grp)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Tags"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setItemDelegateForColumn(1, WordWrapDelegate())
        self.table.itemSelectionChanged.connect(self.show_preview)
        clayout.addWidget(self.table)

        btn_save = QPushButton("SAVE ALL TO DISK")
        btn_save.setFixedHeight(40)
        btn_save.clicked.connect(self.commit_to_disk)
        clayout.addWidget(btn_save)

        layout.addWidget(sidebar)
        layout.addWidget(content)

    def add_model_row(self, name, repo, subfolder, checked=False):
        grp = QGroupBox(name)
        gl = QHBoxLayout(grp)
        chk = QCheckBox("Use")
        # Default checking logic: EVA02 checks by default, others dont
        chk.setChecked(checked or ("EVA02" in name and not self.custom_models)) 
        
        # High Precision SpinBox
        spn = QDoubleSpinBox()
        spn.setRange(0.0000, 1.0000)
        spn.setDecimals(4)
        spn.setSingleStep(0.005) # Fine tuning step
        spn.setValue(0.3500)
        spn.setFixedWidth(80) # Slightly wider for 4 digits

        # Quadratic Slider
        sld = QSlider(Qt.Horizontal)
        sld.setRange(0, 1000)
        # Initialize slider position based on default spin value (Quadratic: sld = sqrt(val)*1000)
        sld.setValue(int(np.sqrt(0.3500) * 1000))

        # Linkage Logic
        # Slider -> Spinbox (Quadratic map: v -> (v/1000)^2)
        sld.valueChanged.connect(lambda v, s=spn: s.setValue((v/1000)**2))
        # Spinbox -> Slider (Inverse map: v -> sqrt(v)*1000)
        spn.valueChanged.connect(lambda v, s=sld: s.blockSignals(True) or s.setValue(int(np.sqrt(v)*1000)) or s.blockSignals(False))

        gl.addWidget(chk)
        gl.addWidget(QLabel("Thresh:"))
        gl.addWidget(spn)
        gl.addWidget(sld) # Slider added back
        
        self.m_layout.addWidget(grp)
        # Store using name as key
        self.model_refs[name] = {"chk": chk, "spn": spn, "repo": repo, "subfolder": subfolder}

    def load_custom_model(self):
        onnx_file, _ = QFileDialog.getOpenFileName(self, "Select Model (.onnx)", "", "ONNX Models (*.onnx)")
        if not onnx_file: return
        
        tags_file, _ = QFileDialog.getOpenFileName(self, "Select Tags (.json, .csv, .txt)", "", "Tag Files (*.json *.csv *.txt)")
        if not tags_file: return
        
        name_guess = os.path.basename(onnx_file).replace(".onnx", "")
        name, ok = QInputDialog.getText(self, "Model Name", "Enter a name for this model:", text=name_guess)
        if not ok: return
        
        # Add to UI
        self.add_model_row(name, onnx_file, tags_file, checked=True)
        self.logs.append(f"Added Custom Model: {name}")

    def select_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select Dataset")
        if p:
            self.dataset_path = p
            self.path_display.setText(p)
            self.load_current_tags()

    def load_current_tags(self):
        files = [f for f in os.listdir(self.dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        self.table.setRowCount(len(files))
        for i, f in enumerate(files):
            self.table.setItem(i, 0, QTableWidgetItem(f))
            self.table.item(i, 0).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            tp = os.path.join(self.dataset_path, os.path.splitext(f)[0] + ".txt")
            val = ""
            if os.path.exists(tp):
                with open(tp, 'r', encoding='utf-8') as fh: val = fh.read()
            self.table.setItem(i, 1, QTableWidgetItem(val))
        self.table.resizeRowsToContents()

    def show_preview(self):
        rows = self.table.selectedItems()
        if rows:
            fn = self.table.item(rows[0].row(), 0).text()
            self.preview_window.display_image(os.path.join(self.dataset_path, fn))
            if not self.preview_window.isVisible(): self.preview_window.show()

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

    def start_tagging(self):
        if not self.dataset_path: return
        active = []
        for n, r in self.model_refs.items():
            if r['chk'].isChecked():
                active.append({'repo': r['repo'], 'subfolder': r['subfolder'], 'thresh': r['spn'].value()})
        if not active: return
        
        self.btn_go.setEnabled(False)
        self.worker = TaggerWorker(self.dataset_path, active, {k: v.isChecked() for k, v in self.f_checks.items()})
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.diag.connect(lambda m: self.logs.append(m))
        self.worker.finished.connect(self.finalize)
        self.worker.start()

    def finalize(self, results):
        for r in range(self.table.rowCount()):
            fn = self.table.item(r, 0).text()
            if fn in results: self.table.setItem(r, 1, QTableWidgetItem(results[fn]))
        self.table.resizeRowsToContents()
        self.btn_go.setEnabled(True)

    def commit_to_disk(self):
        if not self.dataset_path: return
        for r in range(self.table.rowCount()):
            fn = self.table.item(r, 0).text()
            it = self.table.item(r, 1)
            txt = it.text() if it else ""
            with open(os.path.join(self.dataset_path, os.path.splitext(fn)[0] + ".txt"), 'w', encoding='utf-8') as f:
                f.write(txt)
        QMessageBox.information(self, "Saved", "All text files updated.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = App()
    window.show()
    sys.exit(app.exec())






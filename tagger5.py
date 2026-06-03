import sys
import os
import json
import base64
import http.server
import threading
from urllib.parse import urlparse

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_CACHE_DIR = os.path.join(APP_DIR, "models_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_CACHE_DIR

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_tree
import onnxruntime as ort
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QProgressBar, QLabel, QLineEdit, QTextEdit,
                             QMessageBox, QDoubleSpinBox, QSlider, QCheckBox, QGroupBox, 
                             QStyledItemDelegate, QScrollArea, QInputDialog, QTabWidget, QSpinBox)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QTextDocument

# Pre-defined known good models
MODELS = {
    "WD-EVA02-Large-v3": {"repo": "SmilingWolf/wd-eva02-large-tagger-v3", "subfolder": None},
    "WD-ViT-Large-v3": {"repo": "SmilingWolf/wd-vit-large-tagger-v3", "subfolder": None},
    "WD-SwinV2-v3": {"repo": "SmilingWolf/wd-swinv2-tagger-v3", "subfolder": None},
    "WD-ConvNext-v3": {"repo": "SmilingWolf/wd-convnext-tagger-v3", "subfolder": None},
    "CL-Auto-Latest": {"repo": "cella110n/cl_tagger", "subfolder": "AUTO"}
}

# -------------------------------------------------------------------------
# INFERENCE ENGINE (Shared between Batch Worker and Agent Server)
# -------------------------------------------------------------------------
class InferenceEngine:
    def __init__(self, diag_callback=None):
        self.sessions = []
        self.diag = diag_callback if diag_callback else print

    def get_latest_cl_subfolder(self, repo_id):
        try:
            self.diag(f"Scanning {repo_id} for versions...")
            items = list_repo_tree(repo_id)
            folders = [getattr(i, 'path', str(i)) for i in items if (getattr(i, 'type', None) == 'directory' or 'RepoFolder' in str(type(i))) and getattr(i, 'path', str(i)).startswith("cl_")]
            if not folders: return None
            return sorted(folders)[-1]
        except Exception as e:
            self.diag(f"Discovery Error: {str(e)}")
            return None

    def load_models(self, model_configs):
        self.sessions = []
        for cfg in model_configs:
            repo, subfolder, thresh = cfg['repo'], cfg.get('subfolder'), cfg['thresh']
            is_local = os.path.isfile(repo)
            
            try:
                if is_local:
                    self.diag(f"Loading Local Model: {os.path.basename(repo)}")
                    model_path, tags_path = repo, subfolder 
                else:
                    if subfolder == "AUTO" and "cl_tagger" in repo:
                        subfolder = self.get_latest_cl_subfolder(repo)
                        if not subfolder: continue
                    self.diag(f"Downloading: {repo} (Sub: {subfolder})")
                    model_path = hf_hub_download(repo, "model.onnx", subfolder=subfolder)
                    tags_path = hf_hub_download(repo, "tag_mapping.json", subfolder=subfolder) if "cl_tagger" in repo else hf_hub_download(repo, "selected_tags.csv")

                tag_names, tag_categories = [], []
                if tags_path.endswith('.csv'):
                    df = pd.read_csv(tags_path)
                    tag_names, tag_categories = df["name"].tolist(), df["category"].tolist()
                elif tags_path.endswith('.json'):
                    with open(tags_path, 'r', encoding='utf-8') as f: data = json.load(f)
                    if isinstance(data, list):
                        tag_names, tag_categories = data, [0] * len(data)
                    elif isinstance(data, dict):
                        keys = sorted(data.keys(), key=lambda x: int(x))
                        if isinstance(data[keys[0]], dict) and "tag" in data[keys[0]]:
                            tag_names = [data[k]['tag'] for k in keys]
                            tag_categories = [data[k].get('category', 0) for k in keys]
                        else:
                            tag_names, tag_categories = [data[k] for k in keys], [0] * len(keys)
                else:
                    with open(tags_path, 'r', encoding='utf-8') as f: lines = [l.strip() for l in f if l.strip()]
                    tag_names, tag_categories = lines, [0] * len(lines)

                sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                input_meta = sess.get_inputs()[0]
                shape = input_meta.shape
                
                def get_dim(val):
                    try: return int(val)
                    except: return 448 

                is_nchw = False
                if len(shape) >= 4:
                    if shape[1] == 3 or str(shape[1]).lower() == 'channels': 
                        is_nchw, h, w = True, get_dim(shape[2]), get_dim(shape[3])
                    else:
                        h, w = get_dim(shape[1]), get_dim(shape[2])
                else:
                    h, w = 448, 448

                self.sessions.append({
                    "sess": sess, "input_name": input_meta.name, "dim": (w, h),
                    "is_nchw": is_nchw, "tags": tag_names, "cats": tag_categories,
                    "thresh": thresh, "is_cl": "cl_tagger" in repo or is_local
                })
                self.diag(f"Successfully loaded {os.path.basename(model_path)}")
            except Exception as e:
                self.diag(f"Failed to load {repo}: {str(e)}")

    def predict(self, img_rgb, active_categories, negatives, max_tags, use_mcut=False):
        # Accumulator for deduplication and averaging
        # Structure: { category_name: { tag_name: [weight1, weight2] } }
        tag_weights = {cat: {} for cat in active_categories}
        
        for s in self.sessions:
            img_proc = cv2.resize(img_rgb, s["dim"], interpolation=cv2.INTER_AREA).astype(np.float32)
            if s["is_cl"]: img_proc /= 255.0
            if s["is_nchw"]: img_proc = np.transpose(img_proc, (2, 0, 1))
            img_batch = np.expand_dims(img_proc, axis=0)
            
            probs = s["sess"].run(None, {s["input_name"]: img_batch})[0][0]
            
            # MCut Dynamic Thresholding Logic
            if use_mcut:
                valid_probs = [(i, p) for i, p in enumerate(probs) if p >= 0.05] # Floor to prevent cutting extreme noise
                valid_probs.sort(key=lambda x: x[1], reverse=True)
                
                if len(valid_probs) > 1:
                    max_gap = 0
                    cut_index = 0
                    for i in range(len(valid_probs) - 1):
                        gap = valid_probs[i][1] - valid_probs[i+1][1]
                        if gap > max_gap:
                            max_gap = gap
                            cut_index = i
                    dynamic_thresh = valid_probs[cut_index+1][1]
                else:
                    dynamic_thresh = s["thresh"]
            else:
                dynamic_thresh = s["thresh"]
            
            for idx, score in enumerate(probs):
                if score >= dynamic_thresh:
                    if idx >= len(s["tags"]): continue
                    tag = s["tags"][idx].replace("_", " ")
                    
                    # Substring Negatives Check
                    if any(neg.lower() in tag.lower() for neg in negatives if neg.strip()): continue
                    
                    cat_val = str(s["cats"][idx]).lower()
                    
                    # Map to known categories
                    mapped_cat = None
                    if cat_val in ["0", "general", "none"]: mapped_cat = "General Attributes"
                    elif cat_val in ["4", "character"]: mapped_cat = "Characters"
                    elif cat_val == "quality": mapped_cat = "Quality"
                    elif cat_val in ["1", "copyright", "series"]: mapped_cat = "Copyright/Series"
                    elif cat_val in ["9", "rating"]: mapped_cat = "Content Ratings"
                    elif cat_val in ["5", "meta"]: mapped_cat = "Meta"
                    elif cat_val == "model": mapped_cat = "Model Specs"
                    else: mapped_cat = "General Attributes" # Fallback
                    
                    if mapped_cat in active_categories:
                        if tag not in tag_weights[mapped_cat]:
                            tag_weights[mapped_cat][tag] = []
                        tag_weights[mapped_cat][tag].append(score)

        # Average & Sort
        final_bins = {}
        for cat in active_categories:
            avg_tags = []
            for tag, weights in tag_weights[cat].items():
                avg_tags.append((tag, sum(weights)/len(weights)))
            # Sort heavily weighted tags to the front
            avg_tags.sort(key=lambda x: x[1], reverse=True)
            final_bins[cat] = [t[0] for t in avg_tags]

        # Topology Enforcement: General -> Quality -> Character -> (Others)
        priority_order = ["General Attributes", "Quality", "Characters", "Copyright/Series", "Content Ratings", "Meta", "Model Specs"]
        
        ordered_final_tags = []
        for p_cat in priority_order:
            if p_cat in final_bins:
                ordered_final_tags.extend(final_bins[p_cat])

        return ", ".join(ordered_final_tags[:max_tags])


# -------------------------------------------------------------------------
# AGENT / ORCHESTRATOR SERVER
# -------------------------------------------------------------------------
class AgentServerThread(QThread):
    log_msg = Signal(str)
    
    def __init__(self, ip, port, inference_engine, categories, negatives, max_tags, use_mcut):
        super().__init__()
        self.ip = ip
        self.port = port
        self.engine = inference_engine
        self.categories = categories
        self.negatives = negatives
        self.max_tags = max_tags
        self.use_mcut = use_mcut
        self.server = None

    def run(self):
        parent_thread = self
        
        class OrchestratorHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    req = json.loads(post_data)
                    req_id = req.get("request_id", "unknown_req")
                    op_type = req.get("operation_type")
                    
                    if op_type != "llm_generate":
                        raise ValueError(f"Agent unsupported operation: {op_type}")
                    
                    # Extract Base64 from payload
                    messages = req.get("payload", {}).get("messages", [])
                    img_b64 = None
                    for m in messages:
                        for c in m.get("content", []):
                            if c.get("type") == "image_base64":
                                img_b64 = c.get("data")
                                break
                                
                    if not img_b64:
                        raise ValueError("No image_base64 found in payload.")
                        
                    parent_thread.log_msg.emit(f"Orchestrator Request: {req_id} [Processing Image]")
                    
                    # Decode & Process
                    img_bytes = base64.b64decode(img_b64)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    result_tags = parent_thread.engine.predict(
                        img_rgb, parent_thread.categories, parent_thread.negatives, parent_thread.max_tags, parent_thread.use_mcut
                    )
                    
                    # Standard Envelope Response
                    res = {
                        "request_id": req_id,
                        "status": "success",
                        "operation_type": "llm_generate",
                        "response_data": {
                            "reply_text": result_tags,
                            "finish_reason": "stop",
                            "tools_used": ["local_vision_ensemble"]
                        },
                        "error_message": None
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode('utf-8'))
                    parent_thread.log_msg.emit(f"Orchestrator Request: {req_id} [SUCCESS]")
                    
                except Exception as e:
                    parent_thread.log_msg.emit(f"Orchestrator Error: {str(e)}")
                    res = {
                        "request_id": req.get("request_id", "unknown") if 'req' in locals() else "unknown",
                        "status": "error",
                        "operation_type": req.get("operation_type", "unknown") if 'req' in locals() else "unknown",
                        "response_data": None,
                        "error_message": str(e)
                    }
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode('utf-8'))

            def log_message(self, format, *args):
                pass # Suppress default HTTP logs

        self.server = http.server.HTTPServer((self.ip, self.port), OrchestratorHandler)
        self.log_msg.emit(f"Agent Server Online: http://{self.ip}:{self.port}")
        self.server.serve_forever()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.log_msg.emit("Agent Server Offline.")


# -------------------------------------------------------------------------
# BATCH WORKER
# -------------------------------------------------------------------------
class TaggerWorker(QThread):
    progress = Signal(int)
    finished = Signal(dict)
    diag = Signal(str)

    def __init__(self, directory, files, model_configs, active_categories, negatives, max_tags, use_mcut):
        super().__init__()
        self.directory = directory
        self.files = files
        self.engine = InferenceEngine(diag_callback=self.diag.emit)
        self.model_configs = model_configs
        self.active_categories = active_categories
        self.negatives = negatives
        self.max_tags = max_tags
        self.use_mcut = use_mcut
        self._is_running = True

    def run(self):
        try:
            self.engine.load_models(self.model_configs)
            results = {}

            for i, filename in enumerate(self.files):
                if not self._is_running: break
                path = os.path.join(self.directory, filename)
                raw_img = cv2.imread(path)
                if raw_img is None: continue
                img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                
                results[filename] = self.engine.predict(img_rgb, self.active_categories, self.negatives, self.max_tags, self.use_mcut)
                self.progress.emit(int((i + 1) / len(self.files) * 100))

            self.finished.emit(results)
        except Exception as e:
            self.diag.emit(f"Worker Error: {str(e)}")
            self.finished.emit({})


# -------------------------------------------------------------------------
# UI COMPONENTS
# -------------------------------------------------------------------------
class WordWrapDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignTop | Qt.AlignLeft
        super().paint(painter, option, index)
    def sizeHint(self, option, index):
        text = index.model().data(index, Qt.DisplayRole)
        doc = QTextDocument(); doc.setHtml(text); doc.setTextWidth(option.rect.width())
        return QSize(doc.idealWidth(), doc.size().height())

class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Preview")
        self.resize(800, 800)
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("Select an image to preview")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #050505; color: #666;")
        self.layout.addWidget(self.label)
        self.current_pixmap = None

    def display_image(self, path):
        if os.path.exists(path):
            self.current_pixmap = QPixmap(path)
            self._update_label_pixmap()
            self.setWindowTitle(f"Preview: {os.path.basename(path)}")
        else:
            self.label.setText("File Not Found")
            self.current_pixmap = None

    def _update_label_pixmap(self):
        if self.current_pixmap:
            self.label.setPixmap(self.current_pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.current_pixmap: self._update_label_pixmap()
        super().resizeEvent(event)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Booru Tagger Agent Node v2026.4 (Orchestrator Ready)")
        self.resize(1400, 900)
        self.dataset_path = ""
        self.preview_window = PreviewWindow()
        self.agent_thread = None
        self.shared_engine = InferenceEngine(diag_callback=self.log_agent)
        self.init_ui()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        # LEFT SIDEBAR - TABS FOR SETTINGS
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(450)
        
        # --- TAB 1: Models & Categories ---
        tab_models = QWidget(); tm_layout = QVBoxLayout(tab_models)
        
        # Category Group
        cat_grp = QGroupBox("Tag Categories & Limits")
        cl = QVBoxLayout(cat_grp)
        self.f_checks = {
            "General Attributes": QCheckBox("General Attributes"),
            "Quality": QCheckBox("Quality (CL/Custom)"),
            "Characters": QCheckBox("Characters"),
            "Copyright/Series": QCheckBox("Copyright/Series"),
            "Content Ratings": QCheckBox("Content Ratings"),
            "Meta": QCheckBox("Meta"),
            "Model Specs": QCheckBox("Model Specs")
        }
        # ONLY General checked by default per user constraint
        for name, chk in self.f_checks.items():
            chk.setChecked(name == "General Attributes")
            cl.addWidget(chk)
            
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Max Tags Allowed:"))
        self.spin_max_tags = QSpinBox()
        self.spin_max_tags.setRange(10, 300)
        self.spin_max_tags.setValue(75)
        hl.addWidget(self.spin_max_tags)
        cl.addLayout(hl)
        
        self.chk_mcut = QCheckBox("Enable Dynamic Thresholding (MCut)")
        cl.addWidget(self.chk_mcut)
        tm_layout.addWidget(cat_grp)

        # Model Group
        mod_grp = QGroupBox("Active Models")
        ml = QVBoxLayout(mod_grp)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); self.m_layout = QVBoxLayout(container)
        self.model_refs = {}
        for name, data in MODELS.items():
            self.add_model_row(name, data['repo'], data['subfolder'])
        scroll.setWidget(container)
        ml.addWidget(scroll)
        
        btn_custom = QPushButton("+ Load Custom ONNX Model")
        btn_custom.clicked.connect(self.load_custom_model)
        ml.addWidget(btn_custom)
        tm_layout.addWidget(mod_grp)

        # Save/Load settings
        sl_layout = QHBoxLayout()
        btn_save_cfg = QPushButton("Save Config")
        btn_save_cfg.clicked.connect(self.save_config)
        btn_load_cfg = QPushButton("Load Config")
        btn_load_cfg.clicked.connect(self.load_config)
        sl_layout.addWidget(btn_save_cfg); sl_layout.addWidget(btn_load_cfg)
        tm_layout.addLayout(sl_layout)
        
        self.tabs.addTab(tab_models, "Inference Engine")

        # --- TAB 2: Negatives ---
        tab_neg = QWidget(); tn_layout = QVBoxLayout(tab_neg)
        tn_layout.addWidget(QLabel("Negative Substrings (One per line). Tags containing these will be dropped:"))
        self.txt_negatives = QTextEdit()
        tn_layout.addWidget(self.txt_negatives)
        
        n_sl_layout = QHBoxLayout()
        btn_save_neg = QPushButton("Save Negatives")
        btn_save_neg.clicked.connect(self.save_negatives)
        btn_load_neg = QPushButton("Load Negatives")
        btn_load_neg.clicked.connect(self.load_negatives)
        n_sl_layout.addWidget(btn_save_neg); n_sl_layout.addWidget(btn_load_neg)
        tn_layout.addLayout(n_sl_layout)
        
        self.tabs.addTab(tab_neg, "Negatives")

        # --- TAB 3: Agent Orchestrator ---
        tab_agent = QWidget(); ta_layout = QVBoxLayout(tab_agent)
        ag_grp = QGroupBox("Orchestrator Settings")
        agl = QVBoxLayout(ag_grp)
        
        ip_layout = QHBoxLayout()
        ip_layout.addWidget(QLabel("Bind IP:"))
        self.agent_ip = QLineEdit("127.0.0.1")
        ip_layout.addWidget(self.agent_ip)
        ip_layout.addWidget(QLabel("Port:"))
        self.agent_port = QSpinBox()
        self.agent_port.setRange(1024, 65535)
        self.agent_port.setValue(8005)
        ip_layout.addWidget(self.agent_port)
        agl.addLayout(ip_layout)
        
        self.btn_agent_toggle = QPushButton("START AGENT SERVER")
        self.btn_agent_toggle.setStyleSheet("background: #004d40; color: white; font-weight: bold; padding: 10px;")
        self.btn_agent_toggle.clicked.connect(self.toggle_agent)
        agl.addWidget(self.btn_agent_toggle)
        ta_layout.addWidget(ag_grp)
        
        ta_layout.addWidget(QLabel("Agent HTTP Logs:"))
        self.agent_logs = QTextEdit()
        self.agent_logs.setReadOnly(True)
        self.agent_logs.setStyleSheet("background: #000; color: #00ff00; font-family: Consolas; font-size: 11px;")
        ta_layout.addWidget(self.agent_logs)
        
        self.tabs.addTab(tab_agent, "Agent Node")

        layout.addWidget(self.tabs)

        # RIGHT PANE - BATCH PROCESSOR
        content = QWidget(); clayout = QVBoxLayout(content)
        
        dir_layout = QHBoxLayout()
        self.path_display = QLineEdit(); self.path_display.setReadOnly(True)
        btn_sel = QPushButton("Select Dataset Folder")
        btn_sel.clicked.connect(self.select_folder)
        self.chk_recursive = QCheckBox("Recursive Folder Scan")
        dir_layout.addWidget(self.path_display); dir_layout.addWidget(btn_sel)
        dir_layout.addWidget(self.chk_recursive)
        clayout.addLayout(dir_layout)

        self.btn_go = QPushButton("RUN LOCAL BATCH")
        self.btn_go.setFixedHeight(40)
        self.btn_go.setStyleSheet("background: #2e7d32; font-weight: bold; color: white;")
        self.btn_go.clicked.connect(self.start_batch_tagging)
        clayout.addWidget(self.btn_go)
        
        self.pbar = QProgressBar()
        clayout.addWidget(self.pbar)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Generated Tags"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setItemDelegateForColumn(1, WordWrapDelegate())
        self.table.itemSelectionChanged.connect(self.show_preview)
        clayout.addWidget(self.table)

        btn_save_disk = QPushButton("SAVE TAGS TO DISK (.txt)")
        btn_save_disk.setFixedHeight(40)
        btn_save_disk.clicked.connect(self.commit_to_disk)
        clayout.addWidget(btn_save_disk)

        layout.addWidget(content)

    def add_model_row(self, name, repo, subfolder, checked=False):
        grp = QGroupBox(name)
        gl = QHBoxLayout(grp)
        chk = QCheckBox("Use")
        chk.setChecked(checked or ("EVA02" in name)) 
        
        spn = QDoubleSpinBox()
        spn.setRange(0.0000, 1.0000); spn.setDecimals(4); spn.setSingleStep(0.005)
        spn.setValue(0.3500); spn.setFixedWidth(80)

        sld = QSlider(Qt.Horizontal); sld.setRange(0, 1000)
        sld.setValue(int(np.sqrt(0.3500) * 1000))

        sld.valueChanged.connect(lambda v, s=spn: s.setValue((v/1000)**2))
        spn.valueChanged.connect(lambda v, s=sld: s.blockSignals(True) or s.setValue(int(np.sqrt(v)*1000)) or s.blockSignals(False))

        gl.addWidget(chk); gl.addWidget(QLabel("Thresh:")); gl.addWidget(spn); gl.addWidget(sld)
        self.m_layout.addWidget(grp)
        self.model_refs[name] = {"chk": chk, "spn": spn, "repo": repo, "subfolder": subfolder}

    def load_custom_model(self):
        onnx_file, _ = QFileDialog.getOpenFileName(self, "Select Model (.onnx)", "", "ONNX Models (*.onnx)")
        if not onnx_file: return
        tags_file, _ = QFileDialog.getOpenFileName(self, "Select Tags (.json, .csv, .txt)", "", "Tag Files (*.json *.csv *.txt)")
        if not tags_file: return
        name_guess = os.path.basename(onnx_file).replace(".onnx", "")
        name, ok = QInputDialog.getText(self, "Model Name", "Enter a name for this model:", text=name_guess)
        if ok and name: self.add_model_row(name, onnx_file, tags_file, checked=True)

    def get_active_configs(self):
        return [{'repo': r['repo'], 'subfolder': r['subfolder'], 'thresh': r['spn'].value()} 
                for r in self.model_refs.values() if r['chk'].isChecked()]

    def get_active_categories(self):
        return [k for k, v in self.f_checks.items() if v.isChecked()]

    def get_negatives_list(self):
        return [line.strip() for line in self.txt_negatives.toPlainText().split("\n") if line.strip()]

    # Settings Serialization
    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Engine Config", "", "JSON Files (*.json)")
        if not path: return
        data = {
            "max_tags": self.spin_max_tags.value(),
            "categories": {k: v.isChecked() for k, v in self.f_checks.items()},
            "models": {k: {"use": v["chk"].isChecked(), "thresh": v["spn"].value()} for k, v in self.model_refs.items()}
        }
        with open(path, 'w') as f: json.dump(data, f, indent=4)
        QMessageBox.information(self, "Saved", "Config Saved Successfully.")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Engine Config", "", "JSON Files (*.json)")
        if not path: return
        with open(path, 'r') as f: data = json.load(f)
        if "max_tags" in data: self.spin_max_tags.setValue(data["max_tags"])
        if "categories" in data:
            for k, val in data["categories"].items():
                if k in self.f_checks: self.f_checks[k].setChecked(val)
        if "models" in data:
            for k, val in data["models"].items():
                if k in self.model_refs:
                    self.model_refs[k]["chk"].setChecked(val["use"])
                    self.model_refs[k]["spn"].setValue(val["thresh"])

    def save_negatives(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Negatives", "", "Text Files (*.txt)")
        if path:
            with open(path, 'w', encoding='utf-8') as f: f.write(self.txt_negatives.toPlainText())

    def load_negatives(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Negatives", "", "Text Files (*.txt)")
        if path:
            with open(path, 'r', encoding='utf-8') as f: self.txt_negatives.setPlainText(f.read())

    # UI Routing
    def select_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select Dataset")
        if p:
            self.dataset_path = p; self.path_display.setText(p)
            self.dataset_files = []
            
            if self.chk_recursive.isChecked():
                for root, dirs, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            # Generate path relative to dataset root for the table
                            rel_path = os.path.relpath(os.path.join(root, f), p)
                            self.dataset_files.append(rel_path)
            else:
                self.dataset_files = [f for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            self.dataset_files.sort()
            self.table.setRowCount(len(self.dataset_files))
            
            for i, f in enumerate(self.dataset_files):
                self.table.setItem(i, 0, QTableWidgetItem(f))
                self.table.item(i, 0).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                tp = os.path.join(p, os.path.splitext(f)[0] + ".txt")
                val = open(tp, 'r', encoding='utf-8').read() if os.path.exists(tp) else ""
                self.table.setItem(i, 1, QTableWidgetItem(val))
            self.table.resizeRowsToContents()

    def show_preview(self):
        rows = self.table.selectedItems()
        if rows:
            self.preview_window.display_image(os.path.join(self.dataset_path, self.table.item(rows[0].row(), 0).text()))
            if not self.preview_window.isVisible(): self.preview_window.show()

    # Batch Execution
    def start_batch_tagging(self):
        if not self.dataset_path: return
        configs = self.get_active_configs()
        if not configs: return
        
        self.btn_go.setEnabled(False)
        self.worker = TaggerWorker(self.dataset_path, self.dataset_files, configs, self.get_active_categories(), self.get_negatives_list(), self.spin_max_tags.value(), self.chk_mcut.isChecked())
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.diag.connect(self.log_agent)
        self.worker.finished.connect(self.finalize_batch)
        self.worker.start()

    def finalize_batch(self, results):
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
            with open(os.path.join(self.dataset_path, os.path.splitext(fn)[0] + ".txt"), 'w', encoding='utf-8') as f:
                f.write(it.text() if it else "")
        QMessageBox.information(self, "Saved", "All text files written to dataset folder.")

    # Agent Server Control
    def log_agent(self, msg):
        self.agent_logs.append(msg)

    def toggle_agent(self):
        if self.agent_thread and self.agent_thread.isRunning():
            self.agent_thread.stop()
            self.agent_thread.wait()
            self.btn_agent_toggle.setText("START AGENT SERVER")
            self.btn_agent_toggle.setStyleSheet("background: #004d40; color: white; font-weight: bold; padding: 10px;")
        else:
            configs = self.get_active_configs()
            if not configs:
                QMessageBox.warning(self, "No Models", "Please select at least one model in Inference Engine.")
                return
                
            self.log_agent("Initializing resident models for Orchestrator...")
            # Blocking load to ensure server doesn't start until models are hot
            self.shared_engine.load_models(configs) 
            
            self.agent_thread = AgentServerThread(
                self.agent_ip.text(), self.agent_port.value(),
                self.shared_engine, self.get_active_categories(),
                self.get_negatives_list(), self.spin_max_tags.value(),
                self.chk_mcut.isChecked()
            )
            self.agent_thread.log_msg.connect(self.log_agent)
            self.agent_thread.start()
            
            self.btn_agent_toggle.setText("STOP AGENT SERVER")
            self.btn_agent_toggle.setStyleSheet("background: #8b0000; color: white; font-weight: bold; padding: 10px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = App()
    window.show()
    sys.exit(app.exec())
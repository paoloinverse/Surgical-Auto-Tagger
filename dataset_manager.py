import sys
import os
import shutil
from collections import Counter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QLabel, QLineEdit, QTextEdit,
                             QMessageBox, QGroupBox, QStyledItemDelegate, 
                             QTreeView, QFileSystemModel, QSplitter, QTabWidget,
                             QAbstractItemView, QFrame, QComboBox, QCheckBox)
from PySide6.QtCore import Qt, QSize, QDir, QMimeData
from PySide6.QtGui import QPixmap, QTextDocument, QDrag

class WordWrapDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        option.displayAlignment = Qt.AlignTop | Qt.AlignLeft
        super().paint(painter, option, index)

    def sizeHint(self, option, index):
        text = str(index.model().data(index, Qt.DisplayRole))
        doc = QTextDocument()
        doc.setHtml(text)
        doc.setTextWidth(option.rect.width())
        return QSize(doc.idealWidth(), doc.size().height())

class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Preview")
        self.resize(500, 500)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("Select an image")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #050505; color: #666; font-family: 'Consolas';")
        self.layout.addWidget(self.label)
        self.current_pixmap = None

    def display_image(self, path):
        if os.path.exists(path):
            self.current_pixmap = QPixmap(path)
            self._update_label_pixmap()
        else:
            self.label.setText("File Not Found")

    def _update_label_pixmap(self):
        if self.current_pixmap:
            scaled = self.current_pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        if self.current_pixmap: self._update_label_pixmap()
        super().resizeEvent(event)

class DatasetManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Booru Dataset Canvas v2026.2")
        self.resize(1500, 950)
        self.root_dir = ""
        self.current_dir = ""
        self.preview_window = PreviewWindow()
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        self.splitter = QSplitter(Qt.Horizontal)

        # --- LEFT: NAVIGATOR ---
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        
        btn_open = QPushButton("Set Dataset Root")
        btn_open.setFixedHeight(35)
        btn_open.clicked.connect(self.select_root_folder)
        nav_layout.addWidget(btn_open)

        # Tree Navigation Controls
        nav_ctrls = QHBoxLayout()
        btn_root = QPushButton("🏠 Root")
        btn_root.clicked.connect(self.go_to_root)
        btn_up = QPushButton("⤴ Up")
        btn_up.clicked.connect(self.go_up_level)
        nav_ctrls.addWidget(btn_root)
        nav_ctrls.addWidget(btn_up)
        nav_layout.addLayout(nav_ctrls)

        self.tree_model = QFileSystemModel()
        self.tree_model.setReadOnly(False)
        self.tree_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setAcceptDrops(True)
        for i in range(1, 4): self.tree_view.hideColumn(i)
        self.tree_view.clicked.connect(self.on_tree_clicked)
        nav_layout.addWidget(self.tree_view)
        
        self.splitter.addWidget(nav_container)

        # --- RIGHT: WORKSPACE ---
        self.tabs = QTabWidget()
        
        # TAB 1: Editor
        editor_tab = QWidget()
        et_layout = QVBoxLayout(editor_tab)
        
        tools_grp = QGroupBox("Batch & Search Operations")
        tl = QVBoxLayout(tools_grp)
        
        # Row 1: Find/Replace logic
        r1 = QHBoxLayout()
        self.f_in = QLineEdit(); self.f_in.setPlaceholderText("Find tag (e.g. 1girl, solo)...")
        self.r_in = QLineEdit(); self.r_in.setPlaceholderText("Replace with...")
        btn_rep = QPushButton("Replace All"); btn_rep.clicked.connect(self.do_replace)
        r1.addWidget(QLabel("Find:")); r1.addWidget(self.f_in)
        r1.addWidget(QLabel("Replace:")); r1.addWidget(self.r_in); r1.addWidget(btn_rep)
        
        # Row 2: Prepend / Append (with dedicated input)
        r2 = QHBoxLayout()
        self.batch_in = QLineEdit(); self.batch_in.setPlaceholderText("Text to add...")
        btn_prepend = QPushButton("Prepend to All"); btn_prepend.clicked.connect(self.do_prepend)
        btn_append = QPushButton("Append to All"); btn_append.clicked.connect(self.do_append)
        r2.addWidget(QLabel("Add Text:")); r2.addWidget(self.batch_in)
        r2.addWidget(btn_prepend)
        r2.addWidget(btn_append)
        r2.addStretch()

        # Row 3: Scope & Organization
        r3 = QHBoxLayout()
        self.search_scope = QComboBox()
        self.search_scope.addItems(["Current Folder", "Descend (Subtree)", "Global Dataset"])
        
        self.cb_copy = QCheckBox("Copy files")
        self.cb_copy.setToolTip("If unchecked, files will be moved. If checked, files will be duplicated.")
        
        self.dest_in = QLineEdit()
        self.dest_in.setPlaceholderText("Dest folder (auto if empty)")
        
        btn_search = QPushButton("Search/Filter Table")
        btn_search.clicked.connect(self.perform_dry_run_search)
        btn_search.setToolTip("Filter the table to show matches for 'Find' tags across scope (Dry Run).")
        
        btn_organize = QPushButton("Organize Match")
        btn_organize.setStyleSheet("background-color: #d35400; color: white; font-weight: bold;")
        btn_organize.clicked.connect(self.organize_by_tags)
        
        r3.addWidget(QLabel("Scope:")); r3.addWidget(self.search_scope)
        r3.addWidget(self.cb_copy)
        r3.addWidget(QLabel("Target:")); r3.addWidget(self.dest_in)
        r3.addWidget(btn_search)
        r3.addWidget(btn_organize)
        
        tl.addLayout(r1)
        tl.addLayout(r2)
        tl.addLayout(r3)
        et_layout.addWidget(tools_grp)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Tags"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setItemDelegateForColumn(1, WordWrapDelegate())
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setDragEnabled(True)
        self.table.startDrag = self.initiate_drag
        self.table.itemSelectionChanged.connect(self.show_preview)
        et_layout.addWidget(self.table)

        btn_save = QPushButton("SAVE CHANGES IN THIS FOLDER")
        btn_save.setFixedHeight(40)
        btn_save.setStyleSheet("font-weight: bold; background-color: #2c3e50; color: white;")
        btn_save.clicked.connect(self.commit_to_disk)
        et_layout.addWidget(btn_save)
        
        self.tabs.addTab(editor_tab, "Tag Editor")

        # TAB 2: Analysis
        analysis_tab = QWidget()
        at_layout = QVBoxLayout(analysis_tab)
        
        ctrl_frame = QFrame()
        ctrl_layout = QHBoxLayout(ctrl_frame)
        btn_analyze = QPushButton("Scan Entire Dataset")
        btn_analyze.clicked.connect(self.run_global_analysis)
        self.analysis_filter = QLineEdit()
        self.analysis_filter.setPlaceholderText("Filter tags...")
        self.analysis_filter.textChanged.connect(self.filter_analysis_table)
        ctrl_layout.addWidget(btn_analyze)
        ctrl_layout.addWidget(self.analysis_filter)
        at_layout.addWidget(ctrl_frame)

        self.analysis_table = QTableWidget(0, 2)
        self.analysis_table.setHorizontalHeaderLabels(["Tag", "Frequency"])
        self.analysis_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.analysis_table.setSortingEnabled(True)
        self.analysis_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.analysis_table.doubleClicked.connect(self.on_analysis_double_click)
        at_layout.addWidget(self.analysis_table)
        
        self.analysis_stats = QLabel("No analysis performed.")
        at_layout.addWidget(self.analysis_stats)
        
        self.tabs.addTab(analysis_tab, "Global Analysis")

        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(1, 4)
        main_layout.addWidget(self.splitter)

    # --- FILE OPERATIONS ---
    def select_root_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Root")
        if path:
            self.root_dir = path
            self.tree_model.setRootPath(path)
            self.tree_view.setRootIndex(self.tree_model.index(path))
            self.load_folder_contents(path)

    def go_to_root(self):
        if self.root_dir:
            self.tree_view.setRootIndex(self.tree_model.index(self.root_dir))
            self.load_folder_contents(self.root_dir)

    def go_up_level(self):
        if not self.current_dir: return
        parent = os.path.dirname(self.current_dir)
        # Don't go above the root if root is set
        if self.root_dir and not parent.startswith(self.root_dir) and parent != self.root_dir:
            return
        self.load_folder_contents(parent)

    def on_tree_clicked(self, index):
        path = self.tree_model.filePath(index)
        if os.path.isdir(path):
            if self.current_dir and path != self.current_dir:
                try:
                    rel = os.path.relpath(path, self.current_dir)
                    if not rel.startswith(".."):
                        self.dest_in.setText(rel)
                    else:
                        self.dest_in.setText(path)
                except ValueError:
                    self.dest_in.setText(path)
            self.load_folder_contents(path)

    def load_folder_contents(self, path):
        self.current_dir = path
        try:
            files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            self.table.setRowCount(len(files))
            for i, f in enumerate(files):
                self.table.setItem(i, 0, QTableWidgetItem(f))
                tp = os.path.join(path, os.path.splitext(f)[0] + ".txt")
                val = ""
                if os.path.exists(tp):
                    with open(tp, 'r', encoding='utf-8', errors='ignore') as fh: val = fh.read()
                self.table.setItem(i, 1, QTableWidgetItem(val))
            self.table.resizeRowsToContents()
        except Exception as e:
            print(f"Error loading folder: {e}")

    def initiate_drag(self, supportedActions):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows: return
        filenames = [self.table.item(r.row(), 0).text() for r in selected_rows]
        mime_data = QMimeData()
        mime_data.setText(",".join(filenames))
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        if drag.exec(Qt.MoveAction) == Qt.MoveAction:
            self.load_folder_contents(self.current_dir)

    def dragEnterEvent(self, event): event.accept()

    def dropEvent(self, event):
        pos = event.position().toPoint()
        index = self.tree_view.indexAt(self.tree_view.viewport().mapFromParent(pos))
        if not index.isValid(): return
        target_dir = self.tree_model.filePath(index)
        if not os.path.isdir(target_dir): return
        
        filenames = event.mimeData().text().split(",")
        moved = 0
        for f in filenames:
            src_img = os.path.join(self.current_dir, f)
            src_txt = os.path.join(self.current_dir, os.path.splitext(f)[0] + ".txt")
            try:
                shutil.move(src_img, os.path.join(target_dir, f))
                if os.path.exists(src_txt):
                    shutil.move(src_txt, os.path.join(target_dir, os.path.basename(src_txt)))
                moved += 1
            except Exception: pass
        event.setDropAction(Qt.MoveAction)
        event.accept()
        self.load_folder_contents(self.current_dir)

    # --- TAG ANALYSIS & SEARCH ---
    def on_analysis_double_click(self, index):
        new_tag = self.analysis_table.item(index.row(), 0).text().strip()
        current_text = self.f_in.text().strip()
        if not current_text:
            self.f_in.setText(new_tag)
        else:
            existing_tags = [t.strip() for t in current_text.split(",") if t.strip()]
            if new_tag not in existing_tags:
                existing_tags.append(new_tag)
                self.f_in.setText(", ".join(existing_tags))
        self.tabs.setCurrentIndex(0)

    def perform_dry_run_search(self):
        target_tag = self.f_in.text().strip()
        if not target_tag:
            self.load_folder_contents(self.current_dir)
            return

        query_tags = [t.strip().lower() for t in target_tag.split(",") if t.strip()]
        scope = self.search_scope.currentText()
        scan_path = self.current_dir
        recursive = False
        if scope == "Global Dataset":
            scan_path = self.root_dir if self.root_dir else self.current_dir
            recursive = True
        elif scope == "Descend (Subtree)":
            recursive = True

        matches = []
        for root, _, files in (os.walk(scan_path) if recursive else [(scan_path, [], os.listdir(scan_path))]):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    txt_path = os.path.join(root, os.path.splitext(f)[0] + ".txt")
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as fh:
                                content = fh.read()
                                file_tags = [t.strip().lower() for t in content.split(",") if t.strip()]
                                if all(qt in file_tags for qt in query_tags):
                                    matches.append((root, f, content))
                        except: pass

        self.table.setRowCount(len(matches))
        for i, (root, f, tags) in enumerate(matches):
            # Store full path in tooltip or similar if needed, here we just show name
            self.table.setItem(i, 0, QTableWidgetItem(f))
            # Tag the item data with root path for batch editing support
            self.table.item(i, 0).setData(Qt.UserRole, root) 
            self.table.setItem(i, 1, QTableWidgetItem(tags))
        self.table.resizeRowsToContents()

    def organize_by_tags(self):
        target_tag = self.f_in.text().strip()
        if not target_tag:
            QMessageBox.warning(self, "Input Required", "Enter a tag in 'Find' to organize by.")
            return
        if not self.current_dir: return

        is_copy = self.cb_copy.isChecked()
        op_label = "Copy" if is_copy else "Move"
        query_tags = [t.strip().lower() for t in target_tag.split(",") if t.strip()]
        
        scope = self.search_scope.currentText()
        scan_path = self.current_dir
        recursive = scope != "Current Folder"
        if scope == "Global Dataset": scan_path = self.root_dir if self.root_dir else self.current_dir

        manual_dest = self.dest_in.text().strip()
        dest_folder_name = manual_dest if manual_dest else target_tag.replace(",", "_").replace(" ", "_")
        dest_path = dest_folder_name if os.path.isabs(dest_folder_name) else os.path.join(self.current_dir, dest_folder_name)
        
        matches = []
        for root, _, files in (os.walk(scan_path) if recursive else [(scan_path, [], os.listdir(scan_path))]):
            if os.path.abspath(root) == os.path.abspath(dest_path): continue
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    txt_path = os.path.join(root, os.path.splitext(f)[0] + ".txt")
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as fh:
                                file_tags = [t.strip().lower() for t in fh.read().split(",") if t.strip()]
                                if all(qt in file_tags for qt in query_tags):
                                    matches.append((root, f))
                        except: pass

        if not matches:
            QMessageBox.information(self, "No Matches", f"No items found for tags: {target_tag}")
            return

        if QMessageBox.Yes == QMessageBox.question(self, "Confirm", f"Found {len(matches)} pairs. {op_label} to '{os.path.basename(dest_path)}'?", QMessageBox.Yes|QMessageBox.No):
            if not os.path.exists(dest_path): os.makedirs(dest_path)
            count = 0
            for root, f in matches:
                src_img = os.path.join(root, f)
                src_txt = os.path.join(root, os.path.splitext(f)[0] + ".txt")
                try:
                    target_img = os.path.join(dest_path, f)
                    target_txt = os.path.join(dest_path, os.path.basename(src_txt))
                    if is_copy:
                        shutil.copy2(src_img, target_img)
                        shutil.copy2(src_txt, target_txt)
                    else:
                        shutil.move(src_img, target_img)
                        shutil.move(src_txt, target_txt)
                    count += 1
                except Exception as e: print(f"Error: {e}")
            QMessageBox.information(self, "Complete", f"{op_label}ed {count} pairs.")
            self.load_folder_contents(self.current_dir)

    def run_global_analysis(self):
        if not self.root_dir: return
        tag_counts = Counter()
        file_count = 0
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(".txt"):
                    file_count += 1
                    try:
                        with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as fh:
                            tag_counts.update([t.strip() for t in fh.read().split(",") if t.strip()])
                    except: pass
        self.analysis_table.setSortingEnabled(False)
        self.analysis_table.setRowCount(len(tag_counts))
        for i, (tag, count) in enumerate(tag_counts.items()):
            self.analysis_table.setItem(i, 0, QTableWidgetItem(tag))
            count_item = QTableWidgetItem()
            count_item.setData(Qt.EditRole, count)
            self.analysis_table.setItem(i, 1, count_item)
        self.analysis_table.setSortingEnabled(True)
        self.analysis_table.sortByColumn(1, Qt.DescendingOrder)
        self.analysis_stats.setText(f"Files: {file_count} | Unique Tags: {len(tag_counts)}")

    def filter_analysis_table(self):
        txt = self.analysis_filter.text().lower()
        for i in range(self.analysis_table.rowCount()):
            self.analysis_table.setRowHidden(i, txt not in self.analysis_table.item(i, 0).text().lower())

    # --- BATCH TAG UTILS ---
    def do_replace(self):
        f, r = self.f_in.text().strip(), self.r_in.text().strip()
        if not f: return
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 1)
            tags = [t.strip() for t in it.text().split(",") if t.strip()]
            new_tags = [r if t == f else t for t in tags]
            it.setText(", ".join([t for t in new_tags if t]))
        self.table.resizeRowsToContents()

    def do_prepend(self):
        val = self.batch_in.text().strip()
        if not val: return
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 1)
            cur = it.text().strip()
            it.setText(f"{val}, {cur}" if cur else val)
        self.table.resizeRowsToContents()

    def do_append(self):
        val = self.batch_in.text().strip()
        if not val: return
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 1)
            cur = it.text().strip()
            if cur:
                it.setText(f"{cur}, {val}" if not cur.endswith(",") else f"{cur} {val}")
            else:
                it.setText(val)
        self.table.resizeRowsToContents()

    def show_preview(self):
        rows = self.table.selectedItems()
        if rows:
            fn = self.table.item(rows[0].row(), 0).text()
            # Check if we are in a search result (UserRole has the root)
            folder = self.table.item(rows[0].row(), 0).data(Qt.UserRole) or self.current_dir
            self.preview_window.display_image(os.path.join(folder, fn))
            if not self.preview_window.isVisible(): self.preview_window.show()

    def commit_to_disk(self):
        if not self.table.rowCount(): return
        for r in range(self.table.rowCount()):
            fn = self.table.item(r, 0).text()
            txt = self.table.item(r, 1).text()
            folder = self.table.item(r, 0).data(Qt.UserRole) or self.current_dir
            if not folder: continue
            with open(os.path.join(folder, os.path.splitext(fn)[0] + ".txt"), 'w', encoding='utf-8') as f:
                f.write(txt)
        QMessageBox.information(self, "Saved", "All modifications in current table committed to disk.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DatasetManager()
    window.show()
    sys.exit(app.exec())










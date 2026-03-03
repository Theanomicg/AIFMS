from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .classifier import ModelBundle, load_model, save_model, train_model
from .pipeline import (
    OrganizationAssignment,
    PipelineSummary,
    PreviewItem,
    apply_organization_assignments,
    preview_classification_pipeline,
    run_organization_pipeline,
)
from .training import load_training_dataset


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class AIFMSWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AIFMS Desktop")
        self.resize(1150, 760)
        self.thread_pool = QThreadPool()
        self.model_bundle: ModelBundle | None = None
        self.known_labels: list[str] = ["uncategorized"]
        self.preview_items: list[PreviewItem] = []
        self.running_tasks = 0

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.model_tab = self._build_model_tab()
        self.organize_tab = self._build_organize_tab()
        self.review_tab = self._build_review_tab()

        self.tabs.addTab(self.model_tab, "Model")
        self.tabs.addTab(self.organize_tab, "Organize")
        self.tabs.addTab(self.review_tab, "Review")

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def _build_model_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        model_box = QGroupBox("Train / Load Model")
        form = QFormLayout(model_box)

        self.dataset_edit = QLineEdit()
        self.model_out_edit = QLineEdit(str(Path("models") / "aifms_model.joblib"))
        self.min_conf_spin = QDoubleSpinBox()
        self.min_conf_spin.setRange(0.0, 1.0)
        self.min_conf_spin.setSingleStep(0.05)
        self.min_conf_spin.setValue(0.40)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_edit)
        ds_browse = QPushButton("Browse")
        ds_browse.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(ds_browse)

        model_row = QHBoxLayout()
        model_row.addWidget(self.model_out_edit)
        model_browse = QPushButton("Browse")
        model_browse.clicked.connect(self._browse_model_output)
        model_row.addWidget(model_browse)

        form.addRow("Training CSV", dataset_row)
        form.addRow("Model Path", model_row)
        form.addRow("Min Confidence", self.min_conf_spin)

        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("Train And Save")
        self.train_btn.clicked.connect(self._on_train_model)
        self.load_btn = QPushButton("Load Existing Model")
        self.load_btn.clicked.connect(self._on_load_model)
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(self.load_btn)

        self.model_info = QPlainTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setPlaceholderText("Model details and status will appear here.")

        layout.addWidget(model_box)
        layout.addLayout(btn_row)
        layout.addWidget(self.model_info)
        return tab

    def _build_organize_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form_box = QGroupBox("Organization Settings")
        form = QFormLayout(form_box)

        self.input_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit(str(Path("organized_output")))
        self.model_path_edit = QLineEdit(str(Path("models") / "aifms_model.joblib"))
        self.log_path_edit = QLineEdit(str(Path("logs") / "run_log.csv"))
        self.recursive_check = QCheckBox("Scan recursively")
        self.recursive_check.setChecked(True)
        self.copy_only_check = QCheckBox("Copy only (do not move originals)")
        self.copy_only_check.setChecked(True)

        in_row = QHBoxLayout()
        in_row.addWidget(self.input_dir_edit)
        in_btn = QPushButton("Browse")
        in_btn.clicked.connect(lambda: self._browse_dir(self.input_dir_edit))
        in_row.addWidget(in_btn)

        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir_edit)
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        out_row.addWidget(out_btn)

        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit)
        model_btn = QPushButton("Browse")
        model_btn.clicked.connect(self._browse_model_path)
        model_row.addWidget(model_btn)

        log_row = QHBoxLayout()
        log_row.addWidget(self.log_path_edit)
        log_btn = QPushButton("Browse")
        log_btn.clicked.connect(self._browse_log_path)
        log_row.addWidget(log_btn)

        form.addRow("Input Directory", in_row)
        form.addRow("Output Directory", out_row)
        form.addRow("Model Path", model_row)
        form.addRow("Log Path", log_row)
        form.addRow("", self.recursive_check)
        form.addRow("", self.copy_only_check)

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview To Review Tab")
        self.preview_btn.clicked.connect(self._on_preview)
        self.organize_btn = QPushButton("Run Direct Organize")
        self.organize_btn.clicked.connect(self._on_organize_direct)
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.organize_btn)

        self.organize_info = QPlainTextEdit()
        self.organize_info.setReadOnly(True)
        self.organize_info.setPlaceholderText("Pipeline run logs will appear here.")

        layout.addWidget(form_box)
        layout.addLayout(btn_row)
        layout.addWidget(self.organize_info)
        return tab

    def _build_review_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel("Review predicted categories and edit the final category before applying file operations.")
        layout.addWidget(info)

        self.review_table = QTableWidget(0, 5)
        self.review_table.setHorizontalHeaderLabels(
            ["File", "Predicted", "Confidence", "Extraction", "Final Category"]
        )
        self.review_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.review_table)

        btn_row = QHBoxLayout()
        self.apply_review_btn = QPushButton("Apply Reviewed Organization")
        self.apply_review_btn.clicked.connect(self._on_apply_review)
        btn_row.addWidget(self.apply_review_btn)
        layout.addLayout(btn_row)
        return tab

    def _browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Training CSV", "", "CSV Files (*.csv)")
        if path:
            self.dataset_edit.setText(path)

    def _browse_model_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", self.model_out_edit.text(), "Joblib (*.joblib)")
        if path:
            self.model_out_edit.setText(path)
            self.model_path_edit.setText(path)

    def _browse_model_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", self.model_path_edit.text(), "Joblib (*.joblib)")
        if path:
            self.model_path_edit.setText(path)

    def _browse_log_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select Log CSV", self.log_path_edit.text(), "CSV Files (*.csv)")
        if path:
            self.log_path_edit.setText(path)

    def _browse_dir(self, target: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            target.setText(path)

    def _set_busy(self, busy: bool) -> None:
        self.train_btn.setEnabled(not busy)
        self.load_btn.setEnabled(not busy)
        self.preview_btn.setEnabled(not busy)
        self.organize_btn.setEnabled(not busy)
        self.apply_review_btn.setEnabled(not busy)
        self.status_label.setText("Working..." if busy else "Ready")

    def _start_worker(self, fn, on_result, on_error=None) -> None:
        worker = Worker(fn)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(on_error or self._show_error)
        worker.signals.finished.connect(self._on_worker_finished)
        self.running_tasks += 1
        self._set_busy(True)
        self.thread_pool.start(worker)

    def _on_worker_finished(self) -> None:
        self.running_tasks = max(0, self.running_tasks - 1)
        if self.running_tasks == 0:
            self._set_busy(False)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "AIFMS Error", message)

    def _resolve_model_bundle(self) -> ModelBundle:
        model_path = Path(self.model_path_edit.text().strip())
        if self.model_bundle is not None and self.model_out_edit.text().strip() == str(model_path):
            return self.model_bundle
        self.model_bundle = load_model(model_path)
        classes = list(getattr(self.model_bundle.pipeline, "classes_", []))
        self.known_labels = sorted(set(classes + ["uncategorized"]))
        return self.model_bundle

    def _on_train_model(self) -> None:
        dataset = Path(self.dataset_edit.text().strip())
        model_path = Path(self.model_out_edit.text().strip())
        min_conf = float(self.min_conf_spin.value())

        if not dataset.exists():
            self._show_error("Training CSV not found.")
            return
        if not model_path:
            self._show_error("Model path is required.")
            return

        def task():
            texts, labels = load_training_dataset(dataset)
            bundle = train_model(texts, labels, min_confidence=min_conf)
            save_model(bundle, model_path)
            classes = list(getattr(bundle.pipeline, "classes_", []))
            return bundle, len(texts), classes, model_path

        def on_result(result):
            bundle, sample_count, classes, saved_path = result
            self.model_bundle = bundle
            self.model_path_edit.setText(str(saved_path))
            self.known_labels = sorted(set(classes + ["uncategorized"]))
            self.model_info.appendPlainText(
                f"Model trained: {saved_path}\nSamples: {sample_count}\nClasses: {', '.join(classes)}\n"
            )

        self._start_worker(task, on_result)

    def _on_load_model(self) -> None:
        model_path = Path(self.model_out_edit.text().strip())
        if not model_path.exists():
            self._show_error("Model file not found.")
            return

        def task():
            bundle = load_model(model_path)
            classes = list(getattr(bundle.pipeline, "classes_", []))
            return bundle, classes, model_path

        def on_result(result):
            bundle, classes, loaded_path = result
            self.model_bundle = bundle
            self.model_path_edit.setText(str(loaded_path))
            self.known_labels = sorted(set(classes + ["uncategorized"]))
            self.model_info.appendPlainText(f"Model loaded: {loaded_path}\nClasses: {', '.join(classes)}\n")

        self._start_worker(task, on_result)

    def _on_preview(self) -> None:
        input_dir = Path(self.input_dir_edit.text().strip())
        if not input_dir.exists() or not input_dir.is_dir():
            self._show_error("Input directory is invalid.")
            return

        recursive = bool(self.recursive_check.isChecked())

        def task():
            bundle = self._resolve_model_bundle()
            return preview_classification_pipeline(input_dir=input_dir, model_bundle=bundle, recursive=recursive)

        def on_result(items):
            self.preview_items = list(items)
            self._populate_review_table(self.preview_items)
            self.organize_info.appendPlainText(f"Preview completed: {len(self.preview_items)} file(s).")
            self.tabs.setCurrentWidget(self.review_tab)

        self._start_worker(task, on_result)

    def _on_organize_direct(self) -> None:
        input_dir = Path(self.input_dir_edit.text().strip())
        output_dir = Path(self.output_dir_edit.text().strip())
        log_path = Path(self.log_path_edit.text().strip())
        recursive = bool(self.recursive_check.isChecked())
        copy_only = bool(self.copy_only_check.isChecked())

        if not input_dir.exists() or not input_dir.is_dir():
            self._show_error("Input directory is invalid.")
            return
        if not output_dir:
            self._show_error("Output directory is required.")
            return
        if not log_path:
            self._show_error("Log path is required.")
            return

        def task():
            bundle = self._resolve_model_bundle()
            return run_organization_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                model_bundle=bundle,
                log_path=log_path,
                recursive=recursive,
                copy_only=copy_only,
            )

        def on_result(summary):
            s: PipelineSummary = summary
            self.organize_info.appendPlainText(
                "Direct organize done:"
                f"\n- scanned={s.total_files_scanned}"
                f"\n- classified={s.successfully_classified}"
                f"\n- unclassified={s.unclassified}"
                f"\n- seconds={s.total_seconds:.3f}\n"
            )

        self._start_worker(task, on_result)

    def _populate_review_table(self, items: list[PreviewItem]) -> None:
        self.review_table.setRowCount(len(items))
        for row, item in enumerate(items):
            self.review_table.setItem(row, 0, QTableWidgetItem(str(item.source_path)))
            self.review_table.setItem(row, 1, QTableWidgetItem(item.predicted_label))
            self.review_table.setItem(row, 2, QTableWidgetItem(f"{item.confidence:.4f}"))
            self.review_table.setItem(row, 3, QTableWidgetItem("yes" if item.extraction_ok else "no"))

            combo = QComboBox()
            combo.setEditable(True)
            for label in self.known_labels:
                combo.addItem(label)
            if combo.findText(item.predicted_label) < 0:
                combo.addItem(item.predicted_label)
            combo.setCurrentText(item.predicted_label)
            self.review_table.setCellWidget(row, 4, combo)

        self.review_table.resizeColumnsToContents()

    def _on_apply_review(self) -> None:
        if not self.preview_items:
            self._show_error("No preview results available. Run preview first.")
            return

        output_dir = Path(self.output_dir_edit.text().strip())
        log_path = Path(self.log_path_edit.text().strip())
        copy_only = bool(self.copy_only_check.isChecked())
        if not output_dir:
            self._show_error("Output directory is required.")
            return
        if not log_path:
            self._show_error("Log path is required.")
            return

        assignments: list[OrganizationAssignment] = []
        for row, item in enumerate(self.preview_items):
            widget = self.review_table.cellWidget(row, 4)
            if not isinstance(widget, QComboBox):
                final_label = item.predicted_label
            else:
                final_label = widget.currentText().strip() or "uncategorized"
            assignments.append(
                OrganizationAssignment(
                    source_path=item.source_path,
                    category_label=final_label,
                    confidence=item.confidence,
                    extraction_ok=item.extraction_ok,
                )
            )

        def task():
            return apply_organization_assignments(
                assignments=assignments,
                output_dir=output_dir,
                log_path=log_path,
                copy_only=copy_only,
            )

        def on_result(summary):
            s: PipelineSummary = summary
            self.organize_info.appendPlainText(
                "Reviewed organize done:"
                f"\n- scanned={s.total_files_scanned}"
                f"\n- classified={s.successfully_classified}"
                f"\n- unclassified={s.unclassified}"
                f"\n- seconds={s.total_seconds:.3f}\n"
            )
            QMessageBox.information(self, "AIFMS", "Reviewed organization completed.")

        self._start_worker(task, on_result)


def main() -> int:
    app = QApplication([])
    window = AIFMSWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

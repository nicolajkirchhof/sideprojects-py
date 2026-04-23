"""
finance.apps.assistant._error_dialog
======================================
Scrollable error dialog for pipeline failures.

Displays exception type, message, and full traceback so the user can
diagnose the problem and fix the underlying issue before re-running.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets


class ErrorDialog(QtWidgets.QDialog):
    """
    Modal dialog showing a pipeline error with full traceback.

    Parameters
    ----------
    parent:
        Parent widget (pass the main window so the dialog is centred on it).
    message:
        Pre-formatted error string: exception type + message + traceback.
        Produced by the pipeline thread's ``error`` signal.
    """

    def __init__(self, parent: QtWidgets.QWidget | None, message: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pipeline Error")
        self.setMinimumSize(700, 400)
        self.resize(800, 500)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QtWidgets.QLabel("Pipeline stopped — resolve the issue below and re-run.")
        header.setStyleSheet("color: #e87; font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlainText(message)
        self._text.setStyleSheet(
            "QPlainTextEdit { background: #0d0d0d; color: #f88; font-family: monospace;"
            " font-size: 11px; border: 1px solid #333; }"
        )
        layout.addWidget(self._text, 1)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)


def show_pipeline_error(parent: QtWidgets.QWidget | None, message: str) -> None:
    """Convenience wrapper — create, show, and exec the error dialog."""
    dialog = ErrorDialog(parent, message)
    dialog.exec()

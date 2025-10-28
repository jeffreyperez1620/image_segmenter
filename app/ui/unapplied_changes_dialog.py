from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
)
from PySide6.QtCore import Qt


class UnappliedChangesDialog(QDialog):
    """Dialog for handling unapplied changes when switching tabs or closing the application."""
    
    def __init__(self, parent=None, message: str = None):
        super().__init__(parent)
        self.setWindowTitle("Unapplied Changes")
        self.setModal(True)
        self.resize(400, 200)
        
        # Default message if none provided
        if message is None:
            message = (
                "There are unapplied changes that will be lost if you continue.\n\n"
                "What would you like to do?"
            )
        
        self._setup_ui(message)
    
    def _setup_ui(self, message: str) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Message label
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(message_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Apply Changes button
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setStyleSheet(
            "QPushButton { "
            "background-color: #4CAF50; "
            "color: white; "
            "font-weight: bold; "
            "padding: 8px; "
            "border: none; "
            "border-radius: 4px; "
            "}"
        )
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        button_layout.addWidget(self.apply_btn)
        
        # Discard Changes button
        self.discard_btn = QPushButton("Discard Changes")
        self.discard_btn.setStyleSheet(
            "QPushButton { "
            "background-color: #f44336; "
            "color: white; "
            "font-weight: bold; "
            "padding: 8px; "
            "border: none; "
            "border-radius: 4px; "
            "}"
        )
        self.discard_btn.clicked.connect(self._on_discard_clicked)
        button_layout.addWidget(self.discard_btn)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(
            "QPushButton { "
            "background-color: #9E9E9E; "
            "color: white; "
            "font-weight: bold; "
            "padding: 8px; "
            "border: none; "
            "border-radius: 4px; "
            "}"
        )
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _on_apply_clicked(self) -> None:
        """Handle Apply Changes button click."""
        self.done(1)  # Apply Changes
    
    def _on_discard_clicked(self) -> None:
        """Handle Discard Changes button click."""
        self.done(2)  # Discard Changes
    
    def _on_cancel_clicked(self) -> None:
        """Handle Cancel button click."""
        self.done(0)  # Cancel
    
    @staticmethod
    def show_dialog(parent=None, message: str = None) -> str:
        """
        Show the unapplied changes dialog and return the user's choice.
        
        Args:
            parent: Parent widget for the dialog
            message: Custom message to display
            
        Returns:
            str: "apply", "discard", or "cancel"
        """
        dialog = UnappliedChangesDialog(parent, message)
        result = dialog.exec()
        
        if result == 1:
            return "apply"
        elif result == 2:
            return "discard"
        else:
            return "cancel"

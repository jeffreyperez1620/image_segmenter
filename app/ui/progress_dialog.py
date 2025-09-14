"""
Progress dialog for long-running operations.
"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PySide6.QtCore import Qt, QTimer


class ProgressDialog(QDialog):
    """A modal progress dialog with cancel functionality."""
    
    def __init__(self, title: str = "Processing...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 150)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        
        # Track cancellation
        self._cancelled = False
    
    def update_progress(self, current: int, total: int, message: str = ""):
        """Update the progress bar and status message."""
        if self._cancelled:
            return
        
        # Update progress bar
        self.progress_bar.setValue(current)
        
        # Update status message
        if message:
            self.status_label.setText(message)
        
        # Process events to keep UI responsive
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
    
    def set_cancelled(self, cancelled: bool = True):
        """Mark the operation as cancelled."""
        self._cancelled = cancelled
    
    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        return self._cancelled
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        self._cancelled = True
        super().closeEvent(event)


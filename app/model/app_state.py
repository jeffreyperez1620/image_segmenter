from __future__ import annotations

from typing import Optional
import numpy as np
from PySide6.QtCore import Signal, QObject


class AppState(QObject):
    """Singleton class to manage application state including base and working images."""
    
    # Singleton instance
    _instance: Optional[AppState] = None
    
    # Signals for state changes
    base_image_changed = Signal()
    working_image_changed = Signal()
    
    def __new__(cls) -> AppState:
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the application state."""
        if hasattr(self, '_initialized'):
            return  # Avoid re-initialization
        
        super().__init__()
        self._base_image: Optional[np.ndarray] = None
        self._working_image: Optional[np.ndarray] = None
        self._initialized = True
    
    @property
    def base_image(self) -> Optional[np.ndarray]:
        """Get the base image."""
        return self._base_image.copy() if self._base_image is not None else None
    
    @base_image.setter
    def base_image(self, image: Optional[np.ndarray]) -> None:
        """Set the base image and emit signal."""
        self._base_image = image.copy() if image is not None else None
        self.base_image_changed.emit()
    
    @property
    def working_image(self) -> Optional[np.ndarray]:
        """Get the working image."""
        return self._working_image.copy() if self._working_image is not None else None
    
    @working_image.setter
    def working_image(self, image: Optional[np.ndarray]) -> None:
        """Set the working image and emit signal."""
        self._working_image = image.copy() if image is not None else None
        self.working_image_changed.emit()
    
    def reset_working_image(self) -> None:
        """Reset the working image to the base image."""
        if self._base_image is not None:
            self.working_image = self._base_image
    
    def apply_working_to_base(self) -> None:
        """Apply the working image as the new base image."""
        if self._working_image is not None:
            self.base_image = self._working_image
    
    def has_base_image(self) -> bool:
        """Check if base image exists."""
        return self._base_image is not None
    
    def has_working_image(self) -> bool:
        """Check if working image exists."""
        return self._working_image is not None
    
    def clear_all(self) -> None:
        """Clear both base and working images."""
        self._base_image = None
        self._working_image = None
        self.base_image_changed.emit()
        self.working_image_changed.emit()

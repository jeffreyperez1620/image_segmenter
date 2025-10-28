"""
Custom tab widget and tab bar classes that validate tab changes before allowing them.
"""

from PySide6.QtWidgets import QTabWidget, QTabBar
from PySide6.QtCore import Signal


class ValidatingTabBar(QTabBar):
	"""Custom tab bar that validates tab changes before allowing them."""
	
	def __init__(self, parent=None):
		super().__init__(parent)
		self._validation_callback = None
	
	def set_validation_callback(self, callback):
		"""Set the callback function to validate tab changes."""
		self._validation_callback = callback
	
	def mousePressEvent(self, event):
		"""Override mouse press to validate tab changes before they happen."""
		# Check if clicking on a tab
		tab_index = self.tabAt(event.pos())
		
		if (tab_index >= 0 
			and tab_index != self.currentIndex() 
			and self._validation_callback 
			and not self._validation_callback(self.currentIndex(), tab_index)):
			return  # Don't call super() - prevent the tab change
		
		super().mousePressEvent(event)


class ValidatingTabWidget(QTabWidget):
	"""Custom tab widget that uses a validating tab bar."""
	
	def __init__(self, parent=None):
		super().__init__(parent)
		# Replace the default tab bar with our custom one
		self._custom_tab_bar = ValidatingTabBar(self)
		self.setTabBar(self._custom_tab_bar)
	
	def set_validation_callback(self, callback):
		"""Set the callback function to validate tab changes."""
		self._custom_tab_bar.set_validation_callback(callback)

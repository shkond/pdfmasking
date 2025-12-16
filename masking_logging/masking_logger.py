"""Masking Logger - Centralized logging for PII masking operations.

This module provides a logger that can write to file and supports
dynamic log file switching for batch processing.
"""

import logging
from pathlib import Path


class MaskingLogger:
    """
    Logger for PII masking operations.
    
    Supports:
    - Writing masked entity logs to file
    - Dynamic log file switching for batch processing
    - Configurable log format
    """

    _instance = None
    _logger = None

    def __new__(cls):
        """Singleton pattern for shared logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._logger = logging.getLogger("masking")
            cls._logger.setLevel(logging.INFO)
        return cls._instance

    @property
    def logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self._logger

    def setup_file_handler(self, log_file_path: Path) -> None:
        """
        Set up the logger to write to the specified file.
        Removes existing handlers to switch log files dynamically.
        
        Args:
            log_file_path: Path to the log file
        """
        # Remove existing handlers
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        # Add new handler
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(file_handler)

    def log(self, message: str) -> None:
        """
        Log a message.
        
        Args:
            message: Message to log
        """
        self._logger.info(message)

    def info(self, message: str) -> None:
        """Alias for log() - log an info message."""
        self.log(message)

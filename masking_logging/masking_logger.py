"""Masking Logger - Centralized logging for PII masking operations.

This module provides a logger that can write to file and supports
dynamic log file switching for batch processing.

NOTE: Singleton pattern removed for testability. Use dependency injection
to share logger instances across components.
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
    
    Usage:
        logger = MaskingLogger()
        logger.setup_file_handler(Path("output/log.txt"))
        logger.log("Message")
    """

    def __init__(self, name: str = "masking"):
        """Initialize logger with a unique name.
        
        Args:
            name: Logger name (default: "masking")
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        # Prevent duplicate handlers if logger already exists
        if not self._logger.handlers:
            # Add null handler to prevent "No handlers found" warning
            self._logger.addHandler(logging.NullHandler())

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
        # Remove existing handlers (except NullHandler)
        for handler in self._logger.handlers[:]:
            if not isinstance(handler, logging.NullHandler):
                handler.close()
                self._logger.removeHandler(handler)

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

    def close(self) -> None:
        """Close all handlers. Useful for cleanup in tests."""
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)


"""Logging configuration and handlers for the application."""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue
from typing import Optional, Callable, List
import json
import traceback


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format with colors for console output."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'exc_info',
                          'exc_text', 'stack_info', 'pathname', 'processName',
                          'process', 'threadName', 'thread', 'getMessage', 'message']:
                log_obj[key] = value

        return json.dumps(log_obj)


class GUILogHandler(logging.Handler):
    """Custom handler for GUI log display."""

    def __init__(self, callback: Optional[Callable] = None):
        """Initialize GUI log handler.

        Args:
            callback: Function to call with log messages
        """
        super().__init__()
        self.callback = callback
        self.log_buffer = []
        self.max_buffer_size = 1000

    def emit(self, record):
        """Emit a log record."""
        try:
            msg = self.format(record)
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created),
                'level': record.levelname,
                'logger': record.name,
                'message': msg,
            }

            # Add to buffer
            self.log_buffer.append(log_entry)
            if len(self.log_buffer) > self.max_buffer_size:
                self.log_buffer.pop(0)

            # Call callback if provided
            if self.callback:
                self.callback(log_entry)

        except Exception:
            self.handleError(record)

    def get_logs(self, level: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
        """Get buffered logs.

        Args:
            level: Filter by log level
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        logs = self.log_buffer

        if level:
            logs = [log for log in logs if log['level'] == level]

        if limit:
            logs = logs[-limit:]

        return logs

    def clear_buffer(self):
        """Clear the log buffer."""
        self.log_buffer.clear()


class TrainingLogHandler(logging.Handler):
    """Specialized handler for training logs with metrics."""

    def __init__(self, log_dir: str):
        """Initialize training log handler.

        Args:
            log_dir: Directory to save training logs
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create separate files for different log types
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.jsonl"
        self.events_file = self.log_dir / f"events_{timestamp}.log"

    def emit(self, record):
        """Emit a log record."""
        try:
            # Check if this is a metrics log
            if hasattr(record, 'metrics'):
                # Save metrics to JSONL file
                metrics_data = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'step': getattr(record, 'step', None),
                    'epoch': getattr(record, 'epoch', None),
                    'metrics': record.metrics,
                }
                with open(self.metrics_file, 'a') as f:
                    f.write(json.dumps(metrics_data) + '\n')

            # Save all events to regular log file
            with open(self.events_file, 'a') as f:
                f.write(self.format(record) + '\n')

        except Exception:
            self.handleError(record)


class LogManager:
    """Central log manager for the application."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize log manager."""
        if not self._initialized:
            self.logger_name = "grpo_gui"
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)

            self.root_logger = logging.getLogger(self.logger_name)
            self.root_logger.setLevel(logging.DEBUG)

            # Remove default handlers
            self.root_logger.handlers = []

            # Setup default handlers
            self._setup_console_handler()
            self._setup_file_handler()

            # GUI handler (will be set later)
            self.gui_handler = None

            # Training handler (will be set when training starts)
            self.training_handler = None

            # Queue for thread-safe logging
            self.log_queue = Queue()
            self.queue_handler = QueueHandler(self.log_queue)
            self.queue_listener = None

            self._initialized = True

    def _setup_console_handler(self):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Use colored formatter for console
        if sys.stdout.isatty():
            formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(formatter)
        self.root_logger.addHandler(console_handler)
        self.console_handler = console_handler

    def _setup_file_handler(self):
        """Setup rotating file handler."""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"grpo_gui_{timestamp}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)

        # Use JSON formatter for file
        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)

        self.root_logger.addHandler(file_handler)
        self.file_handler = file_handler

    def setup_gui_handler(self, callback: Optional[Callable] = None):
        """Setup GUI log handler.

        Args:
            callback: Function to call with log messages
        """
        if self.gui_handler:
            self.root_logger.removeHandler(self.gui_handler)

        self.gui_handler = GUILogHandler(callback)
        self.gui_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.gui_handler.setFormatter(formatter)

        self.root_logger.addHandler(self.gui_handler)

    def setup_training_handler(self, training_dir: str):
        """Setup training log handler.

        Args:
            training_dir: Directory for training logs
        """
        if self.training_handler:
            self.root_logger.removeHandler(self.training_handler)

        self.training_handler = TrainingLogHandler(training_dir)
        self.training_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.training_handler.setFormatter(formatter)

        self.root_logger.addHandler(self.training_handler)

    def start_queue_listener(self):
        """Start queue listener for thread-safe logging."""
        if not self.queue_listener:
            handlers = [h for h in self.root_logger.handlers]
            self.queue_listener = QueueListener(self.log_queue, *handlers, respect_handler_level=True)
            self.queue_listener.start()

            # Replace all handlers with queue handler
            self.root_logger.handlers = [self.queue_handler]

    def stop_queue_listener(self):
        """Stop queue listener."""
        if self.queue_listener:
            self.queue_listener.stop()
            self.queue_listener = None

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        return logging.getLogger(f"{self.logger_name}.{name}")

    def set_level(self, level: str):
        """Set global log level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")

        self.root_logger.setLevel(numeric_level)

    def set_console_level(self, level: str):
        """Set console handler log level.

        Args:
            level: Log level
        """
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")

        if hasattr(self, 'console_handler'):
            self.console_handler.setLevel(numeric_level)

    def log_metrics(self, metrics: dict, step: Optional[int] = None, epoch: Optional[int] = None):
        """Log training metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step
            epoch: Training epoch
        """
        logger = self.get_logger("training")
        logger.info("Training metrics", extra={'metrics': metrics, 'step': step, 'epoch': epoch})

    def log_exception(self, exc_info=None):
        """Log exception with traceback.

        Args:
            exc_info: Exception info (if None, uses sys.exc_info())
        """
        logger = self.get_logger("error")
        if exc_info is None:
            exc_info = sys.exc_info()

        logger.error("Exception occurred", exc_info=exc_info)

    def cleanup(self):
        """Cleanup log handlers and listeners."""
        self.stop_queue_listener()

        # Close all handlers
        for handler in self.root_logger.handlers[:]:
            handler.close()
            self.root_logger.removeHandler(handler)


# Global log manager instance
log_manager = LogManager()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return log_manager.get_logger(name)


def setup_logging(log_level: str = "INFO",
                  console_level: str = "INFO",
                  log_dir: str = "logs",
                  use_json: bool = False):
    """Setup logging configuration.

    Args:
        log_level: Global log level
        console_level: Console log level
        log_dir: Directory for log files
        use_json: Use JSON formatting
    """
    log_manager.log_dir = Path(log_dir)
    log_manager.log_dir.mkdir(exist_ok=True)

    log_manager.set_level(log_level)
    log_manager.set_console_level(console_level)


if __name__ == "__main__":
    # Test logging
    setup_logging(log_level="DEBUG")
    logger = get_logger("test")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    try:
        1 / 0
    except ZeroDivisionError:
        log_manager.log_exception()

    log_manager.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100, epoch=1)

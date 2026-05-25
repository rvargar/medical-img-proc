from loguru import logger
import sys
from pathlib import Path

class Logger:
    def __init__(self, log_file: str = "./data/logs/medical-img-proc.log", level: str = "DEBUG", rotation: str = "10 MB",
                 retention: str = "7 days", compression: str = "zip"):
        self.log_file = Path(log_file)
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Remove default logger to avoid duplicate logs
        logger.remove()

        # Console handler
        logger.add(sys.stdout, level=level, colorize=True,
                   format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

        # File handler
        logger.add(
            str(self.log_file),
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            enqueue=True,
        )

        self._logger = logger

    def get_logger(self):
        return self._logger

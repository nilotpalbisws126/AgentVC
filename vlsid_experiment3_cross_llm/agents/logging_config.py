"""
Logging Configuration Module.
Sets up structured file and console logging for all agent interactions,
Ollama LLM queries, debate transcripts, and sanity validation checks.
"""

import os
import logging
from datetime import datetime

def setup_agent_logger(log_dir: str = "logs", log_name: str = "agent_execution.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("VLSID_Orchestrator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File Handler
    fh = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Initialized agent logger. Writing log trace to {log_file_path}")
    return logger

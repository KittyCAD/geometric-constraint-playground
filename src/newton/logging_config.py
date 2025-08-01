import logging

logger = logging.getLogger("newton")


# Default configuration
def configure_logging(level=logging.INFO):
    logger.setLevel(level)

    # Create console handler if none exists yet.
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

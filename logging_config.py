    import logging
    from logging.handlers import RotatingFileHandler
    import os

    def setup_logging(log_file="backend/logs/chatbot.log"):
        """
        Setup logging with rotation and console output.
        Ensures no duplicate handlers are added.
        Returns the configured logger.
        """
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Get root logger
        logger = logging.getLogger("HealthChatbotLogger")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Avoid duplicate logs if imported multiple times

        # Prevent adding duplicate handlers
        if not logger.handlers:
            # Rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=2 * 1024 * 1024,  # 2 MB
                backupCount=5,
                encoding="utf-8"
            )
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)

            logger.info("✅ Logging setup complete.")

        return logger

    # ---------------- Usage Example ----------------
    # logger = setup_logging()
    # logger.info("This is an info log")

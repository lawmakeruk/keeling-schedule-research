# app/config.py
"""Configuration settings for the Keeling Schedule Service."""

import os


class Config:
    def __init__(self):
        # Prompt Logging
        self.LOG_PROMPTS = True

        # Paths
        self.LOG_PATH = os.path.join(os.getcwd(), "app", ".log")
        self.PROMPTS_PATH = os.path.join(os.getcwd(), "app", "kernel", "plugins")

        # URL Scheme
        self.PREFERRED_URL_SCHEME = os.environ.get("PREFERRED_URL_SCHEME", "https")

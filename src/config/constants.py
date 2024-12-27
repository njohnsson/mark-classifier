#!/usr/bin/env python3
"""
Constants for the Mark Classifier project. 
This file should be placed in the src/config directory (not in the root directory).
"""

import os

# Directory Paths
# ------------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 3 levels up
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
PROMPTS_DIR = os.path.join(ROOT_DIR, 'src', 'prompts')
TEMP_DIR = os.path.join(ROOT_DIR, 'temp')
# Data directories and paths
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'idm_embeddings')
RAW_IDMANUAL_CSV_PATH = os.path.join(DATA_DIR, 'idmanual.csv')
CLASSES_CSV_PATH = os.path.join(DATA_DIR, 'classes.csv')
IDM_CSV_PATH = os.path.join(DATA_DIR, 'idm.csv')


# Model and API Defaults
# ------------------------------------------------------------------------------
DEFAULT_DEPLOYMENT_NAME = 'gpt-4o-mini'
DEFAULT_API_VERSION = '2024-08-01-preview'

MAX_RETRIES = 3         # For re-trying failed API calls
INITIAL_RETRY_DELAY = 1 # Seconds
EXPONENTIAL_BACKOFF = 2 # Multiplier


# Logging
# ------------------------------------------------------------------------------
DEFAULT_LOGGING_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_RESPONSE_FREQUENCY = 50      # e.g., how often to log full responses

# Other
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# QA (If you need a quick self-test)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print('ROOT_DIR:', ROOT_DIR)
    print('CONFIG_DIR:', CONFIG_DIR)
    print('DATA_DIR:', DATA_DIR)
    print('LOGS_DIR:', LOGS_DIR)
    print('RAW_IDMANUAL_CSV_PATH:', RAW_IDMANUAL_CSV_PATH)
    print('CLASSES_CSV_PATH:', CLASSES_CSV_PATH)
    print('DEFAULT_LOGGING_LEVEL:', DEFAULT_LOGGING_LEVEL)

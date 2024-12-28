# src/config/config.py
import os

# Holds environment-specific settings
 # Values not expected to change by env Model and API version are in constants.py

class Config:
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_TESTAPP")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_SECRET_KEY_TESTAPP")

   
# QA
if __name__ == '__main__':
    print('AZURE_ENDPOINT:', Config.AZURE_ENDPOINT)
    print('AZURE_API_KEY', Config.AZURE_API_KEY[:5])
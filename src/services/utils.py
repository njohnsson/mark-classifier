# src/services/utils.py

import os
from datetime import datetime
import pandas as pd
from src.config.constants import CONVERSATION_HISTORY_PATH


def save_conversation(messages, session_key, filename=CONVERSATION_HISTORY_PATH):
    messages_with_session_and_time = [
        {
            "session_key": session_key,
            "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages if msg["role"] in ["user", "assistant", "system"]
    ]
    df = pd.DataFrame(messages_with_session_and_time)
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False, encoding='utf-8')

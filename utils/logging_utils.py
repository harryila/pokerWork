import os
import json
from datetime import datetime
from typing import List, Dict, Any, Union

class HandHistoryLogger:
    def __init__(self, log_dir: str = "data/hand_history"):
        print("[DEBUG] ✅ USING logging_utils.py from official-llm-poker-collusion-main")
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.history: List[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]):
        self.history.append(record)

    def log_hand(self, hand_log: dict, hand_id: Union[int, str]):
        print(f"[DEBUG] Using HandHistoryLogger from: {__file__}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hand_{hand_id}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        print(f"[DEBUG] Writing log to: {filepath}")
        print(f"[DEBUG] HandHistoryLogger log_dir: {self.log_dir}")
        with open(filepath, "w") as f:
            json.dump(hand_log, f, indent=2)

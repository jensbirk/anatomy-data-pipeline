import csv
import os
import time
from typing import List, Any

class CSVLogger:
    def __init__(self, filepath: str, fieldnames: List[str], append: bool = False):
        self.filepath = filepath
        self.fieldnames = fieldnames
        if "timestamp" not in self.fieldnames:
            self.fieldnames = ["timestamp"] + self.fieldnames
        self.row_buffer = {} 

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        mode = 'a' if append else 'w'
        write_header = not append or not os.path.exists(filepath) or os.path.getsize(filepath) == 0

        self.file = open(self.filepath, mode, newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if write_header:
            self.writer.writeheader()
            self.file.flush()

    def log(self, key: str, value: Any):
        if key not in self.fieldnames:
            print(f"Warning: '{key}' is not in fieldnames and will be ignored.")
            return

        self.row_buffer[key] = value

    def step(self):
        if not self.row_buffer:
            return 

        try:
            if "timestamp" not in self.row_buffer:
                self.row_buffer["timestamp"] = time.time()
            self.writer.writerow(self.row_buffer)
            self.file.flush() 
        except ValueError as e:
            print(f"Logging Error: {e}")
        
        self.row_buffer = {}

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
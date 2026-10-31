import time
from .base import BaseMonitor

class SMIMonitor(BaseMonitor):
    def __init__(self, log_dir):
        cmd = [
            "nvidia-smi", 
            # Added 'temperature.gpu' to the end of the query string
            "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used,temperature.gpu", 
            "--format=csv,nounits", 
            "-l", "1"
        ]
        super().__init__(output_path=log_dir / "smi.csv", command=cmd)

    def parse_line(self, line):
        # Clean whitespace
        line = line.strip()
        
        # header line
        if "uuid" in line.lower() or "utilization" in line.lower():
            if not self.header_written:
                self.header_written = True
                return "timestamp," + line
            return None

        # Handle Data
        # Capture current time (Unix timestamp) to match DCGMI
        current_ts = str(time.time())
        
        # Prepend timestamp to the raw nvidia-smi CSV line
        return f"{current_ts},{line}"
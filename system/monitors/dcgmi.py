import time
from .base import BaseMonitor

class DcgmiMonitor(BaseMonitor):
    def __init__(self, log_dir):
        cmd = [
            "dcgmi", "dmon", "-e", 
            "155,156,200,201,203,204,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012", 
            "-d", "1000"
        ]
        super().__init__(output_path=log_dir / "dcgmi.csv", command=cmd)

    def parse_line(self, line):
        # Skip definition lines
        if line.startswith("ID"): 
            return None
        
        # Handle Header: Just add timestamp, leave column structure as is
        if line.startswith("#Entity"):
            if not self.header_written:
                self.header_written = True
                return "timestamp," + ",".join(line.split())
            return None

        # Handle Data
        parts = line.split()
        if len(parts) < 3:
            return None  # blank lines and error messages (e.g. DCGM not available)
            
        # 1. Combine "GPU" and "0" into "GPU 0" (Single value)
        entity_id = f"{parts[0]} {parts[1]}"
        
        # 2. Clean the rest of the metrics
        clean_metrics = [p if p != "N/A" else "0" for p in parts[2:]]
        
        # 3. Get timestamp
        current_ts = str(time.time())
        
        # Return: timestamp, "GPU 0", metrics...
        return f"{current_ts},{entity_id}," + ",".join(clean_metrics)
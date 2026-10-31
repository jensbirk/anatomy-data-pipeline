import time
from .base import BaseMonitor

class IostatMonitor(BaseMonitor):
    def __init__(self, log_dir):
        # -x: Extended stats, -m: Display in MB, 1: Interval in seconds
        cmd = ["iostat", "-x", "-m", "1"]
        super().__init__(output_path=log_dir / "iostat.csv", command=cmd)

    def parse_line(self, line):
        line = line.strip()
        
        # Skip empty lines and System/CPU header info
        if not line or line.startswith("Linux") or line.startswith("avg-cpu"): 
            return None
        
        # Handle Header: "Device" marks the start of the disk stats section
        if line.startswith("Device"):
            if not self.header_written:
                self.header_written = True
                return "timestamp," + ",".join(line.split())
            return None

        # Handle Data
        parts = line.split()
        
        # 1. Filter out CPU stats rows (rows starting with numbers)
        try:
            float(parts[0])
            return None
        except ValueError:
            pass # It is a text string (Device name), proceed

        # 2. Filter specifically for NVMe devices
        # Only return this line if "nvme" is inside the device name (e.g. "nvme0n1")
        if "nvme" not in parts[0]:
            return None

        # 3. Get timestamp
        current_ts = str(time.time())
        
        # Return: timestamp, metrics...
        return f"{current_ts}," + ",".join(parts)
import time
from .base import BaseMonitor

class VmstatMonitor(BaseMonitor):
    def __init__(self, log_dir):
        # Removed "-t" so we can handle the timestamp in Python
        cmd = ["vmstat", "-n", "1"]
        super().__init__(output_path=log_dir / "vmstat.csv", command=cmd)

    def parse_line(self, line):
        # Skip the very first grouping line (procs -----------memory...)
        if "procs" in line and "memory" in line:
            return None
            
        # Handle the real header line (r b swpd...)
        if "swpd" in line: 
            if not self.header_written:
                self.header_written = True
                # Prepend timestamp to header
                return "timestamp," + ",".join(line.split())
            return None

        # Handle Data lines
        parts = line.split()
        if not parts:
            return None

        # Capture current time (Unix timestamp)
        current_ts = str(time.time())
        
        # Return: timestamp, <vmstat_metrics>
        return f"{current_ts}," + ",".join(parts)
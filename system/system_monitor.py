from pathlib import Path

from system.monitors.dcgmi import DcgmiMonitor
from system.monitors.vmstat import VmstatMonitor
from system.monitors.smi import SMIMonitor
from system.monitors.iostat import IostatMonitor
from system.monitors.psutil import PSUtilMonitor
class SystemMonitor:
    def __init__(self, log_dir_str, process_identifier: str):
        self.log_dir = Path(log_dir_str)
        self.process_identifier = process_identifier
        self.monitors = []

    def __enter__(self):
        print("   Starting System Monitors...")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitors = [
            VmstatMonitor(self.log_dir),
            DcgmiMonitor(self.log_dir),
            SMIMonitor(self.log_dir),
            IostatMonitor(self.log_dir),
            PSUtilMonitor(self.log_dir)
        ]

        for m in self.monitors:
            m.start()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("   Stopping System Monitors...")
        for m in self.monitors:
            m.stop()
        print(f"   Clean Logs ready in: {self.log_dir}/")

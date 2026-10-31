import psutil
import os
import time
import threading
from pathlib import Path

class PSUtilMonitor:
    def __init__(self, output_path: Path, interval: int = 1):
        self.output_path = output_path
        self.parent_pid = os.getpid() 
        self.interval = interval
        self.excluded_processes = ["vmstat", "dcgmi", "nvidia-smi", "iostat", "psutil"] 
        
        self.thread = None
        self._stop_event = threading.Event()
        self.tracked_processes = {}

    def _get_all_pids(self) -> list[int]:
        try:
            parent_process = psutil.Process(self.parent_pid)
            children = parent_process.children(recursive=True)
            all_pids = [self.parent_pid] + [p.pid for p in children]
            
            # Filter excluded names
            all_pids_filtered = []
            for pid in all_pids:
                try:
                    if psutil.Process(pid).name() not in self.excluded_processes:
                        all_pids_filtered.append(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return all_pids_filtered
        except psutil.NoSuchProcess:
            return []

    def _get_or_create_process(self, pid: int) -> psutil.Process | None:
        if pid not in self.tracked_processes:
            try:
                self.tracked_processes[pid] = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return None
        return self.tracked_processes[pid]
    
    def _monitoring_loop(self):
        if not self.output_path.exists():
            os.makedirs(self.output_path, exist_ok=True)

        file_path = self.output_path / "psutil.csv"
        
        # Prime CPU counter (for global system)
        psutil.cpu_percent(interval=None)

        with open(file_path, 'w', buffering=1) as f:
            # CSV header, including system-wide CPU metrics
            header = (
                "timestamp,"
                "App_Process_Count,"       # processes in the monitored tree
                "App_Total_CPU,"           # CPU used by the trainer 
                "System_CPU_Count,"        # Total Logical CPUs
                "System_CPU_Percent,"      # Global System CPU %
                "System_Used_MB,"          # GLOBAL Used
                "System_Shared_MB,"        # GLOBAL Shared
                "System_Buffers_MB,"       # GLOBAL Buffers
                "System_Cache_MB,"         # GLOBAL Cache
                "System_Swap_MB\n"         # GLOBAL Swap
            )
            f.write(header)
            
            while not self._stop_event.is_set():
                current_time = time.time()
                
                # Monitored processes
                all_pids = self._get_all_pids()
                
                # Cleanup cache
                pids_to_remove = [pid for pid in self.tracked_processes if pid not in all_pids]
                for pid in pids_to_remove:
                    del self.tracked_processes[pid]

                # Per-application metrics
                app_cpu = 0.0
                
                for pid in all_pids:
                    p = self._get_or_create_process(pid)
                    if p:
                        try:
                            # Sum CPU
                            app_cpu += p.cpu_percent(interval=None)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                # System-wide metrics
                # CPU Metrics
                sys_cpu_cnt = psutil.cpu_count(logical=True)
                sys_cpu_pct = psutil.cpu_percent(interval=None)

                # Memory Metrics
                sys_mem = psutil.virtual_memory()
                sys_swap = psutil.swap_memory()
                
                shared_mb = getattr(sys_mem, 'shared', 0) / (1024 * 1024)
                buffers_mb = getattr(sys_mem, 'buffers', 0) / (1024 * 1024)
                cache_mb = getattr(sys_mem, 'cached', 0) / (1024 * 1024)
                
                # Htop Green = Used - Shared
                used_mb = (sys_mem.used / (1024 * 1024)) - shared_mb
                
                swap_used_mb = sys_swap.used / (1024 * 1024)

                # Write log row
                log_line = (
                    f"{current_time:.2f},"
                    f"{len(all_pids)},"
                    f"{app_cpu:.2f},"
                    f"{sys_cpu_cnt},"      
                    f"{sys_cpu_pct:.1f},"  
                    f"{used_mb:.2f},"
                    f"{shared_mb:.2f},"
                    f"{buffers_mb:.2f},"
                    f"{cache_mb:.2f},"
                    f"{swap_used_mb:.2f}"
                )
                f.write(log_line + "\n")
                
                self._stop_event.wait(self.interval)

    def start(self):
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True 
        self.thread.start()
        print(f"Monitor started for PID {self.parent_pid}...")

    def stop(self):
        if self.thread and self.thread.is_alive():
            print("Stopping monitor...")
            self._stop_event.set()
            self.thread.join(timeout=2.0)
            print("Monitor stopped.")

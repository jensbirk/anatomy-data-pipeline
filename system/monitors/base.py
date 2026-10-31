import subprocess
import threading
from abc import ABC, abstractmethod

class BaseMonitor(ABC):
    def __init__(self, output_path, command):
        self.output_path = output_path
        self.command = command
        self.process = None
        self.thread = None
        self.header_written = False


    def _stream_writer(self):
        with open(self.output_path, 'w', buffering=1) as f:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                # Delegate specific parsing logic to the subclass
                cleaned_line = self.parse_line(line)
                
                if cleaned_line:
                    f.write(cleaned_line + "\n")

    def start(self):
        # pipe_stdout=True allows us to catch the stream in Python
        self.process = subprocess.Popen(
            self.command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )
        self.thread = threading.Thread(target=self._stream_writer)
        self.thread.start()

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=1.0)
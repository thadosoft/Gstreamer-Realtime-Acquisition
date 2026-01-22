import os
import psutil
from dataclasses import dataclass

@dataclass
class ResourceMetrics:
    cpu_usage: float = 0.0
    cpu_usage_total: float = 0.0
    ram_usage_mb: float = 0.0
    thread_count: int = 0

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.cpu_count = psutil.cpu_count()

    def measure(self) -> ResourceMetrics:
        try:
            cpu_usage = self.process.cpu_percent(interval=None)
            mem_info = self.process.memory_info()

            return ResourceMetrics(
                cpu_usage=cpu_usage,
                cpu_usage_total=cpu_usage / self.cpu_count,
                ram_usage_mb=mem_info.rss / (1024 * 1024),
                thread_count=len(self.process.threads())
            )
        except Exception:
            return ResourceMetrics()
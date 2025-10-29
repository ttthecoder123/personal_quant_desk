"""System latency monitoring"""
from typing import Dict
from datetime import datetime
class LatencyTracker:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.latencies = {}
    def record_latency(self, operation: str, latency_ms: float):
        if operation not in self.latencies:
            self.latencies[operation] = []
        self.latencies[operation].append(latency_ms)
    def get_latency_stats(self, operation: str) -> Dict:
        if operation not in self.latencies or not self.latencies[operation]:
            return {}
        import numpy as np
        lats = self.latencies[operation]
        return {
            'min': min(lats),
            'max': max(lats),
            'avg': np.mean(lats),
            'p50': np.percentile(lats, 50),
            'p95': np.percentile(lats, 95),
            'p99': np.percentile(lats, 99)
        }

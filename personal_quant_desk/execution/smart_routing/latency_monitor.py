"""Venue latency tracking"""
from typing import Dict
class LatencyMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.latencies = {}
    def update_latency(self, venue: str, latency_ms: float):
        self.latencies[venue] = latency_ms
    def get_latency(self, venue: str) -> float:
        return self.latencies.get(venue, 0)

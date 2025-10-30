"""
Process Monitor

Monitors process lifecycle, restarts, crashes, hangs,
resource limits, child processes, and service dependencies.
"""

import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque


@dataclass
class ProcessInfo:
    """Process information."""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_percent: float
    num_threads: int
    create_time: datetime
    restart_count: int = 0
    last_restart: Optional[datetime] = None


class ProcessMonitor:
    """
    Comprehensive process monitoring.

    Features:
    - Process lifecycle tracking
    - Restart detection
    - Crash monitoring
    - Hang detection
    - Resource limit monitoring
    - Child process tracking
    - Service dependency monitoring
    """

    def __init__(self, check_interval: int = 10):
        """
        Initialize process monitor.

        Args:
            check_interval: Seconds between checks
        """
        self.check_interval = check_interval
        self.monitored_processes: Dict[int, ProcessInfo] = {}
        self.process_history: Dict[int, deque] = {}
        self.process_names: Dict[str, Set[int]] = {}  # name -> set of PIDs
        self.dependencies: Dict[int, Set[int]] = {}  # parent -> children
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Thresholds for alerts
        self.cpu_threshold = 90.0
        self.memory_threshold = 90.0
        self.hang_threshold_seconds = 300

    def start(self):
        """Start process monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop process monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def register_process(self, pid: int):
        """
        Register a process for monitoring.

        Args:
            pid: Process ID
        """
        try:
            proc = psutil.Process(pid)
            info = ProcessInfo(
                pid=pid,
                name=proc.name(),
                status=proc.status(),
                cpu_percent=proc.cpu_percent(),
                memory_percent=proc.memory_percent(),
                num_threads=proc.num_threads(),
                create_time=datetime.fromtimestamp(proc.create_time())
            )

            with self.lock:
                self.monitored_processes[pid] = info
                self.process_history[pid] = deque(maxlen=1000)

                # Track by name
                if info.name not in self.process_names:
                    self.process_names[info.name] = set()
                self.process_names[info.name].add(pid)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def unregister_process(self, pid: int):
        """
        Unregister a process from monitoring.

        Args:
            pid: Process ID
        """
        with self.lock:
            if pid in self.monitored_processes:
                info = self.monitored_processes[pid]
                self.process_names[info.name].discard(pid)
                del self.monitored_processes[pid]

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_processes()
                self._detect_hangs()
                self._detect_crashes()
                self._detect_restarts()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in process monitor loop: {e}")

    def _check_all_processes(self):
        """Check all monitored processes."""
        with self.lock:
            for pid in list(self.monitored_processes.keys()):
                try:
                    proc = psutil.Process(pid)
                    info = self.monitored_processes[pid]

                    # Update process info
                    info.status = proc.status()
                    info.cpu_percent = proc.cpu_percent()
                    info.memory_percent = proc.memory_percent()
                    info.num_threads = proc.num_threads()

                    # Record to history
                    self.process_history[pid].append({
                        'timestamp': datetime.now(),
                        'cpu_percent': info.cpu_percent,
                        'memory_percent': info.memory_percent,
                        'num_threads': info.num_threads,
                        'status': info.status
                    })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process no longer exists
                    pass

    def _detect_hangs(self):
        """Detect hung processes."""
        with self.lock:
            for pid, info in self.monitored_processes.items():
                if info.status in [psutil.STATUS_STOPPED, psutil.STATUS_DISK_SLEEP]:
                    # Check if stuck in this state
                    if pid in self.process_history:
                        recent = list(self.process_history[pid])[-10:]
                        if len(recent) >= 10:
                            all_stuck = all(h['status'] == info.status for h in recent)
                            if all_stuck:
                                print(f"Warning: Process {pid} ({info.name}) may be hung")

    def _detect_crashes(self):
        """Detect crashed processes."""
        with self.lock:
            for pid in list(self.monitored_processes.keys()):
                try:
                    psutil.Process(pid)
                except psutil.NoSuchProcess:
                    info = self.monitored_processes[pid]
                    print(f"Alert: Process {pid} ({info.name}) has crashed")
                    # Could trigger alert here

    def _detect_restarts(self):
        """Detect process restarts."""
        with self.lock:
            for name, pids in self.process_names.items():
                # Check if any new PIDs appeared for this name
                try:
                    current_pids = {p.pid for p in psutil.process_iter(['pid', 'name'])
                                   if p.info['name'] == name}

                    new_pids = current_pids - pids
                    if new_pids:
                        # Process was restarted
                        for pid in new_pids:
                            # Update tracked PIDs
                            pids.add(pid)
                            # Record restart
                            for old_pid in (pids - current_pids):
                                if old_pid in self.monitored_processes:
                                    old_info = self.monitored_processes[old_pid]
                                    old_info.restart_count += 1
                                    old_info.last_restart = datetime.now()

                except Exception:
                    pass

    def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """
        Get information for a monitored process.

        Args:
            pid: Process ID

        Returns:
            ProcessInfo or None
        """
        with self.lock:
            return self.monitored_processes.get(pid)

    def get_process_history(self, pid: int, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Get process history.

        Args:
            pid: Process ID
            window_minutes: Time window

        Returns:
            List of historical data points
        """
        with self.lock:
            if pid not in self.process_history:
                return []

            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            return [h for h in self.process_history[pid] if h['timestamp'] >= cutoff]

    def get_high_cpu_processes(self, threshold: float = 80.0) -> List[ProcessInfo]:
        """
        Get processes with high CPU usage.

        Args:
            threshold: CPU percentage threshold

        Returns:
            List of ProcessInfo objects
        """
        with self.lock:
            return [info for info in self.monitored_processes.values()
                   if info.cpu_percent >= threshold]

    def get_high_memory_processes(self, threshold: float = 80.0) -> List[ProcessInfo]:
        """
        Get processes with high memory usage.

        Args:
            threshold: Memory percentage threshold

        Returns:
            List of ProcessInfo objects
        """
        with self.lock:
            return [info for info in self.monitored_processes.values()
                   if info.memory_percent >= threshold]

    def get_crashed_processes(self) -> List[Dict[str, Any]]:
        """
        Get list of processes that have crashed.

        Returns:
            List of crash information
        """
        with self.lock:
            crashed = []
            for pid, info in list(self.monitored_processes.items()):
                try:
                    psutil.Process(pid)
                except psutil.NoSuchProcess:
                    crashed.append({
                        'pid': pid,
                        'name': info.name,
                        'crash_time': datetime.now(),
                        'uptime_seconds': (datetime.now() - info.create_time).total_seconds()
                    })
            return crashed

    def get_summary(self) -> Dict[str, Any]:
        """
        Get process monitoring summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            return {
                'total_monitored': len(self.monitored_processes),
                'high_cpu_count': len(self.get_high_cpu_processes()),
                'high_memory_count': len(self.get_high_memory_processes()),
                'crashed_count': len(self.get_crashed_processes()),
                'timestamp': datetime.now().isoformat()
            }

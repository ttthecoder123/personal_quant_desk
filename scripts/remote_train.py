#!/usr/bin/env python3
"""
Remote GPU Training Orchestration Script

This script automates:
1. SSH connection to remote GPU server
2. Code sync (rsync)
3. Training job submission
4. Model download
5. Auto-shutdown (cost optimization)

Usage:
    python scripts/remote_train.py --config configs/remote_gpu.yaml --model lstm
"""

import argparse
import subprocess
import time
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RemoteTrainer:
    """Orchestrates training on remote GPU server"""

    def __init__(self, config_path: str):
        """Load remote server configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.host = self.config['remote_gpu']['host']
        self.user = self.config['remote_gpu']['user']
        self.port = self.config['remote_gpu'].get('port', 22)
        self.remote_path = self.config['remote_gpu']['remote_path']
        self.local_path = Path(__file__).parent.parent

        logger.info(f"Remote server: {self.user}@{self.host}:{self.port}")

    def sync_code(self):
        """Sync local code to remote server using rsync"""
        logger.info("Syncing code to remote server...")

        exclude = [
            '--exclude', '.git',
            '--exclude', 'data/historical',  # Don't sync large data files
            '--exclude', 'data/features/computed',
            '--exclude', '*.pyc',
            '--exclude', '__pycache__',
            '--exclude', '.pytest_cache',
            '--exclude', 'models/trained',  # Don't sync old models
            '--exclude', 'notebooks/.ipynb_checkpoints'
        ]

        rsync_cmd = [
            'rsync', '-avz', '--progress',
            '-e', f'ssh -p {self.port}',
            *exclude,
            f'{self.local_path}/',
            f'{self.user}@{self.host}:{self.remote_path}/'
        ]

        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✓ Code synced successfully")
        else:
            logger.error(f"✗ Sync failed: {result.stderr}")
            raise RuntimeError("Code sync failed")

    def sync_data(self):
        """Sync only essential data files to remote server"""
        logger.info("Syncing data to remote server...")

        # Only sync processed data catalog (small file)
        data_files = [
            'data/data_catalog.db',
            'config/',
        ]

        for data_path in data_files:
            rsync_cmd = [
                'rsync', '-avz',
                '-e', f'ssh -p {self.port}',
                f'{self.local_path}/{data_path}',
                f'{self.user}@{self.host}:{self.remote_path}/{data_path}'
            ]
            subprocess.run(rsync_cmd)

        logger.info("✓ Data synced")

    def run_training(self, model_type: str, config: Dict[str, Any]):
        """Submit training job to remote server"""
        logger.info(f"Starting training: {model_type}")

        # Build training command
        train_cmd = self._build_train_command(model_type, config)

        # SSH and run training
        ssh_cmd = [
            'ssh',
            '-p', str(self.port),
            f'{self.user}@{self.host}',
            f'cd {self.remote_path} && {train_cmd}'
        ]

        logger.info(f"Executing: {train_cmd}")

        # Run with live output
        process = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode == 0:
            logger.info("✓ Training completed successfully")
        else:
            logger.error("✗ Training failed")
            raise RuntimeError("Training failed")

    def _build_train_command(self, model_type: str, config: Dict[str, Any]) -> str:
        """Build training command based on model type"""

        if model_type == 'lstm':
            return f"python3 models/deep_learning/train_lstm.py --config {config.get('model_config', 'configs/lstm_config.yaml')}"

        elif model_type == 'transformer':
            return f"python3 models/deep_learning/train_transformer.py --config {config.get('model_config', 'configs/transformer_config.yaml')}"

        elif model_type == 'rl_execution':
            return f"python3 models/reinforcement_learning/train_dqn_execution.py --config {config.get('model_config', 'configs/rl_execution_config.yaml')}"

        elif model_type == 'all':
            # Train all models sequentially
            return """
                python3 models/deep_learning/train_lstm.py && \
                python3 models/deep_learning/train_transformer.py && \
                python3 models/reinforcement_learning/train_dqn_execution.py
            """

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def download_models(self, output_dir: str = "models/trained"):
        """Download trained models from remote server"""
        logger.info("Downloading trained models...")

        # Ensure local output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        rsync_cmd = [
            'rsync', '-avz', '--progress',
            '-e', f'ssh -p {self.port}',
            f'{self.user}@{self.host}:{self.remote_path}/models/trained/',
            f'{self.local_path}/{output_dir}/'
        ]

        result = subprocess.run(rsync_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✓ Models downloaded to {output_dir}")
        else:
            logger.error(f"✗ Download failed: {result.stderr}")

    def shutdown_remote(self):
        """Shutdown remote server to save costs (only for cloud instances)"""
        if self.config['remote_gpu'].get('auto_shutdown', False):
            logger.info("Shutting down remote server...")

            ssh_cmd = [
                'ssh',
                '-p', str(self.port),
                f'{self.user}@{self.host}',
                'sudo shutdown -h now'
            ]

            subprocess.run(ssh_cmd)
            logger.info("✓ Shutdown command sent")
        else:
            logger.info("Auto-shutdown disabled")

    def monitor_gpu(self):
        """Monitor GPU utilization on remote server"""
        logger.info("Monitoring GPU...")

        ssh_cmd = [
            'ssh',
            '-p', str(self.port),
            f'{self.user}@{self.host}',
            'nvidia-smi'
        ]

        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        print(result.stdout)

    def full_workflow(self, model_type: str, config: Dict[str, Any]):
        """Complete training workflow"""
        try:
            # 1. Sync code
            self.sync_code()

            # 2. Sync data (optional)
            if config.get('sync_data', False):
                self.sync_data()

            # 3. Run training
            self.run_training(model_type, config)

            # 4. Download models
            self.download_models()

            # 5. Shutdown (if configured)
            self.shutdown_remote()

            logger.info("✓ Full workflow completed successfully")

        except Exception as e:
            logger.error(f"✗ Workflow failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Remote GPU Training Orchestration')
    parser.add_argument('--config', type=str, required=True, help='Remote GPU config file')
    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'transformer', 'rl_execution', 'all'],
                        help='Model type to train')
    parser.add_argument('--sync-only', action='store_true', help='Only sync code, no training')
    parser.add_argument('--download-only', action='store_true', help='Only download models')
    parser.add_argument('--monitor', action='store_true', help='Monitor GPU usage')

    args = parser.parse_args()

    # Load config
    trainer = RemoteTrainer(args.config)

    if args.monitor:
        trainer.monitor_gpu()

    elif args.sync_only:
        trainer.sync_code()

    elif args.download_only:
        trainer.download_models()

    else:
        # Full workflow
        model_config = {}  # Can be extended to load model-specific config
        trainer.full_workflow(args.model, model_config)


if __name__ == '__main__':
    main()

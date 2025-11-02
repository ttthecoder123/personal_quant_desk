#!/usr/bin/env python3
"""
LSTM Price Prediction Model Training
Optimized for remote GPU training, exports to ONNX for M1 inference

Features:
- Multi-horizon prediction (1, 5, 20 days)
- Attention mechanism
- Mixed precision training (FP16)
- ONNX export for M1 compatibility
- WandB/MLflow tracking
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import yaml
import logging
from typing import Tuple, List
import wandb
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceSequenceDataset(Dataset):
    """Dataset for price sequences with multiple features"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 60):
        """
        Args:
            features: Shape (n_samples, n_features)
            targets: Shape (n_samples, n_horizons) - multiple prediction horizons
            seq_length: Sequence length for LSTM input
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.seq_length]

        # Get target (future return)
        y = self.targets[idx + self.seq_length]

        return x, y


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for price prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_horizons: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layers for each horizon
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            for _ in range(num_horizons)
        ])

    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1),  # (batch, seq)
            dim=1
        )

        # Context vector (weighted sum of LSTM outputs)
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq)
            lstm_out  # (batch, seq, hidden)
        ).squeeze(1)  # (batch, hidden)

        # Multi-horizon predictions
        outputs = [layer(context) for layer in self.output_layers]
        predictions = torch.cat(outputs, dim=1)  # (batch, num_horizons)

        return predictions


class LSTMTrainer:
    """Handles LSTM training with mixed precision and checkpointing"""

    def __init__(self, config: dict):
        self.config = config

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed precision

        # Tracking
        self.best_val_loss = float('inf')

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare data"""
        logger.info("Loading data...")

        # Load features (your existing 150+ features)
        features_path = Path("data/features/computed/features_latest.parquet")
        df = pd.read_parquet(features_path)

        # Create multi-horizon targets
        horizons = self.config['training']['horizons']  # e.g., [1, 5, 20]
        targets = []

        for h in horizons:
            df[f'target_{h}d'] = df['return_1d'].shift(-h)
            targets.append(f'target_{h}d')

        # Remove NaN rows
        df = df.dropna()

        # Split train/validation (80/20)
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]

        # Get feature columns (exclude targets and metadata)
        feature_cols = [col for col in df.columns if col not in targets + ['date', 'symbol']]

        # Create datasets
        train_dataset = PriceSequenceDataset(
            features=train_df[feature_cols].values,
            targets=train_df[targets].values,
            seq_length=self.config['model']['seq_length']
        )

        val_dataset = PriceSequenceDataset(
            features=val_df[feature_cols].values,
            targets=val_df[targets].values,
            seq_length=self.config['model']['seq_length']
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        return train_loader, val_loader, len(feature_cols)

    def build_model(self, input_size: int):
        """Build LSTM model"""
        self.model = AttentionLSTM(
            input_size=input_size,
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_horizons=len(self.config['training']['horizons']),
            dropout=self.config['model']['dropout']
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                predictions = self.model(x)
                loss = nn.MSELoss()(predictions, y)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Log progress
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                with torch.cuda.amp.autocast():
                    predictions = self.model(x)
                    loss = nn.MSELoss()(predictions, y)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        logger.info("Starting training...")

        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Log to tracking platforms
            if self.config.get('tracking', {}).get('wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best')
                logger.info(f"✓ New best model saved (val_loss: {val_loss:.6f})")

            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}')

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_dir = Path("models/trained/lstm")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)

    def export_to_onnx(self):
        """Export model to ONNX for M1 inference"""
        logger.info("Exporting to ONNX...")

        self.model.eval()

        # Dummy input for tracing
        seq_length = self.config['model']['seq_length']
        input_size = self.model.lstm.input_size
        dummy_input = torch.randn(1, seq_length, input_size).to(self.device)

        # Export
        onnx_path = Path("models/trained/lstm/lstm_predictor.onnx")

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        logger.info(f"✓ ONNX model saved to {onnx_path}")

        # Test ONNX inference
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        logger.info(f"✓ ONNX inference test passed, output shape: {ort_outputs[0].shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/lstm_config.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize tracking
    if config.get('tracking', {}).get('wandb', False):
        wandb.init(project='quant-desk', name='lstm-training', config=config)

    # Train
    trainer = LSTMTrainer(config)
    train_loader, val_loader, input_size = trainer.load_data()
    trainer.build_model(input_size)
    trainer.train(train_loader, val_loader)

    # Export to ONNX
    trainer.export_to_onnx()

    logger.info("✓ Training complete!")


if __name__ == '__main__':
    main()

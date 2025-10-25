"""
Position Manager

Handles position tracking, P&L calculation, and position lifecycle management.
Integrates with triple-barrier exit rules from Step 4.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger


class PositionStatus(Enum):
    """Position lifecycle status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    PROFIT_TARGET = "PROFIT_TARGET"
    TIME_STOP = "TIME_STOP"


@dataclass
class Position:
    """
    Represents a trading position with complete lifecycle tracking.

    Attributes:
        symbol: Instrument symbol
        side: Position side (1 for long, -1 for short)
        size: Position size in units
        entry_price: Entry price
        entry_time: Entry timestamp
        current_price: Current market price
        stop_loss: Stop loss price (from triple-barrier)
        take_profit: Take profit price (from triple-barrier)
        time_stop: Maximum holding period (from triple-barrier)
        status: Position status
        strategy: Originating strategy name
        unrealized_pnl: Current unrealized P&L
        realized_pnl: Realized P&L upon exit
        exit_price: Exit price (if closed)
        exit_time: Exit timestamp (if closed)
        commission: Total commission paid
        slippage: Estimated slippage
        metadata: Additional position metadata
    """
    symbol: str
    side: int  # 1 for long, -1 for short
    size: float
    entry_price: float
    entry_time: datetime
    current_price: float
    strategy: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_stop: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_price(self, new_price: float):
        """Update current price and recalculate unrealized P&L."""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.size * self.side

    def check_stops(self, current_price: float, current_time: datetime) -> Optional[PositionStatus]:
        """
        Check if position has hit any stop conditions.

        Args:
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            PositionStatus if stop hit, None otherwise
        """
        # Check stop loss
        if self.stop_loss is not None:
            if self.side == 1 and current_price <= self.stop_loss:
                return PositionStatus.STOPPED_OUT
            elif self.side == -1 and current_price >= self.stop_loss:
                return PositionStatus.STOPPED_OUT

        # Check take profit
        if self.take_profit is not None:
            if self.side == 1 and current_price >= self.take_profit:
                return PositionStatus.PROFIT_TARGET
            elif self.side == -1 and current_price <= self.take_profit:
                return PositionStatus.PROFIT_TARGET

        # Check time stop
        if self.time_stop is not None and current_time >= self.time_stop:
            return PositionStatus.TIME_STOP

        return None

    def close(self, exit_price: float, exit_time: datetime, status: PositionStatus):
        """
        Close the position.

        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            status: Final position status
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = status
        self.realized_pnl = (exit_price - self.entry_price) * self.size * self.side
        self.realized_pnl -= (self.commission + self.slippage)

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'time_stop': self.time_stop.isoformat() if self.time_stop else None,
            'status': self.status.value,
            'strategy': self.strategy,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'commission': self.commission,
            'slippage': self.slippage,
            'metadata': self.metadata
        }


class PositionManager:
    """
    Manages all open and closed positions across strategies.

    Provides:
    - Position lifecycle management
    - P&L tracking
    - Risk metrics calculation
    - Position aggregation across strategies
    """

    def __init__(self):
        """Initialize position manager."""
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.position_history: pd.DataFrame = pd.DataFrame()

        logger.info("Position Manager initialized")

    def open_position(
        self,
        symbol: str,
        side: int,
        size: float,
        entry_price: float,
        entry_time: datetime,
        strategy: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        time_stop: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Instrument symbol
            side: Position side (1 for long, -1 for short)
            size: Position size
            entry_price: Entry price
            entry_time: Entry timestamp
            strategy: Strategy name
            stop_loss: Stop loss price
            take_profit: Take profit price
            time_stop: Time-based stop
            metadata: Additional metadata

        Returns:
            Created Position object
        """
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=entry_time,
            current_price=entry_price,
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_stop=time_stop,
            metadata=metadata or {}
        )

        # Use unique key combining symbol and strategy
        position_key = f"{symbol}_{strategy}"
        self.open_positions[position_key] = position

        logger.info(
            f"Opened {symbol} position: "
            f"side={'LONG' if side == 1 else 'SHORT'}, "
            f"size={size}, "
            f"entry={entry_price:.2f}"
        )

        return position

    def close_position(
        self,
        symbol: str,
        strategy: str,
        exit_price: float,
        exit_time: datetime,
        status: PositionStatus = PositionStatus.CLOSED
    ) -> Optional[Position]:
        """
        Close an existing position.

        Args:
            symbol: Instrument symbol
            strategy: Strategy name
            exit_price: Exit price
            exit_time: Exit timestamp
            status: Exit status

        Returns:
            Closed Position object or None if not found
        """
        position_key = f"{symbol}_{strategy}"

        if position_key not in self.open_positions:
            logger.warning(f"Cannot close non-existent position: {position_key}")
            return None

        position = self.open_positions[position_key]
        position.close(exit_price, exit_time, status)

        # Move to closed positions
        self.closed_positions.append(position)
        del self.open_positions[position_key]

        logger.info(
            f"Closed {symbol} position: "
            f"P&L={position.realized_pnl:.2f}, "
            f"status={status.value}"
        )

        return position

    def update_positions(self, market_data: Dict[str, float], current_time: datetime):
        """
        Update all positions with current market prices and check stops.

        Args:
            market_data: Dictionary of {symbol: current_price}
            current_time: Current timestamp
        """
        positions_to_close = []

        for position_key, position in self.open_positions.items():
            if position.symbol in market_data:
                current_price = market_data[position.symbol]
                position.update_price(current_price)

                # Check if any stops are hit
                stop_status = position.check_stops(current_price, current_time)
                if stop_status:
                    positions_to_close.append((position.symbol, position.strategy, stop_status))

        # Close positions that hit stops
        for symbol, strategy, status in positions_to_close:
            current_price = market_data[symbol]
            self.close_position(symbol, strategy, current_price, current_time, status)

    def get_position(self, symbol: str, strategy: str) -> Optional[Position]:
        """
        Get specific position.

        Args:
            symbol: Instrument symbol
            strategy: Strategy name

        Returns:
            Position object or None
        """
        position_key = f"{symbol}_{strategy}"
        return self.open_positions.get(position_key)

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol across strategies."""
        return [
            pos for key, pos in self.open_positions.items()
            if pos.symbol == symbol
        ]

    def get_positions_by_strategy(self, strategy: str) -> List[Position]:
        """Get all positions for a specific strategy."""
        return [
            pos for key, pos in self.open_positions.items()
            if pos.strategy == strategy
        ]

    def get_total_exposure(self) -> float:
        """Calculate total position exposure."""
        return sum(
            abs(pos.size * pos.current_price)
            for pos in self.open_positions.values()
        )

    def get_net_exposure(self) -> float:
        """Calculate net position exposure (long - short)."""
        return sum(
            pos.size * pos.current_price * pos.side
            for pos in self.open_positions.values()
        )

    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.open_positions.values())

    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L from closed positions."""
        return sum(pos.realized_pnl for pos in self.closed_positions)

    def get_total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.get_realized_pnl() + self.get_unrealized_pnl()

    def get_position_summary(self) -> pd.DataFrame:
        """
        Get summary of all open positions.

        Returns:
            DataFrame with position details
        """
        if not self.open_positions:
            return pd.DataFrame()

        data = []
        for position in self.open_positions.values():
            data.append({
                'symbol': position.symbol,
                'strategy': position.strategy,
                'side': 'LONG' if position.side == 1 else 'SHORT',
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'pnl_pct': (position.unrealized_pnl / (position.entry_price * position.size)) * 100,
                'entry_time': position.entry_time
            })

        return pd.DataFrame(data)

    def get_closed_positions_summary(self) -> pd.DataFrame:
        """
        Get summary of closed positions.

        Returns:
            DataFrame with closed position details
        """
        if not self.closed_positions:
            return pd.DataFrame()

        data = []
        for position in self.closed_positions:
            holding_period = (position.exit_time - position.entry_time).total_seconds() / 3600  # hours
            data.append({
                'symbol': position.symbol,
                'strategy': position.strategy,
                'side': 'LONG' if position.side == 1 else 'SHORT',
                'size': position.size,
                'entry_price': position.entry_price,
                'exit_price': position.exit_price,
                'realized_pnl': position.realized_pnl,
                'pnl_pct': (position.realized_pnl / (position.entry_price * position.size)) * 100,
                'holding_period_hours': holding_period,
                'status': position.status.value,
                'entry_time': position.entry_time,
                'exit_time': position.exit_time
            })

        return pd.DataFrame(data)

    def save_state(self, filepath: str):
        """
        Save position manager state to file.

        Args:
            filepath: Path to save file
        """
        state = {
            'open_positions': [pos.to_dict() for pos in self.open_positions.values()],
            'closed_positions': [pos.to_dict() for pos in self.closed_positions]
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved position manager state to {filepath}")

    def load_state(self, filepath: str):
        """
        Load position manager state from file.

        Args:
            filepath: Path to load file
        """
        import json
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Reconstruct positions
        # Note: This is a simplified version - full implementation would properly
        # reconstruct Position objects with all fields
        logger.info(f"Loaded position manager state from {filepath}")

    def __repr__(self) -> str:
        return (
            f"PositionManager(open={len(self.open_positions)}, "
            f"closed={len(self.closed_positions)}, "
            f"total_pnl={self.get_total_pnl():.2f})"
        )

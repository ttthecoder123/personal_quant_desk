"""
Corporate Actions Handler

Handles corporate actions for backtesting:
- Stock splits (forward and reverse)
- Dividends (cash and stock)
- Price adjustments
- Position adjustments
- Ex-date handling
- Historical corporate action tracking

Critical for accurate backtesting as corporate actions affect:
- Historical prices
- Position sizes
- Returns calculation
- Performance attribution
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class CorporateActionType(Enum):
    """Type of corporate action."""
    SPLIT = "SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    CASH_DIVIDEND = "CASH_DIVIDEND"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    RIGHTS_ISSUE = "RIGHTS_ISSUE"
    SPINOFF = "SPINOFF"
    MERGER = "MERGER"


@dataclass
class Split:
    """
    Stock split representation.

    Attributes:
        symbol: Stock symbol
        ex_date: Ex-split date (when split takes effect)
        announcement_date: Announcement date
        ratio: Split ratio (e.g., 2.0 for 2-for-1 split)
        is_reverse: Whether this is a reverse split
    """
    symbol: str
    ex_date: date
    ratio: float
    announcement_date: Optional[date] = None
    is_reverse: bool = False

    def __post_init__(self):
        """Validate split parameters."""
        if self.ratio <= 0:
            raise ValueError("Split ratio must be positive")

        # Determine if reverse split
        if not self.is_reverse and self.ratio < 1.0:
            self.is_reverse = True
            logger.warning(f"Detected reverse split for {self.symbol}: ratio={self.ratio}")

    def adjust_price(self, price: float) -> float:
        """
        Adjust price for split.

        Args:
            price: Pre-split price

        Returns:
            Post-split price
        """
        if self.is_reverse:
            # Reverse split: price increases
            return price * self.ratio
        else:
            # Forward split: price decreases
            return price / self.ratio

    def adjust_quantity(self, quantity: float) -> float:
        """
        Adjust position quantity for split.

        Args:
            quantity: Pre-split quantity

        Returns:
            Post-split quantity
        """
        if self.is_reverse:
            # Reverse split: quantity decreases
            return quantity / self.ratio
        else:
            # Forward split: quantity increases
            return quantity * self.ratio

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'ex_date': self.ex_date,
            'announcement_date': self.announcement_date,
            'ratio': self.ratio,
            'is_reverse': self.is_reverse,
            'type': 'REVERSE_SPLIT' if self.is_reverse else 'SPLIT'
        }


@dataclass
class Dividend:
    """
    Dividend representation.

    Attributes:
        symbol: Stock symbol
        ex_date: Ex-dividend date
        payment_date: Payment date
        amount: Dividend amount (per share for cash, ratio for stock)
        is_cash: True for cash dividend, False for stock dividend
        currency: Currency for cash dividends
    """
    symbol: str
    ex_date: date
    amount: float
    payment_date: Optional[date] = None
    declaration_date: Optional[date] = None
    is_cash: bool = True
    currency: str = "USD"

    def __post_init__(self):
        """Validate dividend parameters."""
        if self.amount < 0:
            raise ValueError("Dividend amount must be non-negative")

    def calculate_dividend_payment(self, shares_held: float) -> float:
        """
        Calculate total dividend payment.

        Args:
            shares_held: Number of shares held

        Returns:
            Total dividend payment
        """
        if self.is_cash:
            return self.amount * shares_held
        else:
            # Stock dividend returns additional shares
            return self.amount * shares_held

    def adjust_price_for_dividend(self, price: float) -> float:
        """
        Adjust price for dividend (ex-dividend adjustment).

        For cash dividends, price typically drops by dividend amount on ex-date.

        Args:
            price: Pre-dividend price

        Returns:
            Ex-dividend price
        """
        if self.is_cash:
            return price - self.amount
        else:
            # Stock dividend dilutes price
            return price / (1 + self.amount)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'ex_date': self.ex_date,
            'payment_date': self.payment_date,
            'declaration_date': self.declaration_date,
            'amount': self.amount,
            'is_cash': self.is_cash,
            'currency': self.currency,
            'type': 'CASH_DIVIDEND' if self.is_cash else 'STOCK_DIVIDEND'
        }


class CorporateActionHandler:
    """
    Handles corporate actions for backtesting.

    Maintains history of corporate actions and applies appropriate
    adjustments to prices and positions.
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize corporate action handler.

        Args:
            symbols: List of symbols to track (None = track all)
        """
        self.symbols = set(symbols) if symbols else set()
        self.splits: Dict[str, List[Split]] = {}
        self.dividends: Dict[str, List[Dividend]] = {}

        # Track applied adjustments
        self.adjustment_history: List[Dict] = []

        logger.info(f"Initialized CorporateActionHandler for {len(self.symbols) if self.symbols else 'all'} symbols")

    def add_split(self, split: Split):
        """
        Add stock split.

        Args:
            split: Split instance
        """
        if self.symbols and split.symbol not in self.symbols:
            logger.debug(f"Ignoring split for {split.symbol} (not in tracked symbols)")
            return

        if split.symbol not in self.splits:
            self.splits[split.symbol] = []

        self.splits[split.symbol].append(split)
        self.splits[split.symbol].sort(key=lambda s: s.ex_date)

        logger.info(
            f"Added {'reverse ' if split.is_reverse else ''}split for {split.symbol}: "
            f"{split.ratio}:1 on {split.ex_date}"
        )

    def add_dividend(self, dividend: Dividend):
        """
        Add dividend.

        Args:
            dividend: Dividend instance
        """
        if self.symbols and dividend.symbol not in self.symbols:
            logger.debug(f"Ignoring dividend for {dividend.symbol} (not in tracked symbols)")
            return

        if dividend.symbol not in self.dividends:
            self.dividends[dividend.symbol] = []

        self.dividends[dividend.symbol].append(dividend)
        self.dividends[dividend.symbol].sort(key=lambda d: d.ex_date)

        logger.info(
            f"Added {'cash' if dividend.is_cash else 'stock'} dividend for {dividend.symbol}: "
            f"${dividend.amount if dividend.is_cash else dividend.amount} on {dividend.ex_date}"
        )

    def get_splits(self, symbol: str, start_date: Optional[date] = None,
                   end_date: Optional[date] = None) -> List[Split]:
        """
        Get splits for a symbol within date range.

        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of splits
        """
        if symbol not in self.splits:
            return []

        splits = self.splits[symbol]

        if start_date:
            splits = [s for s in splits if s.ex_date >= start_date]
        if end_date:
            splits = [s for s in splits if s.ex_date <= end_date]

        return splits

    def get_dividends(self, symbol: str, start_date: Optional[date] = None,
                     end_date: Optional[date] = None) -> List[Dividend]:
        """
        Get dividends for a symbol within date range.

        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of dividends
        """
        if symbol not in self.dividends:
            return []

        dividends = self.dividends[symbol]

        if start_date:
            dividends = [d for d in dividends if d.ex_date >= start_date]
        if end_date:
            dividends = [d for d in dividends if d.ex_date <= end_date]

        return dividends

    def adjust_prices_for_splits(self, symbol: str, prices: pd.Series,
                                 as_of_date: Optional[date] = None) -> pd.Series:
        """
        Adjust historical prices for splits.

        Adjusts all prices before split ex-date to be split-adjusted.

        Args:
            symbol: Stock symbol
            prices: Price series (index must be dates)
            as_of_date: Adjust as of this date (None = all splits)

        Returns:
            Adjusted price series
        """
        if symbol not in self.splits:
            return prices

        adjusted = prices.copy()

        # Get relevant splits
        splits = self.splits[symbol]
        if as_of_date:
            splits = [s for s in splits if s.ex_date <= as_of_date]

        # Apply splits in reverse chronological order
        for split in reversed(splits):
            # Adjust prices before ex-date
            mask = adjusted.index < pd.Timestamp(split.ex_date)
            adjusted[mask] = adjusted[mask].apply(split.adjust_price)

            logger.debug(
                f"Applied {split.ratio}:1 split adjustment for {symbol} "
                f"on {split.ex_date}"
            )

        return adjusted

    def adjust_prices_for_dividends(self, symbol: str, prices: pd.Series,
                                   as_of_date: Optional[date] = None) -> pd.Series:
        """
        Adjust historical prices for dividends (total return adjustment).

        Args:
            symbol: Stock symbol
            prices: Price series
            as_of_date: Adjust as of this date (None = all dividends)

        Returns:
            Adjusted price series
        """
        if symbol not in self.dividends:
            return prices

        adjusted = prices.copy()

        # Get relevant cash dividends
        dividends = [d for d in self.dividends[symbol] if d.is_cash]
        if as_of_date:
            dividends = [d for d in dividends if d.ex_date <= as_of_date]

        # Apply dividends in reverse chronological order
        for dividend in reversed(dividends):
            # Adjust prices before ex-date
            mask = adjusted.index < pd.Timestamp(dividend.ex_date)
            adjustment_factor = 1 - (dividend.amount / adjusted[adjusted.index >= pd.Timestamp(dividend.ex_date)].iloc[0])
            adjusted[mask] = adjusted[mask] * adjustment_factor

        return adjusted

    def adjust_position_for_split(self, symbol: str, quantity: float,
                                  avg_price: float, split_date: date) -> Tuple[float, float]:
        """
        Adjust position for a split occurring on split_date.

        Args:
            symbol: Stock symbol
            quantity: Current position quantity
            avg_price: Average entry price
            split_date: Date of split

        Returns:
            Tuple of (adjusted_quantity, adjusted_price)
        """
        splits = self.get_splits(symbol, start_date=split_date, end_date=split_date)

        if not splits:
            return quantity, avg_price

        # Apply all splits on this date
        for split in splits:
            new_quantity = split.adjust_quantity(quantity)
            new_price = split.adjust_price(avg_price)

            logger.info(
                f"Split adjustment for {symbol}: "
                f"quantity {quantity} -> {new_quantity}, "
                f"price ${avg_price:.2f} -> ${new_price:.2f}"
            )

            self.adjustment_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': 'SPLIT',
                'split_ratio': split.ratio,
                'old_quantity': quantity,
                'new_quantity': new_quantity,
                'old_price': avg_price,
                'new_price': new_price
            })

            quantity = new_quantity
            avg_price = new_price

        return quantity, avg_price

    def process_dividend_payment(self, symbol: str, quantity: float,
                                dividend_date: date) -> Dict[str, float]:
        """
        Process dividend payment.

        Args:
            symbol: Stock symbol
            quantity: Number of shares held
            dividend_date: Date to check for dividends

        Returns:
            Dictionary with dividend payment details
        """
        dividends = self.get_dividends(symbol, start_date=dividend_date, end_date=dividend_date)

        total_cash = 0.0
        additional_shares = 0.0

        for dividend in dividends:
            if dividend.is_cash:
                payment = dividend.calculate_dividend_payment(quantity)
                total_cash += payment

                logger.info(
                    f"Cash dividend for {symbol}: "
                    f"{quantity} shares × ${dividend.amount} = ${payment:.2f}"
                )
            else:
                # Stock dividend
                new_shares = dividend.calculate_dividend_payment(quantity)
                additional_shares += new_shares

                logger.info(
                    f"Stock dividend for {symbol}: "
                    f"{quantity} shares × {dividend.amount} = {new_shares} new shares"
                )

            self.adjustment_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'type': 'DIVIDEND',
                'dividend_type': 'CASH' if dividend.is_cash else 'STOCK',
                'amount': dividend.amount,
                'quantity': quantity,
                'cash_payment': total_cash if dividend.is_cash else 0,
                'additional_shares': additional_shares if not dividend.is_cash else 0
            })

        return {
            'cash_payment': total_cash,
            'additional_shares': additional_shares,
            'num_dividends': len(dividends)
        }

    def check_corporate_actions(self, symbol: str, check_date: date) -> Dict[str, List]:
        """
        Check for corporate actions on a specific date.

        Args:
            symbol: Stock symbol
            check_date: Date to check

        Returns:
            Dictionary with lists of splits and dividends
        """
        splits = self.get_splits(symbol, start_date=check_date, end_date=check_date)
        dividends = self.get_dividends(symbol, start_date=check_date, end_date=check_date)

        return {
            'splits': splits,
            'dividends': dividends,
            'has_actions': len(splits) + len(dividends) > 0
        }

    def calculate_split_factor(self, symbol: str, start_date: date, end_date: date) -> float:
        """
        Calculate cumulative split factor between two dates.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Cumulative split factor
        """
        splits = self.get_splits(symbol, start_date=start_date, end_date=end_date)

        factor = 1.0
        for split in splits:
            if split.is_reverse:
                factor /= split.ratio
            else:
                factor *= split.ratio

        return factor

    def get_adjustment_history(self, symbol: Optional[str] = None,
                              action_type: Optional[str] = None) -> List[Dict]:
        """
        Get history of applied adjustments.

        Args:
            symbol: Filter by symbol (None = all)
            action_type: Filter by type (None = all)

        Returns:
            List of adjustment records
        """
        history = self.adjustment_history

        if symbol:
            history = [h for h in history if h['symbol'] == symbol]

        if action_type:
            history = [h for h in history if h['type'] == action_type]

        return history

    def load_from_dataframe(self, df: pd.DataFrame, action_type: str):
        """
        Load corporate actions from DataFrame.

        Args:
            df: DataFrame with corporate action data
            action_type: 'SPLIT' or 'DIVIDEND'

        Expected columns for splits:
            - symbol, ex_date, ratio, (optional: announcement_date, is_reverse)

        Expected columns for dividends:
            - symbol, ex_date, amount, (optional: payment_date, declaration_date, is_cash, currency)
        """
        if df.empty:
            return

        count = 0

        for _, row in df.iterrows():
            try:
                if action_type == 'SPLIT':
                    split = Split(
                        symbol=row['symbol'],
                        ex_date=pd.to_datetime(row['ex_date']).date(),
                        ratio=float(row['ratio']),
                        announcement_date=pd.to_datetime(row.get('announcement_date')).date() if 'announcement_date' in row else None,
                        is_reverse=bool(row.get('is_reverse', False))
                    )
                    self.add_split(split)
                    count += 1

                elif action_type == 'DIVIDEND':
                    dividend = Dividend(
                        symbol=row['symbol'],
                        ex_date=pd.to_datetime(row['ex_date']).date(),
                        amount=float(row['amount']),
                        payment_date=pd.to_datetime(row.get('payment_date')).date() if 'payment_date' in row else None,
                        declaration_date=pd.to_datetime(row.get('declaration_date')).date() if 'declaration_date' in row else None,
                        is_cash=bool(row.get('is_cash', True)),
                        currency=row.get('currency', 'USD')
                    )
                    self.add_dividend(dividend)
                    count += 1

            except Exception as e:
                logger.error(f"Error loading corporate action: {e}")

        logger.info(f"Loaded {count} {action_type} records from DataFrame")

    def export_to_dataframe(self, action_type: Optional[str] = None) -> pd.DataFrame:
        """
        Export corporate actions to DataFrame.

        Args:
            action_type: 'SPLIT', 'DIVIDEND', or None for all

        Returns:
            DataFrame with corporate actions
        """
        records = []

        if action_type in [None, 'SPLIT']:
            for symbol, splits in self.splits.items():
                for split in splits:
                    records.append(split.to_dict())

        if action_type in [None, 'DIVIDEND']:
            for symbol, dividends in self.dividends.items():
                for dividend in dividends:
                    records.append(dividend.to_dict())

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values(['symbol', 'ex_date'])

        return df

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for corporate actions.

        Returns:
            Dictionary with summary statistics
        """
        num_symbols_with_splits = len(self.splits)
        num_symbols_with_dividends = len(self.dividends)
        total_splits = sum(len(splits) for splits in self.splits.values())
        total_dividends = sum(len(divs) for divs in self.dividends.values())

        return {
            'num_symbols_with_splits': num_symbols_with_splits,
            'num_symbols_with_dividends': num_symbols_with_dividends,
            'total_splits': total_splits,
            'total_dividends': total_dividends,
            'total_adjustments': len(self.adjustment_history),
            'tracked_symbols': len(self.symbols) if self.symbols else 'all'
        }

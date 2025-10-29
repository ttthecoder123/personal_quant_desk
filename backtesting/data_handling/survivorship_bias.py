"""
Survivorship Bias Correction

Implements survivorship bias correction for backtesting:
- Include delisted securities
- Point-in-time universe selection
- IPO date handling
- Index membership tracking
- Merger/acquisition handling
- Bankruptcy modeling
- Simulate realistic universe changes

Survivorship bias is one of the most common sources of overly optimistic
backtest results. This module ensures only securities that actually existed
at each point in time are included in the analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from loguru import logger


class DelistingReason(Enum):
    """Reason for delisting."""
    BANKRUPTCY = "BANKRUPTCY"
    MERGER = "MERGER"
    ACQUISITION = "ACQUISITION"
    PRIVATIZATION = "PRIVATIZATION"
    REGULATORY = "REGULATORY"
    VOLUNTARY = "VOLUNTARY"
    OTHER = "OTHER"


class SecurityStatus(Enum):
    """Security trading status."""
    ACTIVE = "ACTIVE"
    DELISTED = "DELISTED"
    SUSPENDED = "SUSPENDED"
    PENDING_IPO = "PENDING_IPO"


@dataclass
class SecurityLifecycle:
    """
    Lifecycle information for a security.

    Attributes:
        symbol: Trading symbol
        ipo_date: Initial public offering date
        delisting_date: Date security was delisted
        delisting_reason: Reason for delisting
        final_price: Final trading price
        merger_acquirer: Acquiring company (if applicable)
        status: Current status
    """
    symbol: str
    ipo_date: Optional[datetime] = None
    delisting_date: Optional[datetime] = None
    delisting_reason: Optional[DelistingReason] = None
    final_price: Optional[float] = None
    merger_acquirer: Optional[str] = None
    status: SecurityStatus = SecurityStatus.ACTIVE

    def is_tradeable(self, as_of_date: datetime) -> bool:
        """
        Check if security is tradeable as of date.

        Args:
            as_of_date: Date to check

        Returns:
            True if tradeable
        """
        # Not yet IPO'd
        if self.ipo_date and as_of_date < self.ipo_date:
            return False

        # Already delisted
        if self.delisting_date and as_of_date >= self.delisting_date:
            return False

        # Active or suspended (suspended still has data)
        return self.status in [SecurityStatus.ACTIVE, SecurityStatus.SUSPENDED]


class SurvivorshipBiasHandler:
    """
    Primary survivorship bias handler.

    Maintains database of security lifecycles and filters
    data to only include tradeable securities at each point in time.
    """

    def __init__(self):
        """Initialize survivorship bias handler."""
        self.securities: Dict[str, SecurityLifecycle] = {}
        logger.info("Initialized SurvivorshipBiasHandler")

    def add_security(
        self,
        symbol: str,
        ipo_date: Optional[datetime] = None,
        delisting_date: Optional[datetime] = None,
        delisting_reason: Optional[DelistingReason] = None,
        final_price: Optional[float] = None,
        merger_acquirer: Optional[str] = None
    ):
        """
        Add security lifecycle information.

        Args:
            symbol: Trading symbol
            ipo_date: IPO date
            delisting_date: Delisting date
            delisting_reason: Reason for delisting
            final_price: Final price before delisting
            merger_acquirer: Acquiring company
        """
        status = SecurityStatus.ACTIVE

        if delisting_date and delisting_date <= datetime.now():
            status = SecurityStatus.DELISTED
        elif ipo_date and ipo_date > datetime.now():
            status = SecurityStatus.PENDING_IPO

        self.securities[symbol] = SecurityLifecycle(
            symbol=symbol,
            ipo_date=ipo_date,
            delisting_date=delisting_date,
            delisting_reason=delisting_reason,
            final_price=final_price,
            merger_acquirer=merger_acquirer,
            status=status
        )

        logger.debug(f"Added {symbol}: IPO={ipo_date}, Delisting={delisting_date}")

    def get_tradeable_universe(
        self,
        as_of_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> Set[str]:
        """
        Get set of tradeable symbols as of date.

        Args:
            as_of_date: Date to query
            symbols: Optional list to filter (None = all)

        Returns:
            Set of tradeable symbols
        """
        if symbols is None:
            symbols = list(self.securities.keys())

        tradeable = set()

        for symbol in symbols:
            if symbol in self.securities:
                lifecycle = self.securities[symbol]
                if lifecycle.is_tradeable(as_of_date):
                    tradeable.add(symbol)
            else:
                # If no lifecycle info, assume tradeable
                tradeable.add(symbol)

        return tradeable

    def filter_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        as_of_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter data to remove survivorship bias.

        For each timestamp, only include securities that were
        actually tradeable at that time.

        Args:
            data_dict: Dictionary of DataFrames
            as_of_date: If provided, filter for single date

        Returns:
            Filtered dictionary
        """
        if as_of_date:
            # Single date filter
            tradeable = self.get_tradeable_universe(as_of_date)
            return {
                symbol: df for symbol, df in data_dict.items()
                if symbol in tradeable
            }

        # Filter each symbol's data by its lifecycle
        filtered = {}

        for symbol, df in data_dict.items():
            if df.empty:
                continue

            if symbol not in self.securities:
                # No lifecycle info, include all data
                filtered[symbol] = df
                continue

            lifecycle = self.securities[symbol]

            # Filter by IPO date
            if lifecycle.ipo_date:
                df = df[df.index >= lifecycle.ipo_date]

            # Filter by delisting date
            if lifecycle.delisting_date:
                df = df[df.index < lifecycle.delisting_date]

            if not df.empty:
                filtered[symbol] = df

        removed_count = len(data_dict) - len(filtered)
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} securities for survivorship bias")

        return filtered

    def get_delisted_securities(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[SecurityLifecycle]:
        """
        Get securities delisted during period.

        Args:
            start_date: Period start
            end_date: Period end

        Returns:
            List of delisted securities
        """
        delisted = []

        for security in self.securities.values():
            if security.delisting_date:
                if start_date <= security.delisting_date <= end_date:
                    delisted.append(security)

        logger.info(
            f"Found {len(delisted)} delistings between {start_date} and {end_date}"
        )

        return delisted

    def simulate_delisting_return(
        self,
        symbol: str,
        prior_price: float
    ) -> float:
        """
        Simulate return from delisting event.

        Args:
            symbol: Symbol
            prior_price: Price before delisting

        Returns:
            Delisting return (often negative)
        """
        if symbol not in self.securities:
            return 0.0

        lifecycle = self.securities[symbol]

        if lifecycle.final_price is not None:
            # Use actual final price
            return (lifecycle.final_price - prior_price) / prior_price

        # Estimate based on delisting reason
        if lifecycle.delisting_reason == DelistingReason.BANKRUPTCY:
            # Bankruptcy: -100% or close to it
            return -0.95

        elif lifecycle.delisting_reason == DelistingReason.MERGER:
            # Mergers often at premium
            return 0.20

        elif lifecycle.delisting_reason == DelistingReason.ACQUISITION:
            # Acquisitions at premium
            return 0.25

        elif lifecycle.delisting_reason == DelistingReason.PRIVATIZATION:
            # Usually at premium
            return 0.15

        else:
            # Default assumption: small negative
            return -0.10


class UniverseManager:
    """
    Manage point-in-time universe selection.

    Track universe membership over time for:
    - Indices (S&P 500, Russell 2000, etc.)
    - Sectors
    - Custom universes
    """

    def __init__(self):
        """Initialize universe manager."""
        self.universe_history: Dict[str, List[Dict]] = {}
        logger.info("Initialized UniverseManager")

    def add_universe_snapshot(
        self,
        universe_name: str,
        as_of_date: datetime,
        members: Set[str]
    ):
        """
        Add universe snapshot.

        Args:
            universe_name: Universe identifier
            as_of_date: Snapshot date
            members: Set of member symbols
        """
        if universe_name not in self.universe_history:
            self.universe_history[universe_name] = []

        self.universe_history[universe_name].append({
            'date': as_of_date,
            'members': members.copy()
        })

        # Keep sorted by date
        self.universe_history[universe_name].sort(key=lambda x: x['date'])

        logger.debug(
            f"Added {universe_name} snapshot: {as_of_date}, "
            f"{len(members)} members"
        )

    def get_universe(
        self,
        universe_name: str,
        as_of_date: datetime
    ) -> Set[str]:
        """
        Get universe members as of date.

        Args:
            universe_name: Universe identifier
            as_of_date: Query date

        Returns:
            Set of member symbols
        """
        if universe_name not in self.universe_history:
            logger.warning(f"Unknown universe: {universe_name}")
            return set()

        # Find most recent snapshot before as_of_date
        snapshots = self.universe_history[universe_name]

        for snapshot in reversed(snapshots):
            if snapshot['date'] <= as_of_date:
                return snapshot['members'].copy()

        # No snapshot before date
        logger.warning(
            f"No {universe_name} snapshot before {as_of_date}"
        )
        return set()

    def track_universe_changes(
        self,
        universe_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Track universe composition changes.

        Args:
            universe_name: Universe identifier
            start_date: Period start
            end_date: Period end

        Returns:
            DataFrame with additions/deletions
        """
        if universe_name not in self.universe_history:
            return pd.DataFrame()

        snapshots = [
            s for s in self.universe_history[universe_name]
            if start_date <= s['date'] <= end_date
        ]

        if len(snapshots) < 2:
            return pd.DataFrame()

        changes = []

        for i in range(1, len(snapshots)):
            prev_members = snapshots[i-1]['members']
            curr_members = snapshots[i]['members']

            additions = curr_members - prev_members
            deletions = prev_members - curr_members

            if additions or deletions:
                changes.append({
                    'date': snapshots[i]['date'],
                    'additions': list(additions),
                    'deletions': list(deletions),
                    'net_change': len(additions) - len(deletions)
                })

        if changes:
            df = pd.DataFrame(changes)
            logger.info(
                f"Found {len(df)} universe changes for {universe_name}"
            )
            return df

        return pd.DataFrame()


class IPOHandler:
    """
    Handle IPO date tracking and filtering.

    Ensures securities are only included after their IPO date.
    """

    def __init__(self):
        """Initialize IPO handler."""
        self.ipo_dates: Dict[str, datetime] = {}
        logger.info("Initialized IPOHandler")

    def add_ipo(self, symbol: str, ipo_date: datetime):
        """
        Record IPO date.

        Args:
            symbol: Trading symbol
            ipo_date: IPO date
        """
        self.ipo_dates[symbol] = ipo_date
        logger.debug(f"Recorded IPO: {symbol} on {ipo_date}")

    def filter_pre_ipo(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter out pre-IPO data.

        Args:
            data_dict: Dictionary of DataFrames

        Returns:
            Filtered dictionary
        """
        filtered = {}

        for symbol, df in data_dict.items():
            if df.empty:
                continue

            if symbol in self.ipo_dates:
                ipo_date = self.ipo_dates[symbol]
                df = df[df.index >= ipo_date]

                if not df.empty:
                    filtered[symbol] = df
            else:
                filtered[symbol] = df

        return filtered

    def get_recent_ipos(
        self,
        as_of_date: datetime,
        lookback_days: int = 90
    ) -> List[Tuple[str, datetime]]:
        """
        Get recent IPOs.

        Args:
            as_of_date: Reference date
            lookback_days: Days to look back

        Returns:
            List of (symbol, ipo_date) tuples
        """
        cutoff_date = as_of_date - timedelta(days=lookback_days)

        recent = [
            (symbol, date) for symbol, date in self.ipo_dates.items()
            if cutoff_date <= date <= as_of_date
        ]

        recent.sort(key=lambda x: x[1], reverse=True)

        return recent


class MergerHandler:
    """
    Handle merger and acquisition events.

    Track M&A events and adjust returns/positions accordingly.
    """

    def __init__(self):
        """Initialize merger handler."""
        self.merger_events: List[Dict] = []
        logger.info("Initialized MergerHandler")

    def add_merger(
        self,
        target_symbol: str,
        acquirer_symbol: str,
        announcement_date: datetime,
        completion_date: datetime,
        exchange_ratio: Optional[float] = None,
        cash_component: Optional[float] = None
    ):
        """
        Record merger event.

        Args:
            target_symbol: Target company symbol
            acquirer_symbol: Acquiring company symbol
            announcement_date: Announcement date
            completion_date: Deal completion date
            exchange_ratio: Stock exchange ratio
            cash_component: Cash payment per share
        """
        self.merger_events.append({
            'target': target_symbol,
            'acquirer': acquirer_symbol,
            'announcement_date': announcement_date,
            'completion_date': completion_date,
            'exchange_ratio': exchange_ratio,
            'cash_component': cash_component
        })

        logger.debug(
            f"Recorded merger: {target_symbol} → {acquirer_symbol} "
            f"on {completion_date}"
        )

    def get_merger_consideration(
        self,
        target_symbol: str,
        as_of_date: datetime,
        acquirer_price: float
    ) -> Optional[float]:
        """
        Calculate merger consideration value.

        Args:
            target_symbol: Target symbol
            as_of_date: Valuation date
            acquirer_price: Acquirer stock price

        Returns:
            Per-share value of consideration
        """
        for event in self.merger_events:
            if event['target'] != target_symbol:
                continue

            # Check if merger is completed
            if event['completion_date'] > as_of_date:
                continue

            value = 0.0

            # Stock component
            if event['exchange_ratio']:
                value += event['exchange_ratio'] * acquirer_price

            # Cash component
            if event['cash_component']:
                value += event['cash_component']

            return value

        return None

    def get_pending_mergers(self, as_of_date: datetime) -> List[Dict]:
        """
        Get announced but not yet completed mergers.

        Args:
            as_of_date: Reference date

        Returns:
            List of pending merger events
        """
        pending = [
            event for event in self.merger_events
            if event['announcement_date'] <= as_of_date < event['completion_date']
        ]

        return pending


class BankruptcyHandler:
    """
    Handle bankruptcy events.

    Model realistic losses from bankruptcies.
    """

    def __init__(self):
        """Initialize bankruptcy handler."""
        self.bankruptcy_events: Dict[str, Dict] = {}
        logger.info("Initialized BankruptcyHandler")

    def add_bankruptcy(
        self,
        symbol: str,
        filing_date: datetime,
        chapter: int = 11,
        recovery_rate: float = 0.0
    ):
        """
        Record bankruptcy filing.

        Args:
            symbol: Trading symbol
            filing_date: Bankruptcy filing date
            chapter: Chapter 7 or 11
            recovery_rate: Expected recovery rate (0-1)
        """
        self.bankruptcy_events[symbol] = {
            'filing_date': filing_date,
            'chapter': chapter,
            'recovery_rate': recovery_rate
        }

        logger.debug(
            f"Recorded bankruptcy: {symbol} Chapter {chapter} "
            f"on {filing_date}"
        )

    def calculate_bankruptcy_loss(
        self,
        symbol: str,
        position_value: float
    ) -> float:
        """
        Calculate loss from bankruptcy.

        Args:
            symbol: Trading symbol
            position_value: Position value before bankruptcy

        Returns:
            Loss amount (positive = loss)
        """
        if symbol not in self.bankruptcy_events:
            return 0.0

        event = self.bankruptcy_events[symbol]
        recovery_rate = event['recovery_rate']

        # Loss = position value × (1 - recovery rate)
        loss = position_value * (1 - recovery_rate)

        logger.info(
            f"Bankruptcy loss for {symbol}: "
            f"${position_value:.2f} → ${loss:.2f}"
        )

        return loss

    def is_bankrupt(self, symbol: str, as_of_date: datetime) -> bool:
        """
        Check if security is bankrupt as of date.

        Args:
            symbol: Trading symbol
            as_of_date: Query date

        Returns:
            True if bankrupt
        """
        if symbol not in self.bankruptcy_events:
            return False

        return self.bankruptcy_events[symbol]['filing_date'] <= as_of_date


class ComprehensiveSurvivorshipModel:
    """
    Comprehensive survivorship bias model.

    Integrates all components:
    - Security lifecycles
    - Universe tracking
    - IPO handling
    - Merger handling
    - Bankruptcy handling

    Most complete model for production backtesting.
    """

    def __init__(self):
        """Initialize comprehensive model."""
        self.survivorship_handler = SurvivorshipBiasHandler()
        self.universe_manager = UniverseManager()
        self.ipo_handler = IPOHandler()
        self.merger_handler = MergerHandler()
        self.bankruptcy_handler = BankruptcyHandler()

        logger.info("Initialized ComprehensiveSurvivorshipModel")

    def filter_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        universe: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply all survivorship filters.

        Args:
            data_dict: Dictionary of DataFrames
            universe: Optional universe constraint

        Returns:
            Filtered dictionary
        """
        # Filter by lifecycle
        filtered = self.survivorship_handler.filter_data(data_dict)

        # Filter pre-IPO
        filtered = self.ipo_handler.filter_pre_ipo(filtered)

        # Filter by universe if specified
        if universe:
            # Get date range
            dates = []
            for df in filtered.values():
                if not df.empty:
                    dates.extend(df.index.tolist())

            if dates:
                min_date = min(dates)
                max_date = max(dates)

                # Filter to universe members over time
                universe_filtered = {}
                for symbol, df in filtered.items():
                    # For simplicity, check at start date
                    # Production would check at each timestamp
                    members = self.universe_manager.get_universe(universe, min_date)
                    if symbol in members:
                        universe_filtered[symbol] = df

                filtered = universe_filtered

        return filtered

    def get_tradeable_universe(
        self,
        as_of_date: datetime,
        universe: Optional[str] = None
    ) -> Set[str]:
        """
        Get complete tradeable universe.

        Args:
            as_of_date: Query date
            universe: Optional universe constraint

        Returns:
            Set of tradeable symbols
        """
        # Get tradeable securities
        tradeable = self.survivorship_handler.get_tradeable_universe(as_of_date)

        # Filter by universe
        if universe:
            universe_members = self.universe_manager.get_universe(universe, as_of_date)
            tradeable = tradeable.intersection(universe_members)

        return tradeable

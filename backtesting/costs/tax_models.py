"""
Tax Impact Models

Implements tax impact models for backtesting:
- Short-term vs long-term capital gains
- Wash sale rules
- First-in-first-out (FIFO) vs specific identification
- Mark-to-market election (Section 475)
- Tax loss harvesting opportunities

Tax considerations can significantly impact after-tax returns.
Proper tax modeling is essential for realistic backtesting,
especially for taxable accounts.

Note: This models US tax law. Consult tax professionals for
specific situations. Tax laws change frequently.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, List, Tuple
from collections import deque
import pandas as pd
from loguru import logger


class TaxTreatment(Enum):
    """Tax treatment classification."""
    SHORT_TERM_GAINS = "SHORT_TERM_GAINS"  # Held ≤1 year
    LONG_TERM_GAINS = "LONG_TERM_GAINS"  # Held >1 year
    WASH_SALE_DISALLOWED = "WASH_SALE_DISALLOWED"
    SECTION_475_MTM = "SECTION_475_MTM"  # Mark-to-market


class LotMatchingMethod(Enum):
    """Lot matching method for cost basis."""
    FIFO = "FIFO"  # First-In-First-Out
    LIFO = "LIFO"  # Last-In-First-Out
    SPECIFIC_ID = "SPECIFIC_ID"  # Specific identification
    HIFO = "HIFO"  # Highest-In-First-Out (tax optimization)
    MIN_TAX = "MIN_TAX"  # Minimize current tax liability


@dataclass
class TaxLot:
    """
    Individual tax lot.

    Represents a specific purchase of shares with:
    - Purchase date
    - Quantity
    - Cost basis
    - Holding period tracking
    """
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: datetime
    lot_id: str = ""

    def __post_init__(self):
        """Generate lot ID if not provided."""
        if not self.lot_id:
            self.lot_id = f"{self.symbol}_{self.purchase_date.strftime('%Y%m%d_%H%M%S')}"

    @property
    def cost_basis(self) -> float:
        """Total cost basis for lot."""
        return self.quantity * self.purchase_price

    def holding_period_days(self, as_of_date: datetime) -> int:
        """
        Calculate holding period in days.

        Args:
            as_of_date: Date to calculate holding period to

        Returns:
            Number of days held
        """
        return (as_of_date - self.purchase_date).days

    def is_long_term(self, as_of_date: datetime) -> bool:
        """
        Check if holding qualifies as long-term (>1 year).

        Args:
            as_of_date: Date to check

        Returns:
            True if long-term holding
        """
        return self.holding_period_days(as_of_date) > 365


@dataclass
class TaxResult:
    """
    Result of tax calculation.

    Attributes:
        gross_proceeds: Total proceeds from sale
        cost_basis: Cost basis of sold shares
        realized_gain: Realized gain/loss
        short_term_gain: Short-term capital gain
        long_term_gain: Long-term capital gain
        wash_sale_loss_disallowed: Wash sale loss disallowed
        tax_liability: Estimated tax liability
        effective_tax_rate: Effective tax rate on gain
        breakdown: Detailed breakdown
    """
    gross_proceeds: float
    cost_basis: float
    realized_gain: float
    short_term_gain: float = 0.0
    long_term_gain: float = 0.0
    wash_sale_loss_disallowed: float = 0.0
    tax_liability: float = 0.0
    effective_tax_rate: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate totals and populate breakdown."""
        self.realized_gain = self.gross_proceeds - self.cost_basis

        self.breakdown = {
            'gross_proceeds': self.gross_proceeds,
            'cost_basis': self.cost_basis,
            'realized_gain': self.realized_gain,
            'short_term_gain': self.short_term_gain,
            'long_term_gain': self.long_term_gain,
            'wash_sale_loss_disallowed': self.wash_sale_loss_disallowed,
            'tax_liability': self.tax_liability
        }


class CapitalGainsTaxModel:
    """
    Capital gains tax model.

    Models US federal capital gains tax:
    - Short-term gains: Taxed as ordinary income
    - Long-term gains: Preferential rates (0%, 15%, 20%)

    Does not include:
    - State taxes
    - Net Investment Income Tax (3.8%)
    - Alternative Minimum Tax
    """

    def __init__(
        self,
        short_term_rate: float = 0.37,  # Top federal rate
        long_term_rate: float = 0.20,  # Top LTCG rate
        include_niit: bool = False,  # Net Investment Income Tax
        niit_rate: float = 0.038
    ):
        """
        Initialize capital gains tax model.

        Args:
            short_term_rate: Tax rate on short-term gains
            long_term_rate: Tax rate on long-term gains
            include_niit: Include 3.8% NIIT
            niit_rate: NIIT rate
        """
        self.short_term_rate = short_term_rate
        self.long_term_rate = long_term_rate
        self.include_niit = include_niit
        self.niit_rate = niit_rate

        logger.info(
            f"Initialized CapitalGainsTaxModel: "
            f"ST={short_term_rate*100:.1f}%, LT={long_term_rate*100:.1f}%"
        )

    def calculate_tax(
        self,
        short_term_gain: float,
        long_term_gain: float
    ) -> float:
        """
        Calculate total tax liability.

        Args:
            short_term_gain: Short-term capital gain
            long_term_gain: Long-term capital gain

        Returns:
            Total tax liability
        """
        # Short-term tax
        st_tax = max(0, short_term_gain) * self.short_term_rate

        # Long-term tax
        lt_tax = max(0, long_term_gain) * self.long_term_rate

        # NIIT (applies to both)
        niit = 0.0
        if self.include_niit:
            total_gain = max(0, short_term_gain + long_term_gain)
            niit = total_gain * self.niit_rate

        return st_tax + lt_tax + niit

    def calculate_effective_rate(
        self,
        short_term_gain: float,
        long_term_gain: float
    ) -> float:
        """
        Calculate effective tax rate.

        Args:
            short_term_gain: Short-term capital gain
            long_term_gain: Long-term capital gain

        Returns:
            Effective tax rate
        """
        total_gain = short_term_gain + long_term_gain
        if total_gain <= 0:
            return 0.0

        tax = self.calculate_tax(short_term_gain, long_term_gain)
        return tax / total_gain


class WashSaleModel:
    """
    Wash sale rule model.

    IRS wash sale rule (IRC Section 1091):
    - Cannot claim loss if you buy "substantially identical" security
      within 30 days before or after the sale
    - Disallowed loss is added to cost basis of replacement shares
    - Holding period of replacement shares includes disallowed shares

    Critical for realistic tax modeling of active strategies.
    """

    def __init__(self, wash_sale_window_days: int = 30):
        """
        Initialize wash sale model.

        Args:
            wash_sale_window_days: Days before/after for wash sale (30 per IRS)
        """
        self.window_days = wash_sale_window_days
        self.trade_history: List[Dict] = []

        logger.info(
            f"Initialized WashSaleModel: window={wash_sale_window_days} days"
        )

    def check_wash_sale(
        self,
        symbol: str,
        sale_date: datetime,
        loss_amount: float
    ) -> Tuple[bool, float]:
        """
        Check if sale triggers wash sale rule.

        Args:
            symbol: Trading symbol
            sale_date: Date of sale
            loss_amount: Loss amount (negative for loss)

        Returns:
            Tuple of (is_wash_sale, disallowed_loss)
        """
        if loss_amount >= 0:  # Not a loss
            return False, 0.0

        # Check for purchases within window
        window_start = sale_date - timedelta(days=self.window_days)
        window_end = sale_date + timedelta(days=self.window_days)

        # Look for replacement purchases
        for trade in self.trade_history:
            if (trade['symbol'] == symbol and
                trade['action'] == 'BUY' and
                window_start <= trade['date'] <= window_end and
                trade['date'] != sale_date):
                # Wash sale triggered
                return True, abs(loss_amount)

        return False, 0.0

    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        date: datetime
    ):
        """
        Record trade for wash sale tracking.

        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            quantity: Trade quantity
            price: Trade price
            date: Trade date
        """
        self.trade_history.append({
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'date': date
        })

        # Keep only recent history (memory optimization)
        cutoff = datetime.now() - timedelta(days=self.window_days * 3)
        self.trade_history = [
            t for t in self.trade_history if t['date'] > cutoff
        ]


class LotMatchingEngine:
    """
    Lot matching engine for cost basis tracking.

    Tracks individual tax lots and matches them to sales
    using specified method (FIFO, LIFO, Specific ID, etc.).

    Essential for accurate tax calculations.
    """

    def __init__(
        self,
        matching_method: LotMatchingMethod = LotMatchingMethod.FIFO,
        tax_model: Optional[CapitalGainsTaxModel] = None
    ):
        """
        Initialize lot matching engine.

        Args:
            matching_method: Method for matching lots to sales
            tax_model: Tax model for calculations
        """
        self.matching_method = matching_method
        self.tax_model = tax_model or CapitalGainsTaxModel()

        # Track lots by symbol
        self.open_lots: Dict[str, List[TaxLot]] = {}

        logger.info(
            f"Initialized LotMatchingEngine: method={matching_method.value}"
        )

    def add_lot(self, lot: TaxLot):
        """
        Add new tax lot.

        Args:
            lot: TaxLot to add
        """
        if lot.symbol not in self.open_lots:
            self.open_lots[lot.symbol] = []

        self.open_lots[lot.symbol].append(lot)

        logger.debug(
            f"Added lot: {lot.symbol} {lot.quantity} @ ${lot.purchase_price:.2f}"
        )

    def select_lots(
        self,
        symbol: str,
        quantity: float,
        sale_date: datetime,
        sale_price: float
    ) -> List[Tuple[TaxLot, float]]:
        """
        Select lots to match against sale.

        Args:
            symbol: Trading symbol
            quantity: Quantity to sell
            sale_date: Sale date
            sale_price: Sale price

        Returns:
            List of (lot, quantity_from_lot) tuples
        """
        if symbol not in self.open_lots or not self.open_lots[symbol]:
            logger.warning(f"No open lots for {symbol}")
            return []

        lots = self.open_lots[symbol].copy()
        selected: List[Tuple[TaxLot, float]] = []
        remaining_quantity = quantity

        # Sort lots based on matching method
        if self.matching_method == LotMatchingMethod.FIFO:
            lots.sort(key=lambda l: l.purchase_date)
        elif self.matching_method == LotMatchingMethod.LIFO:
            lots.sort(key=lambda l: l.purchase_date, reverse=True)
        elif self.matching_method == LotMatchingMethod.HIFO:
            # Highest cost first (minimize gain)
            lots.sort(key=lambda l: l.purchase_price, reverse=True)
        elif self.matching_method == LotMatchingMethod.MIN_TAX:
            # Prioritize long-term gains, then highest cost
            lots.sort(
                key=lambda l: (
                    not l.is_long_term(sale_date),
                    -l.purchase_price
                )
            )

        # Match lots
        for lot in lots:
            if remaining_quantity <= 0:
                break

            quantity_from_lot = min(lot.quantity, remaining_quantity)
            selected.append((lot, quantity_from_lot))
            remaining_quantity -= quantity_from_lot

        return selected

    def process_sale(
        self,
        symbol: str,
        quantity: float,
        sale_price: float,
        sale_date: datetime
    ) -> TaxResult:
        """
        Process sale and calculate tax impact.

        Args:
            symbol: Trading symbol
            quantity: Quantity sold
            sale_price: Sale price
            sale_date: Sale date

        Returns:
            TaxResult object
        """
        # Select lots
        matched_lots = self.select_lots(symbol, quantity, sale_date, sale_price)

        if not matched_lots:
            logger.warning(f"No lots available to match sale of {symbol}")
            return TaxResult(
                gross_proceeds=quantity * sale_price,
                cost_basis=0.0,
                realized_gain=quantity * sale_price
            )

        # Calculate proceeds and basis
        gross_proceeds = quantity * sale_price
        total_cost_basis = 0.0
        short_term_gain = 0.0
        long_term_gain = 0.0

        # Process each matched lot
        for lot, qty_from_lot in matched_lots:
            # Cost basis from this lot
            cost_basis = qty_from_lot * lot.purchase_price
            total_cost_basis += cost_basis

            # Proceeds allocated to this lot
            proceeds = qty_from_lot * sale_price

            # Gain/loss from this lot
            gain = proceeds - cost_basis

            # Classify as short-term or long-term
            if lot.is_long_term(sale_date):
                long_term_gain += gain
            else:
                short_term_gain += gain

            # Update or remove lot
            lot.quantity -= qty_from_lot
            if lot.quantity <= 0.0001:  # Remove depleted lot
                self.open_lots[symbol].remove(lot)

        # Calculate tax
        tax_liability = self.tax_model.calculate_tax(short_term_gain, long_term_gain)
        effective_rate = self.tax_model.calculate_effective_rate(
            short_term_gain, long_term_gain
        )

        return TaxResult(
            gross_proceeds=gross_proceeds,
            cost_basis=total_cost_basis,
            realized_gain=gross_proceeds - total_cost_basis,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            tax_liability=tax_liability,
            effective_tax_rate=effective_rate
        )

    def get_open_position(self, symbol: str) -> Dict[str, float]:
        """
        Get current open position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with position metrics
        """
        if symbol not in self.open_lots or not self.open_lots[symbol]:
            return {
                'quantity': 0.0,
                'total_cost_basis': 0.0,
                'avg_price': 0.0,
                'num_lots': 0
            }

        lots = self.open_lots[symbol]
        total_quantity = sum(lot.quantity for lot in lots)
        total_cost = sum(lot.cost_basis for lot in lots)

        return {
            'quantity': total_quantity,
            'total_cost_basis': total_cost,
            'avg_price': total_cost / total_quantity if total_quantity > 0 else 0.0,
            'num_lots': len(lots)
        }


class Section475MTMModel:
    """
    Section 475 Mark-to-Market election model.

    Traders can elect mark-to-market accounting:
    - All gains/losses treated as ordinary income
    - No capital loss limitations
    - No wash sale rules
    - Must mark positions to market at year-end

    Beneficial for active traders with consistent profits.
    """

    def __init__(self, ordinary_income_rate: float = 0.37):
        """
        Initialize Section 475 MTM model.

        Args:
            ordinary_income_rate: Ordinary income tax rate
        """
        self.ordinary_income_rate = ordinary_income_rate

        logger.info(
            f"Initialized Section475MTMModel: rate={ordinary_income_rate*100:.1f}%"
        )

    def calculate_tax(
        self,
        total_gain: float,
        unrealized_gain: float = 0.0
    ) -> float:
        """
        Calculate tax under MTM election.

        Args:
            total_gain: Total realized gain
            unrealized_gain: Unrealized gain (marked to market at year-end)

        Returns:
            Tax liability
        """
        # All gains treated as ordinary income
        taxable_income = total_gain + unrealized_gain
        tax = max(0, taxable_income) * self.ordinary_income_rate

        return tax


class TaxLossHarvestingAnalyzer:
    """
    Tax loss harvesting opportunity analyzer.

    Identifies opportunities to:
    - Realize losses to offset gains
    - Avoid wash sales
    - Optimize timing of gains/losses
    - Maximize after-tax returns

    Can be integrated into strategy for tax-aware trading.
    """

    def __init__(
        self,
        lot_engine: Optional[LotMatchingEngine] = None,
        wash_sale_model: Optional[WashSaleModel] = None,
        min_loss_threshold: float = 100.0  # Minimum loss worth harvesting
    ):
        """
        Initialize tax loss harvesting analyzer.

        Args:
            lot_engine: Lot matching engine
            wash_sale_model: Wash sale model
            min_loss_threshold: Minimum loss to consider harvesting
        """
        self.lot_engine = lot_engine or LotMatchingEngine()
        self.wash_sale_model = wash_sale_model or WashSaleModel()
        self.min_loss_threshold = min_loss_threshold

        logger.info("Initialized TaxLossHarvestingAnalyzer")

    def find_harvesting_opportunities(
        self,
        current_date: datetime,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Find tax loss harvesting opportunities.

        Args:
            current_date: Current date
            current_prices: Current prices by symbol

        Returns:
            List of harvesting opportunities
        """
        opportunities = []

        for symbol in self.lot_engine.open_lots:
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            lots = self.lot_engine.open_lots[symbol]

            for lot in lots:
                # Calculate unrealized loss
                current_value = lot.quantity * current_price
                unrealized_loss = current_value - lot.cost_basis

                # Check if loss exceeds threshold
                if unrealized_loss < -self.min_loss_threshold:
                    # Check wash sale implications
                    is_wash_sale, _ = self.wash_sale_model.check_wash_sale(
                        symbol=symbol,
                        sale_date=current_date,
                        loss_amount=unrealized_loss
                    )

                    opportunities.append({
                        'symbol': symbol,
                        'lot_id': lot.lot_id,
                        'quantity': lot.quantity,
                        'purchase_price': lot.purchase_price,
                        'current_price': current_price,
                        'unrealized_loss': unrealized_loss,
                        'holding_days': lot.holding_period_days(current_date),
                        'is_long_term': lot.is_long_term(current_date),
                        'wash_sale_risk': is_wash_sale,
                        'tax_savings': abs(unrealized_loss) * 0.37  # Estimate
                    })

        # Sort by tax savings potential
        opportunities.sort(key=lambda x: x['tax_savings'], reverse=True)

        return opportunities


class ComprehensiveTaxModel:
    """
    Comprehensive tax model combining all components.

    Includes:
    - Capital gains tax (short/long term)
    - Wash sale rules
    - Lot matching/tracking
    - Tax loss harvesting analysis

    Most complete model for production backtesting.
    """

    def __init__(
        self,
        tax_model: Optional[CapitalGainsTaxModel] = None,
        wash_sale_model: Optional[WashSaleModel] = None,
        lot_engine: Optional[LotMatchingEngine] = None,
        track_lots: bool = True
    ):
        """
        Initialize comprehensive tax model.

        Args:
            tax_model: Capital gains tax model
            wash_sale_model: Wash sale model
            lot_engine: Lot matching engine
            track_lots: Whether to track individual lots
        """
        self.tax_model = tax_model or CapitalGainsTaxModel()
        self.wash_sale_model = wash_sale_model or WashSaleModel()
        self.lot_engine = lot_engine or LotMatchingEngine(tax_model=self.tax_model)
        self.track_lots = track_lots

        logger.info("Initialized ComprehensiveTaxModel")

    def process_buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: datetime
    ):
        """
        Process buy transaction.

        Args:
            symbol: Trading symbol
            quantity: Quantity bought
            price: Purchase price
            date: Purchase date
        """
        if self.track_lots:
            lot = TaxLot(
                symbol=symbol,
                quantity=quantity,
                purchase_price=price,
                purchase_date=date
            )
            self.lot_engine.add_lot(lot)

        self.wash_sale_model.record_trade(symbol, 'BUY', quantity, price, date)

    def process_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: datetime
    ) -> TaxResult:
        """
        Process sell transaction with full tax implications.

        Args:
            symbol: Trading symbol
            quantity: Quantity sold
            price: Sale price
            date: Sale date

        Returns:
            TaxResult with full tax analysis
        """
        # Calculate base tax result
        tax_result = self.lot_engine.process_sale(symbol, quantity, price, date)

        # Check wash sale rules for losses
        if tax_result.realized_gain < 0:
            is_wash_sale, disallowed_loss = self.wash_sale_model.check_wash_sale(
                symbol=symbol,
                sale_date=date,
                loss_amount=tax_result.realized_gain
            )

            if is_wash_sale:
                tax_result.wash_sale_loss_disallowed = disallowed_loss
                # Adjust short-term gain (losses disallowed)
                if tax_result.short_term_gain < 0:
                    tax_result.short_term_gain += disallowed_loss
                elif tax_result.long_term_gain < 0:
                    tax_result.long_term_gain += disallowed_loss

                # Recalculate tax
                tax_result.tax_liability = self.tax_model.calculate_tax(
                    tax_result.short_term_gain,
                    tax_result.long_term_gain
                )

        # Record trade
        self.wash_sale_model.record_trade(symbol, 'SELL', quantity, price, date)

        return tax_result

    def get_ytd_tax_summary(self) -> Dict[str, float]:
        """
        Get year-to-date tax summary.

        Returns:
            Dictionary with YTD tax metrics
        """
        # This would aggregate all trades for the year
        # Simplified version - in production would track all trades
        return {
            'ytd_short_term_gains': 0.0,
            'ytd_long_term_gains': 0.0,
            'ytd_wash_sale_adjustments': 0.0,
            'ytd_tax_liability': 0.0
        }

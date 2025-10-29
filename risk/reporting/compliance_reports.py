"""
Regulatory Compliance Reporting System

Generates compliance and regulatory reports:
- Risk limit compliance monitoring
- Regulatory capital requirements
- Leverage ratio reporting
- Position limit monitoring
- Trade surveillance reports
- Audit trail documentation
- Breach reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import hashlib


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACH = "breach"
    CRITICAL = "critical"


class RegulationType(Enum):
    """Types of regulatory requirements"""
    RISK_LIMITS = "risk_limits"
    LEVERAGE_LIMITS = "leverage_limits"
    POSITION_LIMITS = "position_limits"
    CONCENTRATION_LIMITS = "concentration_limits"
    VAR_LIMITS = "var_limits"
    LIQUIDITY_REQUIREMENTS = "liquidity_requirements"
    CAPITAL_REQUIREMENTS = "capital_requirements"
    TRADE_REPORTING = "trade_reporting"
    BEST_EXECUTION = "best_execution"


class ReportingFrequency(Enum):
    """Regulatory reporting frequency"""
    REAL_TIME = "real_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class ComplianceRule:
    """Definition of a compliance rule"""
    rule_id: str
    rule_name: str
    regulation_type: RegulationType
    description: str
    threshold_value: float
    threshold_type: str  # 'max', 'min', 'range'
    severity: ComplianceStatus
    reporting_frequency: ReportingFrequency
    enabled: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'regulation_type': self.regulation_type.value,
            'description': self.description,
            'threshold_value': self.threshold_value,
            'threshold_type': self.threshold_type,
            'severity': self.severity.value,
            'reporting_frequency': self.reporting_frequency.value,
            'enabled': self.enabled
        }


@dataclass
class ComplianceBreach:
    """Record of a compliance breach"""
    breach_id: str
    rule_id: str
    rule_name: str
    timestamp: datetime
    current_value: float
    threshold_value: float
    deviation: float
    deviation_pct: float
    severity: ComplianceStatus
    description: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'breach_id': self.breach_id,
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'timestamp': self.timestamp.isoformat(),
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'deviation': self.deviation,
            'deviation_pct': self.deviation_pct,
            'severity': self.severity.value,
            'description': self.description,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolution_notes': self.resolution_notes
        }


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    report_date: datetime
    period_start: datetime
    period_end: datetime
    total_rules: int
    compliant_rules: int
    warning_rules: int
    breach_rules: int
    breaches: List[ComplianceBreach]
    rule_status: Dict[str, ComplianceStatus]
    summary_statistics: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'report_id': self.report_id,
            'report_date': self.report_date.isoformat(),
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_rules': self.total_rules,
            'compliant_rules': self.compliant_rules,
            'warning_rules': self.warning_rules,
            'breach_rules': self.breach_rules,
            'breaches': [b.to_dict() for b in self.breaches],
            'rule_status': {k: v.value for k, v in self.rule_status.items()},
            'summary_statistics': self.summary_statistics
        }


class ComplianceReporter:
    """
    Regulatory compliance reporting and monitoring system

    Features:
    - Real-time compliance monitoring
    - Breach detection and alerting
    - Regulatory report generation
    - Audit trail maintenance
    - Limit monitoring
    - Capital requirement calculations
    """

    def __init__(
        self,
        output_dir: str = "./compliance_reports",
        rules_config: Optional[List[ComplianceRule]] = None
    ):
        """
        Initialize compliance reporter

        Args:
            output_dir: Directory for compliance reports
            rules_config: List of compliance rules (defaults to standard rules)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize compliance rules
        self.rules = rules_config if rules_config is not None else self._get_default_rules()
        self.rules_dict = {rule.rule_id: rule for rule in self.rules}

        # Breach tracking
        self.active_breaches: List[ComplianceBreach] = []
        self.breach_history: List[ComplianceBreach] = []

        # Report history
        self.report_history: List[ComplianceReport] = []

        # Audit trail
        self.audit_trail: List[Dict] = []

    def check_compliance(
        self,
        portfolio_data: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> Dict[str, ComplianceStatus]:
        """
        Check compliance against all rules

        Args:
            portfolio_data: Current portfolio data
            risk_metrics: Current risk metrics

        Returns:
            Dictionary mapping rule_id to compliance status
        """
        compliance_status = {}
        timestamp = datetime.now()

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Get current value based on rule type
            current_value = self._get_metric_value(
                rule.regulation_type,
                portfolio_data,
                risk_metrics
            )

            # Check against threshold
            status = self._check_threshold(
                current_value,
                rule.threshold_value,
                rule.threshold_type
            )

            compliance_status[rule.rule_id] = status

            # Log breach if non-compliant
            if status in [ComplianceStatus.BREACH, ComplianceStatus.CRITICAL]:
                self._log_breach(rule, current_value, timestamp)

        # Log audit trail
        self._log_audit(
            action="compliance_check",
            details={
                'timestamp': timestamp.isoformat(),
                'status_summary': {k: v.value for k, v in compliance_status.items()}
            }
        )

        return compliance_status

    def generate_compliance_report(
        self,
        period_start: datetime,
        period_end: datetime,
        portfolio_data: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance report

        Args:
            period_start: Report period start
            period_end: Report period end
            portfolio_data: Portfolio data
            risk_metrics: Risk metrics

        Returns:
            ComplianceReport
        """
        timestamp = datetime.now()

        # Check current compliance
        rule_status = self.check_compliance(portfolio_data, risk_metrics)

        # Count status by category
        compliant_rules = sum(1 for s in rule_status.values() if s == ComplianceStatus.COMPLIANT)
        warning_rules = sum(1 for s in rule_status.values() if s == ComplianceStatus.WARNING)
        breach_rules = sum(1 for s in rule_status.values() if s == ComplianceStatus.BREACH)

        # Get breaches in period
        period_breaches = [
            b for b in self.breach_history
            if period_start <= b.timestamp <= period_end
        ]

        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(
            portfolio_data, risk_metrics, period_breaches
        )

        # Create report
        report_id = self._generate_report_id(timestamp)

        report = ComplianceReport(
            report_id=report_id,
            report_date=timestamp,
            period_start=period_start,
            period_end=period_end,
            total_rules=len(self.rules),
            compliant_rules=compliant_rules,
            warning_rules=warning_rules,
            breach_rules=breach_rules,
            breaches=period_breaches,
            rule_status=rule_status,
            summary_statistics=summary_stats
        )

        # Save report
        self.report_history.append(report)
        self._save_report(report)

        # Log audit trail
        self._log_audit(
            action="generate_report",
            details={
                'report_id': report_id,
                'period_start': period_start.isoformat(),
                'period_end': period_end.isoformat()
            }
        )

        return report

    def acknowledge_breach(
        self,
        breach_id: str,
        acknowledged_by: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge a compliance breach

        Args:
            breach_id: Breach ID to acknowledge
            acknowledged_by: User acknowledging the breach
            resolution_notes: Optional notes on resolution

        Returns:
            Success status
        """
        # Find breach in active breaches
        for breach in self.active_breaches:
            if breach.breach_id == breach_id:
                breach.acknowledged = True
                breach.acknowledged_by = acknowledged_by
                breach.acknowledged_at = datetime.now()
                breach.resolution_notes = resolution_notes

                # Log audit trail
                self._log_audit(
                    action="acknowledge_breach",
                    details={
                        'breach_id': breach_id,
                        'acknowledged_by': acknowledged_by,
                        'resolution_notes': resolution_notes
                    }
                )

                return True

        return False

    def add_compliance_rule(self, rule: ComplianceRule):
        """
        Add a new compliance rule

        Args:
            rule: ComplianceRule to add
        """
        self.rules.append(rule)
        self.rules_dict[rule.rule_id] = rule

        self._log_audit(
            action="add_rule",
            details={'rule_id': rule.rule_id, 'rule_name': rule.rule_name}
        )

    def update_compliance_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update an existing compliance rule

        Args:
            rule_id: Rule ID to update
            updates: Dictionary of fields to update

        Returns:
            Success status
        """
        if rule_id not in self.rules_dict:
            return False

        rule = self.rules_dict[rule_id]

        # Update fields
        for field, value in updates.items():
            if hasattr(rule, field):
                setattr(rule, field, value)

        self._log_audit(
            action="update_rule",
            details={'rule_id': rule_id, 'updates': updates}
        )

        return True

    def disable_compliance_rule(self, rule_id: str) -> bool:
        """
        Disable a compliance rule

        Args:
            rule_id: Rule ID to disable

        Returns:
            Success status
        """
        return self.update_compliance_rule(rule_id, {'enabled': False})

    def get_active_breaches(self) -> List[ComplianceBreach]:
        """
        Get all active (unacknowledged) breaches

        Returns:
            List of active breaches
        """
        return [b for b in self.active_breaches if not b.acknowledged]

    def get_breach_statistics(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate breach statistics for a period

        Args:
            period_start: Start of period (None = all time)
            period_end: End of period (None = now)

        Returns:
            Dictionary with breach statistics
        """
        # Filter breaches by period
        breaches = self.breach_history

        if period_start:
            breaches = [b for b in breaches if b.timestamp >= period_start]
        if period_end:
            breaches = [b for b in breaches if b.timestamp <= period_end]

        # Calculate statistics
        total_breaches = len(breaches)
        breaches_by_severity = {}
        breaches_by_rule = {}

        for breach in breaches:
            # By severity
            severity = breach.severity.value
            breaches_by_severity[severity] = breaches_by_severity.get(severity, 0) + 1

            # By rule
            rule_name = breach.rule_name
            breaches_by_rule[rule_name] = breaches_by_rule.get(rule_name, 0) + 1

        acknowledged_breaches = sum(1 for b in breaches if b.acknowledged)

        statistics = {
            'total_breaches': total_breaches,
            'acknowledged': acknowledged_breaches,
            'unacknowledged': total_breaches - acknowledged_breaches,
            'by_severity': breaches_by_severity,
            'by_rule': breaches_by_rule,
            'most_breached_rule': max(breaches_by_rule.items(), key=lambda x: x[1])[0] if breaches_by_rule else None
        }

        return statistics

    def export_audit_trail(
        self,
        filepath: Optional[str] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> str:
        """
        Export audit trail to file

        Args:
            filepath: Output file path (None = auto-generate)
            period_start: Filter start date
            period_end: Filter end date

        Returns:
            Path to exported file
        """
        # Filter audit trail
        audit_records = self.audit_trail

        if period_start:
            audit_records = [
                r for r in audit_records
                if datetime.fromisoformat(r['timestamp']) >= period_start
            ]
        if period_end:
            audit_records = [
                r for r in audit_records
                if datetime.fromisoformat(r['timestamp']) <= period_end
            ]

        # Generate filepath if not provided
        if filepath is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"audit_trail_{timestamp_str}.json"

        # Export to JSON
        with open(filepath, 'w') as f:
            json.dump(audit_records, f, indent=2)

        return str(filepath)

    def _get_default_rules(self) -> List[ComplianceRule]:
        """Get default compliance rules"""
        return [
            ComplianceRule(
                rule_id="RISK001",
                rule_name="Maximum VaR Limit",
                regulation_type=RegulationType.VAR_LIMITS,
                description="Portfolio VaR must not exceed 2% at 95% confidence",
                threshold_value=0.02,
                threshold_type="max",
                severity=ComplianceStatus.BREACH,
                reporting_frequency=ReportingFrequency.DAILY
            ),
            ComplianceRule(
                rule_id="RISK002",
                rule_name="Maximum Drawdown",
                regulation_type=RegulationType.RISK_LIMITS,
                description="Portfolio drawdown must not exceed 20%",
                threshold_value=0.20,
                threshold_type="max",
                severity=ComplianceStatus.CRITICAL,
                reporting_frequency=ReportingFrequency.DAILY
            ),
            ComplianceRule(
                rule_id="LEV001",
                rule_name="Maximum Leverage",
                regulation_type=RegulationType.LEVERAGE_LIMITS,
                description="Portfolio leverage must not exceed 2x",
                threshold_value=2.0,
                threshold_type="max",
                severity=ComplianceStatus.BREACH,
                reporting_frequency=ReportingFrequency.DAILY
            ),
            ComplianceRule(
                rule_id="POS001",
                rule_name="Maximum Position Size",
                regulation_type=RegulationType.POSITION_LIMITS,
                description="Single position must not exceed 20% of portfolio",
                threshold_value=0.20,
                threshold_type="max",
                severity=ComplianceStatus.WARNING,
                reporting_frequency=ReportingFrequency.DAILY
            ),
            ComplianceRule(
                rule_id="CONC001",
                rule_name="Concentration Limit",
                regulation_type=RegulationType.CONCENTRATION_LIMITS,
                description="Top 5 positions must not exceed 50% of portfolio",
                threshold_value=0.50,
                threshold_type="max",
                severity=ComplianceStatus.WARNING,
                reporting_frequency=ReportingFrequency.WEEKLY
            )
        ]

    def _get_metric_value(
        self,
        regulation_type: RegulationType,
        portfolio_data: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> float:
        """Get metric value based on regulation type"""
        if regulation_type == RegulationType.VAR_LIMITS:
            return risk_metrics.get('var_95', 0.0)
        elif regulation_type == RegulationType.RISK_LIMITS:
            return abs(risk_metrics.get('drawdown', 0.0))
        elif regulation_type == RegulationType.LEVERAGE_LIMITS:
            return portfolio_data.get('leverage', 0.0)
        elif regulation_type == RegulationType.POSITION_LIMITS:
            return portfolio_data.get('largest_position_pct', 0.0)
        elif regulation_type == RegulationType.CONCENTRATION_LIMITS:
            return portfolio_data.get('top5_concentration', 0.0)
        else:
            return 0.0

    def _check_threshold(
        self,
        current_value: float,
        threshold_value: float,
        threshold_type: str
    ) -> ComplianceStatus:
        """Check if current value breaches threshold"""
        if threshold_type == "max":
            if current_value <= threshold_value * 0.8:
                return ComplianceStatus.COMPLIANT
            elif current_value <= threshold_value:
                return ComplianceStatus.WARNING
            elif current_value <= threshold_value * 1.2:
                return ComplianceStatus.BREACH
            else:
                return ComplianceStatus.CRITICAL

        elif threshold_type == "min":
            if current_value >= threshold_value * 1.2:
                return ComplianceStatus.COMPLIANT
            elif current_value >= threshold_value:
                return ComplianceStatus.WARNING
            elif current_value >= threshold_value * 0.8:
                return ComplianceStatus.BREACH
            else:
                return ComplianceStatus.CRITICAL

        return ComplianceStatus.COMPLIANT

    def _log_breach(
        self,
        rule: ComplianceRule,
        current_value: float,
        timestamp: datetime
    ):
        """Log a compliance breach"""
        deviation = current_value - rule.threshold_value
        deviation_pct = (deviation / rule.threshold_value * 100) if rule.threshold_value > 0 else 0

        breach_id = self._generate_breach_id(rule.rule_id, timestamp)

        breach = ComplianceBreach(
            breach_id=breach_id,
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            timestamp=timestamp,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            deviation=deviation,
            deviation_pct=deviation_pct,
            severity=rule.severity,
            description=f"{rule.rule_name}: Current value {current_value:.4f} exceeds threshold {rule.threshold_value:.4f}"
        )

        # Add to active and history
        self.active_breaches.append(breach)
        self.breach_history.append(breach)

    def _log_audit(self, action: str, details: Dict[str, Any]):
        """Log action to audit trail"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }

        self.audit_trail.append(audit_entry)

    def _save_report(self, report: ComplianceReport):
        """Save compliance report to file"""
        filename = f"compliance_{report.report_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

    def _calculate_summary_statistics(
        self,
        portfolio_data: Dict[str, Any],
        risk_metrics: Dict[str, float],
        breaches: List[ComplianceBreach]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for report"""
        return {
            'portfolio_value': portfolio_data.get('portfolio_value', 0.0),
            'leverage': portfolio_data.get('leverage', 0.0),
            'var_95': risk_metrics.get('var_95', 0.0),
            'volatility': risk_metrics.get('volatility', 0.0),
            'drawdown': risk_metrics.get('drawdown', 0.0),
            'total_breaches': len(breaches),
            'critical_breaches': sum(1 for b in breaches if b.severity == ComplianceStatus.CRITICAL),
            'average_deviation_pct': np.mean([b.deviation_pct for b in breaches]) if breaches else 0.0
        }

    def _generate_report_id(self, timestamp: datetime) -> str:
        """Generate unique report ID"""
        return f"COMP_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    def _generate_breach_id(self, rule_id: str, timestamp: datetime) -> str:
        """Generate unique breach ID"""
        hash_input = f"{rule_id}_{timestamp.isoformat()}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"BREACH_{hash_value.upper()}"

    def generate_summary_report(self) -> str:
        """
        Generate text summary of compliance status

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("COMPLIANCE SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append(f"Total Rules: {len(self.rules)}")
        lines.append(f"Active Rules: {sum(1 for r in self.rules if r.enabled)}")
        lines.append("")

        lines.append(f"Active Breaches: {len(self.get_active_breaches())}")
        lines.append(f"Total Historical Breaches: {len(self.breach_history)}")
        lines.append("")

        if self.active_breaches:
            lines.append("CURRENT BREACHES:")
            for breach in self.active_breaches[:10]:
                status = "ACK" if breach.acknowledged else "ACTIVE"
                lines.append(
                    f"  [{status}] {breach.rule_name}: "
                    f"{breach.current_value:.4f} (threshold: {breach.threshold_value:.4f}) "
                    f"- {breach.severity.value.upper()}"
                )
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

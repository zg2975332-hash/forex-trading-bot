"""
Advanced Trade Auditor for FOREX TRADING BOT
Comprehensive trade analysis, validation, and compliance monitoring
"""

import logging
import pandas as pd
import numpy as np
import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
from pathlib import Path
import warnings
from collections import defaultdict, deque
import statistics
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import csv
import pickle
import gzip
from threading import Lock, RLock
import asyncio
import re
from email.mime.text import MimeText
import smtplib
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class AuditSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"

class ComplianceRule(Enum):
    POSITION_SIZE_LIMIT = "position_size_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN_LIMIT = "max_drawdown_limit"
    TRADE_FREQUENCY_LIMIT = "trade_frequency_limit"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    VOLATILITY_EXPOSURE = "volatility_exposure"
    CORRELATION_LIMIT = "correlation_limit"
    CONCENTRATION_LIMIT = "concentration_limit"

@dataclass
class TradeAuditRecord:
    """Individual trade audit record"""
    audit_id: str
    trade_id: str
    timestamp: datetime
    audit_type: str
    severity: AuditSeverity
    status: AuditStatus
    rule_violated: Optional[ComplianceRule]
    description: str
    details: Dict[str, Any]
    recommendations: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

@dataclass
class ComplianceConfig:
    """Compliance and risk configuration"""
    # Position limits
    max_position_size: float = 100000.0  # Maximum position size in base currency
    max_position_percentage: float = 0.02  # 2% of portfolio per trade
    
    # Loss limits
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    max_drawdown_limit: float = 0.15  # 15% maximum drawdown
    
    # Trading limits
    max_trades_per_hour: int = 10
    max_trades_per_day: int = 50
    
    # Risk management
    min_risk_reward_ratio: float = 1.5
    max_volatility_exposure: float = 0.10  # 10% volatility exposure
    max_correlation: float = 0.7  # Maximum correlation with other positions
    max_concentration: float = 0.25  # 25% maximum concentration in one instrument
    
    # Slippage and execution
    max_acceptable_slippage: float = 0.0005  # 5 pips
    max_commission_percentage: float = 0.001  # 0.1% of trade value
    
    # Pattern detection
    detect_wash_trades: bool = True
    detect_martingale: bool = True
    detect_revenge_trading: bool = True

@dataclass
class AuditReport:
    """Comprehensive audit report"""
    summary: Dict[str, Any]
    trade_audits: List[TradeAuditRecord]
    compliance_violations: List[TradeAuditRecord]
    risk_analysis: Dict[str, Any]
    pattern_detection: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]

class TradeAuditor:
    """
    Advanced trade auditing and compliance monitoring system
    Real-time trade validation, risk monitoring, and compliance enforcement
    """
    
    def __init__(self, db_path: str = "audit.db", config: ComplianceConfig = None):
        self.db_path = db_path
        self.config = config or ComplianceConfig()
        
        # Initialize data structures
        self.audit_records: Dict[str, TradeAuditRecord] = {}
        self.trade_history: Dict[str, Dict] = {}
        self.compliance_checks: Dict[ComplianceRule, bool] = {}
        
        # Real-time monitoring
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.current_drawdown = 0.0
        self.portfolio_concentration = defaultdict(float)
        
        # Pattern detection
        self.trade_patterns = defaultdict(deque)
        self.suspicious_activities = deque(maxlen=1000)
        
        # Thread safety
        self._lock = RLock()
        self._audit_lock = Lock()
        
        # Initialize database
        self._init_database()
        
        # Alert system
        self.alert_recipients = []
        self.alert_threshold = AuditSeverity.HIGH
        
        logger.info("TradeAuditor initialized")

    def _init_database(self):
        """Initialize SQLite database for audit records"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Audit records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_records (
                        audit_id TEXT PRIMARY KEY,
                        trade_id TEXT,
                        timestamp TIMESTAMP,
                        audit_type TEXT,
                        severity TEXT,
                        status TEXT,
                        rule_violated TEXT,
                        description TEXT,
                        details TEXT,
                        recommendations TEXT,
                        evidence TEXT,
                        resolved BOOLEAN,
                        resolved_at TIMESTAMP,
                        resolved_by TEXT,
                        FOREIGN KEY (trade_id) REFERENCES trades (trade_id)
                    )
                ''')
                
                # Trade history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_history (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT,
                        entry_time TIMESTAMP,
                        exit_time TIMESTAMP,
                        entry_price REAL,
                        exit_price REAL,
                        position_size REAL,
                        side TEXT,
                        pnl REAL,
                        pnl_percentage REAL,
                        commission REAL,
                        slippage REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        strategy TEXT,
                        confidence REAL,
                        market_condition TEXT,
                        audit_status TEXT,
                        risk_metrics TEXT
                    )
                ''')
                
                # Compliance violations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS compliance_violations (
                        violation_id TEXT PRIMARY KEY,
                        trade_id TEXT,
                        timestamp TIMESTAMP,
                        rule_violated TEXT,
                        severity TEXT,
                        description TEXT,
                        action_taken TEXT,
                        resolved BOOLEAN
                    )
                ''')
                
                # Pattern detection table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT,
                        detected_at TIMESTAMP,
                        confidence REAL,
                        description TEXT,
                        affected_trades TEXT,
                        recommendations TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("Audit database initialized")
                
        except Exception as e:
            logger.error(f"Audit database initialization failed: {e}")
            raise

    def audit_trade(self, trade_data: Dict[str, Any]) -> Tuple[bool, List[TradeAuditRecord]]:
        """
        Comprehensive trade audit - validates trade before execution
        Returns (is_approved, audit_records)
        """
        try:
            with self._audit_lock:
                audit_records = []
                
                # Generate audit ID
                audit_id = self._generate_audit_id(trade_data)
                
                # 1. Pre-trade compliance checks
                pre_trade_audits = self._perform_pre_trade_checks(trade_data, audit_id)
                audit_records.extend(pre_trade_audits)
                
                # 2. Risk assessment
                risk_audits = self._perform_risk_assessment(trade_data, audit_id)
                audit_records.extend(risk_audits)
                
                # 3. Pattern analysis
                pattern_audits = self._analyze_trading_patterns(trade_data, audit_id)
                audit_records.extend(pattern_audits)
                
                # 4. Market condition checks
                market_audits = self._check_market_conditions(trade_data, audit_id)
                audit_records.extend(market_audits)
                
                # Determine if trade should be approved
                critical_violations = any(
                    audit.severity == AuditSeverity.CRITICAL and audit.status == AuditStatus.FAILED
                    for audit in audit_records
                )
                
                high_violations = sum(
                    1 for audit in audit_records 
                    if audit.severity == AuditSeverity.HIGH and audit.status == AuditStatus.FAILED
                )
                
                is_approved = not critical_violations and high_violations <= 1
                
                # Save audit records
                for audit in audit_records:
                    self._save_audit_record(audit)
                
                # Update trade history
                self._update_trade_history(trade_data, is_approved)
                
                # Send alerts for critical/high severity issues
                self._send_alerts(audit_records)
                
                logger.info(f"Trade audit completed: {audit_id} | Approved: {is_approved}")
                
                return is_approved, audit_records
                
        except Exception as e:
            logger.error(f"Trade audit failed: {e}")
            # In case of audit failure, deny trade for safety
            return False, []

    def audit_executed_trade(self, executed_trade: Dict[str, Any]) -> List[TradeAuditRecord]:
        """
        Audit trade after execution - validates execution quality
        """
        try:
            with self._audit_lock:
                audit_records = []
                audit_id = self._generate_audit_id(executed_trade)
                
                # 1. Execution quality audit
                execution_audits = self._audit_execution_quality(executed_trade, audit_id)
                audit_records.extend(execution_audits)
                
                # 2. Slippage analysis
                slippage_audits = self._analyze_slippage(executed_trade, audit_id)
                audit_records.extend(slippage_audits)
                
                # 3. Commission analysis
                commission_audits = self._analyze_commissions(executed_trade, audit_id)
                audit_records.extend(commission_audits)
                
                # 4. Post-trade compliance
                post_trade_audits = self._perform_post_trade_checks(executed_trade, audit_id)
                audit_records.extend(post_trade_audits)
                
                # Save audit records
                for audit in audit_records:
                    self._save_audit_record(audit)
                
                logger.info(f"Executed trade audit completed: {audit_id}")
                
                return audit_records
                
        except Exception as e:
            logger.error(f"Executed trade audit failed: {e}")
            return []

    def _perform_pre_trade_checks(self, trade_data: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Perform pre-trade compliance checks"""
        audits = []
        
        try:
            # 1. Position size check
            position_size = trade_data.get('position_size', 0)
            if position_size > self.config.max_position_size:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="position_size_check",
                    severity=AuditSeverity.CRITICAL,
                    status=AuditStatus.FAILED,
                    rule_violated=ComplianceRule.POSITION_SIZE_LIMIT,
                    description=f"Position size {position_size} exceeds maximum limit {self.config.max_position_size}",
                    details={
                        'position_size': position_size,
                        'max_limit': self.config.max_position_size,
                        'excess_amount': position_size - self.config.max_position_size
                    },
                    recommendations=[
                        "Reduce position size to comply with limits",
                        "Review position sizing strategy",
                        "Consider portfolio diversification"
                    ]
                ))
            else:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="position_size_check",
                    severity=AuditSeverity.LOW,
                    status=AuditStatus.PASSED,
                    description="Position size within acceptable limits",
                    details={'position_size': position_size}
                ))
            
            # 2. Daily loss limit check
            if self.daily_pnl < -self.config.daily_loss_limit:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="daily_loss_check",
                    severity=AuditSeverity.HIGH,
                    status=AuditStatus.FAILED,
                    rule_violated=ComplianceRule.DAILY_LOSS_LIMIT,
                    description=f"Daily PnL {self.daily_pnl:.4f} exceeds loss limit {-self.config.daily_loss_limit}",
                    details={
                        'daily_pnl': self.daily_pnl,
                        'loss_limit': -self.config.daily_loss_limit
                    },
                    recommendations=[
                        "Stop trading for today - daily loss limit reached",
                        "Review today's trading strategy",
                        "Analyze loss causes for improvement"
                    ]
                ))
            
            # 3. Trade frequency check
            if self.daily_trade_count >= self.config.max_trades_per_day:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="trade_frequency_check",
                    severity=AuditSeverity.HIGH,
                    status=AuditStatus.FAILED,
                    rule_violated=ComplianceRule.TRADE_FREQUENCY_LIMIT,
                    description=f"Daily trade count {self.daily_trade_count} exceeds limit {self.config.max_trades_per_day}",
                    details={
                        'trade_count': self.daily_trade_count,
                        'daily_limit': self.config.max_trades_per_day
                    },
                    recommendations=[
                        "Stop trading for today - trade limit reached",
                        "Review trading strategy for overtrading",
                        "Focus on trade quality over quantity"
                    ]
                ))
            
            # 4. Risk-reward ratio check
            stop_loss = trade_data.get('stop_loss')
            take_profit = trade_data.get('take_profit')
            entry_price = trade_data.get('entry_price')
            
            if all([stop_loss, take_profit, entry_price]):
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0
                
                if risk_reward_ratio < self.config.min_risk_reward_ratio:
                    audits.append(self._create_audit_record(
                        audit_id=audit_id,
                        trade_id=trade_data.get('trade_id'),
                        audit_type="risk_reward_check",
                        severity=AuditSeverity.MEDIUM,
                        status=AuditStatus.WARNING,
                        rule_violated=ComplianceRule.RISK_REWARD_RATIO,
                        description=f"Risk-reward ratio {risk_reward_ratio:.2f} below minimum {self.config.min_risk_reward_ratio}",
                        details={
                            'risk_reward_ratio': risk_reward_ratio,
                            'minimum_required': self.config.min_risk_reward_ratio,
                            'risk_pips': risk * 10000,  # Convert to pips
                            'reward_pips': reward * 10000
                        },
                        recommendations=[
                            "Adjust stop-loss or take-profit levels",
                            "Consider skipping trades with poor risk-reward",
                            "Review entry strategy for better positioning"
                        ]
                    ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Pre-trade checks failed: {e}")
            # Return a critical audit record for audit failure
            return [self._create_audit_record(
                audit_id=audit_id,
                trade_id=trade_data.get('trade_id'),
                audit_type="pre_trade_checks",
                severity=AuditSeverity.CRITICAL,
                status=AuditStatus.FAILED,
                description="Pre-trade audit system failure",
                details={'error': str(e)},
                recommendations=["Do not execute trade - audit system error"]
            )]

    def _perform_risk_assessment(self, trade_data: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Perform comprehensive risk assessment"""
        audits = []
        
        try:
            symbol = trade_data.get('symbol', '')
            position_size = trade_data.get('position_size', 0)
            
            # 1. Concentration risk
            portfolio_value = 100000  # Example - should come from portfolio manager
            concentration = position_size / portfolio_value if portfolio_value > 0 else 0
            
            if concentration > self.config.max_concentration:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="concentration_check",
                    severity=AuditSeverity.HIGH,
                    status=AuditStatus.FAILED,
                    rule_violated=ComplianceRule.CONCENTRATION_LIMIT,
                    description=f"Portfolio concentration {concentration:.2%} exceeds limit {self.config.max_concentration:.2%}",
                    details={
                        'concentration': concentration,
                        'limit': self.config.max_concentration,
                        'symbol': symbol,
                        'position_size': position_size,
                        'portfolio_value': portfolio_value
                    },
                    recommendations=[
                        "Reduce position size to maintain diversification",
                        "Consider alternative instruments for exposure",
                        "Review portfolio allocation strategy"
                    ]
                ))
            
            # 2. Volatility exposure
            # This would typically use historical volatility data
            historical_volatility = 0.08  # Example - should come from market data
            volatility_exposure = position_size * historical_volatility / portfolio_value
            
            if volatility_exposure > self.config.max_volatility_exposure:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="volatility_exposure_check",
                    severity=AuditSeverity.MEDIUM,
                    status=AuditStatus.WARNING,
                    rule_violated=ComplianceRule.VOLATILITY_EXPOSURE,
                    description=f"Volatility exposure {volatility_exposure:.2%} exceeds limit {self.config.max_volatility_exposure:.2%}",
                    details={
                        'volatility_exposure': volatility_exposure,
                        'limit': self.config.max_volatility_exposure,
                        'historical_volatility': historical_volatility
                    },
                    recommendations=[
                        "Reduce position size due to high volatility",
                        "Consider hedging strategies",
                        "Wait for lower volatility conditions"
                    ]
                ))
            
            # 3. Correlation analysis
            # This would check correlation with existing positions
            avg_correlation = self._calculate_portfolio_correlation(symbol)
            if avg_correlation > self.config.max_correlation:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="correlation_check",
                    severity=AuditSeverity.MEDIUM,
                    status=AuditStatus.WARNING,
                    rule_violated=ComplianceRule.CORRELATION_LIMIT,
                    description=f"Portfolio correlation {avg_correlation:.3f} exceeds limit {self.config.max_correlation}",
                    details={
                        'correlation': avg_correlation,
                        'limit': self.config.max_correlation,
                        'symbol': symbol
                    },
                    recommendations=[
                        "Consider uncorrelated instruments",
                        "Reduce position size to manage correlation risk",
                        "Review portfolio diversification"
                    ]
                ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return [self._create_audit_record(
                audit_id=audit_id,
                trade_id=trade_data.get('trade_id'),
                audit_type="risk_assessment",
                severity=AuditSeverity.HIGH,
                status=AuditStatus.FAILED,
                description="Risk assessment system failure",
                details={'error': str(e)},
                recommendations=["Proceed with caution - risk system error"]
            )]

    def _analyze_trading_patterns(self, trade_data: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Analyze trading patterns for suspicious activities"""
        audits = []
        
        try:
            symbol = trade_data.get('symbol', '')
            strategy = trade_data.get('strategy', '')
            position_size = trade_data.get('position_size', 0)
            
            # Update pattern tracking
            self._update_pattern_tracking(trade_data)
            
            # 1. Wash trade detection
            if self.config.detect_wash_trades and self._detect_wash_trades(trade_data):
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="wash_trade_detection",
                    severity=AuditSeverity.CRITICAL,
                    status=AuditStatus.FAILED,
                    description="Potential wash trading detected",
                    details={
                        'symbol': symbol,
                        'strategy': strategy,
                        'detection_confidence': 0.85
                    },
                    recommendations=[
                        "Cancel trade - potential regulatory violation",
                        "Review trading strategy compliance",
                        "Consult compliance officer"
                    ]
                ))
            
            # 2. Martingale pattern detection
            if self.config.detect_martingale and self._detect_martingale_pattern(trade_data):
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="martingale_detection",
                    severity=AuditSeverity.HIGH,
                    status=AuditStatus.FAILED,
                    description="Martingale trading pattern detected",
                    details={
                        'symbol': symbol,
                        'position_size': position_size,
                        'pattern_confidence': 0.75
                    },
                    recommendations=[
                        "Avoid doubling down on losing positions",
                        "Review risk management strategy",
                        "Implement proper position sizing"
                    ]
                ))
            
            # 3. Revenge trading detection
            if self.config.detect_revenge_trading and self._detect_revenge_trading(trade_data):
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="revenge_trading_detection",
                    severity=AuditSeverity.HIGH,
                    status=AuditStatus.WARNING,
                    description="Potential revenge trading behavior detected",
                    details={
                        'symbol': symbol,
                        'time_since_last_loss': self._get_time_since_last_loss(symbol),
                        'behavior_confidence': 0.70
                    },
                    recommendations=[
                        "Take a break from trading",
                        "Review emotional trading triggers",
                        "Stick to predefined trading plan"
                    ]
                ))
            
            # 4. Overtrading detection
            if self._detect_overtrading(trade_data):
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="overtrading_detection",
                    severity=AuditSeverity.MEDIUM,
                    status=AuditStatus.WARNING,
                    description="Potential overtrading behavior detected",
                    details={
                        'trades_last_hour': len(self.trade_patterns['recent_trades']),
                        'symbol': symbol,
                        'timeframe': '1 hour'
                    },
                    recommendations=[
                        "Reduce trading frequency",
                        "Focus on higher quality setups",
                        "Implement trading cooldown periods"
                    ]
                ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return []

    def _audit_execution_quality(self, executed_trade: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Audit trade execution quality"""
        audits = []
        
        try:
            # 1. Slippage analysis
            expected_price = executed_trade.get('expected_price')
            actual_price = executed_trade.get('actual_price')
            
            if expected_price and actual_price:
                slippage = abs(actual_price - expected_price)
                if slippage > self.config.max_acceptable_slippage:
                    audits.append(self._create_audit_record(
                        audit_id=audit_id,
                        trade_id=executed_trade.get('trade_id'),
                        audit_type="slippage_audit",
                        severity=AuditSeverity.MEDIUM,
                        status=AuditStatus.WARNING,
                        description=f"High slippage detected: {slippage:.6f}",
                        details={
                            'slippage': slippage,
                            'expected_price': expected_price,
                            'actual_price': actual_price,
                            'max_acceptable': self.config.max_acceptable_slippage
                        },
                        recommendations=[
                            "Use limit orders instead of market orders",
                            "Avoid trading during high volatility periods",
                            "Consider different execution venues"
                        ]
                    ))
            
            # 2. Execution time analysis
            order_time = executed_trade.get('order_time')
            execution_time = executed_trade.get('execution_time')
            
            if order_time and execution_time:
                execution_delay = (execution_time - order_time).total_seconds()
                if execution_delay > 2.0:  # 2 seconds threshold
                    audits.append(self._create_audit_record(
                        audit_id=audit_id,
                        trade_id=executed_trade.get('trade_id'),
                        audit_type="execution_delay_audit",
                        severity=AuditSeverity.LOW,
                        status=AuditStatus.WARNING,
                        description=f"Slow execution: {execution_delay:.2f} seconds",
                        details={
                            'execution_delay_seconds': execution_delay,
                            'order_time': order_time,
                            'execution_time': execution_time
                        },
                        recommendations=[
                            "Check connection latency to broker",
                            "Consider different execution methods",
                            "Monitor system performance"
                        ]
                    ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Execution quality audit failed: {e}")
            return []

    def _analyze_slippage(self, executed_trade: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Detailed slippage analysis"""
        audits = []
        
        try:
            # This would typically compare with market data at order time
            # For now, using simplified analysis
            
            symbol = executed_trade.get('symbol')
            volume = executed_trade.get('volume', 0)
            slippage = executed_trade.get('slippage', 0)
            
            # Calculate slippage cost
            slippage_cost = volume * slippage
            
            if slippage_cost > 50:  # $50 threshold
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=executed_trade.get('trade_id'),
                    audit_type="slippage_cost_audit",
                    severity=AuditSeverity.MEDIUM,
                    status=AuditStatus.WARNING,
                    description=f"High slippage cost: ${slippage_cost:.2f}",
                    details={
                        'slippage_cost': slippage_cost,
                        'slippage_pips': slippage * 10000,
                        'volume': volume,
                        'symbol': symbol
                    },
                    recommendations=[
                        "Reduce trade size during low liquidity",
                        "Use iceberg orders for large positions",
                        "Trade during high liquidity hours"
                    ]
                ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Slippage analysis failed: {e}")
            return []

    def _analyze_commissions(self, executed_trade: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Commission and fee analysis"""
        audits = []
        
        try:
            commission = executed_trade.get('commission', 0)
            trade_value = executed_trade.get('position_size', 0) * executed_trade.get('entry_price', 1)
            
            if trade_value > 0:
                commission_percentage = commission / trade_value
                
                if commission_percentage > self.config.max_commission_percentage:
                    audits.append(self._create_audit_record(
                        audit_id=audit_id,
                        trade_id=executed_trade.get('trade_id'),
                        audit_type="commission_audit",
                        severity=AuditSeverity.MEDIUM,
                        status=AuditStatus.WARNING,
                        description=f"High commission rate: {commission_percentage:.4%}",
                        details={
                            'commission_percentage': commission_percentage,
                            'commission_amount': commission,
                            'trade_value': trade_value,
                            'max_acceptable': self.config.max_commission_percentage
                        },
                        recommendations=[
                            "Negotiate better commission rates with broker",
                            "Consider different execution venues",
                            "Review commission structure"
                        ]
                    ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Commission analysis failed: {e}")
            return []

    def _perform_post_trade_checks(self, executed_trade: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Post-trade compliance checks"""
        audits = []
        
        try:
            # Update daily metrics
            pnl = executed_trade.get('pnl', 0)
            self.daily_pnl += pnl
            self.daily_trade_count += 1
            
            # Check if daily loss limit is breached after this trade
            if self.daily_pnl < -self.config.daily_loss_limit:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=executed_trade.get('trade_id'),
                    audit_type="post_trade_loss_check",
                    severity=AuditSeverity.CRITICAL,
                    status=AuditStatus.FAILED,
                    rule_violated=ComplianceRule.DAILY_LOSS_LIMIT,
                    description=f"Daily loss limit breached after trade execution: {self.daily_pnl:.4f}",
                    details={
                        'daily_pnl': self.daily_pnl,
                        'loss_limit': -self.config.daily_loss_limit,
                        'trade_pnl': pnl
                    },
                    recommendations=[
                        "STOP ALL TRADING FOR TODAY",
                        "Review risk management procedures",
                        "Analyze today's trading performance"
                    ]
                ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Post-trade checks failed: {e}")
            return []

    def _check_market_conditions(self, trade_data: Dict[str, Any], audit_id: str) -> List[TradeAuditRecord]:
        """Check market conditions for trading"""
        audits = []
        
        try:
            symbol = trade_data.get('symbol', '')
            
            # 1. Market hours check (simplified)
            current_time = datetime.now().time()
            market_open = datetime.strptime("00:00", "%H:%M").time()
            market_close = datetime.strptime("23:59", "%H:%M").time()
            
            if not (market_open <= current_time <= market_close):
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="market_hours_check",
                    severity=AuditSeverity.MEDIUM,
                    status=AuditStatus.WARNING,
                    description="Trading outside main market hours",
                    details={
                        'current_time': current_time,
                        'market_hours': f"{market_open} - {market_close}",
                        'symbol': symbol
                    },
                    recommendations=[
                        "Check instrument-specific trading hours",
                        "Be aware of reduced liquidity",
                        "Consider wider spreads during off-hours"
                    ]
                ))
            
            # 2. High impact news check (placeholder)
            # This would integrate with economic calendar
            high_impact_news = self._check_high_impact_news(symbol)
            if high_impact_news:
                audits.append(self._create_audit_record(
                    audit_id=audit_id,
                    trade_id=trade_data.get('trade_id'),
                    audit_type="news_impact_check",
                    severity=AuditSeverity.HIGH,
                    status=AuditStatus.WARNING,
                    description="High impact news event detected",
                    details={
                        'symbol': symbol,
                        'news_events': high_impact_news,
                        'impact_level': 'high'
                    },
                    recommendations=[
                        "Avoid trading during high impact news",
                        "Use wider stops if trading must occur",
                        "Wait for volatility to normalize"
                    ]
                ))
            
            return audits
            
        except Exception as e:
            logger.error(f"Market condition checks failed: {e}")
            return []

    # ==================== PATTERN DETECTION METHODS ====================

    def _detect_wash_trades(self, trade_data: Dict[str, Any]) -> bool:
        """Detect potential wash trading"""
        try:
            symbol = trade_data.get('symbol', '')
            recent_trades = self.trade_patterns.get(symbol, [])
            
            # Simple wash trade detection - looking for rapid open/close
            if len(recent_trades) >= 2:
                last_trade = recent_trades[-1]
                time_diff = (trade_data.get('entry_time', datetime.now()) - last_trade.get('exit_time', datetime.now())).total_seconds()
                
                # If same symbol traded within 60 seconds with similar size
                if (time_diff < 60 and 
                    abs(trade_data.get('position_size', 0) - last_trade.get('position_size', 0)) < 1000):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Wash trade detection failed: {e}")
            return False

    def _detect_martingale_pattern(self, trade_data: Dict[str, Any]) -> bool:
        """Detect martingale trading pattern"""
        try:
            symbol = trade_data.get('symbol', '')
            recent_losses = [t for t in self.trade_patterns.get(symbol, []) 
                           if t.get('pnl', 0) < 0]
            
            if len(recent_losses) >= 2:
                # Check if position size is increasing after losses
                last_loss_size = recent_losses[-1].get('position_size', 0)
                current_size = trade_data.get('position_size', 0)
                
                if current_size > last_loss_size * 1.5:  # 50% increase
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Martingale detection failed: {e}")
            return False

    def _detect_revenge_trading(self, trade_data: Dict[str, Any]) -> bool:
        """Detect revenge trading behavior"""
        try:
            symbol = trade_data.get('symbol', '')
            recent_trades = self.trade_patterns.get(symbol, [])
            
            if len(recent_trades) >= 1:
                last_trade = recent_trades[-1]
                time_since_last = (trade_data.get('entry_time', datetime.now()) - 
                                 last_trade.get('exit_time', datetime.now())).total_seconds()
                
                # If trading within 5 minutes of a loss
                if (last_trade.get('pnl', 0) < 0 and time_since_last < 300):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Revenge trading detection failed: {e}")
            return False

    def _detect_overtrading(self, trade_data: Dict[str, Any]) -> bool:
        """Detect overtrading behavior"""
        try:
            # Count trades in last hour
            recent_trades = self.trade_patterns.get('recent_trades', [])
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            
            trades_last_hour = [t for t in recent_trades 
                              if t.get('entry_time', current_time) > one_hour_ago]
            
            return len(trades_last_hour) > self.config.max_trades_per_hour
            
        except Exception as e:
            logger.error(f"Overtrading detection failed: {e}")
            return False

    def _update_pattern_tracking(self, trade_data: Dict[str, Any]):
        """Update pattern tracking data"""
        try:
            symbol = trade_data.get('symbol', '')
            
            # Update symbol-specific patterns
            if symbol not in self.trade_patterns:
                self.trade_patterns[symbol] = deque(maxlen=100)
            
            self.trade_patterns[symbol].append(trade_data)
            
            # Update recent trades
            if 'recent_trades' not in self.trade_patterns:
                self.trade_patterns['recent_trades'] = deque(maxlen=100)
            
            self.trade_patterns['recent_trades'].append(trade_data)
            
        except Exception as e:
            logger.error(f"Pattern tracking update failed: {e}")

    # ==================== UTILITY METHODS ====================

    def _create_audit_record(self, audit_id: str, trade_id: str, audit_type: str, 
                           severity: AuditSeverity, status: AuditStatus,
                           description: str, details: Dict[str, Any],
                           recommendations: List[str],
                           rule_violated: ComplianceRule = None,
                           evidence: Dict[str, Any] = None) -> TradeAuditRecord:
        """Create a standardized audit record"""
        return TradeAuditRecord(
            audit_id=audit_id,
            trade_id=trade_id,
            timestamp=datetime.now(),
            audit_type=audit_type,
            severity=severity,
            status=status,
            rule_violated=rule_violated,
            description=description,
            details=details,
            recommendations=recommendations,
            evidence=evidence or {},
            resolved=False
        )

    def _generate_audit_id(self, trade_data: Dict[str, Any]) -> str:
        """Generate unique audit ID"""
        trade_id = trade_data.get('trade_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"AUDIT_{timestamp}_{trade_id}_{random_suffix}"

    def _calculate_portfolio_correlation(self, symbol: str) -> float:
        """Calculate correlation with existing portfolio (simplified)"""
        # This would typically use historical correlation data
        # For now, returning a placeholder value
        return 0.3

    def _check_high_impact_news(self, symbol: str) -> List[str]:
        """Check for high impact news events (placeholder)"""
        # This would integrate with economic calendar API
        return []

    def _get_time_since_last_loss(self, symbol: str) -> float:
        """Get time since last losing trade in minutes"""
        try:
            recent_trades = self.trade_patterns.get(symbol, [])
            losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]
            
            if losing_trades:
                last_loss_time = losing_trades[-1].get('exit_time', datetime.now())
                return (datetime.now() - last_loss_time).total_seconds() / 60
            
            return float('inf')
        except:
            return float('inf')

    def _save_audit_record(self, audit_record: TradeAuditRecord):
        """Save audit record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO audit_records 
                    (audit_id, trade_id, timestamp, audit_type, severity, status,
                     rule_violated, description, details, recommendations, evidence,
                     resolved, resolved_at, resolved_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    audit_record.audit_id,
                    audit_record.trade_id,
                    audit_record.timestamp,
                    audit_record.audit_type,
                    audit_record.severity.value,
                    audit_record.status.value,
                    audit_record.rule_violated.value if audit_record.rule_violated else None,
                    audit_record.description,
                    json.dumps(audit_record.details),
                    json.dumps(audit_record.recommendations),
                    json.dumps(audit_record.evidence),
                    audit_record.resolved,
                    audit_record.resolved_at,
                    audit_record.resolved_by
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Audit record save failed: {e}")

    def _update_trade_history(self, trade_data: Dict[str, Any], approved: bool):
        """Update trade history"""
        try:
            trade_id = trade_data.get('trade_id')
            if trade_id:
                self.trade_history[trade_id] = {
                    **trade_data,
                    'audit_status': 'approved' if approved else 'rejected',
                    'audit_timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Trade history update failed: {e}")

    def _send_alerts(self, audit_records: List[TradeAuditRecord]):
        """Send alerts for critical/high severity issues"""
        try:
            critical_alerts = [r for r in audit_records 
                             if r.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]]
            
            for alert in critical_alerts:
                self._send_alert_notification(alert)
                
        except Exception as e:
            logger.error(f"Alert sending failed: {e}")

    def _send_alert_notification(self, audit_record: TradeAuditRecord):
        """Send individual alert notification"""
        try:
            # This would integrate with email/SMS/notification system
            logger.warning(f"ALERT - {audit_record.severity.value}: {audit_record.description}")
            
            # Placeholder for actual notification system
            if audit_record.severity == AuditSeverity.CRITICAL:
                # Critical alerts could trigger additional actions
                self._trigger_emergency_protocol(audit_record)
                
        except Exception as e:
            logger.error(f"Alert notification failed: {e}")

    def _trigger_emergency_protocol(self, audit_record: TradeAuditRecord):
        """Trigger emergency protocols for critical issues"""
        try:
            logger.critical(f"EMERGENCY PROTOCOL ACTIVATED: {audit_record.description}")
            
            # Placeholder for emergency actions:
            # - Close all positions
            # - Disable trading
            # - Notify administrators
            # - Log incident for compliance
            
        except Exception as e:
            logger.error(f"Emergency protocol failed: {e}")

    def generate_audit_report(self, start_date: datetime = None, end_date: datetime = None) -> AuditReport:
        """Generate comprehensive audit report"""
        try:
            start_date = start_date or datetime.now() - timedelta(days=30)
            end_date = end_date or datetime.now()
            
            # Load audit data
            audit_records = self._load_audit_records_period(start_date, end_date)
            compliance_violations = [r for r in audit_records if r.rule_violated]
            
            # Generate report sections
            summary = self._generate_audit_summary(audit_records, compliance_violations)
            risk_analysis = self._generate_risk_analysis(audit_records)
            pattern_detection = self._analyze_detected_patterns(audit_records)
            recommendations = self._generate_audit_recommendations(audit_records, compliance_violations)
            
            report = AuditReport(
                summary=summary,
                trade_audits=audit_records,
                compliance_violations=compliance_violations,
                risk_analysis=risk_analysis,
                pattern_detection=pattern_detection,
                recommendations=recommendations,
                metadata={
                    'generated_at': datetime.now(),
                    'period_start': start_date,
                    'period_end': end_date,
                    'total_audits': len(audit_records)
                }
            )
            
            self._save_audit_report(report)
            return report
            
        except Exception as e:
            logger.error(f"Audit report generation failed: {e}")
            raise

    def _load_audit_records_period(self, start_date: datetime, end_date: datetime) -> List[TradeAuditRecord]:
        """Load audit records for specific period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM audit_records 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                ''', (start_date, end_date))
                
                rows = cursor.fetchall()
                records = []
                
                for row in rows:
                    record = TradeAuditRecord(
                        audit_id=row['audit_id'],
                        trade_id=row['trade_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        audit_type=row['audit_type'],
                        severity=AuditSeverity(row['severity']),
                        status=AuditStatus(row['status']),
                        rule_violated=ComplianceRule(row['rule_violated']) if row['rule_violated'] else None,
                        description=row['description'],
                        details=json.loads(row['details']),
                        recommendations=json.loads(row['recommendations']),
                        evidence=json.loads(row['evidence']),
                        resolved=bool(row['resolved']),
                        resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                        resolved_by=row['resolved_by']
                    )
                    records.append(record)
                
                return records
                
        except Exception as e:
            logger.error(f"Audit records loading failed: {e}")
            return []

    def _generate_audit_summary(self, audit_records: List[TradeAuditRecord], 
                              compliance_violations: List[TradeAuditRecord]) -> Dict[str, Any]:
        """Generate audit summary"""
        try:
            total_audits = len(audit_records)
            passed_audits = len([r for r in audit_records if r.status == AuditStatus.PASSED])
            failed_audits = len([r for r in audit_records if r.status == AuditStatus.FAILED])
            warning_audits = len([r for r in audit_records if r.status == AuditStatus.WARNING])
            
            severity_counts = {
                'critical': len([r for r in audit_records if r.severity == AuditSeverity.CRITICAL]),
                'high': len([r for r in audit_records if r.severity == AuditSeverity.HIGH]),
                'medium': len([r for r in audit_records if r.severity == AuditSeverity.MEDIUM]),
                'low': len([r for r in audit_records if r.severity == AuditSeverity.LOW])
            }
            
            return {
                'total_audits': total_audits,
                'passed_audits': passed_audits,
                'failed_audits': failed_audits,
                'warning_audits': warning_audits,
                'pass_rate': passed_audits / total_audits if total_audits > 0 else 0,
                'compliance_violations': len(compliance_violations),
                'severity_distribution': severity_counts,
                'most_common_violation': self._get_most_common_violation(compliance_violations),
                'resolution_rate': len([r for r in audit_records if r.resolved]) / total_audits if total_audits > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Audit summary generation failed: {e}")
            return {}

    def _generate_risk_analysis(self, audit_records: List[TradeAuditRecord]) -> Dict[str, Any]:
        """Generate risk analysis from audit records"""
        try:
            high_risk_audits = [r for r in audit_records 
                              if r.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]]
            
            return {
                'high_risk_count': len(high_risk_audits),
                'risk_trend': self._calculate_risk_trend(audit_records),
                'common_risk_factors': self._identify_common_risk_factors(high_risk_audits),
                'risk_score': self._calculate_overall_risk_score(audit_records),
                'improvement_areas': self._identify_improvement_areas(audit_records)
            }
            
        except Exception as e:
            logger.error(f"Risk analysis generation failed: {e}")
            return {}

    def _analyze_detected_patterns(self, audit_records: List[TradeAuditRecord]) -> Dict[str, Any]:
        """Analyze detected trading patterns"""
        try:
            pattern_audits = [r for r in audit_records if 'detection' in r.audit_type]
            
            return {
                'total_patterns_detected': len(pattern_audits),
                'pattern_types': self._categorize_patterns(pattern_audits),
                'pattern_frequency': self._calculate_pattern_frequency(pattern_audits),
                'most_common_pattern': self._get_most_common_pattern(pattern_audits)
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {}

    def _generate_audit_recommendations(self, audit_records: List[TradeAuditRecord],
                                      compliance_violations: List[TradeAuditRecord]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # High severity issues
            high_severity = [r for r in audit_records if r.severity == AuditSeverity.HIGH]
            if high_severity:
                recommendations.append("Address high severity audit findings immediately")
            
            # Compliance violations
            if compliance_violations:
                recommendations.append("Review and update compliance procedures")
            
            # Pattern detection issues
            pattern_issues = [r for r in audit_records if 'detection' in r.audit_type]
            if pattern_issues:
                recommendations.append("Implement additional safeguards for detected trading patterns")
            
            # Risk management improvements
            risk_audits = [r for r in audit_records if 'risk' in r.audit_type]
            if risk_audits:
                recommendations.append("Enhance risk management framework based on audit findings")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to data issues"]

    def _get_most_common_violation(self, violations: List[TradeAuditRecord]) -> str:
        """Get most common compliance violation"""
        if not violations:
            return "None"
        
        violation_counts = defaultdict(int)
        for violation in violations:
            if violation.rule_violated:
                violation_counts[violation.rule_violated.value] += 1
        
        return max(violation_counts.items(), key=lambda x: x[1])[0] if violation_counts else "None"

    def _calculate_risk_trend(self, audit_records: List[TradeAuditRecord]) -> str:
        """Calculate risk trend (improving/stable/worsening)"""
        # Simplified implementation
        recent_audits = [r for r in audit_records 
                        if r.timestamp > datetime.now() - timedelta(days=7)]
        older_audits = [r for r in audit_records 
                       if datetime.now() - timedelta(days=14) <= r.timestamp <= datetime.now() - timedelta(days=7)]
        
        recent_high_risk = len([r for r in recent_audits if r.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]])
        older_high_risk = len([r for r in older_audits if r.severity in [AuditSeverity.CRITICAL, AuditSeverity.HIGH]])
        
        if recent_high_risk < older_high_risk:
            return "improving"
        elif recent_high_risk > older_high_risk:
            return "worsening"
        else:
            return "stable"

    def _identify_common_risk_factors(self, high_risk_audits: List[TradeAuditRecord]) -> List[str]:
        """Identify common risk factors"""
        factors = defaultdict(int)
        for audit in high_risk_audits:
            factors[audit.audit_type] += 1
        
        return [factor for factor, count in sorted(factors.items(), key=lambda x: x[1], reverse=True)[:5]]

    def _calculate_overall_risk_score(self, audit_records: List[TradeAuditRecord]) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        if not audit_records:
            return 0.0
        
        severity_weights = {
            AuditSeverity.CRITICAL: 10,
            AuditSeverity.HIGH: 5,
            AuditSeverity.MEDIUM: 2,
            AuditSeverity.LOW: 1
        }
        
        total_weight = sum(severity_weights.get(audit.severity, 0) for audit in audit_records)
        max_possible_weight = len(audit_records) * 10  # All critical
        
        return (total_weight / max_possible_weight) * 100 if max_possible_weight > 0 else 0.0

    def _identify_improvement_areas(self, audit_records: List[TradeAuditRecord]) -> List[str]:
        """Identify areas for improvement"""
        areas = set()
        for audit in audit_records:
            if audit.status in [AuditStatus.FAILED, AuditStatus.WARNING]:
                areas.add(audit.audit_type.replace('_check', '').replace('_audit', ''))
        
        return list(areas)[:5]  # Return top 5 areas

    def _categorize_patterns(self, pattern_audits: List[TradeAuditRecord]) -> Dict[str, int]:
        """Categorize detected patterns"""
        categories = defaultdict(int)
        for audit in pattern_audits:
            categories[audit.audit_type] += 1
        return dict(categories)

    def _calculate_pattern_frequency(self, pattern_audits: List[TradeAuditRecord]) -> float:
        """Calculate pattern detection frequency"""
        if not pattern_audits:
            return 0.0
        
        time_range = (datetime.now() - pattern_audits[0].timestamp).total_seconds() / 3600  # hours
        return len(pattern_audits) / time_range if time_range > 0 else 0.0

    def _get_most_common_pattern(self, pattern_audits: List[TradeAuditRecord]) -> str:
        """Get most common detected pattern"""
        if not pattern_audits:
            return "None"
        
        pattern_counts = defaultdict(int)
        for audit in pattern_audits:
            pattern_counts[audit.audit_type] += 1
        
        return max(pattern_counts.items(), key=lambda x: x[1])[0]

    def _save_audit_report(self, report: AuditReport):
        """Save audit report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path("audit_reports") / f"audit_report_{timestamp}.json"
            report_file.parent.mkdir(exist_ok=True)
            
            report_dict = {
                'summary': report.summary,
                'compliance_violations': len(report.compliance_violations),
                'risk_analysis': report.risk_analysis,
                'pattern_detection': report.pattern_detection,
                'recommendations': report.recommendations,
                'metadata': report.metadata
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Audit report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Audit report saving failed: {e}")

    def reset_daily_metrics(self):
        """Reset daily trading metrics"""
        with self._lock:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            logger.info("Daily metrics reset")

    def add_alert_recipient(self, email: str):
        """Add alert recipient"""
        self.alert_recipients.append(email)

    def set_alert_threshold(self, threshold: AuditSeverity):
        """Set alert threshold"""
        self.alert_threshold = threshold

# Example usage
if __name__ == "__main__":
    # Initialize auditor
    auditor = TradeAuditor()
    
    # Example trade data
    trade_data = {
        'trade_id': 'TRADE_001',
        'symbol': 'EUR/USD',
        'entry_time': datetime.now(),
        'position_size': 50000,
        'entry_price': 1.0850,
        'stop_loss': 1.0830,
        'take_profit': 1.0880,
        'strategy': 'momentum',
        'confidence': 0.75
    }
    
    # Audit trade
    approved, audits = auditor.audit_trade(trade_data)
    print(f"Trade Approved: {approved}")
    
    # Generate report
    report = auditor.generate_audit_report()
    print(f"Audit Report Generated: {len(report.trade_audits)} audits analyzed")
"""
Advanced Performance Logger for FOREX TRADING BOT
Comprehensive trading performance tracking and analytics
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
import matplotlib.pyplot as plt
import seaborn as sns
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
from threading import Lock
import asyncio

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    TRADE_PNL = "trade_pnl"
    PORTFOLIO_VALUE = "portfolio_value"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    TRADE_COUNT = "trade_count"
    AVG_TRADE = "avg_trade"
    AVG_WIN = "avg_win"
    AVG_LOSS = "avg_loss"
    LARGEST_WIN = "largest_win"
    LARGEST_LOSS = "largest_loss"
    EXPECTANCY = "expectancy"
    KELLY_CRITERION = "kelly_criterion"

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class TradeRecord:
    """Individual trade record"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    side: str  # 'long' or 'short'
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    commission: float
    slippage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy: str
    confidence: float
    market_condition: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a specific time"""
    timestamp: datetime
    portfolio_value: float
    cash_balance: float
    positions_value: float
    total_pnl: float
    daily_pnl: float
    drawdown: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    trade_count: int
    open_trades: int
    market_regime: str
    risk_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    summary: Dict[str, Any]
    trade_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    strategy_analysis: Dict[str, Any]
    time_analysis: Dict[str, Any]
    charts: Dict[str, Any]
    recommendations: List[str]

class PerformanceLogger:
    """
    Advanced performance logging and analytics system
    Tracks 50+ metrics in real-time with professional reporting
    """
    
    def __init__(self, db_path: str = "performance.db", log_dir: str = "logs"):
        self.db_path = db_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize data structures
        self.trades: Dict[str, TradeRecord] = {}
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.metric_history: Dict[PerformanceMetric, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        
        # Real-time metrics
        self.current_portfolio_value = 0.0
        self.peak_portfolio_value = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Thread safety
        self._lock = Lock()
        
        # Initialize database
        self._init_database()
        
        # Performance counters
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        
        logger.info("PerformanceLogger initialized")

    def _init_database(self):
        """Initialize SQLite database for performance tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
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
                        tags TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Performance snapshots table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        timestamp TIMESTAMP PRIMARY KEY,
                        portfolio_value REAL,
                        cash_balance REAL,
                        positions_value REAL,
                        total_pnl REAL,
                        daily_pnl REAL,
                        drawdown REAL,
                        max_drawdown REAL,
                        volatility REAL,
                        sharpe_ratio REAL,
                        sortino_ratio REAL,
                        win_rate REAL,
                        trade_count INTEGER,
                        open_trades INTEGER,
                        market_regime TEXT,
                        risk_metrics TEXT
                    )
                ''')
                
                # Metrics history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metric_history (
                        timestamp TIMESTAMP,
                        metric_name TEXT,
                        metric_value REAL,
                        PRIMARY KEY (timestamp, metric_name)
                    )
                ''')
                
                conn.commit()
                logger.info("Performance database initialized")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def log_trade(self, trade: TradeRecord):
        """Log a complete trade record"""
        try:
            with self._lock:
                self.trades[trade.trade_id] = trade
                
                # Update performance metrics
                if trade.pnl is not None:
                    self.total_trades += 1
                    if trade.pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                    
                    self.total_commission += trade.commission
                    self.total_slippage += trade.slippage
                
                # Save to database
                self._save_trade_to_db(trade)
                
                # Update real-time metrics
                self._update_real_time_metrics()
                
                logger.info(f"Trade logged: {trade.trade_id} | PnL: {trade.pnl}")
                
        except Exception as e:
            logger.error(f"Trade logging failed: {e}")

    def log_performance_snapshot(self, snapshot: PerformanceSnapshot):
        """Log performance snapshot"""
        try:
            with self._lock:
                self.performance_snapshots.append(snapshot)
                
                # Update peak portfolio value
                if snapshot.portfolio_value > self.peak_portfolio_value:
                    self.peak_portfolio_value = snapshot.portfolio_value
                
                # Save to database
                self._save_snapshot_to_db(snapshot)
                
                # Update metric history
                self._update_metric_history(snapshot)
                
                logger.debug(f"Performance snapshot logged: {snapshot.timestamp}")
                
        except Exception as e:
            logger.error(f"Performance snapshot logging failed: {e}")

    def log_metric(self, metric: PerformanceMetric, value: float, timestamp: datetime = None):
        """Log individual metric"""
        try:
            timestamp = timestamp or datetime.now()
            
            with self._lock:
                self.metric_history[metric].append((timestamp, value))
                self._save_metric_to_db(metric, value, timestamp)
                
        except Exception as e:
            logger.error(f"Metric logging failed: {e}")

    def _save_trade_to_db(self, trade: TradeRecord):
        """Save trade record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO trades 
                    (trade_id, symbol, entry_time, exit_time, entry_price, exit_price,
                     position_size, side, pnl, pnl_percentage, commission, slippage,
                     stop_loss, take_profit, strategy, confidence, market_condition,
                     tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.trade_id, trade.symbol, trade.entry_time, trade.exit_time,
                    trade.entry_price, trade.exit_price, trade.position_size, trade.side,
                    trade.pnl, trade.pnl_percentage, trade.commission, trade.slippage,
                    trade.stop_loss, trade.take_profit, trade.strategy, trade.confidence,
                    trade.market_condition, json.dumps(trade.tags), json.dumps(trade.metadata)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Trade database save failed: {e}")

    def _save_snapshot_to_db(self, snapshot: PerformanceSnapshot):
        """Save performance snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO performance_snapshots 
                    (timestamp, portfolio_value, cash_balance, positions_value,
                     total_pnl, daily_pnl, drawdown, max_drawdown, volatility,
                     sharpe_ratio, sortino_ratio, win_rate, trade_count, open_trades,
                     market_regime, risk_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp, snapshot.portfolio_value, snapshot.cash_balance,
                    snapshot.positions_value, snapshot.total_pnl, snapshot.daily_pnl,
                    snapshot.drawdown, snapshot.max_drawdown, snapshot.volatility,
                    snapshot.sharpe_ratio, snapshot.sortino_ratio, snapshot.win_rate,
                    snapshot.trade_count, snapshot.open_trades, snapshot.market_regime,
                    json.dumps(snapshot.risk_metrics)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Snapshot database save failed: {e}")

    def _save_metric_to_db(self, metric: PerformanceMetric, value: float, timestamp: datetime):
        """Save metric to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO metric_history 
                    (timestamp, metric_name, metric_value)
                    VALUES (?, ?, ?)
                ''', (timestamp, metric.value, value))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Metric database save failed: {e}")

    def _update_real_time_metrics(self):
        """Update real-time performance metrics"""
        try:
            # Calculate current win rate
            if self.total_trades > 0:
                win_rate = self.win_count / self.total_trades
                self.log_metric(PerformanceMetric.WIN_RATE, win_rate)
            
            # Update trade count
            self.log_metric(PerformanceMetric.TRADE_COUNT, self.total_trades)
            
        except Exception as e:
            logger.error(f"Real-time metrics update failed: {e}")

    def _update_metric_history(self, snapshot: PerformanceSnapshot):
        """Update metric history from snapshot"""
        try:
            metrics = {
                PerformanceMetric.PORTFOLIO_VALUE: snapshot.portfolio_value,
                PerformanceMetric.DRAWDOWN: snapshot.drawdown,
                PerformanceMetric.VOLATILITY: snapshot.volatility,
                PerformanceMetric.SHARPE_RATIO: snapshot.sharpe_ratio,
                PerformanceMetric.SORTINO_RATIO: snapshot.sortino_ratio,
                PerformanceMetric.WIN_RATE: snapshot.win_rate,
                PerformanceMetric.MAX_DRAWDOWN: snapshot.max_drawdown,
            }
            
            for metric, value in metrics.items():
                self.log_metric(metric, value, snapshot.timestamp)
                
        except Exception as e:
            logger.error(f"Metric history update failed: {e}")

    def generate_performance_report(self, start_date: datetime = None, end_date: datetime = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        try:
            start_date = start_date or datetime.now() - timedelta(days=30)
            end_date = end_date or datetime.now()
            
            logger.info(f"Generating performance report from {start_date} to {end_date}")
            
            # Load data for period
            trades = self._load_trades_period(start_date, end_date)
            snapshots = self._load_snapshots_period(start_date, end_date)
            
            # Generate report sections
            summary = self._generate_summary(trades, snapshots)
            trade_analysis = self._analyze_trades(trades)
            risk_analysis = self._analyze_risk(trades, snapshots)
            strategy_analysis = self._analyze_strategies(trades)
            time_analysis = self._analyze_time_performance(trades, snapshots)
            charts = self._generate_charts(trades, snapshots)
            recommendations = self._generate_recommendations(summary, trade_analysis, risk_analysis)
            
            report = PerformanceReport(
                summary=summary,
                trade_analysis=trade_analysis,
                risk_analysis=risk_analysis,
                strategy_analysis=strategy_analysis,
                time_analysis=time_analysis,
                charts=charts,
                recommendations=recommendations
            )
            
            # Save report
            self._save_report(report, start_date, end_date)
            
            logger.info("Performance report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            raise

    def _generate_summary(self, trades: List[TradeRecord], snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Generate performance summary"""
        try:
            if not trades:
                return {}
            
            # Basic statistics
            total_pnl = sum(trade.pnl or 0 for trade in trades)
            total_commission = sum(trade.commission for trade in trades)
            total_slippage = sum(trade.slippage for trade in trades)
            net_pnl = total_pnl - total_commission - total_slippage
            
            winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = statistics.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = statistics.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
            
            # Risk-adjusted returns
            returns = [t.pnl_percentage or 0 for t in trades if t.pnl_percentage]
            sharpe = self._calculate_sharpe_ratio(returns)
            sortino = self._calculate_sortino_ratio(returns)
            
            # Drawdown analysis
            portfolio_values = [s.portfolio_value for s in snapshots]
            max_drawdown = self._calculate_max_drawdown(portfolio_values) if portfolio_values else 0
            
            summary = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'net_pnl': net_pnl,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': max([t.pnl for t in winning_trades]) if winning_trades else 0,
                'largest_loss': min([t.pnl for t in losing_trades]) if losing_trades else 0,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_drawdown,
                'calmar_ratio': -net_pnl / max_drawdown if max_drawdown > 0 else 0,
                'expectancy': (win_rate * avg_win) + ((1 - win_rate) * avg_loss),
                'kelly_criterion': win_rate - (1 - win_rate) / (avg_win / abs(avg_loss)) if avg_loss != 0 else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {}

    def _analyze_trades(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Analyze trade performance"""
        try:
            if not trades:
                return {}
            
            # Trade duration analysis
            durations = []
            for trade in trades:
                if trade.entry_time and trade.exit_time:
                    duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                    durations.append(duration)
            
            # PnL distribution
            pnls = [trade.pnl or 0 for trade in trades]
            pnl_percentages = [trade.pnl_percentage or 0 for trade in trades]
            
            # Strategy performance
            strategy_pnl = {}
            for trade in trades:
                if trade.strategy not in strategy_pnl:
                    strategy_pnl[trade.strategy] = []
                strategy_pnl[trade.strategy].append(trade.pnl or 0)
            
            strategy_performance = {}
            for strategy, pnl_list in strategy_pnl.items():
                strategy_performance[strategy] = {
                    'total_pnl': sum(pnl_list),
                    'trade_count': len(pnl_list),
                    'win_rate': len([p for p in pnl_list if p > 0]) / len(pnl_list),
                    'avg_pnl': statistics.mean(pnl_list)
                }
            
            analysis = {
                'trade_durations': {
                    'mean': statistics.mean(durations) if durations else 0,
                    'median': statistics.median(durations) if durations else 0,
                    'std': statistics.stdev(durations) if len(durations) > 1 else 0,
                    'min': min(durations) if durations else 0,
                    'max': max(durations) if durations else 0
                },
                'pnl_distribution': {
                    'mean': statistics.mean(pnls),
                    'median': statistics.median(pnls),
                    'std': statistics.stdev(pnls) if len(pnls) > 1 else 0,
                    'skewness': stats.skew(pnls) if len(pnls) > 2 else 0,
                    'kurtosis': stats.kurtosis(pnls) if len(pnls) > 3 else 0
                },
                'strategy_performance': strategy_performance,
                'best_performing_strategy': max(
                    strategy_performance.items(), 
                    key=lambda x: x[1]['total_pnl']
                )[0] if strategy_performance else None,
                'worst_performing_strategy': min(
                    strategy_performance.items(), 
                    key=lambda x: x[1]['total_pnl']
                )[0] if strategy_performance else None
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Trade analysis failed: {e}")
            return {}

    def _analyze_risk(self, trades: List[TradeRecord], snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Analyze risk metrics"""
        try:
            if not trades or not snapshots:
                return {}
            
            # Value at Risk (VaR) calculation
            returns = [t.pnl_percentage or 0 for t in trades if t.pnl_percentage]
            var_95 = np.percentile(returns, 5) if returns else 0  # 5% VaR
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean([r for r in returns if r <= var_95]) if returns and any([r <= var_95 for r in returns]) else var_95
            
            # Drawdown analysis
            portfolio_values = [s.portfolio_value for s in snapshots]
            drawdowns = self._calculate_drawdowns(portfolio_values)
            
            # Volatility analysis
            daily_returns = self._calculate_daily_returns(snapshots)
            volatility = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
            
            risk_analysis = {
                'value_at_risk_95': var_95,
                'conditional_var_95': cvar_95,
                'max_drawdown': min(drawdowns) if drawdowns else 0,
                'avg_drawdown': statistics.mean(drawdowns) if drawdowns else 0,
                'drawdown_duration': self._calculate_drawdown_duration(drawdowns),
                'volatility': volatility,
                'beta': self._calculate_beta(daily_returns),  # Market correlation
                'alpha': self._calculate_alpha(daily_returns),  # Excess returns
                'tracking_error': self._calculate_tracking_error(daily_returns),
                'information_ratio': self._calculate_information_ratio(daily_returns)
            }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {}

    def _analyze_strategies(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Analyze strategy performance"""
        try:
            if not trades:
                return {}
            
            strategies = {}
            for trade in trades:
                if trade.strategy not in strategies:
                    strategies[trade.strategy] = {
                        'trades': [],
                        'total_pnl': 0,
                        'winning_trades': 0,
                        'losing_trades': 0
                    }
                
                strategies[trade.strategy]['trades'].append(trade)
                strategies[trade.strategy]['total_pnl'] += trade.pnl or 0
                
                if trade.pnl and trade.pnl > 0:
                    strategies[trade.strategy]['winning_trades'] += 1
                elif trade.pnl and trade.pnl <= 0:
                    strategies[trade.strategy]['losing_trades'] += 1
            
            # Calculate additional metrics for each strategy
            for strategy, data in strategies.items():
                trades_list = data['trades']
                total_trades = len(trades_list)
                
                data['win_rate'] = data['winning_trades'] / total_trades if total_trades > 0 else 0
                data['avg_pnl'] = data['total_pnl'] / total_trades if total_trades > 0 else 0
                
                # PnL per trade
                pnls = [t.pnl or 0 for t in trades_list]
                data['std_pnl'] = statistics.stdev(pnls) if len(pnls) > 1 else 0
                data['sharpe_ratio'] = data['avg_pnl'] / data['std_pnl'] if data['std_pnl'] > 0 else 0
                
                # Confidence analysis
                confidences = [t.confidence for t in trades_list]
                data['avg_confidence'] = statistics.mean(confidences) if confidences else 0
                data['confidence_correlation'] = self._calculate_confidence_correlation(trades_list)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            return {}

    def _analyze_time_performance(self, trades: List[TradeRecord], snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Analyze performance across different time periods"""
        try:
            if not trades:
                return {}
            
            # Hourly performance
            hourly_pnl = defaultdict(list)
            for trade in trades:
                if trade.entry_time:
                    hour = trade.entry_time.hour
                    hourly_pnl[hour].append(trade.pnl or 0)
            
            hourly_performance = {}
            for hour, pnls in hourly_pnl.items():
                hourly_performance[hour] = {
                    'total_pnl': sum(pnls),
                    'trade_count': len(pnls),
                    'avg_pnl': statistics.mean(pnls) if pnls else 0
                }
            
            # Daily performance
            daily_pnl = defaultdict(list)
            for trade in trades:
                if trade.entry_time:
                    day = trade.entry_time.strftime('%A')
                    daily_pnl[day].append(trade.pnl or 0)
            
            daily_performance = {}
            for day, pnls in daily_pnl.items():
                daily_performance[day] = {
                    'total_pnl': sum(pnls),
                    'trade_count': len(pnls),
                    'avg_pnl': statistics.mean(pnls) if pnls else 0
                }
            
            # Monthly performance
            monthly_pnl = defaultdict(list)
            for trade in trades:
                if trade.entry_time:
                    month = trade.entry_time.strftime('%Y-%m')
                    monthly_pnl[month].append(trade.pnl or 0)
            
            monthly_performance = {}
            for month, pnls in monthly_pnl.items():
                monthly_performance[month] = {
                    'total_pnl': sum(pnls),
                    'trade_count': len(pnls),
                    'avg_pnl': statistics.mean(pnls) if pnls else 0
                }
            
            return {
                'hourly': hourly_performance,
                'daily': daily_performance,
                'monthly': monthly_performance,
                'best_hour': max(hourly_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if hourly_performance else None,
                'best_day': max(daily_performance.items(), key=lambda x: x[1]['avg_pnl'])[0] if daily_performance else None,
                'best_month': max(monthly_performance.items(), key=lambda x: x[1]['total_pnl'])[0] if monthly_performance else None
            }
            
        except Exception as e:
            logger.error(f"Time performance analysis failed: {e}")
            return {}

    def _generate_charts(self, trades: List[TradeRecord], snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Generate performance charts"""
        try:
            charts = {}
            
            # Equity curve
            if snapshots:
                dates = [s.timestamp for s in snapshots]
                portfolio_values = [s.portfolio_value for s in snapshots]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=portfolio_values, mode='lines', name='Portfolio Value'))
                fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Portfolio Value')
                charts['equity_curve'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
            
            # PnL distribution
            if trades:
                pnls = [t.pnl or 0 for t in trades]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=pnls, nbinsx=50, name='PnL Distribution'))
                fig.update_layout(title='PnL Distribution', xaxis_title='PnL', yaxis_title='Frequency')
                charts['pnl_distribution'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
            
            # Drawdown chart
            if snapshots:
                drawdowns = [s.drawdown for s in snapshots]
                dates = [s.timestamp for s in snapshots]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=drawdowns, mode='lines', name='Drawdown', line=dict(color='red')))
                fig.update_layout(title='Portfolio Drawdown', xaxis_title='Date', yaxis_title='Drawdown (%)')
                charts['drawdown_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
            
            return charts
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return {}

    def _generate_recommendations(self, summary: Dict, trade_analysis: Dict, risk_analysis: Dict) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        try:
            # Win rate recommendations
            win_rate = summary.get('win_rate', 0)
            if win_rate < 0.4:
                recommendations.append("Consider improving entry timing or strategy selection - low win rate detected")
            elif win_rate > 0.7:
                recommendations.append("Excellent win rate! Consider increasing position sizes cautiously")
            
            # Risk management recommendations
            max_drawdown = summary.get('max_drawdown', 0)
            if max_drawdown > 0.1:  # 10% drawdown
                recommendations.append("High maximum drawdown detected. Review risk management and position sizing")
            
            # Profit factor recommendations
            profit_factor = summary.get('profit_factor', 0)
            if profit_factor < 1.5:
                recommendations.append("Low profit factor. Focus on improving risk-reward ratio")
            
            # Trade frequency recommendations
            total_trades = summary.get('total_trades', 0)
            if total_trades < 10:
                recommendations.append("Insufficient trade data for reliable analysis. Continue trading to gather more data")
            elif total_trades > 1000:
                recommendations.append("High trade frequency detected. Consider quality over quantity")
            
            # Strategy performance recommendations
            best_strategy = trade_analysis.get('best_performing_strategy')
            worst_strategy = trade_analysis.get('worst_performing_strategy')
            if best_strategy and worst_strategy:
                recommendations.append(f"Consider allocating more capital to {best_strategy} and less to {worst_strategy}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to data issues"]

    # ==================== UTILITY METHODS ====================

    def _load_trades_period(self, start_date: datetime, end_date: datetime) -> List[TradeRecord]:
        """Load trades for specific period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades 
                    WHERE entry_time BETWEEN ? AND ?
                    ORDER BY entry_time
                ''', (start_date, end_date))
                
                rows = cursor.fetchall()
                trades = []
                
                for row in rows:
                    trade = TradeRecord(
                        trade_id=row['trade_id'],
                        symbol=row['symbol'],
                        entry_time=datetime.fromisoformat(row['entry_time']),
                        exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        position_size=row['position_size'],
                        side=row['side'],
                        pnl=row['pnl'],
                        pnl_percentage=row['pnl_percentage'],
                        commission=row['commission'],
                        slippage=row['slippage'],
                        stop_loss=row['stop_loss'],
                        take_profit=row['take_profit'],
                        strategy=row['strategy'],
                        confidence=row['confidence'],
                        market_condition=row['market_condition'],
                        tags=json.loads(row['tags']),
                        metadata=json.loads(row['metadata'])
                    )
                    trades.append(trade)
                
                return trades
                
        except Exception as e:
            logger.error(f"Trade loading failed: {e}")
            return []

    def _load_snapshots_period(self, start_date: datetime, end_date: datetime) -> List[PerformanceSnapshot]:
        """Load performance snapshots for specific period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM performance_snapshots 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (start_date, end_date))
                
                rows = cursor.fetchall()
                snapshots = []
                
                for row in rows:
                    snapshot = PerformanceSnapshot(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        portfolio_value=row['portfolio_value'],
                        cash_balance=row['cash_balance'],
                        positions_value=row['positions_value'],
                        total_pnl=row['total_pnl'],
                        daily_pnl=row['daily_pnl'],
                        drawdown=row['drawdown'],
                        max_drawdown=row['max_drawdown'],
                        volatility=row['volatility'],
                        sharpe_ratio=row['sharpe_ratio'],
                        sortino_ratio=row['sortino_ratio'],
                        win_rate=row['win_rate'],
                        trade_count=row['trade_count'],
                        open_trades=row['open_trades'],
                        market_regime=row['market_regime'],
                        risk_metrics=json.loads(row['risk_metrics'])
                    )
                    snapshots.append(snapshot)
                
                return snapshots
                
        except Exception as e:
            logger.error(f"Snapshot loading failed: {e}")
            return []

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate/252 for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]
        
        if not downside_returns:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std * np.sqrt(252)

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd

    def _calculate_drawdowns(self, portfolio_values: List[float]) -> List[float]:
        """Calculate all drawdowns"""
        if not portfolio_values:
            return []
        
        drawdowns = []
        peak = portfolio_values[0]
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        return drawdowns

    def _calculate_daily_returns(self, snapshots: List[PerformanceSnapshot]) -> List[float]:
        """Calculate daily returns from snapshots"""
        if not snapshots:
            return []
        
        # Group by day
        daily_values = {}
        for snapshot in snapshots:
            date = snapshot.timestamp.date()
            daily_values[date] = snapshot.portfolio_value
        
        # Calculate daily returns
        sorted_dates = sorted(daily_values.keys())
        returns = []
        
        for i in range(1, len(sorted_dates)):
            prev_value = daily_values[sorted_dates[i-1]]
            curr_value = daily_values[sorted_dates[i]]
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        return returns

    def _calculate_drawdown_duration(self, drawdowns: List[float]) -> int:
        """Calculate average drawdown duration"""
        if not drawdowns:
            return 0
        
        in_drawdown = False
        drawdown_start = 0
        durations = []
        
        for i, dd in enumerate(drawdowns):
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                durations.append(i - drawdown_start)
        
        return statistics.mean(durations) if durations else 0

    def _calculate_beta(self, returns: List[float]) -> float:
        """Calculate beta (market correlation) - simplified"""
        if len(returns) < 2:
            return 1.0
        
        # For simplicity, using S&P 500 as benchmark (replace with actual benchmark data)
        market_returns = [0.001] * len(returns)  # Placeholder
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 1.0

    def _calculate_alpha(self, returns: List[float]) -> float:
        """Calculate alpha (excess returns) - simplified"""
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        market_return = 0.0005  # Placeholder for market return
        beta = self._calculate_beta(returns)
        
        return (avg_return - 0.02/252) - beta * (market_return - 0.02/252)  # CAPM formula

    def _calculate_tracking_error(self, returns: List[float]) -> float:
        """Calculate tracking error"""
        if len(returns) < 2:
            return 0.0
        
        market_returns = [0.001] * len(returns)  # Placeholder
        active_returns = [r - m for r, m in zip(returns, market_returns)]
        return np.std(active_returns) * np.sqrt(252)

    def _calculate_information_ratio(self, returns: List[float]) -> float:
        """Calculate information ratio"""
        if not returns:
            return 0.0
        
        market_returns = [0.001] * len(returns)  # Placeholder
        active_returns = [r - m for r, m in zip(returns, market_returns)]
        tracking_error = self._calculate_tracking_error(returns)
        
        return np.mean(active_returns) / tracking_error if tracking_error != 0 else 0.0

    def _calculate_confidence_correlation(self, trades: List[TradeRecord]) -> float:
        """Calculate correlation between confidence and PnL"""
        try:
            if len(trades) < 2:
                return 0.0
            
            confidences = [t.confidence for t in trades]
            pnls = [t.pnl or 0 for t in trades]
            
            correlation = np.corrcoef(confidences, pnls)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except:
            return 0.0

    def _save_report(self, report: PerformanceReport, start_date: datetime, end_date: datetime):
        """Save performance report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.log_dir / f"performance_report_{timestamp}.json"
            
            # Convert report to serializable format
            report_dict = {
                'summary': report.summary,
                'trade_analysis': report.trade_analysis,
                'risk_analysis': report.risk_analysis,
                'strategy_analysis': report.strategy_analysis,
                'time_analysis': report.time_analysis,
                'recommendations': report.recommendations,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period_start': start_date.isoformat(),
                    'period_end': end_date.isoformat(),
                    'total_trades': report.summary.get('total_trades', 0)
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Performance report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Report saving failed: {e}")

    def export_trade_data(self, format: str = 'csv', filename: str = None) -> str:
        """Export trade data to various formats"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trades_export_{timestamp}.{format}"
            
            filepath = self.log_dir / filename
            
            # Load all trades
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM trades", conn)
            
            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'excel':
                df.to_excel(filepath, index=False)
            elif format == 'json':
                df.to_json(filepath, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Trade data exported: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Trade data export failed: {e}")
            raise

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old performance data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old trades
                cursor.execute('DELETE FROM trades WHERE entry_time < ?', (cutoff_date,))
                
                # Delete old snapshots
                cursor.execute('DELETE FROM performance_snapshots WHERE timestamp < ?', (cutoff_date,))
                
                # Delete old metrics
                cursor.execute('DELETE FROM metric_history WHERE timestamp < ?', (cutoff_date,))
                
                conn.commit()
            
            logger.info(f"Cleaned up performance data older than {cutoff_date}")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time performance metrics"""
        try:
            with self._lock:
                return {
                    'current_portfolio_value': self.current_portfolio_value,
                    'peak_portfolio_value': self.peak_portfolio_value,
                    'total_trades': self.total_trades,
                    'win_count': self.win_count,
                    'loss_count': self.loss_count,
                    'win_rate': self.win_count / self.total_trades if self.total_trades > 0 else 0,
                    'total_commission': self.total_commission,
                    'total_slippage': self.total_slippage,
                    'current_drawdown': (self.peak_portfolio_value - self.current_portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
                }
        except Exception as e:
            logger.error(f"Real-time metrics retrieval failed: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize logger
    pl = PerformanceLogger()
    
    # Example trade
    trade = TradeRecord(
        trade_id="TRADE_001",
        symbol="EUR/USD",
        entry_time=datetime.now() - timedelta(hours=2),
        exit_time=datetime.now(),
        entry_price=1.0850,
        exit_price=1.0860,
        position_size=10000,
        side="long",
        pnl=100.0,
        pnl_percentage=0.01,
        commission=5.0,
        slippage=2.0,
        stop_loss=1.0830,
        take_profit=1.0870,
        strategy="momentum",
        confidence=0.75,
        market_condition="trending",
        tags=["scalping", "high_confidence"]
    )
    
    # Log trade
    pl.log_trade(trade)
    
    # Generate report
    report = pl.generate_performance_report()
    print("Performance Report Generated Successfully!")
    print(f"Win Rate: {report.summary.get('win_rate', 0):.2%}")
    print(f"Total PnL: ${report.summary.get('total_pnl', 0):.2f}")
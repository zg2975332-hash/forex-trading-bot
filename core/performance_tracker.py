"""
Advanced Performance Tracker for Forex Trading Bot
Comprehensive performance monitoring, analytics, and reporting
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sqlite3
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Performance metric types"""
    ABSOLUTE = "absolute"
    RISK_ADJUSTED = "risk_adjusted"
    TRADING = "trading"
    PORTFOLIO = "portfolio"

@dataclass
class TradeRecord:
    """Individual trade record"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_type: str  # 'LONG' or 'SHORT'
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    holding_period: timedelta
    strategy: str
    confidence: float
    stop_loss: float
    take_profit: float
    exit_reason: str

@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at specific time"""
    timestamp: datetime
    total_equity: float
    cash: float
    used_margin: float
    free_margin: float
    open_positions: int
    unrealized_pnl: float
    realized_pnl: float
    portfolio_value: float

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    var_95: float
    cvar_95: float
    
    # Trading metrics
    profit_factor: float
    expectancy: float
    average_win: float
    average_loss: float
    avg_win_to_avg_loss_ratio: float
    largest_winning_trade: float
    largest_losing_trade: float
    
    # Portfolio metrics
    volatility_annual: float
    beta: float
    alpha_annual: float
    information_ratio: float
    tracking_error: float
    
    # Advanced metrics
    kelly_criterion: float
    ulcer_index: float
    tail_ratio: float
    common_sense_ratio: float
    gain_to_pain_ratio: float
    recovery_factor: float
    
    # Time-based metrics
    avg_holding_period_hours: float
    trades_per_day: float
    profit_per_day: float
    
    # Streaks
    longest_win_streak: int
    longest_loss_streak: int
    current_streak: int
    current_streak_type: str
    
    # Confidence intervals
    confidence_interval_95: Tuple[float, float]

class DatabaseManager:
    """Database management for performance data"""
    
    def __init__(self, db_path: str = "data/performance.db"):
        self.db_path = db_path
        self._init_database()
        logger.info(f"Database manager initialized: {db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
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
                position_type TEXT,
                quantity REAL,
                pnl REAL,
                pnl_pct REAL,
                commission REAL,
                slippage REAL,
                holding_period_seconds REAL,
                strategy TEXT,
                confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                exit_reason TEXT
            )
        ''')
        
        # Portfolio snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp TIMESTAMP PRIMARY KEY,
                total_equity REAL,
                cash REAL,
                used_margin REAL,
                free_margin REAL,
                open_positions INTEGER,
                unrealized_pnl REAL,
                realized_pnl REAL,
                portfolio_value REAL
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp TIMESTAMP PRIMARY KEY,
                metrics_json TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_trade(self, trade: TradeRecord):
        """Save trade record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trades VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            trade.trade_id,
            trade.symbol,
            trade.entry_time.isoformat(),
            trade.exit_time.isoformat(),
            trade.entry_price,
            trade.exit_price,
            trade.position_type,
            trade.quantity,
            trade.pnl,
            trade.pnl_pct,
            trade.commission,
            trade.slippage,
            trade.holding_period.total_seconds(),
            trade.strategy,
            trade.confidence,
            trade.stop_loss,
            trade.take_profit,
            trade.exit_reason
        ))
        
        conn.commit()
        conn.close()
    
    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot):
        """Save portfolio snapshot to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_snapshots VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            snapshot.timestamp.isoformat(),
            snapshot.total_equity,
            snapshot.cash,
            snapshot.used_margin,
            snapshot.free_margin,
            snapshot.open_positions,
            snapshot.unrealized_pnl,
            snapshot.realized_pnl,
            snapshot.portfolio_value
        ))
        
        conn.commit()
        conn.close()
    
    def save_performance_metrics(self, timestamp: datetime, metrics: PerformanceMetrics):
        """Save performance metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metrics_dict = asdict(metrics)
        metrics_json = json.dumps(metrics_dict, default=str)
        
        cursor.execute('''
            INSERT OR REPLACE INTO performance_metrics VALUES (?, ?)
        ''', (timestamp.isoformat(), metrics_json))
        
        conn.commit()
        conn.close()
    
    def load_trades(self, start_date: datetime = None, end_date: datetime = None) -> List[TradeRecord]:
        """Load trades from database with optional date range"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades"
        params = []
        
        if start_date or end_date:
            query += " WHERE 1=1"
            if start_date:
                query += " AND exit_time >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND exit_time <= ?"
                params.append(end_date.isoformat())
        
        query += " ORDER BY exit_time"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        trades = []
        for _, row in df.iterrows():
            trade = TradeRecord(
                trade_id=row['trade_id'],
                symbol=row['symbol'],
                entry_time=datetime.fromisoformat(row['entry_time']),
                exit_time=datetime.fromisoformat(row['exit_time']),
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                position_type=row['position_type'],
                quantity=row['quantity'],
                pnl=row['pnl'],
                pnl_pct=row['pnl_pct'],
                commission=row['commission'],
                slippage=row['slippage'],
                holding_period=timedelta(seconds=row['holding_period_seconds']),
                strategy=row['strategy'],
                confidence=row['confidence'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                exit_reason=row['exit_reason']
            )
            trades.append(trade)
        
        return trades
    
    def load_portfolio_snapshots(self, start_date: datetime = None, end_date: datetime = None) -> List[PortfolioSnapshot]:
        """Load portfolio snapshots from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM portfolio_snapshots"
        params = []
        
        if start_date or end_date:
            query += " WHERE 1=1"
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        snapshots = []
        for _, row in df.iterrows():
            snapshot = PortfolioSnapshot(
                timestamp=datetime.fromisoformat(row['timestamp']),
                total_equity=row['total_equity'],
                cash=row['cash'],
                used_margin=row['used_margin'],
                free_margin=row['free_margin'],
                open_positions=row['open_positions'],
                unrealized_pnl=row['unrealized_pnl'],
                realized_pnl=row['realized_pnl'],
                portfolio_value=row['portfolio_value']
            )
            snapshots.append(snapshot)
        
        return snapshots

class PerformanceAnalyzer:
    """Advanced performance analysis and metrics calculation"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.database = DatabaseManager()
        logger.info("Performance Analyzer initialized")
    
    def calculate_comprehensive_metrics(self, trades: List[TradeRecord], 
                                     portfolio_snapshots: List[PortfolioSnapshot]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._create_empty_metrics()
        
        try:
            # Convert to DataFrames for easier analysis
            trades_df = self._trades_to_dataframe(trades)
            portfolio_df = self._portfolio_to_dataframe(portfolio_snapshots)
            
            # Basic metrics
            basic_metrics = self._calculate_basic_metrics(trades_df)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(trades_df, portfolio_df)
            
            # Trading metrics
            trading_metrics = self._calculate_trading_metrics(trades_df)
            
            # Portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(portfolio_df)
            
            # Advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(trades_df, portfolio_df)
            
            # Time-based metrics
            time_metrics = self._calculate_time_metrics(trades_df)
            
            # Streaks
            streak_metrics = self._calculate_streak_metrics(trades_df)
            
            # Confidence intervals
            confidence_interval = self._calculate_confidence_interval(trades_df)
            
            # Combine all metrics
            metrics = PerformanceMetrics(
                **basic_metrics,
                **risk_metrics,
                **trading_metrics,
                **portfolio_metrics,
                **advanced_metrics,
                **time_metrics,
                **streak_metrics,
                confidence_interval_95=confidence_interval
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._create_empty_metrics()
    
    def _trades_to_dataframe(self, trades: List[TradeRecord]) -> pd.DataFrame:
        """Convert trades list to DataFrame"""
        data = []
        for trade in trades:
            data.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_type': trade.position_type,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'holding_period_hours': trade.holding_period.total_seconds() / 3600,
                'strategy': trade.strategy,
                'confidence': trade.confidence,
                'exit_reason': trade.exit_reason
            })
        
        return pd.DataFrame(data)
    
    def _portfolio_to_dataframe(self, snapshots: List[PortfolioSnapshot]) -> pd.DataFrame:
        """Convert portfolio snapshots to DataFrame"""
        if not snapshots:
            return pd.DataFrame()
        
        data = []
        for snapshot in snapshots:
            data.append({
                'timestamp': snapshot.timestamp,
                'total_equity': snapshot.total_equity,
                'portfolio_value': snapshot.portfolio_value,
                'realized_pnl': snapshot.realized_pnl,
                'unrealized_pnl': snapshot.unrealized_pnl
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_basic_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct
        }
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame, 
                              portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-adjusted metrics"""
        # Sharpe Ratio
        returns = trades_df['pnl_pct'] / 100  # Convert to decimal
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < risk_free_rate]
        downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0
        
        if downside_deviation > 0:
            sortino_ratio = (returns.mean() - risk_free_rate) / downside_deviation * np.sqrt(252)
        else:
            sortino_ratio = 0.0
        
        # Maximum Drawdown
        if not portfolio_df.empty:
            portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
            max_drawdown = portfolio_df['drawdown'].min()
            max_drawdown_pct = abs(max_drawdown * 100)
        else:
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
        
        # Calmar Ratio
        if max_drawdown_pct > 0:
            calmar_ratio = (returns.mean() * 252) / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        # Value at Risk and Conditional VaR
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5) * 100  # 5th percentile loss
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_trading_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trading-specific metrics"""
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        # Profit Factor
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Win/Loss ratio
        avg_win_to_avg_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Largest trades
        largest_win = winning_trades['pnl'].max() if not winning_trades.empty else 0
        largest_loss = losing_trades['pnl'].min() if not losing_trades.empty else 0
        
        return {
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'avg_win_to_avg_loss_ratio': avg_win_to_avg_loss_ratio,
            'largest_winning_trade': largest_win,
            'largest_losing_trade': largest_loss
        }
    
    def _calculate_portfolio_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        if portfolio_df.empty:
            return {
                'volatility_annual': 0.0,
                'beta': 0.0,
                'alpha_annual': 0.0,
                'information_ratio': 0.0,
                'tracking_error': 0.0
            }
        
        # Portfolio returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        if len(portfolio_returns) < 2:
            return {
                'volatility_annual': 0.0,
                'beta': 0.0,
                'alpha_annual': 0.0,
                'information_ratio': 0.0,
                'tracking_error': 0.0
            }
        
        # Volatility (annualized)
        volatility_annual = portfolio_returns.std() * np.sqrt(252)
        
        # For beta and alpha, we would need benchmark data
        # Using simplified assumptions for demo
        benchmark_returns = np.random.normal(0.0005, 0.01, len(portfolio_returns))  # Simulated benchmark
        
        # Beta (simplified)
        if len(portfolio_returns) > 1:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        else:
            beta = 1.0
        
        # Alpha (simplified)
        risk_free_rate = 0.02 / 252
        portfolio_excess_return = portfolio_returns.mean() - risk_free_rate
        benchmark_excess_return = benchmark_returns.mean() - risk_free_rate
        alpha_annual = (portfolio_excess_return - beta * benchmark_excess_return) * 252
        
        # Information Ratio
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0.0
        
        return {
            'volatility_annual': volatility_annual,
            'beta': beta,
            'alpha_annual': alpha_annual,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def _calculate_advanced_metrics(self, trades_df: pd.DataFrame, 
                                  portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        returns = trades_df['pnl_pct'] / 100
        
        if len(returns) < 2:
            return {
                'kelly_criterion': 0.0,
                'ulcer_index': 0.0,
                'tail_ratio': 0.0,
                'common_sense_ratio': 0.0,
                'gain_to_pain_ratio': 0.0,
                'recovery_factor': 0.0
            }
        
        # Kelly Criterion
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() / 100 if win_rate > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl_pct'].mean() / 100) if win_rate < 1 else 0
        
        if avg_loss > 0:
            kelly_criterion = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        else:
            kelly_criterion = 0.0
        
        # Ulcer Index
        if not portfolio_df.empty:
            portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
            squared_drawdowns = portfolio_df['drawdown'] ** 2
            ulcer_index = np.sqrt(squared_drawdowns.mean())
        else:
            ulcer_index = 0.0
        
        # Tail Ratio (95th percentile gain / 5th percentile loss)
        if len(returns) > 0:
            tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
        else:
            tail_ratio = 0.0
        
        # Common Sense Ratio (Sharpe * Profit Factor)
        sharpe_ratio = (returns.mean() - 0.02/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        common_sense_ratio = sharpe_ratio * profit_factor
        
        # Gain to Pain Ratio
        total_gain = gross_profit
        total_pain = gross_loss
        gain_to_pain_ratio = total_gain / total_pain if total_pain > 0 else float('inf')
        
        # Recovery Factor
        total_pnl = trades_df['pnl'].sum()
        max_drawdown = abs(self._calculate_max_drawdown(portfolio_df))
        recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else float('inf')
        
        return {
            'kelly_criterion': kelly_criterion,
            'ulcer_index': ulcer_index,
            'tail_ratio': tail_ratio,
            'common_sense_ratio': common_sense_ratio,
            'gain_to_pain_ratio': gain_to_pain_ratio,
            'recovery_factor': recovery_factor
        }
    
    def _calculate_max_drawdown(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown from portfolio data"""
        if portfolio_df.empty:
            return 0.0
        
        portfolio_df = portfolio_df.copy()
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        return portfolio_df['drawdown'].min()
    
    def _calculate_time_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-based metrics"""
        if trades_df.empty:
            return {
                'avg_holding_period_hours': 0.0,
                'trades_per_day': 0.0,
                'profit_per_day': 0.0
            }
        
        # Average holding period
        avg_holding_period = trades_df['holding_period_hours'].mean()
        
        # Trades per day
        if len(trades_df) > 1:
            first_trade = trades_df['exit_time'].min()
            last_trade = trades_df['exit_time'].max()
            total_days = (last_trade - first_trade).days + 1
            trades_per_day = len(trades_df) / total_days
        else:
            trades_per_day = 0.0
        
        # Profit per day
        total_pnl = trades_df['pnl'].sum()
        profit_per_day = total_pnl / total_days if total_days > 0 else 0.0
        
        return {
            'avg_holding_period_hours': avg_holding_period,
            'trades_per_day': trades_per_day,
            'profit_per_day': profit_per_day
        }
    
    def _calculate_streak_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate winning and losing streaks"""
        if trades_df.empty:
            return {
                'longest_win_streak': 0,
                'longest_loss_streak': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }
        
        # Calculate streaks
        current_streak = 0
        current_streak_type = None
        longest_win_streak = 0
        longest_loss_streak = 0
        
        for _, trade in trades_df.iterrows():
            if trade['pnl'] > 0:  # Winning trade
                if current_streak_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = 'win'
                longest_win_streak = max(longest_win_streak, current_streak)
                
            elif trade['pnl'] < 0:  # Losing trade
                if current_streak_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = 'loss'
                longest_loss_streak = max(longest_loss_streak, current_streak)
        
        return {
            'longest_win_streak': longest_win_streak,
            'longest_loss_streak': longest_loss_streak,
            'current_streak': current_streak,
            'current_streak_type': current_streak_type or 'none'
        }
    
    def _calculate_confidence_interval(self, trades_df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate 95% confidence interval for returns"""
        if len(trades_df) < 2:
            return (0.0, 0.0)
        
        returns = trades_df['pnl_pct'] / 100
        mean_return = returns.mean()
        std_error = returns.std() / np.sqrt(len(returns))
        
        # 95% confidence interval
        confidence_interval = stats.norm.interval(0.95, loc=mean_return, scale=std_error)
        
        return (confidence_interval[0] * 100, confidence_interval[1] * 100)  # Convert back to percentage
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics when no data is available"""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            var_95=0.0,
            cvar_95=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            average_win=0.0,
            average_loss=0.0,
            avg_win_to_avg_loss_ratio=0.0,
            largest_winning_trade=0.0,
            largest_losing_trade=0.0,
            volatility_annual=0.0,
            beta=0.0,
            alpha_annual=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
            kelly_criterion=0.0,
            ulcer_index=0.0,
            tail_ratio=0.0,
            common_sense_ratio=0.0,
            gain_to_pain_ratio=0.0,
            recovery_factor=0.0,
            avg_holding_period_hours=0.0,
            trades_per_day=0.0,
            profit_per_day=0.0,
            longest_win_streak=0,
            longest_loss_streak=0,
            current_streak=0,
            current_streak_type='none',
            confidence_interval_95=(0.0, 0.0)
        )

class PerformanceTracker:
    """
    Main Performance Tracker coordinating tracking, analysis, and reporting
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.analyzer = PerformanceAnalyzer(initial_capital)
        self.database = DatabaseManager()
        
        # Live tracking
        self.current_trades: Dict[str, TradeRecord] = {}
        self.portfolio_snapshots: List[PortfolioSnapshot] = []
        
        # Performance history
        self.performance_history: List[PerformanceMetrics] = []
        
        logger.info("Performance Tracker initialized")
    
    def record_trade_entry(self, trade: TradeRecord):
        """Record trade entry"""
        self.current_trades[trade.trade_id] = trade
        logger.info(f"Trade entry recorded: {trade.trade_id}")
    
    def record_trade_exit(self, trade_id: str, exit_price: float, exit_time: datetime, exit_reason: str):
        """Record trade exit and calculate P&L"""
        if trade_id not in self.current_trades:
            logger.warning(f"Trade {trade_id} not found in current trades")
            return
        
        trade = self.current_trades[trade_id]
        
        # Calculate P&L
        if trade.position_type == 'LONG':
            pnl = (exit_price - trade.entry_price) * trade.quantity - trade.commission - trade.slippage
        else:  # SHORT
            pnl = (trade.entry_price - exit_price) * trade.quantity - trade.commission - trade.slippage
        
        pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100
        
        # Update trade record
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.holding_period = exit_time - trade.entry_time
        trade.exit_reason = exit_reason
        
        # Save to database
        self.database.save_trade(trade)
        
        # Remove from current trades
        del self.current_trades[trade_id]
        
        logger.info(f"Trade exit recorded: {trade_id}, P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
    
    def record_portfolio_snapshot(self, snapshot: PortfolioSnapshot):
        """Record portfolio snapshot"""
        self.portfolio_snapshots.append(snapshot)
        self.database.save_portfolio_snapshot(snapshot)
        
        # Keep only recent snapshots in memory
        if len(self.portfolio_snapshots) > 1000:
            self.portfolio_snapshots = self.portfolio_snapshots[-1000:]
    
    def generate_performance_report(self, period: str = "all") -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Determine date range
            end_date = datetime.now()
            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "1w":
                start_date = end_date - timedelta(weeks=1)
            elif period == "1m":
                start_date = end_date - timedelta(days=30)
            elif period == "3m":
                start_date = end_date - timedelta(days=90)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = None
            
            # Load data
            trades = self.database.load_trades(start_date, end_date)
            portfolio_snapshots = self.database.load_portfolio_snapshots(start_date, end_date)
            
            # Calculate metrics
            metrics = self.analyzer.calculate_comprehensive_metrics(trades, portfolio_snapshots)
            
            # Generate report
            report = {
                'period': period,
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_trades': metrics.total_trades,
                    'win_rate': f"{metrics.win_rate:.1%}",
                    'total_pnl': f"${metrics.total_pnl:,.2f}",
                    'total_pnl_pct': f"{metrics.total_pnl_pct:.2f}%",
                    'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                    'max_drawdown': f"{metrics.max_drawdown_pct:.2f}%"
                },
                'detailed_metrics': asdict(metrics),
                'recent_trades': [
                    {
                        'trade_id': trade.trade_id,
                        'symbol': trade.symbol,
                        'entry_time': trade.entry_time.isoformat(),
                        'exit_time': trade.exit_time.isoformat(),
                        'pnl': trade.pnl,
                        'pnl_pct': trade.pnl_pct,
                        'strategy': trade.strategy,
                        'exit_reason': trade.exit_reason
                    }
                    for trade in trades[-10:]  # Last 10 trades
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def create_visualizations(self, output_dir: str = "reports"):
        """Create performance visualization charts"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Load recent data
            trades = self.database.load_trades()
            portfolio_snapshots = self.database.load_portfolio_snapshots()
            
            if not trades or not portfolio_snapshots:
                logger.warning("Insufficient data for visualizations")
                return
            
            # Convert to DataFrames
            trades_df = self.analyzer._trades_to_dataframe(trades)
            portfolio_df = self.analyzer._portfolio_to_dataframe(portfolio_snapshots)
            
            # Create plots
            self._create_equity_curve_plot(portfolio_df, output_dir)
            self._create_drawdown_plot(portfolio_df, output_dir)
            self._create_returns_histogram(trades_df, output_dir)
            self._create_monthly_returns_heatmap(portfolio_df, output_dir)
            
            logger.info(f"Visualizations created in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _create_equity_curve_plot(self, portfolio_df: pd.DataFrame, output_dir: str):
        """Create equity curve plot"""
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], linewidth=2)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_drawdown_plot(self, portfolio_df: pd.DataFrame, output_dir: str):
        """Create drawdown plot"""
        portfolio_df = portfolio_df.copy()
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(portfolio_df.index, portfolio_df['drawdown'] * 100, 0, alpha=0.3, color='red')
        plt.plot(portfolio_df.index, portfolio_df['drawdown'] * 100, color='red', linewidth=1)
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_returns_histogram(self, trades_df: pd.DataFrame, output_dir: str):
        """Create returns distribution histogram"""
        plt.figure(figsize=(10, 6))
        plt.hist(trades_df['pnl_pct'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(trades_df['pnl_pct'].mean(), color='red', linestyle='--', label=f'Mean: {trades_df["pnl_pct"].mean():.2f}%')
        plt.title('Trade Returns Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/returns_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_monthly_returns_heatmap(self, portfolio_df: pd.DataFrame, output_dir: str):
        """Create monthly returns heatmap"""
        try:
            # Calculate monthly returns
            monthly_returns = portfolio_df['portfolio_value'].resample('M').last().pct_change().dropna()
            monthly_returns = monthly_returns * 100  # Convert to percentage
            
            # Create heatmap data
            years = monthly_returns.index.year.unique()
            months = range(1, 13)
            
            heatmap_data = pd.DataFrame(index=years, columns=months)
            
            for date, ret in monthly_returns.items():
                heatmap_data.loc[date.year, date.month] = ret
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, cbar_kws={'label': 'Return (%)'})
            plt.title('Monthly Returns Heatmap (%)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create monthly heatmap: {e}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        try:
            # Load recent data (last 30 days)
            start_date = datetime.now() - timedelta(days=30)
            trades = self.database.load_trades(start_date)
            portfolio_snapshots = self.database.load_portfolio_snapshots(start_date)
            
            metrics = self.analyzer.calculate_comprehensive_metrics(trades, portfolio_snapshots)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_open_trades': len(self.current_trades),
                'recent_performance': asdict(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Test the Performance Tracker
    print("Testing Performance Tracker...")
    
    try:
        # Initialize tracker
        tracker = PerformanceTracker(initial_capital=10000.0)
        
        # Generate sample trades
        print("Generating sample trades...")
        sample_trades = []
        
        for i in range(100):
            trade = TradeRecord(
                trade_id=f"TRADE_{i:03d}",
                symbol="EUR/USD",
                entry_time=datetime.now() - timedelta(hours=i*2),
                exit_time=datetime.now() - timedelta(hours=i*2 - 1),
                entry_price=1.1000 + i * 0.0001,
                exit_price=1.1000 + i * 0.0001 + np.random.normal(0, 0.0005),
                position_type="LONG" if i % 2 == 0 else "SHORT",
                quantity=0.1,
                pnl=0.0,  # Will be calculated
                pnl_pct=0.0,
                commission=0.2,
                slippage=0.1,
                holding_period=timedelta(hours=1),
                strategy="momentum",
                confidence=0.7 + np.random.random() * 0.3,
                stop_loss=1.0950,
                take_profit=1.1050,
                exit_reason="target"
            )
            
            # Calculate P&L
            if trade.position_type == "LONG":
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity - trade.commission - trade.slippage
            else:
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity - trade.commission - trade.slippage
            
            trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
            
            sample_trades.append(trade)
            tracker.database.save_trade(trade)
        
        # Generate sample portfolio snapshots
        print("Generating sample portfolio snapshots...")
        current_value = 10000.0
        
        for i in range(500):
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now() - timedelta(hours=500 - i),
                total_equity=current_value,
                cash=current_value * 0.8,
                used_margin=current_value * 0.2,
                free_margin=current_value * 0.6,
                open_positions=np.random.randint(0, 5),
                unrealized_pnl=np.random.normal(0, 50),
                realized_pnl=i * 10,
                portfolio_value=current_value
            )
            
            # Random walk for portfolio value
            current_value *= 1 + np.random.normal(0, 0.002)
            
            tracker.database.save_portfolio_snapshot(snapshot)
        
        # Generate performance report
        print("Generating performance report...")
        report = tracker.generate_performance_report("all")
        
        print("\n📊 PERFORMANCE REPORT SUMMARY:")
        print(f"Total Trades: {report['summary']['total_trades']}")
        print(f"Win Rate: {report['summary']['win_rate']}")
        print(f"Total P&L: {report['summary']['total_pnl']}")
        print(f"Total P&L %: {report['summary']['total_pnl_pct']}")
        print(f"Sharpe Ratio: {report['summary']['sharpe_ratio']}")
        print(f"Max Drawdown: {report['summary']['max_drawdown']}")
        
        # Test real-time metrics
        print("\nTesting real-time metrics...")
        real_time_metrics = tracker.get_real_time_metrics()
        print(f"Current Open Trades: {real_time_metrics['current_open_trades']}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        tracker.create_visualizations("test_reports")
        print("Visualizations created in 'test_reports' directory")
        
        print(f"\n✅ Performance Tracker test completed successfully!")
        
    except Exception as e:
        print(f"❌ Performance Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
"""
Advanced Backtester for Forex Trading Bot
Comprehensive backtesting with advanced metrics and walk-forward analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import traceback

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade data structure"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_type: str  # 'LONG' or 'SHORT'
    quantity: float
    pnl: float
    pnl_pct: float
    holding_period: timedelta
    signal_strength: float
    stop_loss: float
    take_profit: float

@dataclass
class BacktestResult:
    """Backtest results container"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    average_trade: float
    average_win: float
    average_loss: float
    expectancy: float
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    monthly_returns: pd.Series
    metrics: Dict[str, float]

class AdvancedBacktester:
    """
    Advanced backtesting engine with walk-forward optimization
    and comprehensive performance analytics
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0002):
        self.initial_capital = initial_capital
        self.commission = commission  # 0.02% commission
        self.current_capital = initial_capital
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.dates = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        
        logger.info(f"Advanced Backtester initialized with capital: ${initial_capital:,.2f}")
    
    def calculate_advanced_metrics(self, trades: List[Trade], equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            trades: List of trade objects
            equity_curve: Series of equity values over time
            
        Returns:
            Dict of advanced performance metrics
        """
        try:
            if len(trades) == 0:
                return {}
            
            # Basic metrics
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            total_trades = len(trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t.pnl for t in trades)
            total_pnl_pct = (total_pnl / self.initial_capital) * 100
            
            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average trade metrics
            average_trade = total_pnl / total_trades if total_trades > 0 else 0
            average_win = gross_profit / len(winning_trades) if winning_trades else 0
            average_loss = gross_loss / len(losing_trades) if losing_trades else 0
            
            # Expectancy
            expectancy = (win_rate * average_win) - ((1 - win_rate) * abs(average_loss))
            
            # Sharpe Ratio (annualized)
            returns = [t.pnl_pct / 100 for t in trades]  # Convert to decimal
            if len(returns) > 1:
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum Drawdown
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            max_drawdown_pct = max_drawdown * 100
            
            # Calmar Ratio
            calmar_ratio = (total_pnl_pct / 100) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Recovery Factor
            recovery_factor = total_pnl / abs(self.initial_capital * max_drawdown) if max_drawdown != 0 else 0
            
            # Trade statistics
            holding_periods = [t.holding_period.total_seconds() / 3600 for t in trades]  # in hours
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            # Monthly returns
            monthly_returns = self.calculate_monthly_returns(trades)
            monthly_return_std = monthly_returns.std() if len(monthly_returns) > 0 else 0
            
            # Win/Loss streaks
            streaks = self.calculate_streaks(trades)
            
            # Risk-adjusted metrics
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_returns = np.mean(returns) - (risk_free_rate / 252)
            sortino_ratio = (excess_returns / self.calculate_downside_deviation(returns)) * np.sqrt(252) if self.calculate_downside_deviation(returns) > 0 else 0
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate * 100,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown_pct': max_drawdown_pct,
                'max_drawdown': max_drawdown,
                'recovery_factor': recovery_factor,
                'average_trade': average_trade,
                'average_win': average_win,
                'average_loss': average_loss,
                'expectancy': expectancy,
                'avg_holding_period_hours': avg_holding_period,
                'monthly_return_std': monthly_return_std,
                'longest_win_streak': streaks['longest_win_streak'],
                'longest_loss_streak': streaks['longest_loss_streak'],
                'largest_winning_trade': max(t.pnl for t in trades) if trades else 0,
                'largest_losing_trade': min(t.pnl for t in trades) if trades else 0,
                'avg_win_to_avg_loss_ratio': abs(average_win / average_loss) if average_loss != 0 else 0,
                'commission_costs': total_trades * self.commission * self.initial_capital
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            traceback.print_exc()
            return {}
    
    def calculate_downside_deviation(self, returns: List[float]) -> float:
        """Calculate downside deviation for Sortino ratio"""
        if not returns:
            return 0.0
        downside_returns = [r for r in returns if r < 0]
        return np.std(downside_returns) if downside_returns else 0.0
    
    def calculate_streaks(self, trades: List[Trade]) -> Dict[str, int]:
        """Calculate winning and losing streaks"""
        if not trades:
            return {'longest_win_streak': 0, 'longest_loss_streak': 0}
        
        current_streak = 0
        current_streak_type = None
        longest_win_streak = 0
        longest_loss_streak = 0
        
        for trade in trades:
            if trade.pnl > 0:  # Winning trade
                if current_streak_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = 'win'
                longest_win_streak = max(longest_win_streak, current_streak)
                
            elif trade.pnl < 0:  # Losing trade
                if current_streak_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = 'loss'
                longest_loss_streak = max(longest_loss_streak, current_streak)
        
        return {
            'longest_win_streak': longest_win_streak,
            'longest_loss_streak': longest_loss_streak
        }
    
    def calculate_monthly_returns(self, trades: List[Trade]) -> pd.Series:
        """Calculate monthly returns from trades"""
        if not trades:
            return pd.Series()
        
        # Group trades by month
        monthly_data = {}
        for trade in trades:
            month_key = trade.exit_time.strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = 0.0
            monthly_data[month_key] += trade.pnl
        
        return pd.Series(monthly_data)
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy_function: Callable,
                    position_size: float = 0.1,  # 10% of capital per trade
                    stop_loss_pct: float = 0.02,  # 2% stop loss
                    take_profit_pct: float = 0.04,  # 4% take profit
                    **strategy_params) -> BacktestResult:
        """
        Run comprehensive backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            strategy_function: Function that generates trading signals
            position_size: Fraction of capital to risk per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            **strategy_params: Additional parameters for strategy
            
        Returns:
            BacktestResult object with complete analysis
        """
        try:
            logger.info(f"Starting backtest with {len(data)} data points")
            
            # Reset state
            self.current_capital = self.initial_capital
            self.trades = []
            self.equity_curve = [self.initial_capital]
            self.drawdown_curve = [0.0]
            self.dates = [data.index[0]]
            self.peak_equity = self.initial_capital
            self.current_drawdown = 0.0
            
            current_position = None
            entry_price = 0.0
            entry_time = None
            position_type = None
            quantity = 0.0
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                current_price = row['close']
                
                # Generate trading signal
                signal_data = data.iloc[:i+1].copy() if i > 0 else data.iloc[:1].copy()
                signal = strategy_function(signal_data, **strategy_params)
                
                # Update equity curve
                if current_position:
                    unrealized_pnl = self.calculate_unrealized_pnl(
                        current_price, entry_price, position_type, quantity
                    )
                    current_equity = self.current_capital + unrealized_pnl
                else:
                    current_equity = self.current_capital
                
                self.equity_curve.append(current_equity)
                self.dates.append(timestamp)
                
                # Update drawdown
                if current_equity > self.peak_equity:
                    self.peak_equity = current_equity
                
                current_drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity
                self.drawdown_curve.append(current_drawdown_pct)
                
                # Check for exit conditions
                if current_position:
                    stop_loss_price = self.calculate_stop_loss(entry_price, position_type, stop_loss_pct)
                    take_profit_price = self.calculate_take_profit(entry_price, position_type, take_profit_pct)
                    
                    exit_trade = False
                    exit_reason = ""
                    
                    # Stop loss hit
                    if (position_type == 'LONG' and current_price <= stop_loss_price) or \
                       (position_type == 'SHORT' and current_price >= stop_loss_price):
                        exit_trade = True
                        exit_reason = "Stop Loss"
                    
                    # Take profit hit
                    elif (position_type == 'LONG' and current_price >= take_profit_price) or \
                         (position_type == 'SHORT' and current_price <= take_profit_price):
                        exit_trade = True
                        exit_reason = "Take Profit"
                    
                    # Signal-based exit
                    elif signal['action'] == 'EXIT':
                        exit_trade = True
                        exit_reason = "Signal Exit"
                    
                    if exit_trade:
                        # Close position
                        pnl = self.calculate_pnl(current_price, entry_price, position_type, quantity)
                        pnl_pct = (pnl / (entry_price * quantity)) * 100
                        holding_period = timestamp - entry_time
                        
                        trade = Trade(
                            entry_time=entry_time,
                            exit_time=timestamp,
                            entry_price=entry_price,
                            exit_price=current_price,
                            position_type=position_type,
                            quantity=quantity,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            holding_period=holding_period,
                            signal_strength=signal.get('strength', 0.5),
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price
                        )
                        
                        self.trades.append(trade)
                        self.current_capital += pnl
                        
                        # Reset position
                        current_position = None
                        entry_price = 0.0
                        entry_time = None
                        position_type = None
                        quantity = 0.0
                
                # Check for entry conditions (no current position)
                if not current_position and signal['action'] in ['BUY', 'SELL']:
                    # Calculate position size
                    trade_capital = self.current_capital * position_size
                    quantity = trade_capital / current_price
                    
                    # Enter position
                    current_position = True
                    entry_price = current_price
                    entry_time = timestamp
                    position_type = 'LONG' if signal['action'] == 'BUY' else 'SHORT'
            
            # Close any open position at the end
            if current_position:
                last_price = data.iloc[-1]['close']
                pnl = self.calculate_pnl(last_price, entry_price, position_type, quantity)
                pnl_pct = (pnl / (entry_price * quantity)) * 100
                holding_period = data.index[-1] - entry_time
                
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=data.index[-1],
                    entry_price=entry_price,
                    exit_price=last_price,
                    position_type=position_type,
                    quantity=quantity,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_period=holding_period,
                    signal_strength=0.5,
                    stop_loss=self.calculate_stop_loss(entry_price, position_type, stop_loss_pct),
                    take_profit=self.calculate_take_profit(entry_price, position_type, take_profit_pct)
                )
                
                self.trades.append(trade)
                self.current_capital += pnl
            
            # Calculate final metrics
            equity_series = pd.Series(self.equity_curve[1:], index=self.dates[1:])
            metrics = self.calculate_advanced_metrics(self.trades, equity_series)
            
            result = BacktestResult(
                total_trades=metrics.get('total_trades', 0),
                winning_trades=metrics.get('winning_trades', 0),
                losing_trades=metrics.get('losing_trades', 0),
                win_rate=metrics.get('win_rate', 0),
                total_pnl=metrics.get('total_pnl', 0),
                total_pnl_pct=metrics.get('total_pnl_pct', 0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                max_drawdown_pct=metrics.get('max_drawdown_pct', 0),
                profit_factor=metrics.get('profit_factor', 0),
                average_trade=metrics.get('average_trade', 0),
                average_win=metrics.get('average_win', 0),
                average_loss=metrics.get('average_loss', 0),
                expectancy=metrics.get('expectancy', 0),
                trades=self.trades,
                equity_curve=equity_series,
                drawdown_curve=pd.Series(self.drawdown_curve[1:], index=self.dates[1:]),
                monthly_returns=self.calculate_monthly_returns(self.trades),
                metrics=metrics
            )
            
            logger.info(f"Backtest completed: {result.total_trades} trades, "
                       f"PNL: ${result.total_pnl:,.2f} ({result.total_pnl_pct:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            traceback.print_exc()
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], pd.Series(), pd.Series(), pd.Series(), {})
    
    def calculate_unrealized_pnl(self, current_price: float, entry_price: float, 
                               position_type: str, quantity: float) -> float:
        """Calculate unrealized P&L for open position"""
        if position_type == 'LONG':
            return (current_price - entry_price) * quantity
        else:  # SHORT
            return (entry_price - current_price) * quantity
    
    def calculate_pnl(self, exit_price: float, entry_price: float, 
                     position_type: str, quantity: float) -> float:
        """Calculate realized P&L including commission"""
        if position_type == 'LONG':
            gross_pnl = (exit_price - entry_price) * quantity
        else:  # SHORT
            gross_pnl = (entry_price - exit_price) * quantity
        
        # Apply commission (both entry and exit)
        commission_cost = (entry_price * quantity * self.commission) + (exit_price * quantity * self.commission)
        return gross_pnl - commission_cost
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, stop_loss_pct: float) -> float:
        """Calculate stop loss price"""
        if position_type == 'LONG':
            return entry_price * (1 - stop_loss_pct)
        else:  # SHORT
            return entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, position_type: str, take_profit_pct: float) -> float:
        """Calculate take profit price"""
        if position_type == 'LONG':
            return entry_price * (1 + take_profit_pct)
        else:  # SHORT
            return entry_price * (1 - take_profit_pct)
    
    def walk_forward_analysis(self, 
                             data: pd.DataFrame,
                             strategy_function: Callable,
                             window_size: int = 252,  # 1 year in trading days
                             forward_size: int = 63,   # 3 months forward
                             **strategy_params) -> Dict[str, any]:
        """
        Perform walk-forward analysis for strategy validation
        
        Args:
            data: Historical price data
            strategy_function: Trading strategy function
            window_size: Optimization window size
            forward_size: Forward test window size
            **strategy_params: Strategy parameters
            
        Returns:
            Dict with walk-forward results
        """
        try:
            results = []
            total_windows = len(data) // forward_size
            
            logger.info(f"Starting walk-forward analysis with {total_windows} windows")
            
            for i in range(0, len(data) - window_size - forward_size, forward_size):
                # Split data into in-sample and out-of-sample
                in_sample_data = data.iloc[i:i + window_size]
                out_of_sample_data = data.iloc[i + window_size:i + window_size + forward_size]
                
                # Run backtest on in-sample data
                in_sample_result = self.run_backtest(in_sample_data, strategy_function, **strategy_params)
                
                # Run backtest on out-of-sample data
                out_of_sample_result = self.run_backtest(out_of_sample_data, strategy_function, **strategy_params)
                
                window_result = {
                    'window_id': i,
                    'in_sample_start': in_sample_data.index[0],
                    'in_sample_end': in_sample_data.index[-1],
                    'out_of_sample_start': out_of_sample_data.index[0],
                    'out_of_sample_end': out_of_sample_data.index[-1],
                    'in_sample_metrics': in_sample_result.metrics,
                    'out_of_sample_metrics': out_of_sample_result.metrics,
                    'performance_decay': out_of_sample_result.metrics.get('total_pnl_pct', 0) - in_sample_result.metrics.get('total_pnl_pct', 0)
                }
                
                results.append(window_result)
                
                logger.info(f"Window {i}: In-sample PnL: {in_sample_result.total_pnl_pct:.2f}%, "
                           f"Out-of-sample PnL: {out_of_sample_result.total_pnl_pct:.2f}%")
            
            # Calculate walk-forward efficiency
            if results:
                total_performance_decay = sum(r['performance_decay'] for r in results)
                avg_performance_decay = total_performance_decay / len(results)
                wfa_efficiency = 100 - abs(avg_performance_decay)
            else:
                wfa_efficiency = 0
            
            walk_forward_results = {
                'windows': results,
                'total_windows': len(results),
                'avg_performance_decay': avg_performance_decay,
                'walk_forward_efficiency': wfa_efficiency,
                'consistent_windows': len([r for r in results if r['performance_decay'] > -5])  # Less than 5% decay
            }
            
            return walk_forward_results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis: {e}")
            traceback.print_exc()
            return {}
    
    def generate_report(self, result: BacktestResult, walk_forward_results: Dict = None) -> str:
        """
        Generate comprehensive backtest report
        
        Args:
            result: BacktestResult object
            walk_forward_results: Results from walk-forward analysis
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("ADVANCED BACKTESTING REPORT")
        report.append("=" * 80)
        
        # Basic performance
        report.append(f"\n📊 PERFORMANCE SUMMARY:")
        report.append(f"Total Trades: {result.total_trades}")
        report.append(f"Win Rate: {result.win_rate:.2f}%")
        report.append(f"Total PnL: ${result.total_pnl:,.2f} ({result.total_pnl_pct:.2f}%)")
        report.append(f"Profit Factor: {result.profit_factor:.2f}")
        report.append(f"Expectancy: ${result.expectancy:.2f}")
        
        # Risk metrics
        report.append(f"\n⚡ RISK METRICS:")
        report.append(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio: {result.metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
        report.append(f"Calmar Ratio: {result.metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"Recovery Factor: {result.metrics.get('recovery_factor', 0):.2f}")
        
        # Trade statistics
        report.append(f"\n🎯 TRADE STATISTICS:")
        report.append(f"Average Win: ${result.average_win:.2f}")
        report.append(f"Average Loss: ${result.average_loss:.2f}")
        report.append(f"Win/Loss Ratio: {result.metrics.get('avg_win_to_avg_loss_ratio', 0):.2f}")
        report.append(f"Avg Holding Period: {result.metrics.get('avg_holding_period_hours', 0):.1f} hours")
        report.append(f"Longest Win Streak: {result.metrics.get('longest_win_streak', 0)}")
        report.append(f"Longest Loss Streak: {result.metrics.get('longest_loss_streak', 0)}")
        
        # Walk-forward analysis
        if walk_forward_results:
            report.append(f"\n🔄 WALK-FORWARD ANALYSIS:")
            report.append(f"Total Windows: {walk_forward_results['total_windows']}")
            report.append(f"Walk-Forward Efficiency: {walk_forward_results['walk_forward_efficiency']:.2f}%")
            report.append(f"Consistent Windows: {walk_forward_results['consistent_windows']}")
            report.append(f"Avg Performance Decay: {walk_forward_results['avg_performance_decay']:.2f}%")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Example strategy function for testing
def sample_moving_average_strategy(data: pd.DataFrame, 
                                 short_window: int = 10,
                                 long_window: int = 30) -> Dict[str, any]:
    """
    Sample moving average crossover strategy for testing
    
    Args:
        data: OHLCV DataFrame
        short_window: Short moving average window
        long_window: Long moving average window
        
    Returns:
        Trading signal dictionary
    """
    if len(data) < long_window:
        return {'action': 'HOLD', 'strength': 0.0}
    
    # Calculate moving averages
    data['SMA_short'] = data['close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['close'].rolling(window=long_window).mean()
    
    current_short = data['SMA_short'].iloc[-1]
    current_long = data['SMA_long'].iloc[-1]
    previous_short = data['SMA_short'].iloc[-2] if len(data) > 1 else current_short
    previous_long = data['SMA_long'].iloc[-2] if len(data) > 1 else current_long
    
    # Generate signals
    if current_short > current_long and previous_short <= previous_long:
        return {'action': 'BUY', 'strength': 0.8}
    elif current_short < current_long and previous_short >= previous_long:
        return {'action': 'SELL', 'strength': 0.8}
    else:
        return {'action': 'HOLD', 'strength': 0.2}


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='1H')
    price = 1.1000 + np.cumsum(np.random.randn(len(dates)) * 0.001)
    
    sample_data = pd.DataFrame({
        'open': price + np.random.randn(len(dates)) * 0.0005,
        'high': price + np.abs(np.random.randn(len(dates)) * 0.001),
        'low': price - np.abs(np.random.randn(len(dates)) * 0.001),
        'close': price,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Initialize backtester
    backtester = AdvancedBacktester(initial_capital=10000.0)
    
    # Run backtest
    print("Running backtest...")
    result = backtester.run_backtest(
        data=sample_data,
        strategy_function=sample_moving_average_strategy,
        position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    # Run walk-forward analysis
    print("\nRunning walk-forward analysis...")
    wfa_results = backtester.walk_forward_analysis(
        data=sample_data,
        strategy_function=sample_moving_average_strategy,
        window_size=1000,  # ~42 days in 1H data
        forward_size=168   # 1 week forward
    )
    
    # Generate report
    report = backtester.generate_report(result, wfa_results)
    print(report)
"""
Main Backtesting Script for AI Forex Trading Bot
Orchestrates complete backtesting pipeline with multiple strategies
"""

import pandas as pd
import numpy as np
import logging
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.advanced_metrics import AdvancedMetricsCalculator
from backtesting.advanced_backtester import AdvancedBacktester, BacktestResult
from strategies.deep_learning_strat import DeepLearningStrategy
from strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from strategies.strategy_selector import StrategySelector
from core.data_handler import DataHandler
from config.backtest_config import BacktestConfig
from visualization.advanced_dashboard import create_backtest_dashboard

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForexBacktester:
    """
    Main backtesting class for Forex trading bot
    Coordinates data loading, strategy testing, and result analysis
    """
    
    def __init__(self, config_path: str = 'config/backtest_config.json'):
        self.config = self.load_config(config_path)
        self.data_handler = DataHandler()
        self.advanced_backtester = AdvancedBacktester(
            initial_capital=self.config['initial_capital'],
            commission=self.config['commission']
        )
        self.results = {}
        self.comparison_results = {}
        
        logger.info("Forex Backtester initialized successfully")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load backtesting configuration from JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            # Return default config
            return {
                'initial_capital': 10000.0,
                'commission': 0.0002,
                'test_periods': ['2020-01-01', '2023-12-31'],
                'symbols': ['EUR/USD'],
                'timeframe': '1H',
                'strategies': ['deep_learning', 'multi_timeframe'],
                'risk_parameters': {
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.04,
                    'position_size': 0.1,
                    'max_drawdown': 0.2
                }
            }
    
    def load_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
        """
        Load historical data for backtesting
        
        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date} ({timeframe})")
            
            # For demo purposes, create synthetic data
            # In production, this would load from database or API
            data = self.generate_sample_data(start_date, end_date, timeframe)
            
            logger.info(f"Data loaded: {len(data)} records, from {data.index[0]} to {data.index[-1]}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            # Generate fallback sample data
            return self.generate_sample_data(start_date, end_date, timeframe)
    
    def generate_sample_data(self, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
        """
        Generate realistic sample Forex data for testing
        
        Args:
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            DataFrame with realistic Forex data
        """
        # Calculate number of periods based on timeframe
        if timeframe == '1H':
            freq = '1H'
        elif timeframe == '4H':
            freq = '4H'
        elif timeframe == '1D':
            freq = '1D'
        else:
            freq = '1H'
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Create realistic EUR/USD price series with trends and volatility
        np.random.seed(42)  # For reproducible results
        
        # Start price (typical EUR/USD range)
        start_price = 1.1000
        
        # Generate price movements with realistic volatility
        returns = np.random.normal(0.0001, 0.005, len(dates))  # 50 pips average movement
        
        # Add some trends and seasonality
        trend = np.linspace(0, 0.05, len(dates))  # Small upward trend
        seasonality = 0.001 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Daily seasonality
        
        # Combine components
        cumulative_returns = np.cumsum(returns + trend + seasonality)
        prices = start_price * (1 + cumulative_returns)
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        
        # Generate realistic OHLC from close prices
        data['open'] = data['close'].shift(1).fillna(start_price)
        volatility = np.abs(returns) * 2  # High-low range
        
        data['high'] = data[['open', 'close']].max(axis=1) + volatility * 0.6
        data['low'] = data[['open', 'close']].min(axis=1) - volatility * 0.6
        
        # Ensure high > low
        data['high'] = np.maximum(data['high'].values, data[['open', 'close']].max(axis=1).values + 0.0001)
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1) - 0.0001)
        
        # Volume with some pattern
        base_volume = 10000
        data['volume'] = base_volume + np.random.randint(-2000, 2000, len(dates))
        
        # Clean any NaN values
        data = data.ffill().bfill()
        
        logger.info(f"Generated sample data: {len(data)} records")
        return data
    
    def initialize_strategy(self, strategy_name: str, data: pd.DataFrame) -> Any:
        """
        Initialize trading strategy
        
        Args:
            strategy_name: Name of strategy to initialize
            data: Historical data for strategy initialization
            
        Returns:
            Initialized strategy object
        """
        try:
            if strategy_name == 'deep_learning':
                strategy = DeepLearningStrategy()
                strategy.initialize(data)
                logger.info("Deep Learning strategy initialized")
                
            elif strategy_name == 'multi_timeframe':
                strategy = MultiTimeframeAnalyzer()
                strategy.initialize(data)
                logger.info("Multi-Timeframe strategy initialized")
                
            elif strategy_name == 'strategy_selector':
                strategy = StrategySelector()
                strategy.initialize(data)
                logger.info("Strategy Selector initialized")
                
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error initializing strategy {strategy_name}: {e}")
            # Return a simple fallback strategy
            return self.create_fallback_strategy()
    
    def create_fallback_strategy(self):
        """Create a simple fallback strategy"""
        class FallbackStrategy:
            def generate_signal(self, data):
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Fallback strategy'}
            
            def initialize(self, data):
                pass
        
        return FallbackStrategy()
    
    def run_single_backtest(self, strategy_name: str, data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest for a single strategy
        
        Args:
            strategy_name: Name of strategy to test
            data: Historical data for backtesting
            
        Returns:
            BacktestResult object
        """
        try:
            logger.info(f"Starting backtest for strategy: {strategy_name}")
            
            # Initialize strategy
            strategy = self.initialize_strategy(strategy_name, data)
            
            # Define strategy function for backtester
            def strategy_function(current_data, **params):
                return strategy.generate_signal(current_data)
            
            # Run backtest
            result = self.advanced_backtester.run_backtest(
                data=data,
                strategy_function=strategy_function,
                position_size=self.config['risk_parameters']['position_size'],
                stop_loss_pct=self.config['risk_parameters']['stop_loss_pct'],
                take_profit_pct=self.config['risk_parameters']['take_profit_pct']
            )
            
            # Calculate additional metrics
            metrics_calculator = AdvancedMetricsCalculator()
            additional_metrics = metrics_calculator.calculate_comprehensive_metrics(
                returns=result.returns,
                trades=result.trades,
                initial_balance=self.config['initial_capital']
         )
            result.metrics.update(additional_metrics)
            
            logger.info(f"Backtest completed for {strategy_name}: "
                       f"{result.total_trades} trades, PnL: {result.total_pnl_pct:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in backtest for {strategy_name}: {e}")
            # Return empty result
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], pd.Series(), pd.Series(), pd.Series(), {}, {})
    
    def run_comparative_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtests for all strategies and compare results
        
        Args:
            data: Historical data for backtesting
            
        Returns:
            Dictionary with comparison results
        """
        try:
            logger.info("Starting comparative backtesting for all strategies")
            
            comparison_results = {
                'strategies': {},
                'best_strategy': None,
                'summary': {}
            }
            
            # Run backtest for each strategy
            for strategy_name in self.config['strategies']:
                logger.info(f"Testing strategy: {strategy_name}")
                
                result = self.run_single_backtest(strategy_name, data)
                comparison_results['strategies'][strategy_name] = {
                    'result': result,
                    'metrics': result.metrics
                }
            
            # Find best performing strategy
            best_strategy = None
            best_pnl = -float('inf')
            
            for strategy_name, strategy_data in comparison_results['strategies'].items():
                pnl = strategy_data['metrics'].get('total_pnl_pct', 0)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_strategy = strategy_name
            
            comparison_results['best_strategy'] = best_strategy
            
            # Create summary statistics
            summary = {
                'total_strategies_tested': len(self.config['strategies']),
                'best_strategy': best_strategy,
                'best_pnl_pct': best_pnl,
                'average_pnl_pct': np.mean([s['metrics'].get('total_pnl_pct', 0) 
                                          for s in comparison_results['strategies'].values()]),
                'strategies_positive': len([s for s in comparison_results['strategies'].values() 
                                          if s['metrics'].get('total_pnl_pct', 0) > 0])
            }
            
            comparison_results['summary'] = summary
            
            logger.info(f"Comparative backtesting completed. Best strategy: {best_strategy} "
                       f"with {best_pnl:.2f}% PnL")
            
            self.comparison_results = comparison_results
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in comparative backtesting: {e}")
            return {}
    
    def run_walk_forward_analysis(self, data: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
        """
        Run walk-forward analysis for strategy validation
        
        Args:
            data: Historical data
            strategy_name: Strategy to validate
            
        Returns:
            Walk-forward analysis results
        """
        try:
            logger.info(f"Starting walk-forward analysis for {strategy_name}")
            
            # Initialize strategy
            strategy = self.initialize_strategy(strategy_name, data)
            
            def strategy_function(current_data, **params):
                return strategy.generate_signal(current_data)
            
            # Run walk-forward analysis
            wfa_results = self.advanced_backtester.walk_forward_analysis(
                data=data,
                strategy_function=strategy_function,
                window_size=self.config.get('walk_forward_window', 1000),
                forward_size=self.config.get('walk_forward_forward', 168)
            )
            
            logger.info(f"Walk-forward analysis completed for {strategy_name}. "
                       f"Efficiency: {wfa_results.get('walk_forward_efficiency', 0):.2f}%")
            
            return wfa_results
            
        except Exception as e:
            logger.error(f"Error in walk-forward analysis for {strategy_name}: {e}")
            return {}
    
    def generate_comprehensive_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive backtesting report
        
        Args:
            comparison_results: Results from comparative backtesting
            
        Returns:
            Formatted report string
        """
        try:
            report = []
            report.append("=" * 100)
            report.append("FOREX TRADING BOT - COMPREHENSIVE BACKTESTING REPORT")
            report.append("=" * 100)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Test Period: {self.config['test_periods'][0]} to {self.config['test_periods'][1]}")
            report.append(f"Symbol: {self.config['symbols'][0]}")
            report.append(f"Timeframe: {self.config['timeframe']}")
            report.append("")
            
            # Summary section
            summary = comparison_results.get('summary', {})
            report.append("📊 EXECUTIVE SUMMARY")
            report.append("-" * 50)
            report.append(f"Strategies Tested: {summary.get('total_strategies_tested', 0)}")
            report.append(f"Best Strategy: {summary.get('best_strategy', 'N/A')}")
            report.append(f"Best PnL: {summary.get('best_pnl_pct', 0):.2f}%")
            report.append(f"Average PnL: {summary.get('average_pnl_pct', 0):.2f}%")
            report.append(f"Profitable Strategies: {summary.get('strategies_positive', 0)}")
            report.append("")
            
            # Detailed strategy performance
            report.append("🎯 STRATEGY PERFORMANCE DETAILS")
            report.append("-" * 50)
            
            for strategy_name, strategy_data in comparison_results.get('strategies', {}).items():
                metrics = strategy_data.get('metrics', {})
                result = strategy_data.get('result')
                
                report.append(f"\nStrategy: {strategy_name.upper()}")
                report.append(f"  Trades: {metrics.get('total_trades', 0)}")
                report.append(f"  Win Rate: {metrics.get('win_rate', 0):.2f}%")
                report.append(f"  Total PnL: ${metrics.get('total_pnl', 0):,.2f} ({metrics.get('total_pnl_pct', 0):.2f}%)")
                report.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                report.append(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                report.append(f"  Expectancy: ${metrics.get('expectancy', 0):.2f}")
            
            # Risk analysis
            report.append("\n⚡ RISK ANALYSIS")
            report.append("-" * 50)
            
            best_strategy = comparison_results.get('best_strategy')
            if best_strategy:
                best_metrics = comparison_results['strategies'][best_strategy]['metrics']
                report.append(f"Best Strategy Risk Metrics ({best_strategy}):")
                report.append(f"  Sortino Ratio: {best_metrics.get('sortino_ratio', 0):.2f}")
                report.append(f"  Calmar Ratio: {best_metrics.get('calmar_ratio', 0):.2f}")
                report.append(f"  Recovery Factor: {best_metrics.get('recovery_factor', 0):.2f}")
                report.append(f"  Avg Holding Period: {best_metrics.get('avg_holding_period_hours', 0):.1f} hours")
                report.append(f"  Longest Drawdown: {best_metrics.get('longest_loss_streak', 0)} trades")
            
            # Recommendations
            report.append("\n💡 RECOMMENDATIONS")
            report.append("-" * 50)
            
            if summary.get('best_pnl_pct', 0) > 0:
                report.append(f"✅ RECOMMENDED: {best_strategy} strategy")
                report.append(f"   Expected annual return: {summary['best_pnl_pct']:.2f}%")
                
                best_metrics = comparison_results['strategies'][best_strategy]['metrics']
                if best_metrics.get('max_drawdown_pct', 0) > 15:
                    report.append("   ⚠️  WARNING: High maximum drawdown detected")
                if best_metrics.get('profit_factor', 0) < 1.5:
                    report.append("   ⚠️  WARNING: Low profit factor")
                    
            else:
                report.append("❌ NO STRATEGY: All strategies showed negative performance")
                report.append("   Consider parameter optimization or strategy development")
            
            report.append("\n" + "=" * 100)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"
    
    def save_results(self, comparison_results: Dict[str, Any], filename: str = None):
        """
        Save backtesting results to file
        
        Args:
            comparison_results: Results to save
            filename: Output filename
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_results_{timestamp}.json"
            
            # Convert results to serializable format
            serializable_results = {}
            for strategy_name, strategy_data in comparison_results.get('strategies', {}).items():
                serializable_results[strategy_name] = {
                    'metrics': strategy_data.get('metrics', {}),
                    'trades_count': len(strategy_data.get('result', BacktestResult(0,0,0,0,0,0,0,0,0,0,0,0,0,[],pd.Series(),pd.Series(),pd.Series(),{})).trades)
                }
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'results': serializable_results,
                'summary': comparison_results.get('summary', {})
            }
            
            with open(f"results/{filename}", 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Results saved to results/{filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def plot_comparison_charts(self, comparison_results: Dict[str, Any]):
        """
        Create comparison charts for strategies
        
        Args:
            comparison_results: Results from comparative backtesting
        """
        try:
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Prepare data for plotting
            strategies = list(comparison_results.get('strategies', {}).keys())
            pnl_values = [comparison_results['strategies'][s]['metrics'].get('total_pnl_pct', 0) 
                         for s in strategies]
            win_rates = [comparison_results['strategies'][s]['metrics'].get('win_rate', 0) 
                        for s in strategies]
            sharpe_ratios = [comparison_results['strategies'][s]['metrics'].get('sharpe_ratio', 0) 
                           for s in strategies]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # PnL Comparison
            bars1 = ax1.bar(strategies, pnl_values, color=['green' if x > 0 else 'red' for x in pnl_values])
            ax1.set_title('Strategy PnL Comparison (%)')
            ax1.set_ylabel('Total PnL (%)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom')
            
            # Win Rate Comparison
            bars2 = ax2.bar(strategies, win_rates, color='skyblue')
            ax2.set_title('Win Rate Comparison (%)')
            ax2.set_ylabel('Win Rate (%)')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Sharpe Ratio Comparison
            bars3 = ax3.bar(strategies, sharpe_ratios, color='orange')
            ax3.set_title('Sharpe Ratio Comparison')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Drawdown Comparison
            drawdowns = [comparison_results['strategies'][s]['metrics'].get('max_drawdown_pct', 0) 
                        for s in strategies]
            bars4 = ax4.bar(strategies, drawdowns, color=['red' if x > 10 else 'yellow' for x in drawdowns])
            ax4.set_title('Maximum Drawdown Comparison (%)')
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'results/strategy_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Comparison charts saved to results directory")
            
        except Exception as e:
            logger.error(f"Error creating comparison charts: {e}")

def main():
    """Main function to run complete backtesting pipeline"""
    try:
        logger.info("Starting Forex Trading Bot Backtesting Pipeline")
        
        # Initialize backtester
        backtester = ForexBacktester()
        
        # Load historical data
        data = backtester.load_historical_data(
            symbol=backtester.config['symbols'][0],
            start_date=backtester.config['test_periods'][0],
            end_date=backtester.config['test_periods'][1],
            timeframe=backtester.config['timeframe']
        )
        
        # Run comparative backtesting
        comparison_results = backtester.run_comparative_backtest(data)
        
        # Generate report
        report = backtester.generate_comprehensive_report(comparison_results)
        print(report)
        
        # Save results
        backtester.save_results(comparison_results)
        
        # Create visualizations
        backtester.plot_comparison_charts(comparison_results)
        
        # Run walk-forward analysis for best strategy
        best_strategy = comparison_results.get('best_strategy')
        if best_strategy:
            wfa_results = backtester.run_walk_forward_analysis(data, best_strategy)
            logger.info(f"Walk-forward efficiency for {best_strategy}: "
                       f"{wfa_results.get('walk_forward_efficiency', 0):.2f}%")
        
        logger.info("Backtesting pipeline completed successfully")
        
        # Create interactive dashboard
        try:
            create_backtest_dashboard(comparison_results)
            logger.info("Interactive dashboard created")
        except Exception as e:
            logger.warning(f"Could not create dashboard: {e}")
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Error in main backtesting pipeline: {e}")
        return {}

if __name__ == "__main__":
    # Run backtesting
    results = main()
    
    # Print summary
    if results:
        best_strategy = results.get('best_strategy')
        best_pnl = results.get('summary', {}).get('best_pnl_pct', 0)
        print(f"\n🎉 BACKTESTING COMPLETED!")
        print(f"Best Strategy: {best_strategy}")
        print(f"Best PnL: {best_pnl:.2f}%")
        
        if best_pnl > 0:
            print("✅ Strategy shows profitable potential!")
        else:
            print("❌ Strategy needs optimization")
    else:
        print("❌ Backtesting failed")
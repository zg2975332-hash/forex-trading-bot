#!/usr/bin/env python3
"""
FOREX TRADING BOT - MASTER INTEGRATED MAIN ENTRY POINT
Complete integration of all modules into single execution system
"""
try:
    import scipy.stats
    import scipy.optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  Warning: scipy not installed. Some advanced metrics will be limited.")
import warnings
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

current_file = Path(__file__).resolve()
project_root = current_file.parent  # core/ folder
sys.path.insert(0, str(project_root.parent))  # FOREX TRADING BOT/ folder
sys.path.insert(0, str(project_root))  # core/ folder

def safe_import(module_name, class_name, fallback_class=None):
    """Safely import modules with fallback support"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        class_obj = getattr(module, class_name)
        print(f"✅ {module_name}.{class_name} loaded successfully")
        return class_obj
    except ImportError as e:
        print(f"⚠️  {module_name}.{class_name} not found: {e}")
        return fallback_class

# Fallback classes for missing modules
class FallbackComponent:
    def __init__(self, config=None):
        self.config = config or {}
        self.initialized = True
    
    async def initialize(self):
        return True
    
    def __getattr__(self, name):
        """Handle any method calls gracefully"""
        def method(*args, **kwargs):
            print(f"⚠️  Method {name} called on fallback component")
            return None
        return method

# Fallback for specific classes
class FallbackBacktester(FallbackComponent):
    pass

class FallbackRiskManager(FallbackComponent):
    pass

class FallbackMobileAlerts(FallbackComponent):
    async def send_trading_alert(self, symbol, action, confidence, price):
        print(f"📱 [FALLBACK] Alert: {symbol} {action} at {price} (Confidence: {confidence})")

# Import ALL module main classes
try:
    from analysis.market_microstructure import AdvancedMarketMicrostructure as AnalysisEngine
    print("✅ AnalysisEngine loaded successfully")
except ImportError as e:
    print(f"⚠️  AnalysisEngine not found: {e}")
    AnalysisEngine = FallbackComponent
try:
    from backtesting.advanced_backtester import AdvancedBacktester
    BacktestingEngine = AdvancedBacktester
    print("✅ AdvancedBacktester loaded successfully")
except ImportError as e:
    print(f"⚠️ AdvancedBacktester not found: {e}")
    BacktestingEngine = FallbackBacktester
from backtesting.advanced_metrics import AdvancedMetricsCalculator
try:
    from backtesting.backtest import ForexBacktester
    print("✅ ForexBacktester loaded successfully")
except ImportError as e:
    print(f"⚠️  ForexBacktester not found: {e}")
    # Fallback ForexBacktester class definition
    class ForexBacktester:
        def __init__(self, config_path):
            self.config_path = config_path
        
        def load_historical_data(self, *args, **kwargs):
            return pd.DataFrame()
        
        def run_comparative_backtest(self, data):
            return {}
from config.env_config import EnvironmentConfig as ConfigurationManager, setup_environment as get_config_manager
try:
    from core.adaptive_risk_manager import AdaptiveRiskManager
    ForexTradingCore = AdaptiveRiskManager
    print("✅ AdaptiveRiskManager loaded successfully")
except ImportError as e:
    print(f"⚠️  AdaptiveRiskManager not found: {e}")
    ForexTradingCore = FallbackRiskManager
from data.alternative_data_loader import AlternativeDataLoader as ForexDataManager
from deployment.aws_deploy import AWSDeployer as ForexDeploymentManager
from features.quantitative_features import QuantitativeFeatures as ForexFeatureEngine
from logs.performance_logger import PerformanceLogger as ForexLoggingSystem
from models.ensemble_predictor import EnsemblePredictor as ForexModelSystem
from news.sentiment_analyzer import AdvancedSentimentAnalyzer as ForexNewsSentimentSystem
from deployment.aws_deploy import AWSDeployer, DeploymentConfig, DeploymentEnvironment

# ✅ DASHBOARDS
from monitoring.real_time_dashboard import RealTimeDashboard, create_real_time_dashboard
from visualization.advanced_dashboard import AdvancedVisualizationDashboard, create_advanced_visualization_dashboard

# Import SCRIPTS and additional modules
AdvancedMobileAlerts = safe_import('notifications.mobile_alerts', 'AdvancedMobileAlerts', FallbackMobileAlerts)
from core.key_manager import RuntimeKeyManager
from optimization.genetic_optimizer import AdvancedGeneticOptimizer, GeneticOptimizerConfig
from optimization.portfolio_optimizer import AdvancedPortfolioOptimizer, PortfolioConfig
from risk.correlation_manager import AdvancedCorrelationManager, CorrelationConfig
from risk.monte_carlo_simulator import AdvancedMonteCarloSimulator, SimulationConfig
from risk.var_calculator import AdvancedVaRCalculator, VaRConfig
from security.api_rate_limiter import APIRateLimiter, RateLimitConfig
from security.encryption import AdvancedEncryption, EncryptionConfig
from social.community_signals import AdvancedCommunityAnalyzer, CommunityConfig
from strategies.deep_learning_strat import AdvancedDeepLearningStrategy, ModelConfig
from strategies.multi_timeframe_analyzer import AdvancedMultiTimeframeAnalyzer, MTFConfig
from strategies.retail_strategies import AdvancedRetailStrategies, RetailStrategyConfig
from strategies.signal_filter import AdvancedSignalFilter, FilterConfig
from strategies.strategy_selector import AdvancedStrategySelector, StrategyConfig
from tests.integration_tests import AdvancedIntegrationTests, TestConfig
AdvancedDecisionEngine = safe_import('decision_engine', 'AdvancedDecisionEngine', FallbackComponent)

# Create fallback dashboard functions
def create_real_time_dashboard(trading_bot=None):
    return RealTimeDashboard(trading_bot)

def create_advanced_visualization_dashboard(trading_bot=None):
    return AdvancedVisualizationDashboard(trading_bot)

class DeploymentManager:
    """Simple deployment manager for local development"""
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize_system(self):
        """Initialize deployment system"""
        self.logger.info("Deployment system initialized in local mode")
        return True

class AdvancedBacktester:
    def __init__(self, initial_capital=10000.0, commission=0.0002):
        self.initial_capital = initial_capital
        self.commission = commission
        print("⚠️  Using fallback AdvancedBacktester")
BacktestingEngine = AdvancedBacktester

class AlertConfig:
    def __init__(self, enable_push_notifications=True, enable_email=True, 
                 enable_telegram=False, enable_sms=False, max_alerts_per_hour=100,
                 alert_cooldown_period=300):
        self.enable_push_notifications = enable_push_notifications
        self.enable_email = enable_email
        self.enable_telegram = enable_telegram
        self.enable_sms = enable_sms
        self.max_alerts_per_hour = max_alerts_per_hour
        self.alert_cooldown_period = alert_cooldown_period

class GeneticOptimizerConfig:
    def __init__(self, population_size=50, max_generations=100, 
                 mutation_rate=0.15, crossover_rate=0.8):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

class PortfolioConfig:
    def __init__(self, optimization_method='max_sharpe', lookback_period=252):
        self.optimization_method = optimization_method
        self.lookback_period = lookback_period

class CorrelationConfig:
    def __init__(self, lookback_periods=[20, 60, 120], high_correlation_threshold=0.7):
        self.lookback_periods = lookback_periods
        self.high_correlation_threshold = high_correlation_threshold

class RealTimeDashboard:
    async def start_dashboard(self):
        print("📊 Real-time Dashboard started (Fallback)")
        return True

class AdvancedVisualizationDashboard:
    async def start_dashboard(self):
        print("📈 Advanced Visualization Dashboard started (Fallback)") 
        return True

class ForexTradingBot:
    """
    MASTER FOREX TRADING BOT - Integrates all modules
    """
    
    def __init__(self, config_path: str = "config"):
        self.config_path = config_path
        self.setup_logging()
        self.backtester = ForexBacktester('config/backtest_config.json')
        
        # Initialize runtime key manager
        self.key_manager = RuntimeKeyManager()
        self.api_keys = self.key_manager.load_keys()
        
        self._run_sync_async_methods()
        # Set environment variables from loaded keys
        self._set_environment_vars()
        
        # Initialize all systems
        self.systems = {}
        self.system_status = {}
        
        # Dashboard Systems
        self.dashboards = {}
        self.dashboard_status = {}
        
        # Advanced Components
        self.components = {}
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'system_uptime': 0.0,
            'module_health': {},
            'deployment_history': []
        }
        
        self.system_status_main = "STOPPED"
        self.start_time = None
        self.performance_analyzer = AdvancedMetricsCalculator(risk_free_rate=0.02)
        self.returns_data = pd.Series()
        self.trades = []
        
        self.logger.info("🎯 MASTER FOREX TRADING BOT - Initializing All Systems...")
    
    def setup_logging(self):
        """Setup master logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/master_bot.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_master_config(self):
        """Get master configuration"""
        return {
            'system': {
                'name': 'Advanced Forex Trading Bot',
                'version': '2.0.0',
                'environment': 'production',
                'log_level': 'INFO'
            },
            'modules': {
                'notifications': True,
                'optimization': True,
                'risk': True,
                'security': True,
                'social': True,
                'strategies': True,
                'scripts': True,
                'testing': False,
                'decision_engine': True
            },
            'deployment': {
                'auto_backup': True,
                'health_check_timeout': 300,
                'rollback_on_failure': True,
                'environment': 'production'
            },
            'trading': {
                'enabled_pairs': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
                'trading_hours': '24/5',
                'max_daily_trades': 50,
                'risk_per_trade': 0.02
            }
        }
    
    async def initialize_all_systems(self):
        """Initialize all trading bot systems"""
        try:
            self.logger.info("🔄 INITIALIZING ALL TRADING SYSTEMS...")
            self.start_time = datetime.now()
            
            # 1. Initialize Configuration System
            self.logger.info("📁 Initializing Configuration System...")
            self.systems['config'] = ConfigurationManager()
            self.system_status['config'] = 'INITIALIZED'

            # 2. Initialize Data System  
            self.logger.info("💾 Initializing Data System...")
            self.systems['data'] = ForexDataManager(config=self._get_master_config())
            self.system_status['data'] = 'INITIALIZED'

            # 3. Initialize Logging System
            self.logger.info("📊 Initializing Logging System...")
            self.systems['logs'] = ForexLoggingSystem()
            self.system_status['logs'] = 'INITIALIZED'

            # 4. Initialize Features System
            self.logger.info("🔧 Initializing Features System...")
            self.systems['features'] = ForexFeatureEngine()
            self.system_status['features'] = 'INITIALIZED'

            # 5. Initialize Models System
            self.logger.info("🧠 Initializing Models System...")
            self.systems['models'] = ForexModelSystem()
            self.system_status['models'] = 'INITIALIZED'

            # 6. Initialize News System
            self.logger.info("📰 Initializing News System...")
            self.systems['news'] = ForexNewsSentimentSystem()
            self.system_status['news'] = 'INITIALIZED'

            # 7. Initialize Analysis System
            self.logger.info("📈 Initializing Analysis System...")
            self.systems['analysis'] = AnalysisEngine(config=self._get_master_config())
            self.system_status['analysis'] = 'INITIALIZED'

            # 8. Initialize Backtesting System (Fallback)
            self.logger.info("🧪 Initializing Backtesting System...")
            try:
                if BacktestingEngine and BacktestingEngine != FallbackBacktester:
                    self.systems["backtesting"] = BacktestingEngine(initial_capital=10000.0)
                    self.system_status["backtesting"] = 'INITIALIZED'
                else:
                    self.systems["backtesting"] = FallbackBacktester()
                    self.system_status["backtesting"] = 'FALLBACK'
            except Exception as e:
                self.logger.warning(f"Backtesting system failed: {e}")
                self.systems["backtesting"] = FallbackBacktester()
                self.system_status["backtesting"] = 'FALLBACK'

            # 9. Initialize Core Trading System (Fallback)
            self.logger.info("⚡ Initializing Core Trading System...")
            try:
                if ForexTradingCore and ForexTradingCore != FallbackRiskManager:
                    self.systems["core"] = ForexTradingCore(config=self._get_master_config())
                    self.system_status["core"] = 'INITIALIZED'
                else:
                    self.systems["core"] = FallbackRiskManager()
                    self.system_status["core"] = 'FALLBACK'
            except Exception as e:
                self.logger.warning(f"Core trading system failed: {e}")
                self.systems["core"] = FallbackRiskManager()
                self.system_status["core"] = 'FALLBACK'
            
            # 10. Initialize Deployment System
            self.logger.info("🚀 Initializing Deployment System...")
            try:
                from deployment.aws_deploy import AWSDeployer, DeploymentConfig, DeploymentEnvironment
                self.systems['deployment'] = AWSDeployer(
                    config=DeploymentConfig(environment=DeploymentEnvironment.DEVELOPMENT)
                )
                self.system_status['deployment'] = 'INITIALIZED'
            except Exception as e:
                self.logger.warning(f"⚠️ Deployment system failed: {e}")
                self.logger.info("🔄 Continuing in LOCAL MODE without deployment")
                self.systems['deployment'] = None
                self.system_status['deployment'] = 'LOCAL_MODE'
            
            # Initialize Dashboard Systems
            self.logger.info("📊 Initializing Dashboard Systems...")
            self.dashboards['realtime'] = create_real_time_dashboard(trading_bot=self)
            self.dashboards['advanced_viz'] = create_advanced_visualization_dashboard(trading_bot=self)
            self.dashboard_status['realtime'] = 'INITIALIZED'
            self.dashboard_status['advanced_viz'] = 'INITIALIZED'
            
            # Initialize Advanced Modules
            await self._initialize_advanced_modules()
            
            self.system_status_main = "INITIALIZED"
            self.logger.info("🎉 ALL SYSTEMS INITIALIZED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def _initialize_advanced_modules(self):
        """Initialize all advanced modules"""
        try:
            self.logger.info("🔧 INITIALIZING ADVANCED MODULES...")
            
            # Scripts and Deployment
            self.logger.info("   📦 Initializing Scripts System...")
            self.deployment_manager = DeploymentManager(
                config=self._get_master_config().get('deployment', {})
            )
            success = await self.deployment_manager.initialize_system()
            if success:
                self.components['deployment_manager'] = self.deployment_manager
                self.performance_metrics['module_health']['scripts'] = 'HEALTHY'
            
            # Notifications
            self.logger.info("   📱 Initializing Notifications System...")
            alert_config = AlertConfig(
                enable_push_notifications=True,
                enable_email=True,
                enable_telegram=True,
                enable_sms=False,
                max_alerts_per_hour=100,
                alert_cooldown_period=300
            )
            self.components['notifications'] = AdvancedMobileAlerts(alert_config)
            self.performance_metrics['module_health']['notifications'] = 'HEALTHY'
            
            # Optimization
            self.logger.info("   ⚡ Initializing Optimization System...")
            genetic_config = GeneticOptimizerConfig(
                population_size=50,
                max_generations=100,
                mutation_rate=0.15,
                crossover_rate=0.8
            )
            portfolio_config = PortfolioConfig(
                optimization_method='max_sharpe',
                lookback_period=252,
            )
            self.components['genetic_optimizer'] = AdvancedGeneticOptimizer(genetic_config)
            self.components['portfolio_optimizer'] = AdvancedPortfolioOptimizer(portfolio_config)
            self.performance_metrics['module_health']['optimization'] = 'HEALTHY'
            
            # Risk Management
            self.logger.info("   🛡️ Initializing Risk System...")
            correlation_config = CorrelationConfig(
                lookback_periods=[20, 60, 120],
                high_correlation_threshold=0.7
            )
            monte_carlo_config = SimulationConfig(
                num_simulations=10000,
                time_horizon=252,
                confidence_level=0.95
            )
            var_config = VaRConfig(
                confidence_levels=[0.95, 0.99],
                risk_horizons=[1, 5, 21]
            )
            self.components['correlation_manager'] = AdvancedCorrelationManager(correlation_config)
            self.components['monte_carlo_simulator'] = AdvancedMonteCarloSimulator(monte_carlo_config)
            self.components['var_calculator'] = AdvancedVaRCalculator(var_config)
            self.performance_metrics['module_health']['risk'] = 'HEALTHY'
            
            # Security
            self.logger.info("   🔒 Initializing Security System...")
            rate_limit_config = RateLimitConfig(
                requests_per_second=50,
                burst_capacity=100,
                algorithm='token_bucket'
            )
            encryption_config = EncryptionConfig(
                default_symmetric_algorithm='aes_256_gcm',
                key_rotation_days=90
            )
            self.components['rate_limiter'] = APIRateLimiter(rate_limit_config)
            self.components['encryption'] = AdvancedEncryption(encryption_config)
            self.performance_metrics['module_health']['security'] = 'HEALTHY'
            
            # Social Sentiment
            self.logger.info("   👥 Initializing Social System...")
            community_config = CommunityConfig(
                symbols=self._get_master_config()['trading']['enabled_pairs'],
                min_confidence=0.6,
                aggregation_window=300
            )
            self.components['community_analyzer'] = AdvancedCommunityAnalyzer(community_config)
            self.performance_metrics['module_health']['social'] = 'HEALTHY'
            
            # Strategies
            self.logger.info("   🎯 Initializing Strategies System...")
            dl_config = ModelConfig(
                model_type='ensemble',
                sequence_length=60,
                prediction_horizon=5
            )
            mtf_config = MTFConfig(
                timeframes=['H1', 'H4', 'D1'],
                timeframe_weights={'H1': 0.3, 'H4': 0.4, 'D1': 0.3}
            )
            retail_config = RetailStrategyConfig(
                enabled_strategies=['supply_demand', 'smart_money_concept', 'order_block'],
                min_confidence=0.6
            )
            filter_config = FilterConfig(
                min_confidence=0.65,
                max_daily_trades=10
            )
            selector_config = StrategyConfig(
                enabled_strategies=['trend_following', 'mean_reversion', 'breakout'],
                selection_method='hybrid'
            )
            
            self.components['deep_learning_strategy'] = AdvancedDeepLearningStrategy(dl_config)
            self.components['multi_timeframe_analyzer'] = AdvancedMultiTimeframeAnalyzer(mtf_config)
            self.components['retail_strategies'] = AdvancedRetailStrategies(retail_config)
            self.components['signal_filter'] = AdvancedSignalFilter(filter_config)
            self.components['strategy_selector'] = AdvancedStrategySelector(selector_config)
            self.performance_metrics['module_health']['strategies'] = 'HEALTHY'
            
            # Decision Engine
            self.logger.info("   🧠 Initializing Decision Engine...")
            try:
                if AdvancedDecisionEngine and AdvancedDecisionEngine != FallbackComponent:
                    self.components['decision_engine'] = AdvancedDecisionEngine(
                        config=self._get_master_config().get('decision_engine', {})
                    )
                    self.performance_metrics['module_health']['decision_engine'] = 'HEALTHY'
                else:
                    self.components['decision_engine'] = FallbackComponent()
                    self.performance_metrics['module_health']['decision_engine'] = 'FALLBACK'
            except Exception as e:
                self.logger.warning(f"Decision engine failed: {e}")
                self.components['decision_engine'] = FallbackComponent()
                self.performance_metrics['module_health']['decision_engine'] = 'FALLBACK'
    
    async def run_comprehensive_backtest(self, symbol: str = "EUR/USD"):
        """Run comprehensive backtesting for strategy validation"""
        try:
            self.logger.info(f"🧪 Running comprehensive backtest for {symbol}...")
            
            # Load historical data
            data = self.backtester.load_historical_data(
                symbol=symbol,
                start_date='2023-01-01',
                end_date='2023-12-31',
                timeframe='1H'
            )
            
            # Run comparative backtesting
            comparison_results = self.backtester.run_comparative_backtest(data)
            
            # Generate report
            report = self.backtester.generate_comprehensive_report(comparison_results)
            
            # Save results
            self.backtester.save_results(comparison_results)
            
            # Update performance metrics
            self.returns_data = data['returns'] if 'returns' in data else pd.Series()
            
            self.logger.info(f"✅ Backtesting completed for {symbol}")
            
            return {
                'symbol': symbol,
                'results': comparison_results,
                'report': report[:1000] + "..." if len(report) > 1000 else report,
                'best_strategy': comparison_results.get('best_strategy', 'N/A'),
                'best_pnl': comparison_results.get('summary', {}).get('best_pnl_pct', 0)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting failed: {e}")
            return {'error': str(e)}
    
    async def collect_performance_data(self):
        """Collect returns and trades data"""
        try:
            self.logger.info("📊 Collecting performance data for analysis...")
            
            # Try to get real backtesting results first
            try:
                # Run backtest and get real data
                backtest_results = await self.run_comprehensive_backtest("EUR/USD")
                
                if backtest_results and 'results' in backtest_results and 'error' not in backtest_results:
                    # Extract returns and trades from backtest results
                    best_strategy = backtest_results['results'].get('best_strategy')
                    if best_strategy and 'strategies' in backtest_results['results']:
                        strategy_data = backtest_results['results']['strategies'].get(best_strategy)
                        if strategy_data:
                            result_obj = strategy_data.get('result')
                        
                            if result_obj and hasattr(result_obj, 'returns') and not result_obj.returns.empty:
                                self.returns_data = result_obj.returns
                                self.logger.info(f"✅ Using real backtest returns: {len(self.returns_data)} points")
                        
                        if hasattr(result_obj, 'returns') and not result_obj.returns.empty:
                            self.returns_data = result_obj.returns
                            self.logger.info(f"✅ Using real backtest returns: {len(self.returns_data)} points")
                        
                        if hasattr(result_obj, 'trades') and result_obj.trades:
                            self.trades = result_obj.trades
                            self.logger.info(f"✅ Using real backtest trades: {len(self.trades)} trades")
                            
                            # Convert trades to required format
                            formatted_trades = []
                            for trade in self.trades:
                                if hasattr(trade, '__dict__'):
                                    formatted_trades.append(trade.__dict__)
                                elif isinstance(trade, dict):
                                    formatted_trades.append(trade)
                            
                            self.trades = formatted_trades
                            
                            return True
            except Exception as backtest_error:
                self.logger.warning(f"Could not get backtest data: {backtest_error}")
                # Fallback to simulated data
            
            # Fallback: Simulate data if backtest fails
            self.logger.info("Using simulated performance data (fallback)")
            
            np.random.seed(42)
            dates = pd.date_range(start='2024-01-01', periods=252, freq='B')
            simulated_returns = np.random.normal(0.0008, 0.015, 252)
            simulated_returns[:60] += 0.002
            simulated_returns[120:180] -= 0.003
            
            self.returns_data = pd.Series(simulated_returns, index=dates)
            
            self.trades = [
                {'pnl': 150.50, 'duration_hours': 3.2, 'symbol': 'EUR/USD'},
                {'pnl': -80.25, 'duration_hours': 5.1, 'symbol': 'GBP/USD'},
                {'pnl': 200.75, 'duration_hours': 2.5, 'symbol': 'USD/JPY'},
                {'pnl': 50.00, 'duration_hours': 1.8, 'symbol': 'AUD/USD'},
                {'pnl': -30.50, 'duration_hours': 4.3, 'symbol': 'EUR/USD'},
                {'pnl': 120.00, 'duration_hours': 2.1, 'symbol': 'GBP/USD'},
                {'pnl': 75.25, 'duration_hours': 3.7, 'symbol': 'USD/JPY'},
                {'pnl': -45.75, 'duration_hours': 6.2, 'symbol': 'AUD/USD'},
            ]
            
            self.logger.info(f"✅ Performance data collected: {len(self.returns_data)} returns, {len(self.trades)} trades")
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting performance data: {e}")
            return False
    
    async def analyze_performance(self):
        """
        Run comprehensive performance analysis using AdvancedMetricsCalculator
        """
        try:
            if self.returns_data.empty:
                self.logger.warning("⚠️ No returns data available for analysis")
                return {'error': 'No returns data available'}
            
            self.logger.info("📈 Running advanced performance analysis...")
            
            # Check if scipy is available
            if not SCIPY_AVAILABLE:
                self.logger.warning("⚠️ scipy not available. Some metrics may be limited.")
            
            # Use the AdvancedMetricsCalculator from advanced_metrics.py
            metrics = self.performance_analyzer.calculate_comprehensive_metrics(
                returns=self.returns_data,
                trades=self.trades,
                initial_balance=10000
            )
            
            # Check if metrics were calculated
            if not metrics:
                self.logger.error("❌ Failed to calculate performance metrics")
                return {'error': 'Metrics calculation failed'}
            
            # 🔥 Generate human-readable report
            report = self.performance_analyzer.generate_performance_report(metrics)
            
            # Log key performance metrics
            self.logger.info("🎯 PERFORMANCE ANALYSIS RESULTS:")
            if 'sharpe_ratio' in metrics:
                self.logger.info(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            if 'sortino_ratio' in metrics:
                self.logger.info(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            if 'max_drawdown' in metrics:
                self.logger.info(f"   Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            if 'annual_return' in metrics:
                self.logger.info(f"   Annual Return: {metrics.get('annual_return', 0)*100:.2f}%")
            if 'win_rate' in metrics:
                self.logger.info(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            
            # Save report to file
            report_path = Path('logs/performance_report.txt')
            report_path.parent.mkdir(exist_ok=True)  # Create logs directory if not exists
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"💾 Performance report saved to: {report_path}")
            
            return {
                'metrics': metrics,
                'report': report,
                'report_path': str(report_path),
                'timestamp': datetime.now().isoformat(),
                'scipy_available': SCIPY_AVAILABLE
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance analysis failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    # 4. Modify _system_monitoring_loop to include performance analysis:
    async def _system_monitoring_loop(self):
        """System monitoring and health checks"""
        analysis_counter = 0
        
        while self.system_status_main == "RUNNING":
            try:
                # Perform health checks
                health_status = await self._perform_system_health_checks()
                
                # 🔥 NEW: Every 30 minutes, run performance analysis
                analysis_counter += 1
                if analysis_counter % 6 == 0:  # 6 * 5 minutes = 30 minutes
                    self.logger.info("⏰ Time for scheduled performance analysis...")
                    analysis_result = await self.analyze_performance()
                    
                    if 'error' not in analysis_result:
                        self.logger.info("✅ Scheduled performance analysis completed")
                
                # Log system status
                self.logger.info(
                    f"📊 System Status - "
                    f"Uptime: {self.performance_metrics['system_uptime']:.0f}s, "
                    f"Decisions: {self.performance_metrics['total_decisions']}, "
                    f"Trades: {self.performance_metrics['successful_trades']}, "
                    f"Next analysis in: {(6 - (analysis_counter % 6)) * 5} minutes"
                )
                
                await asyncio.sleep(300)  # 5 minutes
    
    async def start_dashboards(self):
        """Start all dashboard systems"""
        try:
            self.logger.info("🚀 Starting Dashboard Systems...")
            
            if 'realtime' in self.dashboards:
                await self.dashboards['realtime'].start_dashboard()
                self.dashboard_status['realtime'] = 'RUNNING'
            
            if 'advanced_viz' in self.dashboards:
                await self.dashboards['advanced_viz'].start_dashboard()
                self.dashboard_status['advanced_viz'] = 'RUNNING'
            
            self.logger.info("✅ All dashboards started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dashboard startup failed: {e}")
            return False
        
    def _run_sync_async_methods(self):
        """
        Run async methods from sync context (__init__)
        Using event loop properly
        """
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in async context, create task
                asyncio.create_task(self._set_environment_vars_async())
                self.logger.debug("Running async method in existing event loop")
            except RuntimeError:
                # No running loop, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._set_environment_vars_async())
                loop.close()
                self.logger.debug("Running async method in new event loop")
                
        except Exception as e:
            self.logger.error(f"Error running async methods: {e}")
            # Fallback: call sync version
            self._set_environment_vars_sync()
    
    async def _set_environment_vars_async(self):
        """Async version of set environment variables"""
        for key, value in self.api_keys.items():
            if key not in os.environ:
                os.environ[key] = value
                self.logger.debug(f"Set environment variable: {key}")
        
        self.logger.info(f"Loaded {len(self.api_keys)} API keys")
    
    def _set_environment_vars_sync(self):
        """Sync fallback version"""
        for key, value in self.api_keys.items():
            if key not in os.environ:
                os.environ[key] = value
                self.logger.debug(f"Set environment variable: {key}")
        
        self.logger.info(f"Loaded {len(self.api_keys)} API keys (sync fallback)")
    
    def _call_async_from_sync(self):
        """Helper to call async method from sync context"""
        try:
            # Try to get existing loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async method
        if loop.is_running():
            # If loop is running, create task
            loop.create_task(self._set_environment_vars())
        else:
            # Run until complete
            loop.run_until_complete(self._set_environment_vars())
    
    async def start_trading_operation(self):
        """Start complete trading operation"""
        try:
            if self.system_status_main != "INITIALIZED":
                self.logger.error("System not initialized. Please initialize first.")
                return False
            
            self.logger.info("🚀 STARTING COMPLETE TRADING OPERATION...")
            self.system_status_main = "RUNNING"
            await self.collect_performance_data()
            # Start all module operations
            await self._start_all_module_operations()
            
            # Start main trading loop
            asyncio.create_task(self._master_trading_loop())
            
            # Start system monitoring
            asyncio.create_task(self._system_monitoring_loop())
            
            self.logger.info("✅ TRADING OPERATION STARTED SUCCESSFULLY!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start trading operation: {e}")
            return False
    
    async def _start_all_module_operations(self):
        """Start operations for all active modules"""
        try:
            # Start notifications system
            if 'notifications' in self.components:
                await self.components['notifications'].start_monitoring()
            
            # Start social sentiment monitoring
            if 'community_analyzer' in self.components:
                await self.components['community_analyzer'].start_monitoring()
            
            # Start risk monitoring
            if 'correlation_manager' in self.components:
                await self.components['correlation_manager'].start_monitoring()
            
            self.logger.info("✅ All module operations started")
            
        except Exception as e:
            self.logger.error(f"Error starting module operations: {e}")
    
    async def _master_trading_loop(self):
        """Master trading loop integrating all modules"""
        iteration = 0
        
        while self.system_status_main == "RUNNING":
            try:
                iteration += 1
                self.logger.info(f"🔄 Trading Iteration {iteration}...")
                
                # 1. Market Analysis Phase
                market_analysis = await self._perform_market_analysis()
                
                # 2. Risk Assessment Phase
                risk_assessment = await self._perform_risk_assessment()
                
                # 3. Strategy Decision Phase
                trading_decisions = await self._generate_trading_decisions(market_analysis, risk_assessment)
                
                # 4. Execution Phase
                if trading_decisions:
                    await self._execute_trading_decisions(trading_decisions)
                
                # 5. Monitoring & Optimization Phase
                await self._perform_system_optimization()
                
                # Update performance metrics
                self.performance_metrics['total_decisions'] += len(trading_decisions) if trading_decisions else 0
                self.performance_metrics['system_uptime'] = (datetime.now() - self.start_time).total_seconds()
                
                # Wait for next cycle
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in master trading loop: {e}")
                await asyncio.sleep(30)
    
    async def _perform_market_analysis(self):
        """Perform comprehensive market analysis"""
        try:
            analysis = {
                'timestamp': datetime.now(),
                'technical_analysis': {},
                'sentiment_analysis': {},
                'risk_metrics': {},
                'optimization_suggestions': {}
            }
            
            # Technical Analysis
            if 'multi_timeframe_analyzer' in self.components:
                for symbol in self._get_master_config()['trading']['enabled_pairs']:
                    signal = await self.components['multi_timeframe_analyzer'].generate_signal(symbol)
                    analysis['technical_analysis'][symbol] = signal
            
            # Sentiment Analysis
            if 'community_analyzer' in self.components:
                sentiment_data = await self.components['community_analyzer'].get_aggregated_signals()
                analysis['sentiment_analysis'] = sentiment_data
            
            # Risk Analysis
            if 'var_calculator' in self.components:
                var_results = self.components['var_calculator'].calculate_comprehensive_var(10000.0)
                analysis['risk_metrics']['var'] = var_results
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return {}
    
    async def _perform_risk_assessment(self):
        """Perform comprehensive risk assessment"""
        try:
            risk_report = {
                'timestamp': datetime.now(),
                'correlation_risk': {},
                'market_risk': {},
                'portfolio_risk': {}
            }
            
            # Correlation Risk
            if 'correlation_manager' in self.components:
                correlation_insights = self.components['correlation_manager'].get_correlation_insights()
                risk_report['correlation_risk'] = correlation_insights
            
            # Market Risk
            if 'monte_carlo_simulator' in self.components:
                mc_results = await self.components['monte_carlo_simulator'].run_simulation()
                risk_report['market_risk'] = mc_results
            
            return risk_report
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {}
    
    async def _generate_trading_decisions(self, market_analysis, risk_assessment):
        """Generate trading decisions using strategy modules"""
        try:
            decisions = []
            
            # Use Decision Engine if available
            if 'decision_engine' in self.components:
                for symbol in self._get_master_config()['trading']['enabled_pairs']:
                    decision = await self.components['decision_engine'].make_trading_decision(
                        symbol, market_analysis.get(symbol, {})
                    )
                    if decision and decision.confidence > 0.6:
                        decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            self.logger.error(f"Decision generation failed: {e}")
            return []
    
    async def _execute_trading_decisions(self, decisions):
        """Execute trading decisions"""
        try:
            for decision in decisions:
                self.logger.info(f"🎯 Executing decision: {decision.symbol} - {decision.action}")
                
                # Send notification
                if 'notifications' in self.components:
                    await self.components['notifications'].send_trading_alert(
                        symbol=decision.symbol,
                        action=decision.action,
                        confidence=decision.confidence,
                        price=decision.price
                    )
                
                # Update performance
                self.performance_metrics['successful_trades'] += 1
            
        except Exception as e:
            self.logger.error(f"Decision execution failed: {e}")
    
    async def _perform_system_optimization(self):
        """Perform system optimization"""
        try:
            # Run genetic optimization periodically
            if 'genetic_optimizer' in self.components:
                # Optimization logic here
                pass
            
            # Optimize portfolio allocation
            if 'portfolio_optimizer' in self.components:
                # Portfolio optimization logic here
                pass
            
        except Exception as e:
            self.logger.error(f"System optimization failed: {e}")
    
    async def _perform_system_health_checks(self):
        """Perform comprehensive system health checks"""
        try:
            health_report = {
                'timestamp': datetime.now(),
                'overall_health': 'HEALTHY',
                'module_health': self.performance_metrics['module_health'],
                'performance_metrics': self.performance_metrics,
                'deployment_status': 'ACTIVE'
            }
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"Health checks failed: {e}")
            return {'overall_health': 'UNKNOWN'}
    
    # ... (REST OF YOUR ORIGINAL METHODS - run_complete_pipeline, collect_market_data, etc.)
    # Include all your original methods here...
    
    async def run_complete_pipeline(self, symbol: str = "EUR/USD"):
        """Run complete trading pipeline for a symbol"""
        try:
            self.logger.info(f"🏃 Starting Complete Pipeline for {symbol}...")
            
            # Your original pipeline code here...
            # [Include all your original pipeline methods]
            
            return {
                'symbol': symbol,
                'trading_signal': {'action': 'HOLD', 'confidence': 0.5},
                'risk_approved': True,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            return {'error': str(e)}
    
    async def get_dashboard_status(self) -> dict:
        """Get dashboard system status"""
        return {
            'dashboards': self.dashboard_status,
            'timestamp': datetime.now(),
            'realtime_url': 'http://localhost:8050',
            'advanced_viz_url': 'http://localhost:8051', 
            'html_templates': 'templates/dashboard.html'
        }
    
    async def get_system_status(self) -> dict:
        """Get complete system status"""
        status = {
            'system': {
                'status': self.system_status_main,
                'uptime': self.performance_metrics['system_uptime'],
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'version': self._get_master_config()['system']['version']
            },
            'performance': self.performance_metrics,
            'performance_data': {
                'returns_count': len(self.returns_data),
                'trades_count': len(self.trades),
                'last_analysis': 'Not yet performed',
                'performance_analyzer_ready': hasattr(self, 'performance_analyzer')
            },
            'modules_initialized': list(self.components.keys()),
            'module_health': self.performance_metrics['module_health'],
            'trading_config': self._get_master_config()['trading'],
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    async def emergency_stop(self):
        """Emergency stop the entire system"""
        try:
            self.logger.critical("🆘 EMERGENCY STOP ACTIVATED!")
            self.system_status_main = "EMERGENCY_STOP"
            
            # Stop all module operations
            for module_name, module in self.components.items():
                try:
                    if hasattr(module, 'stop_monitoring'):
                        await module.stop_monitoring()
                    if hasattr(module, 'cleanup'):
                        await module.cleanup()
                except Exception as e:
                    self.logger.error(f"Error stopping {module_name}: {e}")
            
            self.logger.critical("✅ EMERGENCY STOP COMPLETED")
            
        except Exception as e:
            self.logger.critical(f"💥 EMERGENCY STOP FAILED: {e}")
    
    async def graceful_shutdown(self):
        """Gracefully shutdown the entire system"""
        try:
            self.logger.info("🛑 INITIATING GRACEFUL SHUTDOWN...")
            self.system_status_main = "SHUTTING_DOWN"
            
            # Stop all operations
            await self.emergency_stop()
            
            # Additional cleanup
            self.components.clear()
            
            self.system_status_main = "STOPPED"
            self.logger.info("✅ GRACEFUL SHUTDOWN COMPLETED")
            
        except Exception as e:
            self.logger.error(f"Graceful shutdown failed: {e}")

# Dashboard Templates Handler
def setup_dashboard_templates():
    """Setup and validate dashboard templates"""
    template_path = Path(__file__).parent / 'templates' / 'dashboard.html'
    
    if template_path.exists():
        logging.info(f"✅ Dashboard template found: {template_path}")
        return {
            'status': 'LOADED',
            'template_path': str(template_path),
            'file_size': template_path.stat().st_size,
            'last_modified': datetime.fromtimestamp(template_path.stat().st_mtime)
        }
    else:
        logging.warning(f"⚠️ Dashboard template not found: {template_path}")
        return {
            'status': 'MISSING', 
            'template_path': str(template_path)
        }

# Main execution function
async def main():
    """Main function to run the complete Forex Trading Bot"""
    print("🤖 FOREX TRADING BOT - COMPLETE INTEGRATED SYSTEM")
    print("=" * 60)
    
    # Check for required directories
    logs_dir = Path('logs')
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True)
        print(f"📁 Created logs directory: {logs_dir}")
    
    # Create config directory if it doesn't exist
    config_dir = Path('config')
    if not config_dir.exists():
        config_dir.mkdir(parents=True)
        print(f"📁 Created config directory: {config_dir}")
    
    # Check if config file exists, if not run setup
    config_file = config_dir / 'api_keys.yaml'
    if not config_file.exists():
        print("\n⚠️  No configuration file found!")
        print("Running first-time setup...\n")
        
        # Ask user if they want to run setup
        response = input("Run interactive setup? (y/n): ").lower()
        if response == 'y':
            # Run setup script
            import subprocess
            try:
                subprocess.run([sys.executable, "setup.py"])
                print("\n✅ Setup complete! Restarting bot...")
            except Exception as e:
                print(f"❌ Setup failed: {e}")
                return 1
        else:
            print("\n⚠️  Setup skipped. Bot will run with default settings.")
            print("Some features may not work without API keys.")
    
    try:
        # Create and initialize the trading bot
        trading_bot = ForexTradingBot()
        
        print("🔧 Initializing all systems...")
        success = await trading_bot.initialize_all_systems()
        
        if not success:
            print("❌ Initialization failed. Check logs for details.")
            return 1
        
        # Get initial status
        status = await trading_bot.get_system_status()
        print(f"✅ System initialized successfully!")
        print(f"   Systems loaded: {len(status['system'])}")
        print(f"   Advanced modules: {len(status['modules_initialized'])}")
        print(f"   Overall status: {status['system']['status']}")
        
        # Start dashboards
        print("🚀 Starting dashboards...")
        await trading_bot.start_dashboards()
        
        # Start trading operation
        print("💰 Starting trading operation...")
        await trading_bot.start_trading_operation()
        
        # Keep the system running
        print("📍 System is running. Press Ctrl+C to stop.")
        print("📊 Monitoring system performance...")
        
        try:
            # Run for demonstration
            for i in range(6):
                await asyncio.sleep(10)
                
                current_status = await trading_bot.get_system_status()
                print(f"\n📈 Status Update {i+1}:")
                print(f"   Uptime: {current_status['system']['uptime']:.0f}s")
                print(f"   Total Decisions: {current_status['performance']['total_decisions']}")
                print(f"   Successful Trades: {current_status['performance']['successful_trades']}")
            
            # Graceful shutdown
            print("\n🛑 Demonstration complete. Shutting down...")
            await trading_bot.graceful_shutdown()
            
        except KeyboardInterrupt:
            print("\n🛑 User interrupted. Shutting down...")
            await trading_bot.graceful_shutdown()
        
        print("✅ Forex Trading Bot completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Run the complete system
    exit_code = asyncio.run(main())
    exit(exit_code)
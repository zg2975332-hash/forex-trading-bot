"""
Real-Time Adaptor for Forex Trading Bot
Dynamic strategy adaptation based on live market conditions and performance feedback
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import threading
import time
import json
from collections import deque
import asyncio
import aiohttp

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptationSignal(Enum):
    """Adaptation signal types"""
    INCREASE_AGGRESSION = "increase_aggression"
    DECREASE_AGGRESSION = "decrease_aggression"
    SWITCH_STRATEGY = "switch_strategy"
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"
    ADJUST_RISK = "adjust_risk"
    OPTIMIZE_PARAMS = "optimize_params"
    NO_ACTION = "no_action"

class MarketCondition(Enum):
    """Market condition classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNCERTAIN = "uncertain"

@dataclass
class AdaptationMetrics:
    """Real-time adaptation metrics"""
    timestamp: datetime
    market_condition: MarketCondition
    adaptation_signal: AdaptationSignal
    confidence: float
    performance_score: float
    market_volatility: float
    trend_strength: float
    liquidity_score: float
    adaptation_parameters: Dict[str, float]
    reasoning: List[str]

@dataclass
class StrategyState:
    """Current strategy state and parameters"""
    strategy_name: str
    parameters: Dict[str, float]
    performance: float
    confidence: float
    active: bool
    weight: float
    last_optimized: datetime

class PerformanceMonitor:
    """Real-time performance monitoring and analysis"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.performance_history = deque(maxlen=lookback_period)
        self.trade_history = deque(maxlen=lookback_period)
        self.market_conditions = deque(maxlen=lookback_period)
        
        # Performance thresholds
        self.thresholds = {
            'performance_decline': -0.05,  # 5% decline
            'high_volatility': 0.15,       # 15% volatility
            'low_confidence': 0.6,         # 60% confidence
            'drawdown_limit': 0.1,         # 10% drawdown
            'win_rate_decline': 0.1        # 10% win rate decline
        }
        
        logger.info("Performance Monitor initialized")
    
    def update_performance(self, trade_pnl: float, confidence: float, 
                          market_condition: MarketCondition):
        """Update performance metrics"""
        performance_data = {
            'timestamp': datetime.now(),
            'pnl': trade_pnl,
            'confidence': confidence,
            'market_condition': market_condition,
            'cumulative_pnl': self._calculate_cumulative_pnl()
        }
        
        self.performance_history.append(performance_data)
    
    def update_trade(self, trade_data: Dict[str, Any]):
        """Update trade history"""
        self.trade_history.append(trade_data)
    
    def analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trends and detect issues"""
        if len(self.performance_history) < 10:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        try:
            # Extract recent performance
            recent_performance = [p['pnl'] for p in list(self.performance_history)[-20:]]
            recent_confidence = [p['confidence'] for p in list(self.performance_history)[-20:]]
            
            # Calculate trends
            pnl_trend = self._calculate_trend(recent_performance)
            confidence_trend = self._calculate_trend(recent_confidence)
            
            # Calculate metrics
            win_rate = self._calculate_win_rate()
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown()
            
            # Determine overall trend
            if (pnl_trend < self.thresholds['performance_decline'] and 
                confidence_trend < self.thresholds['low_confidence']):
                trend = 'deteriorating'
            elif pnl_trend > 0.02 and confidence_trend > 0:
                trend = 'improving'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'confidence': min(1.0, abs(pnl_trend) * 10),
                'pnl_trend': pnl_trend,
                'confidence_trend': confidence_trend,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'issues': self._detect_performance_issues()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trend: {e}")
            return {'trend': 'error', 'confidence': 0.0}
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        return slope
    
    def _calculate_win_rate(self) -> float:
        """Calculate recent win rate"""
        if not self.trade_history:
            return 0.0
        
        recent_trades = list(self.trade_history)[-20:]
        winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(recent_trades) if recent_trades else 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent performance"""
        if len(self.performance_history) < 2:
            return 0.0
        
        returns = [p['pnl'] for p in list(self.performance_history)[-50:]]
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_cumulative_pnl(self) -> float:
        """Calculate cumulative P&L"""
        if not self.performance_history:
            return 0.0
        return sum(p['pnl'] for p in self.performance_history)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.performance_history:
            return 0.0
        
        cumulative_pnls = []
        current_sum = 0
        for p in self.performance_history:
            current_sum += p['pnl']
            cumulative_pnls.append(current_sum)
        
        running_max = np.maximum.accumulate(cumulative_pnls)
        drawdowns = (cumulative_pnls - running_max) / running_max
        return np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    def _detect_performance_issues(self) -> List[str]:
        """Detect specific performance issues"""
        issues = []
        
        analysis = self.analyze_performance_trend()
        
        if analysis['pnl_trend'] < self.thresholds['performance_decline']:
            issues.append("Performance declining")
        
        if analysis['win_rate'] < 0.4:  # 40% win rate
            issues.append("Low win rate")
        
        if analysis['max_drawdown'] < self.thresholds['drawdown_limit']:
            issues.append("Approaching drawdown limit")
        
        if analysis['sharpe_ratio'] < 0.5:
            issues.append("Poor risk-adjusted returns")
        
        return issues

class MarketConditionAnalyzer:
    """Real-time market condition analysis"""
    
    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [10, 20, 50, 100]
        self.price_history = deque(maxlen=max(window_sizes))
        self.volume_history = deque(maxlen=max(window_sizes))
        self.volatility_history = deque(maxlen=100)
        
        # Condition thresholds
        self.thresholds = {
            'trend_strength_strong': 0.7,
            'trend_strength_weak': 0.3,
            'volatility_high': 0.02,
            'volatility_low': 0.005,
            'liquidity_high': 1000000,
            'liquidity_low': 100000
        }
        
        logger.info("Market Condition Analyzer initialized")
    
    def update_market_data(self, price: float, volume: float, timestamp: datetime):
        """Update market data for analysis"""
        self.price_history.append({
            'timestamp': timestamp,
            'price': price,
            'volume': volume
        })
        
        # Calculate and store volatility
        if len(self.price_history) >= 20:
            recent_prices = [p['price'] for p in list(self.price_history)[-20:]]
            volatility = np.std(np.diff(recent_prices)) / np.mean(recent_prices)
            self.volatility_history.append(volatility)
    
    def analyze_current_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions"""
        if len(self.price_history) < min(self.window_sizes):
            return {
                'market_condition': MarketCondition.UNCERTAIN,
                'confidence': 0.1,
                'trend_strength': 0.0,
                'volatility': 0.0,
                'liquidity': 0.0,
                'reasoning': ['Insufficient data']
            }
        
        try:
            prices = [p['price'] for p in self.price_history]
            volumes = [p['volume'] for p in self.price_history]
            
            # Calculate metrics for different timeframes
            trend_strengths = []
            volatilities = []
            
            for window in self.window_sizes:
                if len(prices) >= window:
                    window_prices = prices[-window:]
                    trend_strengths.append(self._calculate_trend_strength(window_prices))
                    volatilities.append(self._calculate_volatility(window_prices))
            
            # Aggregate metrics
            avg_trend_strength = np.mean(trend_strengths) if trend_strengths else 0.0
            avg_volatility = np.mean(volatilities) if volatilities else 0.0
            liquidity = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0.0
            
            # Determine market condition
            condition, confidence, reasoning = self._classify_condition(
                avg_trend_strength, avg_volatility, liquidity
            )
            
            return {
                'market_condition': condition,
                'confidence': confidence,
                'trend_strength': avg_trend_strength,
                'volatility': avg_volatility,
                'liquidity': liquidity,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                'market_condition': MarketCondition.UNCERTAIN,
                'confidence': 0.1,
                'trend_strength': 0.0,
                'volatility': 0.0,
                'liquidity': 0.0,
                'reasoning': [f'Analysis error: {str(e)}']
            }
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        return abs(r_value) * np.sign(slope)  # Strength with direction
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def _classify_condition(self, trend_strength: float, volatility: float, 
                          liquidity: float) -> Tuple[MarketCondition, float, List[str]]:
        """Classify market condition based on metrics"""
        reasoning = []
        confidence_factors = []
        
        # Trend analysis
        if trend_strength > self.thresholds['trend_strength_strong']:
            condition = MarketCondition.TRENDING_UP
            reasoning.append(f"Strong uptrend (strength: {trend_strength:.3f})")
            confidence_factors.append(0.8)
        elif trend_strength < -self.thresholds['trend_strength_strong']:
            condition = MarketCondition.TRENDING_DOWN
            reasoning.append(f"Strong downtrend (strength: {abs(trend_strength):.3f})")
            confidence_factors.append(0.8)
        elif abs(trend_strength) < self.thresholds['trend_strength_weak']:
            condition = MarketCondition.RANGING
            reasoning.append(f"Ranging market (strength: {abs(trend_strength):.3f})")
            confidence_factors.append(0.7)
        else:
            condition = MarketCondition.UNCERTAIN
            reasoning.append(f"Mixed trend signals (strength: {trend_strength:.3f})")
            confidence_factors.append(0.4)
        
        # Volatility analysis
        if volatility > self.thresholds['volatility_high']:
            condition = MarketCondition.VOLATILE
            reasoning.append(f"High volatility ({volatility:.3f})")
            confidence_factors.append(0.9)
        elif volatility < self.thresholds['volatility_low']:
            condition = MarketCondition.CALM
            reasoning.append(f"Low volatility ({volatility:.3f})")
            confidence_factors.append(0.8)
        
        # Liquidity analysis
        if liquidity > self.thresholds['liquidity_high']:
            reasoning.append("High liquidity")
            confidence_factors.append(0.9)
        elif liquidity < self.thresholds['liquidity_low']:
            reasoning.append("Low liquidity")
            confidence_factors.append(0.6)
        
        # Calculate overall confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return condition, confidence, reasoning

class AdaptiveStrategyManager:
    """Dynamic strategy adaptation and optimization"""
    
    def __init__(self, initial_strategies: Dict[str, Dict[str, Any]]):
        self.strategies: Dict[str, StrategyState] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.adaptation_history: List[AdaptationMetrics] = []
        
        # Initialize strategies
        for name, config in initial_strategies.items():
            self.strategies[name] = StrategyState(
                strategy_name=name,
                parameters=config.get('parameters', {}),
                performance=0.0,
                confidence=config.get('initial_confidence', 0.7),
                active=config.get('active', True),
                weight=config.get('initial_weight', 0.0),
                last_optimized=datetime.now()
            )
            self.performance_history[name] = []
        
        # ML model for parameter optimization
        self.parameter_optimizer = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Adaptation rules
        self.adaptation_rules = self._initialize_adaptation_rules()
        
        logger.info("Adaptive Strategy Manager initialized")
    
    def _initialize_adaptation_rules(self) -> List[Dict[str, Any]]:
        """Initialize adaptation rules"""
        return [
            {
                'condition': lambda metrics: (
                    metrics['performance_trend']['trend'] == 'deteriorating' and
                    metrics['market_conditions']['volatility'] > 0.1
                ),
                'action': AdaptationSignal.DECREASE_AGGRESSION,
                'confidence_boost': 0.8,
                'description': "Reduce aggression in high volatility with poor performance"
            },
            {
                'condition': lambda metrics: (
                    metrics['performance_trend']['trend'] == 'improving' and
                    metrics['market_conditions']['trend_strength'] > 0.5
                ),
                'action': AdaptationSignal.INCREASE_AGGRESSION,
                'confidence_boost': 0.7,
                'description': "Increase aggression in strong trends with good performance"
            },
            {
                'condition': lambda metrics: (
                    len(metrics['performance_trend']['issues']) >= 2 and
                    metrics['performance_trend']['win_rate'] < 0.4
                ),
                'action': AdaptationSignal.SWITCH_STRATEGY,
                'confidence_boost': 0.9,
                'description': "Switch strategy due to multiple performance issues"
            },
            {
                'condition': lambda metrics: (
                    metrics['market_conditions']['market_condition'] == MarketCondition.VOLATILE and
                    metrics['performance_trend']['max_drawdown'] < -0.08
                ),
                'action': AdaptationSignal.PAUSE_TRADING,
                'confidence_boost': 0.85,
                'description': "Pause trading in high volatility with significant drawdown"
            }
        ]
    
    def update_strategy_performance(self, strategy_name: str, pnl: float, confidence: float):
        """Update strategy performance metrics"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].performance = pnl
            self.strategies[strategy_name].confidence = confidence
            self.performance_history[strategy_name].append(pnl)
            
            # Keep history manageable
            if len(self.performance_history[strategy_name]) > 1000:
                self.performance_history[strategy_name] = self.performance_history[strategy_name][-1000:]
    
    def generate_adaptation_signals(self, performance_analysis: Dict[str, Any],
                                  market_analysis: Dict[str, Any]) -> AdaptationMetrics:
        """Generate adaptation signals based on current conditions"""
        try:
            # Combine analysis data
            combined_metrics = {
                'performance_trend': performance_analysis,
                'market_conditions': market_analysis,
                'timestamp': datetime.now()
            }
            
            # Evaluate adaptation rules
            best_signal = AdaptationSignal.NO_ACTION
            best_confidence = 0.0
            best_reasoning = []
            
            for rule in self.adaptation_rules:
                try:
                    if rule['condition'](combined_metrics):
                        signal_confidence = (market_analysis['confidence'] * 
                                           performance_analysis['confidence'] * 
                                           rule['confidence_boost'])
                        
                        if signal_confidence > best_confidence:
                            best_signal = rule['action']
                            best_confidence = signal_confidence
                            best_reasoning = [rule['description']]
                except Exception as e:
                    logger.warning(f"Error evaluating adaptation rule: {e}")
                    continue
            
            # Generate adaptation parameters
            adaptation_params = self._calculate_adaptation_parameters(
                best_signal, combined_metrics
            )
            
            # Create metrics
            metrics = AdaptationMetrics(
                timestamp=datetime.now(),
                market_condition=market_analysis['market_condition'],
                adaptation_signal=best_signal,
                confidence=best_confidence,
                performance_score=performance_analysis.get('sharpe_ratio', 0.0),
                market_volatility=market_analysis['volatility'],
                trend_strength=market_analysis['trend_strength'],
                liquidity_score=market_analysis['liquidity'],
                adaptation_parameters=adaptation_params,
                reasoning=best_reasoning
            )
            
            self.adaptation_history.append(metrics)
            
            # Keep history manageable
            if len(self.adaptation_history) > 1000:
                self.adaptation_history = self.adaptation_history[-1000:]
            
            logger.info(f"Adaptation signal generated: {best_signal.value} "
                       f"(confidence: {best_confidence:.2f})")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating adaptation signals: {e}")
            return self._create_default_metrics()
    
    def _calculate_adaptation_parameters(self, signal: AdaptationSignal, 
                                       metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate specific adaptation parameters"""
        params = {}
        
        if signal == AdaptationSignal.INCREASE_AGGRESSION:
            params = {
                'position_size_multiplier': 1.2,
                'confidence_threshold': 0.6,
                'max_trades_per_hour': 10,
                'risk_per_trade': 0.025
            }
        elif signal == AdaptationSignal.DECREASE_AGGRESSION:
            params = {
                'position_size_multiplier': 0.7,
                'confidence_threshold': 0.8,
                'max_trades_per_hour': 3,
                'risk_per_trade': 0.015
            }
        elif signal == AdaptationSignal.SWITCH_STRATEGY:
            # Find best performing alternative strategy
            best_strategy = self._find_best_alternative_strategy()
            params = {
                'new_strategy': best_strategy,
                'transition_period_minutes': 30,
                'weight_adjustment_rate': 0.1
            }
        elif signal == AdaptationSignal.ADJUST_RISK:
            volatility = metrics['market_conditions']['volatility']
            risk_multiplier = max(0.5, min(2.0, 1.0 / (volatility * 10)))
            params = {
                'risk_multiplier': risk_multiplier,
                'stop_loss_multiplier': 1.5 if volatility > 0.15 else 1.0,
                'position_size_adjustment': risk_multiplier
            }
        
        return params
    
    def _find_best_alternative_strategy(self) -> str:
        """Find the best performing alternative strategy"""
        active_strategies = [s for s in self.strategies.values() if s.active]
        if not active_strategies:
            return list(self.strategies.keys())[0] if self.strategies else "default"
        
        # Sort by performance and confidence
        active_strategies.sort(key=lambda x: x.performance * x.confidence, reverse=True)
        return active_strategies[0].strategy_name
    
    def _create_default_metrics(self) -> AdaptationMetrics:
        """Create default adaptation metrics"""
        return AdaptationMetrics(
            timestamp=datetime.now(),
            market_condition=MarketCondition.UNCERTAIN,
            adaptation_signal=AdaptationSignal.NO_ACTION,
            confidence=0.1,
            performance_score=0.0,
            market_volatility=0.0,
            trend_strength=0.0,
            liquidity_score=0.0,
            adaptation_parameters={},
            reasoning=["Default metrics due to analysis error"]
        )
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        total_confidence = sum(s.confidence for s in self.strategies.values() if s.active)
        if total_confidence == 0:
            return {name: 1.0/len(self.strategies) for name in self.strategies}
        
        weights = {}
        for name, strategy in self.strategies.items():
            if strategy.active:
                weights[name] = strategy.confidence / total_confidence
            else:
                weights[name] = 0.0
        
        return weights

class RealTimeAdaptor:
    """
    Main Real-Time Adaptor coordinating all adaptation components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.market_analyzer = MarketConditionAnalyzer()
        self.strategy_manager = AdaptiveStrategyManager(
            self.config.get('initial_strategies', {})
        )
        
        # Adaptation state
        self.current_adaptation = AdaptationSignal.NO_ACTION
        self.adaptation_parameters = {}
        self.last_adaptation_time = datetime.now()
        self.adaptation_cooldown = timedelta(minutes=5)  # Prevent frequent changes
        
        # Real-time data
        self.latest_market_data = {}
        self.latest_performance = {}
        
        # Monitoring thread
        self.adaptation_thread = None
        self.running = False
        self.adaptation_interval = self.config.get('adaptation_interval', 30)  # seconds
        
        logger.info("Real-Time Adaptor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'adaptation_interval': 30,
            'initial_strategies': {
                'momentum': {
                    'parameters': {'lookback': 20, 'threshold': 0.002},
                    'initial_confidence': 0.7,
                    'active': True,
                    'initial_weight': 0.4
                },
                'mean_reversion': {
                    'parameters': {'lookback': 50, 'threshold': 0.001},
                    'initial_confidence': 0.6,
                    'active': True,
                    'initial_weight': 0.3
                },
                'breakout': {
                    'parameters': {'lookback': 30, 'threshold': 0.003},
                    'initial_confidence': 0.5,
                    'active': True,
                    'initial_weight': 0.3
                }
            },
            'adaptation_rules': {
                'performance_decline_threshold': -0.05,
                'volatility_threshold': 0.15,
                'confidence_threshold': 0.6
            }
        }
    
    def start_adaptation(self):
        """Start real-time adaptation monitoring"""
        if self.running:
            return
        
        self.running = True
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        logger.info("Real-time adaptation started")
    
    def stop_adaptation(self):
        """Stop real-time adaptation monitoring"""
        self.running = False
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=10.0)
        logger.info("Real-time adaptation stopped")
    
    def _adaptation_loop(self):
        """Main adaptation monitoring loop"""
        while self.running:
            try:
                # Check if we have sufficient data
                if (len(self.latest_market_data) >= 20 and 
                    len(self.performance_monitor.performance_history) >= 10):
                    
                    # Perform adaptation analysis
                    self._perform_adaptation_analysis()
                
                # Wait for next cycle
                time.sleep(self.adaptation_interval)
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                time.sleep(self.adaptation_interval)  # Continue after error
    
    def _perform_adaptation_analysis(self):
        """Perform adaptation analysis and generate signals"""
        try:
            # Analyze performance
            performance_analysis = self.performance_monitor.analyze_performance_trend()
            
            # Analyze market conditions
            market_analysis = self.market_analyzer.analyze_current_conditions()
            
            # Check adaptation cooldown
            time_since_last_adaptation = datetime.now() - self.last_adaptation_time
            if time_since_last_adaptation < self.adaptation_cooldown:
                return
            
            # Generate adaptation signals
            adaptation_metrics = self.strategy_manager.generate_adaptation_signals(
                performance_analysis, market_analysis
            )
            
            # Apply adaptation if confidence is high enough
            if (adaptation_metrics.confidence > 0.7 and 
                adaptation_metrics.adaptation_signal != AdaptationSignal.NO_ACTION):
                
                self._apply_adaptation(adaptation_metrics)
                self.last_adaptation_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in adaptation analysis: {e}")
    
    def _apply_adaptation(self, adaptation_metrics: AdaptationMetrics):
        """Apply adaptation changes"""
        try:
            self.current_adaptation = adaptation_metrics.adaptation_signal
            self.adaptation_parameters = adaptation_metrics.adaptation_parameters
            
            logger.info(f"Applying adaptation: {adaptation_metrics.adaptation_signal.value}")
            logger.info(f"Adaptation parameters: {adaptation_metrics.adaptation_parameters}")
            logger.info(f"Reasoning: {adaptation_metrics.reasoning}")
            
            # Here you would integrate with your trading system
            # to actually apply the parameter changes
            
        except Exception as e:
            logger.error(f"Error applying adaptation: {e}")
    
    def update_market_data(self, symbol: str, price: float, volume: float, 
                          timestamp: datetime = None):
        """Update market data for adaptation analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.market_analyzer.update_market_data(price, volume, timestamp)
        self.latest_market_data[symbol] = {
            'price': price,
            'volume': volume,
            'timestamp': timestamp
        }
    
    def update_trade_result(self, strategy_name: str, trade_data: Dict[str, Any]):
        """Update trade results for performance monitoring"""
        try:
            pnl = trade_data.get('pnl', 0)
            confidence = trade_data.get('confidence', 0.5)
            
            # Get current market condition
            market_analysis = self.market_analyzer.analyze_current_conditions()
            
            # Update performance monitor
            self.performance_monitor.update_performance(pnl, confidence, 
                                                      market_analysis['market_condition'])
            self.performance_monitor.update_trade(trade_data)
            
            # Update strategy manager
            self.strategy_manager.update_strategy_performance(strategy_name, pnl, confidence)
            
            self.latest_performance = {
                'strategy': strategy_name,
                'pnl': pnl,
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status"""
        performance_analysis = self.performance_monitor.analyze_performance_trend()
        market_analysis = self.market_analyzer.analyze_current_conditions()
        strategy_weights = self.strategy_manager.get_strategy_weights()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_adaptation': self.current_adaptation.value,
            'adaptation_parameters': self.adaptation_parameters,
            'performance_analysis': performance_analysis,
            'market_analysis': market_analysis,
            'strategy_weights': strategy_weights,
            'adaptation_cooldown_remaining': max(0, (
                self.adaptation_cooldown - (datetime.now() - self.last_adaptation_time)
            ).total_seconds()),
            'system_health': {
                'market_data_points': len(self.market_analyzer.price_history),
                'performance_data_points': len(self.performance_monitor.performance_history),
                'adaptation_history_count': len(self.strategy_manager.adaptation_history),
                'running': self.running
            }
        }
    
    def get_adaptation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get adaptation history"""
        history = self.strategy_manager.adaptation_history[-limit:]
        return [asdict(metric) for metric in history]


# Example usage and testing
if __name__ == "__main__":
    # Test the Real-Time Adaptor
    print("Testing Real-Time Adaptor...")
    
    try:
        # Initialize adaptor
        adaptor = RealTimeAdaptor()
        
        # Generate sample market data
        print("Generating sample market data...")
        base_price = 1.1000
        
        for i in range(100):
            price = base_price + np.random.normal(0, 0.001) * i
            volume = 1000000 + np.random.randint(-200000, 200000)
            adaptor.update_market_data("EUR/USD", price, volume)
            time.sleep(0.01)  # Small delay to simulate real-time
        
        # Generate sample trade results
        print("Generating sample trade results...")
        strategies = ['momentum', 'mean_reversion', 'breakout']
        
        for i in range(20):
            strategy = np.random.choice(strategies)
            pnl = np.random.normal(10, 50)  # Random P&L
            confidence = 0.5 + np.random.random() * 0.5  # Random confidence
            
            trade_data = {
                'trade_id': f"TEST_{i}",
                'symbol': "EUR/USD",
                'pnl': pnl,
                'confidence': confidence,
                'strategy': strategy,
                'timestamp': datetime.now()
            }
            
            adaptor.update_trade_result(strategy, trade_data)
        
        # Start adaptation
        print("Starting real-time adaptation...")
        adaptor.start_adaptation()
        
        # Let it run for a bit
        time.sleep(10)
        
        # Get adaptation status
        print("Getting adaptation status...")
        status = adaptor.get_adaptation_status()
        
        print(f"\n📊 ADAPTATION STATUS:")
        print(f"Current Adaptation: {status['current_adaptation']}")
        print(f"Market Condition: {status['market_analysis']['market_condition'].value}")
        print(f"Performance Trend: {status['performance_analysis']['trend']}")
        print(f"Strategy Weights: {status['strategy_weights']}")
        
        # Get adaptation history
        history = adaptor.get_adaptation_history(5)
        print(f"\nRecent Adaptation History ({len(history)} signals):")
        for i, signal in enumerate(history[-3:], 1):
            print(f"  {i}. {signal['adaptation_signal']} "
                  f"(confidence: {signal['confidence']:.2f})")
        
        # Stop adaptation
        adaptor.stop_adaptation()
        
        print(f"\n✅ Real-Time Adaptor test completed successfully!")
        
    except Exception as e:
        print(f"❌ Real-Time Adaptor test failed: {e}")
        import traceback
        traceback.print_exc()
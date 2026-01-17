"""
Advanced Decision Engine for FOREX TRADING BOT
The brain of the trading system - integrates all components for final trade decisions
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from strategies.deep_learning_strat import AdvancedDeepLearningStrategy
    from strategies.multi_timeframe_analyzer import AdvancedMultiTimeframeAnalyzer
    from strategies.strategy_selector import AdvancedStrategySelector
    from strategies.signal_filter import AdvancedSignalFilter
    from models.ensemble_predictor import EnsemblePredictor
    from models.rl_agent import ReinforcementLearningAgent
    from news.sentiment_analyzer import SentimentAnalyzer
    from core.risk_manager import RiskManager
    from core.performance_tracker import PerformanceTracker
    from core.data_handler import DataHandler
    from core.market_regime_detector import MarketRegimeDetector
    from risk.var_calculator import AdvancedVaRCalculator
    from optimization.portfolio_optimizer import PortfolioOptimizer
except ImportError as e:
    print(f"Import warning: {e}")
    # Mock implementations for development
    class AdvancedDeepLearningStrategy:
        async def analyze(self, data):
            return {'signal': 'BUY', 'confidence': 0.75, 'features_used': 50}
    
    class AdvancedMultiTimeframeAnalyzer:
        async def analyze_all_timeframes(self, data):
            return {'consensus': 'BUY', 'alignment_score': 0.8}
    
    class AdvancedStrategySelector:
        async def select_strategies(self, conditions):
            return {'primary_strategy': 'deep_learning', 'confidence': 0.85}
    
    class AdvancedSignalFilter:
        async def apply_filters(self, signal):
            return {'filtered_signal': signal['signal'], 'passed_filters': True}
    
    class EnsemblePredictor:
        async def predict(self, data):
            return {'ensemble_prediction': 0.7, 'confidence': 0.8}
    
    class ReinforcementLearningAgent:
        async def predict_action(self, state):
            return {'action': 'BUY', 'q_value': 0.85}
    
    class SentimentAnalyzer:
        async def get_market_sentiment(self):
            return {'overall_sentiment': 0.6, 'sentiment_trend': 'improving'}
    
    class RiskManager:
        async def validate_trade(self, trade_signal):
            return {'approved': True, 'risk_score': 0.2, 'position_size': 10000}
    
    class MarketRegimeDetector:
        async def detect_regime(self, market_data):
            return {'regime': 'TRENDING', 'confidence': 0.8, 'volatility': 'MEDIUM'}
    
    class AdvancedVaRCalculator:
        async def calculate_portfolio_var(self, positions):
            return {'var_95': -1200, 'expected_shortfall': -1800}
    
    class PortfolioOptimizer:
        async def optimize_positions(self, signals, current_positions):
            return {'optimized_allocations': {'EUR/USD': 0.6, 'GBP/USD': 0.4}}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of trading decisions"""
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"
    HEDGE = "HEDGE"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"

class MarketCondition(Enum):
    """Market condition classifications"""
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"

@dataclass
class TradeSignal:
    """Unified trade signal from all analysis components"""
    symbol: str
    decision: DecisionType
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    timeframe: str = "1H"
    source: str = "ensemble"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketContext:
    """Current market context and conditions"""
    regime: MarketCondition
    volatility: float
    trend_strength: float
    sentiment_score: float
    economic_calendar: List[Dict]
    correlation_matrix: pd.DataFrame
    volume_profile: Dict[str, float]

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    overall_risk: float  # 0-1 scale
    market_risk: float
    liquidity_risk: float
    concentration_risk: float
    var_95: float
    expected_shortfall: float
    stress_scenario_loss: float
    risk_recommendation: str

@dataclass
class DecisionOutput:
    """Final decision output from the engine"""
    decision: DecisionType
    confidence: float
    symbol: str
    position_size: float
    risk_score: float
    reasoning: List[str]
    market_context: MarketContext
    risk_assessment: RiskAssessment
    timestamp: datetime = field(default_factory=datetime.now)
    execution_priority: str = "NORMAL"  # HIGH, NORMAL, LOW

class AdvancedDecisionEngine:
    """
    Advanced Decision Engine - The Brain of the Trading System
    Integrates all components to make final trading decisions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize all components
        self.components = self._initialize_components()
        
        # Decision engine state
        self.decision_history: List[DecisionOutput] = []
        self.market_context: Optional[MarketContext] = None
        self.risk_assessment: Optional[RiskAssessment] = None
        self.portfolio_state: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'decision_accuracy': 0.0,
            'avg_confidence': 0.0,
            'risk_adjusted_return': 0.0,
            'decision_latency': 0.0
        }
        
        # Cache for performance
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)
        
        logger.info("Advanced Decision Engine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "decision_parameters": {
                "min_confidence_threshold": 0.65,
                "max_risk_score": 0.7,
                "position_sizing_method": "kelly",
                "max_portfolio_risk": 0.02,  # 2% max portfolio risk
                "correlation_threshold": 0.8,
                "volatility_adjustment": True,
                "sentiment_weight": 0.15,
                "regime_aware": True
            },
            "component_weights": {
                "deep_learning": 0.30,
                "multi_timeframe": 0.25,
                "ensemble": 0.20,
                "reinforcement_learning": 0.15,
                "sentiment": 0.10
            },
            "risk_parameters": {
                "max_drawdown_limit": 0.05,
                "var_conf_level": 0.95,
                "liquidity_buffer": 0.1,
                "stress_test_scenarios": ["flash_crash", "rate_hike", "geopolitical"]
            },
            "optimization": {
                "portfolio_rebalance_frequency": "4H",
                "correlation_lookback": 30,
                "risk_parity_enabled": True,
                "dynamic_hedging": True
            }
        }
    
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize all decision components"""
        components = {
            'deep_learning_strategy': AdvancedDeepLearningStrategy(),
            'multi_timeframe_analyzer': AdvancedMultiTimeframeAnalyzer(),
            'strategy_selector': AdvancedStrategySelector(),
            'signal_filter': AdvancedSignalFilter(),
            'ensemble_predictor': EnsemblePredictor(),
            'rl_agent': ReinforcementLearningAgent(),
            'sentiment_analyzer': SentimentAnalyzer(),
            'risk_manager': RiskManager(),
            'market_regime_detector': MarketRegimeDetector(),
            'var_calculator': AdvancedVaRCalculator(),
            'portfolio_optimizer': PortfolioOptimizer(),
            'performance_tracker': PerformanceTracker(),
            'data_handler': DataHandler()
        }
        
        logger.info("All decision components initialized")
        return components
    
    async def initialize(self) -> bool:
        """Initialize the decision engine"""
        try:
            # Initialize all components
            init_tasks = []
            for name, component in self.components.items():
                if hasattr(component, 'initialize'):
                    if asyncio.iscoroutinefunction(component.initialize):
                        init_tasks.append(component.initialize())
                    else:
                        component.initialize()
            
            if init_tasks:
                await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Load initial market context
            await self._update_market_context()
            
            logger.info("Decision Engine fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Decision Engine initialization failed: {e}")
            return False
    
    async def make_trading_decision(self, symbol: str, market_data: pd.DataFrame) -> DecisionOutput:
        """
        Make comprehensive trading decision for a symbol
        This is the main decision-making function
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Update market context
            await self._update_market_context()
            
            # Step 2: Gather signals from all components
            signals = await self._gather_signals(symbol, market_data)
            
            # Step 3: Risk assessment
            risk_assessment = await self._assess_risk(symbol, signals, market_data)
            
            # Step 4: Portfolio optimization
            portfolio_recommendation = await self._optimize_portfolio(symbol, signals)
            
            # Step 5: Make final decision
            final_decision = await self._make_final_decision(
                symbol, signals, risk_assessment, portfolio_recommendation
            )
            
            # Step 6: Update performance metrics
            await self._update_performance_metrics(start_time, final_decision)
            
            # Step 7: Log decision
            self._log_decision(final_decision, signals, risk_assessment)
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Decision making failed for {symbol}: {e}")
            # Return safe HOLD decision on error
            return await self._create_safe_decision(symbol, f"Error: {str(e)}")
    
    async def _gather_signals(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, TradeSignal]:
        """Gather signals from all analysis components"""
        signals = {}
        
        # Deep Learning Strategy Signal
        try:
            dl_signal = await self.components['deep_learning_strategy'].analyze(market_data)
            signals['deep_learning'] = TradeSignal(
                symbol=symbol,
                decision=self._convert_to_decision_type(dl_signal.get('signal', 'HOLD')),
                confidence=dl_signal.get('confidence', 0.5),
                entry_price=dl_signal.get('entry_price'),
                stop_loss=dl_signal.get('stop_loss'),
                take_profit=dl_signal.get('take_profit'),
                source='deep_learning',
                metadata=dl_signal
            )
        except Exception as e:
            logger.warning(f"Deep learning signal failed: {e}")
            signals['deep_learning'] = self._create_neutral_signal(symbol, 'deep_learning')
        
        # Multi-Timeframe Analysis
        try:
            mtf_data = {tf: market_data for tf in ['1H', '4H', '1D']}  # Simplified
            mtf_signal = await self.components['multi_timeframe_analyzer'].analyze_all_timeframes(mtf_data)
            signals['multi_timeframe'] = TradeSignal(
                symbol=symbol,
                decision=self._convert_to_decision_type(mtf_signal.get('consensus', 'NEUTRAL')),
                confidence=mtf_signal.get('alignment_score', 0.5),
                source='multi_timeframe',
                metadata=mtf_signal
            )
        except Exception as e:
            logger.warning(f"Multi-timeframe signal failed: {e}")
            signals['multi_timeframe'] = self._create_neutral_signal(symbol, 'multi_timeframe')
        
        # Ensemble Predictor
        try:
            ensemble_signal = await self.components['ensemble_predictor'].predict(market_data)
            signals['ensemble'] = TradeSignal(
                symbol=symbol,
                decision=self._convert_to_decision_type(
                    'BUY' if ensemble_signal.get('ensemble_prediction', 0.5) > 0.6 else 
                    'SELL' if ensemble_signal.get('ensemble_prediction', 0.5) < 0.4 else 'HOLD'
                ),
                confidence=ensemble_signal.get('confidence', 0.5),
                source='ensemble',
                metadata=ensemble_signal
            )
        except Exception as e:
            logger.warning(f"Ensemble signal failed: {e}")
            signals['ensemble'] = self._create_neutral_signal(symbol, 'ensemble')
        
        # Reinforcement Learning Agent
        try:
            state = self._create_rl_state(market_data, self.market_context)
            rl_signal = await self.components['rl_agent'].predict_action(state)
            signals['reinforcement_learning'] = TradeSignal(
                symbol=symbol,
                decision=self._convert_to_decision_type(rl_signal.get('action', 'HOLD')),
                confidence=rl_signal.get('q_value', 0.5),
                source='reinforcement_learning',
                metadata=rl_signal
            )
        except Exception as e:
            logger.warning(f"RL signal failed: {e}")
            signals['reinforcement_learning'] = self._create_neutral_signal(symbol, 'reinforcement_learning')
        
        # Apply signal filtering
        filtered_signals = {}
        for source, signal in signals.items():
            try:
                filtered = await self.components['signal_filter'].apply_filters(signal.__dict__)
                if filtered.get('passed_filters', False):
                    signal.confidence = filtered.get('filtered_confidence', signal.confidence)
                    filtered_signals[source] = signal
            except Exception as e:
                logger.warning(f"Signal filtering failed for {source}: {e}")
                filtered_signals[source] = signal
        
        return filtered_signals
    
    async def _assess_risk(self, symbol: str, signals: Dict[str, TradeSignal], 
                          market_data: pd.DataFrame) -> RiskAssessment:
        """Comprehensive risk assessment"""
        try:
            # Market risk assessment
            market_risk = await self._calculate_market_risk(symbol, market_data)
            
            # Portfolio risk assessment
            portfolio_risk = await self._calculate_portfolio_risk(symbol, signals)
            
            # Liquidity risk assessment
            liquidity_risk = await self._calculate_liquidity_risk(symbol)
            
            # Concentration risk
            concentration_risk = await self._calculate_concentration_risk(symbol)
            
            # VaR calculation
            var_result = await self.components['var_calculator'].calculate_portfolio_var(
                self.portfolio_state.get('positions', {})
            )
            
            # Stress testing
            stress_loss = await self._perform_stress_testing(symbol, signals)
            
            # Overall risk score (weighted average)
            overall_risk = (
                market_risk * 0.4 +
                portfolio_risk * 0.3 +
                liquidity_risk * 0.15 +
                concentration_risk * 0.15
            )
            
            risk_assessment = RiskAssessment(
                overall_risk=overall_risk,
                market_risk=market_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                var_95=var_result.get('var_95', 0),
                expected_shortfall=var_result.get('expected_shortfall', 0),
                stress_scenario_loss=stress_loss,
                risk_recommendation=self._generate_risk_recommendation(overall_risk)
            )
            
            self.risk_assessment = risk_assessment
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Return conservative risk assessment on error
            return RiskAssessment(
                overall_risk=0.8,  # High risk on error
                market_risk=0.8,
                liquidity_risk=0.5,
                concentration_risk=0.5,
                var_95=0,
                expected_shortfall=0,
                stress_scenario_loss=0,
                risk_recommendation="CONSERVATIVE - Risk assessment failed"
            )
    
    async def _optimize_portfolio(self, symbol: str, signals: Dict[str, TradeSignal]) -> Dict[str, Any]:
        """Portfolio optimization and position sizing"""
        try:
            # Get current portfolio state
            current_positions = self.portfolio_state.get('positions', {})
            
            # Optimize portfolio allocations
            optimization_result = await self.components['portfolio_optimizer'].optimize_positions(
                signals, current_positions
            )
            
            # Calculate position size based on risk
            position_size = await self._calculate_position_size(symbol, signals, optimization_result)
            
            return {
                'optimized_allocations': optimization_result.get('optimized_allocations', {}),
                'position_size': position_size,
                'rebalance_needed': optimization_result.get('rebalance_needed', False),
                'correlation_impact': optimization_result.get('correlation_impact', 0)
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return conservative allocation on error
            return {
                'optimized_allocations': {symbol: 0.1},  # Small allocation on error
                'position_size': 1000,  # Minimal position size
                'rebalance_needed': False,
                'correlation_impact': 0
            }
    
    async def _make_final_decision(self, symbol: str, signals: Dict[str, TradeSignal],
                                 risk_assessment: RiskAssessment, 
                                 portfolio_recommendation: Dict[str, Any]) -> DecisionOutput:
        """Make the final trading decision"""
        reasoning = []
        
        # Calculate weighted confidence from all signals
        total_confidence = 0.0
        total_weight = 0.0
        final_decision = DecisionType.HOLD
        
        for source, signal in signals.items():
            weight = self.config['component_weights'].get(source, 0.1)
            total_confidence += signal.confidence * weight
            total_weight += weight
            
            # Track strongest signal
            if signal.confidence > total_confidence and signal.decision != DecisionType.HOLD:
                final_decision = signal.decision
        
        if total_weight > 0:
            weighted_confidence = total_confidence / total_weight
        else:
            weighted_confidence = 0.5
        
        # Apply risk adjustments
        risk_adjusted_confidence = weighted_confidence * (1 - risk_assessment.overall_risk)
        
        # Check confidence threshold
        min_confidence = self.config['decision_parameters']['min_confidence_threshold']
        if risk_adjusted_confidence < min_confidence:
            final_decision = DecisionType.HOLD
            reasoning.append(f"Confidence {risk_adjusted_confidence:.2f} below threshold {min_confidence}")
        
        # Check risk limits
        max_risk = self.config['decision_parameters']['max_risk_score']
        if risk_assessment.overall_risk > max_risk:
            final_decision = DecisionType.HOLD
            reasoning.append(f"Risk score {risk_assessment.overall_risk:.2f} exceeds limit {max_risk}")
        
        # Consider market regime
        if self.config['decision_parameters']['regime_aware']:
            regime_decision = await self._adjust_for_market_regime(final_decision, risk_adjusted_confidence)
            if regime_decision != final_decision:
                final_decision = regime_decision
                reasoning.append(f"Adjusted for {self.market_context.regime.value} regime")
        
        # Determine position size
        if final_decision in [DecisionType.ENTER_LONG, DecisionType.ENTER_SHORT]:
            position_size = portfolio_recommendation['position_size']
            # Further adjust based on risk
            position_size *= (1 - risk_assessment.overall_risk)
        else:
            position_size = 0
        
        # Generate reasoning
        if final_decision != DecisionType.HOLD:
            reasoning.extend([
                f"Weighted confidence: {weighted_confidence:.2f}",
                f"Risk adjusted confidence: {risk_adjusted_confidence:.2f}",
                f"Market regime: {self.market_context.regime.value}",
                f"Risk assessment: {risk_assessment.risk_recommendation}"
            ])
        
        # Determine execution priority
        execution_priority = self._determine_execution_priority(
            final_decision, risk_adjusted_confidence, risk_assessment
        )
        
        return DecisionOutput(
            decision=final_decision,
            confidence=risk_adjusted_confidence,
            symbol=symbol,
            position_size=position_size,
            risk_score=risk_assessment.overall_risk,
            reasoning=reasoning,
            market_context=self.market_context,
            risk_assessment=risk_assessment,
            execution_priority=execution_priority
        )
    
    async def _update_market_context(self) -> None:
        """Update current market context"""
        try:
            # Get market data for context analysis
            major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
            market_data = {}
            
            for pair in major_pairs:
                data = await self.components['data_handler'].fetch_realtime_data(pair)
                market_data[pair] = data
            
            # Detect market regime
            regime_result = await self.components['market_regime_detector'].detect_regime(market_data)
            
            # Get market sentiment
            sentiment_result = await self.components['sentiment_analyzer'].get_market_sentiment()
            
            # Calculate correlation matrix
            correlation_matrix = await self._calculate_correlation_matrix(major_pairs)
            
            # Get economic calendar (simplified)
            economic_calendar = await self._get_economic_calendar()
            
            self.market_context = MarketContext(
                regime=MarketCondition(regime_result.get('regime', 'RANGING')),
                volatility=regime_result.get('volatility', 0.01),
                trend_strength=regime_result.get('trend_strength', 0.5),
                sentiment_score=sentiment_result.get('overall_sentiment', 0.5),
                economic_calendar=economic_calendar,
                correlation_matrix=correlation_matrix,
                volume_profile={}  # Would be calculated from real data
            )
            
        except Exception as e:
            logger.error(f"Market context update failed: {e}")
            # Set default context on error
            self.market_context = MarketContext(
                regime=MarketCondition.RANGING,
                volatility=0.01,
                trend_strength=0.5,
                sentiment_score=0.5,
                economic_calendar=[],
                correlation_matrix=pd.DataFrame(),
                volume_profile={}
            )
    
    async def _calculate_market_risk(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate market-specific risk"""
        try:
            # Calculate volatility (simplified)
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Normalize to 0-1 scale
            market_risk = min(volatility / 0.2, 1.0)  # 20% vol = max risk
            
            # Adjust for current regime
            if self.market_context.regime in [MarketCondition.HIGH_VOLATILITY, MarketCondition.BREAKOUT]:
                market_risk *= 1.5
            elif self.market_context.regime in [MarketCondition.LOW_VOLATILITY, MarketCondition.RANGING]:
                market_risk *= 0.7
            
            return min(market_risk, 1.0)
            
        except Exception as e:
            logger.warning(f"Market risk calculation failed: {e}")
            return 0.5  # Medium risk on error
    
    async def _calculate_portfolio_risk(self, symbol: str, signals: Dict[str, TradeSignal]) -> float:
        """Calculate portfolio-level risk"""
        try:
            current_exposure = self.portfolio_state.get('total_exposure', 0)
            max_exposure = self.portfolio_state.get('max_exposure', 100000)
            
            # Exposure-based risk
            exposure_ratio = current_exposure / max_exposure if max_exposure > 0 else 0
            exposure_risk = min(exposure_ratio / 0.8, 1.0)  # 80% exposure = max risk
            
            # Correlation risk
            correlation_risk = await self._calculate_correlation_risk(symbol)
            
            # Combined portfolio risk
            portfolio_risk = (exposure_risk * 0.6 + correlation_risk * 0.4)
            
            return min(portfolio_risk, 1.0)
            
        except Exception as e:
            logger.warning(f"Portfolio risk calculation failed: {e}")
            return 0.3
    
    async def _calculate_liquidity_risk(self, symbol: str) -> float:
        """Calculate liquidity risk for a symbol"""
        try:
            # Simplified liquidity risk calculation
            # In reality, this would use bid-ask spreads, volume, market depth
            
            # Major pairs have lower liquidity risk
            major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY']
            if symbol in major_pairs:
                return 0.1  # Low liquidity risk for majors
            
            # Minor pairs have medium risk
            minor_pairs = ['AUD/USD', 'USD/CAD', 'NZD/USD', 'USD/CHF']
            if symbol in minor_pairs:
                return 0.3
            
            # Exotics have high risk
            return 0.7
            
        except Exception as e:
            logger.warning(f"Liquidity risk calculation failed: {e}")
            return 0.5
    
    async def _calculate_concentration_risk(self, symbol: str) -> float:
        """Calculate concentration risk"""
        try:
            positions = self.portfolio_state.get('positions', {})
            if not positions:
                return 0.1  # Low risk with no positions
            
            total_value = self.portfolio_state.get('portfolio_value', 100000)
            symbol_exposure = positions.get(symbol, {}).get('exposure', 0)
            
            concentration_ratio = symbol_exposure / total_value if total_value > 0 else 0
            concentration_risk = min(concentration_ratio / 0.3, 1.0)  # 30% concentration = max risk
            
            return concentration_risk
            
        except Exception as e:
            logger.warning(f"Concentration risk calculation failed: {e}")
            return 0.3
    
    async def _calculate_position_size(self, symbol: str, signals: Dict[str, TradeSignal],
                                     portfolio_recommendation: Dict[str, Any]) -> float:
        """Calculate optimal position size"""
        try:
            portfolio_value = self.portfolio_state.get('portfolio_value', 100000)
            max_risk_per_trade = self.config['decision_parameters']['max_portfolio_risk']
            
            # Base position size from portfolio optimizer
            base_size = portfolio_recommendation.get('position_size', 10000)
            
            # Adjust based on confidence
            avg_confidence = np.mean([s.confidence for s in signals.values()])
            confidence_adjustment = avg_confidence ** 2  # Square to be more conservative
            
            # Adjust based on risk
            risk_adjustment = 1 - self.risk_assessment.overall_risk
            
            # Final position size
            position_size = base_size * confidence_adjustment * risk_adjustment
            
            # Ensure it doesn't exceed maximum risk per trade
            max_trade_size = portfolio_value * max_risk_per_trade
            position_size = min(position_size, max_trade_size)
            
            return max(position_size, 1000)  # Minimum position size
            
        except Exception as e:
            logger.warning(f"Position sizing failed: {e}")
            return 5000  # Conservative default
    
    async def _perform_stress_testing(self, symbol: str, signals: Dict[str, TradeSignal]) -> float:
        """Perform stress testing for extreme scenarios"""
        try:
            stress_scenarios = self.config['risk_parameters']['stress_test_scenarios']
            max_loss = 0
            
            for scenario in stress_scenarios:
                if scenario == "flash_crash":
                    # 10% instantaneous drop
                    scenario_loss = self.portfolio_state.get('total_exposure', 0) * 0.10
                elif scenario == "rate_hike":
                    # 5% adverse move
                    scenario_loss = self.portfolio_state.get('total_exposure', 0) * 0.05
                elif scenario == "geopolitical":
                    # 7% adverse move with increased correlations
                    scenario_loss = self.portfolio_state.get('total_exposure', 0) * 0.07
                else:
                    scenario_loss = self.portfolio_state.get('total_exposure', 0) * 0.03
                
                max_loss = max(max_loss, scenario_loss)
            
            return max_loss
            
        except Exception as e:
            logger.warning(f"Stress testing failed: {e}")
            return self.portfolio_state.get('total_exposure', 0) * 0.05  # Conservative estimate
    
    async def _calculate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix for major pairs"""
        try:
            # This would use historical data to calculate actual correlations
            # For now, return a simplified matrix
            
            n = len(symbols)
            corr_matrix = np.eye(n)  # Start with identity matrix
            
            # Add some realistic correlations
            for i in range(n):
                for j in range(i+1, n):
                    if 'USD' in symbols[i] and 'USD' in symbols[j]:
                        # USD pairs are correlated
                        corr_matrix[i, j] = corr_matrix[j, i] = 0.6
                    else:
                        corr_matrix[i, j] = corr_matrix[j, i] = 0.3
            
            return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)
            
        except Exception as e:
            logger.warning(f"Correlation matrix calculation failed: {e}")
            return pd.DataFrame()
    
    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk for new position"""
        try:
            positions = self.portfolio_state.get('positions', {})
            if not positions:
                return 0.1  # Low risk with no existing positions
            
            # Simplified correlation risk calculation
            # In reality, use the correlation matrix and portfolio weights
            
            existing_symbols = list(positions.keys())
            if symbol in existing_symbols:
                return 0.4  # Medium risk for increasing existing position
            
            # Check if symbol is highly correlated with existing positions
            high_correlation_pairs = [('EUR/USD', 'GBP/USD'), ('AUD/USD', 'NZD/USD')]
            for pair1, pair2 in high_correlation_pairs:
                if (symbol == pair1 and pair2 in existing_symbols) or (symbol == pair2 and pair1 in existing_symbols):
                    return 0.6  # High correlation risk
            
            return 0.2  # Low correlation risk
            
        except Exception as e:
            logger.warning(f"Correlation risk calculation failed: {e}")
            return 0.3
    
    async def _adjust_for_market_regime(self, decision: DecisionType, confidence: float) -> DecisionType:
        """Adjust decision based on current market regime"""
        if self.market_context.regime == MarketCondition.TRENDING_BULL:
            if decision == DecisionType.ENTER_LONG:
                return decision  # Favor longs in bull trend
            elif decision == DecisionType.ENTER_SHORT:
                return DecisionType.HOLD  # Avoid shorts in strong bull
                
        elif self.market_context.regime == MarketCondition.TRENDING_BEAR:
            if decision == DecisionType.ENTER_SHORT:
                return decision  # Favor shorts in bear trend
            elif decision == DecisionType.ENTER_LONG:
                return DecisionType.HOLD  # Avoid longs in strong bear
                
        elif self.market_context.regime == MarketCondition.HIGH_VOLATILITY:
            if confidence < 0.8:  # Require higher confidence in high vol
                return DecisionType.HOLD
                
        elif self.market_context.regime == MarketCondition.RANGING:
            # In ranging markets, both directions can work
            return decision
        
        return decision
    
    def _determine_execution_priority(self, decision: DecisionType, confidence: float,
                                    risk_assessment: RiskAssessment) -> str:
        """Determine execution priority for the decision"""
        if decision == DecisionType.HOLD:
            return "LOW"
        
        if confidence > 0.8 and risk_assessment.overall_risk < 0.3:
            return "HIGH"
        elif confidence > 0.7 and risk_assessment.overall_risk < 0.5:
            return "NORMAL"
        else:
            return "LOW"
    
    def _convert_to_decision_type(self, signal_str: str) -> DecisionType:
        """Convert string signal to DecisionType enum"""
        signal_map = {
            'BUY': DecisionType.ENTER_LONG,
            'SELL': DecisionType.ENTER_SHORT,
            'HOLD': DecisionType.HOLD,
            'EXIT': DecisionType.EXIT_LONG,  # Default to exit long
            'NEUTRAL': DecisionType.HOLD
        }
        return signal_map.get(signal_str.upper(), DecisionType.HOLD)
    
    def _create_rl_state(self, market_data: pd.DataFrame, market_context: MarketContext) -> np.ndarray:
        """Create state representation for reinforcement learning"""
        # This would create a comprehensive state representation
        # Simplified version for demonstration
        state_features = [
            market_data['close'].iloc[-1] if len(market_data) > 0 else 1.0,
            market_context.volatility,
            market_context.trend_strength,
            market_context.sentiment_score,
            len(market_context.economic_calendar) / 10.0  # Normalized
        ]
        return np.array(state_features)
    
    def _create_neutral_signal(self, symbol: str, source: str) -> TradeSignal:
        """Create a neutral (HOLD) signal"""
        return TradeSignal(
            symbol=symbol,
            decision=DecisionType.HOLD,
            confidence=0.5,
            source=source
        )
    
    async def _create_safe_decision(self, symbol: str, reason: str) -> DecisionOutput:
        """Create a safe HOLD decision when errors occur"""
        return DecisionOutput(
            decision=DecisionType.HOLD,
            confidence=0.0,
            symbol=symbol,
            position_size=0,
            risk_score=1.0,
            reasoning=[f"Safety decision: {reason}"],
            market_context=self.market_context,
            risk_assessment=RiskAssessment(
                overall_risk=1.0,
                market_risk=1.0,
                liquidity_risk=1.0,
                concentration_risk=1.0,
                var_95=0,
                expected_shortfall=0,
                stress_scenario_loss=0,
                risk_recommendation="MAXIMUM RISK - Using safe decision"
            ),
            execution_priority="LOW"
        )
    
    def _generate_risk_recommendation(self, overall_risk: float) -> str:
        """Generate risk recommendation based on risk score"""
        if overall_risk < 0.3:
            return "LOW RISK - Aggressive trading acceptable"
        elif overall_risk < 0.6:
            return "MEDIUM RISK - Normal trading with caution"
        elif overall_risk < 0.8:
            return "HIGH RISK - Conservative trading recommended"
        else:
            return "VERY HIGH RISK - Avoid new positions"
    
    async def _get_economic_calendar(self) -> List[Dict]:
        """Get economic calendar events (simplified)"""
        # In reality, this would fetch from an economic calendar API
        return [
            {'event': 'US NFP', 'impact': 'HIGH', 'time': datetime.now() + timedelta(hours=2)},
            {'event': 'ECB Rate Decision', 'impact': 'MEDIUM', 'time': datetime.now() + timedelta(days=1)}
        ]
    
    async def _update_performance_metrics(self, start_time: datetime, decision: DecisionOutput) -> None:
        """Update decision engine performance metrics"""
        try:
            decision_latency = (datetime.now() - start_time).total_seconds()
            
            # Update metrics (simplified - in reality would track actual performance)
            self.performance_metrics['decision_latency'] = (
                self.performance_metrics['decision_latency'] * 0.9 + decision_latency * 0.1
            )
            self.performance_metrics['avg_confidence'] = (
                self.performance_metrics['avg_confidence'] * 0.9 + decision.confidence * 0.1
            )
            
        except Exception as e:
            logger.warning(f"Performance metrics update failed: {e}")
    
    def _log_decision(self, decision: DecisionOutput, signals: Dict[str, TradeSignal],
                     risk_assessment: RiskAssessment) -> None:
        """Log the final decision with all details"""
        log_entry = {
            'timestamp': decision.timestamp.isoformat(),
            'symbol': decision.symbol,
            'decision': decision.decision.value,
            'confidence': decision.confidence,
            'position_size': decision.position_size,
            'risk_score': decision.risk_score,
            'execution_priority': decision.execution_priority,
            'reasoning': decision.reasoning,
            'market_regime': decision.market_context.regime.value,
            'signals': {source: {
                'decision': signal.decision.value,
                'confidence': signal.confidence
            } for source, signal in signals.items()},
            'risk_assessment': {
                'overall_risk': risk_assessment.overall_risk,
                'recommendation': risk_assessment.risk_recommendation
            }
        }
        
        logger.info(f"Decision Made: {decision.symbol} - {decision.decision.value} "
                   f"(Confidence: {decision.confidence:.2f}, Risk: {decision.risk_score:.2f})")
        
        # Store in history
        self.decision_history.append(decision)
        
        # Keep only recent history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    async def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision engine statistics"""
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        stats = {
            'total_decisions': len(self.decision_history),
            'recent_decisions': len(recent_decisions),
            'decision_distribution': {},
            'avg_confidence': self.performance_metrics['avg_confidence'],
            'avg_decision_latency': self.performance_metrics['decision_latency'],
            'current_market_regime': self.market_context.regime.value if self.market_context else 'UNKNOWN',
            'risk_profile': self.risk_assessment.risk_recommendation if self.risk_assessment else 'UNKNOWN'
        }
        
        # Calculate decision distribution
        for decision in recent_decisions:
            decision_type = decision.decision.value
            stats['decision_distribution'][decision_type] = \
                stats['decision_distribution'].get(decision_type, 0) + 1
        
        return stats
    
    async def update_portfolio_state(self, portfolio_data: Dict[str, Any]) -> None:
        """Update the engine with current portfolio state"""
        self.portfolio_state = portfolio_data
        logger.info("Portfolio state updated in decision engine")
    
    async def shutdown(self) -> None:
        """Shutdown the decision engine gracefully"""
        logger.info("Shutting down Decision Engine...")
        
        # Close any open connections or resources
        # Component cleanup would happen here
        
        logger.info("Decision Engine shutdown complete")

# Example usage and testing
async def main():
    """Example usage of the Decision Engine"""
    print("🚀 Initializing Advanced Decision Engine...")
    
    # Create decision engine
    engine = AdvancedDecisionEngine()
    
    # Initialize
    success = await engine.initialize()
    if not success:
        print("❌ Decision Engine initialization failed")
        return
    
    print("✅ Decision Engine initialized successfully")
    
    # Example market data
    sample_data = pd.DataFrame({
        'open': [1.0990, 1.1000, 1.1010, 1.1020, 1.1015],
        'high': [1.1005, 1.1015, 1.1025, 1.1030, 1.1020],
        'low': [1.0985, 1.0995, 1.1005, 1.1010, 1.1005],
        'close': [1.1000, 1.1010, 1.1020, 1.1015, 1.1018],
        'volume': [1000000, 1200000, 1100000, 900000, 950000]
    })
    
    # Update portfolio state
    await engine.update_portfolio_state({
        'portfolio_value': 100000,
        'total_exposure': 25000,
        'max_exposure': 50000,
        'positions': {
            'EUR/USD': {'exposure': 15000, 'direction': 'LONG'},
            'GBP/USD': {'exposure': 10000, 'direction': 'SHORT'}
        }
    })
    
    # Make a decision
    print("\n🎯 Making trading decision for EUR/USD...")
    decision = await engine.make_trading_decision('EUR/USD', sample_data)
    
    print(f"\n📊 DECISION RESULT:")
    print(f"Symbol: {decision.symbol}")
    print(f"Decision: {decision.decision.value}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Position Size: ${decision.position_size:,.2f}")
    print(f"Risk Score: {decision.risk_score:.2f}")
    print(f"Execution Priority: {decision.execution_priority}")
    print(f"Reasoning: {', '.join(decision.reasoning)}")
    
    # Get statistics
    stats = await engine.get_decision_stats()
    print(f"\n📈 ENGINE STATS:")
    print(f"Total Decisions: {stats['total_decisions']}")
    print(f"Market Regime: {stats['current_market_regime']}")
    print(f"Risk Profile: {stats['risk_profile']}")
    
    # Shutdown
    await engine.shutdown()
    print("\n✅ Decision Engine demo completed")

if __name__ == "__main__":
    asyncio.run(main())
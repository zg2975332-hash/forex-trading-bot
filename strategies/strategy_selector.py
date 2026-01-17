"""
Advanced Strategy Selector for FOREX TRADING BOT
Dynamic strategy selection based on market conditions and performance
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import json
from pathlib import Path
from scipy import stats
import talib
import hashlib
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    GRID_TRADING = "grid_trading"
    HEDGING = "hedging"

class MarketRegime(Enum):
    STRONG_TREND_UP = "strong_trend_up"
    TREND_UP = "trend_up"
    RANGING = "ranging"
    TREND_DOWN = "trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"

class SelectionMethod(Enum):
    PERFORMANCE_BASED = "performance_based"
    MARKET_REGIME = "market_regime"
    ENSEMBLE = "ensemble"
    ML_PREDICTION = "ml_prediction"
    HYBRID = "hybrid"

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_type: StrategyType
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    recent_performance: List[bool]
    last_used: datetime
    confidence_score: float

@dataclass
class MarketConditions:
    """Current market conditions analysis"""
    regime: MarketRegime
    trend_strength: float
    volatility: float
    volume_profile: str
    support_resistance_strength: float
    momentum_score: float
    market_sentiment: float
    timestamp: datetime

@dataclass
class StrategySelection:
    """Strategy selection result"""
    primary_strategy: StrategyType
    secondary_strategy: StrategyType
    confidence: float
    selection_method: SelectionMethod
    market_conditions: MarketConditions
    strategy_weights: Dict[StrategyType, float]
    reasoning: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyConfig:
    """Configuration for strategy selector"""
    
    # Strategy preferences
    enabled_strategies: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.TREND_FOLLOWING,
        StrategyType.MEAN_REVERSION,
        StrategyType.BREAKOUT,
        StrategyType.MOMENTUM
    ])
    
    # Performance thresholds
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.1
    max_drawdown: float = 0.1
    min_trades_for_evaluation: int = 10
    
    # Market regime mapping
    regime_strategy_mapping: Dict[MarketRegime, List[StrategyType]] = field(default_factory=lambda: {
        MarketRegime.STRONG_TREND_UP: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
        MarketRegime.TREND_UP: [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT],
        MarketRegime.RANGING: [StrategyType.MEAN_REVERSION, StrategyType.GRID_TRADING],
        MarketRegime.TREND_DOWN: [StrategyType.TREND_FOLLOWING, StrategyType.BREAKOUT],
        MarketRegime.STRONG_TREND_DOWN: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
        MarketRegime.HIGH_VOLATILITY: [StrategyType.BREAKOUT, StrategyType.SCALPING],
        MarketRegime.LOW_VOLATILITY: [StrategyType.MEAN_REVERSION, StrategyType.GRID_TRADING],
        MarketRegime.BREAKOUT: [StrategyType.BREAKOUT, StrategyType.MOMENTUM]
    })
    
    # Selection parameters
    selection_method: SelectionMethod = SelectionMethod.HYBRID
    performance_lookback: int = 100
    regime_confidence_threshold: float = 0.7
    enable_ml_selection: bool = True
    ml_retraining_interval: int = 1000
    
    # Risk management
    max_strategies_active: int = 3
    strategy_switch_cooldown: int = 300  # seconds
    performance_decay_factor: float = 0.95

@dataclass
class MLModelConfig:
    """Machine learning model configuration"""
    model_type: str = "random_forest"
    features: List[str] = field(default_factory=lambda: [
        'trend_strength', 'volatility', 'volume_profile', 'rsi', 'macd', 
        'bb_width', 'atr_ratio', 'adx', 'market_sentiment'
    ])
    target_lookforward: int = 10
    training_period: int = 1000
    retrain_interval: int = 500

class AdvancedStrategySelector:
    """
    Advanced Strategy Selector with Dynamic Market Adaptation
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        
        # Strategy performance tracking
        self.strategy_performance: Dict[StrategyType, StrategyPerformance] = {}
        self.trade_history: deque = deque(maxlen=1000)
        self.performance_history: Dict[StrategyType, List[Dict]] = defaultdict(list)
        
        # Market analysis
        self.market_conditions_history: deque = deque(maxlen=500)
        self.current_market_conditions: Optional[MarketConditions] = None
        
        # Selection state
        self.current_strategies: List[StrategyType] = []
        self.strategy_switch_times: Dict[StrategyType, datetime] = {}
        self.selection_history: deque = deque(maxlen=200)
        
        # ML model for strategy prediction
        self.ml_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.ml_training_data: List[Dict] = []
        self.ml_config = MLModelConfig()
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize strategies
        self._initialize_strategies()
        
        logger.info("AdvancedStrategySelector initialized successfully")
    
    def _initialize_strategies(self) -> None:
        """Initialize strategy performance tracking"""
        for strategy in self.config.enabled_strategies:
            self.strategy_performance[strategy] = StrategyPerformance(
                strategy_type=strategy,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                profit_factor=1.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_trade_duration=0.0,
                recent_performance=[],
                last_used=datetime.now(),
                confidence_score=0.5
            )
    
    async def analyze_market_conditions(self, market_data: pd.DataFrame, 
                                      additional_context: Dict[str, Any] = None) -> MarketConditions:
        """Analyze current market conditions"""
        try:
            if len(market_data) < 50:
                raise ValueError("Insufficient market data for analysis")
            
            # Calculate technical indicators
            trend_strength = self._calculate_trend_strength(market_data)
            volatility = self._calculate_volatility(market_data)
            volume_profile = self._analyze_volume_profile(market_data)
            support_resistance_strength = self._analyze_support_resistance(market_data)
            momentum_score = self._calculate_momentum_score(market_data)
            
            # Get market sentiment from context
            market_sentiment = additional_context.get('market_sentiment', 0.5) if additional_context else 0.5
            
            # Determine market regime
            regime = self._determine_market_regime(
                trend_strength, volatility, volume_profile, momentum_score
            )
            
            market_conditions = MarketConditions(
                regime=regime,
                trend_strength=trend_strength,
                volatility=volatility,
                volume_profile=volume_profile,
                support_resistance_strength=support_resistance_strength,
                momentum_score=momentum_score,
                market_sentiment=market_sentiment,
                timestamp=datetime.now()
            )
            
            with self._lock:
                self.current_market_conditions = market_conditions
                self.market_conditions_history.append(market_conditions)
            
            logger.debug(f"Market conditions analyzed: {regime.value}")
            return market_conditions
            
        except Exception as e:
            logger.error(f"Market conditions analysis failed: {e}")
            # Return default conditions on error
            return MarketConditions(
                regime=MarketRegime.RANGING,
                trend_strength=0.0,
                volatility=0.0,
                volume_profile="normal",
                support_resistance_strength=0.5,
                momentum_score=0.0,
                market_sentiment=0.5,
                timestamp=datetime.now()
            )
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple methods"""
        try:
            closes = market_data['close'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            # ADX for trend strength
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0.0
            
            # Linear regression slope
            x = np.arange(len(closes))
            slope, _, r_value, _, _ = stats.linregress(x, closes)
            regression_strength = abs(slope) * r_value ** 2
            
            # Moving average alignment
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            ma_alignment = 1.0 if (sma_20[-1] > sma_50[-1] and sma_50[-1] > sma_50[-10]) else 0.0
            
            # Combine trend strength indicators
            trend_strength = (
                min(1.0, current_adx / 50.0) * 0.4 +
                min(1.0, regression_strength * 100) * 0.3 +
                ma_alignment * 0.3
            )
            
            # Add direction (positive for uptrend, negative for downtrend)
            price_direction = 1.0 if closes[-1] > closes[-10] else -1.0
            trend_strength *= price_direction
            
            return max(-1.0, min(1.0, trend_strength))
            
        except Exception as e:
            logger.warning(f"Trend strength calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate market volatility"""
        try:
            closes = market_data['close'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            # ATR-based volatility
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0.0
            atr_volatility = current_atr / closes[-1] if closes[-1] > 0 else 0.0
            
            # Bollinger Band width
            upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
            bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] > 0 else 0.0
            
            # Historical volatility
            returns = np.diff(np.log(closes))
            hist_volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
            
            # Normalize volatility score (0-1 range)
            volatility_score = min(1.0, (atr_volatility + bb_width + hist_volatility) / 0.1)
            
            return volatility_score
            
        except Exception as e:
            logger.warning(f"Volatility calculation failed: {e}")
            return 0.5
    
    def _analyze_volume_profile(self, market_data: pd.DataFrame) -> str:
        """Analyze volume profile"""
        try:
            volumes = market_data['volume'].values
            price_changes = market_data['close'].pct_change().values
            
            if len(volumes) < 20:
                return "normal"
            
            # Volume trend
            volume_sma = talib.SMA(volumes, timeperiod=20)
            current_volume_ratio = volumes[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1.0
            
            # Volume-price correlation
            valid_indices = ~np.isnan(price_changes) & ~np.isnan(volumes)
            if np.sum(valid_indices) > 10:
                correlation = np.corrcoef(price_changes[valid_indices], volumes[valid_indices])[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Determine volume profile
            if current_volume_ratio > 1.5 and correlation > 0.3:
                return "high_accumulation"
            elif current_volume_ratio > 1.5 and correlation < -0.3:
                return "high_distribution"
            elif current_volume_ratio > 1.2:
                return "high"
            elif current_volume_ratio < 0.8:
                return "low"
            else:
                return "normal"
                
        except Exception as e:
            logger.warning(f"Volume profile analysis failed: {e}")
            return "normal"
    
    def _analyze_support_resistance(self, market_data: pd.DataFrame) -> float:
        """Analyze support/resistance strength"""
        try:
            highs = market_data['high'].values
            lows = market_data['low'].values
            closes = market_data['close'].values
            
            # Find recent swing points
            swing_highs = []
            swing_lows = []
            
            for i in range(5, len(highs) - 5):
                if all(highs[i] > highs[i-j] for j in range(1, 6)) and \
                   all(highs[i] > highs[i+j] for j in range(1, 6)):
                    swing_highs.append(highs[i])
                
                if all(lows[i] < lows[i-j] for j in range(1, 6)) and \
                   all(lows[i] < lows[i+j] for j in range(1, 6)):
                    swing_lows.append(lows[i])
            
            # Calculate strength based on proximity and recent touches
            current_price = closes[-1]
            sr_strength = 0.0
            
            for level in swing_highs[-5:] + swing_lows[-5:]:
                distance_ratio = abs(level - current_price) / current_price
                if distance_ratio < 0.01:  # Within 1%
                    sr_strength += 0.2
            
            return min(1.0, sr_strength)
            
        except Exception as e:
            logger.warning(f"Support/resistance analysis failed: {e}")
            return 0.5
    
    def _calculate_momentum_score(self, market_data: pd.DataFrame) -> float:
        """Calculate market momentum score"""
        try:
            closes = market_data['close'].values
            
            # RSI momentum
            rsi = talib.RSI(closes, timeperiod=14)
            rsi_momentum = (rsi[-1] - 50) / 50 if not np.isnan(rsi[-1]) else 0.0
            
            # MACD momentum
            macd, macd_signal, _ = talib.MACD(closes)
            macd_momentum = (macd[-1] - macd_signal[-1]) / 0.01 if not np.isnan(macd[-1]) else 0.0
            
            # Price momentum
            price_momentum = (closes[-1] / closes[-5] - 1) * 100
            
            # Combined momentum score
            momentum_score = (
                rsi_momentum * 0.3 +
                np.tanh(macd_momentum) * 0.4 +
                np.tanh(price_momentum) * 0.3
            )
            
            return max(-1.0, min(1.0, momentum_score))
            
        except Exception as e:
            logger.warning(f"Momentum calculation failed: {e}")
            return 0.0
    
    def _determine_market_regime(self, trend_strength: float, volatility: float, 
                               volume_profile: str, momentum_score: float) -> MarketRegime:
        """Determine current market regime"""
        try:
            abs_trend_strength = abs(trend_strength)
            
            if volatility > 0.8:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.3:
                return MarketRegime.LOW_VOLATILITY
            elif abs_trend_strength > 0.7:
                if trend_strength > 0:
                    return MarketRegime.STRONG_TREND_UP
                else:
                    return MarketRegime.STRONG_TREND_DOWN
            elif abs_trend_strength > 0.3:
                if trend_strength > 0:
                    return MarketRegime.TREND_UP
                else:
                    return MarketRegime.TREND_DOWN
            elif volume_profile in ["high_accumulation", "high_distribution"]:
                return MarketRegime.BREAKOUT
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            logger.warning(f"Market regime determination failed: {e}")
            return MarketRegime.RANGING
    
    async def update_strategy_performance(self, strategy_type: StrategyType, 
                                       trade_result: Dict[str, Any]) -> None:
        """Update strategy performance metrics"""
        try:
            with self._lock:
                if strategy_type not in self.strategy_performance:
                    self._initialize_strategy(strategy_type)
                
                performance = self.strategy_performance[strategy_type]
                
                # Update basic metrics
                performance.total_trades += 1
                
                if trade_result.get('profit', 0) > 0:
                    performance.winning_trades += 1
                else:
                    performance.losing_trades += 1
                
                # Update win rate
                performance.win_rate = performance.winning_trades / performance.total_trades
                
                # Update profit factor
                total_profit = trade_result.get('total_profit', 0)
                total_loss = trade_result.get('total_loss', 0)
                if total_loss > 0:
                    performance.profit_factor = total_profit / total_loss
                
                # Update Sharpe ratio (simplified)
                returns = [t.get('return', 0) for t in self.performance_history[strategy_type][-20:]]
                if len(returns) > 1:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        performance.sharpe_ratio = avg_return / std_return
                
                # Update max drawdown
                current_drawdown = trade_result.get('drawdown', 0)
                performance.max_drawdown = max(performance.max_drawdown, current_drawdown)
                
                # Update recent performance
                is_win = trade_result.get('profit', 0) > 0
                performance.recent_performance.append(is_win)
                if len(performance.recent_performance) > 20:
                    performance.recent_performance.pop(0)
                
                # Update last used time
                performance.last_used = datetime.now()
                
                # Update confidence score
                performance.confidence_score = self._calculate_confidence_score(performance)
                
                # Store performance history
                self.performance_history[strategy_type].append({
                    'timestamp': datetime.now(),
                    'profit': trade_result.get('profit', 0),
                    'return': trade_result.get('return', 0),
                    'duration': trade_result.get('duration', 0)
                })
                
                logger.debug(f"Updated performance for {strategy_type.value}: Win rate {performance.win_rate:.3f}")
                
        except Exception as e:
            logger.error(f"Strategy performance update failed: {e}")
    
    def _calculate_confidence_score(self, performance: StrategyPerformance) -> float:
        """Calculate strategy confidence score"""
        try:
            confidence = 0.0
            
            # Win rate component
            win_rate_score = max(0.0, (performance.win_rate - 0.4) / 0.6)  # Normalize 0.4-1.0 to 0-1
            
            # Profit factor component
            profit_factor_score = min(1.0, performance.profit_factor / 2.0)
            
            # Recent performance component
            if performance.recent_performance:
                recent_win_rate = sum(performance.recent_performance) / len(performance.recent_performance)
                recent_performance_score = max(0.0, (recent_win_rate - 0.4) / 0.6)
            else:
                recent_performance_score = 0.5
            
            # Sample size component
            sample_size_score = min(1.0, performance.total_trades / 50.0)
            
            # Drawdown penalty
            drawdown_penalty = max(0.0, 1.0 - (performance.max_drawdown / self.config.max_drawdown))
            
            # Combined confidence score
            confidence = (
                win_rate_score * 0.3 +
                profit_factor_score * 0.25 +
                recent_performance_score * 0.2 +
                sample_size_score * 0.15 +
                drawdown_penalty * 0.1
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence score calculation failed: {e}")
            return 0.5
    
    async def select_strategies(self, market_data: pd.DataFrame,
                              additional_context: Dict[str, Any] = None) -> StrategySelection:
        """Select optimal strategies based on current conditions"""
        try:
            # Analyze current market conditions
            market_conditions = await self.analyze_market_conditions(market_data, additional_context)
            
            # Apply selection method
            if self.config.selection_method == SelectionMethod.PERFORMANCE_BASED:
                selection = await self._performance_based_selection(market_conditions)
            elif self.config.selection_method == SelectionMethod.MARKET_REGIME:
                selection = await self._regime_based_selection(market_conditions)
            elif self.config.selection_method == SelectionMethod.ML_PREDICTION:
                selection = await self._ml_based_selection(market_conditions, market_data)
            elif self.config.selection_method == SelectionMethod.ENSEMBLE:
                selection = await self._ensemble_selection(market_conditions, market_data)
            else:  # HYBRID
                selection = await self._hybrid_selection(market_conditions, market_data)
            
            # Update current strategies
            with self._lock:
                self.current_strategies = [selection.primary_strategy, selection.secondary_strategy]
                self.selection_history.append(selection)
            
            logger.info(f"Strategy selection: {selection.primary_strategy.value} "
                       f"(confidence: {selection.confidence:.3f})")
            
            return selection
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return await self._create_fallback_selection()
    
    async def _performance_based_selection(self, market_conditions: MarketConditions) -> StrategySelection:
        """Select strategies based on historical performance"""
        try:
            strategy_scores = {}
            reasoning = []
            
            for strategy_type, performance in self.strategy_performance.items():
                if performance.total_trades < self.config.min_trades_for_evaluation:
                    # Use default score for untested strategies
                    strategy_scores[strategy_type] = 0.5
                    continue
                
                # Check performance thresholds
                if (performance.win_rate < self.config.min_win_rate or 
                    performance.profit_factor < self.config.min_profit_factor):
                    strategy_scores[strategy_type] = 0.0
                    reasoning.append(f"{strategy_type.value} below performance thresholds")
                    continue
                
                # Base score on confidence
                base_score = performance.confidence_score
                
                # Apply performance decay for recently used strategies
                time_since_last_use = (datetime.now() - performance.last_used).total_seconds()
                if time_since_last_use < 3600:  # 1 hour
                    decay_factor = 0.8
                else:
                    decay_factor = 1.0
                
                strategy_scores[strategy_type] = base_score * decay_factor
            
            # Select top strategies
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            valid_strategies = [s for s in sorted_strategies if s[1] > 0]
            
            if len(valid_strategies) >= 2:
                primary_strategy = valid_strategies[0][0]
                secondary_strategy = valid_strategies[1][0]
                confidence = valid_strategies[0][1]
            else:
                # Fallback to default strategies
                primary_strategy = StrategyType.TREND_FOLLOWING
                secondary_strategy = StrategyType.MEAN_REVERSION
                confidence = 0.5
                reasoning.append("Using fallback strategies due to insufficient performance data")
            
            # Create strategy weights
            strategy_weights = {}
            total_score = sum(score for _, score in valid_strategies[:3])
            for strategy, score in valid_strategies[:3]:
                strategy_weights[strategy] = score / total_score if total_score > 0 else 0.33
            
            return StrategySelection(
                primary_strategy=primary_strategy,
                secondary_strategy=secondary_strategy,
                confidence=confidence,
                selection_method=SelectionMethod.PERFORMANCE_BASED,
                market_conditions=market_conditions,
                strategy_weights=strategy_weights,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Performance-based selection failed: {e}")
            raise
    
    async def _regime_based_selection(self, market_conditions: MarketConditions) -> StrategySelection:
        """Select strategies based on market regime"""
        try:
            regime = market_conditions.regime
            recommended_strategies = self.config.regime_strategy_mapping.get(regime, [])
            
            reasoning = [f"Market regime: {regime.value}"]
            
            # Filter enabled strategies
            available_strategies = [s for s in recommended_strategies 
                                  if s in self.config.enabled_strategies]
            
            if not available_strategies:
                # Fallback to all enabled strategies
                available_strategies = self.config.enabled_strategies
                reasoning.append("No regime-specific strategies available, using all enabled")
            
            # Score strategies based on regime alignment and performance
            strategy_scores = {}
            for strategy in available_strategies:
                performance = self.strategy_performance.get(strategy)
                if performance and performance.total_trades >= self.config.min_trades_for_evaluation:
                    base_score = performance.confidence_score
                else:
                    base_score = 0.5
                
                # Boost score for regime alignment
                regime_boost = 0.3 if strategy in recommended_strategies else 0.0
                strategy_scores[strategy] = base_score + regime_boost
            
            # Select top strategies
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_strategies) >= 2:
                primary_strategy = sorted_strategies[0][0]
                secondary_strategy = sorted_strategies[1][0]
                confidence = sorted_strategies[0][1]
            else:
                primary_strategy = available_strategies[0] if available_strategies else StrategyType.TREND_FOLLOWING
                secondary_strategy = available_strategies[1] if len(available_strategies) > 1 else primary_strategy
                confidence = 0.5
            
            # Create strategy weights
            strategy_weights = {primary_strategy: 0.6, secondary_strategy: 0.4}
            
            return StrategySelection(
                primary_strategy=primary_strategy,
                secondary_strategy=secondary_strategy,
                confidence=confidence,
                selection_method=SelectionMethod.MARKET_REGIME,
                market_conditions=market_conditions,
                strategy_weights=strategy_weights,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Regime-based selection failed: {e}")
            raise
    
    async def _ml_based_selection(self, market_conditions: MarketConditions,
                                market_data: pd.DataFrame) -> StrategySelection:
        """Select strategies using machine learning prediction"""
        try:
            if not self.ml_model:
                await self._train_ml_model()
                if not self.ml_model:
                    logger.warning("ML model not available, falling back to hybrid selection")
                    return await self._hybrid_selection(market_conditions, market_data)
            
            # Prepare features for prediction
            features = self._extract_ml_features(market_conditions, market_data)
            features_scaled = self.scaler.transform([features])
            
            # Predict strategy performance
            predictions = self.ml_model.predict_proba(features_scaled)[0]
            
            # Map predictions to strategies
            strategy_scores = {}
            for i, strategy in enumerate(self.config.enabled_strategies):
                if i < len(predictions):
                    strategy_scores[strategy] = predictions[i]
                else:
                    strategy_scores[strategy] = 0.0
            
            # Select top strategies
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            
            primary_strategy = sorted_strategies[0][0]
            secondary_strategy = sorted_strategies[1][0]
            confidence = sorted_strategies[0][1]
            
            reasoning = [f"ML prediction confidence: {confidence:.3f}"]
            
            # Create strategy weights based on prediction scores
            total_score = sum(score for _, score in sorted_strategies[:3])
            strategy_weights = {}
            for strategy, score in sorted_strategies[:3]:
                strategy_weights[strategy] = score / total_score if total_score > 0 else 0.33
            
            return StrategySelection(
                primary_strategy=primary_strategy,
                secondary_strategy=secondary_strategy,
                confidence=confidence,
                selection_method=SelectionMethod.ML_PREDICTION,
                market_conditions=market_conditions,
                strategy_weights=strategy_weights,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"ML-based selection failed: {e}")
            return await self._hybrid_selection(market_conditions, market_data)
    
    async def _ensemble_selection(self, market_conditions: MarketConditions,
                                market_data: pd.DataFrame) -> StrategySelection:
        """Use ensemble of selection methods"""
        try:
            # Get selections from different methods
            performance_selection = await self._performance_based_selection(market_conditions)
            regime_selection = await self._regime_based_selection(market_conditions)
            ml_selection = await self._ml_based_selection(market_conditions, market_data)
            
            # Combine strategy scores
            strategy_scores = defaultdict(float)
            method_weights = {'performance': 0.4, 'regime': 0.3, 'ml': 0.3}
            
            # Performance method scores
            for strategy, weight in performance_selection.strategy_weights.items():
                strategy_scores[strategy] += weight * method_weights['performance']
            
            # Regime method scores
            for strategy, weight in regime_selection.strategy_weights.items():
                strategy_scores[strategy] += weight * method_weights['regime']
            
            # ML method scores
            for strategy, weight in ml_selection.strategy_weights.items():
                strategy_scores[strategy] += weight * method_weights['ml']
            
            # Select top strategies
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            
            primary_strategy = sorted_strategies[0][0]
            secondary_strategy = sorted_strategies[1][0]
            confidence = sorted_strategies[0][1]
            
            reasoning = [
                "Ensemble selection combining performance, regime, and ML methods",
                f"Performance confidence: {performance_selection.confidence:.3f}",
                f"Regime confidence: {regime_selection.confidence:.3f}",
                f"ML confidence: {ml_selection.confidence:.3f}"
            ]
            
            return StrategySelection(
                primary_strategy=primary_strategy,
                secondary_strategy=secondary_strategy,
                confidence=confidence,
                selection_method=SelectionMethod.ENSEMBLE,
                market_conditions=market_conditions,
                strategy_weights=dict(strategy_scores),
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Ensemble selection failed: {e}")
            return await self._hybrid_selection(market_conditions, market_data)
    
    async def _hybrid_selection(self, market_conditions: MarketConditions,
                              market_data: pd.DataFrame) -> StrategySelection:
        """Hybrid selection with fallback mechanisms"""
        try:
            # Try ML first if available
            if self.ml_model and len(self.ml_training_data) > 100:
                ml_selection = await self._ml_based_selection(market_conditions, market_data)
                if ml_selection.confidence > 0.6:
                    return ml_selection
            
            # Fallback to regime-based with performance filtering
            regime_selection = await self._regime_based_selection(market_conditions)
            
            # Filter by performance if available
            valid_strategies = []
            for strategy in [regime_selection.primary_strategy, regime_selection.secondary_strategy]:
                performance = self.strategy_performance.get(strategy)
                if (not performance or 
                    performance.total_trades < self.config.min_trades_for_evaluation or
                    (performance.win_rate >= self.config.min_win_rate and 
                     performance.profit_factor >= self.config.min_profit_factor)):
                    valid_strategies.append(strategy)
            
            if len(valid_strategies) >= 2:
                primary_strategy = valid_strategies[0]
                secondary_strategy = valid_strategies[1]
                confidence = regime_selection.confidence
            else:
                # Ultimate fallback
                primary_strategy = StrategyType.TREND_FOLLOWING
                secondary_strategy = StrategyType.MEAN_REVERSION
                confidence = 0.5
            
            reasoning = [
                "Hybrid selection with performance filtering",
                f"Original regime selection: {regime_selection.primary_strategy.value}",
                f"Filtered to {len(valid_strategies)} valid strategies"
            ]
            
            return StrategySelection(
                primary_strategy=primary_strategy,
                secondary_strategy=secondary_strategy,
                confidence=confidence,
                selection_method=SelectionMethod.HYBRID,
                market_conditions=market_conditions,
                strategy_weights={primary_strategy: 0.6, secondary_strategy: 0.4},
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Hybrid selection failed: {e}")
            return await self._create_fallback_selection()
    
    async def _create_fallback_selection(self) -> StrategySelection:
        """Create fallback strategy selection"""
        return StrategySelection(
            primary_strategy=StrategyType.TREND_FOLLOWING,
            secondary_strategy=StrategyType.MEAN_REVERSION,
            confidence=0.3,
            selection_method=SelectionMethod.PERFORMANCE_BASED,
            market_conditions=MarketConditions(
                regime=MarketRegime.RANGING,
                trend_strength=0.0,
                volatility=0.5,
                volume_profile="normal",
                support_resistance_strength=0.5,
                momentum_score=0.0,
                market_sentiment=0.5,
                timestamp=datetime.now()
            ),
            strategy_weights={
                StrategyType.TREND_FOLLOWING: 0.6,
                StrategyType.MEAN_REVERSION: 0.4
            },
            reasoning=["Fallback selection due to selection failure"],
            timestamp=datetime.now()
        )
    
    def _extract_ml_features(self, market_conditions: MarketConditions, 
                           market_data: pd.DataFrame) -> List[float]:
        """Extract features for ML model"""
        try:
            features = []
            
            # Market condition features
            features.append(market_conditions.trend_strength)
            features.append(market_conditions.volatility)
            features.append(market_conditions.momentum_score)
            features.append(market_conditions.market_sentiment)
            features.append(market_conditions.support_resistance_strength)
            
            # Technical indicators from market data
            closes = market_data['close'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)
            features.append(rsi[-1] if not np.isnan(rsi[-1]) else 50.0)
            
            # MACD
            macd, macd_signal, _ = talib.MACD(closes)
            features.append(macd[-1] if not np.isnan(macd[-1]) else 0.0)
            
            # Bollinger Band width
            upper, middle, lower = talib.BBANDS(closes, timeperiod=20)
            bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] > 0 else 0.0
            features.append(bb_width)
            
            # ATR ratio
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0.0
            avg_atr = np.mean(atr[-20:]) if len(atr) >= 20 else current_atr
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            features.append(atr_ratio)
            
            # ADX
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            features.append(adx[-1] if not np.isnan(adx[-1]) else 0.0)
            
            return features
            
        except Exception as e:
            logger.warning(f"ML feature extraction failed: {e}")
            return [0.0] * len(self.ml_config.features)
    
    async def _train_ml_model(self) -> None:
        """Train ML model for strategy prediction"""
        try:
            if len(self.ml_training_data) < 100:
                logger.warning("Insufficient training data for ML model")
                return
            
            # Prepare training data
            X = []
            y = []
            
            for entry in self.ml_training_data:
                X.append(entry['features'])
                y.append(entry['best_strategy'])
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Train scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.ml_model.fit(X_scaled, y)
            
            logger.info("ML model trained successfully")
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            self.ml_model = None
    
    def get_strategy_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance report"""
        with self._lock:
            report = {
                'timestamp': datetime.now().isoformat(),
                'current_strategies': [s.value for s in self.current_strategies],
                'strategy_performance': {},
                'market_conditions': self.current_market_conditions.__dict__ if self.current_market_conditions else {},
                'selection_history_summary': self._get_selection_history_summary()
            }
            
            for strategy_type, performance in self.strategy_performance.items():
                report['strategy_performance'][strategy_type.value] = {
                    'total_trades': performance.total_trades,
                    'win_rate': performance.win_rate,
                    'profit_factor': performance.profit_factor,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'max_drawdown': performance.max_drawdown,
                    'confidence_score': performance.confidence_score,
                    'last_used': performance.last_used.isoformat()
                }
            
            return report
    
    def _get_selection_history_summary(self) -> Dict[str, Any]:
        """Get summary of selection history"""
        try:
            if not self.selection_history:
                return {}
            
            strategy_counts = defaultdict(int)
            method_counts = defaultdict(int)
            confidence_scores = []
            
            for selection in self.selection_history:
                strategy_counts[selection.primary_strategy.value] += 1
                method_counts[selection.selection_method.value] += 1
                confidence_scores.append(selection.confidence)
            
            return {
                'total_selections': len(self.selection_history),
                'strategy_distribution': dict(strategy_counts),
                'method_distribution': dict(method_counts),
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Selection history summary failed: {e}")
            return {}
    
    def save_selector_state(self, filename: str = "strategy_selector_state.json") -> None:
        """Save current selector state to file"""
        try:
            with self._lock:
                state = {
                    'timestamp': datetime.now().isoformat(),
                    'strategy_performance': {},
                    'config': self.config.__dict__,
                    'current_strategies': [s.value for s in self.current_strategies],
                    'selection_history_count': len(self.selection_history)
                }
                
                # Convert strategy performance to serializable format
                for strategy_type, performance in self.strategy_performance.items():
                    state['strategy_performance'][strategy_type.value] = {
                        'total_trades': performance.total_trades,
                        'winning_trades': performance.winning_trades,
                        'losing_trades': performance.losing_trades,
                        'win_rate': performance.win_rate,
                        'profit_factor': performance.profit_factor,
                        'sharpe_ratio': performance.sharpe_ratio,
                        'max_drawdown': performance.max_drawdown,
                        'avg_trade_duration': performance.avg_trade_duration,
                        'recent_performance': performance.recent_performance,
                        'last_used': performance.last_used.isoformat(),
                        'confidence_score': performance.confidence_score
                    }
                
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2, default=str)
                
                # Save ML model if available
                if self.ml_model and self.scaler:
                    joblib.dump({
                        'model': self.ml_model,
                        'scaler': self.scaler,
                        'training_data_count': len(self.ml_training_data)
                    }, f"{filename}.model")
                
                logger.info(f"Selector state saved to {filename}")
                
        except Exception as e:
            logger.error(f"Selector state saving failed: {e}")

# Example usage and testing
async def main():
    """Example usage of the AdvancedStrategySelector"""
    
    print("=== Testing Advanced Strategy Selector ===")
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
    
    # Create trending market data
    trend = np.cumsum(np.random.normal(0.001, 0.005, 200))
    prices = 1.1000 + trend
    
    market_data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.lognormal(10, 1, 200)
    }, index=dates)
    
    print(f"Generated {len(market_data)} periods of market data")
    
    # Configure strategy selector
    config = StrategyConfig(
        enabled_strategies=[
            StrategyType.TREND_FOLLOWING,
            StrategyType.MEAN_REVERSION,
            StrategyType.BREAKOUT,
            StrategyType.MOMENTUM
        ],
        selection_method=SelectionMethod.HYBRID,
        min_win_rate=0.45,
        min_profit_factor=1.1
    )
    
    # Initialize selector
    selector = AdvancedStrategySelector(config)
    
    # Add some sample performance data
    sample_trades = [
        {'profit': 25, 'return': 0.002, 'duration': 3600, 'total_profit': 100, 'total_loss': 50},
        {'profit': -15, 'return': -0.001, 'duration': 1800, 'total_profit': 100, 'total_loss': 65},
        {'profit': 30, 'return': 0.003, 'duration': 7200, 'total_profit': 130, 'total_loss': 65},
    ]
    
    print("\n=== Adding Sample Performance Data ===")
    for i, trade in enumerate(sample_trades):
        strategy = list(config.enabled_strategies)[i % len(config.enabled_strategies)]
        await selector.update_strategy_performance(strategy, trade)
        print(f"Added trade {i+1} for {strategy.value}")
    
    # Additional context
    context = {
        'market_sentiment': 0.7,
        'economic_events': []
    }
    
    print("\n=== Selecting Strategies ===")
    selection = await selector.select_strategies(market_data, context)
    
    print(f"Primary Strategy: {selection.primary_strategy.value}")
    print(f"Secondary Strategy: {selection.secondary_strategy.value}")
    print(f"Confidence: {selection.confidence:.3f}")
    print(f"Selection Method: {selection.selection_method.value}")
    print(f"Market Regime: {selection.market_conditions.regime.value}")
    
    print("\nStrategy Weights:")
    for strategy, weight in selection.strategy_weights.items():
        print(f"  {strategy.value}: {weight:.3f}")
    
    print("\nReasoning:")
    for reason in selection.reasoning:
        print(f"  - {reason}")
    
    print("\n=== Strategy Performance Report ===")
    report = selector.get_strategy_performance_report()
    
    print("Current Strategies:", report['current_strategies'])
    print("Market Regime:", report['market_conditions']['regime'])
    
    print("\nStrategy Performance:")
    for strategy, perf in report['strategy_performance'].items():
        print(f"  {strategy}:")
        print(f"    Trades: {perf['total_trades']}, Win Rate: {perf['win_rate']:.3f}")
        print(f"    Profit Factor: {perf['profit_factor']:.3f}, Confidence: {perf['confidence_score']:.3f}")
    
    if report['selection_history_summary']:
        summary = report['selection_history_summary']
        print(f"\nSelection History:")
        print(f"  Total Selections: {summary['total_selections']}")
        print(f"  Average Confidence: {summary['avg_confidence']:.3f}")
        
        print("  Strategy Distribution:")
        for strategy, count in summary['strategy_distribution'].items():
            print(f"    {strategy}: {count}")
    
    # Save selector state
    selector.save_selector_state("test_selector_state.json")
    print("\n=== Selector State Saved ===")
    
    print("\n=== Strategy Selector Test Completed ===")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run async main
    asyncio.run(main())
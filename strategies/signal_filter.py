"""
Advanced Signal Filter for FOREX TRADING BOT
Professional signal validation, filtering, and risk assessment
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

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FilterType(Enum):
    CONFIDENCE = "confidence"
    VOLATILITY = "volatility"
    RISK = "risk"
    MARKET_REGIME = "market_regime"
    CORRELATION = "correlation"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"

class SignalStatus(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    MODIFIED = "modified"

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class FilterResult:
    """Result of signal filtering"""
    status: SignalStatus
    original_signal: Any
    filtered_signal: Any
    filter_decisions: Dict[FilterType, Dict[str, Any]]
    overall_score: float
    risk_level: RiskLevel
    rejection_reasons: List[str]
    modifications: List[str]
    timestamp: datetime

@dataclass
class FilterConfig:
    """Configuration for signal filtering"""
    
    # Confidence filtering
    min_confidence: float = 0.65
    min_volume_confidence: float = 0.5
    min_technical_confidence: float = 0.6
    
    # Risk management
    max_daily_trades: int = 10
    max_hourly_trades: int = 3
    max_position_size: float = 0.1
    max_drawdown_limit: float = 0.05
    risk_per_trade: float = 0.02
    
    # Volatility filtering
    max_volatility_ratio: float = 2.0
    min_atr_ratio: float = 0.5
    max_atr_ratio: float = 3.0
    volatility_lookback: int = 20
    
    # Market regime filtering
    enable_regime_filtering: bool = True
    min_trend_strength: float = 0.3
    max_range_volatility: float = 0.02
    
    # Correlation filtering
    enable_correlation_check: bool = True
    max_correlation_threshold: float = 0.8
    correlation_lookback: int = 50
    
    # Sentiment filtering
    enable_sentiment_filter: bool = True
    min_sentiment_score: float = 0.4
    sentiment_weight: float = 0.3
    
    # Technical filtering
    enable_technical_validation: bool = True
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_confirmation: bool = True
    bb_confirmation: bool = True
    
    # Fundamental filtering
    enable_fundamental_filter: bool = True
    economic_event_buffer: int = 30  # minutes
    high_impact_event_multiplier: float = 2.0
    
    # Performance optimization
    cache_ttl: int = 300
    enable_adaptive_filtering: bool = True
    learning_period: int = 1000

@dataclass
class AdaptiveFilterWeights:
    """Adaptive filter weights based on performance"""
    confidence_weight: float = 0.25
    risk_weight: float = 0.20
    volatility_weight: float = 0.15
    regime_weight: float = 0.15
    technical_weight: float = 0.15
    sentiment_weight: float = 0.10

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # trending, ranging, volatile, breakout
    strength: float
    confidence: float
    duration: int
    timestamp: datetime

class AdvancedSignalFilter:
    """
    Advanced Signal Filter with Multi-layer Validation and Risk Assessment
    """
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        
        # Filter state
        self.filter_history: deque = deque(maxlen=1000)
        self.trade_history: deque = deque(maxlen=500)
        self.performance_metrics: Dict[str, Any] = {}
        
        # Adaptive learning
        self.adaptive_weights = AdaptiveFilterWeights()
        self.filter_performance: Dict[FilterType, List[bool]] = defaultdict(list)
        self.learning_data: List[Dict] = []
        
        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_cache: Dict[str, float] = {}
        
        # Risk tracking
        self.daily_trade_count: Dict[str, int] = defaultdict(int)
        self.hourly_trade_count: Dict[str, int] = defaultdict(int)
        self.position_sizes: Dict[str, float] = defaultdict(float)
        self.drawdown_tracker: Dict[str, float] = defaultdict(float)
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        logger.info("AdvancedSignalFilter initialized successfully")
    
    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking metrics"""
        self.performance_metrics = {
            'total_signals_processed': 0,
            'signals_approved': 0,
            'signals_rejected': 0,
            'signals_modified': 0,
            'approval_rate': 0.0,
            'filter_performance': {},
            'rejection_reasons': defaultdict(int),
            'recent_performance': deque(maxlen=100)
        }
        
        for filter_type in FilterType:
            self.performance_metrics['filter_performance'][filter_type.value] = {
                'applications': 0,
                'rejections': 0,
                'modifications': 0
            }
    
    async def filter_signal(self, signal: Any, market_data: pd.DataFrame, 
                          additional_context: Dict[str, Any] = None) -> FilterResult:
        """
        Apply comprehensive signal filtering
        
        Args:
            signal: Trading signal to filter
            market_data: Current market data
            additional_context: Additional context (sentiment, news, etc.)
        
        Returns:
            FilterResult with detailed filtering decisions
        """
        try:
            with self._lock:
                self.performance_metrics['total_signals_processed'] += 1
                
                # Initialize filter decisions
                filter_decisions = {}
                rejection_reasons = []
                modifications = []
                overall_score = 0.0
                
                # Apply each filter type
                confidence_result = await self._apply_confidence_filter(signal, market_data)
                filter_decisions[FilterType.CONFIDENCE] = confidence_result
                
                risk_result = await self._apply_risk_filter(signal, market_data)
                filter_decisions[FilterType.RISK] = risk_result
                
                volatility_result = await self._apply_volatility_filter(signal, market_data)
                filter_decisions[FilterType.VOLATILITY] = volatility_result
                
                regime_result = await self._apply_regime_filter(signal, market_data)
                filter_decisions[FilterType.MARKET_REGIME] = regime_result
                
                technical_result = await self._apply_technical_filter(signal, market_data)
                filter_decisions[FilterType.TECHNICAL] = technical_result
                
                # Apply optional filters
                if self.config.enable_sentiment_filter and additional_context:
                    sentiment_result = await self._apply_sentiment_filter(signal, additional_context)
                    filter_decisions[FilterType.SENTIMENT] = sentiment_result
                
                if self.config.enable_correlation_check:
                    correlation_result = await self._apply_correlation_filter(signal, market_data)
                    filter_decisions[FilterType.CORRELATION] = correlation_result
                
                if self.config.enable_fundamental_filter and additional_context:
                    fundamental_result = await self._apply_fundamental_filter(signal, additional_context)
                    filter_decisions[FilterType.FUNDAMENTAL] = fundamental_result
                
                # Calculate overall score
                overall_score = self._calculate_overall_score(filter_decisions)
                
                # Determine signal status
                status, final_rejection_reasons, final_modifications = self._determine_signal_status(
                    filter_decisions, overall_score
                )
                
                rejection_reasons.extend(final_rejection_reasons)
                modifications.extend(final_modifications)
                
                # Apply modifications to signal if needed
                filtered_signal = self._apply_signal_modifications(signal, modifications)
                
                # Determine risk level
                risk_level = self._determine_risk_level(overall_score, filter_decisions)
                
                # Update performance tracking
                self._update_performance_tracking(status, filter_decisions, rejection_reasons)
                
                # Create filter result
                result = FilterResult(
                    status=status,
                    original_signal=signal,
                    filtered_signal=filtered_signal,
                    filter_decisions=filter_decisions,
                    overall_score=overall_score,
                    risk_level=risk_level,
                    rejection_reasons=rejection_reasons,
                    modifications=modifications,
                    timestamp=datetime.now()
                )
                
                # Store in history
                self.filter_history.append(result)
                
                # Adaptive learning
                if self.config.enable_adaptive_filtering:
                    await self._update_adaptive_weights(result)
                
                logger.info(f"Signal filtering completed: {status.value} (score: {overall_score:.3f})")
                
                return result
                
        except Exception as e:
            logger.error(f"Signal filtering failed: {e}")
            # Return rejected signal on error
            return self._create_error_result(signal, str(e))
    
    async def _apply_confidence_filter(self, signal: Any, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply confidence-based filtering"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            # Extract confidence from signal
            signal_confidence = getattr(signal, 'confidence', 0.5)
            volume_confidence = getattr(signal, 'volume_confidence', 0.5)
            technical_confidence = getattr(signal, 'technical_confidence', 0.5)
            
            # Calculate weighted confidence
            weighted_confidence = (
                signal_confidence * 0.5 +
                volume_confidence * 0.3 +
                technical_confidence * 0.2
            )
            
            decisions['score'] = weighted_confidence
            decisions['details'] = {
                'signal_confidence': signal_confidence,
                'volume_confidence': volume_confidence,
                'technical_confidence': technical_confidence,
                'weighted_confidence': weighted_confidence
            }
            
            # Check against thresholds
            if weighted_confidence < self.config.min_confidence:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Low confidence: {weighted_confidence:.3f} < {self.config.min_confidence}"
            
            if volume_confidence < self.config.min_volume_confidence:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Low volume confidence: {volume_confidence:.3f} < {self.config.min_volume_confidence}"
            
            if technical_confidence < self.config.min_technical_confidence:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Low technical confidence: {technical_confidence:.3f} < {self.config.min_technical_confidence}"
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Confidence filter failed: {e}")
            return {'passed': False, 'score': 0.0, 'details': {}, 'rejection_reason': f'Filter error: {e}'}
    
    async def _apply_risk_filter(self, signal: Any, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply risk management filtering"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            symbol = getattr(signal, 'symbol', 'UNKNOWN')
            current_time = datetime.now()
            
            # Check daily trade limit
            today = current_time.date().isoformat()
            daily_key = f"{symbol}_{today}"
            
            if self.daily_trade_count[daily_key] >= self.config.max_daily_trades:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Daily trade limit exceeded: {self.daily_trade_count[daily_key]}/{self.config.max_daily_trades}"
            
            # Check hourly trade limit
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            hourly_key = f"{symbol}_{current_hour.isoformat()}"
            
            if self.hourly_trade_count[hourly_key] >= self.config.max_hourly_trades:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Hourly trade limit exceeded: {self.hourly_trade_count[hourly_key]}/{self.config.max_hourly_trades}"
            
            # Check position size
            position_size = getattr(signal, 'position_size', 0.0)
            if position_size > self.config.max_position_size:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Position size too large: {position_size:.3f} > {self.config.max_position_size}"
            
            # Check drawdown
            current_drawdown = self.drawdown_tracker[symbol]
            if current_drawdown > self.config.max_drawdown_limit:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Max drawdown exceeded: {current_drawdown:.3f} > {self.config.max_drawdown_limit}"
            
            # Calculate risk score (higher is better)
            risk_score = 1.0
            if decisions['passed']:
                # Normalize based on usage
                daily_usage = self.daily_trade_count[daily_key] / self.config.max_daily_trades
                hourly_usage = self.hourly_trade_count[hourly_key] / self.config.max_hourly_trades
                position_usage = position_size / self.config.max_position_size
                
                risk_score = 1.0 - max(daily_usage, hourly_usage, position_usage)
            
            decisions['score'] = risk_score
            decisions['details'] = {
                'daily_trades': self.daily_trade_count[daily_key],
                'hourly_trades': self.hourly_trade_count[hourly_key],
                'position_size': position_size,
                'current_drawdown': current_drawdown
            }
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Risk filter failed: {e}")
            return {'passed': False, 'score': 0.0, 'details': {}, 'rejection_reason': f'Filter error: {e}'}
    
    async def _apply_volatility_filter(self, signal: Any, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply volatility-based filtering"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            if len(market_data) < self.config.volatility_lookback:
                decisions['passed'] = False
                decisions['rejection_reason'] = "Insufficient data for volatility analysis"
                return decisions
            
            # Calculate current volatility
            closes = market_data['close'].values
            returns = np.diff(np.log(closes))
            current_volatility = np.std(returns[-self.config.volatility_lookback:]) * np.sqrt(252)
            
            # Calculate historical volatility
            historical_volatility = np.std(returns) * np.sqrt(252)
            
            # Calculate ATR
            highs = market_data['high'].values
            lows = market_data['low'].values
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else 0.0
            
            # Calculate ATR ratio (current vs average)
            atr_ratio = current_atr / np.mean(atr[-20:]) if np.mean(atr[-20:]) > 0 else 1.0
            
            # Volatility ratio (current vs historical)
            volatility_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
            
            decisions['details'] = {
                'current_volatility': current_volatility,
                'historical_volatility': historical_volatility,
                'volatility_ratio': volatility_ratio,
                'current_atr': current_atr,
                'atr_ratio': atr_ratio
            }
            
            # Apply volatility filters
            if volatility_ratio > self.config.max_volatility_ratio:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"High volatility ratio: {volatility_ratio:.2f} > {self.config.max_volatility_ratio}"
            
            if atr_ratio < self.config.min_atr_ratio:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Low ATR ratio: {atr_ratio:.2f} < {self.config.min_atr_ratio}"
            
            if atr_ratio > self.config.max_atr_ratio:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"High ATR ratio: {atr_ratio:.2f} > {self.config.max_atr_ratio}"
            
            # Calculate volatility score (higher is better for normal volatility)
            volatility_score = 1.0 - min(1.0, abs(volatility_ratio - 1.0))
            decisions['score'] = volatility_score
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Volatility filter failed: {e}")
            return {'passed': False, 'score': 0.0, 'details': {}, 'rejection_reason': f'Filter error: {e}'}
    
    async def _apply_regime_filter(self, signal: Any, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply market regime filtering"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            if not self.config.enable_regime_filtering:
                decisions['score'] = 1.0
                return decisions
            
            if len(market_data) < 50:
                decisions['passed'] = False
                decisions['rejection_reason'] = "Insufficient data for regime analysis"
                return decisions
            
            # Analyze market regime
            regime_type, regime_strength, regime_confidence = self._analyze_market_regime(market_data)
            
            decisions['details'] = {
                'regime_type': regime_type,
                'regime_strength': regime_strength,
                'regime_confidence': regime_confidence
            }
            
            # Get signal direction
            signal_direction = getattr(signal, 'action', '').lower()
            
            # Filter based on regime and signal alignment
            if regime_type == "ranging" and regime_strength > self.config.max_range_volatility:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"High volatility in ranging market: {regime_strength:.3f}"
            
            elif regime_type == "trending" and regime_strength < self.config.min_trend_strength:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Weak trend strength: {regime_strength:.3f} < {self.config.min_trend_strength}"
            
            # Check signal alignment with regime
            if regime_type == "trending":
                # In trending markets, signals should align with trend
                trend_direction = "bullish" if regime_strength > 0 else "bearish"
                if signal_direction != trend_direction:
                    decisions['score'] *= 0.5  # Penalize counter-trend signals
            
            # Calculate regime score
            regime_score = regime_confidence * (1.0 - abs(regime_strength))
            decisions['score'] = regime_score
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Regime filter failed: {e}")
            return {'passed': True, 'score': 0.5, 'details': {}, 'rejection_reason': None}
    
    async def _apply_technical_filter(self, signal: Any, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply technical indicator validation"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            if not self.config.enable_technical_validation:
                decisions['score'] = 1.0
                return decisions
            
            if len(market_data) < 30:
                decisions['passed'] = False
                decisions['rejection_reason'] = "Insufficient data for technical analysis"
                return decisions
            
            closes = market_data['close'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            # Calculate technical indicators
            rsi = talib.RSI(closes, timeperiod=14)
            current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
            
            macd, macd_signal, macd_hist = talib.MACD(closes)
            current_macd = macd[-1] if not np.isnan(macd[-1]) else 0.0
            current_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0
            
            bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
            bb_position = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if (bb_upper[-1] - bb_lower[-1]) > 0 else 0.5
            
            decisions['details'] = {
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_macd_signal,
                'macd_histogram': current_macd - current_macd_signal,
                'bb_position': bb_position
            }
            
            # Get signal direction
            signal_direction = getattr(signal, 'action', '').lower()
            
            # RSI validation
            if signal_direction == 'buy' and current_rsi > self.config.rsi_overbought:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"RSI overbought: {current_rsi:.1f} > {self.config.rsi_overbought}"
            
            elif signal_direction == 'sell' and current_rsi < self.config.rsi_oversold:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"RSI oversold: {current_rsi:.1f} < {self.config.rsi_oversold}"
            
            # MACD confirmation
            if self.config.macd_confirmation:
                macd_bullish = current_macd > current_macd_signal
                if signal_direction == 'buy' and not macd_bullish:
                    decisions['score'] *= 0.7  # Penalize but don't reject
                elif signal_direction == 'sell' and macd_bullish:
                    decisions['score'] *= 0.7
            
            # Bollinger Bands confirmation
            if self.config.bb_confirmation:
                if signal_direction == 'buy' and bb_position > 0.8:  # Near upper band
                    decisions['score'] *= 0.6
                elif signal_direction == 'sell' and bb_position < 0.2:  # Near lower band
                    decisions['score'] *= 0.6
            
            # Calculate technical score
            technical_score = 1.0
            
            # RSI score (distance from extremes)
            rsi_score = 1.0 - min(1.0, abs(current_rsi - 50) / 50)
            
            # MACD score (strength of signal)
            macd_score = min(1.0, abs(current_macd - current_macd_signal) / 0.01)  # Normalize
            
            # BB score (middle is best)
            bb_score = 1.0 - abs(bb_position - 0.5) * 2
            
            technical_score = (rsi_score * 0.4 + macd_score * 0.3 + bb_score * 0.3)
            decisions['score'] = technical_score
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Technical filter failed: {e}")
            return {'passed': True, 'score': 0.5, 'details': {}, 'rejection_reason': None}
    
    async def _apply_sentiment_filter(self, signal: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sentiment-based filtering"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            sentiment_data = context.get('sentiment', {})
            news_sentiment = sentiment_data.get('news_score', 0.5)
            social_sentiment = sentiment_data.get('social_score', 0.5)
            market_sentiment = sentiment_data.get('market_score', 0.5)
            
            # Calculate weighted sentiment
            weighted_sentiment = (
                news_sentiment * 0.4 +
                social_sentiment * 0.3 +
                market_sentiment * 0.3
            )
            
            decisions['details'] = {
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'market_sentiment': market_sentiment,
                'weighted_sentiment': weighted_sentiment
            }
            
            # Get signal direction
            signal_direction = getattr(signal, 'action', '').lower()
            
            # Check sentiment alignment
            if signal_direction == 'buy' and weighted_sentiment < self.config.min_sentiment_score:
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Low bullish sentiment: {weighted_sentiment:.3f} < {self.config.min_sentiment_score}"
            
            elif signal_direction == 'sell' and weighted_sentiment > (1 - self.config.min_sentiment_score):
                decisions['passed'] = False
                decisions['rejection_reason'] = f"Low bearish sentiment: {weighted_sentiment:.3f} > {1 - self.config.min_sentiment_score}"
            
            # Calculate sentiment score
            if signal_direction == 'buy':
                sentiment_score = weighted_sentiment
            elif signal_direction == 'sell':
                sentiment_score = 1.0 - weighted_sentiment
            else:
                sentiment_score = 0.5
            
            decisions['score'] = sentiment_score
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Sentiment filter failed: {e}")
            return {'passed': True, 'score': 0.5, 'details': {}, 'rejection_reason': None}
    
    async def _apply_correlation_filter(self, signal: Any, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply correlation-based filtering"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            # This would typically check correlation with other positions
            # For now, we'll implement a simplified version
            
            symbol = getattr(signal, 'symbol', 'UNKNOWN')
            
            # Check if we have correlation data for this symbol
            if symbol in self.correlation_cache:
                avg_correlation = self.correlation_cache[symbol]
                
                decisions['details'] = {
                    'average_correlation': avg_correlation
                }
                
                if avg_correlation > self.config.max_correlation_threshold:
                    decisions['passed'] = False
                    decisions['rejection_reason'] = f"High correlation with existing positions: {avg_correlation:.3f}"
            
            # Default score for no correlation data
            decisions['score'] = 0.8
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Correlation filter failed: {e}")
            return {'passed': True, 'score': 0.5, 'details': {}, 'rejection_reason': None}
    
    async def _apply_fundamental_filter(self, signal: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fundamental analysis filtering"""
        try:
            decisions = {
                'passed': True,
                'score': 0.0,
                'details': {},
                'rejection_reason': None
            }
            
            economic_events = context.get('economic_events', [])
            current_time = datetime.now()
            
            # Check for high-impact economic events
            high_impact_events = []
            for event in economic_events:
                event_time = event.get('timestamp')
                impact = event.get('impact', 'low')
                
                if isinstance(event_time, str):
                    event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                
                # Check if event is within buffer period
                time_diff = abs((event_time - current_time).total_seconds() / 60)
                
                if time_diff <= self.config.economic_event_buffer and impact in ['high', 'medium']:
                    high_impact_events.append(event)
            
            decisions['details'] = {
                'high_impact_events': len(high_impact_events),
                'events': high_impact_events
            }
            
            # Apply fundamental filters
            if high_impact_events:
                # High impact events nearby - be cautious
                decisions['score'] = 0.3
                
                # For very high impact events, consider rejecting
                high_impact_count = sum(1 for e in high_impact_events if e.get('impact') == 'high')
                if high_impact_count >= 2:
                    decisions['passed'] = False
                    decisions['rejection_reason'] = "Multiple high-impact economic events nearby"
            
            else:
                decisions['score'] = 0.9
            
            return decisions
            
        except Exception as e:
            logger.warning(f"Fundamental filter failed: {e}")
            return {'passed': True, 'score': 0.5, 'details': {}, 'rejection_reason': None}
    
    def _analyze_market_regime(self, market_data: pd.DataFrame) -> Tuple[str, float, float]:
        """Analyze current market regime"""
        try:
            closes = market_data['close'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            # Calculate trend using linear regression
            x = np.arange(len(closes))
            slope, _, r_value, _, _ = stats.linregress(x, closes)
            
            # Calculate ADX for trend strength
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
            
            # Calculate volatility
            returns = np.diff(np.log(closes))
            volatility = np.std(returns) * np.sqrt(252)
            
            # Determine regime
            if current_adx > 25:  # Strong trend
                regime_type = "trending"
                regime_strength = slope * 1000  # Scale slope
                regime_confidence = min(1.0, current_adx / 50)
            elif volatility > 0.15:  # High volatility
                regime_type = "volatile"
                regime_strength = volatility
                regime_confidence = 0.7
            else:  # Ranging market
                regime_type = "ranging"
                regime_strength = volatility
                regime_confidence = 0.8
            
            return regime_type, regime_strength, regime_confidence
            
        except Exception as e:
            logger.warning(f"Market regime analysis failed: {e}")
            return "unknown", 0.0, 0.5
    
    def _calculate_overall_score(self, filter_decisions: Dict[FilterType, Dict]) -> float:
        """Calculate overall filtering score"""
        try:
            total_score = 0.0
            total_weight = 0.0
            
            for filter_type, decision in filter_decisions.items():
                weight = getattr(self.adaptive_weights, f"{filter_type.value}_weight", 0.1)
                score = decision.get('score', 0.5)
                
                # Apply penalty if filter didn't pass
                if not decision.get('passed', True):
                    score *= 0.3
                
                total_score += score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def _determine_signal_status(self, filter_decisions: Dict[FilterType, Dict], 
                               overall_score: float) -> Tuple[SignalStatus, List[str], List[str]]:
        """Determine final signal status"""
        rejection_reasons = []
        modifications = []
        
        # Check for critical rejections
        for filter_type, decision in filter_decisions.items():
            if not decision.get('passed', True):
                rejection_reason = decision.get('rejection_reason')
                if rejection_reason:
                    rejection_reasons.append(f"{filter_type.value}: {rejection_reason}")
        
        # Check overall score
        if overall_score < 0.3:
            rejection_reasons.append(f"Low overall score: {overall_score:.3f}")
        
        # Determine status
        if rejection_reasons:
            return SignalStatus.REJECTED, rejection_reasons, modifications
        
        elif overall_score < 0.7:
            modifications.append(f"Signal modified due to moderate score: {overall_score:.3f}")
            return SignalStatus.MODIFIED, [], modifications
        
        else:
            return SignalStatus.APPROVED, [], modifications
    
    def _apply_signal_modifications(self, signal: Any, modifications: List[str]) -> Any:
        """Apply modifications to signal based on filtering results"""
        try:
            if not modifications:
                return signal
            
            # Create a copy of the signal to modify
            modified_signal = type(signal)(**signal.__dict__) if hasattr(signal, '__dict__') else signal
            
            # Apply position size reduction for moderate scores
            if any("moderate score" in mod for mod in modifications):
                if hasattr(modified_signal, 'position_size'):
                    modified_signal.position_size *= 0.5
                    modifications.append("Position size reduced by 50%")
            
            # Apply tighter stop loss for volatility concerns
            if any("volatility" in mod.lower() for mod in modifications):
                if hasattr(modified_signal, 'stop_loss') and hasattr(modified_signal, 'price'):
                    current_sl = modified_signal.stop_loss
                    price = modified_signal.price
                    new_sl = price + (current_sl - price) * 0.7  # 30% tighter
                    modified_signal.stop_loss = new_sl
                    modifications.append("Stop loss tightened by 30%")
            
            return modified_signal
            
        except Exception as e:
            logger.warning(f"Signal modification failed: {e}")
            return signal
    
    def _determine_risk_level(self, overall_score: float, filter_decisions: Dict) -> RiskLevel:
        """Determine risk level based on filtering results"""
        try:
            if overall_score >= 0.8:
                return RiskLevel.VERY_LOW
            elif overall_score >= 0.7:
                return RiskLevel.LOW
            elif overall_score >= 0.5:
                return RiskLevel.MEDIUM
            elif overall_score >= 0.3:
                return RiskLevel.HIGH
            else:
                return RiskLevel.VERY_HIGH
                
        except Exception as e:
            logger.warning(f"Risk level determination failed: {e}")
            return RiskLevel.MEDIUM
    
    def _update_performance_tracking(self, status: SignalStatus, 
                                   filter_decisions: Dict[FilterType, Dict],
                                   rejection_reasons: List[str]) -> None:
        """Update performance tracking metrics"""
        try:
            if status == SignalStatus.APPROVED:
                self.performance_metrics['signals_approved'] += 1
            elif status == SignalStatus.REJECTED:
                self.performance_metrics['signals_rejected'] += 1
                for reason in rejection_reasons:
                    self.performance_metrics['rejection_reasons'][reason] += 1
            elif status == SignalStatus.MODIFIED:
                self.performance_metrics['signals_modified'] += 1
            
            # Update filter performance
            for filter_type, decision in filter_decisions.items():
                filter_perf = self.performance_metrics['filter_performance'][filter_type.value]
                filter_perf['applications'] += 1
                
                if not decision.get('passed', True):
                    filter_perf['rejections'] += 1
                elif status == SignalStatus.MODIFIED:
                    filter_perf['modifications'] += 1
            
            # Update approval rate
            total_processed = self.performance_metrics['total_signals_processed']
            approved = self.performance_metrics['signals_approved']
            self.performance_metrics['approval_rate'] = approved / total_processed if total_processed > 0 else 0.0
            
            # Add to recent performance
            self.performance_metrics['recent_performance'].append(status == SignalStatus.APPROVED)
            
        except Exception as e:
            logger.warning(f"Performance tracking update failed: {e}")
    
    async def _update_adaptive_weights(self, filter_result: FilterResult) -> None:
        """Update adaptive filter weights based on performance"""
        try:
            if len(self.learning_data) >= self.config.learning_period:
                # Reset learning data periodically
                self.learning_data.clear()
            
            # Store learning data
            learning_entry = {
                'timestamp': datetime.now(),
                'result': filter_result.status.value,
                'scores': {ft.value: fd.get('score', 0.5) for ft, fd in filter_result.filter_decisions.items()},
                'overall_score': filter_result.overall_score
            }
            self.learning_data.append(learning_entry)
            
            # Simple adaptive learning: adjust weights based on recent performance
            if len(self.learning_data) > 100:
                recent_data = self.learning_data[-100:]
                
                # Calculate performance for each filter
                filter_performance = {}
                for filter_type in FilterType:
                    filter_scores = [entry['scores'].get(filter_type.value, 0.5) for entry in recent_data]
                    approved_entries = [entry for entry in recent_data if entry['result'] == 'approved']
                    
                    if approved_entries:
                        approved_scores = [entry['scores'].get(filter_type.value, 0.5) for entry in approved_entries]
                        avg_approved_score = np.mean(approved_scores)
                        filter_performance[filter_type] = avg_approved_score
                
                # Normalize and update weights
                if filter_performance:
                    total_performance = sum(filter_performance.values())
                    for filter_type, performance in filter_performance.items():
                        new_weight = performance / total_performance
                        current_weight = getattr(self.adaptive_weights, f"{filter_type.value}_weight")
                        # Smooth update
                        updated_weight = current_weight * 0.9 + new_weight * 0.1
                        setattr(self.adaptive_weights, f"{filter_type.value}_weight", updated_weight)
                
                logger.debug(f"Updated adaptive weights: {self.adaptive_weights}")
                
        except Exception as e:
            logger.warning(f"Adaptive weight update failed: {e}")
    
    def _create_error_result(self, signal: Any, error: str) -> FilterResult:
        """Create error result when filtering fails"""
        return FilterResult(
            status=SignalStatus.REJECTED,
            original_signal=signal,
            filtered_signal=signal,
            filter_decisions={},
            overall_score=0.0,
            risk_level=RiskLevel.VERY_HIGH,
            rejection_reasons=[f"Filtering error: {error}"],
            modifications=[],
            timestamp=datetime.now()
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            return self.performance_metrics.copy()
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get detailed filter statistics"""
        with self._lock:
            stats = {
                'total_processed': self.performance_metrics['total_signals_processed'],
                'approval_rate': self.performance_metrics['approval_rate'],
                'recent_performance': list(self.performance_metrics['recent_performance']),
                'filter_performance': self.performance_metrics['filter_performance'],
                'top_rejection_reasons': dict(sorted(
                    self.performance_metrics['rejection_reasons'].items(),
                    key=lambda x: x[1], reverse=True
                )[:10]),
                'adaptive_weights': self.adaptive_weights.__dict__,
                'current_risk_limits': {
                    'max_daily_trades': self.config.max_daily_trades,
                    'max_hourly_trades': self.config.max_hourly_trades,
                    'max_position_size': self.config.max_position_size
                }
            }
            
            return stats
    
    def reset_filters(self) -> None:
        """Reset filter state and history"""
        with self._lock:
            self.filter_history.clear()
            self.trade_history.clear()
            self._initialize_performance_tracking()
            self.adaptive_weights = AdaptiveFilterWeights()
            self.learning_data.clear()
            
            logger.info("Signal filters reset successfully")
    
    def save_filter_state(self, filename: str = "signal_filter_state.json") -> None:
        """Save current filter state to file"""
        try:
            with self._lock:
                state = {
                    'timestamp': datetime.now().isoformat(),
                    'performance_metrics': self.performance_metrics,
                    'adaptive_weights': self.adaptive_weights.__dict__,
                    'filter_config': self.config.__dict__,
                    'recent_history': [
                        {
                            'timestamp': fr.timestamp.isoformat(),
                            'status': fr.status.value,
                            'overall_score': fr.overall_score,
                            'risk_level': fr.risk_level.value
                        }
                        for fr in list(self.filter_history)[-100:]  # Last 100 entries
                    ]
                }
                
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2, default=str)
                
                logger.info(f"Filter state saved to {filename}")
                
        except Exception as e:
            logger.error(f"Filter state saving failed: {e}")
    
    def load_filter_state(self, filename: str = "signal_filter_state.json") -> None:
        """Load filter state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            with self._lock:
                self.performance_metrics = state['performance_metrics']
                self.adaptive_weights = AdaptiveFilterWeights(**state['adaptive_weights'])
                
                logger.info("Filter state loaded successfully")
                
        except Exception as e:
            logger.error(f"Filter state loading failed: {e}")

# Example usage and testing
async def main():
    """Example usage of the AdvancedSignalFilter"""
    
    # Create a sample signal class for testing
    class SampleSignal:
        def __init__(self, symbol, action, confidence, price, position_size=0.05):
            self.symbol = symbol
            self.action = action
            self.confidence = confidence
            self.price = price
            self.position_size = position_size
            self.volume_confidence = 0.7
            self.technical_confidence = 0.8
            self.stop_loss = price * 0.99
            self.take_profit = price * 1.02
    
    print("=== Testing Advanced Signal Filter ===")
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
    prices = 1.1000 + np.cumsum(np.random.normal(0, 0.0005, 200))
    
    market_data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.lognormal(10, 1, 200)
    }, index=dates)
    
    print(f"Generated {len(market_data)} periods of market data")
    
    # Configure signal filter
    config = FilterConfig(
        min_confidence=0.6,
        max_daily_trades=5,
        max_hourly_trades=2,
        enable_regime_filtering=True,
        enable_technical_validation=True,
        enable_sentiment_filter=True
    )
    
    # Initialize filter
    signal_filter = AdvancedSignalFilter(config)
    
    # Test signals
    test_signals = [
        SampleSignal("EUR/USD", "buy", 0.8, 1.1050, 0.03),
        SampleSignal("EUR/USD", "sell", 0.4, 1.1040, 0.08),  # Low confidence
        SampleSignal("EUR/USD", "buy", 0.9, 1.1030, 0.02),  # High confidence
        SampleSignal("EUR/USD", "buy", 0.7, 1.1020, 0.12),  # Large position
    ]
    
    # Additional context for filtering
    context = {
        'sentiment': {
            'news_score': 0.7,
            'social_score': 0.6,
            'market_score': 0.8
        },
        'economic_events': [
            {
                'timestamp': (datetime.now() + timedelta(minutes=45)).isoformat(),
                'impact': 'high',
                'event': 'NFP Report'
            }
        ]
    }
    
    print("\n=== Filtering Test Signals ===")
    
    for i, signal in enumerate(test_signals):
        print(f"\nSignal {i+1}: {signal.action} {signal.symbol} at {signal.price:.4f}")
        print(f"  Confidence: {signal.confidence:.2f}, Position: {signal.position_size:.2f}")
        
        # Apply filtering
        result = await signal_filter.filter_signal(signal, market_data, context)
        
        print(f"  Status: {result.status.value}")
        print(f"  Overall Score: {result.overall_score:.3f}")
        print(f"  Risk Level: {result.risk_level.value}")
        
        if result.rejection_reasons:
            print(f"  Rejection Reasons: {result.rejection_reasons}")
        
        if result.modifications:
            print(f"  Modifications: {result.modifications}")
        
        # Show filter scores
        print("  Filter Scores:")
        for filter_type, decision in result.filter_decisions.items():
            score = decision.get('score', 0.0)
            passed = decision.get('passed', True)
            status = "✓" if passed else "✗"
            print(f"    {filter_type.value}: {status} {score:.3f}")
    
    print("\n=== Filter Performance Statistics ===")
    stats = signal_filter.get_filter_statistics()
    
    print(f"Total Signals Processed: {stats['total_processed']}")
    print(f"Approval Rate: {stats['approval_rate']:.1%}")
    
    print("\nFilter Performance:")
    for filter_type, perf in stats['filter_performance'].items():
        applications = perf['applications']
        rejections = perf['rejections']
        rejection_rate = rejections / applications if applications > 0 else 0.0
        print(f"  {filter_type}: {applications} apps, {rejections} rejects ({rejection_rate:.1%})")
    
    print("\nTop Rejection Reasons:")
    for reason, count in list(stats['top_rejection_reasons'].items())[:3]:
        print(f"  {reason}: {count}")
    
    print("\nAdaptive Weights:")
    for weight_name, weight_value in stats['adaptive_weights'].items():
        print(f"  {weight_name}: {weight_value:.3f}")
    
    # Save filter state
    signal_filter.save_filter_state("test_filter_state.json")
    print("\n=== Filter State Saved ===")
    
    print("\n=== Signal Filter Test Completed ===")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run async main
    asyncio.run(main())
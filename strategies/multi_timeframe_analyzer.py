"""
Advanced Multi-Timeframe Analyzer for FOREX TRADING BOT
Professional timeframe analysis with confluence detection and market structure
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import talib
from scipy import stats
import warnings
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, field
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

class TrendDirection(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class MarketStructure(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class SupportResistanceLevel:
    """Support/Resistance Level with strength scoring"""
    
    def __init__(self, level: float, strength: float, timeframe: Timeframe, 
                 touches: int, last_touch: datetime, is_broken: bool = False):
        self.level = level
        self.strength = strength
        self.timeframe = timeframe
        self.touches = touches
        self.last_touch = last_touch
        self.is_broken = is_broken
        self.breakout_direction: Optional[TrendDirection] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level,
            'strength': self.strength,
            'timeframe': self.timeframe.value,
            'touches': self.touches,
            'last_touch': self.last_touch.isoformat(),
            'is_broken': self.is_broken,
            'breakout_direction': self.breakout_direction.value if self.breakout_direction else None
        }

@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe"""
    timeframe: Timeframe
    trend: TrendDirection
    momentum: float
    volatility: float
    rsi: float
    macd_signal: str
    bb_position: float
    atr: float
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    pivot_points: Dict[str, float]
    market_structure: MarketStructure
    key_levels: List[float]
    timestamp: datetime

@dataclass
class MultiTimeframeSignal:
    """Multi-timeframe trading signal with confluence"""
    symbol: str
    primary_action: str
    confidence: float
    timeframe_confluence: Dict[Timeframe, str]
    key_levels: Dict[str, float]
    trend_alignment: str
    momentum_score: float
    risk_reward_ratio: float
    entry_zones: List[Tuple[float, float]]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfluenceScore:
    """Confluence scoring across timeframes"""
    bullish_score: float
    bearish_score: float
    neutral_score: float
    dominant_bias: TrendDirection
    strength: float
    conflicting_timeframes: List[Timeframe]

@dataclass
class MTFConfig:
    """Configuration for multi-timeframe analysis"""
    # Timeframes to analyze
    timeframes: List[Timeframe] = field(default_factory=lambda: [
        Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1
    ])
    
    # Analysis parameters
    lookback_periods: int = 200
    pivot_period: int = 20
    sr_touch_threshold: int = 3
    volume_threshold: float = 1.2
    
    # Technical indicator settings
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    atr_period: int = 14
    
    # Confluence scoring
    timeframe_weights: Dict[Timeframe, float] = field(default_factory=lambda: {
        Timeframe.M15: 0.1,
        Timeframe.M30: 0.15,
        Timeframe.H1: 0.2,
        Timeframe.H4: 0.25,
        Timeframe.D1: 0.3
    })
    
    # Signal generation
    min_confluence_score: float = 0.7
    min_trend_alignment: int = 2  # Minimum aligned timeframes
    enable_volume_confirmation: bool = True
    
    # Advanced features
    enable_fibonacci: bool = True
    enable_ichimoku: bool = True
    enable_elliott_wave: bool = False
    enable_market_profile: bool = False

class AdvancedMultiTimeframeAnalyzer:
    """
    Advanced Multi-Timeframe Market Analyzer with Confluence Detection
    """
    
    def __init__(self, config: MTFConfig = None):
        self.config = config or MTFConfig()
        
        # Data storage
        self.market_data: Dict[Timeframe, pd.DataFrame] = {}
        self.analysis_results: Dict[Timeframe, TimeframeAnalysis] = {}
        self.support_resistance_levels: Dict[Timeframe, List[SupportResistanceLevel]] = {}
        
        # Historical analysis
        self.historical_signals: deque = deque(maxlen=1000)
        self.confluence_history: Dict[Timeframe, List[ConfluenceScore]] = defaultdict(list)
        
        # Pattern recognition
        self.chart_patterns: Dict[Timeframe, List[Dict]] = defaultdict(list)
        self.market_structure: Dict[Timeframe, MarketStructure] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=8)
        
        # Cache for performance
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        logger.info("AdvancedMultiTimeframeAnalyzer initialized")
    
    def add_market_data(self, timeframe: Timeframe, data: pd.DataFrame) -> None:
        """Add market data for specific timeframe"""
        try:
            with self._lock:
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_columns):
                    raise ValueError(f"Missing required columns: {required_columns}")
                
                self.market_data[timeframe] = data.sort_index().copy()
                logger.info(f"Added market data for {timeframe.value}: {len(data)} candles")
                
        except Exception as e:
            logger.error(f"Market data addition failed for {timeframe.value}: {e}")
            raise
    
    async def analyze_all_timeframes(self) -> Dict[Timeframe, TimeframeAnalysis]:
        """Analyze all timeframes concurrently"""
        try:
            tasks = []
            for timeframe in self.config.timeframes:
                if timeframe in self.market_data:
                    task = asyncio.create_task(
                        self._analyze_single_timeframe(timeframe)
                    )
                    tasks.append(task)
            
            # Wait for all analyses to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, timeframe in enumerate(self.config.timeframes):
                if i < len(results) and not isinstance(results[i], Exception):
                    self.analysis_results[timeframe] = results[i]
                else:
                    logger.error(f"Analysis failed for {timeframe.value}")
            
            logger.info(f"Completed multi-timeframe analysis for {len(self.analysis_results)} timeframes")
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Multi-timeframe analysis failed: {e}")
            return {}
    
    async def _analyze_single_timeframe(self, timeframe: Timeframe) -> TimeframeAnalysis:
        """Analyze single timeframe"""
        try:
            data = self.market_data[timeframe]
            if len(data) < self.config.lookback_periods:
                raise ValueError(f"Insufficient data for {timeframe.value}")
            
            # Use recent data for analysis
            analysis_data = data.tail(self.config.lookback_periods).copy()
            
            # Calculate technical indicators
            trend, trend_strength = self._analyze_trend(analysis_data)
            momentum = self._calculate_momentum(analysis_data)
            volatility = self._calculate_volatility(analysis_data)
            rsi = self._calculate_rsi(analysis_data)
            macd_signal = self._analyze_macd(analysis_data)
            bb_position = self._analyze_bollinger_bands(analysis_data)
            atr = self._calculate_atr(analysis_data)
            
            # Support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance(analysis_data, timeframe)
            
            # Pivot points
            pivot_points = self._calculate_pivot_points(analysis_data)
            
            # Market structure
            market_structure = self._analyze_market_structure(analysis_data)
            
            # Key levels (strong S/R)
            key_levels = self._identify_key_levels(support_levels + resistance_levels)
            
            analysis = TimeframeAnalysis(
                timeframe=timeframe,
                trend=trend,
                momentum=momentum,
                volatility=volatility,
                rsi=rsi,
                macd_signal=macd_signal,
                bb_position=bb_position,
                atr=atr,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                pivot_points=pivot_points,
                market_structure=market_structure,
                key_levels=key_levels,
                timestamp=datetime.now()
            )
            
            logger.debug(f"Timeframe analysis completed for {timeframe.value}")
            return analysis
            
        except Exception as e:
            logger.error(f"Single timeframe analysis failed for {timeframe.value}: {e}")
            raise
    
    def _analyze_trend(self, data: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and strength"""
        try:
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            
            # Multiple trend detection methods
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            sma_200 = talib.SMA(closes, timeperiod=50)  # Using shorter period for demo
            
            # ADX for trend strength
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
            
            # EMA slope analysis
            ema_20 = talib.EMA(closes, timeperiod=20)
            ema_slope = (ema_20[-1] - ema_20[-5]) / 5 if len(ema_20) >= 5 else 0
            
            # Price position relative to MAs
            price_vs_sma20 = (closes[-1] - sma_20[-1]) / sma_20[-1] if sma_20[-1] != 0 else 0
            price_vs_sma50 = (closes[-1] - sma_50[-1]) / sma_50[-1] if sma_50[-1] != 0 else 0
            
            # Trend scoring
            bullish_score = 0
            bearish_score = 0
            
            # MA alignment
            if sma_20[-1] > sma_50[-1] > sma_200[-1]:
                bullish_score += 2
            elif sma_20[-1] < sma_50[-1] < sma_200[-1]:
                bearish_score += 2
            
            # Price vs MA
            if price_vs_sma20 > 0 and price_vs_sma50 > 0:
                bullish_score += 1
            elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
                bearish_score += 1
            
            # EMA slope
            if ema_slope > 0:
                bullish_score += 1
            elif ema_slope < 0:
                bearish_score += 1
            
            # ADX strength
            if current_adx > 25:
                if bullish_score > bearish_score:
                    bullish_score += 1
                elif bearish_score > bullish_score:
                    bearish_score += 1
            
            # Determine trend
            if bullish_score - bearish_score >= 3:
                trend = TrendDirection.STRONG_BULLISH
                strength = min(1.0, (bullish_score - bearish_score) / 4.0)
            elif bullish_score > bearish_score:
                trend = TrendDirection.BULLISH
                strength = min(1.0, (bullish_score - bearish_score) / 3.0)
            elif bearish_score - bullish_score >= 3:
                trend = TrendDirection.STRONG_BEARISH
                strength = min(1.0, (bearish_score - bullish_score) / 4.0)
            elif bearish_score > bullish_score:
                trend = TrendDirection.BEARISH
                strength = min(1.0, (bearish_score - bullish_score) / 3.0)
            else:
                trend = TrendDirection.NEUTRAL
                strength = 0.0
            
            return trend, strength
            
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            return TrendDirection.NEUTRAL, 0.0
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            closes = data['close'].values
            volumes = data['volume'].values
            
            # RSI momentum
            rsi = talib.RSI(closes, timeperiod=14)
            rsi_momentum = (rsi[-1] - 50) / 50  # Normalize to -1 to 1
            
            # Rate of Change
            roc = talib.ROC(closes, timeperiod=10)
            roc_momentum = roc[-1] / 10 if not np.isnan(roc[-1]) else 0
            
            # Stochastic momentum
            slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values)
            stoch_momentum = ((slowk[-1] + slowd[-1]) / 2 - 50) / 50 if not np.isnan(slowk[-1]) else 0
            
            # Volume momentum
            volume_sma = talib.SMA(volumes, timeperiod=20)
            volume_momentum = (volumes[-1] - volume_sma[-1]) / volume_sma[-1] if volume_sma[-1] != 0 else 0
            
            # Combined momentum score
            momentum_score = (rsi_momentum * 0.4 + roc_momentum * 0.3 + 
                            stoch_momentum * 0.2 + volume_momentum * 0.1)
            
            return max(-1.0, min(1.0, momentum_score))
            
        except Exception as e:
            logger.warning(f"Momentum calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility score"""
        try:
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            
            # ATR-based volatility
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            atr_volatility = atr[-1] / closes[-1] if closes[-1] != 0 else 0
            
            # Bollinger Band width
            upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] != 0 else 0
            
            # Historical volatility (standard deviation of returns)
            returns = np.diff(np.log(closes))
            hist_volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
            
            # Combined volatility score (normalized)
            volatility_score = (atr_volatility * 0.4 + bb_width * 0.3 + hist_volatility * 0.3)
            
            return min(1.0, volatility_score * 10)  # Scale to reasonable range
            
        except Exception as e:
            logger.warning(f"Volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_rsi(self, data: pd.DataFrame) -> float:
        """Calculate RSI"""
        try:
            closes = data['close'].values
            rsi = talib.RSI(closes, timeperiod=self.config.rsi_period)
            return rsi[-1] if not np.isnan(rsi[-1]) else 50.0
        except Exception as e:
            logger.warning(f"RSI calculation failed: {e}")
            return 50.0
    
    def _analyze_macd(self, data: pd.DataFrame) -> str:
        """Analyze MACD signal"""
        try:
            closes = data['close'].values
            macd, macd_signal, macd_hist = talib.MACD(
                closes, 
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow, 
                signalperiod=self.config.macd_signal
            )
            
            if np.isnan(macd[-1]) or np.isnan(macd_signal[-1]):
                return "neutral"
            
            if macd[-1] > macd_signal[-1] and macd_hist[-1] > 0:
                return "bullish"
            elif macd[-1] < macd_signal[-1] and macd_hist[-1] < 0:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.warning(f"MACD analysis failed: {e}")
            return "neutral"
    
    def _analyze_bollinger_bands(self, data: pd.DataFrame) -> float:
        """Analyze Bollinger Bands position"""
        try:
            closes = data['close'].values
            upper, middle, lower = talib.BBANDS(
                closes, 
                timeperiod=self.config.bb_period,
                nbdevup=self.config.bb_std, 
                nbdevdn=self.config.bb_std
            )
            
            if np.isnan(upper[-1]) or np.isnan(lower[-1]):
                return 0.5
            
            # Normalize position between 0 (lower band) and 1 (upper band)
            position = (closes[-1] - lower[-1]) / (upper[-1] - lower[-1])
            return max(0.0, min(1.0, position))
            
        except Exception as e:
            logger.warning(f"Bollinger Bands analysis failed: {e}")
            return 0.5
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            atr = talib.ATR(highs, lows, closes, timeperiod=self.config.atr_period)
            return atr[-1] if not np.isnan(atr[-1]) else 0.0
            
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}")
            return 0.0
    
    def _find_support_resistance(self, data: pd.DataFrame, timeframe: Timeframe) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Find support and resistance levels using multiple methods"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            support_levels = []
            resistance_levels = []
            
            # Method 1: Recent swing highs and lows
            swing_highs, swing_lows = self._find_swing_points(highs, lows)
            
            # Method 2: Price clustering
            cluster_levels = self._find_price_clusters(closes)
            
            # Method 3: Fibonacci retracement (if enough data)
            fib_levels = self._calculate_fibonacci_levels(highs, lows)
            
            # Combine all levels
            all_levels = set()
            
            # Add swing points
            for level in swing_highs:
                all_levels.add((level, 'resistance'))
            for level in swing_lows:
                all_levels.add((level, 'support'))
            
            # Add cluster levels
            for level, strength in cluster_levels:
                # Determine if support or resistance based on current price
                if level < closes[-1]:
                    all_levels.add((level, 'support'))
                else:
                    all_levels.add((level, 'resistance'))
            
            # Add Fibonacci levels
            for level in fib_levels.values():
                if level < closes[-1]:
                    all_levels.add((level, 'support'))
                else:
                    all_levels.add((level, 'resistance'))
            
            # Create SupportResistanceLevel objects
            current_time = datetime.now()
            for level, level_type in all_levels:
                # Calculate strength based on various factors
                strength = self._calculate_level_strength(level, data, level_type)
                
                if strength > 0.3:  # Minimum strength threshold
                    sr_level = SupportResistanceLevel(
                        level=level,
                        strength=strength,
                        timeframe=timeframe,
                        touches=self._count_level_touches(level, data, level_type),
                        last_touch=current_time
                    )
                    
                    if level_type == 'support':
                        support_levels.append(sr_level)
                    else:
                        resistance_levels.append(sr_level)
            
            # Sort by strength
            support_levels.sort(key=lambda x: x.strength, reverse=True)
            resistance_levels.sort(key=lambda x: x.strength, reverse=True)
            
            # Keep only top levels
            support_levels = support_levels[:10]
            resistance_levels = resistance_levels[:10]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.warning(f"Support/resistance finding failed: {e}")
            return [], []
    
    def _find_swing_points(self, highs: np.ndarray, lows: np.ndarray, window: int = 5) -> Tuple[List[float], List[float]]:
        """Find swing highs and lows"""
        try:
            swing_highs = []
            swing_lows = []
            
            for i in range(window, len(highs) - window):
                # Check for swing high
                if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] > highs[i+j] for j in range(1, window+1)):
                    swing_highs.append(highs[i])
                
                # Check for swing low
                if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] < lows[i+j] for j in range(1, window+1)):
                    swing_lows.append(lows[i])
            
            return swing_highs, swing_lows
            
        except Exception as e:
            logger.warning(f"Swing points finding failed: {e}")
            return [], []
    
    def _find_price_clusters(self, prices: np.ndarray, bin_size: float = 0.001) -> List[Tuple[float, float]]:
        """Find price clusters using histogram"""
        try:
            if len(prices) == 0:
                return []
            
            # Create price bins
            price_min = np.min(prices)
            price_max = np.max(prices)
            num_bins = int((price_max - price_min) / bin_size) + 1
            
            if num_bins <= 0:
                return []
            
            # Create histogram
            hist, bin_edges = np.histogram(prices, bins=num_bins)
            
            # Find clusters (bins with high frequency)
            clusters = []
            threshold = np.mean(hist) + np.std(hist)
            
            for i, count in enumerate(hist):
                if count > threshold:
                    cluster_price = (bin_edges[i] + bin_edges[i+1]) / 2
                    cluster_strength = min(1.0, count / np.max(hist))
                    clusters.append((cluster_price, cluster_strength))
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Price clustering failed: {e}")
            return []
    
    def _calculate_fibonacci_levels(self, highs: np.ndarray, lows: np.ndarray) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return {}
            
            recent_high = np.max(highs[-50:])  # Last 50 periods
            recent_low = np.min(lows[-50:])
            
            price_range = recent_high - recent_low
            
            fib_levels = {
                '0.0': recent_low,
                '0.236': recent_high - price_range * 0.236,
                '0.382': recent_high - price_range * 0.382,
                '0.5': recent_high - price_range * 0.5,
                '0.618': recent_high - price_range * 0.618,
                '0.786': recent_high - price_range * 0.786,
                '1.0': recent_high
            }
            
            return fib_levels
            
        except Exception as e:
            logger.warning(f"Fibonacci calculation failed: {e}")
            return {}
    
    def _calculate_level_strength(self, level: float, data: pd.DataFrame, level_type: str) -> float:
        """Calculate strength of support/resistance level"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values
            
            strength = 0.0
            
            # Touch count strength
            touch_count = self._count_level_touches(level, data, level_type)
            strength += min(0.4, touch_count * 0.1)
            
            # Volume strength
            touch_indices = self._find_touch_indices(level, data, level_type)
            if touch_indices:
                touch_volumes = volumes[touch_indices]
                avg_volume = np.mean(volumes)
                volume_strength = np.mean(touch_volumes) / avg_volume if avg_volume > 0 else 1.0
                strength += min(0.3, (volume_strength - 1) * 0.15)
            
            # Recency strength (more recent touches are stronger)
            if touch_indices:
                most_recent_touch = max(touch_indices)
                recency_ratio = most_recent_touch / len(data)
                strength += min(0.3, recency_ratio * 0.3)
            
            return min(1.0, strength)
            
        except Exception as e:
            logger.warning(f"Level strength calculation failed: {e}")
            return 0.0
    
    def _count_level_touches(self, level: float, data: pd.DataFrame, level_type: str) -> int:
        """Count how many times price touched the level"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            tolerance = level * 0.001  # 0.1% tolerance
            
            if level_type == 'support':
                touches = np.sum(np.abs(lows - level) <= tolerance)
            else:  # resistance
                touches = np.sum(np.abs(highs - level) <= tolerance)
            
            return int(touches)
            
        except Exception as e:
            logger.warning(f"Touch counting failed: {e}")
            return 0
    
    def _find_touch_indices(self, level: float, data: pd.DataFrame, level_type: str) -> List[int]:
        """Find indices where price touched the level"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            tolerance = level * 0.001
            
            if level_type == 'support':
                touch_indices = np.where(np.abs(lows - level) <= tolerance)[0]
            else:  # resistance
                touch_indices = np.where(np.abs(highs - level) <= tolerance)[0]
            
            return touch_indices.tolist()
            
        except Exception as e:
            logger.warning(f"Touch indices finding failed: {e}")
            return []
    
    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points"""
        try:
            # Use previous day's data for daily pivots
            if len(data) < 2:
                return {}
            
            prev_high = data['high'].iloc[-2]
            prev_low = data['low'].iloc[-2]
            prev_close = data['close'].iloc[-2]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1,
                'r2': r2,
                'r3': r3,
                's1': s1,
                's2': s2,
                's3': s3
            }
            
        except Exception as e:
            logger.warning(f"Pivot points calculation failed: {e}")
            return {}
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> MarketStructure:
        """Analyze market structure (trending, ranging, etc.)"""
        try:
            closes = data['close'].values
            highs = data['high'].values
            lows = data['low'].values
            
            # Calculate trend using linear regression
            x = np.arange(len(closes))
            slope, _, r_value, _, _ = stats.linregress(x, closes)
            
            # ADX for trend strength
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            current_adx = adx[-1] if not np.isnan(adx[-1]) else 0
            
            # Price range analysis
            price_range = (np.max(highs[-20:]) - np.min(lows[-20:])) / closes[-1] if closes[-1] != 0 else 0
            
            # Determine market structure
            if current_adx > 25:  # Strong trend
                if slope > 0:
                    return MarketStructure.UPTREND
                else:
                    return MarketStructure.DOWNTREND
            elif current_adx < 20 and price_range < 0.02:  # Low volatility and weak trend
                return MarketStructure.RANGE
            elif current_adx > 20 and price_range > 0.03:  # High volatility
                return MarketStructure.BREAKOUT
            else:
                return MarketStructure.RANGE
                
        except Exception as e:
            logger.warning(f"Market structure analysis failed: {e}")
            return MarketStructure.RANGE
    
    def _identify_key_levels(self, levels: List[SupportResistanceLevel]) -> List[float]:
        """Identify key levels (strongest support/resistance)"""
        try:
            # Filter strong levels
            strong_levels = [level for level in levels if level.strength > 0.6]
            
            # Sort by strength and get levels
            strong_levels.sort(key=lambda x: x.strength, reverse=True)
            key_prices = [level.level for level in strong_levels[:5]]  # Top 5 levels
            
            return key_prices
            
        except Exception as e:
            logger.warning(f"Key levels identification failed: {e}")
            return []
    
    def calculate_confluence(self) -> ConfluenceScore:
        """Calculate confluence across all timeframes"""
        try:
            bullish_scores = []
            bearish_scores = []
            neutral_scores = []
            conflicting_tfs = []
            
            for timeframe, analysis in self.analysis_results.items():
                weight = self.config.timeframe_weights.get(timeframe, 0.1)
                
                # Convert trend to scores
                if analysis.trend in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH]:
                    bullish_scores.append(weight)
                elif analysis.trend in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH]:
                    bearish_scores.append(weight)
                else:
                    neutral_scores.append(weight)
                
                # Check for conflicts with higher timeframes
                if timeframe != max(self.config.timeframe_weights.keys(), 
                                  key=lambda x: self.config.timeframe_weights[x]):
                    higher_tf_bias = self._get_higher_timeframe_bias(timeframe)
                    if higher_tf_bias and higher_tf_bias != analysis.trend:
                        conflicting_tfs.append(timeframe)
            
            # Calculate total scores
            total_bullish = sum(bullish_scores)
            total_bearish = sum(bearish_scores)
            total_neutral = sum(neutral_scores)
            
            # Determine dominant bias
            if total_bullish > total_bearish and total_bullish > total_neutral:
                dominant_bias = TrendDirection.BULLISH
                strength = total_bullish
            elif total_bearish > total_bullish and total_bearish > total_neutral:
                dominant_bias = TrendDirection.BEARISH
                strength = total_bearish
            else:
                dominant_bias = TrendDirection.NEUTRAL
                strength = total_neutral
            
            confluence = ConfluenceScore(
                bullish_score=total_bullish,
                bearish_score=total_bearish,
                neutral_score=total_neutral,
                dominant_bias=dominant_bias,
                strength=strength,
                conflicting_timeframes=conflicting_tfs
            )
            
            return confluence
            
        except Exception as e:
            logger.error(f"Confluence calculation failed: {e}")
            return ConfluenceScore(0, 0, 1, TrendDirection.NEUTRAL, 0, [])
    
    def _get_higher_timeframe_bias(self, current_tf: Timeframe) -> Optional[TrendDirection]:
        """Get bias from higher timeframes"""
        try:
            higher_tfs = [tf for tf in self.config.timeframe_weights.keys() 
                         if self.config.timeframe_weights[tf] > self.config.timeframe_weights.get(current_tf, 0)]
            
            if not higher_tfs:
                return None
            
            # Get the highest timeframe analysis
            highest_tf = max(higher_tfs, key=lambda x: self.config.timeframe_weights[x])
            if highest_tf in self.analysis_results:
                return self.analysis_results[highest_tf].trend
            
            return None
            
        except Exception as e:
            logger.warning(f"Higher timeframe bias check failed: {e}")
            return None
    
    async def generate_signal(self, symbol: str = "EUR/USD") -> MultiTimeframeSignal:
        """Generate multi-timeframe trading signal"""
        try:
            # Ensure we have fresh analysis
            await self.analyze_all_timeframes()
            
            # Calculate confluence
            confluence = self.calculate_confluence()
            
            # Generate signal based on confluence
            if confluence.strength < self.config.min_confluence_score:
                # No strong signal
                signal = self._create_neutral_signal(symbol, confluence)
            else:
                signal = self._create_trading_signal(symbol, confluence)
            
            # Store signal in history
            self.historical_signals.append(signal)
            
            logger.info(f"Generated MTF signal: {signal.primary_action} (confidence: {signal.confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_error_signal(symbol, str(e))
    
    def _create_trading_signal(self, symbol: str, confluence: ConfluenceScore) -> MultiTimeframeSignal:
        """Create trading signal based on confluence"""
        try:
            # Get key levels from all timeframes
            all_support_levels = []
            all_resistance_levels = []
            
            for analysis in self.analysis_results.values():
                all_support_levels.extend(analysis.support_levels)
                all_resistance_levels.extend(analysis.resistance_levels)
            
            # Find strongest levels
            strong_support = [level for level in all_support_levels if level.strength > 0.7]
            strong_resistance = [level for level in all_resistance_levels if level.strength > 0.7]
            
            # Sort and get top levels
            strong_support.sort(key=lambda x: x.strength, reverse=True)
            strong_resistance.sort(key=lambda x: x.strength, reverse=True)
            
            key_support = strong_support[0].level if strong_support else None
            key_resistance = strong_resistance[0].level if strong_resistance else None
            
            # Determine entry zones
            entry_zones = self._calculate_entry_zones(strong_support, strong_resistance, confluence.dominant_bias)
            
            # Calculate risk-reward ratio
            risk_reward = self._calculate_risk_reward_ratio(entry_zones, key_support, key_resistance, confluence.dominant_bias)
            
            # Create timeframe confluence map
            tf_confluence = {}
            for timeframe, analysis in self.analysis_results.items():
                tf_confluence[timeframe] = analysis.trend.value
            
            signal = MultiTimeframeSignal(
                symbol=symbol,
                primary_action=confluence.dominant_bias.value.replace('_', ''),
                confidence=confluence.strength,
                timeframe_confluence=tf_confluence,
                key_levels={
                    'support': key_support,
                    'resistance': key_resistance
                },
                trend_alignment=self._calculate_trend_alignment(),
                momentum_score=self._calculate_momentum_alignment(),
                risk_reward_ratio=risk_reward,
                entry_zones=entry_zones,
                timestamp=datetime.now(),
                metadata={
                    'confluence_score': confluence.__dict__,
                    'conflicting_timeframes': [tf.value for tf in confluence.conflicting_timeframes]
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Trading signal creation failed: {e}")
            return self._create_neutral_signal(symbol, confluence)
    
    def _create_neutral_signal(self, symbol: str, confluence: ConfluenceScore) -> MultiTimeframeSignal:
        """Create neutral/hold signal"""
        return MultiTimeframeSignal(
            symbol=symbol,
            primary_action="hold",
            confidence=confluence.neutral_score,
            timeframe_confluence={tf: analysis.trend.value 
                                for tf, analysis in self.analysis_results.items()},
            key_levels={},
            trend_alignment="mixed",
            momentum_score=0.0,
            risk_reward_ratio=1.0,
            entry_zones=[],
            timestamp=datetime.now(),
            metadata={'reason': 'low_confluence'}
        )
    
    def _create_error_signal(self, symbol: str, error: str) -> MultiTimeframeSignal:
        """Create error signal"""
        return MultiTimeframeSignal(
            symbol=symbol,
            primary_action="hold",
            confidence=0.0,
            timeframe_confluence={},
            key_levels={},
            trend_alignment="error",
            momentum_score=0.0,
            risk_reward_ratio=1.0,
            entry_zones=[],
            timestamp=datetime.now(),
            metadata={'error': error}
        )
    
    def _calculate_entry_zones(self, support_levels: List[SupportResistanceLevel],
                             resistance_levels: List[SupportResistanceLevel],
                             bias: TrendDirection) -> List[Tuple[float, float]]:
        """Calculate entry zones based on bias"""
        try:
            entry_zones = []
            
            if bias in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH]:
                # For bullish bias, look for entries near support
                for level in support_levels[:3]:  # Top 3 support levels
                    zone_low = level.level * 0.998  # 0.2% below level
                    zone_high = level.level * 1.002  # 0.2% above level
                    entry_zones.append((zone_low, zone_high))
            
            elif bias in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH]:
                # For bearish bias, look for entries near resistance
                for level in resistance_levels[:3]:  # Top 3 resistance levels
                    zone_low = level.level * 0.998
                    zone_high = level.level * 1.002
                    entry_zones.append((zone_low, zone_high))
            
            return entry_zones
            
        except Exception as e:
            logger.warning(f"Entry zone calculation failed: {e}")
            return []
    
    def _calculate_risk_reward_ratio(self, entry_zones: List[Tuple[float, float]],
                                   key_support: float, key_resistance: float,
                                   bias: TrendDirection) -> float:
        """Calculate risk-reward ratio"""
        try:
            if not entry_zones or key_support is None or key_resistance is None:
                return 1.0
            
            # Use first entry zone for calculation
            entry_low, entry_high = entry_zones[0]
            entry_price = (entry_low + entry_high) / 2
            
            if bias in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH]:
                # For long trades: target resistance, stop at support
                target = key_resistance
                stop_loss = key_support
            else:
                # For short trades: target support, stop at resistance
                target = key_support
                stop_loss = key_resistance
            
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            
            if risk > 0:
                return reward / risk
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Risk-reward calculation failed: {e}")
            return 1.0
    
    def _calculate_trend_alignment(self) -> str:
        """Calculate how well timeframes are aligned"""
        try:
            trends = [analysis.trend for analysis in self.analysis_results.values()]
            
            bullish_count = sum(1 for t in trends if t in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH])
            bearish_count = sum(1 for t in trends if t in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH])
            
            total = len(trends)
            
            if bullish_count == total:
                return "perfect_bullish"
            elif bearish_count == total:
                return "perfect_bearish"
            elif bullish_count >= total * 0.7:
                return "strong_bullish"
            elif bearish_count >= total * 0.7:
                return "strong_bearish"
            elif abs(bullish_count - bearish_count) <= 1:
                return "mixed"
            else:
                return "weak_alignment"
                
        except Exception as e:
            logger.warning(f"Trend alignment calculation failed: {e}")
            return "unknown"
    
    def _calculate_momentum_alignment(self) -> float:
        """Calculate momentum alignment across timeframes"""
        try:
            momentums = [analysis.momentum for analysis in self.analysis_results.values()]
            return np.mean(momentums) if momentums else 0.0
        except Exception as e:
            logger.warning(f"Momentum alignment calculation failed: {e}")
            return 0.0
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of multi-timeframe analysis"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'timeframes_analyzed': len(self.analysis_results),
                'confluence': self.calculate_confluence().__dict__,
                'market_conditions': {},
                'key_levels': {},
                'signals_generated': len(self.historical_signals)
            }
            
            # Add per-timeframe summary
            for timeframe, analysis in self.analysis_results.items():
                summary['market_conditions'][timeframe.value] = {
                    'trend': analysis.trend.value,
                    'momentum': analysis.momentum,
                    'volatility': analysis.volatility,
                    'market_structure': analysis.market_structure.value,
                    'rsi': analysis.rsi,
                    'key_support_levels': [level.to_dict() for level in analysis.support_levels[:3]],
                    'key_resistance_levels': [level.to_dict() for level in analysis.resistance_levels[:3]]
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Analysis summary failed: {e}")
            return {}
    
    def save_analysis(self, filename: str = "mtf_analysis.json") -> None:
        """Save analysis results to file"""
        try:
            analysis_data = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__,
                'analysis_results': {},
                'summary': self.get_analysis_summary()
            }
            
            # Convert analysis results to serializable format
            for timeframe, analysis in self.analysis_results.items():
                analysis_data['analysis_results'][timeframe.value] = {
                    'trend': analysis.trend.value,
                    'momentum': analysis.momentum,
                    'volatility': analysis.volatility,
                    'rsi': analysis.rsi,
                    'macd_signal': analysis.macd_signal,
                    'bb_position': analysis.bb_position,
                    'atr': analysis.atr,
                    'market_structure': analysis.market_structure.value,
                    'support_levels': [level.to_dict() for level in analysis.support_levels],
                    'resistance_levels': [level.to_dict() for level in analysis.resistance_levels],
                    'pivot_points': analysis.pivot_points,
                    'key_levels': analysis.key_levels
                }
            
            with open(filename, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"Analysis saved to {filename}")
            
        except Exception as e:
            logger.error(f"Analysis saving failed: {e}")

# Example usage and testing
async def main():
    """Example usage of the AdvancedMultiTimeframeAnalyzer"""
    
    # Generate sample market data for multiple timeframes
    print("=== Generating Sample Market Data ===")
    np.random.seed(42)
    
    # Create sample data for different timeframes
    timeframes = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
    sample_data = {}
    
    base_price = 1.1000
    for tf in timeframes:
        # Different number of periods for different timeframes
        if tf == Timeframe.M15:
            periods = 2000
        elif tf == Timeframe.H1:
            periods = 1000
        elif tf == Timeframe.H4:
            periods = 500
        else:  # D1
            periods = 250
        
        # Generate price data with trends
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=tf.value)
        returns = np.random.normal(0.0001, 0.005, periods)
        
        # Add some structure
        trend = np.sin(np.arange(periods) * 2 * np.pi / 100) * 0.002
        prices = base_price * np.exp(np.cumsum(returns + trend))
        
        df = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001 + np.random.random(periods) * 0.0005,
            'low': prices * 0.998 - np.random.random(periods) * 0.0005,
            'close': prices,
            'volume': np.random.lognormal(10, 1, periods)
        }, index=dates)
        
        sample_data[tf] = df
        print(f"Generated {len(df)} {tf.value} candles")
    
    # Configure multi-timeframe analyzer
    config = MTFConfig(
        timeframes=timeframes,
        lookback_periods=200,
        timeframe_weights={
            Timeframe.M15: 0.15,
            Timeframe.H1: 0.25,
            Timeframe.H4: 0.3,
            Timeframe.D1: 0.3
        }
    )
    
    # Initialize analyzer
    analyzer = AdvancedMultiTimeframeAnalyzer(config)
    
    # Add market data
    for tf, data in sample_data.items():
        analyzer.add_market_data(tf, data)
    
    print("\n=== Running Multi-Timeframe Analysis ===")
    
    # Analyze all timeframes
    analysis_results = await analyzer.analyze_all_timeframes()
    
    print(f"Analysis completed for {len(analysis_results)} timeframes")
    
    # Display analysis results
    print("\n=== Timeframe Analysis Results ===")
    for timeframe, analysis in analysis_results.items():
        print(f"\n{timeframe.value}:")
        print(f"  Trend: {analysis.trend.value}")
        print(f"  Momentum: {analysis.momentum:.3f}")
        print(f"  Volatility: {analysis.volatility:.3f}")
        print(f"  RSI: {analysis.rsi:.1f}")
        print(f"  Market Structure: {analysis.market_structure.value}")
        print(f"  Support Levels: {len(analysis.support_levels)}")
        print(f"  Resistance Levels: {len(analysis.resistance_levels)}")
        print(f"  Key Levels: {analysis.key_levels[:3]}")
    
    # Calculate confluence
    print("\n=== Multi-Timeframe Confluence ===")
    confluence = analyzer.calculate_confluence()
    print(f"Dominant Bias: {confluence.dominant_bias.value}")
    print(f"Confluence Strength: {confluence.strength:.2f}")
    print(f"Bullish Score: {confluence.bullish_score:.2f}")
    print(f"Bearish Score: {confluence.bearish_score:.2f}")
    print(f"Neutral Score: {confluence.neutral_score:.2f}")
    print(f"Conflicting TFs: {[tf.value for tf in confluence.conflicting_timeframes]}")
    
    # Generate trading signal
    print("\n=== Generating Trading Signal ===")
    signal = await analyzer.generate_signal("EUR/USD")
    
    print(f"Primary Action: {signal.primary_action}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Trend Alignment: {signal.trend_alignment}")
    print(f"Momentum Score: {signal.momentum_score:.3f}")
    print(f"Risk-Reward Ratio: {signal.risk_reward_ratio:.2f}")
    
    print("\nTimeframe Confluence:")
    for tf, trend in signal.timeframe_confluence.items():
        print(f"  {tf.value}: {trend}")
    
    if signal.entry_zones:
        print(f"\nEntry Zones:")
        for i, (low, high) in enumerate(signal.entry_zones[:2]):  # Show first 2
            print(f"  Zone {i+1}: {low:.4f} - {high:.4f}")
    
    # Get analysis summary
    print("\n=== Analysis Summary ===")
    summary = analyzer.get_analysis_summary()
    print(f"Timeframes Analyzed: {summary['timeframes_analyzed']}")
    print(f"Signals Generated: {summary['signals_generated']}")
    
    # Save analysis
    analyzer.save_analysis("multi_timeframe_analysis.json")
    print("\n=== Analysis Saved ===")
    
    print("\n=== Multi-Timeframe Analysis Test Completed ===")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run async main
    asyncio.run(main())
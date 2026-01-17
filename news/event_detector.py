"""
Advanced Event Detector for FOREX TRADING BOT
Real-time economic and market event detection with pattern recognition
"""

import logging
import pandas as pd
import numpy as np
import json
import re
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import talib

logger = logging.getLogger(__name__)

class EventType(Enum):
    ECONOMIC_RELEASE = "economic_release"
    CENTRAL_BANK = "central_bank"
    GEOPOLITICAL = "geopolitical"
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_EVENT = "liquidity_event"
    TECHNICAL_BREAKOUT = "technical_breakout"
    NEWS_EVENT = "news_event"
    SOCIAL_SENTIMENT = "social_sentiment"
    WHALE_MOVEMENT = "whale_movement"

class EventSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionMethod(Enum):
    STATISTICAL = "statistical"
    ML_ANOMALY = "ml_anomaly"
    PATTERN_RECOGNITION = "pattern_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    VOLUME_ANALYSIS = "volume_analysis"

@dataclass
class DetectedEvent:
    """Detected market event structure"""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    symbol: str
    timestamp: datetime
    confidence: float
    description: str
    detection_method: DetectionMethod
    impact_score: float
    duration_estimate: timedelta
    metadata: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)

@dataclass
class EventPattern:
    """Pattern template for event detection"""
    name: str
    event_type: EventType
    conditions: List[Dict[str, Any]]
    severity: EventSeverity
    confidence_threshold: float
    cooldown_period: int  # minutes

@dataclass
class EventDetectorConfig:
    """Configuration for event detector"""
    # Detection settings
    enable_statistical_detection: bool = True
    enable_ml_anomaly_detection: bool = True
    enable_pattern_recognition: bool = True
    enable_sentiment_analysis: bool = True
    
    # Thresholds
    volatility_threshold: float = 2.5  # Standard deviations
    volume_threshold: float = 3.0      # Standard deviations
    sentiment_threshold: float = 0.8   # Absolute sentiment score
    confidence_threshold: float = 0.7  # Minimum confidence
    
    # Time windows
    short_window: int = 5      # minutes
    medium_window: int = 30    # minutes
    long_window: int = 240     # minutes
    
    # ML settings
    anomaly_contamination: float = 0.1
    cluster_epsilon: float = 0.5
    min_cluster_size: int = 5
    
    # Real-time settings
    update_frequency: int = 10  # seconds
    max_events_per_minute: int = 10
    
    # Risk management
    auto_hedge_high_impact: bool = True
    max_concurrent_events: int = 5
    event_blacklist: List[str] = field(default_factory=list)

class AdvancedEventDetector:
    """
    Advanced market event detection system using multiple detection methods
    """
    
    def __init__(self, config: EventDetectorConfig = None):
        self.config = config or EventDetectorConfig()
        self.timezone = ZoneInfo("UTC")
        
        # Data storage
        self.detected_events: Dict[str, DetectedEvent] = {}
        self.event_patterns: Dict[str, EventPattern] = {}
        self.market_data_cache = defaultdict(lambda: deque(maxlen=1000))
        self.sentiment_data_cache = defaultdict(lambda: deque(maxlen=500))
        self.volume_profile = defaultdict(lambda: defaultdict(float))
        
        # ML models
        self.anomaly_detector = None
        self.volatility_models = {}
        
        # Statistical baselines
        self.baselines = defaultdict(dict)
        self.correlation_matrix = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._detection_lock = threading.Lock()
        
        # Cooldown tracking
        self.cooldown_tracker = defaultdict(float)
        
        # Initialize patterns and models
        self._initialize_patterns()
        self._initialize_ml_models()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("AdvancedEventDetector initialized")

    def _initialize_patterns(self):
        """Initialize event detection patterns"""
        patterns = [
            EventPattern(
                name="volatility_spike",
                event_type=EventType.VOLATILITY_SPIKE,
                conditions=[
                    {"metric": "volatility", "threshold": 2.5, "window": 5},
                    {"metric": "volume", "threshold": 2.0, "window": 5}
                ],
                severity=EventSeverity.HIGH,
                confidence_threshold=0.75,
                cooldown_period=30
            ),
            EventPattern(
                name="market_crash",
                event_type=EventType.MARKET_CRASH,
                conditions=[
                    {"metric": "price_change", "threshold": -2.0, "window": 10},
                    {"metric": "volume", "threshold": 2.5, "window": 10},
                    {"metric": "volatility", "threshold": 3.0, "window": 10}
                ],
                severity=EventSeverity.CRITICAL,
                confidence_threshold=0.85,
                cooldown_period=60
            ),
            EventPattern(
                name="technical_breakout",
                event_type=EventType.TECHNICAL_BREAKOUT,
                conditions=[
                    {"metric": "price_breakout", "threshold": 1.5, "window": 15},
                    {"metric": "volume_confirmation", "threshold": 1.8, "window": 15}
                ],
                severity=EventSeverity.MEDIUM,
                confidence_threshold=0.70,
                cooldown_period=45
            ),
            EventPattern(
                name="liquidity_event",
                event_type=EventType.LIQUIDITY_EVENT,
                conditions=[
                    {"metric": "spread_widening", "threshold": 3.0, "window": 5},
                    {"metric": "depth_reduction", "threshold": 2.5, "window": 5}
                ],
                severity=EventSeverity.HIGH,
                confidence_threshold=0.80,
                cooldown_period=30
            )
        ]
        
        for pattern in patterns:
            self.event_patterns[pattern.name] = pattern

    def _initialize_ml_models(self):
        """Initialize machine learning models for anomaly detection"""
        try:
            # Isolation Forest for general anomaly detection
            self.anomaly_detector = IsolationForest(
                contamination=self.config.anomaly_contamination,
                random_state=42,
                n_estimators=100
            )
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")

    def _start_background_tasks(self):
        """Start background detection and monitoring tasks"""
        # Real-time detection loop
        detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        detection_thread.start()
        
        # Baseline calculation
        baseline_thread = threading.Thread(target=self._baseline_calculation_loop, daemon=True)
        baseline_thread.start()
        
        # Pattern learning
        learning_thread = threading.Thread(target=self._pattern_learning_loop, daemon=True)
        learning_thread.start()
        
        # Event correlation
        correlation_thread = threading.Thread(target=self._correlation_loop, daemon=True)
        correlation_thread.start()

    def update_market_data(self, symbol: str, data: Dict[str, Any]):
        """Update market data for event detection"""
        try:
            timestamp = datetime.now(self.timezone)
            data_point = {
                'timestamp': timestamp,
                'symbol': symbol,
                'price': data.get('price', 0),
                'volume': data.get('volume', 0),
                'spread': data.get('spread', 0),
                'high': data.get('high', 0),
                'low': data.get('low', 0),
                'bid': data.get('bid', 0),
                'ask': data.get('ask', 0)
            }
            
            with self._lock:
                self.market_data_cache[symbol].append(data_point)
            
            # Update volume profile
            self._update_volume_profile(symbol, data_point)
            
        except Exception as e:
            logger.error(f"Market data update failed for {symbol}: {e}")

    def update_sentiment_data(self, symbol: str, sentiment_score: float, confidence: float):
        """Update sentiment data for event detection"""
        try:
            sentiment_point = {
                'timestamp': datetime.now(self.timezone),
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'confidence': confidence
            }
            
            with self._lock:
                self.sentiment_data_cache[symbol].append(sentiment_point)
                
        except Exception as e:
            logger.error(f"Sentiment data update failed for {symbol}: {e}")

    def _update_volume_profile(self, symbol: str, data_point: Dict[str, Any]):
        """Update volume profile for the symbol"""
        try:
            price = data_point['price']
            volume = data_point['volume']
            
            # Round price to nearest pip for volume profile
            price_level = round(price, 4)
            self.volume_profile[symbol][price_level] += volume
            
        except Exception as e:
            logger.error(f"Volume profile update failed: {e}")

    def detect_events_real_time(self, symbol: str) -> List[DetectedEvent]:
        """Perform real-time event detection for a symbol"""
        events = []
        
        try:
            # Check if in cooldown period
            if self._is_in_cooldown(symbol):
                return events
            
            # Get recent market data
            recent_data = list(self.market_data_cache.get(symbol, []))[-100:]  # Last 100 points
            if len(recent_data) < 20:  # Need minimum data
                return events
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(recent_data)
            df.set_index('timestamp', inplace=True)
            
            # Multiple detection methods
            if self.config.enable_statistical_detection:
                statistical_events = self._statistical_detection(symbol, df)
                events.extend(statistical_events)
            
            if self.config.enable_ml_anomaly_detection:
                ml_events = self._ml_anomaly_detection(symbol, df)
                events.extend(ml_events)
            
            if self.config.enable_pattern_recognition:
                pattern_events = self._pattern_recognition(symbol, df)
                events.extend(pattern_events)
            
            if self.config.enable_sentiment_analysis:
                sentiment_events = self._sentiment_based_detection(symbol)
                events.extend(sentiment_events)
            
            # Filter by confidence and apply cooldown
            filtered_events = []
            for event in events:
                if event.confidence >= self.config.confidence_threshold:
                    filtered_events.append(event)
                    self.cooldown_tracker[symbol] = time.time()
            
            return filtered_events
            
        except Exception as e:
            logger.error(f"Real-time event detection failed for {symbol}: {e}")
            return []

    def _statistical_detection(self, symbol: str, df: pd.DataFrame) -> List[DetectedEvent]:
        """Statistical methods for event detection"""
        events = []
        
        try:
            # Calculate metrics
            returns = df['price'].pct_change().dropna()
            volume = df['volume']
            spread = df['spread']
            
            # Volatility spike detection
            volatility = returns.rolling(window=10).std()
            current_volatility = volatility.iloc[-1] if len(volatility) > 0 else 0
            vol_zscore = self._calculate_zscore(volatility)
            
            if abs(vol_zscore) > self.config.volatility_threshold:
                event = DetectedEvent(
                    event_id=f"vol_spike_{symbol}_{int(time.time())}",
                    event_type=EventType.VOLATILITY_SPIKE,
                    severity=EventSeverity.HIGH,
                    symbol=symbol,
                    timestamp=datetime.now(self.timezone),
                    confidence=min(0.9, abs(vol_zscore) / 5.0),
                    description=f"Volatility spike detected: {current_volatility:.6f}",
                    detection_method=DetectionMethod.STATISTICAL,
                    impact_score=0.7,
                    duration_estimate=timedelta(minutes=30),
                    triggers=[f"Volatility Z-score: {vol_zscore:.2f}"]
                )
                events.append(event)
            
            # Volume anomaly detection
            volume_zscore = self._calculate_zscore(volume)
            if volume_zscore > self.config.volume_threshold:
                event = DetectedEvent(
                    event_id=f"volume_anom_{symbol}_{int(time.time())}",
                    event_type=EventType.VOLATILITY_SPIKE,
                    severity=EventSeverity.MEDIUM,
                    symbol=symbol,
                    timestamp=datetime.now(self.timezone),
                    confidence=min(0.8, volume_zscore / 4.0),
                    description=f"Volume anomaly detected: Z-score {volume_zscore:.2f}",
                    detection_method=DetectionMethod.VOLUME_ANALYSIS,
                    impact_score=0.5,
                    duration_estimate=timedelta(minutes=15),
                    triggers=[f"Volume Z-score: {volume_zscore:.2f}"]
                )
                events.append(event)
            
            # Price crash detection
            price_changes = df['price'].diff()
            large_drops = price_changes[price_changes < -0.005]  # More than 0.5% drop
            if len(large_drops) > 2:  # Multiple large drops
                event = DetectedEvent(
                    event_id=f"price_drop_{symbol}_{int(time.time())}",
                    event_type=EventType.MARKET_CRASH,
                    severity=EventSeverity.HIGH,
                    symbol=symbol,
                    timestamp=datetime.now(self.timezone),
                    confidence=0.75,
                    description="Multiple large price drops detected",
                    detection_method=DetectionMethod.STATISTICAL,
                    impact_score=0.8,
                    duration_estimate=timedelta(minutes=45),
                    triggers=[f"Large drops: {len(large_drops)}"]
                )
                events.append(event)
            
            # Spread widening (liquidity event)
            spread_zscore = self._calculate_zscore(spread)
            if spread_zscore > 2.0:
                event = DetectedEvent(
                    event_id=f"spread_widen_{symbol}_{int(time.time())}",
                    event_type=EventType.LIQUIDITY_EVENT,
                    severity=EventSeverity.MEDIUM,
                    symbol=symbol,
                    timestamp=datetime.now(self.timezone),
                    confidence=min(0.7, spread_zscore / 3.0),
                    description=f"Spread widening detected: Z-score {spread_zscore:.2f}",
                    detection_method=DetectionMethod.STATISTICAL,
                    impact_score=0.6,
                    duration_estimate=timedelta(minutes=20),
                    triggers=[f"Spread Z-score: {spread_zscore:.2f}"]
                )
                events.append(event)
            
        except Exception as e:
            logger.error(f"Statistical detection failed for {symbol}: {e}")
        
        return events

    def _ml_anomaly_detection(self, symbol: str, df: pd.DataFrame) -> List[DetectedEvent]:
        """Machine learning based anomaly detection"""
        events = []
        
        try:
            if self.anomaly_detector is None:
                return events
            
            # Prepare features for ML
            features = self._extract_ml_features(df)
            if len(features) < 10:  # Need enough data
                return events
            
            # Detect anomalies
            predictions = self.anomaly_detector.fit_predict(features)
            anomaly_scores = self.anomaly_detector.decision_function(features)
            
            # Check recent points for anomalies
            recent_predictions = predictions[-5:]  # Last 5 points
            recent_scores = anomaly_scores[-5:]
            
            if any(pred == -1 for pred in recent_predictions):
                avg_anomaly_score = np.mean([abs(score) for score in recent_scores if score < 0])
                
                event = DetectedEvent(
                    event_id=f"ml_anom_{symbol}_{int(time.time())}",
                    event_type=EventType.VOLATILITY_SPIKE,
                    severity=EventSeverity.MEDIUM,
                    symbol=symbol,
                    timestamp=datetime.now(self.timezone),
                    confidence=min(0.85, avg_anomaly_score),
                    description="ML anomaly detected in market data",
                    detection_method=DetectionMethod.ML_ANOMALY,
                    impact_score=0.6,
                    duration_estimate=timedelta(minutes=25),
                    triggers=[f"Anomaly score: {avg_anomaly_score:.3f}"]
                )
                events.append(event)
            
        except Exception as e:
            logger.error(f"ML anomaly detection failed for {symbol}: {e}")
        
        return events

    def _pattern_recognition(self, symbol: str, df: pd.DataFrame) -> List[DetectedEvent]:
        """Pattern recognition based event detection"""
        events = []
        
        try:
            # Technical pattern detection using TA-Lib
            prices = df['price'].values
            
            # RSI based overbought/oversold
            rsi = talib.RSI(prices, timeperiod=14)
            if len(rsi) > 0:
                current_rsi = rsi[-1]
                if current_rsi > 80:  # Overbought
                    event = DetectedEvent(
                        event_id=f"overbought_{symbol}_{int(time.time())}",
                        event_type=EventType.TECHNICAL_BREAKOUT,
                        severity=EventSeverity.MEDIUM,
                        symbol=symbol,
                        timestamp=datetime.now(self.timezone),
                        confidence=0.7,
                        description=f"Overbought condition: RSI {current_rsi:.1f}",
                        detection_method=DetectionMethod.PATTERN_RECOGNITION,
                        impact_score=0.5,
                        duration_estimate=timedelta(minutes=30),
                        triggers=[f"RSI: {current_rsi:.1f}"]
                    )
                    events.append(event)
                elif current_rsi < 20:  # Oversold
                    event = DetectedEvent(
                        event_id=f"oversold_{symbol}_{int(time.time())}",
                        event_type=EventType.TECHNICAL_BREAKOUT,
                        severity=EventSeverity.MEDIUM,
                        symbol=symbol,
                        timestamp=datetime.now(self.timezone),
                        confidence=0.7,
                        description=f"Oversold condition: RSI {current_rsi:.1f}",
                        detection_method=DetectionMethod.PATTERN_RECOGNITION,
                        impact_score=0.5,
                        duration_estimate=timedelta(minutes=30),
                        triggers=[f"RSI: {current_rsi:.1f}"]
                    )
                    events.append(event)
            
            # Bollinger Bands breakout
            upper, middle, lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            if len(upper) > 0 and len(lower) > 0:
                current_price = prices[-1]
                current_upper = upper[-1]
                current_lower = lower[-1]
                
                if current_price > current_upper:  # Upper breakout
                    event = DetectedEvent(
                        event_id=f"bb_breakout_up_{symbol}_{int(time.time())}",
                        event_type=EventType.TECHNICAL_BREAKOUT,
                        severity=EventSeverity.MEDIUM,
                        symbol=symbol,
                        timestamp=datetime.now(self.timezone),
                        confidence=0.75,
                        description="Upper Bollinger Band breakout",
                        detection_method=DetectionMethod.PATTERN_RECOGNITION,
                        impact_score=0.6,
                        duration_estimate=timedelta(minutes=45),
                        triggers=[f"Price: {current_price:.5f}, Upper Band: {current_upper:.5f}"]
                    )
                    events.append(event)
                elif current_price < current_lower:  # Lower breakout
                    event = DetectedEvent(
                        event_id=f"bb_breakout_down_{symbol}_{int(time.time())}",
                        event_type=EventType.TECHNICAL_BREAKOUT,
                        severity=EventSeverity.MEDIUM,
                        symbol=symbol,
                        timestamp=datetime.now(self.timezone),
                        confidence=0.75,
                        description="Lower Bollinger Band breakout",
                        detection_method=DetectionMethod.PATTERN_RECOGNITION,
                        impact_score=0.6,
                        duration_estimate=timedelta(minutes=45),
                        triggers=[f"Price: {current_price:.5f}, Lower Band: {current_lower:.5f}"]
                    )
                    events.append(event)
            
            # Volume spike with price movement
            volume = df['volume'].values
            if len(volume) > 10:
                volume_sma = talib.SMA(volume, timeperiod=10)
                if len(volume_sma) > 0:
                    current_volume = volume[-1]
                    avg_volume = volume_sma[-1]
                    
                    if current_volume > avg_volume * 2.5:  # Volume spike
                        price_change = (prices[-1] - prices[-2]) / prices[-2] * 100
                        
                        event = DetectedEvent(
                            event_id=f"volume_spike_{symbol}_{int(time.time())}",
                            event_type=EventType.VOLATILITY_SPIKE,
                            severity=EventSeverity.MEDIUM,
                            symbol=symbol,
                            timestamp=datetime.now(self.timezone),
                            confidence=0.8,
                            description=f"Volume spike with {price_change:+.2f}% price change",
                            detection_method=DetectionMethod.VOLUME_ANALYSIS,
                            impact_score=0.7,
                            duration_estimate=timedelta(minutes=20),
                            triggers=[f"Volume: {current_volume:.0f}, Avg: {avg_volume:.0f}"]
                        )
                        events.append(event)
            
        except Exception as e:
            logger.error(f"Pattern recognition failed for {symbol}: {e}")
        
        return events

    def _sentiment_based_detection(self, symbol: str) -> List[DetectedEvent]:
        """Sentiment-based event detection"""
        events = []
        
        try:
            recent_sentiment = list(self.sentiment_data_cache.get(symbol, []))[-10:]  # Last 10 points
            if len(recent_sentiment) < 5:
                return events
            
            sentiment_scores = [point['sentiment_score'] for point in recent_sentiment]
            confidence_scores = [point['confidence'] for point in recent_sentiment]
            
            # Extreme sentiment detection
            avg_sentiment = np.mean(sentiment_scores)
            avg_confidence = np.mean(confidence_scores)
            
            if abs(avg_sentiment) > self.config.sentiment_threshold and avg_confidence > 0.6:
                sentiment_type = "bullish" if avg_sentiment > 0 else "bearish"
                
                event = DetectedEvent(
                    event_id=f"sentiment_extreme_{symbol}_{int(time.time())}",
                    event_type=EventType.SOCIAL_SENTIMENT,
                    severity=EventSeverity.MEDIUM,
                    symbol=symbol,
                    timestamp=datetime.now(self.timezone),
                    confidence=avg_confidence,
                    description=f"Extreme {sentiment_type} sentiment detected",
                    detection_method=DetectionMethod.SENTIMENT_ANALYSIS,
                    impact_score=0.5,
                    duration_estimate=timedelta(minutes=60),
                    triggers=[f"Sentiment: {avg_sentiment:.3f}", f"Confidence: {avg_confidence:.3f}"]
                )
                events.append(event)
            
            # Sentiment reversal detection
            if len(sentiment_scores) >= 3:
                recent_trend = np.polyfit(range(3), sentiment_scores[-3:], 1)[0]
                if abs(recent_trend) > 0.2:  # Strong trend
                    direction = "improving" if recent_trend > 0 else "deteriorating"
                    
                    event = DetectedEvent(
                        event_id=f"sentiment_reversal_{symbol}_{int(time.time())}",
                        event_type=EventType.SOCIAL_SENTIMENT,
                        severity=EventSeverity.LOW,
                        symbol=symbol,
                        timestamp=datetime.now(self.timezone),
                        confidence=0.6,
                        description=f"Sentiment trend {direction} rapidly",
                        detection_method=DetectionMethod.SENTIMENT_ANALYSIS,
                        impact_score=0.4,
                        duration_estimate=timedelta(minutes=30),
                        triggers=[f"Trend slope: {recent_trend:.3f}"]
                    )
                    events.append(event)
            
        except Exception as e:
            logger.error(f"Sentiment-based detection failed for {symbol}: {e}")
        
        return events

    def _extract_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML anomaly detection"""
        features = []
        
        try:
            # Price-based features
            returns = df['price'].pct_change().dropna()
            volatility = returns.rolling(window=10).std()
            
            # Volume features
            volume = df['volume']
            volume_ratio = volume / volume.rolling(window=20).mean()
            
            # Spread features
            spread = df['spread']
            spread_ratio = spread / spread.rolling(window=20).mean()
            
            # Combine features
            for i in range(len(df)):
                if i >= 20:  # Ensure we have enough history
                    feature_vector = [
                        returns.iloc[i] if i < len(returns) else 0,
                        volatility.iloc[i] if i < len(volatility) else 0,
                        volume_ratio.iloc[i] if i < len(volume_ratio) else 1,
                        spread_ratio.iloc[i] if i < len(spread_ratio) else 1,
                        df['price'].iloc[i] / df['price'].iloc[i-1] if i > 0 else 1
                    ]
                    features.append(feature_vector)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
        
        return np.array(features) if features else np.array([])

    def _calculate_zscore(self, series: pd.Series) -> float:
        """Calculate Z-score for a series"""
        try:
            if len(series) < 10:
                return 0.0
            
            recent = series.iloc[-10:]  # Last 10 values
            mean = recent.mean()
            std = recent.std()
            
            if std == 0:
                return 0.0
            
            current = series.iloc[-1]
            return (current - mean) / std
            
        except Exception as e:
            logger.error(f"Z-score calculation failed: {e}")
            return 0.0

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        last_event_time = self.cooldown_tracker.get(symbol, 0)
        cooldown_period = 60  # 1 minute cooldown
        return time.time() - last_event_time < cooldown_period

    def get_recent_events(self, symbol: str = None, hours: int = 24) -> List[DetectedEvent]:
        """Get recently detected events"""
        with self._lock:
            now = datetime.now(self.timezone)
            cutoff_time = now - timedelta(hours=hours)
            
            events = []
            for event in self.detected_events.values():
                if event.timestamp >= cutoff_time:
                    if symbol is None or event.symbol == symbol:
                        events.append(event)
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            return events

    def get_high_impact_events(self, hours: int = 24) -> List[DetectedEvent]:
        """Get high and critical impact events"""
        recent_events = self.get_recent_events(hours=hours)
        return [e for e in recent_events if e.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]]

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event detection statistics"""
        try:
            recent_events = self.get_recent_events(hours=24)
            
            stats = {
                'timestamp': datetime.now(),
                'total_events_24h': len(recent_events),
                'events_by_type': defaultdict(int),
                'events_by_severity': defaultdict(int),
                'events_by_symbol': defaultdict(int),
                'avg_confidence': 0.0,
                'detection_methods': defaultdict(int)
            }
            
            if recent_events:
                for event in recent_events:
                    stats['events_by_type'][event.event_type.value] += 1
                    stats['events_by_severity'][event.severity.value] += 1
                    stats['events_by_symbol'][event.symbol] += 1
                    stats['detection_methods'][event.detection_method.value] += 1
                
                stats['avg_confidence'] = np.mean([e.confidence for e in recent_events])
            
            return stats
            
        except Exception as e:
            logger.error(f"Event statistics calculation failed: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def _detection_loop(self):
        """Background detection loop"""
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
        
        while True:
            try:
                for symbol in symbols:
                    events = self.detect_events_real_time(symbol)
                    
                    # Store detected events
                    for event in events:
                        with self._lock:
                            self.detected_events[event.event_id] = event
                        
                        # Log significant events
                        if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL]:
                            logger.info(
                                f"High impact event detected: {event.event_type.value} "
                                f"for {event.symbol} (confidence: {event.confidence:.2f})"
                            )
                
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Detection loop failed: {e}")
                time.sleep(30)

    def _baseline_calculation_loop(self):
        """Background loop for calculating statistical baselines"""
        while True:
            try:
                symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
                
                for symbol in symbols:
                    recent_data = list(self.market_data_cache.get(symbol, []))
                    if len(recent_data) > 50:
                        self._calculate_baselines(symbol, recent_data)
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Baseline calculation loop failed: {e}")
                time.sleep(60)

    def _pattern_learning_loop(self):
        """Background loop for pattern learning"""
        while True:
            try:
                # This would implement adaptive pattern learning
                # For now, it's a placeholder for future enhancement
                time.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Pattern learning loop failed: {e}")
                time.sleep(300)

    def _correlation_loop(self):
        """Background loop for event correlation analysis"""
        while True:
            try:
                self._calculate_event_correlations()
                time.sleep(900)  # Run every 15 minutes
                
            except Exception as e:
                logger.error(f"Correlation loop failed: {e}")
                time.sleep(300)

    def _calculate_baselines(self, symbol: str, data: List[Dict]):
        """Calculate statistical baselines for a symbol"""
        try:
            df = pd.DataFrame(data)
            
            baselines = {
                'volatility_mean': df['price'].pct_change().std(),
                'volume_mean': df['volume'].mean(),
                'volume_std': df['volume'].std(),
                'spread_mean': df['spread'].mean(),
                'spread_std': df['spread'].std(),
                'update_time': datetime.now(self.timezone)
            }
            
            self.baselines[symbol] = baselines
            
        except Exception as e:
            logger.error(f"Baseline calculation failed for {symbol}: {e}")

    def _calculate_event_correlations(self):
        """Calculate correlations between different events"""
        try:
            # This would implement complex event correlation analysis
            # For now, it's a simplified version
            recent_events = self.get_recent_events(hours=6)
            
            if len(recent_events) > 10:
                # Simple co-occurrence analysis
                event_pairs = defaultdict(int)
                for i, event1 in enumerate(recent_events):
                    for event2 in recent_events[i+1:]:
                        time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
                        if time_diff < 300:  # Events within 5 minutes
                            pair_key = f"{event1.event_type.value}-{event2.event_type.value}"
                            event_pairs[pair_key] += 1
                
                self.correlation_matrix = dict(event_pairs)
                
        except Exception as e:
            logger.error(f"Event correlation calculation failed: {e}")

# Example usage and testing
def main():
    """Example usage of the AdvancedEventDetector"""
    
    # Configuration
    config = EventDetectorConfig(
        enable_statistical_detection=True,
        enable_ml_anomaly_detection=True,
        enable_pattern_recognition=True,
        enable_sentiment_analysis=True,
        update_frequency=15  # 15 seconds for testing
    )
    
    # Initialize detector
    detector = AdvancedEventDetector(config)
    
    # Simulate some market data
    print("=== Simulating Market Data ===")
    symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
    
    for i in range(100):
        for symbol in symbols:
            # Simulate price data with occasional anomalies
            base_price = 1.1000 if 'EUR' in symbol else 1.3000 if 'GBP' in symbol else 150.00
            anomaly = 1.0 if i == 50 else 1.0  # Create anomaly at i=50
            
            data = {
                'price': base_price * anomaly + np.random.normal(0, 0.0001),
                'volume': max(1000, np.random.normal(5000, 2000)),
                'spread': max(0.0001, np.random.normal(0.0002, 0.0001)),
                'high': base_price * 1.0005,
                'low': base_price * 0.9995,
                'bid': base_price - 0.0001,
                'ask': base_price + 0.0001
            }
            
            detector.update_market_data(symbol, data)
            
            # Simulate sentiment data
            if i % 10 == 0:
                sentiment_score = np.random.normal(0, 0.3)
                detector.update_sentiment_data(symbol, sentiment_score, 0.8)
        
        time.sleep(0.1)
    
    # Check detected events
    print("\n=== Detected Events ===")
    recent_events = detector.get_recent_events(hours=1)
    print(f"Total events detected: {len(recent_events)}")
    
    for i, event in enumerate(recent_events[:5], 1):
        print(f"\n{i}. {event.symbol} - {event.event_type.value}")
        print(f"   Severity: {event.severity.value}")
        print(f"   Confidence: {event.confidence:.2f}")
        print(f"   Description: {event.description}")
        print(f"   Detection: {event.detection_method.value}")
    
    # Get statistics
    print("\n=== Event Statistics ===")
    stats = detector.get_event_statistics()
    print(f"Total events (24h): {stats['total_events_24h']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print("Events by type:")
    for event_type, count in stats['events_by_type'].items():
        print(f"  - {event_type}: {count}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
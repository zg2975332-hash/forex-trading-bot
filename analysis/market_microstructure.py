"""
Advanced Market Microstructure Analysis for FOREX TRADING BOT
High-frequency market data analysis, order book dynamics, and microstructure features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
import asyncio
import logging
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback statistics functions
    def skew(data):
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean([((x - mean) / std) ** 3 for x in data])
    
    stats = type('MockStats', (), {'skew': skew})()
import warnings
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications based on microstructure"""
    NORMAL = "NORMAL"
    HIGH_FREQUENCY = "HIGH_FREQUENCY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    NEWS_EVENT = "NEWS_EVENT"
    FLASH_CRASH = "FLASH_CRASH"
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_CLOSE = "MARKET_CLOSE"

class OrderBookImbalance(Enum):
    """Order book imbalance states"""
    STRONG_BID = "STRONG_BID"
    STRONG_ASK = "STRONG_ASK"
    BALANCED = "BALANCED"
    UNCERTAIN = "UNCERTAIN"

@dataclass
class TickData:
    """Individual tick data point"""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: Optional[float] = None
    volume: Optional[float] = None
    exchange: str = "FX"

@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    spread: float = field(init=False)
    mid_price: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields"""
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            self.spread = best_ask - best_bid
            self.mid_price = (best_bid + best_ask) / 2
        else:
            self.spread = 0.0
            self.mid_price = 0.0

@dataclass
class MicrostructureFeatures:
    """Computed microstructure features"""
    timestamp: datetime
    symbol: str
    
    # Basic features
    spread_bps: float  # Spread in basis points
    relative_spread: float  # Spread relative to price
    effective_spread: float  # Effective spread considering depth
    quoted_spread: float  # Simple bid-ask spread
    
    # Volume and liquidity features
    order_book_imbalance: float  # -1 to 1, negative for ask pressure
    volume_imbalance: float  # Volume at bid vs ask
    depth_imbalance: float  # Depth at different levels
    total_depth: float  # Total liquidity in order book
    liquidity_consumption: float  # Rate of liquidity consumption
    
    # Price movement features
    price_momentum: float  # Short-term price momentum
    volatility_estimate: float  # Microstructure volatility
    price_impact: float  # Estimated price impact of trades
    adverse_selection: float  # Adverse selection component
    
    # High-frequency features
    tick_velocity: float  # Ticks per second
    quote_velocity: float  # Quote updates per second
    trade_intensity: float  # Trade frequency
    information_flow: float  # Rate of information arrival
    
    # Order flow features
    order_flow_imbalance: float  # Buy vs sell pressure
    large_trade_ratio: float  # Percentage of large trades
    trade_size_skew: float  # Skew in trade sizes
    hidden_liquidity: float  # Estimated hidden liquidity
    
    # Market regime
    market_regime: MarketRegime
    regime_confidence: float
    
    # Derived signals
    micro_pressure: float  # Microstructure pressure (-1 to 1)
    liquidity_score: float  # 0-1, higher = more liquid
    toxicity_score: float  # 0-1, higher = more toxic flow

class AdvancedMarketMicrostructure:
    """
    Advanced Market Microstructure Analysis Engine
    Analyzes high-frequency data, order book dynamics, and market microstructure
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Data buffers
        self.tick_buffer: Deque[TickData] = deque(maxlen=self.config['buffer_sizes']['ticks'])
        self.order_book_buffer: Deque[OrderBookSnapshot] = deque(maxlen=self.config['buffer_sizes']['order_books'])
        self.trade_buffer: Deque[Dict] = deque(maxlen=self.config['buffer_sizes']['trades'])
        
        # Feature storage
        self.features_history: Deque[MicrostructureFeatures] = deque(maxlen=1000)
        self.regime_history: Deque[Tuple[datetime, MarketRegime]] = deque(maxlen=500)
        
        # Statistical models
        self.volatility_estimator = RollingVolatilityEstimator()
        self.regime_detector = MicrostructureRegimeDetector()
        self.liquidity_analyzer = LiquidityAnalyzer()
        
        # Performance tracking
        self.performance_metrics = {
            'processing_latency': 0.0,
            'feature_accuracy': 0.0,
            'regime_detection_accuracy': 0.0
        }
        
        # Cache for expensive computations
        self.cache = {}
        self.cache_ttl = timedelta(minutes=1)
        
        logger.info("Advanced Market Microstructure Analysis initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "buffer_sizes": {
                "ticks": 10000,
                "order_books": 1000,
                "trades": 5000
            },
            "feature_windows": {
                "short_term": 100,   # ticks
                "medium_term": 1000, # ticks
                "long_term": 5000    # ticks
            },
            "thresholds": {
                "high_frequency_threshold": 10,  # ticks per second
                "low_liquidity_threshold": 0.1,  # relative depth
                "large_trade_threshold": 1000000, # USD
                "volatility_threshold": 0.0005   # 5 pips
            },
            "analysis_parameters": {
                "order_book_levels": 5,
                "imbalance_lookback": 50,
                "correlation_window": 100,
                "regime_transition_smoothness": 0.1
            }
        }
    
    async def process_tick_data(self, tick: TickData) -> MicrostructureFeatures:
        """
        Process incoming tick data and compute microstructure features
        """
        start_time = datetime.now()
        
        try:
            # Add to buffer
            self.tick_buffer.append(tick)
            
            # Compute basic features
            basic_features = await self._compute_basic_features(tick)
            
            # Compute order book features (if available)
            order_book_features = await self._compute_order_book_features(tick)
            
            # Compute volume and liquidity features
            liquidity_features = await self._compute_liquidity_features(tick)
            
            # Compute high-frequency features
            hf_features = await self._compute_high_frequency_features()
            
            # Compute order flow features
            order_flow_features = await self._compute_order_flow_features()
            
            # Detect market regime
            regime_info = await self._detect_market_regime(
                basic_features, order_book_features, hf_features
            )
            
            # Compute derived signals
            derived_signals = await self._compute_derived_signals(
                basic_features, order_book_features, liquidity_features, regime_info
            )
            
            # Combine all features
            features = MicrostructureFeatures(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                spread_bps=basic_features['spread_bps'],
                relative_spread=basic_features['relative_spread'],
                effective_spread=basic_features['effective_spread'],
                quoted_spread=basic_features['quoted_spread'],
                order_book_imbalance=order_book_features['order_book_imbalance'],
                volume_imbalance=order_book_features['volume_imbalance'],
                depth_imbalance=order_book_features['depth_imbalance'],
                total_depth=order_book_features['total_depth'],
                liquidity_consumption=liquidity_features['liquidity_consumption'],
                price_momentum=basic_features['price_momentum'],
                volatility_estimate=basic_features['volatility_estimate'],
                price_impact=liquidity_features['price_impact'],
                adverse_selection=liquidity_features['adverse_selection'],
                tick_velocity=hf_features['tick_velocity'],
                quote_velocity=hf_features['quote_velocity'],
                trade_intensity=hf_features['trade_intensity'],
                information_flow=hf_features['information_flow'],
                order_flow_imbalance=order_flow_features['order_flow_imbalance'],
                large_trade_ratio=order_flow_features['large_trade_ratio'],
                trade_size_skew=order_flow_features['trade_size_skew'],
                hidden_liquidity=order_flow_features['hidden_liquidity'],
                market_regime=regime_info['regime'],
                regime_confidence=regime_info['confidence'],
                micro_pressure=derived_signals['micro_pressure'],
                liquidity_score=derived_signals['liquidity_score'],
                toxicity_score=derived_signals['toxicity_score']
            )
            
            # Store features
            self.features_history.append(features)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_metrics['processing_latency'] = (
                self.performance_metrics['processing_latency'] * 0.9 + processing_time * 0.1
            )
            
            logger.debug(f"Processed microstructure features for {tick.symbol} "
                        f"in {processing_time:.4f}s")
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
            # Return default features on error
            return await self._create_default_features(tick)
    
    async def process_order_book(self, order_book: OrderBookSnapshot) -> None:
        """
        Process order book snapshot
        """
        try:
            self.order_book_buffer.append(order_book)
            
            # Update liquidity analyzer
            await self.liquidity_analyzer.update_order_book(order_book)
            
            logger.debug(f"Processed order book for {order_book.symbol}")
            
        except Exception as e:
            logger.error(f"Error processing order book: {e}")
    
    async def process_trade(self, trade: Dict) -> None:
        """
        Process trade data
        """
        try:
            self.trade_buffer.append(trade)
            
            # Update trade-based features
            await self._update_trade_based_features(trade)
            
            logger.debug(f"Processed trade for {trade.get('symbol', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def _compute_basic_features(self, tick: TickData) -> Dict[str, float]:
        """Compute basic microstructure features"""
        features = {}
        
        # Spread features
        features['spread_bps'] = (tick.ask_price - tick.bid_price) / tick.bid_price * 10000
        features['relative_spread'] = (tick.ask_price - tick.bid_price) / ((tick.ask_price + tick.bid_price) / 2)
        features['quoted_spread'] = tick.ask_price - tick.bid_price
        
        # Effective spread (simplified)
        if tick.last_price:
            mid_price = (tick.bid_price + tick.ask_price) / 2
            features['effective_spread'] = 2 * abs(tick.last_price - mid_price)
        else:
            features['effective_spread'] = features['quoted_spread']
        
        # Price momentum (short-term)
        if len(self.tick_buffer) >= 10:
            recent_ticks = list(self.tick_buffer)[-10:]
            price_changes = []
            for i in range(1, len(recent_ticks)):
                mid_prev = (recent_ticks[i-1].bid_price + recent_ticks[i-1].ask_price) / 2
                mid_curr = (recent_ticks[i].bid_price + recent_ticks[i].ask_price) / 2
                price_changes.append((mid_curr - mid_prev) / mid_prev)
            
            if price_changes:
                features['price_momentum'] = np.mean(price_changes) * 10000  # in bps
            else:
                features['price_momentum'] = 0.0
        else:
            features['price_momentum'] = 0.0
        
        # Volatility estimate
        features['volatility_estimate'] = await self.volatility_estimator.estimate_volatility(
            self.tick_buffer
        )
        
        return features
    
    async def _compute_order_book_features(self, tick: TickData) -> Dict[str, float]:
        """Compute order book-based features"""
        features = {
            'order_book_imbalance': 0.0,
            'volume_imbalance': 0.0,
            'depth_imbalance': 0.0,
            'total_depth': 0.0
        }
        
        if not self.order_book_buffer:
            return features
        
        # Get latest order book
        latest_ob = self.order_book_buffer[-1]
        
        if not latest_ob.bids or not latest_ob.asks:
            return features
        
        # Order book imbalance
        total_bid_volume = sum(size for _, size in latest_ob.bids)
        total_ask_volume = sum(size for _, size in latest_ob.asks)
        
        if total_bid_volume + total_ask_volume > 0:
            features['order_book_imbalance'] = (
                (total_bid_volume - total_ask_volume) / 
                (total_bid_volume + total_ask_volume)
            )
        
        # Volume imbalance at best levels
        best_bid_size = latest_ob.bids[0][1] if latest_ob.bids else 0
        best_ask_size = latest_ob.asks[0][1] if latest_ob.asks else 0
        
        if best_bid_size + best_ask_size > 0:
            features['volume_imbalance'] = (
                (best_bid_size - best_ask_size) / 
                (best_bid_size + best_ask_size)
            )
        
        # Depth imbalance across levels
        bid_depth_levels = sum(size for _, size in latest_ob.bids[:3])  # Top 3 levels
        ask_depth_levels = sum(size for _, size in latest_ob.asks[:3])
        
        if bid_depth_levels + ask_depth_levels > 0:
            features['depth_imbalance'] = (
                (bid_depth_levels - ask_depth_levels) / 
                (bid_depth_levels + ask_depth_levels)
            )
        
        # Total depth
        features['total_depth'] = total_bid_volume + total_ask_volume
        
        return features
    
    async def _compute_liquidity_features(self, tick: TickData) -> Dict[str, float]:
        """Compute liquidity-related features"""
        features = {
            'liquidity_consumption': 0.0,
            'price_impact': 0.0,
            'adverse_selection': 0.0
        }
        
        if len(self.trade_buffer) < 10:
            return features
        
        # Liquidity consumption rate
        recent_trades = list(self.trade_buffer)[-10:]
        trade_volumes = [t.get('size', 0) for t in recent_trades]
        
        if trade_volumes:
            features['liquidity_consumption'] = np.mean(trade_volumes)
        
        # Price impact estimation (simplified)
        if len(self.tick_buffer) >= 20:
            recent_ticks = list(self.tick_buffer)[-20:]
            price_changes = []
            
            for i in range(1, len(recent_ticks)):
                mid_prev = (recent_ticks[i-1].bid_price + recent_ticks[i-1].ask_price) / 2
                mid_curr = (recent_ticks[i].bid_price + recent_ticks[i].ask_price) / 2
                price_changes.append(abs(mid_curr - mid_prev) / mid_prev)
            
            if price_changes:
                # Price impact proportional to volatility and trade size
                avg_volume = features['liquidity_consumption']
                avg_volatility = np.mean(price_changes)
                features['price_impact'] = avg_volatility * avg_volume / 1000000  # Normalized
        
        # Adverse selection component (simplified)
        # This would typically use more sophisticated models like Glosten-Harris
        if features['price_impact'] > 0:
            features['adverse_selection'] = min(features['price_impact'] * 0.3, 0.1)
        
        return features
    
    async def _compute_high_frequency_features(self) -> Dict[str, float]:
        """Compute high-frequency data features"""
        features = {
            'tick_velocity': 0.0,
            'quote_velocity': 0.0,
            'trade_intensity': 0.0,
            'information_flow': 0.0
        }
        
        if len(self.tick_buffer) < 10:
            return features
        
        # Tick velocity (ticks per second)
        recent_ticks = list(self.tick_buffer)[-100:]  # Last 100 ticks
        if len(recent_ticks) >= 2:
            time_span = (recent_ticks[-1].timestamp - recent_ticks[0].timestamp).total_seconds()
            if time_span > 0:
                features['tick_velocity'] = len(recent_ticks) / time_span
        
        # Quote velocity (order book updates per second)
        if len(self.order_book_buffer) >= 10:
            recent_obs = list(self.order_book_buffer)[-50:]
            if len(recent_obs) >= 2:
                time_span = (recent_obs[-1].timestamp - recent_obs[0].timestamp).total_seconds()
                if time_span > 0:
                    features['quote_velocity'] = len(recent_obs) / time_span
        
        # Trade intensity
        if len(self.trade_buffer) >= 10:
            recent_trades = list(self.trade_buffer)[-50:]
            if len(recent_trades) >= 2:
                time_span = (recent_trades[-1].get('timestamp', datetime.now()) - 
                           recent_trades[0].get('timestamp', datetime.now())).total_seconds()
                if time_span > 0:
                    features['trade_intensity'] = len(recent_trades) / time_span
        
        # Information flow (combined measure)
        features['information_flow'] = (
            features['tick_velocity'] * 0.4 +
            features['quote_velocity'] * 0.3 +
            features['trade_intensity'] * 0.3
        )
        
        return features
    
    async def _compute_order_flow_features(self) -> Dict[str, float]:
        """Compute order flow analysis features"""
        features = {
            'order_flow_imbalance': 0.0,
            'large_trade_ratio': 0.0,
            'trade_size_skew': 0.0,
            'hidden_liquidity': 0.0
        }
        
        if len(self.trade_buffer) < 20:
            return features
        
        recent_trades = list(self.trade_buffer)[-100:]
        
        # Order flow imbalance
        buy_volume = sum(t.get('size', 0) for t in recent_trades if t.get('side') == 'buy')
        sell_volume = sum(t.get('size', 0) for t in recent_trades if t.get('side') == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            features['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
        
        # Large trade ratio
        large_trade_threshold = self.config['thresholds']['large_trade_threshold']
        large_trades = [t for t in recent_trades if t.get('size', 0) >= large_trade_threshold]
        features['large_trade_ratio'] = len(large_trades) / len(recent_trades) if recent_trades else 0
        
        # Trade size skew
        trade_sizes = [t.get('size', 0) for t in recent_trades]
        if trade_sizes:
            features['trade_size_skew'] = stats.skew(trade_sizes)
        
        # Hidden liquidity estimation (simplified)
        # This would typically use more sophisticated inference
        if len(self.order_book_buffer) >= 10:
            recent_obs = list(self.order_book_buffer)[-10:]
            depth_changes = []
            
            for i in range(1, len(recent_obs)):
                prev_depth = sum(size for _, size in recent_obs[i-1].bids) + sum(size for _, size in recent_obs[i-1].asks)
                curr_depth = sum(size for _, size in recent_obs[i].bids) + sum(size for _, size in recent_obs[i].asks)
                depth_changes.append(abs(curr_depth - prev_depth))
            
            if depth_changes:
                # High depth volatility might indicate hidden liquidity
                features['hidden_liquidity'] = min(np.std(depth_changes) / np.mean(depth_changes) if np.mean(depth_changes) > 0 else 0, 1.0)
        
        return features
    
    async def _detect_market_regime(self, basic_features: Dict, order_book_features: Dict, 
                                  hf_features: Dict) -> Dict[str, Any]:
        """Detect current market regime based on microstructure"""
        
        # Use regime detector
        regime_result = await self.regime_detector.detect_regime(
            basic_features, order_book_features, hf_features, self.tick_buffer
        )
        
        # Store regime history
        self.regime_history.append((datetime.now(), regime_result['regime']))
        
        return regime_result
    
    async def _compute_derived_signals(self, basic_features: Dict, order_book_features: Dict,
                                     liquidity_features: Dict, regime_info: Dict) -> Dict[str, float]:
        """Compute derived trading signals from microstructure features"""
        
        signals = {
            'micro_pressure': 0.0,
            'liquidity_score': 0.0,
            'toxicity_score': 0.0
        }
        
        # Microstructure pressure (-1 to 1)
        # Positive = buying pressure, Negative = selling pressure
        pressure_components = [
            order_book_features['order_book_imbalance'] * 0.3,
            order_book_features['volume_imbalance'] * 0.2,
            liquidity_features['price_impact'] * 10 * 0.2,  # Scaled
            basic_features['price_momentum'] * 0.01 * 0.3   # Scaled
        ]
        
        signals['micro_pressure'] = np.clip(sum(pressure_components), -1, 1)
        
        # Liquidity score (0-1, higher = more liquid)
        liquidity_components = [
            (1 - basic_features['relative_spread'] * 1000) * 0.4,  # Lower spread = better
            min(order_book_features['total_depth'] / 10000000, 1) * 0.3,  # More depth = better
            (1 - liquidity_features['price_impact'] * 10) * 0.3  # Lower impact = better
        ]
        
        signals['liquidity_score'] = np.clip(sum(liquidity_components), 0, 1)
        
        # Toxicity score (0-1, higher = more toxic order flow)
        toxicity_components = [
            liquidity_features['adverse_selection'] * 10 * 0.4,
            basic_features['volatility_estimate'] * 100 * 0.3,  # Higher vol = more toxic
            (1 - signals['liquidity_score']) * 0.3  # Lower liquidity = more toxic
        ]
        
        signals['toxicity_score'] = np.clip(sum(toxicity_components), 0, 1)
        
        return signals
    
    async def _update_trade_based_features(self, trade: Dict) -> None:
        """Update features based on new trade data"""
        # This would update various trade-based estimators
        pass
    
    async def _create_default_features(self, tick: TickData) -> MicrostructureFeatures:
        """Create default features when processing fails"""
        return MicrostructureFeatures(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            spread_bps=2.0,
            relative_spread=0.0002,
            effective_spread=0.0002,
            quoted_spread=0.0002,
            order_book_imbalance=0.0,
            volume_imbalance=0.0,
            depth_imbalance=0.0,
            total_depth=0.0,
            liquidity_consumption=0.0,
            price_momentum=0.0,
            volatility_estimate=0.0001,
            price_impact=0.0,
            adverse_selection=0.0,
            tick_velocity=1.0,
            quote_velocity=0.5,
            trade_intensity=0.2,
            information_flow=0.5,
            order_flow_imbalance=0.0,
            large_trade_ratio=0.0,
            trade_size_skew=0.0,
            hidden_liquidity=0.0,
            market_regime=MarketRegime.NORMAL,
            regime_confidence=0.5,
            micro_pressure=0.0,
            liquidity_score=0.5,
            toxicity_score=0.3
        )
    
    async def get_microstructure_signals(self, symbol: str) -> Dict[str, Any]:
        """Get current microstructure signals for trading"""
        if not self.features_history:
            return await self._get_default_signals(symbol)
        
        latest_features = self.features_history[-1]
        
        signals = {
            'symbol': symbol,
            'timestamp': latest_features.timestamp,
            'market_regime': latest_features.market_regime.value,
            'regime_confidence': latest_features.regime_confidence,
            'micro_pressure': latest_features.micro_pressure,
            'liquidity_score': latest_features.liquidity_score,
            'toxicity_score': latest_features.toxicity_score,
            'trading_recommendation': self._generate_trading_recommendation(latest_features),
            'risk_adjustment': self._calculate_risk_adjustment(latest_features)
        }
        
        return signals
    
    def _generate_trading_recommendation(self, features: MicrostructureFeatures) -> str:
        """Generate trading recommendation based on microstructure"""
        
        if features.toxicity_score > 0.7:
            return "AVOID_TRADING"
        elif features.liquidity_score < 0.3:
            return "REDUCE_SIZE"
        elif features.micro_pressure > 0.6:
            return "CONSIDER_BUY"
        elif features.micro_pressure < -0.6:
            return "CONSIDER_SELL"
        elif features.market_regime == MarketRegime.HIGH_FREQUENCY:
            return "HIGH_FREQ_CAUTION"
        else:
            return "NORMAL_TRADING"
    
    def _calculate_risk_adjustment(self, features: MicrostructureFeatures) -> float:
        """Calculate risk adjustment factor based on microstructure"""
        base_risk = 1.0
        
        # Adjust for toxicity
        if features.toxicity_score > 0.7:
            base_risk *= 0.3
        elif features.toxicity_score > 0.5:
            base_risk *= 0.6
        
        # Adjust for liquidity
        if features.liquidity_score < 0.3:
            base_risk *= 0.5
        elif features.liquidity_score < 0.5:
            base_risk *= 0.8
        
        # Adjust for regime
        if features.market_regime in [MarketRegime.FLASH_CRASH, MarketRegime.NEWS_EVENT]:
            base_risk *= 0.2
        elif features.market_regime == MarketRegime.HIGH_FREQUENCY:
            base_risk *= 0.7
        
        return max(base_risk, 0.1)  # Minimum 10% of normal size
    
    async def _get_default_signals(self, symbol: str) -> Dict[str, Any]:
        """Get default signals when no data available"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'market_regime': 'NORMAL',
            'regime_confidence': 0.5,
            'micro_pressure': 0.0,
            'liquidity_score': 0.5,
            'toxicity_score': 0.3,
            'trading_recommendation': "NORMAL_TRADING",
            'risk_adjustment': 1.0
        }
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics of the microstructure analysis"""
        return self.performance_metrics.copy()
    
    async def reset(self) -> None:
        """Reset the microstructure analyzer"""
        self.tick_buffer.clear()
        self.order_book_buffer.clear()
        self.trade_buffer.clear()
        self.features_history.clear()
        self.regime_history.clear()
        
        logger.info("Market microstructure analyzer reset")

# Supporting Classes

class RollingVolatilityEstimator:
    """Estimates volatility from high-frequency data"""
    
    def __init__(self, window: int = 100):
        self.window = window
        self.price_changes = deque(maxlen=window)
    
    async def estimate_volatility(self, tick_buffer: Deque[TickData]) -> float:
        """Estimate volatility from recent price changes"""
        if len(tick_buffer) < 10:
            return 0.0001  # Default low volatility
        
        recent_ticks = list(tick_buffer)[-self.window:]
        returns = []
        
        for i in range(1, len(recent_ticks)):
            mid_prev = (recent_ticks[i-1].bid_price + recent_ticks[i-1].ask_price) / 2
            mid_curr = (recent_ticks[i].bid_price + recent_ticks[i].ask_price) / 2
            ret = (mid_curr - mid_prev) / mid_prev
            returns.append(ret)
        
        if returns:
            volatility = np.std(returns) * np.sqrt(252 * 24 * 3600)  # Annualized
            return max(volatility, 0.00001)  # Avoid zero volatility
        else:
            return 0.0001

class MicrostructureRegimeDetector:
    """Detects market regimes based on microstructure features"""
    
    def __init__(self):
        self.regime_weights = {
            'volatility': 0.3,
            'liquidity': 0.25,
            'activity': 0.25,
            'imbalance': 0.2
        }
    
    async def detect_regime(self, basic_features: Dict, order_book_features: Dict,
                          hf_features: Dict, tick_buffer: Deque[TickData]) -> Dict[str, Any]:
        """Detect current market regime"""
        
        # Calculate regime scores
        volatility_score = self._calculate_volatility_score(basic_features)
        liquidity_score = self._calculate_liquidity_score(order_book_features)
        activity_score = self._calculate_activity_score(hf_features)
        imbalance_score = self._calculate_imbalance_score(order_book_features)
        
        # Weighted overall score
        overall_score = (
            volatility_score * self.regime_weights['volatility'] +
            liquidity_score * self.regime_weights['liquidity'] +
            activity_score * self.regime_weights['activity'] +
            imbalance_score * self.regime_weights['imbalance']
        )
        
        # Determine regime
        if overall_score > 0.8:
            regime = MarketRegime.HIGH_FREQUENCY
            confidence = overall_score
        elif overall_score > 0.6:
            regime = MarketRegime.NEWS_EVENT
            confidence = overall_score
        elif overall_score < 0.3:
            regime = MarketRegime.LOW_LIQUIDITY
            confidence = 1 - overall_score
        elif volatility_score > 0.7:
            regime = MarketRegime.FLASH_CRASH
            confidence = volatility_score
        else:
            regime = MarketRegime.NORMAL
            confidence = 0.7
        
        return {
            'regime': regime,
            'confidence': min(confidence, 1.0),
            'scores': {
                'volatility': volatility_score,
                'liquidity': liquidity_score,
                'activity': activity_score,
                'imbalance': imbalance_score,
                'overall': overall_score
            }
        }
    
    def _calculate_volatility_score(self, features: Dict) -> float:
        """Calculate volatility-based regime score"""
        volatility = features.get('volatility_estimate', 0.0001)
        # Normalize volatility to 0-1 scale (assuming 0.001 = 10 pips is high)
        return min(volatility / 0.001, 1.0)
    
    def _calculate_liquidity_score(self, features: Dict) -> float:
        """Calculate liquidity-based regime score"""
        depth = features.get('total_depth', 0)
        spread = features.get('order_book_imbalance', 0.5)  # Using imbalance as proxy
        
        # Lower depth and higher imbalance = lower liquidity
        depth_score = 1 - min(depth / 5000000, 1.0)  # Normalize depth
        imbalance_score = abs(spread)  # Higher absolute imbalance = worse liquidity
        
        return (depth_score + imbalance_score) / 2
    
    def _calculate_activity_score(self, features: Dict) -> float:
        """Calculate activity-based regime score"""
        tick_velocity = features.get('tick_velocity', 0)
        trade_intensity = features.get('trade_intensity', 0)
        
        # Normalize to 0-1 scale
        velocity_score = min(tick_velocity / 20, 1.0)  # 20 ticks/sec = high
        intensity_score = min(trade_intensity / 10, 1.0)  # 10 trades/sec = high
        
        return (velocity_score + intensity_score) / 2
    
    def _calculate_imbalance_score(self, features: Dict) -> float:
        """Calculate order book imbalance score"""
        imbalance = abs(features.get('order_book_imbalance', 0))
        volume_imbalance = abs(features.get('volume_imbalance', 0))
        
        return (imbalance + volume_imbalance) / 2

class LiquidityAnalyzer:
    """Analyzes market liquidity conditions"""
    
    def __init__(self):
        self.order_book_history = deque(maxlen=100)
        self.liquidity_metrics = {}
    
    async def update_order_book(self, order_book: OrderBookSnapshot) -> None:
        """Update liquidity analysis with new order book"""
        self.order_book_history.append(order_book)
        await self._compute_liquidity_metrics()
    
    async def _compute_liquidity_metrics(self) -> None:
        """Compute various liquidity metrics"""
        if not self.order_book_history:
            return
        
        latest_ob = self.order_book_history[-1]
        
        # Basic liquidity metrics
        self.liquidity_metrics = {
            'spread': latest_ob.spread,
            'mid_price': latest_ob.mid_price,
            'total_depth': sum(size for _, size in latest_ob.bids) + sum(size for _, size in latest_ob.asks),
            'best_bid_size': latest_ob.bids[0][1] if latest_ob.bids else 0,
            'best_ask_size': latest_ob.asks[0][1] if latest_ob.asks else 0,
            'depth_5_levels': self._calculate_depth_levels(latest_ob, 5),
            'liquidity_skew': self._calculate_liquidity_skew(latest_ob)
        }
    
    def _calculate_depth_levels(self, order_book: OrderBookSnapshot, levels: int) -> float:
        """Calculate total depth at specified levels"""
        bid_depth = sum(size for _, size in order_book.bids[:levels])
        ask_depth = sum(size for _, size in order_book.asks[:levels])
        return bid_depth + ask_depth
    
    def _calculate_liquidity_skew(self, order_book: OrderBookSnapshot) -> float:
        """Calculate liquidity skew between bid and ask sides"""
        total_bid = sum(size for _, size in order_book.bids)
        total_ask = sum(size for _, size in order_book.asks)
        
        if total_bid + total_ask == 0:
            return 0.0
        
        return (total_bid - total_ask) / (total_bid + total_ask)

# Example usage
async def main():
    """Example usage of market microstructure analysis"""
    print("🚀 Starting Market Microstructure Analysis...")
    
    # Create microstructure analyzer
    microstructure = AdvancedMarketMicrostructure()
    
    # Generate sample tick data
    sample_ticks = []
    base_price = 1.1000
    
    for i in range(100):
        tick = TickData(
            symbol="EUR/USD",
            timestamp=datetime.now() + timedelta(seconds=i*0.1),
            bid_price=base_price + np.random.normal(0, 0.0001),
            ask_price=base_price + np.random.normal(0.0002, 0.0001),
            bid_size=1000000 + np.random.normal(0, 100000),
            ask_size=800000 + np.random.normal(0, 100000),
            last_price=base_price + np.random.normal(0.0001, 0.0001),
            volume=5000000
        )
        sample_ticks.append(tick)
    
    # Process sample ticks
    print("📊 Processing sample tick data...")
    for tick in sample_ticks[:50]:  # Process first 50 ticks
        features = await microstructure.process_tick_data(tick)
        print(f"Tick {tick.timestamp}: Spread {features.spread_bps:.2f} bps, "
              f"Pressure: {features.micro_pressure:.3f}, "
              f"Regime: {features.market_regime.value}")
    
    # Get current signals
    signals = await microstructure.get_microstructure_signals("EUR/USD")
    print(f"\n🎯 Current Microstructure Signals:")
    for key, value in signals.items():
        print(f"  {key}: {value}")
    
    # Get performance metrics
    metrics = await microstructure.get_performance_metrics()
    print(f"\n📈 Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Market Microstructure Analysis demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
"""
Advanced Microstructure Analyzer for FOREX TRADING BOT
Real-time order book analysis, market impact modeling, and microstructure-based signals
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
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback functions
    def minimize(func, x0, method='BFGS'):
        return type('Result', (), {'x': x0, 'success': True})()
    
    stats = type('MockStats', (), {
        'skew': lambda x: 0.0,
        'kurtosis': lambda x: 0.0
    })()

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback classes
    class IsolationForest:
        def __init__(self, contamination=0.1, random_state=42):
            pass
        def fit_predict(self, X):
            return [1] * len(X)
    
    class StandardScaler:
        def fit(self, X): return self
        def transform(self, X): return X
import warnings
from enum import Enum
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderBookState(Enum):
    """Order book state classifications"""
    NORMAL = "NORMAL"
    IMBALANCED_BID = "IMBALANCED_BID"
    IMBALANCED_ASK = "IMBALANCED_ASK"
    THIN = "THIN"
    DEEP = "DEEP"
    VOLATILE = "VOLATILE"
    STABLE = "STABLE"

class TradeDirection(Enum):
    """Trade direction classifications"""
    AGGRESSIVE_BUY = "AGGRESSIVE_BUY"
    AGGRESSIVE_SELL = "AGGRESSIVE_SELL"
    PASSIVE_BUY = "PASSIVE_BUY"
    PASSIVE_SELL = "PASSIVE_SELL"
    UNKNOWN = "UNKNOWN"

@dataclass
class OrderBookMetrics:
    """Comprehensive order book metrics"""
    timestamp: datetime
    symbol: str
    
    # Basic metrics
    spread_bps: float
    mid_price: float
    weighted_mid_price: float
    
    # Depth metrics
    total_depth: float
    bid_depth: float
    ask_depth: float
    depth_imbalance: float
    
    # Price levels analysis
    price_levels_bid: int
    price_levels_ask: int
    cumulative_depth_bid: List[float]
    cumulative_depth_ask: List[float]
    
    # Volume analysis
    volume_imbalance: float
    volume_skew: float
    large_order_presence: bool
    
    # Liquidity metrics
    liquidity_score: float
    market_impact_estimate: float
    adverse_selection_risk: float
    
    # Order flow
    order_flow_imbalance: float
    trade_size_distribution: Dict[str, float]
    hidden_liquidity_estimate: float

@dataclass
class MicrostructureSignal:
    """Microstructure-based trading signals"""
    timestamp: datetime
    symbol: str
    
    # Core signals
    order_book_pressure: float  # -1 to 1 (sell to buy pressure)
    liquidity_pressure: float   # -1 to 1 (illiquid to liquid)
    toxicity_score: float       # 0 to 1 (low to high toxicity)
    
    # Derived signals
    short_term_momentum: float  # -1 to 1
    mean_reversion_signal: float # -1 to 1
    breakout_potential: float   # 0 to 1
    
    # Market regime
    market_regime: str
    regime_confidence: float
    
    # Trading recommendations
    suggested_action: str
    confidence: float
    position_size_multiplier: float
    
    # Risk metrics
    execution_risk: float
    adverse_selection_risk: float
    liquidity_risk: float

@dataclass
class MarketImpactModel:
    """Market impact model parameters"""
    temporary_impact: float
    permanent_impact: float
    price_elasticity: float
    liquidity_absorption: float

class AdvancedMicrostructureAnalyzer:
    """
    Advanced Microstructure Analyzer
    Specialized in order book analysis, market impact modeling, and real-time signal generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Data storage
        self.order_book_history: Deque[OrderBookMetrics] = deque(maxlen=1000)
        self.trade_history: Deque[Dict] = deque(maxlen=5000)
        self.microstructure_signals: Deque[MicrostructureSignal] = deque(maxlen=1000)
        
        # Analysis components
        self.order_book_analyzer = OrderBookAnalyzer()
        self.trade_analyzer = TradeAnalyzer()
        self.market_impact_model = MarketImpactEstimator()
        self.regime_detector = AdvancedRegimeDetector()
        self.anomaly_detector = MicrostructureAnomalyDetector()
        
        # Statistical models
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # State tracking
        self.current_regime: str = "NORMAL"
        self.regime_confidence: float = 0.8
        self.last_analysis_time: Optional[datetime] = None
        
        # Performance metrics
        self.performance_stats = {
            'analysis_latency': 0.0,
            'signal_accuracy': 0.0,
            'regime_detection_accuracy': 0.0,
            'anomaly_detection_rate': 0.0
        }
        
        logger.info("Advanced Microstructure Analyzer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "analysis_parameters": {
                "order_book_levels": 10,
                "analysis_frequency": "1s",
                "lookback_period": 60,  # seconds
                "correlation_window": 100
            },
            "thresholds": {
                "high_imbalance": 0.7,
                "low_liquidity": 0.3,
                "high_toxicity": 0.8,
                "large_trade_size": 1000000,
                "volatility_spike": 0.0005
            },
            "signal_parameters": {
                "momentum_window": 20,
                "mean_reversion_window": 50,
                "breakout_threshold": 0.002,
                "confidence_threshold": 0.6
            },
            "risk_parameters": {
                "max_position_size": 0.1,
                "liquidity_buffer": 0.2,
                "impact_tolerance": 0.001
            }
        }
    
    async def analyze_order_book(self, order_book_snapshot: Dict) -> OrderBookMetrics:
        """
        Perform comprehensive order book analysis
        """
        start_time = datetime.now()
        
        try:
            # Extract basic order book data
            symbol = order_book_snapshot['symbol']
            timestamp = order_book_snapshot['timestamp']
            bids = order_book_snapshot['bids']  # List of [price, size]
            asks = order_book_snapshot['asks']  # List of [price, size]
            
            # Compute basic metrics
            basic_metrics = await self._compute_basic_metrics(bids, asks, timestamp, symbol)
            
            # Compute depth analysis
            depth_metrics = await self._compute_depth_analysis(bids, asks, basic_metrics)
            
            # Compute volume analysis
            volume_metrics = await self._compute_volume_analysis(bids, asks)
            
            # Compute liquidity metrics
            liquidity_metrics = await self._compute_liquidity_metrics(basic_metrics, depth_metrics)
            
            # Compute order flow metrics
            order_flow_metrics = await self._compute_order_flow_metrics()
            
            # Combine all metrics
            metrics = OrderBookMetrics(
                timestamp=timestamp,
                symbol=symbol,
                spread_bps=basic_metrics['spread_bps'],
                mid_price=basic_metrics['mid_price'],
                weighted_mid_price=basic_metrics['weighted_mid_price'],
                total_depth=depth_metrics['total_depth'],
                bid_depth=depth_metrics['bid_depth'],
                ask_depth=depth_metrics['ask_depth'],
                depth_imbalance=depth_metrics['depth_imbalance'],
                price_levels_bid=depth_metrics['price_levels_bid'],
                price_levels_ask=depth_metrics['price_levels_ask'],
                cumulative_depth_bid=depth_metrics['cumulative_depth_bid'],
                cumulative_depth_ask=depth_metrics['cumulative_depth_ask'],
                volume_imbalance=volume_metrics['volume_imbalance'],
                volume_skew=volume_metrics['volume_skew'],
                large_order_presence=volume_metrics['large_order_presence'],
                liquidity_score=liquidity_metrics['liquidity_score'],
                market_impact_estimate=liquidity_metrics['market_impact_estimate'],
                adverse_selection_risk=liquidity_metrics['adverse_selection_risk'],
                order_flow_imbalance=order_flow_metrics['order_flow_imbalance'],
                trade_size_distribution=order_flow_metrics['trade_size_distribution'],
                hidden_liquidity_estimate=order_flow_metrics['hidden_liquidity_estimate']
            )
            
            # Store metrics
            self.order_book_history.append(metrics)
            
            # Update performance
            analysis_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats['analysis_latency'] = (
                self.performance_stats['analysis_latency'] * 0.9 + analysis_time * 0.1
            )
            
            logger.debug(f"Order book analysis completed for {symbol} in {analysis_time:.4f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Order book analysis failed: {e}")
            return await self._create_default_metrics(order_book_snapshot)
    
    async def analyze_trade(self, trade_data: Dict) -> None:
        """
        Analyze individual trade for microstructure insights
        """
        try:
            self.trade_history.append(trade_data)
            
            # Classify trade direction
            trade_direction = await self._classify_trade_direction(trade_data)
            trade_data['direction'] = trade_direction
            
            # Update trade-based metrics
            await self._update_trade_based_metrics(trade_data)
            
            # Detect anomalies
            await self._detect_trade_anomalies(trade_data)
            
            logger.debug(f"Trade analysis completed: {trade_direction.value}")
            
        except Exception as e:
            logger.error(f"Trade analysis failed: {e}")
    
    async def generate_microstructure_signals(self, symbol: str) -> MicrostructureSignal:
        """
        Generate comprehensive microstructure-based trading signals
        """
        try:
            # Get latest order book metrics
            if not self.order_book_history:
                return await self._create_default_signal(symbol)
            
            latest_metrics = self.order_book_history[-1]
            
            # Compute core signals
            order_book_pressure = await self._compute_order_book_pressure()
            liquidity_pressure = await self._compute_liquidity_pressure()
            toxicity_score = await self._compute_toxicity_score()
            
            # Compute derived signals
            short_term_momentum = await self._compute_short_term_momentum()
            mean_reversion_signal = await self._compute_mean_reversion_signal()
            breakout_potential = await self._compute_breakout_potential()
            
            # Detect market regime
            regime_info = await self._detect_market_regime()
            
            # Generate trading recommendations
            trading_recommendation = await self._generate_trading_recommendation(
                order_book_pressure, liquidity_pressure, toxicity_score, regime_info
            )
            
            # Compute risk metrics
            risk_metrics = await self._compute_risk_metrics()
            
            # Create signal
            signal = MicrostructureSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                order_book_pressure=order_book_pressure,
                liquidity_pressure=liquidity_pressure,
                toxicity_score=toxicity_score,
                short_term_momentum=short_term_momentum,
                mean_reversion_signal=mean_reversion_signal,
                breakout_potential=breakout_potential,
                market_regime=regime_info['regime'],
                regime_confidence=regime_info['confidence'],
                suggested_action=trading_recommendation['action'],
                confidence=trading_recommendation['confidence'],
                position_size_multiplier=trading_recommendation['size_multiplier'],
                execution_risk=risk_metrics['execution_risk'],
                adverse_selection_risk=risk_metrics['adverse_selection_risk'],
                liquidity_risk=risk_metrics['liquidity_risk']
            )
            
            # Store signal
            self.microstructure_signals.append(signal)
            
            logger.info(f"Microstructure signal generated: {signal.suggested_action} "
                       f"(Confidence: {signal.confidence:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return await self._create_default_signal(symbol)
    
    async def _compute_basic_metrics(self, bids: List, asks: List, 
                                   timestamp: datetime, symbol: str) -> Dict[str, float]:
        """Compute basic order book metrics"""
        metrics = {}
        
        if not bids or not asks:
            return self._get_default_basic_metrics()
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        best_bid_size = bids[0][1]
        best_ask_size = asks[0][1]
        
        # Spread metrics
        spread = best_ask - best_bid
        metrics['spread_bps'] = (spread / best_bid) * 10000
        metrics['mid_price'] = (best_bid + best_ask) / 2
        
        # Weighted mid price
        total_size = best_bid_size + best_ask_size
        if total_size > 0:
            metrics['weighted_mid_price'] = (
                (best_bid * best_ask_size + best_ask * best_bid_size) / total_size
            )
        else:
            metrics['weighted_mid_price'] = metrics['mid_price']
        
        # Basic size metrics
        metrics['best_bid_size'] = best_bid_size
        metrics['best_ask_size'] = best_ask_size
        metrics['size_imbalance'] = (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size) if (best_bid_size + best_ask_size) > 0 else 0
        
        return metrics
    
    async def _compute_depth_analysis(self, bids: List, asks: List, 
                                    basic_metrics: Dict) -> Dict[str, Any]:
        """Compute order book depth analysis"""
        metrics = {
            'total_depth': 0.0,
            'bid_depth': 0.0,
            'ask_depth': 0.0,
            'depth_imbalance': 0.0,
            'price_levels_bid': 0,
            'price_levels_ask': 0,
            'cumulative_depth_bid': [],
            'cumulative_depth_ask': []
        }
        
        if not bids or not asks:
            return metrics
        
        # Calculate depth at different levels
        levels = self.config['analysis_parameters']['order_book_levels']
        
        bid_depth = sum(size for price, size in bids[:levels])
        ask_depth = sum(size for price, size in asks[:levels])
        total_depth = bid_depth + ask_depth
        
        metrics['bid_depth'] = bid_depth
        metrics['ask_depth'] = ask_depth
        metrics['total_depth'] = total_depth
        
        # Depth imbalance
        if total_depth > 0:
            metrics['depth_imbalance'] = (bid_depth - ask_depth) / total_depth
        
        # Price levels
        metrics['price_levels_bid'] = len(bids)
        metrics['price_levels_ask'] = len(asks)
        
        # Cumulative depth
        cumulative_bid = []
        cumulative_ask = []
        current_bid = 0
        current_ask = 0
        
        for i in range(min(levels, len(bids))):
            current_bid += bids[i][1]
            cumulative_bid.append(current_bid)
        
        for i in range(min(levels, len(asks))):
            current_ask += asks[i][1]
            cumulative_ask.append(current_ask)
        
        metrics['cumulative_depth_bid'] = cumulative_bid
        metrics['cumulative_depth_ask'] = cumulative_ask
        
        return metrics
    
    async def _compute_volume_analysis(self, bids: List, asks: List) -> Dict[str, float]:
        """Compute volume-based analysis"""
        metrics = {
            'volume_imbalance': 0.0,
            'volume_skew': 0.0,
            'large_order_presence': False
        }
        
        if not bids or not asks:
            return metrics
        
        # Volume imbalance across levels
        levels = self.config['analysis_parameters']['order_book_levels']
        bid_volumes = [size for _, size in bids[:levels]]
        ask_volumes = [size for _, size in asks[:levels]]
        
        total_bid_volume = sum(bid_volumes)
        total_ask_volume = sum(ask_volumes)
        total_volume = total_bid_volume + total_ask_volume
        
        if total_volume > 0:
            metrics['volume_imbalance'] = (total_bid_volume - total_ask_volume) / total_volume
        
        # Volume skew (using coefficient of variation)
        if bid_volumes and ask_volumes:
            bid_skew = np.std(bid_volumes) / np.mean(bid_volumes) if np.mean(bid_volumes) > 0 else 0
            ask_skew = np.std(ask_volumes) / np.mean(ask_volumes) if np.mean(ask_volumes) > 0 else 0
            metrics['volume_skew'] = (bid_skew + ask_skew) / 2
        
        # Large order detection
        large_order_threshold = self.config['thresholds']['large_trade_size']
        metrics['large_order_presence'] = (
            any(size >= large_order_threshold for _, size in bids[:3]) or
            any(size >= large_order_threshold for _, size in asks[:3])
        )
        
        return metrics
    
    async def _compute_liquidity_metrics(self, basic_metrics: Dict, 
                                       depth_metrics: Dict) -> Dict[str, float]:
        """Compute liquidity-related metrics"""
        metrics = {
            'liquidity_score': 0.5,
            'market_impact_estimate': 0.0,
            'adverse_selection_risk': 0.0
        }
        
        # Liquidity score (0-1, higher = more liquid)
        spread_component = 1 - min(basic_metrics['spread_bps'] / 10, 1)  # Normalize spread
        depth_component = min(depth_metrics['total_depth'] / 10000000, 1)  # Normalize depth
        imbalance_component = 1 - abs(depth_metrics['depth_imbalance'])  # Lower imbalance = better
        
        metrics['liquidity_score'] = (spread_component * 0.4 + 
                                    depth_component * 0.4 + 
                                    imbalance_component * 0.2)
        
        # Market impact estimate
        # Simplified model: impact ~ volatility / liquidity
        if depth_metrics['total_depth'] > 0:
            metrics['market_impact_estimate'] = (
                basic_metrics['spread_bps'] / 10000 / depth_metrics['total_depth'] * 1000000
            )
        else:
            metrics['market_impact_estimate'] = 0.001  # Default impact
        
        # Adverse selection risk
        # Higher when large orders are present and liquidity is low
        large_order_penalty = 0.3 if depth_metrics.get('large_order_presence', False) else 0
        metrics['adverse_selection_risk'] = min(
            (1 - metrics['liquidity_score']) * 0.7 + large_order_penalty, 1.0
        )
        
        return metrics
    
    async def _compute_order_flow_metrics(self) -> Dict[str, Any]:
        """Compute order flow metrics from trade history"""
        metrics = {
            'order_flow_imbalance': 0.0,
            'trade_size_distribution': {},
            'hidden_liquidity_estimate': 0.0
        }
        
        if len(self.trade_history) < 10:
            return metrics
        
        recent_trades = list(self.trade_history)[-100:]
        
        # Order flow imbalance
        buy_trades = [t for t in recent_trades if t.get('direction') in 
                     [TradeDirection.AGGRESSIVE_BUY, TradeDirection.PASSIVE_BUY]]
        sell_trades = [t for t in recent_trades if t.get('direction') in 
                      [TradeDirection.AGGRESSIVE_SELL, TradeDirection.PASSIVE_SELL]]
        
        total_trades = len(buy_trades) + len(sell_trades)
        if total_trades > 0:
            metrics['order_flow_imbalance'] = (len(buy_trades) - len(sell_trades)) / total_trades
        
        # Trade size distribution
        trade_sizes = [t.get('size', 0) for t in recent_trades if t.get('size')]
        if trade_sizes:
            metrics['trade_size_distribution'] = {
                'mean': np.mean(trade_sizes),
                'std': np.std(trade_sizes),
                'skew': stats.skew(trade_sizes),
                'kurtosis': stats.kurtosis(trade_sizes)
            }
        
        # Hidden liquidity estimate (simplified)
        # Based on order book resilience and trade patterns
        if len(self.order_book_history) >= 10:
            recent_metrics = list(self.order_book_history)[-10:]
            depth_changes = []
            
            for i in range(1, len(recent_metrics)):
                depth_change = abs(recent_metrics[i].total_depth - recent_metrics[i-1].total_depth)
                depth_changes.append(depth_change)
            
            if depth_changes and np.mean(depth_changes) > 0:
                # High depth volatility might indicate hidden liquidity
                metrics['hidden_liquidity_estimate'] = min(
                    np.std(depth_changes) / np.mean(depth_changes), 1.0
                )
        
        return metrics
    
    async def _classify_trade_direction(self, trade_data: Dict) -> TradeDirection:
        """Classify trade direction based on microstructure"""
        try:
            price = trade_data.get('price', 0)
            size = trade_data.get('size', 0)
            
            if not self.order_book_history:
                return TradeDirection.UNKNOWN
            
            latest_metrics = self.order_book_history[-1]
            best_bid = latest_metrics.mid_price - latest_metrics.spread_bps * 0.0001 / 2
            best_ask = latest_metrics.mid_price + latest_metrics.spread_bps * 0.0001 / 2
            
            if price >= best_ask:
                return TradeDirection.AGGRESSIVE_BUY
            elif price <= best_bid:
                return TradeDirection.AGGRESSIVE_SELL
            elif price > latest_metrics.mid_price:
                return TradeDirection.PASSIVE_BUY
            elif price < latest_metrics.mid_price:
                return TradeDirection.PASSIVE_SELL
            else:
                return TradeDirection.UNKNOWN
                
        except Exception as e:
            logger.warning(f"Trade direction classification failed: {e}")
            return TradeDirection.UNKNOWN
    
    async def _compute_order_book_pressure(self) -> float:
        """Compute order book pressure signal (-1 to 1)"""
        if len(self.order_book_history) < 10:
            return 0.0
        
        recent_metrics = list(self.order_book_history)[-10:]
        
        pressure_components = []
        
        for metrics in recent_metrics:
            # Combine multiple pressure indicators
            imbalance_pressure = metrics.depth_imbalance * 0.4
            volume_pressure = metrics.volume_imbalance * 0.3
            flow_pressure = metrics.order_flow_imbalance * 0.3
            
            total_pressure = imbalance_pressure + volume_pressure + flow_pressure
            pressure_components.append(total_pressure)
        
        return np.clip(np.mean(pressure_components), -1, 1)
    
    async def _compute_liquidity_pressure(self) -> float:
        """Compute liquidity pressure signal (-1 to 1)"""
        if len(self.order_book_history) < 5:
            return 0.0
        
        recent_metrics = list(self.order_book_history)[-5:]
        
        # Convert liquidity score to pressure (-1 = illiquid, 1 = liquid)
        liquidity_scores = [metrics.liquidity_score for metrics in recent_metrics]
        avg_liquidity = np.mean(liquidity_scores)
        
        # Map to -1 to 1 scale
        liquidity_pressure = 2 * (avg_liquidity - 0.5)
        
        return np.clip(liquidity_pressure, -1, 1)
    
    async def _compute_toxicity_score(self) -> float:
        """Compute order flow toxicity score (0 to 1)"""
        if len(self.order_book_history) < 10:
            return 0.3
        
        recent_metrics = list(self.order_book_history)[-10:]
        
        toxicity_components = []
        
        for metrics in recent_metrics:
            # Higher adverse selection risk = more toxic
            adverse_component = metrics.adverse_selection_risk * 0.4
            
            # Large orders presence increases toxicity
            large_order_component = 0.3 if metrics.large_order_presence else 0.1
            
            # High market impact = more toxic
            impact_component = min(metrics.market_impact_estimate * 100, 1) * 0.3
            
            toxicity = adverse_component + large_order_component + impact_component
            toxicity_components.append(toxicity)
        
        return min(np.mean(toxicity_components), 1.0)
    
    async def _compute_short_term_momentum(self) -> float:
        """Compute short-term momentum from order book (-1 to 1)"""
        if len(self.order_book_history) < 20:
            return 0.0
        
        recent_metrics = list(self.order_book_history)[-20:]
        prices = [metrics.mid_price for metrics in recent_metrics]
        
        if len(prices) >= 2:
            returns = np.diff(prices) / prices[:-1]
            momentum = np.mean(returns) * 10000  # Convert to bps
            return np.clip(momentum / 10, -1, 1)  # Normalize to -1 to 1
        else:
            return 0.0
    
    async def _compute_mean_reversion_signal(self) -> float:
        """Compute mean reversion signal from order book (-1 to 1)"""
        if len(self.order_book_history) < 50:
            return 0.0
        
        recent_metrics = list(self.order_book_history)[-50:]
        prices = [metrics.mid_price for metrics in recent_metrics]
        
        if len(prices) >= 20:
            # Calculate z-score of current price relative to recent history
            current_price = prices[-1]
            mean_price = np.mean(prices[-20:])
            std_price = np.std(prices[-20:])
            
            if std_price > 0:
                z_score = (current_price - mean_price) / std_price
                # Inverse for mean reversion (high z-score = sell signal)
                return np.clip(-z_score / 2, -1, 1)
        
        return 0.0
    
    async def _compute_breakout_potential(self) -> float:
        """Compute breakout potential (0 to 1)"""
        if len(self.order_book_history) < 30:
            return 0.0
        
        recent_metrics = list(self.order_book_history)[-30:]
        
        # Look for consolidation followed by pressure buildup
        if len(recent_metrics) >= 10:
            recent_pressure = [metrics.depth_imbalance for metrics in recent_metrics[-10:]]
            pressure_volatility = np.std(recent_pressure)
            
            # High pressure volatility + strong directional pressure = breakout potential
            current_pressure = abs(recent_pressure[-1])
            breakout_potential = min(pressure_volatility * 10 + current_pressure, 1.0)
            
            return breakout_potential
        
        return 0.0
    
    async def _detect_market_regime(self) -> Dict[str, Any]:
        """Detect current market regime"""
        if len(self.order_book_history) < 20:
            return {'regime': 'NORMAL', 'confidence': 0.5}
        
        recent_metrics = list(self.order_book_history)[-20:]
        
        # Calculate regime features
        avg_spread = np.mean([m.spread_bps for m in recent_metrics])
        avg_liquidity = np.mean([m.liquidity_score for m in recent_metrics])
        pressure_volatility = np.std([m.depth_imbalance for m in recent_metrics])
        toxicity_avg = np.mean([m.adverse_selection_risk for m in recent_metrics])
        
        # Determine regime
        if toxicity_avg > 0.7:
            regime = "TOXIC"
            confidence = toxicity_avg
        elif avg_spread > 5:  # 5 bps
            regime = "ILLIQUID"
            confidence = min(avg_spread / 10, 1.0)
        elif pressure_volatility > 0.3:
            regime = "VOLATILE"
            confidence = min(pressure_volatility, 1.0)
        elif avg_liquidity > 0.8:
            regime = "LIQUID"
            confidence = avg_liquidity
        else:
            regime = "NORMAL"
            confidence = 0.7
        
        self.current_regime = regime
        self.regime_confidence = confidence
        
        return {'regime': regime, 'confidence': confidence}
    
    async def _generate_trading_recommendation(self, order_book_pressure: float,
                                            liquidity_pressure: float,
                                            toxicity_score: float,
                                            regime_info: Dict) -> Dict[str, Any]:
        """Generate trading recommendation based on microstructure"""
        
        # Base recommendation
        if toxicity_score > self.config['thresholds']['high_toxicity']:
            action = "AVOID_TRADING"
            confidence = 1.0 - toxicity_score
            size_multiplier = 0.1
        elif liquidity_pressure < -0.5:
            action = "REDUCE_SIZE"
            confidence = 0.7
            size_multiplier = 0.3
        elif order_book_pressure > 0.6:
            action = "CONSIDER_BUY"
            confidence = min(order_book_pressure, 0.9)
            size_multiplier = min(order_book_pressure, 0.8)
        elif order_book_pressure < -0.6:
            action = "CONSIDER_SELL"
            confidence = min(abs(order_book_pressure), 0.9)
            size_multiplier = min(abs(order_book_pressure), 0.8)
        else:
            action = "HOLD"
            confidence = 0.5
            size_multiplier = 0.5
        
        # Adjust for regime
        if regime_info['regime'] in ["TOXIC", "VOLATILE"]:
            size_multiplier *= 0.5
            confidence *= 0.8
        
        return {
            'action': action,
            'confidence': confidence,
            'size_multiplier': max(size_multiplier, 0.1)
        }
    
    async def _compute_risk_metrics(self) -> Dict[str, float]:
        """Compute execution risk metrics"""
        if not self.order_book_history:
            return {'execution_risk': 0.5, 'adverse_selection_risk': 0.3, 'liquidity_risk': 0.4}
        
        latest_metrics = self.order_book_history[-1]
        
        return {
            'execution_risk': latest_metrics.market_impact_estimate * 100,
            'adverse_selection_risk': latest_metrics.adverse_selection_risk,
            'liquidity_risk': 1 - latest_metrics.liquidity_score
        }
    
    async def _update_trade_based_metrics(self, trade_data: Dict) -> None:
        """Update metrics based on new trade data"""
        # Update trade-based models and estimators
        pass
    
    async def _detect_trade_anomalies(self, trade_data: Dict) -> None:
        """Detect anomalous trade patterns"""
        try:
            if len(self.trade_history) < 50:
                return
            
            # Use isolation forest for anomaly detection
            recent_trades = list(self.trade_history)[-50:]
            features = []
            
            for trade in recent_trades:
                feature_vector = [
                    trade.get('size', 0),
                    trade.get('price', 0),
                    1 if trade.get('direction') in [TradeDirection.AGGRESSIVE_BUY, TradeDirection.AGGRESSIVE_SELL] else 0
                ]
                features.append(feature_vector)
            
            if len(features) >= 50:
                # Fit and predict anomalies
                predictions = self.isolation_forest.fit_predict(features)
                anomalies = [i for i, pred in enumerate(predictions) if pred == -1]
                
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} anomalous trades")
                    
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
    
    async def _create_default_metrics(self, order_book_snapshot: Dict) -> OrderBookMetrics:
        """Create default metrics when analysis fails"""
        return OrderBookMetrics(
            timestamp=order_book_snapshot.get('timestamp', datetime.now()),
            symbol=order_book_snapshot.get('symbol', 'UNKNOWN'),
            spread_bps=2.0,
            mid_price=1.1000,
            weighted_mid_price=1.1000,
            total_depth=1000000,
            bid_depth=500000,
            ask_depth=500000,
            depth_imbalance=0.0,
            price_levels_bid=5,
            price_levels_ask=5,
            cumulative_depth_bid=[100000, 200000, 300000, 400000, 500000],
            cumulative_depth_ask=[100000, 200000, 300000, 400000, 500000],
            volume_imbalance=0.0,
            volume_skew=0.0,
            large_order_presence=False,
            liquidity_score=0.5,
            market_impact_estimate=0.0001,
            adverse_selection_risk=0.3,
            order_flow_imbalance=0.0,
            trade_size_distribution={'mean': 100000, 'std': 50000, 'skew': 0.0, 'kurtosis': 0.0},
            hidden_liquidity_estimate=0.0
        )
    
    async def _create_default_signal(self, symbol: str) -> MicrostructureSignal:
        """Create default signal when generation fails"""
        return MicrostructureSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            order_book_pressure=0.0,
            liquidity_pressure=0.0,
            toxicity_score=0.3,
            short_term_momentum=0.0,
            mean_reversion_signal=0.0,
            breakout_potential=0.0,
            market_regime="NORMAL",
            regime_confidence=0.5,
            suggested_action="HOLD",
            confidence=0.5,
            position_size_multiplier=0.5,
            execution_risk=0.5,
            adverse_selection_risk=0.3,
            liquidity_risk=0.4
        )
    
    def _get_default_basic_metrics(self) -> Dict[str, float]:
        """Get default basic metrics"""
        return {
            'spread_bps': 2.0,
            'mid_price': 1.1000,
            'weighted_mid_price': 1.1000,
            'best_bid_size': 1000000,
            'best_ask_size': 1000000,
            'size_imbalance': 0.0
        }
    
    async def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current analyzer state"""
        return {
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'order_book_history_size': len(self.order_book_history),
            'trade_history_size': len(self.trade_history),
            'signal_history_size': len(self.microstructure_signals),
            'last_analysis_time': self.last_analysis_time
        }

# Supporting Classes

class OrderBookAnalyzer:
    """Specialized order book analysis"""
    
    def __init__(self):
        self.state_history = deque(maxlen=100)
    
    async def analyze_order_book_state(self, order_book_snapshot: Dict) -> OrderBookState:
        """Analyze order book state"""
        # Implementation would go here
        return OrderBookState.NORMAL

class TradeAnalyzer:
    """Specialized trade analysis"""
    
    def __init__(self):
        self.trade_patterns = deque(maxlen=1000)
    
    async def analyze_trade_patterns(self, trade_data: Dict) -> Dict[str, Any]:
        """Analyze trade patterns"""
        # Implementation would go here
        return {}

class MarketImpactEstimator:
    """Market impact estimation"""
    
    def __init__(self):
        self.impact_history = deque(maxlen=500)
    
    async def estimate_impact(self, order_size: float, current_liquidity: float) -> float:
        """Estimate market impact for given order size"""
        # Simplified impact model
        return order_size / current_liquidity * 0.0001

class AdvancedRegimeDetector:
    """Advanced regime detection"""
    
    def __init__(self):
        self.regime_model = None
    
    async def detect_regime(self, features: Dict) -> Dict[str, Any]:
        """Detect market regime"""
        # Implementation would go here
        return {'regime': 'NORMAL', 'confidence': 0.8}

class MicrostructureAnomalyDetector:
    """Microstructure anomaly detection"""
    
    def __init__(self):
        self.anomaly_model = IsolationForest()
    
    async def detect_anomalies(self, data: List[Dict]) -> List[int]:
        """Detect anomalies in microstructure data"""
        # Implementation would go here
        return []

# Example usage
async def main():
    """Example usage of microstructure analyzer"""
    print("🚀 Starting Advanced Microstructure Analyzer...")
    
    # Create analyzer
    analyzer = AdvancedMicrostructureAnalyzer()
    
    # Generate sample order book data
    sample_order_books = []
    base_price = 1.1000
    
    for i in range(100):
        order_book = {
            'symbol': 'EUR/USD',
            'timestamp': datetime.now() + timedelta(seconds=i*0.1),
            'bids': [
                [base_price - 0.0001, 1000000],
                [base_price - 0.0002, 800000],
                [base_price - 0.0003, 600000]
            ],
            'asks': [
                [base_price + 0.0001, 900000],
                [base_price + 0.0002, 700000],
                [base_price + 0.0003, 500000]
            ]
        }
        sample_order_books.append(order_book)
    
    # Process sample order books
    print("📊 Processing sample order books...")
    for i, order_book in enumerate(sample_order_books[:30]):
        metrics = await analyzer.analyze_order_book(order_book)
        if i % 10 == 0:  # Print every 10th result
            print(f"Order Book {i}: Spread {metrics.spread_bps:.2f} bps, "
                  f"Liquidity Score: {metrics.liquidity_score:.3f}")
    
    # Generate microstructure signals
    print("\n🎯 Generating microstructure signals...")
    signal = await analyzer.generate_microstructure_signals("EUR/USD")
    
    print(f"\n📈 MICROSTRUCTURE SIGNAL:")
    print(f"Action: {signal.suggested_action}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Order Book Pressure: {signal.order_book_pressure:.3f}")
    print(f"Liquidity Pressure: {signal.liquidity_pressure:.3f}")
    print(f"Toxicity Score: {signal.toxicity_score:.3f}")
    print(f"Market Regime: {signal.market_regime}")
    print(f"Position Size Multiplier: {signal.position_size_multiplier:.2f}")
    
    # Get performance stats
    stats = await analyzer.get_performance_stats()
    print(f"\n📊 PERFORMANCE STATISTICS:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Get current state
    state = await analyzer.get_current_state()
    print(f"\n🔧 ANALYZER STATE:")
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Advanced Microstructure Analyzer demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
"""
High-Frequency Trading Optimizer for Forex Trading Bot
Ultra-low latency optimization, market microstructure, and execution algorithms
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from scipy import stats
import numba
from numba import jit, float64, int64
import psutil
import socket
import asyncio
import aiohttp

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFTStrategy(Enum):
    """HFT Strategy Types"""
    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    LIQUIDITY_DETECTION = "liquidity_detection"
    LATENCY_ARBITRAGE = "latency_arbitrage"

class OrderType(Enum):
    """Order Types for HFT"""
    LIMIT = "limit"
    MARKET = "market"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    POST_ONLY = "post_only"

@dataclass
class HFTOrder:
    """HFT Order Structure"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    timestamp: datetime
    strategy: HFTStrategy
    priority: int = 1
    time_in_force: float = 0.1  # seconds for HFT orders
    parent_strategy: str = None

@dataclass
class MarketMicrostructure:
    """Market Microstructure Data"""
    timestamp: datetime
    symbol: str
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    spread: float
    mid_price: float
    order_book_imbalance: float
    price_momentum: float
    volatility: float
    trade_flow: float
    market_depth: Dict[float, float]  # price -> quantity

@dataclass
class HFTPortfolio:
    """HFT Portfolio State"""
    timestamp: datetime
    total_equity: float
    used_margin: float
    open_positions: Dict[str, float]
    pending_orders: int
    execution_latency: float
    fill_rate: float
    pnl_1min: float
    pnl_5min: float

class LatencyOptimizer:
    """Ultra-low latency optimization"""
    
    def __init__(self):
        self.latency_measurements = []
        self.optimal_batch_size = 10
        self.parallel_workers = 4
        self.cache_size = 1000
        
        # JIT compilation for critical functions
        self._compile_optimized_functions()
        
        logger.info("Latency Optimizer initialized")
    
    def _compile_optimized_functions(self):
        """Compile performance-critical functions with Numba"""
        
        @jit(float64(float64[:], float64[:]), nopython=True)
        def numba_correlation(x, y):
            return np.corrcoef(x, y)[0, 1]
        
        @jit(float64(float64[:]), nopython=True)
        def numba_volatility(returns):
            return np.std(returns) * np.sqrt(252)
        
        @jit(float64(float64[:], float64[:]), nopython=True)
        def numba_order_book_imbalance(bid_sizes, ask_sizes):
            total_bid = np.sum(bid_sizes)
            total_ask = np.sum(ask_sizes)
            return (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0.0
        
        self.numba_correlation = numba_correlation
        self.numba_volatility = numba_volatility
        self.numba_order_book_imbalance = numba_order_book_imbalance
    
    def measure_latency(self, operation: Callable) -> float:
        """Measure operation latency in microseconds"""
        start_time = time.perf_counter_ns()
        operation()
        end_time = time.perf_counter_ns()
        latency_us = (end_time - start_time) / 1000  # Convert to microseconds
        self.latency_measurements.append(latency_us)
        return latency_us
    
    def optimize_network_latency(self):
        """Optimize network-related latency"""
        # Set TCP_NODELAY for lower latency
        try:
            # This would be implemented in actual network code
            pass
        except Exception as e:
            logger.warning(f"Network optimization failed: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get latency performance report"""
        if not self.latency_measurements:
            return {}
        
        latencies = np.array(self.latency_measurements)
        return {
            'average_latency_us': np.mean(latencies),
            'p95_latency_us': np.percentile(latencies, 95),
            'p99_latency_us': np.percentile(latencies, 99),
            'min_latency_us': np.min(latencies),
            'max_latency_us': np.max(latencies),
            'measurement_count': len(latencies)
        }

class MarketMakingStrategy:
    """Market Making HFT Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spread_target = config.get('spread_target', 0.0002)
        self.position_limit = config.get('position_limit', 0.5)
        self.quote_size = config.get('quote_size', 0.1)
        self.risk_aversion = config.get('risk_aversion', 0.1)
        self.inventory_target = 0.0
        
        # Performance tracking
        self.quotes_posted = 0
        self.quotes_filled = 0
        self.adverse_selection = 0.0
        
        logger.info("Market Making Strategy initialized")
    
    def calculate_quotes(self, microstructure: MarketMicrostructure, 
                        current_inventory: float) -> Tuple[Optional[HFTOrder], Optional[HFTOrder]]:
        """
        Calculate bid and ask quotes for market making
        
        Args:
            microstructure: Current market microstructure
            current_inventory: Current position inventory
            
        Returns:
            Tuple of (bid_order, ask_order)
        """
        try:
            # Inventory adjustment
            inventory_penalty = self.risk_aversion * current_inventory
            half_spread = self.spread_target / 2
            
            # Calculate quote prices with inventory adjustment
            bid_price = microstructure.mid_price - half_spread - inventory_penalty
            ask_price = microstructure.mid_price + half_spread - inventory_penalty
            
            # Adjust for market conditions
            if microstructure.volatility > 0.15:
                # Widen spreads in high volatility
                bid_price -= 0.0001
                ask_price += 0.0001
            
            # Check if quotes are profitable
            if ask_price - bid_price < self.spread_target * 0.8:
                return None, None  # Spread too tight
            
            # Create orders
            bid_order = HFTOrder(
                order_id=f"MM_BID_{int(time.time()*1000)}",
                symbol=microstructure.symbol,
                order_type=OrderType.POST_ONLY,
                side='buy',
                price=bid_price,
                quantity=self.quote_size,
                timestamp=datetime.now(),
                strategy=HFTStrategy.MARKET_MAKING,
                priority=2
            )
            
            ask_order = HFTOrder(
                order_id=f"MM_ASK_{int(time.time()*1000)}",
                symbol=microstructure.symbol,
                order_type=OrderType.POST_ONLY,
                side='sell',
                price=ask_price,
                quantity=self.quote_size,
                timestamp=datetime.now(),
                strategy=HFTStrategy.MARKET_MAKING,
                priority=2
            )
            
            self.quotes_posted += 2
            
            return bid_order, ask_order
            
        except Exception as e:
            logger.error(f"Error calculating market making quotes: {e}")
            return None, None
    
    def update_inventory(self, fill_side: str, quantity: float):
        """Update inventory based on fills"""
        if fill_side == 'buy':
            self.inventory_target -= quantity
        else:  # sell
            self.inventory_target += quantity
        
        self.quotes_filled += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get market making performance metrics"""
        fill_rate = self.quotes_filled / self.quotes_posted if self.quotes_posted > 0 else 0
        
        return {
            'quotes_posted': self.quotes_posted,
            'quotes_filled': self.quotes_filled,
            'fill_rate': fill_rate,
            'inventory_target': self.inventory_target,
            'adverse_selection': self.adverse_selection
        }

class StatisticalArbitrage:
    """Statistical Arbitrage HFT Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cointegration_threshold = config.get('cointegration_threshold', 0.05)
        self.z_score_entry = config.get('z_score_entry', 2.0)
        self.z_score_exit = config.get('z_score_exit', 0.5)
        self.lookback_period = config.get('lookback_period', 100)
        self.position_size = config.get('position_size', 0.05)
        
        # Pair tracking
        self.pairs = {}
        self.historical_spreads = {}
        
        logger.info("Statistical Arbitrage Strategy initialized")
    
    def add_pair(self, pair: Tuple[str, str], hedge_ratio: float):
        """Add trading pair for statistical arbitrage"""
        self.pairs[pair] = {
            'hedge_ratio': hedge_ratio,
            'spread_history': [],
            'current_zscore': 0.0,
            'position': 0.0
        }
    
    def update_pair_spread(self, pair: Tuple[str, str], price1: float, price2: float):
        """Update spread for trading pair"""
        if pair not in self.pairs:
            return
        
        hedge_ratio = self.pairs[pair]['hedge_ratio']
        spread = price1 - hedge_ratio * price2
        
        self.pairs[pair]['spread_history'].append(spread)
        
        # Maintain lookback period
        if len(self.pairs[pair]['spread_history']) > self.lookback_period:
            self.pairs[pair]['spread_history'].pop(0)
        
        # Calculate z-score
        if len(self.pairs[pair]['spread_history']) >= 20:  # Minimum for meaningful stats
            spreads = np.array(self.pairs[pair]['spread_history'])
            mean = np.mean(spreads)
            std = np.std(spreads)
            
            if std > 0:
                z_score = (spread - mean) / std
                self.pairs[pair]['current_zscore'] = z_score
    
    def generate_arbitrage_signals(self, pair: Tuple[str, str]) -> List[HFTOrder]:
        """Generate arbitrage signals for pair"""
        if pair not in self.pairs:
            return []
        
        z_score = self.pairs[pair]['current_zscore']
        current_position = self.pairs[pair]['position']
        
        orders = []
        
        # Entry signals
        if abs(z_score) > self.z_score_entry and abs(current_position) < self.position_size:
            if z_score > self.z_score_entry:
                # Short spread (sell asset1, buy asset2)
                orders.extend(self._create_arbitrage_orders(pair, 'short'))
            elif z_score < -self.z_score_entry:
                # Long spread (buy asset1, sell asset2)
                orders.extend(self._create_arbitrage_orders(pair, 'long'))
        
        # Exit signals
        elif abs(z_score) < self.z_score_exit and abs(current_position) > 0:
            orders.extend(self._create_arbitrage_orders(pair, 'flat'))
        
        return orders
    
    def _create_arbitrage_orders(self, pair: Tuple[str, str], direction: str) -> List[HFTOrder]:
        """Create arbitrage orders for pair and direction"""
        orders = []
        timestamp = datetime.now()
        
        if direction == 'long':
            # Buy asset1, sell asset2
            orders.append(HFTOrder(
                order_id=f"ARB_LONG_{int(time.time()*1000)}_1",
                symbol=pair[0],
                order_type=OrderType.IOC,
                side='buy',
                price=0.0,  # Market order
                quantity=self.position_size,
                timestamp=timestamp,
                strategy=HFTStrategy.ARBITRAGE,
                priority=1
            ))
            orders.append(HFTOrder(
                order_id=f"ARB_LONG_{int(time.time()*1000)}_2",
                symbol=pair[1],
                order_type=OrderType.IOC,
                side='sell',
                price=0.0,  # Market order
                quantity=self.position_size * self.pairs[pair]['hedge_ratio'],
                timestamp=timestamp,
                strategy=HFTStrategy.ARBITRAGE,
                priority=1
            ))
            self.pairs[pair]['position'] = self.position_size
        
        elif direction == 'short':
            # Sell asset1, buy asset2
            orders.append(HFTOrder(
                order_id=f"ARB_SHORT_{int(time.time()*1000)}_1",
                symbol=pair[0],
                order_type=OrderType.IOC,
                side='sell',
                price=0.0,  # Market order
                quantity=self.position_size,
                timestamp=timestamp,
                strategy=HFTStrategy.ARBITRAGE,
                priority=1
            ))
            orders.append(HFTOrder(
                order_id=f"ARB_SHORT_{int(time.time()*1000)}_2",
                symbol=pair[1],
                order_type=OrderType.IOC,
                side='buy',
                price=0.0,  # Market order
                quantity=self.position_size * self.pairs[pair]['hedge_ratio'],
                timestamp=timestamp,
                strategy=HFTStrategy.ARBITRAGE,
                priority=1
            ))
            self.pairs[pair]['position'] = -self.position_size
        
        elif direction == 'flat':
            # Close position
            current_position = self.pairs[pair]['position']
            if current_position > 0:
                # Sell asset1, buy asset2
                orders.extend(self._create_arbitrage_orders(pair, 'short'))
            elif current_position < 0:
                # Buy asset1, sell asset2
                orders.extend(self._create_arbitrage_orders(pair, 'long'))
            self.pairs[pair]['position'] = 0.0
        
        return orders

class LiquidityDetector:
    """Liquidity Detection and Prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.liquidity_threshold = config.get('liquidity_threshold', 1000000)
        self.volume_lookback = config.get('volume_lookback', 50)
        self.flow_decay = config.get('flow_decay', 0.95)
        
        # Liquidity tracking
        self.volume_profiles = {}
        self.trade_flows = {}
        self.liquidity_zones = {}
        
        logger.info("Liquidity Detector initialized")
    
    def update_trade_flow(self, symbol: str, trade_size: float, is_buy: bool):
        """Update trade flow for symbol"""
        if symbol not in self.trade_flows:
            self.trade_flows[symbol] = {'buy_flow': 0.0, 'sell_flow': 0.0}
        
        if is_buy:
            self.trade_flows[symbol]['buy_flow'] = (
                self.flow_decay * self.trade_flows[symbol]['buy_flow'] + trade_size
            )
        else:
            self.trade_flows[symbol]['sell_flow'] = (
                self.flow_decay * self.trade_flows[symbol]['sell_flow'] + trade_size
            )
    
    def detect_liquidity_zones(self, symbol: str, order_book: Dict[float, float]) -> Dict[str, List[float]]:
        """Detect liquidity zones in order book"""
        bids = {k: v for k, v in order_book.items() if k < order_book['mid_price']}
        asks = {k: v for k, v in order_book.items() if k > order_book['mid_price']}
        
        # Find significant liquidity levels
        bid_zones = self._find_liquidity_clusters(bids)
        ask_zones = self._find_liquidity_clusters(asks)
        
        self.liquidity_zones[symbol] = {
            'bid_zones': bid_zones,
            'ask_zones': ask_zones,
            'timestamp': datetime.now()
        }
        
        return self.liquidity_zones[symbol]
    
    def _find_liquidity_clusters(self, levels: Dict[float, float]) -> List[float]:
        """Find liquidity clusters in price levels"""
        if not levels:
            return []
        
        prices = np.array(list(levels.keys()))
        quantities = np.array(list(levels.values()))
        
        # Simple clustering - in production, use more sophisticated methods
        significant_levels = prices[quantities > np.percentile(quantities, 75)]
        
        return significant_levels.tolist()
    
    def predict_liquidity_impact(self, symbol: str, order_size: float) -> float:
        """Predict liquidity impact of order size"""
        if symbol not in self.volume_profiles:
            return 0.01  # Default impact
        
        recent_volume = self.volume_profiles[symbol][-self.volume_lookback:]
        avg_volume = np.mean(recent_volume) if recent_volume else order_size
        
        # Simple impact model - in production, use more sophisticated models
        impact = min(0.05, (order_size / avg_volume) * 0.01)
        
        return impact

class HFTOrderManager:
    """HFT Order Management and Execution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pending_orders: Dict[str, HFTOrder] = {}
        self.order_history: List[HFTOrder] = []
        self.max_pending_orders = config.get('max_pending_orders', 50)
        self.order_timeout = config.get('order_timeout', 0.5)  # seconds
        
        # Execution optimization
        self.batch_processor = ThreadPoolExecutor(max_workers=4)
        self.latency_optimizer = LatencyOptimizer()
        
        # Performance tracking
        self.execution_stats = {
            'orders_sent': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'average_fill_latency': 0.0,
            'fill_rate': 0.0
        }
        
        logger.info("HFT Order Manager initialized")
    
    def submit_order(self, order: HFTOrder) -> bool:
        """Submit HFT order with optimization"""
        try:
            # Check limits
            if len(self.pending_orders) >= self.max_pending_orders:
                logger.warning("Maximum pending orders reached")
                return False
            
            # Optimize order submission
            submission_latency = self.latency_optimizer.measure_latency(
                lambda: self._submit_single_order(order)
            )
            
            self.pending_orders[order.order_id] = order
            self.order_history.append(order)
            self.execution_stats['orders_sent'] += 1
            
            logger.debug(f"Order submitted: {order.order_id} in {submission_latency:.2f}μs")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting order {order.order_id}: {e}")
            return False
    
    def submit_order_batch(self, orders: List[HFTOrder]) -> List[bool]:
        """Submit batch of orders in parallel"""
        if not orders:
            return []
        
        # Sort by priority
        orders.sort(key=lambda x: x.priority, reverse=True)
        
        # Submit in parallel
        futures = []
        for order in orders:
            future = self.batch_processor.submit(self.submit_order, order)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result(timeout=1.0))
            except Exception as e:
                logger.error(f"Batch submission failed: {e}")
                results.append(False)
        
        return results
    
    def _submit_single_order(self, order: HFTOrder):
        """Simulate single order submission"""
        # In production, this would interface with exchange API
        time.sleep(0.001)  # Simulate network latency
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            self.execution_stats['orders_cancelled'] += 1
            logger.debug(f"Order cancelled: {order_id}")
            return True
        return False
    
    def cancel_all_orders(self, symbol: str = None):
        """Cancel all orders or orders for specific symbol"""
        if symbol:
            to_cancel = [oid for oid, order in self.pending_orders.items() 
                        if order.symbol == symbol]
        else:
            to_cancel = list(self.pending_orders.keys())
        
        for order_id in to_cancel:
            self.cancel_order(order_id)
    
    def process_order_fill(self, order_id: str, fill_price: float, fill_quantity: float):
        """Process order fill notification"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            del self.pending_orders[order_id]
            
            self.execution_stats['orders_filled'] += 1
            fill_rate = self.execution_stats['orders_filled'] / self.execution_stats['orders_sent']
            self.execution_stats['fill_rate'] = fill_rate
            
            logger.debug(f"Order filled: {order_id} at {fill_price}")
    
    def cleanup_expired_orders(self):
        """Clean up expired orders"""
        current_time = datetime.now()
        expired_orders = []
        
        for order_id, order in self.pending_orders.items():
            order_age = (current_time - order.timestamp).total_seconds()
            if order_age > self.order_timeout:
                expired_orders.append(order_id)
        
        for order_id in expired_orders:
            self.cancel_order(order_id)
        
        if expired_orders:
            logger.info(f"Cleaned up {len(expired_orders)} expired orders")

class HFTOptimizer:
    """
    Main HFT Optimizer coordinating all HFT strategies and optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.latency_optimizer = LatencyOptimizer()
        self.order_manager = HFTOrderManager(self.config.get('order_manager', {}))
        self.market_maker = MarketMakingStrategy(self.config.get('market_making', {}))
        self.stat_arb = StatisticalArbitrage(self.config.get('statistical_arbitrage', {}))
        self.liquidity_detector = LiquidityDetector(self.config.get('liquidity_detection', {}))
        
        # State tracking
        self.portfolio = HFTPortfolio(
            timestamp=datetime.now(),
            total_equity=10000.0,
            used_margin=0.0,
            open_positions={},
            pending_orders=0,
            execution_latency=0.0,
            fill_rate=0.0,
            pnl_1min=0.0,
            pnl_5min=0.0
        )
        
        # Performance monitoring
        self.performance_history = []
        self.strategy_performance = {}
        
        # Start monitoring thread
        self.monitoring_thread = None
        self.running = False
        
        logger.info("HFT Optimizer initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default HFT configuration"""
        return {
            'market_making': {
                'spread_target': 0.0002,
                'position_limit': 0.5,
                'quote_size': 0.1,
                'risk_aversion': 0.1
            },
            'statistical_arbitrage': {
                'cointegration_threshold': 0.05,
                'z_score_entry': 2.0,
                'z_score_exit': 0.5,
                'lookback_period': 100,
                'position_size': 0.05
            },
            'liquidity_detection': {
                'liquidity_threshold': 1000000,
                'volume_lookback': 50,
                'flow_decay': 0.95
            },
            'order_manager': {
                'max_pending_orders': 50,
                'order_timeout': 0.5
            },
            'monitoring_interval': 1.0  # seconds
        }
    
    def start(self):
        """Start HFT optimization"""
        if self.running:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("HFT Optimizer started")
    
    def stop(self):
        """Stop HFT optimization"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Cancel all pending orders
        self.order_manager.cancel_all_orders()
        
        logger.info("HFT Optimizer stopped")
    
    def _monitoring_loop(self):
        """Main HFT monitoring and optimization loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Update portfolio state
                self._update_portfolio()
                
                # Generate HFT signals
                self._generate_hft_signals()
                
                # Cleanup expired orders
                self.order_manager.cleanup_expired_orders()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Calculate loop duration and sleep if needed
                loop_duration = time.time() - start_time
                sleep_time = max(0, self.config['monitoring_interval'] - loop_duration)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in HFT monitoring loop: {e}")
                time.sleep(1.0)  # Prevent tight loop on errors
    
    def _update_portfolio(self):
        """Update portfolio state"""
        self.portfolio.timestamp = datetime.now()
        self.portfolio.pending_orders = len(self.order_manager.pending_orders)
        self.portfolio.fill_rate = self.order_manager.execution_stats['fill_rate']
        
        # Update P&L (simplified)
        # In production, this would calculate actual P&L from positions
        self.portfolio.pnl_1min = np.random.normal(0, 10)  # Simulated P&L
        self.portfolio.pnl_5min = np.random.normal(0, 50)  # Simulated P&L
    
    def _generate_hft_signals(self):
        """Generate HFT trading signals"""
        # This would use real market data in production
        # For demo, we'll generate simulated signals
        
        # Simulate market making signals
        simulated_microstructure = MarketMicrostructure(
            timestamp=datetime.now(),
            symbol="EUR/USD",
            best_bid=1.0850,
            best_ask=1.0852,
            bid_size=1000000,
            ask_size=800000,
            spread=0.0002,
            mid_price=1.0851,
            order_book_imbalance=0.1,
            price_momentum=0.0001,
            volatility=0.12,
            trade_flow=0.05,
            market_depth={1.0848: 500000, 1.0849: 800000, 1.0850: 1000000,
                        1.0851: 800000, 1.0852: 800000, 1.0853: 600000}
        )
        
        # Market making
        bid_order, ask_order = self.market_maker.calculate_quotes(
            simulated_microstructure, 
            self.portfolio.open_positions.get('EUR/USD', 0.0)
        )
        
        if bid_order:
            self.order_manager.submit_order(bid_order)
        if ask_order:
            self.order_manager.submit_order(ask_order)
        
        # Statistical arbitrage
        # This would require actual pair data in production
    
    def _update_performance_metrics(self):
        """Update HFT performance metrics"""
        performance_snapshot = {
            'timestamp': datetime.now(),
            'portfolio': asdict(self.portfolio),
            'order_stats': self.order_manager.execution_stats.copy(),
            'market_making': self.market_maker.get_performance_metrics(),
            'latency': self.latency_optimizer.get_performance_report()
        }
        
        self.performance_history.append(performance_snapshot)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive HFT optimization report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio': asdict(self.portfolio),
            'performance': {
                'total_orders': self.order_manager.execution_stats['orders_sent'],
                'fill_rate': self.order_manager.execution_stats['fill_rate'],
                'pending_orders': len(self.order_manager.pending_orders),
                'market_making_performance': self.market_maker.get_performance_metrics(),
                'latency_performance': self.latency_optimizer.get_performance_report()
            },
            'strategies_active': {
                'market_making': True,
                'statistical_arbitrage': len(self.stat_arb.pairs) > 0,
                'liquidity_detection': True
            },
            'system_health': {
                'running': self.running,
                'thread_alive': self.monitoring_thread.is_alive() if self.monitoring_thread else False,
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
        }
    
    def add_arbitrage_pair(self, pair: Tuple[str, str], hedge_ratio: float):
        """Add pair for statistical arbitrage"""
        self.stat_arb.add_pair(pair, hedge_ratio)
        logger.info(f"Added arbitrage pair: {pair[0]}/{pair[1]} with hedge ratio {hedge_ratio}")


# Example usage and testing
if __name__ == "__main__":
    # Test the HFT Optimizer
    print("Testing HFT Optimizer...")
    
    try:
        # Initialize HFT optimizer
        hft_optimizer = HFTOptimizer()
        
        # Test latency optimization
        print("Testing latency optimization...")
        latency_optimizer = hft_optimizer.latency_optimizer
        
        def test_operation():
            time.sleep(0.001)  # Simulate some work
        
        for i in range(10):
            latency = latency_optimizer.measure_latency(test_operation)
            print(f"Operation {i+1}: {latency:.2f}μs")
        
        # Test market making
        print("\nTesting market making...")
        market_maker = hft_optimizer.market_maker
        
        simulated_microstructure = MarketMicrostructure(
            timestamp=datetime.now(),
            symbol="EUR/USD",
            best_bid=1.0850,
            best_ask=1.0852,
            bid_size=1000000,
            ask_size=800000,
            spread=0.0002,
            mid_price=1.0851,
            order_book_imbalance=0.1,
            price_momentum=0.0001,
            volatility=0.12,
            trade_flow=0.05,
            market_depth={}
        )
        
        bid_order, ask_order = market_maker.calculate_quotes(simulated_microstructure, 0.0)
        if bid_order and ask_order:
            print(f"Market Making Quotes:")
            print(f"  Bid: {bid_order.price:.5f} for {bid_order.quantity:.2f} lots")
            print(f"  Ask: {ask_order.price:.5f} for {ask_order.quantity:.2f} lots")
        
        # Test order management
        print("\nTesting order management...")
        order_manager = hft_optimizer.order_manager
        
        if bid_order:
            order_manager.submit_order(bid_order)
            print(f"Order submitted: {bid_order.order_id}")
        
        # Test statistical arbitrage
        print("\nTesting statistical arbitrage...")
        hft_optimizer.add_arbitrage_pair(("EUR/USD", "GBP/USD"), 0.9)
        
        # Update spreads
        hft_optimizer.stat_arb.update_pair_spread(("EUR/USD", "GBP/USD"), 1.0850, 1.2650)
        
        # Generate signals
        signals = hft_optimizer.stat_arb.generate_arbitrage_signals(("EUR/USD", "GBP/USD"))
        print(f"Generated {len(signals)} arbitrage signals")
        
        # Start HFT optimizer
        print("\nStarting HFT optimizer...")
        hft_optimizer.start()
        time.sleep(3)  # Let it run for a bit
        
        # Get performance report
        report = hft_optimizer.get_optimization_report()
        print(f"HFT Performance Report:")
        print(f"  Fill Rate: {report['performance']['fill_rate']:.1%}")
        print(f"  Pending Orders: {report['performance']['pending_orders']}")
        print(f"  Total Orders: {report['performance']['total_orders']}")
        
        # Stop HFT optimizer
        hft_optimizer.stop()
        
        print(f"\n✅ HFT Optimizer test completed successfully!")
        
    except Exception as e:
        print(f"❌ HFT Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
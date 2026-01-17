"""
Smart Order Router for FOREX TRADING BOT
Advanced order routing with intelligent execution logic
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class Exchange(Enum):
    BINANCE = "binance"
    EXNESS = "exness"
    DERIBIT = "deribit"  # Alternative

@dataclass
class OrderBookSnapshot:
    """Order book data snapshot"""
    exchange: Exchange
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]
    spread: float
    mid_price: float

@dataclass
class RoutingDecision:
    """Smart routing decision"""
    exchange: Exchange
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    reason: str = ""
    confidence: float = 0.0

class SmartOrderRouter:
    """
    Advanced order router with intelligent execution logic
    Optimizes order execution across multiple exchanges
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self.order_books = {}
        self.latency_data = {}
        self.fee_structures = {}
        
        # Routing parameters
        self.min_liquidity_threshold = config.get('min_liquidity', 10000)
        self.max_spread_ratio = config.get('max_spread_ratio', 0.0002)  # 0.02%
        self.slippage_tolerance = config.get('slippage_tolerance', 0.0005)
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_executions': 0,
            'average_slippage': 0.0,
            'latency_metrics': {}
        }
        
        logger.info("SmartOrderRouter initialized")

    async def initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Binance
            if self.config.get('binance_enabled', True):
                from binance.client import Client
                self.exchanges[Exchange.BINANCE] = Client(
                    api_key=self.config['binance_api_key'],
                    api_secret=self.config['binance_secret_key']
                )
            
            # Initialize Exness
            if self.config.get('exness_enabled', True):
                # Exness API integration would go here
                self.exchanges[Exchange.EXNESS] = None  # Placeholder
            
            logger.info("Exchange connections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
            raise

    async def get_order_book(self, exchange: Exchange, symbol: str) -> OrderBookSnapshot:
        """Fetch real-time order book from exchange"""
        try:
            if exchange == Exchange.BINANCE:
                order_book = await self._fetch_binance_order_book(symbol)
            elif exchange == Exchange.EXNESS:
                order_book = await self._fetch_exness_order_book(symbol)
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
            
            self.order_books[f"{exchange.value}_{symbol}"] = order_book
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book from {exchange}: {e}")
            raise

    async def _fetch_binance_order_book(self, symbol: str) -> OrderBookSnapshot:
        """Fetch order book from Binance"""
        try:
            # Using sync call for simplicity, can be made async
            depth = self.exchanges[Exchange.BINANCE].get_order_book(symbol=symbol)
            
            bids = [(float(bid[0]), float(bid[1])) for bid in depth['bids'][:20]]
            asks = [(float(ask[0]), float(ask[1])) for ask in depth['asks'][:20]]
            
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid if best_ask and best_bid else 0
            mid_price = (best_ask + best_bid) / 2
            
            return OrderBookSnapshot(
                exchange=Exchange.BINANCE,
                symbol=symbol,
                timestamp=time.time(),
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mid_price
            )
            
        except Exception as e:
            logger.error(f"Binance order book fetch failed: {e}")
            raise

    async def _fetch_exness_order_book(self, symbol: str) -> OrderBookSnapshot:
        """Fetch order book from Exness (placeholder implementation)"""
        # This would be replaced with actual Exness API calls
        try:
            # Mock data for demonstration
            mock_price = 1.0850  # EUR/USD mock price
            spread = 0.0001
            
            bids = [(mock_price - spread/2 - i*0.0001, 10000) for i in range(10)]
            asks = [(mock_price + spread/2 + i*0.0001, 10000) for i in range(10)]
            
            return OrderBookSnapshot(
                exchange=Exchange.EXNESS,
                symbol=symbol,
                timestamp=time.time(),
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mock_price
            )
            
        except Exception as e:
            logger.error(f"Exness order book fetch failed: {e}")
            raise

    def calculate_liquidity_metrics(self, order_book: OrderBookSnapshot) -> Dict:
        """Calculate liquidity metrics from order book"""
        try:
            bids_volume = sum(qty for _, qty in order_book.bids[:10])
            asks_volume = sum(qty for _, qty in order_book.asks[:10])
            total_volume = bids_volume + asks_volume
            
            # Calculate depth at different levels
            depth_1 = order_book.bids[0][1] + order_book.asks[0][1] if order_book.bids and order_book.asks else 0
            depth_5 = sum(qty for _, qty in order_book.bids[:5]) + sum(qty for _, qty in order_book.asks[:5])
            
            # Calculate spread percentage
            spread_pct = (order_book.spread / order_book.mid_price) * 100 if order_book.mid_price else 0
            
            return {
                'total_volume': total_volume,
                'bid_ask_ratio': bids_volume / asks_volume if asks_volume > 0 else 1,
                'depth_1_level': depth_1,
                'depth_5_levels': depth_5,
                'spread_percentage': spread_pct,
                'market_impact': self.estimate_market_impact(order_book, 10000)  # Estimate for 10k units
            }
            
        except Exception as e:
            logger.error(f"Liquidity metrics calculation failed: {e}")
            return {}

    def estimate_market_impact(self, order_book: OrderBookSnapshot, quantity: float) -> float:
        """Estimate market impact for a given order size"""
        try:
            if quantity <= 0:
                return 0.0
                
            remaining_qty = quantity
            total_cost = 0.0
            levels_used = 0
            
            # Calculate cost to buy (using asks)
            for price, level_qty in order_book.asks:
                if remaining_qty <= 0:
                    break
                    
                qty_to_take = min(remaining_qty, level_qty)
                total_cost += qty_to_take * price
                remaining_qty -= qty_to_take
                levels_used += 1
            
            if remaining_qty > 0:
                # If order book doesn't have enough liquidity, use last price with penalty
                last_price = order_book.asks[-1][0] if order_book.asks else order_book.mid_price
                total_cost += remaining_qty * last_price * 1.01  # 1% penalty
            
            avg_price = total_cost / quantity
            market_impact = (avg_price - order_book.mid_price) / order_book.mid_price
            
            return max(0.0, market_impact)
            
        except Exception as e:
            logger.error(f"Market impact estimation failed: {e}")
            return 0.0

    async def find_optimal_routing(self, symbol: str, order_type: OrderType, 
                                 quantity: float, direction: str) -> RoutingDecision:
        """
        Find optimal routing for an order
        direction: 'buy' or 'sell'
        """
        try:
            logger.info(f"Finding optimal routing for {direction} {quantity} {symbol}")
            
            # Get order books from all available exchanges
            order_books = []
            for exchange in self.exchanges.keys():
                try:
                    ob = await self.get_order_book(exchange, symbol)
                    order_books.append(ob)
                except Exception as e:
                    logger.warning(f"Failed to get order book from {exchange}: {e}")
                    continue
            
            if not order_books:
                raise Exception("No order books available for routing")
            
            # Analyze each exchange
            exchange_scores = {}
            for ob in order_books:
                score = self._calculate_exchange_score(ob, quantity, direction)
                exchange_scores[ob.exchange] = score
            
            # Select best exchange
            best_exchange = max(exchange_scores.items(), key=lambda x: x[1]['total_score'])
            exchange, scores = best_exchange
            
            # Determine order parameters
            order_params = self._determine_order_parameters(
                order_books, exchange, order_type, quantity, direction
            )
            
            decision = RoutingDecision(
                exchange=exchange,
                order_type=order_type,
                quantity=quantity,
                price=order_params.get('price'),
                stop_price=order_params.get('stop_price'),
                reason=scores['reason'],
                confidence=scores['total_score']
            )
            
            logger.info(f"Routing decision: {decision}")
            return decision
            
        except Exception as e:
            logger.error(f"Optimal routing failed: {e}")
            raise

    def _calculate_exchange_score(self, order_book: OrderBookSnapshot, 
                                quantity: float, direction: str) -> Dict:
        """Calculate score for an exchange based on multiple factors"""
        try:
            liquidity_metrics = self.calculate_liquidity_metrics(order_book)
            
            # Liquidity score (0-100)
            liquidity_score = min(100, liquidity_metrics.get('total_volume', 0) / 1000)
            
            # Spread score (0-100) - lower spread is better
            spread_pct = liquidity_metrics.get('spread_percentage', 0)
            spread_score = max(0, 100 - (spread_pct * 10000))  # Convert to basis points
            
            # Market impact score (0-100)
            market_impact = liquidity_metrics.get('market_impact', 0)
            impact_score = max(0, 100 - (market_impact * 10000))
            
            # Latency score (placeholder)
            latency_score = 95  # Would be calculated from actual latency measurements
            
            # Total weighted score
            weights = {
                'liquidity': 0.35,
                'spread': 0.30,
                'impact': 0.25,
                'latency': 0.10
            }
            
            total_score = (
                liquidity_score * weights['liquidity'] +
                spread_score * weights['spread'] +
                impact_score * weights['impact'] +
                latency_score * weights['latency']
            )
            
            reason_parts = []
            if liquidity_score > 80:
                reason_parts.append("high liquidity")
            if spread_score > 90:
                reason_parts.append("tight spread")
            if impact_score > 85:
                reason_parts.append("low market impact")
                
            reason = f"Best execution: {', '.join(reason_parts)}" if reason_parts else "Adequate execution"
            
            return {
                'total_score': total_score,
                'liquidity_score': liquidity_score,
                'spread_score': spread_score,
                'impact_score': impact_score,
                'latency_score': latency_score,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Exchange score calculation failed: {e}")
            return {'total_score': 0, 'reason': f'Error: {str(e)}'}

    def _determine_order_parameters(self, order_books: List[OrderBookSnapshot],
                                  exchange: Exchange, order_type: OrderType,
                                  quantity: float, direction: str) -> Dict:
        """Determine optimal order parameters"""
        try:
            order_book = next((ob for ob in order_books if ob.exchange == exchange), None)
            if not order_book:
                raise ValueError(f"No order book found for {exchange}")
            
            params = {}
            
            if order_type == OrderType.MARKET:
                # For market orders, use current best price
                if direction == 'buy':
                    params['price'] = order_book.asks[0][0] if order_book.asks else order_book.mid_price
                else:  # sell
                    params['price'] = order_book.bids[0][0] if order_book.bids else order_book.mid_price
                    
            elif order_type == OrderType.LIMIT:
                # For limit orders, set price with slight improvement
                if direction == 'buy':
                    best_bid = order_book.bids[0][0] if order_book.bids else order_book.mid_price
                    params['price'] = best_bid * 0.9999  # Slightly below best bid
                else:  # sell
                    best_ask = order_book.asks[0][0] if order_book.asks else order_book.mid_price
                    params['price'] = best_ask * 1.0001  # Slightly above best ask
                    
            elif order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                # For stop orders, calculate stop price based on volatility
                volatility = self._estimate_volatility(order_book)
                if direction == 'buy':
                    params['stop_price'] = order_book.mid_price * (1 + volatility * 0.5)
                else:  # sell
                    params['stop_price'] = order_book.mid_price * (1 - volatility * 0.5)
                    
                if order_type == OrderType.STOP_LIMIT:
                    # Add limit price for stop-limit orders
                    limit_offset = volatility * 0.1
                    if direction == 'buy':
                        params['price'] = params['stop_price'] * (1 + limit_offset)
                    else:
                        params['price'] = params['stop_price'] * (1 - limit_offset)
            
            return params
            
        except Exception as e:
            logger.error(f"Order parameters determination failed: {e}")
            return {}

    def _estimate_volatility(self, order_book: OrderBookSnapshot) -> float:
        """Estimate current market volatility from order book"""
        try:
            # Simple volatility estimation based on spread and depth
            spread_ratio = order_book.spread / order_book.mid_price if order_book.mid_price else 0
            
            # Calculate depth imbalance
            bid_depth = sum(qty for _, qty in order_book.bids[:5])
            ask_depth = sum(qty for _, qty in order_book.asks[:5])
            depth_imbalance = abs(bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
            
            # Combined volatility estimate
            volatility = spread_ratio + (depth_imbalance * 0.5)
            
            return min(volatility, 0.01)  # Cap at 1%
            
        except Exception as e:
            logger.error(f"Volatility estimation failed: {e}")
            return 0.001  # Default 0.1% volatility

    async def execute_routed_order(self, routing_decision: RoutingDecision) -> Dict:
        """Execute order based on routing decision"""
        try:
            start_time = time.time()
            
            exchange = routing_decision.exchange
            symbol = "EURUSDT"  # Would be from routing decision in real implementation
            
            # Execute on selected exchange
            if exchange == Exchange.BINANCE:
                result = await self._execute_binance_order(routing_decision, symbol)
            elif exchange == Exchange.EXNESS:
                result = await self._execute_exness_order(routing_decision, symbol)
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
            
            # Update execution statistics
            self._update_execution_stats(result, time.time() - start_time)
            
            logger.info(f"Order executed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            raise

    async def _execute_binance_order(self, routing: RoutingDecision, symbol: str) -> Dict:
        """Execute order on Binance"""
        try:
            client = self.exchanges[Exchange.BINANCE]
            
            order_params = {
                'symbol': symbol,
                'side': 'BUY' if routing.quantity > 0 else 'SELL',
                'quantity': abs(routing.quantity),
                'type': routing.order_type.value.upper()
            }
            
            if routing.price:
                order_params['price'] = str(round(routing.price, 6))
                
            if routing.stop_price:
                order_params['stopPrice'] = str(round(routing.stop_price, 6))
            
            # Execute order
            result = client.create_order(**order_params)
            
            return {
                'exchange': 'binance',
                'order_id': result['orderId'],
                'status': result['status'],
                'executed_quantity': float(result.get('executedQty', 0)),
                'fills': result.get('fills', [])
            }
            
        except Exception as e:
            logger.error(f"Binance order execution failed: {e}")
            raise

    async def _execute_exness_order(self, routing: RoutingDecision, symbol: str) -> Dict:
        """Execute order on Exness (placeholder implementation)"""
        try:
            # This would be replaced with actual Exness API calls
            # Mock implementation for demonstration
            await asyncio.sleep(0.1)  # Simulate API call
            
            return {
                'exchange': 'exness',
                'order_id': f"EXNESS_{int(time.time() * 1000)}",
                'status': 'FILLED',
                'executed_quantity': abs(routing.quantity),
                'fills': [{'price': routing.price or 1.0850, 'qty': abs(routing.quantity)}]
            }
            
        except Exception as e:
            logger.error(f"Exness order execution failed: {e}")
            raise

    def _update_execution_stats(self, result: Dict, execution_time: float):
        """Update execution statistics"""
        self.execution_stats['total_orders'] += 1
        
        if result.get('status') in ['FILLED', 'PARTIALLY_FILLED']:
            self.execution_stats['successful_executions'] += 1
            
        # Update latency metrics
        exchange = result.get('exchange', 'unknown')
        if exchange not in self.execution_stats['latency_metrics']:
            self.execution_stats['latency_metrics'][exchange] = []
        
        self.execution_stats['latency_metrics'][exchange].append(execution_time)

    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        total_orders = self.execution_stats['total_orders']
        successful = self.execution_stats['successful_executions']
        
        return {
            'total_orders_routed': total_orders,
            'success_rate': successful / total_orders if total_orders > 0 else 0,
            'average_execution_time': self._calculate_average_latency(),
            'exchange_performance': self.execution_stats['latency_metrics'],
            'current_config': {
                'min_liquidity_threshold': self.min_liquidity_threshold,
                'max_spread_ratio': self.max_spread_ratio,
                'slippage_tolerance': self.slippage_tolerance
            }
        }

    def _calculate_average_latency(self) -> float:
        """Calculate average execution latency"""
        all_latencies = []
        for exchange_latencies in self.execution_stats['latency_metrics'].values():
            all_latencies.extend(exchange_latencies)
        
        return sum(all_latencies) / len(all_latencies) if all_latencies else 0

# Example usage and testing
async def main():
    """Test the Smart Order Router"""
    config = {
        'binance_enabled': True,
        'exness_enabled': True,
        'min_liquidity': 10000,
        'max_spread_ratio': 0.0002,
        'slippage_tolerance': 0.0005,
        'binance_api_key': 'your_api_key',
        'binance_secret_key': 'your_secret_key'
    }
    
    router = SmartOrderRouter(config)
    
    try:
        await router.initialize_exchanges()
        
        # Test order routing
        routing_decision = await router.find_optimal_routing(
            symbol="EURUSDT",
            order_type=OrderType.MARKET,
            quantity=1000,
            direction="buy"
        )
        
        print(f"Optimal routing: {routing_decision}")
        
        # Test order execution
        # result = await router.execute_routed_order(routing_decision)
        # print(f"Execution result: {result}")
        
        # Performance report
        report = router.get_performance_report()
        print(f"Performance: {report}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
"""
Trade Executor for FOREX TRADING BOT
Advanced trade execution with risk management and order management
"""

import logging
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_DOWN
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"

@dataclass
class TradeOrder:
    """Trade order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: float = None
    filled_quantity: float = 0
    average_fill_price: float = 0
    exchange: str = ""
    parent_order_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExecutionResult:
    """Trade execution result"""
    success: bool
    order: TradeOrder
    fills: List[Dict] = None
    error_message: str = ""
    execution_time: float = 0
    exchange_order_id: str = ""

class TradeExecutor:
    """
    Advanced trade executor with comprehensive order management
    Handles order execution, tracking, and risk controls
    """
    
    def __init__(self, config: Dict, risk_manager, api_manager):
        self.config = config
        self.risk_manager = risk_manager
        self.api_manager = api_manager
        
        # Order management
        self.active_orders: Dict[str, TradeOrder] = {}
        self.order_history: List[TradeOrder] = []
        self.position_tracker = {}
        
        # Execution parameters
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.execution_timeout = config.get('execution_timeout', 30.0)
        
        # Performance tracking
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'average_execution_time': 0.0,
            'exchange_metrics': {}
        }
        
        # Order lifecycle management
        self.order_callbacks = {
            'on_fill': [],
            'on_cancel': [],
            'on_reject': []
        }
        
        logger.info("TradeExecutor initialized")

    def register_callback(self, event_type: str, callback):
        """Register callback for order events"""
        if event_type in self.order_callbacks:
            self.order_callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    async def execute_trade(self, symbol: str, side: OrderSide, order_type: OrderType,
                          quantity: float, price: Optional[float] = None,
                          stop_price: Optional[float] = None, 
                          take_profit: Optional[float] = None,
                          stop_loss: Optional[float] = None,
                          exchange: str = "binance",
                          metadata: Dict = None) -> ExecutionResult:
        """
        Execute a trade with comprehensive order management
        """
        start_time = time.time()
        
        try:
            # Generate unique order ID
            order_id = f"{exchange}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            
            # Create trade order
            order = TradeOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                exchange=exchange,
                metadata=metadata or {}
            )
            
            logger.info(f"Executing trade: {order}")
            
            # Pre-execution risk check
            risk_check = await self.risk_manager.pre_trade_check(order)
            if not risk_check['allowed']:
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message=f"Risk check failed: {risk_check['reason']}",
                    execution_time=time.time() - start_time
                )
            
            # Execute based on order type
            if order_type == OrderType.MARKET:
                result = await self._execute_market_order(order)
            elif order_type == OrderType.LIMIT:
                result = await self._execute_limit_order(order)
            elif order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                result = await self._execute_stop_order(order)
            else:
                result = ExecutionResult(
                    success=False,
                    order=order,
                    error_message=f"Unsupported order type: {order_type}"
                )
            
            # Update execution metrics
            self._update_execution_metrics(result, time.time() - start_time)
            
            # Handle order lifecycle
            await self._handle_order_lifecycle(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                order=order,
                error_message=str(e),
                execution_time=execution_time
            )

    async def _execute_market_order(self, order: TradeOrder) -> ExecutionResult:
        """Execute market order"""
        try:
            logger.info(f"Executing market order: {order.order_id}")
            
            # Prepare order parameters
            order_params = {
                'symbol': order.symbol,
                'side': order.side.value.upper(),
                'type': 'MARKET',
                'quantity': self._format_quantity(order.quantity, order.symbol)
            }
            
            # Execute order with retry logic
            result = await self._execute_with_retry(order.exchange, order_params)
            
            if result['success']:
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_fill_price = result.get('average_price', 0)
                
                # Create OCO orders for TP/SL if specified
                if order.take_profit or order.stop_loss:
                    await self._create_oco_orders(order, result)
                
                return ExecutionResult(
                    success=True,
                    order=order,
                    fills=result.get('fills', []),
                    execution_time=result.get('execution_time', 0),
                    exchange_order_id=result.get('exchange_order_id', '')
                )
            else:
                order.status = OrderStatus.REJECTED
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            raise

    async def _execute_limit_order(self, order: TradeOrder) -> ExecutionResult:
        """Execute limit order"""
        try:
            if not order.price:
                raise ValueError("Limit order requires price")
            
            logger.info(f"Executing limit order: {order.order_id} at {order.price}")
            
            order_params = {
                'symbol': order.symbol,
                'side': order.side.value.upper(),
                'type': 'LIMIT',
                'quantity': self._format_quantity(order.quantity, order.symbol),
                'price': self._format_price(order.price, order.symbol),
                'timeInForce': 'GTC'  # Good Till Cancelled
            }
            
            result = await self._execute_with_retry(order.exchange, order_params)
            
            if result['success']:
                order.status = OrderStatus.OPEN
                self.active_orders[order.order_id] = order
                
                return ExecutionResult(
                    success=True,
                    order=order,
                    execution_time=result.get('execution_time', 0),
                    exchange_order_id=result.get('exchange_order_id', '')
                )
            else:
                order.status = OrderStatus.REJECTED
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            raise

    async def _execute_stop_order(self, order: TradeOrder) -> ExecutionResult:
        """Execute stop order (stop loss or stop limit)"""
        try:
            if not order.stop_price:
                raise ValueError("Stop order requires stop_price")
            
            logger.info(f"Executing stop order: {order.order_id} at {order.stop_price}")
            
            if order.order_type == OrderType.STOP:
                order_type = 'STOP_LOSS'
            else:  # STOP_LIMIT
                order_type = 'STOP_LOSS_LIMIT'
                if not order.price:
                    raise ValueError("Stop limit order requires price")
            
            order_params = {
                'symbol': order.symbol,
                'side': order.side.value.upper(),
                'type': order_type,
                'quantity': self._format_quantity(order.quantity, order.symbol),
                'stopPrice': self._format_price(order.stop_price, order.symbol),
                'timeInForce': 'GTC'
            }
            
            if order.order_type == OrderType.STOP_LIMIT:
                order_params['price'] = self._format_price(order.price, order.symbol)
            
            result = await self._execute_with_retry(order.exchange, order_params)
            
            if result['success']:
                order.status = OrderStatus.OPEN
                self.active_orders[order.order_id] = order
                
                return ExecutionResult(
                    success=True,
                    order=order,
                    execution_time=result.get('execution_time', 0),
                    exchange_order_id=result.get('exchange_order_id', '')
                )
            else:
                order.status = OrderStatus.REJECTED
                return ExecutionResult(
                    success=False,
                    order=order,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            logger.error(f"Stop order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            raise

    async def _execute_with_retry(self, exchange: str, order_params: Dict, retry_count: int = 0) -> Dict:
        """Execute order with retry logic"""
        try:
            start_time = time.time()
            
            # Get API client for exchange
            api_client = self.api_manager.get_client(exchange)
            if not api_client:
                raise ValueError(f"No API client for exchange: {exchange}")
            
            # Execute order
            result = await api_client.create_order(**order_params)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'exchange_order_id': result.get('orderId', result.get('id', '')),
                'average_price': self._calculate_average_price(result.get('fills', [])),
                'fills': result.get('fills', []),
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.warning(f"Order execution failed (attempt {retry_count + 1}): {e}")
            
            if retry_count < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                return await self._execute_with_retry(exchange, order_params, retry_count + 1)
            else:
                return {
                    'success': False,
                    'error': str(e)
                }

    async def _create_oco_orders(self, parent_order: TradeOrder, execution_result: Dict):
        """Create OCO (One-Cancels-Other) orders for take profit and stop loss"""
        try:
            if not (parent_order.take_profit or parent_order.stop_loss):
                return
            
            # For Binance, create OCO order
            if parent_order.exchange == 'binance':
                await self._create_binance_oco(parent_order, execution_result)
            else:
                # For other exchanges, create separate orders
                await self._create_separate_orders(parent_order, execution_result)
                
        except Exception as e:
            logger.error(f"OCO order creation failed: {e}")

    async def _create_binance_oco(self, parent_order: TradeOrder, execution_result: Dict):
        """Create Binance OCO order"""
        try:
            side = 'SELL' if parent_order.side == OrderSide.BUY else 'BUY'
            
            order_params = {
                'symbol': parent_order.symbol,
                'side': side,
                'quantity': self._format_quantity(parent_order.quantity, parent_order.symbol),
                'price': self._format_price(parent_order.take_profit, parent_order.symbol),
                'stopPrice': self._format_price(parent_order.stop_loss, parent_order.symbol),
                'stopLimitPrice': self._format_price(parent_order.stop_loss, parent_order.symbol)
            }
            
            api_client = self.api_manager.get_client('binance')
            result = await api_client.create_oco_order(**order_params)
            
            logger.info(f"OCO order created: {result}")
            
        except Exception as e:
            logger.error(f"Binance OCO creation failed: {e}")

    async def _create_separate_orders(self, parent_order: TradeOrder, execution_result: Dict):
        """Create separate take profit and stop loss orders"""
        try:
            # Create take profit order
            if parent_order.take_profit:
                tp_order = TradeOrder(
                    order_id=f"{parent_order.order_id}_TP",
                    symbol=parent_order.symbol,
                    side=OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=parent_order.quantity,
                    price=parent_order.take_profit,
                    parent_order_id=parent_order.order_id,
                    exchange=parent_order.exchange
                )
                
                await self._execute_limit_order(tp_order)
            
            # Create stop loss order
            if parent_order.stop_loss:
                sl_order = TradeOrder(
                    order_id=f"{parent_order.order_id}_SL",
                    symbol=parent_order.symbol,
                    side=OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY,
                    order_type=OrderType.STOP,
                    quantity=parent_order.quantity,
                    stop_price=parent_order.stop_loss,
                    parent_order_id=parent_order.order_id,
                    exchange=parent_order.exchange
                )
                
                await self._execute_stop_order(sl_order)
                
        except Exception as e:
            logger.error(f"Separate orders creation failed: {e}")

    async def cancel_order(self, order_id: str, exchange: str = "binance") -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order not found or not active: {order_id}")
                return False
            
            order = self.active_orders[order_id]
            api_client = self.api_manager.get_client(exchange)
            
            if not api_client:
                logger.error(f"No API client for exchange: {exchange}")
                return False
            
            # Cancel order on exchange
            result = await api_client.cancel_order(
                symbol=order.symbol,
                orderId=order.exchange_order_id
            )
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            self._remove_active_order(order_id)
            
            # Trigger callbacks
            await self._trigger_callbacks('on_cancel', order)
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    async def cancel_all_orders(self, symbol: str = None, exchange: str = "binance") -> int:
        """Cancel all active orders"""
        try:
            cancelled_count = 0
            orders_to_cancel = list(self.active_orders.values())
            
            if symbol:
                orders_to_cancel = [o for o in orders_to_cancel if o.symbol == symbol]
            
            for order in orders_to_cancel:
                if await self.cancel_order(order.order_id, exchange):
                    cancelled_count += 1
            
            logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Cancel all orders failed: {e}")
            return 0

    async def get_order_status(self, order_id: str, exchange: str = "binance") -> Optional[TradeOrder]:
        """Get current order status from exchange"""
        try:
            if order_id not in self.active_orders:
                return None
            
            order = self.active_orders[order_id]
            api_client = self.api_manager.get_client(exchange)
            
            if not api_client:
                return None
            
            # Fetch order status from exchange
            exchange_order = await api_client.get_order(
                symbol=order.symbol,
                orderId=order.exchange_order_id
            )
            
            # Update order status
            self._update_order_status(order, exchange_order)
            
            return order
            
        except Exception as e:
            logger.error(f"Order status check failed: {e}")
            return None

    def _update_order_status(self, order: TradeOrder, exchange_data: Dict):
        """Update order status based on exchange data"""
        try:
            status_map = {
                'NEW': OrderStatus.OPEN,
                'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                'FILLED': OrderStatus.FILLED,
                'CANCELED': OrderStatus.CANCELLED,
                'REJECTED': OrderStatus.REJECTED,
                'EXPIRED': OrderStatus.EXPIRED
            }
            
            new_status = status_map.get(exchange_data.get('status'), OrderStatus.OPEN)
            
            if new_status != order.status:
                order.status = new_status
                order.filled_quantity = float(exchange_data.get('executedQty', 0))
                
                # If order is filled or cancelled, remove from active orders
                if new_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    self._remove_active_order(order.order_id)
                
                logger.info(f"Order status updated: {order.order_id} -> {new_status}")
                
        except Exception as e:
            logger.error(f"Order status update failed: {e}")

    def _remove_active_order(self, order_id: str):
        """Remove order from active orders and add to history"""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            self.order_history.append(order)

    async def _handle_order_lifecycle(self, result: ExecutionResult):
        """Handle order lifecycle events"""
        try:
            if result.success:
                if result.order.status == OrderStatus.FILLED:
                    await self._trigger_callbacks('on_fill', result.order)
                    
                    # Update position tracker
                    self._update_position_tracker(result.order)
                    
            else:
                await self._trigger_callbacks('on_reject', result.order)
                
        except Exception as e:
            logger.error(f"Order lifecycle handling failed: {e}")

    async def _trigger_callbacks(self, event_type: str, order: TradeOrder):
        """Trigger registered callbacks for order events"""
        for callback in self.order_callbacks.get(event_type, []):
            try:
                await callback(order)
            except Exception as e:
                logger.error(f"Callback execution failed: {e}")

    def _update_position_tracker(self, order: TradeOrder):
        """Update position tracker with filled order"""
        try:
            position_key = f"{order.symbol}_{order.exchange}"
            
            if position_key not in self.position_tracker:
                self.position_tracker[position_key] = {
                    'symbol': order.symbol,
                    'exchange': order.exchange,
                    'quantity': 0,
                    'average_price': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0
                }
            
            position = self.position_tracker[position_key]
            
            if order.side == OrderSide.BUY:
                new_quantity = position['quantity'] + order.filled_quantity
                if new_quantity != 0:
                    position['average_price'] = (
                        (position['average_price'] * position['quantity']) +
                        (order.average_fill_price * order.filled_quantity)
                    ) / new_quantity
                position['quantity'] = new_quantity
                
            else:  # SELL
                position['quantity'] -= order.filled_quantity
                
            logger.debug(f"Position updated: {position_key} -> {position['quantity']}")
            
        except Exception as e:
            logger.error(f"Position tracker update failed: {e}")

    def _update_execution_metrics(self, result: ExecutionResult, execution_time: float):
        """Update execution performance metrics"""
        self.execution_metrics['total_orders'] += 1
        
        if result.success:
            self.execution_metrics['successful_orders'] += 1
            self.execution_metrics['total_volume'] += abs(result.order.quantity)
        else:
            self.execution_metrics['failed_orders'] += 1
        
        # Update exchange-specific metrics
        exchange = result.order.exchange
        if exchange not in self.execution_metrics['exchange_metrics']:
            self.execution_metrics['exchange_metrics'][exchange] = {
                'total_orders': 0,
                'successful_orders': 0,
                'total_volume': 0
            }
        
        exchange_metrics = self.execution_metrics['exchange_metrics'][exchange]
        exchange_metrics['total_orders'] += 1
        if result.success:
            exchange_metrics['successful_orders'] += 1
            exchange_metrics['total_volume'] += abs(result.order.quantity)

    def _format_quantity(self, quantity: float, symbol: str) -> float:
        """Format quantity according to exchange rules"""
        # This would implement exchange-specific lot size rules
        # For now, return rounded quantity
        return round(quantity, 2)

    def _format_price(self, price: float, symbol: str) -> float:
        """Format price according to exchange rules"""
        # This would implement exchange-specific price precision
        # For now, return rounded price
        return round(price, 5)

    def _calculate_average_price(self, fills: List[Dict]) -> float:
        """Calculate average fill price from multiple fills"""
        if not fills:
            return 0.0
        
        total_quantity = 0
        total_value = 0
        
        for fill in fills:
            qty = float(fill.get('qty', 0))
            price = float(fill.get('price', 0))
            total_quantity += qty
            total_value += qty * price
        
        return total_value / total_quantity if total_quantity > 0 else 0.0

    def get_active_orders(self, symbol: str = None) -> List[TradeOrder]:
        """Get list of active orders"""
        if symbol:
            return [order for order in self.active_orders.values() if order.symbol == symbol]
        return list(self.active_orders.values())

    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[TradeOrder]:
        """Get order history with optional filtering"""
        history = self.order_history[-limit:] if limit else self.order_history
        
        if symbol:
            history = [order for order in history if order.symbol == symbol]
        
        return history

    def get_execution_metrics(self) -> Dict:
        """Get execution performance metrics"""
        total_orders = self.execution_metrics['total_orders']
        successful = self.execution_metrics['successful_orders']
        
        metrics = self.execution_metrics.copy()
        metrics['success_rate'] = successful / total_orders if total_orders > 0 else 0
        metrics['active_orders_count'] = len(self.active_orders)
        metrics['total_positions'] = len(self.position_tracker)
        
        return metrics

    async def monitor_active_orders(self):
        """Background task to monitor active orders"""
        while True:
            try:
                active_orders = self.get_active_orders()
                
                for order in active_orders:
                    await self.get_order_status(order.order_id, order.exchange)
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Active orders monitoring failed: {e}")
                await asyncio.sleep(30)  # Longer sleep on error

# Example usage and testing
async def main():
    """Test the Trade Executor"""
    
    # Mock dependencies
    class MockRiskManager:
        async def pre_trade_check(self, order):
            return {'allowed': True, 'reason': 'OK'}
    
    class MockAPIManager:
        def get_client(self, exchange):
            return None  # Would return actual API client
    
    config = {
        'max_retries': 3,
        'retry_delay': 1.0,
        'execution_timeout': 30.0
    }
    
    risk_manager = MockRiskManager()
    api_manager = MockAPIManager()
    
    executor = TradeExecutor(config, risk_manager, api_manager)
    
    try:
        # Test market order execution
        result = await executor.execute_trade(
            symbol="EURUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
            take_profit=1.1000,
            stop_loss=1.0800,
            metadata={'strategy': 'momentum', 'confidence': 0.85}
        )
        
        print(f"Execution result: {result.success}")
        print(f"Order status: {result.order.status}")
        print(f"Error: {result.error_message}")
        
        # Get metrics
        metrics = executor.get_execution_metrics()
        print(f"Execution metrics: {metrics}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
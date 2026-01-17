"""
Position Manager for FOREX TRADING BOT
Advanced position tracking, risk management, and portfolio optimization
"""

import logging
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_DOWN
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    HEDGED = "hedged"
    PARTIALLY_CLOSED = "partially_closed"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class Position:
    """Position data structure with comprehensive tracking"""
    position_id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    entry_time: float = field(default_factory=time.time)
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_stop_activation: Optional[float] = None
    exchange: str = ""
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Risk metrics
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    peak_value: float = 0.0
    valley_value: float = 0.0
    
    def __post_init__(self):
        self.update_unrealized_pnl()
        self.peak_value = self.unrealized_pnl
        self.valley_value = self.unrealized_pnl

    def update_unrealized_pnl(self, current_price: Optional[float] = None):
        """Update unrealized P&L based on current price"""
        if current_price is not None:
            self.current_price = current_price
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
        
        # Update MAE/MFE
        self.max_favorable_excursion = max(self.max_favorable_excursion, self.unrealized_pnl)
        self.max_adverse_excursion = min(self.max_adverse_excursion, self.unrealized_pnl)
        self.peak_value = max(self.peak_value, self.unrealized_pnl)
        self.valley_value = min(self.valley_value, self.unrealized_pnl)

    def close_position(self, exit_price: float, exit_time: Optional[float] = None):
        """Close position and calculate realized P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time or time.time()
        self.status = PositionStatus.CLOSED
        
        # Calculate final P&L
        if self.side == PositionSide.LONG:
            self.realized_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.realized_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        self.unrealized_pnl = 0.0
        self.update_unrealized_pnl(self.exit_price)

    def partial_close(self, close_quantity: float, exit_price: float):
        """Partially close position"""
        if close_quantity >= self.quantity:
            self.close_position(exit_price)
            return
        
        # Calculate P&L for closed portion
        if self.side == PositionSide.LONG:
            pnl = (exit_price - self.entry_price) * close_quantity
        else:  # SHORT
            pnl = (self.entry_price - exit_price) * close_quantity
        
        self.realized_pnl += pnl
        self.quantity -= close_quantity
        
        if self.quantity <= 0:
            self.status = PositionStatus.CLOSED
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED

@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio performance metrics"""
    total_value: float = 0.0
    cash_balance: float = 0.0
    margin_used: float = 0.0
    available_margin: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    
    # Position metrics
    total_positions: int = 0
    open_positions: int = 0
    winning_positions: int = 0
    losing_positions: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Exposure metrics
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    concentration_risk: float = 0.0

class PositionManager:
    """
    Advanced position management with risk controls and portfolio optimization
    """
    
    def __init__(self, config: Dict, risk_manager, data_handler):
        self.config = config
        self.risk_manager = risk_manager
        self.data_handler = data_handler
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        self.portfolio_snapshots = deque(maxlen=1000)  # Store portfolio snapshots
        
        # Portfolio state
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.cash_balance = self.initial_capital
        self.leverage = config.get('leverage', 10.0)
        self.margin_requirement = config.get('margin_requirement', 0.01)  # 1% margin
        
        # Risk limits
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.max_daily_loss = config.get('max_daily_loss', 0.02)  # 2% daily loss limit
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.05)  # 5% total risk
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.daily_realized = 0.0
        self.last_reset_time = time.time()
        
        # Correlation matrix (would be populated from data)
        self.correlation_matrix = {}
        
        # Event callbacks
        self.position_callbacks = {
            'on_position_open': [],
            'on_position_close': [],
            'on_position_update': [],
            'on_risk_breach': []
        }
        
        logger.info("PositionManager initialized")

    def register_callback(self, event_type: str, callback):
        """Register callback for position events"""
        if event_type in self.position_callbacks:
            self.position_callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    async def open_position(self, symbol: str, side: PositionSide, quantity: float,
                          entry_price: float, stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          trailing_stop: Optional[float] = None,
                          exchange: str = "binance",
                          strategy: str = "",
                          metadata: Dict = None) -> Optional[Position]:
        """
        Open a new position with comprehensive risk checks
        """
        try:
            # Generate position ID
            position_id = f"{symbol}_{side.value}_{int(time.time() * 1000)}"
            
            # Create position object
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                exchange=exchange,
                strategy=strategy,
                metadata=metadata or {}
            )
            
            logger.info(f"Opening position: {position}")
            
            # Pre-trade risk checks
            risk_check = await self._pre_trade_risk_check(position)
            if not risk_check['allowed']:
                logger.warning(f"Position rejected - Risk check failed: {risk_check['reason']}")
                return None
            
            # Calculate required margin
            required_margin = self._calculate_required_margin(position)
            if required_margin > self.cash_balance:
                logger.warning(f"Insufficient margin: {required_margin} > {self.cash_balance}")
                return None
            
            # Deduct margin
            self.cash_balance -= required_margin
            
            # Store position
            self.positions[position_id] = position
            
            # Trigger callbacks
            await self._trigger_callbacks('on_position_open', position)
            
            logger.info(f"Position opened successfully: {position_id}")
            return position
            
        except Exception as e:
            logger.error(f"Position opening failed: {e}")
            return None

    async def _pre_trade_risk_check(self, position: Position) -> Dict:
        """Comprehensive pre-trade risk assessment"""
        try:
            checks = []
            
            # 1. Position size check
            position_value = position.quantity * position.entry_price
            portfolio_value = await self.get_portfolio_value()
            position_size_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_size_ratio > self.max_position_size:
                checks.append(f"Position size {position_size_ratio:.2%} exceeds limit {self.max_position_size:.2%}")
            
            # 2. Concentration risk
            concentration = await self._calculate_concentration_risk(position.symbol)
            if concentration > 0.3:  # 30% concentration limit
                checks.append(f"Concentration risk: {concentration:.2%}")
            
            # 3. Correlation risk
            correlation_risk = await self._calculate_correlation_risk(position)
            if correlation_risk > 0.8:  # High correlation with existing positions
                checks.append(f"High correlation risk: {correlation_risk:.2%}")
            
            # 4. Daily loss limit
            if self.daily_pnl < -self.max_daily_loss * self.initial_capital:
                checks.append("Daily loss limit breached")
            
            # 5. Portfolio risk limit
            portfolio_risk = await self._calculate_portfolio_risk(position)
            if portfolio_risk > self.max_portfolio_risk:
                checks.append(f"Portfolio risk {portfolio_risk:.2%} exceeds limit")
            
            if checks:
                return {
                    'allowed': False,
                    'reason': '; '.join(checks)
                }
            else:
                return {
                    'allowed': True,
                    'reason': 'All risk checks passed'
                }
                
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return {'allowed': False, 'reason': f'Risk check error: {str(e)}'}

    def _calculate_required_margin(self, position: Position) -> float:
        """Calculate required margin for position"""
        position_value = position.quantity * position.entry_price
        return position_value * self.margin_requirement / self.leverage

    async def close_position(self, position_id: str, exit_price: float,
                           exit_time: Optional[float] = None) -> bool:
        """Close position and calculate P&L"""
        try:
            if position_id not in self.positions:
                logger.warning(f"Position not found: {position_id}")
                return False
            
            position = self.positions[position_id]
            
            # Close position
            position.close_position(exit_price, exit_time)
            
            # Update cash balance (return margin + realized P&L)
            required_margin = self._calculate_required_margin(position)
            self.cash_balance += required_margin + position.realized_pnl
            
            # Update daily P&L
            self.daily_realized += position.realized_pnl
            self.daily_pnl += position.realized_pnl
            
            # Move to history
            self.position_history.append(position)
            del self.positions[position_id]
            
            # Trigger callbacks
            await self._trigger_callbacks('on_position_close', position)
            
            logger.info(f"Position closed: {position_id} P&L: {position.realized_pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Position closing failed: {e}")
            return False

    async def partial_close_position(self, position_id: str, close_quantity: float,
                                   exit_price: float) -> bool:
        """Partially close position"""
        try:
            if position_id not in self.positions:
                return False
            
            position = self.positions[position_id]
            
            # Calculate margin to return
            original_margin = self._calculate_required_margin(position)
            close_ratio = close_quantity / position.quantity
            returned_margin = original_margin * close_ratio
            
            # Partially close
            position.partial_close(close_quantity, exit_price)
            
            # Update cash balance
            self.cash_balance += returned_margin
            
            # If fully closed, remove from active positions
            if position.status == PositionStatus.CLOSED:
                self.position_history.append(position)
                del self.positions[position_id]
                await self._trigger_callbacks('on_position_close', position)
            else:
                await self._trigger_callbacks('on_position_update', position)
            
            logger.info(f"Position partially closed: {position_id} Quantity: {close_quantity}")
            return True
            
        except Exception as e:
            logger.error(f"Partial close failed: {e}")
            return False

    async def update_position_prices(self, price_updates: Dict[str, float]):
        """Update position prices and check for stop losses/take profits"""
        try:
            positions_to_close = []
            
            for position_id, position in self.positions.items():
                if position.symbol in price_updates:
                    new_price = price_updates[position.symbol]
                    old_price = position.current_price
                    
                    # Update position price
                    position.update_unrealized_pnl(new_price)
                    
                    # Check for stop loss/take profit
                    close_reason = await self._check_exit_conditions(position, new_price, old_price)
                    if close_reason:
                        positions_to_close.append((position_id, new_price, close_reason))
                    
                    # Update trailing stop
                    await self._update_trailing_stop(position, new_price)
            
            # Close positions that hit exit conditions
            for position_id, exit_price, reason in positions_to_close:
                await self.close_position(position_id, exit_price)
                logger.info(f"Position auto-closed: {position_id} Reason: {reason}")
            
            # Update portfolio snapshot
            await self._update_portfolio_snapshot()
            
            # Trigger update callbacks
            await self._trigger_callbacks('on_position_update', None)
            
        except Exception as e:
            logger.error(f"Position price update failed: {e}")

    async def _check_exit_conditions(self, position: Position, new_price: float, old_price: float) -> Optional[str]:
        """Check if position should be closed due to stop loss/take profit"""
        try:
            # Check stop loss
            if position.stop_loss:
                if (position.side == PositionSide.LONG and new_price <= position.stop_loss) or \
                   (position.side == PositionSide.SHORT and new_price >= position.stop_loss):
                    return f"Stop loss triggered at {position.stop_loss}"
            
            # Check take profit
            if position.take_profit:
                if (position.side == PositionSide.LONG and new_price >= position.take_profit) or \
                   (position.side == PositionSide.SHORT and new_price <= position.take_profit):
                    return f"Take profit triggered at {position.take_profit}"
            
            # Check trailing stop
            if position.trailing_stop and position.trailing_stop_activation:
                if (position.side == PositionSide.LONG and new_price <= position.trailing_stop) or \
                   (position.side == PositionSide.SHORT and new_price >= position.trailing_stop):
                    return f"Trailing stop triggered at {position.trailing_stop}"
            
            return None
            
        except Exception as e:
            logger.error(f"Exit condition check failed: {e}")
            return None

    async def _update_trailing_stop(self, position: Position, new_price: float):
        """Update trailing stop level"""
        try:
            if not position.trailing_stop:
                return
            
            if position.side == PositionSide.LONG:
                # For long positions, trail below current price
                if new_price > position.entry_price and (position.trailing_stop_activation is None or new_price > position.trailing_stop_activation):
                    position.trailing_stop_activation = new_price
                    position.trailing_stop = new_price * (1 - position.trailing_stop)
                    
            else:  # SHORT
                # For short positions, trail above current price
                if new_price < position.entry_price and (position.trailing_stop_activation is None or new_price < position.trailing_stop_activation):
                    position.trailing_stop_activation = new_price
                    position.trailing_stop = new_price * (1 + position.trailing_stop)
                    
        except Exception as e:
            logger.error(f"Trailing stop update failed: {e}")

    async def _calculate_concentration_risk(self, symbol: str) -> float:
        """Calculate concentration risk for a symbol"""
        try:
            symbol_exposure = 0.0
            total_exposure = 0.0
            
            for position in self.positions.values():
                exposure = position.quantity * position.current_price
                total_exposure += exposure
                if position.symbol == symbol:
                    symbol_exposure += exposure
            
            return symbol_exposure / total_exposure if total_exposure > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return 0.0

    async def _calculate_correlation_risk(self, new_position: Position) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            # This would use historical correlation data
            # For now, return a simple implementation
            correlation_sum = 0.0
            count = 0
            
            for position in self.positions.values():
                if position.symbol != new_position.symbol:
                    # Simple correlation based on same direction
                    if position.side == new_position.side:
                        correlation_sum += 0.6  # Assume moderate correlation
                    else:
                        correlation_sum += 0.2  # Assume low correlation
                    count += 1
            
            return correlation_sum / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return 0.0

    async def _calculate_portfolio_risk(self, new_position: Position) -> float:
        """Calculate overall portfolio risk with new position"""
        try:
            # Simplified risk calculation
            # In production, this would use VaR, CVaR, or other advanced metrics
            portfolio_value = await self.get_portfolio_value()
            new_position_risk = new_position.quantity * new_position.entry_price * 0.01  # 1% risk assumption
            
            existing_risk = 0.0
            for position in self.positions.values():
                existing_risk += position.quantity * position.current_price * 0.01
            
            total_risk = (existing_risk + new_position_risk) / portfolio_value if portfolio_value > 0 else 0
            return total_risk
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return 0.0

    async def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        try:
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            return self.cash_balance + unrealized_pnl
            
        except Exception as e:
            logger.error(f"Portfolio value calculation failed: {e}")
            return self.cash_balance

    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        try:
            metrics = PortfolioMetrics()
            
            # Basic metrics
            metrics.total_value = await self.get_portfolio_value()
            metrics.cash_balance = self.cash_balance
            metrics.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            metrics.total_realized_pnl = sum(pos.realized_pnl for pos in self.position_history[-100:])  # Last 100 trades
            
            # Position metrics
            metrics.total_positions = len(self.position_history)
            metrics.open_positions = len(self.positions)
            
            # Calculate win rate and profit factor
            winning_trades = [p for p in self.position_history if p.realized_pnl > 0]
            losing_trades = [p for p in self.position_history if p.realized_pnl < 0]
            
            metrics.winning_positions = len(winning_trades)
            metrics.losing_positions = len(losing_trades)
            metrics.win_rate = metrics.winning_positions / metrics.total_positions if metrics.total_positions > 0 else 0
            
            total_profit = sum(p.realized_pnl for p in winning_trades)
            total_loss = abs(sum(p.realized_pnl for p in losing_trades))
            metrics.avg_win = total_profit / len(winning_trades) if winning_trades else 0
            metrics.avg_loss = total_loss / len(losing_trades) if losing_trades else 0
            metrics.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Exposure metrics
            long_exposure = sum(p.quantity * p.current_price for p in self.positions.values() if p.side == PositionSide.LONG)
            short_exposure = sum(p.quantity * p.current_price for p in self.positions.values() if p.side == PositionSide.SHORT)
            
            metrics.net_exposure = long_exposure - short_exposure
            metrics.gross_exposure = long_exposure + short_exposure
            
            # Calculate drawdown
            metrics.current_drawdown = await self._calculate_current_drawdown()
            metrics.max_drawdown = await self._calculate_max_drawdown()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return PortfolioMetrics()

    async def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        try:
            if not self.portfolio_snapshots:
                return 0.0
            
            current_value = await self.get_portfolio_value()
            peak_value = max(snapshot['portfolio_value'] for snapshot in self.portfolio_snapshots)
            
            return (peak_value - current_value) / peak_value if peak_value > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Current drawdown calculation failed: {e}")
            return 0.0

    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum historical drawdown"""
        try:
            if len(self.portfolio_snapshots) < 2:
                return 0.0
            
            peak = self.portfolio_snapshots[0]['portfolio_value']
            max_dd = 0.0
            
            for snapshot in self.portfolio_snapshots:
                if snapshot['portfolio_value'] > peak:
                    peak = snapshot['portfolio_value']
                else:
                    dd = (peak - snapshot['portfolio_value']) / peak
                    max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0

    async def _update_portfolio_snapshot(self):
        """Update portfolio snapshot for performance tracking"""
        try:
            snapshot = {
                'timestamp': time.time(),
                'portfolio_value': await self.get_portfolio_value(),
                'cash_balance': self.cash_balance,
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
                'open_positions': len(self.positions)
            }
            
            self.portfolio_snapshots.append(snapshot)
            
        except Exception as e:
            logger.error(f"Portfolio snapshot update failed: {e}")

    async def _trigger_callbacks(self, event_type: str, position: Optional[Position]):
        """Trigger registered callbacks for position events"""
        for callback in self.position_callbacks.get(event_type, []):
            try:
                await callback(position)
            except Exception as e:
                logger.error(f"Position callback execution failed: {e}")

    def get_open_positions(self, symbol: str = None, exchange: str = None) -> List[Position]:
        """Get list of open positions with optional filtering"""
        positions = list(self.positions.values())
        
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        
        if exchange:
            positions = [p for p in positions if p.exchange == exchange]
        
        return positions

    def get_position_history(self, symbol: str = None, limit: int = 100) -> List[Position]:
        """Get position history with optional filtering"""
        history = self.position_history[-limit:] if limit else self.position_history
        
        if symbol:
            history = [p for p in history if p.symbol == symbol]
        
        return history

    async def reset_daily_metrics(self):
        """Reset daily performance metrics"""
        self.daily_pnl = 0.0
        self.daily_realized = 0.0
        self.last_reset_time = time.time()
        logger.info("Daily metrics reset")

    async def run_risk_monitoring(self):
        """Background task for continuous risk monitoring"""
        while True:
            try:
                # Check risk limits
                portfolio_value = await self.get_portfolio_value()
                daily_loss_limit = self.max_daily_loss * self.initial_capital
                
                if self.daily_pnl < -daily_loss_limit:
                    await self._handle_risk_breach("Daily loss limit breached")
                
                # Check portfolio risk
                portfolio_metrics = await self.get_portfolio_metrics()
                if portfolio_metrics.current_drawdown > 0.1:  # 10% drawdown alert
                    await self._handle_risk_breach(f"High drawdown: {portfolio_metrics.current_drawdown:.2%}")
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Risk monitoring failed: {e}")
                await asyncio.sleep(300)  # Longer sleep on error

    async def _handle_risk_breach(self, reason: str):
        """Handle risk limit breaches"""
        try:
            logger.warning(f"Risk breach detected: {reason}")
            
            # Trigger risk breach callbacks
            await self._trigger_callbacks('on_risk_breach', {
                'reason': reason,
                'timestamp': time.time(),
                'portfolio_value': await self.get_portfolio_value(),
                'daily_pnl': self.daily_pnl
            })
            
        except Exception as e:
            logger.error(f"Risk breach handling failed: {e}")

# Example usage and testing
async def main():
    """Test the Position Manager"""
    
    # Mock dependencies
    class MockRiskManager:
        pass
    
    class MockDataHandler:
        pass
    
    config = {
        'initial_capital': 10000.0,
        'leverage': 10.0,
        'margin_requirement': 0.01,
        'max_position_size': 0.1,
        'max_daily_loss': 0.02,
        'max_portfolio_risk': 0.05
    }
    
    risk_manager = MockRiskManager()
    data_handler = MockDataHandler()
    
    position_manager = PositionManager(config, risk_manager, data_handler)
    
    try:
        # Test opening a position
        position = await position_manager.open_position(
            symbol="EURUSDT",
            side=PositionSide.LONG,
            quantity=1000,
            entry_price=1.0850,
            stop_loss=1.0800,
            take_profit=1.0950,
            strategy="trend_following"
        )
        
        if position:
            print(f"Position opened: {position.position_id}")
            
            # Update price
            await position_manager.update_position_prices({'EURUSDT': 1.0900})
            
            # Get portfolio metrics
            metrics = await position_manager.get_portfolio_metrics()
            print(f"Portfolio value: {metrics.total_value:.2f}")
            print(f"Unrealized P&L: {metrics.total_unrealized_pnl:.2f}")
            
            # Close position
            await position_manager.close_position(position.position_id, 1.0920)
            
        else:
            print("Position opening failed")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
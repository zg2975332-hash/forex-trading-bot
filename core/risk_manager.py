"""
Advanced Risk Manager for Forex Trading Bot
Comprehensive risk management with dynamic position sizing, drawdown control, and real-time monitoring
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from scipy import stats
import json
import threading
import time
from collections import deque
import asyncio

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class RiskEvent(Enum):
    """Risk event types"""
    MARGIN_CALL_WARNING = "margin_call_warning"
    DRAWDOWN_LIMIT = "drawdown_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    POSITION_LIMIT = "position_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_RISK = "correlation_risk"
    CONCENTRATION_RISK = "concentration_risk"

@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    symbol: str
    position_type: str
    entry_price: float
    current_price: float
    quantity: float
    notional_value: float
    margin_used: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    time_in_trade: timedelta

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    timestamp: datetime
    total_equity: float
    used_margin: float
    free_margin: float
    margin_level: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    cvar_95: float
    volatility_annual: float
    sharpe_ratio: float
    correlation_risk: float
    concentration_risk: float
    risk_score: float

@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_id: str
    timestamp: datetime
    risk_event: RiskEvent
    severity: str
    message: str
    triggered_metrics: Dict[str, float]
    action_required: bool
    resolved: bool = False

class PositionSizingCalculator:
    """Advanced position sizing calculations"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Position sizing methods weights
        self.sizing_weights = {
            'kelly': 0.4,
            'volatility_adjusted': 0.3,
            'risk_parity': 0.2,
            'confidence_based': 0.1
        }
        
        logger.info("Position Sizing Calculator initialized")
    
    def calculate_kelly_position(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0.02  # Conservative default
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Use fractional Kelly for safety (1/4 Kelly)
        fractional_kelly = kelly_fraction * 0.25
        
        # Apply bounds
        return max(0.01, min(0.1, fractional_kelly))
    
    def calculate_volatility_adjusted_size(self, volatility: float, 
                                         base_risk: float = 0.02) -> float:
        """Calculate position size adjusted for volatility"""
        # Normalize volatility (assuming 15% is average)
        volatility_ratio = volatility / 0.15
        
        # Inverse relationship: higher volatility = smaller position
        adjusted_risk = base_risk / max(0.5, volatility_ratio)
        
        return max(0.005, min(0.1, adjusted_risk))
    
    def calculate_risk_parity_size(self, correlation: float, 
                                 portfolio_volatility: float) -> float:
        """Calculate position size using risk parity principles"""
        # Adjust for correlation (higher correlation = smaller position)
        correlation_penalty = 1.0 - min(0.7, abs(correlation))
        
        # Adjust for portfolio volatility
        volatility_adjustment = 0.15 / max(0.05, portfolio_volatility)
        
        base_size = 0.02 * correlation_penalty * volatility_adjustment
        
        return max(0.005, min(0.08, base_size))
    
    def calculate_confidence_based_size(self, confidence: float, 
                                      base_risk: float = 0.02) -> float:
        """Calculate position size based on model confidence"""
        if confidence < 0.5:
            multiplier = 0.5
        elif confidence < 0.7:
            multiplier = 0.8
        elif confidence < 0.85:
            multiplier = 1.0
        else:
            multiplier = 1.2
        
        adjusted_risk = base_risk * multiplier
        
        return max(0.005, min(0.15, adjusted_risk))
    
    def calculate_dynamic_position_size(self, symbol: str, current_price: float,
                                     win_rate: float, avg_win: float, avg_loss: float,
                                     volatility: float, correlation: float,
                                     portfolio_volatility: float, confidence: float) -> Dict[str, Any]:
        """Calculate dynamic position size using multiple methods"""
        try:
            # Calculate using different methods
            kelly_size = self.calculate_kelly_position(win_rate, avg_win, avg_loss)
            vol_adjusted_size = self.calculate_volatility_adjusted_size(volatility)
            risk_parity_size = self.calculate_risk_parity_size(correlation, portfolio_volatility)
            confidence_size = self.calculate_confidence_based_size(confidence)
            
            # Weighted average
            weighted_size = (
                kelly_size * self.sizing_weights['kelly'] +
                vol_adjusted_size * self.sizing_weights['volatility_adjusted'] +
                risk_parity_size * self.sizing_weights['risk_parity'] +
                confidence_size * self.sizing_weights['confidence_based']
            )
            
            # Apply overall limits
            final_size_pct = max(0.005, min(0.1, weighted_size))
            
            # Calculate lot size
            risk_amount = self.current_capital * final_size_pct
            lot_size = risk_amount / current_price
            
            return {
                'position_size_pct': final_size_pct,
                'lot_size': lot_size,
                'risk_amount': risk_amount,
                'method_breakdown': {
                    'kelly': kelly_size,
                    'volatility_adjusted': vol_adjusted_size,
                    'risk_parity': risk_parity_size,
                    'confidence_based': confidence_size
                },
                'weighted_average': weighted_size
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic position size: {e}")
            # Return conservative default
            return {
                'position_size_pct': 0.01,
                'lot_size': (self.current_capital * 0.01) / current_price,
                'risk_amount': self.current_capital * 0.01,
                'method_breakdown': {},
                'weighted_average': 0.01
            }

class StopLossCalculator:
    """Advanced stop loss calculations"""
    
    def __init__(self):
        self.atr_period = 14
        self.volatility_lookback = 20
        
        # Stop loss methods weights
        self.stop_loss_weights = {
            'atr_based': 0.5,
            'volatility_based': 0.3,
            'support_resistance': 0.2
        }
        
        logger.info("Stop Loss Calculator initialized")
    
    def calculate_atr_stop_loss(self, highs: List[float], lows: List[float], 
                              closes: List[float], position_type: str) -> float:
        """Calculate stop loss using Average True Range"""
        if len(highs) < self.atr_period:
            return 0.02  # Default 2% stop loss
        
        try:
            # Calculate ATR
            true_ranges = []
            for i in range(1, len(highs)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            atr = np.mean(true_ranges[-self.atr_period:])
            current_price = closes[-1]
            
            # ATR-based stop loss (2 ATRs)
            atr_stop_distance = atr * 2.0
            
            if position_type == 'LONG':
                stop_loss_price = current_price - atr_stop_distance
                stop_loss_pct = atr_stop_distance / current_price
            else:  # SHORT
                stop_loss_price = current_price + atr_stop_distance
                stop_loss_pct = atr_stop_distance / current_price
            
            return stop_loss_pct
            
        except Exception as e:
            logger.error(f"Error calculating ATR stop loss: {e}")
            return 0.02
    
    def calculate_volatility_stop_loss(self, prices: List[float], 
                                     position_type: str) -> float:
        """Calculate stop loss based on volatility"""
        if len(prices) < self.volatility_lookback:
            return 0.02
        
        try:
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Volatility-based stop (2 standard deviations)
            vol_stop_distance = volatility * 2.0
            
            return vol_stop_distance
            
        except Exception as e:
            logger.error(f"Error calculating volatility stop loss: {e}")
            return 0.02
    
    def calculate_support_resistance_stop_loss(self, prices: List[float],
                                            position_type: str) -> float:
        """Calculate stop loss based on support/resistance levels"""
        if len(prices) < 20:
            return 0.02
        
        try:
            # Simplified support/resistance detection
            recent_low = min(prices[-20:])
            recent_high = max(prices[-20:])
            current_price = prices[-1]
            
            if position_type == 'LONG':
                # Stop below recent low
                stop_distance = (current_price - recent_low) / current_price
            else:  # SHORT
                # Stop above recent high
                stop_distance = (recent_high - current_price) / current_price
            
            return max(0.01, min(0.05, stop_distance))
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance stop loss: {e}")
            return 0.02
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float,
                                 position_type: str, highs: List[float],
                                 lows: List[float], closes: List[float]) -> Dict[str, Any]:
        """Calculate dynamic stop loss using multiple methods"""
        try:
            # Calculate using different methods
            atr_stop_pct = self.calculate_atr_stop_loss(highs, lows, closes, position_type)
            vol_stop_pct = self.calculate_volatility_stop_loss(closes, position_type)
            sr_stop_pct = self.calculate_support_resistance_stop_loss(closes, position_type)
            
            # Weighted average
            weighted_stop_pct = (
                atr_stop_pct * self.stop_loss_weights['atr_based'] +
                vol_stop_pct * self.stop_loss_weights['volatility_based'] +
                sr_stop_pct * self.stop_loss_weights['support_resistance']
            )
            
            # Apply bounds
            final_stop_pct = max(0.005, min(0.05, weighted_stop_pct))
            
            # Calculate stop loss price
            if position_type == 'LONG':
                stop_loss_price = entry_price * (1 - final_stop_pct)
                take_profit_price = entry_price * (1 + final_stop_pct * 2)  # 1:2 risk-reward
            else:  # SHORT
                stop_loss_price = entry_price * (1 + final_stop_pct)
                take_profit_price = entry_price * (1 - final_stop_pct * 2)
            
            return {
                'stop_loss_pct': final_stop_pct,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'risk_reward_ratio': 2.0,
                'method_breakdown': {
                    'atr_based': atr_stop_pct,
                    'volatility_based': vol_stop_pct,
                    'support_resistance': sr_stop_pct
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {e}")
            # Return conservative default
            default_stop_pct = 0.02
            if position_type == 'LONG':
                stop_loss = entry_price * (1 - default_stop_pct)
                take_profit = entry_price * (1 + default_stop_pct * 2)
            else:
                stop_loss = entry_price * (1 + default_stop_pct)
                take_profit = entry_price * (1 - default_stop_pct * 2)
            
            return {
                'stop_loss_pct': default_stop_pct,
                'stop_loss_price': stop_loss,
                'take_profit_price': take_profit,
                'risk_reward_ratio': 2.0,
                'method_breakdown': {}
            }

class PortfolioRiskAnalyzer:
    """Comprehensive portfolio risk analysis"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_equity = initial_capital
        self.daily_peak = initial_capital
        
        # Risk limits
        self.risk_limits = {
            'max_drawdown': 0.15,           # 15% maximum drawdown
            'daily_loss_limit': 0.05,       # 5% daily loss limit
            'margin_call_level': 0.8,       # 80% margin level for warning
            'stop_out_level': 0.5,          # 50% margin level for stop out
            'max_position_size': 0.2,       # 20% maximum position size
            'max_correlation': 0.7,         # 70% maximum correlation
            'max_open_positions': 5,        # Maximum 5 open positions
            'var_confidence': 0.95          # 95% VaR confidence level
        }
        
        # Performance tracking
        self.equity_history = deque(maxlen=1000)
        self.drawdown_history = deque(maxlen=1000)
        self.risk_alerts: List[RiskAlert] = []
        
        logger.info("Portfolio Risk Analyzer initialized")
    
    def update_portfolio_state(self, positions: Dict[str, PositionRisk], 
                             cash: float, total_equity: float):
        """Update portfolio state and calculate risk metrics"""
        try:
            # Calculate basic metrics
            used_margin = sum(pos.margin_used for pos in positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
            
            free_margin = total_equity - used_margin
            margin_level = (total_equity / used_margin) * 100 if used_margin > 0 else float('inf')
            
            # Update peak equity for drawdown calculation
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity
            
            current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
            
            # Update daily peak
            if total_equity > self.daily_peak:
                self.daily_peak = total_equity
            
            # Calculate advanced risk metrics
            volatility = self._calculate_portfolio_volatility()
            var_95 = self._calculate_value_at_risk(0.95)
            cvar_95 = self._calculate_conditional_var(0.95)
            correlation_risk = self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions)
            
            # Calculate overall risk score (0-100, higher = riskier)
            risk_score = self._calculate_risk_score(
                margin_level, current_drawdown, volatility, 
                correlation_risk, concentration_risk
            )
            
            portfolio_risk = PortfolioRisk(
                timestamp=datetime.now(),
                total_equity=total_equity,
                used_margin=used_margin,
                free_margin=free_margin,
                margin_level=margin_level,
                total_unrealized_pnl=total_unrealized_pnl,
                total_realized_pnl=total_equity - self.initial_capital - total_unrealized_pnl,
                daily_pnl=total_equity - self.daily_peak,
                max_drawdown=abs(min(self.drawdown_history)) if self.drawdown_history else 0.0,
                current_drawdown=current_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                volatility_annual=volatility,
                sharpe_ratio=self._calculate_sharpe_ratio(),
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                risk_score=risk_score
            )
            
            # Update history
            self.equity_history.append(total_equity)
            self.drawdown_history.append(current_drawdown)
            
            # Check risk limits and generate alerts
            self._check_risk_limits(portfolio_risk, positions)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
            return self._create_default_risk_metrics()
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility from equity history"""
        if len(self.equity_history) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.equity_history)):
            ret = (self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
            returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        daily_volatility = np.std(returns)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
    
    def _calculate_value_at_risk(self, confidence: float) -> float:
        """Calculate Value at Risk"""
        if len(self.equity_history) < 20:
            return 0.0
        
        returns = []
        for i in range(1, len(self.equity_history)):
            ret = (self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
            returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(var * self.current_capital)
    
    def _calculate_conditional_var(self, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(self.equity_history) < 20:
            return 0.0
        
        returns = []
        for i in range(1, len(self.equity_history)):
            ret = (self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
            returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = [r for r in returns if r <= var_threshold]
        
        if not tail_returns:
            return abs(var_threshold * self.current_capital)
        
        cvar = np.mean(tail_returns)
        return abs(cvar * self.current_capital)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from equity history"""
        if len(self.equity_history) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.equity_history)):
            ret = (self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
            returns.append(ret)
        
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = np.mean(returns) - risk_free_rate
        sharpe_ratio = excess_returns / np.std(returns) * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_correlation_risk(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate correlation risk in portfolio"""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation risk calculation
        # In production, use actual correlation matrix between positions
        symbols = list(positions.keys())
        
        # Assume some base correlation risk based on number of positions
        # and their notional values
        total_notional = sum(abs(pos.notional_value) for pos in positions.values())
        max_position_notional = max(abs(pos.notional_value) for pos in positions.values())
        
        concentration = max_position_notional / total_notional if total_notional > 0 else 0.0
        correlation_risk = min(1.0, len(symbols) * 0.1 + concentration * 0.5)
        
        return correlation_risk
    
    def _calculate_concentration_risk(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate concentration risk using Herfindahl index"""
        if not positions:
            return 0.0
        
        position_sizes = [abs(pos.notional_value) for pos in positions.values()]
        total_exposure = sum(position_sizes)
        
        if total_exposure == 0:
            return 0.0
        
        # Herfindahl-Hirschman Index for concentration
        herfindahl = sum((size / total_exposure) ** 2 for size in position_sizes)
        
        return herfindahl
    
    def _calculate_risk_score(self, margin_level: float, drawdown: float,
                            volatility: float, correlation_risk: float,
                            concentration_risk: float) -> float:
        """Calculate overall risk score (0-100)"""
        risk_factors = []
        
        # Margin level risk
        if margin_level < 100:
            risk_factors.append(90)
        elif margin_level < 200:
            risk_factors.append(70)
        elif margin_level < 500:
            risk_factors.append(30)
        else:
            risk_factors.append(10)
        
        # Drawdown risk
        drawdown_risk = min(100, abs(drawdown) * 1000)  # Convert to 0-100 scale
        risk_factors.append(drawdown_risk)
        
        # Volatility risk
        volatility_risk = min(100, volatility * 500)  # Convert to 0-100 scale
        risk_factors.append(volatility_risk)
        
        # Correlation risk
        correlation_risk_score = correlation_risk * 100
        risk_factors.append(correlation_risk_score)
        
        # Concentration risk
        concentration_risk_score = concentration_risk * 100
        risk_factors.append(concentration_risk_score)
        
        return np.mean(risk_factors)
    
    def _check_risk_limits(self, portfolio_risk: PortfolioRisk, 
                          positions: Dict[str, PositionRisk]):
        """Check risk limits and generate alerts"""
        alerts = []
        
        # Margin call warning
        if portfolio_risk.margin_level < self.risk_limits['margin_call_level'] * 100:
            alerts.append(RiskAlert(
                alert_id=f"MARGIN_{int(time.time())}",
                timestamp=datetime.now(),
                risk_event=RiskEvent.MARGIN_CALL_WARNING,
                severity="high",
                message=f"Margin level critical: {portfolio_risk.margin_level:.1f}%",
                triggered_metrics={'margin_level': portfolio_risk.margin_level},
                action_required=True
            ))
        
        # Drawdown limit
        if portfolio_risk.current_drawdown < -self.risk_limits['max_drawdown']:
            alerts.append(RiskAlert(
                alert_id=f"DRAWDOWN_{int(time.time())}",
                timestamp=datetime.now(),
                risk_event=RiskEvent.DRAWDOWN_LIMIT,
                severity="high",
                message=f"Drawdown limit exceeded: {abs(portfolio_risk.current_drawdown):.1%}",
                triggered_metrics={'drawdown': portfolio_risk.current_drawdown},
                action_required=True
            ))
        
        # Daily loss limit
        if portfolio_risk.daily_pnl < -self.risk_limits['daily_loss_limit'] * self.initial_capital:
            alerts.append(RiskAlert(
                alert_id=f"DAILY_LOSS_{int(time.time())}",
                timestamp=datetime.now(),
                risk_event=RiskEvent.DAILY_LOSS_LIMIT,
                severity="medium",
                message=f"Daily loss limit exceeded: ${abs(portfolio_risk.daily_pnl):.2f}",
                triggered_metrics={'daily_loss': portfolio_risk.daily_pnl},
                action_required=True
            ))
        
        # Position limit
        if len(positions) >= self.risk_limits['max_open_positions']:
            alerts.append(RiskAlert(
                alert_id=f"POSITION_LIMIT_{int(time.time())}",
                timestamp=datetime.now(),
                risk_event=RiskEvent.POSITION_LIMIT,
                severity="medium",
                message=f"Maximum open positions reached: {len(positions)}",
                triggered_metrics={'open_positions': len(positions)},
                action_required=False
            ))
        
        # Add new alerts
        for alert in alerts:
            self.risk_alerts.append(alert)
            logger.warning(f"RISK ALERT: {alert.message}")
        
        # Keep alerts manageable
        if len(self.risk_alerts) > 100:
            self.risk_alerts = self.risk_alerts[-100:]
    
    def _create_default_risk_metrics(self) -> PortfolioRisk:
        """Create default risk metrics when calculation fails"""
        return PortfolioRisk(
            timestamp=datetime.now(),
            total_equity=self.current_capital,
            used_margin=0.0,
            free_margin=self.current_capital,
            margin_level=float('inf'),
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
            daily_pnl=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            volatility_annual=0.0,
            sharpe_ratio=0.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            risk_score=0.0
        )
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get active (unresolved) risk alerts"""
        return [alert for alert in self.risk_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert in self.risk_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Risk alert resolved: {alert_id}")

class RiskManager:
    """
    Main Risk Manager coordinating all risk management components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.position_sizer = PositionSizingCalculator(
            self.config.get('initial_capital', 10000.0)
        )
        self.stop_loss_calculator = StopLossCalculator()
        self.portfolio_analyzer = PortfolioRiskAnalyzer(
            self.config.get('initial_capital', 10000.0)
        )
        
        # Current state
        self.positions: Dict[str, PositionRisk] = {}
        self.cash = self.config.get('initial_capital', 10000.0)
        self.total_equity = self.config.get('initial_capital', 10000.0)
        
        # Risk limits from config
        self.risk_limits = self.config.get('risk_limits', {})
        
        # Monitoring
        self.risk_monitor_thread = None
        self.running = False
        self.monitor_interval = self.config.get('monitor_interval', 30)  # seconds
        
        logger.info("Risk Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk management configuration"""
        return {
            'initial_capital': 10000.0,
            'monitor_interval': 30,
            'risk_limits': {
                'max_drawdown': 0.15,
                'daily_loss_limit': 0.05,
                'margin_call_level': 0.8,
                'stop_out_level': 0.5,
                'max_position_size': 0.2,
                'max_correlation': 0.7,
                'max_open_positions': 5,
                'var_confidence': 0.95
            }
        }
    
    def start_risk_monitoring(self):
        """Start real-time risk monitoring"""
        if self.running:
            return
        
        self.running = True
        self.risk_monitor_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
        self.risk_monitor_thread.start()
        logger.info("Risk monitoring started")
    
    def stop_risk_monitoring(self):
        """Stop real-time risk monitoring"""
        self.running = False
        if self.risk_monitor_thread:
            self.risk_monitor_thread.join(timeout=10.0)
        logger.info("Risk monitoring stopped")
    
    def _risk_monitoring_loop(self):
        """Main risk monitoring loop"""
        while self.running:
            try:
                # Update portfolio risk analysis
                portfolio_risk = self.portfolio_analyzer.update_portfolio_state(
                    self.positions, self.cash, self.total_equity
                )
                
                # Check for critical alerts that require immediate action
                critical_alerts = [
                    alert for alert in self.portfolio_analyzer.get_active_alerts()
                    if alert.action_required and alert.severity == "high"
                ]
                
                if critical_alerts:
                    logger.critical(f"CRITICAL RISK ALERTS: {len(critical_alerts)} require immediate action")
                    # In production, this would trigger automatic risk reduction
                
                # Wait for next monitoring cycle
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(self.monitor_interval)  # Continue after error
    
    def calculate_position_size(self, symbol: str, current_price: float,
                              strategy_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate position size for a new trade"""
        return self.position_sizer.calculate_dynamic_position_size(
            symbol=symbol,
            current_price=current_price,
            win_rate=strategy_metrics.get('win_rate', 0.5),
            avg_win=strategy_metrics.get('avg_win', 0.02),
            avg_loss=strategy_metrics.get('avg_loss', 0.01),
            volatility=strategy_metrics.get('volatility', 0.15),
            correlation=strategy_metrics.get('correlation', 0.0),
            portfolio_volatility=strategy_metrics.get('portfolio_volatility', 0.15),
            confidence=strategy_metrics.get('confidence', 0.7)
        )
    
    def calculate_stop_loss(self, symbol: str, entry_price: float,
                          position_type: str, market_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate stop loss and take profit levels"""
        return self.stop_loss_calculator.calculate_dynamic_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            position_type=position_type,
            highs=market_data.get('highs', []),
            lows=market_data.get('lows', []),
            closes=market_data.get('closes', [])
        )
    
    def can_open_position(self, symbol: str, proposed_size: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a new position can be opened"""
        try:
            # Check position size limit
            position_size_pct = proposed_size['position_size_pct']
            if position_size_pct > self.risk_limits['max_position_size']:
                return {
                    'allowed': False,
                    'reason': f"Position size {position_size_pct:.1%} exceeds maximum {self.risk_limits['max_position_size']:.1%}",
                    'suggested_size': self.risk_limits['max_position_size'] * 0.8
                }
            
            # Check open positions limit
            if len(self.positions) >= self.risk_limits['max_open_positions']:
                return {
                    'allowed': False,
                    'reason': f"Maximum open positions ({self.risk_limits['max_open_positions']}) reached",
                    'suggested_size': 0.0
                }
            
            # Check margin requirements
            required_margin = proposed_size['risk_amount'] * 0.1  # 10:1 leverage
            if required_margin > self.portfolio_analyzer.current_capital * 0.2:  # 20% of capital
                return {
                    'allowed': False,
                    'reason': "Insufficient margin available",
                    'suggested_size': position_size_pct * 0.5
                }
            
            # Check current risk level
            portfolio_risk = self.portfolio_analyzer.update_portfolio_state(
                self.positions, self.cash, self.total_equity
            )
            
            if portfolio_risk.risk_score > 70:  # High risk level
                return {
                    'allowed': False,
                    'reason': "Portfolio risk level too high",
                    'suggested_size': 0.0
                }
            
            return {
                'allowed': True,
                'reason': "Position meets all risk criteria",
                'suggested_size': position_size_pct
            }
            
        except Exception as e:
            logger.error(f"Error checking position opening: {e}")
            return {
                'allowed': False,
                'reason': f"Error in risk check: {str(e)}",
                'suggested_size': 0.0
            }
    
    def add_position(self, symbol: str, position_data: Dict[str, Any]):
        """Add a new position to risk management"""
        position_risk = PositionRisk(
            symbol=symbol,
            position_type=position_data['position_type'],
            entry_price=position_data['entry_price'],
            current_price=position_data['entry_price'],
            quantity=position_data['quantity'],
            notional_value=position_data['entry_price'] * position_data['quantity'],
            margin_used=position_data['quantity'] * position_data['entry_price'] * 0.1,  # 10:1 leverage
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss=position_data['stop_loss'],
            take_profit=position_data['take_profit'],
            risk_reward_ratio=position_data.get('risk_reward_ratio', 2.0),
            time_in_trade=timedelta(0)
        )
        
        self.positions[symbol] = position_risk
        
        # Update cash and equity
        self.cash -= position_risk.margin_used
        self.total_equity = self.cash + sum(
            pos.margin_used + pos.unrealized_pnl for pos in self.positions.values()
        )
        
        logger.info(f"Position added: {symbol}, Margin used: ${position_risk.margin_used:.2f}")
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update position with current market price"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.current_price = current_price
        
        # Calculate unrealized P&L
        if position.position_type == 'LONG':
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        else:  # SHORT
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
        
        position.unrealized_pnl_pct = (position.unrealized_pnl / 
                                     (position.entry_price * position.quantity)) * 100
        
        position.time_in_trade = datetime.now() - position.time_in_trade  # This would need proper tracking
        
        # Update total equity
        self.total_equity = self.cash + sum(
            pos.margin_used + pos.unrealized_pnl for pos in self.positions.values()
        )
    
    def close_position(self, symbol: str, exit_price: float):
        """Close a position and update risk metrics"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate final P&L
        if position.position_type == 'LONG':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Update cash and remove position
        self.cash += position.margin_used + pnl
        del self.positions[symbol]
        
        # Update total equity
        self.total_equity = self.cash + sum(
            pos.margin_used + pos.unrealized_pnl for pos in self.positions.values()
        )
        
        logger.info(f"Position closed: {symbol}, P&L: ${pnl:.2f}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        portfolio_risk = self.portfolio_analyzer.update_portfolio_state(
            self.positions, self.cash, self.total_equity
        )
        
        active_alerts = self.portfolio_analyzer.get_active_alerts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': asdict(portfolio_risk),
            'positions_summary': {
                'total_positions': len(self.positions),
                'symbols': list(self.positions.keys()),
                'total_margin_used': sum(pos.margin_used for pos in self.positions.values()),
                'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
            },
            'risk_alerts': {
                'active_alerts': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.severity == 'high']),
                'recent_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'event': alert.risk_event.value,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in active_alerts[-5:]  # Last 5 alerts
                ]
            },
            'risk_limits': self.risk_limits,
            'system_status': {
                'monitoring_active': self.running,
                'total_equity': self.total_equity,
                'cash_available': self.cash,
                'margin_available': portfolio_risk.free_margin
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the Risk Manager
    print("Testing Risk Manager...")
    
    try:
        # Initialize risk manager
        risk_manager = RiskManager({
            'initial_capital': 10000.0,
            'risk_limits': {
                'max_drawdown': 0.15,
                'daily_loss_limit': 0.05,
                'max_position_size': 0.2,
                'max_open_positions': 3
            }
        })
        
        # Start risk monitoring
        risk_manager.start_risk_monitoring()
        
        # Test position sizing
        print("Testing position sizing...")
        position_size = risk_manager.calculate_position_size(
            symbol="EUR/USD",
            current_price=1.0850,
            strategy_metrics={
                'win_rate': 0.6,
                'avg_win': 0.02,
                'avg_loss': 0.01,
                'volatility': 0.12,
                'correlation': 0.2,
                'portfolio_volatility': 0.15,
                'confidence': 0.75
            }
        )
        
        print(f"Position Size Calculation:")
        print(f"  Size: {position_size['position_size_pct']:.2%}")
        print(f"  Lot Size: {position_size['lot_size']:.4f}")
        print(f"  Risk Amount: ${position_size['risk_amount']:.2f}")
        
        # Test stop loss calculation
        print("\nTesting stop loss calculation...")
        # Generate sample market data
        sample_prices = [1.0800 + i * 0.0001 for i in range(100)]
        sample_highs = [p + 0.0005 for p in sample_prices]
        sample_lows = [p - 0.0005 for p in sample_prices]
        
        stop_loss_data = risk_manager.calculate_stop_loss(
            symbol="EUR/USD",
            entry_price=1.0850,
            position_type="LONG",
            market_data={
                'highs': sample_highs,
                'lows': sample_lows,
                'closes': sample_prices
            }
        )
        
        print(f"Stop Loss Calculation:")
        print(f"  Stop Loss %: {stop_loss_data['stop_loss_pct']:.2%}")
        print(f"  Stop Loss Price: {stop_loss_data['stop_loss_price']:.5f}")
        print(f"  Take Profit Price: {stop_loss_data['take_profit_price']:.5f}")
        print(f"  Risk/Reward: 1:{stop_loss_data['risk_reward_ratio']:.1f}")
        
        # Test position opening approval
        print("\nTesting position opening approval...")
        approval = risk_manager.can_open_position("EUR/USD", position_size)
        
        print(f"Position Approval: {'APPROVED' if approval['allowed'] else 'REJECTED'}")
        if not approval['allowed']:
            print(f"  Reason: {approval['reason']}")
            print(f"  Suggested Size: {approval['suggested_size']:.2%}")
        
        # Add a sample position if approved
        if approval['allowed']:
            risk_manager.add_position("EUR/USD", {
                'position_type': 'LONG',
                'entry_price': 1.0850,
                'quantity': position_size['lot_size'],
                'stop_loss': stop_loss_data['stop_loss_price'],
                'take_profit': stop_loss_data['take_profit_price'],
                'risk_reward_ratio': stop_loss_data['risk_reward_ratio']
            })
            
            # Update position with price movement
            risk_manager.update_position_price("EUR/USD", 1.0860)
        
        # Generate risk report
        print("\nGenerating risk report...")
        risk_report = risk_manager.get_risk_report()
        
        print(f"📊 RISK REPORT:")
        print(f"  Total Equity: ${risk_report['portfolio_metrics']['total_equity']:,.2f}")
        print(f"  Margin Level: {risk_report['portfolio_metrics']['margin_level']:.1f}%")
        print(f"  Current Drawdown: {risk_report['portfolio_metrics']['current_drawdown']:.2%}")
        print(f"  Risk Score: {risk_report['portfolio_metrics']['risk_score']:.1f}/100")
        print(f"  Active Alerts: {risk_report['risk_alerts']['active_alerts']}")
        
        # Stop risk monitoring
        risk_manager.stop_risk_monitoring()
        
        print(f"\n✅ Risk Manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Risk Manager test failed: {e}")
        import traceback
        traceback.print_exc()
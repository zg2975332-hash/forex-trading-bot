"""
Adaptive Risk Manager for Forex Trading Bot
Dynamic risk management with market condition adaptation and machine learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import warnings
from enum import Enum
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    CRASH = "crash"
    UNCERTAIN = "uncertain"

class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    position_type: str
    entry_price: float
    current_price: float
    quantity: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    margin_used: float
    risk_score: float
    stop_loss_price: float
    take_profit_price: float
    time_in_trade: timedelta

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_equity: float
    used_margin: float
    free_margin: float
    margin_level: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    volatility: float
    var_95: float
    cvar_95: float
    correlation_risk: float
    concentration_risk: float

class AdaptiveRiskManager:
    """
    Advanced adaptive risk management system with machine learning
    and real-time market condition analysis
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_drawdown_limit: float = 0.15,
                 daily_loss_limit: float = 0.05,
                 max_position_size: float = 0.2):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.max_position_size = max_position_size
        
        # Risk parameters
        self.risk_parameters = self._initialize_risk_parameters()
        self.market_regime = MarketRegime.UNCERTAIN
        self.current_risk_level = RiskLevel.MEDIUM
        
        # Tracking
        self.positions = {}
        self.portfolio_history = []
        self.trade_history = []
        self.risk_history = []
        
        # Machine learning features
        self.volatility_lookback = 20
        self.correlation_lookback = 60
        self.regime_lookback = 50
        
        # Performance metrics
        self.peak_equity = initial_capital
        self.daily_peak = initial_capital
        self.daily_realized_pnl = 0.0
        
        logger.info("Adaptive Risk Manager initialized")
    
    def _initialize_risk_parameters(self) -> Dict[str, Any]:
        """Initialize adaptive risk parameters"""
        return {
            'base_risk_per_trade': 0.02,
            'volatility_multipliers': {
                'very_low': 1.5,
                'low': 1.2,
                'medium': 1.0,
                'high': 0.7,
                'very_high': 0.5,
                'extreme': 0.3
            },
            'regime_multipliers': {
                'trending_bull': 1.2,
                'trending_bear': 1.1,
                'ranging': 0.8,
                'volatile': 0.6,
                'calm': 1.0,
                'crash': 0.2,
                'uncertain': 0.5
            },
            'correlation_penalties': {
                'low': 1.0,
                'medium': 0.8,
                'high': 0.5,
                'very_high': 0.3
            },
            'position_sizing_methods': {
                'kelly': 0.5,
                'volatility_adjusted': 0.3,
                'regime_based': 0.2
            },
            'stop_loss_methods': {
                'atr_based': 0.6,
                'volatility_based': 0.3,
                'support_resistance': 0.1
            }
        }
    
    def update_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime using multiple indicators
        
        Args:
            market_data: OHLCV data for analysis
            
        Returns:
            Detected market regime
        """
        try:
            if len(market_data) < self.regime_lookback:
                return MarketRegime.UNCERTAIN
            
            closes = market_data['close'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            # Calculate regime indicators
            trend_strength = self._calculate_trend_strength(closes)
            volatility = self._calculate_volatility(closes)
            adx_value = self._calculate_adx(highs, lows, closes)
            rsi_value = self._calculate_rsi(closes)
            
            # Regime detection logic
            if volatility > 0.02:  # High volatility
                if trend_strength > 0.7:
                    regime = MarketRegime.VOLATILE
                else:
                    regime = MarketRegime.CRASH
            elif adx_value > 25:  # Strong trend
                if trend_strength > 0:
                    regime = MarketRegime.TRENDING_BULL
                else:
                    regime = MarketRegime.TRENDING_BEAR
            elif volatility < 0.005:  # Low volatility
                regime = MarketRegime.CALM
            elif adx_value < 20:  # Weak trend
                regime = MarketRegime.RANGING
            else:
                regime = MarketRegime.UNCERTAIN
            
            self.market_regime = regime
            logger.info(f"Market regime detected: {regime.value}")
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.UNCERTAIN
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        trend_direction = np.sign(slope)
        trend_strength = abs(r_value)
        
        return trend_strength * trend_direction
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate volatility as annualized standard deviation"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        daily_volatility = np.std(returns)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
    
    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                      period: int = 14) -> float:
        """Calculate Average Directional Index"""
        if len(highs) < period * 2:
            return 0.0
        
        try:
            # Simplified ADX calculation
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
            atr = np.mean(tr[-period:])
            
            if atr == 0:
                return 0.0
            
            # Simplified ADX (in production, use proper ADX calculation)
            adx = min(50.0, (atr / np.mean(closes[-period:])) * 100)
            return adx
            
        except:
            return 0.0
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except:
            return 50.0
    
    def calculate_dynamic_position_size(self, 
                                      symbol: str,
                                      current_price: float,
                                      confidence: float,
                                      volatility: float,
                                      correlation: float = 0.0) -> Dict[str, float]:
        """
        Calculate dynamic position size based on multiple factors
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            confidence: Model confidence (0-1)
            volatility: Current volatility
            correlation: Correlation with existing positions
            
        Returns:
            Dictionary with position sizing details
        """
        try:
            # Base position size from Kelly criterion
            kelly_size = self._calculate_kelly_position_size(confidence)
            
            # Volatility adjustment
            vol_adjustment = self._calculate_volatility_adjustment(volatility)
            
            # Regime adjustment
            regime_adjustment = self.risk_parameters['regime_multipliers'][self.market_regime.value]
            
            # Correlation penalty
            correlation_penalty = self._calculate_correlation_penalty(correlation)
            
            # Confidence multiplier
            confidence_multiplier = self._calculate_confidence_multiplier(confidence)
            
            # Calculate final position size
            base_size = self.risk_parameters['base_risk_per_trade']
            adjusted_size = (base_size * kelly_size * vol_adjustment * 
                           regime_adjustment * correlation_penalty * confidence_multiplier)
            
            # Apply limits
            final_size = min(adjusted_size, self.max_position_size)
            final_size = max(final_size, 0.01)  # Minimum 1%
            
            # Calculate lot size
            risk_amount = self.current_capital * final_size
            lot_size = risk_amount / current_price
            
            position_details = {
                'position_size_pct': final_size,
                'lot_size': lot_size,
                'risk_amount': risk_amount,
                'kelly_size': kelly_size,
                'volatility_adjustment': vol_adjustment,
                'regime_adjustment': regime_adjustment,
                'correlation_penalty': correlation_penalty,
                'confidence_multiplier': confidence_multiplier,
                'market_regime': self.market_regime.value
            }
            
            logger.info(f"Position size calculated for {symbol}: {final_size:.2%}")
            
            return position_details
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            # Return conservative default
            return {
                'position_size_pct': 0.01,
                'lot_size': (self.current_capital * 0.01) / current_price,
                'risk_amount': self.current_capital * 0.01,
                'kelly_size': 0.5,
                'volatility_adjustment': 1.0,
                'regime_adjustment': 1.0,
                'correlation_penalty': 1.0,
                'confidence_multiplier': 1.0,
                'market_regime': self.market_regime.value
            }
    
    def _calculate_kelly_position_size(self, confidence: float) -> float:
        """Calculate position size using Kelly criterion"""
        # Simplified Kelly: f = (bp - q) / b
        # Where b is odds (2:1 for forex), p is win probability, q is loss probability
        win_probability = confidence
        win_loss_ratio = 2.0  # Typical forex risk:reward
        
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Apply fractional Kelly for safety
        fractional_kelly = kelly_fraction * 0.5  # Use half Kelly
        
        return max(fractional_kelly, 0.05)  # Minimum 5% of Kelly
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Adjust position size based on volatility"""
        if volatility < 0.05:
            risk_level = RiskLevel.VERY_LOW
        elif volatility < 0.1:
            risk_level = RiskLevel.LOW
        elif volatility < 0.15:
            risk_level = RiskLevel.MEDIUM
        elif volatility < 0.25:
            risk_level = RiskLevel.HIGH
        elif volatility < 0.4:
            risk_level = RiskLevel.VERY_HIGH
        else:
            risk_level = RiskLevel.EXTREME
        
        self.current_risk_level = risk_level
        return self.risk_parameters['volatility_multipliers'][risk_level.value]
    
    def _calculate_correlation_penalty(self, correlation: float) -> float:
        """Apply correlation penalty to position size"""
        abs_correlation = abs(correlation)
        
        if abs_correlation < 0.3:
            level = 'low'
        elif abs_correlation < 0.6:
            level = 'medium'
        elif abs_correlation < 0.8:
            level = 'high'
        else:
            level = 'very_high'
        
        return self.risk_parameters['correlation_penalties'][level]
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Adjust position size based on model confidence"""
        if confidence >= 0.8:
            return 1.5
        elif confidence >= 0.7:
            return 1.2
        elif confidence >= 0.6:
            return 1.0
        elif confidence >= 0.5:
            return 0.8
        else:
            return 0.5
    
    def calculate_dynamic_stop_loss(self,
                                  symbol: str,
                                  entry_price: float,
                                  position_type: str,
                                  volatility: float,
                                  atr: float) -> Dict[str, float]:
        """
        Calculate dynamic stop loss levels
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_type: LONG or SHORT
            volatility: Current volatility
            atr: Average True Range
            
        Returns:
            Dictionary with stop loss details
        """
        try:
            # ATR-based stop loss
            atr_stop_distance = atr * 2.0  # 2 ATRs
            
            # Volatility-based stop loss
            vol_stop_distance = entry_price * volatility * 2.0
            
            # Regime-based adjustment
            regime_multiplier = self.risk_parameters['regime_multipliers'][self.market_regime.value]
            
            # Combine methods
            base_stop_distance = (atr_stop_distance * 0.6 + vol_stop_distance * 0.4)
            adjusted_stop_distance = base_stop_distance * regime_multiplier
            
            # Calculate stop loss price
            if position_type == 'LONG':
                stop_loss_price = entry_price - adjusted_stop_distance
                take_profit_price = entry_price + (adjusted_stop_distance * 2.0)  # 1:2 risk:reward
            else:  # SHORT
                stop_loss_price = entry_price + adjusted_stop_distance
                take_profit_price = entry_price - (adjusted_stop_distance * 2.0)
            
            # Ensure stop loss is reasonable
            stop_loss_pct = abs(stop_loss_price - entry_price) / entry_price
            if stop_loss_pct > 0.1:  # Max 10% stop loss
                stop_loss_price = entry_price * (0.9 if position_type == 'LONG' else 1.1)
            
            stop_details = {
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'stop_distance_pct': abs(stop_loss_price - entry_price) / entry_price,
                'risk_reward_ratio': 2.0,
                'method': 'dynamic_combined',
                'atr_distance': atr_stop_distance,
                'vol_distance': vol_stop_distance
            }
            
            logger.info(f"Stop loss calculated for {symbol}: {stop_loss_price:.5f}")
            
            return stop_details
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Return conservative default
            default_stop_pct = 0.02  # 2% default stop loss
            
            if position_type == 'LONG':
                stop_loss = entry_price * (1 - default_stop_pct)
                take_profit = entry_price * (1 + default_stop_pct * 2)
            else:
                stop_loss = entry_price * (1 + default_stop_pct)
                take_profit = entry_price * (1 - default_stop_pct * 2)
            
            return {
                'stop_loss_price': stop_loss,
                'take_profit_price': take_profit,
                'stop_distance_pct': default_stop_pct,
                'risk_reward_ratio': 2.0,
                'method': 'default_fallback'
            }
    
    def assess_portfolio_risk(self, positions: Dict[str, PositionRisk]) -> PortfolioRisk:
        """
        Assess overall portfolio risk
        
        Args:
            positions: Dictionary of current positions
            
        Returns:
            PortfolioRisk object with risk metrics
        """
        try:
            # Calculate basic metrics
            total_equity = self.current_capital
            used_margin = 0.0
            total_unrealized_pnl = 0.0
            
            for position in positions.values():
                used_margin += position.margin_used
                total_unrealized_pnl += position.unrealized_pnl
            
            total_equity += total_unrealized_pnl
            free_margin = total_equity - used_margin
            margin_level = (total_equity / used_margin) * 100 if used_margin > 0 else float('inf')
            
            # Update peak equity for drawdown calculation
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity
            
            current_drawdown = (self.peak_equity - total_equity) / self.peak_equity
            
            # Calculate additional risk metrics
            volatility = self._calculate_portfolio_volatility()
            var_95 = self._calculate_value_at_risk(0.95)
            cvar_95 = self._calculate_conditional_var(0.95)
            correlation_risk = self._calculate_correlation_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions)
            
            portfolio_risk = PortfolioRisk(
                total_equity=total_equity,
                used_margin=used_margin,
                free_margin=free_margin,
                margin_level=margin_level,
                total_unrealized_pnl=total_unrealized_pnl,
                total_realized_pnl=self.daily_realized_pnl,
                daily_pnl=self.daily_realized_pnl,
                max_drawdown=current_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=0.0,  # Would require historical returns
                volatility=volatility,
                var_95=var_95,
                cvar_95=cvar_95,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk
            )
            
            # Log risk assessment
            if margin_level < 100:
                logger.warning(f"Margin level critical: {margin_level:.1f}%")
            if current_drawdown > self.max_drawdown_limit:
                logger.warning(f"Drawdown exceeded limit: {current_drawdown:.1%}")
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            # Return safe default
            return PortfolioRisk(
                total_equity=self.current_capital,
                used_margin=0.0,
                free_margin=self.current_capital,
                margin_level=float('inf'),
                total_unrealized_pnl=0.0,
                total_realized_pnl=0.0,
                daily_pnl=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
                var_95=0.0,
                cvar_95=0.0,
                correlation_risk=0.0,
                concentration_risk=0.0
            )
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility from historical returns"""
        # Simplified implementation
        # In production, use actual portfolio return history
        return 0.15  # 15% annual volatility as default
    
    def _calculate_value_at_risk(self, confidence: float) -> float:
        """Calculate Value at Risk"""
        # Simplified VaR calculation
        portfolio_value = self.current_capital
        volatility = 0.15  # Annual volatility
        z_score = stats.norm.ppf(confidence)
        var = portfolio_value * z_score * volatility / np.sqrt(252)
        
        return abs(var)
    
    def _calculate_conditional_var(self, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        # Simplified CVaR calculation
        var = self._calculate_value_at_risk(confidence)
        cvar = var * 1.3  # CVaR is typically 20-30% higher than VaR
        
        return cvar
    
    def _calculate_correlation_risk(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate correlation risk in portfolio"""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation risk calculation
        # In production, use actual correlation matrix
        symbols = list(positions.keys())
        return min(1.0, len(symbols) * 0.1)  # Assume some correlation risk
    
    def _calculate_concentration_risk(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate concentration risk"""
        if not positions:
            return 0.0
        
        position_sizes = [abs(pos.quantity * pos.current_price) for pos in positions.values()]
        total_exposure = sum(position_sizes)
        
        if total_exposure == 0:
            return 0.0
        
        # Herfindahl index for concentration
        herfindahl = sum((size / total_exposure) ** 2 for size in position_sizes)
        
        return herfindahl
    
    def should_enter_trade(self, 
                          symbol: str,
                          signal_confidence: float,
                          portfolio_risk: PortfolioRisk,
                          correlation: float = 0.0) -> Dict[str, Any]:
        """
        Determine if a trade should be entered based on risk assessment
        
        Args:
            symbol: Trading symbol
            signal_confidence: Model confidence
            portfolio_risk: Current portfolio risk
            correlation: Correlation with existing positions
            
        Returns:
            Dictionary with trade decision and reasons
        """
        try:
            reasons = []
            allowed = True
            
            # Check confidence threshold
            min_confidence = 0.6
            if signal_confidence < min_confidence:
                allowed = False
                reasons.append(f"Low confidence: {signal_confidence:.2f} < {min_confidence}")
            
            # Check margin level
            min_margin_level = 200  # 200% margin level
            if portfolio_risk.margin_level < min_margin_level:
                allowed = False
                reasons.append(f"Low margin level: {portfolio_risk.margin_level:.1f}%")
            
            # Check drawdown limit
            if portfolio_risk.current_drawdown > self.max_drawdown_limit:
                allowed = False
                reasons.append(f"Drawdown limit exceeded: {portfolio_risk.current_drawdown:.1%}")
            
            # Check daily loss limit
            if portfolio_risk.daily_pnl < -self.daily_loss_limit * self.initial_capital:
                allowed = False
                reasons.append("Daily loss limit exceeded")
            
            # Check correlation limit
            max_correlation = 0.7
            if abs(correlation) > max_correlation:
                allowed = False
                reasons.append(f"High correlation: {correlation:.2f}")
            
            # Check position count limit
            max_positions = 5
            if len(self.positions) >= max_positions:
                allowed = False
                reasons.append(f"Maximum positions reached: {len(self.positions)}")
            
            decision = {
                'allowed': allowed,
                'reasons': reasons,
                'risk_level': self.current_risk_level.value,
                'market_regime': self.market_regime.value,
                'timestamp': datetime.now()
            }
            
            if not allowed:
                logger.warning(f"Trade rejected for {symbol}: {', '.join(reasons)}")
            else:
                logger.info(f"Trade approved for {symbol}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in trade decision: {e}")
            return {
                'allowed': False,
                'reasons': [f"Error: {str(e)}"],
                'risk_level': 'error',
                'market_regime': 'uncertain',
                'timestamp': datetime.now()
            }
    
    def update_position(self, position: PositionRisk):
        """Update position in risk manager"""
        self.positions[position.symbol] = position
    
    def close_position(self, symbol: str, realized_pnl: float):
        """Close position and update realized P&L"""
        if symbol in self.positions:
            del self.positions[symbol]
            self.daily_realized_pnl += realized_pnl
            self.current_capital += realized_pnl
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        portfolio_risk = self.assess_portfolio_risk(self.positions)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': {
                'total_equity': portfolio_risk.total_equity,
                'used_margin': portfolio_risk.used_margin,
                'free_margin': portfolio_risk.free_margin,
                'margin_level': portfolio_risk.margin_level,
                'current_drawdown': portfolio_risk.current_drawdown,
                'max_drawdown': portfolio_risk.max_drawdown,
                'daily_pnl': portfolio_risk.daily_pnl
            },
            'risk_metrics': {
                'volatility': portfolio_risk.volatility,
                'var_95': portfolio_risk.var_95,
                'cvar_95': portfolio_risk.cvar_95,
                'correlation_risk': portfolio_risk.correlation_risk,
                'concentration_risk': portfolio_risk.concentration_risk
            },
            'market_conditions': {
                'regime': self.market_regime.value,
                'risk_level': self.current_risk_level.value
            },
            'positions_count': len(self.positions),
            'risk_limits': {
                'max_drawdown': self.max_drawdown_limit,
                'daily_loss_limit': self.daily_loss_limit,
                'max_position_size': self.max_position_size
            }
        }
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Test the adaptive risk manager
    print("Testing Adaptive Risk Manager...")
    
    try:
        # Initialize risk manager
        risk_manager = AdaptiveRiskManager(
            initial_capital=10000.0,
            max_drawdown_limit=0.15,
            daily_loss_limit=0.05,
            max_position_size=0.2
        )
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        prices = 1.1000 + np.cumsum(np.random.randn(100) * 0.001)
        
        market_data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.0002,
            'high': prices + np.abs(np.random.randn(100) * 0.0005),
            'low': prices - np.abs(np.random.randn(100) * 0.0005),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test market regime detection
        regime = risk_manager.update_market_regime(market_data)
        print(f"Detected market regime: {regime.value}")
        
        # Test position sizing
        position_size = risk_manager.calculate_dynamic_position_size(
            symbol="EUR/USD",
            current_price=1.0850,
            confidence=0.75,
            volatility=0.12,
            correlation=0.3
        )
        
        print("\nPosition Sizing Results:")
        for key, value in position_size.items():
            print(f"  {key}: {value}")
        
        # Test stop loss calculation
        stop_loss = risk_manager.calculate_dynamic_stop_loss(
            symbol="EUR/USD",
            entry_price=1.0850,
            position_type="LONG",
            volatility=0.12,
            atr=0.0008
        )
        
        print("\nStop Loss Results:")
        for key, value in stop_loss.items():
            print(f"  {key}: {value}")
        
        # Test trade decision
        portfolio_risk = risk_manager.assess_portfolio_risk({})
        trade_decision = risk_manager.should_enter_trade(
            symbol="EUR/USD",
            signal_confidence=0.8,
            portfolio_risk=portfolio_risk,
            correlation=0.2
        )
        
        print(f"\nTrade Decision: {'APPROVED' if trade_decision['allowed'] else 'REJECTED'}")
        if trade_decision['reasons']:
            print("Reasons:", trade_decision['reasons'])
        
        # Generate risk report
        risk_report = risk_manager.get_risk_report()
        print(f"\nRisk Report Generated")
        print(f"Equity: ${risk_report['portfolio_metrics']['total_equity']:,.2f}")
        print(f"Drawdown: {risk_report['portfolio_metrics']['current_drawdown']:.2%}")
        
        print(f"\n✅ Adaptive Risk Manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Adaptive Risk Manager test failed: {e}")
"""
Advanced VaR Calculator for FOREX TRADING BOT
Value at Risk calculation using multiple methods with risk management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import threading
from scipy import stats
from scipy.optimize import minimize
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VaRMethod(Enum):
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    GARCH = "garch"
    EXPECTED_SHORTFALL = "expected_shortfall"
    CONDITIONAL_VAR = "conditional_var"

class ConfidenceLevel(Enum):
    LEVEL_95 = 0.95
    LEVEL_99 = 0.99
    LEVEL_99_5 = 0.995
    LEVEL_99_9 = 0.999

class RiskHorizon(Enum):
    DAILY = 1
    WEEKLY = 5
    MONTHLY = 21
    QUARTERLY = 63
    YEARLY = 252

@dataclass
class VaRResult:
    """VaR calculation result"""
    var_method: VaRMethod
    confidence_level: float
    risk_horizon: int
    var_value: float
    var_percentage: float
    cvar_value: float
    cvar_percentage: float
    portfolio_value: float
    timestamp: datetime
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskLimit:
    """Risk limit configuration"""
    var_limit_95: float
    var_limit_99: float
    cvar_limit_95: float
    max_drawdown_limit: float
    concentration_limit: float
    sector_limits: Dict[str, float] = field(default_factory=dict)

@dataclass
class VaRConfig:
    """Configuration for VaR calculator"""
    # Calculation settings
    default_method: VaRMethod = VaRMethod.PARAMETRIC
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.995])
    risk_horizons: List[int] = field(default_factory=lambda: [1, 5, 21])
    
    # Historical settings
    historical_lookback: int = 252
    min_historical_data: int = 50
    
    # Parametric settings
    distribution_type: str = "student_t"  # normal, student_t, skewed_t
    degrees_freedom: int = 5
    
    # Monte Carlo settings
    mc_simulations: int = 10000
    mc_time_steps: int = 1
    
    # GARCH settings
    garch_p: int = 1
    garch_q: int = 1
    
    # Risk management
    enable_stress_testing: bool = True
    enable_backtesting: bool = True
    enable_component_var: bool = True
    
    # Reporting
    save_reports: bool = True
    report_frequency: str = "daily"

class AdvancedVaRCalculator:
    """
    Advanced Value at Risk calculator with multiple methodologies
    """
    
    def __init__(self, config: VaRConfig = None):
        self.config = config or VaRConfig()
        
        # Data storage
        self.portfolio_returns: pd.Series = None
        self.asset_returns: Dict[str, pd.Series] = {}
        self.portfolio_weights: Dict[str, float] = {}
        self.correlation_matrix: pd.DataFrame = None
        
        # Results storage
        self.var_results: Dict[str, VaRResult] = {}
        self.risk_limits: Optional[RiskLimit] = None
        self.backtest_results: Dict[str, Any] = {}
        
        # Statistical models
        self.distribution_params: Dict[str, Any] = {}
        self.garch_models: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("AdvancedVaRCalculator initialized successfully")

    def set_portfolio_data(self, portfolio_returns: pd.Series, 
                          asset_returns: Dict[str, pd.Series] = None,
                          portfolio_weights: Dict[str, float] = None) -> None:
        """Set portfolio and asset data for VaR calculation"""
        try:
            self.portfolio_returns = portfolio_returns
            
            if asset_returns:
                self.asset_returns = asset_returns
            
            if portfolio_weights:
                self.portfolio_weights = portfolio_weights
                self._validate_portfolio_weights()
                
            # Calculate correlation matrix if we have asset returns
            if asset_returns and len(asset_returns) > 1:
                self._calculate_correlation_matrix()
            
            logger.info(f"Portfolio data set with {len(portfolio_returns)} returns")
            
        except Exception as e:
            logger.error(f"Portfolio data setting failed: {e}")
            raise

    def _validate_portfolio_weights(self) -> None:
        """Validate portfolio weights sum to 1"""
        total_weight = sum(self.portfolio_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Portfolio weights sum to {total_weight}, normalizing to 1.0")
            self.portfolio_weights = {k: v/total_weight for k, v in self.portfolio_weights.items()}

    def _calculate_correlation_matrix(self) -> None:
        """Calculate correlation matrix from asset returns"""
        try:
            returns_data = {}
            for symbol, returns in self.asset_returns.items():
                returns_data[symbol] = returns
            
            returns_df = pd.DataFrame(returns_data)
            self.correlation_matrix = returns_df.corr()
            
            logger.info("Correlation matrix calculated successfully")
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")

    def calculate_var(self, portfolio_value: float = 100000.0,
                     method: VaRMethod = None,
                     confidence_level: float = None,
                     risk_horizon: int = None) -> VaRResult:
        """Calculate Value at Risk using specified method"""
        try:
            if self.portfolio_returns is None:
                raise ValueError("Portfolio data not set. Call set_portfolio_data() first.")
            
            method = method or self.config.default_method
            confidence_level = confidence_level or self.config.confidence_levels[0]
            risk_horizon = risk_horizon or self.config.risk_horizons[0]
            
            if method == VaRMethod.PARAMETRIC:
                return self._parametric_var(portfolio_value, confidence_level, risk_horizon)
            elif method == VaRMethod.HISTORICAL:
                return self._historical_var(portfolio_value, confidence_level, risk_horizon)
            elif method == VaRMethod.MONTE_CARLO:
                return self._monte_carlo_var(portfolio_value, confidence_level, risk_horizon)
            elif method == VaRMethod.GARCH:
                return self._garch_var(portfolio_value, confidence_level, risk_horizon)
            elif method == VaRMethod.EXPECTED_SHORTFALL:
                return self._expected_shortfall(portfolio_value, confidence_level, risk_horizon)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
                
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise

    def _parametric_var(self, portfolio_value: float, confidence_level: float, 
                       risk_horizon: int) -> VaRResult:
        """Calculate parametric VaR (variance-covariance method)"""
        try:
            returns = self.portfolio_returns
            
            if self.config.distribution_type == "normal":
                # Normal distribution VaR
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                var_value = portfolio_value * (mean * risk_horizon - z_score * std * np.sqrt(risk_horizon))
                
            elif self.config.distribution_type == "student_t":
                # Student's t-distribution VaR
                df, loc, scale = stats.t.fit(returns)
                t_score = stats.t.ppf(1 - confidence_level, df)
                var_value = portfolio_value * (loc * risk_horizon - t_score * scale * np.sqrt(risk_horizon))
                
            else:
                # Fallback to normal
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                var_value = portfolio_value * (mean * risk_horizon - z_score * std * np.sqrt(risk_horizon))
            
            # Calculate CVaR
            cvar_value = self._calculate_parametric_cvar(portfolio_value, confidence_level, risk_horizon)
            
            # Calculate component VaR if weights are available
            components = {}
            if self.portfolio_weights and self.asset_returns:
                components = self._calculate_component_var(portfolio_value, confidence_level, risk_horizon)
            
            result = VaRResult(
                var_method=VaRMethod.PARAMETRIC,
                confidence_level=confidence_level,
                risk_horizon=risk_horizon,
                var_value=abs(var_value),
                var_percentage=abs(var_value) / portfolio_value,
                cvar_value=abs(cvar_value),
                cvar_percentage=abs(cvar_value) / portfolio_value,
                portfolio_value=portfolio_value,
                timestamp=datetime.now(),
                components=components,
                metadata={
                    'distribution_type': self.config.distribution_type,
                    'data_points': len(returns)
                }
            )
            
            logger.info(f"Parametric VaR calculated: {confidence_level:.1%} = ${abs(var_value):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Parametric VaR calculation failed: {e}")
            raise

    def _calculate_parametric_cvar(self, portfolio_value: float, confidence_level: float,
                                 risk_horizon: int) -> float:
        """Calculate parametric Conditional VaR"""
        try:
            returns = self.portfolio_returns
            
            if self.config.distribution_type == "normal":
                # Normal distribution CVaR
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                pdf_z = stats.norm.pdf(z_score)
                cvar_value = portfolio_value * (mean * risk_horizon - (pdf_z / (1 - confidence_level)) * std * np.sqrt(risk_horizon))
                
            elif self.config.distribution_type == "student_t":
                # Student's t-distribution CVaR
                df, loc, scale = stats.t.fit(returns)
                t_score = stats.t.ppf(1 - confidence_level, df)
                pdf_t = stats.t.pdf(t_score, df)
                cvar_value = portfolio_value * (loc * risk_horizon - (pdf_t / (1 - confidence_level)) * scale * np.sqrt(risk_horizon))
                
            else:
                # Fallback to normal
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                pdf_z = stats.norm.pdf(z_score)
                cvar_value = portfolio_value * (mean * risk_horizon - (pdf_z / (1 - confidence_level)) * std * np.sqrt(risk_horizon))
            
            return abs(cvar_value)
            
        except Exception as e:
            logger.error(f"Parametric CVaR calculation failed: {e}")
            # Fallback to simple approximation
            return self._parametric_var(portfolio_value, confidence_level, risk_horizon).var_value * 1.2

    def _historical_var(self, portfolio_value: float, confidence_level: float,
                       risk_horizon: int) -> VaRResult:
        """Calculate historical VaR"""
        try:
            returns = self.portfolio_returns
            
            if len(returns) < self.config.min_historical_data:
                raise ValueError(f"Insufficient historical data: {len(returns)} < {self.config.min_historical_data}")
            
            # Calculate historical returns for the risk horizon
            if risk_horizon > 1:
                # Create rolling returns for the horizon
                rolling_returns = returns.rolling(window=risk_horizon).sum().dropna()
            else:
                rolling_returns = returns
            
            # Calculate VaR as percentile of historical returns
            var_percentile = np.percentile(rolling_returns, (1 - confidence_level) * 100)
            var_value = portfolio_value * abs(var_percentile)
            
            # Calculate CVaR (average of returns worse than VaR)
            tail_returns = rolling_returns[rolling_returns <= var_percentile]
            cvar_percentile = tail_returns.mean() if len(tail_returns) > 0 else var_percentile
            cvar_value = portfolio_value * abs(cvar_percentile)
            
            result = VaRResult(
                var_method=VaRMethod.HISTORICAL,
                confidence_level=confidence_level,
                risk_horizon=risk_horizon,
                var_value=var_value,
                var_percentage=var_value / portfolio_value,
                cvar_value=cvar_value,
                cvar_percentage=cvar_value / portfolio_value,
                portfolio_value=portfolio_value,
                timestamp=datetime.now(),
                metadata={
                    'historical_periods': len(rolling_returns),
                    'tail_observations': len(tail_returns)
                }
            )
            
            logger.info(f"Historical VaR calculated: {confidence_level:.1%} = ${var_value:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Historical VaR calculation failed: {e}")
            raise

    def _monte_carlo_var(self, portfolio_value: float, confidence_level: float,
                        risk_horizon: int) -> VaRResult:
        """Calculate Monte Carlo VaR"""
        try:
            if not self.portfolio_weights or not self.asset_returns:
                raise ValueError("Asset returns and weights required for Monte Carlo VaR")
            
            n_simulations = self.config.mc_simulations
            symbols = list(self.portfolio_weights.keys())
            n_assets = len(symbols)
            
            # Generate correlated returns
            correlated_returns = self._generate_correlated_returns(symbols, n_simulations, risk_horizon)
            
            # Calculate portfolio returns for each simulation
            weights_array = np.array([self.portfolio_weights[sym] for sym in symbols])
            portfolio_returns = np.zeros(n_simulations)
            
            for i in range(n_simulations):
                asset_returns = correlated_returns[i, :]
                portfolio_returns[i] = np.dot(weights_array, asset_returns)
            
            # Calculate portfolio values
            portfolio_values = portfolio_value * (1 + portfolio_returns)
            
            # Calculate VaR and CVaR
            var_value = portfolio_value - np.percentile(portfolio_values, (1 - confidence_level) * 100)
            tail_values = portfolio_values[portfolio_values <= portfolio_value - var_value]
            cvar_value = portfolio_value - np.mean(tail_values) if len(tail_values) > 0 else var_value
            
            result = VaRResult(
                var_method=VaRMethod.MONTE_CARLO,
                confidence_level=confidence_level,
                risk_horizon=risk_horizon,
                var_value=abs(var_value),
                var_percentage=abs(var_value) / portfolio_value,
                cvar_value=abs(cvar_value),
                cvar_percentage=abs(cvar_value) / portfolio_value,
                portfolio_value=portfolio_value,
                timestamp=datetime.now(),
                metadata={
                    'simulations': n_simulations,
                    'assets': n_assets
                }
            )
            
            logger.info(f"Monte Carlo VaR calculated: {confidence_level:.1%} = ${abs(var_value):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            raise

    def _generate_correlated_returns(self, symbols: List[str], n_simulations: int,
                                   risk_horizon: int) -> np.ndarray:
        """Generate correlated returns for Monte Carlo simulation"""
        try:
            n_assets = len(symbols)
            
            # Get correlation matrix
            if self.correlation_matrix is not None:
                corr_matrix = self.correlation_matrix.loc[symbols, symbols].values
            else:
                corr_matrix = np.eye(n_assets)
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                # Add small noise for numerical stability
                corr_matrix += np.eye(n_assets) * 1e-6
                L = np.linalg.cholesky(corr_matrix)
            
            # Generate correlated normal random numbers
            uncorrelated_normals = np.random.normal(0, 1, (n_simulations, n_assets))
            correlated_normals = np.dot(uncorrelated_normals, L.T)
            
            # Transform to asset returns using historical parameters
            returns = np.zeros((n_simulations, n_assets))
            
            for i, symbol in enumerate(symbols):
                if symbol in self.asset_returns:
                    asset_returns = self.asset_returns[symbol]
                    mean = asset_returns.mean()
                    std = asset_returns.std()
                    returns[:, i] = mean + std * correlated_normals[:, i]
            
            return returns
            
        except Exception as e:
            logger.error(f"Correlated returns generation failed: {e}")
            # Fallback to independent returns
            returns = np.zeros((n_simulations, n_assets))
            for i, symbol in enumerate(symbols):
                if symbol in self.asset_returns:
                    asset_returns = self.asset_returns[symbol]
                    mean = asset_returns.mean()
                    std = asset_returns.std()
                    returns[:, i] = np.random.normal(mean, std, n_simulations)
            return returns

    def _garch_var(self, portfolio_value: float, confidence_level: float,
                  risk_horizon: int) -> VaRResult:
        """Calculate GARCH VaR"""
        try:
            # This is a simplified GARCH implementation
            # In production, you would use a proper GARCH library like arch
            returns = self.portfolio_returns
            
            # Simple GARCH(1,1) estimation
            omega, alpha, beta = self._estimate_garch_parameters(returns)
            
            # Forecast volatility
            forecast_volatility = self._garch_volatility_forecast(omega, alpha, beta, returns, risk_horizon)
            
            # Calculate VaR
            mean = returns.mean()
            z_score = stats.norm.ppf(1 - confidence_level)
            var_value = portfolio_value * (mean * risk_horizon - z_score * forecast_volatility)
            
            # Calculate CVaR
            pdf_z = stats.norm.pdf(z_score)
            cvar_value = portfolio_value * (mean * risk_horizon - (pdf_z / (1 - confidence_level)) * forecast_volatility)
            
            result = VaRResult(
                var_method=VaRMethod.GARCH,
                confidence_level=confidence_level,
                risk_horizon=risk_horizon,
                var_value=abs(var_value),
                var_percentage=abs(var_value) / portfolio_value,
                cvar_value=abs(cvar_value),
                cvar_percentage=abs(cvar_value) / portfolio_value,
                portfolio_value=portfolio_value,
                timestamp=datetime.now(),
                metadata={
                    'garch_omega': omega,
                    'garch_alpha': alpha,
                    'garch_beta': beta,
                    'forecast_volatility': forecast_volatility
                }
            )
            
            logger.info(f"GARCH VaR calculated: {confidence_level:.1%} = ${abs(var_value):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"GARCH VaR calculation failed: {e}")
            # Fallback to parametric VaR
            return self._parametric_var(portfolio_value, confidence_level, risk_horizon)

    def _estimate_garch_parameters(self, returns: pd.Series) -> Tuple[float, float, float]:
        """Estimate GARCH(1,1) parameters"""
        try:
            # Simplified GARCH parameter estimation
            # In production, use a proper GARCH estimation method
            returns_array = returns.values
            n = len(returns_array)
            
            # Initial parameter estimates
            omega = 0.1
            alpha = 0.1
            beta = 0.8
            
            # Simple optimization (this is simplified)
            def garch_likelihood(params):
                omega, alpha, beta = params
                if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                    return 1e10
                
                # Calculate conditional variances
                variances = np.zeros(n)
                variances[0] = np.var(returns_array)
                
                for t in range(1, n):
                    variances[t] = omega + alpha * returns_array[t-1]**2 + beta * variances[t-1]
                
                # Calculate log-likelihood
                log_likelihood = -0.5 * np.sum(np.log(variances) + (returns_array**2) / variances)
                return -log_likelihood  # Minimize negative log-likelihood
            
            # Constrained optimization
            constraints = [
                {'type': 'ineq', 'fun': lambda x: x[0]},  # omega > 0
                {'type': 'ineq', 'fun': lambda x: x[1]},  # alpha >= 0
                {'type': 'ineq', 'fun': lambda x: x[2]},  # beta >= 0
                {'type': 'ineq', 'fun': lambda x: 0.99 - (x[1] + x[2])}  # alpha + beta < 1
            ]
            
            result = minimize(garch_likelihood, [omega, alpha, beta], 
                            method='SLSQP', constraints=constraints)
            
            if result.success:
                return tuple(result.x)
            else:
                return 0.1, 0.1, 0.8  # Default values
                
        except Exception as e:
            logger.warning(f"GARCH parameter estimation failed: {e}")
            return 0.1, 0.1, 0.8  # Default values

    def _garch_volatility_forecast(self, omega: float, alpha: float, beta: float,
                                 returns: pd.Series, horizon: int) -> float:
        """Forecast volatility using GARCH model"""
        try:
            returns_array = returns.values
            n = len(returns_array)
            
            # Calculate current conditional variance
            current_variance = np.var(returns_array[-100:])  # Use recent variance
            
            # Forecast variance for horizon
            for _ in range(horizon):
                current_variance = omega + alpha * returns_array[-1]**2 + beta * current_variance
            
            return np.sqrt(current_variance)
            
        except Exception as e:
            logger.warning(f"GARCH volatility forecast failed: {e}")
            return returns.std()

    def _expected_shortfall(self, portfolio_value: float, confidence_level: float,
                           risk_horizon: int) -> VaRResult:
        """Calculate Expected Shortfall (CVaR) directly"""
        try:
            # Use historical method for ES calculation
            var_result = self._historical_var(portfolio_value, confidence_level, risk_horizon)
            
            # ES is the same as CVaR in our implementation
            result = VaRResult(
                var_method=VaRMethod.EXPECTED_SHORTFALL,
                confidence_level=confidence_level,
                risk_horizon=risk_horizon,
                var_value=var_result.cvar_value,
                var_percentage=var_result.cvar_percentage,
                cvar_value=var_result.cvar_value,
                cvar_percentage=var_result.cvar_percentage,
                portfolio_value=portfolio_value,
                timestamp=datetime.now(),
                metadata=var_result.metadata
            )
            
            logger.info(f"Expected Shortfall calculated: {confidence_level:.1%} = ${var_result.cvar_value:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Expected Shortfall calculation failed: {e}")
            raise

    def _calculate_component_var(self, portfolio_value: float, confidence_level: float,
                               risk_horizon: int) -> Dict[str, float]:
        """Calculate Component VaR for each asset"""
        try:
            if not self.portfolio_weights or not self.asset_returns:
                return {}
            
            components = {}
            symbols = list(self.portfolio_weights.keys())
            
            # Calculate marginal VaR for each asset
            for symbol in symbols:
                # Calculate portfolio VaR without this asset
                temp_weights = self.portfolio_weights.copy()
                original_weight = temp_weights.pop(symbol)
                
                if temp_weights:
                    # Re-normalize remaining weights
                    total_weight = sum(temp_weights.values())
                    temp_weights = {k: v/total_weight for k, v in temp_weights.items()}
                    
                    # Calculate VaR without this asset
                    temp_returns = self._calculate_portfolio_returns(temp_weights)
                    temp_var = self._historical_var(portfolio_value, confidence_level, risk_horizon, temp_returns)
                    
                    # Component VaR is the difference
                    full_var = self._historical_var(portfolio_value, confidence_level, risk_horizon)
                    component_var = full_var.var_value - temp_var.var_value
                    
                    components[symbol] = component_var
            
            return components
            
        except Exception as e:
            logger.error(f"Component VaR calculation failed: {e}")
            return {}

    def _calculate_portfolio_returns(self, weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns from asset returns and weights"""
        try:
            symbols = list(weights.keys())
            weighted_returns = None
            
            for symbol in symbols:
                if symbol in self.asset_returns:
                    asset_return = self.asset_returns[symbol] * weights[symbol]
                    if weighted_returns is None:
                        weighted_returns = asset_return
                    else:
                        weighted_returns = weighted_returns.add(asset_return, fill_value=0)
            
            return weighted_returns.fillna(0) if weighted_returns is not None else pd.Series()
            
        except Exception as e:
            logger.error(f"Portfolio returns calculation failed: {e}")
            return pd.Series()

    def calculate_comprehensive_var(self, portfolio_value: float = 100000.0) -> Dict[str, VaRResult]:
        """Calculate comprehensive VaR using all methods"""
        try:
            results = {}
            
            for method in VaRMethod:
                try:
                    for confidence in self.config.confidence_levels:
                        for horizon in self.config.risk_horizons:
                            key = f"{method.value}_{confidence}_{horizon}"
                            results[key] = self.calculate_var(
                                portfolio_value, method, confidence, horizon
                            )
                except Exception as e:
                    logger.warning(f"VaR calculation failed for {method.value}: {e}")
                    continue
            
            self.var_results.update(results)
            
            logger.info(f"Comprehensive VaR calculation completed: {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive VaR calculation failed: {e}")
            return {}

    def backtest_var(self, actual_returns: pd.Series, 
                    var_results: Dict[str, VaRResult] = None) -> Dict[str, Any]:
        """Backtest VaR models against actual returns"""
        try:
            if var_results is None:
                var_results = self.var_results
            
            backtest_results = {}
            
            for key, var_result in var_results.items():
                try:
                    # Get VaR threshold
                    var_threshold = -var_result.var_percentage  # Negative for losses
                    
                    # Count exceptions (actual returns worse than VaR)
                    exceptions = actual_returns[actual_returns < var_threshold]
                    n_exceptions = len(exceptions)
                    n_observations = len(actual_returns)
                    
                    # Calculate exception rate
                    exception_rate = n_exceptions / n_observations
                    expected_rate = 1 - var_result.confidence_level
                    
                    # Kupiec test for unconditional coverage
                    kupiec_stat = self._kupiec_test(n_exceptions, n_observations, expected_rate)
                    
                    # Christoffersen test for independence
                    christoffersen_stat = self._christoffersen_test(actual_returns, var_threshold)
                    
                    backtest_results[key] = {
                        'var_method': var_result.var_method.value,
                        'confidence_level': var_result.confidence_level,
                        'risk_horizon': var_result.risk_horizon,
                        'exceptions': n_exceptions,
                        'observations': n_observations,
                        'exception_rate': exception_rate,
                        'expected_rate': expected_rate,
                        'kupiec_statistic': kupiec_stat,
                        'christoffersen_statistic': christoffersen_stat,
                        'var_threshold': var_threshold,
                        'average_exception': exceptions.mean() if n_exceptions > 0 else 0
                    }
                    
                except Exception as e:
                    logger.warning(f"Backtest failed for {key}: {e}")
                    continue
            
            self.backtest_results = backtest_results
            
            logger.info(f"VaR backtesting completed: {len(backtest_results)} models tested")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"VaR backtesting failed: {e}")
            return {}

    def _kupiec_test(self, n_exceptions: int, n_observations: int, expected_rate: float) -> float:
        """Kupiec test for unconditional coverage"""
        try:
            if n_observations == 0:
                return 0.0
            
            actual_rate = n_exceptions / n_observations
            
            # Likelihood ratio test
            if actual_rate == 0:
                lr = -2 * np.log((1 - expected_rate) ** n_observations)
            else:
                lr = -2 * np.log(
                    ((1 - expected_rate) ** (n_observations - n_exceptions)) *
                    (expected_rate ** n_exceptions) /
                    (((1 - actual_rate) ** (n_observations - n_exceptions)) *
                     (actual_rate ** n_exceptions))
                )
            
            return lr
            
        except Exception as e:
            logger.warning(f"Kupiec test failed: {e}")
            return 0.0

    def _christoffersen_test(self, returns: pd.Series, var_threshold: float) -> float:
        """Christoffersen test for independence of exceptions"""
        try:
            # Create exception indicators
            exceptions = (returns < var_threshold).astype(int)
            
            # Count transitions
            n00 = n01 = n10 = n11 = 0
            
            for i in range(1, len(exceptions)):
                if exceptions.iloc[i-1] == 0 and exceptions.iloc[i] == 0:
                    n00 += 1
                elif exceptions.iloc[i-1] == 0 and exceptions.iloc[i] == 1:
                    n01 += 1
                elif exceptions.iloc[i-1] == 1 and exceptions.iloc[i] == 0:
                    n10 += 1
                else:
                    n11 += 1
            
            # Calculate probabilities
            pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            pi = (n01 + n11) / (n00 + n01 + n10 + n11)
            
            # Likelihood ratio test
            lr = -2 * np.log(
                ((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) /
                ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11)
            )
            
            return lr
            
        except Exception as e:
            logger.warning(f"Christoffersen test failed: {e}")
            return 0.0

    def set_risk_limits(self, risk_limits: RiskLimit) -> None:
        """Set risk limits for monitoring"""
        self.risk_limits = risk_limits
        logger.info("Risk limits set successfully")

    def check_risk_limits(self, portfolio_value: float = 100000.0) -> Dict[str, bool]:
        """Check if current VaR exceeds risk limits"""
        try:
            if self.risk_limits is None:
                logger.warning("Risk limits not set")
                return {}
            
            limit_checks = {}
            
            # Calculate current VaR
            var_95 = self.calculate_var(portfolio_value, confidence_level=0.95)
            var_99 = self.calculate_var(portfolio_value, confidence_level=0.99)
            
            # Check limits
            limit_checks['var_95_limit'] = var_95.var_value <= self.risk_limits.var_limit_95
            limit_checks['var_99_limit'] = var_99.var_value <= self.risk_limits.var_limit_99
            limit_checks['cvar_95_limit'] = var_95.cvar_value <= self.risk_limits.cvar_limit_95
            
            # Log violations
            for check, passed in limit_checks.items():
                if not passed:
                    logger.warning(f"Risk limit violated: {check}")
            
            return limit_checks
            
        except Exception as e:
            logger.error(f"Risk limit check failed: {e}")
            return {}

    def generate_risk_report(self, portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            # Calculate comprehensive VaR
            var_results = self.calculate_comprehensive_var(portfolio_value)
            
            # Backtest if we have actual returns
            backtest_results = {}
            if len(self.portfolio_returns) > self.config.min_historical_data:
                backtest_results = self.backtest_var(self.portfolio_returns, var_results)
            
            # Check risk limits
            limit_checks = self.check_risk_limits(portfolio_value)
            
            # Compile report
            report = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'var_results': {
                    key: {
                        'method': result.var_method.value,
                        'confidence': result.confidence_level,
                        'horizon': result.risk_horizon,
                        'var_value': result.var_value,
                        'var_percentage': result.var_percentage,
                        'cvar_value': result.cvar_value,
                        'cvar_percentage': result.cvar_percentage
                    }
                    for key, result in var_results.items()
                },
                'backtest_results': backtest_results,
                'risk_limit_checks': limit_checks,
                'portfolio_statistics': {
                    'mean_return': self.portfolio_returns.mean(),
                    'volatility': self.portfolio_returns.std(),
                    'skewness': self.portfolio_returns.skew(),
                    'kurtosis': self.portfolio_returns.kurtosis(),
                    'sharpe_ratio': self.portfolio_returns.mean() / self.portfolio_returns.std(),
                    'max_drawdown': self._calculate_max_drawdown(self.portfolio_returns)
                }
            }
            
            # Save report if configured
            if self.config.save_reports:
                self._save_risk_report(report)
            
            logger.info("Risk report generated successfully")
            
            return report
            
        except Exception as e:
            logger.error(f"Risk report generation failed: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            return drawdowns.min()
        except Exception as e:
            logger.warning(f"Max drawdown calculation failed: {e}")
            return 0.0

    def _save_risk_report(self, report: Dict[str, Any]) -> None:
        """Save risk report to file"""
        try:
            reports_dir = Path("risk_reports")
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"risk_report_{timestamp}.json"
            
            # Convert non-serializable objects
            serializable_report = json.loads(json.dumps(report, default=str))
            
            with open(report_file, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            logger.info(f"Risk report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Risk report saving failed: {e}")

# Example usage and testing
def main():
    """Example usage of the AdvancedVaRCalculator"""
    
    # Generate sample portfolio data
    print("=== Generating Sample Portfolio Data ===")
    np.random.seed(42)
    
    # Create sample returns
    n_periods = 1000
    base_volatility = 0.01
    
    # Generate correlated returns for multiple assets
    symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
    
    # Create correlation structure
    correlations = np.array([
        [1.0, 0.6, 0.3, 0.4],
        [0.6, 1.0, 0.2, 0.5],
        [0.3, 0.2, 1.0, 0.3],
        [0.4, 0.5, 0.3, 1.0]
    ])
    
    L = np.linalg.cholesky(correlations)
    uncorrelated_returns = np.random.normal(0, base_volatility, (n_periods, len(symbols)))
    correlated_returns = uncorrelated_returns @ L.T
    
    # Create portfolio returns (equal weighted)
    portfolio_returns = pd.Series(correlated_returns.mean(axis=1))
    
    # Create individual asset returns
    asset_returns = {}
    for i, symbol in enumerate(symbols):
        asset_returns[symbol] = pd.Series(correlated_returns[:, i])
    
    # Portfolio weights
    portfolio_weights = {symbol: 0.25 for symbol in symbols}
    
    print(f"Generated {n_periods} return periods for {len(symbols)} assets")
    
    # Configuration
    config = VaRConfig(
        default_method=VaRMethod.PARAMETRIC,
        confidence_levels=[0.95, 0.99],
        risk_horizons=[1, 5, 21],
        historical_lookback=252,
        distribution_type="student_t",
        degrees_freedom=5,
        enable_stress_testing=True,
        enable_backtesting=True
    )
    
    # Initialize VaR calculator
    var_calculator = AdvancedVaRCalculator(config)
    var_calculator.set_portfolio_data(portfolio_returns, asset_returns, portfolio_weights)
    
    # Set risk limits
    risk_limits = RiskLimit(
        var_limit_95=5000,   # $5,000 at 95% confidence
        var_limit_99=8000,   # $8,000 at 99% confidence  
        cvar_limit_95=6000,  # $6,000 at 95% confidence
        max_drawdown_limit=0.15  # 15% maximum drawdown
    )
    var_calculator.set_risk_limits(risk_limits)
    
    portfolio_value = 100000.0
    
    print("\n=== VaR Calculation Examples ===")
    
    # Calculate individual VaR
    print("1. Parametric VaR (95%, 1-day):")
    parametric_var = var_calculator.calculate_var(portfolio_value, VaRMethod.PARAMETRIC, 0.95, 1)
    print(f"   VaR: ${parametric_var.var_value:.2f} ({parametric_var.var_percentage:.2%})")
    print(f"   CVaR: ${parametric_var.cvar_value:.2f} ({parametric_var.cvar_percentage:.2%})")
    
    print("\n2. Historical VaR (99%, 1-day):")
    historical_var = var_calculator.calculate_var(portfolio_value, VaRMethod.HISTORICAL, 0.99, 1)
    print(f"   VaR: ${historical_var.var_value:.2f} ({historical_var.var_percentage:.2%})")
    print(f"   CVaR: ${historical_var.cvar_value:.2f} ({historical_var.cvar_percentage:.2%})")
    
    print("\n3. Monte Carlo VaR (95%, 5-day):")
    mc_var = var_calculator.calculate_var(portfolio_value, VaRMethod.MONTE_CARLO, 0.95, 5)
    print(f"   VaR: ${mc_var.var_value:.2f} ({mc_var.var_percentage:.2%})")
    print(f"   CVaR: ${mc_var.cvar_value:.2f} ({mc_var.cvar_percentage:.2%})")
    
    print("\n=== Comprehensive Risk Analysis ===")
    
    # Generate comprehensive report
    risk_report = var_calculator.generate_risk_report(portfolio_value)
    
    print("Portfolio Statistics:")
    stats = risk_report['portfolio_statistics']
    print(f"  Mean Return: {stats['mean_return']:.4%}")
    print(f"  Volatility: {stats['volatility']:.4%}")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2%}")
    
    print("\nVaR Summary (95%, 1-day):")
    var_key = "parametric_0.95_1"
    if var_key in risk_report['var_results']:
        var_result = risk_report['var_results'][var_key]
        print(f"  Method: {var_result['method']}")
        print(f"  VaR: ${var_result['var_value']:.2f} ({var_result['var_percentage']:.2%})")
        print(f"  CVaR: ${var_result['cvar_value']:.2f} ({var_result['cvar_percentage']:.2%})")
    
    print("\nRisk Limit Checks:")
    for check, passed in risk_report['risk_limit_checks'].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")
    
    print("\n=== Backtesting Results ===")
    if risk_report['backtest_results']:
        for key, result in list(risk_report['backtest_results'].items())[:2]:  # Show first 2
            print(f"  {result['var_method']} ({result['confidence_level']:.1%}):")
            print(f"    Exceptions: {result['exceptions']}/{result['observations']}")
            print(f"    Exception Rate: {result['exception_rate']:.2%}")
            print(f"    Expected Rate: {result['expected_rate']:.2%}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asma)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
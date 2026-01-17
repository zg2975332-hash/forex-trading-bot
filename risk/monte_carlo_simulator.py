"""
Advanced Monte Carlo Simulator for FOREX TRADING BOT
Risk analysis, portfolio simulation, and scenario modeling using Monte Carlo methods
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
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
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SimulationType(Enum):
    PORTFOLIO_RETURNS = "portfolio_returns"
    VALUE_AT_RISK = "value_at_risk"
    MAX_DRAWDOWN = "max_drawdown"
    STRESS_TESTING = "stress_testing"
    OPTION_PRICING = "option_pricing"
    REGIME_SWITCHING = "regime_switching"

class DistributionType(Enum):
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    STUDENT_T = "student_t"
    HISTORICAL = "historical"
    GARCH = "garch"
    MIXTURE = "mixture"

class RiskMetric(Enum):
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation"""
    # Simulation parameters
    simulation_type: SimulationType = SimulationType.PORTFOLIO_RETURNS
    num_simulations: int = 10000
    time_horizon: int = 252  # Trading days
    time_steps: int = 1
    
    # Distribution settings
    distribution: DistributionType = DistributionType.STUDENT_T
    confidence_level: float = 0.95
    degrees_freedom: int = 5  # For Student's t-distribution
    
    # Portfolio settings
    initial_portfolio_value: float = 100000.0
    risk_free_rate: float = 0.02
    
    # Advanced settings
    enable_copula: bool = True
    enable_regime_switching: bool = False
    enable_fat_tails: bool = True
    correlation_method: str = "pearson"
    
    # GARCH settings
    garch_p: int = 1
    garch_q: int = 1
    
    # Mixture model settings
    mixture_components: int = 2
    regime_probabilities: List[float] = field(default_factory=lambda: [0.7, 0.3])
    
    # Performance settings
    chunk_size: int = 1000
    enable_parallel: bool = True

@dataclass
class SimulationResult:
    """Monte Carlo simulation result"""
    simulation_type: SimulationType
    config: SimulationConfig
    results: Dict[str, Any]
    metrics: Dict[str, float]
    percentiles: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskReport:
    """Comprehensive risk analysis report"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    expected_shortfall: float
    portfolio_metrics: Dict[str, float]
    stress_scenarios: Dict[str, float]
    timestamp: datetime

class AdvancedMonteCarloSimulator:
    """
    Advanced Monte Carlo simulator for Forex trading risk analysis and scenario modeling
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # Data storage
        self.historical_returns: Dict[str, pd.Series] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_estimates: Dict[str, float] = {}
        
        # Simulation results
        self.simulation_results: Dict[str, SimulationResult] = {}
        self.risk_reports: Dict[str, RiskReport] = {}
        
        # Statistical models
        self.distribution_params: Dict[str, Dict] = {}
        self.garch_models: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("AdvancedMonteCarloSimulator initialized successfully")

    def set_historical_data(self, prices: Dict[str, pd.Series]) -> None:
        """Set historical price data for simulation"""
        try:
            self.historical_returns = {}
            
            for symbol, price_series in prices.items():
                # Calculate returns
                returns = price_series.pct_change().dropna()
                self.historical_returns[symbol] = returns
                
                # Calculate volatility
                self.volatility_estimates[symbol] = returns.std()
                
                # Fit distribution parameters
                self._fit_distribution_parameters(symbol, returns)
            
            # Calculate correlation matrix
            self._calculate_correlation_matrix()
            
            logger.info(f"Historical data set for {len(prices)} symbols")
            
        except Exception as e:
            logger.error(f"Historical data setting failed: {e}")
            raise

    def _fit_distribution_parameters(self, symbol: str, returns: pd.Series) -> None:
        """Fit distribution parameters for returns"""
        try:
            params = {}
            
            if self.config.distribution == DistributionType.NORMAL:
                params['mean'] = returns.mean()
                params['std'] = returns.std()
                
            elif self.config.distribution == DistributionType.LOGNORMAL:
                log_returns = np.log(1 + returns)
                params['mean'] = log_returns.mean()
                params['std'] = log_returns.std()
                
            elif self.config.distribution == DistributionType.STUDENT_T:
                # Fit Student's t-distribution
                df, loc, scale = stats.t.fit(returns)
                params['df'] = df
                params['loc'] = loc
                params['scale'] = scale
                
            elif self.config.distribution == DistributionType.MIXTURE:
                # Fit Gaussian mixture model
                from sklearn.mixture import GaussianMixture
                returns_reshaped = returns.values.reshape(-1, 1)
                gmm = GaussianMixture(n_components=self.config.mixture_components)
                gmm.fit(returns_reshaped)
                params['weights'] = gmm.weights_
                params['means'] = gmm.means_.flatten()
                params['covariances'] = gmm.covariances_.flatten()
            
            self.distribution_params[symbol] = params
            logger.debug(f"Distribution parameters fitted for {symbol}")
            
        except Exception as e:
            logger.warning(f"Distribution fitting failed for {symbol}: {e}")

    def _calculate_correlation_matrix(self) -> None:
        """Calculate correlation matrix from historical returns"""
        try:
            # Create DataFrame from returns
            returns_data = {}
            for symbol, returns in self.historical_returns.items():
                returns_data[symbol] = returns
            
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            if self.config.correlation_method == "pearson":
                self.correlation_matrix = returns_df.corr()
            elif self.config.correlation_method == "spearman":
                self.correlation_matrix = returns_df.corr(method='spearman')
            elif self.config.correlation_method == "kendall":
                self.correlation_matrix = returns_df.corr(method='kendall')
            else:
                self.correlation_matrix = returns_df.corr()
            
            logger.info("Correlation matrix calculated successfully")
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")

    def simulate_portfolio_returns(self, portfolio_weights: Dict[str, float],
                                 simulation_id: str = None) -> SimulationResult:
        """Simulate portfolio returns using Monte Carlo"""
        try:
            if not self.historical_returns:
                raise ValueError("Historical data not set. Call set_historical_data() first.")
            
            symbols = list(portfolio_weights.keys())
            n_assets = len(symbols)
            n_simulations = self.config.num_simulations
            time_horizon = self.config.time_horizon
            
            # Validate weights
            total_weight = sum(portfolio_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"Portfolio weights sum to {total_weight}, normalizing to 1.0")
                portfolio_weights = {k: v/total_weight for k, v in portfolio_weights.items()}
            
            # Generate correlated random returns
            correlated_returns = self._generate_correlated_returns(symbols, n_simulations, time_horizon)
            
            # Calculate portfolio returns for each simulation
            portfolio_returns = np.zeros((n_simulations, time_horizon))
            weights_array = np.array([portfolio_weights[sym] for sym in symbols])
            
            for i in range(n_simulations):
                for t in range(time_horizon):
                    asset_returns = correlated_returns[i, t, :]
                    portfolio_returns[i, t] = np.dot(weights_array, asset_returns)
            
            # Calculate cumulative returns and portfolio values
            cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
            portfolio_values = self.config.initial_portfolio_value * cumulative_returns
            
            # Calculate performance metrics
            final_values = portfolio_values[:, -1]
            total_returns = (final_values - self.config.initial_portfolio_value) / self.config.initial_portfolio_value
            
            metrics = self._calculate_portfolio_metrics(portfolio_returns, total_returns)
            percentiles = self._calculate_percentiles(final_values, total_returns)
            
            # Create simulation result
            result = SimulationResult(
                simulation_type=SimulationType.PORTFOLIO_RETURNS,
                config=self.config,
                results={
                    'portfolio_returns': portfolio_returns,
                    'portfolio_values': portfolio_values,
                    'cumulative_returns': cumulative_returns,
                    'final_values': final_values,
                    'total_returns': total_returns
                },
                metrics=metrics,
                percentiles=percentiles,
                timestamp=datetime.now(),
                metadata={
                    'portfolio_weights': portfolio_weights,
                    'symbols': symbols,
                    'n_simulations': n_simulations,
                    'time_horizon': time_horizon
                }
            )
            
            # Store result
            sim_id = simulation_id or f"portfolio_sim_{int(datetime.now().timestamp())}"
            self.simulation_results[sim_id] = result
            
            logger.info(f"Portfolio simulation completed: {sim_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio simulation failed: {e}")
            raise

    def _generate_correlated_returns(self, symbols: List[str], n_simulations: int, 
                                   time_horizon: int) -> np.ndarray:
        """Generate correlated random returns for Monte Carlo simulation"""
        n_assets = len(symbols)
        
        try:
            # Get correlation matrix for selected symbols
            if self.correlation_matrix is not None:
                corr_subset = self.correlation_matrix.loc[symbols, symbols].values
            else:
                # Identity matrix if no correlation data
                corr_subset = np.eye(n_assets)
            
            # Cholesky decomposition for correlation
            try:
                L = np.linalg.cholesky(corr_subset)
            except np.linalg.LinAlgError:
                # Add small noise for numerical stability
                corr_subset += np.eye(n_assets) * 1e-6
                L = np.linalg.cholesky(corr_subset)
            
            # Generate correlated random numbers
            if self.config.enable_copula and n_assets > 1:
                correlated_normals = self._generate_copula_returns(n_assets, n_simulations, time_horizon, L)
            else:
                # Standard multivariate normal approach
                uncorrelated_normals = np.random.normal(0, 1, (n_simulations, time_horizon, n_assets))
                correlated_normals = np.dot(uncorrelated_normals, L.T)
            
            # Transform to desired distribution
            returns = np.zeros_like(correlated_normals)
            
            for i, symbol in enumerate(symbols):
                if symbol in self.distribution_params:
                    params = self.distribution_params[symbol]
                    
                    if self.config.distribution == DistributionType.NORMAL:
                        returns[:, :, i] = params['mean'] + params['std'] * correlated_normals[:, :, i]
                        
                    elif self.config.distribution == DistributionType.LOGNORMAL:
                        returns[:, :, i] = np.exp(params['mean'] + params['std'] * correlated_normals[:, :, i]) - 1
                        
                    elif self.config.distribution == DistributionType.STUDENT_T:
                        # Use inverse CDF of t-distribution
                        uniform_values = stats.norm.cdf(correlated_normals[:, :, i])
                        returns[:, :, i] = stats.t.ppf(uniform_values, df=params['df'], 
                                                     loc=params['loc'], scale=params['scale'])
                        
                    elif self.config.distribution == DistributionType.MIXTURE:
                        # Gaussian mixture model
                        returns[:, :, i] = self._generate_mixture_returns(
                            correlated_normals[:, :, i], params, n_simulations * time_horizon)
                
                else:
                    # Fallback to normal distribution
                    mean = self.historical_returns[symbol].mean()
                    std = self.historical_returns[symbol].std()
                    returns[:, :, i] = mean + std * correlated_normals[:, :, i]
            
            return returns
            
        except Exception as e:
            logger.error(f"Correlated returns generation failed: {e}")
            # Fallback to independent returns
            return self._generate_independent_returns(symbols, n_simulations, time_horizon)

    def _generate_copula_returns(self, n_assets: int, n_simulations: int, 
                               time_horizon: int, L: np.ndarray) -> np.ndarray:
        """Generate returns using copula methods for better dependency modeling"""
        try:
            # Gaussian copula approach
            uncorrelated_uniform = np.random.uniform(0, 1, (n_simulations, time_horizon, n_assets))
            correlated_uniform = stats.norm.cdf(np.dot(stats.norm.ppf(uncorrelated_uniform), L.T))
            correlated_normals = stats.norm.ppf(correlated_uniform)
            
            return correlated_normals
            
        except Exception as e:
            logger.warning(f"Copula generation failed: {e}")
            # Fallback to standard method
            uncorrelated_normals = np.random.normal(0, 1, (n_simulations, time_horizon, n_assets))
            return np.dot(uncorrelated_normals, L.T)

    def _generate_mixture_returns(self, correlated_normals: np.ndarray, params: Dict, 
                                n_samples: int) -> np.ndarray:
        """Generate returns from Gaussian mixture model"""
        try:
            weights = params['weights']
            means = params['means']
            covariances = params['covariances']
            
            # Reshape for processing
            flat_normals = correlated_normals.flatten()
            returns = np.zeros_like(flat_normals)
            
            # Assign samples to mixture components
            component_assignments = np.random.choice(len(weights), size=n_samples, p=weights)
            
            for comp_idx in range(len(weights)):
                mask = component_assignments == comp_idx
                n_component = mask.sum()
                
                if n_component > 0:
                    # Transform using component parameters
                    component_std = np.sqrt(covariances[comp_idx])
                    returns[mask] = means[comp_idx] + component_std * flat_normals[mask]
            
            return returns.reshape(correlated_normals.shape)
            
        except Exception as e:
            logger.warning(f"Mixture model generation failed: {e}")
            # Fallback to normal distribution
            return correlated_normals

    def _generate_independent_returns(self, symbols: List[str], n_simulations: int, 
                                    time_horizon: int) -> np.ndarray:
        """Generate independent returns (fallback method)"""
        returns = np.zeros((n_simulations, time_horizon, len(symbols)))
        
        for i, symbol in enumerate(symbols):
            if symbol in self.distribution_params:
                params = self.distribution_params[symbol]
                
                if self.config.distribution == DistributionType.NORMAL:
                    returns[:, :, i] = np.random.normal(params['mean'], params['std'], 
                                                      (n_simulations, time_horizon))
                else:
                    # Fallback to historical bootstrap
                    historical_data = self.historical_returns[symbol].values
                    indices = np.random.randint(0, len(historical_data), (n_simulations, time_horizon))
                    returns[:, :, i] = historical_data[indices]
        
        return returns

    def calculate_value_at_risk(self, portfolio_weights: Dict[str, float],
                              confidence_level: float = None) -> RiskReport:
        """Calculate Value at Risk using Monte Carlo simulation"""
        try:
            confidence_level = confidence_level or self.config.confidence_level
            
            # Run portfolio simulation
            sim_result = self.simulate_portfolio_returns(portfolio_weights)
            final_values = sim_result.results['final_values']
            total_returns = sim_result.results['total_returns']
            
            # Calculate VaR and CVaR
            var_95 = self._calculate_var(final_values, 0.95)
            var_99 = self._calculate_var(final_values, 0.99)
            cvar_95 = self._calculate_cvar(final_values, 0.95)
            cvar_99 = self._calculate_cvar(final_values, 0.99)
            
            # Calculate maximum drawdown from simulations
            portfolio_values = sim_result.results['portfolio_values']
            max_drawdown = self._calculate_simulated_max_drawdown(portfolio_values)
            
            # Calculate expected shortfall
            expected_shortfall = self._calculate_expected_shortfall(final_values)
            
            # Stress testing
            stress_scenarios = self._run_stress_tests(portfolio_weights)
            
            # Create risk report
            risk_report = RiskReport(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                expected_shortfall=expected_shortfall,
                portfolio_metrics=sim_result.metrics,
                stress_scenarios=stress_scenarios,
                timestamp=datetime.now()
            )
            
            # Store report
            report_id = f"risk_report_{int(datetime.now().timestamp())}"
            self.risk_reports[report_id] = risk_report
            
            logger.info(f"VaR calculation completed: 95% VaR = {var_95:.2f}, 99% VaR = {var_99:.2f}")
            
            return risk_report
            
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            raise

    def _calculate_var(self, values: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(values, (1 - confidence_level) * 100)

    def _calculate_cvar(self, values: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(values, confidence_level)
        tail_losses = values[values <= var]
        return np.mean(tail_losses) if len(tail_losses) > 0 else var

    def _calculate_simulated_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown from simulated portfolio values"""
        max_drawdowns = []
        
        for i in range(portfolio_values.shape[0]):
            running_max = np.maximum.accumulate(portfolio_values[i, :])
            drawdowns = (portfolio_values[i, :] - running_max) / running_max
            max_drawdowns.append(np.min(drawdowns))
        
        return np.mean(max_drawdowns)

    def _calculate_expected_shortfall(self, final_values: np.ndarray) -> float:
        """Calculate expected shortfall beyond VaR"""
        var_95 = self._calculate_var(final_values, 0.95)
        tail_losses = final_values[final_values <= var_95]
        return np.mean(tail_losses) if len(tail_losses) > 0 else var_95

    def _run_stress_tests(self, portfolio_weights: Dict[str, float]) -> Dict[str, float]:
        """Run stress testing scenarios"""
        stress_results = {}
        
        try:
            # Market crash scenario (increased volatility)
            crash_config = SimulationConfig(
                num_simulations=5000,
                time_horizon=252,
                distribution=DistributionType.STUDENT_T,
                degrees_freedom=3  # Fatter tails
            )
            
            crash_simulator = AdvancedMonteCarloSimulator(crash_config)
            crash_simulator.set_historical_data(
                {sym: pd.Series(self.historical_returns[sym]) for sym in portfolio_weights.keys()}
            )
            
            crash_result = crash_simulator.simulate_portfolio_returns(portfolio_weights)
            stress_results['market_crash'] = self._calculate_var(
                crash_result.results['final_values'], 0.95
            )
            
            # High correlation scenario
            high_corr_config = SimulationConfig(
                num_simulations=5000,
                time_horizon=252,
                enable_copula=True
            )
            
            high_corr_simulator = AdvancedMonteCarloSimulator(high_corr_config)
            high_corr_simulator.set_historical_data(
                {sym: pd.Series(self.historical_returns[sym]) for sym in portfolio_weights.keys()}
            )
            
            # Modify correlation matrix for stress test
            high_corr_simulator.correlation_matrix = self._create_high_correlation_matrix(
                list(portfolio_weights.keys())
            )
            
            high_corr_result = high_corr_simulator.simulate_portfolio_returns(portfolio_weights)
            stress_results['high_correlation'] = self._calculate_var(
                high_corr_result.results['final_values'], 0.95
            )
            
        except Exception as e:
            logger.warning(f"Stress testing failed: {e}")
        
        return stress_results

    def _create_high_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Create high correlation matrix for stress testing"""
        n_assets = len(symbols)
        high_corr_matrix = np.full((n_assets, n_assets), 0.8)  # High correlation
        np.fill_diagonal(high_corr_matrix, 1.0)
        
        return pd.DataFrame(high_corr_matrix, index=symbols, columns=symbols)

    def _calculate_portfolio_metrics(self, portfolio_returns: np.ndarray, 
                                   total_returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            # Annualized metrics
            annualized_returns = np.mean(total_returns) * 252 / self.config.time_horizon
            annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
            
            # Risk-adjusted ratios
            sharpe_ratio = (annualized_returns - self.config.risk_free_rate) / annualized_volatility
            
            # Sortino ratio (only downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_returns - self.config.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown from simulations
            cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
            running_max = np.maximum.accumulate(cumulative_returns, axis=1)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Calmar ratio
            calmar_ratio = annualized_returns / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Win rate
            positive_returns = np.sum(total_returns > 0) / len(total_returns)
            
            return {
                'annualized_return': annualized_returns,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': positive_returns,
                'skewness': stats.skew(total_returns),
                'kurtosis': stats.kurtosis(total_returns)
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {}

    def _calculate_percentiles(self, final_values: np.ndarray, 
                             total_returns: np.ndarray) -> Dict[str, float]:
        """Calculate percentile statistics"""
        percentiles = {}
        
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[f'value_p{p}'] = np.percentile(final_values, p)
            percentiles[f'return_p{p}'] = np.percentile(total_returns, p)
        
        return percentiles

    def plot_simulation_results(self, simulation_id: str, save_path: str = None):
        """Plot Monte Carlo simulation results"""
        try:
            if simulation_id not in self.simulation_results:
                raise ValueError(f"Simulation ID not found: {simulation_id}")
            
            result = self.simulation_results[simulation_id]
            portfolio_values = result.results['portfolio_values']
            final_values = result.results['final_values']
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Portfolio value paths
            n_paths_to_plot = min(100, portfolio_values.shape[0])
            for i in range(n_paths_to_plot):
                ax1.plot(portfolio_values[i, :], alpha=0.1, color='blue')
            
            # Plot mean path
            mean_path = np.mean(portfolio_values, axis=0)
            ax1.plot(mean_path, color='red', linewidth=2, label='Mean Path')
            
            ax1.set_title('Monte Carlo Simulation Paths')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Portfolio Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Final value distribution
            ax2.hist(final_values, bins=50, density=True, alpha=0.7, color='green')
            ax2.axvline(x=self.config.initial_portfolio_value, color='red', linestyle='--', label='Initial Value')
            ax2.axvline(x=np.mean(final_values), color='blue', linestyle='--', label='Mean Final Value')
            
            ax2.set_title('Final Portfolio Value Distribution')
            ax2.set_xlabel('Portfolio Value')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Drawdown distribution
            max_drawdowns = []
            for i in range(portfolio_values.shape[0]):
                running_max = np.maximum.accumulate(portfolio_values[i, :])
                drawdowns = (portfolio_values[i, :] - running_max) / running_max
                max_drawdowns.append(np.min(drawdowns))
            
            ax3.hist(max_drawdowns, bins=50, density=True, alpha=0.7, color='orange')
            ax3.set_title('Maximum Drawdown Distribution')
            ax3.set_xlabel('Maximum Drawdown')
            ax3.set_ylabel('Density')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Risk-return scatter
            returns = (final_values - self.config.initial_portfolio_value) / self.config.initial_portfolio_value
            volatilities = np.std(portfolio_values, axis=1)
            
            ax4.scatter(volatilities, returns, alpha=0.5, color='purple')
            ax4.set_title('Risk-Return Scatter Plot')
            ax4.set_xlabel('Volatility')
            ax4.set_ylabel('Return')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Simulation plot saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Simulation plotting failed: {e}")

    def optimize_portfolio_weights(self, symbols: List[str], 
                                 objective: str = "sharpe") -> Dict[str, float]:
        """Optimize portfolio weights using Monte Carlo simulation"""
        try:
            def objective_function(weights):
                # Convert weights to dictionary
                weight_dict = {sym: w for sym, w in zip(symbols, weights)}
                
                # Run simulation
                sim_result = self.simulate_portfolio_returns(weight_dict)
                metrics = sim_result.metrics
                
                if objective == "sharpe":
                    return -metrics['sharpe_ratio']  # Minimize negative Sharpe
                elif objective == "sortino":
                    return -metrics['sortino_ratio']  # Minimize negative Sortino
                elif objective == "min_volatility":
                    return metrics['annualized_volatility']
                else:
                    return -metrics['sharpe_ratio']
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum to 1
            ]
            
            # Bounds
            bounds = [(0.0, 1.0) for _ in range(len(symbols))]
            
            # Initial guess (equal weights)
            initial_weights = np.ones(len(symbols)) / len(symbols)
            
            # Optimize
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimized_weights = {sym: w for sym, w in zip(symbols, result.x)}
                
                logger.info(f"Portfolio optimization completed for objective: {objective}")
                
                return optimized_weights
            else:
                logger.warning("Portfolio optimization did not converge")
                return {sym: 1.0/len(symbols) for sym in symbols}
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {sym: 1.0/len(symbols) for sym in symbols}

    def get_simulation_statistics(self, simulation_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a simulation"""
        if simulation_id not in self.simulation_results:
            raise ValueError(f"Simulation ID not found: {simulation_id}")
        
        result = self.simulation_results[simulation_id]
        
        stats = {
            'simulation_id': simulation_id,
            'timestamp': result.timestamp,
            'simulation_type': result.simulation_type.value,
            'metrics': result.metrics,
            'percentiles': result.percentiles,
            'metadata': result.metadata
        }
        
        return stats

# Example usage and testing
def main():
    """Example usage of the AdvancedMonteCarloSimulator"""
    
    # Generate sample price data
    print("=== Generating Sample Price Data ===")
    np.random.seed(42)
    
    symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
    n_periods = 1000
    
    # Create correlated price series
    base_volatility = 0.01
    correlations = np.array([
        [1.0, 0.6, 0.3, 0.4],
        [0.6, 1.0, 0.2, 0.5],
        [0.3, 0.2, 1.0, 0.3],
        [0.4, 0.5, 0.3, 1.0]
    ])
    
    # Generate correlated returns
    L = np.linalg.cholesky(correlations)
    uncorrelated_returns = np.random.normal(0, base_volatility, (n_periods, len(symbols)))
    correlated_returns = uncorrelated_returns @ L.T
    
    # Convert to price series
    prices = {}
    base_prices = [1.1000, 1.3000, 150.00, 0.6500]
    
    for i, symbol in enumerate(symbols):
        price_series = [base_prices[i]]
        for ret in correlated_returns[:, i]:
            new_price = price_series[-1] * (1 + ret)
            price_series.append(new_price)
        
        prices[symbol] = pd.Series(price_series)
    
    print(f"Generated {n_periods} price points for {len(symbols)} symbols")
    
    # Configuration
    config = SimulationConfig(
        num_simulations=10000,
        time_horizon=252,
        distribution=DistributionType.STUDENT_T,
        degrees_freedom=5,
        initial_portfolio_value=100000.0,
        risk_free_rate=0.02,
        enable_copula=True
    )
    
    # Initialize simulator
    simulator = AdvancedMonteCarloSimulator(config)
    simulator.set_historical_data(prices)
    
    # Define portfolio weights
    portfolio_weights = {
        'EUR/USD': 0.3,
        'GBP/USD': 0.25,
        'USD/JPY': 0.25,
        'AUD/USD': 0.2
    }
    
    print("\n=== Running Portfolio Simulation ===")
    
    # Run Monte Carlo simulation
    sim_result = simulator.simulate_portfolio_returns(portfolio_weights)
    
    print("Simulation Metrics:")
    for metric, value in sim_result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nKey Percentiles:")
    for p in [5, 50, 95]:
        value_key = f'value_p{p}'
        return_key = f'return_p{p}'
        print(f"  {p}%: Value = ${sim_result.percentiles[value_key]:.2f}, "
              f"Return = {sim_result.percentiles[return_key]:.2%}")
    
    print("\n=== Risk Analysis ===")
    
    # Calculate Value at Risk
    risk_report = simulator.calculate_value_at_risk(portfolio_weights)
    
    print("Risk Metrics:")
    print(f"  95% VaR: ${risk_report.var_95:.2f}")
    print(f"  99% VaR: ${risk_report.var_99:.2f}")
    print(f"  95% CVaR: ${risk_report.cvar_95:.2f}")
    print(f"  Max Drawdown: {risk_report.max_drawdown:.2%}")
    print(f"  Expected Shortfall: ${risk_report.expected_shortfall:.2f}")
    
    print("\nStress Test Results:")
    for scenario, loss in risk_report.stress_scenarios.items():
        print(f"  {scenario}: ${loss:.2f}")
    
    print("\n=== Portfolio Optimization ===")
    
    # Optimize portfolio weights
    optimized_weights = simulator.optimize_portfolio_weights(symbols, objective="sharpe")
    
    print("Optimized Portfolio Weights:")
    for symbol, weight in optimized_weights.items():
        print(f"  {symbol}: {weight:.3f}")
    
    # Run simulation with optimized weights
    optimized_result = simulator.simulate_portfolio_returns(optimized_weights)
    print(f"Optimized Sharpe Ratio: {optimized_result.metrics['sharpe_ratio']:.4f}")
    
    # Plot results
    try:
        simulator.plot_simulation_results(list(simulator.simulation_results.keys())[0], 
                                        "monte_carlo_simulation.png")
        print("\nSimulation plot saved as 'monte_carlo_simulation.png'")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
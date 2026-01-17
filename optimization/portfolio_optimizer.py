"""
Advanced Portfolio Optimizer for FOREX TRADING BOT
Modern portfolio theory with risk parity, Black-Litterman, and hierarchical risk parity
"""

import logging
import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HRP = "hierarchical_risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MAX_DIVERSIFICATION = "max_diversification"

class RiskMeasure(Enum):
    VARIANCE = "variance"
    VOLATILITY = "volatility"
    CVAR = "cvar"
    VAR = "var"
    MAX_DRAWDOWN = "max_drawdown"

class ConstraintType(Enum):
    WEIGHT_BOUNDS = "weight_bounds"
    SECTOR_NEUTRAL = "sector_neutral"
    LEVERAGE_LIMIT = "leverage_limit"
    TURNOVER_LIMIT = "turnover_limit"
    CONCENTRATION_LIMIT = "concentration_limit"

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_leverage: float = 1.0
    max_turnover: float = 0.5
    max_concentration: float = 0.3
    sector_limits: Dict[str, float] = field(default_factory=dict)
    group_constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class PortfolioResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_contributions: Dict[str, float]
    diversification_ratio: float
    optimization_method: OptimizationMethod
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketData:
    """Market data for portfolio optimization"""
    returns: pd.DataFrame
    prices: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    risk_free_rate: float = 0.02

@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimizer"""
    # Optimization settings
    optimization_method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    risk_measure: RiskMeasure = RiskMeasure.VOLATILITY
    lookback_period: int = 252  # Trading days
    
    # Constraints
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    
    # Black-Litterman settings
    bl_confidence: float = 0.5
    bl_tau: float = 0.05
    
    # Risk parity settings
    risk_budget_method: str = "equal"  # equal, volatility, or custom
    
    # HRP settings
    linkage_method: str = "ward"
    distance_metric: str = "euclidean"
    
    # Regularization
    enable_regularization: bool = True
    l2_regularization: float = 0.01
    
    # Numerical stability
    min_positive_weight: float = 1e-6
    max_iterations: int = 1000

class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization using modern portfolio theory and risk-based approaches
    """
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        self.market_data: Optional[MarketData] = None
        self.view_registry = defaultdict(list)
        
        logger.info("AdvancedPortfolioOptimizer initialized")

    def set_market_data(self, prices: pd.DataFrame, risk_free_rate: float = 0.02) -> None:
        """Set market data for optimization"""
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate covariance matrix
            covariance = returns.cov()
            
            # Calculate correlation matrix
            correlation = returns.corr()
            
            self.market_data = MarketData(
                returns=returns,
                prices=prices,
                covariance=covariance,
                correlation=correlation,
                risk_free_rate=risk_free_rate
            )
            
            logger.info(f"Market data set with {len(returns.columns)} assets and {len(returns)} periods")
            
        except Exception as e:
            logger.error(f"Market data setting failed: {e}")
            raise

    def add_black_litterman_view(self, asset: str, view_return: float, confidence: float = None) -> None:
        """Add Black-Litterman view for an asset"""
        if confidence is None:
            confidence = self.config.bl_confidence
        
        self.view_registry[asset].append({
            'return': view_return,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Added BL view for {asset}: return={view_return:.4f}, confidence={confidence:.2f}")

    def optimize_portfolio(self, method: OptimizationMethod = None) -> PortfolioResult:
        """Optimize portfolio using specified method"""
        if self.market_data is None:
            raise ValueError("Market data not set. Call set_market_data() first.")
        
        method = method or self.config.optimization_method
        
        try:
            if method == OptimizationMethod.MIN_VARIANCE:
                return self._min_variance_optimization()
            elif method == OptimizationMethod.MAX_SHARPE:
                return self._max_sharpe_optimization()
            elif method == OptimizationMethod.RISK_PARITY:
                return self._risk_parity_optimization()
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                return self._black_litterman_optimization()
            elif method == OptimizationMethod.HRP:
                return self._hierarchical_risk_parity_optimization()
            elif method == OptimizationMethod.EQUAL_WEIGHT:
                return self._equal_weight_optimization()
            elif method == OptimizationMethod.MAX_DIVERSIFICATION:
                return self._max_diversification_optimization()
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise

    def _min_variance_optimization(self) -> PortfolioResult:
        """Minimum variance portfolio optimization"""
        n_assets = len(self.market_data.returns.columns)
        
        def objective(weights):
            return self._portfolio_variance(weights)
        
        constraints = self._get_weight_constraints()
        bounds = self._get_weight_bounds(n_assets)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = sco.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Minimum variance optimization did not converge")
        
        return self._create_portfolio_result(result.x, OptimizationMethod.MIN_VARIANCE)

    def _max_sharpe_optimization(self) -> PortfolioResult:
        """Maximum Sharpe ratio portfolio optimization"""
        n_assets = len(self.market_data.returns.columns)
        
        def objective(weights):
            portfolio_return = self._portfolio_return(weights)
            portfolio_volatility = self._portfolio_volatility(weights)
            sharpe = (portfolio_return - self.market_data.risk_free_rate) / portfolio_volatility
            return -sharpe  # Minimize negative Sharpe
        
        constraints = self._get_weight_constraints()
        bounds = self._get_weight_bounds(n_assets)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = sco.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Max Sharpe optimization did not converge")
        
        return self._create_portfolio_result(result.x, OptimizationMethod.MAX_SHARPE)

    def _risk_parity_optimization(self) -> PortfolioResult:
        """Risk parity portfolio optimization"""
        n_assets = len(self.market_data.returns.columns)
        covariance_matrix = self.market_data.covariance.values
        
        def objective(weights):
            portfolio_volatility = self._portfolio_volatility(weights)
            risk_contributions = (weights * (covariance_matrix @ weights)) / portfolio_volatility
            target_contributions = np.ones(n_assets) / n_assets  # Equal risk contribution
            
            # Sum of squared differences from equal risk contribution
            return np.sum((risk_contributions - target_contributions) ** 2)
        
        constraints = self._get_weight_constraints()
        bounds = self._get_weight_bounds(n_assets)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = sco.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Risk parity optimization did not converge")
        
        return self._create_portfolio_result(result.x, OptimizationMethod.RISK_PARITY)

    def _black_litterman_optimization(self) -> PortfolioResult:
        """Black-Litterman portfolio optimization"""
        try:
            # Get equilibrium returns (market implied)
            equilibrium_weights = self._get_market_cap_weights()
            equilibrium_returns = self._implied_equilibrium_returns(equilibrium_weights)
            
            # Create views matrix
            P, Q, Omega = self._create_black_litterman_views()
            
            if P.shape[0] == 0:  # No views
                logger.info("No BL views specified, using equilibrium returns")
                expected_returns = equilibrium_returns
            else:
                # Calculate posterior returns
                tau = self.config.bl_tau
                sigma = self.market_data.covariance.values
                
                # Black-Litterman formula
                first_term = np.linalg.inv(np.linalg.inv(tau * sigma) + P.T @ np.linalg.inv(Omega) @ P)
                second_term = np.linalg.inv(tau * sigma) @ equilibrium_returns + P.T @ np.linalg.inv(Omega) @ Q
                
                expected_returns = first_term @ second_term
            
            # Optimize with new expected returns
            n_assets = len(self.market_data.returns.columns)
            
            def objective(weights):
                portfolio_return = weights @ expected_returns
                portfolio_volatility = self._portfolio_volatility(weights)
                sharpe = (portfolio_return - self.market_data.risk_free_rate) / portfolio_volatility
                return -sharpe
            
            constraints = self._get_weight_constraints()
            bounds = self._get_weight_bounds(n_assets)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = sco.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                logger.warning("Black-Litterman optimization did not converge")
            
            return self._create_portfolio_result(result.x, OptimizationMethod.BLACK_LITTERMAN)
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            # Fallback to max Sharpe
            return self._max_sharpe_optimization()

    def _hierarchical_risk_parity_optimization(self) -> PortfolioResult:
        """Hierarchical Risk Parity portfolio optimization"""
        try:
            # Get correlation matrix and convert to distance matrix
            correlation = self.market_data.correlation.values
            distance_matrix = np.sqrt(2 * (1 - correlation))
            
            # Perform hierarchical clustering
            from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
            linkage_matrix = linkage(distance_matrix, method=self.config.linkage_method)
            
            # Get cluster ordering
            leaves = leaves_list(linkage_matrix)
            
            # Recursive bisection for weight allocation
            weights = self._hrp_recursive_bisection(correlation, leaves)
            
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            return self._create_portfolio_result(weights, OptimizationMethod.HRP)
            
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            # Fallback to risk parity
            return self._risk_parity_optimization()

    def _hrp_recursive_bisection(self, correlation: np.ndarray, leaves: List[int]) -> np.ndarray:
        """Recursive bisection for HRP"""
        n_assets = len(leaves)
        weights = np.ones(n_assets)
        
        if n_assets == 1:
            return weights
        
        # Reorder correlation matrix
        ordered_correlation = correlation[leaves, :][:, leaves]
        
        # Split into two clusters
        split_point = n_assets // 2
        left_cluster = leaves[:split_point]
        right_cluster = leaves[split_point:]
        
        # Calculate variance for each cluster
        left_variance = self._cluster_variance(ordered_correlation[:split_point, :split_point])
        right_variance = self._cluster_variance(ordered_correlation[split_point:, split_point:])
        
        # Allocate weights inversely proportional to variance
        left_weight = right_variance / (left_variance + right_variance)
        right_weight = left_variance / (left_variance + right_variance)
        
        # Recursive allocation
        left_weights = self._hrp_recursive_bisection(correlation, left_cluster)
        right_weights = self._hrp_recursive_bisection(correlation, right_cluster)
        
        # Combine weights
        weights[:split_point] = left_weights * left_weight
        weights[split_point:] = right_weights * right_weight
        
        return weights

    def _cluster_variance(self, cluster_correlation: np.ndarray) -> float:
        """Calculate cluster variance for HRP"""
        n = cluster_correlation.shape[0]
        if n == 1:
            return 1.0
        
        # Use inverse-variance weighting within cluster
        weights = np.ones(n) / n
        variance = weights @ cluster_correlation @ weights
        return variance

    def _equal_weight_optimization(self) -> PortfolioResult:
        """Equal weight portfolio (1/N)"""
        n_assets = len(self.market_data.returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        return self._create_portfolio_result(weights, OptimizationMethod.EQUAL_WEIGHT)

    def _max_diversification_optimization(self) -> PortfolioResult:
        """Maximum diversification portfolio"""
        n_assets = len(self.market_data.returns.columns)
        volatilities = np.sqrt(np.diag(self.market_data.covariance.values))
        
        def objective(weights):
            portfolio_volatility = self._portfolio_volatility(weights)
            weighted_volatility = weights @ volatilities
            diversification_ratio = weighted_volatility / portfolio_volatility
            return -diversification_ratio  # Minimize negative diversification
        
        constraints = self._get_weight_constraints()
        bounds = self._get_weight_bounds(n_assets)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = sco.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            logger.warning("Max diversification optimization did not converge")
        
        return self._create_portfolio_result(result.x, OptimizationMethod.MAX_DIVERSIFICATION)

    def _get_weight_constraints(self) -> List[Dict]:
        """Get weight constraints for optimization"""
        constraints = []
        
        # Sum to 1 constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1.0
        })
        
        # Leverage constraint
        if self.config.constraints.max_leverage != 1.0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self.config.constraints.max_leverage - np.sum(np.abs(x))
            })
        
        # Add sector constraints if specified
        for sector, limit in self.config.constraints.sector_limits.items():
            # This would require mapping assets to sectors
            # For now, it's a placeholder
            pass
        
        return constraints

    def _get_weight_bounds(self, n_assets: int) -> List[Tuple]:
        """Get weight bounds for optimization"""
        min_weight = self.config.constraints.min_weight
        max_weight = self.config.constraints.max_weight
        
        # Apply concentration limit
        if self.config.constraints.max_concentration < 1.0:
            max_weight = min(max_weight, self.config.constraints.max_concentration)
        
        return [(min_weight, max_weight) for _ in range(n_assets)]

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return"""
        expected_returns = self.market_data.returns.mean().values
        return weights @ expected_returns

    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """Calculate portfolio variance"""
        covariance_matrix = self.market_data.covariance.values
        
        if self.config.enable_regularization:
            # Add L2 regularization for numerical stability
            identity_matrix = np.eye(len(weights))
            regularized_covariance = covariance_matrix + self.config.l2_regularization * identity_matrix
            return weights @ regularized_covariance @ weights
        else:
            return weights @ covariance_matrix @ weights

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(self._portfolio_variance(weights))

    def _portfolio_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions for each asset"""
        portfolio_volatility = self._portfolio_volatility(weights)
        covariance_matrix = self.market_data.covariance.values
        marginal_contributions = covariance_matrix @ weights
        risk_contributions = (weights * marginal_contributions) / portfolio_volatility
        return risk_contributions

    def _diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate diversification ratio"""
        weighted_volatility = weights @ np.sqrt(np.diag(self.market_data.covariance.values))
        portfolio_volatility = self._portfolio_volatility(weights)
        return weighted_volatility / portfolio_volatility

    def _create_portfolio_result(self, weights: np.ndarray, method: OptimizationMethod) -> PortfolioResult:
        """Create portfolio result from optimized weights"""
        asset_names = self.market_data.returns.columns.tolist()
        weight_dict = {asset: float(weight) for asset, weight in zip(asset_names, weights)}
        
        # Filter out very small weights
        weight_dict = {k: v for k, v in weight_dict.items() if v > self.config.min_positive_weight}
        
        # Re-normalize
        total_weight = sum(weight_dict.values())
        weight_dict = {k: v/total_weight for k, v in weight_dict.items()}
        
        # Calculate portfolio metrics
        expected_return = self._portfolio_return(weights)
        volatility = self._portfolio_volatility(weights)
        sharpe_ratio = (expected_return - self.market_data.risk_free_rate) / volatility
        risk_contributions = self._portfolio_risk_contributions(weights)
        diversification_ratio = self._diversification_ratio(weights)
        
        risk_contributions_dict = {
            asset: float(contribution) 
            for asset, contribution in zip(asset_names, risk_contributions)
        }
        
        return PortfolioResult(
            weights=weight_dict,
            expected_return=float(expected_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe_ratio),
            risk_contributions=risk_contributions_dict,
            diversification_ratio=float(diversification_ratio),
            optimization_method=method,
            metadata={
                'total_assets': len(weight_dict),
                'effective_assets': 1 / (np.sum(np.array(list(weight_dict.values())) ** 2)),
                'concentration_index': np.sum(np.array(list(weight_dict.values())) ** 2)
            }
        )

    def _get_market_cap_weights(self) -> np.ndarray:
        """Get market capitalization weights (placeholder implementation)"""
        n_assets = len(self.market_data.returns.columns)
        # In real implementation, this would use actual market cap data
        # For now, use equal weights as proxy
        return np.ones(n_assets) / n_assets

    def _implied_equilibrium_returns(self, market_weights: np.ndarray) -> np.ndarray:
        """Calculate implied equilibrium returns"""
        delta = 2.5  # Risk aversion coefficient
        covariance_matrix = self.market_data.covariance.values
        return delta * covariance_matrix @ market_weights

    def _create_black_litterman_views(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create Black-Litterman views matrices"""
        asset_names = self.market_data.returns.columns.tolist()
        n_assets = len(asset_names)
        
        if not self.view_registry:
            return np.zeros((0, n_assets)), np.zeros(0), np.zeros((0, 0))
        
        # Create views matrix P and view returns Q
        P_list = []
        Q_list = []
        confidence_list = []
        
        for asset, views in self.view_registry.items():
            if asset in asset_names:
                asset_index = asset_names.index(asset)
                
                for view in views:
                    # Create view vector
                    p_vector = np.zeros(n_assets)
                    p_vector[asset_index] = 1.0
                    
                    P_list.append(p_vector)
                    Q_list.append(view['return'])
                    confidence_list.append(view['confidence'])
        
        if not P_list:
            return np.zeros((0, n_assets)), np.zeros(0), np.zeros((0, 0))
        
        P = np.array(P_list)
        Q = np.array(Q_list)
        
        # Create uncertainty matrix Omega (diagonal)
        Omega = np.diag([1.0 / (conf ** 2) if conf > 0 else 1e6 for conf in confidence_list])
        
        return P, Q, Omega

    def calculate_efficient_frontier(self, points: int = 50) -> pd.DataFrame:
        """Calculate efficient frontier"""
        if self.market_data is None:
            raise ValueError("Market data not set")
        
        n_assets = len(self.market_data.returns.columns)
        expected_returns = self.market_data.returns.mean().values
        
        # Find minimum and maximum expected returns
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        target_returns = np.linspace(min_return, max_return, points)
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                constraints = self._get_weight_constraints()
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x: self._portfolio_return(x) - target_return
                })
                
                bounds = self._get_weight_bounds(n_assets)
                initial_weights = np.array([1/n_assets] * n_assets)
                
                result = sco.minimize(
                    self._portfolio_variance,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    volatility = self._portfolio_volatility(result.x)
                    sharpe = (target_return - self.market_data.risk_free_rate) / volatility
                    
                    efficient_portfolios.append({
                        'return': target_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe,
                        'weights': result.x
                    })
                    
            except Exception as e:
                logger.warning(f"Efficient frontier point failed for return {target_return}: {e}")
                continue
        
        return pd.DataFrame(efficient_portfolios)

    def get_portfolio_metrics(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        try:
            # Convert weight dict to array
            asset_names = self.market_data.returns.columns.tolist()
            weight_array = np.array([weights.get(asset, 0.0) for asset in asset_names])
            
            # Basic metrics
            expected_return = self._portfolio_return(weight_array)
            volatility = self._portfolio_volatility(weight_array)
            sharpe_ratio = (expected_return - self.market_data.risk_free_rate) / volatility
            
            # Risk metrics
            risk_contributions = self._portfolio_risk_contributions(weight_array)
            diversification_ratio = self._diversification_ratio(weight_array)
            
            # Drawdown analysis (simplified)
            portfolio_returns = self.market_data.returns @ weight_array
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Concentration metrics
            weight_values = np.array(list(weights.values()))
            herfindahl = np.sum(weight_values ** 2)
            effective_n = 1 / herfindahl
            
            return {
                'expected_return_annual': float(expected_return * 252),
                'volatility_annual': float(volatility * np.sqrt(252)),
                'sharpe_ratio_annual': float(sharpe_ratio * np.sqrt(252)),
                'max_drawdown': float(max_drawdown),
                'diversification_ratio': float(diversification_ratio),
                'concentration_index': float(herfindahl),
                'effective_number_assets': float(effective_n),
                'tail_risk_ratio': float(-max_drawdown / (volatility * np.sqrt(252))),
                'risk_contribution_std': float(np.std(risk_contributions))
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {}

    def compare_optimization_methods(self) -> Dict[str, PortfolioResult]:
        """Compare all optimization methods"""
        methods = [
            OptimizationMethod.MIN_VARIANCE,
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.EQUAL_WEIGHT,
            OptimizationMethod.MAX_DIVERSIFICATION
        ]
        
        results = {}
        
        for method in methods:
            try:
                result = self.optimize_portfolio(method)
                results[method.value] = result
                logger.info(f"Completed {method.value} optimization")
            except Exception as e:
                logger.error(f"Optimization failed for {method.value}: {e}")
        
        return results

# Example usage and testing
def main():
    """Example usage of the AdvancedPortfolioOptimizer"""
    
    # Generate sample market data
    np.random.seed(42)
    n_assets = 8
    n_periods = 1000
    
    # Create correlated returns
    means = np.random.uniform(0.05, 0.15, n_assets)
    stds = np.random.uniform(0.1, 0.3, n_assets)
    
    # Generate correlation matrix
    corr_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Convert to covariance matrix
    cov_matrix = np.outer(stds, stds) * corr_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(means, cov_matrix, n_periods)
    
    # Create DataFrame
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    # Create price data (cumulative returns from 100)
    prices_df = 100 * (1 + returns_df).cumprod()
    
    # Configuration
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.3,
        max_leverage=1.0,
        max_concentration=0.25
    )
    
    config = PortfolioConfig(
        optimization_method=OptimizationMethod.MAX_SHARPE,
        constraints=constraints,
        lookback_period=252,
        enable_regularization=True
    )
    
    # Initialize optimizer
    optimizer = AdvancedPortfolioOptimizer(config)
    optimizer.set_market_data(prices_df, risk_free_rate=0.03)
    
    # Add some Black-Litterman views
    optimizer.add_black_litterman_view("ASSET_0", 0.12, 0.7)
    optimizer.add_black_litterman_view("ASSET_1", 0.08, 0.6)
    
    print("=== Portfolio Optimization Demo ===")
    
    # Test different optimization methods
    methods_to_test = [
        OptimizationMethod.MIN_VARIANCE,
        OptimizationMethod.MAX_SHARPE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.BLACK_LITTERMAN,
        OptimizationMethod.EQUAL_WEIGHT
    ]
    
    results = {}
    
    for method in methods_to_test:
        print(f"\n--- {method.value.upper()} ---")
        try:
            result = optimizer.optimize_portfolio(method)
            results[method.value] = result
            
            print(f"Expected Return: {result.expected_return:.4f}")
            print(f"Volatility: {result.volatility:.4f}")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
            print(f"Diversification Ratio: {result.diversification_ratio:.4f}")
            print(f"Top 3 Weights:")
            
            sorted_weights = sorted(result.weights.items(), key=lambda x: x[1], reverse=True)[:3]
            for asset, weight in sorted_weights:
                risk_contribution = result.risk_contributions.get(asset, 0)
                print(f"  {asset}: {weight:.3f} (Risk: {risk_contribution:.3f})")
                
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    # Compare methods
    print("\n=== Method Comparison ===")
    comparison_data = []
    
    for method_name, result in results.items():
        metrics = optimizer.get_portfolio_metrics(result.weights)
        comparison_data.append({
            'Method': method_name,
            'Return': metrics.get('expected_return_annual', 0),
            'Volatility': metrics.get('volatility_annual', 0),
            'Sharpe': metrics.get('sharpe_ratio_annual', 0),
            'Max DD': metrics.get('max_drawdown', 0),
            'Diversification': metrics.get('diversification_ratio', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    # Calculate efficient frontier
    print("\n=== Efficient Frontier ===")
    try:
        frontier = optimizer.calculate_efficient_frontier(points=20)
        if not frontier.empty:
            max_sharpe_point = frontier.loc[frontier['sharpe_ratio'].idxmax()]
            print(f"Maximum Sharpe Portfolio:")
            print(f"  Return: {max_sharpe_point['return']:.4f}")
            print(f"  Volatility: {max_sharpe_point['volatility']:.4f}")
            print(f"  Sharpe: {max_sharpe_point['sharpe_ratio']:.4f}")
    except Exception as e:
        print(f"Efficient frontier calculation failed: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
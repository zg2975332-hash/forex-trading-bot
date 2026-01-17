"""
Advanced Correlation Manager for FOREX TRADING BOT
Real-time correlation analysis, regime detection, and risk management
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
import time
from scipy import stats
import scipy.optimize as sco
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import json
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CorrelationMethod(Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING = "rolling"
    EXPONENTIAL = "exponential"

class CorrelationRegime(Enum):
    NORMAL = "normal"
    HIGH_CORRELATION = "high_correlation"
    DECORRELATION = "decorrelation"
    FLIGHT_TO_SAFETY = "flight_to_safety"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"

class ClusterMethod(Enum):
    HIERARCHICAL = "hierarchical"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    SPECTRAL = "spectral"

@dataclass
class CorrelationMatrix:
    """Correlation matrix with metadata"""
    matrix: pd.DataFrame
    timestamp: datetime
    method: CorrelationMethod
    lookback_period: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelationAlert:
    """Correlation-based alert"""
    alert_id: str
    alert_type: str
    symbol_pair: str
    current_correlation: float
    threshold: float
    direction: str
    confidence: float
    timestamp: datetime
    recommendation: str

@dataclass
class ClusterResult:
    """Clustering result"""
    clusters: Dict[int, List[str]]
    linkage_matrix: np.ndarray
    distances: Dict[str, float]
    silhouette_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrelationConfig:
    """Configuration for correlation manager"""
    # Correlation calculation
    default_method: CorrelationMethod = CorrelationMethod.PEARSON
    lookback_periods: List[int] = field(default_factory=lambda: [20, 60, 120])
    min_data_points: int = 10
    
    # Alert thresholds
    high_correlation_threshold: float = 0.7
    low_correlation_threshold: float = 0.2
    significant_change_threshold: float = 0.3
    volatility_threshold: float = 0.1
    
    # Clustering settings
    cluster_method: ClusterMethod = ClusterMethod.HIERARCHICAL
    max_clusters: int = 5
    distance_threshold: float = 0.5
    
    # Risk management
    max_correlation_exposure: float = 0.8
    diversification_target: float = 0.3
    regime_change_sensitivity: float = 0.7
    
    # Real-time processing
    update_frequency: int = 300  # seconds
    rolling_window: int = 50
    decay_factor: float = 0.94  # For exponential weighting
    
    # Data management
    max_history_days: int = 365
    save_correlations: bool = True

class AdvancedCorrelationManager:
    """
    Advanced correlation analysis and management for Forex trading
    """
    
    def __init__(self, config: CorrelationConfig = None):
        self.config = config or CorrelationConfig()
        
        # Data storage
        self.price_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.return_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.correlation_matrices: Dict[str, CorrelationMatrix] = {}
        self.correlation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.regime_history: deque = deque(maxlen=200)
        
        # Clustering results
        self.cluster_results: Optional[ClusterResult] = None
        self.asset_clusters: Dict[str, int] = {}
        
        # Alert system
        self.active_alerts: Dict[str, CorrelationAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Statistical baselines
        self.correlation_baselines: Dict[str, float] = {}
        self.volatility_estimates: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("AdvancedCorrelationManager initialized successfully")

    def _start_background_tasks(self):
        """Start background correlation monitoring tasks"""
        # Correlation update loop
        update_thread = threading.Thread(target=self._correlation_update_loop, daemon=True)
        update_thread.start()
        
        # Regime detection loop
        regime_thread = threading.Thread(target=self._regime_detection_loop, daemon=True)
        regime_thread.start()
        
        # Clustering update loop
        cluster_thread = threading.Thread(target=self._clustering_update_loop, daemon=True)
        cluster_thread.start()
        
        # Data cleanup loop
        cleanup_thread = threading.Thread(target=self._data_cleanup_loop, daemon=True)
        cleanup_thread.start()

    def update_price_data(self, symbol: str, price: float, timestamp: datetime = None):
        """Update price data for a symbol"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            with self._lock:
                self.price_data[symbol].append({
                    'timestamp': timestamp,
                    'price': price
                })
                
                # Calculate returns if we have enough data
                if len(self.price_data[symbol]) > 1:
                    prev_price = self.price_data[symbol][-2]['price']
                    if prev_price != 0:
                        returns = (price - prev_price) / prev_price
                        self.return_data[symbol].append({
                            'timestamp': timestamp,
                            'return': returns
                        })
            
            logger.debug(f"Price data updated for {symbol}: {price}")
            
        except Exception as e:
            logger.error(f"Price data update failed for {symbol}: {e}")

    def calculate_correlation_matrix(self, symbols: List[str] = None, 
                                  method: CorrelationMethod = None,
                                  lookback: int = None) -> CorrelationMatrix:
        """Calculate correlation matrix for given symbols"""
        try:
            if symbols is None:
                symbols = list(self.return_data.keys())
            
            if len(symbols) < 2:
                raise ValueError("At least 2 symbols required for correlation calculation")
            
            method = method or self.config.default_method
            lookback = lookback or self.config.lookback_periods[0]
            
            # Prepare return data
            returns_df = self._prepare_returns_data(symbols, lookback)
            
            if returns_df.empty or len(returns_df) < self.config.min_data_points:
                raise ValueError("Insufficient data for correlation calculation")
            
            # Calculate correlation matrix
            if method == CorrelationMethod.PEARSON:
                corr_matrix = returns_df.corr(method='pearson')
            elif method == CorrelationMethod.SPEARMAN:
                corr_matrix = returns_df.corr(method='spearman')
            elif method == CorrelationMethod.KENDALL:
                corr_matrix = returns_df.corr(method='kendall')
            elif method == CorrelationMethod.ROLLING:
                corr_matrix = returns_df.rolling(window=self.config.rolling_window).corr().iloc[-1]
            elif method == CorrelationMethod.EXPONENTIAL:
                corr_matrix = self._exponential_correlation(returns_df)
            else:
                corr_matrix = returns_df.corr()
            
            # Create correlation matrix object
            correlation_matrix = CorrelationMatrix(
                matrix=corr_matrix,
                timestamp=datetime.now(),
                method=method,
                lookback_period=lookback,
                metadata={
                    'symbols': symbols,
                    'data_points': len(returns_df),
                    'completeness_ratio': self._calculate_completeness_ratio(returns_df)
                }
            )
            
            # Store correlation matrix
            matrix_key = f"{method.value}_{lookback}"
            self.correlation_matrices[matrix_key] = correlation_matrix
            
            # Update correlation history
            self._update_correlation_history(correlation_matrix)
            
            logger.info(f"Correlation matrix calculated: {method.value}, {lookback} periods, {len(symbols)} symbols")
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            raise

    def _prepare_returns_data(self, symbols: List[str], lookback: int) -> pd.DataFrame:
        """Prepare returns data for correlation calculation"""
        returns_data = {}
        
        for symbol in symbols:
            if symbol in self.return_data:
                # Get recent returns
                recent_returns = list(self.return_data[symbol])[-lookback:]
                if recent_returns:
                    returns = [r['return'] for r in recent_returns]
                    returns_data[symbol] = returns
        
        # Create DataFrame and handle missing data
        max_length = max(len(returns) for returns in returns_data.values()) if returns_data else 0
        
        if max_length == 0:
            return pd.DataFrame()
        
        # Align return series
        aligned_returns = {}
        for symbol, returns in returns_data.items():
            if len(returns) < max_length:
                # Pad with NaN for shorter series
                padded_returns = [np.nan] * (max_length - len(returns)) + returns
                aligned_returns[symbol] = padded_returns
            else:
                aligned_returns[symbol] = returns[-max_length:]
        
        df = pd.DataFrame(aligned_returns)
        
        # Forward fill and drop remaining NaN
        df = df.ffill().dropna()
        
        return df

    def _exponential_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate exponential weighted correlation"""
        try:
            # Calculate exponential weights
            n_periods = len(returns_df)
            weights = [self.config.decay_factor ** i for i in range(n_periods-1, -1, -1)]
            weights = np.array(weights) / sum(weights)
            
            # Weighted correlation calculation
            weighted_returns = returns_df.multiply(np.sqrt(weights), axis=0)
            corr_matrix = weighted_returns.corr()
            
            return corr_matrix
            
        except Exception as e:
            logger.warning(f"Exponential correlation failed, using Pearson: {e}")
            return returns_df.corr()

    def _calculate_completeness_ratio(self, df: pd.DataFrame) -> float:
        """Calculate data completeness ratio"""
        total_cells = df.shape[0] * df.shape[1]
        non_nan_cells = df.count().sum()
        return non_nan_cells / total_cells if total_cells > 0 else 0.0

    def _update_correlation_history(self, corr_matrix: CorrelationMatrix):
        """Update correlation history for pairs"""
        symbols = corr_matrix.matrix.columns.tolist()
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:  # Avoid duplicates and self-correlation
                    pair = f"{sym1}_{sym2}"
                    correlation = corr_matrix.matrix.iloc[i, j]
                    
                    self.correlation_history[pair].append({
                        'timestamp': corr_matrix.timestamp,
                        'correlation': correlation,
                        'method': corr_matrix.method.value,
                        'lookback': corr_matrix.lookback_period
                    })

    def get_correlation(self, symbol1: str, symbol2: str, 
                       method: CorrelationMethod = None,
                       lookback: int = None) -> float:
        """Get correlation between two symbols"""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            # Calculate correlation matrix if not available
            matrix_key = f"{method.value if method else self.config.default_method.value}_{lookback or self.config.lookback_periods[0]}"
            
            if matrix_key not in self.correlation_matrices:
                self.calculate_correlation_matrix([symbol1, symbol2], method, lookback)
            
            corr_matrix = self.correlation_matrices[matrix_key].matrix
            
            if symbol1 in corr_matrix.columns and symbol2 in corr_matrix.columns:
                return float(corr_matrix.loc[symbol1, symbol2])
            else:
                logger.warning(f"Symbols not found in correlation matrix: {symbol1}, {symbol2}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Correlation retrieval failed for {symbol1}-{symbol2}: {e}")
            return 0.0

    def detect_correlation_regime(self) -> CorrelationRegime:
        """Detect current correlation regime"""
        try:
            # Get current correlation matrix
            corr_matrix = self.calculate_correlation_matrix()
            symbols = corr_matrix.matrix.columns.tolist()
            
            if len(symbols) < 3:
                return CorrelationRegime.NORMAL
            
            # Calculate regime metrics
            avg_correlation = self._average_correlation(corr_matrix.matrix)
            correlation_volatility = self._correlation_volatility(corr_matrix.matrix)
            risk_appetite = self._calculate_risk_appetite(symbols)
            
            # Determine regime
            if avg_correlation > self.config.high_correlation_threshold:
                if risk_appetite > 0.6:
                    regime = CorrelationRegime.RISK_ON
                elif risk_appetite < 0.4:
                    regime = CorrelationRegime.FLIGHT_TO_SAFETY
                else:
                    regime = CorrelationRegime.HIGH_CORRELATION
                    
            elif avg_correlation < self.config.low_correlation_threshold:
                regime = CorrelationRegime.DECORRELATION
                
            elif correlation_volatility > self.config.volatility_threshold:
                regime = CorrelationRegime.DECORRELATION
                
            else:
                regime = CorrelationRegime.NORMAL
            
            # Store regime history
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'avg_correlation': avg_correlation,
                'correlation_volatility': correlation_volatility,
                'risk_appetite': risk_appetite
            })
            
            logger.info(f"Correlation regime detected: {regime.value}")
            
            return regime
            
        except Exception as e:
            logger.error(f"Correlation regime detection failed: {e}")
            return CorrelationRegime.NORMAL

    def _average_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average correlation (excluding diagonal)"""
        matrix_values = corr_matrix.values
        n = len(matrix_values)
        
        if n <= 1:
            return 0.0
        
        # Exclude diagonal and get upper triangle
        total_correlation = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                total_correlation += abs(matrix_values[i, j])  # Use absolute value
                count += 1
        
        return total_correlation / count if count > 0 else 0.0

    def _correlation_volatility(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate volatility of correlations"""
        matrix_values = corr_matrix.values
        n = len(matrix_values)
        
        if n <= 1:
            return 0.0
        
        correlations = []
        for i in range(n):
            for j in range(i+1, n):
                correlations.append(matrix_values[i, j])
        
        return np.std(correlations) if correlations else 0.0

    def _calculate_risk_appetite(self, symbols: List[str]) -> float:
        """Calculate market risk appetite (simplified)"""
        try:
            # This would typically use volatility indices, currency strengths, etc.
            # For now, use a simplified approach based on USD pairs
            usd_pairs = [s for s in symbols if 'USD' in s]
            
            if not usd_pairs:
                return 0.5
            
            # Calculate average strength (simplified)
            strengths = []
            for pair in usd_pairs:
                if pair.startswith('USD'):
                    # USD is base currency - higher price means stronger USD
                    if self.price_data[pair]:
                        current_price = self.price_data[pair][-1]['price']
                        # Normalize to 0-1 range (this is simplified)
                        strength = min(1.0, current_price / 2.0)  # Assuming major pairs < 2.0
                        strengths.append(strength)
                else:
                    # USD is quote currency - lower price means stronger USD
                    if self.price_data[pair]:
                        current_price = self.price_data[pair][-1]['price']
                        strength = min(1.0, (2.0 - current_price) / 2.0)  # Inverse relationship
                        strengths.append(strength)
            
            avg_strength = np.mean(strengths) if strengths else 0.5
            
            # Convert to risk appetite (strong USD often indicates risk-off)
            risk_appetite = 1.0 - avg_strength
            
            return float(risk_appetite)
            
        except Exception as e:
            logger.warning(f"Risk appetite calculation failed: {e}")
            return 0.5

    def perform_clustering(self, symbols: List[str] = None) -> ClusterResult:
        """Perform hierarchical clustering on correlation matrix"""
        try:
            if symbols is None:
                symbols = list(self.return_data.keys())
            
            if len(symbols) < 3:
                raise ValueError("At least 3 symbols required for clustering")
            
            # Get correlation matrix
            corr_matrix = self.calculate_correlation_matrix(symbols)
            correlation_values = corr_matrix.matrix.values
            
            # Convert correlation to distance (1 - |correlation|)
            distance_matrix = 1 - np.abs(correlation_values)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            
            # Determine optimal number of clusters
            optimal_clusters = self._find_optimal_clusters(linkage_matrix, distance_matrix)
            
            # Form clusters
            clusters = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')
            
            # Organize clusters
            cluster_groups = defaultdict(list)
            for i, symbol in enumerate(symbols):
                cluster_groups[clusters[i]].append(symbol)
                self.asset_clusters[symbol] = clusters[i]
            
            # Calculate silhouette score
            silhouette_avg = self._calculate_silhouette_score(distance_matrix, clusters)
            
            # Calculate cluster distances
            cluster_distances = self._calculate_cluster_distances(cluster_groups, corr_matrix.matrix)
            
            # Create cluster result
            self.cluster_results = ClusterResult(
                clusters=dict(cluster_groups),
                linkage_matrix=linkage_matrix,
                distances=cluster_distances,
                silhouette_score=silhouette_avg,
                metadata={
                    'symbols': symbols,
                    'optimal_clusters': optimal_clusters,
                    'clustering_method': 'hierarchical'
                }
            )
            
            logger.info(f"Clustering completed: {optimal_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
            
            return self.cluster_results
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise

    def _find_optimal_clusters(self, linkage_matrix: np.ndarray, distance_matrix: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis"""
        max_clusters = min(self.config.max_clusters, len(linkage_matrix) + 1)
        
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            clusters = fcluster(linkage_matrix, k, criterion='maxclust')
            score = self._calculate_silhouette_score(distance_matrix, clusters)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k

    def _calculate_silhouette_score(self, distance_matrix: np.ndarray, clusters: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        try:
            n = len(clusters)
            if n <= 1:
                return 0.0
            
            # Calculate silhouette scores manually
            silhouette_scores = []
            
            for i in range(n):
                # Average distance to points in same cluster
                same_cluster_indices = np.where(clusters == clusters[i])[0]
                if len(same_cluster_indices) > 1:
                    a_i = np.mean([distance_matrix[i, j] for j in same_cluster_indices if j != i])
                else:
                    a_i = 0
                
                # Average distance to points in nearest other cluster
                b_i = float('inf')
                for cluster_id in set(clusters):
                    if cluster_id != clusters[i]:
                        other_cluster_indices = np.where(clusters == cluster_id)[0]
                        if other_cluster_indices.size > 0:
                            distance = np.mean([distance_matrix[i, j] for j in other_cluster_indices])
                            b_i = min(b_i, distance)
                
                if a_i == 0 and b_i == 0:
                    silhouette_scores.append(0.0)
                else:
                    silhouette_scores.append((b_i - a_i) / max(a_i, b_i))
            
            return np.mean(silhouette_scores) if silhouette_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Silhouette score calculation failed: {e}")
            return 0.0

    def _calculate_cluster_distances(self, clusters: Dict[int, List[str]], corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate distances between clusters"""
        cluster_distances = {}
        cluster_ids = list(clusters.keys())
        
        for i, cluster1 in enumerate(cluster_ids):
            for j, cluster2 in enumerate(cluster_ids):
                if i < j:
                    key = f"cluster_{cluster1}_cluster_{cluster2}"
                    
                    # Calculate average correlation between clusters
                    correlations = []
                    for sym1 in clusters[cluster1]:
                        for sym2 in clusters[cluster2]:
                            if sym1 in corr_matrix.columns and sym2 in corr_matrix.columns:
                                correlations.append(corr_matrix.loc[sym1, sym2])
                    
                    avg_correlation = np.mean(correlations) if correlations else 0.0
                    distance = 1 - abs(avg_correlation)  # Convert to distance
                    
                    cluster_distances[key] = distance
        
        return cluster_distances

    def check_correlation_alerts(self) -> List[CorrelationAlert]:
        """Check for correlation-based alerts"""
        alerts = []
        
        try:
            # Get current correlation matrix
            corr_matrix = self.calculate_correlation_matrix()
            symbols = corr_matrix.matrix.columns.tolist()
            
            # Check for high correlations
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i < j:
                        correlation = corr_matrix.matrix.iloc[i, j]
                        pair = f"{sym1}_{sym2}"
                        
                        # High correlation alert
                        if abs(correlation) > self.config.high_correlation_threshold:
                            alert = CorrelationAlert(
                                alert_id=f"high_corr_{pair}_{int(time.time())}",
                                alert_type="HIGH_CORRELATION",
                                symbol_pair=pair,
                                current_correlation=correlation,
                                threshold=self.config.high_correlation_threshold,
                                direction="positive" if correlation > 0 else "negative",
                                confidence=min(0.95, abs(correlation)),
                                timestamp=datetime.now(),
                                recommendation=f"Consider reducing exposure to {pair} due to high correlation"
                            )
                            alerts.append(alert)
                        
                        # Significant change alert
                        historical_corr = self._get_historical_correlation(pair)
                        if historical_corr is not None:
                            change = abs(correlation - historical_corr)
                            if change > self.config.significant_change_threshold:
                                alert = CorrelationAlert(
                                    alert_id=f"corr_change_{pair}_{int(time.time())}",
                                    alert_type="CORRELATION_CHANGE",
                                    symbol_pair=pair,
                                    current_correlation=correlation,
                                    threshold=self.config.significant_change_threshold,
                                    direction="increasing" if correlation > historical_corr else "decreasing",
                                    confidence=min(0.9, change),
                                    timestamp=datetime.now(),
                                    recommendation=f"Monitor {pair} for correlation regime change"
                                )
                                alerts.append(alert)
            
            # Update active alerts
            for alert in alerts:
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
            
            if alerts:
                logger.info(f"Generated {len(alerts)} correlation alerts")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Correlation alert check failed: {e}")
            return []

    def _get_historical_correlation(self, pair: str, lookback: int = 20) -> Optional[float]:
        """Get historical correlation for a pair"""
        try:
            if pair in self.correlation_history and len(self.correlation_history[pair]) > lookback:
                historical_correlations = [h['correlation'] for h in list(self.correlation_history[pair])[-lookback:]]
                return np.mean(historical_correlations)
            return None
        except Exception as e:
            logger.warning(f"Historical correlation retrieval failed for {pair}: {e}")
            return None

    def get_diversification_score(self, portfolio_weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification score"""
        try:
            if not portfolio_weights:
                return 0.0
            
            symbols = list(portfolio_weights.keys())
            
            if len(symbols) < 2:
                return 0.0
            
            # Get correlation matrix
            corr_matrix = self.calculate_correlation_matrix(symbols)
            correlation_matrix = corr_matrix.matrix.values
            
            # Convert weights to array
            weights = np.array([portfolio_weights[sym] for sym in symbols])
            
            # Calculate portfolio variance
            portfolio_variance = weights @ correlation_matrix @ weights
            
            # Calculate weighted average variance
            individual_variances = np.diag(correlation_matrix)
            weighted_variance = weights @ individual_variances
            
            # Diversification ratio
            if weighted_variance > 0:
                diversification_ratio = weighted_variance / portfolio_variance
            else:
                diversification_ratio = 1.0
            
            # Normalize to 0-1 scale
            diversification_score = min(1.0, diversification_ratio / 3.0)  # Max theoretical is ~n
            
            return float(diversification_score)
            
        except Exception as e:
            logger.error(f"Diversification score calculation failed: {e}")
            return 0.0

    def optimize_for_diversification(self, symbols: List[str], 
                                   target_score: float = None) -> Dict[str, float]:
        """Optimize portfolio weights for maximum diversification"""
        try:
            if target_score is None:
                target_score = self.config.diversification_target
            
            n_assets = len(symbols)
            
            def objective(weights):
                return -self.get_diversification_score(dict(zip(symbols, weights)))  # Minimize negative diversification
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum to 1
            ]
            
            # Bounds
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimize
            result = sco.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimized_weights = {sym: weight for sym, weight in zip(symbols, result.x)}
                final_score = self.get_diversification_score(optimized_weights)
                
                logger.info(f"Diversification optimization completed: score = {final_score:.3f}")
                
                return optimized_weights
            else:
                logger.warning("Diversification optimization did not converge")
                return {sym: 1.0/n_assets for sym in symbols}
                
        except Exception as e:
            logger.error(f"Diversification optimization failed: {e}")
            return {sym: 1.0/n_assets for sym in symbols}

    def _correlation_update_loop(self):
        """Background correlation update loop"""
        while True:
            try:
                # Update correlation matrices for different lookbacks
                for lookback in self.config.lookback_periods:
                    self.calculate_correlation_matrix(lookback=lookback)
                
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Correlation update loop failed: {e}")
                time.sleep(60)

    def _regime_detection_loop(self):
        """Background regime detection loop"""
        while True:
            try:
                self.detect_correlation_regime()
                time.sleep(self.config.update_frequency * 2)  # Half frequency
                
            except Exception as e:
                logger.error(f"Regime detection loop failed: {e}")
                time.sleep(120)

    def _clustering_update_loop(self):
        """Background clustering update loop"""
        while True:
            try:
                if len(self.return_data) >= 3:
                    self.perform_clustering()
                
                time.sleep(self.config.update_frequency * 5)  # Even lower frequency
                
            except Exception as e:
                logger.error(f"Clustering update loop failed: {e}")
                time.sleep(300)

    def _data_cleanup_loop(self):
        """Background data cleanup loop"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.config.max_history_days)
                
                with self._lock:
                    # Clean old price data
                    for symbol in list(self.price_data.keys()):
                        self.price_data[symbol] = deque(
                            [p for p in self.price_data[symbol] if p['timestamp'] > cutoff_time],
                            maxlen=1000
                        )
                    
                    # Clean old return data
                    for symbol in list(self.return_data.keys()):
                        self.return_data[symbol] = deque(
                            [r for r in self.return_data[symbol] if r['timestamp'] > cutoff_time],
                            maxlen=1000
                        )
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Data cleanup loop failed: {e}")
                time.sleep(1800)

    def get_correlation_insights(self) -> Dict[str, Any]:
        """Get comprehensive correlation insights"""
        try:
            # Current regime
            current_regime = self.detect_correlation_regime()
            
            # Recent alerts
            recent_alerts = list(self.alert_history)[-10:]
            
            # Clustering info
            clustering_info = {}
            if self.cluster_results:
                clustering_info = {
                    'number_of_clusters': len(self.cluster_results.clusters),
                    'silhouette_score': self.cluster_results.silhouette_score,
                    'cluster_sizes': {f"Cluster_{k}": len(v) for k, v in self.cluster_results.clusters.items()}
                }
            
            # Correlation statistics
            corr_matrix = self.calculate_correlation_matrix()
            avg_correlation = self._average_correlation(corr_matrix.matrix)
            corr_volatility = self._correlation_volatility(corr_matrix.matrix)
            
            insights = {
                'timestamp': datetime.now(),
                'current_regime': current_regime.value,
                'average_correlation': avg_correlation,
                'correlation_volatility': corr_volatility,
                'number_of_symbols': len(corr_matrix.matrix.columns),
                'recent_alerts_count': len(recent_alerts),
                'clustering_info': clustering_info,
                'diversification_opportunity': max(0, self.config.diversification_target - avg_correlation)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Correlation insights generation failed: {e}")
            return {'timestamp': datetime.now(), 'error': str(e)}

# Example usage and testing
def main():
    """Example usage of the AdvancedCorrelationManager"""
    
    # Configuration
    config = CorrelationConfig(
        lookback_periods=[20, 60, 120],
        high_correlation_threshold=0.7,
        low_correlation_threshold=0.2,
        update_frequency=60  # 1 minute for testing
    )
    
    # Initialize correlation manager
    corr_manager = AdvancedCorrelationManager(config)
    
    # Generate sample price data
    print("=== Generating Sample Price Data ===")
    symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD']
    
    np.random.seed(42)
    base_prices = {
        'EUR/USD': 1.1000,
        'GBP/USD': 1.3000,
        'USD/JPY': 150.00,
        'AUD/USD': 0.6500,
        'USD/CAD': 1.3500,
        'USD/CHF': 0.9000,
        'NZD/USD': 0.6000
    }
    
    # Generate correlated price movements
    n_periods = 500
    for i in range(n_periods):
        for symbol in symbols:
            # Create some correlation in movements
            base_noise = np.random.normal(0, 0.001)
            correlated_noise = base_noise * 0.7 + np.random.normal(0, 0.0005)
            
            if i == 0:
                price = base_prices[symbol]
            else:
                price = corr_manager.price_data[symbol][-1]['price'] * (1 + correlated_noise)
            
            corr_manager.update_price_data(symbol, price)
    
    print(f"Generated {n_periods} price points for {len(symbols)} symbols")
    
    # Test correlation calculations
    print("\n=== Correlation Analysis ===")
    
    # Calculate correlation matrix
    corr_matrix = corr_manager.calculate_correlation_matrix(symbols)
    print("Correlation Matrix:")
    print(corr_matrix.matrix.round(3))
    
    # Get specific correlation
    eur_gbp_corr = corr_manager.get_correlation('EUR/USD', 'GBP/USD')
    print(f"\nEUR/USD - GBP/USD Correlation: {eur_gbp_corr:.3f}")
    
    # Detect correlation regime
    regime = corr_manager.detect_correlation_regime()
    print(f"Current Correlation Regime: {regime.value}")
    
    # Perform clustering
    print("\n=== Asset Clustering ===")
    cluster_result = corr_manager.perform_clustering(symbols)
    print("Clusters:")
    for cluster_id, cluster_symbols in cluster_result.clusters.items():
        print(f"  Cluster {cluster_id}: {cluster_symbols}")
    print(f"Silhouette Score: {cluster_result.silhouette_score:.3f}")
    
    # Check alerts
    print("\n=== Correlation Alerts ===")
    alerts = corr_manager.check_correlation_alerts()
    for alert in alerts[:3]:  # Show first 3 alerts
        print(f"Alert: {alert.alert_type} - {alert.symbol_pair} (corr: {alert.current_correlation:.3f})")
    
    # Diversification analysis
    print("\n=== Diversification Analysis ===")
    portfolio_weights = {sym: 1.0/len(symbols) for sym in symbols}
    div_score = corr_manager.get_diversification_score(portfolio_weights)
    print(f"Equal Weight Portfolio Diversification Score: {div_score:.3f}")
    
    # Optimized weights
    optimized_weights = corr_manager.optimize_for_diversification(symbols)
    optimized_score = corr_manager.get_diversification_score(optimized_weights)
    print(f"Optimized Portfolio Diversification Score: {optimized_score:.3f}")
    print("Optimized Weights:")
    for symbol, weight in sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {symbol}: {weight:.3f}")
    
    # Get insights
    print("\n=== Correlation Insights ===")
    insights = corr_manager.get_correlation_insights()
    for key, value in insights.items():
        if key != 'clustering_info':
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
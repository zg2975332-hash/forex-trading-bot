"""
Advanced Market Regime Detector for Forex Trading Bot
Real-time market regime classification using machine learning and statistical methods
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ta
from ta import volatility, trend, momentum
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    STRONG_BULL = "strong_bull"
    WEAK_BULL = "weak_bull" 
    SIDEWAYS = "sideways"
    WEAK_BEAR = "weak_bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RALLY = "rally"
    UNCERTAIN = "uncertain"

@dataclass
class RegimeMetrics:
    """Comprehensive regime metrics"""
    timestamp: datetime
    symbol: str
    regime: MarketRegime
    confidence: float
    trend_strength: float
    volatility_level: float
    momentum_score: float
    market_health: float
    regime_duration: timedelta
    features: Dict[str, float]
    transition_probability: float

@dataclass
class RegimeHistory:
    """Regime history tracking"""
    start_time: datetime
    end_time: datetime
    regime: MarketRegime
    duration: timedelta
    performance: float
    volatility: float

class StatisticalRegimeDetector:
    """Statistical-based regime detection using technical indicators"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.regime_history: List[RegimeHistory] = []
        self.current_regime = MarketRegime.UNCERTAIN
        self.regime_start_time = datetime.now()
        
        # Thresholds for regime classification
        self.thresholds = {
            'trend_strength_strong': 0.7,
            'trend_strength_weak': 0.3,
            'volatility_high': 0.15,
            'volatility_low': 0.05,
            'momentum_strong': 0.6,
            'momentum_weak': 0.3
        }
        
        logger.info("Statistical Regime Detector initialized")
    
    def calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 10:
            return 0.0
        
        try:
            # Linear regression to determine trend strength
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices.values)
            
            # Normalize trend strength
            trend_strength = abs(r_value)
            trend_direction = np.sign(slope)
            
            return trend_strength * trend_direction
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate annualized volatility"""
        if len(prices) < 2:
            return 0.0
        
        try:
            returns = np.log(prices / prices.shift(1)).dropna()
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            return annualized_volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def calculate_momentum_score(self, prices: pd.Series) -> float:
        """Calculate momentum score using multiple timeframes"""
        if len(prices) < 50:
            return 0.0
        
        try:
            # Calculate momentum across different periods
            momentum_5 = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
            momentum_10 = (prices.iloc[-1] / prices.iloc[-10] - 1) if len(prices) >= 10 else 0
            momentum_20 = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
            
            # Weighted average momentum
            momentum_score = (momentum_5 * 0.4 + momentum_10 * 0.3 + momentum_20 * 0.3)
            
            return momentum_score
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0.0
    
    def calculate_market_health(self, prices: pd.Series) -> float:
        """Calculate overall market health score"""
        if len(prices) < 20:
            return 0.5
        
        try:
            # Price above moving averages
            sma_20 = prices.rolling(window=20).mean().iloc[-1]
            sma_50 = prices.rolling(window=50).mean().iloc[-1]
            
            price_vs_sma20 = 1.0 if prices.iloc[-1] > sma_20 else 0.0
            price_vs_sma50 = 1.0 if prices.iloc[-1] > sma_50 else 0.0
            
            # Volatility stability
            volatility = self.calculate_volatility(prices)
            volatility_score = 1.0 - min(volatility / 0.3, 1.0)  # Normalize
            
            # Trend consistency
            trend_strength = abs(self.calculate_trend_strength(prices))
            
            # Combined health score
            health_score = (price_vs_sma20 * 0.3 + price_vs_sma50 * 0.3 + 
                          volatility_score * 0.2 + trend_strength * 0.2)
            
            return health_score
            
        except Exception as e:
            logger.error(f"Error calculating market health: {e}")
            return 0.5
    
    def detect_regime_statistical(self, prices: pd.Series) -> RegimeMetrics:
        """Detect market regime using statistical methods"""
        try:
            if len(prices) < self.lookback_period:
                return self._create_uncertain_metrics(prices)
            
            # Calculate features
            trend_strength = self.calculate_trend_strength(prices)
            volatility_level = self.calculate_volatility(prices)
            momentum_score = self.calculate_momentum_score(prices)
            market_health = self.calculate_market_health(prices)
            
            # Determine regime
            regime, confidence = self._classify_regime(
                trend_strength, volatility_level, momentum_score, market_health
            )
            
            # Calculate regime duration
            current_time = datetime.now()
            regime_duration = current_time - self.regime_start_time
            
            # Update regime history if changed
            if regime != self.current_regime:
                self._update_regime_history(regime, prices)
                self.current_regime = regime
                self.regime_start_time = current_time
            
            # Create metrics
            metrics = RegimeMetrics(
                timestamp=current_time,
                symbol="EUR/USD",  # Would be parameterized in production
                regime=regime,
                confidence=confidence,
                trend_strength=trend_strength,
                volatility_level=volatility_level,
                momentum_score=momentum_score,
                market_health=market_health,
                regime_duration=regime_duration,
                features={
                    'trend_strength': trend_strength,
                    'volatility': volatility_level,
                    'momentum': momentum_score,
                    'market_health': market_health,
                    'sma_20_ratio': prices.iloc[-1] / prices.rolling(20).mean().iloc[-1],
                    'sma_50_ratio': prices.iloc[-1] / prices.rolling(50).mean().iloc[-1]
                },
                transition_probability=self._calculate_transition_probability()
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in statistical regime detection: {e}")
            return self._create_uncertain_metrics(prices)
    
    def _classify_regime(self, trend_strength: float, volatility: float, 
                        momentum: float, health: float) -> Tuple[MarketRegime, float]:
        """Classify regime based on calculated metrics"""
        
        # High volatility regimes
        if volatility > self.thresholds['volatility_high']:
            if trend_strength > self.thresholds['trend_strength_strong']:
                return MarketRegime.CRASH, 0.8
            else:
                return MarketRegime.HIGH_VOLATILITY, 0.7
        
        # Low volatility regimes
        elif volatility < self.thresholds['volatility_low']:
            if abs(trend_strength) < self.thresholds['trend_strength_weak']:
                return MarketRegime.LOW_VOLATILITY, 0.8
            else:
                return MarketRegime.SIDEWAYS, 0.6
        
        # Trending regimes
        elif abs(trend_strength) > self.thresholds['trend_strength_strong']:
            if trend_strength > 0:
                if momentum > self.thresholds['momentum_strong']:
                    return MarketRegime.STRONG_BULL, 0.9
                else:
                    return MarketRegime.WEAK_BULL, 0.7
            else:
                if momentum < -self.thresholds['momentum_strong']:
                    return MarketRegime.STRONG_BEAR, 0.9
                else:
                    return MarketRegime.WEAK_BEAR, 0.7
        
        # Sideways market
        elif abs(trend_strength) < self.thresholds['trend_strength_weak']:
            return MarketRegime.SIDEWAYS, 0.8
        
        # Rally (strong momentum but moderate trend)
        elif momentum > self.thresholds['momentum_strong']:
            return MarketRegime.RALLY, 0.7
        
        else:
            return MarketRegime.UNCERTAIN, 0.5
    
    def _update_regime_history(self, new_regime: MarketRegime, prices: pd.Series):
        """Update regime history when regime changes"""
        current_time = datetime.now()
        duration = current_time - self.regime_start_time
        
        # Calculate performance during the regime
        if len(prices) >= 10:
            start_price = prices.iloc[-min(len(prices), 100)]  # Use available data
            end_price = prices.iloc[-1]
            performance = (end_price - start_price) / start_price
        else:
            performance = 0.0
        
        # Add to history
        regime_record = RegimeHistory(
            start_time=self.regime_start_time,
            end_time=current_time,
            regime=self.current_regime,
            duration=duration,
            performance=performance,
            volatility=self.calculate_volatility(prices)
        )
        
        self.regime_history.append(regime_record)
        
        # Keep history manageable
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        logger.info(f"Regime changed from {self.current_regime.value} to {new_regime.value}")
    
    def _calculate_transition_probability(self) -> float:
        """Calculate probability of regime transition"""
        if len(self.regime_history) < 5:
            return 0.1
        
        # Calculate average regime duration
        durations = [r.duration.total_seconds() for r in self.regime_history[-10:]]
        avg_duration = np.mean(durations)
        current_duration = (datetime.now() - self.regime_start_time).total_seconds()
        
        # Probability increases as current duration approaches average
        if avg_duration > 0:
            probability = min(0.9, current_duration / avg_duration)
        else:
            probability = 0.1
        
        return probability
    
    def _create_uncertain_metrics(self, prices: pd.Series) -> RegimeMetrics:
        """Create uncertain regime metrics for insufficient data"""
        return RegimeMetrics(
            timestamp=datetime.now(),
            symbol="EUR/USD",
            regime=MarketRegime.UNCERTAIN,
            confidence=0.1,
            trend_strength=0.0,
            volatility_level=0.0,
            momentum_score=0.0,
            market_health=0.5,
            regime_duration=timedelta(0),
            features={},
            transition_probability=0.1
        )

class MLRegimeDetector:
    """Machine Learning-based regime detection using clustering"""
    
    def __init__(self, n_clusters: int = 5, lookback_period: int = 200):
        self.n_clusters = n_clusters
        self.lookback_period = lookback_period
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_labels = {}
        
        logger.info("ML Regime Detector initialized")
    
    def extract_features(self, prices: pd.Series) -> np.ndarray:
        """Extract comprehensive features for ML model"""
        if len(prices) < self.lookback_period:
            return np.array([])
        
        try:
            features = []
            
            # Price-based features
            returns = np.log(prices / prices.shift(1)).dropna()
            
            # Statistical features
            features.extend([
                returns.mean(), returns.std(), returns.skew(), returns.kurtosis(),
                prices.pct_change().std(),  # Volatility
                (prices.iloc[-1] / prices.rolling(20).mean().iloc[-1] - 1),  # Price vs MA
                (prices.iloc[-1] / prices.rolling(50).mean().iloc[-1] - 1),
            ])
            
            # Technical indicators
            # RSI
            rsi = ta.momentum.RSIIndicator(prices, window=14).rsi()
            features.append(rsi.iloc[-1] / 100.0 if not pd.isna(rsi.iloc[-1]) else 0.5)
            
            # MACD
            macd = ta.trend.MACD(prices)
            features.append(macd.macd_diff().iloc[-1] if not pd.isna(macd.macd_diff().iloc[-1]) else 0.0)
            
            # Bollinger Bands
            bb = volatility.BollingerBands(prices)
            bb_position = (prices.iloc[-1] - bb.bollinger_lband().iloc[-1]) / (
                bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])
            features.append(bb_position if not pd.isna(bb_position) else 0.5)
            
            # ATR
            high = prices * 1.001  # Simulated high/low
            low = prices * 0.999
            atr = volatility.AverageTrueRange(high, low, prices).average_true_range()
            features.append(atr.iloc[-1] / prices.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0)
            
            # Additional features
            features.extend([
                prices.autocorr(lag=1) if len(prices) > 1 else 0.0,  # Autocorrelation
                self._hurst_exponent(prices),  # Hurst exponent
                self._fractal_dimension(prices),  # Fractal dimension
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting ML features: {e}")
            return np.array([])
    
    def _hurst_exponent(self, prices: pd.Series) -> float:
        """Calculate Hurst exponent for market efficiency"""
        try:
            if len(prices) < 100:
                return 0.5
            
            lags = range(2, 20)
            tau = [np.std(np.subtract(prices[lag:].values, prices[:-lag].values)) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            
            return poly[0]  # Hurst exponent
            
        except:
            return 0.5
    
    def _fractal_dimension(self, prices: pd.Series) -> float:
        """Calculate fractal dimension using Katz method"""
        try:
            if len(prices) < 10:
                return 1.0
            
            n = len(prices)
            L = np.sum(np.abs(np.diff(prices)))
            d = np.max(np.abs(prices - prices.iloc[0]))
            
            if d == 0:
                return 1.0
            
            return np.log(n) / (np.log(n) + np.log(d/L))
            
        except:
            return 1.0
    
    def fit_model(self, historical_prices: pd.Series):
        """Fit ML model on historical data"""
        try:
            if len(historical_prices) < self.lookback_period * 2:
                logger.warning("Insufficient data for ML model fitting")
                return
            
            # Extract features for entire history
            features_list = []
            valid_indices = []
            
            for i in range(self.lookback_period, len(historical_prices)):
                window = historical_prices.iloc[i-self.lookback_period:i]
                features = self.extract_features(window)
                if features.size > 0:
                    features_list.append(features.flatten())
                    valid_indices.append(i)
            
            if not features_list:
                logger.warning("No valid features extracted for ML training")
                return
            
            X = np.array(features_list)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit KMeans
            self.kmeans.fit(X_scaled)
            
            # Fit Isolation Forest for anomaly detection
            self.isolation_forest.fit(X_scaled)
            
            # Label regimes based on cluster characteristics
            self._label_regimes(X_scaled, historical_prices.iloc[valid_indices])
            
            self.is_fitted = True
            logger.info(f"ML model fitted on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error fitting ML model: {e}")
    
    def _label_regimes(self, X: np.ndarray, prices: pd.Series):
        """Label clusters based on their characteristics"""
        cluster_centers = self.kmeans.cluster_centers_
        
        for i, center in enumerate(cluster_centers):
            # Analyze cluster characteristics
            volatility = center[4]  # Volatility feature index
            trend = center[5]  # Trend feature index
            
            if volatility > 0.1:
                if trend > 0.05:
                    self.regime_labels[i] = MarketRegime.HIGH_VOLATILITY
                else:
                    self.regime_labels[i] = MarketRegime.CRASH
            elif trend > 0.02:
                self.regime_labels[i] = MarketRegime.STRONG_BULL
            elif trend < -0.02:
                self.regime_labels[i] = MarketRegime.STRONG_BEAR
            else:
                self.regime_labels[i] = MarketRegime.SIDEWAYS
    
    def detect_regime_ml(self, prices: pd.Series) -> Optional[RegimeMetrics]:
        """Detect regime using ML model"""
        if not self.is_fitted or len(prices) < self.lookback_period:
            return None
        
        try:
            # Extract features
            features = self.extract_features(prices)
            if features.size == 0:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict cluster
            cluster = self.kmeans.predict(features_scaled)[0]
            
            # Detect anomalies
            is_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
            
            # Get regime from cluster label
            regime = self.regime_labels.get(cluster, MarketRegime.UNCERTAIN)
            
            # If anomaly, mark as high volatility or crash
            if is_anomaly:
                regime = MarketRegime.HIGH_VOLATILITY
            
            # Calculate confidence based on distance to cluster center
            distances = np.linalg.norm(
                features_scaled - self.kmeans.cluster_centers_[cluster], axis=1
            )
            confidence = 1.0 / (1.0 + distances[0])  # Convert distance to confidence
            
            # Create metrics
            metrics = RegimeMetrics(
                timestamp=datetime.now(),
                symbol="EUR/USD",
                regime=regime,
                confidence=confidence,
                trend_strength=features[0][5],  # Trend feature
                volatility_level=features[0][4],  # Volatility feature
                momentum_score=features[0][6],  # RSI-based momentum
                market_health=0.7,  # Would need separate calculation
                regime_duration=timedelta(0),  # ML doesn't track duration
                features=dict(zip(range(len(features[0])), features[0])),
                transition_probability=0.3  # Default
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in ML regime detection: {e}")
            return None

class MultiTimeframeRegimeDetector:
    """Multi-timeframe regime detection for confirmation"""
    
    def __init__(self):
        self.timeframes = ['15m', '1H', '4H', '1D']
        self.detectors = {
            '15m': StatisticalRegimeDetector(lookback_period=50),
            '1H': StatisticalRegimeDetector(lookback_period=100),
            '4H': StatisticalRegimeDetector(lookback_period=150),
            '1D': StatisticalRegimeDetector(lookback_period=200)
        }
        self.ml_detector = MLRegimeDetector()
        
        # Weighting for different timeframes
        self.timeframe_weights = {
            '15m': 0.1,
            '1H': 0.2,
            '4H': 0.3,
            '1D': 0.4
        }
        
        logger.info("Multi-Timeframe Regime Detector initialized")
    
    def detect_multi_timeframe_regime(self, 
                                    price_data: Dict[str, pd.Series]) -> RegimeMetrics:
        """Detect regime using multiple timeframes"""
        try:
            timeframe_metrics = {}
            
            # Get regime from each timeframe
            for tf in self.timeframes:
                if tf in price_data and len(price_data[tf]) > 0:
                    metrics = self.detectors[tf].detect_regime_statistical(price_data[tf])
                    timeframe_metrics[tf] = metrics
            
            if not timeframe_metrics:
                return self._create_uncertain_metrics()
            
            # Combine regimes using weighted voting
            combined_regime, combined_confidence = self._combine_regimes(timeframe_metrics)
            
            # Get ML regime if available
            ml_metrics = None
            if '1H' in price_data:
                ml_metrics = self.ml_detector.detect_regime_ml(price_data['1H'])
            
            # Final decision with ML confirmation
            final_regime, final_confidence = self._finalize_regime(
                combined_regime, combined_confidence, ml_metrics
            )
            
            # Create comprehensive metrics
            primary_metrics = list(timeframe_metrics.values())[0]  # Use 1H as primary
            
            final_metrics = RegimeMetrics(
                timestamp=datetime.now(),
                symbol="EUR/USD",
                regime=final_regime,
                confidence=final_confidence,
                trend_strength=primary_metrics.trend_strength,
                volatility_level=primary_metrics.volatility_level,
                momentum_score=primary_metrics.momentum_score,
                market_health=primary_metrics.market_health,
                regime_duration=primary_metrics.regime_duration,
                features={
                    'timeframe_agreement': len(timeframe_metrics),
                    'ml_confidence': ml_metrics.confidence if ml_metrics else 0.0,
                    **primary_metrics.features
                },
                transition_probability=primary_metrics.transition_probability
            )
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe regime detection: {e}")
            return self._create_uncertain_metrics()
    
    def _combine_regimes(self, timeframe_metrics: Dict[str, RegimeMetrics]) -> Tuple[MarketRegime, float]:
        """Combine regimes from different timeframes using weighted voting"""
        regime_votes = {}
        total_confidence = 0.0
        
        for tf, metrics in timeframe_metrics.items():
            weight = self.timeframe_weights.get(tf, 0.1)
            vote_strength = metrics.confidence * weight
            
            if metrics.regime not in regime_votes:
                regime_votes[metrics.regime] = 0.0
            
            regime_votes[metrics.regime] += vote_strength
            total_confidence += vote_strength
        
        if not regime_votes:
            return MarketRegime.UNCERTAIN, 0.1
        
        # Find regime with highest votes
        best_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
        confidence = regime_votes[best_regime] / total_confidence if total_confidence > 0 else 0.1
        
        return best_regime, confidence
    
    def _finalize_regime(self, statistical_regime: MarketRegime, 
                        statistical_confidence: float,
                        ml_metrics: Optional[RegimeMetrics]) -> Tuple[MarketRegime, float]:
        """Finalize regime decision with ML confirmation"""
        if ml_metrics is None:
            return statistical_regime, statistical_confidence
        
        # If ML agrees, boost confidence
        if ml_metrics.regime == statistical_regime:
            final_confidence = (statistical_confidence + ml_metrics.confidence) / 2
            return statistical_regime, final_confidence
        
        # If ML disagrees but has low confidence, trust statistical
        elif ml_metrics.confidence < 0.6:
            return statistical_regime, statistical_confidence * 0.8
        
        # If ML strongly disagrees, use ML with reduced confidence
        else:
            return ml_metrics.regime, ml_metrics.confidence * 0.7
    
    def _create_uncertain_metrics(self) -> RegimeMetrics:
        """Create uncertain regime metrics"""
        return RegimeMetrics(
            timestamp=datetime.now(),
            symbol="EUR/USD",
            regime=MarketRegime.UNCERTAIN,
            confidence=0.1,
            trend_strength=0.0,
            volatility_level=0.0,
            momentum_score=0.0,
            market_health=0.5,
            regime_duration=timedelta(0),
            features={},
            transition_probability=0.1
        )
    
    def train_ml_model(self, historical_data: Dict[str, pd.Series]):
        """Train ML model on historical data"""
        try:
            # Use 1H data for ML training
            if '1H' in historical_data:
                self.ml_detector.fit_model(historical_data['1H'])
                logger.info("ML model trained on historical data")
        except Exception as e:
            logger.error(f"Error training ML model: {e}")

class MarketRegimeDetector:
    """
    Main Market Regime Detector coordinating all detection methods
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize detectors
        self.statistical_detector = StatisticalRegimeDetector()
        self.ml_detector = MLRegimeDetector()
        self.multi_timeframe_detector = MultiTimeframeRegimeDetector()
        
        # Regime tracking
        self.current_regime = MarketRegime.UNCERTAIN
        self.regime_history: List[RegimeMetrics] = []
        self.regime_start_time = datetime.now()
        
        # Performance tracking
        self.regime_performance = {}
        self.detection_accuracy = 0.0
        
        logger.info("Market Regime Detector initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'primary_method': 'multi_timeframe',  # statistical, ml, multi_timeframe
            'confidence_threshold': 0.7,
            'min_regime_duration': timedelta(minutes=30),
            'enable_ml': True,
            'enable_multi_timeframe': True
        }
    
    def detect_regime(self, price_data: Dict[str, pd.Series]) -> RegimeMetrics:
        """
        Detect current market regime using configured methods
        
        Args:
            price_data: Dictionary of price series for different timeframes
            
        Returns:
            RegimeMetrics object
        """
        try:
            method = self.config['primary_method']
            
            if method == 'multi_timeframe' and self.config['enable_multi_timeframe']:
                metrics = self.multi_timeframe_detector.detect_multi_timeframe_regime(price_data)
            elif method == 'ml' and self.config['enable_ml']:
                metrics = self.ml_detector.detect_regime_ml(price_data.get('1H', pd.Series()))
                if metrics is None:
                    metrics = self.statistical_detector.detect_regime_statistical(
                        price_data.get('1H', pd.Series())
                    )
            else:
                metrics = self.statistical_detector.detect_regime_statistical(
                    price_data.get('1H', pd.Series())
                )
            
            # Update regime history
            self._update_regime_history(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self._create_fallback_metrics()
    
    def _update_regime_history(self, metrics: RegimeMetrics):
        """Update regime history and track changes"""
        # Check if regime changed
        if metrics.regime != self.current_regime:
            # Only update if confidence is above threshold and minimum duration passed
            current_duration = datetime.now() - self.regime_start_time
            if (metrics.confidence >= self.config['confidence_threshold'] and 
                current_duration >= self.config['min_regime_duration']):
                
                logger.info(f"Regime changed from {self.current_regime.value} to {metrics.regime.value}")
                self.current_regime = metrics.regime
                self.regime_start_time = datetime.now()
        
        # Add to history
        self.regime_history.append(metrics)
        
        # Keep history manageable
        if len(self.regime_history) > 10000:
            self.regime_history = self.regime_history[-10000:]
    
    def _create_fallback_metrics(self) -> RegimeMetrics:
        """Create fallback metrics when detection fails"""
        return RegimeMetrics(
            timestamp=datetime.now(),
            symbol="EUR/USD",
            regime=MarketRegime.UNCERTAIN,
            confidence=0.1,
            trend_strength=0.0,
            volatility_level=0.0,
            momentum_score=0.0,
            market_health=0.5,
            regime_duration=timedelta(0),
            features={},
            transition_probability=0.1
        )
    
    def get_regime_report(self) -> Dict[str, Any]:
        """Get comprehensive regime report"""
        if not self.regime_history:
            return {'current_regime': 'uncertain', 'confidence': 0.0}
        
        current_metrics = self.regime_history[-1]
        
        # Calculate regime statistics
        regime_durations = {}
        for metrics in self.regime_history[-100:]:  # Last 100 detections
            regime = metrics.regime.value
            if regime not in regime_durations:
                regime_durations[regime] = []
            regime_durations[regime].append(metrics.regime_duration.total_seconds())
        
        avg_durations = {
            regime: np.mean(durations) if durations else 0
            for regime, durations in regime_durations.items()
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_regime': {
                'regime': current_metrics.regime.value,
                'confidence': current_metrics.confidence,
                'trend_strength': current_metrics.trend_strength,
                'volatility': current_metrics.volatility_level,
                'momentum': current_metrics.momentum_score,
                'market_health': current_metrics.market_health,
                'duration_seconds': current_metrics.regime_duration.total_seconds(),
                'transition_probability': current_metrics.transition_probability
            },
            'regime_statistics': {
                'average_durations': avg_durations,
                'total_detections': len(self.regime_history),
                'current_regime_duration': (datetime.now() - self.regime_start_time).total_seconds()
            },
            'detection_methods': {
                'primary': self.config['primary_method'],
                'ml_enabled': self.config['enable_ml'],
                'multi_timeframe_enabled': self.config['enable_multi_timeframe']
            }
        }
    
    def train_models(self, historical_data: Dict[str, pd.Series]):
        """Train ML models on historical data"""
        try:
            if self.config['enable_ml'] and '1H' in historical_data:
                self.ml_detector.fit_model(historical_data['1H'])
            
            if self.config['enable_multi_timeframe']:
                self.multi_timeframe_detector.train_ml_model(historical_data)
            
            logger.info("All regime detection models trained")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the Market Regime Detector
    print("Testing Market Regime Detector...")
    
    try:
        # Initialize detector
        regime_detector = MarketRegimeDetector()
        
        # Generate sample price data
        print("Generating sample price data...")
        np.random.seed(42)
        
        # Create sample data for different timeframes
        base_price = 1.1000
        dates_1h = pd.date_range(start='2024-01-01', periods=500, freq='1H')
        
        # Simulate different market regimes in the data
        prices_1h = []
        current_price = base_price
        
        # Bull market segment
        for i in range(150):
            current_price *= 1 + np.random.normal(0.001, 0.005)
            prices_1h.append(current_price)
        
        # Sideways market segment
        for i in range(150):
            current_price *= 1 + np.random.normal(0.000, 0.003)
            prices_1h.append(current_price)
        
        # Bear market segment
        for i in range(200):
            current_price *= 1 + np.random.normal(-0.001, 0.006)
            prices_1h.append(current_price)
        
        price_series_1h = pd.Series(prices_1h, index=dates_1h[:len(prices_1h)])
        
        # Create multi-timeframe data (simplified)
        price_data = {
            '1H': price_series_1h,
            '4H': price_series_1h.iloc[::4],  # Sample every 4th point for 4H
            '1D': price_series_1h.iloc[::24]   # Sample every 24th point for 1D
        }
        
        # Train models
        print("Training ML models...")
        regime_detector.train_models(price_data)
        
        # Test regime detection
        print("Testing regime detection...")
        regime_metrics = regime_detector.detect_regime(price_data)
        
        print(f"Detected Regime: {regime_metrics.regime.value}")
        print(f"Confidence: {regime_metrics.confidence:.2f}")
        print(f"Trend Strength: {regime_metrics.trend_strength:.3f}")
        print(f"Volatility: {regime_metrics.volatility_level:.3f}")
        print(f"Momentum: {regime_metrics.momentum_score:.3f}")
        print(f"Market Health: {regime_metrics.market_health:.3f}")
        
        # Get comprehensive report
        print("\nGenerating regime report...")
        report = regime_detector.get_regime_report()
        print(f"Current Regime: {report['current_regime']['regime']}")
        print(f"Regime Confidence: {report['current_regime']['confidence']:.2f}")
        print(f"Regime Duration: {report['current_regime']['duration_seconds']:.0f} seconds")
        
        # Test multiple detections
        print("\nTesting multiple detections...")
        for i in range(5):
            # Use different segments of the data
            segment_start = i * 100
            segment_end = segment_start + 200
            segment_data = {
                '1H': price_series_1h.iloc[segment_start:segment_end],
                '4H': price_series_1h.iloc[segment_start:segment_end:4],
                '1D': price_series_1h.iloc[segment_start:segment_end:24]
            }
            
            metrics = regime_detector.detect_regime(segment_data)
            print(f"Detection {i+1}: {metrics.regime.value} (confidence: {metrics.confidence:.2f})")
        
        print(f"\n✅ Market Regime Detector test completed successfully!")
        
    except Exception as e:
        print(f"❌ Market Regime Detector test failed: {e}")
        import traceback
        traceback.print_exc()
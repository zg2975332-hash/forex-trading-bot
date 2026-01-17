"""
Advanced Feature Engineering for FOREX TRADING BOT
Comprehensive feature extraction and transformation for quantitative trading
"""

import logging
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import numba
from numba import jit, prange
from collections import defaultdict, deque
import pywt
from hurst import compute_Hc
from typing import Callable
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    PRICE_BASED = "price_based"
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SPECTRAL = "spectral"
    WAVELET = "wavelet"
    MICROSTRUCTURE = "microstructure"
    REGIME = "regime"
    NOVELTY = "novelty"

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    CALM = "calm"
    RANGING = "ranging"

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Price-based features
    enable_returns: bool = True
    enable_log_returns: bool = True
    enable_ohlc_ratios: bool = True
    
    # Technical indicators
    enable_technical: bool = True
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    bb_period: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_period: int = 14
    atr_period: int = 14
    
    # Statistical features
    enable_statistical: bool = True
    rolling_window: int = 20
    zscore_threshold: float = 2.0
    entropy_window: int = 50
    autocorrelation_lags: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Volatility features
    enable_volatility: bool = True
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20, 30, 50])
    garch_estimation: bool = False
    
    # Volume features
    enable_volume: bool = True
    volume_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Advanced features
    enable_advanced: bool = True
    spectral_window: int = 100
    wavelet_level: int = 4
    hurst_window: int = 100
    fractal_window: int = 50
    
    # Feature selection
    enable_feature_selection: bool = True
    max_features: int = 100
    correlation_threshold: float = 0.95
    
    # Normalization
    normalization_method: str = "robust"  # standard, robust, minmax

@dataclass
class FeatureSet:
    """Computed feature set"""
    features: pd.DataFrame
    feature_importance: Dict[str, float]
    feature_groups: Dict[FeatureType, List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    scaler: Any = None

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for Forex trading
    Implements 200+ features across multiple domains with professional-grade algorithms
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_cache = {}
        self.scalers = {}
        self.feature_stats = {}
        self.performance_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_correlation = {}
        
        # Statistical properties
        self.feature_distributions = {}
        
        logger.info("AdvancedFeatureEngineer initialized")

    def compute_features(self, data: pd.DataFrame, target: pd.Series = None) -> FeatureSet:
        """
        Compute comprehensive feature set from market data
        """
        try:
            logger.info(f"Computing features for {len(data)} data points")
            
            features_dict = {}
            feature_groups = defaultdict(list)
            
            # 1. Price-based Features
            price_features = self._compute_price_features(data)
            features_dict.update(price_features)
            feature_groups[FeatureType.PRICE_BASED].extend(price_features.keys())
            logger.info(f"Computed {len(price_features)} price-based features")
            
            # 2. Technical Indicators
            if self.config.enable_technical:
                technical_features = self._compute_technical_features(data)
                features_dict.update(technical_features)
                feature_groups[FeatureType.TECHNICAL].extend(technical_features.keys())
                logger.info(f"Computed {len(technical_features)} technical features")
            
            # 3. Statistical Features
            if self.config.enable_statistical:
                statistical_features = self._compute_statistical_features(data)
                features_dict.update(statistical_features)
                feature_groups[FeatureType.STATISTICAL].extend(statistical_features.keys())
                logger.info(f"Computed {len(statistical_features)} statistical features")
            
            # 4. Volatility Features
            if self.config.enable_volatility:
                volatility_features = self._compute_volatility_features(data)
                features_dict.update(volatility_features)
                feature_groups[FeatureType.VOLATILITY].extend(volatility_features.keys())
                logger.info(f"Computed {len(volatility_features)} volatility features")
            
            # 5. Volume Features
            if self.config.enable_volume and 'volume' in data.columns:
                volume_features = self._compute_volume_features(data)
                features_dict.update(volume_features)
                feature_groups[FeatureType.VOLUME].extend(volume_features.keys())
                logger.info(f"Computed {len(volume_features)} volume features")
            
            # 6. Advanced Features
            if self.config.enable_advanced:
                advanced_features = self._compute_advanced_features(data)
                features_dict.update(advanced_features)
                feature_groups[FeatureType.NOVELTY].extend(advanced_features.keys())
                logger.info(f"Computed {len(advanced_features)} advanced features")
            
            # Create feature DataFrame
            features_df = pd.DataFrame(features_dict, index=data.index)
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            # Feature selection
            if self.config.enable_feature_selection and target is not None:
                features_df = self._select_features(features_df, target)
            
            # Normalize features
            features_df, scaler = self._normalize_features(features_df)
            
            # Compute feature importance if target is provided
            feature_importance = {}
            if target is not None:
                feature_importance = self._compute_feature_importance(features_df, target)
            
            # Update performance metrics
            self._update_performance_metrics(features_df)
            
            logger.info(f"Feature computation completed: {len(features_df.columns)} total features")
            
            return FeatureSet(
                features=features_df,
                feature_importance=feature_importance,
                feature_groups=dict(feature_groups),
                metadata={
                    'total_features': len(features_df.columns),
                    'computation_timestamp': pd.Timestamp.now(),
                    'data_points': len(data),
                    'feature_groups_summary': {k.value: len(v) for k, v in feature_groups.items()}
                },
                scaler=scaler
            )
            
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            raise

    def _compute_price_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute price-based features"""
        features = {}
        
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            open_ = data['open'].values
            
            # Basic returns
            if self.config.enable_returns:
                features['returns'] = self._compute_returns(close)
                features['returns_abs'] = np.abs(features['returns'])
            
            if self.config.enable_log_returns:
                features['log_returns'] = self._compute_log_returns(close)
                features['log_returns_abs'] = np.abs(features['log_returns'])
            
            # OHLC ratios and relationships
            if self.config.enable_ohlc_ratios:
                features['hl_ratio'] = (high - low) / close  # High-Low ratio
                features['oc_ratio'] = (open_ - close) / close  # Open-Close ratio
                features['co_ratio'] = (close - open_) / open_  # Close-Open ratio
                
                # Price position within daily range
                features['price_position'] = (close - low) / (high - low)
                features['price_position'] = np.clip(features['price_position'], 0, 1)  # Handle division by zero
            
            # Price extremes
            features['daily_range'] = high - low
            features['normalized_range'] = features['daily_range'] / close
            
            # Gap features
            features['overnight_gap'] = (open_ - np.roll(close, 1)) / np.roll(close, 1)
            features['overnight_gap'][0] = 0  # Handle first element
            
            # Momentum-like features
            features['price_acceleration'] = self._compute_acceleration(close)
            features['price_jerk'] = self._compute_jerk(close)
            
            # Support/resistance levels (simplified)
            features['proximity_to_high'] = (high - close) / close
            features['proximity_to_low'] = (close - low) / close
            
            return features
            
        except Exception as e:
            logger.error(f"Price feature computation failed: {e}")
            return {}

    def _compute_technical_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute technical indicators"""
        features = {}
        
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data.get('volume', np.ones(len(data)))
            
            # Moving Averages
            for period in self.config.ma_periods:
                if len(close) >= period:
                    # Simple Moving Average
                    sma = talib.SMA(close, timeperiod=period)
                    features[f'SMA_{period}'] = sma
                    
                    # Exponential Moving Average
                    ema = talib.EMA(close, timeperiod=period)
                    features[f'EMA_{period}'] = ema
                    
                    # Price relative to MA
                    features[f'price_vs_sma_{period}'] = (close - sma) / sma
                    features[f'price_vs_ema_{period}'] = (close - ema) / ema
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, 
                timeperiod=self.config.bb_period,
                nbdevup=2, 
                nbdevdn=2
            )
            features['BB_upper'] = bb_upper
            features['BB_lower'] = bb_lower
            features['BB_middle'] = bb_middle
            features['BB_width'] = (bb_upper - bb_lower) / bb_middle
            features['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            features['BB_position'] = np.clip(features['BB_position'], 0, 1)
            
            # RSI
            rsi = talib.RSI(close, timeperiod=self.config.rsi_period)
            features['RSI'] = rsi
            features['RSI_normalized'] = (rsi - 50) / 50  # Normalize around 0
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            features['MACD'] = macd
            features['MACD_signal'] = macd_signal
            features['MACD_histogram'] = macd_hist
            features['MACD_ratio'] = macd / (np.abs(macd_signal) + 1e-8)
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=self.config.stoch_period,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            features['Stoch_K'] = slowk
            features['Stoch_D'] = slowd
            features['Stoch_KD_ratio'] = slowk / (slowd + 1e-8)
            
            # ATR
            atr = talib.ATR(high, low, close, timeperiod=self.config.atr_period)
            features['ATR'] = atr
            features['ATR_ratio'] = atr / close
            
            # Additional indicators
            features['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            features['CCI'] = talib.CCI(high, low, close, timeperiod=20)
            features['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
            features['Momentum'] = talib.MOM(close, timeperiod=10)
            features['ROC'] = talib.ROC(close, timeperiod=10)
            
            # Volume-based indicators (if volume available)
            if 'volume' in data.columns:
                features['OBV'] = talib.OBV(close, volume)
                features['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            # Remove any completely NaN features
            features = {k: v for k, v in features.items() if not np.all(np.isnan(v))}
            
            return features
            
        except Exception as e:
            logger.error(f"Technical feature computation failed: {e}")
            return {}

    def _compute_statistical_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute statistical features"""
        features = {}
        
        try:
            close = data['close'].values
            returns = self._compute_returns(close)
            
            window = self.config.rolling_window
            
            # Rolling statistics
            if len(close) > window:
                # Central moments
                rolling_mean = pd.Series(close).rolling(window).mean()
                rolling_std = pd.Series(close).rolling(window).std()
                rolling_skew = pd.Series(close).rolling(window).skew()
                rolling_kurtosis = pd.Series(close).rolling(window).kurt()
                
                features['rolling_mean'] = rolling_mean
                features['rolling_std'] = rolling_std
                features['rolling_skew'] = rolling_skew
                features['rolling_kurtosis'] = rolling_kurtosis
                
                # Z-score based features
                zscore = (close - rolling_mean) / rolling_std
                features['zscore'] = zscore
                features['zscore_abs'] = np.abs(zscore)
                features['zscore_squared'] = zscore ** 2
                
                # Extreme value detection
                features['extreme_high'] = (zscore > self.config.zscore_threshold).astype(float)
                features['extreme_low'] = (zscore < -self.config.zscore_threshold).astype(float)
                features['extreme_any'] = (np.abs(zscore) > self.config.zscore_threshold).astype(float)
                
                # Quantile-based features
                rolling_q25 = pd.Series(close).rolling(window).quantile(0.25)
                rolling_q75 = pd.Series(close).rolling(window).quantile(0.75)
                features['rolling_iqr'] = rolling_q75 - rolling_q25
                features['rolling_q25'] = rolling_q25
                features['rolling_q75'] = rolling_q75
                features['price_vs_q25'] = (close - rolling_q25) / rolling_q25
                features['price_vs_q75'] = (close - rolling_q75) / rolling_q75
            
            # Return distribution features
            if len(returns) > window:
                return_skew = pd.Series(returns).rolling(window).skew()
                return_kurtosis = pd.Series(returns).rolling(window).kurt()
                features['return_skew'] = return_skew
                features['return_kurtosis'] = return_kurtosis
                
                # Variance ratio test (simplified)
                features['variance_ratio'] = self._compute_variance_ratio(returns, window)
            
            # Autocorrelation features
            for lag in self.config.autocorrelation_lags:
                acf_feature = self._compute_autocorrelation(returns, lag, window)
                features[f'return_acf_lag_{lag}'] = acf_feature
            
            # Entropy features
            entropy_features = self._compute_entropy_features(close, window=self.config.entropy_window)
            features.update(entropy_features)
            
            # Normality tests
            if len(returns) > 30:
                normality_features = self._compute_normality_tests(returns, window=30)
                features.update(normality_features)
            
            # Stationarity tests
            stationarity_features = self._compute_stationarity_tests(close, window=50)
            features.update(stationarity_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Statistical feature computation failed: {e}")
            return {}

    def _compute_volatility_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute volatility features"""
        features = {}
        
        try:
            close = data['close'].values
            open_ = data['open'].values
            high = data['high'].values
            low = data['low'].values
            returns = self._compute_returns(close)
            
            # Historical volatility (multiple timeframes)
            for window in self.config.volatility_windows:
                if len(returns) > window:
                    # Simple historical volatility
                    hist_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
                    features[f'hist_vol_{window}'] = hist_vol
                    
                    # Parkinson volatility (high-low estimator)
                    parkinson_vol = self._compute_parkinson_volatility(high, low, window)
                    features[f'parkinson_vol_{window}'] = parkinson_vol
                    
                    # Garman-Klass volatility
                    gk_vol = self._compute_garman_klass_volatility(open_, high, low, close, window)
                    features[f'gk_vol_{window}'] = gk_vol
            
            # Volatility ratios
            if 'hist_vol_20' in features and 'hist_vol_50' in features:
                features['vol_ratio_20_50'] = features['hist_vol_20'] / features['hist_vol_50']
            
            # Volatility regimes
            if 'hist_vol_20' in features:
                vol_mean = features['hist_vol_20'].mean()
                vol_std = features['hist_vol_20'].std()
                features['vol_regime'] = (features['hist_vol_20'] - vol_mean) / vol_std
            
            # Volatility clustering (autocorrelation of squared returns)
            if len(returns) > 20:
                squared_returns = returns ** 2
                vol_clustering = pd.Series(squared_returns).rolling(20).apply(
                    lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan,
                    raw=True
                )
                features['vol_clustering'] = vol_clustering
            
            # Realized volatility (using intraday high-low if available)
            if all(col in data.columns for col in ['high', 'low']):
                for window in [5, 10, 20]:
                    realized_vol = self._compute_realized_volatility(high, low, window)
                    features[f'realized_vol_{window}'] = realized_vol
            
            return features
            
        except Exception as e:
            logger.error(f"Volatility feature computation failed: {e}")
            return {}

    def _compute_volume_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute volume-based features"""
        features = {}
        
        try:
            if 'volume' not in data.columns:
                return features
                
            volume = data['volume'].values
            close = data['close'].values
            
            # Basic volume statistics
            for window in self.config.volume_windows:
                if len(volume) > window:
                    volume_sma = pd.Series(volume).rolling(window).mean()
                    volume_std = pd.Series(volume).rolling(window).std()
                    
                    features[f'volume_sma_{window}'] = volume_sma
                    features[f'volume_std_{window}'] = volume_std
                    features[f'volume_zscore_{window}'] = (volume - volume_sma) / volume_std
            
            # Volume-price relationships
            price_change = np.abs(self._compute_returns(close))
            features['volume_price_correlation'] = pd.Series(volume).rolling(20).corr(pd.Series(price_change))
            
            # Volume spikes
            volume_median = pd.Series(volume).rolling(50).median()
            volume_mad = pd.Series(volume).rolling(50).apply(lambda x: np.median(np.abs(x - np.median(x))))
            features['volume_spike'] = (volume - volume_median) / (volume_mad + 1e-8)
            
            # On-balance volume features
            obv = talib.OBV(close, volume)
            features['OBV'] = obv
            features['OBV_slope'] = self._compute_slope(obv, window=10)
            
            # Volume profile (simplified)
            features['volume_vwap'] = self._compute_volume_vwap(close, volume, window=20)
            
            return features
            
        except Exception as e:
            logger.error(f"Volume feature computation failed: {e}")
            return {}

    def _compute_advanced_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute advanced quantitative features"""
        features = {}
        
        try:
            close = data['close'].values
            returns = self._compute_returns(close)
            
            # 1. Hurst Exponent
            if len(close) >= self.config.hurst_window:
                hurst_features = self._compute_hurst_features(close, window=self.config.hurst_window)
                features.update(hurst_features)
            
            # 2. Fractal Analysis
            fractal_features = self._compute_fractal_features(close, window=self.config.fractal_window)
            features.update(fractal_features)
            
            # 3. Spectral Analysis
            spectral_features = self._compute_spectral_features(close, window=self.config.spectral_window)
            features.update(spectral_features)
            
            # 4. Wavelet Analysis
            wavelet_features = self._compute_wavelet_features(close, level=self.config.wavelet_level)
            features.update(wavelet_features)
            
            # 5. Market Regime Detection
            regime_features = self._compute_regime_features(close)
            features.update(regime_features)
            
            # 6. Correlation Structure
            correlation_features = self._compute_correlation_features(returns, lookback=50)
            features.update(correlation_features)
            
            # 7. Nonlinear Dynamics
            nonlinear_features = self._compute_nonlinear_features(close)
            features.update(nonlinear_features)
            
            # 8. Risk Measures
            risk_features = self._compute_risk_features(returns)
            features.update(risk_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Advanced feature computation failed: {e}")
            return {}

    def _compute_hurst_features(self, prices: np.ndarray, window: int) -> Dict[str, pd.Series]:
        """Compute Hurst exponent features"""
        features = {}
        
        try:
            hurst_values = []
            regime_values = []
            
            for i in range(len(prices) - window + 1):
                window_data = prices[i:i+window]
                try:
                    H, _, _ = compute_Hc(window_data, kind='price', simplified=True)
                    hurst_values.append(H)
                    
                    # Interpret Hurst exponent
                    if H > 0.6:
                        regime = 1  # Trending
                    elif H < 0.4:
                        regime = -1  # Mean-reverting
                    else:
                        regime = 0  # Random
                    regime_values.append(regime)
                    
                except:
                    hurst_values.append(np.nan)
                    regime_values.append(np.nan)
            
            # Pad with NaN
            pad_length = len(prices) - len(hurst_values)
            features['hurst_exponent'] = pd.Series([np.nan] * pad_length + hurst_values)
            features['hurst_regime'] = pd.Series([np.nan] * pad_length + regime_values)
            
            return features
            
        except Exception as e:
            logger.warning(f"Hurst computation failed: {e}")
            return {}

    def _compute_fractal_features(self, prices: np.ndarray, window: int) -> Dict[str, pd.Series]:
        """Compute fractal dimension features"""
        features = {}
        
        try:
            fd_values = []
            
            for i in range(len(prices) - window + 1):
                window_data = prices[i:i+window]
                if len(window_data) > 10:
                    # Higuchi fractal dimension
                    fd = self._compute_higuchi_fractal_dimension(window_data)
                    fd_values.append(fd)
                else:
                    fd_values.append(np.nan)
            
            pad_length = len(prices) - len(fd_values)
            features['fractal_dimension'] = pd.Series([np.nan] * pad_length + fd_values)
            
            return features
            
        except Exception as e:
            logger.warning(f"Fractal computation failed: {e}")
            return {}

    def _compute_spectral_features(self, prices: np.ndarray, window: int) -> Dict[str, pd.Series]:
        """Compute spectral analysis features"""
        features = {}
        
        try:
            dominant_freqs = []
            spectral_entropy = []
            spectral_centroid = []
            
            for i in range(len(prices) - window + 1):
                window_data = prices[i:i+window]
                window_data = window_data[~np.isnan(window_data)]
                
                if len(window_data) >= 32:
                    # Remove trend
                    detrended = signal.detrend(window_data)
                    
                    # Compute FFT
                    fft_result = fft(detrended)
                    freqs = fftfreq(len(detrended))
                    
                    # Get magnitude spectrum
                    magnitude = np.abs(fft_result)
                    positive_freqs = freqs > 0
                    magnitude = magnitude[positive_freqs]
                    
                    if len(magnitude) > 0:
                        # Dominant frequency
                        dominant_idx = np.argmax(magnitude)
                        dominant_freq = freqs[positive_freqs][dominant_idx]
                        dominant_freqs.append(dominant_freq)
                        
                        # Spectral entropy
                        prob = magnitude / np.sum(magnitude)
                        prob = prob[prob > 0]
                        entropy = -np.sum(prob * np.log(prob))
                        spectral_entropy.append(entropy)
                        
                        # Spectral centroid
                        centroid = np.sum(freqs[positive_freqs] * magnitude) / np.sum(magnitude)
                        spectral_centroid.append(centroid)
                    else:
                        dominant_freqs.append(np.nan)
                        spectral_entropy.append(np.nan)
                        spectral_centroid.append(np.nan)
                else:
                    dominant_freqs.append(np.nan)
                    spectral_entropy.append(np.nan)
                    spectral_centroid.append(np.nan)
            
            pad_length = len(prices) - len(dominant_freqs)
            features['spectral_dominant_freq'] = pd.Series([np.nan] * pad_length + dominant_freqs)
            features['spectral_entropy'] = pd.Series([np.nan] * pad_length + spectral_entropy)
            features['spectral_centroid'] = pd.Series([np.nan] * pad_length + spectral_centroid)
            
            return features
            
        except Exception as e:
            logger.warning(f"Spectral computation failed: {e}")
            return {}

    def _compute_wavelet_features(self, prices: np.ndarray, level: int) -> Dict[str, pd.Series]:
        """Compute wavelet transform features"""
        features = {}
        
        try:
            if len(prices) < 2 ** level:
                return features
            
            # Discrete Wavelet Transform
            coeffs = pywt.wavedec(prices, 'db4', level=level)
            
            # Energy distribution across levels
            total_energy = np.sum(prices ** 2)
            
            for i, coeff in enumerate(coeffs):
                level_energy = np.sum(coeff ** 2)
                relative_energy = level_energy / total_energy if total_energy > 0 else 0
                
                # Create constant series for each feature
                features[f'wavelet_energy_level_{i}'] = pd.Series([relative_energy] * len(prices))
                features[f'wavelet_std_level_{i}'] = pd.Series([np.std(coeff)] * len(prices))
            
            # Wavelet entropy
            energy_levels = [np.sum(coeff ** 2) for coeff in coeffs]
            total_energy = np.sum(energy_levels)
            
            if total_energy > 0:
                probabilities = [e / total_energy for e in energy_levels]
                probabilities = [p for p in probabilities if p > 0]
                wavelet_entropy = -np.sum(probabilities * np.log(probabilities))
                features['wavelet_entropy'] = pd.Series([wavelet_entropy] * len(prices))
            
            return features
            
        except Exception as e:
            logger.warning(f"Wavelet computation failed: {e}")
            return {}

    def _compute_regime_features(self, prices: np.ndarray) -> Dict[str, pd.Series]:
        """Compute market regime detection features"""
        features = {}
        
        try:
            returns = self._compute_returns(prices)
            
            # Volatility regime
            volatility = pd.Series(returns).rolling(20).std()
            vol_threshold_high = volatility.quantile(0.7)
            vol_threshold_low = volatility.quantile(0.3)
            
            features['high_vol_regime'] = (volatility > vol_threshold_high).astype(float)
            features['low_vol_regime'] = (volatility < vol_threshold_low).astype(float)
            
            # Trend regime
            ma_short = pd.Series(prices).rolling(10).mean()
            ma_long = pd.Series(prices).rolling(30).mean()
            features['trend_strength'] = (ma_short - ma_long) / pd.Series(prices).rolling(30).std()
            features['trend_direction'] = np.sign(ma_short - ma_long)
            
            # Mean reversion regime
            zscore = (prices - pd.Series(prices).rolling(30).mean()) / pd.Series(prices).rolling(30).std()
            features['mean_reversion_potential'] = np.abs(zscore)
            features['mean_reversion_signal'] = (np.abs(zscore) > 2).astype(float)
            
            # Combined regime classification
            def classify_regime(row):
                vol_high = row['high_vol_regime'] if 'high_vol_regime' in row else 0
                trend_str = row['trend_strength'] if 'trend_strength' in row else 0
                mr_potential = row['mean_reversion_potential'] if 'mean_reversion_potential' in row else 0
                
                if vol_high > 0.5 and abs(trend_str) > 1:
                    return 3  # Trending volatile
                elif vol_high > 0.5:
                    return 2  # Volatile ranging
                elif abs(trend_str) > 1:
                    return 1  # Trending calm
                elif mr_potential > 2:
                    return -1  # Mean-reverting
                else:
                    return 0  # Random
            
            # This would need to be applied to each row
            # For simplicity, we'll compute a simplified version
            regime_simple = []
            for i in range(len(prices)):
                if i < 30:
                    regime_simple.append(0)
                    continue
                
                vol = volatility.iloc[i] if i < len(volatility) else 0
                ts = features['trend_strength'].iloc[i] if i < len(features['trend_strength']) else 0
                mr = features['mean_reversion_potential'].iloc[i] if i < len(features['mean_reversion_potential']) else 0
                
                if vol > vol_threshold_high and abs(ts) > 1:
                    regime_simple.append(3)
                elif vol > vol_threshold_high:
                    regime_simple.append(2)
                elif abs(ts) > 1:
                    regime_simple.append(1)
                elif mr > 2:
                    regime_simple.append(-1)
                else:
                    regime_simple.append(0)
            
            features['market_regime'] = pd.Series(regime_simple)
            
            return features
            
        except Exception as e:
            logger.warning(f"Regime computation failed: {e}")
            return {}

    # ==================== CORE COMPUTATIONAL METHODS ====================

    @staticmethod
    @jit(nopython=True)
    def _compute_returns(prices: np.ndarray) -> np.ndarray:
        """Compute percentage returns"""
        returns = np.zeros_like(prices)
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
        return returns

    @staticmethod
    @jit(nopython=True)
    def _compute_log_returns(prices: np.ndarray) -> np.ndarray:
        """Compute log returns"""
        log_returns = np.zeros_like(prices)
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                log_returns[i] = np.log(prices[i] / prices[i-1])
        return log_returns

    @staticmethod
    def _compute_acceleration(prices: np.ndarray, window: int = 5) -> np.ndarray:
        """Compute price acceleration (second derivative)"""
        if len(prices) < window + 1:
            return np.zeros_like(prices)
        
        acceleration = np.zeros_like(prices)
        for i in range(window, len(prices)):
            x = np.arange(window)
            y = prices[i-window:i]
            if len(y) == window and not np.any(np.isnan(y)):
                coeffs = np.polyfit(x, y, 2)  # Quadratic fit
                acceleration[i] = 2 * coeffs[0]  # Second derivative
        return acceleration

    @staticmethod
    def _compute_jerk(prices: np.ndarray, window: int = 5) -> np.ndarray:
        """Compute price jerk (third derivative)"""
        if len(prices) < window + 2:
            return np.zeros_like(prices)
        
        jerk = np.zeros_like(prices)
        for i in range(window, len(prices)):
            x = np.arange(window)
            y = prices[i-window:i]
            if len(y) == window and not np.any(np.isnan(y)):
                coeffs = np.polyfit(x, y, 3)  # Cubic fit
                jerk[i] = 6 * coeffs[0]  # Third derivative
        return jerk

    def _compute_variance_ratio(self, returns: np.ndarray, window: int) -> pd.Series:
        """Compute variance ratio test for random walk"""
        vr_values = []
        for i in range(len(returns) - window + 1):
            window_returns = returns[i:i+window]
            if len(window_returns) > 1:
                # Variance of 1-period returns
                var_1 = np.var(window_returns)
                
                # Variance of 2-period returns
                returns_2 = window_returns[:-1] + window_returns[1:]
                var_2 = np.var(returns_2) if len(returns_2) > 1 else np.nan
                
                vr = var_2 / (2 * var_1) if var_1 != 0 else np.nan
                vr_values.append(vr)
            else:
                vr_values.append(np.nan)
        
        pad_length = len(returns) - len(vr_values)
        return pd.Series([np.nan] * pad_length + vr_values)

    def _compute_autocorrelation(self, returns: np.ndarray, lag: int, window: int) -> pd.Series:
        """Compute rolling autocorrelation"""
        acf_values = []
        for i in range(len(returns) - window + 1):
            window_data = returns[i:i+window]
            if len(window_data) > lag and not np.any(np.isnan(window_data)):
                if i >= lag:
                    corr = np.corrcoef(window_data[:-lag], window_data[lag:])[0, 1]
                    acf_values.append(corr if not np.isnan(corr) else 0.0)
                else:
                    acf_values.append(np.nan)
            else:
                acf_values.append(np.nan)
        
        pad_length = len(returns) - len(acf_values)
        return pd.Series([np.nan] * pad_length + acf_values)

    def _compute_entropy_features(self, prices: np.ndarray, window: int) -> Dict[str, pd.Series]:
        """Compute entropy-based features"""
        features = {}
        
        if len(prices) < window:
            return features
        
        # Sample entropy (approximate)
        sample_entropy = []
        for i in range(len(prices) - window + 1):
            window_data = prices[i:i+window]
            if len(window_data) > 2:
                # Simple entropy approximation using histogram
                hist, _ = np.histogram(window_data, bins=10, density=True)
                hist = hist[hist > 0]  # Remove zero bins
                entropy = -np.sum(hist * np.log(hist))
                sample_entropy.append(entropy)
            else:
                sample_entropy.append(np.nan)
        
        pad_length = len(prices) - len(sample_entropy)
        features['sample_entropy'] = pd.Series([np.nan] * pad_length + sample_entropy)
        
        return features

    def _compute_normality_tests(self, returns: np.ndarray, window: int) -> Dict[str, pd.Series]:
        """Compute rolling normality tests"""
        features = {}
        
        if len(returns) < window:
            return features
        
        jarque_bera = []
        shapiro_p = []
        
        for i in range(len(returns) - window + 1):
            window_data = returns[i:i+window]
            if len(window_data) > 3 and not np.any(np.isnan(window_data)):
                try:
                    # Jarque-Bera test
                    jb_stat, jb_p = stats.jarque_bera(window_data)
                    jarque_bera.append(jb_p)
                    
                    # Shapiro-Wilk test (for smaller samples)
                    if len(window_data) < 5000:
                        sh_stat, sh_p = stats.shapiro(window_data)
                        shapiro_p.append(sh_p)
                    else:
                        shapiro_p.append(np.nan)
                except:
                    jarque_bera.append(np.nan)
                    shapiro_p.append(np.nan)
            else:
                jarque_bera.append(np.nan)
                shapiro_p.append(np.nan)
        
        pad_length = len(returns) - len(jarque_bera)
        features['jarque_bera_p'] = pd.Series([np.nan] * pad_length + jarque_bera)
        features['shapiro_p'] = pd.Series([np.nan] * pad_length + shapiro_p)
        
        return features

    def _compute_stationarity_tests(self, prices: np.ndarray, window: int) -> Dict[str, pd.Series]:
        """Compute rolling stationarity tests"""
        features = {}
        
        if len(prices) < window:
            return features
        
        # Simplified stationarity measure using rolling statistics
        rolling_mean = pd.Series(prices).rolling(window).mean()
        rolling_std = pd.Series(prices).rolling(window).std()
        
        # Coefficient of variation of rolling statistics as stationarity proxy
        mean_cv = rolling_std / np.abs(rolling_mean)
        features['stationarity_proxy'] = 1 / (1 + mean_cv)  # Higher = more stationary
        
        return features

    @staticmethod
    def _compute_parkinson_volatility(high: np.ndarray, low: np.ndarray, window: int) -> pd.Series:
        """Compute Parkinson volatility estimator"""
        parkinson = np.log(high / low) ** 2
        parkinson_vol = pd.Series(parkinson).rolling(window).mean() * np.sqrt(252 / (4 * np.log(2)))
        return parkinson_vol

    @staticmethod
    def _compute_garman_klass_volatility(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> pd.Series:
        """Compute Garman-Klass volatility estimator"""
        gk = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
        gk_vol = pd.Series(gk).rolling(window).mean() * np.sqrt(252)
        return gk_vol

    @staticmethod
    def _compute_realized_volatility(high: np.ndarray, low: np.ndarray, window: int) -> pd.Series:
        """Compute realized volatility from high-low range"""
        daily_range = np.log(high / low)
        realized_vol = pd.Series(daily_range).rolling(window).std() * np.sqrt(252)
        return realized_vol

    @staticmethod
    def _compute_slope(series: np.ndarray, window: int) -> pd.Series:
        """Compute rolling slope of a series"""
        slopes = []
        for i in range(len(series)):
            if i < window:
                slopes.append(np.nan)
            else:
                y = series[i-window:i]
                x = np.arange(len(y))
                if not np.any(np.isnan(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=range(len(series)))

    @staticmethod
    def _compute_volume_vwap(close: np.ndarray, volume: np.ndarray, window: int) -> pd.Series:
        """Compute volume-weighted average price"""
        vwap = pd.Series(close * volume).rolling(window).sum() / pd.Series(volume).rolling(window).sum()
        return vwap

    @staticmethod
    def _compute_higuchi_fractal_dimension(data: np.ndarray, k_max: int = 10) -> float:
        """Compute Higuchi fractal dimension"""
        try:
            N = len(data)
            L = []
            
            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    # Create segments
                    segments = [data[i] for i in range(m, N, k)]
                    if len(segments) > 1:
                        # Calculate length of segment
                        Lkm = np.sum(np.abs(np.diff(segments))) * (N - 1) / (len(segments) * k)
                        Lk += Lkm
                L.append(Lk / k)
            
            # Fit line to log-log plot
            if len(L) > 1:
                x = np.log(np.arange(1, k_max + 1))
                y = np.log(np.array(L))
                if not np.any(np.isnan(y)) and not np.any(np.isinf(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    return -slope
            return 1.5  # Default value for random walk
        except:
            return 1.5

    def _compute_correlation_features(self, returns: np.ndarray, lookback: int) -> Dict[str, pd.Series]:
        """Compute correlation structure features"""
        features = {}
        
        if len(returns) < lookback:
            return features
        
        # Autocorrelation of absolute returns (volatility clustering)
        abs_returns = np.abs(returns)
        acf_abs = []
        
        for i in range(len(returns) - lookback + 1):
            window_data = abs_returns[i:i+lookback]
            if len(window_data) > 1:
                # Lag-1 autocorrelation
                if i > 0:
                    corr = np.corrcoef(window_data[:-1], window_data[1:])[0, 1]
                    acf_abs.append(corr if not np.isnan(corr) else 0.0)
                else:
                    acf_abs.append(np.nan)
            else:
                acf_abs.append(np.nan)
        
        pad_length = len(returns) - len(acf_abs)
        features['vol_clustering_acf'] = pd.Series([np.nan] * pad_length + acf_abs)
        
        return features

    def _compute_nonlinear_features(self, prices: np.ndarray) -> Dict[str, pd.Series]:
        """Compute nonlinear dynamics features"""
        features = {}
        
        try:
            # Lyapunov exponent approximation (simplified)
            # This is a placeholder for more sophisticated nonlinear analysis
            returns = self._compute_returns(prices)
            
            # Simple nonlinearity measure using rolling skewness of returns
            nonlinearity = pd.Series(returns).rolling(20).skew() ** 2  # Squared skewness
            features['nonlinearity_measure'] = nonlinearity
            
            return features
            
        except Exception as e:
            logger.warning(f"Nonlinear feature computation failed: {e}")
            return {}

    def _compute_risk_features(self, returns: np.ndarray) -> Dict[str, pd.Series]:
        """Compute risk measurement features"""
        features = {}
        
        if len(returns) < 20:
            return features
        
        # Value at Risk (Historical)
        var_95 = pd.Series(returns).rolling(20).quantile(0.05)
        features['VaR_95'] = var_95
        
        # Conditional VaR (Expected Shortfall)
        def compute_cvar(window_returns):
            if len(window_returns) > 0:
                var = np.quantile(window_returns, 0.05)
                cvar = window_returns[window_returns <= var].mean()
                return cvar if not np.isnan(cvar) else 0.0
            return np.nan
        
        cvar_95 = pd.Series(returns).rolling(20).apply(compute_cvar, raw=True)
        features['CVaR_95'] = cvar_95
        
        # Maximum Drawdown
        features['max_drawdown'] = self._compute_rolling_max_drawdown(returns, window=20)
        
        # Sharpe Ratio (annualized)
        rolling_mean = pd.Series(returns).rolling(20).mean()
        rolling_std = pd.Series(returns).rolling(20).std()
        features['sharpe_ratio'] = (rolling_mean / rolling_std) * np.sqrt(252)
        
        return features

    @staticmethod
    def _compute_rolling_max_drawdown(returns: np.ndarray, window: int) -> pd.Series:
        """Compute rolling maximum drawdown"""
        drawdowns = []
        
        for i in range(len(returns) - window + 1):
            window_returns = returns[i:i+window]
            cumulative = np.cumprod(1 + window_returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            drawdowns.append(max_drawdown)
        
        pad_length = len(returns) - len(drawdowns)
        return pd.Series([np.nan] * pad_length + drawdowns)

    def _handle_missing_values(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Forward fill, then backward fill
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaN, fill with 0
        features_df = features_df.fillna(0)
        
        return features_df

    def _select_features(self, features_df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most important features"""
        try:
            # Remove highly correlated features
            corr_matrix = features_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.config.correlation_threshold)]
            
            features_reduced = features_df.drop(columns=to_drop)
            logger.info(f"Dropped {len(to_drop)} highly correlated features")
            
            # Further feature selection if still too many features
            if len(features_reduced.columns) > self.config.max_features:
                selector = SelectKBest(score_func=f_regression, k=self.config.max_features)
                X_selected = selector.fit_transform(features_reduced, target)
                selected_features = features_reduced.columns[selector.get_support()]
                features_reduced = features_reduced[selected_features]
                logger.info(f"Selected {len(selected_features)} best features")
            
            return features_reduced
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return features_df

    def _normalize_features(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """Normalize features using specified method"""
        try:
            if self.config.normalization_method == "standard":
                scaler = StandardScaler()
            elif self.config.normalization_method == "robust":
                scaler = RobustScaler()
            elif self.config.normalization_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()  # Default
            
            features_normalized = scaler.fit_transform(features_df)
            features_df_normalized = pd.DataFrame(features_normalized, 
                                               columns=features_df.columns, 
                                               index=features_df.index)
            
            return features_df_normalized, scaler
            
        except Exception as e:
            logger.error(f"Feature normalization failed: {e}")
            return features_df, None

    def _compute_feature_importance(self, features_df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Compute feature importance scores"""
        try:
            # Use mutual information for feature importance
            mi_scores = mutual_info_regression(features_df, target, random_state=42)
            importance_dict = dict(zip(features_df.columns, mi_scores))
            
            # Normalize scores to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v / total_importance for k, v in importance_dict.items()}
            
            self.feature_importance = importance_dict
            return importance_dict
            
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            return {}

    def _update_performance_metrics(self, features_df: pd.DataFrame):
        """Update performance metrics for monitoring"""
        try:
            # Track feature statistics
            for column in features_df.columns:
                if column not in self.feature_stats:
                    self.feature_stats[column] = deque(maxlen=1000)
                
                self.feature_stats[column].extend(features_df[column].values)
                
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of computed features"""
        return {
            'total_features': len(self.feature_importance),
            'feature_importance': dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]),
            'feature_stats': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in self.feature_stats.items()},
            'cache_size': len(self.feature_cache)
        }

# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    config = FeatureConfig(
        enable_technical=True,
        enable_statistical=True,
        enable_volatility=True,
        enable_advanced=True
    )
    
    engineer = AdvancedFeatureEngineer(config)
    
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='H')
    np.random.seed(42)
    
    # Create realistic Forex data
    prices = [1.1000]
    for i in range(1, len(dates)):
        change = np.random.normal(0.0001, 0.005)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.normal(1000000, 100000, len(dates))
    })
    sample_data.set_index('date', inplace=True)
    
    # Compute features
    feature_set = engineer.compute_features(sample_data)
    
    print(f"Computed {len(feature_set.features.columns)} features")
    print(f"Feature groups: {feature_set.metadata['feature_groups_summary']}")
    
    # Show top 10 most important features
    top_features = sorted(feature_set.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 features:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
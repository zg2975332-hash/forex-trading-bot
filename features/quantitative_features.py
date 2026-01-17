"""
Quantitative Features Engine for FOREX TRADING BOT
Advanced feature engineering for quantitative trading strategies
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import talib
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import numba
from numba import jit, prange
from collections import deque, defaultdict
import pywt
from hurst import compute_Hc
from typing import Union
import requests
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
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

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Technical indicators
    enable_technical: bool = True
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    volatility_periods: List[int] = field(default_factory=lambda: [10, 20, 30])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Statistical features
    enable_statistical: bool = True
    rolling_window: int = 20
    zscore_threshold: float = 2.0
    entropy_window: int = 50
    
    # Spectral analysis
    enable_spectral: bool = True
    fft_window: int = 256
    dominant_frequencies: int = 5
    
    # Wavelet features
    enable_wavelet: bool = True
    wavelet_type: str = 'db4'
    decomposition_level: int = 4
    
    # Market microstructure
    enable_microstructure: bool = True
    tick_window: int = 100
    volume_profile_bins: int = 20
    
    # Advanced features
    enable_advanced: bool = True
    hurst_window: int = 100
    correlation_lookback: int = 50

@dataclass
class FeatureSet:
    """Computed feature set"""
    features: pd.DataFrame
    feature_importance: Dict[str, float]
    feature_groups: Dict[FeatureType, List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

class QuantitativeFeatures:
    """
    Advanced quantitative feature engineering for Forex trading
    Implements 100+ features across multiple domains
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_cache = {}
        self.performance_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_correlation = {}
        
        # Statistical normalization
        self.scalers = {}
        self.feature_stats = {}
        
        logger.info("QuantitativeFeatures engine initialized")

    def compute_features(self, data: pd.DataFrame, target: pd.Series = None) -> FeatureSet:
        """
        Compute comprehensive feature set from price data
        """
        try:
            logger.info(f"Computing features for {len(data)} data points")
            
            features_dict = {}
            feature_groups = defaultdict(list)
            
            # 1. Technical Indicators
            if self.config.enable_technical:
                technical_features = self._compute_technical_features(data)
                features_dict.update(technical_features)
                feature_groups[FeatureType.TECHNICAL].extend(technical_features.keys())
                logger.info(f"Computed {len(technical_features)} technical features")
            
            # 2. Statistical Features
            if self.config.enable_statistical:
                statistical_features = self._compute_statistical_features(data)
                features_dict.update(statistical_features)
                feature_groups[FeatureType.STATISTICAL].extend(statistical_features.keys())
                logger.info(f"Computed {len(statistical_features)} statistical features")
            
            # 3. Spectral Features
            if self.config.enable_spectral:
                spectral_features = self._compute_spectral_features(data)
                features_dict.update(spectral_features)
                feature_groups[FeatureType.SPECTRAL].extend(spectral_features.keys())
                logger.info(f"Computed {len(spectral_features)} spectral features")
            
            # 4. Wavelet Features
            if self.config.enable_wavelet:
                wavelet_features = self._compute_wavelet_features(data)
                features_dict.update(wavelet_features)
                feature_groups[FeatureType.WAVELET].extend(wavelet_features.keys())
                logger.info(f"Computed {len(wavelet_features)} wavelet features")
            
            # 5. Market Microstructure Features
            if self.config.enable_microstructure:
                microstructure_features = self._compute_microstructure_features(data)
                features_dict.update(microstructure_features)
                feature_groups[FeatureType.MICROSTRUCTURE].extend(microstructure_features.keys())
                logger.info(f"Computed {len(microstructure_features)} microstructure features")
            
            # 6. Advanced Quantitative Features
            if self.config.enable_advanced:
                advanced_features = self._compute_advanced_features(data)
                features_dict.update(advanced_features)
                feature_groups[FeatureType.NOVELTY].extend(advanced_features.keys())
                logger.info(f"Computed {len(advanced_features)} advanced features")
            
            # Create feature DataFrame
            features_df = pd.DataFrame(features_dict, index=data.index)
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
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
                    'data_points': len(data)
                }
            )
            
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            raise

    def _compute_technical_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute technical indicators"""
        try:
            features = {}
            prices = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data.get('volume', pd.Series(np.ones(len(data))).values)
            
            # Moving Averages
            for period in self.config.ma_periods:
                if len(prices) >= period:
                    # Simple Moving Average
                    features[f'SMA_{period}'] = talib.SMA(prices, timeperiod=period)
                    
                    # Exponential Moving Average
                    features[f'EMA_{period}'] = talib.EMA(prices, timeperiod=period)
                    
                    # Moving Average Convergence Divergence (MACD)
                    if period == self.config.macd_fast:
                        macd, macd_signal, macd_hist = talib.MACD(
                            prices, 
                            fastperiod=self.config.macd_fast,
                            slowperiod=self.config.macd_slow,
                            signalperiod=self.config.macd_signal
                        )
                        features['MACD'] = macd
                        features['MACD_Signal'] = macd_signal
                        features['MACD_Histogram'] = macd_hist
            
            # Relative Strength Index (RSI)
            rsi = talib.RSI(prices, timeperiod=self.config.rsi_period)
            features['RSI'] = rsi
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                prices, 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            features['BB_Upper'] = bb_upper
            features['BB_Middle'] = bb_middle
            features['BB_Lower'] = bb_lower
            features['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            features['BB_Position'] = (prices - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                high, low, prices,
                fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
            )
            features['Stoch_K'] = slowk
            features['Stoch_D'] = slowd
            
            # Average True Range (ATR)
            atr = talib.ATR(high, low, prices, timeperiod=14)
            features['ATR'] = atr
            features['ATR_Ratio'] = atr / prices
            
            # Commodity Channel Index (CCI)
            cci = talib.CCI(high, low, prices, timeperiod=20)
            features['CCI'] = cci
            
            # Williams %R
            willr = talib.WILLR(high, low, prices, timeperiod=14)
            features['Williams_R'] = willr
            
            # Rate of Change (ROC)
            roc = talib.ROC(prices, timeperiod=10)
            features['ROC'] = roc
            
            # Money Flow Index (MFI)
            if 'volume' in data.columns:
                mfi = talib.MFI(high, low, prices, volume, timeperiod=14)
                features['MFI'] = mfi
            
            # On Balance Volume (OBV)
            if 'volume' in data.columns:
                obv = talib.OBV(prices, volume)
                features['OBV'] = obv
            
            # Price-based features
            features['Price_Returns'] = self._compute_returns(prices)
            features['Log_Returns'] = self._compute_log_returns(prices)
            
            # Volatility features
            for period in self.config.volatility_periods:
                if len(prices) > period:
                    returns = self._compute_returns(prices)
                    volatility = returns.rolling(window=period).std() * np.sqrt(252)
                    features[f'Volatility_{period}'] = volatility
            
            # Trend strength
            features['ADX'] = talib.ADX(high, low, prices, timeperiod=14)
            features['ADXR'] = talib.ADXR(high, low, prices, timeperiod=14)
            
            # Momentum indicators
            features['Momentum'] = talib.MOM(prices, timeperiod=10)
            
            # Remove any features with all NaN values
            features = {k: v for k, v in features.items() if not np.all(np.isnan(v))}
            
            return features
            
        except Exception as e:
            logger.error(f"Technical feature computation failed: {e}")
            return {}

    def _compute_statistical_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute statistical features"""
        try:
            features = {}
            prices = data['close'].values
            returns = self._compute_returns(prices)
            
            window = self.config.rolling_window
            
            # Rolling statistics
            if len(prices) > window:
                # Central moments
                features['Rolling_Mean'] = pd.Series(prices).rolling(window=window).mean()
                features['Rolling_Std'] = pd.Series(prices).rolling(window=window).std()
                features['Rolling_Skew'] = pd.Series(prices).rolling(window=window).skew()
                features['Rolling_Kurtosis'] = pd.Series(prices).rolling(window=window).kurt()
                
                # Quantiles
                features['Rolling_Q25'] = pd.Series(prices).rolling(window=window).quantile(0.25)
                features['Rolling_Q75'] = pd.Series(prices).rolling(window=window).quantile(0.75)
                features['Rolling_IQR'] = features['Rolling_Q75'] - features['Rolling_Q25']
                
                # Z-score based features
                zscore = (prices - features['Rolling_Mean']) / features['Rolling_Std']
                features['Z_Score'] = zscore
                features['Z_Score_Abs'] = np.abs(zscore)
                features['Z_Score_Squared'] = zscore ** 2
                
                # Extreme value indicators
                features['Extreme_High'] = (zscore > self.config.zscore_threshold).astype(float)
                features['Extreme_Low'] = (zscore < -self.config.zscore_threshold).astype(float)
            
            # Return distribution features
            if len(returns) > window:
                # Higher moments of returns
                features['Return_Skew'] = pd.Series(returns).rolling(window=window).skew()
                features['Return_Kurtosis'] = pd.Series(returns).rolling(window=window).kurt()
                
                # Variance ratio (random walk test)
                features['Variance_Ratio'] = self._compute_variance_ratio(returns, window)
            
            # Autocorrelation features
            autocorr_features = self._compute_autocorrelation_features(returns, max_lag=10)
            features.update(autocorr_features)
            
            # Entropy and information theory features
            entropy_features = self._compute_entropy_features(prices, window=self.config.entropy_window)
            features.update(entropy_features)
            
            # Normality tests
            if len(returns) > 30:
                features['Jarque_Bera'] = self._compute_rolling_jarque_bera(returns, window=30)
                features['Shapiro_Wilk'] = self._compute_rolling_shapiro_wilk(returns, window=30)
            
            # Stationarity tests
            adf_features = self._compute_rolling_adf(prices, window=50)
            features.update(adf_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Statistical feature computation failed: {e}")
            return {}

    def _compute_spectral_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute spectral analysis features using FFT"""
        try:
            features = {}
            prices = data['close'].values
            
            if len(prices) < self.config.fft_window:
                return features
            
            # Compute rolling FFT features
            fft_results = self._compute_rolling_fft(prices, window=self.config.fft_window)
            features.update(fft_results)
            
            # Spectral density features
            spectral_features = self._compute_spectral_density_features(prices)
            features.update(spectral_features)
            
            # Periodogram features
            periodogram_features = self._compute_periodogram_features(prices)
            features.update(periodogram_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Spectral feature computation failed: {e}")
            return {}

    def _compute_wavelet_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute wavelet transform features"""
        try:
            features = {}
            prices = data['close'].values
            
            if len(prices) < 2 ** self.config.decomposition_level:
                return features
            
            # Discrete Wavelet Transform
            wavelet_features = self._compute_dwt_features(
                prices, 
                wavelet=self.config.wavelet_type, 
                level=self.config.decomposition_level
            )
            features.update(wavelet_features)
            
            # Wavelet energy features
            energy_features = self._compute_wavelet_energy_features(prices)
            features.update(energy_features)
            
            # Wavelet entropy features
            entropy_features = self._compute_wavelet_entropy_features(prices)
            features.update(entropy_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Wavelet feature computation failed: {e}")
            return {}

    def _compute_microstructure_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute market microstructure features"""
        try:
            features = {}
            
            # Requires high-frequency data with bid/ask prices
            if all(col in data.columns for col in ['bid', 'ask', 'volume']):
                bid = data['bid'].values
                ask = data['ask'].values
                volume = data['volume'].values
                
                # Spread features
                spread = ask - bid
                features['Bid_Ask_Spread'] = spread
                features['Relative_Spread'] = spread / ((bid + ask) / 2)
                
                # Mid-price features
                mid_price = (bid + ask) / 2
                features['Mid_Price_Returns'] = self._compute_returns(mid_price)
                
                # Microstructure noise
                noise_features = self._compute_microstructure_noise(bid, ask, volume)
                features.update(noise_features)
                
                # Volume-based features
                volume_features = self._compute_volume_features(volume, self.config.tick_window)
                features.update(volume_features)
                
                # Order imbalance
                if 'bid_volume' in data.columns and 'ask_volume' in data.columns:
                    imbalance_features = self._compute_order_imbalance(
                        data['bid_volume'].values, 
                        data['ask_volume'].values
                    )
                    features.update(imbalance_features)
            
            # Price impact features (simplified)
            if 'volume' in data.columns:
                price_impact = self._compute_price_impact(data['close'].values, data['volume'].values)
                features['Price_Impact'] = price_impact
            
            # Volatility clustering
            if len(data) > 50:
                vol_clustering = self._compute_volatility_clustering(data['close'].values)
                features['Volatility_Clustering'] = vol_clustering
            
            return features
            
        except Exception as e:
            logger.error(f"Microstructure feature computation failed: {e}")
            return {}

    def _compute_advanced_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute advanced quantitative features"""
        try:
            features = {}
            prices = data['close'].values
            returns = self._compute_returns(prices)
            
            # Hurst Exponent
            if len(prices) >= self.config.hurst_window:
                hurst_features = self._compute_hurst_features(prices, window=self.config.hurst_window)
                features.update(hurst_features)
            
            # Fractal dimensions
            fractal_features = self._compute_fractal_dimensions(prices)
            features.update(fractal_features)
            
            # Regime detection features
            regime_features = self._compute_regime_features(prices)
            features.update(regime_features)
            
            # Correlation structure
            correlation_features = self._compute_correlation_features(returns, self.config.correlation_lookback)
            features.update(correlation_features)
            
            # PCA-based features
            pca_features = self._compute_pca_features(data)
            features.update(pca_features)
            
            # Nonlinear dynamics features
            nonlinear_features = self._compute_nonlinear_features(prices)
            features.update(nonlinear_features)
            
            # Risk measures
            risk_features = self._compute_risk_features(returns)
            features.update(risk_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Advanced feature computation failed: {e}")
            return {}

    # ==================== MISSING METHODS IMPLEMENTATION ====================

    def _compute_spectral_density_features(self, prices: np.ndarray) -> Dict[str, pd.Series]:
        """Compute spectral density features"""
        features = {}
        
        try:
            window = 100
            if len(prices) < window:
                return features
            
            spectral_rolloff = []
            spectral_flux = []
            spectral_contrast = []
            
            for i in range(len(prices) - window + 1):
                window_data = prices[i:i+window]
                window_data = window_data[~np.isnan(window_data)]
                
                if len(window_data) >= 32:
                    # Compute periodogram
                    freqs, psd = signal.periodogram(window_data - np.mean(window_data))
                    
                    if len(psd) > 0:
                        # Spectral rolloff (85th percentile)
                        cumulative_energy = np.cumsum(psd) / np.sum(psd)
                        rolloff_idx = np.argmax(cumulative_energy >= 0.85)
                        spectral_rolloff.append(freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1])
                        
                        # Spectral flux (change from previous window)
                        if i > 0 and len(spectral_flux) > 0:
                            prev_psd = features.get('Spectral_Flux', [0])[-1]
                            flux = np.sum((psd - prev_psd) ** 2)
                            spectral_flux.append(flux)
                        else:
                            spectral_flux.append(0.0)
                        
                        # Spectral contrast (high vs low frequencies)
                        mid_idx = len(psd) // 2
                        low_energy = np.mean(psd[:mid_idx])
                        high_energy = np.mean(psd[mid_idx:])
                        contrast = high_energy / low_energy if low_energy > 0 else 1.0
                        spectral_contrast.append(contrast)
                    else:
                        spectral_rolloff.append(np.nan)
                        spectral_flux.append(np.nan)
                        spectral_contrast.append(np.nan)
                else:
                    spectral_rolloff.append(np.nan)
                    spectral_flux.append(np.nan)
                    spectral_contrast.append(np.nan)
            
            # Pad with NaN
            pad_length = len(prices) - len(spectral_rolloff)
            features['Spectral_Rolloff'] = pd.Series([np.nan] * pad_length + spectral_rolloff)
            features['Spectral_Flux'] = pd.Series([np.nan] * pad_length + spectral_flux)
            features['Spectral_Contrast'] = pd.Series([np.nan] * pad_length + spectral_contrast)
            
        except Exception as e:
            logger.warning(f"Spectral density computation failed: {e}")
        
        return features

    def _compute_periodogram_features(self, prices: np.ndarray) -> Dict[str, pd.Series]:
        """Compute periodogram-based features"""
        features = {}
        
        try:
            window = 100
            if len(prices) < window:
                return features
            
            peak_frequencies = []
            peak_heights = []
            bandwidths = []
            
            for i in range(len(prices) - window + 1):
                window_data = prices[i:i+window]
                window_data = window_data[~np.isnan(window_data)]
                
                if len(window_data) >= 32:
                    freqs, psd = signal.periodogram(window_data - np.mean(window_data))
                    
                    if len(psd) > 0:
                        # Find dominant peak
                        peak_idx = np.argmax(psd)
                        peak_frequencies.append(freqs[peak_idx])
                        peak_heights.append(psd[peak_idx])
                        
                        # Bandwidth (width at half maximum)
                        half_max = psd[peak_idx] / 2
                        above_half = psd >= half_max
                        if np.any(above_half):
                            bandwidth = np.max(freqs[above_half]) - np.min(freqs[above_half])
                            bandwidths.append(bandwidth)
                        else:
                            bandwidths.append(0.0)
                    else:
                        peak_frequencies.append(np.nan)
                        peak_heights.append(np.nan)
                        bandwidths.append(np.nan)
                else:
                    peak_frequencies.append(np.nan)
                    peak_heights.append(np.nan)
                    bandwidths.append(np.nan)
            
            # Pad with NaN
            pad_length = len(prices) - len(peak_frequencies)
            features['Peak_Frequency'] = pd.Series([np.nan] * pad_length + peak_frequencies)
            features['Peak_Height'] = pd.Series([np.nan] * pad_length + peak_heights)
            features['Spectral_Bandwidth'] = pd.Series([np.nan] * pad_length + bandwidths)
            
        except Exception as e:
            logger.warning(f"Periodogram computation failed: {e}")
        
        return features

    def _compute_wavelet_energy_features(self, prices: np.ndarray) -> Dict[str, pd.Series]:
        """Compute wavelet energy features"""
        features = {}
        
        try:
            if len(prices) < 2 ** self.config.decomposition_level:
                return features
            
            # Perform DWT
            coeffs = pywt.wavedec(prices, self.config.wavelet_type, level=self.config.decomposition_level)
            
            # Compute energy for each level
            total_energy = np.sum(prices ** 2)
            
            for i, coeff in enumerate(coeffs):
                level_energy = np.sum(coeff ** 2)
                relative_energy = level_energy / total_energy if total_energy > 0 else 0.0
                
                # Create series with constant value
                features[f'Wavelet_Energy_Level_{i}'] = pd.Series([relative_energy] * len(prices))
                features[f'Wavelet_Energy_Ratio_{i}'] = pd.Series([relative_energy] * len(prices))
            
        except Exception as e:
            logger.warning(f"Wavelet energy computation failed: {e}")
        
        return features

    def _compute_wavelet_entropy_features(self, prices: np.ndarray) -> Dict[str, pd.Series]:
        """Compute wavelet entropy features"""
        features = {}
        
        try:
            if len(prices) < 2 ** self.config.decomposition_level:
                return features
            
            # Perform DWT
            coeffs = pywt.wavedec(prices, self.config.wavelet_type, level=self.config.decomposition_level)
            
            # Compute wavelet entropy
            energy_levels = [np.sum(coeff ** 2) for coeff in coeffs]
            total_energy = np.sum(energy_levels)
            
            if total_energy > 0:
                probabilities = [e / total_energy for e in energy_levels]
                probabilities = [p for p in probabilities if p > 0]  # Remove zeros
                wavelet_entropy = -np.sum(probabilities * np.log(probabilities))
                
                features['Wavelet_Entropy'] = pd.Series([wavelet_entropy] * len(prices))
            
        except Exception as e:
            logger.warning(f"Wavelet entropy computation failed: {e}")
        
        return features

def _compute_microstructure_noise(self, bid: np.ndarray, ask: np.ndarray, volume: np.ndarray) -> Dict[str, pd.Series]:
    """Compute microstructure noise features"""
    features = {}

    try:
        if len(bid) < 10:
            return features

        # Mid-price (average of bid/ask)
        mid_price = (bid + ask) / 2

        # Roll measure (microstructure noise)
        roll_measure = np.sqrt(np.maximum(0, -np.cov(mid_price[:-1], mid_price[1:])[0, 1]))
        features['Roll_Measure'] = pd.Series([roll_measure] * len(bid))

        # Spread-related noise
        spread = ask - bid
        features['Spread_Volatility'] = pd.Series(pd.Series(spread).rolling(window=10).std())

        # Volume imbalance indicator
        features['Volume_Imbalance'] = pd.Series(np.gradient(volume))

    except Exception as e:
        logger.warning(f"Microstructure noise computation failed: {e}")

    return features

                   
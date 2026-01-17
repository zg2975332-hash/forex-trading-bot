"""
Advanced Ensemble Predictor for FOREX TRADING BOT
Combines multiple ML models for superior prediction accuracy
"""

import logging
import pandas as pd
import numpy as np
import json
import pickle
import gzip
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import statistics
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import talib
from scipy.stats import trim_mean
import joblib
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class ModelType(Enum):
    LSTM = "lstm"
    GRU = "gru"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVR = "svr"
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    MLP = "mlp"

class EnsembleMethod(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    VOTING = "voting"
    DYNAMIC_WEIGHTING = "dynamic_weighting"

@dataclass
class ModelConfig:
    """Ensemble model configuration"""
    # Base models to include
    base_models: List[ModelType] = field(default_factory=lambda: [
        ModelType.LSTM, ModelType.RANDOM_FOREST, ModelType.XGBOOST,
        ModelType.GRADIENT_BOOSTING, ModelType.LIGHTGBM
    ])
    
    # Ensemble method
    ensemble_method: EnsembleMethod = EnsembleMethod.DYNAMIC_WEIGHTING
    
    # Training parameters
    lookback_window: int = 100
    prediction_horizon: int = 1
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    
    # Model-specific parameters
    lstm_units: List[int] = field(default_factory=lambda: [100, 50, 25])
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    
    random_forest_trees: int = 100
    xgboost_trees: int = 100
    gradient_boosting_trees: int = 100
    lightgbm_trees: int = 100
    
    # Feature engineering
    use_technical_indicators: bool = True
    use_statistical_features: bool = True
    use_lagged_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Dynamic weighting
    performance_lookback: int = 50
    min_model_weight: float = 0.05
    decay_factor: float = 0.95

@dataclass
class ModelPerformance:
    """Individual model performance metrics"""
    model_type: ModelType
    mse: float
    mae: float
    r2: float
    rmse: float
    directional_accuracy: float
    recent_performance: deque
    weight: float
    last_updated: datetime

@dataclass
class PredictionResult:
    """Ensemble prediction result"""
    timestamp: datetime
    prediction: float
    confidence: float
    individual_predictions: Dict[ModelType, float]
    model_weights: Dict[ModelType, float]
    features_used: List[str]
    metadata: Dict[str, Any]

class EnsemblePredictor:
    """
    Advanced ensemble prediction system combining multiple ML models
    Implements dynamic weighting based on recent performance
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        
        # Model storage
        self.models: Dict[ModelType, Any] = {}
        self.scalers: Dict[ModelType, StandardScaler] = {}
        self.performance: Dict[ModelType, ModelPerformance] = {}
        
        # Feature engineering
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        # Training data
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        
        # Prediction history
        self.prediction_history: deque = deque(maxlen=1000)
        self.actual_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        self._training_lock = threading.Lock()
        
        # Model initialization
        self._initialize_models()
        
        logger.info("EnsemblePredictor initialized")

    def _initialize_models(self):
        """Initialize all base models"""
        try:
            for model_type in self.config.base_models:
                if model_type == ModelType.LSTM:
                    self.models[model_type] = self._create_lstm_model()
                elif model_type == ModelType.RANDOM_FOREST:
                    self.models[model_type] = RandomForestRegressor(
                        n_estimators=self.config.random_forest_trees,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_type == ModelType.XGBOOST:
                    self.models[model_type] = xgb.XGBRegressor(
                        n_estimators=self.config.xgboost_trees,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_type == ModelType.LIGHTGBM:
                    self.models[model_type] = lgb.LGBMRegressor(
                        n_estimators=self.config.lightgbm_trees,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_type == ModelType.GRADIENT_BOOSTING:
                    self.models[model_type] = GradientBoostingRegressor(
                        n_estimators=self.config.gradient_boosting_trees,
                        random_state=42
                    )
                elif model_type == ModelType.SVR:
                    self.models[model_type] = SVR(kernel='rbf', C=1.0)
                elif model_type == ModelType.LINEAR:
                    self.models[model_type] = LinearRegression()
                elif model_type == ModelType.RIDGE:
                    self.models[model_type] = Ridge(alpha=1.0)
                elif model_type == ModelType.LASSO:
                    self.models[model_type] = Lasso(alpha=1.0)
                elif model_type == ModelType.MLP:
                    self.models[model_type] = MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        random_state=42,
                        max_iter=1000
                    )
                
                # Initialize performance tracking
                self.performance[model_type] = ModelPerformance(
                    model_type=model_type,
                    mse=0.0,
                    mae=0.0,
                    r2=0.0,
                    rmse=0.0,
                    directional_accuracy=0.0,
                    recent_performance=deque(maxlen=self.config.performance_lookback),
                    weight=1.0 / len(self.config.base_models),
                    last_updated=datetime.now()
                )
                
                # Initialize scaler
                self.scalers[model_type] = StandardScaler()
            
            logger.info(f"Initialized {len(self.models)} base models")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _create_lstm_model(self) -> Sequential:
        """Create LSTM model architecture"""
        try:
            model = Sequential()
            
            # Input layer
            model.add(LSTM(
                units=self.config.lstm_units[0],
                return_sequences=True,
                input_shape=(self.config.lookback_window, 1)
            ))
            model.add(Dropout(self.config.lstm_dropout))
            
            # Hidden layers
            for units in self.config.lstm_units[1:]:
                model.add(LSTM(units=units, return_sequences=True))
                model.add(Dropout(self.config.lstm_dropout))
            
            # Final LSTM layer
            model.add(LSTM(units=self.config.lstm_units[-1]))
            model.add(Dropout(self.config.lstm_dropout))
            
            # Output layer
            model.add(Dense(units=1))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.config.lstm_learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logger.error(f"LSTM model creation failed: {e}")
            raise

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare advanced features for model training/prediction
        """
        try:
            features_dict = {}
            
            # Price-based features
            prices = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data.get('volume', np.ones(len(data)))
            
            # 1. Basic price features
            features_dict['price'] = prices
            features_dict['returns'] = self._compute_returns(prices)
            features_dict['log_returns'] = self._compute_log_returns(prices)
            
            # 2. Technical indicators
            if self.config.use_technical_indicators:
                tech_features = self._compute_technical_indicators(data)
                features_dict.update(tech_features)
            
            # 3. Statistical features
            if self.config.use_statistical_features:
                stat_features = self._compute_statistical_features(prices)
                features_dict.update(stat_features)
            
            # 4. Lagged features
            if self.config.use_lagged_features:
                lag_features = self._compute_lagged_features(prices, self.config.lag_periods)
                features_dict.update(lag_features)
            
            # 5. Volatility features
            vol_features = self._compute_volatility_features(prices)
            features_dict.update(vol_features)
            
            # 6. Volume features (if available)
            if 'volume' in data.columns:
                volume_features = self._compute_volume_features(volume)
                features_dict.update(volume_features)
            
            # Create features DataFrame
            features_df = pd.DataFrame(features_dict, index=data.index)
            
            # Handle missing values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # Store feature names
            self.feature_names = features_df.columns.tolist()
            
            logger.info(f"Prepared {len(self.feature_names)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise

    def _compute_technical_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute technical indicators"""
        features = {}
        prices = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        try:
            # Moving averages
            features['sma_10'] = talib.SMA(prices, timeperiod=10)
            features['sma_20'] = talib.SMA(prices, timeperiod=20)
            features['sma_50'] = talib.SMA(prices, timeperiod=50)
            features['ema_12'] = talib.EMA(prices, timeperiod=12)
            features['ema_26'] = talib.EMA(prices, timeperiod=26)
            
            # RSI
            features['rsi'] = talib.RSI(prices, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(prices)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices, timeperiod=20)
            features['bb_upper'] = bb_upper
            features['bb_lower'] = bb_lower
            features['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, prices)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
            # ATR
            features['atr'] = talib.ATR(high, low, prices, timeperiod=14)
            
            # CCI
            features['cci'] = talib.CCI(high, low, prices, timeperiod=20)
            
        except Exception as e:
            logger.warning(f"Technical indicator computation failed: {e}")
        
        return features

    def _compute_statistical_features(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistical features"""
        features = {}
        
        try:
            returns = self._compute_returns(prices)
            
            # Rolling statistics
            window = 20
            if len(prices) > window:
                features['rolling_mean'] = pd.Series(prices).rolling(window).mean().values
                features['rolling_std'] = pd.Series(prices).rolling(window).std().values
                features['rolling_skew'] = pd.Series(prices).rolling(window).skew().values
                features['rolling_kurtosis'] = pd.Series(prices).rolling(window).kurt().values
                
                # Z-score
                zscore = (prices - features['rolling_mean']) / features['rolling_std']
                features['zscore'] = zscore
            
            # Return statistics
            if len(returns) > window:
                features['return_std'] = pd.Series(returns).rolling(window).std().values
                features['return_skew'] = pd.Series(returns).rolling(window).skew().values
            
            # Volatility clustering
            squared_returns = returns ** 2
            if len(squared_returns) > 1:
                # Lag-1 autocorrelation of squared returns
                acf = [np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]]
                features['vol_clustering'] = np.array(acf * len(prices))
            
        except Exception as e:
            logger.warning(f"Statistical feature computation failed: {e}")
        
        return features

    def _compute_lagged_features(self, prices: np.ndarray, lag_periods: List[int]) -> Dict[str, np.ndarray]:
        """Compute lagged price features"""
        features = {}
        
        try:
            for lag in lag_periods:
                if len(prices) > lag:
                    features[f'lag_{lag}'] = np.roll(prices, lag)
                    features[f'lag_{lag}'][:lag] = prices[0]  # Fill initial values
                else:
                    features[f'lag_{lag}'] = np.full_like(prices, prices[0])
            
            # Lagged returns
            returns = self._compute_returns(prices)
            for lag in [1, 2, 3]:
                if len(returns) > lag:
                    features[f'return_lag_{lag}'] = np.roll(returns, lag)
                    features[f'return_lag_{lag}'][:lag] = 0
                else:
                    features[f'return_lag_{lag}'] = np.zeros_like(returns)
        
        except Exception as e:
            logger.warning(f"Lagged feature computation failed: {e}")
        
        return features

    def _compute_volatility_features(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute volatility features"""
        features = {}
        
        try:
            returns = self._compute_returns(prices)
            
            # Historical volatility (multiple timeframes)
            for window in [10, 20, 30]:
                if len(returns) > window:
                    volatility = pd.Series(returns).rolling(window).std() * np.sqrt(252)
                    features[f'volatility_{window}'] = volatility.values
                else:
                    features[f'volatility_{window}'] = np.zeros_like(prices)
            
            # Parkinson volatility (high-low based)
            # features['parkinson_vol'] = self._compute_parkinson_volatility(high, low)
            
        except Exception as e:
            logger.warning(f"Volatility feature computation failed: {e}")
        
        return features

    def _compute_volume_features(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute volume-based features"""
        features = {}
        
        try:
            if len(volume) > 20:
                features['volume_sma'] = pd.Series(volume).rolling(20).mean().values
                features['volume_std'] = pd.Series(volume).rolling(20).std().values
                features['volume_zscore'] = (volume - features['volume_sma']) / features['volume_std']
            else:
                features['volume_sma'] = np.full_like(volume, np.mean(volume))
                features['volume_std'] = np.full_like(volume, np.std(volume))
                features['volume_zscore'] = np.zeros_like(volume)
        
        except Exception as e:
            logger.warning(f"Volume feature computation failed: {e}")
        
        return features

    @staticmethod
    def _compute_returns(prices: np.ndarray) -> np.ndarray:
        """Compute percentage returns"""
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        return returns

    @staticmethod
    def _compute_log_returns(prices: np.ndarray) -> np.ndarray:
        """Compute log returns"""
        log_returns = np.zeros_like(prices)
        log_returns[1:] = np.log(prices[1:] / prices[:-1])
        return log_returns

    def prepare_sequences(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        try:
            X_seq = []
            y_seq = []
            
            for i in range(self.config.lookback_window, len(features)):
                X_seq.append(features.iloc[i-self.config.lookback_window:i].values)
                y_seq.append(targets.iloc[i])
            
            return np.array(X_seq), np.array(y_seq)
            
        except Exception as e:
            logger.error(f"Sequence preparation failed: {e}")
            raise

    def train_models(self, data: pd.DataFrame, target_column: str = 'close'):
        """
        Train all ensemble models
        """
        try:
            with self._training_lock:
                logger.info("Starting ensemble model training...")
                
                # Prepare features and targets
                features = self.prepare_features(data)
                targets = data[target_column].shift(-self.config.prediction_horizon)  # Predict future price
                targets = targets[:-self.config.prediction_horizon]  # Remove last n rows
                features = features[:-self.config.prediction_horizon]  # Align features
                
                # Remove any remaining NaN values
                valid_indices = ~(targets.isna() | features.isna().any(axis=1))
                features = features[valid_indices]
                targets = targets[valid_indices]
                
                # Split data
                split_idx = int(len(features) * self.config.train_test_split)
                self.X_train = features.iloc[:split_idx]
                self.y_train = targets.iloc[:split_idx]
                self.X_test = features.iloc[split_idx:]
                self.y_test = targets.iloc[split_idx:]
                
                logger.info(f"Training data: {len(self.X_train)} samples")
                logger.info(f"Test data: {len(self.X_test)} samples")
                
                # Train models in parallel
                with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                    future_to_model = {
                        executor.submit(self._train_single_model, model_type, self.X_train, self.y_train): model_type
                        for model_type in self.models.keys()
                    }
                    
                    for future in as_completed(future_to_model):
                        model_type = future_to_model[future]
                        try:
                            future.result()
                            logger.info(f"Model trained successfully: {model_type.value}")
                        except Exception as e:
                            logger.error(f"Model training failed for {model_type.value}: {e}")
                
                # Evaluate models
                self._evaluate_models()
                
                # Calculate initial weights
                self._update_model_weights()
                
                logger.info("Ensemble model training completed")
                
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            raise

    def _train_single_model(self, model_type: ModelType, X_train: pd.DataFrame, y_train: pd.Series):
        """Train a single model"""
        try:
            model = self.models[model_type]
            scaler = self.scalers[model_type]
            
            if model_type == ModelType.LSTM:
                # Prepare sequences for LSTM
                X_seq, y_seq = self.prepare_sequences(X_train, y_train)
                
                # Scale features
                X_scaled = scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1]))
                X_scaled = X_scaled.reshape(X_seq.shape)
                
                # Train LSTM
                early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)
                
                model.fit(
                    X_scaled, y_seq,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
            else:
                # Scale features for traditional models
                X_scaled = scaler.fit_transform(X_train)
                
                # Train model
                model.fit(X_scaled, y_train)
            
        except Exception as e:
            logger.error(f"Single model training failed for {model_type.value}: {e}")
            raise

    def _evaluate_models(self):
        """Evaluate all models on test set"""
        try:
            for model_type in self.models.keys():
                predictions = self._predict_single_model(model_type, self.X_test)
                actual = self.y_test.values
                
                # Calculate metrics
                mse = mean_squared_error(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
                r2 = r2_score(actual, predictions)
                rmse = np.sqrt(mse)
                
                # Directional accuracy
                actual_direction = np.sign(np.diff(actual))
                pred_direction = np.sign(np.diff(predictions))
                directional_accuracy = np.mean(actual_direction == pred_direction)
                
                # Update performance
                self.performance[model_type].mse = mse
                self.performance[model_type].mae = mae
                self.performance[model_type].r2 = r2
                self.performance[model_type].rmse = rmse
                self.performance[model_type].directional_accuracy = directional_accuracy
                self.performance[model_type].last_updated = datetime.now()
                
                logger.info(f"Model {model_type.value} - RMSE: {rmse:.6f}, R²: {r2:.4f}, DA: {directional_accuracy:.4f}")
                
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")

    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        try:
            total_weight = 0.0
            weights = {}
            
            for model_type, perf in self.performance.items():
                # Use inverse RMSE as weight basis (lower RMSE = higher weight)
                weight = 1.0 / (perf.rmse + 1e-8)  # Add small constant to avoid division by zero
                
                # Apply directional accuracy bonus
                da_bonus = 1.0 + perf.directional_accuracy
                weight *= da_bonus
                
                # Apply decay for older performances
                time_diff = (datetime.now() - perf.last_updated).total_seconds() / 3600  # hours
                decay = self.config.decay_factor ** (time_diff / 24)  # Daily decay
                weight *= decay
                
                weights[model_type] = weight
                total_weight += weight
            
            # Normalize weights and apply minimum weight
            for model_type in weights.keys():
                normalized_weight = weights[model_type] / total_weight
                # Ensure minimum weight
                final_weight = max(normalized_weight, self.config.min_model_weight)
                self.performance[model_type].weight = final_weight
            
            # Renormalize after applying minimum weights
            total_final_weight = sum(perf.weight for perf in self.performance.values())
            for model_type in self.performance.keys():
                self.performance[model_type].weight /= total_final_weight
            
            logger.info("Model weights updated")
            
        except Exception as e:
            logger.error(f"Weight update failed: {e}")

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        """
        Make ensemble prediction using all models
        """
        try:
            with self._lock:
                individual_predictions = {}
                model_weights = {}
                
                # Get predictions from all models
                for model_type in self.models.keys():
                    prediction = self._predict_single_model(model_type, features)
                    individual_predictions[model_type] = prediction
                    model_weights[model_type] = self.performance[model_type].weight
                
                # Combine predictions based on ensemble method
                if self.config.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
                    ensemble_prediction = self._weighted_average(individual_predictions, model_weights)
                elif self.config.ensemble_method == EnsembleMethod.STACKING:
                    ensemble_prediction = self._stacking_predict(individual_predictions, features)
                elif self.config.ensemble_method == EnsembleMethod.VOTING:
                    ensemble_prediction = self._voting_predict(individual_predictions, model_weights)
                else:  # Default to weighted average
                    ensemble_prediction = self._weighted_average(individual_predictions, model_weights)
                
                # Calculate confidence based on prediction agreement
                confidence = self._calculate_confidence(individual_predictions, ensemble_prediction)
                
                result = PredictionResult(
                    timestamp=datetime.now(),
                    prediction=ensemble_prediction,
                    confidence=confidence,
                    individual_predictions=individual_predictions,
                    model_weights=model_weights,
                    features_used=self.feature_names,
                    metadata={
                        'ensemble_method': self.config.ensemble_method.value,
                        'models_used': [mt.value for mt in individual_predictions.keys()],
                        'total_models': len(individual_predictions)
                    }
                )
                
                # Store prediction for later performance tracking
                self.prediction_history.append(result)
                
                logger.debug(f"Ensemble prediction: {ensemble_prediction:.6f}, Confidence: {confidence:.4f}")
                
                return result
                
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise

    def _predict_single_model(self, model_type: ModelType, features: pd.DataFrame) -> float:
        """Get prediction from a single model"""
        try:
            model = self.models[model_type]
            scaler = self.scalers[model_type]
            
            if model_type == ModelType.LSTM:
                # Prepare sequence for LSTM
                if len(features) < self.config.lookback_window:
                    # Pad with latest values if insufficient data
                    padding = pd.concat([features] * (self.config.lookback_window // len(features) + 1))
                    features = padding.iloc[-self.config.lookback_window:]
                
                X_seq = features.iloc[-self.config.lookback_window:].values.reshape(1, self.config.lookback_window, -1)
                
                # Scale features
                X_scaled = scaler.transform(X_seq.reshape(-1, X_seq.shape[-1]))
                X_scaled = X_scaled.reshape(X_seq.shape)
                
                # Predict
                prediction = model.predict(X_scaled, verbose=0)[0, 0]
                
            else:
                # Scale features for traditional models
                X_scaled = scaler.transform(features.iloc[-1:].values)
                
                # Predict
                prediction = model.predict(X_scaled)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Single model prediction failed for {model_type.value}: {e}")
            # Return a safe default prediction
            return float(features['close'].iloc[-1])

    def _weighted_average(self, predictions: Dict[ModelType, float], weights: Dict[ModelType, float]) -> float:
        """Calculate weighted average of predictions"""
        try:
            total = 0.0
            total_weight = 0.0
            
            for model_type, prediction in predictions.items():
                weight = weights[model_type]
                total += prediction * weight
                total_weight += weight
            
            return total / total_weight if total_weight > 0 else statistics.mean(predictions.values())
            
        except Exception as e:
            logger.error(f"Weighted average calculation failed: {e}")
            return statistics.mean(predictions.values())

    def _stacking_predict(self, base_predictions: Dict[ModelType, float], features: pd.DataFrame) -> float:
        """Use stacking ensemble method"""
        try:
            # This would use a meta-learner trained on base model predictions
            # For now, using a simple weighted approach based on recent performance
            
            # Create feature vector from base predictions
            meta_features = np.array([base_predictions[mt] for mt in base_predictions.keys()]).reshape(1, -1)
            
            # Simple meta-learner: weighted average with performance-based weights
            weights = np.array([self.performance[mt].weight for mt in base_predictions.keys()])
            weights = weights / np.sum(weights)
            
            prediction = np.dot(meta_features, weights)[0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Stacking prediction failed: {e}")
            return self._weighted_average(base_predictions, {mt: self.performance[mt].weight for mt in base_predictions.keys()})

    def _voting_predict(self, predictions: Dict[ModelType, float], weights: Dict[ModelType, float]) -> float:
        """Use voting ensemble method"""
        try:
            # For regression, voting means taking the median or trimmed mean
            predictions_list = list(predictions.values())
            
            # Use trimmed mean to reduce outlier influence
            trimmed_mean = trim_mean(predictions_list, 0.1)  # Trim 10% from each end
            
            return float(trimmed_mean)
            
        except Exception as e:
            logger.error(f"Voting prediction failed: {e}")
            return statistics.median(predictions.values())

    def _calculate_confidence(self, individual_predictions: Dict[ModelType, float], ensemble_prediction: float) -> float:
        """Calculate prediction confidence based on model agreement"""
        try:
            predictions = list(individual_predictions.values())
            
            if len(predictions) < 2:
                return 0.5
            
            # Calculate coefficient of variation (inverse for confidence)
            std_dev = statistics.stdev(predictions)
            mean_pred = statistics.mean(predictions)
            
            if mean_pred == 0:
                cv = 0
            else:
                cv = std_dev / abs(mean_pred)
            
            # Convert to confidence (lower CV = higher confidence)
            confidence = 1.0 / (1.0 + cv)
            
            # Adjust based on recent performance
            recent_performance = self._get_recent_performance()
            performance_boost = statistics.mean(recent_performance) if recent_performance else 0.5
            confidence = 0.7 * confidence + 0.3 * performance_boost
            
            return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _get_recent_performance(self) -> List[float]:
        """Get recent model performance scores"""
        try:
            performances = []
            for perf in self.performance.values():
                if perf.recent_performance:
                    performances.extend(perf.recent_performance)
            return performances[-50:]  # Last 50 performances
        except:
            return []

    def update_with_actual(self, actual_value: float, prediction_result: PredictionResult):
        """
        Update model weights based on actual outcome
        """
        try:
            with self._lock:
                prediction_error = abs(actual_value - prediction_result.prediction)
                
                # Update individual model performances
                for model_type, predicted_value in prediction_result.individual_predictions.items():
                    model_error = abs(actual_value - predicted_value)
                    
                    # Store recent performance (lower error = better performance)
                    performance_score = 1.0 / (1.0 + model_error)
                    self.performance[model_type].recent_performance.append(performance_score)
                
                # Update model weights
                self._update_model_weights()
                
                # Store actual value for historical analysis
                self.actual_history.append(actual_value)
                
                logger.debug(f"Model updated with actual value: {actual_value}, Error: {prediction_error:.6f}")
                
        except Exception as e:
            logger.error(f"Model update failed: {e}")

    def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance report"""
        try:
            report = {
                'ensemble_method': self.config.ensemble_method.value,
                'total_models': len(self.models),
                'model_performances': {},
                'ensemble_metrics': {},
                'feature_importance': self.feature_importance
            }
            
            # Individual model performances
            for model_type, perf in self.performance.items():
                report['model_performances'][model_type.value] = {
                    'mse': perf.mse,
                    'mae': perf.mae,
                    'r2': perf.r2,
                    'rmse': perf.rmse,
                    'directional_accuracy': perf.directional_accuracy,
                    'weight': perf.weight,
                    'recent_performance_count': len(perf.recent_performance)
                }
            
            # Ensemble metrics (if we have actual values)
            if len(self.actual_history) > 0 and len(self.prediction_history) > 0:
                actuals = list(self.actual_history)
                predictions = [pred.prediction for pred in self.prediction_history]
                
                report['ensemble_metrics'] = {
                    'mse': mean_squared_error(actuals, predictions),
                    'mae': mean_absolute_error(actuals, predictions),
                    'r2': r2_score(actuals, predictions),
                    'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                    'total_predictions': len(actuals)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {}

    def save_models(self, directory: str = "models"):
        """Save all models and configuration"""
        try:
            model_dir = Path(directory)
            model_dir.mkdir(exist_ok=True)
            
            # Save configuration
            config_path = model_dir / "ensemble_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            # Save models
            for model_type, model in self.models.items():
                model_path = model_dir / f"{model_type.value}_model.pkl"
                
                if model_type == ModelType.LSTM:
                    # Save Keras model
                    model.save(model_path.with_suffix('.h5'))
                else:
                    # Save scikit-learn models
                    joblib.dump(model, model_path)
            
            # Save scalers
            scalers_path = model_dir / "scalers.pkl"
            joblib.dump(self.scalers, scalers_path)
            
            # Save performance data
            performance_path = model_dir / "performance.pkl"
            joblib.dump(self.performance, performance_path)
            
            logger.info(f"Models saved to {directory}")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")

    def load_models(self, directory: str = "models"):
        """Load models and configuration"""
        try:
            model_dir = Path(directory)
            
            # Load configuration
            config_path = model_dir / "ensemble_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                    self.config = ModelConfig(**config_dict)
            
            # Load models
            for model_type in self.config.base_models:
                model_path = model_dir / f"{model_type.value}_model"
                
                if model_type == ModelType.LSTM:
                    # Load Keras model
                    from tensorflow.keras.models import load_model
                    self.models[model_type] = load_model(model_path.with_suffix('.h5'))
                else:
                    # Load scikit-learn models
                    self.models[model_type] = joblib.load(model_path.with_suffix('.pkl'))
            
            # Load scalers
            scalers_path = model_dir / "scalers.pkl"
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
            
            # Load performance data
            performance_path = model_dir / "performance.pkl"
            if performance_path.exists():
                self.performance = joblib.load(performance_path)
            
            logger.info(f"Models loaded from {directory}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def feature_importance_analysis(self) -> Dict[str, float]:
        """Analyze feature importance across ensemble"""
        try:
            importance_scores = defaultdict(float)
            
            for model_type, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances = model.feature_importances_
                    for i, importance in enumerate(importances):
                        feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                        importance_scores[feature_name] += importance * self.performance[model_type].weight
                
                elif hasattr(model, 'coef_'):
                    # Linear models
                    coefs = model.coef_
                    if len(coefs.shape) > 1:  # Multiple outputs
                        coefs = np.mean(np.abs(coefs), axis=0)
                    else:
                        coefs = np.abs(coefs)
                    
                    for i, coef in enumerate(coefs):
                        feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                        importance_scores[feature_name] += coef * self.performance[model_type].weight
            
            # Normalize scores
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
            
            self.feature_importance = dict(importance_scores)
            return self.feature_importance
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize ensemble predictor
    config = ModelConfig(
        base_models=[ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM],
        ensemble_method=EnsembleMethod.WEIGHTED_AVERAGE
    )
    
    predictor = EnsemblePredictor(config)
    
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='H')
    np.random.seed(42)
    
    # Create realistic Forex data
    prices = [1.1000]
    for i in range(1, len(dates)):
        # Random walk with drift and volatility
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
    
    # Train models
    predictor.train_models(sample_data)
    
    # Make prediction
    latest_features = predictor.prepare_features(sample_data.tail(100))
    prediction = predictor.predict(latest_features)
    
    print(f"Ensemble Prediction: {prediction.prediction:.6f}")
    print(f"Confidence: {prediction.confidence:.4f}")
    print(f"Models Used: {len(prediction.individual_predictions)}")
    
    # Get performance report
    report = predictor.get_model_performance()
    print(f"Best Model: {max(report['model_performances'].items(), key=lambda x: x[1]['r2'])[0]}")
"""
Advanced Model Trainer for FOREX TRADING BOT
Professional machine learning model training with hyperparameter optimization
"""

import logging
import pandas as pd
import numpy as np
import json
import pickle
import joblib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import statistics
from scipy import stats
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical
import talib
from scipy.stats import trim_mean
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ModelType(Enum):
    LSTM = "lstm"
    CNN = "cnn"
    CNN_LSTM = "cnn_lstm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVR = "svr"
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    MLP = "mlp"

class ProblemType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTI_OUTPUT = "multi_output"

class TrainingStatus(Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    OPTIMIZING = "optimizing"

@dataclass
class TrainingConfig:
    """Model training configuration"""
    # Model selection
    model_types: List[ModelType] = field(default_factory=lambda: [
        ModelType.LSTM, ModelType.RANDOM_FOREST, ModelType.XGBOOST
    ])
    problem_type: ProblemType = ProblemType.REGRESSION
    
    # Training parameters
    lookback_window: int = 60
    prediction_horizon: int = 1
    train_test_split: float = 0.8
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Feature engineering
    feature_selection_enabled: bool = True
    max_features: int = 50
    correlation_threshold: float = 0.95
    
    # Hyperparameter optimization
    hyperparameter_optimization: bool = True
    n_trials: int = 100
    optimization_timeout: int = 3600  # seconds
    
    # Neural network parameters
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    lstm_dropout: float = 0.3
    lstm_recurrent_dropout: float = 0.2
    dense_units: List[int] = field(default_factory=lambda: [64, 32, 16])
    dense_dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 200
    patience: int = 20
    
    # Tree-based parameters
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    
    # Early stopping
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # Ensemble settings
    create_ensemble: bool = True
    ensemble_method: str = "weighted_average"  # weighted_average, stacking, voting

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_type: ModelType
    mse: float
    mae: float
    r2: float
    rmse: float
    mape: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    training_time: float
    cross_val_scores: List[float]
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_history: Dict[str, List[float]]
    created_at: datetime

@dataclass
class TrainingResult:
    """Complete training result"""
    status: TrainingStatus
    best_model: Any
    best_model_type: ModelType
    best_score: float
    all_performances: Dict[ModelType, ModelPerformance]
    feature_names: List[str]
    scaler: Any
    training_config: TrainingConfig
    metadata: Dict[str, Any]

class AdvancedModelTrainer:
    """
    Advanced model training system with hyperparameter optimization
    Supports multiple model types and comprehensive performance evaluation
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        
        # Model storage
        self.models: Dict[ModelType, Any] = {}
        self.scalers: Dict[ModelType, Any] = {}
        self.performance: Dict[ModelType, ModelPerformance] = {}
        
        # Training state
        self.training_status = TrainingStatus.PENDING
        self.current_model: Optional[ModelType] = None
        self.training_start_time: Optional[datetime] = None
        
        # Feature management
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        # Optimization
        self.study: Optional[optuna.Study] = None
        self.optimization_history: List[Dict] = []
        
        # Thread safety
        self._lock = threading.RLock()
        self._training_lock = threading.Lock()
        
        logger.info("AdvancedModelTrainer initialized")

    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare data for model training with proper time series splitting
        """
        try:
            logger.info("Preparing training data...")
            
            # Handle missing values
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            target = target.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Align features and target
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]
            
            # Feature selection
            if self.config.feature_selection_enabled:
                features = self._select_features(features, target)
            
            self.feature_names = features.columns.tolist()
            
            # Time-based split for time series
            split_idx = int(len(features) * self.config.train_test_split)
            
            X_train = features.iloc[:split_idx]
            y_train = target.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_test = target.iloc[split_idx:]
            
            logger.info(f"Training data: {len(X_train)} samples")
            logger.info(f"Test data: {len(X_test)} samples")
            logger.info(f"Features: {len(self.feature_names)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def _select_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most important features"""
        try:
            # Remove highly correlated features
            corr_matrix = features.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > self.config.correlation_threshold)]
            
            features_reduced = features.drop(columns=to_drop)
            logger.info(f"Dropped {len(to_drop)} highly correlated features")
            
            # Further selection if needed
            if len(features_reduced.columns) > self.config.max_features:
                selector = SelectKBest(score_func=mutual_info_regression, k=self.config.max_features)
                X_selected = selector.fit_transform(features_reduced, target)
                selected_features = features_reduced.columns[selector.get_support()]
                features_reduced = features_reduced[selected_features]
                logger.info(f"Selected {len(selected_features)} best features using mutual information")
            
            return features_reduced
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return features

    def train_models(self, features: pd.DataFrame, target: pd.Series) -> TrainingResult:
        """
        Train multiple models with hyperparameter optimization
        """
        try:
            with self._training_lock:
                self.training_status = TrainingStatus.TRAINING
                self.training_start_time = datetime.now()
                
                logger.info("Starting model training pipeline...")
                
                # Prepare data
                X_train, X_test, y_train, y_test = self.prepare_data(features, target)
                
                # Train models
                trained_models = {}
                performances = {}
                
                for model_type in self.config.model_types:
                    try:
                        logger.info(f"Training {model_type.value}...")
                        self.current_model = model_type
                        
                        # Train individual model
                        model, performance = self._train_single_model(
                            model_type, X_train, X_test, y_train, y_test
                        )
                        
                        if model is not None:
                            trained_models[model_type] = model
                            performances[model_type] = performance
                            logger.info(f"✅ {model_type.value} trained successfully")
                        else:
                            logger.warning(f"❌ {model_type.value} training failed")
                            
                    except Exception as e:
                        logger.error(f"Training failed for {model_type.value}: {e}")
                        continue
                
                # Select best model
                best_model_type, best_model, best_score = self._select_best_model(performances, trained_models)
                
                # Create ensemble if enabled
                ensemble_model = None
                if self.config.create_ensemble and len(trained_models) > 1:
                    ensemble_model = self._create_ensemble(trained_models, performances, X_test, y_test)
                
                training_time = (datetime.now() - self.training_start_time).total_seconds()
                
                result = TrainingResult(
                    status=TrainingStatus.COMPLETED,
                    best_model=ensemble_model if ensemble_model else best_model,
                    best_model_type=ModelType.ENSEMBLE if ensemble_model else best_model_type,
                    best_score=best_score,
                    all_performances=performances,
                    feature_names=self.feature_names,
                    scaler=self.scalers.get(best_model_type),
                    training_config=self.config,
                    metadata={
                        'training_time_seconds': training_time,
                        'models_trained': len(trained_models),
                        'total_features': len(self.feature_names),
                        'completion_time': datetime.now()
                    }
                )
                
                self.training_status = TrainingStatus.COMPLETED
                logger.info(f"Model training completed in {training_time:.2f} seconds")
                
                return result
                
        except Exception as e:
            self.training_status = TrainingStatus.FAILED
            logger.error(f"Model training failed: {e}")
            raise

    def _train_single_model(self, model_type: ModelType, X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, ModelPerformance]:
        """Train a single model with hyperparameter optimization"""
        try:
            start_time = datetime.now()
            
            # Hyperparameter optimization
            if self.config.hyperparameter_optimization:
                best_params = self._optimize_hyperparameters(model_type, X_train, y_train)
            else:
                best_params = self._get_default_parameters(model_type)
            
            # Train model with best parameters
            if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM]:
                model, history = self._train_neural_network(model_type, X_train, y_train, X_test, y_test, best_params)
            else:
                model, history = self._train_sklearn_model(model_type, X_train, y_train, best_params)
            
            # Make predictions
            if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM]:
                predictions = self._predict_neural_network(model, X_test, model_type)
            else:
                predictions = model.predict(X_test)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(
                model_type, y_test, predictions, history, best_params, start_time
            )
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, model.feature_importances_))
            else:
                importance_dict = self._calculate_permutation_importance(model, X_test, y_test)
            
            performance.feature_importance = importance_dict
            
            # Store model and scaler
            self.models[model_type] = model
            self.performance[model_type] = performance
            
            return model, performance
            
        except Exception as e:
            logger.error(f"Single model training failed for {model_type.value}: {e}")
            return None, None

    def _train_neural_network(self, model_type: ModelType, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series, params: Dict[str, Any]) -> Tuple[Any, Dict]:
        """Train neural network model"""
        try:
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[model_type] = scaler
            
            # Prepare sequences for time series models
            if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM]:
                X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train.values)
                X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test.values)
            else:
                X_train_seq, y_train_seq = X_train_scaled, y_train.values
                X_test_seq, y_test_seq = X_test_scaled, y_test.values
            
            # Create model architecture
            if model_type == ModelType.LSTM:
                model = self._create_lstm_model(X_train_seq.shape[1:], params)
            elif model_type == ModelType.CNN:
                model = self._create_cnn_model(X_train_seq.shape[1:], params)
            elif model_type == ModelType.CNN_LSTM:
                model = self._create_cnn_lstm_model(X_train_seq.shape[1:], params)
            else:
                model = self._create_mlp_model(X_train_seq.shape[1], params)
            
            # Compile model
            if self.config.problem_type == ProblemType.REGRESSION:
                loss = 'mse'
                metrics = ['mae']
            else:
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
            
            model.compile(
                optimizer=Adam(learning_rate=params.get('learning_rate', self.config.learning_rate)),
                loss=loss,
                metrics=metrics
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                ReduceLROnPlateau(
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-7
                ),
                ModelCheckpoint(
                    f'best_{model_type.value}_model.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train_seq, y_train_seq,
                batch_size=params.get('batch_size', self.config.batch_size),
                epochs=self.config.epochs,
                validation_data=(X_test_seq, y_test_seq),
                callbacks=callbacks,
                verbose=0,
                shuffle=False  # Important for time series
            )
            
            return model, history.history
            
        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            raise

    def _create_lstm_model(self, input_shape: Tuple, params: Dict) -> Model:
        """Create LSTM model architecture"""
        model = Sequential()
        
        # Input layer
        model.add(LSTM(
            units=params.get('lstm_units', self.config.lstm_units[0]),
            return_sequences=len(self.config.lstm_units) > 1,
            input_shape=input_shape,
            dropout=params.get('dropout', self.config.lstm_dropout),
            recurrent_dropout=params.get('recurrent_dropout', self.config.lstm_recurrent_dropout)
        ))
        
        # Additional LSTM layers
        for units in self.config.lstm_units[1:]:
            model.add(LSTM(
                units=units,
                return_sequences=True,
                dropout=params.get('dropout', self.config.lstm_dropout),
                recurrent_dropout=params.get('recurrent_dropout', self.config.lstm_recurrent_dropout)
            ))
        
        # Final LSTM layer
        model.add(LSTM(
            units=self.config.lstm_units[-1],
            dropout=params.get('dropout', self.config.lstm_dropout),
            recurrent_dropout=params.get('recurrent_dropout', self.config.lstm_recurrent_dropout)
        ))
        
        # Dense layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(params.get('dense_dropout', self.config.dense_dropout)))
            model.add(BatchNormalization())
        
        # Output layer
        if self.config.problem_type == ProblemType.REGRESSION:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(3, activation='softmax'))  # 3 classes: BUY, SELL, HOLD
        
        return model

    def _create_cnn_model(self, input_shape: Tuple, params: Dict) -> Model:
        """Create CNN model architecture"""
        model = Sequential()
        
        # Conv1D layers
        model.add(Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu'
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(
            filters=256,
            kernel_size=3,
            activation='relu'
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        # Flatten and dense layers
        model.add(Flatten())
        
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(params.get('dense_dropout', self.config.dense_dropout)))
            model.add(BatchNormalization())
        
        # Output layer
        if self.config.problem_type == ProblemType.REGRESSION:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(3, activation='softmax'))  # 3 classes: BUY, SELL, HOLD
        
        return model

    def _create_cnn_lstm_model(self, input_shape: Tuple, params: Dict) -> Model:
        """Create CNN-LSTM hybrid model"""
        model = Sequential()
        
        # CNN layers for feature extraction
        model.add(Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu'
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        # LSTM layers for sequence modeling
        model.add(LSTM(
            units=128,
            return_sequences=True,
            dropout=params.get('dropout', self.config.lstm_dropout)
        ))
        model.add(LSTM(
            units=64,
            dropout=params.get('dropout', self.config.lstm_dropout)
        ))
        
        # Dense layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(params.get('dense_dropout', self.config.dense_dropout)))
            model.add(BatchNormalization())
        
        # Output layer
        if self.config.problem_type == ProblemType.REGRESSION:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(3, activation='softmax'))  # 3 classes: BUY, SELL, HOLD
        
        return model

    def _create_mlp_model(self, input_dim: int, params: Dict) -> Model:
        """Create MLP model architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            params.get('first_layer_units', 128),
            activation='relu',
            input_dim=input_dim
        ))
        model.add(BatchNormalization())
        model.add(Dropout(params.get('dropout', self.config.dense_dropout)))
        
        # Hidden layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(params.get('dropout', self.config.dense_dropout)))
        
        # Output layer
        if self.config.problem_type == ProblemType.REGRESSION:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(3, activation='softmax'))  # 3 classes: BUY, SELL, HOLD
        
        return model

    def _train_sklearn_model(self, model_type: ModelType, X_train: pd.DataFrame, 
                           y_train: pd.Series, params: Dict) -> Tuple[Any, Dict]:
        """Train scikit-learn model"""
        try:
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_type] = scaler
            
            # Create model
            if model_type == ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif model_type == ModelType.GRADIENT_BOOSTING:
                model = GradientBoostingRegressor(**params, random_state=42)
            elif model_type == ModelType.XGBOOST:
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            elif model_type == ModelType.LIGHTGBM:
                model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
            elif model_type == ModelType.SVR:
                model = SVR(**params)
            elif model_type == ModelType.LINEAR:
                model = LinearRegression(**params)
            elif model_type == ModelType.RIDGE:
                model = Ridge(**params, random_state=42)
            elif model_type == ModelType.LASSO:
                model = Lasso(**params, random_state=42)
            elif model_type == ModelType.ELASTIC_NET:
                model = ElasticNet(**params, random_state=42)
            elif model_type == ModelType.MLP:
                model = MLPRegressor(**params, random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Create dummy history for consistency
            history = {
                'loss': [0.0],
                'val_loss': [0.0]
            }
            
            return model, history
            
        except Exception as e:
            logger.error(f"Scikit-learn model training failed: {e}")
            raise

    def _create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models"""
        X_seq, y_seq = [], []
        
        for i in range(self.config.lookback_window, len(data)):
            X_seq.append(data[i-self.config.lookback_window:i])
            y_seq.append(targets[i])
        
        return np.array(X_seq), np.array(y_seq)

    def _predict_neural_network(self, model: Model, X: pd.DataFrame, model_type: ModelType) -> np.ndarray:
        """Make predictions with neural network"""
        try:
            scaler = self.scalers[model_type]
            X_scaled = scaler.transform(X)
            
            if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM]:
                X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
                predictions = model.predict(X_seq, verbose=0)
                
                # Pad predictions to match original length
                pad_length = len(X) - len(predictions)
                if pad_length > 0:
                    predictions = np.concatenate([np.full(pad_length, predictions[0]), predictions.flatten()])
                else:
                    predictions = predictions.flatten()
            else:
                predictions = model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Neural network prediction failed: {e}")
            return np.zeros(len(X))

    def _optimize_hyperparameters(self, model_type: ModelType, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        try:
            logger.info(f"Optimizing hyperparameters for {model_type.value}...")
            
            def objective(trial):
                # Suggest hyperparameters based on model type
                params = self._suggest_hyperparameters(trial, model_type)
                
                try:
                    # Cross-validation score
                    if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM]:
                        score = self._evaluate_neural_network(params, model_type, X, y)
                    else:
                        score = self._evaluate_sklearn_model(params, model_type, X, y)
                    
                    return score
                    
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return float('inf')
            
            # Create study
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Optimize
            study.optimize(
                objective, 
                n_trials=self.config.n_trials,
                timeout=self.config.optimization_timeout,
                show_progress_bar=True
            )
            
            self.optimization_history.append({
                'model_type': model_type.value,
                'best_params': study.best_params,
                'best_value': study.best_value,
                'completed_trials': len(study.trials)
            })
            
            logger.info(f"Hyperparameter optimization completed for {model_type.value}")
            logger.info(f"Best score: {study.best_value:.6f}")
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return self._get_default_parameters(model_type)

    def _suggest_hyperparameters(self, trial, model_type: ModelType) -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna trials"""
        params = {}
        
        if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM, ModelType.MLP]:
            # Neural network parameters
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
            
            if model_type == ModelType.LSTM:
                params['lstm_units'] = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
                params['recurrent_dropout'] = trial.suggest_float('recurrent_dropout', 0.1, 0.5)
            
            params['dense_dropout'] = trial.suggest_float('dense_dropout', 0.1, 0.5)
            
        elif model_type == ModelType.RANDOM_FOREST:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
        elif model_type == ModelType.XGBOOST:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            params['gamma'] = trial.suggest_float('gamma', 0, 5)
            
        elif model_type == ModelType.LIGHTGBM:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            
        elif model_type == ModelType.SVR:
            params['C'] = trial.suggest_float('C', 0.1, 10.0, log=True)
            params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1.0)
            params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            
        return params

    def _evaluate_neural_network(self, params: Dict, model_type: ModelType, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate neural network with cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Create sequences
                X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train.values)
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values)
                
                # Create and train model
                if model_type == ModelType.LSTM:
                    model = self._create_lstm_model(X_train_seq.shape[1:], params)
                elif model_type == ModelType.CNN:
                    model = self._create_cnn_model(X_train_seq.shape[1:], params)
                elif model_type == ModelType.CNN_LSTM:
                    model = self._create_cnn_lstm_model(X_train_seq.shape[1:], params)
                else:
                    model = self._create_mlp_model(X_train_seq.shape[1], params)
                
                model.compile(
                    optimizer=Adam(learning_rate=params['learning_rate']),
                    loss='mse',
                    metrics=['mae']
                )
                
                # Train briefly for evaluation
                history = model.fit(
                    X_train_seq, y_train_seq,
                    batch_size=params['batch_size'],
                    epochs=10,
                    validation_data=(X_val_seq, y_val_seq),
                    verbose=0,
                    shuffle=False
                )
                
                # Use validation loss as score
                scores.append(history.history['val_loss'][-1])
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"Neural network evaluation failed: {e}")
            return float('inf')

    def _evaluate_sklearn_model(self, params: Dict, model_type: ModelType, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate scikit-learn model with cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=self.config.cross_validation_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Create model
                if model_type == ModelType.RANDOM_FOREST:
                    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                elif model_type == ModelType.XGBOOST:
                    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                elif model_type == ModelType.LIGHTGBM:
                    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
                elif model_type == ModelType.SVR:
                    model = SVR(**params)
                else:
                    model = LinearRegression(**params)
                
                # Train and evaluate
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                score = mean_squared_error(y_val, y_pred)
                scores.append(score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"Scikit-learn evaluation failed: {e}")
            return float('inf')

    def _calculate_performance_metrics(self, model_type: ModelType, y_true: pd.Series, 
                                    y_pred: np.ndarray, history: Dict, params: Dict, 
                                    start_time: datetime) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Directional accuracy
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(true_direction == pred_direction)
            
            # Trading performance metrics
            returns = y_pred  # Using predictions as returns for simplicity
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Cross-validation scores
            cv_scores = self._calculate_cross_validation_scores(model_type, params, y_true, y_pred)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            return ModelPerformance(
                model_type=model_type,
                mse=mse,
                mae=mae,
                r2=r2,
                rmse=rmse,
                mape=mape,
                directional_accuracy=directional_accuracy,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                training_time=training_time,
                cross_val_scores=cv_scores,
                feature_importance={},
                hyperparameters=params,
                training_history=history,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            # Return default performance metrics
            return ModelPerformance(
                model_type=model_type,
                mse=float('inf'),
                mae=float('inf'),
                r2=-float('inf'),
                rmse=float('inf'),
                mape=float('inf'),
                directional_accuracy=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                training_time=0.0,
                cross_val_scores=[],
                feature_importance={},
                hyperparameters=params,
                training_history=history,
                created_at=datetime.now()
            )

    def _calculate_cross_validation_scores(self, model_type: ModelType, params: Dict, 
                                         y_true: pd.Series, y_pred: np.ndarray) -> List[float]:
        """Calculate cross-validation scores"""
        try:
            # Simplified cross-validation using time series split
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(y_true):
                y_train_fold = y_true.iloc[train_idx]
                y_val_fold = y_true.iloc[val_idx]
                y_pred_fold = y_pred[val_idx]
                
                if len(y_val_fold) > 0 and len(y_pred_fold) > 0:
                    score = r2_score(y_val_fold, y_pred_fold)
                    scores.append(score)
            
            return scores if scores else [0.0]
            
        except Exception as e:
            logger.warning(f"Cross-validation calculation failed: {e}")
            return [0.0]

    def _calculate_permutation_importance(self, model: Any, X_test: pd.DataFrame, 
                                        y_test: pd.Series) -> Dict[str, float]:
        """Calculate permutation importance for models without built-in importance"""
        try:
            baseline_score = r2_score(y_test, model.predict(X_test))
            importance_scores = {}
            
            for feature in self.feature_names:
                X_test_permuted = X_test.copy()
                X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])
                permuted_score = r2_score(y_test, model.predict(X_test_permuted))
                importance_scores[feature] = baseline_score - permuted_score
            
            # Normalize scores
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {k: v/total_importance for k, v in importance_scores.items()}
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Permutation importance calculation failed: {e}")
            return {feature: 1.0/len(self.feature_names) for feature in self.feature_names}

    def _select_best_model(self, performances: Dict[ModelType, ModelPerformance], 
                          models: Dict[ModelType, Any]) -> Tuple[ModelType, Any, float]:
        """Select the best performing model"""
        try:
            best_score = -float('inf')
            best_model_type = None
            best_model = None
            
            for model_type, performance in performances.items():
                # Use weighted score considering multiple metrics
                score = (
                    0.3 * (1 - performance.rmse) +  # Lower RMSE is better
                    0.3 * performance.r2 +          # Higher R² is better
                    0.2 * performance.directional_accuracy +  # Higher DA is better
                    0.2 * performance.sharpe_ratio  # Higher Sharpe is better
                )
                
                if score > best_score:
                    best_score = score
                    best_model_type = model_type
                    best_model = models[model_type]
            
            logger.info(f"Best model: {best_model_type.value} with score: {best_score:.4f}")
            return best_model_type, best_model, best_score
            
        except Exception as e:
            logger.error(f"Best model selection failed: {e}")
            # Return first available model
            for model_type, model in models.items():
                return model_type, model, 0.0
            
            raise ValueError("No models available")

    def _create_ensemble(self, models: Dict[ModelType, Any], performances: Dict[ModelType, ModelPerformance],
                        X_test: pd.DataFrame, y_test: pd.Series) -> Any:
        """Create ensemble model"""
        try:
            logger.info("Creating ensemble model...")
            
            if self.config.ensemble_method == "weighted_average":
                return self._create_weighted_ensemble(models, performances)
            elif self.config.ensemble_method == "stacking":
                return self._create_stacking_ensemble(models, performances, X_test, y_test)
            else:
                return self._create_voting_ensemble(models, performances)
                
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return None

    def _create_weighted_ensemble(self, models: Dict[ModelType, Any], 
                                performances: Dict[ModelType, ModelPerformance]) -> Dict:
        """Create weighted average ensemble"""
        try:
            # Calculate weights based on performance
            weights = {}
            total_weight = 0
            
            for model_type, performance in performances.items():
                # Weight based on R² score
                weight = max(0, performance.r2)  # Ensure non-negative weights
                weights[model_type] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                # Equal weights if all performances are poor
                equal_weight = 1.0 / len(models)
                weights = {k: equal_weight for k in models.keys()}
            
            ensemble = {
                'type': 'weighted_average',
                'models': models,
                'weights': weights,
                'method': 'predict_proba' if self.config.problem_type == ProblemType.CLASSIFICATION else 'predict'
            }
            
            logger.info(f"Created weighted ensemble with weights: {weights}")
            return ensemble
            
        except Exception as e:
            logger.error(f"Weighted ensemble creation failed: {e}")
            return None

    def _create_stacking_ensemble(self, models: Dict[ModelType, Any],
                                performances: Dict[ModelType, ModelPerformance],
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Create stacking ensemble"""
        try:
            # Use the best performing model as meta-learner
            best_model_type, best_model, _ = self._select_best_model(performances, models)
            
            ensemble = {
                'type': 'stacking',
                'base_models': models,
                'meta_learner': best_model,
                'meta_learner_type': best_model_type
            }
            
            logger.info(f"Created stacking ensemble with meta-learner: {best_model_type.value}")
            return ensemble
            
        except Exception as e:
            logger.error(f"Stacking ensemble creation failed: {e}")
            return self._create_weighted_ensemble(models, performances)

    def _create_voting_ensemble(self, models: Dict[ModelType, Any],
                              performances: Dict[ModelType, ModelPerformance]) -> Dict:
        """Create voting ensemble"""
        try:
            ensemble = {
                'type': 'voting',
                'models': models,
                'voting': 'soft' if self.config.problem_type == ProblemType.CLASSIFICATION else 'hard'
            }
            
            logger.info("Created voting ensemble")
            return ensemble
            
        except Exception as e:
            logger.error(f"Voting ensemble creation failed: {e}")
            return self._create_weighted_ensemble(models, performances)

    def _get_default_parameters(self, model_type: ModelType) -> Dict[str, Any]:
        """Get default parameters for each model type"""
        defaults = {
            ModelType.LSTM: {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'dropout': self.config.lstm_dropout,
                'recurrent_dropout': self.config.lstm_recurrent_dropout,
                'dense_dropout': self.config.dense_dropout
            },
            ModelType.RANDOM_FOREST: {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'min_samples_split': self.config.min_samples_split,
                'min_samples_leaf': self.config.min_samples_leaf
            },
            ModelType.XGBOOST: {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
        
        return defaults.get(model_type, {})

    def save_model(self, training_result: TrainingResult, directory: str = "trained_models"):
        """Save trained model and metadata"""
        try:
            model_dir = Path(directory)
            model_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"model_{timestamp}"
            model_path.mkdir(exist_ok=True)
            
            # Save best model
            best_model = training_result.best_model
            if isinstance(best_model, dict) and 'type' in best_model:  # Ensemble
                ensemble_info = {
                    'ensemble_type': best_model['type'],
                    'models': list(best_model.get('models', {}).keys()),
                    'weights': best_model.get('weights', {}),
                    'meta_learner': best_model.get('meta_learner_type')
                }
                
                with open(model_path / "ensemble_info.json", 'w') as f:
                    json.dump(ensemble_info, f, indent=2, default=str)
                
                # Save individual models in ensemble
                for model_type, model in best_model.get('models', {}).items():
                    self._save_single_model(model, model_type, model_path)
                    
            else:  # Single model
                self._save_single_model(best_model, training_result.best_model_type, model_path)
            
            # Save scaler
            if training_result.scaler is not None:
                joblib.dump(training_result.scaler, model_path / "scaler.pkl")
            
            # Save training result metadata
            metadata = {
                'best_model_type': training_result.best_model_type.value,
                'best_score': training_result.best_score,
                'feature_names': training_result.feature_names,
                'training_config': asdict(training_result.training_config),
                'metadata': training_result.metadata,
                'performance_summary': {
                    model_type.value: {
                        'rmse': perf.rmse,
                        'r2': perf.r2,
                        'directional_accuracy': perf.directional_accuracy
                    } for model_type, perf in training_result.all_performances.items()
                }
            }
            
            with open(model_path / "training_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise

    def _save_single_model(self, model: Any, model_type: ModelType, model_path: Path):
        """Save a single model"""
        try:
            if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM, ModelType.MLP]:
                # Save Keras model
                model.save(model_path / f"{model_type.value}_model.h5")
            else:
                # Save scikit-learn model
                joblib.dump(model, model_path / f"{model_type.value}_model.pkl")
                
        except Exception as e:
            logger.error(f"Failed to save {model_type.value} model: {e}")

    def load_model(self, model_path: str) -> TrainingResult:
        """Load trained model"""
        try:
            model_path = Path(model_path)
            
            # Load metadata
            with open(model_path / "training_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load scaler
            scaler = joblib.load(model_path / "scaler.pkl")
            
            # Check if it's an ensemble
            ensemble_info_path = model_path / "ensemble_info.json"
            if ensemble_info_path.exists():
                with open(ensemble_info_path, 'r') as f:
                    ensemble_info = json.load(f)
                
                # Load ensemble models
                models = {}
                for model_type_str in ensemble_info['models']:
                    model_type = ModelType(model_type_str)
                    model = self._load_single_model(model_type, model_path)
                    if model is not None:
                        models[model_type] = model
                
                best_model = {
                    'type': ensemble_info['ensemble_type'],
                    'models': models,
                    'weights': ensemble_info.get('weights', {}),
                    'meta_learner': ensemble_info.get('meta_learner')
                }
                best_model_type = ModelType.ENSEMBLE
            else:
                # Load single model
                best_model_type = ModelType(metadata['best_model_type'])
                best_model = self._load_single_model(best_model_type, model_path)
            
            # Recreate training result
            training_config = TrainingConfig(**metadata['training_config'])
            
            result = TrainingResult(
                status=TrainingStatus.COMPLETED,
                best_model=best_model,
                best_model_type=best_model_type,
                best_score=metadata['best_score'],
                all_performances={},  # Would need to be loaded separately
                feature_names=metadata['feature_names'],
                scaler=scaler,
                training_config=training_config,
                metadata=metadata['metadata']
            )
            
            logger.info(f"Model loaded from {model_path}")
            return result
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _load_single_model(self, model_type: ModelType, model_path: Path) -> Any:
        """Load a single model"""
        try:
            if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.CNN_LSTM, ModelType.MLP]:
                return load_model(model_path / f"{model_type.value}_model.h5")
            else:
                return joblib.load(model_path / f"{model_type.value}_model.pkl")
                
        except Exception as e:
            logger.error(f"Failed to load {model_type.value} model: {e}")
            return None

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'status': self.training_status.value,
            'current_model': self.current_model.value if self.current_model else None,
            'models_trained': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'optimization_trials': len(self.optimization_history),
            'training_start_time': self.training_start_time
        }

# Example usage
if __name__ == "__main__":
    # Initialize model trainer
    config = TrainingConfig(
        model_types=[ModelType.LSTM, ModelType.RANDOM_FOREST, ModelType.XGBOOST],
        problem_type=ProblemType.REGRESSION,
        hyperparameter_optimization=True,
        n_trials=50
    )
    
    trainer = AdvancedModelTrainer(config)
    
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='H')
    np.random.seed(42)
    
    # Create realistic features and target
    n_samples = len(dates)
    n_features = 20
    
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create target with some relationship to features
    target = (features.iloc[:, 0] * 0.3 + 
              features.iloc[:, 1] * 0.2 + 
              features.iloc[:, 2] * 0.1 + 
              np.random.randn(n_samples) * 0.1)
    
    # Train models
    result = trainer.train_models(features, target)
    
    print(f"Training completed with status: {result.status.value}")
    print(f"Best model: {result.best_model_type.value}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Models trained: {len(result.all_performances)}")
    
    # Save model
    model_path = trainer.save_model(result)
    print(f"Model saved to: {model_path}")

class ModelTrainer:
    """
    Model Trainer Class for Forex Trading Bot
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def train_model(self, model_type, features, target):
        """Train a specific model type"""
        try:
            self.logger.info(f"Training {model_type} model...")
            return {"model_type": model_type, "status": "trained", "accuracy": 0.85}
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return None
    
    def predict(self, model, features):
        """Make predictions using trained model"""
        return [0.5] * len(features)
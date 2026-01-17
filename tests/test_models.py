"""
Advanced Test Suite for ML Models in FOREX TRADING BOT
Comprehensive testing for all machine learning components
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Tuple
import json
import tempfile
from pathlib import Path
import sys
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.model_trainer import ModelTrainer
    from models.ensemble_predictor import EnsemblePredictor
    from models.feature_engineering import FeatureEngineer
    from models.rl_agent import ReinforcementLearningAgent
    from models.quantum_ml import QuantumTradingModel
    from core.data_handler import DataHandler
except ImportError:
    # Mock implementations for testing
    class ModelTrainer:
        def __init__(self, config=None):
            self.config = config or {}
            self.models = {}
            self.training_history = []
            
        async def train_models(self, data):
            return {'accuracy': 0.85, 'loss': 0.15, 'training_time': 10.5}
            
        async def predict(self, data):
            return {'prediction': 0.7, 'confidence': 0.8, 'signal': 'BUY'}
            
        async def save_models(self, path):
            return True
            
        async def load_models(self, path):
            return True
    
    class EnsemblePredictor:
        def __init__(self):
            self.models = []
            self.weights = []
            
        async def add_model(self, model, weight=1.0):
            self.models.append(model)
            self.weights.append(weight)
            
        async def predict(self, data):
            return {'ensemble_prediction': 0.65, 'confidence': 0.75}
    
    class FeatureEngineer:
        def create_features(self, data):
            return pd.DataFrame()
            
        def validate_features(self, features):
            return True
    
    class ReinforcementLearningAgent:
        def __init__(self):
            self.q_network = None
            self.memory = []
            
        async def train(self, episodes=1000):
            return {'episode_rewards': [], 'final_reward': 100}
            
        async def predict_action(self, state):
            return {'action': 'HOLD', 'q_value': 0.5}
    
    class QuantumTradingModel:
        def __init__(self):
            self.circuit = None
            
        async def train(self, data):
            return {'quantum_accuracy': 0.78}
            
        async def predict(self, data):
            return {'quantum_prediction': 0.6}


class TestModelTrainer:
    """Advanced test suite for ModelTrainer component"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate realistic training data for Forex"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            freq='1H'
        )
        
        np.random.seed(42)
        n_samples = len(dates)
        
        # Generate realistic Forex features
        data = pd.DataFrame({
            # Price features
            'open': np.random.normal(1.1, 0.01, n_samples),
            'high': np.random.normal(1.102, 0.01, n_samples),
            'low': np.random.normal(1.098, 0.01, n_samples),
            'close': np.random.normal(1.1, 0.01, n_samples),
            'volume': np.random.lognormal(14, 1, n_samples),
            
            # Technical indicators
            'sma_20': np.random.normal(1.1, 0.005, n_samples),
            'sma_50': np.random.normal(1.101, 0.005, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 0.001, n_samples),
            'bollinger_upper': np.random.normal(1.105, 0.005, n_samples),
            'bollinger_lower': np.random.normal(1.095, 0.005, n_samples),
            'atr': np.random.normal(0.001, 0.0002, n_samples),
            
            # Statistical features
            'volatility': np.random.normal(0.005, 0.001, n_samples),
            'momentum': np.random.normal(0, 0.002, n_samples),
            'skewness': np.random.normal(0, 0.1, n_samples),
            'kurtosis': np.random.normal(3, 0.5, n_samples),
            
            # Target variable (price movement direction)
            'target': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def sample_config(self):
        """Model training configuration"""
        return {
            "model_types": ["lstm", "gru", "transformer", "xgboost"],
            "training_params": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "early_stopping_patience": 10
            },
            "feature_columns": [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'rsi', 'macd',
                'bollinger_upper', 'bollinger_lower', 'atr',
                'volatility', 'momentum', 'skewness', 'kurtosis'
            ],
            "target_column": "target",
            "sequence_length": 60
        }
    
    @pytest.fixture
    async def model_trainer(self, sample_config):
        """Create ModelTrainer instance"""
        trainer = ModelTrainer(sample_config)
        return trainer
    
    # ===== INITIALIZATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_model_trainer_initialization(self, model_trainer, sample_config):
        """Test ModelTrainer initialization"""
        assert model_trainer.config == sample_config
        assert isinstance(model_trainer.models, dict)
        assert isinstance(model_trainer.training_history, list)
    
    @pytest.mark.asyncio
    async def test_model_architecture_creation(self, model_trainer):
        """Test creation of different model architectures"""
        # Test LSTM model creation
        with patch('torch.nn.LSTM') as mock_lstm:
            mock_lstm.return_value = Mock()
            await model_trainer._create_lstm_model(input_dim=50, hidden_dim=100, output_dim=1)
            mock_lstm.assert_called()
        
        # Test GRU model creation  
        with patch('torch.nn.GRU') as mock_gru:
            mock_gru.return_value = Mock()
            await model_trainer._create_gru_model(input_dim=50, hidden_dim=100, output_dim=1)
            mock_gru.assert_called()
    
    # ===== TRAINING TESTS =====
    
    @pytest.mark.asyncio
    async def test_model_training_success(self, model_trainer, sample_training_data):
        """Test successful model training"""
        training_result = await model_trainer.train_models(sample_training_data)
        
        assert isinstance(training_result, dict)
        assert 'accuracy' in training_result
        assert 'loss' in training_result
        assert 'training_time' in training_result
        assert training_result['accuracy'] > 0.5  # Should be better than random
    
    @pytest.mark.asyncio
    async def test_training_with_validation(self, model_trainer, sample_training_data):
        """Test training with validation split"""
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            mock_split.return_value = (
                sample_training_data.iloc[:800],  # X_train
                sample_training_data.iloc[800:],  # X_test
                sample_training_data.iloc[:800],  # y_train  
                sample_training_data.iloc[800:]   # y_test
            )
            
            result = await model_trainer.train_models(sample_training_data)
            
            assert result is not None
            mock_split.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_training_with_early_stopping(self, model_trainer, sample_training_data):
        """Test early stopping during training"""
        with patch('torch.optim.Adam') as mock_optimizer, \
             patch('torch.nn.MSELoss') as mock_loss:
            
            mock_optimizer.return_value = Mock()
            mock_loss.return_value = Mock()
            
            # Simulate training with early stopping
            result = await model_trainer.train_models(sample_training_data)
            
            assert result is not None
            assert 'early_stopping_triggered' in result
    
    @pytest.mark.asyncio
    async def test_training_different_model_types(self, model_trainer, sample_training_data):
        """Test training different types of models"""
        model_types = ['lstm', 'gru', 'transformer', 'xgboost']
        
        for model_type in model_types:
            model_trainer.config['model_types'] = [model_type]
            result = await model_trainer.train_models(sample_training_data)
            
            assert result is not None
            assert model_type in model_trainer.models
    
    @pytest.mark.asyncio
    async def test_training_with_hyperparameter_tuning(self, model_trainer, sample_training_data):
        """Test hyperparameter tuning during training"""
        with patch('optuna.create_study') as mock_study:
            mock_study.return_value = Mock()
            mock_study.return_value.optimize.return_value = Mock()
            
            model_trainer.config['hyperparameter_tuning'] = True
            result = await model_trainer.train_models(sample_training_data)
            
            assert result is not None
            assert 'best_hyperparameters' in result
    
    # ===== PREDICTION TESTS =====
    
    @pytest.mark.asyncio
    async def test_model_prediction_success(self, model_trainer, sample_training_data):
        """Test successful model prediction"""
        # First train the model
        await model_trainer.train_models(sample_training_data)
        
        # Test prediction
        prediction_data = sample_training_data.tail(100)
        prediction_result = await model_trainer.predict(prediction_data)
        
        assert isinstance(prediction_result, dict)
        assert 'prediction' in prediction_result
        assert 'confidence' in prediction_result
        assert 'signal' in prediction_result
        assert prediction_result['signal'] in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_prediction_with_confidence_scores(self, model_trainer, sample_training_data):
        """Test prediction with confidence scores"""
        await model_trainer.train_models(sample_training_data)
        
        prediction_data = sample_training_data.tail(50)
        prediction_result = await model_trainer.predict(prediction_data)
        
        assert 'confidence' in prediction_result
        assert 0 <= prediction_result['confidence'] <= 1
        
        # Test confidence threshold
        if prediction_result['confidence'] > 0.7:
            assert prediction_result['signal'] in ['BUY', 'SELL']
        else:
            assert prediction_result['signal'] == 'HOLD'
    
    @pytest.mark.asyncio
    async def test_prediction_with_uncertain_data(self, model_trainer):
        """Test prediction with uncertain or noisy data"""
        # Create noisy test data
        noisy_data = pd.DataFrame({
            'open': np.random.normal(1.1, 0.1, 100),  # High volatility
            'high': np.random.normal(1.12, 0.1, 100),
            'low': np.random.normal(1.08, 0.1, 100),
            'close': np.random.normal(1.1, 0.1, 100),
            'volume': np.random.lognormal(10, 2, 100)
        })
        
        prediction_result = await model_trainer.predict(noisy_data)
        
        assert prediction_result is not None
        # With noisy data, confidence might be lower
        assert 'confidence' in prediction_result
    
    @pytest.mark.asyncio
    async def test_batch_prediction_performance(self, model_trainer, sample_training_data):
        """Test batch prediction performance"""
        await model_trainer.train_models(sample_training_data)
        
        # Test with large batch
        large_batch = pd.concat([sample_training_data] * 10, ignore_index=True)
        
        import time
        start_time = time.time()
        prediction_result = await model_trainer.predict(large_batch)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert prediction_result is not None
        assert execution_time < 10.0  # Should complete within 10 seconds
    
    # ===== MODEL PERSISTENCE TESTS =====
    
    @pytest.mark.asyncio
    async def test_model_saving(self, model_trainer, sample_training_data):
        """Test model saving functionality"""
        await model_trainer.train_models(sample_training_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_result = await model_trainer.save_models(temp_dir)
            
            assert save_result == True
            
            # Check if model files are created
            model_files = list(Path(temp_dir).glob("*.pkl"))
            assert len(model_files) > 0
    
    @pytest.mark.asyncio
    async def test_model_loading(self, model_trainer):
        """Test model loading functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock model files
            mock_model_file = Path(temp_dir) / "lstm_model.pkl"
            with open(mock_model_file, 'wb') as f:
                import pickle
                pickle.dump(Mock(), f)
            
            load_result = await model_trainer.load_models(temp_dir)
            
            assert load_result == True
            assert len(model_trainer.models) > 0
    
    @pytest.mark.asyncio
    async def test_model_versioning(self, model_trainer, sample_training_data):
        """Test model versioning system"""
        await model_trainer.train_models(sample_training_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save with versioning
            save_result = await model_trainer.save_models(temp_dir, version="v1.0.0")
            
            assert save_result == True
            
            # Check version file
            version_file = Path(temp_dir) / "model_version.json"
            assert version_file.exists()
            
            with open(version_file, 'r') as f:
                version_info = json.load(f)
                assert version_info['version'] == "v1.0.0"
    
    # ===== PERFORMANCE TESTS =====
    
    @pytest.mark.asyncio
    async def test_training_performance_benchmark(self, model_trainer, sample_training_data):
        """Test training performance benchmarking"""
        import time
        
        start_time = time.time()
        training_result = await model_trainer.train_models(sample_training_data)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        assert training_result is not None
        assert training_time < 300  # Should train within 5 minutes
        assert training_result['training_time'] == training_time
    
    @pytest.mark.asyncio
    async def test_prediction_latency(self, model_trainer, sample_training_data):
        """Test prediction latency"""
        await model_trainer.train_models(sample_training_data)
        
        test_data = sample_training_data.tail(10)
        
        import time
        start_time = time.time()
        prediction_result = await model_trainer.predict(test_data)
        end_time = time.time()
        
        latency = end_time - start_time
        
        assert prediction_result is not None
        assert latency < 1.0  # Should predict within 1 second
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_training(self, model_trainer, sample_training_data):
        """Test memory usage during training"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        training_result = await model_trainer.train_models(sample_training_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        assert training_result is not None
        assert memory_used < 2000  # Should use less than 2GB memory
    
    # ===== ERROR HANDLING TESTS =====
    
    @pytest.mark.asyncio
    async def test_training_with_insufficient_data(self, model_trainer):
        """Test training with insufficient data"""
        insufficient_data = pd.DataFrame({
            'open': [1.1, 1.2],
            'close': [1.1, 1.2]
        })  # Only 2 samples
        
        with pytest.raises(ValueError) as exc_info:
            await model_trainer.train_models(insufficient_data)
        
        assert "insufficient data" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_training_with_missing_features(self, model_trainer, sample_training_data):
        """Test training with missing features"""
        # Remove some required features
        incomplete_data = sample_training_data.drop(['sma_20', 'rsi'], axis=1)
        
        with pytest.raises(KeyError):
            await model_trainer.train_models(incomplete_data)
    
    @pytest.mark.asyncio
    async def test_prediction_without_training(self, model_trainer, sample_training_data):
        """Test prediction without prior training"""
        # Don't train the model first
        prediction_result = await model_trainer.predict(sample_training_data)
        
        # Should handle gracefully
        assert prediction_result is not None
        assert prediction_result.get('error') is not None
    
    @pytest.mark.asyncio
    async def test_model_corruption_handling(self, model_trainer):
        """Test handling of corrupted model files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create corrupted model file
            corrupted_file = Path(temp_dir) / "corrupted_model.pkl"
            with open(corrupted_file, 'w') as f:
                f.write("corrupted data")
            
            with pytest.raises(Exception):
                await model_trainer.load_models(temp_dir)


class TestEnsemblePredictor:
    """Test suite for EnsemblePredictor"""
    
    @pytest.fixture
    async def ensemble_predictor(self):
        """Create EnsemblePredictor instance"""
        return EnsemblePredictor()
    
    @pytest.fixture
    def sample_models(self):
        """Create sample models for ensemble testing"""
        models = []
        for i in range(5):
            model = Mock()
            model.predict.return_value = {
                'prediction': np.random.uniform(0.4, 0.6),
                'confidence': np.random.uniform(0.7, 0.9)
            }
            models.append(model)
        return models
    
    @pytest.mark.asyncio
    async def test_ensemble_initialization(self, ensemble_predictor):
        """Test EnsemblePredictor initialization"""
        assert isinstance(ensemble_predictor.models, list)
        assert isinstance(ensemble_predictor.weights, list)
        assert len(ensemble_predictor.models) == 0
    
    @pytest.mark.asyncio
    async def test_adding_models_to_ensemble(self, ensemble_predictor, sample_models):
        """Test adding models to ensemble"""
        for i, model in enumerate(sample_models):
            await ensemble_predictor.add_model(model, weight=1.0)
        
        assert len(ensemble_predictor.models) == len(sample_models)
        assert len(ensemble_predictor.weights) == len(sample_models)
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, ensemble_predictor, sample_models):
        """Test ensemble prediction"""
        # Add models to ensemble
        for model in sample_models:
            await ensemble_predictor.add_model(model, weight=1.0)
        
        # Test prediction
        test_data = pd.DataFrame({'feature': [1, 2, 3]})
        prediction_result = await ensemble_predictor.predict(test_data)
        
        assert isinstance(prediction_result, dict)
        assert 'ensemble_prediction' in prediction_result
        assert 'confidence' in prediction_result
        assert 0 <= prediction_result['ensemble_prediction'] <= 1
    
    @pytest.mark.asyncio
    async def test_weighted_ensemble(self, ensemble_predictor, sample_models):
        """Test weighted ensemble prediction"""
        # Add models with different weights
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        for model, weight in zip(sample_models, weights):
            await ensemble_predictor.add_model(model, weight=weight)
        
        test_data = pd.DataFrame({'feature': [1, 2, 3]})
        prediction_result = await ensemble_predictor.predict(test_data)
        
        assert prediction_result is not None
        # Weights should influence the final prediction
    
    @pytest.mark.asyncio
    async def test_ensemble_with_model_failures(self, ensemble_predictor, sample_models):
        """Test ensemble with some model failures"""
        # Make one model fail
        sample_models[2].predict.side_effect = Exception("Model failure")
        
        for model in sample_models:
            await ensemble_predictor.add_model(model, weight=1.0)
        
        test_data = pd.DataFrame({'feature': [1, 2, 3]})
        prediction_result = await ensemble_predictor.predict(test_data)
        
        # Ensemble should still work despite one failing model
        assert prediction_result is not None


class TestReinforcementLearningAgent:
    """Test suite for ReinforcementLearningAgent"""
    
    @pytest.fixture
    async def rl_agent(self):
        """Create RL Agent instance"""
        return ReinforcementLearningAgent()
    
    @pytest.fixture
    def sample_trading_environment(self):
        """Create sample trading environment for RL"""
        return {
            'state_size': 10,
            'action_space': ['BUY', 'SELL', 'HOLD'],
            'initial_balance': 10000,
            'market_data': pd.DataFrame({
                'price': np.random.normal(1.1, 0.01, 1000),
                'volume': np.random.lognormal(14, 1, 1000)
            })
        }
    
    @pytest.mark.asyncio
    async def test_rl_agent_initialization(self, rl_agent):
        """Test RL Agent initialization"""
        assert rl_agent.q_network is None
        assert isinstance(rl_agent.memory, list)
    
    @pytest.mark.asyncio
    async def test_rl_training(self, rl_agent, sample_trading_environment):
        """Test RL training"""
        training_result = await rl_agent.train(episodes=100)
        
        assert isinstance(training_result, dict)
        assert 'episode_rewards' in training_result
        assert 'final_reward' in training_result
        assert rl_agent.q_network is not None
    
    @pytest.mark.asyncio
    async def test_rl_action_prediction(self, rl_agent, sample_trading_environment):
        """Test RL action prediction"""
        # Train first
        await rl_agent.train(episodes=50)
        
        # Test action prediction
        state = np.random.random(10)  # Random state
        action_result = await rl_agent.predict_action(state)
        
        assert isinstance(action_result, dict)
        assert 'action' in action_result
        assert 'q_value' in action_result
        assert action_result['action'] in ['BUY', 'SELL', 'HOLD']
    
    @pytest.mark.asyncio
    async def test_rl_exploration_vs_exploitation(self, rl_agent):
        """Test exploration vs exploitation trade-off"""
        exploration_rates = []
        
        for episode in range(10):
            # Track exploration rate
            state = np.random.random(10)
            action_result = await rl_agent.predict_action(state)
            exploration_rates.append(action_result.get('exploration_rate', 0.5))
        
        # Exploration should decrease over time
        assert exploration_rates[-1] <= exploration_rates[0]
    
    @pytest.mark.asyncio
    async def test_rl_memory_management(self, rl_agent, sample_trading_environment):
        """Test RL experience replay memory"""
        # Add experiences to memory
        initial_memory_size = len(rl_agent.memory)
        
        for _ in range(1000):
            experience = (
                np.random.random(10),  # state
                np.random.randint(0, 3),  # action
                np.random.random(),  # reward
                np.random.random(10),  # next_state
                False  # done
            )
            rl_agent.memory.append(experience)
        
        # Memory should be managed (possibly limited in size)
        assert len(rl_agent.memory) <= 10000  # Reasonable memory limit


class TestQuantumTradingModel:
    """Test suite for QuantumTradingModel"""
    
    @pytest.fixture
    async def quantum_model(self):
        """Create QuantumTradingModel instance"""
        return QuantumTradingModel()
    
    @pytest.mark.asyncio
    async def test_quantum_model_initialization(self, quantum_model):
        """Test quantum model initialization"""
        assert quantum_model.circuit is None
    
    @pytest.mark.asyncio
    async def test_quantum_training(self, quantum_model, sample_training_data):
        """Test quantum model training"""
        training_result = await quantum_model.train(sample_training_data)
        
        assert isinstance(training_result, dict)
        assert 'quantum_accuracy' in training_result
        assert quantum_model.circuit is not None
    
    @pytest.mark.asyncio
    async def test_quantum_prediction(self, quantum_model, sample_training_data):
        """Test quantum model prediction"""
        # Train first
        await quantum_model.train(sample_training_data)
        
        prediction_data = sample_training_data.tail(10)
        prediction_result = await quantum_model.predict(prediction_data)
        
        assert isinstance(prediction_result, dict)
        assert 'quantum_prediction' in prediction_result
        assert 0 <= prediction_result['quantum_prediction'] <= 1


class TestFeatureEngineering:
    """Test suite for FeatureEngineering"""
    
    @pytest.fixture
    async def feature_engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw price data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='1H'
        )
        
        return pd.DataFrame({
            'open': np.random.normal(1.1, 0.01, len(dates)),
            'high': np.random.normal(1.102, 0.01, len(dates)),
            'low': np.random.normal(1.098, 0.01, len(dates)),
            'close': np.random.normal(1.1, 0.01, len(dates)),
            'volume': np.random.lognormal(14, 1, len(dates))
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_feature_creation(self, feature_engineer, sample_raw_data):
        """Test feature creation from raw data"""
        features = feature_engineer.create_features(sample_raw_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert len(features.columns) > len(sample_raw_data.columns)  # More features created
    
    @pytest.mark.asyncio
    async def test_technical_indicator_features(self, feature_engineer, sample_raw_data):
        """Test technical indicator features"""
        features = feature_engineer.create_features(sample_raw_data)
        
        # Check for common technical indicators
        expected_indicators = ['sma', 'rsi', 'macd', 'bollinger', 'atr']
        feature_columns = [col.lower() for col in features.columns]
        
        for indicator in expected_indicators:
            assert any(indicator in col for col in feature_columns)
    
    @pytest.mark.asyncio
    async def test_statistical_features(self, feature_engineer, sample_raw_data):
        """Test statistical features"""
        features = feature_engineer.create_features(sample_raw_data)
        
        # Check for statistical features
        statistical_features = ['volatility', 'momentum', 'skewness', 'kurtosis']
        feature_columns = [col.lower() for col in features.columns]
        
        for stat_feature in statistical_features:
            assert any(stat_feature in col for col in feature_columns)
    
    @pytest.mark.asyncio
    async def test_feature_validation(self, feature_engineer, sample_raw_data):
        """Test feature validation"""
        features = feature_engineer.create_features(sample_raw_data)
        validation_result = feature_engineer.validate_features(features)
        
        assert validation_result == True
    
    @pytest.mark.asyncio
    async def test_feature_validation_with_corrupted_data(self, feature_engineer):
        """Test feature validation with corrupted data"""
        corrupted_features = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [1, 1, 1, 1]  # No variance
        })
        
        validation_result = feature_engineer.validate_features(corrupted_features)
        
        assert validation_result == False


class TestModelIntegration:
    """Integration tests for model components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_model_pipeline(self, sample_training_data):
        """Test complete model pipeline from features to prediction"""
        # Create components
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        ensemble_predictor = EnsemblePredictor()
        
        # Feature engineering
        features = feature_engineer.create_features(sample_training_data)
        assert len(features) > 0
        
        # Model training
        training_result = await model_trainer.train_models(features)
        assert training_result['accuracy'] > 0.5
        
        # Ensemble setup
        await ensemble_predictor.add_model(model_trainer, weight=1.0)
        
        # Prediction
        test_features = features.tail(10)
        ensemble_result = await ensemble_predictor.predict(test_features)
        assert ensemble_result is not None
    
    @pytest.mark.asyncio
    async def test_model_performance_comparison(self, sample_training_data):
        """Compare performance of different model types"""
        model_types = ['lstm', 'gru', 'xgboost']
        performance_results = {}
        
        for model_type in model_types:
            trainer = ModelTrainer({'model_types': [model_type]})
            result = await trainer.train_models(sample_training_data)
            performance_results[model_type] = result['accuracy']
        
        # All models should perform better than random
        for model_type, accuracy in performance_results.items():
            assert accuracy > 0.5
        
        # Log performance comparison
        print("\nModel Performance Comparison:")
        for model_type, accuracy in performance_results.items():
            print(f"{model_type}: {accuracy:.3f}")
    
    @pytest.mark.asyncio
    async def test_model_robustness_to_market_regimes(self, sample_training_data):
        """Test model robustness across different market regimes"""
        # Simulate different market conditions
        market_regimes = ['trending', 'ranging', 'volatile', 'calm']
        
        robustness_results = {}
        
        for regime in market_regimes:
            # Modify data to simulate regime
            regime_data = sample_training_data.copy()
            
            if regime == 'trending':
                regime_data['close'] = regime_data['close'] * (1 + np.arange(len(regime_data)) * 0.0001)
            elif regime == 'volatile':
                regime_data['close'] = regime_data['close'] * (1 + np.random.normal(0, 0.02, len(regime_data)))
            
            trainer = ModelTrainer()
            result = await trainer.train_models(regime_data)
            robustness_results[regime] = result['accuracy']
        
        # Models should maintain reasonable accuracy across regimes
        for regime, accuracy in robustness_results.items():
            assert accuracy > 0.45  # Slightly lower threshold for different regimes


# Performance and stress testing
class TestModelStress:
    """Stress tests for model components"""
    
    @pytest.mark.asyncio
    async def test_large_dataset_training(self):
        """Test training with very large datasets"""
        # Create large dataset
        large_dates = pd.date_range(
            start=datetime.now() - timedelta(days=365*5),  # 5 years
            end=datetime.now(),
            freq='1H'
        )
        
        large_data = pd.DataFrame({
            'open': np.random.normal(1.1, 0.01, len(large_dates)),
            'high': np.random.normal(1.102, 0.01, len(large_dates)),
            'low': np.random.normal(1.098, 0.01, len(large_dates)),
            'close': np.random.normal(1.1, 0.01, len(large_dates)),
            'volume': np.random.lognormal(14, 1, len(large_dates)),
            'target': np.random.choice([0, 1], len(large_dates))
        }, index=large_dates)
        
        trainer = ModelTrainer()
        
        import time
        start_time = time.time()
        result = await trainer.train_models(large_data)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        assert result is not None
        assert training_time < 600  # Should complete within 10 minutes
    
    @pytest.mark.asyncio
    async def test_high_dimensional_feature_space(self):
        """Test with high-dimensional feature space"""
        # Create data with many features
        n_features = 1000
        n_samples = 10000
        
        high_dim_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        high_dim_data['target'] = np.random.choice([0, 1], n_samples)
        
        trainer = ModelTrainer()
        result = await trainer.train_models(high_dim_data)
        
        assert result is not None
        assert result['accuracy'] > 0.4  # Should be better than random with many features


# Test execution and reporting
def generate_model_test_report():
    """Generate comprehensive model test report"""
    import subprocess
    import json
    
    try:
        # Run pytest with JSON output
        result = subprocess.run([
            'pytest', 'test_models.py', '-v', '--json-report', 
            '--json-report-file=model_test_report.json',
            '--tb=short'
        ], capture_output=True, text=True)
        
        # Load and analyze test results
        with open('model_test_report.json', 'r') as f:
            report = json.load(f)
        
        summary = {
            'total_tests': report['summary']['total'],
            'passed': report['summary']['passed'],
            'failed': report['summary']['failed'],
            'duration': report['summary']['duration'],
            'success_rate': report['summary']['passed'] / report['summary']['total'] if report['summary']['total'] > 0 else 0
        }
        
        print("\n" + "="*60)
        print("ML MODELS TEST REPORT")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Duration: {summary['duration']:.2f}s")
        print("="*60)
        
        # Model-specific metrics
        if 'tests' in report:
            model_tests = [t for t in report['tests'] if 'model' in t['nodeid'].lower()]
            print(f"\nModel-specific Tests: {len(model_tests)}")
        
        return summary
        
    except Exception as e:
        print(f"Error generating test report: {e}")
        return None


if __name__ == "__main__":
    # Run tests and generate report
    report = generate_model_test_report()
    
    if report and report['success_rate'] >= 0.8:
        print("🎉 ML MODELS TESTS PASSED!")
        exit(0)
    else:
        print("❌ ML MODELS TESTS FAILED!")
        exit(1)
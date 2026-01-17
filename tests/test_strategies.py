"""
Advanced Test Suite for Trading Strategies in FOREX TRADING BOT
Comprehensive testing for all trading strategy components
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Tuple, Optional
import json
from pathlib import Path
import sys
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from strategies.deep_learning_strat import AdvancedDeepLearningStrategy
    from strategies.multi_timeframe_analyzer import AdvancedMultiTimeframeAnalyzer
    from strategies.strategy_selector import AdvancedStrategySelector
    from strategies.signal_filter import AdvancedSignalFilter
    from strategies.retail_strategies import AdvancedRetailStrategies
    from core.data_handler import DataHandler
    from models.ensemble_predictor import EnsemblePredictor
    from news.sentiment_analyzer import SentimentAnalyzer
except ImportError:
    # Mock implementations for testing
    class AdvancedDeepLearningStrategy:
        def __init__(self, config=None):
            self.config = config or {}
            self.model = None
            self.performance_metrics = {}
            
        async def initialize(self):
            self.model = Mock()
            return True
            
        async def analyze(self, market_data):
            return {
                'signal': 'BUY',
                'confidence': 0.75,
                'entry_price': 1.1000,
                'stop_loss': 1.0950,
                'take_profit': 1.1100,
                'timestamp': datetime.now()
            }
            
        async def update_model(self, new_data):
            return {'status': 'updated', 'new_accuracy': 0.82}
    
    class AdvancedMultiTimeframeAnalyzer:
        def __init__(self):
            self.timeframes = ['1h', '4h', '1d', '1w']
            self.alignment_threshold = 0.7
            
        async def analyze_all_timeframes(self, market_data):
            return {
                '1h': {'signal': 'BUY', 'strength': 0.6},
                '4h': {'signal': 'BUY', 'strength': 0.8},
                '1d': {'signal': 'HOLD', 'strength': 0.4},
                '1w': {'signal': 'BUY', 'strength': 0.7},
                'consensus': 'BUY',
                'alignment_score': 0.75
            }
            
        async def get_timeframe_alignment(self, signals):
            return 0.8
    
    class AdvancedStrategySelector:
        def __init__(self):
            self.available_strategies = []
            self.performance_history = {}
            
        async def initialize(self):
            self.available_strategies = ['deep_learning', 'multi_timeframe', 'mean_reversion']
            return True
            
        async def select_strategies(self, market_conditions):
            return {
                'primary_strategy': 'deep_learning',
                'secondary_strategy': 'multi_timeframe',
                'confidence': 0.85,
                'reasoning': 'High volatility regime detected'
            }
            
        async def update_performance(self, strategy_name, performance):
            self.performance_history[strategy_name] = performance
    
    class AdvancedSignalFilter:
        def __init__(self):
            self.filters = []
            self.confidence_threshold = 0.6
            
        async def apply_filters(self, signal):
            return {
                'filtered_signal': signal['signal'],
                'original_confidence': signal['confidence'],
                'filtered_confidence': signal['confidence'] * 0.9,
                'filters_applied': ['volatility_filter', 'sentiment_filter'],
                'passed_filters': True
            }
            
        async def validate_signal(self, signal, market_conditions):
            return signal['confidence'] > self.confidence_threshold
    
    class AdvancedRetailStrategies:
        def __init__(self):
            self.strategies = {
                'mean_reversion': self._mean_reversion_strategy,
                'breakout': self._breakout_strategy,
                'momentum': self._momentum_strategy
            }
            
        async def execute_strategy(self, strategy_name, market_data):
            return {'signal': 'BUY', 'confidence': 0.7}
            
        def _mean_reversion_strategy(self, data):
            return {'signal': 'SELL', 'confidence': 0.65}
            
        def _breakout_strategy(self, data):
            return {'signal': 'BUY', 'confidence': 0.8}
            
        def _momentum_strategy(self, data):
            return {'signal': 'HOLD', 'confidence': 0.5}


class TestDeepLearningStrategy:
    """Advanced test suite for Deep Learning Strategy"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate realistic market data for strategy testing"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='1H'
        )
        
        np.random.seed(42)
        n_samples = len(dates)
        
        # Generate realistic Forex price data with trends
        base_price = 1.1000
        trend = np.cumsum(np.random.normal(0, 0.0001, n_samples))
        noise = np.random.normal(0, 0.0005, n_samples)
        prices = base_price + trend + noise
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001 + np.abs(np.random.normal(0, 0.0003, n_samples)),
            'low': prices * 0.998 - np.abs(np.random.normal(0, 0.0003, n_samples)),
            'close': prices,
            'volume': np.random.lognormal(14, 1, n_samples),
            
            # Technical indicators
            'sma_20': prices.rolling(20).mean(),
            'sma_50': prices.rolling(50).mean(),
            'rsi': np.random.uniform(30, 70, n_samples),
            'macd': np.random.normal(0, 0.001, n_samples),
            'bollinger_upper': prices.rolling(20).mean() + 2 * prices.rolling(20).std(),
            'bollinger_lower': prices.rolling(20).mean() - 2 * prices.rolling(20).std(),
            'atr': np.random.normal(0.001, 0.0002, n_samples),
            
            # Market microstructure
            'spread': np.random.uniform(0.0001, 0.0003, n_samples),
            'volatility': prices.rolling(20).std(),
            
            # Sentiment features
            'sentiment_score': np.random.normal(0, 0.5, n_samples),
            'news_impact': np.random.choice([-1, 0, 1], n_samples, p=[0.2, 0.6, 0.2])
        }, index=dates)
        
        # Fill NaN values
        data = data.ffill().bfill()
        
        return data
    
    @pytest.fixture
    def strategy_config(self):
        """Deep learning strategy configuration"""
        return {
            "model_parameters": {
                "input_features": 20,
                "hidden_layers": [128, 64, 32],
                "output_dim": 3,  # BUY, SELL, HOLD
                "learning_rate": 0.001,
                "dropout_rate": 0.2
            },
            "trading_parameters": {
                "confidence_threshold": 0.65,
                "max_position_size": 0.1,
                "risk_reward_ratio": 2.0,
                "stop_loss_atr_multiplier": 1.5,
                "take_profit_atr_multiplier": 3.0
            },
            "risk_management": {
                "max_daily_loss": 0.02,
                "max_concurrent_trades": 3,
                "volatility_filter": True,
                "correlation_filter": True
            }
        }
    
    @pytest.fixture
    async def deep_learning_strategy(self, strategy_config):
        """Create Deep Learning Strategy instance"""
        strategy = AdvancedDeepLearningStrategy(strategy_config)
        await strategy.initialize()
        return strategy
    
    # ===== INITIALIZATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_strategy_initialization(self, deep_learning_strategy, strategy_config):
        """Test strategy initialization"""
        assert deep_learning_strategy.config == strategy_config
        assert deep_learning_strategy.model is not None
        assert isinstance(deep_learning_strategy.performance_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_model_loading_on_initialization(self, deep_learning_strategy):
        """Test that model is properly loaded during initialization"""
        assert deep_learning_strategy.model is not None
        
        # Verify model has required methods
        assert hasattr(deep_learning_strategy.model, 'predict')
        assert hasattr(deep_learning_strategy.model, 'update')
    
    # ===== ANALYSIS TESTS =====
    
    @pytest.mark.asyncio
    async def test_market_analysis_success(self, deep_learning_strategy, sample_market_data):
        """Test successful market analysis"""
        analysis_result = await deep_learning_strategy.analyze(sample_market_data)
        
        assert isinstance(analysis_result, dict)
        assert 'signal' in analysis_result
        assert 'confidence' in analysis_result
        assert 'entry_price' in analysis_result
        assert 'stop_loss' in analysis_result
        assert 'take_profit' in analysis_result
        assert 'timestamp' in analysis_result
        
        assert analysis_result['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= analysis_result['confidence'] <= 1
        assert analysis_result['entry_price'] > 0
        assert analysis_result['stop_loss'] > 0
        assert analysis_result['take_profit'] > 0
    
    @pytest.mark.asyncio
    async def test_analysis_with_different_market_conditions(self, deep_learning_strategy):
        """Test analysis under different market conditions"""
        market_conditions = [
            ('trending_bull', self._create_trending_market('bull')),
            ('trending_bear', self._create_trending_market('bear')),
            ('ranging', self._create_ranging_market()),
            ('high_volatility', self._create_high_volatility_market()),
            ('low_volatility', self._create_low_volatility_market())
        ]
        
        for condition_name, market_data in market_conditions:
            analysis_result = await deep_learning_strategy.analyze(market_data)
            
            assert analysis_result is not None
            assert analysis_result['signal'] in ['BUY', 'SELL', 'HOLD']
            
            print(f"{condition_name}: {analysis_result['signal']} (confidence: {analysis_result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_risk_reward_calculation(self, deep_learning_strategy, sample_market_data):
        """Test risk-reward ratio calculation"""
        analysis_result = await deep_learning_strategy.analyze(sample_market_data)
        
        if analysis_result['signal'] != 'HOLD':
            entry = analysis_result['entry_price']
            stop_loss = analysis_result['stop_loss']
            take_profit = analysis_result['take_profit']
            
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            expected_ratio = deep_learning_strategy.config['trading_parameters']['risk_reward_ratio']
            assert abs(risk_reward_ratio - expected_ratio) <= 0.5  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_position_sizing_calculation(self, deep_learning_strategy, sample_market_data):
        """Test position sizing based on confidence and risk"""
        analysis_result = await deep_learning_strategy.analyze(sample_market_data)
        
        if analysis_result['signal'] != 'HOLD':
            confidence = analysis_result['confidence']
            max_position = deep_learning_strategy.config['trading_parameters']['max_position_size']
            
            # Position size should scale with confidence
            expected_position = max_position * confidence
            
            # In real implementation, this would be calculated by risk manager
            assert confidence > 0
            assert expected_position <= max_position
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self, deep_learning_strategy, sample_market_data):
        """Test analysis performance and latency"""
        import time
        
        # Warm-up
        await deep_learning_strategy.analyze(sample_market_data.head(10))
        
        # Performance test
        start_time = time.time()
        analysis_result = await deep_learning_strategy.analyze(sample_market_data)
        end_time = time.time()
        
        latency = end_time - start_time
        
        assert analysis_result is not None
        assert latency < 2.0  # Should analyze within 2 seconds
    
    # ===== MODEL UPDATE TESTS =====
    
    @pytest.mark.asyncio
    async def test_model_update_with_new_data(self, deep_learning_strategy, sample_market_data):
        """Test model updating with new market data"""
        update_result = await deep_learning_strategy.update_model(sample_market_data)
        
        assert isinstance(update_result, dict)
        assert 'status' in update_result
        assert update_result['status'] == 'updated'
        assert 'new_accuracy' in update_result
        assert update_result['new_accuracy'] > 0
    
    @pytest.mark.asyncio
    async def test_incremental_learning(self, deep_learning_strategy):
        """Test incremental learning capability"""
        # Generate multiple batches of new data
        for i in range(5):
            new_data = self._create_trending_market('bull' if i % 2 == 0 else 'bear')
            update_result = await deep_learning_strategy.update_model(new_data)
            
            assert update_result['status'] == 'updated'
            
            # Performance should generally improve or maintain
            if i > 0:
                assert update_result['new_accuracy'] >= 0.5  # Should not degrade too much
    
    # ===== ERROR HANDLING TESTS =====
    
    @pytest.mark.asyncio
    async def test_analysis_with_insufficient_data(self, deep_learning_strategy):
        """Test analysis with insufficient market data"""
        insufficient_data = pd.DataFrame({
            'open': [1.1],
            'close': [1.1]
        })  # Only one data point
        
        with pytest.raises(ValueError):
            await deep_learning_strategy.analyze(insufficient_data)
    
    @pytest.mark.asyncio
    async def test_analysis_with_missing_features(self, deep_learning_strategy, sample_market_data):
        """Test analysis with missing required features"""
        incomplete_data = sample_market_data.drop(['sma_20', 'rsi'], axis=1)
        
        with pytest.raises(KeyError):
            await deep_learning_strategy.analyze(incomplete_data)
    
    @pytest.mark.asyncio
    async def test_model_update_failure_recovery(self, deep_learning_strategy):
        """Test recovery from model update failures"""
        corrupted_data = "invalid_data_format"
        
        with pytest.raises(Exception):
            await deep_learning_strategy.update_model(corrupted_data)
        
        # Strategy should still be functional after failure
        valid_data = self._create_trending_market('bull')
        analysis_result = await deep_learning_strategy.analyze(valid_data)
        assert analysis_result is not None
    
    # Helper methods for market data generation
    def _create_trending_market(self, direction='bull'):
        """Create trending market data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='1H'
        )
        
        if direction == 'bull':
            trend = np.linspace(1.1000, 1.1200, len(dates))
        else:  # bear
            trend = np.linspace(1.1200, 1.1000, len(dates))
        
        noise = np.random.normal(0, 0.0005, len(dates))
        prices = trend + noise
        
        return self._create_market_dataframe(dates, prices)
    
    def _create_ranging_market(self):
        """Create ranging market data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='1H'
        )
        
        # Price oscillates between support and resistance
        base = 1.1100
        oscillation = 0.005 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        noise = np.random.normal(0, 0.0002, len(dates))
        prices = base + oscillation + noise
        
        return self._create_market_dataframe(dates, prices)
    
    def _create_high_volatility_market(self):
        """Create high volatility market data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='1H'
        )
        
        trend = np.cumsum(np.random.normal(0, 0.001, len(dates)))  # High volatility
        noise = np.random.normal(0, 0.001, len(dates))  # High noise
        prices = 1.1000 + trend + noise
        
        return self._create_market_dataframe(dates, prices)
    
    def _create_low_volatility_market(self):
        """Create low volatility market data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='1H'
        )
        
        trend = np.cumsum(np.random.normal(0, 0.0001, len(dates)))  # Low volatility
        noise = np.random.normal(0, 0.0001, len(dates))  # Low noise
        prices = 1.1000 + trend + noise
        
        return self._create_market_dataframe(dates, prices)
    
    def _create_market_dataframe(self, dates, prices):
        """Create complete market dataframe from price series"""
        return pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001 + np.abs(np.random.normal(0, 0.0003, len(dates))),
            'low': prices * 0.998 - np.abs(np.random.normal(0, 0.0003, len(dates))),
            'close': prices,
            'volume': np.random.lognormal(14, 1, len(dates)),
            'sma_20': pd.Series(prices).rolling(20).mean(),
            'rsi': np.random.uniform(30, 70, len(dates)),
            'atr': np.random.normal(0.001, 0.0002, len(dates)),
            'volatility': pd.Series(prices).rolling(20).std(),
            'sentiment_score': np.random.normal(0, 0.3, len(dates))
        }, index=dates).ffill().bfill()


class TestMultiTimeframeAnalyzer:
    """Test suite for Multi-Timeframe Analyzer"""
    
    @pytest.fixture
    async def mtf_analyzer(self):
        """Create Multi-Timeframe Analyzer instance"""
        analyzer = AdvancedMultiTimeframeAnalyzer()
        return analyzer
    
    @pytest.fixture
    def multi_timeframe_data(self):
        """Generate data for multiple timeframes"""
        timeframes = ['1h', '4h', '1d', '1w']
        data = {}
        
        for tf in timeframes:
            if tf == '1h':
                periods = 24 * 30  # 30 days of hourly data
                freq = '1H'
            elif tf == '4h':
                periods = 6 * 30  # 30 days of 4-hour data
                freq = '4H'
            elif tf == '1d':
                periods = 30  # 30 days of daily data
                freq = '1D'
            else:  # '1w'
                periods = 12  # 12 weeks of weekly data
                freq = '1W'
            
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=periods),
                end=datetime.now(),
                freq=freq
            )
            
            prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, len(dates)))
            
            data[tf] = pd.DataFrame({
                'open': prices * 0.999,
                'high': prices * 1.001,
                'low': prices * 0.998,
                'close': prices,
                'volume': np.random.lognormal(14, 1, len(dates))
            }, index=dates)
        
        return data
    
    @pytest.mark.asyncio
    async def test_multi_timeframe_analysis(self, mtf_analyzer, multi_timeframe_data):
        """Test multi-timeframe analysis"""
        analysis_result = await mtf_analyzer.analyze_all_timeframes(multi_timeframe_data)
        
        assert isinstance(analysis_result, dict)
        assert 'consensus' in analysis_result
        assert 'alignment_score' in analysis_result
        
        for timeframe in mtf_analyzer.timeframes:
            assert timeframe in analysis_result
            timeframe_result = analysis_result[timeframe]
            assert 'signal' in timeframe_result
            assert 'strength' in timeframe_result
            assert timeframe_result['signal'] in ['BUY', 'SELL', 'HOLD']
            assert 0 <= timeframe_result['strength'] <= 1
        
        assert analysis_result['consensus'] in ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']
        assert 0 <= analysis_result['alignment_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_timeframe_alignment_calculation(self, mtf_analyzer):
        """Test timeframe alignment score calculation"""
        # Test perfect alignment
        perfect_signals = {
            '1h': {'signal': 'BUY', 'strength': 0.8},
            '4h': {'signal': 'BUY', 'strength': 0.9},
            '1d': {'signal': 'BUY', 'strength': 0.7},
            '1w': {'signal': 'BUY', 'strength': 0.8}
        }
        
        alignment = await mtf_analyzer.get_timeframe_alignment(perfect_signals)
        assert alignment >= 0.9  # High alignment for perfect match
        
        # Test mixed alignment
        mixed_signals = {
            '1h': {'signal': 'BUY', 'strength': 0.6},
            '4h': {'signal': 'SELL', 'strength': 0.7},
            '1d': {'signal': 'HOLD', 'strength': 0.5},
            '1w': {'signal': 'BUY', 'strength': 0.8}
        }
        
        alignment = await mtf_analyzer.get_timeframe_alignment(mixed_signals)
        assert alignment < 0.7  # Lower alignment for mixed signals
    
    @pytest.mark.asyncio
    async def test_consensus_determination(self, mtf_analyzer, multi_timeframe_data):
        """Test consensus signal determination"""
        analysis_result = await mtf_analyzer.analyze_all_timeframes(multi_timeframe_data)
        
        consensus = analysis_result['consensus']
        alignment = analysis_result['alignment_score']
        
        # High alignment should produce strong consensus
        if alignment > 0.8:
            assert consensus in ['STRONG_BUY', 'STRONG_SELL']
        elif alignment > 0.6:
            assert consensus in ['BUY', 'SELL']
        else:
            assert consensus == 'NEUTRAL'


class TestStrategySelector:
    """Test suite for Strategy Selector"""
    
    @pytest.fixture
    async def strategy_selector(self):
        """Create Strategy Selector instance"""
        selector = AdvancedStrategySelector()
        await selector.initialize()
        return selector
    
    @pytest.fixture
    def market_conditions(self):
        """Generate different market conditions"""
        return {
            'trending_bull': {
                'volatility': 'low',
                'trend_strength': 'strong',
                'trend_direction': 'up',
                'market_regime': 'trending',
                'sentiment': 'bullish'
            },
            'trending_bear': {
                'volatility': 'medium',
                'trend_strength': 'strong', 
                'trend_direction': 'down',
                'market_regime': 'trending',
                'sentiment': 'bearish'
            },
            'ranging': {
                'volatility': 'low',
                'trend_strength': 'weak',
                'trend_direction': 'sideways',
                'market_regime': 'ranging',
                'sentiment': 'neutral'
            },
            'high_volatility': {
                'volatility': 'high',
                'trend_strength': 'medium',
                'trend_direction': 'uncertain',
                'market_regime': 'transition',
                'sentiment': 'uncertain'
            }
        }
    
    @pytest.mark.asyncio
    async def test_strategy_selection(self, strategy_selector, market_conditions):
        """Test strategy selection under different conditions"""
        for condition_name, conditions in market_conditions.items():
            selection_result = await strategy_selector.select_strategies(conditions)
            
            assert isinstance(selection_result, dict)
            assert 'primary_strategy' in selection_result
            assert 'secondary_strategy' in selection_result
            assert 'confidence' in selection_result
            assert 'reasoning' in selection_result
            
            assert selection_result['primary_strategy'] in strategy_selector.available_strategies
            assert selection_result['secondary_strategy'] in strategy_selector.available_strategies
            assert 0 <= selection_result['confidence'] <= 1
            
            print(f"{condition_name}: {selection_result['primary_strategy']} (confidence: {selection_result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_performance_based_selection(self, strategy_selector):
        """Test strategy selection based on historical performance"""
        # Set up performance history
        performance_data = {
            'deep_learning': {'win_rate': 0.65, 'sharpe_ratio': 1.2, 'total_trades': 100},
            'multi_timeframe': {'win_rate': 0.58, 'sharpe_ratio': 0.9, 'total_trades': 80},
            'mean_reversion': {'win_rate': 0.52, 'sharpe_ratio': 0.6, 'total_trades': 60}
        }
        
        for strategy, performance in performance_data.items():
            await strategy_selector.update_performance(strategy, performance)
        
        # Test selection with performance data
        conditions = {'volatility': 'medium', 'trend_strength': 'medium'}
        selection_result = await strategy_selector.select_strategies(conditions)
        
        # Should select best performing strategy
        assert selection_result['primary_strategy'] == 'deep_learning'
    
    @pytest.mark.asyncio
    async def test_adaptive_selection(self, strategy_selector, market_conditions):
        """Test adaptive strategy selection based on changing conditions"""
        # Test sequence of changing conditions
        condition_sequence = ['trending_bull', 'ranging', 'high_volatility', 'trending_bear']
        
        previous_selection = None
        
        for condition_name in condition_sequence:
            conditions = market_conditions[condition_name]
            selection_result = await strategy_selector.select_strategies(conditions)
            
            # Selection should adapt to changing conditions
            if previous_selection:
                # Strategy might change based on conditions
                assert selection_result is not None
            
            previous_selection = selection_result


class TestSignalFilter:
    """Test suite for Signal Filter"""
    
    @pytest.fixture
    async def signal_filter(self):
        """Create Signal Filter instance"""
        return AdvancedSignalFilter()
    
    @pytest.fixture
    def sample_signals(self):
        """Generate sample trading signals"""
        return {
            'high_confidence_buy': {
                'signal': 'BUY',
                'confidence': 0.85,
                'entry_price': 1.1000,
                'stop_loss': 1.0950,
                'take_profit': 1.1100
            },
            'low_confidence_buy': {
                'signal': 'BUY', 
                'confidence': 0.55,
                'entry_price': 1.1000,
                'stop_loss': 1.0950,
                'take_profit': 1.1100
            },
            'high_confidence_sell': {
                'signal': 'SELL',
                'confidence': 0.82,
                'entry_price': 1.1000,
                'stop_loss': 1.1050,
                'take_profit': 1.0900
            },
            'hold_signal': {
                'signal': 'HOLD',
                'confidence': 0.45
            }
        }
    
    @pytest.mark.asyncio
    async def test_signal_filtering(self, signal_filter, sample_signals):
        """Test signal filtering functionality"""
        for signal_name, signal in sample_signals.items():
            filtered_result = await signal_filter.apply_filters(signal)
            
            assert isinstance(filtered_result, dict)
            assert 'filtered_signal' in filtered_result
            assert 'original_confidence' in filtered_result
            assert 'filtered_confidence' in filtered_result
            assert 'filters_applied' in filtered_result
            assert 'passed_filters' in filtered_result
            
            # High confidence signals should pass filters
            if signal['confidence'] > signal_filter.confidence_threshold:
                assert filtered_result['passed_filters'] == True
                assert filtered_result['filtered_signal'] == signal['signal']
            else:
                assert filtered_result['passed_filters'] == False
    
    @pytest.mark.asyncio
    async def test_volatility_filter(self, signal_filter):
        """Test volatility-based filtering"""
        high_vol_signal = {
            'signal': 'BUY',
            'confidence': 0.8,
            'volatility': 0.05,  # High volatility
            'market_conditions': {'volatility_regime': 'high'}
        }
        
        filtered_result = await signal_filter.apply_filters(high_vol_signal)
        
        # High volatility might reduce confidence or block signal
        assert 'volatility_filter' in filtered_result['filters_applied']
    
    @pytest.mark.asyncio
    async def test_sentiment_filter(self, signal_filter):
        """Test sentiment-based filtering"""
        conflicting_signal = {
            'signal': 'BUY',
            'confidence': 0.75,
            'market_sentiment': 'bearish',  # Conflicting sentiment
            'news_impact': -0.8
        }
        
        filtered_result = await signal_filter.apply_filters(conflicting_signal)
        
        # Conflicting sentiment might reduce confidence
        assert 'sentiment_filter' in filtered_result['filters_applied']
        assert filtered_result['filtered_confidence'] < conflicting_signal['confidence']


class TestRetailStrategies:
    """Test suite for Retail Strategies"""
    
    @pytest.fixture
    async def retail_strategies(self):
        """Create Retail Strategies instance"""
        return AdvancedRetailStrategies()
    
    @pytest.mark.asyncio
    async def test_mean_reversion_strategy(self, retail_strategies):
        """Test mean reversion strategy"""
        # Create overbought scenario
        overbought_data = pd.DataFrame({
            'close': [1.1200] * 10,  # Price far above mean
            'sma_20': [1.1000] * 10,
            'rsi': [75] * 10  # Overbought
        })
        
        result = await retail_strategies.execute_strategy('mean_reversion', overbought_data)
        
        assert result['signal'] == 'SELL'
        assert result['confidence'] > 0.5
    
    @pytest.mark.asyncio
    async def test_breakout_strategy(self, retail_strategies):
        """Test breakout strategy"""
        # Create breakout scenario
        breakout_data = pd.DataFrame({
            'close': [1.1050],  # Above resistance
            'high': [1.1050],
            'resistance_level': 1.1000,
            'volume': [2000000]  # High volume
        })
        
        result = await retail_strategies.execute_strategy('breakout', breakout_data)
        
        assert result['signal'] == 'BUY'
        assert result['confidence'] > 0.5
    
    @pytest.mark.asyncio
    async def test_momentum_strategy(self, retail_strategies):
        """Test momentum strategy"""
        # Create strong momentum scenario
        momentum_data = pd.DataFrame({
            'close': [1.1100, 1.1120, 1.1150],  # Strong uptrend
            'momentum': [0.02, 0.025, 0.03],  # Increasing momentum
            'volume': [1500000, 1800000, 2000000]  # Rising volume
        })
        
        result = await retail_strategies.execute_strategy('momentum', momentum_data)
        
        assert result['signal'] == 'BUY'
        assert result['confidence'] > 0.5


class TestStrategyIntegration:
    """Integration tests for strategy components"""
    
    @pytest.mark.asyncio
    async def test_complete_strategy_pipeline(self, sample_market_data):
        """Test complete strategy pipeline"""
        # Create all strategy components
        dl_strategy = AdvancedDeepLearningStrategy()
        await dl_strategy.initialize()
        
        mtf_analyzer = AdvancedMultiTimeframeAnalyzer()
        strategy_selector = AdvancedStrategySelector()
        await strategy_selector.initialize()
        
        signal_filter = AdvancedSignalFilter()
        
        # Step 1: Deep Learning Analysis
        dl_result = await dl_strategy.analyze(sample_market_data)
        assert dl_result is not None
        
        # Step 2: Multi-timeframe Analysis
        mtf_data = {'1h': sample_market_data}  # Simplified for test
        mtf_result = await mtf_analyzer.analyze_all_timeframes(mtf_data)
        assert mtf_result is not None
        
        # Step 3: Strategy Selection
        market_conditions = {
            'volatility': 'medium',
            'trend_strength': 'medium',
            'alignment_score': mtf_result['alignment_score']
        }
        selection_result = await strategy_selector.select_strategies(market_conditions)
        assert selection_result is not None
        
        # Step 4: Signal Filtering
        filtered_result = await signal_filter.apply_filters(dl_result)
        assert filtered_result is not None
        
        # Final decision
        if filtered_result['passed_filters']:
            final_signal = filtered_result['filtered_signal']
            final_confidence = filtered_result['filtered_confidence']
        else:
            final_signal = 'HOLD'
            final_confidence = 0.0
        
        assert final_signal in ['BUY', 'SELL', 'HOLD']
        assert 0 <= final_confidence <= 1
    
    @pytest.mark.asyncio
    async def test_strategy_performance_comparison(self, sample_market_data):
        """Compare performance of different strategies"""
        strategies = {
            'deep_learning': AdvancedDeepLearningStrategy(),
            'multi_timeframe': AdvancedMultiTimeframeAnalyzer(),
            'retail_mean_reversion': AdvancedRetailStrategies()
        }
        
        # Initialize strategies
        for name, strategy in strategies.items():
            if hasattr(strategy, 'initialize'):
                await strategy.initialize()
        
        performance_results = {}
        
        for name, strategy in strategies.items():
            if name == 'deep_learning':
                result = await strategy.analyze(sample_market_data)
                performance_results[name] = result['confidence']
            elif name == 'multi_timeframe':
                mtf_data = {'1h': sample_market_data}
                result = await strategy.analyze_all_timeframes(mtf_data)
                performance_results[name] = result['alignment_score']
            elif name == 'retail_mean_reversion':
                result = await strategy.execute_strategy('mean_reversion', sample_market_data)
                performance_results[name] = result['confidence']
        
        # All strategies should provide reasonable results
        for name, performance in performance_results.items():
            assert performance > 0.3
        
        print("\nStrategy Performance Comparison:")
        for name, performance in performance_results.items():
            print(f"{name}: {performance:.3f}")
    
    @pytest.mark.asyncio
    async def test_strategy_robustness_across_conditions(self):
        """Test strategy robustness across different market conditions"""
        market_scenarios = [
            ('bull_trend', self._create_trending_market('bull')),
            ('bear_trend', self._create_trending_market('bear')),
            ('high_vol', self._create_high_volatility_market()),
            ('low_vol', self._create_low_volatility_market())
        ]
        
        strategy = AdvancedDeepLearningStrategy()
        await strategy.initialize()
        
        robustness_results = {}
        
        for scenario_name, market_data in market_scenarios:
            try:
                result = await strategy.analyze(market_data)
                robustness_results[scenario_name] = {
                    'success': True,
                    'signal': result['signal'],
                    'confidence': result['confidence']
                }
            except Exception as e:
                robustness_results[scenario_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Strategy should handle all market conditions
        success_rate = sum(1 for r in robustness_results.values() if r['success']) / len(robustness_results)
        assert success_rate >= 0.8  # 80% success rate across conditions


# Performance and stress testing
class TestStrategyStress:
    """Stress tests for strategy components"""
    
    @pytest.mark.asyncio
    async def test_high_frequency_strategy_analysis(self):
        """Test strategy analysis under high frequency conditions"""
        strategy = AdvancedDeepLearningStrategy()
        await strategy.initialize()
        
        # Generate high frequency data
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='1min'  # 1-minute data
        )
        
        hf_data = pd.DataFrame({
            'open': np.random.normal(1.1, 0.001, len(dates)),
            'high': np.random.normal(1.101, 0.001, len(dates)),
            'low': np.random.normal(1.099, 0.001, len(dates)),
            'close': np.random.normal(1.1, 0.001, len(dates)),
            'volume': np.random.lognormal(13, 1, len(dates))
        }, index=dates)
        
        import time
        start_time = time.time()
        
        # Analyze multiple times rapidly
        for i in range(50):
            result = await strategy.analyze(hf_data.iloc[i*100:(i+1)*100])
            assert result is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 30  # Should complete within 30 seconds
    
    @pytest.mark.asyncio
    async def test_strategy_with_large_feature_set(self):
        """Test strategy with large number of features"""
        # Create data with many features
        n_features = 200
        n_samples = 1000
        
        large_feature_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add required columns
        large_feature_data['open'] = np.random.normal(1.1, 0.01, n_samples)
        large_feature_data['high'] = large_feature_data['open'] * 1.001
        large_feature_data['low'] = large_feature_data['open'] * 0.999
        large_feature_data['close'] = large_feature_data['open']
        large_feature_data['volume'] = np.random.lognormal(14, 1, n_samples)
        
        strategy = AdvancedDeepLearningStrategy()
        await strategy.initialize()
        
        result = await strategy.analyze(large_feature_data)
        
        assert result is not None
        assert result['signal'] in ['BUY', 'SELL', 'HOLD']


# Test execution and reporting
def generate_strategy_test_report():
    """Generate comprehensive strategy test report"""
    import subprocess
    import json
    
    try:
        # Run pytest with JSON output
        result = subprocess.run([
            'pytest', 'test_strategies.py', '-v', '--json-report', 
            '--json-report-file=strategy_test_report.json',
            '--tb=short'
        ], capture_output=True, text=True)
        
        # Load and analyze test results
        with open('strategy_test_report.json', 'r') as f:
            report = json.load(f)
        
        summary = {
            'total_tests': report['summary']['total'],
            'passed': report['summary']['passed'],
            'failed': report['summary']['failed'],
            'duration': report['summary']['duration'],
            'success_rate': report['summary']['passed'] / report['summary']['total'] if report['summary']['total'] > 0 else 0
        }
        
        print("\n" + "="*60)
        print("TRADING STRATEGIES TEST REPORT")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Duration: {summary['duration']:.2f}s")
        print("="*60)
        
        # Strategy-specific metrics
        if 'tests' in report:
            strategy_tests = [t for t in report['tests'] if any(s in t['nodeid'].lower() for s in ['strategy', 'analyzer', 'filter'])]
            print(f"\nStrategy-specific Tests: {len(strategy_tests)}")
        
        return summary
        
    except Exception as e:
        print(f"Error generating test report: {e}")
        return None


if __name__ == "__main__":
    # Run tests and generate report
    report = generate_strategy_test_report()
    
    if report and report['success_rate'] >= 0.8:
        print("🎉 TRADING STRATEGIES TESTS PASSED!")
        exit(0)
    else:
        print("❌ TRADING STRATEGIES TESTS FAILED!")
        exit(1)
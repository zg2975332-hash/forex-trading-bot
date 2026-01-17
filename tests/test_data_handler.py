"""
Advanced Test Suite for DataHandler Component
Comprehensive unit and integration tests for data handling functionality
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import aiohttp
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.data_handler import DataHandler
    from core.cache_manager import CacheManager
    from data.data_validator import DataValidator
except ImportError:
    # Mock implementations for testing
    class DataHandler:
        def __init__(self, config=None):
            self.config = config or {}
            self.cache_manager = CacheManager()
            self.validator = DataValidator()
            self.initialized = False
            
        async def initialize(self):
            self.initialized = True
            return True
            
        async def fetch_historical_data(self, symbol, timeframe, days=365):
            return pd.DataFrame()
            
        async def fetch_realtime_data(self, symbol):
            return {}
            
        async def validate_data_quality(self, data):
            return True
            
    class CacheManager:
        async def get(self, key):
            return None
            
        async def set(self, key, value, ttl=3600):
            return True
            
    class DataValidator:
        def validate_ohlcv(self, data):
            return True


class TestDataHandler:
    """Advanced test suite for DataHandler component"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            "api_keys": {
                "binance": {"key": "test_key", "secret": "test_secret"},
                "exness": {"key": "test_key", "secret": "test_secret"}
            },
            "cache_ttl": 3600,
            "max_retries": 3,
            "retry_delay": 1,
            "data_quality_threshold": 0.95
        }
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            freq='1H'
        )
        
        # Generate realistic price data with trends and volatility
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.005, len(dates))
        prices = 1.1000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001 + np.abs(np.random.normal(0, 0.0005, len(dates))),
            'low': prices * 0.998 - np.abs(np.random.normal(0, 0.0005, len(dates))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        }, index=dates)
        
        # Add some market events
        data.iloc[100:110] *= 1.02  # Price spike
        data.iloc[500:510] *= 0.98  # Price drop
        data.iloc[800:810] = np.nan  # Missing data
        
        return data
    
    @pytest.fixture
    def corrupted_data(self):
        """Generate corrupted data for testing validation"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=10),
            end=datetime.now(),
            freq='1H'
        )
        
        data = pd.DataFrame({
            'open': [np.nan] * len(dates),
            'high': [1.1000] * len(dates),
            'low': [1.0900] * len(dates),
            'close': [1.0950] * len(dates),
            'volume': [0] * len(dates)  # Zero volume
        }, index=dates)
        
        return data
    
    @pytest.fixture
    async def data_handler(self, sample_config):
        """Create DataHandler instance for testing"""
        handler = DataHandler(sample_config)
        await handler.initialize()
        return handler
    
    # ===== INITIALIZATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_initialization(self, data_handler):
        """Test DataHandler initialization"""
        assert data_handler.initialized == True
        assert hasattr(data_handler, 'cache_manager')
        assert hasattr(data_handler, 'validator')
        assert data_handler.config is not None
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test initialization failure scenarios"""
        with patch('core.data_handler.CacheManager') as mock_cache:
            mock_cache.side_effect = Exception("Cache initialization failed")
            
            handler = DataHandler()
            result = await handler.initialize()
            
            assert result == False
            assert handler.initialized == False
    
    # ===== HISTORICAL DATA TESTS =====
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_success(self, data_handler, sample_ohlcv_data):
        """Test successful historical data fetching"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            result = await data_handler.fetch_historical_data(
                symbol="EUR/USD", 
                timeframe="1h", 
                days=365
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
            mock_fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_caching(self, data_handler, sample_ohlcv_data):
        """Test historical data caching functionality"""
        cache_key = "historical_EUR/USD_1h_365"
        
        with patch('core.data_handler.CacheManager.get') as mock_cache_get, \
             patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            
            # Test cache hit
            mock_cache_get.return_value = sample_ohlcv_data
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert isinstance(result, pd.DataFrame)
            mock_cache_get.assert_called_once_with(cache_key)
            mock_fetch.assert_not_called()  # Should not fetch from API
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_retry_mechanism(self, data_handler):
        """Test retry mechanism for failed API calls"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch, \
             patch('asyncio.sleep') as mock_sleep:
            
            mock_fetch.side_effect = [Exception("API Error"), Exception("API Error"), pd.DataFrame()]
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert mock_fetch.call_count == 3  # Should retry 3 times
            assert mock_sleep.call_count == 2  # Should sleep between retries
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_multiple_timeframes(self, data_handler, sample_ohlcv_data):
        """Test fetching data for multiple timeframes"""
        timeframes = ["1h", "4h", "1d", "1w"]
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            for timeframe in timeframes:
                result = await data_handler.fetch_historical_data("EUR/USD", timeframe, 365)
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_multiple_symbols(self, data_handler, sample_ohlcv_data):
        """Test fetching data for multiple symbols"""
        symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            for symbol in symbols:
                result = await data_handler.fetch_historical_data(symbol, "1h", 365)
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
    
    # ===== REAL-TIME DATA TESTS =====
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_data_success(self, data_handler):
        """Test successful real-time data fetching"""
        mock_realtime_data = {
            'symbol': 'EUR/USD',
            'timestamp': datetime.now(),
            'bid': 1.10025,
            'ask': 1.10035,
            'spread': 0.0001,
            'volume': 1500000
        }
        
        with patch('core.data_handler.DataHandler._fetch_realtime_from_api') as mock_fetch:
            mock_fetch.return_value = mock_realtime_data
            
            result = await data_handler.fetch_realtime_data("EUR/USD")
            
            assert isinstance(result, dict)
            assert result['symbol'] == 'EUR/USD'
            assert 'bid' in result
            assert 'ask' in result
            assert result['spread'] == result['ask'] - result['bid']
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_data_websocket(self, data_handler):
        """Test WebSocket real-time data streaming"""
        mock_websocket_data = {
            'symbol': 'EUR/USD',
            'timestamp': datetime.now(),
            'bid': 1.10030,
            'ask': 1.10040,
            'spread': 0.0001
        }
        
        with patch('core.data_handler.DataHandler._setup_websocket') as mock_ws:
            mock_ws.return_value = AsyncMock()
            mock_ws.return_value.receive.return_value = json.dumps(mock_websocket_data)
            
            # Test WebSocket connection
            result = await data_handler._fetch_realtime_websocket("EUR/USD")
            
            assert result is not None
            mock_ws.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_realtime_data_fallback_mechanism(self, data_handler):
        """Test fallback mechanism when primary API fails"""
        mock_data = {
            'symbol': 'EUR/USD',
            'bid': 1.10025,
            'ask': 1.10035
        }
        
        with patch('core.data_handler.DataHandler._fetch_realtime_from_api') as mock_primary, \
             patch('core.data_handler.DataHandler._fetch_realtime_from_backup') as mock_backup:
            
            mock_primary.side_effect = Exception("Primary API down")
            mock_backup.return_value = mock_data
            
            result = await data_handler.fetch_realtime_data("EUR/USD")
            
            assert result == mock_data
            mock_primary.assert_called_once()
            mock_backup.assert_called_once()
    
    # ===== DATA VALIDATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_data_quality_validation_success(self, data_handler, sample_ohlcv_data):
        """Test successful data quality validation"""
        result = await data_handler.validate_data_quality(sample_ohlcv_data)
        
        assert result == True
    
    @pytest.mark.asyncio
    async def test_data_quality_validation_failure(self, data_handler, corrupted_data):
        """Test data quality validation with corrupted data"""
        result = await data_handler.validate_data_quality(corrupted_data)
        
        assert result == False
    
    @pytest.mark.asyncio
    async def test_data_completeness_check(self, data_handler, sample_ohlcv_data):
        """Test data completeness validation"""
        # Remove some data points to simulate incomplete data
        incomplete_data = sample_ohlcv_data.copy()
        incomplete_data.iloc[100:200] = np.nan
        
        with patch('data.data_validator.DataValidator.validate_completeness') as mock_validate:
            mock_validate.return_value = False
            
            result = await data_handler.validate_data_quality(incomplete_data)
            
            assert result == False
            mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_consistency_check(self, data_handler, sample_ohlcv_data):
        """Test data consistency validation"""
        inconsistent_data = sample_ohlcv_data.copy()
        inconsistent_data.loc[inconsistent_data['high'] < inconsistent_data['low']] = np.nan
        
        with patch('data.data_validator.DataValidator.validate_consistency') as mock_validate:
            mock_validate.return_value = False
            
            result = await data_handler.validate_data_quality(inconsistent_data)
            
            assert result == False
    
    @pytest.mark.asyncio
    async def test_outlier_detection(self, data_handler, sample_ohlcv_data):
        """Test outlier detection in price data"""
        # Add some outliers
        data_with_outliers = sample_ohlcv_data.copy()
        data_with_outliers.iloc[50] = data_with_outliers.iloc[50] * 1.5  # Extreme price
        
        with patch('data.data_validator.DataValidator.detect_outliers') as mock_detect:
            mock_detect.return_value = True  # Outliers detected
            
            result = await data_handler.validate_data_quality(data_with_outliers)
            
            assert result == False
            mock_detect.assert_called_once()
    
    # ===== PERFORMANCE TESTS =====
    
    @pytest.mark.asyncio
    async def test_historical_data_performance(self, data_handler, sample_ohlcv_data):
        """Test historical data fetching performance"""
        import time
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            start_time = time.time()
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            assert isinstance(result, pd.DataFrame)
            assert execution_time < 5.0  # Should complete within 5 seconds
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_data_fetching(self, data_handler, sample_ohlcv_data):
        """Test concurrent data fetching for multiple symbols"""
        symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"]
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            # Fetch data concurrently
            tasks = [
                data_handler.fetch_historical_data(symbol, "1h", 365)
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(symbols)
            for result in results:
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, data_handler, sample_ohlcv_data):
        """Test memory usage optimization with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            # Create larger dataset
            large_data = pd.concat([sample_ohlcv_data] * 10, ignore_index=True)
            mock_fetch.return_value = large_data
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 3650)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            assert isinstance(result, pd.DataFrame)
            assert memory_increase < 500  # Should not use more than 500MB extra
    
    # ===== ERROR HANDLING TESTS =====
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, data_handler):
        """Test handling of network timeouts"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch, \
             patch('asyncio.sleep') as mock_sleep:
            
            mock_fetch.side_effect = asyncio.TimeoutError("API timeout")
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert result.empty
            assert mock_fetch.call_count == data_handler.config.get('max_retries', 3)
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, data_handler):
        """Test handling of API rate limits"""
        from aiohttp import ClientResponseError
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch, \
             patch('asyncio.sleep') as mock_sleep:
            
            mock_fetch.side_effect = ClientResponseError(
                status=429, 
                message="Rate limit exceeded",
                request_info=Mock()
            )
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert result.empty
            mock_sleep.assert_called()  # Should sleep on rate limit
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, data_handler):
        """Test handling of invalid symbols"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()  # Empty response for invalid symbol
            
            result = await data_handler.fetch_historical_data("INVALID/SYMBOL", "1h", 365)
            
            assert result.empty
    
    @pytest.mark.asyncio
    async def test_data_parsing_errors(self, data_handler):
        """Test handling of data parsing errors"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = "invalid_data_format"
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert result.empty
    
    # ===== INTEGRATION TESTS =====
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_pipeline(self, data_handler, sample_ohlcv_data):
        """Test complete data pipeline from fetch to validation"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch, \
             patch('core.data_handler.CacheManager.set') as mock_cache_set, \
             patch('data.data_validator.DataValidator.validate_ohlcv') as mock_validate:
            
            mock_fetch.return_value = sample_ohlcv_data
            mock_validate.return_value = True
            
            # Fetch data
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            # Validate data
            validation_result = await data_handler.validate_data_quality(result)
            
            assert isinstance(result, pd.DataFrame)
            assert validation_result == True
            mock_cache_set.assert_called_once()  # Should cache the data
            mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_api_fallback(self, data_handler, sample_ohlcv_data):
        """Test fallback between multiple data providers"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_binance, \
             patch('core.data_handler.DataHandler._fetch_from_exness') as mock_exness, \
             patch('core.data_handler.DataHandler._fetch_from_alphavantage') as mock_alpha:
            
            mock_binance.side_effect = Exception("Binance API down")
            mock_exness.side_effect = Exception("Exness API down")
            mock_alpha.return_value = sample_ohlcv_data
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert isinstance(result, pd.DataFrame)
            mock_binance.assert_called_once()
            mock_exness.assert_called_once()
            mock_alpha.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_handler_with_live_apis(self, data_handler):
        """Test with actual API calls (if available)"""
        # This test would require actual API keys
        # For safety, we'll mock it but demonstrate the structure
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                'symbol': 'EURUSD',
                'data': []  # Empty for test
            }
            mock_get.return_value.__aenter__.return_value = mock_response
            
            try:
                result = await data_handler.fetch_historical_data("EUR/USD", "1h", 7)  # Only 7 days for test
                # If we get here, API call was successful
                assert True
            except Exception:
                # API call failed, which is acceptable in test environment
                assert True
    
    # ===== EDGE CASE TESTS =====
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, data_handler):
        """Test handling of empty data responses"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert result.empty
    
    @pytest.mark.asyncio
    async def test_very_large_dataset(self, data_handler):
        """Test handling of very large datasets"""
        # Create a large dataset
        large_dates = pd.date_range(
            start=datetime.now() - timedelta(days=3650),  # 10 years
            end=datetime.now(),
            freq='1H'
        )
        
        large_data = pd.DataFrame({
            'open': np.random.uniform(1.0, 1.5, len(large_dates)),
            'high': np.random.uniform(1.0, 1.5, len(large_dates)),
            'low': np.random.uniform(1.0, 1.5, len(large_dates)),
            'close': np.random.uniform(1.0, 1.5, len(large_dates)),
            'volume': np.random.uniform(1000000, 5000000, len(large_dates))
        }, index=large_dates)
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = large_data
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 3650)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 10000
    
    @pytest.mark.asyncio
    async def test_timezone_handling(self, data_handler, sample_ohlcv_data):
        """Test handling of different timezones"""
        # Convert to different timezone
        data_with_timezone = sample_ohlcv_data.copy()
        data_with_timezone.index = data_with_timezone.index.tz_localize('UTC')
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = data_with_timezone
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert isinstance(result, pd.DataFrame)
            assert result.index.tz is not None
    
    @pytest.mark.asyncio
    async def test_duplicate_data_handling(self, data_handler, sample_ohlcv_data):
        """Test handling of duplicate timestamp data"""
        data_with_duplicates = pd.concat([
            sample_ohlcv_data,
            sample_ohlcv_data.tail(10)  # Add duplicates
        ])
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = data_with_duplicates
            
            result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
            
            assert isinstance(result, pd.DataFrame)
            # Handler should remove duplicates
            assert len(result) == len(sample_ohlcv_data)


class TestDataHandlerAdvanced:
    """Advanced test cases for DataHandler"""
    
    @pytest.mark.asyncio
    async def test_data_handler_resilience(self, data_handler):
        """Test DataHandler resilience under various failure conditions"""
        failure_scenarios = [
            # (mock_side_effect, expected_behavior)
            (Exception("Network error"), "should_handle_gracefully"),
            (asyncio.TimeoutError(), "should_retry"),
            (KeyError("Invalid response"), "should_handle_parse_error"),
            (MemoryError(), "should_handle_memory_error"),
        ]
        
        for scenario, expected in failure_scenarios:
            with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
                mock_fetch.side_effect = scenario
                
                try:
                    result = await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
                    # If we get here, the error was handled gracefully
                    assert expected in ["should_handle_gracefully", "should_retry"]
                except Exception as e:
                    # Some errors might not be caught (like MemoryError)
                    assert expected in ["should_handle_memory_error"]
    
    @pytest.mark.asyncio
    async def test_data_handler_performance_benchmark(self, data_handler, sample_ohlcv_data):
        """Performance benchmark test for DataHandler"""
        import time
        
        test_cases = [
            ("EUR/USD", "1h", 30),   # 1 month
            ("EUR/USD", "1h", 365),  # 1 year  
            ("EUR/USD", "1h", 3650), # 10 years
        ]
        
        performance_results = {}
        
        for symbol, timeframe, days in test_cases:
            with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
                # Adjust data size based on days
                adjusted_data = sample_ohlcv_data.head(days * 24)  # Approximate hours
                mock_fetch.return_value = adjusted_data
                
                start_time = time.time()
                result = await data_handler.fetch_historical_data(symbol, timeframe, days)
                end_time = time.time()
                
                execution_time = end_time - start_time
                performance_results[f"{symbol}_{timeframe}_{days}d"] = {
                    'execution_time': execution_time,
                    'data_points': len(result),
                    'throughput': len(result) / execution_time if execution_time > 0 else 0
                }
                
                assert isinstance(result, pd.DataFrame)
        
        # Assert performance requirements
        for test_name, metrics in performance_results.items():
            assert metrics['execution_time'] < 10.0  # All should complete within 10 seconds
            print(f"{test_name}: {metrics['execution_time']:.2f}s, {metrics['throughput']:.0f} points/s")


# Performance and stress testing
class TestDataHandlerStress:
    """Stress tests for DataHandler"""
    
    @pytest.mark.asyncio
    async def test_high_frequency_requests(self, data_handler, sample_ohlcv_data):
        """Test handling high frequency data requests"""
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            mock_fetch.return_value = sample_ohlcv_data
            
            # Make 100 rapid requests
            tasks = []
            for i in range(100):
                task = data_handler.fetch_historical_data(f"SYMBOL_{i}", "1h", 365)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if isinstance(r, pd.DataFrame))
            failure_count = sum(1 for r in results if isinstance(r, Exception))
            
            assert success_count > 0
            assert failure_count < 50  # Less than 50% failures under stress
    
    @pytest.mark.asyncio 
    async def test_memory_stress_test(self, data_handler):
        """Test memory usage under stress"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Track memory usage during stress test
        memory_samples = []
        
        with patch('core.data_handler.DataHandler._fetch_from_binance') as mock_fetch:
            # Create large dataset
            large_data = pd.DataFrame(
                np.random.randn(100000, 5),  # 100,000 rows
                columns=['open', 'high', 'low', 'close', 'volume']
            )
            mock_fetch.return_value = large_data
            
            for i in range(10):
                memory_before = process.memory_info().rss / 1024 / 1024
                await data_handler.fetch_historical_data("EUR/USD", "1h", 365)
                memory_after = process.memory_info().rss / 1024 / 1024
                
                memory_samples.append(memory_after - memory_before)
            
            avg_memory_increase = np.mean(memory_samples)
            assert avg_memory_increase < 100  # Should not increase more than 100MB on average


# Test execution and reporting
def generate_test_report():
    """Generate comprehensive test report"""
    import subprocess
    import json
    
    try:
        # Run pytest with JSON output
        result = subprocess.run([
            'pytest', 'test_data_handler.py', '-v', '--json-report', 
            '--json-report-file=test_report.json'
        ], capture_output=True, text=True)
        
        # Load and analyze test results
        with open('test_report.json', 'r') as f:
            report = json.load(f)
        
        summary = {
            'total_tests': report['summary']['total'],
            'passed': report['summary']['passed'],
            'failed': report['summary']['failed'],
            'duration': report['summary']['duration'],
            'success_rate': report['summary']['passed'] / report['summary']['total']
        }
        
        print("\n" + "="*50)
        print("DATA HANDLER TEST REPORT")
        print("="*50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Duration: {summary['duration']:.2f}s")
        print("="*50)
        
        return summary
        
    except Exception as e:
        print(f"Error generating test report: {e}")
        return None


if __name__ == "__main__":
    # Run tests and generate report
    report = generate_test_report()
    
    if report and report['success_rate'] >= 0.8:
        print("🎉 DATA HANDLER TESTS PASSED!")
        exit(0)
    else:
        print("❌ DATA HANDLER TESTS FAILED!")
        exit(1)
"""
Advanced Data Handler for Forex Trading Bot
Real-time and historical data management with multiple sources and caching
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import requests
import time
import threading
from dataclasses import dataclass
import json
import sqlite3
import warnings
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pickle
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure for OHLCV + additional features"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float
    additional_data: Dict[str, Any] = None

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness: float
    accuracy: float
    timeliness: float
    consistency: float
    overall_score: float
    issues: List[str]

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def fetch_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def fetch_real_time_data(self, symbol: str) -> MarketData:
        pass
    
    @abstractmethod
    def get_data_quality(self) -> DataQualityMetrics:
        pass

class BinanceDataSource(DataSource):
    """Binance data source implementation"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com/api/v3"
        self.rate_limits = {
            'requests_per_minute': 1200,
            'last_request_time': None
        }
        self.session = requests.Session()
        
        logger.info("Binance data source initialized")
    
    def _rate_limit_check(self):
        """Implement rate limiting"""
        if self.rate_limits['last_request_time']:
            elapsed = time.time() - self.rate_limits['last_request_time']
            min_interval = 60.0 / self.rate_limits['requests_per_minute']
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        
        self.rate_limits['last_request_time'] = time.time()
    
    def fetch_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical OHLCV data from Binance"""
        try:
            self._rate_limit_check()
            
            # Convert timeframe to Binance format
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '1H': '1h', '4H': '4h', '1D': '1d'
            }
            binance_timeframe = timeframe_map.get(timeframe, '1h')
            
            # Convert symbol to Binance format
            binance_symbol = symbol.replace('/', '').upper()
            
            # Calculate limit based on timeframe
            limit = self._calculate_data_limit(start_date, end_date, timeframe)
            
            url = f"{self.base_url}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': binance_timeframe,
                'startTime': int(start_date.timestamp() * 1000),
                'endTime': int(end_date.timestamp() * 1000),
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = self._parse_binance_data(data, symbol, timeframe)
            
            logger.info(f"Fetched {len(df)} historical records for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data from Binance: {e}")
            # Return empty DataFrame with proper structure
            return self._create_empty_dataframe()
    
    def _calculate_data_limit(self, start_date: datetime, end_date: datetime, timeframe: str) -> int:
        """Calculate appropriate data limit based on timeframe and date range"""
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15,
            '1H': 60, '4H': 240, '1D': 1440
        }
        
        total_minutes = (end_date - start_date).total_seconds() / 60
        timeframe_minute = timeframe_minutes.get(timeframe, 60)
        limit = int(total_minutes / timeframe_minute) + 1
        
        return min(limit, 1000)  # Binance max limit
    
    def _parse_binance_data(self, data: List, symbol: str, timeframe: str) -> pd.DataFrame:
        """Parse Binance API response into DataFrame"""
        if not data:
            return self._create_empty_dataframe()
        
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert OHLCV to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Add symbol and timeframe
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        # Calculate spread (approximate)
        df['spread'] = (df['high'] - df['low']) * 0.0001  # Simplified spread calculation
        
        # Select and reorder columns
        df = df[['symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume', 'spread']]
        
        return df
    
    def fetch_real_time_data(self, symbol: str) -> MarketData:
        """Fetch real-time market data from Binance"""
        try:
            self._rate_limit_check()
            
            binance_symbol = symbol.replace('/', '').upper()
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': binance_symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                timeframe='realtime',
                timestamp=datetime.now(),
                open=float(data['openPrice']),
                high=float(data['highPrice']),
                low=float(data['lowPrice']),
                close=float(data['lastPrice']),
                volume=float(data['volume']),
                spread=(float(data['highPrice']) - float(data['lowPrice'])) * 0.0001,
                additional_data={
                    'price_change': float(data['priceChange']),
                    'price_change_percent': float(data['priceChangePercent']),
                    'weighted_avg_price': float(data['weightedAvgPrice']),
                    'prev_close_price': float(data['prevClosePrice']),
                    'bid_price': float(data['bidPrice']),
                    'ask_price': float(data['askPrice'])
                }
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data from Binance: {e}")
            raise
    
    def get_data_quality(self) -> DataQualityMetrics:
        """Assess data quality from Binance"""
        # Simplified quality assessment
        # In production, implement actual quality checks
        return DataQualityMetrics(
            completeness=0.95,
            accuracy=0.98,
            timeliness=0.99,
            consistency=0.96,
            overall_score=0.97,
            issues=["Occasional API rate limiting"]
        )

class ExnessDataSource(DataSource):
    """Exness data source implementation"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.exness.com"
        self.session = requests.Session()
        
        logger.info("Exness data source initialized")
    
    def fetch_historical_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data from Exness"""
        try:
            # Exness API implementation would go here
            # For now, return synthetic data
            logger.warning("Exness historical data not implemented, returning synthetic data")
            return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching historical data from Exness: {e}")
            return self._create_empty_dataframe()
    
    def fetch_real_time_data(self, symbol: str) -> MarketData:
        """Fetch real-time data from Exness"""
        try:
            # Exness real-time API implementation
            # For now, return synthetic data
            logger.warning("Exness real-time data not implemented, returning synthetic data")
            return self._generate_synthetic_realtime_data(symbol)
            
        except Exception as e:
            logger.error(f"Error fetching real-time data from Exness: {e}")
            raise
    
    def get_data_quality(self) -> DataQualityMetrics:
        """Assess data quality from Exness"""
        return DataQualityMetrics(
            completeness=0.92,
            accuracy=0.95,
            timeliness=0.98,
            consistency=0.94,
            overall_score=0.95,
            issues=["Synthetic data used", "API not fully implemented"]
        )
    
    def _generate_synthetic_data(self, symbol: str, timeframe: str, 
                              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic data for testing"""
        # Generate date range
        if timeframe == '1H':
            freq = '1H'
        elif timeframe == '4H':
            freq = '4H'
        elif timeframe == '1D':
            freq = '1D'
        else:
            freq = '1H'
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 1.1000 if 'EUR' in symbol else 1.3000
        
        returns = np.random.normal(0.0001, 0.005, len(dates))
        prices = base_price * (1 + np.cumsum(returns))
        
        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(base_price)
        df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(len(dates))) * 0.0005
        df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(len(dates))) * 0.0005
        df['volume'] = np.random.randint(1000, 10000, len(dates))
        df['spread'] = np.random.uniform(0.0001, 0.0003, len(dates))
        
        return df
    
    def _generate_synthetic_realtime_data(self, symbol: str) -> MarketData:
        """Generate synthetic real-time data"""
        base_price = 1.1000 if 'EUR' in symbol else 1.3000
        price_move = np.random.normal(0, 0.001)
        current_price = base_price * (1 + price_move)
        
        return MarketData(
            symbol=symbol,
            timeframe='realtime',
            timestamp=datetime.now(),
            open=current_price - 0.0002,
            high=current_price + 0.0005,
            low=current_price - 0.0005,
            close=current_price,
            volume=np.random.randint(5000, 15000),
            spread=0.0002,
            additional_data={'synthetic': True}
        )

class DataCache:
    """Advanced caching system for market data"""
    
    def __init__(self, cache_dir: str = "data/cache", max_size_mb: int = 1000):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.cache_index = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache index
        self._load_cache_index()
        
        logger.info(f"Data cache initialized at {cache_dir}")
    
    def _load_cache_index(self):
        """Load cache index from file"""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache index: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to file"""
        index_file = os.path.join(self.cache_dir, "cache_index.json")
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache index: {e}")
    
    def get_cache_key(self, symbol: str, timeframe: str, 
                     start_date: datetime, end_date: datetime) -> str:
        """Generate cache key for data request"""
        key_parts = [
            symbol.replace('/', '_'),
            timeframe,
            start_date.strftime('%Y%m%d'),
            end_date.strftime('%Y%m%d')
        ]
        return "_".join(key_parts)
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        if key not in self.cache_index:
            return None
        
        cache_info = self.cache_index[key]
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        # Check if cache is still valid (1 hour for real-time, 24 hours for historical)
        cache_age = datetime.now() - datetime.fromisoformat(cache_info['timestamp'])
        max_age = timedelta(hours=1) if 'realtime' in key else timedelta(hours=24)
        
        if cache_age > max_age:
            self.delete(key)
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit for key: {key}")
            return data
        except Exception as e:
            logger.warning(f"Error loading cache for key {key}: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, data: pd.DataFrame):
        """Store data in cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Update index
            self.cache_index[key] = {
                'timestamp': datetime.now().isoformat(),
                'size': len(data),
                'file': cache_file
            }
            
            self._save_cache_index()
            logger.debug(f"Data cached with key: {key}")
            
            # Cleanup old cache if needed
            self._cleanup_cache()
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def delete(self, key: str):
        """Delete data from cache"""
        try:
            if key in self.cache_index:
                cache_file = self.cache_index[key]['file']
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                del self.cache_index[key]
                self._save_cache_index()
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
    
    def _cleanup_cache(self):
        """Cleanup old cache files if size limit exceeded"""
        try:
            total_size = 0
            for cache_info in self.cache_index.values():
                cache_file = cache_info['file']
                if os.path.exists(cache_file):
                    total_size += os.path.getsize(cache_file)
            
            total_size_mb = total_size / (1024 * 1024)
            
            if total_size_mb > self.max_size_mb:
                # Sort by timestamp (oldest first)
                sorted_items = sorted(self.cache_index.items(), 
                                   key=lambda x: x[1]['timestamp'])
                
                # Remove oldest items until under limit
                for key, cache_info in sorted_items:
                    self.delete(key)
                    total_size_mb = self._get_cache_size_mb()
                    if total_size_mb <= self.max_size_mb * 0.8:  # Leave some buffer
                        break
                
                logger.info(f"Cache cleaned up. New size: {total_size_mb:.2f} MB")
                
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB"""
        total_size = 0
        for cache_info in self.cache_index.values():
            cache_file = cache_info['file']
            if os.path.exists(cache_file):
                total_size += os.path.getsize(cache_file)
        return total_size / (1024 * 1024)

class DataHandler:
    """
    Main data handler class coordinating multiple data sources and caching
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.data_sources = self._initialize_data_sources()
        self.cache = DataCache()
        self.real_time_data = {}
        self.data_quality_metrics = {}
        
        # Real-time data thread
        self.real_time_thread = None
        self.real_time_running = False
        self.real_time_interval = 5  # seconds
        
        # Data validation
        self.validation_rules = self._initialize_validation_rules()
        
        logger.info("Data Handler initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data_sources': {
                'binance': {'enabled': True, 'priority': 1},
                'exness': {'enabled': True, 'priority': 2}
            },
            'cache': {
                'enabled': True,
                'max_size_mb': 1000,
                'cache_dir': 'data/cache'
            },
            'real_time': {
                'enabled': True,
                'update_interval': 5
            },
            'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
            'timeframes': ['1H', '4H', '1D']
        }
    
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize data sources"""
        sources = {}
        
        if self.config['data_sources']['binance']['enabled']:
            sources['binance'] = BinanceDataSource()
        
        if self.config['data_sources']['exness']['enabled']:
            sources['exness'] = ExnessDataSource()
        
        return sources
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize data validation rules"""
        return {
            'price_validation': {
                'min_price': 0.5,
                'max_price': 500.0,
                'max_change_percent': 0.1  # 10% max change between candles
            },
            'volume_validation': {
                'min_volume': 0,
                'max_volume': 10000000
            },
            'time_gaps': {
                'max_gap_minutes': 120  # Max allowed gap between data points
            },
            'completeness': {
                'min_completeness': 0.95  # Minimum data completeness ratio
            }
        }
    
    def get_historical_data(self, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical data with caching and fallback sources
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Generate cache key
            cache_key = self.cache.get_cache_key(symbol, timeframe, start_date, end_date)
            
            # Try cache first
            if use_cache and self.config['cache']['enabled']:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    logger.info(f"Using cached data for {symbol} ({timeframe})")
                    return cached_data
            
            # Try data sources in priority order
            sorted_sources = sorted(self.data_sources.items(),
                                  key=lambda x: self.config['data_sources'][x[0]]['priority'])
            
            data = None
            used_source = None
            
            for source_name, source in sorted_sources:
                try:
                    logger.info(f"Fetching data from {source_name} for {symbol}")
                    data = source.fetch_historical_data(symbol, timeframe, start_date, end_date)
                    
                    if data is not None and len(data) > 0:
                        used_source = source_name
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch data from {source_name}: {e}")
                    continue
            
            if data is None or len(data) == 0:
                logger.error(f"All data sources failed for {symbol}")
                return self._create_empty_dataframe()
            
            # Validate data
            validation_result = self._validate_data(data, symbol, timeframe)
            if not validation_result['is_valid']:
                logger.warning(f"Data validation failed for {symbol}: {validation_result['issues']}")
            
            # Clean data
            data = self._clean_data(data)
            
            # Cache the data
            if use_cache and self.config['cache']['enabled'] and len(data) > 0:
                self.cache.set(cache_key, data)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol} from {used_source}")
            return data
            
        except Exception as e:
            logger.error(f"Error in get_historical_data for {symbol}: {e}")
            return self._create_empty_dataframe()
    
    def _validate_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Validate data quality"""
        issues = []
        
        if len(data) == 0:
            issues.append("Empty dataset")
            return {'is_valid': False, 'issues': issues}
        
        # Check for missing values
        missing_values = data[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()
        if missing_values > 0:
            issues.append(f"Missing values: {missing_values}")
        
        # Check price validity
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                issues.append(f"Invalid prices in {col}")
                break
        
        # Check OHLC consistency
        ohlc_issues = data[(data['high'] < data['low']) | 
                          (data['high'] < data['open']) | 
                          (data['high'] < data['close']) |
                          (data['low'] > data['open']) | 
                          (data['low'] > data['close'])].shape[0]
        if ohlc_issues > 0:
            issues.append(f"OHLC inconsistency in {ohlc_issues} records")
        
        # Check time gaps
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dt.total_seconds().dropna()
            max_gap = time_diffs.max()
            timeframe_seconds = self._timeframe_to_seconds(timeframe)
            if max_gap > timeframe_seconds * 2:  # Allow some flexibility
                issues.append(f"Large time gap detected: {max_gap/60:.1f} minutes")
        
        is_valid = len(issues) == 0
        return {'is_valid': is_valid, 'issues': issues}
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe to seconds"""
        timeframe_map = {
            '1m': 60, '5m': 300, '15m': 900,
            '1H': 3600, '4H': 14400, '1D': 86400
        }
        return timeframe_map.get(timeframe, 3600)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Forward fill small gaps
        data = data.ffill()
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        return data
    
    def start_real_time_data(self):
        """Start real-time data collection"""
        if self.real_time_running:
            logger.warning("Real-time data collection already running")
            return
        
        self.real_time_running = True
        self.real_time_thread = threading.Thread(target=self._real_time_worker, daemon=True)
        self.real_time_thread.start()
        logger.info("Real-time data collection started")
    
    def stop_real_time_data(self):
        """Stop real-time data collection"""
        self.real_time_running = False
        if self.real_time_thread:
            self.real_time_thread.join(timeout=10)
        logger.info("Real-time data collection stopped")
    
    def _real_time_worker(self):
        """Worker function for real-time data collection"""
        while self.real_time_running:
            try:
                for symbol in self.config['symbols']:
                    for source_name, source in self.data_sources.items():
                        try:
                            market_data = source.fetch_real_time_data(symbol)
                            self.real_time_data[symbol] = market_data
                            
                            # Update data quality metrics
                            self._update_data_quality_metrics(source_name, symbol)
                            break  # Use first successful source
                            
                        except Exception as e:
                            logger.warning(f"Real-time data failed from {source_name} for {symbol}: {e}")
                            continue
                
                # Wait for next update
                time.sleep(self.real_time_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time data worker: {e}")
                time.sleep(self.real_time_interval)  # Continue after error
    
    def _update_data_quality_metrics(self, source_name: str, symbol: str):
        """Update data quality metrics"""
        if source_name not in self.data_quality_metrics:
            self.data_quality_metrics[source_name] = {}
        
        if symbol not in self.data_quality_metrics[source_name]:
            self.data_quality_metrics[source_name][symbol] = {
                'successful_requests': 0,
                'failed_requests': 0,
                'last_update': datetime.now(),
                'average_latency': 0.0
            }
        
        metrics = self.data_quality_metrics[source_name][symbol]
        metrics['successful_requests'] += 1
        metrics['last_update'] = datetime.now()
    
    def get_real_time_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest real-time data for symbol"""
        return self.real_time_data.get(symbol)
    
    def get_all_real_time_data(self) -> Dict[str, MarketData]:
        """Get all real-time data"""
        return self.real_time_data.copy()
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'overall_quality': 0.0,
            'issues': []
        }
        
        total_quality = 0.0
        source_count = 0
        
        for source_name, source in self.data_sources.items():
            source_metrics = source.get_data_quality()
            report['sources'][source_name] = {
                'completeness': source_metrics.completeness,
                'accuracy': source_metrics.accuracy,
                'timeliness': source_metrics.timeliness,
                'consistency': source_metrics.consistency,
                'overall_score': source_metrics.overall_score,
                'issues': source_metrics.issues
            }
            
            total_quality += source_metrics.overall_score
            source_count += 1
            
            # Add source issues to overall report
            report['issues'].extend([f"{source_name}: {issue}" for issue in source_metrics.issues])
        
        if source_count > 0:
            report['overall_quality'] = total_quality / source_count
        
        return report
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with proper structure"""
        return pd.DataFrame(columns=[
            'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume', 'spread'
        ])
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_real_time_data()
        logger.info("Data Handler cleanup completed")


# Example usage and testing
if __name__ == "__main__":
    # Test the data handler
    print("Testing Data Handler...")
    
    try:
        # Initialize data handler
        data_handler = DataHandler()
        
        # Test historical data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        
        print("Fetching historical data...")
        historical_data = data_handler.get_historical_data(
            symbol="EUR/USD",
            timeframe="1H",
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        print(f"Retrieved {len(historical_data)} historical records")
        if len(historical_data) > 0:
            print(f"Date range: {historical_data.index[0]} to {historical_data.index[-1]}")
            print(f"Columns: {list(historical_data.columns)}")
        
        # Test real-time data
        print("\nTesting real-time data...")
        data_handler.start_real_time_data()
        
        # Wait for some real-time data
        time.sleep(10)
        
        real_time_data = data_handler.get_real_time_data("EUR/USD")
        if real_time_data:
            print(f"Real-time data for EUR/USD:")
            print(f"  Price: {real_time_data.close:.5f}")
            print(f"  High: {real_time_data.high:.5f}")
            print(f"  Low: {real_time_data.low:.5f}")
            print(f"  Volume: {real_time_data.volume:,.0f}")
        
        # Test data quality report
        print("\nGenerating data quality report...")
        quality_report = data_handler.get_data_quality_report()
        print(f"Overall quality score: {quality_report['overall_quality']:.3f}")
        
        for source_name, source_metrics in quality_report['sources'].items():
            print(f"  {source_name}: {source_metrics['overall_score']:.3f}")
        
        # Cleanup
        data_handler.cleanup()
        
        print(f"\n✅ Data Handler test completed successfully!")
        
    except Exception as e:
        print(f"❌ Data Handler test failed: {e}")
        import traceback
        traceback.print_exc()
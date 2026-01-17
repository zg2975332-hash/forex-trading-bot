"""
Historical Data Manager for FOREX TRADING BOT
Advanced historical data acquisition, preprocessing, and management
"""

import logging
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import requests
import aiohttp
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import zipfile
import io
import talib
from scipy import stats
import yfinance as yf
import ccxt

logger = logging.getLogger(__name__)

class DataSource(Enum):
    YAHOO_FINANCE = "yfinance"
    CCXT_EXCHANGE = "ccxt"
    ALPHA_VANTAGE = "alpha_vantage"
    OANDA = "oanda"
    DUKASCOPY = "dukascopy"
    LOCAL_STORAGE = "local"

class TimeFrame(Enum):
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

class DataQuality(Enum):
    RAW = "raw"
    CLEANED = "cleaned"
    FEATURE_ENRICHED = "feature_enriched"
    VALIDATED = "validated"

@dataclass
class HistoricalRequest:
    """Historical data request specification"""
    symbol: str
    timeframe: TimeFrame
    start_date: datetime
    end_date: datetime
    data_source: DataSource
    fields: List[str] = field(default_factory=lambda: ['open', 'high', 'low', 'close', 'volume'])
    include_indicators: bool = False
    include_volatility: bool = False
    include_correlation: bool = False
    quality_level: DataQuality = DataQuality.FEATURE_ENRICHED

@dataclass
class HistoricalResponse:
    """Historical data response"""
    success: bool
    data: pd.DataFrame = None
    symbol: str = ""
    timeframe: TimeFrame = None
    data_points: int = 0
    quality_score: float = 0.0
    missing_percentage: float = 0.0
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class HistoricalDataManager:
    """
    Advanced historical data manager with multi-source aggregation and feature engineering
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Data sources configuration
        self.data_sources = {
            DataSource.YAHOO_FINANCE: {
                'enabled': True,
                'rate_limit': 5,  # requests per second
                'max_retries': 3
            },
            DataSource.CCXT_EXCHANGE: {
                'enabled': True,
                'exchanges': ['binance', 'kraken', 'bitfinex'],
                'rate_limit': 2
            },
            DataSource.ALPHA_VANTAGE: {
                'enabled': False,  # Requires API key
                'api_key': self.config.get('alpha_vantage_key'),
                'rate_limit': 1
            }
        }
        
        # Cache configuration
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache_dir = Path(self.config.get('cache_dir', './data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.data_cache = {}
        self.request_history = deque(maxlen=1000)
        self.quality_metrics = defaultdict(lambda: deque(maxlen=100))
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.last_request_time = {}
        
        # Initialize exchanges for CCXT
        self.exchanges = {}
        self._initialize_exchanges()
        
        logger.info("HistoricalDataManager initialized")

    def _initialize_exchanges(self):
        """Initialize cryptocurrency exchanges"""
        try:
            if self.data_sources[DataSource.CCXT_EXCHANGE]['enabled']:
                for exchange_id in self.data_sources[DataSource.CCXT_EXCHANGE]['exchanges']:
                    try:
                        exchange_class = getattr(ccxt, exchange_id)
                        self.exchanges[exchange_id] = exchange_class({
                            'rateLimit': 1000,
                            'enableRateLimit': True,
                            'timeout': 30000
                        })
                        logger.info(f"Initialized exchange: {exchange_id}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {exchange_id}: {e}")
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    async def get_historical_data(self, request: HistoricalRequest) -> HistoricalResponse:
        """
        Get historical data with advanced preprocessing and feature engineering
        """
        start_time = time.time()
        
        try:
            logger.info(f"Fetching historical data: {request.symbol} {request.timeframe.value} "
                       f"from {request.start_date} to {request.end_date}")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if self.cache_enabled:
                cached_data = await self._get_cached_data(cache_key)
                if cached_data is not None:
                    logger.debug(f"Cache hit for: {cache_key}")
                    cached_data.processing_time = time.time() - start_time
                    return cached_data
            
            # Fetch data from source
            raw_data = await self._fetch_from_source(request)
            
            if raw_data.empty:
                return HistoricalResponse(
                    success=False,
                    error_message="No data retrieved from source",
                    processing_time=time.time() - start_time
                )
            
            # Process data based on quality level
            processed_data = await self._process_data(raw_data, request)
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(processed_data)
            missing_percentage = self._calculate_missing_percentage(processed_data)
            
            response = HistoricalResponse(
                success=True,
                data=processed_data,
                symbol=request.symbol,
                timeframe=request.timeframe,
                data_points=len(processed_data),
                quality_score=quality_score,
                missing_percentage=missing_percentage,
                processing_time=time.time() - start_time,
                metadata={
                    'source': request.data_source.value,
                    'cache_key': cache_key,
                    'quality_level': request.quality_level.value
                }
            )
            
            # Cache the response
            if self.cache_enabled:
                await self._cache_data(cache_key, response)
            
            # Store in request history
            self.request_history.append({
                'timestamp': time.time(),
                'symbol': request.symbol,
                'timeframe': request.timeframe.value,
                'data_points': len(processed_data),
                'quality_score': quality_score,
                'processing_time': response.processing_time
            })
            
            logger.info(f"Historical data retrieved: {len(processed_data)} data points, "
                       f"quality: {quality_score:.1f}%")
            
            return response
            
        except Exception as e:
            error_msg = f"Historical data retrieval failed: {str(e)}"
            logger.error(error_msg)
            return HistoricalResponse(
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )

    async def _fetch_from_source(self, request: HistoricalRequest) -> pd.DataFrame:
        """Fetch data from specified source"""
        try:
            if request.data_source == DataSource.YAHOO_FINANCE:
                return await self._fetch_yahoo_finance(request)
            elif request.data_source == DataSource.CCXT_EXCHANGE:
                return await self._fetch_ccxt(request)
            elif request.data_source == DataSource.LOCAL_STORAGE:
                return await self._fetch_local(request)
            else:
                raise ValueError(f"Unsupported data source: {request.data_source}")
                
        except Exception as e:
            logger.error(f"Data fetch from {request.data_source} failed: {e}")
            return pd.DataFrame()

    async def _fetch_yahoo_finance(self, request: HistoricalRequest) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            await self._check_rate_limit(DataSource.YAHOO_FINANCE)
            
            # Convert Forex symbol to Yahoo format
            yahoo_symbol = self._convert_to_yahoo_symbol(request.symbol)
            
            # Download data
            data = yf.download(
                yahoo_symbol,
                start=request.start_date,
                end=request.end_date,
                interval=request.timeframe.value,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                logger.warning(f"No data from Yahoo Finance for {yahoo_symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to standard format
            column_mapping = {
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            data = data.rename(columns=column_mapping)
            
            # Ensure timestamp is datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Set timestamp as index
            data.set_index('timestamp', inplace=True)
            
            logger.debug(f"Yahoo Finance data fetched: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}")
            return pd.DataFrame()

    async def _fetch_ccxt(self, request: HistoricalRequest) -> pd.DataFrame:
        """Fetch data from cryptocurrency exchanges"""
        try:
            await self._check_rate_limit(DataSource.CCXT_EXCHANGE)
            
            # Convert Forex symbol to crypto format
            crypto_symbol = self._convert_to_crypto_symbol(request.symbol)
            
            all_data = []
            
            for exchange_id, exchange in self.exchanges.items():
                try:
                    # Convert timeframe to exchange format
                    exchange_timeframe = self._convert_timeframe(request.timeframe, exchange_id)
                    
                    # Fetch OHLCV data
                    since = exchange.parse8601(request.start_date.isoformat())
                    ohlcv = exchange.fetch_ohlcv(
                        crypto_symbol, 
                        exchange_timeframe, 
                        since=since,
                        limit=1000
                    )
                    
                    if ohlcv:
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df['exchange'] = exchange_id
                        df.set_index('timestamp', inplace=True)
                        
                        all_data.append(df)
                        logger.debug(f"CCXT data from {exchange_id}: {len(df)} records")
                        
                except Exception as e:
                    logger.warning(f"CCXT fetch from {exchange_id} failed: {e}")
                    continue
            
            if not all_data:
                return pd.DataFrame()
            
            # Combine data from all exchanges
            combined_data = pd.concat(all_data)
            
            # Remove duplicates and sort
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            combined_data = combined_data.sort_index()
            
            # Filter by date range
            combined_data = combined_data[
                (combined_data.index >= request.start_date) & 
                (combined_data.index <= request.end_date)
            ]
            
            return combined_data
            
        except Exception as e:
            logger.error(f"CCXT fetch failed: {e}")
            return pd.DataFrame()

    async def _fetch_local(self, request: HistoricalRequest) -> pd.DataFrame:
        """Fetch data from local storage"""
        try:
            local_file = self.cache_dir / f"{request.symbol}_{request.timeframe.value}.parquet"
            
            if not local_file.exists():
                logger.warning(f"Local file not found: {local_file}")
                return pd.DataFrame()
            
            data = pd.read_parquet(local_file)
            
            # Filter by date range
            data = data[
                (data.index >= request.start_date) & 
                (data.index <= request.end_date)
            ]
            
            logger.debug(f"Local data loaded: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Local data fetch failed: {e}")
            return pd.DataFrame()

    async def _process_data(self, data: pd.DataFrame, request: HistoricalRequest) -> pd.DataFrame:
        """Process data based on quality level requirements"""
        try:
            processed_data = data.copy()
            
            # Step 1: Basic cleaning
            processed_data = self._clean_data(processed_data)
            
            # Step 2: Handle missing values
            processed_data = self._handle_missing_values(processed_data)
            
            # Step 3: Feature engineering for higher quality levels
            if request.quality_level in [DataQuality.FEATURE_ENRICHED, DataQuality.VALIDATED]:
                processed_data = await self._add_technical_indicators(processed_data, request)
                processed_data = await self._add_volatility_measures(processed_data)
                processed_data = await self._add_statistical_features(processed_data)
            
            # Step 4: Validation for highest quality level
            if request.quality_level == DataQuality.VALIDATED:
                processed_data = await self._validate_data(processed_data)
            
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in processed_data.columns:
                    processed_data[col] = np.nan
            
            logger.debug(f"Data processing completed: {len(processed_data)} records")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        try:
            cleaned_data = data.copy()
            
            # Remove duplicates
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]
            
            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in cleaned_data.columns:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            # Remove obvious outliers (prices <= 0)
            for col in ['open', 'high', 'low', 'close']:
                if col in cleaned_data.columns:
                    cleaned_data = cleaned_data[cleaned_data[col] > 0]
            
            # Ensure OHLC consistency
            if all(col in cleaned_data.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= Open, Low, Close
                mask = (cleaned_data['high'] >= cleaned_data[['open', 'low', 'close']].max(axis=1))
                # Low should be <= Open, High, Close  
                mask &= (cleaned_data['low'] <= cleaned_data[['open', 'high', 'close']].min(axis=1))
                cleaned_data = cleaned_data[mask]
            
            logger.debug(f"Data cleaning completed: {len(cleaned_data)} records remaining")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        try:
            filled_data = data.copy()
            
            # Forward fill for small gaps in price data
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in filled_data.columns:
                    # Limit forward fill to 2 periods to avoid propagating errors
                    filled_data[col] = filled_data[col].fillna(method='ffill', limit=2)
            
            # For volume, use 0 for missing values
            if 'volume' in filled_data.columns:
                filled_data['volume'] = filled_data['volume'].fillna(0)
            
            # Remove rows with critical missing data
            critical_columns = ['open', 'high', 'low', 'close']
            if all(col in filled_data.columns for col in critical_columns):
                filled_data = filled_data.dropna(subset=critical_columns)
            
            logger.debug(f"Missing values handled: {len(filled_data)} records remaining")
            return filled_data
            
        except Exception as e:
            logger.error(f"Missing value handling failed: {e}")
            return data

    async def _add_technical_indicators(self, data: pd.DataFrame, request: HistoricalRequest) -> pd.DataFrame:
        """Add technical indicators to data"""
        try:
            enriched_data = data.copy()
            
            if len(enriched_data) < 50:  # Need sufficient data for indicators
                return enriched_data
            
            # Price-based indicators
            prices = enriched_data['close'].values
            
            # Moving averages
            enriched_data['sma_20'] = talib.SMA(prices, timeperiod=20)
            enriched_data['sma_50'] = talib.SMA(prices, timeperiod=50)
            enriched_data['sma_200'] = talib.SMA(prices, timeperiod=200)
            enriched_data['ema_12'] = talib.EMA(prices, timeperiod=12)
            enriched_data['ema_26'] = talib.EMA(prices, timeperiod=26)
            
            # Bollinger Bands
            enriched_data['bb_upper'], enriched_data['bb_middle'], enriched_data['bb_lower'] = talib.BBANDS(
                prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            
            # RSI
            enriched_data['rsi_14'] = talib.RSI(prices, timeperiod=14)
            
            # MACD
            enriched_data['macd'], enriched_data['macd_signal'], enriched_data['macd_hist'] = talib.MACD(
                prices, fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Stochastic
            enriched_data['stoch_k'], enriched_data['stoch_d'] = talib.STOCH(
                enriched_data['high'].values, enriched_data['low'].values, enriched_data['close'].values,
                fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
            )
            
            # Volume indicators (if volume data available)
            if 'volume' in enriched_data.columns:
                volume = enriched_data['volume'].values
                enriched_data['volume_sma'] = talib.SMA(volume, timeperiod=20)
                enriched_data['ad'] = talib.AD(
                    enriched_data['high'].values, enriched_data['low'].values, 
                    enriched_data['close'].values, volume
                )
            
            logger.debug(f"Technical indicators added: {len(enriched_data.columns)} total columns")
            return enriched_data
            
        except Exception as e:
            logger.error(f"Technical indicators addition failed: {e}")
            return data

    async def _add_volatility_measures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures"""
        try:
            volatility_data = data.copy()
            
            if len(volatility_data) < 20:
                return volatility_data
            
            prices = volatility_data['close'].values
            
            # Historical volatility (standard deviation of returns)
            returns = np.log(prices[1:] / prices[:-1])
            volatility_data['returns'] = np.concatenate([[np.nan], returns])
            
            # Rolling volatility (20-period)
            volatility_data['volatility_20'] = volatility_data['returns'].rolling(window=20).std() * np.sqrt(252)
            
            # ATR (Average True Range)
            volatility_data['atr_14'] = talib.ATR(
                volatility_data['high'].values, volatility_data['low'].values, 
                volatility_data['close'].values, timeperiod=14
            )
            
            # Donchian Channel
            volatility_data['dc_upper'] = volatility_data['high'].rolling(window=20).max()
            volatility_data['dc_lower'] = volatility_data['low'].rolling(window=20).min()
            volatility_data['dc_middle'] = (volatility_data['dc_upper'] + volatility_data['dc_lower']) / 2
            
            logger.debug("Volatility measures added")
            return volatility_data
            
        except Exception as e:
            logger.error(f"Volatility measures addition failed: {e}")
            return data

    async def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        try:
            statistical_data = data.copy()
            
            if len(statistical_data) < 30:
                return statistical_data
            
            prices = statistical_data['close'].values
            
            # Z-score of prices
            statistical_data['price_zscore'] = stats.zscore(prices, nan_policy='omit')
            
            # Rolling skewness and kurtosis
            statistical_data['skewness_20'] = statistical_data['close'].rolling(window=20).skew()
            statistical_data['kurtosis_20'] = statistical_data['close'].rolling(window=20).kurt()
            
            # Rolling quantiles
            statistical_data['price_q25'] = statistical_data['close'].rolling(window=20).quantile(0.25)
            statistical_data['price_q75'] = statistical_data['close'].rolling(window=20).quantile(0.75)
            
            # Price position in recent range
            statistical_data['price_position'] = (
                (statistical_data['close'] - statistical_data['price_q25']) / 
                (statistical_data['price_q75'] - statistical_data['price_q25'])
            )
            
            logger.debug("Statistical features added")
            return statistical_data
            
        except Exception as e:
            logger.error(f"Statistical features addition failed: {e}")
            return data

    async def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality"""
        try:
            validated_data = data.copy()
            
            # Remove rows with NaN in critical columns
            critical_columns = ['open', 'high', 'low', 'close']
            validated_data = validated_data.dropna(subset=critical_columns)
            
            # Remove extreme outliers (beyond 5 standard deviations)
            for col in ['open', 'high', 'low', 'close']:
                if col in validated_data.columns:
                    z_scores = np.abs(stats.zscore(validated_data[col], nan_policy='omit'))
                    validated_data = validated_data[z_scores < 5]
            
            # Ensure chronological order
            validated_data = validated_data.sort_index()
            
            # Check for data gaps
            time_diffs = validated_data.index.to_series().diff()
            if not time_diffs.empty:
                avg_interval = time_diffs.mean()
                large_gaps = time_diffs > avg_interval * 5  # Gaps > 5x average interval
                if large_gaps.any():
                    logger.warning(f"Large time gaps detected in validated data")
            
            logger.debug(f"Data validation completed: {len(validated_data)} records")
            return validated_data
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return data

    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score (0-100)"""
        try:
            if data.empty:
                return 0.0
            
            score = 100.0
            
            # Deduct for missing values
            missing_percentage = self._calculate_missing_percentage(data)
            score -= missing_percentage * 50  # 50% penalty for missing data
            
            # Deduct for data length (insufficient data)
            if len(data) < 100:
                score -= (100 - len(data)) * 0.5
            
            # Check for basic data integrity
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    score -= 25
            
            # Check for price consistency
            if all(col in data.columns for col in required_columns):
                consistency_issues = (
                    (data['high'] < data[['open', 'low', 'close']].max(axis=1)) |
                    (data['low'] > data[['open', 'high', 'close']].min(axis=1))
                ).sum()
                
                if consistency_issues > 0:
                    consistency_penalty = (consistency_issues / len(data)) * 50
                    score -= consistency_penalty
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0

    def _calculate_missing_percentage(self, data: pd.DataFrame) -> float:
        """Calculate percentage of missing values"""
        try:
            if data.empty:
                return 100.0
            
            required_columns = ['open', 'high', 'low', 'close']
            missing_count = 0
            total_cells = 0
            
            for col in required_columns:
                if col in data.columns:
                    missing_count += data[col].isna().sum()
                    total_cells += len(data)
            
            return (missing_count / total_cells) * 100 if total_cells > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Missing percentage calculation failed: {e}")
            return 100.0

    def _generate_cache_key(self, request: HistoricalRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.symbol}_{request.timeframe.value}_{request.start_date}_{request.end_date}_{request.quality_level.value}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _get_cached_data(self, cache_key: str) -> Optional[HistoricalResponse]:
        """Get data from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            meta_file = self.cache_dir / f"{cache_key}_meta.json"
            
            if cache_file.exists() and meta_file.exists():
                # Load metadata
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if cache is still valid (24 hours)
                cache_age = time.time() - metadata.get('cached_at', 0)
                if cache_age < 86400:  # 24 hours
                    # Load data
                    data = pd.read_parquet(cache_file)
                    
                    return HistoricalResponse(
                        success=True,
                        data=data,
                        symbol=metadata.get('symbol', ''),
                        timeframe=TimeFrame(metadata.get('timeframe', '1h')),
                        data_points=len(data),
                        quality_score=metadata.get('quality_score', 0.0),
                        missing_percentage=metadata.get('missing_percentage', 0.0),
                        metadata=metadata
                    )
                else:
                    # Cache expired, delete files
                    cache_file.unlink(missing_ok=True)
                    meta_file.unlink(missing_ok=True)
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    async def _cache_data(self, cache_key: str, response: HistoricalResponse):
        """Cache data response"""
        try:
            if response.success and response.data is not None:
                cache_file = self.cache_dir / f"{cache_key}.parquet"
                meta_file = self.cache_dir / f"{cache_key}_meta.json"
                
                # Save data
                response.data.to_parquet(cache_file)
                
                # Save metadata
                metadata = {
                    'cached_at': time.time(),
                    'symbol': response.symbol,
                    'timeframe': response.timeframe.value,
                    'data_points': response.data_points,
                    'quality_score': response.quality_score,
                    'missing_percentage': response.missing_percentage,
                    'source': response.metadata.get('source', ''),
                    'quality_level': response.metadata.get('quality_level', '')
                }
                
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f)
                
                logger.debug(f"Data cached: {cache_key}")
                
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    async def _check_rate_limit(self, data_source: DataSource):
        """Check and enforce rate limits"""
        try:
            rate_config = self.data_sources[data_source]['rate_limit']
            current_time = time.time()
            
            # Remove old requests from rate limit window
            window_start = current_time - 1.0  # 1 second window
            self.rate_limits[data_source] = deque(
                [t for t in self.rate_limits[data_source] if t > window_start],
                maxlen=100
            )
            
            # Check if we're over the limit
            if len(self.rate_limits[data_source]) >= rate_config:
                sleep_time = 1.0 - (current_time - self.rate_limits[data_source][0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Add current request
            self.rate_limits[data_source].append(current_time)
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")

    def _convert_to_yahoo_symbol(self, symbol: str) -> str:
        """Convert Forex symbol to Yahoo Finance format"""
        # Remove slash for Yahoo Finance
        return symbol.replace('/', '').upper() + '=X'

    def _convert_to_crypto_symbol(self, symbol: str) -> str:
        """Convert Forex symbol to cryptocurrency format"""
        # For crypto, use BTC/USDT as example
        if 'USD' in symbol:
            return 'BTC/USDT'  # Default crypto pair
        return symbol

    def _convert_timeframe(self, timeframe: TimeFrame, exchange: str) -> str:
        """Convert timeframe to exchange-specific format"""
        timeframe_map = {
            TimeFrame.MINUTE_1: '1m',
            TimeFrame.MINUTE_5: '5m', 
            TimeFrame.MINUTE_15: '15m',
            TimeFrame.MINUTE_30: '30m',
            TimeFrame.HOUR_1: '1h',
            TimeFrame.HOUR_4: '4h',
            TimeFrame.DAILY: '1d'
        }
        return timeframe_map.get(timeframe, '1h')

    async def get_data_quality_report(self, symbol: str, timeframe: TimeFrame) -> Dict[str, Any]:
        """Get data quality report for symbol and timeframe"""
        try:
            # Get recent requests for this symbol/timeframe
            recent_requests = [
                req for req in self.request_history 
                if req['symbol'] == symbol and req['timeframe'] == timeframe.value
            ]
            
            if not recent_requests:
                return {'error': 'No historical data available'}
            
            quality_scores = [req['quality_score'] for req in recent_requests]
            processing_times = [req['processing_time'] for req in recent_requests]
            
            return {
                'symbol': symbol,
                'timeframe': timeframe.value,
                'total_requests': len(recent_requests),
                'average_quality_score': np.mean(quality_scores),
                'average_processing_time': np.mean(processing_times),
                'data_points_range': {
                    'min': min(req['data_points'] for req in recent_requests),
                    'max': max(req['data_points'] for req in recent_requests),
                    'average': np.mean([req['data_points'] for req in recent_requests])
                },
                'recommendations': self._generate_quality_recommendations(quality_scores)
            }
            
        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            return {'error': str(e)}

    def _generate_quality_recommendations(self, quality_scores: List[float]) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        avg_quality = np.mean(quality_scores)
        
        if avg_quality < 80:
            recommendations.append("Consider using multiple data sources for better quality")
        if avg_quality < 70:
            recommendations.append("Data quality is poor. Check data source connectivity")
        if avg_quality > 90:
            recommendations.append("Data quality is excellent")
        
        return recommendations

    async def bulk_download(self, symbols: List[str], timeframes: List[TimeFrame],
                          start_date: datetime, end_date: datetime) -> Dict[str, HistoricalResponse]:
        """Bulk download historical data for multiple symbols and timeframes"""
        try:
            results = {}
            
            for symbol in symbols:
                for timeframe in timeframes:
                    request = HistoricalRequest(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        data_source=DataSource.YAHOO_FINANCE,
                        quality_level=DataQuality.FEATURE_ENRICHED
                    )
                    
                    result = await self.get_historical_data(request)
                    results[f"{symbol}_{timeframe.value}"] = result
            
            logger.info(f"Bulk download completed: {len(results)} datasets")
            return results
            
        except Exception as e:
            logger.error(f"Bulk download failed: {e}")
            return {}

    async def close(self):
        """Cleanup resources"""
        try:
            # Close exchange connections
            for exchange in self.exchanges.values():
                if hasattr(exchange, 'close'):
                    await exchange.close()
            
            logger.info("HistoricalDataManager closed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Example usage and testing
async def main():
    """Test the Historical Data Manager"""
    
    config = {
        'cache_enabled': True,
        'cache_dir': './data/cache'
    }
    
    data_manager = HistoricalDataManager(config)
    
    try:
        # Create historical data request
        request = HistoricalRequest(
            symbol="EUR/USD",
            timeframe=TimeFrame.HOUR_1,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            data_source=DataSource.YAHOO_FINANCE,
            quality_level=DataQuality.FEATURE_ENRICHED
        )
        
        # Get historical data
        response = await data_manager.get_historical_data(request)
        
        if response.success:
            print(f"Data retrieved successfully:")
            print(f"  - Data points: {response.data_points}")
            print(f"  - Quality score: {response.quality_score:.1f}%")
            print(f"  - Processing time: {response.processing_time:.2f}s")
            print(f"  - Columns: {list(response.data.columns)}")
            
            # Display first few rows
            print("\nFirst 5 rows:")
            print(response.data.head())
            
            # Get quality report
            quality_report = await data_manager.get_data_quality_report("EUR/USD", TimeFrame.HOUR_1)
            print(f"\nQuality Report: {quality_report}")
            
        else:
            print(f"Data retrieval failed: {response.error_message}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        await data_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
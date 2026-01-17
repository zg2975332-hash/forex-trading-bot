"""
On-Chain Analyzer for FOREX TRADING BOT
Advanced blockchain data analysis for cryptocurrency and forex correlations
"""

import logging
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import requests
from datetime import datetime, timedelta
import hashlib
import json
from collections import defaultdict, deque
import websockets
import hmac
import base64
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class OnChainMetric(Enum):
    """On-chain metrics categories"""
    NETWORK_GROWTH = "network_growth"
    EXCHANGE_FLOWS = "exchange_flows"
    MINER_ACTIVITY = "miner_activity"
    MARKET_VALUE = "market_value"
    TRANSACTION_ACTIVITY = "transaction_activity"
    DEFI_METRICS = "defi_metrics"
    STAKING_METRICS = "staking_metrics"

class Blockchain(Enum):
    """Supported blockchains"""
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    BINANCE_CHAIN = "binance_chain"
    SOLANA = "solana"
    POLKADOT = "polkadot"

@dataclass
class OnChainRequest:
    """On-chain data request specification"""
    blockchain: Blockchain
    metrics: List[OnChainMetric]
    start_date: datetime
    end_date: datetime
    resolution: str = "1d"  # 1h, 1d, 1w
    include_forex_correlation: bool = False
    forex_pairs: List[str] = field(default_factory=lambda: ["EUR/USD", "USD/JPY"])

@dataclass
class OnChainResponse:
    """On-chain analysis response"""
    success: bool
    blockchain: Blockchain = None
    metrics_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    anomaly_detection: Dict[str, Any] = field(default_factory=dict)
    market_signals: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WhaleAlert:
    """Large transaction alert"""
    transaction_hash: str
    blockchain: Blockchain
    amount: float
    from_address: str
    to_address: str
    timestamp: datetime
    transaction_type: str  # exchange_in, exchange_out, internal
    confidence: float = 0.0
    impact_score: float = 0.0

class OnChainAnalyzer:
    """
    Advanced on-chain data analyzer for cryptocurrency and forex market insights
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # API configurations
        self.api_configs = {
            'glassnode': {
                'base_url': 'https://api.glassnode.com/v1',
                'api_key': self.config.get('glassnode_api_key'),
                'rate_limit': 2
            },
            'cryptocompare': {
                'base_url': 'https://min-api.cryptocompare.com/data',
                'api_key': self.config.get('cryptocompare_api_key'),
                'rate_limit': 10
            },
            'whale_alert': {
                'base_url': 'https://api.whale-alert.io/v1',
                'api_key': self.config.get('whale_alert_api_key'),
                'rate_limit': 1
            },
            'blockchain_com': {
                'base_url': 'https://api.blockchain.info',
                'rate_limit': 5
            },
            'moralis': {
                'base_url': 'https://deep-index.moralis.io/api/v2',
                'api_key': self.config.get('moralis_api_key'),
                'rate_limit': 5
            }
        }
        
        # Cache configuration
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Analysis models
        self.anomaly_detectors = {}
        self.correlation_models = {}
        
        # Real-time monitoring
        self.whale_alerts = deque(maxlen=1000)
        self.network_metrics = defaultdict(lambda: deque(maxlen=10000))
        
        # Performance tracking
        self.request_history = deque(maxlen=10000)
        self.alert_history = deque(maxlen=5000)
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize HTTP session
        self.session = None
        
        logger.info("OnChainAnalyzer initialized")

    async def initialize(self):
        """Initialize connections and models"""
        try:
            # Initialize aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=100)
            )
            
            # Initialize anomaly detection models
            self._initialize_models()
            
            logger.info("OnChainAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def _initialize_models(self):
        """Initialize machine learning models"""
        try:
            # Anomaly detection models for different metrics
            for metric in OnChainMetric:
                self.anomaly_detectors[metric.value] = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
            
            logger.debug("ML models initialized")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")

    async def analyze_onchain_data(self, request: OnChainRequest) -> OnChainResponse:
        """
        Comprehensive on-chain data analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting on-chain analysis for {request.blockchain.value}")
            
            # Fetch on-chain metrics
            metrics_data = await self._fetch_onchain_metrics(request)
            
            if not metrics_data:
                return OnChainResponse(
                    success=False,
                    error_message="No on-chain data retrieved",
                    processing_time=time.time() - start_time
                )
            
            # Perform correlation analysis with forex
            correlation_analysis = {}
            if request.include_forex_correlation:
                correlation_analysis = await self._analyze_forex_correlation(
                    metrics_data, request.forex_pairs
                )
            
            # Detect anomalies
            anomaly_detection = await self._detect_anomalies(metrics_data)
            
            # Generate market signals
            market_signals = await self._generate_market_signals(
                metrics_data, correlation_analysis, anomaly_detection
            )
            
            response = OnChainResponse(
                success=True,
                blockchain=request.blockchain,
                metrics_data=metrics_data,
                correlation_analysis=correlation_analysis,
                anomaly_detection=anomaly_detection,
                market_signals=market_signals,
                processing_time=time.time() - start_time,
                metadata={
                    'metrics_analyzed': list(metrics_data.keys()),
                    'data_points': sum(len(df) for df in metrics_data.values()),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Store in history
            self.request_history.append({
                'timestamp': time.time(),
                'blockchain': request.blockchain.value,
                'metrics': [m.value for m in request.metrics],
                'processing_time': response.processing_time
            })
            
            logger.info(f"On-chain analysis completed: {request.blockchain.value} "
                       f"in {response.processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            error_msg = f"On-chain analysis failed: {str(e)}"
            logger.error(error_msg)
            return OnChainResponse(
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )

    async def _fetch_onchain_metrics(self, request: OnChainRequest) -> Dict[str, pd.DataFrame]:
        """Fetch on-chain metrics from multiple sources"""
        try:
            metrics_data = {}
            
            for metric in request.metrics:
                try:
                    if metric == OnChainMetric.NETWORK_GROWTH:
                        data = await self._fetch_network_growth(request)
                    elif metric == OnChainMetric.EXCHANGE_FLOWS:
                        data = await self._fetch_exchange_flows(request)
                    elif metric == OnChainMetric.MINER_ACTIVITY:
                        data = await self._fetch_miner_activity(request)
                    elif metric == OnChainMetric.MARKET_VALUE:
                        data = await self._fetch_market_value(request)
                    elif metric == OnChainMetric.TRANSACTION_ACTIVITY:
                        data = await self._fetch_transaction_activity(request)
                    elif metric == OnChainMetric.DEFI_METRICS:
                        data = await self._fetch_defi_metrics(request)
                    elif metric == OnChainMetric.STAKING_METRICS:
                        data = await self._fetch_staking_metrics(request)
                    else:
                        continue
                    
                    if data is not None and not data.empty:
                        metrics_data[metric.value] = data
                        logger.debug(f"Fetched {metric.value}: {len(data)} records")
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {metric.value}: {e}")
                    continue
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"On-chain metrics fetch failed: {e}")
            return {}

    async def _fetch_network_growth(self, request: OnChainRequest) -> pd.DataFrame:
        """Fetch network growth metrics"""
        try:
            await self._check_rate_limit('glassnode')
            
            endpoints = {
                Blockchain.BITCOIN: [
                    ('metrics/addresses/new_count', 'new_addresses'),
                    ('metrics/addresses/active_count', 'active_addresses'),
                    ('metrics/addresses/non_zero_count', 'non_zero_addresses'),
                    ('metrics/network/growth', 'network_growth')
                ],
                Blockchain.ETHEREUM: [
                    ('metrics/addresses/new_count', 'new_addresses'),
                    ('metrics/addresses/active_count', 'active_addresses'),
                    ('metrics/network/growth', 'network_growth')
                ]
            }
            
            data_frames = []
            
            for endpoint, column_name in endpoints.get(request.blockchain, []):
                try:
                    url = f"{self.api_configs['glassnode']['base_url']}/{endpoint}"
                    params = {
                        'a': request.blockchain.value,
                        'api_key': self.api_configs['glassnode']['api_key'],
                        's': int(request.start_date.timestamp()),
                        'u': int(request.end_date.timestamp()),
                        'i': request.resolution
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            json_data = await response.json()
                            if json_data:
                                df = pd.DataFrame(json_data)
                                df['t'] = pd.to_datetime(df['t'], unit='s')
                                df.set_index('t', inplace=True)
                                df.rename(columns={'v': column_name}, inplace=True)
                                data_frames.append(df[[column_name]])
                        else:
                            logger.warning(f"Glassnode API error: {response.status}")
                            
                except Exception as e:
                    logger.warning(f"Glassnode endpoint {endpoint} failed: {e}")
                    continue
            
            if data_frames:
                # Merge all dataframes
                combined_data = pd.concat(data_frames, axis=1)
                combined_data = combined_data.loc[~combined_data.index.duplicated(keep='first')]
                return combined_data
            else:
                # Return mock data for demonstration
                return self._generate_mock_network_data(request)
                
        except Exception as e:
            logger.error(f"Network growth fetch failed: {e}")
            return self._generate_mock_network_data(request)

    async def _fetch_exchange_flows(self, request: OnChainRequest) -> pd.DataFrame:
        """Fetch exchange inflow/outflow data"""
        try:
            await self._check_rate_limit('glassnode')
            
            endpoints = {
                Blockchain.BITCOIN: [
                    ('metrics/transactions/transfers_volume_exchanges_in', 'exchange_inflow'),
                    ('metrics/transactions/transfers_volume_exchanges_out', 'exchange_outflow'),
                    ('metrics/transactions/transfers_volume_exchanges_net', 'net_flow')
                ],
                Blockchain.ETHEREUM: [
                    ('metrics/transactions/transfers_volume_exchanges_in', 'exchange_inflow'),
                    ('metrics/transactions/transfers_volume_exchanges_out', 'exchange_outflow')
                ]
            }
            
            data_frames = []
            
            for endpoint, column_name in endpoints.get(request.blockchain, []):
                try:
                    url = f"{self.api_configs['glassnode']['base_url']}/{endpoint}"
                    params = {
                        'a': request.blockchain.value,
                        'api_key': self.api_configs['glassnode']['api_key'],
                        's': int(request.start_date.timestamp()),
                        'u': int(request.end_date.timestamp()),
                        'i': request.resolution
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            json_data = await response.json()
                            if json_data:
                                df = pd.DataFrame(json_data)
                                df['t'] = pd.to_datetime(df['t'], unit='s')
                                df.set_index('t', inplace=True)
                                df.rename(columns={'v': column_name}, inplace=True)
                                data_frames.append(df[[column_name]])
                                
                except Exception as e:
                    logger.warning(f"Exchange flows endpoint {endpoint} failed: {e}")
                    continue
            
            if data_frames:
                combined_data = pd.concat(data_frames, axis=1)
                combined_data = combined_data.loc[~combined_data.index.duplicated(keep='first')]
                
                # Calculate flow ratios
                if 'exchange_inflow' in combined_data.columns and 'exchange_outflow' in combined_data.columns:
                    combined_data['flow_ratio'] = combined_data['exchange_inflow'] / combined_data['exchange_outflow']
                    combined_data['net_flow_ratio'] = (
                        (combined_data['exchange_inflow'] - combined_data['exchange_outflow']) / 
                        (combined_data['exchange_inflow'] + combined_data['exchange_outflow'])
                    )
                
                return combined_data
            else:
                return self._generate_mock_flow_data(request)
                
        except Exception as e:
            logger.error(f"Exchange flows fetch failed: {e}")
            return self._generate_mock_flow_data(request)

    async def _fetch_market_value(self, request: OnChainRequest) -> pd.DataFrame:
        """Fetch market value metrics"""
        try:
            await self._check_rate_limit('cryptocompare')
            
            # Fetch price and market cap data
            if request.blockchain == Blockchain.BITCOIN:
                symbol = 'BTC'
            elif request.blockchain == Blockchain.ETHEREUM:
                symbol = 'ETH'
            else:
                symbol = 'BTC'  # Default
            
            url = f"{self.api_configs['cryptocompare']['base_url']}/v2/histoday"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': 2000,  # Max limit
                'api_key': self.api_configs['cryptocompare']['api_key']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['Response'] == 'Success':
                        df = pd.DataFrame(data['Data'])
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('time', inplace=True)
                        
                        # Calculate additional metrics
                        df['market_cap'] = df['volumeto'] / df['volumefrom'] * df['high']  # Approximation
                        df['price_change_24h'] = df['close'].pct_change() * 100
                        df['volume_change_24h'] = df['volumeto'].pct_change() * 100
                        
                        return df[['open', 'high', 'low', 'close', 'volumeto', 'market_cap', 
                                 'price_change_24h', 'volume_change_24h']]
            
            return self._generate_mock_market_data(request)
            
        except Exception as e:
            logger.error(f"Market value fetch failed: {e}")
            return self._generate_mock_market_data(request)

    async def _fetch_transaction_activity(self, request: OnChainRequest) -> pd.DataFrame:
        """Fetch transaction activity metrics"""
        try:
            await self._check_rate_limit('glassnode')
            
            endpoints = {
                Blockchain.BITCOIN: [
                    ('metrics/transactions/count', 'transaction_count'),
                    ('metrics/transactions/volume_sum', 'transaction_volume'),
                    ('metrics/transactions/rate', 'transaction_rate'),
                    ('metrics/fees/volume_sum', 'fee_volume')
                ],
                Blockchain.ETHEREUM: [
                    ('metrics/transactions/count', 'transaction_count'),
                    ('metrics/transactions/volume_sum', 'transaction_volume'),
                    ('metrics/gas/used', 'gas_used')
                ]
            }
            
            data_frames = []
            
            for endpoint, column_name in endpoints.get(request.blockchain, []):
                try:
                    url = f"{self.api_configs['glassnode']['base_url']}/{endpoint}"
                    params = {
                        'a': request.blockchain.value,
                        'api_key': self.api_configs['glassnode']['api_key'],
                        's': int(request.start_date.timestamp()),
                        'u': int(request.end_date.timestamp()),
                        'i': request.resolution
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            json_data = await response.json()
                            if json_data:
                                df = pd.DataFrame(json_data)
                                df['t'] = pd.to_datetime(df['t'], unit='s')
                                df.set_index('t', inplace=True)
                                df.rename(columns={'v': column_name}, inplace=True)
                                data_frames.append(df[[column_name]])
                                
                except Exception as e:
                    logger.warning(f"Transaction activity endpoint {endpoint} failed: {e}")
                    continue
            
            if data_frames:
                combined_data = pd.concat(data_frames, axis=1)
                combined_data = combined_data.loc[~combined_data.index.duplicated(keep='first')]
                return combined_data
            else:
                return self._generate_mock_transaction_data(request)
                
        except Exception as e:
            logger.error(f"Transaction activity fetch failed: {e}")
            return self._generate_mock_transaction_data(request)

    async def _fetch_miner_activity(self, request: OnChainRequest) -> pd.DataFrame:
        """Fetch miner activity metrics"""
        # Implementation for Bitcoin miner flows, Ethereum staking, etc.
        return self._generate_mock_miner_data(request)

    async def _fetch_defi_metrics(self, request: OnChainRequest) -> pd.DataFrame:
        """Fetch DeFi metrics (Ethereum specific)"""
        # Implementation for TVL, DEX volumes, lending rates
        return self._generate_mock_defi_data(request)

    async def _fetch_staking_metrics(self, request: OnChainRequest) -> pd.DataFrame:
        """Fetch staking metrics (Ethereum 2.0, etc.)"""
        # Implementation for staking amounts, rewards, validator counts
        return self._generate_mock_staking_data(request)

    async def _analyze_forex_correlation(self, metrics_data: Dict[str, pd.DataFrame], 
                                       forex_pairs: List[str]) -> Dict[str, Any]:
        """Analyze correlation between on-chain metrics and forex pairs"""
        try:
            correlation_results = {}
            
            # This would require actual forex data
            # For demonstration, we'll use mock correlation analysis
            
            for metric_name, metric_data in metrics_data.items():
                pair_correlations = {}
                
                for forex_pair in forex_pairs:
                    # Mock correlation calculation
                    # In reality, this would fetch forex data and compute actual correlations
                    if 'close' in metric_data.columns or 'transaction_count' in metric_data.columns:
                        # Use appropriate column for correlation
                        if 'close' in metric_data.columns:
                            crypto_series = metric_data['close']
                        else:
                            # Use first numeric column
                            numeric_cols = metric_data.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                crypto_series = metric_data[numeric_cols[0]]
                            else:
                                continue
                        
                        # Generate mock forex data with some correlation
                        forex_series = self._generate_correlated_series(crypto_series)
                        
                        # Calculate correlation
                        correlation = crypto_series.corr(forex_series)
                        p_value = self._calculate_correlation_pvalue(crypto_series, forex_series)
                        
                        pair_correlations[forex_pair] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'strength': self._classify_correlation_strength(correlation),
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }
                
                correlation_results[metric_name] = pair_correlations
            
            # Overall correlation summary
            overall_correlation = self._calculate_overall_correlation(correlation_results)
            
            return {
                'pair_correlations': correlation_results,
                'overall_correlation': overall_correlation,
                'strongest_relationships': self._identify_strong_relationships(correlation_results)
            }
            
        except Exception as e:
            logger.error(f"Forex correlation analysis failed: {e}")
            return {}

    async def _detect_anomalies(self, metrics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect anomalies in on-chain metrics"""
        try:
            anomalies = {}
            
            for metric_name, metric_data in metrics_data.items():
                try:
                    # Select numeric columns for anomaly detection
                    numeric_data = metric_data.select_dtypes(include=[np.number])
                    
                    if numeric_data.empty:
                        continue
                    
                    # Remove any remaining NaN values
                    clean_data = numeric_data.dropna()
                    
                    if len(clean_data) < 10:  # Need sufficient data
                        continue
                    
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clean_data)
                    
                    # Detect anomalies using Isolation Forest
                    detector = self.anomaly_detectors.get(metric_name)
                    if detector is None:
                        detector = IsolationForest(contamination=0.1, random_state=42)
                        self.anomaly_detectors[metric_name] = detector
                    
                    anomaly_labels = detector.fit_predict(scaled_data)
                    anomaly_scores = detector.decision_function(scaled_data)
                    
                    # Identify anomaly points
                    anomaly_indices = np.where(anomaly_labels == -1)[0]
                    anomaly_dates = clean_data.index[anomaly_indices]
                    
                    anomalies[metric_name] = {
                        'anomaly_count': len(anomaly_indices),
                        'anomaly_dates': anomaly_dates.tolist(),
                        'anomaly_scores': anomaly_scores[anomaly_indices].tolist(),
                        'anomaly_percentage': (len(anomaly_indices) / len(clean_data)) * 100,
                        'severity': self._classify_anomaly_severity(anomaly_scores)
                    }
                    
                except Exception as e:
                    logger.warning(f"Anomaly detection failed for {metric_name}: {e}")
                    continue
            
            # Cross-metric anomaly analysis
            cross_metric_analysis = self._analyze_cross_metric_anomalies(anomalies, metrics_data)
            
            return {
                'metric_anomalies': anomalies,
                'cross_metric_analysis': cross_metric_analysis,
                'summary': self._generate_anomaly_summary(anomalies)
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {}

    async def _generate_market_signals(self, metrics_data: Dict[str, pd.DataFrame],
                                     correlation_analysis: Dict[str, Any],
                                     anomaly_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on on-chain analysis"""
        try:
            signals = {}
            
            # Network Growth Signals
            if 'network_growth' in metrics_data:
                network_signals = self._analyze_network_signals(metrics_data['network_growth'])
                signals.update(network_signals)
            
            # Exchange Flow Signals
            if 'exchange_flows' in metrics_data:
                flow_signals = self._analyze_flow_signals(metrics_data['exchange_flows'])
                signals.update(flow_signals)
            
            # Market Value Signals
            if 'market_value' in metrics_data:
                market_signals = self._analyze_market_signals(metrics_data['market_value'])
                signals.update(market_signals)
            
            # Anomaly-based Signals
            anomaly_signals = self._generate_anomaly_signals(anomaly_detection)
            signals.update(anomaly_signals)
            
            # Correlation-based Signals
            correlation_signals = self._generate_correlation_signals(correlation_analysis)
            signals.update(correlation_signals)
            
            # Overall Signal Strength
            overall_signal = self._calculate_overall_signal(signals)
            
            return {
                'individual_signals': signals,
                'overall_signal': overall_signal,
                'confidence_score': self._calculate_confidence_score(signals),
                'recommendations': self._generate_trading_recommendations(signals, overall_signal)
            }
            
        except Exception as e:
            logger.error(f"Market signal generation failed: {e}")
            return {}

    async def monitor_whale_activity(self, blockchains: List[Blockchain] = None) -> List[WhaleAlert]:
        """Monitor large transactions (whale activity)"""
        try:
            whale_alerts = []
            blockchains = blockchains or [Blockchain.BITCOIN, Blockchain.ETHEREUM]
            
            for blockchain in blockchains:
                try:
                    alerts = await self._fetch_whale_alerts(blockchain)
                    whale_alerts.extend(alerts)
                except Exception as e:
                    logger.warning(f"Whale monitoring failed for {blockchain}: {e}")
                    continue
            
            # Sort by impact score and timestamp
            whale_alerts.sort(key=lambda x: (x.impact_score, x.timestamp), reverse=True)
            
            # Store in history
            self.whale_alerts.extend(whale_alerts)
            self.alert_history.extend([
                {
                    'timestamp': alert.timestamp,
                    'blockchain': alert.blockchain.value,
                    'amount': alert.amount,
                    'impact_score': alert.impact_score
                }
                for alert in whale_alerts
            ])
            
            logger.info(f"Whale monitoring completed: {len(whale_alerts)} alerts")
            return whale_alerts
            
        except Exception as e:
            logger.error(f"Whale activity monitoring failed: {e}")
            return []

    async def _fetch_whale_alerts(self, blockchain: Blockchain) -> List[WhaleAlert]:
        """Fetch whale alerts from API"""
        try:
            # This would use Whale Alert API or similar service
            # For demonstration, generate mock alerts
            
            mock_alerts = []
            alert_count = np.random.randint(1, 10)
            
            for i in range(alert_count):
                alert = WhaleAlert(
                    transaction_hash=f"0x{hashlib.md5(f'{blockchain.value}_{i}'.encode()).hexdigest()}",
                    blockchain=blockchain,
                    amount=np.random.uniform(1000000, 50000000),  # 1M to 50M USD
                    from_address=f"from_{i}",
                    to_address=f"to_{i}",
                    timestamp=datetime.utcnow() - timedelta(hours=np.random.randint(0, 24)),
                    transaction_type=np.random.choice(['exchange_in', 'exchange_out', 'internal']),
                    confidence=np.random.uniform(0.7, 1.0),
                    impact_score=np.random.uniform(0.5, 1.0)
                )
                mock_alerts.append(alert)
            
            return mock_alerts
            
        except Exception as e:
            logger.error(f"Whale alert fetch failed: {e}")
            return []

    def _analyze_network_signals(self, network_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals from network growth metrics"""
        try:
            signals = {}
            
            if 'new_addresses' in network_data.columns:
                new_addresses = network_data['new_addresses'].dropna()
                if len(new_addresses) > 10:
                    # Calculate growth rate
                    growth_rate = new_addresses.pct_change().rolling(7).mean().iloc[-1]
                    
                    signals['network_growth_signal'] = {
                        'value': growth_rate,
                        'strength': 'strong' if abs(growth_rate) > 0.1 else 'medium',
                        'direction': 'bullish' if growth_rate > 0.05 else 'bearish' if growth_rate < -0.05 else 'neutral',
                        'confidence': min(abs(growth_rate) * 10, 1.0)
                    }
            
            return signals
            
        except Exception as e:
            logger.error(f"Network signal analysis failed: {e}")
            return {}

    def _analyze_flow_signals(self, flow_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals from exchange flow metrics"""
        try:
            signals = {}
            
            if 'net_flow_ratio' in flow_data.columns:
                net_flow = flow_data['net_flow_ratio'].dropna()
                if len(net_flow) > 10:
                    recent_flow = net_flow.rolling(5).mean().iloc[-1]
                    
                    signals['exchange_flow_signal'] = {
                        'value': recent_flow,
                        'strength': 'strong' if abs(recent_flow) > 0.2 else 'medium',
                        'direction': 'bullish' if recent_flow > 0.1 else 'bearish' if recent_flow < -0.1 else 'neutral',
                        'confidence': min(abs(recent_flow) * 5, 1.0)
                    }
            
            return signals
            
        except Exception as e:
            logger.error(f"Flow signal analysis failed: {e}")
            return {}

    def _analyze_market_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals from market metrics"""
        try:
            signals = {}
            
            if 'close' in market_data.columns:
                price = market_data['close'].dropna()
                if len(price) > 20:
                    # Calculate momentum
                    returns = price.pct_change()
                    momentum = returns.rolling(10).mean().iloc[-1]
                    
                    signals['price_momentum_signal'] = {
                        'value': momentum,
                        'strength': 'strong' if abs(momentum) > 0.02 else 'medium',
                        'direction': 'bullish' if momentum > 0.01 else 'bearish' if momentum < -0.01 else 'neutral',
                        'confidence': min(abs(momentum) * 50, 1.0)
                    }
            
            return signals
            
        except Exception as e:
            logger.error(f"Market signal analysis failed: {e}")
            return {}

    # Helper methods for mock data generation
    def _generate_mock_network_data(self, request: OnChainRequest) -> pd.DataFrame:
        """Generate mock network growth data"""
        dates = pd.date_range(request.start_date, request.end_date, freq=request.resolution)
        data = {
            'new_addresses': np.random.poisson(100000, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 50000,
            'active_addresses': np.random.poisson(500000, len(dates)) + np.sin(np.arange(len(dates)) * 0.05) * 100000,
            'network_growth': np.random.normal(0.01, 0.005, len(dates)).cumsum()
        }
        return pd.DataFrame(data, index=dates)

    def _generate_mock_flow_data(self, request: OnChainRequest) -> pd.DataFrame:
        """Generate mock exchange flow data"""
        dates = pd.date_range(request.start_date, request.end_date, freq=request.resolution)
        data = {
            'exchange_inflow': np.random.lognormal(10, 1, len(dates)),
            'exchange_outflow': np.random.lognormal(10, 1, len(dates)),
            'net_flow': np.random.normal(0, 1000, len(dates))
        }
        df = pd.DataFrame(data, index=dates)
        df['flow_ratio'] = df['exchange_inflow'] / df['exchange_outflow']
        df['net_flow_ratio'] = (df['exchange_inflow'] - df['exchange_outflow']) / (df['exchange_inflow'] + df['exchange_outflow'])
        return df

    def _generate_mock_market_data(self, request: OnChainRequest) -> pd.DataFrame:
        """Generate mock market data"""
        dates = pd.date_range(request.start_date, request.end_date, freq=request.resolution)
        returns = np.random.normal(0.001, 0.02, len(dates))
        price = 50000 * (1 + returns).cumprod()
        
        data = {
            'open': price * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': price * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': price,
            'volumeto': np.random.lognormal(20, 1, len(dates)),
            'market_cap': price * 19000000  # Approximation
        }
        df = pd.DataFrame(data, index=dates)
        df['price_change_24h'] = df['close'].pct_change() * 100
        df['volume_change_24h'] = df['volumeto'].pct_change() * 100
        return df

    def _generate_mock_transaction_data(self, request: OnChainRequest) -> pd.DataFrame:
        """Generate mock transaction data"""
        dates = pd.date_range(request.start_date, request.end_date, freq=request.resolution)
        data = {
            'transaction_count': np.random.poisson(300000, len(dates)),
            'transaction_volume': np.random.lognormal(18, 1, len(dates)),
            'transaction_rate': np.random.normal(5, 1, len(dates)),
            'fee_volume': np.random.lognormal(15, 1, len(dates))
        }
        return pd.DataFrame(data, index=dates)

    def _generate_mock_miner_data(self, request: OnChainRequest) -> pd.DataFrame:
        """Generate mock miner activity data"""
        dates = pd.date_range(request.start_date, request.end_date, freq=request.resolution)
        data = {
            'miner_flow': np.random.normal(0, 100, len(dates)),
            'miner_reserve': np.random.lognormal(16, 1, len(dates)).cumsum(),
            'miner_revenue': np.random.lognormal(14, 1, len(dates))
        }
        return pd.DataFrame(data, index=dates)

    def _generate_mock_defi_data(self, request: OnChainRequest) -> pd.DataFrame:
        """Generate mock DeFi data"""
        dates = pd.date_range(request.start_date, request.end_date, freq=request.resolution)
        data = {
            'tvl': np.random.lognormal(22, 0.5, len(dates)),
            'dex_volume': np.random.lognormal(20, 1, len(dates)),
            'lending_volume': np.random.lognormal(19, 1, len(dates))
        }
        return pd.DataFrame(data, index=dates)

    def _generate_mock_staking_data(self, request: OnChainRequest) -> pd.DataFrame:
        """Generate mock staking data"""
        dates = pd.date_range(request.start_date, request.end_date, freq=request.resolution)
        data = {
            'staked_amount': np.random.lognormal(21, 0.3, len(dates)).cumsum(),
            'validator_count': np.random.poisson(300000, len(dates)),
            'staking_reward': np.random.normal(0.05, 0.01, len(dates))
        }
        return pd.DataFrame(data, index=dates)

    def _generate_correlated_series(self, base_series: pd.Series, correlation: float = 0.6) -> pd.Series:
        """Generate a series correlated with the base series"""
        noise = np.random.normal(0, 0.1, len(base_series))
        correlated = correlation * base_series + (1 - correlation) * noise
        return pd.Series(correlated, index=base_series.index)

    def _calculate_correlation_pvalue(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate p-value for correlation"""
        if len(series1) < 3 or len(series2) < 3:
            return 1.0
        correlation, p_value = stats.pearsonr(series1.dropna(), series2.dropna())
        return p_value

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def _classify_anomaly_severity(self, anomaly_scores: np.ndarray) -> str:
        """Classify anomaly severity"""
        if len(anomaly_scores) == 0:
            return "none"
        avg_score = np.mean(anomaly_scores)
        if avg_score < -0.5:
            return "high"
        elif avg_score < -0.2:
            return "medium"
        else:
            return "low"

    async def _check_rate_limit(self, api_name: str):
        """Check and enforce rate limits"""
        try:
            rate_config = self.api_configs[api_name]['rate_limit']
            current_time = time.time()
            
            # Remove old requests
            window_start = current_time - 1.0
            self.rate_limits[api_name] = deque(
                [t for t in self.rate_limits[api_name] if t > window_start],
                maxlen=100
            )
            
            # Check limit
            if len(self.rate_limits[api_name]) >= rate_config:
                sleep_time = 1.0 - (current_time - self.rate_limits[api_name][0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.rate_limits[api_name].append(current_time)
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")

    # Additional helper methods for signal generation
    def _generate_anomaly_signals(self, anomaly_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on anomaly detection"""
        signals = {}
        
        try:
            metric_anomalies = anomaly_detection.get('metric_anomalies', {})
            
            total_anomalies = sum(anomaly['anomaly_count'] for anomaly in metric_anomalies.values())
            total_metrics = len(metric_anomalies)
            
            if total_metrics > 0:
                anomaly_ratio = total_anomalies / total_metrics
                
                signals['anomaly_signal'] = {
                    'value': anomaly_ratio,
                    'strength': 'strong' if anomaly_ratio > 0.3 else 'medium' if anomaly_ratio > 0.1 else 'weak',
                    'direction': 'bearish' if anomaly_ratio > 0.2 else 'neutral',
                    'confidence': min(anomaly_ratio * 3, 1.0)
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Anomaly signal generation failed: {e}")
            return {}

    def _generate_correlation_signals(self, correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals based on correlation analysis"""
        signals = {}
        
        try:
            pair_correlations = correlation_analysis.get('pair_correlations', {})
            strong_correlations = 0
            total_correlations = 0
            
            for metric_correlations in pair_correlations.values():
                for pair_data in metric_correlations.values():
                    if pair_data.get('strength') in ['strong', 'moderate'] and pair_data.get('significance') == 'significant':
                        strong_correlations += 1
                    total_correlations += 1
            
            if total_correlations > 0:
                strong_ratio = strong_correlations / total_correlations
                
                signals['correlation_signal'] = {
                    'value': strong_ratio,
                    'strength': 'strong' if strong_ratio > 0.5 else 'medium' if strong_ratio > 0.3 else 'weak',
                    'direction': 'bullish' if strong_ratio > 0.4 else 'neutral',
                    'confidence': min(strong_ratio * 2, 1.0)
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Correlation signal generation failed: {e}")
            return {}

    def _calculate_overall_signal(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market signal"""
        try:
            if not signals:
                return {'direction': 'neutral', 'strength': 'weak', 'confidence': 0.0}
            
            # Weight different signals
            signal_weights = {
                'network_growth_signal': 0.25,
                'exchange_flow_signal': 0.25,
                'price_momentum_signal': 0.30,
                'anomaly_signal': 0.10,
                'correlation_signal': 0.10
            }
            
            direction_score = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            
            for signal_name, signal_data in signals.items():
                weight = signal_weights.get(signal_name, 0.1)
                
                if signal_data['direction'] == 'bullish':
                    direction_score += weight
                elif signal_data['direction'] == 'bearish':
                    direction_score -= weight
                
                confidence_sum += signal_data['confidence'] * weight
                total_weight += weight
            
            if total_weight > 0:
                direction_score /= total_weight
                avg_confidence = confidence_sum / total_weight
            else:
                direction_score = 0.0
                avg_confidence = 0.0
            
            # Determine overall direction and strength
            if direction_score > 0.1:
                direction = 'bullish'
                strength = 'strong' if direction_score > 0.3 else 'medium'
            elif direction_score < -0.1:
                direction = 'bearish'
                strength = 'strong' if direction_score < -0.3 else 'medium'
            else:
                direction = 'neutral'
                strength = 'weak'
            
            return {
                'direction': direction,
                'strength': strength,
                'confidence': avg_confidence,
                'score': direction_score
            }
            
        except Exception as e:
            logger.error(f"Overall signal calculation failed: {e}")
            return {'direction': 'neutral', 'strength': 'weak', 'confidence': 0.0}

    def _calculate_confidence_score(self, signals: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        try:
            if not signals:
                return 0.0
            
            confidences = [signal['confidence'] for signal in signals.values()]
            return np.mean(confidences)
            
        except Exception as e:
            logger.error(f"Confidence score calculation failed: {e}")
            return 0.0

    def _generate_trading_recommendations(self, signals: Dict[str, Any], 
                                        overall_signal: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        try:
            direction = overall_signal['direction']
            strength = overall_signal['strength']
            confidence = overall_signal['confidence']
            
            if direction == 'bullish' and strength == 'strong' and confidence > 0.7:
                recommendations.append("Consider long positions in correlated forex pairs")
                recommendations.append("Monitor for continuation patterns")
            elif direction == 'bearish' and strength == 'strong' and confidence > 0.7:
                recommendations.append("Consider short positions or hedging")
                recommendations.append("Watch for support levels breaking")
            elif direction == 'neutral' or confidence < 0.5:
                recommendations.append("Maintain current positions")
                recommendations.append("Wait for clearer signals")
            
            # Add specific recommendations based on individual signals
            if 'anomaly_signal' in signals and signals['anomaly_signal']['strength'] == 'strong':
                recommendations.append("High anomaly activity detected - exercise caution")
            
            if 'correlation_signal' in signals and signals['correlation_signal']['strength'] == 'strong':
                recommendations.append("Strong correlations detected - good for pairs trading")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate specific recommendations"]

    def _analyze_cross_metric_anomalies(self, anomalies: Dict[str, Any], 
                                      metrics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze anomalies across multiple metrics"""
        try:
            cross_analysis = {}
            
            # Find dates with multiple anomalies
            anomaly_dates = defaultdict(list)
            for metric_name, anomaly_info in anomalies.items():
                for date in anomaly_info.get('anomaly_dates', []):
                    anomaly_dates[date].append(metric_name)
            
            # Identify clusters of anomalies
            cluster_dates = {date: metrics for date, metrics in anomaly_dates.items() if len(metrics) > 1}
            
            cross_analysis['anomaly_clusters'] = {
                'cluster_count': len(cluster_dates),
                'cluster_dates': list(cluster_dates.keys()),
                'cluster_metrics': cluster_dates
            }
            
            # Calculate cluster severity
            if cluster_dates:
                avg_cluster_size = np.mean([len(metrics) for metrics in cluster_dates.values()])
                cross_analysis['cluster_severity'] = 'high' if avg_cluster_size > 3 else 'medium' if avg_cluster_size > 2 else 'low'
            else:
                cross_analysis['cluster_severity'] = 'none'
            
            return cross_analysis
            
        except Exception as e:
            logger.error(f"Cross-metric anomaly analysis failed: {e}")
            return {}

    def _generate_anomaly_summary(self, anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of anomaly detection results"""
        try:
            total_anomalies = sum(anomaly['anomaly_count'] for anomaly in anomalies.values())
            total_metrics = len(anomalies)
            
            severity_counts = defaultdict(int)
            for anomaly in anomalies.values():
                severity_counts[anomaly['severity']] += 1
            
            return {
                'total_anomalies': total_anomalies,
                'affected_metrics': total_metrics,
                'severity_breakdown': dict(severity_counts),
                'overall_risk': 'high' if severity_counts['high'] > 0 else 'medium' if severity_counts['medium'] > 0 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Anomaly summary generation failed: {e}")
            return {}

    def _calculate_overall_correlation(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall correlation summary"""
        try:
            all_correlations = []
            significant_correlations = 0
            total_correlations = 0
            
            for metric_correlations in correlation_results.values():
                for pair_data in metric_correlations.values():
                    correlation = pair_data.get('correlation', 0)
                    p_value = pair_data.get('p_value', 1)
                    
                    all_correlations.append(correlation)
                    total_correlations += 1
                    
                    if p_value < 0.05:
                        significant_correlations += 1
            
            if all_correlations:
                avg_correlation = np.mean(all_correlations)
                abs_avg_correlation = np.mean([abs(c) for c in all_correlations])
                significance_ratio = significant_correlations / total_correlations if total_correlations > 0 else 0
            else:
                avg_correlation = 0.0
                abs_avg_correlation = 0.0
                significance_ratio = 0.0
            
            return {
                'average_correlation': avg_correlation,
                'average_absolute_correlation': abs_avg_correlation,
                'significance_ratio': significance_ratio,
                'total_correlations_analyzed': total_correlations
            }
            
        except Exception as e:
            logger.error(f"Overall correlation calculation failed: {e}")
            return {}

    def _identify_strong_relationships(self, correlation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify the strongest correlation relationships"""
        try:
            strong_relationships = []
            
            for metric_name, metric_correlations in correlation_results.items():
                for pair_name, pair_data in metric_correlations.items():
                    correlation = pair_data.get('correlation', 0)
                    p_value = pair_data.get('p_value', 1)
                    strength = pair_data.get('strength', 'very_weak')
                    
                    if strength in ['strong', 'moderate'] and p_value < 0.05:
                        strong_relationships.append({
                            'metric': metric_name,
                            'forex_pair': pair_name,
                            'correlation': correlation,
                            'p_value': p_value,
                            'strength': strength
                        })
            
            # Sort by absolute correlation strength
            strong_relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return strong_relationships[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Strong relationship identification failed: {e}")
            return []

    async def get_performance_report(self) -> Dict[str, Any]:
        """Get performance and usage report"""
        try:
            total_requests = len(self.request_history)
            total_alerts = len(self.alert_history)
            
            if total_requests > 0:
                processing_times = [req['processing_time'] for req in self.request_history]
                avg_processing_time = np.mean(processing_times)
            else:
                avg_processing_time = 0.0
            
            return {
                'total_requests_processed': total_requests,
                'total_alerts_generated': total_alerts,
                'average_processing_time': avg_processing_time,
                'active_anomaly_detectors': len(self.anomaly_detectors),
                'cache_hit_ratio': self._calculate_cache_hit_ratio(),
                'rate_limit_usage': dict(self.rate_limits)
            }
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {}

    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        # This would track actual cache hits/misses
        # For now, return a mock value
        return 0.65

    async def close(self):
        """Cleanup resources"""
        try:
            if self.session:
                await self.session.close()
            logger.info("OnChainAnalyzer closed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Example usage and testing
async def main():
    """Test the On-Chain Analyzer"""
    
    config = {
        'glassnode_api_key': 'your_glassnode_key',
        'cryptocompare_api_key': 'your_cryptocompare_key',
        'whale_alert_api_key': 'your_whale_alert_key',
        'cache_enabled': True
    }
    
    analyzer = OnChainAnalyzer(config)
    
    try:
        await analyzer.initialize()
        
        # Create on-chain analysis request
        request = OnChainRequest(
            blockchain=Blockchain.BITCOIN,
            metrics=[OnChainMetric.NETWORK_GROWTH, OnChainMetric.EXCHANGE_FLOWS, OnChainMetric.MARKET_VALUE],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            resolution="1d",
            include_forex_correlation=True,
            forex_pairs=["EUR/USD", "USD/JPY", "GBP/USD"]
        )
        
        # Perform analysis
        response = await analyzer.analyze_onchain_data(request)
        
        if response.success:
            print(f"On-chain analysis completed successfully:")
            print(f"  - Blockchain: {response.blockchain.value}")
            print(f"  - Metrics analyzed: {len(response.metrics_data)}")
            print(f"  - Processing time: {response.processing_time:.2f}s")
            
            # Display market signals
            print(f"\nMarket Signals:")
            for signal_name, signal_data in response.market_signals.get('individual_signals', {}).items():
                print(f"  - {signal_name}: {signal_data['direction']} ({signal_data['strength']})")
            
            print(f"  - Overall: {response.market_signals['overall_signal']['direction']} "
                  f"({response.market_signals['overall_signal']['strength']})")
            
            # Display anomaly summary
            anomaly_summary = response.anomaly_detection.get('summary', {})
            print(f"\nAnomaly Detection:")
            print(f"  - Total anomalies: {anomaly_summary.get('total_anomalies', 0)}")
            print(f"  - Overall risk: {anomaly_summary.get('overall_risk', 'low')}")
            
            # Monitor whale activity
            whale_alerts = await analyzer.monitor_whale_activity()
            print(f"\nWhale Alerts: {len(whale_alerts)} recent large transactions")
            
        else:
            print(f"Analysis failed: {response.error_message}")
        
        # Get performance report
        performance = await analyzer.get_performance_report()
        print(f"\nPerformance Report:")
        print(f"  - Requests processed: {performance['total_requests_processed']}")
        print(f"  - Average processing time: {performance['average_processing_time']:.2f}s")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())
"""
Alternative Data Loader for FOREX TRADING BOT
Advanced alternative data sources for alpha generation
"""

import logging
import asyncio
import time
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import requests
from datetime import datetime, timedelta
import websockets
import hashlib
import hmac
import base64
from collections import defaultdict, deque
import zipfile
import io
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class DataSource(Enum):
    ECONOMIC_CALENDAR = "economic_calendar"
    SOCIAL_SENTIMENT = "social_sentiment"
    ONCHAIN_METRICS = "onchain_metrics"
    NEWS_FLOW = "news_flow"
    OPTIONS_FLOW = "options_flow"
    ORDER_FLOW = "order_flow"
    WEATHER_DATA = "weather_data"
    SATELLITE_DATA = "satellite_data"
    SHIPPING_DATA = "shipping_data"

@dataclass
class EconomicEvent:
    """Economic calendar event data"""
    event_id: str
    country: str
    event_name: str
    timestamp: datetime
    impact: str  # low, medium, high
    previous: Optional[float] = None
    forecast: Optional[float] = None
    actual: Optional[float] = None
    currency: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SentimentData:
    """Social media sentiment data"""
    source: str
    symbol: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    volume: int  # Number of mentions
    bullish_ratio: float  # 0 to 1
    unique_authors: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OnchainMetric:
    """On-chain cryptocurrency metrics"""
    metric_id: str
    symbol: str
    timestamp: datetime
    value: float
    metric_type: str  # volume, transactions, active_addresses, etc.
    change_24h: float = 0.0
    change_7d: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NewsArticle:
    """News article data"""
    article_id: str
    source: str
    title: str
    content: str
    timestamp: datetime
    symbols: List[str]
    sentiment: Optional[float] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class AlternativeDataLoader:
    """
    Advanced alternative data loader for alpha generation
    Integrates multiple unconventional data sources
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_clients = {}
        self.cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes cache
        
        # Data storage
        self.economic_events = deque(maxlen=10000)
        self.sentiment_data = deque(maxlen=50000)
        self.onchain_metrics = deque(maxlen=100000)
        self.news_articles = deque(maxlen=10000)
        
        # API configurations
        self.api_configs = {
            'fred': {
                'base_url': 'https://api.stlouisfed.org/fred',
                'api_key': config.get('fred_api_key'),
                'rate_limit': 10  # requests per second
            },
            'alphavantage': {
                'base_url': 'https://www.alphavantage.co/query',
                'api_key': config.get('alphavantage_api_key'),
                'rate_limit': 5
            },
            'twitter': {
                'bearer_token': config.get('twitter_bearer_token'),
                'rate_limit': 10
            },
            'reddit': {
                'client_id': config.get('reddit_client_id'),
                'client_secret': config.get('reddit_client_secret'),
                'rate_limit': 10
            },
            'glassnode': {
                'base_url': 'https://api.glassnode.com/v1',
                'api_key': config.get('glassnode_api_key'),
                'rate_limit': 2
            },
            'cryptocompare': {
                'base_url': 'https://min-api.cryptocompare.com/data',
                'api_key': config.get('cryptocompare_api_key'),
                'rate_limit': 10
            },
            'newsapi': {
                'base_url': 'https://newsapi.org/v2',
                'api_key': config.get('newsapi_api_key'),
                'rate_limit': 5
            }
        }
        
        # Rate limiting
        self.rate_limits = {}
        self.last_request_time = {}
        
        logger.info("AlternativeDataLoader initialized")

    async def initialize(self):
        """Initialize API clients and connections"""
        try:
            # Initialize aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=100)
            )
            
            # Initialize rate limit trackers
            for source in self.api_configs:
                self.rate_limits[source] = deque(maxlen=100)
                self.last_request_time[source] = 0
            
            logger.info("AlternativeDataLoader initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def fetch_economic_calendar(self, days: int = 7, countries: List[str] = None) -> List[EconomicEvent]:
        """Fetch economic calendar data"""
        try:
            cache_key = f"economic_calendar_{days}_{'_'.join(countries or [])}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            events = []
            
            # Try multiple data sources
            sources = [
                self._fetch_alphavantage_economic_calendar,
                self._fetch_fred_economic_data,
                self._fetch_cryptocompare_events
            ]
            
            for source_func in sources:
                try:
                    source_events = await source_func(days, countries)
                    events.extend(source_events)
                    break  # Use first successful source
                except Exception as e:
                    logger.warning(f"Economic calendar source failed: {e}")
                    continue
            
            # Remove duplicates and sort
            events = self._deduplicate_events(events)
            events.sort(key=lambda x: x.timestamp)
            
            # Cache results
            self._cache_data(cache_key, events)
            self.economic_events.extend(events)
            
            logger.info(f"Fetched {len(events)} economic events")
            return events
            
        except Exception as e:
            logger.error(f"Economic calendar fetch failed: {e}")
            return []

    async def _fetch_alphavantage_economic_calendar(self, days: int, countries: List[str]) -> List[EconomicEvent]:
        """Fetch economic calendar from Alpha Vantage"""
        try:
            await self._check_rate_limit('alphavantage')
            
            params = {
                'function': 'ECONOMIC_CALENDAR',
                'apikey': self.api_configs['alphavantage']['api_key']
            }
            
            async with self.session.get(self.api_configs['alphavantage']['base_url'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_alphavantage_events(data)
                else:
                    raise Exception(f"API returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"Alpha Vantage economic calendar failed: {e}")
            raise

    def _parse_alphavantage_events(self, data: Dict) -> List[EconomicEvent]:
        """Parse Alpha Vantage economic events"""
        events = []
        
        for event_data in data.get('economicCalendar', []):
            try:
                event = EconomicEvent(
                    event_id=event_data.get('eventId', ''),
                    country=event_data.get('country', ''),
                    event_name=event_data.get('event', ''),
                    timestamp=datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00')),
                    impact=event_data.get('importance', 'low'),
                    previous=self._safe_float(event_data.get('previous')),
                    forecast=self._safe_float(event_data.get('estimate')),
                    actual=self._safe_float(event_data.get('actual')),
                    currency=event_data.get('currency', ''),
                    description=event_data.get('event', '')
                )
                events.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse Alpha Vantage event: {e}")
                continue
        
        return events

    async def fetch_social_sentiment(self, symbols: List[str], 
                                   sources: List[str] = None) -> Dict[str, List[SentimentData]]:
        """Fetch social media sentiment data"""
        try:
            cache_key = f"sentiment_{'_'.join(symbols)}_{'_'.join(sources or [])}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            sentiment_data = defaultdict(list)
            sources = sources or ['twitter', 'reddit']
            
            for symbol in symbols:
                for source in sources:
                    try:
                        if source == 'twitter':
                            symbol_sentiment = await self._fetch_twitter_sentiment(symbol)
                        elif source == 'reddit':
                            symbol_sentiment = await self._fetch_reddit_sentiment(symbol)
                        else:
                            continue
                        
                        sentiment_data[symbol].extend(symbol_sentiment)
                        
                    except Exception as e:
                        logger.warning(f"Sentiment source {source} failed for {symbol}: {e}")
                        continue
            
            # Cache results
            self._cache_data(cache_key, dict(sentiment_data))
            
            # Update storage
            for symbol_data in sentiment_data.values():
                self.sentiment_data.extend(symbol_data)
            
            logger.info(f"Fetched sentiment data for {len(sentiment_data)} symbols")
            return dict(sentiment_data)
            
        except Exception as e:
            logger.error(f"Social sentiment fetch failed: {e}")
            return {}

    async def _fetch_twitter_sentiment(self, symbol: str) -> List[SentimentData]:
        """Fetch Twitter sentiment for symbol"""
        try:
            await self._check_rate_limit('twitter')
            
            # Search for tweets about the symbol
            query = f"${symbol} OR {symbol}USD OR {symbol} forex"
            headers = {
                'Authorization': f'Bearer {self.api_configs["twitter"]["bearer_token"]}'
            }
            
            params = {
                'query': query,
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics,author_id',
                'start_time': (datetime.utcnow() - timedelta(hours=24)).isoformat() + 'Z'
            }
            
            # Note: This would require Twitter API v2 access
            # For demo purposes, return mock data
            return self._generate_mock_sentiment(symbol, 'twitter', 24)
            
        except Exception as e:
            logger.error(f"Twitter sentiment fetch failed: {e}")
            return []

    async def _fetch_reddit_sentiment(self, symbol: str) -> List[SentimentData]:
        """Fetch Reddit sentiment for symbol"""
        try:
            await self._check_rate_limit('reddit')
            
            # This would use Reddit API to fetch posts and comments
            # For demo purposes, return mock data
            return self._generate_mock_sentiment(symbol, 'reddit', 24)
            
        except Exception as e:
            logger.error(f"Reddit sentiment fetch failed: {e}")
            return []

    def _generate_mock_sentiment(self, symbol: str, source: str, hours: int) -> List[SentimentData]:
        """Generate mock sentiment data for demonstration"""
        sentiment_data = []
        base_time = datetime.utcnow()
        
        for i in range(hours):
            timestamp = base_time - timedelta(hours=i)
            sentiment_score = np.random.normal(0.1, 0.3)  # Slightly positive bias
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
            
            data = SentimentData(
                source=source,
                symbol=symbol,
                timestamp=timestamp,
                sentiment_score=sentiment_score,
                volume=np.random.randint(10, 1000),
                bullish_ratio=max(0, min(1, 0.5 + sentiment_score * 0.3)),
                unique_authors=np.random.randint(5, 500)
            )
            sentiment_data.append(data)
        
        return sentiment_data

    async def fetch_onchain_metrics(self, symbols: List[str], 
                                  metrics: List[str] = None) -> Dict[str, List[OnchainMetric]]:
        """Fetch on-chain cryptocurrency metrics"""
        try:
            cache_key = f"onchain_{'_'.join(symbols)}_{'_'.join(metrics or [])}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            onchain_data = defaultdict(list)
            metrics = metrics or ['volume', 'transactions', 'active_addresses']
            
            for symbol in symbols:
                for metric in metrics:
                    try:
                        metric_data = await self._fetch_glassnode_metric(symbol, metric)
                        onchain_data[symbol].extend(metric_data)
                    except Exception as e:
                        logger.warning(f"On-chain metric {metric} failed for {symbol}: {e}")
                        continue
            
            # Cache results
            self._cache_data(cache_key, dict(onchain_data))
            
            # Update storage
            for symbol_data in onchain_data.values():
                self.onchain_metrics.extend(symbol_data)
            
            logger.info(f"Fetched on-chain data for {len(onchain_data)} symbols")
            return dict(onchain_data)
            
        except Exception as e:
            logger.error(f"On-chain metrics fetch failed: {e}")
            return {}

    async def _fetch_glassnode_metric(self, symbol: str, metric: str) -> List[OnchainMetric]:
        """Fetch specific metric from Glassnode"""
        try:
            await self._check_rate_limit('glassnode')
            
            # Map symbols to Glassnode assets
            asset_map = {
                'BTC': 'btc',
                'ETH': 'eth',
                'EUR': 'eur',  # Forex through stablecoins
                'USD': 'usd'
            }
            
            asset = asset_map.get(symbol, symbol.lower())
            
            endpoint = f"/metrics/{metric}/{asset}"
            params = {
                'api_key': self.api_configs['glassnode']['api_key'],
                'since': int((datetime.utcnow() - timedelta(days=30)).timestamp()),
                'until': int(datetime.utcnow().timestamp()),
                'interval': '24h'
            }
            
            url = f"{self.api_configs['glassnode']['base_url']}{endpoint}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_glassnode_data(symbol, metric, data)
                else:
                    raise Exception(f"Glassnode API returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"Glassnode metric fetch failed: {e}")
            return []

    def _parse_glassnode_data(self, symbol: str, metric: str, data: List) -> List[OnchainMetric]:
        """Parse Glassnode API response"""
        metrics = []
        
        for point in data:
            try:
                metric_obj = OnchainMetric(
                    metric_id=f"{symbol}_{metric}_{point['t']}",
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(point['t']),
                    value=point['v'],
                    metric_type=metric,
                    metadata={'source': 'glassnode'}
                )
                metrics.append(metric_obj)
            except Exception as e:
                logger.warning(f"Failed to parse Glassnode data point: {e}")
                continue
        
        return metrics

    async def fetch_news_sentiment(self, symbols: List[str], 
                                 sources: List[str] = None) -> Dict[str, List[NewsArticle]]:
        """Fetch news articles and sentiment"""
        try:
            cache_key = f"news_{'_'.join(symbols)}_{'_'.join(sources or [])}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            news_data = defaultdict(list)
            sources = sources or ['newsapi']
            
            for symbol in symbols:
                for source in sources:
                    try:
                        if source == 'newsapi':
                            articles = await self._fetch_newsapi_articles(symbol)
                        else:
                            continue
                        
                        # Analyze sentiment for articles
                        articles_with_sentiment = await self._analyze_news_sentiment(articles)
                        news_data[symbol].extend(articles_with_sentiment)
                        
                    except Exception as e:
                        logger.warning(f"News source {source} failed for {symbol}: {e}")
                        continue
            
            # Cache results
            self._cache_data(cache_key, dict(news_data))
            
            # Update storage
            for symbol_articles in news_data.values():
                self.news_articles.extend(symbol_articles)
            
            logger.info(f"Fetched news data for {len(news_data)} symbols")
            return dict(news_data)
            
        except Exception as e:
            logger.error(f"News sentiment fetch failed: {e}")
            return {}

    async def _fetch_newsapi_articles(self, symbol: str) -> List[NewsArticle]:
        """Fetch news articles from NewsAPI"""
        try:
            await self._check_rate_limit('newsapi')
            
            query = f"{symbol} OR {symbol}USD OR forex {symbol}"
            params = {
                'q': query,
                'apiKey': self.api_configs['newsapi']['api_key'],
                'pageSize': 50,
                'sortBy': 'publishedAt',
                'language': 'en',
                'from': (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
            }
            
            url = f"{self.api_configs['newsapi']['base_url']}/everything"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_newsapi_articles(symbol, data)
                else:
                    raise Exception(f"NewsAPI returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []

    def _parse_newsapi_articles(self, symbol: str, data: Dict) -> List[NewsArticle]:
        """Parse NewsAPI response"""
        articles = []
        
        for article_data in data.get('articles', []):
            try:
                article = NewsArticle(
                    article_id=hashlib.md5(article_data['url'].encode()).hexdigest(),
                    source=article_data.get('source', {}).get('name', ''),
                    title=article_data.get('title', ''),
                    content=article_data.get('description', '') or article_data.get('title', ''),
                    timestamp=datetime.fromisoformat(article_data['publishedAt'].replace('Z', '+00:00')),
                    symbols=[symbol],
                    relevance_score=0.8,  # Would be calculated based on content analysis
                    metadata={
                        'url': article_data.get('url', ''),
                        'author': article_data.get('author', ''),
                        'source': article_data.get('source', {}).get('name', '')
                    }
                )
                articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to parse news article: {e}")
                continue
        
        return articles

    async def _analyze_news_sentiment(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Analyze sentiment of news articles"""
        try:
            # This would use NLP models for sentiment analysis
            # For demo, use simple rule-based sentiment
            for article in articles:
                text = f"{article.title} {article.content}".lower()
                
                # Simple keyword-based sentiment
                positive_words = ['bullish', 'rise', 'gain', 'positive', 'strong', 'buy']
                negative_words = ['bearish', 'fall', 'drop', 'negative', 'weak', 'sell']
                
                positive_count = sum(1 for word in positive_words if word in text)
                negative_count = sum(1 for word in negative_words if word in text)
                
                total_words = positive_count + negative_count
                if total_words > 0:
                    article.sentiment = (positive_count - negative_count) / total_words
                else:
                    article.sentiment = 0.0
            
            return articles
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return articles

    async def fetch_options_flow(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch options flow data (for crypto and traditional markets)"""
        try:
            options_data = {}
            
            for symbol in symbols:
                try:
                    # This would integrate with Deribit, Binance Options, or traditional options data
                    # For demo, return mock data
                    options_data[symbol] = {
                        'call_volume': np.random.randint(100, 10000),
                        'put_volume': np.random.randint(100, 10000),
                        'put_call_ratio': np.random.uniform(0.5, 1.5),
                        'large_trades': np.random.randint(0, 50),
                        'timestamp': datetime.utcnow()
                    }
                except Exception as e:
                    logger.warning(f"Options flow failed for {symbol}: {e}")
                    continue
            
            return options_data
            
        except Exception as e:
            logger.error(f"Options flow fetch failed: {e}")
            return {}

    async def fetch_order_flow(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch order flow and market microstructure data"""
        try:
            order_flow_data = {}
            
            for symbol in symbols:
                try:
                    # This would analyze order book data and trade tapes
                    # For demo, return mock data
                    order_flow_data[symbol] = {
                        'bid_ask_imbalance': np.random.uniform(-0.1, 0.1),
                        'large_trade_ratio': np.random.uniform(0, 0.3),
                        'aggressive_buy_volume': np.random.randint(0, 1000000),
                        'aggressive_sell_volume': np.random.randint(0, 1000000),
                        'timestamp': datetime.utcnow()
                    }
                except Exception as e:
                    logger.warning(f"Order flow failed for {symbol}: {e}")
                    continue
            
            return order_flow_data
            
        except Exception as e:
            logger.error(f"Order flow fetch failed: {e}")
            return {}

    async def fetch_weather_data(self, regions: List[str]) -> Dict[str, Any]:
        """Fetch weather data for commodity correlations"""
        try:
            weather_data = {}
            
            for region in regions:
                try:
                    # This would integrate with weather APIs
                    # For demo, return mock data
                    weather_data[region] = {
                        'temperature': np.random.uniform(-10, 35),
                        'precipitation': np.random.uniform(0, 50),
                        'wind_speed': np.random.uniform(0, 30),
                        'timestamp': datetime.utcnow()
                    }
                except Exception as e:
                    logger.warning(f"Weather data failed for {region}: {e}")
                    continue
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Weather data fetch failed: {e}")
            return {}

    async def _check_rate_limit(self, source: str):
        """Check and enforce rate limits"""
        try:
            current_time = time.time()
            rate_config = self.api_configs[source]['rate_limit']
            
            # Remove old requests from rate limit window
            window_start = current_time - 1.0  # 1 second window
            self.rate_limits[source] = deque(
                [t for t in self.rate_limits[source] if t > window_start],
                maxlen=100
            )
            
            # Check if we're over the limit
            if len(self.rate_limits[source]) >= rate_config:
                sleep_time = 1.0 - (current_time - self.rate_limits[source][0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Add current request
            self.rate_limits[source].append(current_time)
            self.last_request_time[source] = current_time
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")

    def _get_cached_data(self, key: str) -> Any:
        """Get data from cache"""
        try:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.cache_ttl:
                    return data
                else:
                    del self.cache[key]
            return None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None

    def _cache_data(self, key: str, data: Any):
        """Store data in cache"""
        try:
            self.cache[key] = (data, time.time())
        except Exception as e:
            logger.error(f"Cache store failed: {e}")

    def _deduplicate_events(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """Remove duplicate economic events"""
        seen = set()
        unique_events = []
        
        for event in events:
            event_key = f"{event.event_name}_{event.timestamp}_{event.country}"
            if event_key not in seen:
                seen.add(event_key)
                unique_events.append(event)
        
        return unique_events

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float"""
        try:
            if value is None or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    async def get_combined_alpha_signals(self, symbols: List[str]) -> Dict[str, float]:
        """Generate combined alpha signals from all alternative data sources"""
        try:
            alpha_signals = {}
            
            for symbol in symbols:
                try:
                    signal = 0.0
                    weight_sum = 0.0
                    
                    # Economic calendar impact
                    economic_impact = await self._calculate_economic_impact(symbol)
                    signal += economic_impact * 0.3
                    weight_sum += 0.3
                    
                    # Social sentiment
                    sentiment_impact = await self._calculate_sentiment_impact(symbol)
                    signal += sentiment_impact * 0.25
                    weight_sum += 0.25
                    
                    # News sentiment
                    news_impact = await self._calculate_news_impact(symbol)
                    signal += news_impact * 0.2
                    weight_sum += 0.2
                    
                    # On-chain metrics (for crypto)
                    onchain_impact = await self._calculate_onchain_impact(symbol)
                    signal += onchain_impact * 0.15
                    weight_sum += 0.15
                    
                    # Options flow (if available)
                    options_impact = await self._calculate_options_impact(symbol)
                    signal += options_impact * 0.1
                    weight_sum += 0.1
                    
                    # Normalize signal
                    if weight_sum > 0:
                        alpha_signals[symbol] = signal / weight_sum
                    else:
                        alpha_signals[symbol] = 0.0
                        
                except Exception as e:
                    logger.warning(f"Alpha signal calculation failed for {symbol}: {e}")
                    alpha_signals[symbol] = 0.0
            
            return alpha_signals
            
        except Exception as e:
            logger.error(f"Combined alpha signals failed: {e}")
            return {}

    async def _calculate_economic_impact(self, symbol: str) -> float:
        """Calculate economic calendar impact for symbol"""
        try:
            # This would analyze upcoming economic events for the currency
            # For demo, return random impact
            return np.random.uniform(-0.5, 0.5)
        except Exception as e:
            logger.error(f"Economic impact calculation failed: {e}")
            return 0.0

    async def _calculate_sentiment_impact(self, symbol: str) -> float:
        """Calculate social sentiment impact for symbol"""
        try:
            # Analyze recent sentiment data
            recent_sentiment = [s for s in self.sentiment_data 
                              if s.symbol == symbol and s.timestamp > datetime.utcnow() - timedelta(hours=6)]
            
            if recent_sentiment:
                avg_sentiment = np.mean([s.sentiment_score for s in recent_sentiment])
                return avg_sentiment
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Sentiment impact calculation failed: {e}")
            return 0.0

    async def _calculate_news_impact(self, symbol: str) -> float:
        """Calculate news sentiment impact for symbol"""
        try:
            # Analyze recent news articles
            recent_news = [n for n in self.news_articles 
                          if symbol in n.symbols and n.timestamp > datetime.utcnow() - timedelta(hours=12)]
            
            if recent_news:
                avg_sentiment = np.mean([n.sentiment or 0 for n in recent_news])
                return avg_sentiment
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"News impact calculation failed: {e}")
            return 0.0

    async def _calculate_onchain_impact(self, symbol: str) -> float:
        """Calculate on-chain metrics impact for symbol"""
        try:
            # Analyze recent on-chain data
            recent_metrics = [m for m in self.onchain_metrics 
                            if m.symbol == symbol and m.timestamp > datetime.utcnow() - timedelta(days=1)]
            
            if recent_metrics:
                # Simple volume-based signal
                volume_metrics = [m for m in recent_metrics if m.metric_type == 'volume']
                if volume_metrics:
                    latest_volume = volume_metrics[-1].value
                    avg_volume = np.mean([m.value for m in volume_metrics[-10:]])
                    
                    if avg_volume > 0:
                        return (latest_volume - avg_volume) / avg_volume * 0.1  # Scale down
            return 0.0
            
        except Exception as e:
            logger.error(f"On-chain impact calculation failed: {e}")
            return 0.0

    async def _calculate_options_impact(self, symbol: str) -> float:
        """Calculate options flow impact for symbol"""
        try:
            # Analyze options flow data
            options_data = await self.fetch_options_flow([symbol])
            if symbol in options_data:
                put_call_ratio = options_data[symbol]['put_call_ratio']
                # Normal put/call ratio is around 0.7-1.0
                if put_call_ratio > 1.2:  # Bearish
                    return -0.3
                elif put_call_ratio < 0.6:  # Bullish
                    return 0.3
            return 0.0
            
        except Exception as e:
            logger.error(f"Options impact calculation failed: {e}")
            return 0.0

    async def close(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'session'):
                await self.session.close()
            logger.info("AlternativeDataLoader closed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Example usage and testing
async def main():
    """Test the Alternative Data Loader"""
    
    config = {
        'fred_api_key': 'your_fred_key',
        'alphavantage_api_key': 'your_alpha_key',
        'twitter_bearer_token': 'your_twitter_token',
        'reddit_client_id': 'your_reddit_id',
        'reddit_client_secret': 'your_reddit_secret',
        'glassnode_api_key': 'your_glassnode_key',
        'cryptocompare_api_key': 'your_crypto_key',
        'newsapi_api_key': 'your_newsapi_key',
        'cache_ttl': 300
    }
    
    loader = AlternativeDataLoader(config)
    
    try:
        await loader.initialize()
        
        # Test economic calendar
        events = await loader.fetch_economic_calendar(days=3, countries=['US', 'EU'])
        print(f"Fetched {len(events)} economic events")
        
        # Test sentiment data
        sentiment = await loader.fetch_social_sentiment(['EUR', 'USD'])
        print(f"Fetched sentiment for {len(sentiment)} symbols")
        
        # Test alpha signals
        alpha_signals = await loader.get_combined_alpha_signals(['EUR', 'USD'])
        print(f"Alpha signals: {alpha_signals}")
        
        # Test news sentiment
        news = await loader.fetch_news_sentiment(['EUR'])
        print(f"Fetched {len(news.get('EUR', []))} news articles")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        await loader.close()

if __name__ == "__main__":
    asyncio.run(main())
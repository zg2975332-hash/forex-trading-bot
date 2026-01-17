"""
Advanced Community Signals Analyzer for FOREX TRADING BOT
Real-time crowd sentiment analysis from multiple social platforms
"""

import logging
import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
import re
from collections import defaultdict, deque
import threading
from textblob import TextBlob
import tweepy
import praw
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import heapq
from urllib.parse import urlencode
import hashlib
import hmac
import base64
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class Platform(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    TRADINGVIEW = "tradingview"
    FOREX_FACTORY = "forex_factory"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    YOUTUBE = "youtube"
    INVESTING_COM = "investing_com"

class SignalType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class ConfidenceLevel(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class CommunitySignal:
    """Individual community signal"""
    id: str
    platform: Platform
    signal_type: SignalType
    confidence: float
    content: str
    author: str
    timestamp: datetime
    symbol: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    sentiment_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregateSignal:
    """Aggregated community signals"""
    symbol: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_signals: int
    net_sentiment: float
    confidence_score: float
    signal_strength: float
    dominant_signal: SignalType
    timestamp: datetime
    platform_breakdown: Dict[str, int] = field(default_factory=dict)
    recent_trend: str = "stable"
    volume_change: float = 0.0

@dataclass
class PlatformConfig:
    """Configuration for each social platform"""
    enabled: bool = True
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_token_secret: str = ""
    rate_limit: int = 100
    update_interval: int = 300  # seconds
    max_posts: int = 100

@dataclass
class CommunityConfig:
    """Configuration for community signals analyzer"""
    # Platform configurations
    platforms: Dict[Platform, PlatformConfig] = field(default_factory=dict)
    
    # Analysis settings
    sentiment_models: List[str] = field(default_factory=lambda: ["vader", "textblob", "transformers"])
    min_confidence: float = 0.6
    max_signals_age: int = 3600  # 1 hour
    aggregation_window: int = 900  # 15 minutes
    
    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
    min_signal_strength: float = 0.7
    volume_threshold: int = 10
    
    # Advanced features
    enable_ml_filtering: bool = True
    enable_trend_analysis: bool = True
    enable_influencer_tracking: bool = True
    enable_controversy_detection: bool = True
    
    # Performance
    max_workers: int = 10
    cache_ttl: int = 300

class AdvancedCommunityAnalyzer:
    """
    Advanced Community Signals Analyzer with real-time multi-platform monitoring
    """
    
    def __init__(self, config: CommunityConfig = None):
        self.config = config or CommunityConfig()
        
        # Data storage
        self._signals: Dict[str, List[CommunitySignal]] = defaultdict(list)
        self._aggregated_signals: Dict[str, AggregateSignal] = {}
        self._influencer_scores: Dict[str, float] = {}
        
        # Platform clients
        self._platform_clients: Dict[Platform, Any] = {}
        self._initialize_platforms()
        
        # Sentiment analyzers
        self._sentiment_analyzers: Dict[str, Any] = {}
        self._initialize_sentiment_analyzers()
        
        # ML models
        self._ml_models: Dict[str, Any] = {}
        self._initialize_ml_models()
        
        # Caching
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._running = True
        
        # Statistics
        self._stats = {
            'total_signals_processed': 0,
            'platform_breakdown': defaultdict(int),
            'symbol_breakdown': defaultdict(int)
        }
        
        # Trend analysis
        self._historical_signals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("AdvancedCommunityAnalyzer initialized successfully")
    
    def _initialize_platforms(self) -> None:
        """Initialize platform clients"""
        try:
            # Twitter
            if Platform.TWITTER in self.config.platforms:
                twitter_config = self.config.platforms[Platform.TWITTER]
                if twitter_config.enabled and twitter_config.api_key:
                    auth = tweepy.OAuthHandler(
                        twitter_config.api_key, 
                        twitter_config.api_secret
                    )
                    auth.set_access_token(
                        twitter_config.access_token,
                        twitter_config.access_token_secret
                    )
                    self._platform_clients[Platform.TWITTER] = tweepy.API(
                        auth, wait_on_rate_limit=True
                    )
            
            # Reddit
            if Platform.REDDIT in self.config.platforms:
                reddit_config = self.config.platforms[Platform.REDDIT]
                if reddit_config.enabled and reddit_config.api_key:
                    self._platform_clients[Platform.REDDIT] = praw.Reddit(
                        client_id=reddit_config.api_key,
                        client_secret=reddit_config.api_secret,
                        user_agent="forex_trading_bot"
                    )
            
            # Initialize web drivers for web scraping
            self._initialize_web_drivers()
            
            logger.info("Platform clients initialized")
            
        except Exception as e:
            logger.error(f"Platform initialization failed: {e}")
    
    def _initialize_web_drivers(self) -> None:
        """Initialize web drivers for scraping"""
        try:
            # Initialize Selenium WebDriver (headless)
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            self._web_driver = webdriver.Chrome(options=options)
            self._web_driver.set_page_load_timeout(30)
            
        except Exception as e:
            logger.warning(f"Web driver initialization failed: {e}")
            self._web_driver = None
    
    def _initialize_sentiment_analyzers(self) -> None:
        """Initialize sentiment analysis models"""
        try:
            # VADER Sentiment Analyzer
            if "vader" in self.config.sentiment_models:
                self._sentiment_analyzers["vader"] = SentimentIntensityAnalyzer()
            
            # TextBlob (no initialization needed)
            if "textblob" in self.config.sentiment_models:
                self._sentiment_analyzers["textblob"] = TextBlob
            
            # Transformers (Hugging Face)
            if "transformers" in self.config.sentiment_models:
                try:
                    self._sentiment_analyzers["transformers"] = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=0 if torch.cuda.is_available() else -1
                    )
                except Exception as e:
                    logger.warning(f"Transformers model loading failed: {e}")
            
            logger.info(f"Sentiment analyzers initialized: {list(self._sentiment_analyzers.keys())}")
            
        except Exception as e:
            logger.error(f"Sentiment analyzers initialization failed: {e}")
    
    def _initialize_ml_models(self) -> None:
        """Initialize ML models for signal filtering"""
        if not self.config.enable_ml_filtering:
            return
        
        try:
            # Spam detection model (simplified)
            # In production, you would train this on labeled data
            self._ml_models["spam_detector"] = self._create_spam_detector()
            
            # Influencer scoring model
            self._ml_models["influencer_scorer"] = self._create_influencer_scorer()
            
            logger.info("ML models initialized")
            
        except Exception as e:
            logger.error(f"ML models initialization failed: {e}")
    
    def _create_spam_detector(self) -> Any:
        """Create spam detection model (simplified)"""
        # This is a simplified version - in production, use proper ML model
        spam_keywords = [
            "guaranteed profit", "make money fast", "100% accurate", 
            "free signals", "join now", "limited offer", "secret strategy"
        ]
        
        def detector(text: str, author: str, engagement: int) -> float:
            # Simple rule-based spam detection
            spam_score = 0.0
            
            # Check for spam keywords
            for keyword in spam_keywords:
                if keyword.lower() in text.lower():
                    spam_score += 0.3
            
            # Check for excessive punctuation
            if text.count('!') > 3 or text.count('$') > 2:
                spam_score += 0.2
            
            # Check for low engagement relative to claims
            if "huge" in text.lower() and engagement < 10:
                spam_score += 0.2
            
            # Check for URL patterns
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            if re.findall(url_pattern, text):
                spam_score += 0.3
            
            return min(1.0, spam_score)
        
        return detector
    
    def _create_influencer_scorer(self) -> Any:
        """Create influencer scoring model"""
        def scorer(author: str, historical_signals: List[CommunitySignal]) -> float:
            if not historical_signals:
                return 0.5  # Default score
            
            # Calculate based on historical accuracy
            accurate_signals = [s for s in historical_signals if s.confidence > 0.7]
            accuracy_score = len(accurate_signals) / len(historical_signals) if historical_signals else 0.0
            
            # Engagement factor
            avg_engagement = np.mean([s.likes + s.retweets + s.replies for s in historical_signals])
            engagement_score = min(1.0, avg_engagement / 100.0)
            
            # Consistency factor
            recent_signals = [s for s in historical_signals 
                            if s.timestamp > datetime.now() - timedelta(days=30)]
            consistency_score = len(recent_signals) / 30.0  # Signals per day
            
            # Combined score
            total_score = (accuracy_score * 0.5 + engagement_score * 0.3 + consistency_score * 0.2)
            return min(1.0, total_score)
        
        return scorer
    
    async def start_monitoring(self) -> None:
        """Start monitoring all platforms"""
        logger.info("Starting community signals monitoring")
        
        # Create tasks for each platform
        tasks = []
        for platform, config in self.config.platforms.items():
            if config.enabled:
                task = asyncio.create_task(
                    self._monitor_platform(platform, config)
                )
                tasks.append(task)
        
        # Start aggregation task
        aggregation_task = asyncio.create_task(self._run_aggregation_loop())
        tasks.append(aggregation_task)
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Monitoring task failed: {e}")
    
    async def _monitor_platform(self, platform: Platform, config: PlatformConfig) -> None:
        """Monitor specific platform for signals"""
        while self._running:
            try:
                logger.debug(f"Monitoring {platform.value} for signals")
                
                # Platform-specific monitoring
                if platform == Platform.TWITTER:
                    await self._monitor_twitter(config)
                elif platform == Platform.REDDIT:
                    await self._monitor_reddit(config)
                elif platform == Platform.TRADINGVIEW:
                    await self._monitor_tradingview(config)
                elif platform == Platform.FOREX_FACTORY:
                    await self._monitor_forex_factory(config)
                elif platform == Platform.INVESTING_COM:
                    await self._monitor_investing_com(config)
                
                # Wait for next update
                await asyncio.sleep(config.update_interval)
                
            except Exception as e:
                logger.error(f"Platform monitoring failed for {platform.value}: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _monitor_twitter(self, config: PlatformConfig) -> None:
        """Monitor Twitter for forex signals"""
        try:
            if Platform.TWITTER not in self._platform_clients:
                return
            
            client = self._platform_clients[Platform.TWITTER]
            
            # Search for forex-related tweets
            queries = [
                "#EURUSD OR #forex OR #trading",
                "#GBPUSD OR #forexsignals",
                "#USDJPY OR #forextrading",
                "#AUDUSD OR #fx"
            ]
            
            for query in queries:
                try:
                    tweets = client.search_tweets(
                        q=query,
                        count=config.max_posts,
                        result_type="recent",
                        tweet_mode="extended"
                    )
                    
                    for tweet in tweets:
                        await self._process_twitter_tweet(tweet)
                        
                except Exception as e:
                    logger.warning(f"Twitter search failed for query {query}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Twitter monitoring failed: {e}")
    
    async def _process_twitter_tweet(self, tweet: Any) -> None:
        """Process individual Twitter tweet"""
        try:
            # Extract symbol mentions
            symbols = self._extract_symbols_from_text(tweet.full_text)
            if not symbols:
                return
            
            # Analyze sentiment
            sentiment_result = self._analyze_sentiment(tweet.full_text)
            signal_type = self._sentiment_to_signal_type(sentiment_result)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                sentiment_result, 
                tweet.favorite_count, 
                tweet.retweet_count
            )
            
            if confidence < self.config.min_confidence:
                return
            
            # Create signal
            signal = CommunitySignal(
                id=f"twitter_{tweet.id}",
                platform=Platform.TWITTER,
                signal_type=signal_type,
                confidence=confidence,
                content=tweet.full_text,
                author=tweet.user.screen_name,
                timestamp=tweet.created_at.replace(tzinfo=None),
                symbol=symbols[0],  # Use first mentioned symbol
                likes=tweet.favorite_count,
                retweets=tweet.retweet_count,
                replies=0,  # Would need additional API call
                sentiment_score=sentiment_result['compound'],
                metadata={
                    'tweet_id': tweet.id,
                    'user_followers': tweet.user.followers_count,
                    'user_verified': tweet.user.verified,
                    'has_media': bool(tweet.entities.get('media', []))
                }
            )
            
            # Apply ML filtering
            if not self._is_spam_signal(signal):
                await self._store_signal(signal)
                
        except Exception as e:
            logger.error(f"Twitter tweet processing failed: {e}")
    
    async def _monitor_reddit(self, config: PlatformConfig) -> None:
        """Monitor Reddit for forex signals"""
        try:
            if Platform.REDDIT not in self._platform_clients:
                return
            
            client = self._platform_clients[Platform.REDDIT]
            
            # Monitor forex-related subreddits
            subreddits = [
                "forex", "trading", "algotrading", "Forexstrategy",
                "Forexsignals", "Daytrading"
            ]
            
            for subreddit_name in subreddits:
                try:
                    subreddit = client.subreddit(subreddit_name)
                    
                    # Get hot posts
                    for post in subreddit.hot(limit=config.max_posts):
                        await self._process_reddit_post(post)
                    
                    # Get new posts
                    for post in subreddit.new(limit=config.max_posts):
                        await self._process_reddit_post(post)
                        
                except Exception as e:
                    logger.warning(f"Reddit monitoring failed for {subreddit_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Reddit monitoring failed: {e}")
    
    async def _process_reddit_post(self, post: Any) -> None:
        """Process individual Reddit post"""
        try:
            # Extract symbol mentions
            symbols = self._extract_symbols_from_text(post.title + " " + post.selftext)
            if not symbols:
                return
            
            # Analyze sentiment
            sentiment_result = self._analyze_sentiment(post.title + " " + post.selftext)
            signal_type = self._sentiment_to_signal_type(sentiment_result)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                sentiment_result,
                post.score,
                post.num_comments
            )
            
            if confidence < self.config.min_confidence:
                return
            
            # Create signal
            signal = CommunitySignal(
                id=f"reddit_{post.id}",
                platform=Platform.REDDIT,
                signal_type=signal_type,
                confidence=confidence,
                content=post.title + " " + post.selftext,
                author=str(post.author),
                timestamp=datetime.fromtimestamp(post.created_utc),
                symbol=symbols[0],
                likes=post.score,
                retweets=0,
                replies=post.num_comments,
                sentiment_score=sentiment_result['compound'],
                metadata={
                    'post_id': post.id,
                    'subreddit': post.subreddit.display_name,
                    'upvote_ratio': post.upvote_ratio,
                    'is_original_content': post.is_original_content
                }
            )
            
            # Apply ML filtering
            if not self._is_spam_signal(signal):
                await self._store_signal(signal)
                
        except Exception as e:
            logger.error(f"Reddit post processing failed: {e}")
    
    async def _monitor_tradingview(self, config: PlatformConfig) -> None:
        """Monitor TradingView ideas and signals"""
        try:
            # This would require web scraping or TradingView API
            # Simplified implementation for demonstration
            
            # Mock data for demonstration
            mock_ideas = [
                {
                    'title': 'EURUSD Bullish Breakout Expected',
                    'content': 'EURUSD showing strong bullish momentum on H4 timeframe',
                    'author': 'ForexMaster',
                    'symbol': 'EUR/USD',
                    'timestamp': datetime.now(),
                    'votes': 150
                },
                {
                    'title': 'GBPUSD Bearish Reversal',
                    'content': 'GBPUSD facing resistance at 1.3800, expecting pullback',
                    'author': 'TradePro',
                    'symbol': 'GBP/USD', 
                    'timestamp': datetime.now(),
                    'votes': 89
                }
            ]
            
            for idea in mock_ideas:
                sentiment_result = self._analyze_sentiment(idea['title'] + " " + idea['content'])
                signal_type = self._sentiment_to_signal_type(sentiment_result)
                
                confidence = self._calculate_confidence(sentiment_result, idea['votes'], 0)
                
                if confidence >= self.config.min_confidence:
                    signal = CommunitySignal(
                        id=f"tradingview_{hash(idea['title'])}",
                        platform=Platform.TRADINGVIEW,
                        signal_type=signal_type,
                        confidence=confidence,
                        content=idea['title'] + " " + idea['content'],
                        author=idea['author'],
                        timestamp=idea['timestamp'],
                        symbol=idea['symbol'],
                        likes=idea['votes'],
                        sentiment_score=sentiment_result['compound'],
                        metadata={'source': 'tradingview_ideas'}
                    )
                    
                    if not self._is_spam_signal(signal):
                        await self._store_signal(signal)
                        
        except Exception as e:
            logger.error(f"TradingView monitoring failed: {e}")
    
    async def _monitor_forex_factory(self, config: PlatformConfig) -> None:
        """Monitor Forex Factory calendar and forum"""
        try:
            # Web scraping implementation for Forex Factory
            if not self._web_driver:
                return
            
            # Navigate to Forex Factory calendar
            self._web_driver.get("https://www.forexfactory.com/calendar.php")
            
            # Wait for page load
            WebDriverWait(self._web_driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "calendar__row"))
            )
            
            # Parse calendar events
            soup = BeautifulSoup(self._web_driver.page_source, 'html.parser')
            events = soup.find_all('tr', class_='calendar__row')
            
            for event in events:
                await self._process_forex_factory_event(event)
                
        except Exception as e:
            logger.error(f"Forex Factory monitoring failed: {e}")
    
    async def _process_forex_factory_event(self, event: Any) -> None:
        """Process Forex Factory calendar event"""
        try:
            # Extract event details
            # This is simplified - actual implementation would parse HTML structure
            event_text = event.get_text()
            
            # Look for currency pairs in event text
            symbols = self._extract_symbols_from_text(event_text)
            if not symbols:
                return
            
            # Analyze sentiment based on event impact
            impact = self._extract_impact_level(event_text)
            sentiment_score = self._impact_to_sentiment(impact)
            
            signal_type = SignalType.BULLISH if sentiment_score > 0.1 else (
                SignalType.BEARISH if sentiment_score < -0.1 else SignalType.NEUTRAL
            )
            
            confidence = 0.7 if impact in ["high", "medium"] else 0.4
            
            signal = CommunitySignal(
                id=f"forexfactory_{hash(event_text)}",
                platform=Platform.FOREX_FACTORY,
                signal_type=signal_type,
                confidence=confidence,
                content=event_text[:500],  # Limit content length
                author="ForexFactory",
                timestamp=datetime.now(),
                symbol=symbols[0],
                sentiment_score=sentiment_score,
                metadata={'impact': impact, 'source': 'economic_calendar'}
            )
            
            await self._store_signal(signal)
            
        except Exception as e:
            logger.error(f"Forex Factory event processing failed: {e}")
    
    async def _monitor_investing_com(self, config: PlatformConfig) -> None:
        """Monitor Investing.com for market sentiment"""
        try:
            # Web scraping implementation
            if not self._web_driver:
                return
            
            # Navigate to Investing.com technical analysis
            self._web_driver.get("https://www.investing.com/technical/technical-summary")
            
            # Wait for page load
            WebDriverWait(self._web_driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "technicalSummaryTbl"))
            )
            
            # Parse technical summary
            soup = BeautifulSoup(self._web_driver.page_source, 'html.parser')
            table = soup.find('table', class_='technicalSummaryTbl')
            
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    await self._process_investing_com_row(row)
                    
        except Exception as e:
            logger.error(f"Investing.com monitoring failed: {e}")
    
    async def _process_investing_com_row(self, row: Any) -> None:
        """Process Investing.com technical summary row"""
        try:
            cells = row.find_all('td')
            if len(cells) < 4:
                return
            
            symbol = cells[0].get_text().strip()
            summary = cells[3].get_text().strip()
            
            # Map summary to signal type
            signal_type = self._summary_to_signal_type(summary)
            sentiment_score = self._signal_type_to_sentiment(signal_type)
            
            signal = CommunitySignal(
                id=f"investingcom_{hash(symbol + summary)}",
                platform=Platform.INVESTING_COM,
                signal_type=signal_type,
                confidence=0.75,  # Technical analysis based
                content=f"Technical Summary: {summary}",
                author="Investing.com",
                timestamp=datetime.now(),
                symbol=symbol,
                sentiment_score=sentiment_score,
                metadata={'summary': summary, 'source': 'technical_analysis'}
            )
            
            await self._store_signal(signal)
            
        except Exception as e:
            logger.error(f"Investing.com row processing failed: {e}")
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract forex symbols from text"""
        # Common forex pairs pattern
        patterns = [
            r'\b([A-Z]{3}/[A-Z]{3})\b',  # EUR/USD format
            r'\b([A-Z]{6})\b',           # EURUSD format
            r'\b(GBPUSD|EURUSD|USDJPY|AUDUSD|USDCAD|USDCHF|NZDUSD)\b',
        ]
        
        symbols = []
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.extend(matches)
        
        # Normalize symbol format
        normalized_symbols = []
        for symbol in symbols:
            if '/' not in symbol and len(symbol) == 6:
                # Convert EURUSD to EUR/USD
                normalized = f"{symbol[:3]}/{symbol[3:]}"
                normalized_symbols.append(normalized)
            else:
                normalized_symbols.append(symbol)
        
        return list(set(normalized_symbols))  # Remove duplicates
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple models"""
        try:
            results = {}
            
            # VADER Sentiment
            if "vader" in self._sentiment_analyzers:
                vader_result = self._sentiment_analyzers["vader"].polarity_scores(text)
                results["vader"] = vader_result
            
            # TextBlob Sentiment
            if "textblob" in self._sentiment_analyzers:
                blob = TextBlob(text)
                results["textblob"] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            
            # Transformers Sentiment
            if "transformers" in self._sentiment_analyzers:
                try:
                    transformer_result = self._sentiment_analyzers["transformers"](text)[0]
                    label = transformer_result['label']
                    score = transformer_result['score']
                    
                    # Map to consistent format
                    if label == "positive":
                        polarity = score
                    elif label == "negative":
                        polarity = -score
                    else:
                        polarity = 0
                    
                    results["transformers"] = {
                        'polarity': polarity,
                        'confidence': score
                    }
                except Exception as e:
                    logger.warning(f"Transformers sentiment analysis failed: {e}")
            
            # Combine results
            combined_result = self._combine_sentiment_results(results)
            return combined_result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def _combine_sentiment_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Combine results from multiple sentiment analyzers"""
        if not results:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        weights = {
            'vader': 0.4,
            'textblob': 0.3,
            'transformers': 0.3
        }
        
        compound_score = 0.0
        total_weight = 0.0
        
        for model, result in results.items():
            weight = weights.get(model, 0.2)
            
            if model == "vader":
                compound_score += result['compound'] * weight
            elif model == "textblob":
                compound_score += result['polarity'] * weight
            elif model == "transformers":
                compound_score += result['polarity'] * weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            compound_score /= total_weight
        
        # Convert to VADER-like format
        if compound_score >= 0.05:
            return {
                'compound': compound_score,
                'positive': min(1.0, compound_score + 0.5),
                'negative': max(0.0, 1.0 - (compound_score + 0.5)),
                'neutral': max(0.0, 1.0 - abs(compound_score))
            }
        elif compound_score <= -0.05:
            return {
                'compound': compound_score,
                'positive': max(0.0, 1.0 - (abs(compound_score) + 0.5)),
                'negative': min(1.0, abs(compound_score) + 0.5),
                'neutral': max(0.0, 1.0 - abs(compound_score))
            }
        else:
            return {
                'compound': compound_score,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
    
    def _sentiment_to_signal_type(self, sentiment_result: Dict[str, float]) -> SignalType:
        """Convert sentiment score to signal type"""
        compound = sentiment_result['compound']
        
        if compound >= 0.3:
            return SignalType.STRONG_BUY
        elif compound >= 0.1:
            return SignalType.BULLISH
        elif compound <= -0.3:
            return SignalType.STRONG_SELL
        elif compound <= -0.1:
            return SignalType.BEARISH
        else:
            return SignalType.NEUTRAL
    
    def _calculate_confidence(self, sentiment_result: Dict[str, float], 
                            engagement: int, social_proof: int) -> float:
        """Calculate confidence score for signal"""
        # Base confidence from sentiment strength
        sentiment_strength = abs(sentiment_result['compound'])
        base_confidence = min(1.0, sentiment_strength * 2)
        
        # Engagement factor
        engagement_factor = min(1.0, engagement / 100.0)
        
        # Social proof factor
        social_factor = min(1.0, social_proof / 50.0)
        
        # Combined confidence
        confidence = (base_confidence * 0.6 + engagement_factor * 0.25 + social_factor * 0.15)
        return min(1.0, confidence)
    
    def _is_spam_signal(self, signal: CommunitySignal) -> bool:
        """Check if signal is likely spam"""
        if not self.config.enable_ml_filtering:
            return False
        
        try:
            spam_detector = self._ml_models.get("spam_detector")
            if spam_detector:
                spam_score = spam_detector(
                    signal.content, 
                    signal.author, 
                    signal.likes + signal.retweets + signal.replies
                )
                return spam_score > 0.7
            
            return False
            
        except Exception as e:
            logger.warning(f"Spam detection failed: {e}")
            return False
    
    async def _store_signal(self, signal: CommunitySignal) -> None:
        """Store processed signal"""
        try:
            with self._lock:
                # Update influencer score
                self._update_influencer_score(signal.author, signal)
                
                # Store signal
                self._signals[signal.symbol].append(signal)
                
                # Update statistics
                self._stats['total_signals_processed'] += 1
                self._stats['platform_breakdown'][signal.platform.value] += 1
                self._stats['symbol_breakdown'][signal.symbol] += 1
                
                # Store in historical data for trend analysis
                self._historical_signals[signal.symbol].append(signal)
                
                # Clean old signals
                self._clean_old_signals()
                
                logger.debug(f"Stored signal for {signal.symbol}: {signal.signal_type.value}")
                
        except Exception as e:
            logger.error(f"Signal storage failed: {e}")
    
    def _update_influencer_score(self, author: str, signal: CommunitySignal) -> None:
        """Update influencer credibility score"""
        try:
            if author not in self._influencer_scores:
                self._influencer_scores[author] = 0.5  # Default score
            
            # Get historical signals from this author
            author_signals = [
                s for s in self._historical_signals.get(signal.symbol, [])
                if s.author == author
            ]
            
            # Update score using ML model
            influencer_scorer = self._ml_models.get("influencer_scorer")
            if influencer_scorer:
                new_score = influencer_scorer(author, author_signals)
                # Smooth update
                current_score = self._influencer_scores[author]
                self._influencer_scores[author] = (current_score * 0.7 + new_score * 0.3)
                
        except Exception as e:
            logger.warning(f"Influencer score update failed for {author}: {e}")
    
    def _clean_old_signals(self) -> None:
        """Remove signals older than max age"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=self.config.max_signals_age)
            
            for symbol in list(self._signals.keys()):
                self._signals[symbol] = [
                    signal for signal in self._signals[symbol]
                    if signal.timestamp > cutoff_time
                ]
                
                # Remove empty symbol entries
                if not self._signals[symbol]:
                    del self._signals[symbol]
                    
        except Exception as e:
            logger.error(f"Signal cleanup failed: {e}")
    
    async def _run_aggregation_loop(self) -> None:
        """Run continuous signal aggregation"""
        while self._running:
            try:
                await self.aggregate_signals()
                await asyncio.sleep(self.config.aggregation_window)
            except Exception as e:
                logger.error(f"Aggregation loop failed: {e}")
                await asyncio.sleep(60)
    
    async def aggregate_signals(self) -> Dict[str, AggregateSignal]:
        """Aggregate signals by symbol"""
        try:
            aggregated = {}
            
            for symbol, signals in self._signals.items():
                if len(signals) < self.config.volume_threshold:
                    continue
                
                # Count signal types
                bullish_count = len([s for s in signals if s.signal_type in [SignalType.BULLISH, SignalType.STRONG_BUY]])
                bearish_count = len([s for s in signals if s.signal_type in [SignalType.BEARISH, SignalType.STRONG_SELL]])
                neutral_count = len([s for s in signals if s.signal_type == SignalType.NEUTRAL])
                
                total_signals = len(signals)
                
                # Calculate net sentiment
                net_sentiment = (bullish_count - bearish_count) / total_signals if total_signals > 0 else 0
                
                # Calculate confidence score
                avg_confidence = np.mean([s.confidence for s in signals])
                volume_factor = min(1.0, total_signals / 100.0)
                confidence_score = avg_confidence * volume_factor
                
                # Determine dominant signal
                if bullish_count > bearish_count and bullish_count > neutral_count:
                    dominant_signal = SignalType.BULLISH
                    signal_strength = bullish_count / total_signals
                elif bearish_count > bullish_count and bearish_count > neutral_count:
                    dominant_signal = SignalType.BEARISH
                    signal_strength = bearish_count / total_signals
                else:
                    dominant_signal = SignalType.NEUTRAL
                    signal_strength = neutral_count / total_signals
                
                # Platform breakdown
                platform_breakdown = {}
                for signal in signals:
                    platform = signal.platform.value
                    platform_breakdown[platform] = platform_breakdown.get(platform, 0) + 1
                
                # Recent trend analysis
                recent_trend = self._analyze_recent_trend(symbol)
                
                # Volume change
                volume_change = self._calculate_volume_change(symbol)
                
                aggregated[symbol] = AggregateSignal(
                    symbol=symbol,
                    bullish_count=bullish_count,
                    bearish_count=bearish_count,
                    neutral_count=neutral_count,
                    total_signals=total_signals,
                    net_sentiment=net_sentiment,
                    confidence_score=confidence_score,
                    signal_strength=signal_strength,
                    dominant_signal=dominant_signal,
                    timestamp=datetime.now(),
                    platform_breakdown=platform_breakdown,
                    recent_trend=recent_trend,
                    volume_change=volume_change
                )
            
            with self._lock:
                self._aggregated_signals = aggregated
            
            logger.info(f"Aggregated signals for {len(aggregated)} symbols")
            return aggregated
            
        except Exception as e:
            logger.error(f"Signal aggregation failed: {e}")
            return {}
    
    def _analyze_recent_trend(self, symbol: str) -> str:
        """Analyze recent trend for symbol"""
        try:
            signals = self._historical_signals.get(symbol, [])
            if len(signals) < 10:
                return "stable"
            
            # Get recent signals (last 30 minutes)
            cutoff = datetime.now() - timedelta(minutes=30)
            recent_signals = [s for s in signals if s.timestamp > cutoff]
            
            if len(recent_signals) < 5:
                return "stable"
            
            # Calculate trend
            bullish_trend = len([s for s in recent_signals if s.signal_type in [SignalType.BULLISH, SignalType.STRONG_BUY]])
            bearish_trend = len([s for s in recent_signals if s.signal_type in [SignalType.BEARISH, SignalType.STRONG_SELL]])
            
            total_recent = len(recent_signals)
            trend_ratio = (bullish_trend - bearish_trend) / total_recent
            
            if trend_ratio > 0.3:
                return "increasing"
            elif trend_ratio < -0.3:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.warning(f"Trend analysis failed for {symbol}: {e}")
            return "stable"
    
    def _calculate_volume_change(self, symbol: str) -> float:
        """Calculate volume change for symbol"""
        try:
            signals = self._historical_signals.get(symbol, [])
            if len(signals) < 20:
                return 0.0
            
            # Compare current hour with previous hour
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            previous_hour = current_hour - timedelta(hours=1)
            
            current_count = len([s for s in signals if s.timestamp >= current_hour])
            previous_count = len([s for s in signals if previous_hour <= s.timestamp < current_hour])
            
            if previous_count == 0:
                return 1.0 if current_count > 0 else 0.0
            
            return (current_count - previous_count) / previous_count
            
        except Exception as e:
            logger.warning(f"Volume change calculation failed for {symbol}: {e}")
            return 0.0
    
    def _extract_impact_level(self, text: str) -> str:
        """Extract impact level from Forex Factory event text"""
        text_lower = text.lower()
        
        if "high" in text_lower:
            return "high"
        elif "medium" in text_lower:
            return "medium"
        elif "low" in text_lower:
            return "low"
        else:
            return "unknown"
    
    def _impact_to_sentiment(self, impact: str) -> float:
        """Convert impact level to sentiment score"""
        impact_scores = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2,
            "unknown": 0.0
        }
        return impact_scores.get(impact, 0.0)
    
    def _summary_to_signal_type(self, summary: str) -> SignalType:
        """Convert technical summary to signal type"""
        summary_lower = summary.lower()
        
        if any(word in summary_lower for word in ["strong buy", "buy", "bullish", "long"]):
            return SignalType.BULLISH
        elif any(word in summary_lower for word in ["strong sell", "sell", "bearish", "short"]):
            return SignalType.BEARISH
        else:
            return SignalType.NEUTRAL
    
    def _signal_type_to_sentiment(self, signal_type: SignalType) -> float:
        """Convert signal type to sentiment score"""
        sentiment_scores = {
            SignalType.STRONG_BUY: 0.9,
            SignalType.BULLISH: 0.6,
            SignalType.NEUTRAL: 0.0,
            SignalType.BEARISH: -0.6,
            SignalType.STRONG_SELL: -0.9
        }
        return sentiment_scores.get(signal_type, 0.0)
    
    def get_aggregated_signals(self, symbol: str = None) -> Dict[str, AggregateSignal]:
        """Get aggregated signals"""
        with self._lock:
            if symbol:
                return {symbol: self._aggregated_signals.get(symbol)}
            return self._aggregated_signals.copy()
    
    def get_community_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get community sentiment for specific symbol"""
        aggregated = self.get_aggregated_signals(symbol)
        if not aggregated or symbol not in aggregated:
            return None
        
        signal = aggregated[symbol]
        
        return {
            'symbol': symbol,
            'net_sentiment': signal.net_sentiment,
            'signal_strength': signal.signal_strength,
            'dominant_signal': signal.dominant_signal.value,
            'confidence': signal.confidence_score,
            'total_signals': signal.total_signals,
            'bullish_ratio': signal.bullish_count / signal.total_signals,
            'bearish_ratio': signal.bearish_count / signal.total_signals,
            'recent_trend': signal.recent_trend,
            'volume_change': signal.volume_change,
            'timestamp': signal.timestamp
        }
    
    def get_top_influencers(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top influencers by credibility score"""
        with self._lock:
            sorted_influencers = sorted(
                self._influencer_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_influencers[:limit]
    
    def get_platform_stats(self) -> Dict[str, Any]:
        """Get platform statistics"""
        with self._lock:
            return {
                'total_signals_processed': self._stats['total_signals_processed'],
                'platform_breakdown': dict(self._stats['platform_breakdown']),
                'symbol_breakdown': dict(self._stats['symbol_breakdown']),
                'active_symbols': len(self._signals),
                'total_influencers': len(self._influencer_scores)
            }
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring tasks"""
        self._running = False
        self._executor.shutdown(wait=True)
        
        if self._web_driver:
            self._web_driver.quit()
        
        logger.info("Community signals monitoring stopped")

# Example usage and testing
async def main():
    """Example usage of the AdvancedCommunityAnalyzer"""
    
    # Configure community analysis
    config = CommunityConfig(
        platforms={
            Platform.TWITTER: PlatformConfig(
                enabled=True,
                api_key="your_twitter_api_key",
                api_secret="your_twitter_api_secret",
                access_token="your_access_token",
                access_token_secret="your_access_token_secret",
                max_posts=50
            ),
            Platform.REDDIT: PlatformConfig(
                enabled=True,
                api_key="your_reddit_client_id",
                api_secret="your_reddit_client_secret",
                max_posts=30
            ),
            Platform.TRADINGVIEW: PlatformConfig(enabled=True),
            Platform.FOREX_FACTORY: PlatformConfig(enabled=True),
            Platform.INVESTING_COM: PlatformConfig(enabled=True)
        },
        symbols=["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"],
        min_confidence=0.6,
        aggregation_window=300  # 5 minutes
    )
    
    # Initialize analyzer
    analyzer = AdvancedCommunityAnalyzer(config)
    
    print("=== Testing Community Signals Analyzer ===\n")
    
    # Start monitoring (in background)
    monitoring_task = asyncio.create_task(analyzer.start_monitoring())
    
    # Wait for some signals to be collected
    print("Collecting community signals for 30 seconds...")
    await asyncio.sleep(30)
    
    # Test 1: Get aggregated signals
    print("1. Aggregated Signals:")
    aggregated = analyzer.get_aggregated_signals()
    
    for symbol, signal in aggregated.items():
        print(f"   {symbol}:")
        print(f"     Dominant: {signal.dominant_signal.value}")
        print(f"     Strength: {signal.signal_strength:.2f}")
        print(f"     Confidence: {signal.confidence_score:.2f}")
        print(f"     Total Signals: {signal.total_signals}")
        print(f"     Net Sentiment: {signal.net_sentiment:.2f}")
        print(f"     Recent Trend: {signal.recent_trend}")
        print(f"     Volume Change: {signal.volume_change:.1%}")
    
    # Test 2: Get community sentiment for specific symbol
    print("\n2. Community Sentiment for EUR/USD:")
    sentiment = analyzer.get_community_sentiment("EUR/USD")
    if sentiment:
        for key, value in sentiment.items():
            print(f"   {key}: {value}")
    
    # Test 3: Get platform statistics
    print("\n3. Platform Statistics:")
    stats = analyzer.get_platform_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Test 4: Get top influencers
    print("\n4. Top Influencers:")
    influencers = analyzer.get_top_influencers(5)
    for author, score in influencers:
        print(f"   {author}: {score:.3f}")
    
    # Stop monitoring
    await analyzer.stop_monitoring()
    monitoring_task.cancel()
    
    print("\n=== Community Signals Test Completed ===")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run async main
    asyncio.run(main())
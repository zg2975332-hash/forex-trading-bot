"""
Advanced Social Sentiment Analyzer for FOREX TRADING BOT
Real-time social media sentiment analysis from Twitter, Reddit, and financial forums
"""

import logging
import pandas as pd
import numpy as np
import re
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3
import warnings

# Social Media APIs
import tweepy
import praw
from bs4 import BeautifulSoup

# Sentiment Analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Utility
import requests
from urllib.parse import quote
import html

logger = logging.getLogger(__name__)

class SocialPlatform(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    TRADING_VIEW = "tradingview"
    FOREX_FACTORY = "forex_factory"
    INVESTING_COM = "investing_com"
    STOCKTWITS = "stocktwits"
    YAHOO_FINANCE = "yahoo_finance"
    TELEGRAM = "telegram"

class SentimentImpact(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ContentType(Enum):
    TWEET = "tweet"
    POST = "post"
    COMMENT = "comment"
    THREAD = "thread"
    NEWS = "news"
    ANALYSIS = "analysis"

@dataclass
class SocialPost:
    """Social media post structure"""
    post_id: str
    platform: SocialPlatform
    author: str
    content: str
    timestamp: datetime
    likes: int = 0
    shares: int = 0
    comments: int = 0
    url: str = ""
    symbol: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SocialSentiment:
    """Social sentiment analysis result"""
    platform: SocialPlatform
    symbol: str
    overall_score: float
    confidence: float
    sentiment_label: str
    impact: SentimentImpact
    volume: int
    engagement_rate: float
    dominant_topics: List[str]
    influencer_impact: float
    timestamp: datetime
    posts_analyzed: int
    trend_direction: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SocialConfig:
    """Configuration for social sentiment analyzer"""
    # API Credentials
    twitter_bearer_token: str = ""
    twitter_consumer_key: str = ""
    twitter_consumer_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_secret: str = ""
    
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "ForexTradingBot v1.0"
    
    # Platform Settings
    enable_twitter: bool = True
    enable_reddit: bool = True
    enable_web_scraping: bool = True
    
    # Analysis Parameters
    max_posts_per_source: int = 200
    min_engagement_threshold: int = 5
    influencer_follower_threshold: int = 10000
    
    # Time Windows
    lookback_hours: int = 24
    update_frequency: int = 300  # seconds
    
    # Sentiment Weights
    platform_weights: Dict[SocialPlatform, float] = field(default_factory=lambda: {
        SocialPlatform.TWITTER: 0.8,
        SocialPlatform.REDDIT: 0.7,
        SocialPlatform.TRADING_VIEW: 0.6,
        SocialPlatform.FOREX_FACTORY: 0.9,
        SocialPlatform.INVESTING_COM: 0.7,
        SocialPlatform.STOCKTWITS: 0.5
    })
    
    # Content Filters
    spam_keywords: List[str] = field(default_factory=lambda: [
        "signals", "guaranteed", "profit", "make money", "free", "click here",
        "join now", "limited time", "100%", "guarantee"
    ])
    
    # Risk Management
    max_api_calls_per_minute: int = 50
    rate_limit_sleep: int = 60

class AdvancedSocialSentiment:
    """
    Advanced social media sentiment analysis for Forex trading
    """
    
    def __init__(self, config: SocialConfig = None):
        self.config = config or SocialConfig()
        
        # Initialize APIs
        self._initialize_apis()
        
        # Sentiment analyzers
        self._initialize_sentiment_analyzers()
        
        # Data storage
        self.social_posts = defaultdict(lambda: deque(maxlen=1000))
        self.sentiment_results = {}
        self.influencer_scores = defaultdict(float)
        self.topic_trends = defaultdict(lambda: deque(maxlen=500))
        
        # Rate limiting
        self.api_call_times = deque(maxlen=self.config.max_api_calls_per_minute)
        
        # Thread safety
        self._lock = threading.RLock()
        self._analysis_lock = threading.Lock()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("AdvancedSocialSentiment initialized successfully")

    def _initialize_apis(self):
        """Initialize social media API clients"""
        try:
            # Twitter API
            if self.config.enable_twitter and self.config.twitter_bearer_token:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.config.twitter_bearer_token,
                    consumer_key=self.config.twitter_consumer_key,
                    consumer_secret=self.config.twitter_consumer_secret,
                    access_token=self.config.twitter_access_token,
                    access_token_secret=self.config.twitter_access_secret,
                    wait_on_rate_limit=True
                )
                logger.info("Twitter API client initialized")
            else:
                self.twitter_client = None
                logger.warning("Twitter API not configured")
            
            # Reddit API
            if self.config.enable_reddit and self.config.reddit_client_id:
                self.reddit_client = praw.Reddit(
                    client_id=self.config.reddit_client_id,
                    client_secret=self.config.reddit_client_secret,
                    user_agent=self.config.reddit_user_agent
                )
                logger.info("Reddit API client initialized")
            else:
                self.reddit_client = None
                logger.warning("Reddit API not configured")
            
            # HTTP session for web scraping
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
        except Exception as e:
            logger.error(f"API initialization failed: {e}")

    def _initialize_sentiment_analyzers(self):
        """Initialize sentiment analysis models"""
        try:
            # VADER for social media text
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Enhance VADER with trading terms
            trading_lexicon = {
                'bullish': 2.0, 'bearish': -2.0, 'long': 1.0, 'short': -1.0,
                'buy': 1.5, 'sell': -1.5, 'rally': 1.8, 'crash': -2.5,
                'breakout': 1.2, 'breakdown': -1.2, 'support': 0.5, 'resistance': -0.5,
                'moon': 2.0, 'dump': -2.0, 'pump': 1.5, 'hodl': 0.8
            }
            
            for word, score in trading_lexicon.items():
                self.vader_analyzer.lexicon[word] = score
            
            # Transformer model for complex analysis
            try:
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logger.info("Transformer sentiment model initialized")
            except Exception as e:
                logger.warning(f"Transformer model failed: {e}")
                self.transformer_pipeline = None
            
            logger.info("Sentiment analyzers initialized")
            
        except Exception as e:
            logger.error(f"Sentiment analyzer initialization failed: {e}")

    def _start_background_tasks(self):
        """Start background data collection tasks"""
        # Social media monitoring
        monitoring_thread = threading.Thread(target=self._social_monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # Sentiment analysis
        analysis_thread = threading.Thread(target=self._sentiment_analysis_loop, daemon=True)
        analysis_thread.start()
        
        # Trend detection
        trend_thread = threading.Thread(target=self._trend_detection_loop, daemon=True)
        trend_thread.start()
        
        # Data cleanup
        cleanup_thread = threading.Thread(target=self._data_cleanup_loop, daemon=True)
        cleanup_thread.start()

    def _rate_limit_check(self):
        """Check and enforce API rate limits"""
        current_time = time.time()
        
        # Remove old API calls
        while (self.api_call_times and 
               current_time - self.api_call_times[0] > 60):
            self.api_call_times.popleft()
        
        # Check if we're at rate limit
        if len(self.api_call_times) >= self.config.max_api_calls_per_minute:
            sleep_time = 60 - (current_time - self.api_call_times[0])
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.api_call_times.append(current_time)

    async def fetch_twitter_sentiment(self, symbol: str, query: str = None) -> List[SocialPost]:
        """Fetch and analyze Twitter sentiment for a symbol"""
        if not self.twitter_client:
            return []
        
        posts = []
        
        try:
            self._rate_limit_check()
            
            # Build search query
            base_query = query or f"#{symbol} OR ${symbol} OR {symbol.replace('/', ' ')} -is:retweet lang:en"
            
            # Search for tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=base_query,
                max_results=min(100, self.config.max_posts_per_source),
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                user_fields=['username', 'public_metrics'],
                expansions=['author_id']
            )
            
            if tweets and tweets.data:
                # Get user information
                users = {user.id: user for user in tweets.includes.get('users', [])}
                
                for tweet in tweets.data:
                    user = users.get(tweet.author_id)
                    
                    # Skip low engagement posts
                    engagement = (tweet.public_metrics.get('like_count', 0) +
                                 tweet.public_metrics.get('retweet_count', 0))
                    
                    if engagement < self.config.min_engagement_threshold:
                        continue
                    
                    # Check for spam
                    if self._is_spam_content(tweet.text):
                        continue
                    
                    post = SocialPost(
                        post_id=f"twitter_{tweet.id}",
                        platform=SocialPlatform.TWITTER,
                        author=user.username if user else "unknown",
                        content=tweet.text,
                        timestamp=tweet.created_at,
                        likes=tweet.public_metrics.get('like_count', 0),
                        shares=tweet.public_metrics.get('retweet_count', 0),
                        comments=tweet.public_metrics.get('reply_count', 0),
                        url=f"https://twitter.com/user/status/{tweet.id}",
                        symbol=symbol,
                        metadata={
                            'followers': user.public_metrics.get('followers_count', 0) if user else 0,
                            'engagement_rate': engagement,
                            'context_annotations': tweet.context_annotations
                        }
                    )
                    
                    posts.append(post)
            
            logger.info(f"Fetched {len(posts)} Twitter posts for {symbol}")
            
        except Exception as e:
            logger.error(f"Twitter sentiment fetch failed for {symbol}: {e}")
        
        return posts

    async def fetch_reddit_sentiment(self, symbol: str) -> List[SocialPost]:
        """Fetch and analyze Reddit sentiment for a symbol"""
        if not self.reddit_client:
            return []
        
        posts = []
        
        try:
            self._rate_limit_check()
            
            # Relevant subreddits for Forex
            subreddits = [
                'Forex', 'trading', 'investing', 'stocks', 
                'economics', 'wallstreetbets', 'smallstreetbets'
            ]
            
            search_query = symbol.replace('/', ' ') if '/' in symbol else symbol
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Search for posts
                    for post in subreddit.search(
                        search_query, 
                        limit=self.config.max_posts_per_source // len(subreddits),
                        time_filter='day'
                    ):
                        # Skip removed or low score posts
                        if (post.removed or post.score < self.config.min_engagement_threshold or
                            self._is_spam_content(post.title + " " + post.selftext)):
                            continue
                        
                        reddit_post = SocialPost(
                            post_id=f"reddit_{post.id}",
                            platform=SocialPlatform.REDDIT,
                            author=str(post.author) if post.author else "unknown",
                            content=f"{post.title}. {post.selftext}",
                            timestamp=datetime.fromtimestamp(post.created_utc),
                            likes=post.score,
                            shares=post.num_crossposts,
                            comments=post.num_comments,
                            url=post.url,
                            symbol=symbol,
                            metadata={
                                'subreddit': subreddit_name,
                                'upvote_ratio': post.upvote_ratio,
                                'awards': len(post.all_awardings)
                            }
                        )
                        
                        posts.append(reddit_post)
                        
                        # Also get top comments
                        try:
                            post.comments.replace_more(limit=0)
                            for comment in post.comments.list()[:10]:  # Top 10 comments
                                if (isinstance(comment, praw.models.Comment) and 
                                    comment.score > 5 and 
                                    not self._is_spam_content(comment.body)):
                                    
                                    comment_post = SocialPost(
                                        post_id=f"reddit_comment_{comment.id}",
                                        platform=SocialPlatform.REDDIT,
                                        author=str(comment.author) if comment.author else "unknown",
                                        content=comment.body,
                                        timestamp=datetime.fromtimestamp(comment.created_utc),
                                        likes=comment.score,
                                        shares=0,
                                        comments=0,
                                        url=f"https://reddit.com{comment.permalink}",
                                        symbol=symbol,
                                        metadata={
                                            'subreddit': subreddit_name,
                                            'parent_post': post.id,
                                            'is_comment': True
                                        }
                                    )
                                    
                                    posts.append(comment_post)
                        except Exception as comment_error:
                            logger.warning(f"Error fetching comments: {comment_error}")
                            
                except Exception as subreddit_error:
                    logger.error(f"Error fetching from subreddit {subreddit_name}: {subreddit_error}")
                    continue
            
            logger.info(f"Fetched {len(posts)} Reddit posts for {symbol}")
            
        except Exception as e:
            logger.error(f"Reddit sentiment fetch failed for {symbol}: {e}")
        
        return posts

    async def fetch_tradingview_sentiment(self, symbol: str) -> List[SocialPost]:
        """Fetch sentiment from TradingView (web scraping)"""
        if not self.config.enable_web_scraping:
            return []
        
        posts = []
        
        try:
            self._rate_limit_check()
            
            # TradingView ideas page
            symbol_clean = symbol.replace('/', '').replace('USD', '')
            url = f"https://www.tradingview.com/symbols/{symbol_clean}/ideas/"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract ideas (this would need to be adapted based on actual page structure)
                ideas = soup.find_all('div', class_='tv-widget-idea')[:20]
                
                for i, idea in enumerate(ideas):
                    try:
                        title_elem = idea.find('div', class_='tv-widget-idea__title')
                        content_elem = idea.find('div', class_='tv-widget-idea__description')
                        
                        if title_elem and content_elem:
                            title = title_elem.get_text(strip=True)
                            content = content_elem.get_text(strip=True)
                            
                            if self._is_spam_content(title + " " + content):
                                continue
                            
                            post = SocialPost(
                                post_id=f"tradingview_{i}_{int(time.time())}",
                                platform=SocialPlatform.TRADING_VIEW,
                                author="TradingView User",
                                content=f"{title}. {content}",
                                timestamp=datetime.now(),
                                likes=0,  # Would need to extract actual metrics
                                shares=0,
                                comments=0,
                                url=url,
                                symbol=symbol,
                                metadata={
                                    'source': 'TradingView',
                                    'scraped_at': datetime.now()
                                }
                            )
                            
                            posts.append(post)
                    except Exception as idea_error:
                        logger.warning(f"Error parsing TradingView idea: {idea_error}")
                        continue
            
            logger.info(f"Fetched {len(posts)} TradingView posts for {symbol}")
            
        except Exception as e:
            logger.error(f"TradingView sentiment fetch failed for {symbol}: {e}")
        
        return posts

    async def fetch_forex_factory_sentiment(self, symbol: str) -> List[SocialPost]:
        """Fetch sentiment from Forex Factory forum"""
        if not self.config.enable_web_scraping:
            return []
        
        posts = []
        
        try:
            self._rate_limit_check()
            
            # Forex Factory forum (simulated - would need actual scraping logic)
            # This is a simplified implementation
            simulated_posts = [
                {
                    'content': f"{symbol} showing strong bullish momentum on daily chart. Targeting resistance at 1.1200",
                    'author': 'ForexTrader123',
                    'likes': 15
                },
                {
                    'content': f"Bearish divergence on {symbol} 4H chart. Expecting pullback to 1.0950 support",
                    'author': 'TechnicalAnalyst',
                    'likes': 8
                }
            ]
            
            for i, post_data in enumerate(simulated_posts):
                post = SocialPost(
                    post_id=f"forexfactory_{i}_{int(time.time())}",
                    platform=SocialPlatform.FOREX_FACTORY,
                    author=post_data['author'],
                    content=post_data['content'],
                    timestamp=datetime.now(),
                    likes=post_data['likes'],
                    shares=0,
                    comments=3,  # Simulated
                    url=f"https://www.forexfactory.com/thread/{i}",
                    symbol=symbol,
                    metadata={
                        'source': 'ForexFactory',
                        'simulated': True
                    }
                )
                
                posts.append(post)
            
            logger.info(f"Fetched {len(posts)} Forex Factory posts for {symbol}")
            
        except Exception as e:
            logger.error(f"Forex Factory sentiment fetch failed for {symbol}: {e}")
        
        return posts

    def _is_spam_content(self, content: str) -> bool:
        """Check if content contains spam keywords"""
        content_lower = content.lower()
        return any(spam_keyword in content_lower for spam_keyword in self.config.spam_keywords)

    def _analyze_post_sentiment(self, post: SocialPost) -> Dict[str, float]:
        """Analyze sentiment of a single post using multiple methods"""
        try:
            sentiment_scores = {}
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(post.content)
            sentiment_scores['vader'] = vader_scores['compound']
            
            # TextBlob sentiment
            blob = TextBlob(post.content)
            sentiment_scores['textblob'] = blob.sentiment.polarity
            
            # Transformer sentiment (if available)
            if self.transformer_pipeline:
                try:
                    transformer_result = self.transformer_pipeline(post.content[:512])[0]
                    if transformer_result['label'] == 'positive':
                        sentiment_scores['transformer'] = transformer_result['score']
                    elif transformer_result['label'] == 'negative':
                        sentiment_scores['transformer'] = -transformer_result['score']
                    else:
                        sentiment_scores['transformer'] = 0.0
                except Exception as e:
                    logger.warning(f"Transformer analysis failed: {e}")
                    sentiment_scores['transformer'] = 0.0
            
            # Engagement-weighted score
            engagement_weight = min(1.0, (post.likes + post.shares) / 100.0)
            
            # Calculate weighted average
            weights = {'vader': 0.4, 'textblob': 0.3, 'transformer': 0.3}
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method, score in sentiment_scores.items():
                weight = weights.get(method, 0.1)
                weighted_sum += score * weight
                total_weight += weight
            
            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            final_score *= (1.0 + engagement_weight * 0.2)  # Boost for engagement
            
            return {
                'final_score': final_score,
                'method_scores': sentiment_scores,
                'engagement_weight': engagement_weight
            }
            
        except Exception as e:
            logger.error(f"Post sentiment analysis failed: {e}")
            return {'final_score': 0.0, 'method_scores': {}, 'engagement_weight': 0.0}

    def _calculate_influencer_impact(self, post: SocialPost) -> float:
        """Calculate influencer impact score for a post"""
        try:
            base_impact = 1.0
            
            # Platform-specific impact factors
            platform_factors = {
                SocialPlatform.TWITTER: 1.2,
                SocialPlatform.REDDIT: 1.0,
                SocialPlatform.TRADING_VIEW: 0.8,
                SocialPlatform.FOREX_FACTORY: 1.1
            }
            
            platform_factor = platform_factors.get(post.platform, 0.5)
            
            # Follower-based impact (for Twitter)
            if post.platform == SocialPlatform.TWITTER:
                followers = post.metadata.get('followers', 0)
                if followers > self.config.influencer_follower_threshold:
                    follower_impact = min(2.0, followers / 50000.0)  # Cap at 2x
                    base_impact *= follower_impact
            
            # Engagement-based impact
            engagement = post.likes + post.shares + post.comments
            engagement_impact = min(1.5, 1.0 + (engagement / 100.0))
            
            # Historical credibility (simplified)
            author_credibility = self.influencer_scores.get(post.author, 1.0)
            
            final_impact = base_impact * platform_factor * engagement_impact * author_credibility
            
            return min(3.0, final_impact)  # Cap at 3x impact
            
        except Exception as e:
            logger.error(f"Influencer impact calculation failed: {e}")
            return 1.0

    async def analyze_social_sentiment(self, symbol: str) -> SocialSentiment:
        """Perform comprehensive social sentiment analysis for a symbol"""
        try:
            # Fetch posts from all platforms
            fetch_tasks = []
            
            if self.config.enable_twitter:
                fetch_tasks.append(self.fetch_twitter_sentiment(symbol))
            
            if self.config.enable_reddit:
                fetch_tasks.append(self.fetch_reddit_sentiment(symbol))
            
            if self.config.enable_web_scraping:
                fetch_tasks.append(self.fetch_tradingview_sentiment(symbol))
                fetch_tasks.append(self.fetch_forex_factory_sentiment(symbol))
            
            # Execute all fetch tasks
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # Process results
            all_posts = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Social fetch task failed: {result}")
                elif isinstance(result, list):
                    all_posts.extend(result)
            
            # Store posts
            with self._lock:
                for post in all_posts:
                    self.social_posts[symbol].append(post)
            
            # Analyze sentiment
            if not all_posts:
                return self._create_neutral_sentiment(symbol)
            
            # Calculate platform-specific scores
            platform_scores = defaultdict(list)
            platform_volumes = defaultdict(int)
            total_engagement = 0
            influencer_impact_total = 0.0
            influencer_count = 0
            
            topics = defaultdict(int)
            
            for post in all_posts:
                # Analyze post sentiment
                sentiment_analysis = self._analyze_post_sentiment(post)
                post_score = sentiment_analysis['final_score']
                
                # Calculate influencer impact
                influencer_impact = self._calculate_influencer_impact(post)
                
                # Apply platform weight and influencer impact
                platform_weight = self.config.platform_weights.get(post.platform, 0.5)
                weighted_score = post_score * platform_weight * influencer_impact
                
                platform_scores[post.platform].append(weighted_score)
                platform_volumes[post.platform] += 1
                total_engagement += post.likes + post.shares + post.comments
                
                if influencer_impact > 1.5:
                    influencer_impact_total += influencer_impact
                    influencer_count += 1
                
                # Extract topics (simplified)
                self._extract_topics(post.content, topics)
            
            # Calculate overall sentiment
            overall_score = 0.0
            total_weighted_posts = 0
            
            for platform, scores in platform_scores.items():
                platform_avg = np.mean(scores) if scores else 0.0
                platform_volume = platform_volumes[platform]
                platform_weight = self.config.platform_weights.get(platform, 0.5)
                
                overall_score += platform_avg * platform_volume * platform_weight
                total_weighted_posts += platform_volume * platform_weight
            
            if total_weighted_posts > 0:
                overall_score /= total_weighted_posts
            
            # Calculate confidence
            confidence = min(1.0, len(all_posts) / 100.0)
            if len(platform_scores) > 1:
                score_agreement = 1.0 - np.std([np.mean(scores) for scores in platform_scores.values() if scores]) / 2.0
                confidence = (confidence + score_agreement) / 2.0
            
            # Determine sentiment label and impact
            sentiment_label = self._classify_sentiment(overall_score)
            impact = self._determine_impact(overall_score, confidence, len(all_posts), total_engagement)
            
            # Calculate influencer impact average
            avg_influencer_impact = influencer_impact_total / influencer_count if influencer_count > 0 else 1.0
            
            # Get dominant topics
            dominant_topics = self._get_dominant_topics(topics)
            
            # Calculate trend
            trend_direction = self._calculate_sentiment_trend(symbol)
            
            # Create sentiment result
            sentiment = SocialSentiment(
                platform=SocialPlatform.TWITTER,  # Primary platform
                symbol=symbol,
                overall_score=overall_score,
                confidence=confidence,
                sentiment_label=sentiment_label,
                impact=impact,
                volume=len(all_posts),
                engagement_rate=total_engagement / len(all_posts) if all_posts else 0,
                dominant_topics=dominant_topics,
                influencer_impact=avg_influencer_impact,
                timestamp=datetime.now(),
                posts_analyzed=len(all_posts),
                trend_direction=trend_direction,
                metadata={
                    'platform_breakdown': {p.value: len(scores) for p, scores in platform_scores.items()},
                    'total_engagement': total_engagement
                }
            )
            
            # Store result
            with self._lock:
                self.sentiment_results[symbol] = sentiment
            
            logger.info(f"Social sentiment analysis completed for {symbol}: {sentiment_label} (score: {overall_score:.3f})")
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Social sentiment analysis failed for {symbol}: {e}")
            return self._create_neutral_sentiment(symbol)

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on score"""
        if score > 0.3:
            return "strongly_bullish"
        elif score > 0.1:
            return "bullish"
        elif score < -0.3:
            return "strongly_bearish"
        elif score < -0.1:
            return "bearish"
        else:
            return "neutral"

    def _determine_impact(self, score: float, confidence: float, volume: int, engagement: int) -> SentimentImpact:
        """Determine the impact level of the sentiment"""
        impact_score = (abs(score) * 0.4 + confidence * 0.3 + 
                       min(1.0, volume / 50.0) * 0.2 + min(1.0, engagement / 100.0) * 0.1)
        
        if impact_score > 0.7:
            return SentimentImpact.VERY_HIGH
        elif impact_score > 0.5:
            return SentimentImpact.HIGH
        elif impact_score > 0.3:
            return SentimentImpact.MEDIUM
        elif impact_score > 0.1:
            return SentimentImpact.LOW
        else:
            return SentimentImpact.VERY_LOW

    def _extract_topics(self, content: str, topics: Dict[str, int]):
        """Extract topics from content"""
        try:
            # Simple keyword-based topic extraction
            topic_keywords = {
                'technical_analysis': ['support', 'resistance', 'breakout', 'breakdown', 'rsi', 'macd', 'bollinger'],
                'fundamental_analysis': ['interest rates', 'inflation', 'gdp', 'employment', 'fed', 'ecb'],
                'market_sentiment': ['fear', 'greed', 'momentum', 'trend', 'volatility'],
                'trading_strategy': ['long', 'short', 'entry', 'exit', 'stop loss', 'take profit'],
                'economic_events': ['nfp', 'cpi', 'rate decision', 'central bank', 'fomc']
            }
            
            content_lower = content.lower()
            
            for topic, keywords in topic_keywords.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        topics[topic] += 1
                        break
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")

    def _get_dominant_topics(self, topics: Dict[str, int]) -> List[str]:
        """Get dominant topics from frequency counts"""
        try:
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
            return [topic for topic, count in sorted_topics[:3] if count > 0]
        except Exception as e:
            logger.warning(f"Dominant topics extraction failed: {e}")
            return []

    def _calculate_sentiment_trend(self, symbol: str) -> str:
        """Calculate sentiment trend direction"""
        try:
            recent_posts = list(self.social_posts.get(symbol, []))[-50:]  # Last 50 posts
            if len(recent_posts) < 10:
                return "stable"
            
            # Split into time windows
            window_size = max(1, len(recent_posts) // 3)
            windows = [recent_posts[i:i + window_size] for i in range(0, len(recent_posts), window_size)]
            
            if len(windows) < 2:
                return "stable"
            
            # Calculate average sentiment for each window
            window_scores = []
            for window in windows[:3]:
                if window:
                    window_sentiments = [self._analyze_post_sentiment(post)['final_score'] for post in window]
                    window_scores.append(np.mean(window_sentiments))
            
            # Determine trend
            if len(window_scores) >= 2:
                trend = window_scores[-1] - window_scores[0]
                
                if trend > 0.1:
                    return "improving"
                elif trend < -0.1:
                    return "deteriorating"
            
            return "stable"
            
        except Exception as e:
            logger.error(f"Sentiment trend calculation failed: {e}")
            return "stable"

    def _create_neutral_sentiment(self, symbol: str) -> SocialSentiment:
        """Create neutral sentiment result"""
        return SocialSentiment(
            platform=SocialPlatform.TWITTER,
            symbol=symbol,
            overall_score=0.0,
            confidence=0.1,
            sentiment_label="neutral",
            impact=SentimentImpact.VERY_LOW,
            volume=0,
            engagement_rate=0.0,
            dominant_topics=[],
            influencer_impact=1.0,
            timestamp=datetime.now(),
            posts_analyzed=0,
            trend_direction="stable",
            metadata={'error': 'insufficient_data'}
        )

    def _social_monitoring_loop(self):
        """Background social media monitoring loop"""
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
        
        while True:
            try:
                for symbol in symbols:
                    asyncio.run(self.analyze_social_sentiment(symbol))
                
                logger.info("Social media monitoring cycle completed")
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Social monitoring loop failed: {e}")
                time.sleep(60)

    def _sentiment_analysis_loop(self):
        """Background sentiment analysis loop"""
        while True:
            try:
                # Update influencer scores based on recent performance
                self._update_influencer_scores()
                time.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Sentiment analysis loop failed: {e}")
                time.sleep(300)

    def _trend_detection_loop(self):
        """Background trend detection loop"""
        while True:
            try:
                # Analyze topic trends over time
                self._analyze_topic_trends()
                time.sleep(900)  # Run every 15 minutes
                
            except Exception as e:
                logger.error(f"Trend detection loop failed: {e}")
                time.sleep(300)

    def _data_cleanup_loop(self):
        """Background data cleanup loop"""
        while True:
            try:
                with self._lock:
                    cutoff = datetime.now() - timedelta(hours=self.config.lookback_hours)
                    
                    # Clean old posts
                    for symbol in list(self.social_posts.keys()):
                        self.social_posts[symbol] = deque(
                            [post for post in self.social_posts[symbol] if post.timestamp >= cutoff],
                            maxlen=1000
                        )
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Data cleanup loop failed: {e}")
                time.sleep(1800)

    def _update_influencer_scores(self):
        """Update influencer credibility scores"""
        try:
            # This would implement more sophisticated influencer scoring
            # For now, it's a simplified version
            pass
        except Exception as e:
            logger.error(f"Influencer score update failed: {e}")

    def _analyze_topic_trends(self):
        """Analyze trending topics over time"""
        try:
            # Track topic frequency over time
            for symbol in self.social_posts.keys():
                recent_posts = list(self.social_posts[symbol])[-100:]
                
                current_topics = defaultdict(int)
                for post in recent_posts:
                    self._extract_topics(post.content, current_topics)
                
                self.topic_trends[symbol].append({
                    'timestamp': datetime.now(),
                    'topics': dict(current_topics)
                })
                
        except Exception as e:
            logger.error(f"Topic trend analysis failed: {e}")

    def get_current_sentiment(self, symbol: str) -> Optional[SocialSentiment]:
        """Get current social sentiment for a symbol"""
        return self.sentiment_results.get(symbol)

    def get_sentiment_alerts(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get alerts for significant social sentiment movements"""
        alerts = []
        
        try:
            for symbol, sentiment in self.sentiment_results.items():
                if (sentiment.confidence > 0.6 and 
                    abs(sentiment.overall_score) > threshold and
                    sentiment.impact in [SentimentImpact.HIGH, SentimentImpact.VERY_HIGH]):
                    
                    alert = {
                        'symbol': symbol,
                        'sentiment_score': sentiment.overall_score,
                        'sentiment_label': sentiment.sentiment_label,
                        'confidence': sentiment.confidence,
                        'impact': sentiment.impact.value,
                        'volume': sentiment.volume,
                        'trend': sentiment.trend_direction,
                        'dominant_topics': sentiment.dominant_topics,
                        'timestamp': sentiment.timestamp
                    }
                    
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Sentiment alerts generation failed: {e}")
            return []

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the social sentiment analyzer"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'total_posts_analyzed': sum(len(posts) for posts in self.social_posts.values()),
                'symbol_coverage': {symbol: len(posts) for symbol, posts in self.social_posts.items()},
                'platform_distribution': defaultdict(int),
                'average_confidence': 0.0,
                'recent_alerts': len(self.get_sentiment_alerts(0.3))
            }
            
            # Calculate platform distribution
            for symbol_posts in self.social_posts.values():
                for post in symbol_posts:
                    metrics['platform_distribution'][post.platform.value] += 1
            
            # Calculate average confidence
            confidences = [sentiment.confidence for sentiment in self.sentiment_results.values()]
            if confidences:
                metrics['average_confidence'] = np.mean(confidences)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {'timestamp': datetime.now(), 'error': str(e)}

# Example usage and testing
def main():
    """Example usage of the AdvancedSocialSentiment"""
    
    # Configuration
    config = SocialConfig(
        enable_twitter=False,  # Set to True if you have API keys
        enable_reddit=False,   # Set to True if you have API keys
        enable_web_scraping=True,
        update_frequency=600   # 10 minutes for testing
    )
    
    # Initialize analyzer
    analyzer = AdvancedSocialSentiment(config)
    
    # Wait for initial data collection
    print("=== Social Sentiment Analysis Demo ===")
    print("Collecting social data...")
    time.sleep(10)
    
    # Get current sentiment
    symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
    
    for symbol in symbols:
        sentiment = analyzer.get_current_sentiment(symbol)
        if sentiment:
            print(f"\n{symbol} Social Sentiment:")
            print(f"  Score: {sentiment.overall_score:.3f}")
            print(f"  Label: {sentiment.sentiment_label}")
            print(f"  Confidence: {sentiment.confidence:.3f}")
            print(f"  Impact: {sentiment.impact.value}")
            print(f"  Volume: {sentiment.volume} posts")
            print(f"  Trend: {sentiment.trend_direction}")
            print(f"  Topics: {sentiment.dominant_topics}")
        else:
            print(f"\n{symbol}: No sentiment data available yet")
    
    # Get performance metrics
    print("\n=== Performance Metrics ===")
    metrics = analyzer.get_performance_metrics()
    print(f"Total Posts Analyzed: {metrics['total_posts_analyzed']}")
    print(f"Symbol Coverage: {metrics['symbol_coverage']}")
    print(f"Platform Distribution: {dict(metrics['platform_distribution'])}")
    print(f"Average Confidence: {metrics['average_confidence']:.3f}")
    
    # Get alerts
    print("\n=== Social Sentiment Alerts ===")
    alerts = analyzer.get_sentiment_alerts(threshold=0.3)
    for alert in alerts:
        print(f"Alert: {alert['symbol']} - {alert['sentiment_label']} (Score: {alert['sentiment_score']:.3f})")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
"""
Advanced Sentiment Analysis for FOREX TRADING BOT
Multi-source sentiment analysis with deep learning and real-time processing
"""

import logging
import pandas as pd
import numpy as np
import json
import requests
import re
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import tweepy
from newsapi import NewsApiClient
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    FORUMS = "forums"
    PRESS_RELEASES = "press_releases"
    ECONOMIC_CALENDAR = "economic_calendar"

class SentimentType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

class LanguageModel(Enum):
    BERT = "bert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    FINBERT = "finbert"
    CUSTOM = "custom"

@dataclass
class SentimentConfig:
    """Advanced sentiment analysis configuration"""
    # Data sources
    enable_news: bool = True
    enable_twitter: bool = True
    enable_reddit: bool = True
    enable_forums: bool = True
    enable_press_releases: bool = True
    
    # API configurations
    newsapi_key: str = ""
    twitter_bearer_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    
    # Analysis methods
    use_bert: bool = True
    use_vader: bool = True
    use_textblob: bool = True
    use_custom_model: bool = True
    
    # Model settings
    language_model: LanguageModel = LanguageModel.FINBERT
    model_confidence_threshold: float = 0.7
    max_sequence_length: int = 512
    
    # Processing settings
    batch_size: int = 32
    max_concurrent_requests: int = 10
    cache_duration: int = 300  # seconds
    
    # Filtering
    min_confidence: float = 0.6
    relevance_threshold: float = 0.5
    spam_filter: bool = True
    
    # Real-time settings
    streaming_enabled: bool = True
    update_frequency: int = 60  # seconds

@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result"""
    source: SentimentSource
    text: str
    sentiment_score: float
    confidence: float
    sentiment_type: SentimentType
    keywords: List[str]
    entities: Dict[str, List[str]]
    timestamp: datetime
    metadata: Dict[str, Any]
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregateSentiment:
    """Aggregated sentiment across multiple sources"""
    symbol: str
    overall_score: float
    confidence: float
    sentiment_type: SentimentType
    source_breakdown: Dict[SentimentSource, float]
    trend_direction: str  # improving, deteriorating, stable
    volatility_impact: float
    timestamp: datetime
    recommendations: List[str]

class AdvancedSentimentAnalyzer:
    """
    Advanced multi-source sentiment analysis for Forex trading
    Combines NLP models with real-time data processing
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        
        # Initialize NLP models
        self._initialize_models()
        
        # Data storage
        self.sentiment_cache = defaultdict(lambda: deque(maxlen=1000))
        self.aggregate_sentiments = {}
        self.trend_analysis = defaultdict(lambda: deque(maxlen=500))
        
        # API clients
        self.news_client = None
        self.twitter_client = None
        self.reddit_client = None
        
        # Initialize APIs
        self._initialize_apis()
        
        # Thread safety
        self._lock = threading.RLock()
        self._analysis_lock = threading.Lock()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("AdvancedSentimentAnalyzer initialized")

    def _initialize_models(self):
        """Initialize NLP models for sentiment analysis"""
        try:
            logger.info("Initializing sentiment analysis models...")
            
            # VADER Sentiment Analyzer
            if self.config.use_vader:
                self.vader_analyzer = VaderAnalyzer()
                logger.info("VADER sentiment analyzer initialized")
            
            # TextBlob (no initialization needed)
            if self.config.use_textblob:
                logger.info("TextBlob sentiment analyzer ready")
            
            # Transformer models (BERT, RoBERTa, FinBERT)
            if self.config.use_bert:
                self._initialize_transformer_models()
            
            # Custom financial sentiment model
            if self.config.use_custom_model:
                self._initialize_custom_model()
            
            # TF-IDF Vectorizer for keyword extraction
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Topic modeling
            self.lda_model = LatentDirichletAllocation(
                n_components=5,
                random_state=42
            )
            
            # Entity recognition patterns
            self.entity_patterns = {
                'currency_pairs': re.compile(r'\b[A-Z]{3}/[A-Z]{3}\b'),
                'central_banks': re.compile(r'\b(ECB|Fed|BOE|BOJ|PBOC|RBA|BOC|SNB)\b', re.IGNORECASE),
                'economic_indicators': re.compile(r'\b(GDP|CPI|NFP|Unemployment|Inflation|Retail Sales)\b', re.IGNORECASE),
                'companies': re.compile(r'\b(Apple|Google|Amazon|Microsoft|Tesla|Meta)\b', re.IGNORECASE),
                'currencies': re.compile(r'\b(USD|EUR|GBP|JPY|AUD|CAD|CHF|NZD|CNY)\b', re.IGNORECASE),
                'trading_terms': re.compile(r'\b(bullish|bearish|rally|crash|support|resistance|breakout|breakdown)\b', re.IGNORECASE)
            }
            
            logger.info("All sentiment models initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _initialize_transformer_models(self):
        """Initialize transformer-based models"""
        try:
            model_map = {
                LanguageModel.BERT: "bert-base-uncased",
                LanguageModel.ROBERTA: "roberta-base",
                LanguageModel.DISTILBERT: "distilbert-base-uncased",
                LanguageModel.FINBERT: "ProsusAI/finbert"
            }
            
            model_name = model_map.get(self.config.language_model, "bert-base-uncased")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Create pipeline for easy inference
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                truncation=True,
                max_length=self.config.max_sequence_length
            )
            
            logger.info(f"Transformer model {model_name} initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Transformer model initialization failed: {e}")
            # Fallback to smaller model
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Fallback transformer model initialized")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                self.config.use_bert = False

    def _initialize_custom_model(self):
        """Initialize custom financial sentiment model"""
        try:
            # Load fine-tuned financial sentiment model
            self.custom_tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
            self.custom_model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
            
            self.custom_model.to(self.device)
            self.custom_model.eval()  # Set to evaluation mode
            
            logger.info("Custom financial sentiment model initialized")
            
        except Exception as e:
            logger.warning(f"Custom model initialization failed: {e}")
            self.config.use_custom_model = False

    def _initialize_apis(self):
        """Initialize API clients for data sources"""
        try:
            # NewsAPI
            if self.config.enable_news and self.config.newsapi_key:
                self.news_client = NewsApiClient(api_key=self.config.newsapi_key)
                logger.info("NewsAPI client initialized")
            
            # Twitter API
            if self.config.enable_twitter and self.config.twitter_bearer_token:
                self.twitter_client = tweepy.Client(bearer_token=self.config.twitter_bearer_token)
                logger.info("Twitter API client initialized")
            
            # Reddit API (PRAW initialization)
            if self.config.enable_reddit and self.config.reddit_client_id and self.config.reddit_client_secret:
                try:
                    import praw
                    self.reddit_client = praw.Reddit(
                        client_id=self.config.reddit_client_id,
                        client_secret=self.config.reddit_client_secret,
                        user_agent="forex_trading_bot"
                    )
                    logger.info("Reddit API client initialized")
                except ImportError:
                    logger.warning("PRAW not installed, Reddit disabled")
                    self.config.enable_reddit = False
                
        except Exception as e:
            logger.error(f"API initialization failed: {e}")

    def _start_background_tasks(self):
        """Start background data collection and analysis tasks"""
        # Real-time sentiment streaming
        if self.config.streaming_enabled:
            streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            streaming_thread.start()
        
        # Cache cleanup
        cleanup_thread = threading.Thread(target=self._cache_cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        # Trend analysis
        trend_thread = threading.Thread(target=self._trend_analysis_loop, daemon=True)
        trend_thread.start()
        
        # Database maintenance
        db_thread = threading.Thread(target=self._db_maintenance_loop, daemon=True)
        db_thread.start()

    def analyze_text(self, text: str, source: SentimentSource, 
                    symbol: str = None) -> SentimentResult:
        """
        Perform comprehensive sentiment analysis on text
        """
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            if not cleaned_text or len(cleaned_text) < 10:
                return self._create_neutral_result(text, source, symbol)
            
            # Multi-model sentiment analysis
            sentiment_scores = self._multi_model_analysis(cleaned_text)
            
            # Aggregate scores
            final_score, confidence = self._aggregate_scores(sentiment_scores)
            
            # Determine sentiment type
            sentiment_type = self._classify_sentiment(final_score, confidence)
            
            # Extract features
            keywords = self._extract_keywords(cleaned_text)
            entities = self._extract_entities(cleaned_text)
            
            # Create result
            result = SentimentResult(
                source=source,
                text=cleaned_text,
                sentiment_score=final_score,
                confidence=confidence,
                sentiment_type=sentiment_type,
                keywords=keywords,
                entities=entities,
                timestamp=datetime.now(),
                metadata={
                    'symbol': symbol,
                    'text_length': len(cleaned_text),
                    'model_scores': sentiment_scores,
                    'processing_time': datetime.now(),
                    'source': source.value
                }
            )
            
            # Cache result
            if symbol:
                self._cache_sentiment(symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_neutral_result(text, source, symbol)

    def _multi_model_analysis(self, text: str) -> Dict[str, float]:
        """Perform sentiment analysis using multiple models"""
        scores = {}
        
        try:
            # VADER Sentiment
            if self.config.use_vader:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                scores['vader'] = vader_scores['compound']
            
            # TextBlob Sentiment
            if self.config.use_textblob:
                blob = TextBlob(text)
                scores['textblob'] = blob.sentiment.polarity
            
            # Transformer-based sentiment
            if self.config.use_bert:
                bert_score = self._bert_analysis(text)
                scores['bert'] = bert_score
            
            # Custom financial sentiment
            if self.config.use_custom_model:
                custom_score = self._custom_model_analysis(text)
                scores['custom'] = custom_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Multi-model analysis failed: {e}")
            return {'fallback': 0.0}

    def _bert_analysis(self, text: str) -> float:
        """Perform BERT-based sentiment analysis"""
        try:
            # Truncate text if too long
            if len(text) > self.config.max_sequence_length:
                text = text[:self.config.max_sequence_length]
            
            result = self.sentiment_pipeline(text)[0]
            
            # Convert to numeric score
            if result['label'] == 'POSITIVE':
                return result['score']
            elif result['label'] == 'NEGATIVE':
                return -result['score']
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"BERT analysis failed: {e}")
            return 0.0

    def _custom_model_analysis(self, text: str) -> float:
        """Perform custom financial sentiment analysis"""
        try:
            inputs = self.custom_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.custom_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Model returns [negative, neutral, positive]
            positive_score = predictions[0][2].item()
            negative_score = predictions[0][0].item()
            neutral_score = predictions[0][1].item()
            
            # Weighted score giving more importance to strong sentiments
            financial_score = (positive_score * 1.0) + (negative_score * -1.0) + (neutral_score * 0.0)
            
            return financial_score
            
        except Exception as e:
            logger.error(f"Custom model analysis failed: {e}")
            return 0.0

    def _aggregate_scores(self, scores: Dict[str, float]) -> Tuple[float, float]:
        """Aggregate scores from multiple models with confidence"""
        if not scores:
            return 0.0, 0.0
        
        # Weighted average based on model reliability for financial context
        weights = {
            'custom': 0.5,    # Financial-specific model (highest weight)
            'bert': 0.3,      # General transformer
            'vader': 0.15,    # Rule-based
            'textblob': 0.05  # Simple statistical
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        valid_scores = []
        
        for model, score in scores.items():
            weight = weights.get(model, 0.1)
            weighted_sum += score * weight
            total_weight += weight
            valid_scores.append(score)
        
        if total_weight == 0:
            return 0.0, 0.0
        
        final_score = weighted_sum / total_weight
        
        # Confidence based on score agreement and number of models
        if len(valid_scores) > 1:
            score_agreement = 1.0 - (np.std(valid_scores) / 2.0)  # Normalize
            model_count_confidence = min(1.0, len(valid_scores) / 4.0)  # More models = more confidence
            confidence = (score_agreement + model_count_confidence) / 2.0
        else:
            confidence = 0.3  # Low confidence for single model
        
        confidence = max(0.1, min(1.0, confidence))
        
        return final_score, confidence

    def _classify_sentiment(self, score: float, confidence: float) -> SentimentType:
        """Classify sentiment based on score and confidence"""
        if confidence < self.config.min_confidence:
            return SentimentType.NEUTRAL
        
        # More conservative thresholds for financial sentiment
        if score > 0.25:
            return SentimentType.BULLISH
        elif score < -0.25:
            return SentimentType.BEARISH
        elif abs(score) > 0.15:
            return SentimentType.VOLATILE
        else:
            return SentimentType.NEUTRAL

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        try:
            # Enhanced keyword extraction for financial context
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            if len(words) < 5:
                return []
            
            # Financial stop words to exclude
            financial_stop_words = {
                'this', 'that', 'with', 'have', 'from', 'they', 'what', 'when',
                'where', 'which', 'would', 'could', 'should', 'about', 'after'
            }
            
            # Financial context words to prioritize
            financial_priority_words = {
                'bullish', 'bearish', 'rally', 'crash', 'surge', 'plunge', 'soar', 'tumble',
                'inflation', 'deflation', 'interest', 'rates', 'fed', 'ecb', 'boj', 'boe',
                'gdp', 'cpi', 'nfp', 'unemployment', 'retail', 'sales', 'manufacturing',
                'currency', 'forex', 'trading', 'market', 'price', 'volatility', 'liquidity',
                'support', 'resistance', 'breakout', 'breakdown', 'trend', 'momentum'
            }
            
            word_freq = defaultdict(int)
            for word in words:
                if (word not in financial_stop_words and 
                    len(word) > 3 and 
                    word.isalpha()):
                    word_freq[word] += 1
            
            # Boost scores for financial priority words
            for word in word_freq:
                if word in financial_priority_words:
                    word_freq[word] *= 2
            
            # Return top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
            return [kw[0] for kw in keywords if kw[1] > 1]  # Only include words appearing more than once
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text"""
        entities = {}
        
        try:
            for entity_type, pattern in self.entity_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    # Clean and deduplicate matches
                    cleaned_matches = list(set([match.upper() if entity_type in ['currency_pairs', 'currencies'] else match for match in matches]))
                    entities[entity_type] = cleaned_matches
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {}

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        try:
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            
            # Remove user mentions and hashtags but keep the text
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remove special characters but keep financial symbols and basic punctuation
            text = re.sub(r'[^\w\s$%\.\-/&+]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Convert to lowercase but preserve currency pairs (like EUR/USD)
            lines = text.split()
            processed_lines = []
            for line in lines:
                if re.match(r'^[A-Z]{3}/[A-Z]{3}$', line):
                    processed_lines.append(line)
                else:
                    processed_lines.append(line.lower())
            
            text = ' '.join(processed_lines)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return ""

    def _create_neutral_result(self, text: str, source: SentimentSource, 
                             symbol: str = None) -> SentimentResult:
        """Create a neutral sentiment result for fallback"""
        return SentimentResult(
            source=source,
            text=text[:100] + "..." if len(text) > 100 else text,
            sentiment_score=0.0,
            confidence=0.1,
            sentiment_type=SentimentType.NEUTRAL,
            keywords=[],
            entities={},
            timestamp=datetime.now(),
            metadata={
                'symbol': symbol,
                'fallback_reason': 'insufficient_text',
                'processing_time': datetime.now(),
                'source': source.value
            }
        )

    def _cache_sentiment(self, symbol: str, result: SentimentResult):
        """Cache sentiment result for trend analysis"""
        with self._lock:
            self.sentiment_cache[symbol].append(result)
            
            # Also store in trend analysis for technical analysis
            self.trend_analysis[symbol].append({
                'timestamp': result.timestamp,
                'score': result.sentiment_score,
                'confidence': result.confidence,
                'source': result.source.value
            })

    async def fetch_news_sentiment(self, symbol: str, query: str = None, 
                                 hours: int = 24) -> List[SentimentResult]:
        """Fetch and analyze news sentiment"""
        if not self.news_client:
            logger.warning("News client not initialized")
            return []
        
        try:
            # Calculate time range
            from_date = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d')
            
            # Build query
            search_query = query or f"{symbol} OR forex OR currency OR {symbol.replace('/', ' ')}"
            
            # Fetch news articles
            articles = self.news_client.get_everything(
                q=search_query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=50
            )
            
            # Analyze articles
            sentiment_results = []
            for article in articles.get('articles', []):
                content = f"{article.get('title', '')}. {article.get('description', '')}"
                
                if len(content) > 50:  # Minimum content length
                    result = self.analyze_text(content, SentimentSource.NEWS, symbol)
                    result.raw_data = {
                        'title': article.get('title'),
                        'url': article.get('url'),
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'author': article.get('author')
                    }
                    sentiment_results.append(result)
            
            logger.info(f"Fetched {len(sentiment_results)} news sentiment results for {symbol}")
            return sentiment_results
            
        except Exception as e:
            logger.error(f"News sentiment fetch failed: {e}")
            return []

    async def fetch_twitter_sentiment(self, symbol: str, 
                                    tweet_count: int = 100) -> List[SentimentResult]:
        """Fetch and analyze Twitter sentiment"""
        if not self.twitter_client:
            logger.warning("Twitter client not initialized")
            return []
        
        try:
            # Build query for financial tweets
            base_symbol = symbol.replace('/', '').replace('USD', '')
            query = f"#{symbol} OR ${symbol} OR {symbol} OR #{base_symbol} OR ${base_symbol} -is:retweet lang:en"
            
            # Fetch tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(tweet_count, 100),
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations']
            )
            
            sentiment_results = []
            if tweets and tweets.data:
                for tweet in tweets.data:
                    result = self.analyze_text(tweet.text, SentimentSource.SOCIAL_MEDIA, symbol)
                    result.raw_data = {
                        'tweet_id': tweet.id,
                        'author_id': tweet.author_id,
                        'created_at': tweet.created_at,
                        'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                        'like_count': tweet.public_metrics.get('like_count', 0),
                        'reply_count': tweet.public_metrics.get('reply_count', 0),
                        'quote_count': tweet.public_metrics.get('quote_count', 0)
                    }
                    sentiment_results.append(result)
            
            logger.info(f"Fetched {len(sentiment_results)} Twitter sentiment results for {symbol}")
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Twitter sentiment fetch failed: {e}")
            return []

    async def fetch_reddit_sentiment(self, symbol: str, 
                                   post_count: int = 50) -> List[SentimentResult]:
        """Fetch and analyze Reddit sentiment"""
        if not self.reddit_client:
            logger.warning("Reddit client not initialized")
            return []
        
        try:
            sentiment_results = []
            subreddits = ['Forex', 'trading', 'investing', 'stocks', 'economics']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Search for posts related to the symbol
                    search_query = f"{symbol} OR {symbol.replace('/', ' ')} OR forex"
                    
                    for post in subreddit.search(search_query, limit=post_count//len(subreddits)):
                        content = f"{post.title}. {post.selftext}"
                        
                        if len(content) > 30:
                            result = self.analyze_text(content, SentimentSource.FORUMS, symbol)
                            result.raw_data = {
                                'post_id': post.id,
                                'subreddit': subreddit_name,
                                'title': post.title,
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'created_utc': post.created_utc,
                                'url': post.url
                            }
                            sentiment_results.append(result)
                            
                except Exception as subreddit_error:
                    logger.error(f"Error fetching from subreddit {subreddit_name}: {subreddit_error}")
                    continue
            
            logger.info(f"Fetched {len(sentiment_results)} Reddit sentiment results for {symbol}")
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Reddit sentiment fetch failed: {e}")
            return []

    def get_aggregate_sentiment(self, symbol: str, 
                              lookback_hours: int = 24) -> AggregateSentiment:
        """
        Get aggregated sentiment across all sources for a symbol
        """
        try:
            with self._lock:
                # Get recent sentiment results
                recent_results = [
                    result for result in self.sentiment_cache[symbol]
                    if datetime.now() - result.timestamp <= timedelta(hours=lookback_hours)
                ]
                
                if not recent_results:
                    return self._create_neutral_aggregate(symbol)
                
                # Calculate aggregate metrics
                source_scores = defaultdict(list)
                source_confidences = defaultdict(list)
                all_scores = []
                all_confidences = []
                
                for result in recent_results:
                    source_scores[result.source].append(result.sentiment_score)
                    source_confidences[result.source].append(result.confidence)
                    all_scores.append(result.sentiment_score)
                    all_confidences.append(result.confidence)
                
                # Calculate overall score (weighted by confidence and source reliability)
                weighted_sum = 0.0
                total_weight = 0.0
                
                # Source reliability weights (news is most reliable for Forex)
                source_weights = {
                    SentimentSource.NEWS: 0.4,
                    SentimentSource.PRESS_RELEASES: 0.3,
                    SentimentSource.SOCIAL_MEDIA: 0.2,
                    SentimentSource.FORUMS: 0.1,
                    SentimentSource.ECONOMIC_CALENDAR: 0.3
                }
                
                for result in recent_results:
                    source_weight = source_weights.get(result.source, 0.1)
                    confidence_weight = result.confidence
                    combined_weight = source_weight * confidence_weight
                    
                    weighted_sum += result.sentiment_score * combined_weight
                    total_weight += combined_weight
                
                overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
                
                # Calculate source breakdown (average scores per source)
                source_breakdown = {}
                for source, scores in source_scores.items():
                    if scores:
                        # Weight source average by number of samples and confidence
                        avg_confidence = np.mean(source_confidences[source])
                        sample_weight = min(1.0, len(scores) / 20.0)  # Normalize by sample size
                        source_breakdown[source] = np.mean(scores) * avg_confidence * sample_weight
                
                # Calculate overall confidence
                volume_confidence = min(1.0, len(recent_results) / 100.0)  # Normalize by volume
                agreement_confidence = 1.0 - (np.std(all_scores) / 2.0) if len(all_scores) > 1 else 0.5
                avg_source_confidence = np.mean(all_confidences) if all_confidences else 0.5
                
                confidence = (volume_confidence + agreement_confidence + avg_source_confidence) / 3.0
                confidence = max(0.1, min(1.0, confidence))
                
                # Determine sentiment type
                sentiment_type = self._classify_sentiment(overall_score, confidence)
                
                # Calculate trend
                trend_direction = self._calculate_trend(symbol, lookback_hours)
                
                # Calculate volatility impact
                volatility_impact = self._calculate_volatility_impact(all_scores)
                
                # Generate recommendations
                recommendations = self._generate_recommendations(
                    overall_score, sentiment_type, trend_direction, volatility_impact, confidence
                )
                
                aggregate = AggregateSentiment(
                    symbol=symbol,
                    overall_score=overall_score,
                    confidence=confidence,
                    sentiment_type=sentiment_type,
                    source_breakdown=source_breakdown,
                    trend_direction=trend_direction,
                    volatility_impact=volatility_impact,
                    timestamp=datetime.now(),
                    recommendations=recommendations
                )
                
                self.aggregate_sentiments[symbol] = aggregate
                return aggregate
                
        except Exception as e:
            logger.error(f"Aggregate sentiment calculation failed: {e}")
            return self._create_neutral_aggregate(symbol)

    def _calculate_trend(self, symbol: str, lookback_hours: int) -> str:
        """Calculate sentiment trend direction"""
        try:
            # Get historical sentiment scores with timestamps
            recent_results = [
                (result.timestamp, result.sentiment_score) 
                for result in self.sentiment_cache[symbol]
                if datetime.now() - result.timestamp <= timedelta(hours=lookback_hours)
            ]
            
            if len(recent_results) < 10:
                return "stable"
            
            # Sort by timestamp and split into time windows
            recent_results.sort(key=lambda x: x[0])
            
            # Create 3 time windows
            total_results = len(recent_results)
            window_size = max(1, total_results // 3)
            windows = [recent_results[i:i + window_size] for i in range(0, total_results, window_size)]
            
            if len(windows) < 2:
                return "stable"
            
            # Calculate weighted average scores for each window (recent windows have more weight)
            window_scores = []
            weights = [0.2, 0.3, 0.5]  # Weights for [oldest, middle, newest]
            
            for i, window in enumerate(windows[:3]):  # Use up to 3 most recent windows
                if window:
                    window_score = np.mean([score for _, score in window])
                    weighted_score = window_score * weights[i]
                    window_scores.append(weighted_score)
            
            # Calculate trend from weighted scores
            if len(window_scores) >= 2:
                # Compare most recent window to previous windows
                recent_score = window_scores[-1]
                previous_avg = np.mean(window_scores[:-1]) if len(window_scores) > 1 else window_scores[0]
                
                trend_magnitude = recent_score - previous_avg
                
                if trend_magnitude > 0.15:
                    return "improving"
                elif trend_magnitude < -0.15:
                    return "deteriorating"
                elif abs(trend_magnitude) < 0.05:
                    return "stable"
                else:
                    return "stable"  # Small changes are considered stable
            
            return "stable"
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return "stable"

    def _calculate_volatility_impact(self, scores: List[float]) -> float:
        """Calculate potential volatility impact from sentiment"""
        if len(scores) < 5:
            return 0.0
        
        try:
            # Calculate basic volatility
            score_volatility = np.std(scores)
            
            # Count extreme sentiments (strong bullish/bearish)
            strong_bullish = sum(1 for score in scores if score > 0.5)
            strong_bearish = sum(1 for score in scores if score < -0.5)
            extreme_ratio = (strong_bullish + strong_bearish) / len(scores)
            
            # Calculate sentiment polarization
            positive_scores = [s for s in scores if s > 0]
            negative_scores = [s for s in scores if s < 0]
            
            polarization = 0.0
            if positive_scores and negative_scores:
                avg_positive = np.mean(positive_scores)
                avg_negative = np.mean(negative_scores)
                polarization = abs(avg_positive - avg_negative) / 2.0
            
            # Combined volatility impact (weighted average)
            volatility_impact = (
                score_volatility * 0.4 +
                extreme_ratio * 0.3 +
                polarization * 0.3
            )
            
            return min(1.0, max(0.0, volatility_impact))
            
        except Exception as e:
            logger.error(f"Volatility impact calculation failed: {e}")
            return 0.5  # Default medium volatility

    def _generate_recommendations(self, overall_score: float, sentiment_type: SentimentType,
                                trend_direction: str, volatility_impact: float, 
                                confidence: float) -> List[str]:
        """Generate trading recommendations based on sentiment analysis"""
        recommendations = []
        
        # Confidence-based warnings
        if confidence < 0.3:
            recommendations.append("Low confidence in sentiment analysis - verify with technical analysis")
        elif confidence < 0.6:
            recommendations.append("Moderate confidence - use smaller position sizes")
        
        # Sentiment-based recommendations
        if sentiment_type == SentimentType.BULLISH:
            if trend_direction == "improving" and confidence > 0.6:
                recommendations.append("Strong bullish sentiment trending upward - favorable for long positions")
            elif trend_direction == "deteriorating":
                recommendations.append("Bullish sentiment but trend weakening - monitor for reversal signals")
            else:
                recommendations.append("Bullish sentiment detected - consider long opportunities with proper risk management")
        
        elif sentiment_type == SentimentType.BEARISH:
            if trend_direction == "deteriorating" and confidence > 0.6:
                recommendations.append("Strong bearish sentiment trending downward - favorable for short positions")
            elif trend_direction == "improving":
                recommendations.append("Bearish sentiment but trend improving - monitor for reversal signals")
            else:
                recommendations.append("Bearish sentiment detected - consider short opportunities with proper risk management")
        
        elif sentiment_type == SentimentType.VOLATILE:
            recommendations.append("High sentiment volatility - market uncertainty detected")
            recommendations.append("Consider range-bound strategies or reduce position sizes")
        
        # Volatility-based recommendations
        if volatility_impact > 0.7:
            recommendations.append("Very high sentiment volatility - use wider stops and smaller positions")
            recommendations.append("Consider hedging strategies to manage risk")
        elif volatility_impact > 0.5:
            recommendations.append("Elevated sentiment volatility - adjust position sizing accordingly")
        
        # Trend-based recommendations
        if trend_direction == "improving" and sentiment_type != SentimentType.BEARISH:
            if confidence > 0.6:
                recommendations.append("Sentiment trend improving - favorable for trend-following strategies")
        elif trend_direction == "deteriorating" and sentiment_type != SentimentType.BULLISH:
            recommendations.append("Sentiment trend deteriorating - consider contrarian strategies with caution")
        
        # Risk management recommendations
        if abs(overall_score) > 0.6 and confidence > 0.7:
            recommendations.append("Strong sentiment signal detected - consider increasing position size moderately")
        elif abs(overall_score) < 0.2:
            recommendations.append("Weak sentiment signal - maintain standard position sizes")
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append("Neutral market sentiment - maintain current strategy with standard monitoring")
        
        return recommendations[:4]  # Return top 4 most relevant recommendations

    def _create_neutral_aggregate(self, symbol: str) -> AggregateSentiment:
        """Create neutral aggregate sentiment for fallback"""
        return AggregateSentiment(
            symbol=symbol,
            overall_score=0.0,
            confidence=0.1,
            sentiment_type=SentimentType.NEUTRAL,
            source_breakdown={},
            trend_direction="stable",
            volatility_impact=0.0,
            timestamp=datetime.now(),
            recommendations=[
                "Insufficient data for reliable sentiment analysis", 
                "Collecting more market data...",
                "Use technical analysis as primary decision tool"
            ]
        )

    def _streaming_loop(self):
        """Background loop for real-time sentiment streaming"""
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD']
        
        while True:
            try:
                # Fetch real-time sentiment for all symbols
                for symbol in symbols:
                    try:
                        asyncio.run(self._update_real_time_sentiment(symbol))
                    except Exception as e:
                        logger.error(f"Real-time update failed for {symbol}: {e}")
                
                logger.debug(f"Completed real-time sentiment update cycle for {len(symbols)} symbols")
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Streaming loop failed: {e}")
                time.sleep(30)  # Wait before retrying

    async def _update_real_time_sentiment(self, symbol: str):
        """Update real-time sentiment for a symbol"""
        try:
            # Fetch from multiple sources concurrently
            tasks = []
            
            if self.config.enable_news:
                tasks.append(self.fetch_news_sentiment(symbol, hours=1))
            
            if self.config.enable_twitter:
                tasks.append(self.fetch_twitter_sentiment(symbol, tweet_count=30))
            
            if self.config.enable_reddit:
                tasks.append(self.fetch_reddit_sentiment(symbol, post_count=20))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            successful_fetches = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Sentiment fetch task {i} failed: {result}")
                elif isinstance(result, list):
                    successful_fetches += 1
            
            logger.debug(f"Real-time update for {symbol}: {successful_fetches}/{len(tasks)} sources successful")
            
        except Exception as e:
            logger.error(f"Real-time sentiment update failed for {symbol}: {e}")

    def _cache_cleanup_loop(self):
        """Background loop for cache cleanup"""
        while True:
            try:
                with self._lock:
                    current_time = datetime.now()
                    cleanup_threshold = timedelta(days=2)  # Keep 2 days of data
                    
                    for symbol in list(self.sentiment_cache.keys()):
                        # Remove old entries
                        self.sentiment_cache[symbol] = deque(
                            [r for r in self.sentiment_cache[symbol] 
                             if current_time - r.timestamp <= cleanup_threshold],
                            maxlen=1000
                        )
                    
                    # Clean aggregate sentiments older than 1 hour
                    expired_symbols = []
                    for symbol, aggregate in self.aggregate_sentiments.items():
                        if current_time - aggregate.timestamp > timedelta(hours=1):
                            expired_symbols.append(symbol)
                    
                    for symbol in expired_symbols:
                        del self.aggregate_sentiments[symbol]
                
                logger.debug("Cache cleanup completed")
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def _trend_analysis_loop(self):
        """Background loop for trend analysis"""
        while True:
            try:
                symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
                
                for symbol in symbols:
                    try:
                        # Update aggregate sentiment (which includes trend analysis)
                        aggregate = self.get_aggregate_sentiment(symbol)
                        
                        # Log significant trend changes
                        if aggregate.confidence > 0.6 and abs(aggregate.overall_score) > 0.4:
                            logger.info(
                                f"Strong sentiment for {symbol}: {aggregate.sentiment_type.value} "
                                f"(score: {aggregate.overall_score:.3f}, confidence: {aggregate.confidence:.3f})"
                            )
                            
                    except Exception as e:
                        logger.error(f"Trend analysis failed for {symbol}: {e}")
                        continue
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Trend analysis loop failed: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _db_maintenance_loop(self):
        """Background loop for database maintenance"""
        while True:
            try:
                # This would handle database cleanup and optimization
                # For now, just a placeholder for future database functionality
                time.sleep(7200)  # Run every 2 hours
            except Exception as e:
                logger.error(f"DB maintenance loop failed: {e}")
                time.sleep(3600)

    def get_sentiment_alert(self, symbol: str, threshold: float = 0.7) -> Optional[Dict]:
        """Get sentiment alert if significant change detected"""
        try:
            aggregate = self.get_aggregate_sentiment(symbol)
            
            if aggregate.confidence < 0.5:
                return None
            
            # Check for significant sentiment or high volatility
            if (abs(aggregate.overall_score) > threshold or 
                aggregate.volatility_impact > 0.7 or
                aggregate.trend_direction in ["improving", "deteriorating"]):
                
                alert_level = "HIGH" if (abs(aggregate.overall_score) > 0.8 or aggregate.volatility_impact > 0.8) else "MEDIUM"
                
                return {
                    'symbol': symbol,
                    'alert_level': alert_level,
                    'sentiment_score': aggregate.overall_score,
                    'sentiment_type': aggregate.sentiment_type.value,
                    'confidence': aggregate.confidence,
                    'trend': aggregate.trend_direction,
                    'volatility_impact': aggregate.volatility_impact,
                    'recommendations': aggregate.recommendations,
                    'timestamp': aggregate.timestamp,
                    'source_breakdown': {k.value: v for k, v in aggregate.source_breakdown.items()}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Sentiment alert check failed: {e}")
            return None

    def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market sentiment overview"""
        try:
            symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD']
            overview = {
                'timestamp': datetime.now(),
                'market_sentiment': 'NEUTRAL',
                'symbol_breakdown': {},
                'key_insights': [],
                'risk_level': 'MEDIUM',
                'dominant_themes': [],
                'market_volatility': 'MODERATE'
            }
            
            sentiment_counts = defaultdict(int)
            total_confidence = 0.0
            volatility_scores = []
            
            for symbol in symbols:
                aggregate = self.get_aggregate_sentiment(symbol)
                overview['symbol_breakdown'][symbol] = {
                    'sentiment': aggregate.sentiment_type.value,
                    'score': aggregate.overall_score,
                    'confidence': aggregate.confidence,
                    'trend': aggregate.trend_direction,
                    'volatility_impact': aggregate.volatility_impact
                }
                
                sentiment_counts[aggregate.sentiment_type] += 1
                total_confidence += aggregate.confidence
                volatility_scores.append(aggregate.volatility_impact)
            
            # Determine overall market sentiment
            total_symbols = len(symbols)
            if sentiment_counts[SentimentType.BULLISH] >= total_symbols * 0.6:
                overview['market_sentiment'] = 'BULLISH'
            elif sentiment_counts[SentimentType.BEARISH] >= total_symbols * 0.6:
                overview['market_sentiment'] = 'BEARISH'
            elif (sentiment_counts[SentimentType.BULLISH] + sentiment_counts[SentimentType.BEARISH]) >= total_symbols * 0.7:
                overview['market_sentiment'] = 'MIXED'
            elif sentiment_counts[SentimentType.VOLATILE] >= total_symbols * 0.5:
                overview['market_sentiment'] = 'VOLATILE'
            
            # Calculate average confidence and volatility
            avg_confidence = total_confidence / len(symbols)
            avg_volatility = np.mean(volatility_scores) if volatility_scores else 0.5
            
            # Set risk level based on confidence and volatility
            if avg_confidence > 0.7 and avg_volatility < 0.4:
                overview['risk_level'] = 'LOW'
            elif avg_confidence < 0.4 or avg_volatility > 0.7:
                overview['risk_level'] = 'HIGH'
            
            # Set market volatility
            if avg_volatility > 0.7:
                overview['market_volatility'] = 'HIGH'
            elif avg_volatility > 0.5:
                overview['market_volatility'] = 'ELEVATED'
            elif avg_volatility < 0.3:
                overview['market_volatility'] = 'LOW'
            
            # Generate insights and themes
            overview['key_insights'] = self._generate_market_insights(overview['symbol_breakdown'])
            overview['dominant_themes'] = self._extract_market_themes()
            
            return overview
            
        except Exception as e:
            logger.error(f"Market overview generation failed: {e}")
            return {
                'timestamp': datetime.now(),
                'market_sentiment': 'UNKNOWN',
                'symbol_breakdown': {},
                'key_insights': ['Data collection in progress', 'Use caution until system stabilizes'],
                'risk_level': 'HIGH',
                'dominant_themes': [],
                'market_volatility': 'UNKNOWN'
            }

    def _generate_market_insights(self, symbol_breakdown: Dict) -> List[str]:
        """Generate market insights from symbol breakdown"""
        insights = []
        
        try:
            # Count sentiment types
            sentiment_counts = defaultdict(int)
            trending_symbols = []
            volatile_symbols = []
            
            for symbol, data in symbol_breakdown.items():
                sentiment_counts[data['sentiment']] += 1
                
                if data['trend'] in ['improving', 'deteriorating']:
                    trending_symbols.append(symbol)
                
                if data['volatility_impact'] > 0.6:
                    volatile_symbols.append(symbol)
            
            # Generate insights based on patterns
            total_symbols = len(symbol_breakdown)
            
            if sentiment_counts['bullish'] >= total_symbols * 0.7:
                insights.append("Strong bullish bias across most major currency pairs")
            elif sentiment_counts['bearish'] >= total_symbols * 0.7:
                insights.append("Strong bearish bias across most major currency pairs")
            
            if sentiment_counts['volatile'] >= total_symbols * 0.5:
                insights.append("High sentiment volatility detected across multiple pairs")
            
            if len(trending_symbols) >= 3:
                insights.append(f"Multiple pairs showing strong trend direction: {', '.join(trending_symbols[:3])}")
            
            if len(volatile_symbols) >= 2:
                insights.append(f"Elevated volatility in: {', '.join(volatile_symbols[:2])}")
            
            # USD strength/weakness analysis
            usd_pairs = [sym for sym in symbol_breakdown.keys() if 'USD' in sym]
            if usd_pairs:
                usd_sentiments = [symbol_breakdown[sym]['sentiment'] for sym in usd_pairs]
                bullish_usd = usd_sentiments.count('bullish')
                bearish_usd = usd_sentiments.count('bearish')
                
                if bullish_usd > bearish_usd * 1.5:
                    insights.append("USD showing strength across multiple pairs")
                elif bearish_usd > bullish_usd * 1.5:
                    insights.append("USD showing weakness across multiple pairs")
            
            # Default insight if none generated
            if not insights:
                insights.append("Market sentiment relatively balanced - monitor individual pair dynamics")
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Market insights generation failed: {e}")
            return ["Analyzing market sentiment patterns..."]

    def _extract_market_themes(self) -> List[str]:
        """Extract dominant market themes from recent sentiment data"""
        try:
            themes = []
            
            # Analyze recent sentiment results for common keywords and entities
            all_keywords = []
            all_entities = defaultdict(list)
            
            for symbol_results in self.sentiment_cache.values():
                for result in list(symbol_results)[-50:]:  # Last 50 results per symbol
                    all_keywords.extend(result.keywords)
                    for entity_type, entities in result.entities.items():
                        all_entities[entity_type].extend(entities)
            
            # Find most common themes
            if all_keywords:
                keyword_counts = defaultdict(int)
                for keyword in all_keywords:
                    keyword_counts[keyword] += 1
                
                top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Group related keywords into themes
                economic_terms = {'inflation', 'rates', 'fed', 'interest', 'gdp', 'cpi'}
                market_terms = {'volatility', 'trading', 'market', 'price', 'liquidity'}
                sentiment_terms = {'bullish', 'bearish', 'rally', 'crash', 'surge'}
                
                economic_count = sum(count for kw, count in top_keywords if kw in economic_terms)
                market_count = sum(count for kw, count in top_keywords if kw in market_terms)
                sentiment_count = sum(count for kw, count in top_keywords if kw in sentiment_terms)
                
                if economic_count > market_count and economic_count > sentiment_count:
                    themes.append("Economic Fundamentals Driven")
                elif market_count > economic_count and market_count > sentiment_count:
                    themes.append("Market Technicals Driven")
                elif sentiment_count > economic_count and sentiment_count > market_count:
                    themes.append("Sentiment Driven")
            
            # Analyze entity frequency
            if all_entities:
                for entity_type, entities in all_entities.items():
                    if entities:
                        entity_counts = defaultdict(int)
                        for entity in entities:
                            entity_counts[entity] += 1
                        
                        top_entity = max(entity_counts.items(), key=lambda x: x[1]) if entity_counts else None
                        if top_entity and top_entity[1] > 5:
                            themes.append(f"Focus on {entity_type.replace('_', ' ').title()}: {top_entity[0]}")
            
            return themes[:3]  # Return top 3 themes
            
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            return ["Analyzing market themes..."]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the sentiment analyzer"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'total_analyzed_texts': 0,
                'symbol_coverage': {},
                'model_performance': {},
                'data_freshness': {},
                'system_health': 'HEALTHY'
            }
            
            # Calculate total analyzed texts
            total_texts = 0
            for symbol_results in self.sentiment_cache.values():
                total_texts += len(symbol_results)
            metrics['total_analyzed_texts'] = total_texts
            
            # Symbol coverage
            for symbol, results in self.sentiment_cache.items():
                metrics['symbol_coverage'][symbol] = {
                    'total_texts': len(results),
                    'last_update': max([r.timestamp for r in results]) if results else None,
                    'sources': defaultdict(int)
                }
                for result in results:
                    metrics['symbol_coverage'][symbol]['sources'][result.source.value] += 1
            
            # Model performance (simplified - would be more complex in production)
            active_models = []
            if self.config.use_vader:
                active_models.append('VADER')
            if self.config.use_textblob:
                active_models.append('TextBlob')
            if self.config.use_bert:
                active_models.append('Transformer')
            if self.config.use_custom_model:
                active_models.append('Custom Financial')
            
            metrics['model_performance']['active_models'] = active_models
            metrics['model_performance']['total_models'] = len(active_models)
            
            # Data freshness
            current_time = datetime.now()
            for symbol, results in self.sentiment_cache.items():
                if results:
                    latest_timestamp = max(r.timestamp for r in results)
                    freshness = (current_time - latest_timestamp).total_seconds() / 60  # minutes
                    metrics['data_freshness'][symbol] = f"{freshness:.1f} minutes ago"
            
            # System health check
            error_count = 0
            warning_count = 0
            
            # Check if models are loaded
            if not active_models:
                metrics['system_health'] = 'CRITICAL'
                error_count += 1
            elif len(active_models) < 2:
                metrics['system_health'] = 'DEGRADED'
                warning_count += 1
            
            # Check data freshness
            stale_symbols = [sym for sym, fresh in metrics['data_freshness'].items() 
                           if float(fresh.split()[0]) > 120]  # More than 2 hours
            if stale_symbols:
                metrics['system_health'] = 'DEGRADED'
                warning_count += 1
            
            metrics['error_count'] = error_count
            metrics['warning_count'] = warning_count
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {
                'timestamp': datetime.now(),
                'system_health': 'UNKNOWN',
                'error': str(e)
            }

# Example usage and testing
def main():
    """Example usage of the AdvancedSentimentAnalyzer"""
    
    # Configuration
    config = SentimentConfig(
        enable_news=True,
        enable_twitter=True,
        enable_reddit=False,  # Disable Reddit for example
        use_bert=True,
        use_vader=True,
        use_textblob=True,
        use_custom_model=True,
        streaming_enabled=True,
        update_frequency=120  # 2 minutes for testing
    )
    
    # Initialize analyzer
    analyzer = AdvancedSentimentAnalyzer(config)
    
    # Test with sample text
    sample_texts = [
        "EUR/USD is showing strong bullish momentum after the ECB decision yesterday.",
        "The Fed's hawkish stance is putting pressure on USD pairs, particularly GBP/USD.",
        "Market volatility is increasing due to geopolitical tensions and oil price shocks.",
        "Technical analysis suggests support levels are holding for AUD/USD."
    ]
    
    print("=== Testing Sentiment Analysis ===")
    for i, text in enumerate(sample_texts, 1):
        result = analyzer.analyze_text(text, SentimentSource.NEWS, "EUR/USD")
        print(f"\nSample {i}:")
        print(f"Text: {text}")
        print(f"Sentiment: {result.sentiment_type.value} (Score: {result.sentiment_score:.3f})")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Keywords: {result.keywords}")
        print(f"Entities: {result.entities}")
    
    # Test aggregate sentiment
    print("\n=== Testing Aggregate Sentiment ===")
    aggregate = analyzer.get_aggregate_sentiment("EUR/USD", lookback_hours=1)
    print(f"Symbol: {aggregate.symbol}")
    print(f"Overall Score: {aggregate.overall_score:.3f}")
    print(f"Confidence: {aggregate.confidence:.3f}")
    print(f"Sentiment Type: {aggregate.sentiment_type.value}")
    print(f"Trend: {aggregate.trend_direction}")
    print(f"Volatility Impact: {aggregate.volatility_impact:.3f}")
    print("Recommendations:")
    for rec in aggregate.recommendations:
        print(f"  - {rec}")
    
    # Test market overview
    print("\n=== Testing Market Overview ===")
    overview = analyzer.get_market_overview()
    print(f"Market Sentiment: {overview['market_sentiment']}")
    print(f"Risk Level: {overview['risk_level']}")
    print(f"Market Volatility: {overview['market_volatility']}")
    print("Key Insights:")
    for insight in overview['key_insights']:
        print(f"  - {insight}")
    
    # Test performance metrics
    print("\n=== Testing Performance Metrics ===")
    metrics = analyzer.get_performance_metrics()
    print(f"System Health: {metrics['system_health']}")
    print(f"Total Analyzed Texts: {metrics['total_analyzed_texts']}")
    print(f"Active Models: {metrics['model_performance']['active_models']}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
"""
Advanced Sentiment Analyzer for FOREX TRADING BOT
Real-time sentiment analysis with multi-model ensemble and financial context
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
import sqlite3
import json
import warnings

# NLP Libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer

# Machine Learning
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FOREX_FACTORY = "forex_factory"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    SOCIAL_MEDIA = "social_media"
    PRESS_RELEASES = "press_releases"

class SentimentLabel(Enum):
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"

class AnalysisMethod(Enum):
    VADER = "vader"
    TEXTBLOB = "textblob"
    BERT = "bert"
    FINBERT = "finbert"
    ENSEMBLE = "ensemble"
    CUSTOM_ML = "custom_ml"

@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result"""
    text: str
    source: SentimentSource
    symbol: str
    overall_score: float
    confidence: float
    sentiment_label: SentimentLabel
    analysis_method: AnalysisMethod
    timestamp: datetime
    keyword_scores: Dict[str, float] = field(default_factory=dict)
    entity_mentions: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregateSentiment:
    """Aggregated sentiment across multiple analyses"""
    symbol: str
    timeframe: str
    overall_score: float
    confidence: float
    sentiment_label: SentimentLabel
    source_breakdown: Dict[SentimentSource, float]
    trend_direction: str
    volatility_impact: float
    recommendation: str
    timestamp: datetime

@dataclass
class SentimentConfig:
    """Configuration for sentiment analyzer"""
    # Model settings
    use_vader: bool = True
    use_textblob: bool = True
    use_bert: bool = True
    use_finbert: bool = True
    use_custom_model: bool = True
    
    # Analysis parameters
    confidence_threshold: float = 0.7
    min_text_length: int = 10
    max_text_length: int = 512
    
    # Financial context
    enable_financial_context: bool = True
    financial_lexicon_path: str = "data/financial_lexicon.json"
    
    # Real-time processing
    batch_size: int = 32
    max_concurrent_requests: int = 10
    cache_duration: int = 300  # seconds
    
    # Source weights
    source_weights: Dict[SentimentSource, float] = field(default_factory=lambda: {
        SentimentSource.BLOOMBERG: 0.9,
        SentimentSource.REUTERS: 0.9,
        SentimentSource.FOREX_FACTORY: 0.8,
        SentimentSource.NEWS: 0.7,
        SentimentSource.PRESS_RELEASES: 0.7,
        SentimentSource.REDDIT: 0.5,
        SentimentSource.TWITTER: 0.4,
        SentimentSource.SOCIAL_MEDIA: 0.3
    })

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analysis with multi-model ensemble and financial context
    """
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Data storage
        self.sentiment_cache = defaultdict(lambda: deque(maxlen=1000))
        self.aggregate_sentiments = {}
        self.performance_metrics = defaultdict(lambda: deque(maxlen=500))
        
        # Financial lexicon
        self.financial_lexicon = self._load_financial_lexicon()
        
        # Thread safety
        self._lock = threading.RLock()
        self._analysis_lock = threading.Lock()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("AdvancedSentimentAnalyzer initialized successfully")

    def _initialize_nlp(self):
        """Initialize NLP models and resources"""
        try:
            logger.info("Initializing NLP components...")
            
            # Download required NLTK data
            try:
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except Exception as e:
                logger.warning(f"NLTK downloads may have issues: {e}")
            
            # VADER Sentiment Analyzer
            if self.config.use_vader:
                self.vader_analyzer = VaderAnalyzer()
                # Enhance VADER with financial terms
                self._enhance_vader_lexicon()
                logger.info("VADER sentiment analyzer initialized")
            
            # TextBlob (no specific initialization needed)
            if self.config.use_textblob:
                logger.info("TextBlob sentiment analyzer ready")
            
            # BERT-based models
            if self.config.use_bert or self.config.use_finbert:
                self._initialize_transformer_models()
            
            # TF-IDF Vectorizer for custom ML
            if self.config.use_custom_model:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.custom_classifier = None
            
            # Stop words for preprocessing
            self.stop_words = set(stopwords.words('english'))
            
            # Financial entity patterns
            self.entity_patterns = {
                'currency_pairs': re.compile(r'\b[A-Z]{3}/[A-Z]{3}\b'),
                'central_banks': re.compile(r'\b(ECB|Federal Reserve|Fed|BOE|Bank of England|BOJ|Bank of Japan|PBOC|RBA|BOC|SNB)\b', re.IGNORECASE),
                'economic_indicators': re.compile(r'\b(GDP|CPI|PPI|NFP|unemployment|inflation|retail sales|manufacturing|services PMI)\b', re.IGNORECASE),
                'companies': re.compile(r'\b(Apple|Google|Amazon|Microsoft|Tesla|Meta|Facebook|Netflix|NVIDIA)\b', re.IGNORECASE),
                'currencies': re.compile(r'\b(USD|EUR|GBP|JPY|AUD|CAD|CHF|NZD|CNY|yuan|dollar|euro|pound|yen)\b', re.IGNORECASE)
            }
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"NLP initialization failed: {e}")
            raise
        
    def _initialize_ml_models(self):
        """Initialize machine learning models - ADDED METHOD"""
        try:
            logger.info("Initializing ML models...")
            
            # Custom ML model initialization (if needed)
            if self.config.use_custom_model:
                # Placeholder for custom model training/loading
                logger.info("Custom ML models placeholder initialized")
            
            logger.info("ML models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"ML models initialization failed: {e}")
            return False
    
    def _enhance_vader_lexicon(self):
        """Enhance VADER lexicon with financial terms"""
        financial_terms = {
            'bullish': 2.0,
            'bearish': -2.0,
            'rally': 1.5,
            'crash': -2.5,
            'surge': 1.8,
            'plunge': -2.2,
            'soar': 1.7,
            'tumble': -1.8,
            'inflation': -0.5,
            'deflation': -0.7,
            'hawkish': 0.8,
            'dovish': -0.8,
            'rate hike': -1.2,
            'rate cut': 0.8,
            'quantitative easing': 0.7,
            'tightening': -0.9,
            'recession': -2.0,
            'growth': 1.2,
            'expansion': 1.1,
            'contraction': -1.3
        }
        
        for word, score in financial_terms.items():
            self.vader_analyzer.lexicon.update({word: score})

    def _initialize_transformer_models(self):
        """Initialize transformer-based models"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Standard BERT for general sentiment
            if self.config.use_bert:
                self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
                self.bert_model.to(self.device)
                self.bert_model.eval()
                
                self.bert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="bert-base-uncased",
                    tokenizer="bert-base-uncased",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("BERT model initialized")
            
            # FinBERT for financial sentiment
            if self.config.use_finbert:
                try:
                    self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                    self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                    self.finbert_model.to(self.device)
                    self.finbert_model.eval()
                    logger.info("FinBERT model initialized")
                except Exception as e:
                    logger.warning(f"FinBERT initialization failed: {e}")
                    self.config.use_finbert = False
            
        except Exception as e:
            logger.error(f"Transformer models initialization failed: {e}")
            self.config.use_bert = False
            self.config.use_finbert = False

    def _load_financial_lexicon(self) -> Dict[str, float]:
        """Load financial sentiment lexicon"""
        default_lexicon = {
            # Positive financial terms
            "profit": 1.5, "gain": 1.4, "growth": 1.3, "surge": 2.0, "rally": 1.8,
            "bullish": 2.0, "optimistic": 1.2, "strong": 1.1, "recovery": 1.3,
            "expansion": 1.2, "outperform": 1.6, "beat": 1.4, "positive": 1.0,
            
            # Negative financial terms  
            "loss": -1.5, "decline": -1.3, "drop": -1.4, "fall": -1.3, "crash": -2.5,
            "bearish": -2.0, "pessimistic": -1.2, "weak": -1.1, "recession": -2.0,
            "contraction": -1.5, "underperform": -1.6, "miss": -1.4, "negative": -1.0,
            
            # Central bank terms
            "hawkish": -0.8, "dovish": 0.8, "tightening": -1.0, "easing": 0.8,
            "inflation": -0.7, "deflation": -0.5, "rates": -0.3, "hike": -1.2,
            
            # Economic indicators
            "unemployment": -1.0, "employment": 1.0, "gdp": 0.5, "retail": 0.3
        }
        
        try:
            if Path(self.config.financial_lexicon_path).exists():
                with open(self.config.financial_lexicon_path, 'r') as f:
                    custom_lexicon = json.load(f)
                    default_lexicon.update(custom_lexicon)
                logger.info("Custom financial lexicon loaded")
        except Exception as e:
            logger.warning(f"Custom lexicon loading failed: {e}")
        
        return default_lexicon

    def _start_background_tasks(self):
        """Start background processing tasks"""
        # Model performance monitoring
        perf_thread = threading.Thread(target=self._performance_monitoring_loop, daemon=True)
        perf_thread.start()
        
        # Cache cleanup
        cleanup_thread = threading.Thread(target=self._cache_cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        # Aggregate sentiment updates
        aggregate_thread = threading.Thread(target=self._aggregate_update_loop, daemon=True)
        aggregate_thread.start()

    def analyze_text(self, text: str, source: SentimentSource, symbol: str = None) -> SentimentResult:
        """
        Perform comprehensive sentiment analysis on text
        """
        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            if len(cleaned_text) < self.config.min_text_length:
                return self._create_neutral_result(text, source, symbol)
            
            # Multi-model analysis
            model_scores = self._multi_model_analysis(cleaned_text)
            
            # Apply financial context if enabled
            if self.config.enable_financial_context:
                financial_score = self._apply_financial_context(cleaned_text)
                model_scores['financial_context'] = financial_score
            
            # Ensemble scoring
            final_score, confidence = self._ensemble_scoring(model_scores, source)
            
            # Determine sentiment label
            sentiment_label = self._classify_sentiment(final_score, confidence)
            
            # Extract keywords and entities
            keyword_scores = self._extract_keyword_scores(cleaned_text)
            entity_mentions = self._extract_entities(cleaned_text)
            
            # Create result
            result = SentimentResult(
                text=cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
                source=source,
                symbol=symbol or "GLOBAL",
                overall_score=final_score,
                confidence=confidence,
                sentiment_label=sentiment_label,
                analysis_method=AnalysisMethod.ENSEMBLE,
                timestamp=datetime.now(),
                keyword_scores=keyword_scores,
                entity_mentions=entity_mentions,
                metadata={
                    'model_scores': model_scores,
                    'text_length': len(cleaned_text),
                    'processing_time': datetime.now()
                }
            )
            
            # Cache result
            self._cache_sentiment_result(result)
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._create_neutral_result(text, source, symbol)

    def _multi_model_analysis(self, text: str) -> Dict[str, float]:
        """Perform sentiment analysis using multiple models"""
        scores = {}
        
        try:
            # VADER analysis
            if self.config.use_vader:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                scores['vader'] = vader_scores['compound']
            
            # TextBlob analysis
            if self.config.use_textblob:
                blob = TextBlob(text)
                scores['textblob'] = blob.sentiment.polarity
            
            # BERT analysis
            if self.config.use_bert:
                bert_score = self._bert_analysis(text)
                scores['bert'] = bert_score
            
            # FinBERT analysis
            if self.config.use_finbert:
                finbert_score = self._finbert_analysis(text)
                scores['finbert'] = finbert_score
            
            # Custom ML analysis (if trained)
            if self.config.use_custom_model and self.custom_classifier is not None:
                custom_score = self._custom_ml_analysis(text)
                scores['custom_ml'] = custom_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Multi-model analysis failed: {e}")
            return {'fallback': 0.0}

    def _bert_analysis(self, text: str) -> float:
        """Perform BERT-based sentiment analysis"""
        try:
            # Truncate if too long
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
            
            result = self.bert_pipeline(text)[0]
            
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

    def _finbert_analysis(self, text: str) -> float:
        """Perform FinBERT financial sentiment analysis"""
        try:
            inputs = self.finbert_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.max_text_length,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT returns: [positive, negative, neutral]
            positive_score = predictions[0][0].item()
            negative_score = predictions[0][1].item()
            neutral_score = predictions[0][2].item()
            
            # Calculate weighted score favoring strong sentiments
            financial_score = (positive_score * 1.0) + (negative_score * -1.0) + (neutral_score * 0.0)
            
            return financial_score
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return 0.0

    def _custom_ml_analysis(self, text: str) -> float:
        """Perform custom ML model analysis"""
        try:
            # This would use a custom-trained model
            # For now, return a simple TF-IDF based score
            features = self.tfidf_vectorizer.transform([text])
            if hasattr(self.custom_classifier, 'predict_proba'):
                probabilities = self.custom_classifier.predict_proba(features)[0]
                return probabilities[1] - probabilities[0]  # Positive - Negative
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Custom ML analysis failed: {e}")
            return 0.0

    def _apply_financial_context(self, text: str) -> float:
        """Apply financial context scoring"""
        try:
            words = word_tokenize(text.lower())
            financial_score = 0.0
            financial_terms_found = 0
            
            for word in words:
                if word in self.financial_lexicon:
                    financial_score += self.financial_lexicon[word]
                    financial_terms_found += 1
            
            if financial_terms_found > 0:
                return financial_score / financial_terms_found
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Financial context application failed: {e}")
            return 0.0

    def _ensemble_scoring(self, model_scores: Dict[str, float], source: SentimentSource) -> Tuple[float, float]:
        """Combine scores from multiple models with weighting"""
        if not model_scores:
            return 0.0, 0.0
        
        # Model weights (financial models weighted higher)
        model_weights = {
            'finbert': 0.25,
            'custom_ml': 0.20,
            'bert': 0.15,
            'vader': 0.15,
            'textblob': 0.10,
            'financial_context': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        valid_scores = []
        
        for model, score in model_scores.items():
            weight = model_weights.get(model, 0.05)
            weighted_sum += score * weight
            total_weight += weight
            valid_scores.append(score)
        
        if total_weight == 0:
            return 0.0, 0.0
        
        final_score = weighted_sum / total_weight
        
        # Apply source weight
        source_weight = self.config.source_weights.get(source, 0.5)
        final_score *= source_weight
        
        # Calculate confidence based on agreement and number of models
        if len(valid_scores) > 1:
            agreement = 1.0 - (np.std(valid_scores) / 2.0)
            model_count_confidence = min(1.0, len(valid_scores) / 5.0)
            confidence = (agreement + model_count_confidence) / 2.0
        else:
            confidence = 0.3
        
        confidence = max(0.1, min(1.0, confidence))
        
        return final_score, confidence

    def _classify_sentiment(self, score: float, confidence: float) -> SentimentLabel:
        """Classify sentiment based on score and confidence"""
        if confidence < self.config.confidence_threshold:
            return SentimentLabel.NEUTRAL
        
        if score > 0.6:
            return SentimentLabel.STRONGLY_BULLISH
        elif score > 0.2:
            return SentimentLabel.BULLISH
        elif score < -0.6:
            return SentimentLabel.STRONGLY_BEARISH
        elif score < -0.2:
            return SentimentLabel.BEARISH
        else:
            return SentimentLabel.NEUTRAL

    def _extract_keyword_scores(self, text: str) -> Dict[str, float]:
        """Extract and score important keywords"""
        try:
            words = word_tokenize(text.lower())
            word_scores = {}
            
            for word in words:
                if (len(word) > 3 and 
                    word.isalpha() and 
                    word not in self.stop_words and
                    word in self.financial_lexicon):
                    word_scores[word] = self.financial_lexicon[word]
            
            # Return top 10 scoring keywords
            sorted_keywords = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            return dict(sorted_keywords)
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return {}

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities from text"""
        entities = {}
        
        try:
            for entity_type, pattern in self.entity_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    # Clean and deduplicate
                    cleaned_matches = list(set([m.upper() if entity_type in ['currency_pairs', 'currencies'] else m for m in matches]))
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
            
            # Remove user mentions and hashtags but keep text
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remove special characters but keep financial symbols
            text = re.sub(r'[^\w\s$%\.\-/&+]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Convert to lowercase but preserve currency pairs
            lines = text.split()
            processed_lines = []
            for line in lines:
                if re.match(r'^[A-Z]{3}/[A-Z]{3}$', line):
                    processed_lines.append(line)
                else:
                    processed_lines.append(line.lower())
            
            return ' '.join(processed_lines).strip()
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return ""

    def _create_neutral_result(self, text: str, source: SentimentSource, symbol: str = None) -> SentimentResult:
        """Create a neutral sentiment result for fallback"""
        return SentimentResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            source=source,
            symbol=symbol or "GLOBAL",
            overall_score=0.0,
            confidence=0.1,
            sentiment_label=SentimentLabel.NEUTRAL,
            analysis_method=AnalysisMethod.ENSEMBLE,
            timestamp=datetime.now(),
            metadata={'fallback_reason': 'insufficient_text'}
        )

    def _cache_sentiment_result(self, result: SentimentResult):
        """Cache sentiment result"""
        with self._lock:
            key = result.symbol
            self.sentiment_cache[key].append(result)

    def _update_performance_metrics(self, result: SentimentResult):
        """Update performance tracking metrics"""
        metric_key = f"{result.source.value}_{result.analysis_method.value}"
        self.performance_metrics[metric_key].append({
            'timestamp': result.timestamp,
            'confidence': result.confidence,
            'processing_time': datetime.now()
        })

    def get_aggregate_sentiment(self, symbol: str, timeframe: str = "1h") -> AggregateSentiment:
        """Get aggregated sentiment for a symbol"""
        try:
            with self._lock:
                # Get relevant time window
                if timeframe == "1h":
                    cutoff = datetime.now() - timedelta(hours=1)
                elif timeframe == "4h":
                    cutoff = datetime.now() - timedelta(hours=4)
                elif timeframe == "24h":
                    cutoff = datetime.now() - timedelta(hours=24)
                else:
                    cutoff = datetime.now() - timedelta(hours=1)
                
                # Filter recent results
                recent_results = [
                    r for r in self.sentiment_cache.get(symbol, [])
                    if r.timestamp >= cutoff
                ]
                
                if not recent_results:
                    return self._create_neutral_aggregate(symbol, timeframe)
                
                # Calculate aggregate metrics
                source_scores = defaultdict(list)
                all_scores = []
                all_confidences = []
                
                for result in recent_results:
                    source_scores[result.source].append(result.overall_score)
                    all_scores.append(result.overall_score)
                    all_confidences.append(result.confidence)
                
                # Weighted average score
                weighted_sum = 0.0
                total_weight = 0.0
                
                for result in recent_results:
                    source_weight = self.config.source_weights.get(result.source, 0.5)
                    confidence_weight = result.confidence
                    combined_weight = source_weight * confidence_weight
                    
                    weighted_sum += result.overall_score * combined_weight
                    total_weight += combined_weight
                
                overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
                
                # Source breakdown
                source_breakdown = {}
                for source, scores in source_scores.items():
                    if scores:
                        source_breakdown[source] = np.mean(scores)
                
                # Confidence calculation
                avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
                volume_confidence = min(1.0, len(recent_results) / 50.0)
                final_confidence = (avg_confidence + volume_confidence) / 2.0
                
                # Sentiment label
                sentiment_label = self._classify_sentiment(overall_score, final_confidence)
                
                # Trend analysis
                trend_direction = self._calculate_trend(symbol, timeframe)
                
                # Volatility impact
                volatility_impact = self._calculate_volatility_impact(all_scores)
                
                # Recommendation
                recommendation = self._generate_recommendation(
                    overall_score, sentiment_label, trend_direction, volatility_impact
                )
                
                aggregate = AggregateSentiment(
                    symbol=symbol,
                    timeframe=timeframe,
                    overall_score=overall_score,
                    confidence=final_confidence,
                    sentiment_label=sentiment_label,
                    source_breakdown=source_breakdown,
                    trend_direction=trend_direction,
                    volatility_impact=volatility_impact,
                    recommendation=recommendation,
                    timestamp=datetime.now()
                )
                
                self.aggregate_sentiments[f"{symbol}_{timeframe}"] = aggregate
                return aggregate
                
        except Exception as e:
            logger.error(f"Aggregate sentiment calculation failed: {e}")
            return self._create_neutral_aggregate(symbol, timeframe)

    def _calculate_trend(self, symbol: str, timeframe: str) -> str:
        """Calculate sentiment trend direction"""
        try:
            # Get historical sentiment data
            recent_results = list(self.sentiment_cache.get(symbol, []))
            if len(recent_results) < 10:
                return "stable"
            
            # Split into time windows for trend analysis
            window_size = max(1, len(recent_results) // 3)
            windows = [recent_results[i:i + window_size] for i in range(0, len(recent_results), window_size)]
            
            if len(windows) < 2:
                return "stable"
            
            # Calculate average scores for recent windows
            window_scores = []
            for window in windows[:3]:  # Use up to 3 most recent windows
                if window:
                    window_scores.append(np.mean([r.overall_score for r in window]))
            
            # Determine trend
            if len(window_scores) >= 2:
                recent_change = window_scores[-1] - window_scores[0]
                
                if recent_change > 0.1:
                    return "improving"
                elif recent_change < -0.1:
                    return "deteriorating"
            
            return "stable"
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return "stable"

    def _calculate_volatility_impact(self, scores: List[float]) -> float:
        """Calculate volatility impact from sentiment scores"""
        if len(scores) < 5:
            return 0.0
        
        try:
            # Score volatility
            score_volatility = np.std(scores)
            
            # Extreme sentiment ratio
            extreme_count = sum(1 for s in scores if abs(s) > 0.5)
            extreme_ratio = extreme_count / len(scores)
            
            # Combined volatility impact
            volatility_impact = (score_volatility + extreme_ratio) / 2.0
            return min(1.0, volatility_impact)
            
        except Exception as e:
            logger.error(f"Volatility impact calculation failed: {e}")
            return 0.5

    def _generate_recommendation(self, score: float, label: SentimentLabel, 
                               trend: str, volatility: float) -> str:
        """Generate trading recommendation"""
        recommendations = []
        
        # Sentiment-based recommendations
        if label == SentimentLabel.STRONGLY_BULLISH:
            recommendations.append("Strong bullish sentiment - favorable for long positions")
        elif label == SentimentLabel.BULLISH:
            recommendations.append("Bullish sentiment - consider long opportunities")
        elif label == SentimentLabel.STRONGLY_BEARISH:
            recommendations.append("Strong bearish sentiment - favorable for short positions")
        elif label == SentimentLabel.BEARISH:
            recommendations.append("Bearish sentiment - consider short opportunities")
        
        # Trend-based recommendations
        if trend == "improving" and label not in [SentimentLabel.BEARISH, SentimentLabel.STRONGLY_BEARISH]:
            recommendations.append("Sentiment trend improving - monitor for continuation")
        elif trend == "deteriorating" and label not in [SentimentLabel.BULLISH, SentimentLabel.STRONGLY_BULLISH]:
            recommendations.append("Sentiment trend deteriorating - exercise caution")
        
        # Volatility-based recommendations
        if volatility > 0.7:
            recommendations.append("High sentiment volatility - use wider stops")
        elif volatility > 0.4:
            recommendations.append("Moderate sentiment volatility - standard risk management")
        
        return " | ".join(recommendations) if recommendations else "Neutral market sentiment"

    def _create_neutral_aggregate(self, symbol: str, timeframe: str) -> AggregateSentiment:
        """Create neutral aggregate sentiment"""
        return AggregateSentiment(
            symbol=symbol,
            timeframe=timeframe,
            overall_score=0.0,
            confidence=0.1,
            sentiment_label=SentimentLabel.NEUTRAL,
            source_breakdown={},
            trend_direction="stable",
            volatility_impact=0.0,
            recommendation="Insufficient data for analysis",
            timestamp=datetime.now()
        )

    def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        while True:
            try:
                self._calculate_performance_metrics()
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Performance monitoring failed: {e}")
                time.sleep(60)

    def _cache_cleanup_loop(self):
        """Background cache cleanup"""
        while True:
            try:
                with self._lock:
                    cutoff = datetime.now() - timedelta(hours=24)
                    
                    # Clean old sentiment results
                    for symbol in list(self.sentiment_cache.keys()):
                        self.sentiment_cache[symbol] = deque(
                            [r for r in self.sentiment_cache[symbol] if r.timestamp >= cutoff],
                            maxlen=1000
                        )
                
                time.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
                time.sleep(300)

    def _aggregate_update_loop(self):
        """Background aggregate sentiment updates"""
        while True:
            try:
                symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
                timeframes = ['1h', '4h', '24h']
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        self.get_aggregate_sentiment(symbol, timeframe)
                
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Aggregate update failed: {e}")
                time.sleep(60)

    def _calculate_performance_metrics(self):
        """Calculate and log performance metrics"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'total_analyzed_texts': sum(len(results) for results in self.sentiment_cache.values()),
                'average_confidence': 0.0,
                'source_distribution': defaultdict(int),
                'model_performance': defaultdict(dict)
            }
            
            # Calculate average confidence
            all_confidences = []
            for symbol_results in self.sentiment_cache.values():
                for result in symbol_results:
                    all_confidences.append(result.confidence)
                    metrics['source_distribution'][result.source.value] += 1
            
            if all_confidences:
                metrics['average_confidence'] = np.mean(all_confidences)
            
            logger.info(f"Performance Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")

    def get_sentiment_alerts(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get sentiment alerts for significant movements"""
        alerts = []
        
        try:
            symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
            
            for symbol in symbols:
                aggregate = self.get_aggregate_sentiment(symbol, "1h")
                
                if (aggregate.confidence > 0.6 and 
                    abs(aggregate.overall_score) > threshold):
                    
                    alert = {
                        'symbol': symbol,
                        'sentiment_score': aggregate.overall_score,
                        'sentiment_label': aggregate.sentiment_label.value,
                        'confidence': aggregate.confidence,
                        'trend': aggregate.trend_direction,
                        'volatility': aggregate.volatility_impact,
                        'recommendation': aggregate.recommendation,
                        'timestamp': aggregate.timestamp
                    }
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Sentiment alerts generation failed: {e}")
            return []

# Example usage and testing
def main():
    """Example usage of the AdvancedSentimentAnalyzer"""
    
    # Configuration
    config = SentimentConfig(
        use_vader=True,
        use_textblob=True,
        use_bert=True,
        use_finbert=True,
        confidence_threshold=0.6
    )
    
    # Initialize analyzer
    analyzer = AdvancedSentimentAnalyzer(config)
    
    # Test samples
    test_texts = [
        {
            'text': "EUR/USD shows strong bullish momentum after ECB decision. The pair rallied to 1.1000 amid positive economic data.",
            'source': SentimentSource.NEWS,
            'symbol': 'EUR/USD'
        },
        {
            'text': "Federal Reserve hawkish stance puts pressure on USD pairs. GBP/USD falls below 1.3000 as inflation concerns grow.",
            'source': SentimentSource.BLOOMBERG,
            'symbol': 'GBP/USD'
        },
        {
            'text': "Market volatility increases due to geopolitical tensions. Traders seek safe havens as risk appetite diminishes.",
            'source': SentimentSource.REUTERS,
            'symbol': 'USD/JPY'
        }
    ]
    
    print("=== Sentiment Analysis Demo ===")
    
    for i, test_data in enumerate(test_texts, 1):
        result = analyzer.analyze_text(
            test_data['text'],
            test_data['source'],
            test_data['symbol']
        )
        
        print(f"\n{i}. {test_data['symbol']} - {test_data['source'].value}")
        print(f"Text: {test_data['text'][:100]}...")
        print(f"Sentiment: {result.sentiment_label.value} (Score: {result.overall_score:.3f})")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Keywords: {list(result.keyword_scores.keys())[:5]}")
        print(f"Entities: {result.entity_mentions}")
    
    # Test aggregate sentiment
    print("\n=== Aggregate Sentiment ===")
    for symbol in ['EUR/USD', 'GBP/USD', 'USD/JPY']:
        aggregate = analyzer.get_aggregate_sentiment(symbol, "1h")
        print(f"\n{symbol}:")
        print(f"  Overall Score: {aggregate.overall_score:.3f}")
        print(f"  Confidence: {aggregate.confidence:.3f}")
        print(f"  Label: {aggregate.sentiment_label.value}")
        print(f"  Trend: {aggregate.trend_direction}")
        print(f"  Recommendation: {aggregate.recommendation}")
    
    # Test alerts
    print("\n=== Sentiment Alerts ===")
    alerts = analyzer.get_sentiment_alerts(threshold=0.5)
    for alert in alerts:
        print(f"Alert for {alert['symbol']}: {alert['sentiment_label']} (Score: {alert['sentiment_score']:.3f})")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
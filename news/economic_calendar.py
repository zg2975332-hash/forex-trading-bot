"""
Advanced Economic Calendar for FOREX TRADING BOT
Real-time economic event tracking with impact analysis and trading signals
"""

import logging
import pandas as pd
import numpy as np
import json
import requests
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
import re
from zoneinfo import ZoneInfo
import warnings

logger = logging.getLogger(__name__)

class ImpactLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class EventStatus(Enum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    REVISED = "revised"
    CANCELLED = "cancelled"

class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    NZD = "NZD"
    CNY = "CNY"

@dataclass
class EconomicEvent:
    """Economic calendar event data structure"""
    event_id: str
    title: str
    country: str
    currency: Currency
    date: datetime
    impact: ImpactLevel
    previous: Optional[float] = None
    forecast: Optional[float] = None
    actual: Optional[float] = None
    unit: str = ""
    description: str = ""
    source: str = ""
    event_status: EventStatus = EventStatus.SCHEDULED
    revision_from: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EventAnalysis:
    """Analysis results for economic events"""
    event: EconomicEvent
    deviation_score: float  # How much actual deviated from forecast
    surprise_factor: float  # Statistical surprise level
    market_impact: float    # Expected market impact (0-1)
    volatility_forecast: float  # Expected volatility (0-1)
    trading_bias: str       # bullish, bearish, neutral
    confidence: float       # Analysis confidence (0-1)
    recommended_actions: List[str]
    historical_volatility: float
    correlation_assets: List[str]

@dataclass
class EconomicCalendarConfig:
    """Configuration for economic calendar"""
    # Data sources
    enable_fred: bool = True
    enable_forex_factory: bool = True
    enable_investing_com: bool = True
    enable_trading_economics: bool = True
    
    # API keys
    fred_api_key: str = ""
    trading_economics_api_key: str = ""
    
    # Impact thresholds
    high_impact_threshold: float = 0.7
    medium_impact_threshold: float = 0.4
    surprise_threshold: float = 1.5  # Standard deviations
    
    # Time settings
    lookahead_days: int = 7
    lookback_days: int = 30
    update_frequency: int = 300  # seconds
    timezone: str = "America/New_York"
    
    # Trading settings
    pre_event_buffer: int = 30  # minutes before event
    post_event_buffer: int = 60  # minutes after event
    max_concurrent_events: int = 3
    
    # Risk management
    enable_auto_hedging: bool = True
    max_volatility_exposure: float = 0.3
    event_blacklist: List[str] = field(default_factory=list)

class EconomicCalendar:
    """
    Advanced Economic Calendar with real-time event tracking and impact analysis
    """
    
    def __init__(self, config: EconomicCalendarConfig = None):
        self.config = config or EconomicCalendarConfig()
        self.timezone = ZoneInfo(self.config.timezone)
        
        # Data storage
        self.events: Dict[str, EconomicEvent] = {}
        self.event_analysis: Dict[str, EventAnalysis] = {}
        self.historical_data = defaultdict(lambda: deque(maxlen=1000))
        self.volatility_cache = {}
        
        # API clients
        self.session = None
        self.fred_session = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._update_lock = threading.Lock()
        
        # Impact mapping for common events
        self._initialize_impact_mapping()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("EconomicCalendar initialized")

    def _initialize_impact_mapping(self):
        """Initialize impact levels for common economic events"""
        self.impact_mapping = {
            # HIGH IMPACT EVENTS
            "Non-Farm Payrolls": ImpactLevel.VERY_HIGH,
            "Interest Rate Decision": ImpactLevel.VERY_HIGH,
            "CPI": ImpactLevel.VERY_HIGH,
            "GDP": ImpactLevel.VERY_HIGH,
            "FOMC": ImpactLevel.VERY_HIGH,
            "ECB Press Conference": ImpactLevel.VERY_HIGH,
            "Retail Sales": ImpactLevel.HIGH,
            "Unemployment Rate": ImpactLevel.HIGH,
            "Inflation Rate": ImpactLevel.HIGH,
            "PMI": ImpactLevel.HIGH,
            
            # MEDIUM IMPACT EVENTS
            "PPI": ImpactLevel.MEDIUM,
            "Industrial Production": ImpactLevel.MEDIUM,
            "Consumer Confidence": ImpactLevel.MEDIUM,
            "Business Confidence": ImpactLevel.MEDIUM,
            "Trade Balance": ImpactLevel.MEDIUM,
            "Current Account": ImpactLevel.MEDIUM,
            
            # LOW IMPACT EVENTS
            "Building Permits": ImpactLevel.LOW,
            "Housing Starts": ImpactLevel.LOW,
            "Factory Orders": ImpactLevel.LOW,
            "Wholesale Inventories": ImpactLevel.LOW
        }
        
        # Currency-specific important events
        self.currency_important_events = {
            Currency.USD: ["Non-Farm Payrolls", "FOMC", "CPI", "GDP", "Retail Sales"],
            Currency.EUR: ["ECB Press Conference", "CPI", "GDP", "German ZEW"],
            Currency.GBP: ["BOE Inflation Report", "CPI", "GDP", "Retail Sales"],
            Currency.JPY: ["BOJ Policy Rate", "CPI", "GDP", "Tankan Survey"],
            Currency.AUD: ["RBA Rate Statement", "CPI", "GDP", "Employment Change"],
            Currency.CAD: ["BOC Rate Statement", "CPI", "GDP", "Employment Change"],
            Currency.CHF: ["SNB Monetary Policy", "CPI", "GDP"],
            Currency.NZD: ["RBNZ Rate Statement", "CPI", "GDP", "Employment Change"]
        }

    def _start_background_tasks(self):
        """Start background data collection tasks"""
        # Real-time event updates
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()
        
        # Volatility monitoring
        volatility_thread = threading.Thread(target=self._volatility_monitor_loop, daemon=True)
        volatility_thread.start()
        
        # Analysis updates
        analysis_thread = threading.Thread(target=self._analysis_update_loop, daemon=True)
        analysis_thread.start()
        
        # Database maintenance
        db_thread = threading.Thread(target=self._db_maintenance_loop, daemon=True)
        db_thread.start()

    async def _initialize_session(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    def _determine_impact_level(self, event_title: str, currency: Currency) -> ImpactLevel:
        """Determine impact level based on event title and currency"""
        # Check exact matches first
        for key, impact in self.impact_mapping.items():
            if key.lower() in event_title.lower():
                return impact
        
        # Check currency-specific important events
        important_events = self.currency_important_events.get(currency, [])
        for important_event in important_events:
            if important_event.lower() in event_title.lower():
                return ImpactLevel.HIGH
        
        # Default to medium impact for economic events
        economic_terms = ["rate", "cpi", "gdp", "employment", "sales", "production", "confidence"]
        if any(term in event_title.lower() for term in economic_terms):
            return ImpactLevel.MEDIUM
        
        return ImpactLevel.LOW

    async def fetch_forex_factory_events(self, days: int = 7) -> List[EconomicEvent]:
        """Fetch economic events from Forex Factory"""
        events = []
        try:
            # Note: Forex Factory requires web scraping
            # This is a simplified implementation
            base_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            
            async with self.session.get(base_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data:
                        try:
                            # Parse date
                            event_date = datetime.fromtimestamp(item['timestamp'], tz=self.timezone)
                            
                            # Only include future events and recent past events
                            if event_date > datetime.now(self.timezone) - timedelta(hours=24):
                                # Determine currency
                                currency_str = item.get('currency', 'USD')
                                try:
                                    currency = Currency(currency_str)
                                except ValueError:
                                    continue
                                
                                # Create event
                                event = EconomicEvent(
                                    event_id=f"ff_{item['title'].replace(' ', '_').lower()}_{int(item['timestamp'])}",
                                    title=item['title'],
                                    country=item.get('country', ''),
                                    currency=currency,
                                    date=event_date,
                                    impact=self._determine_impact_level(item['title'], currency),
                                    previous=item.get('previous'),
                                    forecast=item.get('forecast'),
                                    actual=item.get('actual'),
                                    unit=item.get('unit', ''),
                                    description=item.get('description', ''),
                                    source="Forex Factory",
                                    event_status=EventStatus.SCHEDULED
                                )
                                
                                events.append(event)
                                
                        except Exception as e:
                            logger.warning(f"Error parsing Forex Factory event: {e}")
                            continue
                            
            logger.info(f"Fetched {len(events)} events from Forex Factory")
            
        except Exception as e:
            logger.error(f"Forex Factory fetch failed: {e}")
            
        return events

    async def fetch_fred_events(self, days: int = 7) -> List[EconomicEvent]:
        """Fetch economic events from FRED API"""
        events = []
        
        if not self.config.fred_api_key:
            return events
            
        try:
            # FRED API endpoint for economic releases
            base_url = "https://api.stlouisfed.org/fred/releases"
            params = {
                'api_key': self.config.fred_api_key,
                'file_type': 'json'
            }
            
            async with self.session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # This would need more complex parsing for actual FRED data
                    # Simplified for example purposes
                    logger.info("FRED API response received")
                    
        except Exception as e:
            logger.error(f"FRED API fetch failed: {e}")
            
        return events

    async def fetch_investing_com_events(self, days: int = 7) -> List[EconomicEvent]:
        """Fetch economic events from Investing.com (simulated)"""
        events = []
        try:
            # Note: Investing.com requires web scraping or paid API
            # This is a simulated implementation
            simulated_events = [
                {
                    'title': 'US Core CPI',
                    'currency': 'USD',
                    'date': datetime.now(self.timezone) + timedelta(hours=2),
                    'impact': 'high',
                    'forecast': 0.3,
                    'previous': 0.4
                },
                {
                    'title': 'German ZEW Economic Sentiment',
                    'currency': 'EUR',
                    'date': datetime.now(self.timezone) + timedelta(hours=5),
                    'impact': 'medium',
                    'forecast': 15.2,
                    'previous': 12.8
                }
            ]
            
            for item in simulated_events:
                currency = Currency(item['currency'])
                event = EconomicEvent(
                    event_id=f"inv_{item['title'].replace(' ', '_').lower()}_{int(datetime.now().timestamp())}",
                    title=item['title'],
                    country="US" if currency == Currency.USD else "EU",
                    currency=currency,
                    date=item['date'],
                    impact=ImpactLevel(item['impact']),
                    forecast=item.get('forecast'),
                    previous=item.get('previous'),
                    source="Investing.com",
                    event_status=EventStatus.SCHEDULED
                )
                events.append(event)
                
            logger.info(f"Simulated {len(events)} events from Investing.com")
            
        except Exception as e:
            logger.error(f"Investing.com fetch failed: {e}")
            
        return events

    async def fetch_all_events(self, days: int = None) -> List[EconomicEvent]:
        """Fetch events from all configured sources"""
        if days is None:
            days = self.config.lookahead_days
            
        await self._initialize_session()
        
        all_events = []
        tasks = []
        
        # Add tasks for enabled sources
        if self.config.enable_forex_factory:
            tasks.append(self.fetch_forex_factory_events(days))
            
        if self.config.enable_fred and self.config.fred_api_key:
            tasks.append(self.fetch_fred_events(days))
            
        if self.config.enable_investing_com:
            tasks.append(self.fetch_investing_com_events(days))
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Event fetch task failed: {result}")
                elif isinstance(result, list):
                    all_events.extend(result)
        
        # Remove duplicates and sort by date
        unique_events = self._deduplicate_events(all_events)
        unique_events.sort(key=lambda x: x.date)
        
        # Update events cache
        with self._lock:
            for event in unique_events:
                self.events[event.event_id] = event
        
        logger.info(f"Total unique events fetched: {len(unique_events)}")
        return unique_events

    def _deduplicate_events(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """Remove duplicate events from different sources"""
        unique_events = {}
        
        for event in events:
            # Create a unique key based on title, currency, and date (within 1 hour)
            event_key = f"{event.title}_{event.currency.value}_{event.date.strftime('%Y%m%d%H')}"
            
            if event_key not in unique_events:
                unique_events[event_key] = event
            else:
                # Keep the event with more complete data
                existing = unique_events[event_key]
                if (event.forecast is not None and existing.forecast is None) or \
                   (event.previous is not None and existing.previous is None):
                    unique_events[event_key] = event
        
        return list(unique_events.values())

    def get_upcoming_events(self, hours: int = 24, currency: Currency = None) -> List[EconomicEvent]:
        """Get upcoming events within specified time frame"""
        with self._lock:
            now = datetime.now(self.timezone)
            end_time = now + timedelta(hours=hours)
            
            upcoming = []
            for event in self.events.values():
                if now <= event.date <= end_time:
                    if currency is None or event.currency == currency:
                        upcoming.append(event)
            
            # Sort by date and impact
            upcoming.sort(key=lambda x: (x.date, -self._impact_to_numeric(x.impact)))
            return upcoming

    def get_high_impact_events(self, hours: int = 24) -> List[EconomicEvent]:
        """Get only high and very high impact events"""
        upcoming = self.get_upcoming_events(hours)
        return [e for e in upcoming if e.impact in [ImpactLevel.HIGH, ImpactLevel.VERY_HIGH]]

    def _impact_to_numeric(self, impact: ImpactLevel) -> float:
        """Convert impact level to numeric value for sorting"""
        impact_map = {
            ImpactLevel.LOW: 1,
            ImpactLevel.MEDIUM: 2,
            ImpactLevel.HIGH: 3,
            ImpactLevel.VERY_HIGH: 4
        }
        return impact_map.get(impact, 0)

    def analyze_event(self, event: EconomicEvent) -> EventAnalysis:
        """Perform comprehensive analysis of an economic event"""
        try:
            # Calculate deviation score
            deviation_score = self._calculate_deviation_score(event)
            
            # Calculate surprise factor
            surprise_factor = self._calculate_surprise_factor(event, deviation_score)
            
            # Estimate market impact
            market_impact = self._estimate_market_impact(event, deviation_score, surprise_factor)
            
            # Forecast volatility
            volatility_forecast = self._forecast_volatility(event, market_impact)
            
            # Determine trading bias
            trading_bias = self._determine_trading_bias(event, deviation_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(event, deviation_score)
            
            # Generate recommendations
            recommended_actions = self._generate_recommendations(
                event, deviation_score, market_impact, volatility_forecast
            )
            
            # Historical volatility
            historical_volatility = self._get_historical_volatility(event)
            
            # Correlation assets
            correlation_assets = self._get_correlation_assets(event.currency)
            
            analysis = EventAnalysis(
                event=event,
                deviation_score=deviation_score,
                surprise_factor=surprise_factor,
                market_impact=market_impact,
                volatility_forecast=volatility_forecast,
                trading_bias=trading_bias,
                confidence=confidence,
                recommended_actions=recommended_actions,
                historical_volatility=historical_volatility,
                correlation_assets=correlation_assets
            )
            
            # Cache analysis
            with self._lock:
                self.event_analysis[event.event_id] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Event analysis failed for {event.title}: {e}")
            # Return default analysis
            return self._create_default_analysis(event)

    def _calculate_deviation_score(self, event: EconomicEvent) -> float:
        """Calculate how much actual deviated from forecast"""
        if event.actual is None or event.forecast is None:
            return 0.0
        
        try:
            # Handle percentage values
            if event.unit == '%' or any(term in event.title.lower() for term in ['%', 'percent', 'rate']):
                deviation = abs(event.actual - event.forecast)
            else:
                # For absolute values, calculate percentage deviation
                if event.forecast != 0:
                    deviation = abs((event.actual - event.forecast) / event.forecast)
                else:
                    deviation = abs(event.actual) if event.actual != 0 else 0.0
            
            # Normalize deviation score (0-1)
            deviation_score = min(1.0, deviation / 2.0)  # Cap at 200% deviation
            
            return deviation_score
            
        except Exception as e:
            logger.warning(f"Deviation calculation failed: {e}")
            return 0.0

    def _calculate_surprise_factor(self, event: EconomicEvent, deviation_score: float) -> float:
        """Calculate statistical surprise factor"""
        if event.actual is None or event.forecast is None:
            return 0.0
        
        try:
            # Get historical deviations for this event type
            historical_deviations = self._get_historical_deviations(event.title)
            
            if not historical_deviations:
                # Default surprise threshold
                return deviation_score / 0.1  # Normalize by 10% threshold
            
            # Calculate z-score of current deviation
            mean_dev = np.mean(historical_deviations)
            std_dev = np.std(historical_deviations) if len(historical_deviations) > 1 else 0.1
            
            if std_dev == 0:
                surprise = abs(deviation_score - mean_dev) / 0.01
            else:
                surprise = abs(deviation_score - mean_dev) / std_dev
            
            # Normalize surprise factor
            surprise_factor = min(3.0, surprise) / 3.0  # Cap at 3 sigma
            
            return surprise_factor
            
        except Exception as e:
            logger.warning(f"Surprise factor calculation failed: {e}")
            return deviation_score

    def _estimate_market_impact(self, event: EconomicEvent, deviation_score: float, 
                              surprise_factor: float) -> float:
        """Estimate potential market impact"""
        # Base impact from event type
        impact_weights = {
            ImpactLevel.LOW: 0.2,
            ImpactLevel.MEDIUM: 0.5,
            ImpactLevel.HIGH: 0.8,
            ImpactLevel.VERY_HIGH: 1.0
        }
        
        base_impact = impact_weights.get(event.impact, 0.3)
        
        # Adjust for deviation and surprise
        deviation_impact = deviation_score * 0.6
        surprise_impact = surprise_factor * 0.4
        
        # Combined impact
        market_impact = base_impact * (0.4 + 0.3 * deviation_impact + 0.3 * surprise_impact)
        
        return min(1.0, market_impact)

    def _forecast_volatility(self, event: EconomicEvent, market_impact: float) -> float:
        """Forecast volatility based on event and impact"""
        # Base volatility from impact level
        base_volatility = {
            ImpactLevel.LOW: 0.1,
            ImpactLevel.MEDIUM: 0.3,
            ImpactLevel.HIGH: 0.6,
            ImpactLevel.VERY_HIGH: 0.9
        }.get(event.impact, 0.2)
        
        # Adjust for market impact
        volatility = base_volatility * (0.7 + 0.3 * market_impact)
        
        # Increase volatility for USD events (most traded)
        if event.currency == Currency.USD:
            volatility *= 1.2
        
        return min(1.0, volatility)

    def _determine_trading_bias(self, event: EconomicEvent, deviation_score: float) -> str:
        """Determine trading bias based on event data"""
        if event.actual is None or event.forecast is None:
            return "neutral"
        
        # For most economic indicators, higher than expected is bullish for the currency
        # Except for indicators like unemployment where lower is better
        bullish_indicators = ['gdp', 'retail sales', 'cpi', 'employment', 'payrolls', 'production']
        bearish_indicators = ['unemployment', 'claims', 'deficit']
        
        event_lower = event.title.lower()
        
        if any(indicator in event_lower for indicator in bullish_indicators):
            if event.actual > event.forecast:
                return "bullish"
            elif event.actual < event.forecast:
                return "bearish"
                
        elif any(indicator in event_lower for indicator in bearish_indicators):
            if event.actual < event.forecast:
                return "bullish" 
            elif event.actual > event.forecast:
                return "bearish"
        
        return "neutral"

    def _calculate_confidence(self, event: EconomicEvent, deviation_score: float) -> float:
        """Calculate confidence in analysis"""
        confidence_factors = []
        
        # Data completeness
        if event.actual is not None and event.forecast is not None:
            confidence_factors.append(0.8)
        elif event.forecast is not None:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Event impact level
        impact_confidence = {
            ImpactLevel.VERY_HIGH: 0.9,
            ImpactLevel.HIGH: 0.7,
            ImpactLevel.MEDIUM: 0.5,
            ImpactLevel.LOW: 0.3
        }.get(event.impact, 0.3)
        confidence_factors.append(impact_confidence)
        
        # Historical data availability
        historical_data = self._get_historical_deviations(event.title)
        if len(historical_data) > 10:
            confidence_factors.append(0.8)
        elif len(historical_data) > 5:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)

    def _generate_recommendations(self, event: EconomicEvent, deviation_score: float,
                                market_impact: float, volatility_forecast: float) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Impact-based recommendations
        if market_impact > 0.7:
            recommendations.append("High impact event - consider reducing position sizes")
            recommendations.append("Monitor for breakout opportunities")
        elif market_impact > 0.4:
            recommendations.append("Medium impact - standard risk management applies")
        
        # Volatility recommendations
        if volatility_forecast > 0.7:
            recommendations.append("High volatility expected - use wider stop losses")
        elif volatility_forecast > 0.5:
            recommendations.append("Elevated volatility - adjust position sizing")
        
        # Timing recommendations
        time_to_event = (event.date - datetime.now(self.timezone)).total_seconds() / 3600
        if 0 < time_to_event <= 2:
            recommendations.append("Event imminent - consider closing sensitive positions")
        elif time_to_event <= 0.5:
            recommendations.append("Event very soon - avoid new positions")
        
        # Currency-specific recommendations
        if event.currency == Currency.USD:
            recommendations.append("USD event - monitor DXY and major USD pairs")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Monitor event for trading opportunities")
        
        return recommendations[:5]  # Return top 5 recommendations

    def _get_historical_deviations(self, event_title: str) -> List[float]:
        """Get historical deviations for similar events"""
        # This would typically query a database
        # For now, return simulated data
        return [0.1, 0.05, 0.15, 0.08, 0.12, 0.03, 0.18, 0.07, 0.09, 0.11]

    def _get_historical_volatility(self, event: EconomicEvent) -> float:
        """Get historical volatility for this event type"""
        # Simulated historical volatility data
        volatility_map = {
            "Non-Farm Payrolls": 0.8,
            "CPI": 0.7,
            "Interest Rate Decision": 0.9,
            "GDP": 0.6,
            "Retail Sales": 0.5
        }
        
        for key, vol in volatility_map.items():
            if key.lower() in event.title.lower():
                return vol
        
        return 0.3  # Default volatility

    def _get_correlation_assets(self, currency: Currency) -> List[str]:
        """Get assets that correlate with this currency's events"""
        correlation_map = {
            Currency.USD: ["DXY", "SPX", "UST10Y", "XAU/USD"],
            Currency.EUR: ["EUR/USD", "DAX", "BUND", "XAU/EUR"],
            Currency.GBP: ["GBP/USD", "FTSE", "GILT", "GBP/JPY"],
            Currency.JPY: ["USD/JPY", "NKY", "JGB", "AUD/JPY"],
            Currency.AUD: ["AUD/USD", "ASX", "AUD/NZD", "XAU/AUD"],
            Currency.CAD: ["USD/CAD", "SPTSX", "WTI", "CAD/JPY"],
            Currency.CHF: ["USD/CHF", "SMI", "EUR/CHF", "XAU/CHF"],
            Currency.NZD: ["NZD/USD", "NZD/JPY", "AUD/NZD", "Dairy"]
        }
        
        return correlation_map.get(currency, [])

    def _create_default_analysis(self, event: EconomicEvent) -> EventAnalysis:
        """Create default analysis when detailed analysis fails"""
        return EventAnalysis(
            event=event,
            deviation_score=0.0,
            surprise_factor=0.0,
            market_impact=0.3,
            volatility_forecast=0.3,
            trading_bias="neutral",
            confidence=0.1,
            recommended_actions=["Insufficient data for detailed analysis"],
            historical_volatility=0.3,
            correlation_assets=[]
        )

    def get_event_alert(self, event: EconomicEvent, minutes_before: int = 30) -> Optional[Dict]:
        """Generate alert for upcoming event"""
        try:
            time_to_event = (event.date - datetime.now(self.timezone)).total_seconds() / 60
            
            if 0 <= time_to_event <= minutes_before:
                analysis = self.analyze_event(event)
                
                alert = {
                    'event_id': event.event_id,
                    'title': event.title,
                    'currency': event.currency.value,
                    'scheduled_time': event.date,
                    'time_to_event': time_to_event,
                    'impact': event.impact.value,
                    'market_impact': analysis.market_impact,
                    'volatility_forecast': analysis.volatility_forecast,
                    'recommendations': analysis.recommended_actions,
                    'analysis_confidence': analysis.confidence
                }
                
                return alert
                
        except Exception as e:
            logger.error(f"Event alert generation failed: {e}")
        
        return None

    def get_market_overview(self, hours: int = 24) -> Dict[str, Any]:
        """Get market overview based on economic events"""
        try:
            upcoming_events = self.get_upcoming_events(hours)
            high_impact_count = len(self.get_high_impact_events(hours))
            
            # Calculate overall market stress
            total_impact = 0.0
            total_volatility = 0.0
            event_count = 0
            
            for event in upcoming_events:
                analysis = self.analyze_event(event)
                total_impact += analysis.market_impact
                total_volatility += analysis.volatility_forecast
                event_count += 1
            
            avg_impact = total_impact / event_count if event_count > 0 else 0.0
            avg_volatility = total_volatility / event_count if event_count > 0 else 0.0
            
            # Determine market status
            if high_impact_count >= 3 or avg_impact > 0.7:
                market_status = "HIGH_STRESS"
            elif high_impact_count >= 1 or avg_impact > 0.4:
                market_status = "ELEVATED_STRESS"
            else:
                market_status = "NORMAL"
            
            # Currency exposure
            currency_exposure = defaultdict(int)
            for event in upcoming_events:
                currency_exposure[event.currency.value] += self._impact_to_numeric(event.impact)
            
            overview = {
                'timestamp': datetime.now(),
                'market_status': market_status,
                'total_upcoming_events': len(upcoming_events),
                'high_impact_events': high_impact_count,
                'average_impact': avg_impact,
                'average_volatility': avg_volatility,
                'currency_exposure': dict(currency_exposure),
                'key_events': [e.title for e in upcoming_events[:5]],
                'recommendations': self._generate_market_recommendations(market_status, high_impact_count)
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Market overview generation failed: {e}")
            return {
                'timestamp': datetime.now(),
                'market_status': 'UNKNOWN',
                'error': str(e)
            }

    def _generate_market_recommendations(self, market_status: str, high_impact_count: int) -> List[str]:
        """Generate market-wide recommendations"""
        recommendations = []
        
        if market_status == "HIGH_STRESS":
            recommendations.append("Multiple high-impact events - reduce overall exposure")
            recommendations.append("Consider hedging strategies")
            recommendations.append("Monitor news flow closely")
            
        elif market_status == "ELEVATED_STRESS":
            recommendations.append("Elevated event risk - review position sizes")
            recommendations.append("Set wider stops for volatility")
            
        else:
            recommendations.append("Normal market conditions - standard trading rules apply")
        
        if high_impact_count > 0:
            recommendations.append(f"{high_impact_count} high-impact event(s) scheduled")
        
        return recommendations

    def _update_loop(self):
        """Background loop for updating events"""
        while True:
            try:
                asyncio.run(self.fetch_all_events())
                logger.info("Economic calendar updated successfully")
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Update loop failed: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _volatility_monitor_loop(self):
        """Background loop for volatility monitoring"""
        while True:
            try:
                # Update volatility estimates for upcoming events
                upcoming = self.get_upcoming_events(hours=48)
                for event in upcoming:
                    if event.event_id not in self.event_analysis:
                        self.analyze_event(event)
                
                time.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Volatility monitor loop failed: {e}")
                time.sleep(300)

    def _analysis_update_loop(self):
        """Background loop for updating analyses"""
        while True:
            try:
                # Update analyses for events happening soon
                upcoming = self.get_upcoming_events(hours=6)
                for event in upcoming:
                    self.analyze_event(event)
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Analysis update loop failed: {e}")
                time.sleep(60)

    def _db_maintenance_loop(self):
        """Background loop for database maintenance"""
        while True:
            try:
                # Clean up old events
                with self._lock:
                    current_time = datetime.now(self.timezone)
                    expired_events = []
                    
                    for event_id, event in self.events.items():
                        if event.date < current_time - timedelta(days=2):
                            expired_events.append(event_id)
                    
                    for event_id in expired_events:
                        del self.events[event_id]
                        if event_id in self.event_analysis:
                            del self.event_analysis[event_id]
                
                logger.debug(f"Cleaned up {len(expired_events)} expired events")
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"DB maintenance loop failed: {e}")
                time.sleep(1800)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the economic calendar"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'total_tracked_events': len(self.events),
                'upcoming_events': len(self.get_upcoming_events(24)),
                'high_impact_upcoming': len(self.get_high_impact_events(24)),
                'analysis_coverage': len(self.event_analysis),
                'data_freshness': {},
                'system_health': 'HEALTHY'
            }
            
            # Data freshness
            now = datetime.now(self.timezone)
            for event in list(self.events.values())[:10]:  # Sample 10 events
                time_diff = (now - event.timestamp).total_seconds() / 60
                metrics['data_freshness'][event.event_id] = f"{time_diff:.1f} minutes ago"
            
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
    """Example usage of the EconomicCalendar"""
    
    # Configuration
    config = EconomicCalendarConfig(
        enable_forex_factory=True,
        enable_investing_com=True,
        lookahead_days=3,
        update_frequency=600  # 10 minutes for testing
    )
    
    # Initialize calendar
    calendar = EconomicCalendar(config)
    
    # Wait for initial data load
    time.sleep(5)
    
    print("=== Economic Calendar Demo ===")
    
    # Get upcoming events
    upcoming = calendar.get_upcoming_events(hours=24)
    print(f"\nUpcoming events (next 24 hours): {len(upcoming)}")
    
    for i, event in enumerate(upcoming[:5], 1):
        print(f"\n{i}. {event.title} ({event.currency.value})")
        print(f"   Time: {event.date.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"   Impact: {event.impact.value}")
        if event.forecast is not None:
            print(f"   Forecast: {event.forecast}")
    
    # Get high impact events
    high_impact = calendar.get_high_impact_events(hours=24)
    print(f"\nHigh impact events: {len(high_impact)}")
    
    # Analyze an event
    if upcoming:
        analysis = calendar.analyze_event(upcoming[0])
        print(f"\n=== Analysis for {upcoming[0].title} ===")
        print(f"Deviation Score: {analysis.deviation_score:.3f}")
        print(f"Market Impact: {analysis.market_impact:.3f}")
        print(f"Volatility Forecast: {analysis.volatility_forecast:.3f}")
        print(f"Trading Bias: {analysis.trading_bias}")
        print(f"Confidence: {analysis.confidence:.3f}")
        print("Recommendations:")
        for rec in analysis.recommended_actions:
            print(f"  - {rec}")
    
    # Get market overview
    overview = calendar.get_market_overview()
    print(f"\n=== Market Overview ===")
    print(f"Market Status: {overview['market_status']}")
    print(f"Total Events: {overview['total_upcoming_events']}")
    print(f"High Impact: {overview['high_impact_events']}")
    print(f"Average Impact: {overview['average_impact']:.3f}")
    print("Recommendations:")
    for rec in overview['recommendations']:
        print(f"  - {rec}")
    
    # Performance metrics
    metrics = calendar.get_performance_metrics()
    print(f"\n=== System Metrics ===")
    print(f"Total Tracked Events: {metrics['total_tracked_events']}")
    print(f"System Health: {metrics['system_health']}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
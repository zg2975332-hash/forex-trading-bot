"""
Advanced Retail Trading Strategies for FOREX TRADING BOT
Professional retail trader strategies with risk management
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import json
from pathlib import Path
import talib
from scipy import stats
import hashlib

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RetailStrategyType(Enum):
    SUPPLY_DEMAND = "supply_demand"
    SMART_MONEY_CONCEPT = "smart_money_concept"
    ICT = "ict"
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    BREAKER = "breaker"
    LIQUIDITY_GRAB = "liquidity_grab"
    MARKET_STRUCTURE = "market_structure"
    MULTI_TIMEFRAME = "multi_timeframe"

class MarketStructure(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    BREAKOUT = "breakout"

class OrderBlockType(Enum):
    BULLISH_OB = "bullish_ob"
    BEARISH_OB = "bearish_ob"
    BREAKER_OB = "breaker_ob"

@dataclass
class RetailSignal:
    """Retail trading signal"""
    strategy_type: RetailStrategyType
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timeframe: str
    market_structure: MarketStructure
    reasoning: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderBlock:
    """Order Block identification"""
    block_type: OrderBlockType
    high: float
    low: float
    open: float
    close: float
    volume: float
    strength: float
    timeframe: str
    formed_at: datetime
    is_activated: bool = False
    is_breached: bool = False

@dataclass
class FairValueGap:
    """Fair Value Gap identification"""
    high: float
    low: float
    gap_size: float
    direction: str  # bullish, bearish
    strength: float
    timeframe: str
    formed_at: datetime
    is_filled: bool = False

@dataclass
class MarketLiquidity:
    """Market liquidity levels"""
    level: float
    type: str  # buy_side, sell_side
    strength: float
    timeframe: str
    timestamp: datetime

@dataclass
class RetailStrategyConfig:
    """Configuration for retail strategies"""
    
    # Strategy enablement
    enabled_strategies: List[RetailStrategyType] = field(default_factory=lambda: [
        RetailStrategyType.SUPPLY_DEMAND,
        RetailStrategyType.SMART_MONEY_CONCEPT,
        RetailStrategyType.ORDER_BLOCK,
        RetailStrategyType.FAIR_VALUE_GAP
    ])
    
    # Supply Demand parameters
    sd_lookback_period: int = 100
    sd_min_touch_count: int = 2
    sd_strength_threshold: float = 0.6
    
    # Smart Money Concept parameters
    smc_liquidity_lookback: int = 50
    smc_volume_threshold: float = 1.5
    smc_break_of_structure_confirmation: bool = True
    
    # Order Block parameters
    ob_lookback_period: int = 20
    ob_min_volume_ratio: float = 1.2
    ob_activation_zone: float = 0.001  # 0.1%
    
    # Fair Value Gap parameters
    fvg_min_gap_size: float = 0.0005  # 0.05%
    fvg_lookback_period: int = 10
    fvg_fill_threshold: float = 0.8
    
    # Risk management
    default_risk_reward: float = 1.5
    max_position_size: float = 0.02
    min_confidence: float = 0.65
    
    # Multi-timeframe analysis
    primary_timeframe: str = "1h"
    confirmation_timeframes: List[str] = field(default_factory=lambda: ["4h", "15m"])
    require_mtf_confirmation: bool = True

class AdvancedRetailStrategies:
    """
    Advanced Retail Trading Strategies Implementation
    """
    
    def __init__(self, config: RetailStrategyConfig = None):
        self.config = config or RetailStrategyConfig()
        
        # Strategy state
        self.order_blocks: Dict[str, List[OrderBlock]] = defaultdict(list)
        self.fair_value_gaps: Dict[str, List[FairValueGap]] = defaultdict(list)
        self.supply_demand_zones: Dict[str, List[Dict]] = defaultdict(list)
        self.liquidity_levels: Dict[str, List[MarketLiquidity]] = defaultdict(list)
        
        # Market structure
        self.market_structure: Dict[str, MarketStructure] = {}
        self.swing_points: Dict[str, List[Dict]] = defaultdict(list)
        
        # Performance tracking
        self.strategy_performance: Dict[RetailStrategyType, Dict] = defaultdict(dict)
        self.signal_history: deque = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("AdvancedRetailStrategies initialized successfully")
    
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame, 
                           timeframe: str = "1h") -> Dict[str, Any]:
        """Comprehensive market analysis for retail strategies"""
        try:
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'market_structure': None,
                'order_blocks': [],
                'fair_value_gaps': [],
                'supply_demand_zones': [],
                'liquidity_levels': [],
                'swing_points': []
            }
            
            # Analyze market structure
            market_structure = await self._analyze_market_structure(market_data)
            analysis['market_structure'] = market_structure
            
            # Identify order blocks
            order_blocks = await self._identify_order_blocks(market_data, timeframe)
            analysis['order_blocks'] = order_blocks
            
            # Find fair value gaps
            fair_value_gaps = await self._identify_fair_value_gaps(market_data, timeframe)
            analysis['fair_value_gaps'] = fair_value_gaps
            
            # Identify supply demand zones
            supply_demand_zones = await self._identify_supply_demand_zones(market_data, timeframe)
            analysis['supply_demand_zones'] = supply_demand_zones
            
            # Find liquidity levels
            liquidity_levels = await self._identify_liquidity_levels(market_data, timeframe)
            analysis['liquidity_levels'] = liquidity_levels
            
            # Identify swing points
            swing_points = await self._identify_swing_points(market_data)
            analysis['swing_points'] = swing_points
            
            # Update internal state
            with self._lock:
                self.order_blocks[symbol].extend(order_blocks)
                self.fair_value_gaps[symbol].extend(fair_value_gaps)
                self.supply_demand_zones[symbol].extend(supply_demand_zones)
                self.liquidity_levels[symbol].extend(liquidity_levels)
                self.swing_points[symbol].extend(swing_points)
                self.market_structure[symbol] = market_structure
            
            logger.debug(f"Market analysis completed for {symbol} ({timeframe})")
            return analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed for {symbol}: {e}")
            return {}
    
    async def generate_signals(self, symbol: str, market_data: pd.DataFrame,
                             timeframe: str = "1h") -> List[RetailSignal]:
        """Generate trading signals using retail strategies"""
        try:
            signals = []
            
            # Perform market analysis
            market_analysis = await self.analyze_market(symbol, market_data, timeframe)
            if not market_analysis:
                return signals
            
            current_price = market_data['close'].iloc[-1]
            
            # Generate signals for each enabled strategy
            for strategy_type in self.config.enabled_strategies:
                try:
                    if strategy_type == RetailStrategyType.SUPPLY_DEMAND:
                        signal = await self._supply_demand_signal(symbol, market_data, market_analysis, current_price)
                    elif strategy_type == RetailStrategyType.SMART_MONEY_CONCEPT:
                        signal = await self._smart_money_signal(symbol, market_data, market_analysis, current_price)
                    elif strategy_type == RetailStrategyType.ORDER_BLOCK:
                        signal = await self._order_block_signal(symbol, market_data, market_analysis, current_price)
                    elif strategy_type == RetailStrategyType.FAIR_VALUE_GAP:
                        signal = await self._fair_value_gap_signal(symbol, market_data, market_analysis, current_price)
                    elif strategy_type == RetailStrategyType.BREAKER:
                        signal = await self._breaker_signal(symbol, market_data, market_analysis, current_price)
                    elif strategy_type == RetailStrategyType.LIQUIDITY_GRAB:
                        signal = await self._liquidity_grab_signal(symbol, market_data, market_analysis, current_price)
                    else:
                        continue
                    
                    if signal and signal.confidence >= self.config.min_confidence:
                        signals.append(signal)
                        logger.info(f"Generated {strategy_type.value} signal: {signal.action} (confidence: {signal.confidence:.3f})")
                        
                except Exception as e:
                    logger.warning(f"Signal generation failed for {strategy_type.value}: {e}")
                    continue
            
            # Store signals in history
            with self._lock:
                self.signal_history.extend(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return []
    
    async def _analyze_market_structure(self, market_data: pd.DataFrame) -> MarketStructure:
        """Analyze market structure (bullish, bearish, ranging)"""
        try:
            highs = market_data['high'].values
            lows = market_data['low'].values
            closes = market_data['close'].values
            
            # Identify higher highs/lows and lower highs/lows
            higher_highs = 0
            higher_lows = 0
            lower_highs = 0
            lower_lows = 0
            
            for i in range(2, len(highs)):
                # Check for higher high
                if highs[i] > highs[i-1] > highs[i-2]:
                    higher_highs += 1
                # Check for lower high
                elif highs[i] < highs[i-1] < highs[i-2]:
                    lower_highs += 1
                
                # Check for higher low
                if lows[i] > lows[i-1] > lows[i-2]:
                    higher_lows += 1
                # Check for lower low
                elif lows[i] < lows[i-1] < lows[i-2]:
                    lower_lows += 1
            
            # Determine market structure
            if higher_highs > 2 and higher_lows > 2:
                return MarketStructure.BULLISH
            elif lower_highs > 2 and lower_lows > 2:
                return MarketStructure.BEARISH
            elif abs(higher_highs - lower_highs) <= 1 and abs(higher_lows - lower_lows) <= 1:
                return MarketStructure.RANGING
            else:
                # Check for breakout
                recent_high = np.max(highs[-10:])
                recent_low = np.min(lows[-10:])
                if closes[-1] > recent_high:
                    return MarketStructure.BREAKOUT
                elif closes[-1] < recent_low:
                    return MarketStructure.BREAKOUT
                else:
                    return MarketStructure.RANGING
                    
        except Exception as e:
            logger.warning(f"Market structure analysis failed: {e}")
            return MarketStructure.RANGING
    
    async def _identify_order_blocks(self, market_data: pd.DataFrame, timeframe: str) -> List[OrderBlock]:
        """Identify order blocks using Smart Money Concept"""
        try:
            order_blocks = []
            highs = market_data['high'].values
            lows = market_data['low'].values
            opens = market_data['open'].values
            closes = market_data['close'].values
            volumes = market_data['volume'].values
            
            for i in range(1, len(market_data) - 1):
                current_candle = i
                prev_candle = i - 1
                next_candle = i + 1
                
                # Bullish Order Block: Bearish candle followed by bullish candle
                if (closes[prev_candle] < opens[prev_candle] and  # Bearish candle
                    closes[current_candle] > opens[current_candle] and  # Bullish candle
                    closes[current_candle] > closes[prev_candle] and  # Break above previous close
                    volumes[current_candle] > np.mean(volumes[max(0, i-10):i]) * self.config.ob_min_volume_ratio):
                    
                    block = OrderBlock(
                        block_type=OrderBlockType.BULLISH_OB,
                        high=highs[prev_candle],
                        low=lows[prev_candle],
                        open=opens[prev_candle],
                        close=closes[prev_candle],
                        volume=volumes[prev_candle],
                        strength=min(1.0, volumes[current_candle] / np.mean(volumes[max(0, i-10):i])),
                        timeframe=timeframe,
                        formed_at=market_data.index[prev_candle]
                    )
                    order_blocks.append(block)
                
                # Bearish Order Block: Bullish candle followed by bearish candle
                elif (closes[prev_candle] > opens[prev_candle] and  # Bullish candle
                      closes[current_candle] < opens[current_candle] and  # Bearish candle
                      closes[current_candle] < closes[prev_candle] and  # Break below previous close
                      volumes[current_candle] > np.mean(volumes[max(0, i-10):i]) * self.config.ob_min_volume_ratio):
                    
                    block = OrderBlock(
                        block_type=OrderBlockType.BEARISH_OB,
                        high=highs[prev_candle],
                        low=lows[prev_candle],
                        open=opens[prev_candle],
                        close=closes[prev_candle],
                        volume=volumes[prev_candle],
                        strength=min(1.0, volumes[current_candle] / np.mean(volumes[max(0, i-10):i])),
                        timeframe=timeframe,
                        formed_at=market_data.index[prev_candle]
                    )
                    order_blocks.append(block)
            
            # Keep only recent order blocks
            recent_blocks = [block for block in order_blocks 
                           if (datetime.now() - block.formed_at).days < 7]
            
            return recent_blocks[:10]  # Return top 10 blocks
            
        except Exception as e:
            logger.warning(f"Order block identification failed: {e}")
            return []
    
    async def _identify_fair_value_gaps(self, market_data: pd.DataFrame, timeframe: str) -> List[FairValueGap]:
        """Identify Fair Value Gaps (FVG)"""
        try:
            fair_value_gaps = []
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            for i in range(2, len(market_data)):
                current_high = highs[i]
                current_low = lows[i]
                prev_high = highs[i-1]
                prev_low = lows[i-1]
                prev_prev_high = highs[i-2]
                prev_prev_low = lows[i-2]
                
                # Bullish FVG: Previous low > current high
                if prev_low > current_high:
                    gap_size = prev_low - current_high
                    if gap_size >= self.config.fvg_min_gap_size:
                        fvg = FairValueGap(
                            high=prev_low,
                            low=current_high,
                            gap_size=gap_size,
                            direction="bullish",
                            strength=min(1.0, gap_size / current_high),
                            timeframe=timeframe,
                            formed_at=market_data.index[i]
                        )
                        fair_value_gaps.append(fvg)
                
                # Bearish FVG: Previous high < current low
                elif prev_high < current_low:
                    gap_size = current_low - prev_high
                    if gap_size >= self.config.fvg_min_gap_size:
                        fvg = FairValueGap(
                            high=current_low,
                            low=prev_high,
                            gap_size=gap_size,
                            direction="bearish",
                            strength=min(1.0, gap_size / current_low),
                            timeframe=timeframe,
                            formed_at=market_data.index[i]
                        )
                        fair_value_gaps.append(fvg)
            
            return fair_value_gaps[:5]  # Return top 5 FVGs
            
        except Exception as e:
            logger.warning(f"Fair value gap identification failed: {e}")
            return []
    
    async def _identify_supply_demand_zones(self, market_data: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Identify supply and demand zones"""
        try:
            zones = []
            highs = market_data['high'].values
            lows = market_data['low'].values
            closes = market_data['close'].values
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(highs) - 2):
                # Swing high
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    swing_highs.append((i, highs[i]))
                
                # Swing low
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    swing_lows.append((i, lows[i]))
            
            # Create supply zones from swing highs
            for idx, price in swing_highs:
                # Check if this is a significant level
                touch_count = 0
                for j in range(max(0, idx - 20), min(len(highs), idx + 20)):
                    if abs(highs[j] - price) / price < 0.001:  # 0.1% tolerance
                        touch_count += 1
                
                if touch_count >= self.config.sd_min_touch_count:
                    zone_strength = min(1.0, touch_count / 5.0)
                    zones.append({
                        'type': 'supply',
                        'price': price,
                        'strength': zone_strength,
                        'touch_count': touch_count,
                        'timeframe': timeframe,
                        'formed_at': market_data.index[idx]
                    })
            
            # Create demand zones from swing lows
            for idx, price in swing_lows:
                # Check if this is a significant level
                touch_count = 0
                for j in range(max(0, idx - 20), min(len(lows), idx + 20)):
                    if abs(lows[j] - price) / price < 0.001:  # 0.1% tolerance
                        touch_count += 1
                
                if touch_count >= self.config.sd_min_touch_count:
                    zone_strength = min(1.0, touch_count / 5.0)
                    zones.append({
                        'type': 'demand',
                        'price': price,
                        'strength': zone_strength,
                        'touch_count': touch_count,
                        'timeframe': timeframe,
                        'formed_at': market_data.index[idx]
                    })
            
            # Filter by strength and recency
            strong_zones = [zone for zone in zones if zone['strength'] >= self.config.sd_strength_threshold]
            recent_zones = [zone for zone in strong_zones 
                          if (datetime.now() - zone['formed_at']).days < 30]
            
            return recent_zones[:8]  # Return top 8 zones
            
        except Exception as e:
            logger.warning(f"Supply demand zone identification failed: {e}")
            return []
    
    async def _identify_liquidity_levels(self, market_data: pd.DataFrame, timeframe: str) -> List[MarketLiquidity]:
        """Identify market liquidity levels"""
        try:
            liquidity_levels = []
            highs = market_data['high'].values
            lows = market_data['low'].values
            volumes = market_data['volume'].values
            
            # Find significant highs and lows with high volume
            for i in range(10, len(market_data) - 10):
                current_high = highs[i]
                current_low = lows[i]
                current_volume = volumes[i]
                
                # Check if this is a significant high
                if (current_high == np.max(highs[i-10:i+10]) and 
                    current_volume > np.mean(volumes[i-10:i]) * self.config.smc_volume_threshold):
                    
                    liquidity = MarketLiquidity(
                        level=current_high,
                        type='sell_side',
                        strength=min(1.0, current_volume / np.mean(volumes[i-10:i])),
                        timeframe=timeframe,
                        timestamp=market_data.index[i]
                    )
                    liquidity_levels.append(liquidity)
                
                # Check if this is a significant low
                if (current_low == np.min(lows[i-10:i+10]) and 
                    current_volume > np.mean(volumes[i-10:i]) * self.config.smc_volume_threshold):
                    
                    liquidity = MarketLiquidity(
                        level=current_low,
                        type='buy_side',
                        strength=min(1.0, current_volume / np.mean(volumes[i-10:i])),
                        timeframe=timeframe,
                        timestamp=market_data.index[i]
                    )
                    liquidity_levels.append(liquidity)
            
            return liquidity_levels[:6]  # Return top 6 levels
            
        except Exception as e:
            logger.warning(f"Liquidity level identification failed: {e}")
            return []
    
    async def _identify_swing_points(self, market_data: pd.DataFrame) -> List[Dict]:
        """Identify swing highs and lows"""
        try:
            swing_points = []
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            for i in range(5, len(market_data) - 5):
                # Swing high
                if all(highs[i] > highs[i-j] for j in range(1, 6)) and \
                   all(highs[i] > highs[i+j] for j in range(1, 6)):
                    swing_points.append({
                        'type': 'swing_high',
                        'price': highs[i],
                        'index': i,
                        'timestamp': market_data.index[i]
                    })
                
                # Swing low
                if all(lows[i] < lows[i-j] for j in range(1, 6)) and \
                   all(lows[i] < lows[i+j] for j in range(1, 6)):
                    swing_points.append({
                        'type': 'swing_low',
                        'price': lows[i],
                        'index': i,
                        'timestamp': market_data.index[i]
                    })
            
            return swing_points[-10:]  # Return last 10 swing points
            
        except Exception as e:
            logger.warning(f"Swing point identification failed: {e}")
            return []
    
    async def _supply_demand_signal(self, symbol: str, market_data: pd.DataFrame,
                                  market_analysis: Dict, current_price: float) -> Optional[RetailSignal]:
        """Generate signal based on supply demand zones"""
        try:
            zones = market_analysis.get('supply_demand_zones', [])
            if not zones:
                return None
            
            current_time = datetime.now()
            reasoning = []
            confidence = 0.0
            
            # Find nearest zones
            demand_zones = [z for z in zones if z['type'] == 'demand']
            supply_zones = [z for z in zones if z['type'] == 'supply']
            
            # Sort by proximity to current price
            demand_zones.sort(key=lambda z: abs(z['price'] - current_price))
            supply_zones.sort(key=lambda z: abs(z['price'] - current_price))
            
            nearest_demand = demand_zones[0] if demand_zones else None
            nearest_supply = supply_zones[0] if supply_zones else None
            
            # Check if price is approaching a zone
            if nearest_demand and current_price <= nearest_demand['price'] * 1.002:  # Within 0.2%
                # Price approaching demand zone - potential buy
                zone_strength = nearest_demand['strength']
                distance_ratio = abs(current_price - nearest_demand['price']) / current_price
                
                confidence = zone_strength * (1 - distance_ratio * 10)  # Closer is better
                reasoning.append(f"Price approaching strong demand zone (strength: {zone_strength:.2f})")
                
                if confidence >= self.config.min_confidence:
                    # Calculate risk management
                    stop_loss = nearest_demand['price'] * 0.995  # 0.5% below zone
                    take_profit = current_price + (current_price - stop_loss) * self.config.default_risk_reward
                    
                    return RetailSignal(
                        strategy_type=RetailStrategyType.SUPPLY_DEMAND,
                        symbol=symbol,
                        action="buy",
                        confidence=confidence,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=self.config.default_risk_reward,
                        timeframe=self.config.primary_timeframe,
                        market_structure=market_analysis['market_structure'],
                        reasoning=reasoning,
                        timestamp=current_time,
                        metadata={'zone_strength': zone_strength, 'zone_price': nearest_demand['price']}
                    )
            
            elif nearest_supply and current_price >= nearest_supply['price'] * 0.998:  # Within 0.2%
                # Price approaching supply zone - potential sell
                zone_strength = nearest_supply['strength']
                distance_ratio = abs(current_price - nearest_supply['price']) / current_price
                
                confidence = zone_strength * (1 - distance_ratio * 10)  # Closer is better
                reasoning.append(f"Price approaching strong supply zone (strength: {zone_strength:.2f})")
                
                if confidence >= self.config.min_confidence:
                    # Calculate risk management
                    stop_loss = nearest_supply['price'] * 1.005  # 0.5% above zone
                    take_profit = current_price - (stop_loss - current_price) * self.config.default_risk_reward
                    
                    return RetailSignal(
                        strategy_type=RetailStrategyType.SUPPLY_DEMAND,
                        symbol=symbol,
                        action="sell",
                        confidence=confidence,
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=self.config.default_risk_reward,
                        timeframe=self.config.primary_timeframe,
                        market_structure=market_analysis['market_structure'],
                        reasoning=reasoning,
                        timestamp=current_time,
                        metadata={'zone_strength': zone_strength, 'zone_price': nearest_supply['price']}
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Supply demand signal generation failed: {e}")
            return None
    
    async def _smart_money_signal(self, symbol: str, market_data: pd.DataFrame,
                                market_analysis: Dict, current_price: float) -> Optional[RetailSignal]:
        """Generate signal based on Smart Money Concept"""
        try:
            liquidity_levels = market_analysis.get('liquidity_levels', [])
            order_blocks = market_analysis.get('order_blocks', [])
            market_structure = market_analysis['market_structure']
            
            if not liquidity_levels or not order_blocks:
                return None
            
            current_time = datetime.now()
            reasoning = []
            confidence = 0.0
            
            # Find nearest liquidity levels
            buy_liquidity = [l for l in liquidity_levels if l.type == 'buy_side']
            sell_liquidity = [l for l in liquidity_levels if l.type == 'sell_side']
            
            buy_liquidity.sort(key=lambda l: abs(l.level - current_price))
            sell_liquidity.sort(key=lambda l: abs(l.level - current_price))
            
            nearest_buy_liquidity = buy_liquidity[0] if buy_liquidity else None
            nearest_sell_liquidity = sell_liquidity[0] if sell_liquidity else None
            
            # Find valid order blocks
            valid_bullish_obs = [ob for ob in order_blocks 
                               if ob.block_type == OrderBlockType.BULLISH_OB and not ob.is_breached]
            valid_bearish_obs = [ob for ob in order_blocks 
                               if ob.block_type == OrderBlockType.BEARISH_OB and not ob.is_breached]
            
            # Smart Money Logic: Liquidity grab + order block
            if (nearest_sell_liquidity and 
                current_price >= nearest_sell_liquidity.level * 0.999 and  # Near sell liquidity
                valid_bullish_obs and market_structure == MarketStructure.BULLISH):
                
                # Potential liquidity grab above, then move down to bullish OB
                reasoning.append("Potential sell-side liquidity grab detected")
                reasoning.append("Bullish order block present for reaction")
                
                # Find best bullish OB for entry
                valid_bullish_obs.sort(key=lambda ob: ob.low)  # Lowest OB first
                best_ob = valid_bullish_obs[0]
                
                confidence = min(1.0, best_ob.strength * 0.7 + nearest_sell_liquidity.strength * 0.3)
                
                if confidence >= self.config.min_confidence:
                    entry_price = best_ob.low * 1.001  # Just above OB low
                    stop_loss = best_ob.low * 0.995
                    take_profit = entry_price + (entry_price - stop_loss) * self.config.default_risk_reward
                    
                    return RetailSignal(
                        strategy_type=RetailStrategyType.SMART_MONEY_CONCEPT,
                        symbol=symbol,
                        action="buy",
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=self.config.default_risk_reward,
                        timeframe=self.config.primary_timeframe,
                        market_structure=market_structure,
                        reasoning=reasoning,
                        timestamp=current_time,
                        metadata={
                            'liquidity_level': nearest_sell_liquidity.level,
                            'order_block_low': best_ob.low,
                            'order_block_high': best_ob.high
                        }
                    )
            
            elif (nearest_buy_liquidity and 
                  current_price <= nearest_buy_liquidity.level * 1.001 and  # Near buy liquidity
                  valid_bearish_obs and market_structure == MarketStructure.BEARISH):
                
                # Potential liquidity grab below, then move up to bearish OB
                reasoning.append("Potential buy-side liquidity grab detected")
                reasoning.append("Bearish order block present for reaction")
                
                # Find best bearish OB for entry
                valid_bearish_obs.sort(key=lambda ob: ob.high, reverse=True)  # Highest OB first
                best_ob = valid_bearish_obs[0]
                
                confidence = min(1.0, best_ob.strength * 0.7 + nearest_buy_liquidity.strength * 0.3)
                
                if confidence >= self.config.min_confidence:
                    entry_price = best_ob.high * 0.999  # Just below OB high
                    stop_loss = best_ob.high * 1.005
                    take_profit = entry_price - (stop_loss - entry_price) * self.config.default_risk_reward
                    
                    return RetailSignal(
                        strategy_type=RetailStrategyType.SMART_MONEY_CONCEPT,
                        symbol=symbol,
                        action="sell",
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=self.config.default_risk_reward,
                        timeframe=self.config.primary_timeframe,
                        market_structure=market_structure,
                        reasoning=reasoning,
                        timestamp=current_time,
                        metadata={
                            'liquidity_level': nearest_buy_liquidity.level,
                            'order_block_low': best_ob.low,
                            'order_block_high': best_ob.high
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Smart money signal generation failed: {e}")
            return None
    
    async def _order_block_signal(self, symbol: str, market_data: pd.DataFrame,
                                market_analysis: Dict, current_price: float) -> Optional[RetailSignal]:
        """Generate signal based on order blocks"""
        try:
            order_blocks = market_analysis.get('order_blocks', [])
            if not order_blocks:
                return None
            
            current_time = datetime.now()
            reasoning = []
            
            # Find order blocks that haven't been activated yet
            inactive_bullish_obs = [ob for ob in order_blocks 
                                  if ob.block_type == OrderBlockType.BULLISH_OB and not ob.is_activated]
            inactive_bearish_obs = [ob for ob in order_blocks 
                                  if ob.block_type == OrderBlockType.BEARISH_OB and not ob.is_activated]
            
            # Check for bullish OB activation
            for ob in inactive_bullish_obs:
                if current_price <= ob.high * (1 + self.config.ob_activation_zone) and \
                   current_price >= ob.low * (1 - self.config.ob_activation_zone):
                    
                    reasoning.append(f"Bullish order block activated (strength: {ob.strength:.2f})")
                    reasoning.append(f"Block range: {ob.low:.4f} - {ob.high:.4f}")
                    
                    confidence = ob.strength * 0.8
                    
                    if confidence >= self.config.min_confidence:
                        entry_price = current_price
                        stop_loss = ob.low * 0.995
                        take_profit = entry_price + (entry_price - stop_loss) * self.config.default_risk_reward
                        
                        # Mark OB as activated
                        ob.is_activated = True
                        
                        return RetailSignal(
                            strategy_type=RetailStrategyType.ORDER_BLOCK,
                            symbol=symbol,
                            action="buy",
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=self.config.default_risk_reward,
                            timeframe=self.config.primary_timeframe,
                            market_structure=market_analysis['market_structure'],
                            reasoning=reasoning,
                            timestamp=current_time,
                            metadata={
                                'order_block_low': ob.low,
                                'order_block_high': ob.high,
                                'block_strength': ob.strength
                            }
                        )
            
            # Check for bearish OB activation
            for ob in inactive_bearish_obs:
                if current_price <= ob.high * (1 + self.config.ob_activation_zone) and \
                   current_price >= ob.low * (1 - self.config.ob_activation_zone):
                    
                    reasoning.append(f"Bearish order block activated (strength: {ob.strength:.2f})")
                    reasoning.append(f"Block range: {ob.low:.4f} - {ob.high:.4f}")
                    
                    confidence = ob.strength * 0.8
                    
                    if confidence >= self.config.min_confidence:
                        entry_price = current_price
                        stop_loss = ob.high * 1.005
                        take_profit = entry_price - (stop_loss - entry_price) * self.config.default_risk_reward
                        
                        # Mark OB as activated
                        ob.is_activated = True
                        
                        return RetailSignal(
                            strategy_type=RetailStrategyType.ORDER_BLOCK,
                            symbol=symbol,
                            action="sell",
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=self.config.default_risk_reward,
                            timeframe=self.config.primary_timeframe,
                            market_structure=market_analysis['market_structure'],
                            reasoning=reasoning,
                            timestamp=current_time,
                            metadata={
                                'order_block_low': ob.low,
                                'order_block_high': ob.high,
                                'block_strength': ob.strength
                            }
                        )
            
            return None
            
        except Exception as e:
            logger.warning(f"Order block signal generation failed: {e}")
            return None
    
    async def _fair_value_gap_signal(self, symbol: str, market_data: pd.DataFrame,
                                   market_analysis: Dict, current_price: float) -> Optional[RetailSignal]:
        """Generate signal based on fair value gaps"""
        try:
            fvgs = market_analysis.get('fair_value_gaps', [])
            if not fvgs:
                return None
            
            current_time = datetime.now()
            reasoning = []
            
            # Find unfilled FVGs
            unfilled_bullish_fvgs = [fvg for fvg in fvgs 
                                   if fvg.direction == "bullish" and not fvg.is_filled]
            unfilled_bearish_fvgs = [fvg for fvg in fvgs 
                                   if fvg.direction == "bearish" and not fvg.is_filled]
            
            # Check for bullish FVG fill (price enters FVG from below)
            for fvg in unfilled_bullish_fvgs:
                if current_price >= fvg.low and current_price <= fvg.high:
                    reasoning.append(f"Bullish FVG being filled (size: {fvg.gap_size:.4f})")
                    reasoning.append("Expecting continuation in FVG direction")
                    
                    confidence = fvg.strength * 0.7
                    
                    if confidence >= self.config.min_confidence:
                        entry_price = current_price
                        stop_loss = fvg.low * 0.995
                        take_profit = fvg.high + (fvg.high - fvg.low) * 1.5  # 1.5x FVG size
                        
                        # Mark FVG as filled
                        fvg.is_filled = True
                        
                        return RetailSignal(
                            strategy_type=RetailStrategyType.FAIR_VALUE_GAP,
                            symbol=symbol,
                            action="buy",
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=self.config.default_risk_reward,
                            timeframe=self.config.primary_timeframe,
                            market_structure=market_analysis['market_structure'],
                            reasoning=reasoning,
                            timestamp=current_time,
                            metadata={
                                'fvg_low': fvg.low,
                                'fvg_high': fvg.high,
                                'gap_size': fvg.gap_size
                            }
                        )
            
            # Check for bearish FVG fill (price enters FVG from above)
            for fvg in unfilled_bearish_fvgs:
                if current_price >= fvg.low and current_price <= fvg.high:
                    reasoning.append(f"Bearish FVG being filled (size: {fvg.gap_size:.4f})")
                    reasoning.append("Expecting continuation in FVG direction")
                    
                    confidence = fvg.strength * 0.7
                    
                    if confidence >= self.config.min_confidence:
                        entry_price = current_price
                        stop_loss = fvg.high * 1.005
                        take_profit = fvg.low - (fvg.high - fvg.low) * 1.5  # 1.5x FVG size
                        
                        # Mark FVG as filled
                        fvg.is_filled = True
                        
                        return RetailSignal(
                            strategy_type=RetailStrategyType.FAIR_VALUE_GAP,
                            symbol=symbol,
                            action="sell",
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=self.config.default_risk_reward,
                            timeframe=self.config.primary_timeframe,
                            market_structure=market_analysis['market_structure'],
                            reasoning=reasoning,
                            timestamp=current_time,
                            metadata={
                                'fvg_low': fvg.low,
                                'fvg_high': fvg.high,
                                'gap_size': fvg.gap_size
                            }
                        )
            
            return None
            
        except Exception as e:
            logger.warning(f"Fair value gap signal generation failed: {e}")
            return None
    
    async def _breaker_signal(self, symbol: str, market_data: pd.DataFrame,
                            market_analysis: Dict, current_price: float) -> Optional[RetailSignal]:
        """Generate signal based on breaker blocks"""
        try:
            # Breaker blocks are order blocks that get "broken" then act as support/resistance
            order_blocks = market_analysis.get('order_blocks', [])
            if not order_blocks:
                return None
            
            current_time = datetime.now()
            reasoning = []
            
            # Find order blocks that have been breached but not yet acted as breaker
            breached_bullish_obs = [ob for ob in order_blocks 
                                  if ob.block_type == OrderBlockType.BULLISH_OB and ob.is_breached and not ob.is_activated]
            breached_bearish_obs = [ob for ob in order_blocks 
                                  if ob.block_type == OrderBlockType.BEARISH_OB and ob.is_breached and not ob.is_activated]
            
            # Check for bullish breaker (price returns to breached bullish OB)
            for ob in breached_bullish_obs:
                if current_price >= ob.low and current_price <= ob.high:
                    reasoning.append("Price returning to breached bullish order block (Breaker)")
                    reasoning.append("Expecting rejection and continuation down")
                    
                    confidence = ob.strength * 0.6
                    
                    if confidence >= self.config.min_confidence:
                        entry_price = current_price
                        stop_loss = ob.high * 1.005
                        take_profit = entry_price - (stop_loss - entry_price) * self.config.default_risk_reward
                        
                        return RetailSignal(
                            strategy_type=RetailStrategyType.BREAKER,
                            symbol=symbol,
                            action="sell",
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=self.config.default_risk_reward,
                            timeframe=self.config.primary_timeframe,
                            market_structure=market_analysis['market_structure'],
                            reasoning=reasoning,
                            timestamp=current_time,
                            metadata={
                                'breaker_low': ob.low,
                                'breaker_high': ob.high
                            }
                        )
            
            # Check for bearish breaker (price returns to breached bearish OB)
            for ob in breached_bearish_obs:
                if current_price >= ob.low and current_price <= ob.high:
                    reasoning.append("Price returning to breached bearish order block (Breaker)")
                    reasoning.append("Expecting rejection and continuation up")
                    
                    confidence = ob.strength * 0.6
                    
                    if confidence >= self.config.min_confidence:
                        entry_price = current_price
                        stop_loss = ob.low * 0.995
                        take_profit = entry_price + (entry_price - stop_loss) * self.config.default_risk_reward
                        
                        return RetailSignal(
                            strategy_type=RetailStrategyType.BREAKER,
                            symbol=symbol,
                            action="buy",
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=self.config.default_risk_reward,
                            timeframe=self.config.primary_timeframe,
                            market_structure=market_analysis['market_structure'],
                            reasoning=reasoning,
                            timestamp=current_time,
                            metadata={
                                'breaker_low': ob.low,
                                'breaker_high': ob.high
                            }
                        )
            
            return None
            
        except Exception as e:
            logger.warning(f"Breaker signal generation failed: {e}")
            return None
    
    async def _liquidity_grab_signal(self, symbol: str, market_data: pd.DataFrame,
                                   market_analysis: Dict, current_price: float) -> Optional[RetailSignal]:
        """Generate signal based on liquidity grabs"""
        try:
            liquidity_levels = market_analysis.get('liquidity_levels', [])
            if not liquidity_levels:
                return None
            
            current_time = datetime.now()
            reasoning = []
            
            # Find recent liquidity levels
            recent_liquidity = [l for l in liquidity_levels 
                              if (datetime.now() - l.timestamp).total_seconds() < 86400]  # Last 24 hours
            
            # Check for liquidity grabs (price moves beyond liquidity then reverses)
            for liquidity in recent_liquidity:
                if liquidity.type == 'sell_side':
                    # Price moved above sell-side liquidity and is now pulling back
                    recent_high = np.max(market_data['high'].values[-10:])
                    if recent_high >= liquidity.level and current_price < liquidity.level * 0.998:
                        reasoning.append("Sell-side liquidity grab detected")
                        reasoning.append("Expecting move down after liquidity taken")
                        
                        confidence = liquidity.strength * 0.5
                        
                        if confidence >= self.config.min_confidence:
                            entry_price = current_price
                            stop_loss = liquidity.level * 1.005
                            take_profit = entry_price - (stop_loss - entry_price) * self.config.default_risk_reward
                            
                            return RetailSignal(
                                strategy_type=RetailStrategyType.LIQUIDITY_GRAB,
                                symbol=symbol,
                                action="sell",
                                confidence=confidence,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=self.config.default_risk_reward,
                                timeframe=self.config.primary_timeframe,
                                market_structure=market_analysis['market_structure'],
                                reasoning=reasoning,
                                timestamp=current_time,
                                metadata={'liquidity_level': liquidity.level}
                            )
                
                elif liquidity.type == 'buy_side':
                    # Price moved below buy-side liquidity and is now pulling back
                    recent_low = np.min(market_data['low'].values[-10:])
                    if recent_low <= liquidity.level and current_price > liquidity.level * 1.002:
                        reasoning.append("Buy-side liquidity grab detected")
                        reasoning.append("Expecting move up after liquidity taken")
                        
                        confidence = liquidity.strength * 0.5
                        
                        if confidence >= self.config.min_confidence:
                            entry_price = current_price
                            stop_loss = liquidity.level * 0.995
                            take_profit = entry_price + (entry_price - stop_loss) * self.config.default_risk_reward
                            
                            return RetailSignal(
                                strategy_type=RetailStrategyType.LIQUIDITY_GRAB,
                                symbol=symbol,
                                action="buy",
                                confidence=confidence,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=self.config.default_risk_reward,
                                timeframe=self.config.primary_timeframe,
                                market_structure=market_analysis['market_structure'],
                                reasoning=reasoning,
                                timestamp=current_time,
                                metadata={'liquidity_level': liquidity.level}
                            )
            
            return None
            
        except Exception as e:
            logger.warning(f"Liquidity grab signal generation failed: {e}")
            return None
    
    def get_strategy_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive strategy analysis for symbol"""
        with self._lock:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_structure': self.market_structure.get(symbol, MarketStructure.RANGING).value,
                'order_blocks_count': len(self.order_blocks.get(symbol, [])),
                'fair_value_gaps_count': len(self.fair_value_gaps.get(symbol, [])),
                'supply_demand_zones_count': len(self.supply_demand_zones.get(symbol, [])),
                'liquidity_levels_count': len(self.liquidity_levels.get(symbol, [])),
                'recent_signals': []
            }
            
            # Add recent signals
            recent_signals = [s for s in self.signal_history 
                            if s.symbol == symbol and 
                            (datetime.now() - s.timestamp).total_seconds() < 3600]  # Last hour
            
            for signal in recent_signals[-5:]:  # Last 5 signals
                analysis['recent_signals'].append({
                    'strategy': signal.strategy_type.value,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'timestamp': signal.timestamp.isoformat()
                })
            
            return analysis
    
    def save_strategy_state(self, filename: str = "retail_strategies_state.json") -> None:
        """Save current strategy state to file"""
        try:
            with self._lock:
                state = {
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config.__dict__,
                    'market_structures': {k: v.value for k, v in self.market_structure.items()},
                    'signal_history_count': len(self.signal_history)
                }
                
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2, default=str)
                
                logger.info(f"Strategy state saved to {filename}")
                
        except Exception as e:
            logger.error(f"Strategy state saving failed: {e}")

# Example usage and testing
async def main():
    """Example usage of the AdvancedRetailStrategies"""
    
    print("=== Testing Advanced Retail Strategies ===")
    
    # Generate sample market data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
    
    # Create market data with some structure
    base_trend = np.cumsum(np.random.normal(0.0002, 0.003, 200))
    noise = np.random.normal(0, 0.001, 200)
    prices = 1.1000 + base_trend + noise
    
    # Add some clear swing points
    prices[50] = 1.1050  # Swing high
    prices[80] = 1.0950  # Swing low
    prices[120] = 1.1080  # Swing high
    prices[150] = 1.0920  # Swing low
    
    market_data = pd.DataFrame({
        'open': prices * 0.9995,
        'high': prices * 1.001 + np.abs(np.random.normal(0, 0.0003, 200)),
        'low': prices * 0.998 - np.abs(np.random.normal(0, 0.0003, 200)),
        'close': prices,
        'volume': np.random.lognormal(8, 1, 200)
    }, index=dates)
    
    print(f"Generated {len(market_data)} periods of market data")
    
    # Configure retail strategies
    config = RetailStrategyConfig(
        enabled_strategies=[
            RetailStrategyType.SUPPLY_DEMAND,
            RetailStrategyType.SMART_MONEY_CONCEPT,
            RetailStrategyType.ORDER_BLOCK,
            RetailStrategyType.FAIR_VALUE_GAP
        ],
        min_confidence=0.6,
        default_risk_reward=1.5
    )
    
    # Initialize strategies
    strategies = AdvancedRetailStrategies(config)
    
    print("\n=== Performing Market Analysis ===")
    analysis = await strategies.analyze_market("EUR/USD", market_data, "1h")
    
    print(f"Market Structure: {analysis['market_structure'].value}")
    print(f"Order Blocks Found: {len(analysis['order_blocks'])}")
    print(f"Fair Value Gaps Found: {len(analysis['fair_value_gaps'])}")
    print(f"Supply/Demand Zones: {len(analysis['supply_demand_zones'])}")
    print(f"Liquidity Levels: {len(analysis['liquidity_levels'])}")
    
    if analysis['order_blocks']:
        print("\nOrder Blocks:")
        for i, ob in enumerate(analysis['order_blocks'][:3]):
            print(f"  {i+1}. {ob.block_type.value} - Strength: {ob.strength:.2f}")
    
    if analysis['supply_demand_zones']:
        print("\nSupply/Demand Zones:")
        for i, zone in enumerate(analysis['supply_demand_zones'][:3]):
            print(f"  {i+1}. {zone['type']} at {zone['price']:.4f} - Strength: {zone['strength']:.2f}")
    
    print("\n=== Generating Trading Signals ===")
    signals = await strategies.generate_signals("EUR/USD", market_data, "1h")
    
    print(f"Generated {len(signals)} signals:")
    for i, signal in enumerate(signals):
        print(f"\nSignal {i+1}:")
        print(f"  Strategy: {signal.strategy_type.value}")
        print(f"  Action: {signal.action}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Entry: {signal.entry_price:.4f}")
        print(f"  Stop Loss: {signal.stop_loss:.4f}")
        print(f"  Take Profit: {signal.take_profit:.4f}")
        print(f"  R:R Ratio: {signal.risk_reward_ratio:.2f}")
        print("  Reasoning:")
        for reason in signal.reasoning:
            print(f"    - {reason}")
    
    print("\n=== Strategy Analysis Report ===")
    report = strategies.get_strategy_analysis("EUR/USD")
    
    print(f"Symbol: {report['symbol']}")
    print(f"Market Structure: {report['market_structure']}")
    print(f"Order Blocks: {report['order_blocks_count']}")
    print(f"Fair Value Gaps: {report['fair_value_gaps_count']}")
    print(f"Supply/Demand Zones: {report['supply_demand_zones_count']}")
    print(f"Liquidity Levels: {report['liquidity_levels_count']}")
    
    if report['recent_signals']:
        print("\nRecent Signals:")
        for signal in report['recent_signals']:
            print(f"  {signal['strategy']}: {signal['action']} (confidence: {signal['confidence']:.3f})")
    
    # Save strategy state
    strategies.save_strategy_state("test_retail_strategies_state.json")
    print("\n=== Strategy State Saved ===")
    
    print("\n=== Retail Strategies Test Completed ===")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run async main
    asyncio.run(main())
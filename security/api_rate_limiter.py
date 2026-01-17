"""
Advanced API Rate Limiter for FOREX TRADING BOT
Professional rate limiting with multiple algorithms and real-time monitoring
"""

import logging
import time
import threading
import asyncio
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import redis
import heapq
from collections import defaultdict, deque
import functools
import inspect
from contextlib import contextmanager
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_LOG = "sliding_window_log"

class RateLimitStrategy(Enum):
    FAIL_FAST = "fail_fast"
    BLOCKING = "blocking"
    SLOW_DOWN = "slow_down"
    DYNAMIC_BACKOFF = "dynamic_backoff"

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    # Basic limits
    requests_per_second: float = 10.0
    requests_per_minute: float = 300.0
    requests_per_hour: float = 10000.0
    burst_capacity: int = 50
    
    # Algorithm settings
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    strategy: RateLimitStrategy = RateLimitStrategy.DYNAMIC_BACKOFF
    
    # Redis settings (for distributed rate limiting)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Advanced settings
    enable_dynamic_limits: bool = True
    enable_cost_aware: bool = True
    enable_priority: bool = True
    enable_retry_after: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    monitor_window: int = 300  # 5 minutes
    
    # Cost weights for different API endpoints
    endpoint_costs: Dict[str, float] = field(default_factory=lambda: {
        'market_data': 1.0,
        'order_status': 0.5,
        'place_order': 2.0,
        'cancel_order': 1.5,
        'account_info': 0.3,
        'historical_data': 3.0
    })

@dataclass
class RateLimitStats:
    """Statistics for rate limiting"""
    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    throttled_requests: int = 0
    total_wait_time: float = 0.0
    current_rps: float = 0.0
    peak_rps: float = 0.0
    endpoint_stats: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    limit_remaining: int
    reset_time: float
    wait_time: float = 0.0
    retry_after: float = 0.0
    cost: float = 1.0
    endpoint: str = "default"
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

class APIRateLimiter:
    """
    Advanced API Rate Limiter with multiple algorithms and real-time monitoring
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        
        # Rate limit stores
        self._token_buckets: Dict[str, Dict] = {}
        self._window_counts: Dict[str, Dict] = {}
        self._request_logs: Dict[str, deque] = {}
        
        # Statistics
        self.stats = RateLimitStats()
        self._request_timestamps: deque = deque()
        self._peak_rps_window: deque = deque()
        
        # Redis client for distributed rate limiting
        self.redis_client = None
        self._init_redis()
        
        # Thread safety
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # Dynamic limits
        self._dynamic_limits: Dict[str, float] = {}
        self._limit_adjustment_factor: float = 1.0
        
        # Priority queues
        self._priority_queues: Dict[int, deque] = defaultdict(deque)
        
        # Monitoring
        self._monitoring_data: deque = deque(maxlen=self.config.monitor_window)
        
        logger.info(f"APIRateLimiter initialized with {self.config.algorithm.value} algorithm")
    
    def _init_redis(self) -> None:
        """Initialize Redis client for distributed rate limiting"""
        try:
            if self.config.redis_host:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                self.redis_client.ping()
                logger.info("Redis client connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using local rate limiting only.")
            self.redis_client = None
    
    def _update_stats(self, endpoint: str, allowed: bool, wait_time: float = 0.0) -> None:
        """Update rate limiting statistics"""
        with self._lock:
            current_time = time.time()
            
            # Update basic stats
            self.stats.total_requests += 1
            if allowed:
                self.stats.allowed_requests += 1
                if wait_time > 0:
                    self.stats.throttled_requests += 1
                    self.stats.total_wait_time += wait_time
            else:
                self.stats.blocked_requests += 1
            
            # Update request timestamps for RPS calculation
            self._request_timestamps.append(current_time)
            
            # Remove old timestamps (older than 1 second)
            one_second_ago = current_time - 1.0
            while self._request_timestamps and self._request_timestamps[0] < one_second_ago:
                self._request_timestamps.popleft()
            
            # Calculate current RPS
            self.stats.current_rps = len(self._request_timestamps)
            self.stats.peak_rps = max(self.stats.peak_rps, self.stats.current_rps)
            
            # Update endpoint-specific stats
            if endpoint not in self.stats.endpoint_stats:
                self.stats.endpoint_stats[endpoint] = {
                    'total_requests': 0,
                    'allowed_requests': 0,
                    'blocked_requests': 0,
                    'throttled_requests': 0,
                    'total_wait_time': 0.0
                }
            
            endpoint_stats = self.stats.endpoint_stats[endpoint]
            endpoint_stats['total_requests'] += 1
            if allowed:
                endpoint_stats['allowed_requests'] += 1
                if wait_time > 0:
                    endpoint_stats['throttled_requests'] += 1
                    endpoint_stats['total_wait_time'] += wait_time
            else:
                endpoint_stats['blocked_requests'] += 1
            
            # Update monitoring data
            self._monitoring_data.append({
                'timestamp': current_time,
                'endpoint': endpoint,
                'allowed': allowed,
                'wait_time': wait_time,
                'current_rps': self.stats.current_rps
            })
    
    def _get_endpoint_cost(self, endpoint: str) -> float:
        """Get cost weight for an endpoint"""
        return self.config.endpoint_costs.get(endpoint, 1.0)
    
    def _calculate_dynamic_limit(self, endpoint: str) -> float:
        """Calculate dynamic rate limit based on recent performance"""
        if not self.config.enable_dynamic_limits:
            return self.config.requests_per_second
        
        # Simple dynamic adjustment based on recent success rate
        recent_requests = list(self._monitoring_data)[-100:]  # Last 100 requests
        if len(recent_requests) < 10:
            return self.config.requests_per_second
        
        success_count = sum(1 for req in recent_requests if req['allowed'])
        success_rate = success_count / len(recent_requests)
        
        # Adjust limit based on success rate
        if success_rate > 0.95:
            # Increase limit if we're doing well
            adjustment = 1.2
        elif success_rate < 0.8:
            # Decrease limit if we're having issues
            adjustment = 0.8
        else:
            adjustment = 1.0
        
        return self.config.requests_per_second * adjustment
    
    def _token_bucket_algorithm(self, endpoint: str, cost: float = 1.0) -> RateLimitResult:
        """Token Bucket rate limiting algorithm"""
        with self._lock:
            current_time = time.time()
            bucket_key = f"token_bucket:{endpoint}"
            
            if bucket_key not in self._token_buckets:
                # Initialize bucket
                self._token_buckets[bucket_key] = {
                    'tokens': self.config.burst_capacity,
                    'last_refill': current_time
                }
            
            bucket = self._token_buckets[bucket_key]
            
            # Refill tokens based on time passed
            time_passed = current_time - bucket['last_refill']
            tokens_to_add = time_passed * self.config.requests_per_second
            bucket['tokens'] = min(
                self.config.burst_capacity,
                bucket['tokens'] + tokens_to_add
            )
            bucket['last_refill'] = current_time
            
            # Check if we have enough tokens
            if bucket['tokens'] >= cost:
                bucket['tokens'] -= cost
                wait_time = 0.0
                allowed = True
            else:
                # Calculate wait time
                tokens_needed = cost - bucket['tokens']
                wait_time = tokens_needed / self.config.requests_per_second
                allowed = False
            
            limit_remaining = int(bucket['tokens'])
            reset_time = current_time + ((self.config.burst_capacity - bucket['tokens']) / self.config.requests_per_second)
            
            return RateLimitResult(
                allowed=allowed,
                limit_remaining=limit_remaining,
                reset_time=reset_time,
                wait_time=wait_time,
                retry_after=wait_time,
                cost=cost,
                endpoint=endpoint
            )
    
    def _fixed_window_algorithm(self, endpoint: str, cost: float = 1.0) -> RateLimitResult:
        """Fixed Window rate limiting algorithm"""
        with self._lock:
            current_time = time.time()
            window_key = f"fixed_window:{endpoint}:{int(current_time)}"
            
            if window_key not in self._window_counts:
                # Initialize new window
                self._window_counts[window_key] = {
                    'count': 0,
                    'window_start': current_time
                }
                # Clean up old windows
                self._cleanup_old_windows()
            
            window = self._window_counts[window_key]
            
            # Check limit
            if window['count'] + cost <= self.config.requests_per_second:
                window['count'] += cost
                allowed = True
                wait_time = 0.0
            else:
                allowed = False
                # Wait until next window
                wait_time = 1.0 - (current_time - window['window_start'])
            
            limit_remaining = int(self.config.requests_per_second - window['count'])
            reset_time = window['window_start'] + 1.0
            
            return RateLimitResult(
                allowed=allowed,
                limit_remaining=limit_remaining,
                reset_time=reset_time,
                wait_time=wait_time,
                retry_after=wait_time,
                cost=cost,
                endpoint=endpoint
            )
    
    def _sliding_window_algorithm(self, endpoint: str, cost: float = 1.0) -> RateLimitResult:
        """Sliding Window rate limiting algorithm"""
        with self._lock:
            current_time = time.time()
            window_key = f"sliding_window:{endpoint}"
            
            if window_key not in self._request_logs:
                self._request_logs[window_key] = deque()
            
            request_log = self._request_logs[window_key]
            
            # Remove requests outside the current window (1 second)
            window_start = current_time - 1.0
            while request_log and request_log[0] < window_start:
                request_log.popleft()
            
            # Calculate current window count
            current_count = sum(1 for _ in request_log)
            
            # Check limit
            if current_count + cost <= self.config.requests_per_second:
                request_log.append(current_time)
                allowed = True
                wait_time = 0.0
            else:
                allowed = False
                # Wait until oldest request expires
                oldest_request = request_log[0] if request_log else current_time
                wait_time = max(0.0, oldest_request + 1.0 - current_time)
            
            limit_remaining = int(self.config.requests_per_second - current_count)
            reset_time = current_time + wait_time if wait_time > 0 else current_time
            
            return RateLimitResult(
                allowed=allowed,
                limit_remaining=limit_remaining,
                reset_time=reset_time,
                wait_time=wait_time,
                retry_after=wait_time,
                cost=cost,
                endpoint=endpoint
            )
    
    def _leaky_bucket_algorithm(self, endpoint: str, cost: float = 1.0) -> RateLimitResult:
        """Leaky Bucket rate limiting algorithm"""
        with self._lock:
            current_time = time.time()
            bucket_key = f"leaky_bucket:{endpoint}"
            
            if bucket_key not in self._token_buckets:
                # Initialize bucket
                self._token_buckets[bucket_key] = {
                    'water_level': 0.0,
                    'last_leak': current_time
                }
            
            bucket = self._token_buckets[bucket_key]
            
            # Leak water based on time passed
            time_passed = current_time - bucket['last_leak']
            water_leaked = time_passed * self.config.requests_per_second
            bucket['water_level'] = max(0.0, bucket['water_level'] - water_leaked)
            bucket['last_leak'] = current_time
            
            # Check if we can add more water
            if bucket['water_level'] + cost <= self.config.burst_capacity:
                bucket['water_level'] += cost
                allowed = True
                wait_time = 0.0
            else:
                allowed = False
                # Calculate wait time until enough water leaks
                water_to_leak = (bucket['water_level'] + cost) - self.config.burst_capacity
                wait_time = water_to_leak / self.config.requests_per_second
            
            limit_remaining = int(self.config.burst_capacity - bucket['water_level'])
            reset_time = current_time + (bucket['water_level'] / self.config.requests_per_second)
            
            return RateLimitResult(
                allowed=allowed,
                limit_remaining=limit_remaining,
                reset_time=reset_time,
                wait_time=wait_time,
                retry_after=wait_time,
                cost=cost,
                endpoint=endpoint
            )
    
    def _cleanup_old_windows(self) -> None:
        """Clean up old window data to prevent memory leaks"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, window in self._window_counts.items():
            if current_time - window['window_start'] > 2.0:  # Keep only recent windows
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._window_counts[key]
    
    def check_rate_limit(self, endpoint: str = "default", 
                        cost: float = None,
                        priority: int = 1) -> RateLimitResult:
        """
        Check if request is allowed under rate limits
        
        Args:
            endpoint: API endpoint identifier
            cost: Cost of the request (default: endpoint cost)
            priority: Request priority (1-10, 1=highest)
        
        Returns:
            RateLimitResult with decision and metadata
        """
        try:
            if cost is None:
                cost = self._get_endpoint_cost(endpoint)
            
            # Get dynamic limit if enabled
            effective_limit = self._calculate_dynamic_limit(endpoint)
            
            # Select algorithm
            if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                result = self._token_bucket_algorithm(endpoint, cost)
            elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                result = self._fixed_window_algorithm(endpoint, cost)
            elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                result = self._sliding_window_algorithm(endpoint, cost)
            elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                result = self._leaky_bucket_algorithm(endpoint, cost)
            else:
                # Default to token bucket
                result = self._token_bucket_algorithm(endpoint, cost)
            
            # Apply strategy
            final_result = self._apply_strategy(result, priority)
            
            # Update statistics
            self._update_stats(endpoint, final_result.allowed, final_result.wait_time)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {endpoint}: {e}")
            # Allow request if rate limiting fails
            return RateLimitResult(
                allowed=True,
                limit_remaining=999,
                reset_time=time.time() + 1.0,
                wait_time=0.0,
                retry_after=0.0,
                cost=cost or 1.0,
                endpoint=endpoint,
                priority=priority
            )
    
    def _apply_strategy(self, result: RateLimitResult, priority: int) -> RateLimitResult:
        """Apply rate limiting strategy to the result"""
        if result.allowed:
            return result
        
        if self.config.strategy == RateLimitStrategy.FAIL_FAST:
            # Immediately deny request
            return result
        
        elif self.config.strategy == RateLimitStrategy.BLOCKING:
            # Block until request can be processed
            if result.wait_time > 0:
                time.sleep(result.wait_time)
                result.allowed = True
                result.wait_time = 0.0
            return result
        
        elif self.config.strategy == RateLimitStrategy.SLOW_DOWN:
            # Add jitter and slow down
            jitter = (result.wait_time * 0.1) * (1 - (priority / 10.0))
            actual_wait = result.wait_time + jitter
            if actual_wait < 0.1:  # Don't wait for very short times
                time.sleep(actual_wait)
                result.allowed = True
                result.wait_time = 0.0
            return result
        
        elif self.config.strategy == RateLimitStrategy.DYNAMIC_BACKOFF:
            # Dynamic backoff based on priority and system load
            load_factor = min(1.0, self.stats.current_rps / self.config.requests_per_second)
            priority_factor = (11 - priority) / 10.0  # Higher priority = lower factor
            
            backoff_time = result.wait_time * load_factor * priority_factor
            jitter = backoff_time * 0.2 * (1 - load_factor)
            
            actual_wait = max(0.0, backoff_time + jitter)
            
            if actual_wait < 0.5:  # Wait only for reasonable times
                time.sleep(actual_wait)
                result.allowed = True
                result.wait_time = actual_wait
            else:
                result.retry_after = actual_wait
            
            return result
        
        else:
            return result
    
    async def check_rate_limit_async(self, endpoint: str = "default",
                                   cost: float = None,
                                   priority: int = 1) -> RateLimitResult:
        """Async version of check_rate_limit"""
        try:
            # Run synchronous check in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.check_rate_limit, endpoint, cost, priority
            )
            return result
        except Exception as e:
            logger.error(f"Async rate limit check failed for {endpoint}: {e}")
            return RateLimitResult(
                allowed=True,
                limit_remaining=999,
                reset_time=time.time() + 1.0,
                wait_time=0.0,
                retry_after=0.0,
                cost=cost or 1.0,
                endpoint=endpoint,
                priority=priority
            )
    
    def wait_if_needed(self, result: RateLimitResult) -> bool:
        """Wait if needed based on rate limit result"""
        if result.allowed and result.wait_time > 0:
            time.sleep(result.wait_time)
            return True
        return result.allowed
    
    async def wait_if_needed_async(self, result: RateLimitResult) -> bool:
        """Async wait if needed based on rate limit result"""
        if result.allowed and result.wait_time > 0:
            await asyncio.sleep(result.wait_time)
            return True
        return result.allowed
    
    # Decorator for synchronous functions
    def rate_limit(self, endpoint: str = "default", cost: float = None, priority: int = 1):
        """Decorator for rate limiting synchronous functions"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = self.check_rate_limit(endpoint, cost, priority)
                if not result.allowed:
                    raise RateLimitExceededError(f"Rate limit exceeded for {endpoint}", result)
                
                self.wait_if_needed(result)
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # Decorator for asynchronous functions
    def rate_limit_async(self, endpoint: str = "default", cost: float = None, priority: int = 1):
        """Decorator for rate limiting asynchronous functions"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                result = await self.check_rate_limit_async(endpoint, cost, priority)
                if not result.allowed:
                    raise RateLimitExceededError(f"Rate limit exceeded for {endpoint}", result)
                
                await self.wait_if_needed_async(result)
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    # Context manager for rate limiting
    @contextmanager
    def rate_limit_context(self, endpoint: str = "default", cost: float = None, priority: int = 1):
        """Context manager for rate limiting"""
        result = self.check_rate_limit(endpoint, cost, priority)
        if not result.allowed:
            raise RateLimitExceededError(f"Rate limit exceeded for {endpoint}", result)
        
        self.wait_if_needed(result)
        try:
            yield result
        finally:
            pass
    
    def get_stats(self) -> RateLimitStats:
        """Get current rate limiting statistics"""
        with self._lock:
            return RateLimitStats(
                total_requests=self.stats.total_requests,
                allowed_requests=self.stats.allowed_requests,
                blocked_requests=self.stats.blocked_requests,
                throttled_requests=self.stats.throttled_requests,
                total_wait_time=self.stats.total_wait_time,
                current_rps=self.stats.current_rps,
                peak_rps=self.stats.peak_rps,
                endpoint_stats=self.stats.endpoint_stats.copy()
            )
    
    def reset_stats(self) -> None:
        """Reset rate limiting statistics"""
        with self._lock:
            self.stats = RateLimitStats()
            self._request_timestamps.clear()
            self._peak_rps_window.clear()
            self._monitoring_data.clear()
    
    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get statistics for specific endpoint"""
        with self._lock:
            if endpoint in self.stats.endpoint_stats:
                return self.stats.endpoint_stats[endpoint].copy()
            return {}
    
    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        if self.config.requests_per_second == 0:
            return 0.0
        return min(1.0, self.stats.current_rps / self.config.requests_per_second)

class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str, rate_limit_result: RateLimitResult = None):
        super().__init__(message)
        self.rate_limit_result = rate_limit_result
        self.retry_after = rate_limit_result.retry_after if rate_limit_result else 0.0

# Example usage and testing
def main():
    """Example usage of the APIRateLimiter"""
    
    # Configure rate limiting
    config = RateLimitConfig(
        requests_per_second=5.0,  # 5 requests per second
        burst_capacity=10,        # Allow bursts up to 10 requests
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        strategy=RateLimitStrategy.DYNAMIC_BACKOFF,
        enable_dynamic_limits=True,
        enable_cost_aware=True
    )
    
    # Initialize rate limiter
    rate_limiter = APIRateLimiter(config)
    
    print("=== Testing API Rate Limiter ===\n")
    
    # Test 1: Basic rate limiting
    print("1. Basic Rate Limiting Test:")
    for i in range(15):
        result = rate_limiter.check_rate_limit("market_data", priority=1)
        status = "ALLOWED" if result.allowed else "BLOCKED"
        wait_info = f" (wait: {result.wait_time:.2f}s)" if result.wait_time > 0 else ""
        print(f"   Request {i+1:2d}: {status} - Remaining: {result.limit_remaining:2d}{wait_info}")
        time.sleep(0.1)
    
    # Test 2: Different endpoints with different costs
    print("\n2. Endpoint Cost Testing:")
    endpoints = [
        ("market_data", 1.0),
        ("order_status", 0.5),
        ("place_order", 2.0),
        ("historical_data", 3.0)
    ]
    
    for endpoint, cost in endpoints:
        result = rate_limiter.check_rate_limit(endpoint, cost)
        status = "ALLOWED" if result.allowed else "BLOCKED"
        print(f"   {endpoint:15s} (cost: {cost:3.1f}): {status} - Remaining: {result.limit_remaining}")
    
    # Test 3: Priority testing
    print("\n3. Priority Testing:")
    for priority in [1, 5, 10]:
        result = rate_limiter.check_rate_limit("market_data", priority=priority)
        wait_info = f" (wait: {result.wait_time:.2f}s)" if result.wait_time > 0 else ""
        print(f"   Priority {priority:2d}: Remaining: {result.limit_remaining:2d}{wait_info}")
    
    # Test 4: Decorator usage
    print("\n4. Decorator Test:")
    
    @rate_limiter.rate_limit("place_order", cost=2.0, priority=1)
    def place_order(symbol: str, quantity: float):
        print(f"   Placing order: {quantity} of {symbol}")
        return f"ORDER_{symbol}_{int(time.time())}"
    
    try:
        order_id = place_order("EUR/USD", 1000.0)
        print(f"   Order placed: {order_id}")
    except RateLimitExceededError as e:
        print(f"   Rate limit exceeded: {e}")
    
    # Test 5: Statistics
    print("\n5. Statistics:")
    stats = rate_limiter.get_stats()
    print(f"   Total Requests: {stats.total_requests}")
    print(f"   Allowed: {stats.allowed_requests}")
    print(f"   Blocked: {stats.blocked_requests}")
    print(f"   Throttled: {stats.throttled_requests}")
    print(f"   Current RPS: {stats.current_rps:.2f}")
    print(f"   Peak RPS: {stats.peak_rps:.2f}")
    print(f"   Utilization: {rate_limiter.get_utilization():.1%}")
    
    # Test 6: Context manager
    print("\n6. Context Manager Test:")
    try:
        with rate_limiter.rate_limit_context("account_info", cost=0.3) as result:
            print(f"   Context allowed: {result.allowed}")
            print(f"   Making API call...")
    except RateLimitExceededError as e:
        print(f"   Context blocked: {e}")
    
    print("\n=== Rate Limiter Test Completed ===")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
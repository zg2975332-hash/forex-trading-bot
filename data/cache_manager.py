"""
Cache Manager for FOREX TRADING BOT
Advanced caching system with multi-layer storage and intelligent invalidation
"""

import logging
import asyncio
import time
import pickle
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis
import diskcache
from collections import OrderedDict, defaultdict
import threading
from datetime import datetime, timedelta
import zlib
import lz4.frame
import psutil
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class CacheLayer(Enum):
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"

class CachePriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EvictionPolicy(Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    RANDOM = "random"

@dataclass
class CacheConfig:
    """Cache configuration settings"""
    # Memory cache settings
    memory_max_size: int = 1000000  # 1MB
    memory_max_items: int = 10000
    memory_ttl: int = 300  # 5 minutes
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = None
    redis_ttl: int = 3600  # 1 hour
    
    # Disk cache settings
    disk_cache_dir: str = "./cache"
    disk_max_size: int = 100000000  # 100MB
    disk_ttl: int = 86400  # 24 hours
    
    # General settings
    default_ttl: int = 600  # 10 minutes
    compression_enabled: bool = True
    compression_threshold: int = 1024  # 1KB
    statistics_enabled: bool = True
    background_cleanup: bool = True
    cleanup_interval: int = 60  # seconds

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    compression_savings: float = 0.0
    total_size: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        total = self.hits + self.misses
        return self.misses / total if total > 0 else 0.0

@dataclass
class CacheItem:
    """Cache item with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: int
    size: int
    priority: CachePriority
    layer: CacheLayer
    tags: List[str] = field(default_factory=list)
    compressed: bool = False

class CacheManager:
    """
    Advanced multi-layer cache manager with intelligent eviction and compression
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        
        # Initialize cache layers
        self.memory_cache = OrderedDict()
        self.redis_client = None
        self.disk_cache = None
        
        # Statistics
        self.stats = CacheStats()
        self.layer_stats = {
            CacheLayer.MEMORY: CacheStats(),
            CacheLayer.REDIS: CacheStats(),
            CacheLayer.DISK: CacheStats()
        }
        
        # Background tasks
        self.cleanup_task = None
        self.running = False
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize caches
        self._initialize_caches()
        
        logger.info("CacheManager initialized")

    def _initialize_caches(self):
        """Initialize all cache layers"""
        try:
            # Initialize memory cache
            self.memory_cache = OrderedDict()
            
            # Initialize Redis
            self._initialize_redis()
            
            # Initialize disk cache
            self._initialize_disk_cache()
            
            # Start background tasks
            if self.config.background_cleanup:
                self._start_background_tasks()
                
            logger.info("All cache layers initialized")
            
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            raise

    def _initialize_redis(self):
        """Initialize Redis client"""
        try:
            if hasattr(self.config, 'redis_host'):
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache layer initialized")
            else:
                logger.info("Redis not configured")
                
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None

    def _initialize_disk_cache(self):
        """Initialize disk cache"""
        try:
            self.disk_cache = diskcache.Cache(
                directory=self.config.disk_cache_dir,
                size_limit=self.config.disk_max_size,
                eviction_policy='least-recently-used'
            )
            logger.info("Disk cache layer initialized")
            
        except Exception as e:
            logger.error(f"Disk cache initialization failed: {e}")
            self.disk_cache = None

    def _start_background_tasks(self):
        """Start background cleanup tasks"""
        self.running = True
        self.cleanup_task = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_task.start()
        logger.info("Background cleanup tasks started")

    def _cleanup_worker(self):
        """Background worker for cache maintenance"""
        while self.running:
            try:
                self._cleanup_expired_items()
                self._cleanup_memory_cache()
                self._update_statistics()
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(self.config.cleanup_interval * 2)

    def _cleanup_expired_items(self):
        """Cleanup expired items from all cache layers"""
        try:
            # Clean memory cache
            with self._lock:
                expired_keys = []
                for key, item in self.memory_cache.items():
                    if self._is_expired(item):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                    self.stats.evictions += 1
                    self.layer_stats[CacheLayer.MEMORY].evictions += 1
                
                if expired_keys:
                    logger.debug(f"Cleaned {len(expired_keys)} expired items from memory cache")
            
            # Clean Redis (handled by Redis TTL)
            # Clean disk cache (handled by diskcache library)
            
        except Exception as e:
            logger.error(f"Expired items cleanup failed: {e}")

    def _cleanup_memory_cache(self):
        """Cleanup memory cache based on size limits"""
        try:
            with self._lock:
                # Check size limits
                current_size = sum(item.size for item in self.memory_cache.values())
                current_items = len(self.memory_cache)
                
                if (current_size > self.config.memory_max_size or 
                    current_items > self.config.memory_max_items):
                    
                    # Evict based on LRU policy
                    items_to_remove = max(1, current_items // 10)  # Remove 10%
                    
                    for _ in range(items_to_remove):
                        if self.memory_cache:
                            key, item = self.memory_cache.popitem(last=False)
                            self.stats.evictions += 1
                            self.layer_stats[CacheLayer.MEMORY].evictions += 1
                    
                    logger.debug(f"Evicted {items_to_remove} items from memory cache")
                    
        except Exception as e:
            logger.error(f"Memory cache cleanup failed: {e}")

    def _is_expired(self, item: CacheItem) -> bool:
        """Check if cache item is expired"""
        if item.ttl == 0:  # Never expire
            return False
        return time.time() - item.created_at > item.ttl

    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        try:
            if isinstance(data, (str, int, float, bool)):
                key_data = str(data)
            else:
                # Serialize complex objects
                key_data = pickle.dumps(data)
            
            # Create hash
            key_hash = hashlib.md5(key_data.encode() if isinstance(key_data, str) else key_data).hexdigest()
            return f"cache_{key_hash}"
            
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            # Fallback to simple hash
            return f"cache_{hash(str(data)) & 0xFFFFFFFF}"

    def _compress_data(self, data: Any) -> Tuple[bytes, bool]:
        """Compress data if beneficial"""
        try:
            if not self.config.compression_enabled:
                return data, False
            
            # Serialize data
            serialized_data = pickle.dumps(data)
            original_size = len(serialized_data)
            
            if original_size < self.config.compression_threshold:
                return serialized_data, False
            
            # Compress using LZ4
            compressed_data = lz4.frame.compress(serialized_data)
            compressed_size = len(compressed_data)
            
            # Only use compression if it saves space
            if compressed_size < original_size * 0.9:  # At least 10% savings
                compression_ratio = compressed_size / original_size
                self.stats.compression_savings += (1 - compression_ratio)
                return compressed_data, True
            else:
                return serialized_data, False
                
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return pickle.dumps(data), False

    def _decompress_data(self, data: bytes, compressed: bool) -> Any:
        """Decompress data if it was compressed"""
        try:
            if compressed:
                decompressed_data = lz4.frame.decompress(data)
                return pickle.loads(decompressed_data)
            else:
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes"""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (int, float, bool)):
                return 8  # Approximate size for primitive types
            else:
                return len(pickle.dumps(data))
                
        except Exception as e:
            logger.warning(f"Size calculation failed: {e}")
            return 1024  # Default size

    async def get(self, key: Any, layer: CacheLayer = None) -> Optional[Any]:
        """
        Get item from cache with multi-layer fallback
        """
        try:
            cache_key = self._generate_key(key)
            
            # Try specified layer first, then fallback
            layers_to_try = [layer] if layer else [CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.DISK]
            
            for current_layer in layers_to_try:
                try:
                    item = await self._get_from_layer(cache_key, current_layer)
                    if item is not None and not self._is_expired(item):
                        # Update access statistics
                        item.accessed_at = time.time()
                        item.access_count += 1
                        
                        # Promote to memory cache if frequently accessed
                        if (current_layer != CacheLayer.MEMORY and 
                            item.access_count > 5 and 
                            item.priority in [CachePriority.HIGH, CachePriority.CRITICAL]):
                            await self._set_to_layer(cache_key, item, CacheLayer.MEMORY)
                        
                        self.stats.hits += 1
                        self.layer_stats[current_layer].hits += 1
                        
                        logger.debug(f"Cache hit: {cache_key} from {current_layer}")
                        return item.value
                        
                except Exception as e:
                    logger.warning(f"Cache get failed from {current_layer}: {e}")
                    continue
            
            self.stats.misses += 1
            logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get operation failed: {e}")
            return None

    async def _get_from_layer(self, key: str, layer: CacheLayer) -> Optional[CacheItem]:
        """Get item from specific cache layer"""
        try:
            if layer == CacheLayer.MEMORY:
                with self._lock:
                    if key in self.memory_cache:
                        item = self.memory_cache[key]
                        # Move to end (most recently used)
                        self.memory_cache.move_to_end(key)
                        return item
                    return None
                    
            elif layer == CacheLayer.REDIS and self.redis_client:
                redis_data = self.redis_client.get(key)
                if redis_data:
                    item_dict = pickle.loads(redis_data)
                    return CacheItem(**item_dict)
                return None
                
            elif layer == CacheLayer.DISK and self.disk_cache:
                disk_data = self.disk_cache.get(key)
                if disk_data:
                    item_dict = pickle.loads(disk_data)
                    return CacheItem(**item_dict)
                return None
                
            return None
            
        except Exception as e:
            logger.error(f"Get from {layer} failed: {e}")
            return None

    async def set(self, key: Any, value: Any, ttl: int = None, 
                 priority: CachePriority = CachePriority.MEDIUM,
                 layers: List[CacheLayer] = None, tags: List[str] = None) -> bool:
        """
        Set item in cache with multi-layer storage
        """
        try:
            cache_key = self._generate_key(key)
            ttl = ttl or self.config.default_ttl
            
            # Determine which layers to use
            if layers is None:
                layers = self._get_default_layers(priority)
            
            # Create cache item
            compressed_value, is_compressed = self._compress_data(value)
            item_size = self._calculate_size(compressed_value)
            
            cache_item = CacheItem(
                key=cache_key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=0,
                ttl=ttl,
                size=item_size,
                priority=priority,
                layer=layers[0],  # Primary layer
                tags=tags or [],
                compressed=is_compressed
            )
            
            # Store in all specified layers
            success = True
            for layer in layers:
                try:
                    layer_success = await self._set_to_layer(cache_key, cache_item, layer)
                    success = success and layer_success
                except Exception as e:
                    logger.warning(f"Set to {layer} failed: {e}")
                    success = False
            
            if success:
                self.stats.sets += 1
                self.stats.total_size += item_size
                logger.debug(f"Cache set: {cache_key} in {len(layers)} layers")
            else:
                logger.warning(f"Partial cache set failure: {cache_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache set operation failed: {e}")
            return False

    async def _set_to_layer(self, key: str, item: CacheItem, layer: CacheLayer) -> bool:
        """Set item to specific cache layer"""
        try:
            # Create layer-specific item copy
            layer_item = CacheItem(
                key=item.key,
                value=item.value,
                created_at=item.created_at,
                accessed_at=item.accessed_at,
                access_count=item.access_count,
                ttl=item.ttl,
                size=item.size,
                priority=item.priority,
                layer=layer,
                tags=item.tags,
                compressed=item.compressed
            )
            
            if layer == CacheLayer.MEMORY:
                with self._lock:
                    self.memory_cache[key] = layer_item
                    # Move to end (most recently used)
                    self.memory_cache.move_to_end(key)
                return True
                
            elif layer == CacheLayer.REDIS and self.redis_client:
                # Compress and serialize for Redis
                compressed_data, _ = self._compress_data(layer_item)
                redis_ttl = min(item.ttl, self.config.redis_ttl) if item.ttl > 0 else None
                return self.redis_client.set(key, compressed_data, ex=redis_ttl)
                
            elif layer == CacheLayer.DISK and self.disk_cache:
                # Serialize for disk
                serialized_data = pickle.dumps(layer_item.__dict__)
                disk_ttl = min(item.ttl, self.config.disk_ttl) if item.ttl > 0 else None
                return self.disk_cache.set(key, serialized_data, expire=disk_ttl)
                
            return False
            
        except Exception as e:
            logger.error(f"Set to {layer} failed: {e}")
            return False

    def _get_default_layers(self, priority: CachePriority) -> List[CacheLayer]:
        """Get default cache layers based on priority"""
        if priority == CachePriority.CRITICAL:
            return [CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.DISK]
        elif priority == CachePriority.HIGH:
            return [CacheLayer.MEMORY, CacheLayer.REDIS]
        elif priority == CachePriority.MEDIUM:
            return [CacheLayer.REDIS, CacheLayer.DISK]
        else:  # LOW
            return [CacheLayer.DISK]

    async def delete(self, key: Any, layers: List[CacheLayer] = None) -> bool:
        """Delete item from cache"""
        try:
            cache_key = self._generate_key(key)
            
            if layers is None:
                layers = [CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.DISK]
            
            success = True
            for layer in layers:
                try:
                    layer_success = await self._delete_from_layer(cache_key, layer)
                    success = success and layer_success
                except Exception as e:
                    logger.warning(f"Delete from {layer} failed: {e}")
                    success = False
            
            if success:
                self.stats.deletes += 1
                logger.debug(f"Cache delete: {cache_key}")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete operation failed: {e}")
            return False

    async def _delete_from_layer(self, key: str, layer: CacheLayer) -> bool:
        """Delete item from specific cache layer"""
        try:
            if layer == CacheLayer.MEMORY:
                with self._lock:
                    if key in self.memory_cache:
                        del self.memory_cache[key]
                        return True
                    return False
                    
            elif layer == CacheLayer.REDIS and self.redis_client:
                return self.redis_client.delete(key) > 0
                
            elif layer == CacheLayer.DISK and self.disk_cache:
                return self.disk_cache.delete(key)
                
            return False
            
        except Exception as e:
            logger.error(f"Delete from {layer} failed: {e}")
            return False

    async def clear(self, layer: CacheLayer = None) -> bool:
        """Clear cache layer or all layers"""
        try:
            if layer:
                layers = [layer]
            else:
                layers = [CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.DISK]
            
            success = True
            for current_layer in layers:
                try:
                    layer_success = await self._clear_layer(current_layer)
                    success = success and layer_success
                except Exception as e:
                    logger.warning(f"Clear {current_layer} failed: {e}")
                    success = False
            
            logger.info(f"Cache clear completed for {len(layers)} layers")
            return success
            
        except Exception as e:
            logger.error(f"Cache clear operation failed: {e}")
            return False

    async def _clear_layer(self, layer: CacheLayer) -> bool:
        """Clear specific cache layer"""
        try:
            if layer == CacheLayer.MEMORY:
                with self._lock:
                    self.memory_cache.clear()
                return True
                
            elif layer == CacheLayer.REDIS and self.redis_client:
                return self.redis_client.flushdb()
                
            elif layer == CacheLayer.DISK and self.disk_cache:
                self.disk_cache.clear()
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Clear {layer} failed: {e}")
            return False

    async def get_or_set(self, key: Any, factory: Callable, ttl: int = None,
                        priority: CachePriority = CachePriority.MEDIUM,
                        layers: List[CacheLayer] = None, **factory_kwargs) -> Any:
        """
        Get item from cache or set it using factory function
        """
        try:
            # Try to get from cache first
            cached_value = await self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Generate value using factory
            if asyncio.iscoroutinefunction(factory):
                value = await factory(**factory_kwargs)
            else:
                value = factory(**factory_kwargs)
            
            # Store in cache
            await self.set(key, value, ttl=ttl, priority=priority, layers=layers)
            
            return value
            
        except Exception as e:
            logger.error(f"Get or set operation failed: {e}")
            # Try factory directly as fallback
            try:
                if asyncio.iscoroutinefunction(factory):
                    return await factory(**factory_kwargs)
                else:
                    return factory(**factory_kwargs)
            except Exception as factory_error:
                logger.error(f"Factory function also failed: {factory_error}")
                raise

    async def exists(self, key: Any, layer: CacheLayer = None) -> bool:
        """Check if key exists in cache"""
        try:
            cache_key = self._generate_key(key)
            
            if layer:
                layers = [layer]
            else:
                layers = [CacheLayer.MEMORY, CacheLayer.REDIS, CacheLayer.DISK]
            
            for current_layer in layers:
                try:
                    if await self._exists_in_layer(cache_key, current_layer):
                        return True
                except Exception as e:
                    logger.warning(f"Exists check in {current_layer} failed: {e}")
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Exists operation failed: {e}")
            return False

    async def _exists_in_layer(self, key: str, layer: CacheLayer) -> bool:
        """Check if key exists in specific layer"""
        try:
            if layer == CacheLayer.MEMORY:
                with self._lock:
                    return key in self.memory_cache
                    
            elif layer == CacheLayer.REDIS and self.redis_client:
                return self.redis_client.exists(key) > 0
                
            elif layer == CacheLayer.DISK and self.disk_cache:
                return key in self.disk_cache
                
            return False
            
        except Exception as e:
            logger.error(f"Exists in {layer} failed: {e}")
            return False

    async def get_stats(self, detailed: bool = False) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            base_stats = {
                'hit_rate': self.stats.hit_rate,
                'miss_rate': self.stats.miss_rate,
                'total_operations': self.stats.hits + self.stats.misses + self.stats.sets,
                'total_size_bytes': self.stats.total_size,
                'compression_savings': self.stats.compression_savings,
                'eviction_count': self.stats.evictions,
                'memory_usage': self._get_memory_usage()
            }
            
            if detailed:
                base_stats['layer_stats'] = {
                    layer.value: {
                        'hits': layer_stat.hits,
                        'misses': layer_stat.misses,
                        'hit_rate': layer_stat.hit_rate,
                        'evictions': layer_stat.evictions
                    }
                    for layer, layer_stat in self.layer_stats.items()
                }
                
                base_stats['memory_cache'] = {
                    'items': len(self.memory_cache),
                    'total_size': sum(item.size for item in self.memory_cache.values())
                }
            
            return base_stats
            
        except Exception as e:
            logger.error(f"Stats collection failed: {e}")
            return {}

    def _get_memory_usage(self) -> int:
        """Get current memory usage"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception as e:
            logger.warning(f"Memory usage check failed: {e}")
            return 0

    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all items with specified tags"""
        try:
            deleted_count = 0
            
            # This would require maintaining a tag index
            # For now, implement simple version that scans memory cache
            with self._lock:
                keys_to_delete = []
                for key, item in self.memory_cache.items():
                    if any(tag in item.tags for tag in tags):
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} items by tags: {tags}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Delete by tags failed: {e}")
            return 0

    async def prefetch(self, keys: List[Any], factory: Callable, 
                      ttl: int = None, priority: CachePriority = CachePriority.MEDIUM) -> Dict[Any, Any]:
        """Prefetch multiple keys and return results"""
        try:
            results = {}
            missing_keys = []
            
            # Try to get all keys from cache first
            for key in keys:
                cached_value = await self.get(key)
                if cached_value is not None:
                    results[key] = cached_value
                else:
                    missing_keys.append(key)
            
            # Generate missing values
            if missing_keys:
                if asyncio.iscoroutinefunction(factory):
                    missing_values = await factory(missing_keys)
                else:
                    missing_values = factory(missing_keys)
                
                # Store missing values in cache
                for key, value in zip(missing_keys, missing_values):
                    await self.set(key, value, ttl=ttl, priority=priority)
                    results[key] = value
            
            return results
            
        except Exception as e:
            logger.error(f"Prefetch operation failed: {e}")
            # Fallback to individual gets
            results = {}
            for key in keys:
                try:
                    results[key] = await self.get_or_set(key, lambda: None, ttl=ttl, priority=priority)
                except Exception:
                    results[key] = None
            return results

    @asynccontextmanager
    async def transaction(self):
        """Context manager for cache transactions"""
        # For memory cache, use the existing lock
        # For Redis, would use Redis transactions
        with self._lock:
            try:
                yield self
            except Exception as e:
                logger.error(f"Cache transaction failed: {e}")
                raise

    async def close(self):
        """Cleanup resources"""
        try:
            self.running = False
            
            if self.cleanup_task:
                self.cleanup_task.join(timeout=5)
            
            if self.redis_client:
                self.redis_client.close()
            
            if self.disk_cache:
                self.disk_cache.close()
            
            logger.info("CacheManager closed")
            
        except Exception as e:
            logger.error(f"CacheManager close failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self.close())

# Example usage and testing
async def main():
    """Test the Cache Manager"""
    
    config = CacheConfig(
        memory_max_size=5000000,  # 5MB
        memory_max_items=5000,
        redis_host="localhost",
        redis_port=6379,
        disk_cache_dir="./test_cache"
    )
    
    cache_manager = CacheManager(config)
    
    try:
        # Test basic set/get
        await cache_manager.set("test_key", "test_value", ttl=60)
        value = await cache_manager.get("test_key")
        print(f"Retrieved value: {value}")
        
        # Test get_or_set with async function
        async def expensive_operation():
            await asyncio.sleep(0.1)
            return {"data": "expensive_result", "timestamp": time.time()}
        
        result = await cache_manager.get_or_set(
            "expensive_key", 
            expensive_operation,
            ttl=300,
            priority=CachePriority.HIGH
        )
        print(f"Expensive operation result: {result}")
        
        # Test statistics
        stats = await cache_manager.get_stats(detailed=True)
        print(f"Cache stats: {json.dumps(stats, indent=2, default=str)}")
        
        # Test prefetch
        keys = ["key1", "key2", "key3"]
        def batch_factory(keys):
            return [f"value_for_{key}" for key in keys]
        
        results = await cache_manager.prefetch(keys, batch_factory)
        print(f"Prefetch results: {results}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        await cache_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
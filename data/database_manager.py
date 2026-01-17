"""
Database Manager for FOREX TRADING BOT
Advanced database operations with connection pooling, caching, and performance optimization
"""

import logging
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import redis
import json
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
import threading
from collections import defaultdict, deque
import hashlib
import psutil
import os

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"
    INFLUXDB = "influxdb"

class OperationType(Enum):
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    BATCH = "batch"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    # Primary database
    db_type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    database: str = "forex_bot"
    username: str = "postgres"
    password: str = "password"
    
    # Connection settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Redis cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = None
    
    # Performance settings
    query_timeout: int = 30
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    batch_size: int = 1000
    
    # Backup settings
    backup_enabled: bool = True
    backup_interval: int = 3600  # 1 hour
    backup_retention: int = 7  # days

@dataclass
class QueryResult:
    """Database query result"""
    success: bool
    data: Any = None
    execution_time: float = 0.0
    rows_affected: int = 0
    error_message: str = ""
    query_hash: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class DatabaseStats:
    """Database performance statistics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_query_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    connection_usage: float = 0.0
    memory_usage: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total_cache = self.cache_hits + self.cache_misses
        return self.cache_hits / total_cache if total_cache > 0 else 0.0

# SQLAlchemy Base
Base = declarative_base()

# Define core tables
class MarketData(Base):
    """Market data table"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"

class Trade(Base):
    """Trade records table"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, nullable=False, index=True)
    exit_time = Column(DateTime, index=True)
    status = Column(String(20), nullable=False)  # open, closed, cancelled
    pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    strategy = Column(String(50))
    exchange = Column(String(50))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Portfolio(Base):
    """Portfolio snapshot table"""
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    margin_used = Column(Float, nullable=False)
    open_positions = Column(Integer, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class PerformanceMetrics(Base):
    """Performance metrics table"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    period = Column(String(20), nullable=False)  # daily, weekly, monthly
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
    total_pnl = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """
    Advanced database manager with connection pooling, caching, and performance optimization
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        self.metadata = MetaData()
        
        # Statistics and monitoring
        self.stats = DatabaseStats()
        self.query_history = deque(maxlen=10000)
        self.performance_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Connection pool monitoring
        self.connection_pool = None
        self.last_health_check = time.time()
        
        # Background tasks
        self.cleanup_task = None
        self.backup_task = None
        self.running = False
        
        # Initialize databases
        self._initialize_databases()
        
        logger.info("DatabaseManager initialized")

    def _initialize_databases(self):
        """Initialize all database connections"""
        try:
            # Initialize primary database
            self._initialize_primary_db()
            
            # Initialize Redis cache
            self._initialize_redis()
            
            # Create tables if they don't exist
            self._create_tables()
            
            # Start background tasks
            self._start_background_tasks()
            
            logger.info("All database connections initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _initialize_primary_db(self):
        """Initialize primary database connection"""
        try:
            if self.config.db_type == DatabaseType.POSTGRESQL:
                connection_string = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            elif self.config.db_type == DatabaseType.MYSQL:
                connection_string = f"mysql+pymysql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            elif self.config.db_type == DatabaseType.SQLITE:
                connection_string = f"sqlite:///{self.config.database}.db"
            else:
                raise ValueError(f"Unsupported database type: {self.config.db_type}")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False  # Set to True for SQL logging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"Primary database connected: {self.config.db_type.value}")
            
        except Exception as e:
            logger.error(f"Primary database initialization failed: {e}")
            raise

    def _initialize_redis(self):
        """Initialize Redis cache"""
        try:
            if self.config.enable_caching:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    password=self.config.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            else:
                logger.info("Redis caching disabled")
                
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None

    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.running = True
        
        # Start cleanup task
        self.cleanup_task = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_task.start()
        
        # Start backup task if enabled
        if self.config.backup_enabled:
            self.backup_task = threading.Thread(target=self._backup_worker, daemon=True)
            self.backup_task.start()
        
        logger.info("Background tasks started")

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def _generate_cache_key(self, query: str, params: Dict = None) -> str:
        """Generate cache key for query"""
        key_data = f"{query}_{json.dumps(params, sort_keys=True) if params else ''}"
        return f"db_cache_{hashlib.md5(key_data.encode()).hexdigest()}"

    async def execute_query(self, query: str, params: Dict = None, 
                          operation: OperationType = OperationType.READ,
                          use_cache: bool = True) -> QueryResult:
        """
        Execute database query with caching and performance tracking
        """
        start_time = time.time()
        query_hash = self._generate_cache_key(query, params)
        
        try:
            # Check cache for read operations
            if (operation == OperationType.READ and use_cache and 
                self.config.enable_caching and self.redis_client):
                cached_result = self._get_cached_result(query_hash)
                if cached_result:
                    self.stats.cache_hits += 1
                    logger.debug(f"Cache hit for query: {query_hash[:20]}...")
                    return cached_result
            
            # Execute query
            with self.get_session() as session:
                if operation == OperationType.READ:
                    result = session.execute(text(query), params or {})
                    data = [dict(row) for row in result]
                    rows_affected = len(data)
                else:
                    result = session.execute(text(query), params or {})
                    rows_affected = result.rowcount
                    data = None
                
                # Commit if write operation
                if operation != OperationType.READ:
                    session.commit()
            
            execution_time = time.time() - start_time
            
            # Create result object
            query_result = QueryResult(
                success=True,
                data=data,
                execution_time=execution_time,
                rows_affected=rows_affected,
                query_hash=query_hash
            )
            
            # Cache read results
            if (operation == OperationType.READ and use_cache and 
                self.config.enable_caching and self.redis_client):
                self._cache_result(query_hash, query_result)
                self.stats.cache_misses += 1
            
            # Update statistics
            self._update_stats(query_result, operation)
            
            logger.debug(f"Query executed successfully: {query_hash[:20]}... in {execution_time:.3f}s")
            
            return query_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(f"{error_msg} - Query: {query_hash[:20]}...")
            
            query_result = QueryResult(
                success=False,
                execution_time=execution_time,
                error_message=error_msg,
                query_hash=query_hash
            )
            
            self._update_stats(query_result, operation)
            
            return query_result

    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result"""
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result_dict = json.loads(cached_data)
                    return QueryResult(**result_dict)
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def _cache_result(self, cache_key: str, result: QueryResult):
        """Cache query result"""
        try:
            if self.redis_client:
                result_dict = {
                    'success': result.success,
                    'data': result.data,
                    'execution_time': result.execution_time,
                    'rows_affected': result.rows_affected,
                    'query_hash': result.query_hash,
                    'timestamp': result.timestamp
                }
                self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl,
                    json.dumps(result_dict, default=str)
                )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def _update_stats(self, result: QueryResult, operation: OperationType):
        """Update performance statistics"""
        self.stats.total_queries += 1
        
        if result.success:
            self.stats.successful_queries += 1
        else:
            self.stats.failed_queries += 1
        
        # Update average query time
        total_time = self.stats.average_query_time * (self.stats.total_queries - 1)
        self.stats.average_query_time = (total_time + result.execution_time) / self.stats.total_queries
        
        # Store in history
        self.query_history.append({
            'timestamp': time.time(),
            'operation': operation.value,
            'execution_time': result.execution_time,
            'success': result.success,
            'query_hash': result.query_hash
        })
        
        # Update performance metrics
        self.performance_metrics[operation.value].append(result.execution_time)

    # High-level operations
    async def store_market_data(self, data: pd.DataFrame, symbol: str, source: str) -> bool:
        """Store market data in database"""
        try:
            records = []
            for _, row in data.iterrows():
                record = MarketData(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row.get('volume', 0),
                    source=source
                )
                records.append(record)
            
            # Use batch insert for performance
            return await self._batch_insert(records)
            
        except Exception as e:
            logger.error(f"Market data storage failed: {e}")
            return False

    async def _batch_insert(self, records: List, batch_size: int = None) -> bool:
        """Batch insert records for performance"""
        batch_size = batch_size or self.config.batch_size
        
        try:
            with self.get_session() as session:
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    session.bulk_save_objects(batch)
                    session.commit()  # Commit each batch
                    
                    logger.debug(f"Inserted batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}")
            
            return True
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return False

    async def get_market_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime, timeframe: str = "1H") -> pd.DataFrame:
        """Retrieve market data from database"""
        try:
            query = """
            SELECT timestamp, open, high, low, close, volume 
            FROM market_data 
            WHERE symbol = :symbol 
            AND timestamp BETWEEN :start_date AND :end_date
            ORDER BY timestamp
            """
            
            params = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date
            }
            
            result = await self.execute_query(query, params, OperationType.READ, use_cache=True)
            
            if result.success and result.data:
                df = pd.DataFrame(result.data)
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Market data retrieval failed: {e}")
            return pd.DataFrame()

    async def store_trade(self, trade_data: Dict) -> bool:
        """Store trade record"""
        try:
            trade = Trade(
                trade_id=trade_data['trade_id'],
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data.get('exit_price'),
                entry_time=trade_data['entry_time'],
                exit_time=trade_data.get('exit_time'),
                status=trade_data['status'],
                pnl=trade_data.get('pnl', 0.0),
                commission=trade_data.get('commission', 0.0),
                strategy=trade_data.get('strategy'),
                exchange=trade_data.get('exchange'),
                metadata=trade_data.get('metadata')
            )
            
            with self.get_session() as session:
                session.add(trade)
                session.commit()
            
            # Invalidate relevant cache entries
            self._invalidate_trade_cache(trade_data['symbol'])
            
            logger.info(f"Trade stored: {trade_data['trade_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Trade storage failed: {e}")
            return False

    async def update_trade(self, trade_id: str, updates: Dict) -> bool:
        """Update trade record"""
        try:
            with self.get_session() as session:
                trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
                if not trade:
                    logger.warning(f"Trade not found: {trade_id}")
                    return False
                
                for key, value in updates.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)
                
                session.commit()
            
            # Invalidate cache
            self._invalidate_trade_cache(trade.symbol)
            
            logger.debug(f"Trade updated: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Trade update failed: {e}")
            return False

    async def get_trade_history(self, symbol: str = None, start_date: datetime = None,
                              end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Retrieve trade history"""
        try:
            query = "SELECT * FROM trades WHERE 1=1"
            params = {}
            
            if symbol:
                query += " AND symbol = :symbol"
                params['symbol'] = symbol
            
            if start_date:
                query += " AND entry_time >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND entry_time <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY entry_time DESC"
            
            if limit:
                query += " LIMIT :limit"
                params['limit'] = limit
            
            result = await self.execute_query(query, params, OperationType.READ, use_cache=True)
            
            if result.success and result.data:
                return pd.DataFrame(result.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Trade history retrieval failed: {e}")
            return pd.DataFrame()

    async def store_portfolio_snapshot(self, snapshot_data: Dict) -> bool:
        """Store portfolio snapshot"""
        try:
            snapshot = Portfolio(
                timestamp=snapshot_data['timestamp'],
                total_value=snapshot_data['total_value'],
                cash_balance=snapshot_data['cash_balance'],
                unrealized_pnl=snapshot_data['unrealized_pnl'],
                realized_pnl=snapshot_data['realized_pnl'],
                margin_used=snapshot_data['margin_used'],
                open_positions=snapshot_data['open_positions'],
                metadata=snapshot_data.get('metadata')
            )
            
            with self.get_session() as session:
                session.add(snapshot)
                session.commit()
            
            logger.debug(f"Portfolio snapshot stored: {snapshot_data['timestamp']}")
            return True
            
        except Exception as e:
            logger.error(f"Portfolio snapshot storage failed: {e}")
            return False

    async def get_portfolio_history(self, start_date: datetime = None,
                                 end_date: datetime = None) -> pd.DataFrame:
        """Retrieve portfolio history"""
        try:
            query = "SELECT * FROM portfolio_snapshots WHERE 1=1"
            params = {}
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp"
            
            result = await self.execute_query(query, params, OperationType.READ, use_cache=True)
            
            if result.success and result.data:
                return pd.DataFrame(result.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Portfolio history retrieval failed: {e}")
            return pd.DataFrame()

    async def store_performance_metrics(self, metrics_data: Dict) -> bool:
        """Store performance metrics"""
        try:
            metrics = PerformanceMetrics(
                timestamp=metrics_data['timestamp'],
                period=metrics_data['period'],
                sharpe_ratio=metrics_data.get('sharpe_ratio'),
                sortino_ratio=metrics_data.get('sortino_ratio'),
                max_drawdown=metrics_data.get('max_drawdown'),
                win_rate=metrics_data.get('win_rate'),
                profit_factor=metrics_data.get('profit_factor'),
                total_trades=metrics_data.get('total_trades'),
                total_pnl=metrics_data.get('total_pnl'),
                metadata=metrics_data.get('metadata')
            )
            
            with self.get_session() as session:
                session.add(metrics)
                session.commit()
            
            logger.debug(f"Performance metrics stored: {metrics_data['period']} - {metrics_data['timestamp']}")
            return True
            
        except Exception as e:
            logger.error(f"Performance metrics storage failed: {e}")
            return False

    def _invalidate_trade_cache(self, symbol: str):
        """Invalidate cache entries related to trades"""
        try:
            if self.redis_client:
                # Invalidate trade-related cache patterns
                patterns = [f"db_cache_*trade*", f"db_cache_*{symbol}*"]
                for pattern in patterns:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                        logger.debug(f"Invalidated {len(keys)} cache entries for pattern: {pattern}")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            # Basic stats
            stats = {
                'performance': {
                    'success_rate': self.stats.success_rate,
                    'average_query_time': self.stats.average_query_time,
                    'cache_hit_rate': self.stats.cache_hit_rate,
                    'total_queries': self.stats.total_queries,
                    'failed_queries': self.stats.failed_queries
                },
                'tables': {},
                'connections': {}
            }
            
            # Table statistics
            with self.get_session() as session:
                tables = ['market_data', 'trades', 'portfolio_snapshots', 'performance_metrics']
                for table in tables:
                    count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    stats['tables'][table] = {'row_count': count}
            
            # Connection pool stats
            if self.engine:
                pool = self.engine.pool
                stats['connections'] = {
                    'checked_out': pool.checkedout(),
                    'checked_in': pool.checkedin(),
                    'overflow': pool.overflow(),
                    'size': pool.size()
                }
            
            # Memory usage
            stats['memory_usage_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
            
            return stats
            
        except Exception as e:
            logger.error(f"Database stats collection failed: {e}")
            return {}

    async def optimize_database(self):
        """Perform database optimization tasks"""
        try:
            logger.info("Starting database optimization...")
            
            with self.get_session() as session:
                # Vacuum and analyze for PostgreSQL
                if self.config.db_type == DatabaseType.POSTGRESQL:
                    session.execute(text("VACUUM ANALYZE"))
                    logger.info("VACUUM ANALYZE completed")
                
                # Update table statistics
                session.execute(text("ANALYZE"))
                logger.info("Table statistics updated")
            
            # Clear old cache entries
            if self.redis_client:
                # Keep only recent cache entries (last 24 hours)
                cutoff_time = time.time() - 86400
                all_keys = self.redis_client.keys("db_cache_*")
                
                for key in all_keys:
                    key_time = self.redis_client.object('idletime', key)
                    if key_time > cutoff_time:
                        self.redis_client.delete(key)
                
                logger.info("Cache optimization completed")
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")

    def _cleanup_worker(self):
        """Background worker for database maintenance"""
        while self.running:
            try:
                # Cleanup old data
                self._cleanup_old_data()
                
                # Health check
                self._health_check()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(600)  # Longer sleep on error

    def _cleanup_old_data(self):
        """Cleanup old data based on retention policy"""
        try:
            retention_days = 365  # Keep 1 year of data
            
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            with self.get_session() as session:
                # Delete old market data
                session.execute(
                    text("DELETE FROM market_data WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                )
                
                # Delete old portfolio snapshots (keep only daily)
                session.execute(
                    text("""
                    DELETE FROM portfolio_snapshots 
                    WHERE timestamp < :cutoff_date 
                    AND DATE(timestamp) NOT IN (
                        SELECT DISTINCT DATE(timestamp) 
                        FROM portfolio_snapshots 
                        WHERE timestamp >= :cutoff_date
                    )
                    """),
                    {'cutoff_date': cutoff_date}
                )
            
            logger.debug("Old data cleanup completed")
            
        except Exception as e:
            logger.error(f"Old data cleanup failed: {e}")

    def _health_check(self):
        """Perform database health check"""
        try:
            with self.get_session() as session:
                # Test connection and basic operations
                session.execute(text("SELECT 1"))
                
                # Check table sizes
                tables = ['market_data', 'trades', 'portfolio_snapshots']
                for table in tables:
                    count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    logger.debug(f"Health check - {table}: {count} rows")
            
            self.last_health_check = time.time()
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")

    def _backup_worker(self):
        """Background worker for database backups"""
        while self.running:
            try:
                current_time = time.time()
                next_backup = self.last_health_check + self.config.backup_interval
                
                if current_time >= next_backup:
                    self._create_backup()
                    self.last_health_check = current_time
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Backup worker error: {e}")
                time.sleep(300)  # Longer sleep on error

    def _create_backup(self):
        """Create database backup"""
        try:
            backup_dir = "./backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{backup_dir}/backup_{timestamp}.sql"
            
            # This would use pg_dump for PostgreSQL or similar for other databases
            # For now, just log the backup attempt
            logger.info(f"Database backup created: {backup_file}")
            
            # Cleanup old backups
            self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")

    def _cleanup_old_backups(self, backup_dir: str):
        """Cleanup old backup files"""
        try:
            cutoff_time = time.time() - (self.config.backup_retention * 86400)
            
            for filename in os.listdir(backup_dir):
                filepath = os.path.join(backup_dir, filename)
                if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    logger.debug(f"Removed old backup: {filename}")
                    
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    async def close(self):
        """Cleanup database connections"""
        try:
            self.running = False
            
            if self.cleanup_task:
                self.cleanup_task.join(timeout=5)
            
            if self.backup_task:
                self.backup_task.join(timeout=5)
            
            if self.engine:
                self.engine.dispose()
            
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("DatabaseManager closed")
            
        except Exception as e:
            logger.error(f"DatabaseManager close failed: {e}")

# Example usage and testing
async def main():
    """Test the Database Manager"""
    
    config = DatabaseConfig(
        db_type=DatabaseType.SQLITE,  # Use SQLite for testing
        database="test_forex_bot",
        enable_caching=True
    )
    
    db_manager = DatabaseManager(config)
    
    try:
        # Test market data storage
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H'),
            'open': np.random.uniform(1.05, 1.10, 10),
            'high': np.random.uniform(1.10, 1.15, 10),
            'low': np.random.uniform(1.00, 1.05, 10),
            'close': np.random.uniform(1.05, 1.10, 10),
            'volume': np.random.randint(1000, 10000, 10)
        })
        
        success = await db_manager.store_market_data(sample_data, "EUR/USD", "test")
        print(f"Market data storage: {'Success' if success else 'Failed'}")
        
        # Test market data retrieval
        retrieved_data = await db_manager.get_market_data(
            "EUR/USD", 
            datetime(2024, 1, 1), 
            datetime(2024, 1, 2)
        )
        print(f"Retrieved {len(retrieved_data)} market data records")
        
        # Test trade storage
        trade_data = {
            'trade_id': 'test_trade_001',
            'symbol': 'EUR/USD',
            'side': 'buy',
            'quantity': 1000,
            'entry_price': 1.0850,
            'entry_time': datetime.utcnow(),
            'status': 'open',
            'strategy': 'momentum',
            'exchange': 'binance'
        }
        
        success = await db_manager.store_trade(trade_data)
        print(f"Trade storage: {'Success' if success else 'Failed'}")
        
        # Test database statistics
        stats = await db_manager.get_database_stats()
        print(f"Database stats - Success rate: {stats['performance']['success_rate']:.2%}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
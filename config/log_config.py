"""
Advanced Logging Configuration for Forex Trading Bot
Comprehensive logging setup with multiple handlers, formats, and monitoring
"""

import logging
import logging.config
import logging.handlers
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import colorama
from colorama import Fore, Style, Back
import threading

# Initialize colorama for colored console output
colorama.init(autoreset=True)

class CustomFormatter(logging.Formatter):
    """
    Custom log formatter with colors and detailed formatting
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    # Format strings
    CONSOLE_FORMAT = '{color}{levelname:8}{reset} │ {name:20} │ {message}'
    FILE_FORMAT = '{asctime} │ {levelname:8} │ {name:20} │ {message}'
    DETAILED_FORMAT = '{asctime} │ {levelname:8} │ {name:20} │ {funcName}:{lineno} │ {message}'
    
    def __init__(self, use_colors=True, detailed=False):
        super().__init__()
        self.use_colors = use_colors
        self.detailed = detailed
        
        if detailed:
            self._format = self.DETAILED_FORMAT
        else:
            self._format = self.FILE_FORMAT
    
    def format(self, record):
        """Format log record with colors and custom styling"""
        # Create custom format string
        if self.use_colors and hasattr(record, 'color'):
            format_str = self.CONSOLE_FORMAT.format(
                color=record.color,
                levelname=record.levelname,
                reset=Style.RESET_ALL,
                name=record.name,
                message=record.getMessage()
            )
        else:
            if self.detailed:
                format_str = self.DETAILED_FORMAT
            else:
                format_str = self.FILE_FORMAT
            
            # Use default formatter for file output
            formatter = logging.Formatter(format_str, style='{')
            return formatter.format(record)
        
        return format_str

class TradeFilter(logging.Filter):
    """Custom filter for trade-related log messages"""
    
    def filter(self, record):
        record.is_trade = hasattr(record, 'trade_id') and record.trade_id
        record.is_signal = 'signal' in record.getMessage().lower()
        record.is_execution = 'execut' in record.getMessage().lower()
        return True

class PerformanceFilter(logging.Filter):
    """Custom filter for performance-related log messages"""
    
    def filter(self, record):
        record.is_performance = any(keyword in record.getMessage().lower() 
                                  for keyword in ['pnl', 'profit', 'loss', 'drawdown', 'sharpe', 'win rate'])
        return True

class ErrorTrackingFilter(logging.Filter):
    """Custom filter for error tracking"""
    
    def filter(self, record):
        record.is_error = record.levelno >= logging.ERROR
        record.is_critical = record.levelno >= logging.CRITICAL
        return True

class TradingFileHandler(logging.handlers.RotatingFileHandler):
    """Custom file handler for trading-specific logs"""
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)

class TradeLogger:
    """
    Specialized logger for trade operations with structured logging
    """
    
    def __init__(self, main_logger):
        self.logger = main_logger.getChild('trades')
        self.trade_logger = logging.getLogger('trading_bot.trades')
    
    def log_trade_entry(self, symbol: str, direction: str, entry_price: float, 
                       quantity: float, trade_id: str, confidence: float = 0.0):
        """Log trade entry with structured data"""
        extra = {
            'trade_id': trade_id,
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'quantity': quantity,
            'confidence': confidence,
            'action': 'entry'
        }
        
        self.trade_logger.info(
            f"TRADE ENTRY │ {symbol} │ {direction} │ Price: {entry_price:.5f} │ "
            f"Qty: {quantity:.2f} │ Confidence: {confidence:.2f} │ ID: {trade_id}",
            extra=extra
        )
    
    def log_trade_exit(self, trade_id: str, exit_price: float, pnl: float, 
                      pnl_pct: float, duration: str, reason: str):
        """Log trade exit with P&L information"""
        extra = {
            'trade_id': trade_id,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'duration': duration,
            'reason': reason,
            'action': 'exit'
        }
        
        # Color code based on P&L
        pnl_color = Fore.GREEN if pnl > 0 else Fore.RED
        pnl_text = f"{pnl:+.2f} ({pnl_pct:+.2f}%)"
        
        self.trade_logger.info(
            f"TRADE EXIT  │ ID: {trade_id} │ Price: {exit_price:.5f} │ "
            f"P&L: {pnl_color}{pnl_text}{Style.RESET_ALL} │ Duration: {duration} │ Reason: {reason}",
            extra=extra
        )
    
    def log_trade_signal(self, symbol: str, signal: str, strength: float, 
                        indicators: Dict[str, Any], model_confidence: float):
        """Log trading signal with detailed information"""
        extra = {
            'symbol': symbol,
            'signal': signal,
            'strength': strength,
            'model_confidence': model_confidence,
            'action': 'signal'
        }
        
        indicators_str = ' │ '.join([f"{k}: {v:.3f}" for k, v in indicators.items()])
        
        self.trade_logger.info(
            f"SIGNAL     │ {symbol} │ {signal} │ Strength: {strength:.2f} │ "
            f"Model: {model_confidence:.2f} │ {indicators_str}",
            extra=extra
        )

class PerformanceLogger:
    """
    Specialized logger for performance metrics and analytics
    """
    
    def __init__(self, main_logger):
        self.logger = main_logger.getChild('performance')
        self.performance_logger = logging.getLogger('trading_bot.performance')
    
    def log_daily_performance(self, date: str, pnl: float, pnl_pct: float, 
                            trades: int, win_rate: float, drawdown: float):
        """Log daily performance summary"""
        extra = {
            'date': date,
            'daily_pnl': pnl,
            'daily_pnl_pct': pnl_pct,
            'trades_count': trades,
            'win_rate': win_rate,
            'drawdown': drawdown
        }
        
        self.performance_logger.info(
            f"DAILY PERFORMANCE │ Date: {date} │ P&L: {pnl:+.2f} ({pnl_pct:+.2f}%) │ "
            f"Trades: {trades} │ Win Rate: {win_rate:.1f}% │ Drawdown: {drawdown:.2f}%",
            extra=extra
        )
    
    def log_portfolio_update(self, total_value: float, cash: float, 
                           unrealized_pnl: float, realized_pnl: float):
        """Log portfolio value update"""
        extra = {
            'total_value': total_value,
            'cash': cash,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl
        }
        
        self.performance_logger.info(
            f"PORTFOLIO UPDATE │ Total: ${total_value:,.2f} │ Cash: ${cash:,.2f} │ "
            f"Unrealized: ${unrealized_pnl:+.2f} │ Realized: ${realized_pnl:+.2f}",
            extra=extra
        )
    
    def log_risk_metrics(self, sharpe_ratio: float, sortino_ratio: float, 
                        max_drawdown: float, volatility: float, var_95: float):
        """Log risk management metrics"""
        extra = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'var_95': var_95
        }
        
        self.performance_logger.info(
            f"RISK METRICS │ Sharpe: {sharpe_ratio:.2f} │ Sortino: {sortino_ratio:.2f} │ "
            f"Max DD: {max_drawdown:.2f}% │ Vol: {volatility:.2f}% │ VaR 95%: {var_95:.2f}%",
            extra=extra
        )

class LogManager:
    """
    Comprehensive log management system for Forex Trading Bot
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.setup_complete = False
        self.trade_logger = None
        self.performance_logger = None
        
        # Setup logging
        self.setup_logging()
        
        # Initialize specialized loggers
        self.main_logger = logging.getLogger('trading_bot')
        self.trade_logger = TradeLogger(self.main_logger)
        self.performance_logger = PerformanceLogger(self.main_logger)
        
        logging.info("Logging system initialized successfully")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '{asctime} │ {levelname:8} │ {name:20} │ {message}',
                    'style': '{'
                },
                'detailed': {
                    'format': '{asctime} │ {levelname:8} │ {name:20} │ {funcName}:{lineno} │ {message}',
                    'style': '{'
                },
                'colored': {
                    '()': CustomFormatter,
                    'use_colors': True,
                    'detailed': False
                }
            },
            'filters': {
                'trade_filter': {
                    '()': TradeFilter
                },
                'performance_filter': {
                    '()': PerformanceFilter
                },
                'error_filter': {
                    '()': ErrorTrackingFilter
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'colored',
                    'stream': 'ext://sys.stdout'
                },
                'file_general': {
                    '()': TradingFileHandler,
                    'level': 'INFO',
                    'formatter': 'standard',
                    'filename': 'logs/trading_bot.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf8'
                },
                'file_trades': {
                    '()': TradingFileHandler,
                    'level': 'INFO',
                    'formatter': 'standard',
                    'filename': 'logs/trades.log',
                    'maxBytes': 10485760,
                    'backupCount': 10,
                    'encoding': 'utf8',
                    'filters': ['trade_filter']
                },
                'file_errors': {
                    '()': TradingFileHandler,
                    'level': 'WARNING',
                    'formatter': 'detailed',
                    'filename': 'logs/errors.log',
                    'maxBytes': 5242880,  # 5MB
                    'backupCount': 3,
                    'encoding': 'utf8',
                    'filters': ['error_filter']
                },
                'file_performance': {
                    '()': TradingFileHandler,
                    'level': 'INFO',
                    'formatter': 'standard',
                    'filename': 'logs/performance.log',
                    'maxBytes': 5242880,
                    'backupCount': 5,
                    'encoding': 'utf8',
                    'filters': ['performance_filter']
                },
                'timed_rotating': {
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'filename': 'logs/daily/daily.log',
                    'when': 'midnight',
                    'interval': 1,
                    'backupCount': 7,
                    'encoding': 'utf8'
                }
            },
            'loggers': {
                'trading_bot': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_general', 'timed_rotating'],
                    'propagate': False
                },
                'trading_bot.trades': {
                    'level': 'INFO',
                    'handlers': ['file_trades'],
                    'propagate': False
                },
                'trading_bot.performance': {
                    'level': 'INFO',
                    'handlers': ['file_performance'],
                    'propagate': False
                },
                'trading_bot.errors': {
                    'level': 'WARNING',
                    'handlers': ['file_errors'],
                    'propagate': False
                },
                'core': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_general'],
                    'propagate': False
                },
                'models': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_general'],
                    'propagate': False
                },
                'strategies': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_general'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console']
            }
        }
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        try:
            # Create logs directory structure
            self._create_log_directories()
            
            # Apply logging configuration
            logging.config.dictConfig(self.config)
            
            # Add custom filters and handlers
            self._setup_custom_logging()
            
            self.setup_complete = True
            print(f"{Fore.GREEN}✅ Logging system configured successfully{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error setting up logging: {e}{Style.RESET_ALL}")
            # Fallback to basic configuration
            logging.basicConfig(level=logging.INFO)
    
    def _create_log_directories(self):
        """Create necessary log directories"""
        directories = [
            'logs',
            'logs/daily',
            'logs/backtest',
            'logs/performance'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_custom_logging(self):
        """Setup custom logging features"""
        # Add thread safety
        self._make_logging_thread_safe()
        
        # Setup exception hook for unhandled exceptions
        self._setup_exception_hook()
    
    def _make_logging_thread_safe(self):
        """Make logging thread-safe"""
        logging._acquireLock()
        try:
            # This is handled by logging.config, but we ensure thread safety
            pass
        finally:
            logging._releaseLock()
    
    def _setup_exception_hook(self):
        """Setup global exception handler"""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupts
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            logger = logging.getLogger('trading_bot.errors')
            logger.critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        
        sys.excepthook = handle_exception
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger with the specified name"""
        return logging.getLogger(name)
    
    def set_log_level(self, level: str, logger_name: str = None):
        """Set log level for specific logger or all loggers"""
        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, level.upper()))
        else:
            logging.getLogger().setLevel(getattr(logging, level.upper()))
        
        logging.info(f"Log level set to {level.upper()} for {logger_name or 'all loggers'}")
    
    def log_system_startup(self):
        """Log system startup information"""
        logger = self.get_logger('trading_bot')
        
        logger.info("=" * 80)
        logger.info("FOREX TRADING BOT STARTUP")
        logger.info("=" * 80)
        logger.info(f"Startup Time: {datetime.now().isoformat()}")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Working Directory: {os.getcwd()}")
        logger.info(f"Log Directory: {os.path.abspath('logs')}")
        logger.info("=" * 80)
    
    def log_system_shutdown(self):
        """Log system shutdown information"""
        logger = self.get_logger('trading_bot')
        
        logger.info("=" * 80)
        logger.info("FOREX TRADING BOT SHUTDOWN")
        logger.info("=" * 80)
        logger.info(f"Shutdown Time: {datetime.now().isoformat()}")
        logger.info("=" * 80)
    
    def get_log_files_info(self) -> Dict[str, Any]:
        """Get information about log files"""
        log_dir = Path('logs')
        log_files = {}
        
        if log_dir.exists():
            for log_file in log_dir.rglob('*.log'):
                try:
                    stat = log_file.stat()
                    log_files[str(log_file)] = {
                        'size_mb': stat.st_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'lines_count': self._count_lines(log_file)
                    }
                except Exception as e:
                    log_files[str(log_file)] = {'error': str(e)}
        
        return log_files
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file efficiently"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def cleanup_old_logs(self, days_old: int = 30):
        """Clean up log files older than specified days"""
        try:
            log_dir = Path('logs')
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            deleted_files = []
            
            for log_file in log_dir.rglob('*.log'):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    deleted_files.append(str(log_file))
            
            if deleted_files:
                logging.info(f"Cleaned up {len(deleted_files)} old log files")
            
            return deleted_files
            
        except Exception as e:
            logging.error(f"Error cleaning up old logs: {e}")
            return []


# Global log manager instance
_log_manager: Optional[LogManager] = None

def setup_logging(config: Dict[str, Any] = None) -> LogManager:
    """
    Setup and return global log manager
    
    Args:
        config: Optional logging configuration
        
    Returns:
        LogManager instance
    """
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager(config)
    return _log_manager

def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _log_manager
    if _log_manager is None:
        setup_logging()
    return _log_manager.get_logger(name)

def get_trade_logger() -> TradeLogger:
    """Get trade logger instance"""
    global _log_manager
    if _log_manager is None:
        setup_logging()
    return _log_manager.trade_logger

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance"""
    global _log_manager
    if _log_manager is None:
        setup_logging()
    return _log_manager.performance_logger


# Example usage and testing
if __name__ == "__main__":
    # Test the logging system
    print("Testing Logging Configuration...")
    
    try:
        # Setup logging
        log_manager = setup_logging()
        
        # Log system startup
        log_manager.log_system_startup()
        
        # Get different loggers
        main_logger = get_logger('trading_bot')
        core_logger = get_logger('core.data_handler')
        models_logger = get_logger('models.lstm')
        
        # Test different log levels
        main_logger.debug("This is a debug message")
        main_logger.info("This is an info message")
        main_logger.warning("This is a warning message")
        main_logger.error("This is an error message")
        
        # Test trade logging
        trade_logger = get_trade_logger()
        trade_logger.log_trade_signal(
            symbol="EUR/USD",
            signal="BUY",
            strength=0.85,
            indicators={'rsi': 35.2, 'macd': 0.0012, 'momentum': 0.045},
            model_confidence=0.78
        )
        
        trade_logger.log_trade_entry(
            symbol="EUR/USD",
            direction="LONG",
            entry_price=1.08542,
            quantity=0.1,
            trade_id="TRADE_001",
            confidence=0.82
        )
        
        trade_logger.log_trade_exit(
            trade_id="TRADE_001",
            exit_price=1.08715,
            pnl=17.30,
            pnl_pct=1.59,
            duration="2h 15m",
            reason="Take Profit"
        )
        
        # Test performance logging
        performance_logger = get_performance_logger()
        performance_logger.log_daily_performance(
            date="2024-01-15",
            pnl=245.50,
            pnl_pct=2.45,
            trades=8,
            win_rate=62.5,
            drawdown=1.2
        )
        
        performance_logger.log_risk_metrics(
            sharpe_ratio=1.85,
            sortino_ratio=2.12,
            max_drawdown=8.5,
            volatility=12.3,
            var_95=3.2
        )
        
        # Test error logging
        try:
            # Simulate an error
            raise ValueError("This is a test error for logging")
        except Exception as e:
            main_logger.error("Test error occurred", exc_info=True)
        
        # Get log files info
        log_files_info = log_manager.get_log_files_info()
        print(f"\n📊 Log Files Information:")
        for file_path, info in log_files_info.items():
            print(f"  {file_path}: {info.get('size_mb', 0):.2f}MB, {info.get('lines_count', 0)} lines")
        
        # Log system shutdown
        log_manager.log_system_shutdown()
        
        print(f"\n{Fore.GREEN}✅ Logging system test completed successfully!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Logging system test failed: {e}{Style.RESET_ALL}")
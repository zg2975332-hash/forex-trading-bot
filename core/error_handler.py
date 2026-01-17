"""
Advanced Error Handler for Forex Trading Bot
Comprehensive error management, recovery, and monitoring system
"""

import logging
import traceback
import sys
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading
import time
from contextlib import contextmanager
import inspect
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA_ERROR = "data_error"
    TRADING_ERROR = "trading_error"
    CONNECTION_ERROR = "connection_error"
    STRATEGY_ERROR = "strategy_error"
    RISK_ERROR = "risk_error"
    SYSTEM_ERROR = "system_error"
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    CONFIG_ERROR = "config_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorInfo:
    """Comprehensive error information container"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    exception_message: str
    traceback: str
    module: str
    function: str
    line_number: int
    context: Dict[str, Any]
    system_state: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    resolved: bool = False

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    name: str
    description: str
    action_function: Callable
    conditions: List[Callable]
    priority: int = 1
    max_attempts: int = 3
    cooldown_seconds: int = 60

class CircuitBreaker:
    """
    Circuit breaker pattern for error prevention
    """
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.RLock()
        
        logger.info(f"Circuit breaker initialized: threshold={failure_threshold}, timeout={reset_timeout}s")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        with self.lock:
            if self.state == "OPEN":
                # Check if reset timeout has passed
                if (datetime.now() - self.last_failure_time).total_seconds() > self.reset_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                    return True
                return False
            return True
    
    def record_success(self):
        """Record successful execution"""
        with self.lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.last_failure_time = None
                logger.info("Circuit breaker reset to CLOSED state")
    
    def record_failure(self):
        """Record failed execution"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold and self.state != "OPEN":
                self.state = "OPEN"
                logger.warning(f"Circuit breaker triggered to OPEN state after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self.lock:
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'failure_threshold': self.failure_threshold,
                'reset_timeout': self.reset_timeout
            }

class ErrorHandler:
    """
    Advanced error handling and recovery system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.error_history: List[ErrorInfo] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.alert_handlers: List[Callable] = []
        
        # Statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_category': {category.value: 0 for category in ErrorCategory},
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'recovery_attempts': 0,
            'recovery_successes': 0,
            'last_error_time': None
        }
        
        # Initialize components
        self._initialize_circuit_breakers()
        self._initialize_recovery_actions()
        self._initialize_alert_handlers()
        
        # Start monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info("Advanced Error Handler initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_error_history': 1000,
            'alert_thresholds': {
                'critical_errors_per_hour': 5,
                'recovery_failure_rate': 0.3,
                'memory_usage_mb': 1024
            },
            'circuit_breakers': {
                'trading': {'failure_threshold': 3, 'reset_timeout': 300},
                'data_fetch': {'failure_threshold': 5, 'reset_timeout': 60},
                'api_calls': {'failure_threshold': 10, 'reset_timeout': 120}
            },
            'notifications': {
                'email_alerts': False,
                'telegram_alerts': False,
                'webhook_alerts': False
            },
            'auto_recovery': True,
            'log_to_file': True,
            'error_reporting': True
        }
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for different components"""
        for breaker_name, breaker_config in self.config['circuit_breakers'].items():
            self.circuit_breakers[breaker_name] = CircuitBreaker(
                failure_threshold=breaker_config['failure_threshold'],
                reset_timeout=breaker_config['reset_timeout']
            )
    
    def _initialize_recovery_actions(self):
        """Initialize automatic recovery actions"""
        
        # Data connection recovery
        self.recovery_actions['reset_data_connection'] = RecoveryAction(
            name="Reset Data Connection",
            description="Reset data source connections",
            action_function=self._recover_data_connection,
            conditions=[
                lambda error: error.category in [ErrorCategory.DATA_ERROR, ErrorCategory.CONNECTION_ERROR],
                lambda error: "connection" in error.message.lower() or "timeout" in error.message.lower()
            ],
            priority=1,
            max_attempts=3
        )
        
        # API rate limit recovery
        self.recovery_actions['handle_rate_limit'] = RecoveryAction(
            name="Handle Rate Limit",
            description="Wait and retry after rate limit",
            action_function=self._recover_rate_limit,
            conditions=[
                lambda error: "rate limit" in error.message.lower() or "429" in error.message
            ],
            priority=2,
            cooldown_seconds=60
        )
        
        # Memory cleanup recovery
        self.recovery_actions['cleanup_memory'] = RecoveryAction(
            name="Memory Cleanup",
            description="Perform garbage collection and memory cleanup",
            action_function=self._recover_memory,
            conditions=[
                lambda error: "memory" in error.message.lower() or "MemoryError" in error.exception_type
            ],
            priority=3
        )
        
        # Strategy reset recovery
        self.recovery_actions['reset_strategy'] = RecoveryAction(
            name="Strategy Reset",
            description="Reset trading strategy state",
            action_function=self._recover_strategy,
            conditions=[
                lambda error: error.category == ErrorCategory.STRATEGY_ERROR
            ],
            priority=4,
            max_attempts=2
        )
        
        logger.info(f"Initialized {len(self.recovery_actions)} recovery actions")
    
    def _initialize_alert_handlers(self):
        """Initialize alert handlers"""
        if self.config['notifications']['email_alerts']:
            self.alert_handlers.append(self._send_email_alert)
        
        if self.config['notifications']['telegram_alerts']:
            self.alert_handlers.append(self._send_telegram_alert)
        
        if self.config['notifications']['webhook_alerts']:
            self.alert_handlers.append(self._send_webhook_alert)
    
    def handle_error(self, 
                    error: Exception,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                    context: Dict[str, Any] = None,
                    module: str = None,
                    function: str = None) -> ErrorInfo:
        """
        Main error handling method
        
        Args:
            error: The exception that occurred
            severity: Error severity level
            category: Error category
            context: Additional context information
            module: Module where error occurred
            function: Function where error occurred
            
        Returns:
            ErrorInfo object with error details
        """
        try:
            # Get caller information if not provided
            if not module or not function:
                frame = inspect.currentframe().f_back
                module = frame.f_globals.get('__name__', 'unknown')
                function = frame.f_code.co_name
            
            # Create error info
            error_info = self._create_error_info(
                error=error,
                severity=severity,
                category=category,
                context=context or {},
                module=module,
                function=function
            )
            
            # Update statistics
            self._update_error_stats(error_info)
            
            # Log the error
            self._log_error(error_info)
            
            # Attempt automatic recovery
            if self.config['auto_recovery']:
                recovery_success = self._attempt_recovery(error_info)
                error_info.recovery_attempted = True
                error_info.recovery_successful = recovery_success
            
            # Send alerts if needed
            if severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
                self._send_alerts(error_info)
            
            # Check alert thresholds
            self._check_alert_thresholds()
            
            # Store in history
            self._store_error_history(error_info)
            
            logger.info(f"Error handled: {error_info.error_id} - {error_info.message}")
            
            return error_info
            
        except Exception as e:
            # Emergency fallback for error handler errors
            logger.critical(f"CRITICAL: Error handler failed: {e}")
            emergency_info = ErrorInfo(
                error_id="EMERGENCY_" + str(int(time.time())),
                timestamp=datetime.now(),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM_ERROR,
                message="Error handler failure",
                exception_type=type(e).__name__,
                exception_message=str(e),
                traceback=traceback.format_exc(),
                module="error_handler",
                function="handle_error",
                line_number=0,
                context={},
                system_state=self._get_system_state()
            )
            return emergency_info
    
    def _create_error_info(self, error: Exception, severity: ErrorSeverity,
                          category: ErrorCategory, context: Dict[str, Any],
                          module: str, function: str) -> ErrorInfo:
        """Create ErrorInfo object from error details"""
        # Get traceback
        tb = traceback.format_exc()
        if not tb or "NoneType: None" in tb:
            tb = "".join(traceback.format_stack()[:-1])
        
        # Get line number
        frame = inspect.currentframe().f_back.f_back
        line_number = frame.f_lineno if frame else 0
        
        return ErrorInfo(
            error_id=f"ERR_{int(time.time())}_{hash(str(error)) % 10000:04d}",
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=str(error),
            exception_type=type(error).__name__,
            exception_message=str(error),
            traceback=tb,
            module=module,
            function=function,
            line_number=line_number,
            context=context,
            system_state=self._get_system_state()
        )
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'thread_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'system_memory_percent': psutil.virtual_memory().percent,
                'system_cpu_percent': psutil.cpu_percent(),
                'python_version': sys.version,
                'platform': sys.platform
            }
        except Exception as e:
            return {'error': f"Failed to get system state: {str(e)}"}
    
    def _update_error_stats(self, error_info: ErrorInfo):
        """Update error statistics"""
        self.error_stats['total_errors'] += 1
        self.error_stats['errors_by_category'][error_info.category.value] += 1
        self.error_stats['errors_by_severity'][error_info.severity.value] += 1
        self.error_stats['last_error_time'] = error_info.timestamp
        
        if error_info.recovery_attempted:
            self.error_stats['recovery_attempts'] += 1
            if error_info.recovery_successful:
                self.error_stats['recovery_successes'] += 1
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = f"{error_info.error_id} - {error_info.message} in {error_info.module}.{error_info.function}:{error_info.line_number}"
        
        if error_info.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif error_info.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif error_info.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif error_info.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        
        # Log traceback for errors and criticals
        if error_info.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
            logger.error(f"Traceback for {error_info.error_id}:\n{error_info.traceback}")
    
    def _attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt automatic recovery for the error"""
        applicable_actions = []
        
        # Find applicable recovery actions
        for action_name, action in self.recovery_actions.items():
            if all(condition(error_info) for condition in action.conditions):
                applicable_actions.append(action)
        
        # Sort by priority (highest first)
        applicable_actions.sort(key=lambda x: x.priority, reverse=True)
        
        # Execute recovery actions
        recovery_success = False
        for action in applicable_actions:
            try:
                logger.info(f"Attempting recovery: {action.name}")
                result = action.action_function(error_info)
                if result:
                    recovery_success = True
                    logger.info(f"Recovery successful: {action.name}")
                    break  # Stop after first successful recovery
                else:
                    logger.warning(f"Recovery failed: {action.name}")
            except Exception as e:
                logger.error(f"Recovery action {action.name} failed: {e}")
        
        return recovery_success
    
    def _recover_data_connection(self, error_info: ErrorInfo) -> bool:
        """Recovery action for data connection issues"""
        try:
            # Simulate reconnection logic
            logger.info("Attempting data connection recovery...")
            time.sleep(2)  # Simulate reconnection delay
            logger.info("Data connection recovery completed")
            return True
        except Exception as e:
            logger.error(f"Data connection recovery failed: {e}")
            return False
    
    def _recover_rate_limit(self, error_info: ErrorInfo) -> bool:
        """Recovery action for API rate limits"""
        try:
            logger.info("Handling API rate limit - waiting 60 seconds")
            time.sleep(60)
            logger.info("Rate limit cooldown completed")
            return True
        except Exception as e:
            logger.error(f"Rate limit recovery failed: {e}")
            return False
    
    def _recover_memory(self, error_info: ErrorInfo) -> bool:
        """Recovery action for memory issues"""
        try:
            logger.info("Performing memory cleanup...")
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear various caches if they exist
            if 'data_handler' in globals():
                data_handler = globals()['data_handler']
                if hasattr(data_handler, 'cache'):
                    data_handler.cache.clear()
            
            logger.info("Memory cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _recover_strategy(self, error_info: ErrorInfo) -> bool:
        """Recovery action for strategy errors"""
        try:
            logger.info("Resetting trading strategy...")
            # Reset strategy state
            # This would interact with your strategy manager
            time.sleep(1)
            logger.info("Strategy reset completed")
            return True
        except Exception as e:
            logger.error(f"Strategy recovery failed: {e}")
            return False
    
    def _send_alerts(self, error_info: ErrorInfo):
        """Send alerts through configured channels"""
        for alert_handler in self.alert_handlers:
            try:
                alert_handler(error_info)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _send_email_alert(self, error_info: ErrorInfo):
        """Send email alert"""
        # Implementation would depend on your email configuration
        logger.info(f"Would send email alert for error: {error_info.error_id}")
    
    def _send_telegram_alert(self, error_info: ErrorInfo):
        """Send Telegram alert"""
        # Implementation would depend on your Telegram bot configuration
        logger.info(f"Would send Telegram alert for error: {error_info.error_id}")
    
    def _send_webhook_alert(self, error_info: ErrorInfo):
        """Send webhook alert"""
        # Implementation would depend on your webhook configuration
        logger.info(f"Would send webhook alert for error: {error_info.error_id}")
    
    def _check_alert_thresholds(self):
        """Check if any alert thresholds have been exceeded"""
        try:
            # Check critical errors per hour
            hour_ago = datetime.now() - timedelta(hours=1)
            recent_critical_errors = len([
                e for e in self.error_history 
                if e.severity == ErrorSeverity.CRITICAL and e.timestamp > hour_ago
            ])
            
            if recent_critical_errors >= self.config['alert_thresholds']['critical_errors_per_hour']:
                self._trigger_threshold_alert(
                    f"Critical error threshold exceeded: {recent_critical_errors} errors in last hour"
                )
            
            # Check recovery failure rate
            if self.error_stats['recovery_attempts'] > 10:
                failure_rate = 1 - (self.error_stats['recovery_successes'] / self.error_stats['recovery_attempts'])
                if failure_rate > self.config['alert_thresholds']['recovery_failure_rate']:
                    self._trigger_threshold_alert(
                        f"High recovery failure rate: {failure_rate:.1%}"
                    )
            
            # Check memory usage
            system_state = self._get_system_state()
            memory_usage = system_state.get('memory_usage_mb', 0)
            if memory_usage > self.config['alert_thresholds']['memory_usage_mb']:
                self._trigger_threshold_alert(
                    f"High memory usage: {memory_usage:.0f} MB"
                )
                
        except Exception as e:
            logger.error(f"Error checking alert thresholds: {e}")
    
    def _trigger_threshold_alert(self, message: str):
        """Trigger threshold alert"""
        logger.warning(f"ALERT: {message}")
        # Could send additional notifications here
    
    def _store_error_history(self, error_info: ErrorInfo):
        """Store error in history with size limits"""
        self.error_history.append(error_info)
        
        # Trim history if too large
        if len(self.error_history) > self.config['max_error_history']:
            self.error_history = self.error_history[-self.config['max_error_history']:]
    
    @contextmanager
    def handle_errors(self, 
                     severity: ErrorSeverity = ErrorSeverity.ERROR,
                     category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                     context: Dict[str, Any] = None,
                     circuit_breaker: str = None):
        """
        Context manager for error handling
        
        Usage:
            with error_handler.handle_errors(category=ErrorCategory.DATA_ERROR):
                # code that might raise exceptions
        """
        try:
            if circuit_breaker and circuit_breaker in self.circuit_breakers:
                breaker = self.circuit_breakers[circuit_breaker]
                if not breaker.can_execute():
                    raise Exception(f"Circuit breaker {circuit_breaker} is OPEN")
            
            yield
            
            if circuit_breaker and circuit_breaker in self.circuit_breakers:
                self.circuit_breakers[circuit_breaker].record_success()
                
        except Exception as e:
            if circuit_breaker and circuit_breaker in self.circuit_breakers:
                self.circuit_breakers[circuit_breaker].record_failure()
            
            # Get caller information
            frame = inspect.currentframe().f_back
            module = frame.f_globals.get('__name__', 'unknown')
            function = frame.f_code.co_name
            
            self.handle_error(
                error=e,
                severity=severity,
                category=category,
                context=context,
                module=module,
                function=function
            )
            raise
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.error_stats,
            'recent_errors': [
                {
                    'error_id': error.error_id,
                    'timestamp': error.timestamp.isoformat(),
                    'severity': error.severity.value,
                    'category': error.category.value,
                    'message': error.message,
                    'module': error.module,
                    'function': error.function,
                    'recovery_attempted': error.recovery_attempted,
                    'recovery_successful': error.recovery_successful
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ],
            'circuit_breaker_states': {
                name: breaker.get_state() 
                for name, breaker in self.circuit_breakers.items()
            },
            'system_state': self._get_system_state()
        }
    
    def start_monitoring(self):
        """Start error monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        logger.info("Error monitoring started")
    
    def stop_monitoring(self):
        """Stop error monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Error monitoring stopped")
    
    def _monitoring_worker(self):
        """Background monitoring worker"""
        while self.monitoring_active:
            try:
                # Check for stuck states
                self._check_stuck_states()
                
                # Generate periodic report
                if len(self.error_history) > 0:
                    last_error = self.error_history[-1]
                    if (datetime.now() - last_error.timestamp).total_seconds() > 3600:  # 1 hour
                        logger.info("No errors in last hour - system stable")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _check_stuck_states(self):
        """Check for stuck circuit breakers or other issues"""
        for breaker_name, breaker in self.circuit_breakers.items():
            state = breaker.get_state()
            if state['state'] == 'OPEN':
                open_time = datetime.now() - datetime.fromisoformat(state['last_failure_time'])
                if open_time.total_seconds() > breaker.reset_timeout * 2:  # Double timeout
                    logger.warning(f"Circuit breaker {breaker_name} stuck in OPEN state")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        logger.info("Error Handler cleanup completed")


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None

def get_error_handler(config: Dict[str, Any] = None) -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(config)
    return _error_handler

def handle_error(error: Exception, **kwargs):
    """Convenience function to handle errors"""
    handler = get_error_handler()
    return handler.handle_error(error, **kwargs)

@contextmanager
def error_context(**kwargs):
    """Convenience context manager for error handling"""
    handler = get_error_handler()
    with handler.handle_errors(**kwargs):
        yield


# Example usage and testing
if __name__ == "__main__":
    # Test the error handler
    print("Testing Error Handler...")
    
    try:
        # Initialize error handler
        error_handler = get_error_handler()
        
        # Test basic error handling
        print("Testing basic error handling...")
        try:
            raise ValueError("This is a test error")
        except Exception as e:
            error_info = error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM_ERROR,
                context={'test': True, 'phase': 'basic_test'}
            )
        
        print(f"Error handled: {error_info.error_id}")
        
        # Test context manager
        print("Testing context manager...")
        try:
            with error_handler.handle_errors(
                category=ErrorCategory.DATA_ERROR,
                circuit_breaker='data_fetch'
            ):
                raise ConnectionError("Simulated connection error")
        except Exception as e:
            print(f"Context manager caught: {e}")
        
        # Test circuit breaker
        print("Testing circuit breaker...")
        for i in range(5):
            try:
                with error_handler.handle_errors(circuit_breaker='api_calls'):
                    if i < 3:
                        raise Exception(f"API call failed {i+1}")
                    else:
                        print("API call would succeed now")
            except Exception as e:
                print(f"Attempt {i+1}: {e}")
        
        # Test recovery actions
        print("Testing recovery actions...")
        try:
            raise MemoryError("Simulated memory error")
        except Exception as e:
            error_info = error_handler.handle_error(
                error=e,
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM_ERROR
            )
            print(f"Recovery attempted: {error_info.recovery_attempted}")
            print(f"Recovery successful: {error_info.recovery_successful}")
        
        # Generate report
        print("Generating error report...")
        report = error_handler.get_error_report()
        print(f"Total errors: {report['statistics']['total_errors']}")
        print(f"Error categories: {report['statistics']['errors_by_category']}")
        
        # Start monitoring
        error_handler.start_monitoring()
        time.sleep(2)  # Let monitoring run briefly
        
        # Cleanup
        error_handler.cleanup()
        
        print(f"\n✅ Error Handler test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error Handler test failed: {e}")
        import traceback
        traceback.print_exc()
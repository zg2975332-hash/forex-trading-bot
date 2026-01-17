"""
Environment Configuration Manager for Forex Trading Bot
Secure environment variable handling and configuration management
"""

import os
import logging
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import hashlib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "forex_bot"
    username: str = "forex_user"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20

@dataclass
class APIConfig:
    """API configuration settings"""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    rate_limit_requests: int = 1200
    rate_limit_seconds: int = 60

@dataclass
class TradingConfig:
    """Trading configuration settings"""
    risk_per_trade: float = 0.02
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.05
    default_lot_size: float = 0.1
    max_position_size: float = 1.0
    leverage: int = 30

@dataclass
class ModelConfig:
    """AI Model configuration settings"""
    sequence_length: int = 60
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_layers: List[int] = None
    dropout_rate: float = 0.2
    confidence_threshold: float = 0.7

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [50, 25]

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    file_path: str = "logs/forex_bot.log"
    max_file_size: str = "100MB"
    backup_count: int = 10
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class MonitoringConfig:
    """Monitoring configuration settings"""
    health_check_interval: int = 60
    performance_check_interval: int = 3600
    alert_enabled: bool = True
    metrics_export: bool = True

class EnvironmentConfig:
    """
    Secure environment configuration manager for Forex Trading Bot
    Handles environment variables, encryption, and configuration validation
    """
    
    def __init__(self, env_file: str = ".env", config_dir: str = "config"):
        self.env_file = env_file
        self.config_dir = config_dir
        self.encryption_key = None
        self.cipher_suite = None
        self.config_cache = {}
        
        # Load environment variables
        self._load_environment()
        
        # Initialize encryption
        self._initialize_encryption()
        
        # Load configurations
        self._load_all_configs()
        
        logger.info("Environment Configuration Manager initialized")
    
    def _load_environment(self):
        """Load environment variables from .env file and system"""
        try:
            # Load from .env file
            if os.path.exists(self.env_file):
                load_dotenv(self.env_file)
                logger.info(f"Environment variables loaded from {self.env_file}")
            else:
                logger.warning(f"Environment file {self.env_file} not found, using system environment")
            
            # Set default environment if not set
            if not os.getenv('ENVIRONMENT'):
                os.environ['ENVIRONMENT'] = 'development'
                logger.info("Default environment set to 'development'")
                
        except Exception as e:
            logger.error(f"Error loading environment: {e}")
            raise
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive data"""
        try:
            # Get or generate encryption key
            encryption_key = os.getenv('ENCRYPTION_KEY')
            if not encryption_key:
                # Generate a new key (in production, this should be set as environment variable)
                encryption_key = Fernet.generate_key().decode()
                os.environ['ENCRYPTION_KEY'] = encryption_key
                logger.warning("New encryption key generated - save this key securely!")
            
            self.encryption_key = encryption_key.encode()
            self.cipher_suite = Fernet(self.encryption_key)
            logger.info("Encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
            raise
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value"""
        try:
            if not value:
                return ""
            encrypted_value = self.cipher_suite.encrypt(value.encode())
            return encrypted_value.decode()
        except Exception as e:
            logger.error(f"Error encrypting value: {e}")
            return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted value"""
        try:
            if not encrypted_value:
                return ""
            decrypted_value = self.cipher_suite.decrypt(encrypted_value.encode())
            return decrypted_value.decode()
        except Exception as e:
            logger.error(f"Error decrypting value: {e}")
            return encrypted_value
    
    def _load_all_configs(self):
        """Load all configuration sections"""
        try:
            # Load API configurations
            self.binance_config = self._load_binance_config()
            self.exness_config = self._load_exness_config()
            self.news_config = self._load_news_config()
            
            # Load service configurations
            self.database_config = self._load_database_config()
            self.trading_config = self._load_trading_config()
            self.model_config = self._load_model_config()
            self.logging_config = self._load_logging_config()
            self.monitoring_config = self._load_monitoring_config()
            
            logger.info("All configurations loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise
    
    def _load_binance_config(self) -> APIConfig:
        """Load Binance API configuration"""
        return APIConfig(
            api_key=os.getenv('BINANCE_API_KEY', ''),
            api_secret=self.decrypt_value(os.getenv('BINANCE_API_SECRET', '')),
            testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true',
            rate_limit_requests=int(os.getenv('BINANCE_RATE_LIMIT', '1200')),
            rate_limit_seconds=60
        )
    
    def _load_exness_config(self) -> APIConfig:
        """Load Exness API configuration"""
        return APIConfig(
            api_key=os.getenv('EXNESS_API_KEY', ''),
            api_secret=self.decrypt_value(os.getenv('EXNESS_API_SECRET', '')),
            testnet=os.getenv('EXNESS_DEMO_ACCOUNT', 'true').lower() == 'true',
            rate_limit_requests=int(os.getenv('EXNESS_RATE_LIMIT', '1000')),
            rate_limit_seconds=60
        )
    
    def _load_news_config(self) -> Dict[str, Any]:
        """Load news and sentiment API configurations"""
        return {
            'news_api': {
                'api_key': os.getenv('NEWS_API_KEY', ''),
                'sources': ['bloomberg', 'reuters', 'financial-times'],
                'categories': ['business', 'finance', 'economics']
            },
            'twitter': {
                'api_key': os.getenv('TWITTER_API_KEY', ''),
                'api_secret': self.decrypt_value(os.getenv('TWITTER_API_SECRET', '')),
                'bearer_token': self.decrypt_value(os.getenv('TWITTER_BEARER_TOKEN', ''))
            },
            'reddit': {
                'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
                'client_secret': self.decrypt_value(os.getenv('REDDIT_CLIENT_SECRET', ''))
            }
        }
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            host=os.getenv('DATABASE_HOST', 'localhost'),
            port=int(os.getenv('DATABASE_PORT', '5432')),
            database=os.getenv('DATABASE_NAME', 'forex_bot'),
            username=os.getenv('DATABASE_USER', 'forex_user'),
            password=self.decrypt_value(os.getenv('DATABASE_PASSWORD', '')),
            ssl_mode=os.getenv('DATABASE_SSL_MODE', 'prefer'),
            pool_size=int(os.getenv('DATABASE_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DATABASE_MAX_OVERFLOW', '20'))
        )
    
    def _load_trading_config(self) -> TradingConfig:
        """Load trading configuration"""
        return TradingConfig(
            risk_per_trade=float(os.getenv('RISK_PER_TRADE', '0.02')),
            max_drawdown=float(os.getenv('MAX_DRAWDOWN', '0.15')),
            daily_loss_limit=float(os.getenv('DAILY_LOSS_LIMIT', '0.05')),
            default_lot_size=float(os.getenv('DEFAULT_LOT_SIZE', '0.1')),
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '1.0')),
            leverage=int(os.getenv('LEVERAGE', '30'))
        )
    
    def _load_model_config(self) -> ModelConfig:
        """Load AI model configuration"""
        return ModelConfig(
            sequence_length=int(os.getenv('MODEL_SEQUENCE_LENGTH', '60')),
            batch_size=int(os.getenv('MODEL_BATCH_SIZE', '32')),
            learning_rate=float(os.getenv('MODEL_LEARNING_RATE', '0.001')),
            hidden_layers=json.loads(os.getenv('MODEL_HIDDEN_LAYERS', '[50, 25]')),
            dropout_rate=float(os.getenv('MODEL_DROPOUT_RATE', '0.2')),
            confidence_threshold=float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.7'))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration"""
        return LoggingConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            file_path=os.getenv('LOG_FILE_PATH', 'logs/forex_bot.log'),
            max_file_size=os.getenv('LOG_MAX_FILE_SIZE', '100MB'),
            backup_count=int(os.getenv('LOG_BACKUP_COUNT', '10')),
            format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    def _load_monitoring_config(self) -> MonitoringConfig:
        """Load monitoring configuration"""
        return MonitoringConfig(
            health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '60')),
            performance_check_interval=int(os.getenv('PERFORMANCE_CHECK_INTERVAL', '3600')),
            alert_enabled=os.getenv('ALERT_ENABLED', 'true').lower() == 'true',
            metrics_export=os.getenv('METRICS_EXPORT', 'true').lower() == 'true'
        )
    
    def get_environment(self) -> str:
        """Get current environment"""
        return os.getenv('ENVIRONMENT', 'development')
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.get_environment() == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.get_environment() == 'production'
    
    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self.get_environment() == 'staging'
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate all configurations and return issues
    
        Returns:
            Dict with validation results
        """
    
        if self.is_development():
            return {'errors': [], 'warnings': [], 'info': ['Development mode - validation bypassed']}
    
        validation_results = {
            'errors': [],
            'warnings': [],
            'info': []
       }
        
        try:
            # Validate API configurations
            if not self.binance_config.api_key:
                validation_results['errors'].append("Binance API key is missing")
            if not self.binance_config.api_secret:
                validation_results['errors'].append("Binance API secret is missing")
            
            if not self.exness_config.api_key:
                validation_results['warnings'].append("Exness API key is missing")
            if not self.exness_config.api_secret:
                validation_results['warnings'].append("Exness API secret is missing")
            
            # Validate database configuration
            if not self.database_config.password:
                validation_results['errors'].append("Database password is missing")
            
            # Validate trading configuration
            if self.trading_config.risk_per_trade > 0.05:
                validation_results['warnings'].append("Risk per trade is high (above 5%)")
            
            if self.trading_config.leverage > 50:
                validation_results['warnings'].append("Leverage is high (above 50:1)")
            
            # Environment specific validations
            if self.is_production():
                if self.binance_config.testnet:
                    validation_results['errors'].append("Production environment should not use testnet")
                
                if not self.database_config.ssl_mode == 'require':
                    validation_results['warnings'].append("Production database should use SSL require mode")
            
            # Info messages
            validation_results['info'].append(f"Environment: {self.get_environment()}")
            validation_results['info'].append(f"Database: {self.database_config.host}:{self.database_config.port}")
            validation_results['info'].append(f"Risk per trade: {self.trading_config.risk_per_trade:.1%}")
            
            logger.info(f"Configuration validation completed: {len(validation_results['errors'])} errors, "
                       f"{len(validation_results['warnings'])} warnings")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during configuration validation: {e}")
            validation_results['errors'].append(f"Validation error: {str(e)}")
            return validation_results
    
    def save_encrypted_config(self, file_path: str = "config/encrypted_config.yaml"):
        """
        Save encrypted configuration to file (for backup)
        
        Args:
            file_path: Path to save encrypted configuration
        """
        try:
            config_data = {
                'binance': {
                    'api_key': self.encrypt_value(self.binance_config.api_key),
                    'api_secret': self.encrypt_value(self.binance_config.api_secret)
                },
                'exness': {
                    'api_key': self.encrypt_value(self.exness_config.api_key),
                    'api_secret': self.encrypt_value(self.exness_config.api_secret)
                },
                'database': {
                    'password': self.encrypt_value(self.database_config.password)
                },
                'environment': self.get_environment(),
                'timestamp': self._get_timestamp()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info(f"Encrypted configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving encrypted configuration: {e}")
    
    def create_env_template(self, file_path: str = ".env.template"):
        """
        Create environment template file for documentation
        
        Args:
            file_path: Path to save template file
        """
        try:
            template = """# Forex Trading Bot Environment Configuration
# Copy this file to .env and fill in your actual values

# Environment Settings
ENVIRONMENT=development  # development, staging, production

# Encryption Key (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
ENCRYPTION_KEY=2vLzubzkA-9-ItH40lnHz-ytWaS00XtpaY6SHnjaonc=

# Binance API Configuration
BINANCE_API_KEY=RkXrL1EHUPavsnrUNnQogNwMnYfMYPkbjcim1lQEnZ5FL9RNqFbDZ6dJhenHuzXJ
BINANCE_API_SECRET=BRtQwZmiLXhH8GEbW1yrR1aMVQetR9VPdyEGmxMhbPMJoJVmhAA9Llgb9l5dr2qR
BINANCE_TESTNET=true
BINANCE_RATE_LIMIT=1200

# Exness API Configuration
EXNESS_API_KEY=your_exness_api_key_here
EXNESS_API_SECRET=your_exness_api_secret_here
EXNESS_DEMO_ACCOUNT=true
EXNESS_RATE_LIMIT=1000

# News and Sentiment APIs
NEWS_API_KEY=your_news_api_key_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=forex_bot
DATABASE_USER=forex_user
DATABASE_PASSWORD=Umais106332@##5325#55404#zohaib#159357
DATABASE_SSL_MODE=prefer
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Trading Parameters
RISK_PER_TRADE=0.02
MAX_DRAWDOWN=0.15
DAILY_LOSS_LIMIT=0.05
DEFAULT_LOT_SIZE=0.1
MAX_POSITION_SIZE=1.0
LEVERAGE=30

# AI Model Configuration
MODEL_SEQUENCE_LENGTH=60
MODEL_BATCH_SIZE=32
MODEL_LEARNING_RATE=0.001
MODEL_HIDDEN_LAYERS=[50, 25]
MODEL_DROPOUT_RATE=0.2
MODEL_CONFIDENCE_THRESHOLD=0.7

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/forex_bot.log
LOG_MAX_FILE_SIZE=100MB
LOG_BACKUP_COUNT=10
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Monitoring Configuration
HEALTH_CHECK_INTERVAL=60
PERFORMANCE_CHECK_INTERVAL=3600
ALERT_ENABLED=true
METRICS_EXPORT=true
"""
            
            with open(file_path, 'w') as f:
                f.write(template)
            
            logger.info(f"Environment template created at {file_path}")
            
        except Exception as e:
            logger.error(f"Error creating environment template: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get safe configuration summary (without sensitive data)
        
        Returns:
            Configuration summary dictionary
        """
        return {
            'environment': self.get_environment(),
            'binance': {
                'testnet': self.binance_config.testnet,
                'rate_limit': f"{self.binance_config.rate_limit_requests}/{self.binance_config.rate_limit_seconds}s"
            },
            'exness': {
                'demo_account': self.exness_config.testnet,
                'rate_limit': f"{self.exness_config.rate_limit_requests}/{self.exness_config.rate_limit_seconds}s"
            },
            'database': {
                'host': self.database_config.host,
                'port': self.database_config.port,
                'database': self.database_config.database,
                'ssl_mode': self.database_config.ssl_mode
            },
            'trading': asdict(self.trading_config),
            'model': asdict(self.model_config),
            'logging': {
                'level': self.logging_config.level,
                'file_path': self.logging_config.file_path
            },
            'monitoring': asdict(self.monitoring_config)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for configuration versioning"""
        from datetime import datetime
        return datetime.now().isoformat()


# Utility functions
def setup_environment(env_file: str = ".env") -> EnvironmentConfig:
    """
    Setup and validate environment configuration
    
    Args:
        env_file: Path to environment file
        
    Returns:
        Initialized EnvironmentConfig instance
    """
    try:
        config = EnvironmentConfig(env_file)
        
        # Validate configuration
        validation = config.validate_configuration()
        
        # Log validation results
        for error in validation['errors']:
            logger.error(f"Configuration Error: {error}")
        
        for warning in validation['warnings']:
            logger.warning(f"Configuration Warning: {warning}")
        
        for info in validation['info']:
            logger.info(f"Configuration Info: {info}")
        
        # Raise exception if there are critical errors
        if validation['errors']:
            raise ValueError("Critical configuration errors found. Please fix before proceeding.")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Test the environment configuration
    try:
        print("Testing Environment Configuration...")
        
        # Initialize configuration
        env_config = setup_environment()
        
        # Print configuration summary
        summary = env_config.get_config_summary()
        print("\nConfiguration Summary:")
        print(json.dumps(summary, indent=2))
        
        # Validate configuration
        validation = env_config.validate_configuration()
        print(f"\nValidation Results: {len(validation['errors'])} errors, {len(validation['warnings'])} warnings")
        
        # Create environment template
        env_config.create_env_template()
        print("\nEnvironment template created: .env.template")
        
        # Test encryption
        test_value = "sensitive_data"
        encrypted = env_config.encrypt_value(test_value)
        decrypted = env_config.decrypt_value(encrypted)
        print(f"\nEncryption Test: '{test_value}' -> encrypted -> '{decrypted}'")
        
        print("\n✅ Environment configuration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Environment configuration test failed: {e}")
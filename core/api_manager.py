import time
import threading
import requests
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from collections import deque, defaultdict
import json
import hmac
import hashlib
import urllib.parse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class APIManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('forex_bot.api_manager')
        
        # API rate limiting and monitoring
        self.rate_limits = {}
        self.request_history = defaultdict(deque)
        self.api_errors = deque(maxlen=100)
        self.connection_status = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # API clients
        self.exchange_clients = {}
        self.news_clients = {}
        self.sentiment_clients = {}
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1
        self.timeout = 30
        
        # Initialize all API connections
        self._initialize_exchanges()
        self._initialize_news_apis()
        self._initialize_sentiment_apis()
        
    def _initialize_exchanges(self):
        """Initialize exchange API connections"""
        try:
            exchanges_config = self.config.get('api', {})
            
            # Binance
            if 'binance' in exchanges_config:
                self.exchange_clients['binance'] = ccxt.binance({
                    'apiKey': exchanges_config['binance']['api_key'],
                    'secret': exchanges_config['binance']['api_secret'],
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True
                    },
                    'sandbox': exchanges_config['binance'].get('sandbox', False)
                })
                self.rate_limits['binance'] = {
                    'requests_per_minute': 1200,
                    'orders_per_minute': 50,
                    'last_reset': datetime.now(),
                    'request_count': 0,
                    'order_count': 0
                }
                self.connection_status['binance'] = True
                self.logger.info("✅ Binance API initialized")
            
            # Exness (custom implementation)
            if 'exness' in exchanges_config:
                self.exchange_clients['exness'] = self._create_exness_client(
                    exchanges_config['exness']
                )
                self.rate_limits['exness'] = {
                    'requests_per_minute': 600,
                    'orders_per_minute': 30,
                    'last_reset': datetime.now(),
                    'request_count': 0,
                    'order_count': 0
                }
                self.connection_status['exness'] = True
                self.logger.info("✅ Exness API initialized")
                
            # Other exchanges can be added similarly
            self.logger.info(f"Initialized {len(self.exchange_clients)} exchange APIs")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {e}")

    def _create_exness_client(self, config: Dict) -> Dict:
        """Create custom Exness client (since it's not in CCXT)"""
        return {
            'api_key': config['api_key'],
            'api_secret': config['api_secret'],
            'base_url': config.get('base_url', 'https://api.exness.com'),
            'sandbox': config.get('sandbox', False)
        }

    def _initialize_news_apis(self):
        """Initialize news API connections"""
        try:
            news_config = self.config.get('news_apis', {})
            
            # Alpha Vantage
            if 'alpha_vantage' in news_config:
                self.news_clients['alpha_vantage'] = {
                    'api_key': news_config['alpha_vantage']['api_key'],
                    'base_url': 'https://www.alphavantage.co/query'
                }
                self.rate_limits['alpha_vantage'] = {
                    'requests_per_minute': 5,  # Free tier limit
                    'last_reset': datetime.now(),
                    'request_count': 0
                }
            
            # NewsAPI
            if 'newsapi' in news_config:
                self.news_clients['newsapi'] = {
                    'api_key': news_config['newsapi']['api_key'],
                    'base_url': 'https://newsapi.org/v2'
                }
                self.rate_limits['newsapi'] = {
                    'requests_per_minute': 100,  # Free tier limit
                    'last_reset': datetime.now(),
                    'request_count': 0
                }
            
            # Forex Factory (web scraping)
            self.news_clients['forex_factory'] = {
                'base_url': 'https://www.forexfactory.com'
            }
            
            self.logger.info(f"Initialized {len(self.news_clients)} news APIs")
            
        except Exception as e:
            self.logger.error(f"Error initializing news APIs: {e}")

    def _initialize_sentiment_apis(self):
        """Initialize sentiment analysis APIs"""
        try:
            sentiment_config = self.config.get('sentiment_apis', {})
            
            # Twitter API
            if 'twitter' in sentiment_config:
                self.sentiment_clients['twitter'] = {
                    'bearer_token': sentiment_config['twitter']['bearer_token'],
                    'base_url': 'https://api.twitter.com/2'
                }
                self.rate_limits['twitter'] = {
                    'requests_per_minute': 300,
                    'last_reset': datetime.now(),
                    'request_count': 0
                }
            
            # Reddit API
            if 'reddit' in sentiment_config:
                self.sentiment_clients['reddit'] = {
                    'client_id': sentiment_config['reddit']['client_id'],
                    'client_secret': sentiment_config['reddit']['client_secret'],
                    'user_agent': sentiment_config['reddit']['user_agent']
                }
            
            self.logger.info(f"Initialized {len(self.sentiment_clients)} sentiment APIs")
            
        except Exception as e:
            self.logger.error(f"Error initializing sentiment APIs: {e}")

    def make_request(self, service: str, endpoint: str, method: str = 'GET', 
                    params: Dict = None, data: Dict = None, 
                    retry_on_failure: bool = True) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        try:
            # Check rate limits
            if not self._check_rate_limit(service):
                self.logger.warning(f"Rate limit exceeded for {service}")
                return None
            
            # Update request count
            self._update_request_count(service)
            
            # Make request based on service type
            if service in self.exchange_clients:
                return self._make_exchange_request(service, endpoint, method, params, data)
            elif service in self.news_clients:
                return self._make_news_request(service, endpoint, method, params)
            elif service in self.sentiment_clients:
                return self._make_sentiment_request(service, endpoint, method, params)
            else:
                self.logger.error(f"Unknown service: {service}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error making API request to {service}: {e}")
            self._record_error(service, str(e))
            
            if retry_on_failure:
                return self._retry_request(service, endpoint, method, params, data)
            
            return None

    def _make_exchange_request(self, exchange: str, endpoint: str, method: str,
                             params: Dict, data: Dict) -> Optional[Dict]:
        """Make exchange API request"""
        try:
            client = self.exchange_clients[exchange]
            
            if exchange == 'binance':
                # Use CCXT for Binance
                if endpoint == 'fetch_ohlcv':
                    return client.fetch_ohlcv(
                        params['symbol'], 
                        params['timeframe'], 
                        since=params.get('since'),
                        limit=params.get('limit', 100)
                    )
                elif endpoint == 'create_order':
                    return client.create_order(
                        params['symbol'],
                        params['type'],
                        params['side'],
                        params['amount'],
                        params.get('price')
                    )
                elif endpoint == 'fetch_balance':
                    return client.fetch_balance()
                elif endpoint == 'fetch_order_book':
                    return client.fetch_order_book(params['symbol'])
                    
            elif exchange == 'exness':
                # Custom Exness implementation
                return self._make_exness_request(endpoint, method, params, data)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error making {exchange} request: {e}")
            raise

    def _make_exness_request(self, endpoint: str, method: str, 
                           params: Dict, data: Dict) -> Optional[Dict]:
        """Make Exness API request (custom implementation)"""
        try:
            client = self.exchange_clients['exness']
            base_url = client['base_url']
            api_key = client['api_key']
            api_secret = client['api_secret']
            
            # Add authentication
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_exness_signature(api_secret, timestamp)
            
            headers = {
                'X-API-KEY': api_key,
                'X-API-SIGNATURE': signature,
                'X-API-TIMESTAMP': timestamp,
                'Content-Type': 'application/json'
            }
            
            url = f"{base_url}/{endpoint}"
            
            session = self._create_session()
            
            if method.upper() == 'GET':
                response = session.get(url, params=params, headers=headers, timeout=self.timeout)
            else:
                response = session.post(url, json=data, headers=headers, timeout=self.timeout)
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error making Exness request: {e}")
            raise

    def _generate_exness_signature(self, secret: str, timestamp: str) -> str:
        """Generate Exness API signature"""
        message = timestamp + secret
        return hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_news_request(self, service: str, endpoint: str, method: str,
                          params: Dict) -> Optional[Dict]:
        """Make news API request"""
        try:
            client = self.news_clients[service]
            base_url = client['base_url']
            api_key = client.get('api_key')
            
            # Add API key to params
            if api_key and service != 'forex_factory':
                params = params or {}
                params['apikey'] = api_key
            
            url = f"{base_url}/{endpoint}"
            
            session = self._create_session()
            
            if method.upper() == 'GET':
                response = session.get(url, params=params, timeout=self.timeout)
            else:
                response = session.post(url, json=params, timeout=self.timeout)
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error making {service} news request: {e}")
            raise

    def _make_sentiment_request(self, service: str, endpoint: str, method: str,
                               params: Dict) -> Optional[Dict]:
        """Make sentiment API request"""
        try:
            client = self.sentiment_clients[service]
            
            if service == 'twitter':
                return self._make_twitter_request(endpoint, method, params)
            elif service == 'reddit':
                return self._make_reddit_request(endpoint, method, params)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error making {service} sentiment request: {e}")
            raise

    def _make_twitter_request(self, endpoint: str, method: str, 
                             params: Dict) -> Optional[Dict]:
        """Make Twitter API request"""
        try:
            client = self.sentiment_clients['twitter']
            base_url = client['base_url']
            bearer_token = client['bearer_token']
            
            headers = {
                'Authorization': f'Bearer {bearer_token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{base_url}/{endpoint}"
            
            session = self._create_session()
            
            if method.upper() == 'GET':
                response = session.get(url, params=params, headers=headers, timeout=self.timeout)
            else:
                response = session.post(url, json=params, headers=headers, timeout=self.timeout)
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error making Twitter request: {e}")
            raise

    def _make_reddit_request(self, endpoint: str, method: str, 
                            params: Dict) -> Optional[Dict]:
        """Make Reddit API request"""
        try:
            # Reddit API implementation would go here
            # This is a simplified version
            pass
            
        except Exception as e:
            self.logger.error(f"Error making Reddit request: {e}")
            raise

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"],
            backoff_factor=self.retry_delay
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _check_rate_limit(self, service: str) -> bool:
        """Check if request is within rate limits"""
        with self._lock:
            if service not in self.rate_limits:
                return True
            
            limits = self.rate_limits[service]
            now = datetime.now()
            
            # Reset counter if minute has passed
            if now - limits['last_reset'] > timedelta(minutes=1):
                limits['request_count'] = 0
                limits['order_count'] = 0
                limits['last_reset'] = now
            
            # Check request limits
            max_requests = limits.get('requests_per_minute', float('inf'))
            if limits['request_count'] >= max_requests:
                return False
            
            # Check order limits (for exchanges)
            max_orders = limits.get('orders_per_minute', float('inf'))
            if limits.get('order_count', 0) >= max_orders:
                return False
            
            return True

    def _update_request_count(self, service: str, is_order: bool = False):
        """Update request and order counts"""
        with self._lock:
            if service in self.rate_limits:
                self.rate_limits[service]['request_count'] += 1
                if is_order:
                    self.rate_limits[service]['order_count'] += 1

    def _retry_request(self, service: str, endpoint: str, method: str,
                      params: Dict, data: Dict) -> Optional[Dict]:
        """Retry failed request with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Retry attempt {attempt + 1} for {service}")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
                return self.make_request(
                    service, endpoint, method, params, data, retry_on_failure=False
                )
                
            except Exception as e:
                self.logger.warning(f"Retry {attempt + 1} failed for {service}: {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"All retry attempts failed for {service}")
                    return None

    def _record_error(self, service: str, error: str):
        """Record API error for monitoring"""
        error_record = {
            'service': service,
            'error': error,
            'timestamp': datetime.now(),
            'retry_count': 0
        }
        self.api_errors.append(error_record)

    def get_market_data(self, symbol: str, timeframe: str = '1h', 
                       limit: int = 100) -> Optional[List]:
        """Get market data from exchanges"""
        try:
            # Try primary exchange first
            exchange = 'binance'
            ohlcv = self.make_request(
                exchange, 
                'fetch_ohlcv',
                params={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'limit': limit
                }
            )
            
            if ohlcv:
                return ohlcv
            
            # Fallback to secondary exchange
            if 'exness' in self.exchange_clients:
                self.logger.info(f"Falling back to Exness for {symbol}")
                # Implement Exness market data fetch
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    def place_order(self, exchange: str, symbol: str, order_type: str, 
                   side: str, amount: float, price: float = None) -> Optional[Dict]:
        """Place order on exchange"""
        try:
            if exchange not in self.exchange_clients:
                self.logger.error(f"Exchange {exchange} not available")
                return None
            
            params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount
            }
            
            if price:
                params['price'] = price
            
            result = self.make_request(
                exchange,
                'create_order',
                method='POST',
                params=params,
                is_order=True  # Mark as order for rate limiting
            )
            
            if result:
                self.logger.info(f"Order placed on {exchange}: {side} {amount} {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order on {exchange}: {e}")
            return None

    def get_account_balance(self, exchange: str) -> Optional[Dict]:
        """Get account balance from exchange"""
        try:
            return self.make_request(exchange, 'fetch_balance')
        except Exception as e:
            self.logger.error(f"Error getting balance from {exchange}: {e}")
            return None

    def get_news_sentiment(self, query: str, days: int = 7) -> Optional[Dict]:
        """Get news sentiment data"""
        try:
            # Try multiple news sources
            news_data = {}
            
            # Alpha Vantage
            if 'alpha_vantage' in self.news_clients:
                news_data['alpha_vantage'] = self.make_request(
                    'alpha_vantage',
                    'query',
                    params={
                        'function': 'NEWS_SENTIMENT',
                        'tickers': 'EURUSD',
                        'apikey': self.news_clients['alpha_vantage']['api_key']
                    }
                )
            
            # NewsAPI
            if 'newsapi' in self.news_clients:
                news_data['newsapi'] = self.make_request(
                    'newsapi',
                    'everything',
                    params={
                        'q': query,
                        'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                        'sortBy': 'relevancy',
                        'language': 'en'
                    }
                )
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {e}")
            return None

    def get_social_sentiment(self, platform: str, query: str) -> Optional[Dict]:
        """Get social media sentiment data"""
        try:
            if platform == 'twitter':
                return self.make_request(
                    'twitter',
                    'tweets/search/recent',
                    params={
                        'query': query,
                        'max_results': 100,
                        'tweet.fields': 'created_at,public_metrics'
                    }
                )
            elif platform == 'reddit':
                # Implement Reddit sentiment
                pass
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting {platform} sentiment: {e}")
            return None

    def get_api_status(self) -> Dict:
        """Get comprehensive API status"""
        status = {
            'exchanges': {},
            'news_apis': {},
            'sentiment_apis': {},
            'rate_limits': {},
            'recent_errors': list(self.api_errors)[-10:],  # Last 10 errors
            'total_requests': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Exchange status
        for exchange in self.exchange_clients:
            status['exchanges'][exchange] = {
                'connected': self.connection_status.get(exchange, False),
                'rate_limit': self.rate_limits.get(exchange, {}),
                'last_used': None  # Would track last successful request
            }
        
        # News API status
        for news_api in self.news_clients:
            status['news_apis'][news_api] = {
                'configured': True,
                'rate_limit': self.rate_limits.get(news_api, {})
            }
        
        # Sentiment API status
        for sentiment_api in self.sentiment_clients:
            status['sentiment_apis'][sentiment_api] = {
                'configured': True,
                'rate_limit': self.rate_limits.get(sentiment_api, {})
            }
        
        # Rate limit status
        for service, limits in self.rate_limits.items():
            status['rate_limits'][service] = {
                'requests_this_minute': limits.get('request_count', 0),
                'max_requests_per_minute': limits.get('requests_per_minute', 'N/A'),
                'orders_this_minute': limits.get('order_count', 0),
                'max_orders_per_minute': limits.get('orders_per_minute', 'N/A'),
                'last_reset': limits.get('last_reset').isoformat() if limits.get('last_reset') else 'N/A'
            }
        
        return status

    def health_check(self) -> Dict:
        """Perform health check on all APIs"""
        health = {
            'overall_status': 'healthy',
            'services': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test exchange connections
        for exchange in self.exchange_clients:
            try:
                if exchange == 'binance':
                    # Simple API call to test connection
                    self.exchange_clients[exchange].fetch_time()
                    health['services'][exchange] = 'healthy'
                else:
                    health['services'][exchange] = 'configured'
            except Exception as e:
                health['services'][exchange] = 'unhealthy'
                health['overall_status'] = 'degraded'
                self.logger.error(f"Health check failed for {exchange}: {e}")
        
        # Update overall status
        unhealthy_services = [s for s, status in health['services'].items() if status == 'unhealthy']
        if unhealthy_services:
            health['overall_status'] = 'unhealthy' if len(unhealthy_services) == len(health['services']) else 'degraded'
        
        return health

    def cleanup(self):
        """Cleanup API connections"""
        try:
            # Close exchange connections
            for exchange, client in self.exchange_clients.items():
                if hasattr(client, 'close'):
                    client.close()
            
            self.logger.info("API manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during API manager cleanup: {e}")
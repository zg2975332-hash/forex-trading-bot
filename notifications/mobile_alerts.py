"""
Advanced Mobile Alerts System for FOREX TRADING BOT
Real-time push notifications for trading signals, market events, and system alerts
"""

import logging
import json
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import hmac
import hashlib
import base64

logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    TRADING_SIGNAL = "trading_signal"
    MARKET_EVENT = "market_event"
    PRICE_ALERT = "price_alert"
    RISK_ALERT = "risk_alert"
    SYSTEM_ALERT = "system_alert"
    SENTIMENT_ALERT = "sentiment_alert"
    ECONOMIC_EVENT = "economic_event"
    VOLATILITY_ALERT = "volatility_alert"

class NotificationChannel(Enum):
    PUSH_NOTIFICATION = "push"
    SMS = "sms"
    EMAIL = "email"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SLACK = "slack"
    WEBHOOK = "webhook"

@dataclass
class MobileAlert:
    """Mobile alert data structure"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    symbol: str
    timestamp: datetime
    expiration: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels: List[NotificationChannel] = field(default_factory=list)
    acknowledged: bool = False
    delivered: bool = False

@dataclass
class AlertConfig:
    """Configuration for mobile alerts system"""
    # Push Notification Services
    enable_push_notifications: bool = True
    onesignal_app_id: str = ""
    onesignal_rest_api_key: str = ""
    firebase_server_key: str = ""
    
    # SMS Gateway
    enable_sms: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    
    # Email Settings
    enable_email: bool = True
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    from_email: str = ""
    
    # Telegram Bot
    enable_telegram: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Discord Webhook
    enable_discord: bool = False
    discord_webhook_url: str = ""
    
    # Slack Webhook
    enable_slack: bool = False
    slack_webhook_url: str = ""
    
    # Custom Webhook
    enable_webhook: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""
    
    # Alert Settings
    max_alerts_per_hour: int = 50
    alert_cooldown_period: int = 300  # seconds
    priority_routing: Dict[AlertPriority, List[NotificationChannel]] = field(default_factory=lambda: {
        AlertPriority.CRITICAL: [NotificationChannel.PUSH_NOTIFICATION, NotificationChannel.SMS, NotificationChannel.EMAIL],
        AlertPriority.HIGH: [NotificationChannel.PUSH_NOTIFICATION, NotificationChannel.EMAIL],
        AlertPriority.MEDIUM: [NotificationChannel.PUSH_NOTIFICATION],
        AlertPriority.LOW: [NotificationChannel.EMAIL]
    })
    
    # User Management
    user_preferences: Dict[str, Dict] = field(default_factory=dict)
    
    # Retry Settings
    max_retry_attempts: int = 3
    retry_delay: int = 30  # seconds

class AdvancedMobileAlerts:
    """
    Advanced mobile alerts system for real-time trading notifications
    """
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        
        # Alert storage
        self.alerts_queue = deque(maxlen=1000)
        self.sent_alerts = defaultdict(lambda: deque(maxlen=500))
        self.failed_alerts = defaultdict(lambda: deque(maxlen=200))
        
        # Rate limiting
        self.alert_timestamps = deque(maxlen=self.config.max_alerts_per_hour)
        self.cooldown_tracker = defaultdict(float)
        
        # User device tokens (would typically come from database)
        self.user_devices = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        self._sending_lock = threading.Lock()
        
        # Initialize services
        self._initialize_services()
        
        # Background tasks
        self._start_background_tasks()
        
        logger.info("AdvancedMobileAlerts initialized successfully")

    def _initialize_services(self):
        """Initialize notification services"""
        try:
            # Initialize HTTP session
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'ForexTradingBot/1.0',
                'Content-Type': 'application/json'
            })
            
            # OneSignal headers if configured
            if self.config.onesignal_app_id and self.config.onesignal_rest_api_key:
                self.onesignal_headers = {
                    'Authorization': f'Basic {self.config.onesignal_rest_api_key}',
                    'Content-Type': 'application/json'
                }
            
            logger.info("Notification services initialized")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")

    def _start_background_tasks(self):
        """Start background alert processing tasks"""
        # Alert processing loop
        processing_thread = threading.Thread(target=self._alert_processing_loop, daemon=True)
        processing_thread.start()
        
        # Retry failed alerts
        retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
        retry_thread.start()
        
        # Cleanup old alerts
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        # Delivery monitoring
        monitoring_thread = threading.Thread(target=self._delivery_monitoring_loop, daemon=True)
        monitoring_thread.start()

    def create_alert(self, alert_type: AlertType, priority: AlertPriority, 
                    title: str, message: str, symbol: str = "GLOBAL",
                    metadata: Dict[str, Any] = None) -> str:
        """
        Create a new mobile alert
        """
        try:
            # Check rate limiting
            if self._is_rate_limited():
                logger.warning("Rate limit exceeded, alert queued for later delivery")
            
            # Check cooldown for similar alerts
            cooldown_key = f"{alert_type.value}_{symbol}"
            if self._is_in_cooldown(cooldown_key):
                logger.debug(f"Alert in cooldown: {cooldown_key}")
                return ""
            
            # Create alert
            alert_id = f"{alert_type.value}_{symbol}_{int(time.time())}"
            expiration = datetime.now() + timedelta(hours=24)
            
            alert = MobileAlert(
                alert_id=alert_id,
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                symbol=symbol,
                timestamp=datetime.now(),
                expiration=expiration,
                metadata=metadata or {},
                channels=self.config.priority_routing.get(priority, [NotificationChannel.PUSH_NOTIFICATION])
            )
            
            # Add to queue
            with self._lock:
                self.alerts_queue.append(alert)
            
            # Update cooldown tracker
            self.cooldown_tracker[cooldown_key] = time.time()
            
            logger.info(f"Alert created: {alert_id} - {title}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
            return ""

    def _is_rate_limited(self) -> bool:
        """Check if we're exceeding rate limits"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Remove old timestamps
        while self.alert_timestamps and self.alert_timestamps[0] < hour_ago:
            self.alert_timestamps.popleft()
        
        return len(self.alert_timestamps) >= self.config.max_alerts_per_hour

    def _is_in_cooldown(self, cooldown_key: str) -> bool:
        """Check if alert type is in cooldown period"""
        last_alert_time = self.cooldown_tracker.get(cooldown_key, 0)
        return time.time() - last_alert_time < self.config.alert_cooldown_period

    def _alert_processing_loop(self):
        """Background loop for processing alerts"""
        while True:
            try:
                # Process alerts from queue
                if self.alerts_queue:
                    alert = self.alerts_queue[0]  # Peek at first alert
                    
                    # Check if alert is expired
                    if datetime.now() > alert.expiration:
                        with self._lock:
                            self.alerts_queue.popleft()
                        continue
                    
                    # Send alert
                    success = self._send_alert(alert)
                    
                    if success:
                        with self._lock:
                            sent_alert = self.alerts_queue.popleft()
                            sent_alert.delivered = True
                            self.sent_alerts[sent_alert.symbol].append(sent_alert)
                        
                        # Update rate limiting
                        self.alert_timestamps.append(time.time())
                        
                        logger.info(f"Alert delivered: {alert.alert_id}")
                    else:
                        # Move to failed queue for retry
                        with self._lock:
                            failed_alert = self.alerts_queue.popleft()
                            self.failed_alerts[failed_alert.alert_id].append({
                                'alert': failed_alert,
                                'attempts': 1,
                                'last_attempt': datetime.now()
                            })
                        
                        logger.warning(f"Alert delivery failed: {alert.alert_id}")
                
                time.sleep(1)  # Small delay between processing
                
            except Exception as e:
                logger.error(f"Alert processing loop failed: {e}")
                time.sleep(5)

    def _send_alert(self, alert: MobileAlert) -> bool:
        """
        Send alert through configured channels
        """
        try:
            success_count = 0
            total_channels = len(alert.channels)
            
            for channel in alert.channels:
                try:
                    channel_success = False
                    
                    if channel == NotificationChannel.PUSH_NOTIFICATION:
                        channel_success = self._send_push_notification(alert)
                    
                    elif channel == NotificationChannel.EMAIL:
                        channel_success = self._send_email_alert(alert)
                    
                    elif channel == NotificationChannel.TELEGRAM:
                        channel_success = self._send_telegram_alert(alert)
                    
                    elif channel == NotificationChannel.SMS:
                        channel_success = self._send_sms_alert(alert)
                    
                    elif channel == NotificationChannel.DISCORD:
                        channel_success = self._send_discord_alert(alert)
                    
                    elif channel == NotificationChannel.SLACK:
                        channel_success = self._send_slack_alert(alert)
                    
                    elif channel == NotificationChannel.WEBHOOK:
                        channel_success = self._send_webhook_alert(alert)
                    
                    if channel_success:
                        success_count += 1
                        logger.debug(f"Alert sent via {channel.value}: {alert.alert_id}")
                    else:
                        logger.warning(f"Alert failed via {channel.value}: {alert.alert_id}")
                        
                except Exception as channel_error:
                    logger.error(f"Channel {channel.value} failed: {channel_error}")
                    continue
            
            # Consider successful if at least one channel worked
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Alert sending failed: {e}")
            return False

    def _send_push_notification(self, alert: MobileAlert) -> bool:
        """Send push notification via OneSignal or Firebase"""
        try:
            if self.config.enable_push_notifications and self.config.onesignal_app_id:
                return self._send_onesignal_notification(alert)
            elif self.config.enable_push_notifications and self.config.firebase_server_key:
                return self._send_firebase_notification(alert)
            else:
                logger.warning("Push notifications not configured")
                return False
                
        except Exception as e:
            logger.error(f"Push notification failed: {e}")
            return False

    def _send_onesignal_notification(self, alert: MobileAlert) -> bool:
        """Send notification via OneSignal"""
        try:
            # Get user device tokens (in real implementation, this would come from database)
            device_tokens = self._get_user_device_tokens(alert.symbol)
            
            if not device_tokens:
                logger.warning("No device tokens found for OneSignal")
                return False
            
            payload = {
                'app_id': self.config.onesignal_app_id,
                'include_player_ids': device_tokens,
                'headings': {'en': alert.title},
                'contents': {'en': alert.message},
                'data': alert.metadata,
                'priority': 10 if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL] else 5,
                'ttl': 3600,  # 1 hour time to live
                'android_visibility': 1,
                'small_icon': 'ic_stat_onesignal_default',
                'large_icon': 'ic_launcher',
                'android_accent_color': 'FF9977D0'
            }
            
            response = requests.post(
                'https://onesignal.com/api/v1/notifications',
                headers=self.onesignal_headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"OneSignal notification sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"OneSignal API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"OneSignal notification failed: {e}")
            return False

    def _send_firebase_notification(self, alert: MobileAlert) -> bool:
        """Send notification via Firebase Cloud Messaging"""
        try:
            device_tokens = self._get_user_device_tokens(alert.symbol)
            
            if not device_tokens:
                return False
            
            payload = {
                'registration_ids': device_tokens,
                'notification': {
                    'title': alert.title,
                    'body': alert.message,
                    'sound': 'default',
                    'badge': '1'
                },
                'data': alert.metadata,
                'priority': 'high' if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL] else 'normal'
            }
            
            headers = {
                'Authorization': f'key={self.config.firebase_server_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                'https://fcm.googleapis.com/fcm/send',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', 0) > 0:
                    logger.info(f"Firebase notification sent: {alert.alert_id}")
                    return True
            
            logger.error(f"Firebase notification failed: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Firebase notification failed: {e}")
            return False

    def _send_email_alert(self, alert: MobileAlert) -> bool:
        """Send alert via email"""
        try:
            if not self.config.enable_email:
                return False
            
            # Get user emails (in real implementation, from database)
            user_emails = self._get_user_emails(alert.symbol)
            
            if not user_emails:
                logger.warning("No email addresses found")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ', '.join(user_emails)
            msg['Subject'] = f"[{alert.priority.value.upper()}] {alert.title}"
            
            # Create HTML email body
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .alert {{ border-left: 4px solid {self._get_priority_color(alert.priority)}; padding: 15px; background: #f9f9f9; }}
                    .priority {{ font-weight: bold; color: {self._get_priority_color(alert.priority)}; }}
                    .symbol {{ color: #2c3e50; font-weight: bold; }}
                    .timestamp {{ color: #7f8c8d; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="alert">
                    <div class="priority">{alert.priority.value.upper()} ALERT</div>
                    <h2>{alert.title}</h2>
                    <p>{alert.message}</p>
                    <div class="symbol">Symbol: {alert.symbol}</div>
                    <div class="timestamp">Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False

    def _send_telegram_alert(self, alert: MobileAlert) -> bool:
        """Send alert via Telegram bot"""
        try:
            if not self.config.enable_telegram or not self.config.telegram_bot_token:
                return False
            
            # Format message for Telegram
            message = f"""
            🚨 *{alert.title}* 🚨

            *Priority*: {alert.priority.value.upper()}
            *Symbol*: {alert.symbol}
            *Time*: {alert.timestamp.strftime('%H:%M:%S')}

            {alert.message}

            📊 _Forex Trading Bot Alert_
            """
            
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.config.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"Telegram API error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
            return False

    def _send_sms_alert(self, alert: MobileAlert) -> bool:
        """Send alert via SMS (Twilio)"""
        try:
            if not self.config.enable_sms or not self.config.twilio_account_sid:
                return False
            
            # This would require Twilio API integration
            # For now, return False as placeholder
            logger.warning("SMS alerts not fully implemented")
            return False
            
        except Exception as e:
            logger.error(f"SMS alert failed: {e}")
            return False

    def _send_discord_alert(self, alert: MobileAlert) -> bool:
        """Send alert via Discord webhook"""
        try:
            if not self.config.enable_discord or not self.config.discord_webhook_url:
                return False
            
            # Create embed for Discord
            embed = {
                'title': alert.title,
                'description': alert.message,
                'color': self._get_priority_color_discord(alert.priority),
                'timestamp': alert.timestamp.isoformat(),
                'fields': [
                    {
                        'name': 'Priority',
                        'value': alert.priority.value.upper(),
                        'inline': True
                    },
                    {
                        'name': 'Symbol',
                        'value': alert.symbol,
                        'inline': True
                    },
                    {
                        'name': 'Type',
                        'value': alert.alert_type.value.replace('_', ' ').title(),
                        'inline': True
                    }
                ],
                'footer': {
                    'text': 'Forex Trading Bot'
                }
            }
            
            payload = {
                'embeds': [embed],
                'username': 'Trading Bot Alerts',
                'avatar_url': 'https://example.com/bot_avatar.png'
            }
            
            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Discord alert sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"Discord webhook error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Discord alert failed: {e}")
            return False

    def _send_slack_alert(self, alert: MobileAlert) -> bool:
        """Send alert via Slack webhook"""
        try:
            if not self.config.enable_slack or not self.config.slack_webhook_url:
                return False
            
            # Create Slack message
            priority_color = self._get_priority_color_slack(alert.priority)
            
            payload = {
                'attachments': [
                    {
                        'color': priority_color,
                        'title': alert.title,
                        'text': alert.message,
                        'fields': [
                            {
                                'title': 'Priority',
                                'value': alert.priority.value.upper(),
                                'short': True
                            },
                            {
                                'title': 'Symbol',
                                'value': alert.symbol,
                                'short': True
                            }
                        ],
                        'ts': time.time(),
                        'footer': 'Forex Trading Bot'
                    }
                ]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"Slack webhook error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            return False

    def _send_webhook_alert(self, alert: MobileAlert) -> bool:
        """Send alert via custom webhook"""
        try:
            if not self.config.enable_webhook or not self.config.webhook_url:
                return False
            
            payload = {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'priority': alert.priority.value,
                'title': alert.title,
                'message': alert.message,
                'symbol': alert.symbol,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            # Add signature if secret is provided
            headers = {'Content-Type': 'application/json'}
            if self.config.webhook_secret:
                signature = hmac.new(
                    self.config.webhook_secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-Signature'] = signature
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook alert sent: {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            return False

    def _get_priority_color(self, priority: AlertPriority) -> str:
        """Get color for priority level"""
        colors = {
            AlertPriority.CRITICAL: "#e74c3c",  # Red
            AlertPriority.HIGH: "#e67e22",      # Orange
            AlertPriority.MEDIUM: "#f39c12",    # Yellow
            AlertPriority.LOW: "#3498db"        # Blue
        }
        return colors.get(priority, "#95a5a6")  # Default gray

    def _get_priority_color_discord(self, priority: AlertPriority) -> int:
        """Get Discord color integer for priority"""
        colors = {
            AlertPriority.CRITICAL: 0xe74c3c,  # Red
            AlertPriority.HIGH: 0xe67e22,      # Orange
            AlertPriority.MEDIUM: 0xf39c12,    # Yellow
            AlertPriority.LOW: 0x3498db        # Blue
        }
        return colors.get(priority, 0x95a5a6)  # Default gray

    def _get_priority_color_slack(self, priority: AlertPriority) -> str:
        """Get Slack color for priority"""
        colors = {
            AlertPriority.CRITICAL: "danger",
            AlertPriority.HIGH: "warning",
            AlertPriority.MEDIUM: "good",
            AlertPriority.LOW: "#439FE0"
        }
        return colors.get(priority, "#95a5a6")

    def _get_user_device_tokens(self, symbol: str) -> List[str]:
        """Get device tokens for users interested in this symbol"""
        # In real implementation, this would query a database
        # For demo purposes, return some mock tokens
        return ["mock_device_token_1", "mock_device_token_2"]

    def _get_user_emails(self, symbol: str) -> List[str]:
        """Get email addresses for users interested in this symbol"""
        # In real implementation, this would query a database
        return ["trader@example.com"]

    def _retry_loop(self):
        """Background loop for retrying failed alerts"""
        while True:
            try:
                current_time = datetime.now()
                retry_count = 0
                
                with self._lock:
                    failed_keys = list(self.failed_alerts.keys())
                    
                    for alert_id in failed_keys:
                        if alert_id in self.failed_alerts:
                            failed_entry = self.failed_alerts[alert_id][-1]  # Get latest attempt
                            
                            if (failed_entry['attempts'] < self.config.max_retry_attempts and
                                (current_time - failed_entry['last_attempt']).total_seconds() > self.config.retry_delay):
                                
                                # Retry the alert
                                alert = failed_entry['alert']
                                success = self._send_alert(alert)
                                
                                if success:
                                    # Move to sent alerts
                                    alert.delivered = True
                                    self.sent_alerts[alert.symbol].append(alert)
                                    del self.failed_alerts[alert_id]
                                    logger.info(f"Retry successful for alert: {alert_id}")
                                else:
                                    # Update retry information
                                    failed_entry['attempts'] += 1
                                    failed_entry['last_attempt'] = current_time
                                    logger.warning(f"Retry failed for alert: {alert_id} (attempt {failed_entry['attempts']})")
                                
                                retry_count += 1
                
                if retry_count > 0:
                    logger.info(f"Retried {retry_count} failed alerts")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Retry loop failed: {e}")
                time.sleep(30)

    def _cleanup_loop(self):
        """Background loop for cleaning up old alerts"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                cleanup_count = 0
                
                with self._lock:
                    # Clean old sent alerts
                    for symbol in list(self.sent_alerts.keys()):
                        self.sent_alerts[symbol] = deque(
                            [alert for alert in self.sent_alerts[symbol] if alert.timestamp > cutoff_time],
                            maxlen=500
                        )
                        cleanup_count += 1
                    
                    # Clean old failed alerts
                    failed_keys = list(self.failed_alerts.keys())
                    for alert_id in failed_keys:
                        if alert_id in self.failed_alerts:
                            failed_entry = self.failed_alerts[alert_id][-1]
                            if failed_entry['last_attempt'] < cutoff_time:
                                del self.failed_alerts[alert_id]
                                cleanup_count += 1
                
                if cleanup_count > 0:
                    logger.debug(f"Cleaned up {cleanup_count} old alerts")
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Cleanup loop failed: {e}")
                time.sleep(1800)

    def _delivery_monitoring_loop(self):
        """Background loop for monitoring delivery statistics"""
        while True:
            try:
                stats = self.get_delivery_statistics()
                logger.info(f"Delivery Statistics: {stats}")
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Delivery monitoring failed: {e}")
                time.sleep(60)

    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get delivery statistics for alerts"""
        try:
            total_sent = sum(len(alerts) for alerts in self.sent_alerts.values())
            total_failed = sum(len(entries) for entries in self.failed_alerts.values())
            total_queued = len(self.alerts_queue)
            
            # Calculate delivery rate
            total_attempted = total_sent + total_failed
            delivery_rate = (total_sent / total_attempted * 100) if total_attempted > 0 else 0
            
            # Platform breakdown
            platform_stats = defaultdict(int)
            for symbol_alerts in self.sent_alerts.values():
                for alert in symbol_alerts:
                    for channel in alert.channels:
                        platform_stats[channel.value] += 1
            
            return {
                'timestamp': datetime.now(),
                'total_sent': total_sent,
                'total_failed': total_failed,
                'total_queued': total_queued,
                'delivery_rate': round(delivery_rate, 2),
                'platform_breakdown': dict(platform_stats),
                'recent_alerts': [
                    {
                        'id': alert.alert_id,
                        'type': alert.alert_type.value,
                        'priority': alert.priority.value,
                        'symbol': alert.symbol,
                        'timestamp': alert.timestamp.isoformat(),
                        'delivered': alert.delivered
                    }
                    for alert in list(self.alerts_queue)[-5:]  # Last 5 queued alerts
                ]
            }
            
        except Exception as e:
            logger.error(f"Delivery statistics calculation failed: {e}")
            return {'timestamp': datetime.now(), 'error': str(e)}

    def send_trading_signal(self, symbol: str, signal_type: str, price: float, 
                          confidence: float, message: str) -> str:
        """Send trading signal alert"""
        priority = AlertPriority.HIGH if confidence > 0.7 else AlertPriority.MEDIUM
        
        title = f"Trading Signal: {symbol} {signal_type.upper()}"
        full_message = f"{message}\n\nPrice: {price}\nConfidence: {confidence:.1%}"
        
        metadata = {
            'signal_type': signal_type,
            'price': price,
            'confidence': confidence,
            'action': 'BUY' if signal_type.lower() == 'bullish' else 'SELL'
        }
        
        return self.create_alert(
            AlertType.TRADING_SIGNAL,
            priority,
            title,
            full_message,
            symbol,
            metadata
        )

    def send_price_alert(self, symbol: str, current_price: float, 
                        target_price: float, condition: str) -> str:
        """Send price level alert"""
        title = f"Price Alert: {symbol} {condition.upper()}"
        message = f"{symbol} has {condition} {target_price}. Current price: {current_price}"
        
        metadata = {
            'current_price': current_price,
            'target_price': target_price,
            'condition': condition,
            'price_difference': abs(current_price - target_price)
        }
        
        return self.create_alert(
            AlertType.PRICE_ALERT,
            AlertPriority.MEDIUM,
            title,
            message,
            symbol,
            metadata
        )

    def send_risk_alert(self, symbol: str, risk_level: str, message: str, 
                       metrics: Dict[str, float]) -> str:
        """Send risk management alert"""
        priority_map = {
            'extreme': AlertPriority.CRITICAL,
            'high': AlertPriority.HIGH,
            'medium': AlertPriority.MEDIUM,
            'low': AlertPriority.LOW
        }
        
        priority = priority_map.get(risk_level.lower(), AlertPriority.MEDIUM)
        title = f"Risk Alert: {symbol} - {risk_level.upper()} Risk"
        
        metadata = {
            'risk_level': risk_level,
            'risk_metrics': metrics
        }
        
        return self.create_alert(
            AlertType.RISK_ALERT,
            priority,
            title,
            message,
            symbol,
            metadata
        )

    def send_system_alert(self, component: str, issue: str, severity: str, 
                         details: str) -> str:
        """Send system health alert"""
        priority_map = {
            'critical': AlertPriority.CRITICAL,
            'error': AlertPriority.HIGH,
            'warning': AlertPriority.MEDIUM,
            'info': AlertPriority.LOW
        }
        
        priority = priority_map.get(severity.lower(), AlertPriority.MEDIUM)
        title = f"System Alert: {component} - {issue}"
        
        metadata = {
            'component': component,
            'issue': issue,
            'severity': severity,
            'details': details
        }
        
        return self.create_alert(
            AlertType.SYSTEM_ALERT,
            priority,
            title,
            details,
            "SYSTEM",
            metadata
        )

# Example usage and testing
def main():
    """Example usage of the AdvancedMobileAlerts system"""
    
    # Configuration
    config = AlertConfig(
        enable_email=True,
        enable_telegram=False,  # Set to True with actual bot token
        enable_push_notifications=False,  # Set to True with actual API keys
        max_alerts_per_hour=10
    )
    
    # Initialize alerts system
    alerts = AdvancedMobileAlerts(config)
    
    print("=== Mobile Alerts System Demo ===")
    
    # Send various types of alerts
    alert_ids = []
    
    # Trading signal
    alert_ids.append(
        alerts.send_trading_signal(
            symbol="EUR/USD",
            signal_type="bullish",
            price=1.0950,
            confidence=0.82,
            message="Strong bullish momentum with RSI oversold and MACD crossover"
        )
    )
    
    # Price alert
    alert_ids.append(
        alerts.send_price_alert(
            symbol="GBP/USD",
            current_price=1.2750,
            target_price=1.2800,
            condition="broken above resistance"
        )
    )
    
    # Risk alert
    alert_ids.append(
        alerts.send_risk_alert(
            symbol="USD/JPY",
            risk_level="high",
            message="Unusual volatility detected, consider reducing position size",
            metrics={'volatility': 2.5, 'drawdown': 0.15}
        )
    )
    
    # System alert
    alert_ids.append(
        alerts.send_system_alert(
            component="Data Feed",
            issue="Connection interrupted",
            severity="warning",
            details="Binance API connection lost, attempting reconnect..."
        )
    )
    
    print(f"Sent {len(alert_ids)} demo alerts")
    
    # Wait for processing
    print("Waiting for alert processing...")
    time.sleep(5)
    
    # Get delivery statistics
    print("\n=== Delivery Statistics ===")
    stats = alerts.get_delivery_statistics()
    print(f"Total Sent: {stats['total_sent']}")
    print(f"Total Failed: {stats['total_failed']}")
    print(f"Delivery Rate: {stats['delivery_rate']}%")
    print(f"Platform Breakdown: {stats['platform_breakdown']}")
    
    # Show recent alerts
    print("\n=== Recent Alerts ===")
    for alert in stats['recent_alerts']:
        print(f"  - {alert['type']}: {alert['symbol']} ({alert['priority']})")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
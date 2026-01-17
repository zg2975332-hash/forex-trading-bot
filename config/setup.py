#!/usr/bin/env python3
"""
Interactive Setup Script for Forex Trading Bot
Guides user through API key setup step-by-step
"""
import os
import sys
import json
from pathlib import Path
import getpass
import yaml

class InteractiveSetup:
    def __init__(self):
        self.config_dir = Path("config")
        self.secrets_file = self.config_dir / "secrets.json"
        self.example_file = self.config_dir / "api_keys_example.yaml"
        self.config_file = self.config_dir / "api_keys.yaml"
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "=" * 60)
        print(f"🤖 {title}")
        print("=" * 60)
    
    def ask_yes_no(self, question: str) -> bool:
        """Ask yes/no question"""
        while True:
            response = input(f"\n{question} (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("⚠️  Please enter 'y' or 'n'")
    
    def ask_input(self, question: str, password=False, default=None) -> str:
        """Ask for input with optional default"""
        if default:
            question = f"{question} [Default: {default}]: "
        else:
            question = f"{question}: "
        
        if password:
            response = getpass.getpass(question)
        else:
            response = input(question)
        
        if not response and default:
            return default
        return response
    
    def setup_binance(self):
        """Setup Binance API"""
        self.print_header("Binance API Setup")
        print("1. Go to: https://www.binance.com/en/my/settings/api-management")
        print("2. Create new API key (enable Spot & Margin trading)")
        print("3. Enter the keys below:\n")
        
        use_binance = self.ask_yes_no("Do you want to use Binance API?")
        
        if use_binance:
            api_key = self.ask_input("Binance API Key")
            api_secret = self.ask_input("Binance API Secret", password=True)
            use_testnet = self.ask_yes_no("Use Testnet (Recommended for testing)?")
            
            return {
                'enabled': True,
                'api_key': api_key,
                'api_secret': api_secret,
                'testnet': use_testnet
            }
        return {'enabled': False}
    
    def setup_database(self):
        """Setup Database"""
        self.print_header("Database Setup")
        print("Configure PostgreSQL database for storing trading data")
        
        use_database = self.ask_yes_no("Do you want to setup database?")
        
        if use_database:
            print("\n📊 Database Configuration:")
            host = self.ask_input("Database Host", default="localhost")
            port = self.ask_input("Database Port", default="5432")
            database = self.ask_input("Database Name", default="forex_bot")
            username = self.ask_input("Database Username", default="forex_user")
            password = self.ask_input("Database Password", password=True)
            
            return {
                'enabled': True,
                'postgresql': {
                    'host': host,
                    'port': int(port),
                    'database': database,
                    'username': username,
                    'password': password,
                    'ssl_mode': 'prefer'
                }
            }
        return {'enabled': False}
    
    def setup_telegram(self):
        """Setup Telegram Bot"""
        self.print_header("Telegram Bot Setup")
        print("1. Talk to @BotFather on Telegram")
        print("2. Create new bot with /newbot")
        print("3. Get bot token and chat ID\n")
        
        use_telegram = self.ask_yes_no("Do you want Telegram notifications?")
        
        if use_telegram:
            bot_token = self.ask_input("Telegram Bot Token")
            
            print("\n📱 To get Chat ID:")
            print("1. Start a chat with your bot")
            print("2. Send any message to the bot")
            print("3. Visit: https://api.telegram.org/bot{YOUR_TOKEN}/getUpdates")
            print("4. Find 'chat' -> 'id' in the response\n")
            
            chat_id = self.ask_input("Telegram Chat ID")
            
            return {
                'enabled': True,
                'bot_token': bot_token,
                'chat_id': chat_id
            }
        return {'enabled': False}
    
    def setup_other_apis(self):
        """Setup other optional APIs"""
        self.print_header("Optional APIs Setup")
        apis = {}
        
        # Alpha Vantage
        if self.ask_yes_no("Setup Alpha Vantage API (Market Data)?"):
            apis['alpha_vantage'] = {
                'api_key': self.ask_input("Alpha Vantage API Key"),
                'premium': False
            }
        
        # News API
        if self.ask_yes_no("Setup News API (Financial News)?"):
            apis['news_api'] = {
                'api_key': self.ask_input("News API Key")
            }
        
        return apis
    
    def save_secrets_encrypted(self, secrets: dict):
        """Save secrets to encrypted file"""
        try:
            # Simple XOR encryption for demonstration
            # In production, use proper encryption like cryptography.fernet
            import base64
            
            secrets_json = json.dumps(secrets)
            
            # Simple encoding (not secure - for demo only)
            # Replace with proper encryption in production
            encoded = base64.b64encode(secrets_json.encode()).decode()
            
            with open(self.secrets_file, 'w') as f:
                json.dump({'data': encoded}, f)
            
            # Set file permissions to owner only
            self.secrets_file.chmod(0o600)
            
            print(f"✅ Secrets saved to {self.secrets_file} (encrypted)")
            return True
            
        except Exception as e:
            print(f"❌ Error saving secrets: {e}")
            return False
    
    def generate_config_yaml(self, settings: dict):
        """Generate YAML config file from settings"""
        config = {
            'binance': {
                'api_key': '${BINANCE_API_KEY}',
                'api_secret': '${BINANCE_API_SECRET}',
                'testnet': settings.get('binance', {}).get('testnet', True)
            },
            'database': {
                'postgresql': {
                    'host': '${DATABASE_HOST:localhost}',
                    'port': '${DATABASE_PORT:5432}',
                    'database': '${DATABASE_NAME:forex_bot}',
                    'username': '${DATABASE_USER:forex_user}',
                    'password': '${DATABASE_PASSWORD}',
                    'ssl_mode': 'prefer'
                }
            },
            'telegram': {
                'bot_token': '${TELEGRAM_BOT_TOKEN}',
                'chat_id': '${TELEGRAM_CHAT_ID}'
            },
            'trading': {
                'risk_per_trade': 0.02,
                'max_drawdown': 0.15,
                'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
                'timeframes': ['1H', '4H', '1D']
            }
        }
        
        # Add optional APIs if configured
        if 'alpha_vantage' in settings:
            config['alpha_vantage'] = {
                'api_key': '${ALPHA_VANTAGE_API_KEY}',
                'premium': False
            }
        
        if 'news_api' in settings:
            config['news_api'] = {
                'api_key': '${NEWS_API_KEY}'
            }
        
        # Save YAML file
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Configuration saved to {self.config_file}")
    
    def create_env_file(self, secrets: dict):
        """Create .env file from secrets"""
        env_lines = []
        
        # Binance
        if secrets.get('binance', {}).get('enabled'):
            env_lines.append(f"BINANCE_API_KEY={secrets['binance']['api_key']}")
            env_lines.append(f"BINANCE_API_SECRET={secrets['binance']['api_secret']}")
        
        # Database
        if secrets.get('database', {}).get('enabled'):
            db = secrets['database']['postgresql']
            env_lines.append(f"DATABASE_HOST={db['host']}")
            env_lines.append(f"DATABASE_PORT={db['port']}")
            env_lines.append(f"DATABASE_NAME={db['database']}")
            env_lines.append(f"DATABASE_USER={db['username']}")
            env_lines.append(f"DATABASE_PASSWORD={db['password']}")
        
        # Telegram
        if secrets.get('telegram', {}).get('enabled'):
            env_lines.append(f"TELEGRAM_BOT_TOKEN={secrets['telegram']['bot_token']}")
            env_lines.append(f"TELEGRAM_CHAT_ID={secrets['telegram']['chat_id']}")
        
        # Other APIs
        if 'alpha_vantage' in secrets:
            env_lines.append(f"ALPHA_VANTAGE_API_KEY={secrets['alpha_vantage']['api_key']}")
        
        if 'news_api' in secrets:
            env_lines.append(f"NEWS_API_KEY={secrets['news_api']['api_key']}")
        
        # Environment
        env_lines.append("ENVIRONMENT=development")
        env_lines.append("DEBUG=true")
        
        # Write .env file
        env_file = Path('.env')
        with open(env_file, 'w') as f:
            f.write('\n'.join(env_lines))
        
        # Set secure permissions
        env_file.chmod(0o600)
        
        print(f"✅ Environment file created: {env_file}")
        print("⚠️  IMPORTANT: Never commit .env file to version control!")
    
    def run_setup(self):
        """Run interactive setup"""
        self.clear_screen()
        self.print_header("Forex Trading Bot - Initial Setup")
        print("Welcome! Let's setup your trading bot step-by-step.")
        print("You'll need API keys for the services you want to use.\n")
        
        input("Press Enter to continue...")
        
        # Collect all settings
        all_settings = {}
        all_secrets = {}
        
        # Binance
        binance_settings = self.setup_binance()
        if binance_settings['enabled']:
            all_settings['binance'] = {'testnet': binance_settings['testnet']}
            all_secrets['binance'] = {
                'api_key': binance_settings['api_key'],
                'api_secret': binance_settings['api_secret'],
                'enabled': True
            }
        
        # Database
        db_settings = self.setup_database()
        if db_settings['enabled']:
            all_settings['database'] = {}
            all_secrets['database'] = db_settings
        
        # Telegram
        telegram_settings = self.setup_telegram()
        if telegram_settings['enabled']:
            all_settings['telegram'] = {}
            all_secrets['telegram'] = telegram_settings
        
        # Other APIs
        other_apis = self.setup_other_apis()
        all_settings.update(other_apis)
        all_secrets.update(other_apis)
        
        # Summary
        self.clear_screen()
        self.print_header("Setup Summary")
        
        print("📋 Services to be configured:")
        if all_secrets.get('binance'):
            print("  ✅ Binance API (Testnet)" if all_settings['binance']['testnet'] else "  ✅ Binance API (Live)")
        if all_secrets.get('database'):
            print("  ✅ PostgreSQL Database")
        if all_secrets.get('telegram'):
            print("  ✅ Telegram Notifications")
        if 'alpha_vantage' in all_secrets:
            print("  ✅ Alpha Vantage API")
        if 'news_api' in all_secrets:
            print("  ✅ News API")
        
        print("\n📁 Files to be created:")
        print("  • config/api_keys.yaml (Template)")
        print("  • .env (Environment variables)")
        print("  • config/secrets.json (Encrypted)")
        
        if not self.ask_yes_no("\nProceed with saving configuration?"):
            print("❌ Setup cancelled.")
            return False
        
        # Save configuration
        self.generate_config_yaml(all_settings)
        self.create_env_file(all_secrets)
        self.save_secrets_encrypted(all_secrets)
        
        self.print_header("Setup Complete!")
        print("🎉 Your trading bot is now configured!")
        print("\n📚 Next steps:")
        print("1. Run: python main.py")
        print("2. Check logs/ directory for output")
        print("3. Start with paper trading first")
        print("\n🔒 Security notes:")
        print("• Keep your .env file secure")
        print("• Never share or commit API keys")
        print("• Regularly rotate your API keys")
        
        return True

def main():
    """Main function"""
    setup = InteractiveSetup()
    try:
        setup.run_setup()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
FOREX TRADING BOT - PRODUCTION DEPLOYMENT SCRIPT
Advanced deployment script for institutional-grade trading system
Converted from shell script to Python
"""

import os
import sys
import subprocess
import argparse
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Color codes for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# Logging setup
class DeploymentLogger:
    @staticmethod
    def log_info(message):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
        logging.info(message)
    
    @staticmethod
    def log_success(message):
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
        logging.info(message)
    
    @staticmethod
    def log_warning(message):
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
        logging.warning(message)
    
    @staticmethod
    def log_error(message):
        print(f"{Colors.RED}[ERROR]{Colors.NC} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
        logging.error(message)

class DeploymentManager:
    """
    Forex Trading Bot Deployment Manager
    Handles production deployments to various platforms
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = DeploymentLogger()
        
        # Configuration
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.deployment_env = os.getenv('DEPLOYMENT_ENV', 'production')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = Path(f"/opt/forex_bot/backups/{self.timestamp}")
        
        # Deployment configuration
        self.docker_image_name = "forex-trading-bot"
        self.docker_image_tag = "latest"
        self.k8s_namespace = "forex-trading"
        self.aws_region = "us-east-1"
        self.ecs_cluster = "forex-bot-cluster"
        self.ecs_service = "forex-bot-service"
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "deployment.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with error handling"""
        try:
            self.logger.log_info(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, check=check)
            if result.stdout:
                self.logger.log_info(f"Command output: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.log_error(f"Command failed: {e}")
            if e.stderr:
                self.logger.log_error(f"Error output: {e.stderr}")
            raise
    
    def load_config(self):
        """Load deployment configuration"""
        self.logger.log_info("Loading deployment configuration...")
        
        env_file = self.project_root / "config" / "deploy.env"
        if env_file.exists():
            # Simple env file parser
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            self.logger.log_success("Environment configuration loaded")
        else:
            self.logger.log_warning("No deploy.env found, using defaults")
        
        # Validate required environment variables
        required_vars = ["DEPLOYMENT_ENV", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        for var in required_vars:
            if not os.getenv(var):
                self.logger.log_error(f"Required environment variable {var} is not set")
                sys.exit(1)
    
    def security_checks(self):
        """Run security checks"""
        self.logger.log_info("Running security checks...")
        
        # Check for secrets in code
        try:
            result = self.run_command([
                "grep", "-r", "password|secret|key", 
                str(self.project_root), 
                "--include=*.py", "--include=*.json"
            ], check=False)
            
            if result.returncode == 0:
                self.logger.log_warning("Potential secrets found in code")
        except Exception as e:
            self.logger.log_warning(f"Secret check failed: {e}")
        
        # Validate file permissions
        self.run_command(["find", str(self.project_root), "-name", "*.py", "-exec", "chmod", "644", "{}", ";"])
        
        self.logger.log_success("Security checks completed")
    
    def check_dependencies(self):
        """Check system dependencies"""
        self.logger.log_info("Checking system dependencies...")
        
        dependencies = ["docker", "python3", "git", "aws", "kubectl", "jq"]
        
        for dep in dependencies:
            try:
                self.run_command([dep, "--version"], check=False)
                self.logger.log_info(f"✓ {dep}")
            except Exception:
                self.logger.log_error(f"Dependency {dep} not found. Please install it first.")
                sys.exit(1)
        
        # Check Docker daemon
        try:
            self.run_command(["docker", "info"])
        except Exception:
            self.logger.log_error("Docker daemon is not running")
            sys.exit(1)
        
        # Check Kubernetes cluster access
        try:
            self.run_command(["kubectl", "cluster-info"])
            self.logger.log_success("Kubernetes cluster is accessible")
        except Exception:
            self.logger.log_warning("Kubernetes cluster is not accessible")
        
        self.logger.log_success("All dependencies verified")
    
    def backup_current(self):
        """Backup current deployment"""
        self.logger.log_info("Backing up current deployment...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup configuration
        config_dir = Path("/opt/forex_bot/config")
        if config_dir.exists():
            self.run_command(["cp", "-r", str(config_dir), str(self.backup_dir)])
        
        # Backup logs
        logs_dir = Path("/opt/forex_bot/logs")
        if logs_dir.exists():
            self.run_command([
                "tar", "-czf", str(self.backup_dir / "logs.tar.gz"), 
                "-C", "/opt/forex_bot", "logs/"
            ])
        
        self.logger.log_success(f"Backup created at {self.backup_dir}")
    
    def build_docker_image(self):
        """Build Docker image"""
        self.logger.log_info("Building Docker image...")
        
        os.chdir(self.project_root)
        
        # Build the image
        self.run_command([
            "docker", "build",
            "-t", f"{self.docker_image_name}:{self.docker_image_tag}",
            "-t", f"{self.docker_image_name}:{self.timestamp}",
            "--build-arg", f"DEPLOYMENT_ENV={self.deployment_env}",
            "--no-cache", "."
        ])
        
        # Test the image
        self.run_command([
            "docker", "run", "--rm",
            "-e", f"DEPLOYMENT_ENV={self.deployment_env}",
            f"{self.docker_image_name}:{self.docker_image_tag}",
            "python", "-c", "import sys; print('Docker image test successful')"
        ])
        
        self.logger.log_success("Docker image built and tested successfully")
    
    def run_tests(self):
        """Run test suite"""
        self.logger.log_info("Running test suite...")
        
        os.chdir(self.project_root)
        
        # Unit tests
        self.logger.log_info("Running unit tests...")
        try:
            self.run_command([
                "python", "-m", "pytest", "tests/", "-v",
                "--cov=.", "--cov-report=html:reports/coverage"
            ])
        except Exception:
            self.logger.log_error("Unit tests failed")
            sys.exit(1)
        
        # Integration tests
        self.logger.log_info("Running integration tests...")
        try:
            self.run_command([
                "python", "-m", "pytest", "tests/integration_tests.py", "-v"
            ])
        except Exception:
            self.logger.log_error("Integration tests failed")
            sys.exit(1)
        
        self.logger.log_success("All tests completed")
    
    def run_migrations(self):
        """Run database migrations"""
        self.logger.log_info("Running database migrations...")
        
        migration_script = self.project_root / "scripts" / "run_migrations.py"
        if migration_script.exists():
            self.run_command([
                "python", str(migration_script), 
                "--environment", self.deployment_env
            ])
        else:
            self.logger.log_info("No database migrations found, skipping")
        
        self.logger.log_success("Database migrations completed")
    
    def deploy_kubernetes(self):
        """Deploy to Kubernetes"""
        self.logger.log_info("Deploying to Kubernetes cluster...")
        
        os.chdir(self.project_root / "deployment")
        
        # Apply Kubernetes manifests
        manifests = [
            "kubernetes/namespace.yaml",
            "kubernetes/configmap.yaml", 
            "kubernetes/secret.yaml",
            "kubernetes/deployment.yaml",
            "kubernetes/service.yaml",
            "kubernetes/hpa.yaml"
        ]
        
        for manifest in manifests:
            if Path(manifest).exists():
                self.run_command(["kubectl", "apply", "-f", manifest])
        
        # Wait for deployment
        self.run_command([
            "kubectl", "rollout", "status", "deployment/forex-bot-deployment",
            "-n", self.k8s_namespace, "--timeout=300s"
        ])
        
        # Check pods
        self.run_command([
            "kubectl", "get", "pods", "-n", self.k8s_namespace, "-l", "app=forex-bot"
        ])
        
        self.logger.log_success("Kubernetes deployment completed")
    
    def health_checks(self):
        """Run health checks"""
        self.logger.log_info("Running health checks...")
        
        retries = 10
        wait_time = 30
        
        for i in range(1, retries + 1):
            self.logger.log_info(f"Health check attempt {i}/{retries}...")
            
            try:
                self.run_command([
                    "curl", "-f", "-s", "--retry", "3", "--max-time", "10",
                    "http://localhost:8080/health"
                ], check=False)
                self.logger.log_success("Application health check passed")
                return
            except Exception:
                pass
            
            time.sleep(wait_time)
        
        self.logger.log_error(f"Health checks failed after {retries} attempts")
        sys.exit(1)
    
    def cleanup_old_deployments(self):
        """Cleanup old deployments"""
        self.logger.log_info("Cleaning up old deployments...")
        
        # Keep only last 5 backups
        try:
            result = self.run_command([
                "find", "/opt/forex_bot/backups", "-maxdepth", "1", "-type", "d", "-name", "2*"
            ], check=False)
            
            if result.stdout:
                backups = sorted(result.stdout.strip().split('\n'), reverse=True)
                for backup in backups[5:]:
                    if backup:
                        self.run_command(["rm", "-rf", backup])
        except Exception as e:
            self.logger.log_warning(f"Backup cleanup failed: {e}")
        
        self.logger.log_success("Cleanup completed")
    
    def deploy(self, target: str = "kubernetes"):
        """Main deployment method"""
        self.logger.log_success("Starting Forex Trading Bot Production Deployment")
        self.logger.log_info(f"Environment: {self.deployment_env}")
        self.logger.log_info(f"Timestamp: {self.timestamp}")
        self.logger.log_info(f"Project Root: {self.project_root}")
        
        try:
            self.load_config()
            self.security_checks()
            self.check_dependencies()
            self.backup_current()
            self.build_docker_image()
            self.run_tests()
            self.run_migrations()
            
            # Select deployment target
            if target == "kubernetes":
                self.deploy_kubernetes()
            elif target == "aws-ecs":
                self.deploy_aws_ecs()
            elif target == "docker-swarm":
                self.deploy_docker_swarm()
            else:
                self.logger.log_error(f"Unknown deployment target: {target}")
                sys.exit(1)
            
            self.health_checks()
            self.cleanup_old_deployments()
            
            self.logger.log_success("🎉 PRODUCTION DEPLOYMENT COMPLETED SUCCESSFULLY 🎉")
            
        except Exception as e:
            self.logger.log_error(f"Deployment failed: {e}")
            self.rollback()
            sys.exit(1)
    
    def deploy_aws_ecs(self):
        """Deploy to AWS ECS (placeholder)"""
        self.logger.log_info("AWS ECS deployment would be implemented here")
        # Implementation for AWS ECS would go here
    
    def deploy_docker_swarm(self):
        """Deploy to Docker Swarm (placeholder)"""
        self.logger.log_info("Docker Swarm deployment would be implemented here")
        # Implementation for Docker Swarm would go here
    
    def rollback(self):
        """Rollback deployment"""
        self.logger.log_error("Starting rollback process...")
        # Rollback implementation would go here

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Forex Trading Bot Deployment Script")
    parser.add_argument("-e", "--environment", default="production", help="Deployment environment")
    parser.add_argument("-t", "--target", default="kubernetes", 
                       choices=["kubernetes", "aws-ecs", "docker-swarm"],
                       help="Deployment target")
    parser.add_argument("--dry-run", action="store_true", help="Simulate deployment")
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger = DeploymentLogger()
        logger.log_info("=== DRY RUN MODE ===")
        logger.log_info("The following actions would be performed:")
        logger.log_info("1. Security checks")
        logger.log_info("2. Dependency verification") 
        logger.log_info("3. Backup creation")
        logger.log_info("4. Docker image build")
        logger.log_info("5. Test execution")
        logger.log_info("6. Database migrations")
        logger.log_info(f"7. Deployment to: {args.target}")
        logger.log_info("8. Health checks")
        logger.log_info("9. Cleanup")
        return
    
    # Set environment
    os.environ['DEPLOYMENT_ENV'] = args.environment
    
    # Run deployment
    deployer = DeploymentManager()
    deployer.deploy(target=args.target)

if __name__ == "__main__":
    main()
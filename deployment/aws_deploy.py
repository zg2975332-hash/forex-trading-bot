"""
AWS Deployment Manager for FOREX TRADING BOT
Advanced cloud deployment with auto-scaling, monitoring, and security
"""

import base64
import logging
import asyncio
import time
import boto3
import json
import yaml
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import shutil
from pathlib import Path
import hashlib
from datetime import datetime, timedelta
import zipfile
import io
import paramiko
from botocore.exceptions import ClientError, BotoCoreError
import requests

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"

class ServiceType(Enum):
    TRADING_BOT = "trading_bot"
    DATA_PIPELINE = "data_pipeline"
    API_GATEWAY = "api_gateway"
    MONITORING = "monitoring"
    DATABASE = "database"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    environment: DeploymentEnvironment
    region: str = "us-east-1"
    vpc_id: str = None
    subnet_ids: List[str] = field(default_factory=list)
    instance_type: str = "t3.medium"
    min_instances: int = 1
    max_instances: int = 10
    desired_instances: int = 2
    key_pair_name: str = "forex-bot-key"
    security_groups: List[str] = field(default_factory=list)
    docker_image: str = "forex-trading-bot:latest"
    docker_registry: str = "ECR"  # ECR, DockerHub, Private
    enable_ssl: bool = True
    domain_name: str = None
    monitoring_enabled: bool = True
    backup_enabled: bool = True
    auto_scaling_enabled: bool = True

@dataclass
class DeploymentResult:
    """Deployment result"""
    status: DeploymentStatus
    service_name: str
    environment: DeploymentEnvironment
    deployment_id: str
    start_time: float
    end_time: float = None
    resources_created: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    cost_estimate: float = 0.0

@dataclass
class CloudFormationTemplate:
    """CloudFormation template definition"""
    template_name: str
    template_body: Dict[str, Any]
    parameters: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=lambda: ['CAPABILITY_IAM'])

class AWSDeployer:
    """
    Advanced AWS deployment manager for Forex Trading Bot
    Handles infrastructure provisioning, deployment, and monitoring
    """
    
    def __init__(self, config: DeploymentConfig, aws_profile: str = None):
        self.config = config
        self.aws_profile = aws_profile
        
        # AWS clients
        self.ec2_client = None
        self.ecs_client = None
        self.ecr_client = None
        self.elbv2_client = None
        self.autoscaling_client = None
        self.cloudformation_client = None
        self.s3_client = None
        self.cloudwatch_client = None
        self.rds_client = None
        self.elasticache_client = None
        self.iam_client = None
        
        # Deployment state
        self.deployment_history = []
        self.resource_map = {}
        self.active_deployments = {}
        
        # Initialize AWS clients
        self._initialize_aws_clients()
        
        # Infrastructure templates
        self.templates = self._load_infrastructure_templates()
        
        logger.info(f"AWSDeployer initialized for {config.environment.value}")
    
    def initialize_system(self):
        """Initialize deployment system"""
        try:
            self.logger.info("Initializing AWS deployment system...")
        
            # Check AWS credentials
            try:
                import boto3
                sts = boto3.client('sts')
                sts.get_caller_identity()
                self.logger.info("AWS credentials verified")
                return True
            except Exception as e:
                self.logger.warning(f"AWS credentials not available: {e}")
                self.logger.info("Running in local mode without AWS deployment")
                return False
            
        except Exception as e:
            self.logger.error(f"Deployment system initialization failed: {e}")
            return False
    
    def _initialize_aws_clients(self):
        """Initialize AWS service clients"""
        try:
            session = boto3.Session(profile_name=self.aws_profile, region_name=self.config.region)
            
            self.ec2_client = session.client('ec2')
            self.ecs_client = session.client('ecs')
            self.ecr_client = session.client('ecr')
            self.elbv2_client = session.client('elbv2')
            self.autoscaling_client = session.client('autoscaling')
            self.cloudformation_client = session.client('cloudformation')
            self.s3_client = session.client('s3')
            self.cloudwatch_client = session.client('cloudwatch')
            self.rds_client = session.client('rds')
            self.elasticache_client = session.client('elasticache')
            self.iam_client = session.client('iam')
            
            # Verify credentials
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS identity: {identity['Arn']}")
            
        except Exception as e:
            logger.error(f"AWS client initialization failed: {e}")
            raise

    def _load_infrastructure_templates(self) -> Dict[str, CloudFormationTemplate]:
        """Load CloudFormation templates for different services"""
        templates = {}
        
        # VPC Template
        templates['vpc'] = CloudFormationTemplate(
            template_name="ForexBotVPC",
            template_body=self._create_vpc_template(),
            parameters={
                'Environment': self.config.environment.value,
                'VPCCIDR': '10.0.0.0/16'
            }
        )
        
        # ECS Cluster Template
        templates['ecs_cluster'] = CloudFormationTemplate(
            template_name="ForexBotECSCluster",
            template_body=self._create_ecs_cluster_template(),
            parameters={
                'Environment': self.config.environment.value,
                'InstanceType': self.config.instance_type,
                'MinSize': str(self.config.min_instances),
                'MaxSize': str(self.config.max_instances),
                'DesiredCapacity': str(self.config.desired_instances),
                'KeyName': self.config.key_pair_name
            }
        )
        
        # Load Balancer Template
        templates['load_balancer'] = CloudFormationTemplate(
            template_name="ForexBotLoadBalancer",
            template_body=self._create_load_balancer_template(),
            parameters={
                'Environment': self.config.environment.value,
                'CertificateArn': self._get_certificate_arn() if self.config.enable_ssl else ''
            }
        )
        
        # RDS Database Template
        templates['database'] = CloudFormationTemplate(
            template_name="ForexBotDatabase",
            template_body=self._create_database_template(),
            parameters={
                'Environment': self.config.environment.value,
                'DBInstanceClass': 'db.t3.medium',
                'DBName': 'forex_bot'
            }
        )
        
        # Redis Cache Template
        templates['redis'] = CloudFormationTemplate(
            template_name="ForexBotRedis",
            template_body=self._create_redis_template(),
            parameters={
                'Environment': self.config.environment.value,
                'CacheNodeType': 'cache.t3.medium'
            }
        )
        
        return templates

    async def deploy_infrastructure(self) -> DeploymentResult:
        """
        Deploy complete infrastructure for Forex Trading Bot
        """
        deployment_id = f"deploy-{int(time.time())}"
        start_time = time.time()
        
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            service_name="full_infrastructure",
            environment=self.config.environment,
            deployment_id=deployment_id,
            start_time=start_time
        )
        
        try:
            logger.info(f"Starting infrastructure deployment: {deployment_id}")
            
            # Step 1: Create VPC and networking
            vpc_result = await self._deploy_vpc()
            if not vpc_result:
                raise Exception("VPC deployment failed")
            result.resources_created['vpc'] = vpc_result
            
            # Step 2: Create ECR repository for Docker images
            ecr_result = await self._create_ecr_repository()
            if not ecr_result:
                raise Exception("ECR repository creation failed")
            result.resources_created['ecr'] = ecr_result
            
            # Step 3: Deploy ECS cluster
            ecs_result = await self._deploy_ecs_cluster()
            if not ecs_result:
                raise Exception("ECS cluster deployment failed")
            result.resources_created['ecs'] = ecs_result
            
            # Step 4: Deploy load balancer
            lb_result = await self._deploy_load_balancer()
            if not lb_result:
                raise Exception("Load balancer deployment failed")
            result.resources_created['load_balancer'] = lb_result
            
            # Step 5: Deploy database
            db_result = await self._deploy_database()
            if not db_result:
                raise Exception("Database deployment failed")
            result.resources_created['database'] = db_result
            
            # Step 6: Deploy Redis cache
            redis_result = await self._deploy_redis()
            if not redis_result:
                raise Exception("Redis deployment failed")
            result.resources_created['redis'] = redis_result
            
            # Step 7: Configure monitoring and alerts
            monitoring_result = await self._setup_monitoring()
            if monitoring_result:
                result.resources_created['monitoring'] = monitoring_result
            
            # Step 8: Update deployment status
            result.status = DeploymentStatus.SUCCESS
            result.end_time = time.time()
            result.cost_estimate = await self._estimate_monthly_cost()
            
            logger.info(f"Infrastructure deployment completed: {deployment_id}")
            
        except Exception as e:
            error_msg = f"Infrastructure deployment failed: {str(e)}"
            logger.error(error_msg)
            result.status = DeploymentStatus.FAILED
            result.end_time = time.time()
            result.errors.append(error_msg)
            
            # Attempt rollback
            await self._rollback_deployment(result)
        
        finally:
            self.deployment_history.append(result)
            return result

    async def _deploy_vpc(self) -> List[str]:
        """Deploy VPC and networking infrastructure"""
        try:
            template = self.templates['vpc']
            stack_name = f"forex-bot-vpc-{self.config.environment.value}"
            
            logger.info(f"Deploying VPC stack: {stack_name}")
            
            response = self.cloudformation_client.create_stack(
                StackName=stack_name,
                TemplateBody=json.dumps(template.template_body),
                Parameters=[
                    {'ParameterKey': key, 'ParameterValue': value}
                    for key, value in template.parameters.items()
                ],
                Capabilities=template.capabilities,
                Tags=[
                    {'Key': 'Environment', 'Value': self.config.environment.value},
                    {'Key': 'Project', 'Value': 'ForexTradingBot'},
                    {'Key': 'ManagedBy', 'Value': 'AWSDeployer'}
                ]
            )
            
            # Wait for stack creation
            await self._wait_for_stack_completion(stack_name)
            
            # Get stack outputs
            stack_outputs = self._get_stack_outputs(stack_name)
            vpc_id = stack_outputs.get('VPCId')
            subnet_ids = [
                stack_outputs.get('PublicSubnet1'),
                stack_outputs.get('PublicSubnet2'),
                stack_outputs.get('PrivateSubnet1'),
                stack_outputs.get('PrivateSubnet2')
            ]
            
            # Update config with created resources
            self.config.vpc_id = vpc_id
            self.config.subnet_ids = [sid for sid in subnet_ids if sid]
            
            logger.info(f"VPC deployed: {vpc_id}")
            return [vpc_id] + self.config.subnet_ids
            
        except Exception as e:
            logger.error(f"VPC deployment failed: {e}")
            return []

    async def _create_ecr_repository(self) -> List[str]:
        """Create ECR repository for Docker images"""
        try:
            repository_name = f"forex-trading-bot/{self.config.environment.value}"
            
            # Check if repository already exists
            try:
                response = self.ecr_client.describe_repositories(
                    repositoryNames=[repository_name]
                )
                repository_uri = response['repositories'][0]['repositoryUri']
                logger.info(f"ECR repository already exists: {repository_uri}")
                return [repository_uri]
                
            except ClientError:
                # Create new repository
                response = self.ecr_client.create_repository(
                    repositoryName=repository_name,
                    imageTagMutability='MUTABLE',
                    imageScanningConfiguration={
                        'scanOnPush': True
                    },
                    tags=[
                        {'Key': 'Environment', 'Value': self.config.environment.value},
                        {'Key': 'Project', 'Value': 'ForexTradingBot'}
                    ]
                )
                
                repository_uri = response['repository']['repositoryUri']
                logger.info(f"ECR repository created: {repository_uri}")
                
                # Get login token for Docker
                login_token = self.ecr_client.get_authorization_token()
                auth_data = login_token['authorizationData'][0]
                
                return [repository_uri, auth_data['authorizationToken']]
                
        except Exception as e:
            logger.error(f"ECR repository creation failed: {e}")
            return []

    async def _deploy_ecs_cluster(self) -> List[str]:
        """Deploy ECS cluster for container orchestration"""
        try:
            template = self.templates['ecs_cluster']
            stack_name = f"forex-bot-ecs-{self.config.environment.value}"
            
            # Update template parameters with VPC info
            template.parameters.update({
                'VpcId': self.config.vpc_id,
                'SubnetIds': ','.join(self.config.subnet_ids[:2])  # Use first two subnets
            })
            
            logger.info(f"Deploying ECS cluster stack: {stack_name}")
            
            response = self.cloudformation_client.create_stack(
                StackName=stack_name,
                TemplateBody=json.dumps(template.template_body),
                Parameters=[
                    {'ParameterKey': key, 'ParameterValue': value}
                    for key, value in template.parameters.items()
                ],
                Capabilities=template.capabilities,
                Tags=[
                    {'Key': 'Environment', 'Value': self.config.environment.value},
                    {'Key': 'Project', 'Value': 'ForexTradingBot'}
                ]
            )
            
            # Wait for stack creation
            await self._wait_for_stack_completion(stack_name)
            
            # Get stack outputs
            stack_outputs = self._get_stack_outputs(stack_name)
            cluster_name = stack_outputs.get('ECSClusterName')
            auto_scaling_group = stack_outputs.get('AutoScalingGroup')
            
            logger.info(f"ECS cluster deployed: {cluster_name}")
            return [cluster_name, auto_scaling_group]
            
        except Exception as e:
            logger.error(f"ECS cluster deployment failed: {e}")
            return []

    async def _deploy_load_balancer(self) -> List[str]:
        """Deploy Application Load Balancer"""
        try:
            template = self.templates['load_balancer']
            stack_name = f"forex-bot-alb-{self.config.environment.value}"
            
            # Update template parameters
            template.parameters.update({
                'VpcId': self.config.vpc_id,
                'SubnetIds': ','.join(self.config.subnet_ids[:2])
            })
            
            logger.info(f"Deploying Load Balancer stack: {stack_name}")
            
            response = self.cloudformation_client.create_stack(
                StackName=stack_name,
                TemplateBody=json.dumps(template.template_body),
                Parameters=[
                    {'ParameterKey': key, 'ParameterValue': value}
                    for key, value in template.parameters.items()
                ],
                Capabilities=template.capabilities
            )
            
            await self._wait_for_stack_completion(stack_name)
            
            stack_outputs = self._get_stack_outputs(stack_name)
            load_balancer_dns = stack_outputs.get('LoadBalancerDNS')
            target_group_arn = stack_outputs.get('TargetGroupARN')
            
            logger.info(f"Load Balancer deployed: {load_balancer_dns}")
            return [load_balancer_dns, target_group_arn]
            
        except Exception as e:
            logger.error(f"Load Balancer deployment failed: {e}")
            return []

    async def _deploy_database(self) -> List[str]:
        """Deploy RDS PostgreSQL database"""
        try:
            template = self.templates['database']
            stack_name = f"forex-bot-db-{self.config.environment.value}"
            
            # Update template parameters
            template.parameters.update({
                'VpcId': self.config.vpc_id,
                'SubnetIds': ','.join(self.config.subnet_ids[2:])  # Use private subnets
            })
            
            logger.info(f"Deploying Database stack: {stack_name}")
            
            response = self.cloudformation_client.create_stack(
                StackName=stack_name,
                TemplateBody=json.dumps(template.template_body),
                Parameters=[
                    {'ParameterKey': key, 'ParameterValue': value}
                    for key, value in template.parameters.items()
                ],
                Capabilities=template.capabilities
            )
            
            await self._wait_for_stack_completion(stack_name)
            
            stack_outputs = self._get_stack_outputs(stack_name)
            db_endpoint = stack_outputs.get('DBEndpoint')
            db_secret_arn = stack_outputs.get('DBSecretARN')
            
            logger.info(f"Database deployed: {db_endpoint}")
            return [db_endpoint, db_secret_arn]
            
        except Exception as e:
            logger.error(f"Database deployment failed: {e}")
            return []

    async def _deploy_redis(self) -> List[str]:
        """Deploy Redis ElastiCache cluster"""
        try:
            template = self.templates['redis']
            stack_name = f"forex-bot-redis-{self.config.environment.value}"
            
            # Update template parameters
            template.parameters.update({
                'VpcId': self.config.vpc_id,
                'SubnetIds': ','.join(self.config.subnet_ids[2:])  # Use private subnets
            })
            
            logger.info(f"Deploying Redis stack: {stack_name}")
            
            response = self.cloudformation_client.create_stack(
                StackName=stack_name,
                TemplateBody=json.dumps(template.template_body),
                Parameters=[
                    {'ParameterKey': key, 'ParameterValue': value}
                    for key, value in template.parameters.items()
                ],
                Capabilities=template.capabilities
            )
            
            await self._wait_for_stack_completion(stack_name)
            
            stack_outputs = self._get_stack_outputs(stack_name)
            redis_endpoint = stack_outputs.get('RedisEndpoint')
            
            logger.info(f"Redis deployed: {redis_endpoint}")
            return [redis_endpoint]
            
        except Exception as e:
            logger.error(f"Redis deployment failed: {e}")
            return []

    async def _setup_monitoring(self) -> List[str]:
        """Setup CloudWatch monitoring and alerts"""
        try:
            # Create CloudWatch dashboard
            dashboard_name = f"forex-bot-{self.config.environment.value}"
            
            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                ["AWS/ECS", "CPUUtilization", "ServiceName", "forex-bot-service"],
                                [".", "MemoryUtilization", ".", "."]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config.region,
                            "title": "ECS Service Metrics"
                        }
                    },
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", "forex-bot-db"],
                                [".", "DatabaseConnections", ".", "."]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config.region,
                            "title": "RDS Metrics"
                        }
                    }
                ]
            }
            
            self.cloudwatch_client.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            # Create alarms
            alarm_names = []
            
            # ECS CPU alarm
            cpu_alarm_name = f"forex-bot-ecs-cpu-{self.config.environment.value}"
            self.cloudwatch_client.put_metric_alarm(
                AlarmName=cpu_alarm_name,
                AlarmDescription='High CPU utilization for Forex Bot',
                MetricName='CPUUtilization',
                Namespace='AWS/ECS',
                Statistic='Average',
                Dimensions=[
                    {
                        'Name': 'ServiceName',
                        'Value': 'forex-bot-service'
                    },
                    {
                        'Name': 'ClusterName',
                        'Value': f"forex-bot-cluster-{self.config.environment.value}"
                    }
                ],
                Period=300,
                EvaluationPeriods=2,
                Threshold=80.0,
                ComparisonOperator='GreaterThanThreshold',
                AlarmActions=[],  # Would add SNS topics for notifications
                OKActions=[]
            )
            alarm_names.append(cpu_alarm_name)
            
            logger.info(f"Monitoring setup completed: {dashboard_name}")
            return [dashboard_name] + alarm_names
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return []

    async def deploy_trading_bot(self, docker_image: str = None) -> DeploymentResult:
        """Deploy the Forex Trading Bot application"""
        deployment_id = f"bot-deploy-{int(time.time())}"
        start_time = time.time()
        
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            service_name="trading_bot",
            environment=self.config.environment,
            deployment_id=deployment_id,
            start_time=start_time
        )
        
        try:
            logger.info(f"Starting Trading Bot deployment: {deployment_id}")
            
            # Step 1: Build and push Docker image
            image_uri = await self._build_and_push_docker_image(docker_image)
            if not image_uri:
                raise Exception("Docker image build/push failed")
            result.outputs['docker_image'] = image_uri
            
            # Step 2: Create ECS task definition
            task_definition_arn = await self._create_task_definition(image_uri)
            if not task_definition_arn:
                raise Exception("Task definition creation failed")
            result.outputs['task_definition'] = task_definition_arn
            
            # Step 3: Create ECS service
            service_arn = await self._create_ecs_service(task_definition_arn)
            if not service_arn:
                raise Exception("ECS service creation failed")
            result.outputs['service'] = service_arn
            
            # Step 4: Wait for service stability
            await self._wait_for_service_stability(service_arn)
            
            # Step 5: Run health checks
            health_status = await self._run_health_checks()
            if not health_status:
                raise Exception("Health checks failed")
            
            result.status = DeploymentStatus.SUCCESS
            result.end_time = time.time()
            
            logger.info(f"Trading Bot deployment completed: {deployment_id}")
            
        except Exception as e:
            error_msg = f"Trading Bot deployment failed: {str(e)}"
            logger.error(error_msg)
            result.status = DeploymentStatus.FAILED
            result.end_time = time.time()
            result.errors.append(error_msg)
        
        finally:
            self.deployment_history.append(result)
            return result

    async def _build_and_push_docker_image(self, docker_image: str = None) -> str:
        """Build and push Docker image to ECR"""
        try:
            if not docker_image:
                docker_image = self.config.docker_image
            
            # Get ECR repository URI
            repository_name = f"forex-trading-bot/{self.config.environment.value}"
            response = self.ecr_client.describe_repositories(
                repositoryNames=[repository_name]
            )
            repository_uri = response['repositories'][0]['repositoryUri']
            
            # Build Docker image
            build_result = subprocess.run([
                'docker', 'build', 
                '-t', f"{repository_uri}:latest",
                '-t', f"{repository_uri}:{datetime.now().strftime('%Y%m%d%H%M%S')}",
                '.'
            ], capture_output=True, text=True)
            
            if build_result.returncode != 0:
                logger.error(f"Docker build failed: {build_result.stderr}")
                return None
            
            # Get ECR login token
            login_token = self.ecr_client.get_authorization_token()
            auth_data = login_token['authorizationData'][0]
            token = base64.b64decode(auth_data['authorizationToken']).decode()
            username, password = token.split(':')
            registry = auth_data['proxyEndpoint']
            
            # Login to ECR
            login_result = subprocess.run([
                'docker', 'login', 
                '-u', username,
                '-p', password,
                registry
            ], capture_output=True, text=True)
            
            if login_result.returncode != 0:
                logger.error(f"ECR login failed: {login_result.stderr}")
                return None
            
            # Push image to ECR
            push_result = subprocess.run([
                'docker', 'push', f"{repository_uri}:latest"
            ], capture_output=True, text=True)
            
            if push_result.returncode != 0:
                logger.error(f"Docker push failed: {push_result.stderr}")
                return None
            
            logger.info(f"Docker image pushed: {repository_uri}:latest")
            return f"{repository_uri}:latest"
            
        except Exception as e:
            logger.error(f"Docker image build/push failed: {e}")
            return None

    async def _create_task_definition(self, image_uri: str) -> str:
        """Create ECS task definition for Trading Bot"""
        try:
            task_definition = {
                'family': f"forex-trading-bot-{self.config.environment.value}",
                'networkMode': 'awsvpc',
                'requiresCompatibilities': ['EC2'],
                'cpu': '1024',
                'memory': '2048',
                'executionRoleArn': self._get_or_create_task_execution_role(),
                'taskRoleArn': self._get_or_create_task_role(),
                'containerDefinitions': [
                    {
                        'name': 'forex-trading-bot',
                        'image': image_uri,
                        'essential': True,
                        'portMappings': [
                            {
                                'containerPort': 8000,
                                'hostPort': 8000,
                                'protocol': 'tcp'
                            }
                        ],
                        'environment': [
                            {'name': 'ENVIRONMENT', 'value': self.config.environment.value},
                            {'name': 'AWS_REGION', 'value': self.config.region},
                            {'name': 'LOG_LEVEL', 'value': 'INFO'}
                        ],
                        'logConfiguration': {
                            'logDriver': 'awslogs',
                            'options': {
                                'awslogs-group': f"/ecs/forex-bot-{self.config.environment.value}",
                                'awslogs-region': self.config.region,
                                'awslogs-stream-prefix': 'ecs'
                            }
                        },
                        'healthCheck': {
                            'command': ['CMD-SHELL', 'curl -f http://localhost:8000/health || exit 1'],
                            'interval': 30,
                            'timeout': 5,
                            'retries': 3,
                            'startPeriod': 60
                        }
                    }
                ],
                'tags': [
                    {'key': 'Environment', 'value': self.config.environment.value},
                    {'key': 'Project', 'value': 'ForexTradingBot'}
                ]
            }
            
            response = self.ecs_client.register_task_definition(**task_definition)
            task_definition_arn = response['taskDefinition']['taskDefinitionArn']
            
            logger.info(f"Task definition created: {task_definition_arn}")
            return task_definition_arn
            
        except Exception as e:
            logger.error(f"Task definition creation failed: {e}")
            return None

    async def _create_ecs_service(self, task_definition_arn: str) -> str:
        """Create ECS service for Trading Bot"""
        try:
            cluster_name = f"forex-bot-cluster-{self.config.environment.value}"
            service_name = f"forex-bot-service-{self.config.environment.value}"
            
            # Get target group ARN from load balancer stack
            stack_name = f"forex-bot-alb-{self.config.environment.value}"
            stack_outputs = self._get_stack_outputs(stack_name)
            target_group_arn = stack_outputs.get('TargetGroupARN')
            
            response = self.ecs_client.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_definition_arn,
                loadBalancers=[
                    {
                        'targetGroupArn': target_group_arn,
                        'containerName': 'forex-trading-bot',
                        'containerPort': 8000
                    }
                ],
                desiredCount=self.config.desired_instances,
                launchType='EC2',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': self.config.subnet_ids,
                        'securityGroups': self.config.security_groups,
                        'assignPublicIp': 'ENABLED'
                    }
                },
                healthCheckGracePeriodSeconds=60,
                deploymentConfiguration={
                    'deploymentCircuitBreaker': {
                        'enable': True,
                        'rollback': True
                    },
                    'maximumPercent': 200,
                    'minimumHealthyPercent': 100
                },
                enableECSManagedTags=True,
                propagateTags='SERVICE'
            )
            
            service_arn = response['service']['serviceArn']
            logger.info(f"ECS service created: {service_arn}")
            return service_arn
            
        except Exception as e:
            logger.error(f"ECS service creation failed: {e}")
            return None

    async def _wait_for_service_stability(self, service_arn: str, timeout: int = 600):
        """Wait for ECS service to become stable"""
        try:
            cluster_name = f"forex-bot-cluster-{self.config.environment.value}"
            service_name = service_arn.split('/')[-1]
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = self.ecs_client.describe_services(
                    cluster=cluster_name,
                    services=[service_name]
                )
                
                service = response['services'][0]
                deployments = service['deployments']
                
                # Check if primary deployment is running and stable
                primary_deployment = next(
                    (d for d in deployments if d['status'] == 'PRIMARY'),
                    None
                )
                
                if (primary_deployment and 
                    primary_deployment['runningCount'] == primary_deployment['desiredCount'] and
                    len(deployments) == 1):  # Only primary deployment
                    logger.info("ECS service is stable")
                    return True
                
                logger.info(f"Waiting for service stability... "
                           f"Running: {primary_deployment['runningCount'] if primary_deployment else 0}/"
                           f"{primary_deployment['desiredCount'] if primary_deployment else 0}")
                
                await asyncio.sleep(30)
            
            raise Exception("Service stability timeout")
            
        except Exception as e:
            logger.error(f"Service stability check failed: {e}")
            return False

    async def _run_health_checks(self) -> bool:
        """Run health checks on deployed application"""
        try:
            # Get load balancer DNS
            stack_name = f"forex-bot-alb-{self.config.environment.value}"
            stack_outputs = self._get_stack_outputs(stack_name)
            load_balancer_dns = stack_outputs.get('LoadBalancerDNS')
            
            if not load_balancer_dns:
                logger.error("Load balancer DNS not found")
                return False
            
            health_url = f"http://{load_balancer_dns}/health"
            
            for i in range(10):  # Retry 10 times
                try:
                    response = requests.get(health_url, timeout=10)
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get('status') == 'healthy':
                            logger.info("Health check passed")
                            return True
                
                except requests.RequestException as e:
                    logger.warning(f"Health check attempt {i+1} failed: {e}")
                
                await asyncio.sleep(30)
            
            logger.error("All health checks failed")
            return False
            
        except Exception as e:
            logger.error(f"Health check execution failed: {e}")
            return False

    def _get_or_create_task_execution_role(self) -> str:
        """Get or create ECS task execution role"""
        try:
            role_name = f"forex-bot-task-execution-{self.config.environment.value}"
            
            try:
                response = self.iam_client.get_role(RoleName=role_name)
                return response['Role']['Arn']
            except ClientError:
                # Create new role
                assume_role_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }
                
                response = self.iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                    Description='ECS Task Execution Role for Forex Trading Bot',
                    Tags=[
                        {'Key': 'Environment', 'Value': self.config.environment.value},
                        {'Key': 'Project', 'Value': 'ForexTradingBot'}
                    ]
                )
                
                # Attach managed policies
                self.iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
                )
                
                return response['Role']['Arn']
                
        except Exception as e:
            logger.error(f"Task execution role creation failed: {e}")
            raise

    def _get_or_create_task_role(self) -> str:
        """Get or create ECS task role with necessary permissions"""
        try:
            role_name = f"forex-bot-task-{self.config.environment.value}"
            
            try:
                response = self.iam_client.get_role(RoleName=role_name)
                return response['Role']['Arn']
            except ClientError:
                # Create new role
                assume_role_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }
                
                response = self.iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                    Description='ECS Task Role for Forex Trading Bot',
                    Tags=[
                        {'Key': 'Environment', 'Value': self.config.environment.value},
                        {'Key': 'Project', 'Value': 'ForexTradingBot'}
                    ]
                )
                
                # Create and attach custom policy
                policy_document = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": [
                                "ec2:Describe*",
                                "elasticloadbalancing:Describe*",
                                "cloudwatch:PutMetricData",
                                "cloudwatch:GetMetricStatistics",
                                "cloudwatch:ListMetrics",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                                "logs:DescribeLogStreams",
                                "secretsmanager:GetSecretValue"
                            ],
                            "Resource": "*"
                        }
                    ]
                }
                
                policy_name = f"forex-bot-task-policy-{self.config.environment.value}"
                self.iam_client.put_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(policy_document)
                )
                
                return response['Role']['Arn']
                
        except Exception as e:
            logger.error(f"Task role creation failed: {e}")
            raise

    async def _wait_for_stack_completion(self, stack_name: str, timeout: int = 1800):
        """Wait for CloudFormation stack to complete"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                response = self.cloudformation_client.describe_stacks(StackName=stack_name)
                stack = response['Stacks'][0]
                status = stack['StackStatus']
                
                if status.endswith('_COMPLETE'):
                    logger.info(f"Stack {stack_name} completed with status: {status}")
                    return True
                elif status.endswith('_FAILED') or status.endswith('_ROLLBACK_COMPLETE'):
                    raise Exception(f"Stack {stack_name} failed with status: {status}")
                
                logger.info(f"Waiting for stack {stack_name}... Current status: {status}")
                await asyncio.sleep(30)
            
            raise Exception(f"Stack creation timeout for {stack_name}")
            
        except Exception as e:
            logger.error(f"Stack completion wait failed: {e}")
            raise

    def _get_stack_outputs(self, stack_name: str) -> Dict[str, str]:
        """Get CloudFormation stack outputs"""
        try:
            response = self.cloudformation_client.describe_stacks(StackName=stack_name)
            stack = response['Stacks'][0]
            outputs = {}
            
            for output in stack.get('Outputs', []):
                outputs[output['OutputKey']] = output['OutputValue']
            
            return outputs
            
        except Exception as e:
            logger.error(f"Failed to get stack outputs: {e}")
            return {}

    def _get_certificate_arn(self) -> str:
        """Get SSL certificate ARN for domain"""
        # This would typically use ACM to get or create a certificate
        # For now, return empty string (no SSL)
        return ""

    async def _estimate_monthly_cost(self) -> float:
        """Estimate monthly infrastructure cost"""
        try:
            # Simple cost estimation based on resource types
            cost_components = {
                'ec2_instances': self.config.desired_instances * 50,  # $50 per instance/month
                'rds_instance': 100,  # $100 for RDS
                'elasticache': 50,    # $50 for Redis
                'alb': 20,           # $20 for ALB
                'data_transfer': 10,  # $10 for data transfer
                'cloudwatch': 10,     # $10 for monitoring
            }
            
            total_cost = sum(cost_components.values())
            logger.info(f"Estimated monthly cost: ${total_cost}")
            return total_cost
            
        except Exception as e:
            logger.error(f"Cost estimation failed: {e}")
            return 0.0

    async def _rollback_deployment(self, result: DeploymentResult):
        """Rollback failed deployment"""
        try:
            logger.info(f"Starting rollback for deployment: {result.deployment_id}")
            
            # Delete CloudFormation stacks in reverse order
            stacks_to_delete = []
            
            if 'redis' in result.resources_created:
                stacks_to_delete.append(f"forex-bot-redis-{self.config.environment.value}")
            if 'database' in result.resources_created:
                stacks_to_delete.append(f"forex-bot-db-{self.config.environment.value}")
            if 'load_balancer' in result.resources_created:
                stacks_to_delete.append(f"forex-bot-alb-{self.config.environment.value}")
            if 'ecs' in result.resources_created:
                stacks_to_delete.append(f"forex-bot-ecs-{self.config.environment.value}")
            if 'vpc' in result.resources_created:
                stacks_to_delete.append(f"forex-bot-vpc-{self.config.environment.value}")
            
            for stack_name in stacks_to_delete:
                try:
                    self.cloudformation_client.delete_stack(StackName=stack_name)
                    logger.info(f"Deleted stack: {stack_name}")
                except Exception as e:
                    logger.error(f"Failed to delete stack {stack_name}: {e}")
            
            result.status = DeploymentStatus.ROLLBACK
            logger.info("Rollback completed")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def destroy_infrastructure(self) -> DeploymentResult:
        """Destroy all infrastructure resources"""
        deployment_id = f"destroy-{int(time.time())}"
        start_time = time.time()
        
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            service_name="infrastructure_destroy",
            environment=self.config.environment,
            deployment_id=deployment_id,
            start_time=start_time
        )
        
        try:
            logger.info(f"Starting infrastructure destruction: {deployment_id}")
            
            # Delete stacks in reverse creation order
            stacks = [
                f"forex-bot-redis-{self.config.environment.value}",
                f"forex-bot-db-{self.config.environment.value}",
                f"forex-bot-alb-{self.config.environment.value}",
                f"forex-bot-ecs-{self.config.environment.value}",
                f"forex-bot-vpc-{self.config.environment.value}"
            ]
            
            for stack_name in stacks:
                try:
                    self.cloudformation_client.delete_stack(StackName=stack_name)
                    result.resources_created.setdefault('deleted_stacks', []).append(stack_name)
                    logger.info(f"Deleted stack: {stack_name}")
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ValidationError':
                        logger.warning(f"Stack {stack_name} not found or already deleted")
                    else:
                        raise
            
            # Delete ECR repository
            try:
                repository_name = f"forex-trading-bot/{self.config.environment.value}"
                self.ecr_client.delete_repository(
                    repositoryName=repository_name,
                    force=True
                )
                result.resources_created.setdefault('deleted_repositories', []).append(repository_name)
                logger.info(f"Deleted ECR repository: {repository_name}")
            except ClientError as e:
                logger.warning(f"ECR repository deletion failed: {e}")
            
            result.status = DeploymentStatus.SUCCESS
            result.end_time = time.time()
            logger.info("Infrastructure destruction completed")
            
        except Exception as e:
            error_msg = f"Infrastructure destruction failed: {str(e)}"
            logger.error(error_msg)
            result.status = DeploymentStatus.FAILED
            result.end_time = time.time()
            result.errors.append(error_msg)
        
        finally:
            self.deployment_history.append(result)
            return result

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment"""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None

    def get_deployment_history(self, limit: int = 10) -> List[DeploymentResult]:
        """Get deployment history"""
        return self.deployment_history[-limit:]

    # CloudFormation template creation methods
    def _create_vpc_template(self) -> Dict[str, Any]:
        """Create VPC CloudFormation template"""
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "VPC for Forex Trading Bot",
            "Parameters": {
                "Environment": {"Type": "String"},
                "VPCCIDR": {"Type": "String", "Default": "10.0.0.0/16"}
            },
            "Resources": {
                "VPC": {
                    "Type": "AWS::EC2::VPC",
                    "Properties": {
                        "CidrBlock": {"Ref": "VPCCIDR"},
                        "EnableDnsHostnames": True,
                        "EnableDnsSupport": True,
                        "Tags": [{"Key": "Name", "Value": {"Fn::Sub": "forex-bot-vpc-${Environment}"}}]
                    }
                },
                "PublicSubnet1": {
                    "Type": "AWS::EC2::Subnet",
                    "Properties": {
                        "VpcId": {"Ref": "VPC"},
                        "CidrBlock": "10.0.1.0/24",
                        "AvailabilityZone": {"Fn::Select": [0, {"Fn::GetAZs": ""}]},
                        "Tags": [{"Key": "Name", "Value": {"Fn::Sub": "public-subnet-1-${Environment}"}}]
                    }
                },
                # Additional subnet and networking resources would be defined here
            },
            "Outputs": {
                "VPCId": {"Value": {"Ref": "VPC"}, "Description": "VPC ID"},
                "PublicSubnet1": {"Value": {"Ref": "PublicSubnet1"}, "Description": "Public Subnet 1"}
            }
        }

    def _create_ecs_cluster_template(self) -> Dict[str, Any]:
        """Create ECS Cluster CloudFormation template"""
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "ECS Cluster for Forex Trading Bot",
            "Parameters": {
                "Environment": {"Type": "String"},
                "VpcId": {"Type": "AWS::EC2::VPC::Id"},
                "SubnetIds": {"Type": "List<AWS::EC2::Subnet::Id>"},
                "InstanceType": {"Type": "String", "Default": "t3.medium"},
                "KeyName": {"Type": "AWS::EC2::KeyPair::KeyName"},
                "MinSize": {"Type": "Number", "Default": "1"},
                "MaxSize": {"Type": "Number", "Default": "10"},
                "DesiredCapacity": {"Type": "Number", "Default": "2"}
            },
            "Resources": {
                "ECSCluster": {
                    "Type": "AWS::ECS::Cluster",
                    "Properties": {
                        "ClusterName": {"Fn::Sub": "forex-bot-cluster-${Environment}"},
                        "Tags": [{"Key": "Environment", "Value": {"Ref": "Environment"}}]
                    }
                },
                # Additional resources for auto-scaling group, launch configuration, etc.
            },
            "Outputs": {
                "ECSClusterName": {"Value": {"Ref": "ECSCluster"}, "Description": "ECS Cluster Name"},
                "AutoScalingGroup": {"Value": {"Ref": "AutoScalingGroup"}, "Description": "Auto Scaling Group Name"}
            }
        }

    def _create_load_balancer_template(self) -> Dict[str, Any]:
        """Create Load Balancer CloudFormation template"""
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "Application Load Balancer for Forex Trading Bot",
            "Parameters": {
                "Environment": {"Type": "String"},
                "VpcId": {"Type": "AWS::EC2::VPC::Id"},
                "SubnetIds": {"Type": "List<AWS::EC2::Subnet::Id>"},
                "CertificateArn": {"Type": "String", "Default": ""}
            },
            "Resources": {
                "LoadBalancer": {
                    "Type": "AWS::ElasticLoadBalancingV2::LoadBalancer",
                    "Properties": {
                        "Name": {"Fn::Sub": "forex-bot-alb-${Environment}"},
                        "Subnets": {"Ref": "SubnetIds"},
                        "SecurityGroups": [{"Ref": "LoadBalancerSecurityGroup"}],
                        "Scheme": "internet-facing",
                        "Type": "application",
                        "Tags": [{"Key": "Environment", "Value": {"Ref": "Environment"}}]
                    }
                },
                # Additional resources for listeners, target groups, security groups
            },
            "Outputs": {
                "LoadBalancerDNS": {"Value": {"Fn::GetAtt": ["LoadBalancer", "DNSName"]}, "Description": "Load Balancer DNS"},
                "TargetGroupARN": {"Value": {"Ref": "TargetGroup"}, "Description": "Target Group ARN"}
            }
        }

    def _create_database_template(self) -> Dict[str, Any]:
        """Create RDS Database CloudFormation template"""
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "RDS PostgreSQL Database for Forex Trading Bot",
            "Parameters": {
                "Environment": {"Type": "String"},
                "VpcId": {"Type": "AWS::EC2::VPC::Id"},
                "SubnetIds": {"Type": "List<AWS::EC2::Subnet::Id>"},
                "DBInstanceClass": {"Type": "String", "Default": "db.t3.medium"},
                "DBName": {"Type": "String", "Default": "forex_bot"}
            },
            "Resources": {
                "DBSubnetGroup": {
                    "Type": "AWS::RDS::DBSubnetGroup",
                    "Properties": {
                        "DBSubnetGroupDescription": "Subnet group for Forex Bot database",
                        "SubnetIds": {"Ref": "SubnetIds"},
                        "Tags": [{"Key": "Environment", "Value": {"Ref": "Environment"}}]
                    }
                },
                "Database": {
                    "Type": "AWS::RDS::DBInstance",
                    "Properties": {
                        "DBInstanceIdentifier": {"Fn::Sub": "forex-bot-db-${Environment}"},
                        "DBName": {"Ref": "DBName"},
                        "DBInstanceClass": {"Ref": "DBInstanceClass"},
                        "Engine": "postgres",
                        "EngineVersion": "13.7",
                        "MasterUsername": "forexbot",
                        "MasterUserPassword": "{{resolve:secretsmanager:forex-bot-db-password:SecretString:password}}",
                        "AllocatedStorage": "20",
                        "StorageType": "gp2",
                        "DBSubnetGroupName": {"Ref": "DBSubnetGroup"},
                        "VPCSecurityGroups": [{"Ref": "DBSecurityGroup"}],
                        "BackupRetentionPeriod": 7,
                        "MultiAZ": False,
                        "PubliclyAccessible": False,
                        "Tags": [{"Key": "Environment", "Value": {"Ref": "Environment"}}]
                    }
                }
            },
            "Outputs": {
                "DBEndpoint": {"Value": {"Fn::GetAtt": ["Database", "Endpoint.Address"]}, "Description": "Database Endpoint"},
                "DBSecretARN": {"Value": "{{resolve:secretsmanager:forex-bot-db-password:SecretString:password}}", "Description": "Database Secret ARN"}
            }
        }

    def _create_redis_template(self) -> Dict[str, Any]:
        """Create Redis ElastiCache CloudFormation template"""
        return {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "Redis ElastiCache for Forex Trading Bot",
            "Parameters": {
                "Environment": {"Type": "String"},
                "VpcId": {"Type": "AWS::EC2::VPC::Id"},
                "SubnetIds": {"Type": "List<AWS::EC2::Subnet::Id>"},
                "CacheNodeType": {"Type": "String", "Default": "cache.t3.medium"}
            },
            "Resources": {
                "CacheSubnetGroup": {
                    "Type": "AWS::ElastiCache::SubnetGroup",
                    "Properties": {
                        "Description": "Subnet group for Forex Bot Redis",
                        "SubnetIds": {"Ref": "SubnetIds"}
                    }
                },
                "RedisCluster": {
                    "Type": "AWS::ElastiCache::CacheCluster",
                    "Properties": {
                        "CacheNodeType": {"Ref": "CacheNodeType"},
                        "Engine": "redis",
                        "EngineVersion": "6.x",
                        "NumCacheNodes": 1,
                        "CacheSubnetGroupName": {"Ref": "CacheSubnetGroup"},
                        "VpcSecurityGroupIds": [{"Ref": "RedisSecurityGroup"}],
                        "Tags": [{"Key": "Environment", "Value": {"Ref": "Environment"}}]
                    }
                }
            },
            "Outputs": {
                "RedisEndpoint": {"Value": {"Fn::GetAtt": ["RedisCluster", "RedisEndpoint.Address"]}, "Description": "Redis Endpoint"}
            }
        }

# Example usage and testing
async def main():
    """Test the AWS Deployer"""
    
    # Configuration for development environment
    config = DeploymentConfig(
        environment=DeploymentEnvironment.DEVELOPMENT,
        region="us-east-1",
        instance_type="t3.small",
        min_instances=1,
        max_instances=3,
        desired_instances=1,
        key_pair_name="forex-bot-dev-key",
        docker_image="forex-trading-bot:latest",
        enable_ssl=False,
        monitoring_enabled=True,
        backup_enabled=True,
        auto_scaling_enabled=True
    )
    
    # Initialize deployer
    deployer = AWSDeployer(config)
    
    try:
        print("Starting AWS deployment test...")
        
        # Deploy infrastructure
        print("1. Deploying infrastructure...")
        infra_result = await deployer.deploy_infrastructure()
        
        if infra_result.status == DeploymentStatus.SUCCESS:
            print(f"✓ Infrastructure deployed successfully in {infra_result.processing_time:.2f}s")
            print(f"  Estimated monthly cost: ${infra_result.cost_estimate}")
            
            # Deploy trading bot application
            print("2. Deploying Trading Bot application...")
            app_result = await deployer.deploy_trading_bot()
            
            if app_result.status == DeploymentStatus.SUCCESS:
                print(f"✓ Trading Bot deployed successfully in {app_result.processing_time:.2f}s")
                print(f"  Docker image: {app_result.outputs.get('docker_image', 'N/A')}")
                print(f"  Service ARN: {app_result.outputs.get('service', 'N/A')}")
            else:
                print(f"✗ Trading Bot deployment failed: {app_result.errors}")
                
        else:
            print(f"✗ Infrastructure deployment failed: {infra_result.errors}")
        
        # Show deployment history
        print("\n3. Deployment History:")
        history = deployer.get_deployment_history()
        for deployment in history:
            print(f"  - {deployment.deployment_id}: {deployment.status.value} "
                  f"({deployment.processing_time:.2f}s)")
        
        # Cleanup (comment out to keep resources running)
        # print("4. Cleaning up resources...")
        # destroy_result = await deployer.destroy_infrastructure()
        # print(f"  Cleanup: {destroy_result.status.value}")
        
    except Exception as e:
        print(f"Deployment test failed: {e}")
    
    print("AWS deployment test completed!")

if __name__ == "__main__":
    asyncio.run(main())
"""
Cloud Configuration Manager for FOREX TRADING BOT
Advanced cloud infrastructure configuration and management
"""

import logging
import boto3
import json
import yaml
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import docker
from docker.types import Mount, ServiceMode
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
import paramiko
import requests
import time
import hashlib
from pathlib import Path
import zipfile
import io

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITAL_OCEAN = "digital_ocean"
    HEROKU = "heroku"

class DeploymentTier(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ServiceType(Enum):
    COMPUTE = "compute"
    DATABASE = "database"
    CACHE = "cache"
    LOAD_BALANCER = "load_balancer"
    STORAGE = "storage"
    MONITORING = "monitoring"
    NETWORKING = "networking"

@dataclass
class CloudConfig:
    """Cloud configuration data class"""
    provider: CloudProvider
    tier: DeploymentTier
    region: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    project_id: Optional[str] = None  # For GCP
    subscription_id: Optional[str] = None  # For Azure
    resource_group: str = "forex-trading-bot"
    
    # Compute configuration
    instance_type: str = "t3.medium"
    min_instances: int = 1
    max_instances: int = 10
    desired_instances: int = 2
    
    # Storage configuration
    storage_size_gb: int = 100
    storage_type: str = "gp3"
    
    # Database configuration
    db_instance_type: str = "db.t3.medium"
    db_engine: str = "postgres"
    db_version: str = "13.7"
    
    # Cache configuration
    cache_node_type: str = "cache.t3.medium"
    cache_engine: str = "redis"
    
    # Networking
    vpc_cidr: str = "10.0.0.0/16"
    enable_public_ip: bool = True
    enable_ssl: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    enable_backups: bool = True
    backup_retention_days: int = 7
    
    # Security
    security_groups: List[str] = field(default_factory=list)
    allowed_cidrs: List[str] = field(default_factory=lambda: ["0.0.0.0/0"])
    
    # Tags
    tags: Dict[str, str] = field(default_factory=lambda: {
        "Project": "ForexTradingBot",
        "Environment": "production",
        "ManagedBy": "CloudConfigManager"
    })

@dataclass
class DeploymentResult:
    """Deployment result data class"""
    success: bool
    service_type: ServiceType
    resource_id: str = ""
    resource_arn: str = ""
    endpoint: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class CloudConfigManager:
    """
    Advanced cloud configuration manager for multi-cloud deployment
    """
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.clients = {}
        self.resources = {}
        self.deployment_history = []
        
        # Initialize cloud provider clients
        self._initialize_cloud_clients()
        
        # Initialize Docker client
        self._initialize_docker_client()
        
        logger.info(f"CloudConfigManager initialized for {config.provider.value}")

    def _initialize_cloud_clients(self):
        """Initialize cloud provider specific clients"""
        try:
            if self.config.provider == CloudProvider.AWS:
                self._initialize_aws_clients()
            elif self.config.provider == CloudProvider.GCP:
                self._initialize_gcp_clients()
            elif self.config.provider == CloudProvider.AZURE:
                self._initialize_azure_clients()
            elif self.config.provider == CloudProvider.DIGITAL_OCEAN:
                self._initialize_do_clients()
            else:
                raise ValueError(f"Unsupported cloud provider: {self.config.provider}")
                
            logger.info(f"{self.config.provider.value} clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Cloud client initialization failed: {e}")
            raise

    def _initialize_aws_clients(self):
        """Initialize AWS service clients"""
        try:
            # Use provided credentials or fallback to environment/default
            if self.config.access_key and self.config.secret_key:
                session = boto3.Session(
                    aws_access_key_id=self.config.access_key,
                    aws_secret_access_key=self.config.secret_key,
                    region_name=self.config.region
                )
            else:
                session = boto3.Session(region_name=self.config.region)
            
            # Initialize all AWS service clients
            self.clients['ec2'] = session.client('ec2')
            self.clients['ecs'] = session.client('ecs')
            self.clients['rds'] = session.client('rds')
            self.clients['elasticache'] = session.client('elasticache')
            self.clients['elbv2'] = session.client('elbv2')
            self.clients['autoscaling'] = session.client('autoscaling')
            self.clients['cloudformation'] = session.client('cloudformation')
            self.clients['s3'] = session.client('s3')
            self.clients['cloudwatch'] = session.client('cloudwatch')
            self.clients['iam'] = session.client('iam')
            self.clients['secretsmanager'] = session.client('secretsmanager')
            self.clients['lambda'] = session.client('lambda')
            
            # Verify credentials
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS Identity: {identity['Arn']}")
            
        except Exception as e:
            logger.error(f"AWS client initialization failed: {e}")
            raise

    def _initialize_gcp_clients(self):
        """Initialize Google Cloud Platform clients"""
        try:
            # This would use google-cloud-python libraries
            # For now, placeholder implementation
            logger.info("GCP client initialization would go here")
            
        except Exception as e:
            logger.error(f"GCP client initialization failed: {e}")
            raise

    def _initialize_azure_clients(self):
        """Initialize Microsoft Azure clients"""
        try:
            # This would use azure-mgmt libraries
            # For now, placeholder implementation
            logger.info("Azure client initialization would go here")
            
        except Exception as e:
            logger.error(f"Azure client initialization failed: {e}")
            raise

    def _initialize_do_clients(self):
        """Initialize Digital Ocean clients"""
        try:
            # This would use digitalocean library
            # For now, placeholder implementation
            logger.info("Digital Ocean client initialization would go here")
            
        except Exception as e:
            logger.error(f"Digital Ocean client initialization failed: {e}")
            raise

    def _initialize_docker_client(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Docker client initialization failed: {e}")
            self.docker_client = None

    def deploy_infrastructure(self) -> Dict[str, DeploymentResult]:
        """
        Deploy complete cloud infrastructure for Forex Trading Bot
        """
        try:
            logger.info(f"Starting infrastructure deployment for {self.config.tier.value}")
            
            deployment_results = {}
            
            # 1. Deploy Networking
            networking_result = self._deploy_networking_infrastructure()
            deployment_results['networking'] = networking_result
            
            if not networking_result.success:
                raise Exception("Networking deployment failed")
            
            # 2. Deploy Compute
            compute_result = self._deploy_compute_infrastructure()
            deployment_results['compute'] = compute_result
            
            # 3. Deploy Database
            database_result = self._deploy_database_infrastructure()
            deployment_results['database'] = database_result
            
            # 4. Deploy Cache
            cache_result = self._deploy_cache_infrastructure()
            deployment_results['cache'] = cache_result
            
            # 5. Deploy Storage
            storage_result = self._deploy_storage_infrastructure()
            deployment_results['storage'] = storage_result
            
            # 6. Deploy Load Balancer
            lb_result = self._deploy_load_balancer()
            deployment_results['load_balancer'] = lb_result
            
            # 7. Setup Monitoring
            monitoring_result = self._setup_monitoring()
            deployment_results['monitoring'] = monitoring_result
            
            # 8. Configure Security
            security_result = self._configure_security()
            deployment_results['security'] = security_result
            
            # Generate deployment summary
            summary = self._generate_deployment_summary(deployment_results)
            
            logger.info("Infrastructure deployment completed successfully")
            
            return deployment_results
            
        except Exception as e:
            logger.error(f"Infrastructure deployment failed: {e}")
            # Attempt rollback
            self._rollback_deployment()
            raise

    def _deploy_networking_infrastructure(self) -> DeploymentResult:
        """Deploy networking infrastructure (VPC, subnets, etc.)"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._deploy_aws_networking()
            elif self.config.provider == CloudProvider.GCP:
                return self._deploy_gcp_networking()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.NETWORKING,
                    error_message=f"Networking not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Networking deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.NETWORKING,
                error_message=str(e)
            )

    def _deploy_aws_networking(self) -> DeploymentResult:
        """Deploy AWS networking infrastructure"""
        try:
            ec2 = self.clients['ec2']
            
            # Create VPC
            vpc_response = ec2.create_vpc(
                CidrBlock=self.config.vpc_cidr,
                AmazonProvidedIpv6CidrBlock=False,
                InstanceTenancy='default',
                TagSpecifications=[{
                    'ResourceType': 'vpc',
                    'Tags': [{'Key': 'Name', 'Value': f"forex-bot-vpc-{self.config.tier.value}"}]
                }]
            )
            vpc_id = vpc_response['Vpc']['VpcId']
            
            # Enable DNS support and hostnames
            ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsSupport={'Value': True})
            ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={'Value': True})
            
            # Create Internet Gateway
            igw_response = ec2.create_internet_gateway(
                TagSpecifications=[{
                    'ResourceType': 'internet-gateway',
                    'Tags': [{'Key': 'Name', 'Value': f"forex-bot-igw-{self.config.tier.value}"}]
                }]
            )
            igw_id = igw_response['InternetGateway']['InternetGatewayId']
            
            # Attach Internet Gateway to VPC
            ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=vpc_id)
            
            # Create subnets
            subnet_configs = [
                {'cidr': '10.0.1.0/24', 'az': 'a', 'public': True},
                {'cidr': '10.0.2.0/24', 'az': 'b', 'public': True},
                {'cidr': '10.0.3.0/24', 'az': 'a', 'public': False},
                {'cidr': '10.0.4.0/24', 'az': 'b', 'public': False}
            ]
            
            subnets = {}
            for config in subnet_configs:
                subnet_response = ec2.create_subnet(
                    VpcId=vpc_id,
                    CidrBlock=config['cidr'],
                    AvailabilityZone=f"{self.config.region}{config['az']}",
                    TagSpecifications=[{
                        'ResourceType': 'subnet',
                        'Tags': [{'Key': 'Name', 'Value': f"forex-bot-subnet-{config['az']}-{'public' if config['public'] else 'private'}"}]
                    }]
                )
                subnet_id = subnet_response['Subnet']['SubnetId']
                subnets[f"{'public' if config['public'] else 'private'}_{config['az']}"] = subnet_id
            
            # Create route tables
            public_rt_response = ec2.create_route_table(
                VpcId=vpc_id,
                TagSpecifications=[{
                    'ResourceType': 'route-table',
                    'Tags': [{'Key': 'Name', 'Value': 'forex-bot-public-rt'}]
                }]
            )
            public_rt_id = public_rt_response['RouteTable']['RouteTableId']
            
            # Add route to internet gateway
            ec2.create_route(
                RouteTableId=public_rt_id,
                DestinationCidrBlock='0.0.0.0/0',
                GatewayId=igw_id
            )
            
            # Associate public subnets with public route table
            for subnet_id in [subnets['public_a'], subnets['public_b']]:
                ec2.associate_route_table(
                    RouteTableId=public_rt_id,
                    SubnetId=subnet_id
                )
            
            # Store resource IDs
            self.resources['vpc_id'] = vpc_id
            self.resources['igw_id'] = igw_id
            self.resources['subnets'] = subnets
            self.resources['public_route_table_id'] = public_rt_id
            
            logger.info(f"AWS networking deployed: VPC={vpc_id}")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.NETWORKING,
                resource_id=vpc_id,
                metadata={
                    'vpc_id': vpc_id,
                    'igw_id': igw_id,
                    'subnets': subnets
                }
            )
            
        except Exception as e:
            logger.error(f"AWS networking deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.NETWORKING,
                error_message=str(e)
            )

    def _deploy_compute_infrastructure(self) -> DeploymentResult:
        """Deploy compute infrastructure"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._deploy_aws_compute()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.COMPUTE,
                    error_message=f"Compute not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Compute deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.COMPUTE,
                error_message=str(e)
            )

    def _deploy_aws_compute(self) -> DeploymentResult:
        """Deploy AWS compute infrastructure (ECS Cluster)"""
        try:
            ecs = self.clients['ecs']
            autoscaling = self.clients['autoscaling']
            ec2 = self.clients['ec2']
            
            cluster_name = f"forex-bot-cluster-{self.config.tier.value}"
            
            # Create ECS cluster
            cluster_response = ecs.create_cluster(
                clusterName=cluster_name,
                tags=[{'key': 'Environment', 'value': self.config.tier.value}]
            )
            cluster_arn = cluster_response['cluster']['clusterArn']
            
            # Create Auto Scaling Group for ECS instances
            launch_config_name = f"forex-bot-lc-{self.config.tier.value}"
            asg_name = f"forex-bot-asg-{self.config.tier.value}"
            
            # Get latest Amazon Linux 2 AMI
            ami_response = ec2.describe_images(
                Owners=['amazon'],
                Filters=[
                    {'Name': 'name', 'Values': ['amzn2-ami-ecs-hvm-2.0.*-x86_64-ebs']},
                    {'Name': 'state', 'Values': ['available']}
                ]
            )
            latest_ami = sorted(ami_response['Images'], key=lambda x: x['CreationDate'], reverse=True)[0]
            ami_id = latest_ami['ImageId']
            
            # Create launch configuration
            user_data = f"""#!/bin/bash
echo ECS_CLUSTER={cluster_name} >> /etc/ecs/ecs.config
echo ECS_ENABLE_SPOT_INSTANCE_DRAINING=true >> /etc/ecs/ecs.config
echo ECS_ENABLE_TASK_IAM_ROLE=true >> /etc/ecs/ecs.config
"""
            
            lc_response = autoscaling.create_launch_configuration(
                LaunchConfigurationName=launch_config_name,
                ImageId=ami_id,
                InstanceType=self.config.instance_type,
                KeyName='forex-bot-key',  # Would be configurable
                SecurityGroups=self.config.security_groups,
                IamInstanceProfile='ecsInstanceRole',  # Would be created
                UserData=user_data,
                InstanceMonitoring={'Enabled': True}
            )
            
            # Create Auto Scaling Group
            asg_response = autoscaling.create_auto_scaling_group(
                AutoScalingGroupName=asg_name,
                LaunchConfigurationName=launch_config_name,
                MinSize=self.config.min_instances,
                MaxSize=self.config.max_instances,
                DesiredCapacity=self.config.desired_instances,
                VPCZoneIdentifier=','.join([self.resources['subnets']['public_a'], self.resources['subnets']['public_b']]),
                Tags=[
                    {'Key': 'Name', 'Value': f"forex-bot-instance-{self.config.tier.value}", 'PropagateAtLaunch': True},
                    {'Key': 'Environment', 'Value': self.config.tier.value, 'PropagateAtLaunch': True}
                ]
            )
            
            self.resources['ecs_cluster_arn'] = cluster_arn
            self.resources['asg_name'] = asg_name
            self.resources['launch_config_name'] = launch_config_name
            
            logger.info(f"AWS compute deployed: Cluster={cluster_arn}")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.COMPUTE,
                resource_id=cluster_arn,
                metadata={
                    'cluster_arn': cluster_arn,
                    'asg_name': asg_name,
                    'instance_type': self.config.instance_type
                }
            )
            
        except Exception as e:
            logger.error(f"AWS compute deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.COMPUTE,
                error_message=str(e)
            )

    def _deploy_database_infrastructure(self) -> DeploymentResult:
        """Deploy database infrastructure"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._deploy_aws_database()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.DATABASE,
                    error_message=f"Database not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Database deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.DATABASE,
                error_message=str(e)
            )

    def _deploy_aws_database(self) -> DeploymentResult:
        """Deploy AWS RDS PostgreSQL database"""
        try:
            rds = self.clients['rds']
            secretsmanager = self.clients['secretsmanager']
            
            db_identifier = f"forex-bot-db-{self.config.tier.value}"
            
            # Generate secure password
            import secrets
            import string
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            password = ''.join(secrets.choice(alphabet) for i in range(32))
            
            # Store password in Secrets Manager
            secret_response = secretsmanager.create_secret(
                Name=f"forex-bot-db-password-{self.config.tier.value}",
                Description="Database password for Forex Trading Bot",
                SecretString=json.dumps({
                    'username': 'forexbot',
                    'password': password,
                    'engine': 'postgres',
                    'host': '',  # Will be updated after DB creation
                    'port': 5432,
                    'dbname': 'forex_bot'
                }),
                Tags=[
                    {'Key': 'Environment', 'Value': self.config.tier.value},
                    {'Key': 'Project', 'Value': 'ForexTradingBot'}
                ]
            )
            
            # Create DB subnet group
            subnet_group_name = f"forex-bot-db-subnet-group-{self.config.tier.value}"
            rds.create_db_subnet_group(
                DBSubnetGroupName=subnet_group_name,
                DBSubnetGroupDescription="Subnet group for Forex Bot database",
                SubnetIds=[self.resources['subnets']['private_a'], self.resources['subnets']['private_b']],
                Tags=[{'Key': 'Environment', 'Value': self.config.tier.value}]
            )
            
            # Create security group for database
            ec2 = self.clients['ec2']
            sg_response = ec2.create_security_group(
                GroupName=f"forex-bot-db-sg-{self.config.tier.value}",
                Description="Security group for Forex Bot database",
                VpcId=self.resources['vpc_id']
            )
            db_security_group_id = sg_response['GroupId']
            
            # Add ingress rule for database access
            ec2.authorize_security_group_ingress(
                GroupId=db_security_group_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 5432,
                    'ToPort': 5432,
                    'IpRanges': [{'CidrIp': '10.0.0.0/16'}]  # Allow from VPC
                }]
            )
            
            # Create RDS instance
            db_response = rds.create_db_instance(
                DBInstanceIdentifier=db_identifier,
                DBName='forex_bot',
                DBInstanceClass=self.config.db_instance_type,
                Engine=self.config.db_engine,
                EngineVersion=self.config.db_version,
                MasterUsername='forexbot',
                MasterUserPassword=password,
                AllocatedStorage=self.config.storage_size_gb,
                StorageType=self.config.storage_type,
                DBSubnetGroupName=subnet_group_name,
                VpcSecurityGroupIds=[db_security_group_id],
                BackupRetentionPeriod=self.config.backup_retention_days,
                MultiAZ=False,
                AutoMinorVersionUpgrade=True,
                PubliclyAccessible=False,
                DeletionProtection=(self.config.tier == DeploymentTier.PRODUCTION),
                Tags=[{'Key': 'Environment', 'Value': self.config.tier.value}]
            )
            
            # Wait for DB to be available
            waiter = rds.get_waiter('db_instance_available')
            waiter.wait(DBInstanceIdentifier=db_identifier)
            
            # Get endpoint
            db_info = rds.describe_db_instances(DBInstanceIdentifier=db_identifier)
            endpoint = db_info['DBInstances'][0]['Endpoint']['Address']
            
            # Update secret with endpoint
            secretsmanager.update_secret(
                SecretId=f"forex-bot-db-password-{self.config.tier.value}",
                SecretString=json.dumps({
                    'username': 'forexbot',
                    'password': password,
                    'engine': 'postgres',
                    'host': endpoint,
                    'port': 5432,
                    'dbname': 'forex_bot'
                })
            )
            
            self.resources['db_identifier'] = db_identifier
            self.resources['db_endpoint'] = endpoint
            self.resources['db_secret_arn'] = secret_response['ARN']
            self.resources['db_security_group_id'] = db_security_group_id
            
            logger.info(f"AWS database deployed: {endpoint}")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.DATABASE,
                resource_id=db_identifier,
                endpoint=endpoint,
                metadata={
                    'endpoint': endpoint,
                    'engine': self.config.db_engine,
                    'version': self.config.db_version
                }
            )
            
        except Exception as e:
            logger.error(f"AWS database deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.DATABASE,
                error_message=str(e)
            )

    def _deploy_cache_infrastructure(self) -> DeploymentResult:
        """Deploy cache infrastructure"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._deploy_aws_cache()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.CACHE,
                    error_message=f"Cache not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Cache deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.CACHE,
                error_message=str(e)
            )

    def _deploy_aws_cache(self) -> DeploymentResult:
        """Deploy AWS ElastiCache Redis"""
        try:
            elasticache = self.clients['elasticache']
            ec2 = self.clients['ec2']
            
            cache_cluster_id = f"forex-bot-cache-{self.config.tier.value}"
            
            # Create cache subnet group
            subnet_group_response = elasticache.create_cache_subnet_group(
                CacheSubnetGroupName=f"forex-bot-cache-subnet-{self.config.tier.value}",
                CacheSubnetGroupDescription="Subnet group for Forex Bot cache",
                SubnetIds=[self.resources['subnets']['private_a'], self.resources['subnets']['private_b']]
            )
            
            # Create security group for cache
            sg_response = ec2.create_security_group(
                GroupName=f"forex-bot-cache-sg-{self.config.tier.value}",
                Description="Security group for Forex Bot cache",
                VpcId=self.resources['vpc_id']
            )
            cache_security_group_id = sg_response['GroupId']
            
            # Add ingress rule for cache access
            ec2.authorize_security_group_ingress(
                GroupId=cache_security_group_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 6379,
                    'ToPort': 6379,
                    'IpRanges': [{'CidrIp': '10.0.0.0/16'}]  # Allow from VPC
                }]
            )
            
            # Create Redis cluster
            cache_response = elasticache.create_cache_cluster(
                CacheClusterId=cache_cluster_id,
                Engine=self.config.cache_engine,
                CacheNodeType=self.config.cache_node_type,
                NumCacheNodes=1,
                CacheSubnetGroupName=f"forex-bot-cache-subnet-{self.config.tier.value}",
                SecurityGroupIds=[cache_security_group_id],
                Tags=[{'Key': 'Environment', 'Value': self.config.tier.value}]
            )
            
            # Wait for cache cluster to be available
            waiter = elasticache.get_waiter('cache_cluster_available')
            waiter.wait(CacheClusterId=cache_cluster_id)
            
            # Get endpoint
            cache_info = elasticache.describe_cache_clusters(CacheClusterId=cache_cluster_id)
            endpoint = cache_info['CacheClusters'][0]['ConfigurationEndpoint']['Address']
            
            self.resources['cache_cluster_id'] = cache_cluster_id
            self.resources['cache_endpoint'] = endpoint
            self.resources['cache_security_group_id'] = cache_security_group_id
            
            logger.info(f"AWS cache deployed: {endpoint}")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.CACHE,
                resource_id=cache_cluster_id,
                endpoint=endpoint,
                metadata={
                    'endpoint': endpoint,
                    'engine': self.config.cache_engine,
                    'node_type': self.config.cache_node_type
                }
            )
            
        except Exception as e:
            logger.error(f"AWS cache deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.CACHE,
                error_message=str(e)
            )

    def _deploy_storage_infrastructure(self) -> DeploymentResult:
        """Deploy storage infrastructure"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._deploy_aws_storage()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.STORAGE,
                    error_message=f"Storage not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Storage deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.STORAGE,
                error_message=str(e)
            )

    def _deploy_aws_storage(self) -> DeploymentResult:
        """Deploy AWS S3 buckets"""
        try:
            s3 = self.clients['s3']
            
            buckets = {}
            
            # Create data bucket
            data_bucket_name = f"forex-bot-data-{self.config.tier.value}-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            s3.create_bucket(
                Bucket=data_bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.config.region}
            )
            
            # Configure bucket policy
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:*",
                        "Resource": [
                            f"arn:aws:s3:::{data_bucket_name}",
                            f"arn:aws:s3:::{data_bucket_name}/*"
                        ],
                        "Condition": {
                            "Bool": {"aws:SecureTransport": False}
                        }
                    }
                ]
            }
            
            s3.put_bucket_policy(
                Bucket=data_bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            # Enable versioning
            s3.put_bucket_versioning(
                Bucket=data_bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Enable server-side encryption
            s3.put_bucket_encryption(
                Bucket=data_bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [
                        {
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            }
                        }
                    ]
                }
            )
            
            buckets['data'] = data_bucket_name
            
            self.resources['s3_buckets'] = buckets
            
            logger.info(f"AWS storage deployed: {data_bucket_name}")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.STORAGE,
                resource_id=data_bucket_name,
                metadata={'buckets': buckets}
            )
            
        except Exception as e:
            logger.error(f"AWS storage deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.STORAGE,
                error_message=str(e)
            )

    def _deploy_load_balancer(self) -> DeploymentResult:
        """Deploy load balancer"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._deploy_aws_load_balancer()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.LOAD_BALANCER,
                    error_message=f"Load balancer not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Load balancer deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.LOAD_BALANCER,
                error_message=str(e)
            )

    def _deploy_aws_load_balancer(self) -> DeploymentResult:
        """Deploy AWS Application Load Balancer"""
        try:
            elbv2 = self.clients['elbv2']
            ec2 = self.clients['ec2']
            
            lb_name = f"forex-bot-alb-{self.config.tier.value}"
            
            # Create security group for ALB
            sg_response = ec2.create_security_group(
                GroupName=f"forex-bot-alb-sg-{self.config.tier.value}",
                Description="Security group for Forex Bot ALB",
                VpcId=self.resources['vpc_id']
            )
            alb_security_group_id = sg_response['GroupId']
            
            # Add ingress rules for ALB
            ec2.authorize_security_group_ingress(
                GroupId=alb_security_group_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }, {
                    'IpProtocol': 'tcp',
                    'FromPort': 443,
                    'ToPort': 443,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
            
            # Create ALB
            alb_response = elbv2.create_load_balancer(
                Name=lb_name,
                Subnets=[self.resources['subnets']['public_a'], self.resources['subnets']['public_b']],
                SecurityGroups=[alb_security_group_id],
                Scheme='internet-facing',
                Tags=[{'Key': 'Environment', 'Value': self.config.tier.value}],
                Type='application',
                IpAddressType='ipv4'
            )
            
            alb_arn = alb_response['LoadBalancers'][0]['LoadBalancerArn']
            alb_dns = alb_response['LoadBalancers'][0]['DNSName']
            
            # Create target group
            tg_response = elbv2.create_target_group(
                Name=f"forex-bot-tg-{self.config.tier.value}",
                Protocol='HTTP',
                Port=8050,
                VpcId=self.resources['vpc_id'],
                HealthCheckProtocol='HTTP',
                HealthCheckPort='8050',
                HealthCheckPath='/health',
                HealthCheckIntervalSeconds=30,
                HealthCheckTimeoutSeconds=5,
                HealthyThresholdCount=2,
                UnhealthyThresholdCount=2,
                TargetType='ip',
                Tags=[{'Key': 'Environment', 'Value': self.config.tier.value}]
            )
            
            tg_arn = tg_response['TargetGroups'][0]['TargetGroupArn']
            
            # Create listener
            listener_response = elbv2.create_listener(
                LoadBalancerArn=alb_arn,
                Protocol='HTTP',
                Port=80,
                DefaultActions=[{
                    'Type': 'forward',
                    'TargetGroupArn': tg_arn
                }]
            )
            
            self.resources['alb_arn'] = alb_arn
            self.resources['alb_dns'] = alb_dns
            self.resources['target_group_arn'] = tg_arn
            self.resources['alb_security_group_id'] = alb_security_group_id
            
            logger.info(f"AWS load balancer deployed: {alb_dns}")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.LOAD_BALANCER,
                resource_id=alb_arn,
                endpoint=alb_dns,
                metadata={
                    'dns_name': alb_dns,
                    'target_group_arn': tg_arn
                }
            )
            
        except Exception as e:
            logger.error(f"AWS load balancer deployment failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.LOAD_BALANCER,
                error_message=str(e)
            )

    def _setup_monitoring(self) -> DeploymentResult:
        """Setup monitoring and alerting"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._setup_aws_monitoring()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.MONITORING,
                    error_message=f"Monitoring not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.MONITORING,
                error_message=str(e)
            )

    def _setup_aws_monitoring(self) -> DeploymentResult:
        """Setup AWS CloudWatch monitoring"""
        try:
            cloudwatch = self.clients['cloudwatch']
            
            # Create CloudWatch dashboard
            dashboard_name = f"forex-bot-{self.config.tier.value}"
            
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
                                ["AWS/ECS", "CPUUtilization"],
                                ["AWS/ECS", "MemoryUtilization"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config.region,
                            "title": "ECS Cluster Metrics"
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
                                ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", self.resources.get('db_identifier', '')],
                                ["AWS/RDS", "DatabaseConnections", "DBInstanceIdentifier", self.resources.get('db_identifier', '')]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config.region,
                            "title": "RDS Metrics"
                        }
                    }
                ]
            }
            
            cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            # Create alarms
            alarms_created = []
            
            # ECS CPU alarm
            cpu_alarm_name = f"forex-bot-ecs-cpu-{self.config.tier.value}"
            cloudwatch.put_metric_alarm(
                AlarmName=cpu_alarm_name,
                AlarmDescription='High CPU utilization for Forex Bot ECS cluster',
                MetricName='CPUUtilization',
                Namespace='AWS/ECS',
                Statistic='Average',
                Period=300,
                EvaluationPeriods=2,
                Threshold=80.0,
                ComparisonOperator='GreaterThanThreshold',
                AlarmActions=[],  # Would add SNS topic ARN
                OKActions=[],
                Dimensions=[{'Name': 'ClusterName', 'Value': f"forex-bot-cluster-{self.config.tier.value}"}]
            )
            alarms_created.append(cpu_alarm_name)
            
            # RDS CPU alarm
            rds_cpu_alarm_name = f"forex-bot-rds-cpu-{self.config.tier.value}"
            cloudwatch.put_metric_alarm(
                AlarmName=rds_cpu_alarm_name,
                AlarmDescription='High CPU utilization for Forex Bot RDS',
                MetricName='CPUUtilization',
                Namespace='AWS/RDS',
                Statistic='Average',
                Period=300,
                EvaluationPeriods=2,
                Threshold=90.0,
                ComparisonOperator='GreaterThanThreshold',
                AlarmActions=[],
                OKActions=[],
                Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': self.resources.get('db_identifier', '')}]
            )
            alarms_created.append(rds_cpu_alarm_name)
            
            logger.info(f"AWS monitoring setup: {dashboard_name}")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.MONITORING,
                resource_id=dashboard_name,
                metadata={
                    'dashboard': dashboard_name,
                    'alarms_created': alarms_created
                }
            )
            
        except Exception as e:
            logger.error(f"AWS monitoring setup failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.MONITORING,
                error_message=str(e)
            )

    def _configure_security(self) -> DeploymentResult:
        """Configure security settings"""
        try:
            if self.config.provider == CloudProvider.AWS:
                return self._configure_aws_security()
            else:
                return DeploymentResult(
                    success=False,
                    service_type=ServiceType.NETWORKING,
                    error_message=f"Security configuration not implemented for {self.config.provider.value}"
                )
                
        except Exception as e:
            logger.error(f"Security configuration failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.NETWORKING,
                error_message=str(e)
            )

    def _configure_aws_security(self) -> DeploymentResult:
        """Configure AWS security settings"""
        try:
            iam = self.clients['iam']
            
            # Create IAM roles for ECS tasks
            task_role_name = f"forex-bot-task-role-{self.config.tier.value}"
            execution_role_name = f"forex-bot-execution-role-{self.config.tier.value}"
            
            # Task role for ECS tasks
            task_role_response = iam.create_role(
                RoleName=task_role_name,
                AssumeRolePolicyDocument=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }),
                Description='ECS Task Role for Forex Trading Bot',
                Tags=[{'Key': 'Environment', 'Value': self.config.tier.value}]
            )
            
            # Execution role for ECS tasks
            execution_role_response = iam.create_role(
                RoleName=execution_role_name,
                AssumeRolePolicyDocument=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }),
                Description='ECS Task Execution Role for Forex Trading Bot',
                Tags=[{'Key': 'Environment', 'Value': self.config.tier.value}]
            )
            
            # Attach managed policies to execution role
            iam.attach_role_policy(
                RoleName=execution_role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
            )
            
            # Create and attach custom policy for task role
            task_policy_document = {
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
                            "secretsmanager:GetSecretValue",
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:ListBucket"
                        ],
                        "Resource": "*"
                    }
                ]
            }
            
            task_policy_name = f"forex-bot-task-policy-{self.config.tier.value}"
            iam.put_role_policy(
                RoleName=task_role_name,
                PolicyName=task_policy_name,
                PolicyDocument=json.dumps(task_policy_document)
            )
            
            self.resources['task_role_arn'] = task_role_response['Role']['Arn']
            self.resources['execution_role_arn'] = execution_role_response['Role']['Arn']
            
            logger.info("AWS security configuration completed")
            
            return DeploymentResult(
                success=True,
                service_type=ServiceType.NETWORKING,
                resource_id=task_role_name,
                metadata={
                    'task_role_arn': task_role_response['Role']['Arn'],
                    'execution_role_arn': execution_role_response['Role']['Arn']
                }
            )
            
        except Exception as e:
            logger.error(f"AWS security configuration failed: {e}")
            return DeploymentResult(
                success=False,
                service_type=ServiceType.NETWORKING,
                error_message=str(e)
            )

    def _generate_deployment_summary(self, results: Dict[str, DeploymentResult]) -> Dict[str, Any]:
        """Generate deployment summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'provider': self.config.provider.value,
                'tier': self.config.tier.value,
                'region': self.config.region,
                'successful_services': 0,
                'failed_services': 0,
                'services': {},
                'endpoints': {},
                'estimated_monthly_cost': self._estimate_monthly_cost()
            }
            
            for service_name, result in results.items():
                summary['services'][service_name] = {
                    'success': result.success,
                    'resource_id': result.resource_id,
                    'endpoint': result.endpoint,
                    'error': result.error_message
                }
                
                if result.success:
                    summary['successful_services'] += 1
                    if result.endpoint:
                        summary['endpoints'][service_name] = result.endpoint
                else:
                    summary['failed_services'] += 1
            
            summary['overall_status'] = 'SUCCESS' if summary['failed_services'] == 0 else 'PARTIAL_SUCCESS' if summary['successful_services'] > 0 else 'FAILED'
            
            return summary
            
        except Exception as e:
            logger.error(f"Deployment summary generation failed: {e}")
            return {'error': str(e)}

    def _estimate_monthly_cost(self) -> float:
        """Estimate monthly infrastructure cost"""
        try:
            # Simple cost estimation based on resource types and region
            cost_map = {
                't3.medium': 30.0,
                't3.large': 60.0,
                't3.xlarge': 120.0,
                'db.t3.medium': 50.0,
                'db.t3.large': 100.0,
                'cache.t3.medium': 40.0,
                'cache.t3.large': 80.0
            }
            
            base_cost = cost_map.get(self.config.instance_type, 30.0)
            db_cost = cost_map.get(self.config.db_instance_type, 50.0)
            cache_cost = cost_map.get(self.config.cache_node_type, 40.0)
            
            total_cost = (
                base_cost * self.config.desired_instances +  # EC2 instances
                db_cost +                                   # RDS instance
                cache_cost +                                # ElastiCache
                20.0 +                                      # ALB
                10.0 +                                      # S3
                5.0 +                                       # CloudWatch
                5.0                                         # Data transfer
            )
            
            return round(total_cost, 2)
            
        except Exception as e:
            logger.error(f"Cost estimation failed: {e}")
            return 0.0

    def _rollback_deployment(self):
        """Rollback failed deployment"""
        try:
            logger.info("Starting deployment rollback...")
            
            # Delete resources in reverse order
            if 'alb_arn' in self.resources:
                self.clients['elbv2'].delete_load_balancer(LoadBalancerArn=self.resources['alb_arn'])
            
            if 'cache_cluster_id' in self.resources:
                self.clients['elasticache'].delete_cache_cluster(CacheClusterId=self.resources['cache_cluster_id'])
            
            if 'db_identifier' in self.resources:
                self.clients['rds'].delete_db_instance(
                    DBInstanceIdentifier=self.resources['db_identifier'],
                    SkipFinalSnapshot=True
                )
            
            if 'asg_name' in self.resources:
                self.clients['autoscaling'].delete_auto_scaling_group(
                    AutoScalingGroupName=self.resources['asg_name'],
                    ForceDelete=True
                )
            
            if 'ecs_cluster_arn' in self.resources:
                self.clients['ecs'].delete_cluster(cluster=self.resources['ecs_cluster_arn'])
            
            if 'vpc_id' in self.resources:
                # Delete VPC and all associated resources
                self._delete_vpc_resources()
            
            logger.info("Deployment rollback completed")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    def _delete_vpc_resources(self):
        """Delete VPC and all associated resources"""
        try:
            ec2 = self.clients['ec2']
            vpc_id = self.resources['vpc_id']
            
            # Delete internet gateway
            if 'igw_id' in self.resources:
                ec2.detach_internet_gateway(
                    InternetGatewayId=self.resources['igw_id'],
                    VpcId=vpc_id
                )
                ec2.delete_internet_gateway(InternetGatewayId=self.resources['igw_id'])
            
            # Delete subnets
            if 'subnets' in self.resources:
                for subnet_id in self.resources['subnets'].values():
                    ec2.delete_subnet(SubnetId=subnet_id)
            
            # Delete VPC
            ec2.delete_vpc(VpcId=vpc_id)
            
        except Exception as e:
            logger.error(f"VPC deletion failed: {e}")

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'provider': self.config.provider.value,
                'tier': self.config.tier.value,
                'region': self.config.region,
                'resources_deployed': list(self.resources.keys()),
                'services_healthy': self._check_services_health(),
                'estimated_cost': self._estimate_monthly_cost()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {'error': str(e)}

    def _check_services_health(self) -> Dict[str, bool]:
        """Check health of deployed services"""
        try:
            health_status = {}
            
            if self.config.provider == CloudProvider.AWS:
                # Check ECS cluster
                if 'ecs_cluster_arn' in self.resources:
                    try:
                        response = self.clients['ecs'].describe_clusters(
                            clusters=[self.resources['ecs_cluster_arn']]
                        )
                        health_status['ecs'] = response['clusters'][0]['status'] == 'ACTIVE'
                    except:
                        health_status['ecs'] = False
                
                # Check RDS instance
                if 'db_identifier' in self.resources:
                    try:
                        response = self.clients['rds'].describe_db_instances(
                            DBInstanceIdentifier=self.resources['db_identifier']
                        )
                        health_status['rds'] = response['DBInstances'][0]['DBInstanceStatus'] == 'available'
                    except:
                        health_status['rds'] = False
                
                # Check ElastiCache
                if 'cache_cluster_id' in self.resources:
                    try:
                        response = self.clients['elasticache'].describe_cache_clusters(
                            CacheClusterId=self.resources['cache_cluster_id']
                        )
                        health_status['cache'] = response['CacheClusters'][0]['CacheClusterStatus'] == 'available'
                    except:
                        health_status['cache'] = False
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {}

# Example usage and testing
def main():
    """Test the Cloud Config Manager"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create cloud configuration
    config = CloudConfig(
        provider=CloudProvider.AWS,
        tier=DeploymentTier.DEVELOPMENT,
        region="us-east-1",
        instance_type="t3.small",
        min_instances=1,
        max_instances=3,
        desired_instances=1,
        db_instance_type="db.t3.micro",
        cache_node_type="cache.t3.micro",
        enable_public_ip=True,
        enable_ssl=False,
        enable_monitoring=True,
        enable_backups=True
    )
    
    # Initialize cloud config manager
    manager = CloudConfigManager(config)
    
    try:
        print("Starting cloud infrastructure deployment...")
        
        # Deploy infrastructure
        results = manager.deploy_infrastructure()
        
        # Print results
        print("\n=== DEPLOYMENT RESULTS ===")
        for service, result in results.items():
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"{service}: {status}")
            if result.success and result.endpoint:
                print(f"  Endpoint: {result.endpoint}")
            if result.error_message:
                print(f"  Error: {result.error_message}")
        
        # Get deployment status
        status = manager.get_deployment_status()
        print(f"\n=== DEPLOYMENT STATUS ===")
        print(f"Overall Status: {status.get('services_healthy', {})}")
        print(f"Estimated Monthly Cost: ${status.get('estimated_cost', 0)}")
        
        # Generate deployment summary
        summary = manager._generate_deployment_summary(results)
        print(f"\n=== DEPLOYMENT SUMMARY ===")
        print(f"Successful Services: {summary['successful_services']}")
        print(f"Failed Services: {summary['failed_services']}")
        print(f"Overall Status: {summary['overall_status']}")
        
    except Exception as e:
        print(f"Deployment failed: {e}")
    
    print("Cloud infrastructure deployment test completed!")

if __name__ == "__main__":
    main()
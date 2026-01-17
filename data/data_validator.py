"""
Data Validator for FOREX TRADING BOT
Advanced data validation, cleaning, and quality assurance
"""

import logging
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import json
from scipy import stats
import warnings
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

logger = logging.getLogger(__name__)

class DataQuality(Enum):
    EXCELLENT = "excellent"  # 95-100% quality
    GOOD = "good"          # 85-94% quality  
    FAIR = "fair"          # 70-84% quality
    POOR = "poor"          # 50-69% quality
    UNUSABLE = "unusable"  # <50% quality

class ValidationLevel(Enum):
    BASIC = "basic"        # Basic syntax validation
    BUSINESS = "business"  # Business logic validation
    STATISTICAL = "statistical"  # Statistical anomaly detection
    ADVANCED = "advanced"  # Machine learning anomaly detection

@dataclass
class ValidationRule:
    """Data validation rule definition"""
    name: str
    field: str
    validator: Callable
    severity: str  # "error", "warning", "info"
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Validation result for a single check"""
    rule_name: str
    passed: bool
    message: str
    severity: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    dataset_id: str
    timestamp: float
    overall_quality: DataQuality
    quality_score: float  # 0-100
    total_records: int
    valid_records: int
    invalid_records: int
    missing_values: int
    validation_results: List[ValidationResult]
    anomalies_detected: int
    data_freshness: float  # Hours since last update
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataValidator:
    """
    Advanced data validator with statistical analysis and ML anomaly detection
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Validation rules storage
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.field_validators: Dict[str, List[ValidationRule]] = defaultdict(list)
        
        # Statistical models for anomaly detection
        self.anomaly_detectors = {}
        self.data_profiles = {}
        
        # Quality thresholds
        self.quality_thresholds = {
            DataQuality.EXCELLENT: 95.0,
            DataQuality.GOOD: 85.0,
            DataQuality.FAIR: 70.0,
            DataQuality.POOR: 50.0
        }
        
        # Performance tracking
        self.validation_history = deque(maxlen=1000)
        self.quality_trends = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize built-in validators
        self._initialize_builtin_validators()
        
        logger.info("DataValidator initialized")

    def _initialize_builtin_validators(self):
        """Initialize built-in validation rules"""
        # Price data validators
        self.add_validation_rule(
            ValidationRule(
                name="price_positive",
                field="price",
                validator=self._validate_positive_price,
                severity="error",
                description="Price must be positive"
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="price_sanity_check",
                field="price",
                validator=self._validate_price_sanity,
                severity="error",
                description="Price within reasonable bounds"
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="volume_non_negative",
                field="volume",
                validator=self._validate_non_negative,
                severity="error",
                description="Volume must be non-negative"
            )
        )
        
        # Timestamp validators
        self.add_validation_rule(
            ValidationRule(
                name="timestamp_recency",
                field="timestamp",
                validator=self._validate_timestamp_recency,
                severity="warning",
                description="Timestamp should be recent",
                params={"max_age_hours": 24}
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="timestamp_ordering",
                field="timestamp",
                validator=self._validate_timestamp_ordering,
                severity="error",
                description="Timestamps should be sequential"
            )
        )
        
        # Forex-specific validators
        self.add_validation_rule(
            ValidationRule(
                name="forex_pair_format",
                field="symbol",
                validator=self._validate_forex_pair,
                severity="error",
                description="Forex pair must be valid format"
            )
        )
        
        self.add_validation_rule(
            ValidationRule(
                name="spread_reasonable",
                field="spread",
                validator=self._validate_spread_reasonable,
                severity="warning",
                description="Spread should be within reasonable bounds"
            )
        )

    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.validation_rules[rule.name] = rule
        self.field_validators[rule.field].append(rule)
        logger.debug(f"Added validation rule: {rule.name}")

    def remove_validation_rule(self, rule_name: str):
        """Remove a validation rule"""
        if rule_name in self.validation_rules:
            rule = self.validation_rules[rule_name]
            self.field_validators[rule.field].remove(rule)
            del self.validation_rules[rule_name]
            logger.debug(f"Removed validation rule: {rule_name}")

    async def validate_dataset(self, data: pd.DataFrame, dataset_id: str = None,
                             validation_level: ValidationLevel = ValidationLevel.BUSINESS) -> DataQualityReport:
        """
        Comprehensive dataset validation
        """
        start_time = time.time()
        
        try:
            dataset_id = dataset_id or f"dataset_{int(time.time())}"
            
            logger.info(f"Starting validation for dataset: {dataset_id}")
            
            # Basic structure validation
            structure_results = await self._validate_data_structure(data)
            
            # Field-level validation
            field_results = await self._validate_fields(data)
            
            # Business logic validation
            business_results = await self._validate_business_logic(data)
            
            # Statistical validation based on level
            statistical_results = []
            if validation_level in [ValidationLevel.STATISTICAL, ValidationLevel.ADVANCED]:
                statistical_results = await self._validate_statistical(data)
            
            # ML anomaly detection for advanced level
            ml_results = []
            if validation_level == ValidationLevel.ADVANCED:
                ml_results = await self._detect_anomalies_ml(data)
            
            # Combine all results
            all_results = structure_results + field_results + business_results + statistical_results + ml_results
            
            # Generate quality report
            report = await self._generate_quality_report(
                dataset_id, data, all_results, time.time() - start_time
            )
            
            # Store in history
            self.validation_history.append(report)
            self.quality_trends[dataset_id].append(report.quality_score)
            
            logger.info(f"Validation completed: {dataset_id} - Quality: {report.overall_quality.value}")
            
            return report
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            # Return minimal error report
            return DataQualityReport(
                dataset_id=dataset_id or "unknown",
                timestamp=time.time(),
                overall_quality=DataQuality.UNUSABLE,
                quality_score=0.0,
                total_records=0,
                valid_records=0,
                invalid_records=0,
                missing_values=0,
                validation_results=[],
                anomalies_detected=0,
                data_freshness=float('inf'),
                recommendations=["Validation process failed"]
            )

    async def _validate_data_structure(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate basic data structure"""
        results = []
        
        try:
            # Check if DataFrame
            if not isinstance(data, pd.DataFrame):
                results.append(ValidationResult(
                    rule_name="dataframe_type",
                    passed=False,
                    message="Data must be a pandas DataFrame",
                    severity="error"
                ))
                return results
            
            # Check empty dataset
            if data.empty:
                results.append(ValidationResult(
                    rule_name="non_empty_dataset",
                    passed=False,
                    message="Dataset is empty",
                    severity="error"
                ))
                return results
            
            results.append(ValidationResult(
                rule_name="dataframe_type",
                passed=True,
                message="Data is valid DataFrame",
                severity="info"
            ))
            
            # Check required columns for Forex data
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                results.append(ValidationResult(
                    rule_name="required_columns",
                    passed=False,
                    message=f"Missing required columns: {missing_columns}",
                    severity="error",
                    details={"missing_columns": missing_columns}
                ))
            else:
                results.append(ValidationResult(
                    rule_name="required_columns",
                    passed=True,
                    message="All required columns present",
                    severity="info"
                ))
            
            # Check data types
            type_issues = []
            if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                type_issues.append("timestamp should be datetime")
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    type_issues.append(f"{col} should be numeric")
            
            if type_issues:
                results.append(ValidationResult(
                    rule_name="data_types",
                    passed=False,
                    message=f"Data type issues: {type_issues}",
                    severity="error",
                    details={"type_issues": type_issues}
                ))
            else:
                results.append(ValidationResult(
                    rule_name="data_types",
                    passed=True,
                    message="Data types are correct",
                    severity="info"
                ))
                
        except Exception as e:
            logger.error(f"Data structure validation failed: {e}")
            results.append(ValidationResult(
                rule_name="structure_validation",
                passed=False,
                message=f"Structure validation error: {str(e)}",
                severity="error"
            ))
        
        return results

    async def _validate_fields(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate individual fields using registered rules"""
        results = []
        
        try:
            for field_name, rules in self.field_validators.items():
                if field_name not in data.columns:
                    continue
                
                for rule in rules:
                    try:
                        # Apply validator to each non-null value
                        field_data = data[field_name].dropna()
                        
                        if field_data.empty:
                            continue
                        
                        # Check if validator works on series or individual values
                        validator_signature = rule.validator.__code__.co_varnames
                        
                        if 'series' in validator_signature:
                            # Validator accepts series
                            passed, message, details = rule.validator(field_data, **rule.params)
                        else:
                            # Validator accepts individual values - apply to each
                            individual_results = []
                            for value in field_data:
                                individual_passed, individual_message, individual_details = rule.validator(value, **rule.params)
                                individual_results.append(individual_passed)
                            
                            passed = all(individual_results)
                            message = f"Field validation: {sum(individual_results)}/{len(individual_results)} passed"
                            details = {"passed_count": sum(individual_results), "total_count": len(individual_results)}
                        
                        results.append(ValidationResult(
                            rule_name=rule.name,
                            passed=passed,
                            message=message,
                            severity=rule.severity,
                            details=details
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Field validator {rule.name} failed: {e}")
                        results.append(ValidationResult(
                            rule_name=rule.name,
                            passed=False,
                            message=f"Validator error: {str(e)}",
                            severity="error"
                        ))
                        
        except Exception as e:
            logger.error(f"Field validation failed: {e}")
        
        return results

    async def _validate_business_logic(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Validate business logic rules"""
        results = []
        
        try:
            # OHLC consistency checks
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= Open, Low, Close
                high_violations = data[data['high'] < data[['open', 'low', 'close']].max(axis=1)]
                if not high_violations.empty:
                    results.append(ValidationResult(
                        rule_name="ohlc_high_consistency",
                        passed=False,
                        message=f"High price violations: {len(high_violations)} records",
                        severity="error",
                        details={"violation_count": len(high_violations)}
                    ))
                
                # Low should be <= Open, High, Close
                low_violations = data[data['low'] > data[['open', 'high', 'close']].min(axis=1)]
                if not low_violations.empty:
                    results.append(ValidationResult(
                        rule_name="ohlc_low_consistency",
                        passed=False,
                        message=f"Low price violations: {len(low_violations)} records",
                        severity="error",
                        details={"violation_count": len(low_violations)}
                    ))
            
            # Volume-price relationship
            if all(col in data.columns for col in ['volume', 'close']):
                # Check for zero volume with price movement
                zero_volume_movement = data[
                    (data['volume'] == 0) & 
                    (data['close'].diff().abs() > 0.0001)
                ]
                if not zero_volume_movement.empty:
                    results.append(ValidationResult(
                        rule_name="volume_price_relationship",
                        passed=False,
                        message=f"Zero volume with price movement: {len(zero_volume_movement)} records",
                        severity="warning",
                        details={"violation_count": len(zero_volume_movement)}
                    ))
            
            # Time gaps detection
            if 'timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                time_diffs = data['timestamp'].diff().dropna()
                large_gaps = time_diffs[time_diffs > timedelta(hours=2)]
                
                if not large_gaps.empty:
                    results.append(ValidationResult(
                        rule_name="timestamp_continuity",
                        passed=False,
                        message=f"Large time gaps detected: {len(large_gaps)} gaps > 2 hours",
                        severity="warning",
                        details={"gap_count": len(large_gaps), "max_gap": large_gaps.max().total_seconds() / 3600}
                    ))
                    
        except Exception as e:
            logger.error(f"Business logic validation failed: {e}")
            results.append(ValidationResult(
                rule_name="business_logic_validation",
                passed=False,
                message=f"Business logic validation error: {str(e)}",
                severity="error"
            ))
        
        return results

    async def _validate_statistical(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Statistical validation and anomaly detection"""
        results = []
        
        try:
            # Z-score outlier detection for price columns
            price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in data.columns]
            
            for col in price_columns:
                col_data = data[col].dropna()
                
                if len(col_data) < 10:  # Need sufficient data
                    continue
                
                z_scores = np.abs(zscore(col_data))
                outliers = z_scores > 3  # 3 standard deviations
                
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    results.append(ValidationResult(
                        rule_name=f"statistical_outliers_{col}",
                        passed=False,
                        message=f"Statistical outliers in {col}: {outlier_count} records",
                        severity="warning",
                        details={
                            "column": col,
                            "outlier_count": outlier_count,
                            "outlier_percentage": (outlier_count / len(col_data)) * 100
                        }
                    ))
            
            # Volume spike detection
            if 'volume' in data.columns:
                volume_data = data['volume'].dropna()
                if len(volume_data) > 10:
                    volume_z_scores = np.abs(zscore(volume_data))
                    volume_spikes = volume_z_scores > 4  # Higher threshold for volume
                    
                    spike_count = volume_spikes.sum()
                    if spike_count > 0:
                        results.append(ValidationResult(
                            rule_name="volume_spikes",
                            passed=False,
                            message=f"Volume spikes detected: {spike_count} records",
                            severity="warning",
                            details={"spike_count": spike_count}
                        ))
            
            # Price change distribution
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                if len(returns) > 20:
                    # Check for normality (returns should be roughly normal)
                    _, p_value = stats.normaltest(returns)
                    if p_value < 0.05:  # Not normal distribution
                        results.append(ValidationResult(
                            rule_name="returns_distribution",
                            passed=False,
                            message="Price returns distribution is not normal",
                            severity="info",
                            details={"p_value": p_value}
                        ))
                    
                    # Check for excessive volatility
                    volatility = returns.std()
                    if volatility > 0.05:  # 5% daily volatility threshold
                        results.append(ValidationResult(
                            rule_name="excessive_volatility",
                            passed=False,
                            message=f"Excessive volatility detected: {volatility:.4f}",
                            severity="warning",
                            details={"volatility": volatility}
                        ))
                        
        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")
            results.append(ValidationResult(
                rule_name="statistical_validation",
                passed=False,
                message=f"Statistical validation error: {str(e)}",
                severity="error"
            ))
        
        return results

    async def _detect_anomalies_ml(self, data: pd.DataFrame) -> List[ValidationResult]:
        """Machine learning based anomaly detection"""
        results = []
        
        try:
            # Prepare features for anomaly detection
            feature_columns = []
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data.columns:
                    feature_columns.append(col)
            
            if len(feature_columns) < 2:
                return results  # Need multiple features
            
            feature_data = data[feature_columns].dropna()
            
            if len(feature_data) < 20:  # Need sufficient data for ML
                return results
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42
            )
            
            anomalies = iso_forest.fit_predict(feature_data)
            anomaly_count = (anomalies == -1).sum()
            
            if anomaly_count > 0:
                results.append(ValidationResult(
                    rule_name="ml_anomalies",
                    passed=False,
                    message=f"ML detected anomalies: {anomaly_count} records",
                    severity="warning",
                    details={
                        "anomaly_count": anomaly_count,
                        "anomaly_percentage": (anomaly_count / len(anomalies)) * 100,
                        "method": "IsolationForest"
                    }
                ))
            
            # Store detector for future use
            dataset_id = f"anomaly_detector_{hashlib.md5(str(feature_columns).encode()).hexdigest()}"
            self.anomaly_detectors[dataset_id] = iso_forest
                
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
            results.append(ValidationResult(
                rule_name="ml_anomaly_detection",
                passed=False,
                message=f"ML anomaly detection error: {str(e)}",
                severity="error"
            ))
        
        return results

    async def _generate_quality_report(self, dataset_id: str, data: pd.DataFrame, 
                                     results: List[ValidationResult], 
                                     processing_time: float) -> DataQualityReport:
        """Generate comprehensive quality report"""
        try:
            total_records = len(data)
            
            # Count missing values
            missing_values = data.isnull().sum().sum()
            
            # Calculate validation results
            passed_results = [r for r in results if r.passed]
            failed_results = [r for r in results if not r.passed]
            
            # Count errors and warnings
            error_results = [r for r in failed_results if r.severity == "error"]
            warning_results = [r for r in failed_results if r.severity == "warning"]
            
            # Calculate quality score (0-100)
            base_score = 100.0
            
            # Deductions for errors
            error_deduction = len(error_results) * 10  # 10 points per error
            warning_deduction = len(warning_results) * 5  # 5 points per warning
            
            # Deduction for missing values
            missing_deduction = (missing_values / (total_records * len(data.columns))) * 50 if total_records > 0 else 0
            
            quality_score = max(0, base_score - error_deduction - warning_deduction - missing_deduction)
            
            # Determine overall quality
            overall_quality = DataQuality.UNUSABLE
            for quality, threshold in self.quality_thresholds.items():
                if quality_score >= threshold:
                    overall_quality = quality
                    break
            
            # Calculate data freshness
            data_freshness = float('inf')
            if 'timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                latest_timestamp = data['timestamp'].max()
                if pd.notna(latest_timestamp):
                    if isinstance(latest_timestamp, pd.Timestamp):
                        data_freshness = (pd.Timestamp.now() - latest_timestamp).total_seconds() / 3600
            
            # Generate recommendations
            recommendations = self._generate_recommendations(results, quality_score, data)
            
            report = DataQualityReport(
                dataset_id=dataset_id,
                timestamp=time.time(),
                overall_quality=overall_quality,
                quality_score=quality_score,
                total_records=total_records,
                valid_records=total_records - len(error_results),
                invalid_records=len(error_results),
                missing_values=missing_values,
                validation_results=results,
                anomalies_detected=len([r for r in results if "anomaly" in r.rule_name and not r.passed]),
                data_freshness=data_freshness,
                recommendations=recommendations,
                metadata={
                    "processing_time_seconds": processing_time,
                    "error_count": len(error_results),
                    "warning_count": len(warning_results),
                    "info_count": len([r for r in results if r.severity == "info"])
                }
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            raise

    def _generate_recommendations(self, results: List[ValidationResult], 
                                quality_score: float, data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            failed_rules = [r for r in results if not r.passed]
            
            if quality_score < 70:
                recommendations.append("Dataset quality is poor. Consider acquiring better quality data.")
            
            # Specific recommendations based on failed rules
            error_rules = [r for r in failed_rules if r.severity == "error"]
            for rule in error_rules:
                if "required_columns" in rule.rule_name:
                    recommendations.append("Add missing required columns to dataset")
                elif "data_types" in rule.rule_name:
                    recommendations.append("Fix data type issues in the dataset")
                elif "price_positive" in rule.rule_name:
                    recommendations.append("Investigate and fix negative price values")
            
            warning_rules = [r for r in failed_rules if r.severity == "warning"]
            for rule in warning_rules:
                if "anomaly" in rule.rule_name:
                    recommendations.append("Review detected anomalies for potential data issues")
                elif "volume_spikes" in rule.rule_name:
                    recommendations.append("Investigate volume spikes for data quality issues")
                elif "timestamp_continuity" in rule.rule_name:
                    recommendations.append("Address large time gaps in the data")
            
            # Data freshness recommendation
            if 'timestamp' in data.columns:
                latest_time = data['timestamp'].max()
                if pd.notna(latest_time):
                    age_hours = (pd.Timestamp.now() - latest_time).total_seconds() / 3600
                    if age_hours > 24:
                        recommendations.append(f"Data is {age_hours:.1f} hours old. Consider updating with more recent data.")
            
            # Add general recommendations if none specific
            if not recommendations and quality_score < 90:
                recommendations.append("Dataset quality is acceptable but could be improved with additional validation")
                
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Error generating specific recommendations")
        
        return recommendations

    # Built-in validator functions
    def _validate_positive_price(self, value: float) -> Tuple[bool, str, Dict]:
        """Validate that price is positive"""
        if value <= 0:
            return False, f"Price must be positive, got {value}", {"value": value}
        return True, "Price is positive", {}

    def _validate_price_sanity(self, value: float) -> Tuple[bool, str, Dict]:
        """Validate price is within reasonable bounds"""
        if value < 0.5 or value > 2.0:  # Reasonable EUR/USD bounds
            return False, f"Price {value} outside reasonable bounds", {"value": value}
        return True, "Price within reasonable bounds", {}

    def _validate_non_negative(self, value: float) -> Tuple[bool, str, Dict]:
        """Validate value is non-negative"""
        if value < 0:
            return False, f"Value must be non-negative, got {value}", {"value": value}
        return True, "Value is non-negative", {}

    def _validate_timestamp_recency(self, value: pd.Timestamp, max_age_hours: int = 24) -> Tuple[bool, str, Dict]:
        """Validate timestamp is recent"""
        if pd.isna(value):
            return False, "Timestamp is missing", {}
        
        age_hours = (pd.Timestamp.now() - value).total_seconds() / 3600
        if age_hours > max_age_hours:
            return False, f"Timestamp is {age_hours:.1f} hours old", {"age_hours": age_hours}
        return True, "Timestamp is recent", {}

    def _validate_timestamp_ordering(self, series: pd.Series) -> Tuple[bool, str, Dict]:
        """Validate timestamps are sequential"""
        if series.is_monotonic_increasing:
            return True, "Timestamps are sequential", {}
        else:
            non_sequential = series[series.diff() < pd.Timedelta(0)]
            return False, f"Found {len(non_sequential)} non-sequential timestamps", {
                "non_sequential_count": len(non_sequential)
            }

    def _validate_forex_pair(self, value: str) -> Tuple[bool, str, Dict]:
        """Validate forex pair format"""
        if not isinstance(value, str):
            return False, "Forex pair must be string", {}
        
        pattern = r'^[A-Z]{3}/[A-Z]{3}$'
        if re.match(pattern, value):
            return True, "Valid forex pair format", {}
        else:
            return False, f"Invalid forex pair format: {value}", {"value": value}

    def _validate_spread_reasonable(self, value: float) -> Tuple[bool, str, Dict]:
        """Validate spread is reasonable"""
        if value < 0:
            return False, "Spread cannot be negative", {"value": value}
        elif value > 0.01:  # 100 pips
            return False, f"Spread {value} seems too high", {"value": value}
        return True, "Spread is reasonable", {}

    async def get_validation_history(self, dataset_id: str = None, limit: int = 100) -> List[DataQualityReport]:
        """Get validation history for a dataset"""
        if dataset_id:
            return list(self.quality_trends[dataset_id])[-limit:]
        else:
            return list(self.validation_history)[-limit:]

    async def get_quality_trend(self, dataset_id: str, window: int = 30) -> Dict[str, Any]:
        """Get quality trend analysis for a dataset"""
        trends = list(self.quality_trends[dataset_id])
        
        if len(trends) < 2:
            return {"trend": "insufficient_data", "change": 0.0}
        
        recent_trends = trends[-window:]
        if len(recent_trends) < 2:
            return {"trend": "insufficient_data", "change": 0.0}
        
        # Calculate trend
        oldest = recent_trends[0]
        newest = recent_trends[-1]
        change = newest - oldest
        
        if change > 1.0:
            trend = "improving"
        elif change < -1.0:
            trend = "deteriorating"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change": change,
            "current_quality": newest,
            "periods_analyzed": len(recent_trends)
        }

    async def validate_real_time(self, data_point: Dict, dataset_id: str) -> bool:
        """Validate a single real-time data point"""
        try:
            # Convert to DataFrame for validation
            df = pd.DataFrame([data_point])
            
            # Run basic validation
            report = await self.validate_dataset(df, dataset_id, ValidationLevel.BASIC)
            
            return report.overall_quality in [DataQuality.EXCELLENT, DataQuality.GOOD, DataQuality.FAIR]
            
        except Exception as e:
            logger.error(f"Real-time validation failed: {e}")
            return False

# Example usage and testing
async def main():
    """Test the Data Validator"""
    
    validator = DataValidator()
    
    try:
        # Create sample Forex data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'open': np.random.uniform(1.05, 1.10, 100),
            'high': np.random.uniform(1.10, 1.15, 100),
            'low': np.random.uniform(1.00, 1.05, 100),
            'close': np.random.uniform(1.05, 1.10, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'symbol': ['EUR/USD'] * 100
        })
        
        # Add some intentional issues
        sample_data.loc[5, 'close'] = -1.0  # Negative price
        sample_data.loc[10, 'volume'] = -100  # Negative volume
        sample_data.loc[15:20, 'timestamp'] = pd.NaT  # Missing timestamps
        
        # Run validation
        report = await validator.validate_dataset(
            sample_data, 
            "test_dataset",
            ValidationLevel.ADVANCED
        )
        
        print(f"Quality Score: {report.quality_score:.1f}%")
        print(f"Overall Quality: {report.overall_quality.value}")
        print(f"Valid Records: {report.valid_records}/{report.total_records}")
        print(f"Errors: {len([r for r in report.validation_results if not r.passed and r.severity == 'error'])}")
        
        # Print failed validations
        print("\nFailed Validations:")
        for result in report.validation_results:
            if not result.passed:
                print(f"  - {result.rule_name}: {result.message} ({result.severity})")
        
        print(f"\nRecommendations: {report.recommendations}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
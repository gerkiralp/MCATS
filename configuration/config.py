#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Module for MCATS (Multi-Currency Automated Trading System)

This module provides class-based configuration using Python dataclasses
as specified in the MCATS Technical Specification v4.9.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class RiskConfig:
    """Risk management configuration parameters."""
    
    drawdown_threshold: float = 0.05
    margin_rates: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': 0.02,
        'GBPUSD': 0.02,
        'USDJPY': 0.01,
        'USDCHF': 0.01,
        'AUDUSD': 0.02,
        'NZDUSD': 0.02,
        'USDCAD': 0.02,
        'USDNOK': 0.02,
        'USDSEK': 0.02,
    })
    max_position_size: float = 100.0  # Normalized units


@dataclass
class FeatureConfig:
    """Feature engineering configuration parameters."""
    
    base_features: List[str] = field(default_factory=lambda: ['log_returns', 'RSI', 'MACD'])
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        'RSI': 14,
        'MACD_fast': 12,
        'MACD_slow': 26
    })
    scaling_params: Dict[str, float] = field(default_factory=lambda: {
        'normalized_price_scale': 100.0
    })


@dataclass
class DataPipelineConfig:
    """Data pipeline configuration parameters."""
    
    raw_data_dir: str = "data/raw_tick_data"
    ohlcv_output_dir: str = "data/ohlcv_data"
    bar_size: str = '1min'
    currency_pairs: List[str] = field(default_factory=lambda: [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
        'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK'
    ])


@dataclass
class MarketSimulatorConfig:
    """Market simulator configuration parameters."""
    
    spread_table: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': 0.0001,
        'GBPUSD': 0.00012,
        'USDJPY': 0.015,
        'USDCHF': 0.009,
        'AUDUSD': 0.00011,
        'NZDUSD': 0.00013,
        'USDCAD': 0.00014,
        'USDNOK': 0.0018,
        'USDSEK': 0.0019
    })
    commission_rate: float = 0.000025  # 0.0025% commission per trade


@dataclass
class TrainingPipelineConfig:
    """Training pipeline configuration parameters."""
    
    episodes_per_training_run: int = 100
    rl_algorithm: str = "PPO"  # Example default RL algorithm
    # Other RL hyperparameters would be added here


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration parameters."""
    
    failure_threshold: int = 5  # Number of consecutive failures before opening circuit
    recovery_timeout: int = 30  # Seconds to wait before transitioning to half-open state


@dataclass
class SystemConfig:
    """
    Main configuration class for the MCATS system.
    
    Combines all sub-configurations into a single configuration object.
    """
    
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    data_pipeline_config: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    market_simulator_config: MarketSimulatorConfig = field(default_factory=MarketSimulatorConfig)
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    # Project paths
    project_root: Path = None
    
    def __post_init__(self):
        """Perform post-initialization setup."""
        if self.project_root is None:
            # Try to determine project root (assuming this module is in project/configuration/)
            try:
                self.project_root = Path(__file__).parent.parent
            except:
                # Fallback to current working directory
                self.project_root = Path.cwd()


def load_config(config_path=None) -> SystemConfig:
    """
    Load system configuration.
    
    If config_path is specified, loads configuration from a JSON or YAML file.
    Otherwise, returns default configuration.
    
    Args:
        config_path (str, optional): Path to configuration file. If None, uses defaults.
        
    Returns:
        SystemConfig: Configuration object for the MCATS system.
    """
    if config_path is None:
        return SystemConfig()
    
    # TODO: Implement loading from JSON/YAML files
    # For now, just return default configuration
    return SystemConfig()


# Default configuration instance
default_config = SystemConfig() 
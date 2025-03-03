**MCATS: Multi-Currency Automated Trading System**
**Technical Specification Document v4.9**

**Introduction**

This document provides the technical specification for the Multi-Currency Automated Trading System (MCATS). MCATS is designed as a modular, scalable, and robust platform for developing and deploying reinforcement learning (RL) agents for automated trading in multi-currency foreign exchange (FX) markets.

**Project Goals:**

*   Develop a Reinforcement Learning based Trading System: Leverage RL to create intelligent agents capable of learning optimal trading strategies directly from market data.
*   Multi-Currency Trading:  Enable agents to trade across a portfolio of major currency pairs, capturing potential diversification benefits and cross-currency trading opportunities.
*   Modular and Scalable Architecture: Design a system with clearly defined layers and interfaces to facilitate independent development, testing, and future enhancements.
*   Realistic Simulation Environment: Create a simulation environment that accurately models market dynamics, transaction costs (spread and commission), and risk factors (drawdown).
*   Production Readiness (Future Goal):  Architect the system with considerations for eventual deployment in a production trading environment, including robust risk management and efficient market execution.

**Target Audience:**

This document is intended for software engineers, machine learning engineers, quantitative analysts, and researchers with expertise in:

*   Software Engineering:  Experience in designing and implementing modular, object-oriented systems in Python.
*   Machine Learning and Reinforcement Learning:  Understanding of fundamental RL concepts, neural networks, and training pipelines.
*   Financial Markets and Algorithmic Trading:  Familiarity with FX markets, trading terminology (OHLCV, tick data, bid/ask spread, commission, Net Liquidation Value, exposure), and common trading strategies and risk metrics (drawdown).
*   Data Engineering: Experience with handling large datasets, efficient data storage (Parquet, NPZ), and data processing pipelines.

**Architecture Overview**

MCATS employs a layered architecture to clearly separate concerns and promote modularity, scalability, and maintainability. The system is structured into the following distinct layers:

*   **Data Layer:**  (See Section 2) Responsible for raw data ingestion and processing into **Parquet tick data files**, and generation of **1-minute OHLCV Parquet data files**. Functionality to generate longer period OHLCV bars from 1-minute data is included.
*   **Feature Engineering System:** (See Section 3) Extracts relevant features from 1-minute OHLCV data, with potential for **distributed computation (Dask/Ray)** and **dynamic feature normalization**.
*   **Trading Agent Layer (Decision Layer):** (See Section 4) Houses reinforcement learning agents.
*   **Order & Risk Management Layer (Broker & Internal Fill Layer):** (See Section 5) Acts as an intermediary, enforcing risk management, simulating internal fills, and aggregating positions, incorporating **circuit breaker** patterns for resilience.
*   **Market Execution Layer (Aggregation & Real Market Interface):** (See Section 6) Handles market execution of aggregated positions.
*   **Training Pipeline:** (See Section 7) Orchestrates the end-to-end training process, integrated with **experiment tracking tools (MLflow/Weights & Biases)**.
*   **User Interface (UI) & Command Line Interface (CLI):** (See Section 9) Provides user interaction and system control via both CLI and a planned Web-based UI.
*   **Compute Infrastructure:** (See Section 10) Outlines the plan for scalable compute infrastructure, starting with Google Colab and transitioning to Cloud VMs.

**(Sequence Diagrams - Example: `train_episode` workflow)**

```mermaid
sequenceDiagram
    participant TrainingPipeline
    participant OrderRiskManagementLayer
    participant MarketExecutionLayer
    participant TradingAgentLayer
    participant FeatureEngine
    participant DataLayer

    TrainingPipeline->>DataLayer: Load Episode Data (Parquet)
    TrainingPipeline->>FeatureEngine: Generate Features (OHLCV Data)
    loop For each bar in episode
        TrainingPipeline->>TradingAgentLayer: Get State Vector
        TradingAgentLayer->>TrainingPipeline: State Vector
        TrainingPipeline->>TradingAgentLayer: agent.get_action(state)
        TradingAgentLayer->>TrainingPipeline: target_exposure_vector
        TrainingPipeline->>OrderRiskManagementLayer: process_agent_actions(current_prices)
        OrderRiskManagementLayer->>RiskManager: check_risk_limits(agent)
        alt Risk Limits Passed
            OrderRiskManagementLayer->>MarketExecutionLayer: _execute_orders (internal fills)
            MarketExecutionLayer-->>OrderRiskManagementLayer: internal_fills, transaction_costs
            OrderRiskManagementLayer->>MarketExecutionLayer: _calculate_nlv_change
            MarketExecutionLayer-->>OrderRiskManagementLayer: nlv_change_vector
            OrderRiskManagementLayer->>MarketExecutionLayer: _update_exposure_vector
            MarketExecutionLayer-->>OrderRiskManagementLayer: updated_exposure_vector
        else Stop-Out Triggered
            OrderRiskManagementLayer->>OrderRiskManagementLayer: Generate zero exposure action
        end
        OrderRiskManagementLayer->>TrainingPipeline: agent_execution_results, aggregated_internal_positions
        TrainingPipeline->>MarketExecutionLayer: execute_aggregated_positions(aggregated_internal_positions, current_prices)
        MarketExecutionLayer->>TrainingPipeline: market_execution_results
        TrainingPipeline->>TradingAgentLayer: agent.calculate_reward(nlv_change_scalar)
        TradingAgentLayer->>TrainingPipeline: reward
        TrainingPipeline->>TradingAgentLayer: agent.update(state, action, reward, next_state)
    end
    TrainingPipeline->>TrainingPipeline: Generate Episode Summary
(Sequence Diagrams - Example: Agent - ORM - Market Execution Interaction for a single time step)

Code snippet

sequenceDiagram
    participant TradingAgent
    participant OrderRiskManagementLayer
    participant MarketExecutionLayer

    TradingAgent->>OrderRiskManagementLayer: target_exposure_vector (get_action output)
    OrderRiskManagementLayer->>RiskManager: check_risk_limits(agent)
    alt Risk Limits Passed
        OrderRiskManagementLayer->>OrderRiskManagementLayer: Enforce Currency Constraints
        OrderRiskManagementLayer->>OrderRiskManagementLayer: _convert_exposure_change_to_internal_orders
        OrderRiskManagementLayer->>MarketExecutionLayer: _execute_orders (internal fills)
        MarketExecutionLayer-->>OrderRiskManagementLayer: internal_fills, transaction_costs
        OrderRiskManagementLayer->>MarketExecutionLayer: _calculate_nlv_change
        MarketExecutionLayer-->>OrderRiskManagementLayer: nlv_change_vector
        OrderRiskManagementLayer->>MarketExecutionLayer: _update_exposure_vector
        MarketExecutionLayer-->>OrderRiskManagementLayer: updated_exposure_vector
        OrderRiskManagementLayer->>TradingAgent: agent_execution_results (feedback)
    else Stop-Out Triggered
        OrderRiskManagementLayer->>TradingAgent: Stop-out notification (optional)
    end
    OrderRiskManagementLayer->>MarketExecutionLayer: aggregated_internal_positions (from all agents)
    MarketExecutionLayer->>MarketExecutionLayer: execute_aggregated_positions
    MarketExecutionLayer->>OrderRiskManagementLayer: market_execution_results (optional feedback)

Error Handling and Validation:

Robust error handling and input validation are critical throughout the MCATS system.  Each layer should implement appropriate error handling mechanisms to gracefully manage unexpected data, system failures, or invalid inputs.  Key areas for error handling and validation include:

Data Pipeline:
Raw Data Parsing Errors: Handle potential errors during JSON parsing of raw tick data. Implement logging and error reporting for corrupted or invalid data files.
Timestamp Overlap Detection: Implement robust detection and handling of timestamp overlaps in raw data. Decide on a strategy for resolving overlaps (e.g., logging, skipping duplicate data, averaging prices).
Data Integrity Checks: Implement checks to validate the integrity of data after each processing step (e.g., data type validation, range checks, missing value checks).
Feature Engineering System:
Invalid Feature Configuration: Validate the feature_config dictionary to ensure that requested features are valid and supported.
Numerical Stability: Implement checks for numerical issues (e.g., division by zero, NaNs, Infs) during feature calculations, especially for technical indicators and volatility measures.
Trading Agent Layer:
Invalid State Input: Validate the input state vector to get_action() to ensure it conforms to the expected format and data types.
Model Loading Errors: Implement error handling for loading neural network models, including checks for model file existence and compatibility.
Order & Risk Management Layer:
Invalid Target Exposure Vectors: Validate target_exposure_vector received from agents to ensure it has the correct shape and data types.
Risk Rule Evaluation Errors: Implement error handling for risk rule evaluation and potential exceptions during risk checks.
Market Simulator Errors: Handle potential errors from the injected MarketSimulator instance (e.g., connection errors, invalid order requests).
Market Execution Layer:
Broker API Errors (Future): In a production system, robust error handling for brokerage API interactions is paramount, including handling connection errors, authentication failures, rate limiting, and order rejection messages.
Execution Simulation Errors: While simpler for the prototype simulator, implement basic error handling for unexpected conditions during execution simulation.
Training Pipeline:
Data Loading Errors: Handle potential errors during loading of episode data (e.g., file not found, corrupted Parquet files).
Layer Initialization Errors: Implement error handling during initialization of OrderRiskManagementLayer, FeatureEngine, etc., to ensure that dependencies are correctly injected.
Error Logging:

A comprehensive logging system should be implemented to record errors, warnings, and informational messages throughout the MCATS system.  Logs should include timestamps, error levels, source modules, and detailed error messages to facilitate debugging and system monitoring.  Consider using Python's logging module for structured logging.

Configuration Management Enhancement: Class-Based Configuration

Instead of relying solely on YAML/JSON configuration files, MCATS v4.9 will adopt a class-based configuration system using Python dataclasses. This approach offers several advantages:

Type Safety: dataclasses enforce type hints, ensuring that configuration parameters are of the expected data types, reducing configuration errors and improving code robustness.
Validation: dataclasses and libraries like pydantic (consider for future iterations) can be used to implement data validation logic within the configuration classes, ensuring that configuration parameters are within valid ranges or adhere to specific constraints.
Code Readability and Maintainability: Configuration classes provide a more structured and self-documenting way to manage configuration parameters compared to flat dictionaries loaded from YAML/JSON.
IDE Support: Type hints in dataclasses enable better IDE support for autocompletion, type checking, and refactoring of configuration parameters.
Example Configuration Classes (Illustrative):

Python

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class RiskConfig:
    drawdown_threshold: float = 0.05
    margin_rates: Dict[str, float] = field(default_factory=lambda: { # Example defaults
        'EURUSD': 0.02,
        'GBPUSD': 0.02,
        'USDJPY': 0.01,
        # ... other currency pairs
    })
    max_position_size: float = 100.0  # Normalized units

@dataclass
class FeatureConfig:
    base_features: List[str] = field(default_factory=lambda: ['log_returns', 'RSI', 'MACD'])
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        'RSI': 14,
        'MACD_fast': 12,
        'MACD_slow': 26
        # ... other indicator lookback periods
    })
    scaling_params: Dict[str, float] = field(default_factory=lambda: { # Example scaling parameters
        'normalized_price_scale': 100.0
    })

@dataclass
class DataPipelineConfig:
    raw_data_dir: str = "data/raw_tick_data"
    ohlcv_output_dir: str = "data/ohlcv_data"
    bar_size: str = '5min'
    currency_pairs: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK'])

@dataclass
class MarketSimulatorConfig:
    spread_table: Dict[str, float] = field(default_factory=lambda: { # Example spread table
        'EURUSD': 0.0001,
        'GBPUSD': 0.00012,
        # ... other currency pairs
    })
    commission_rate: float = 0.000025

@dataclass
class TrainingPipelineConfig:
    episodes_per_training_run: int = 100
    rl_algorithm: str = "PPO" # Example default RL algorithm
    # ... RL algorithm hyperparameters, etc.
Configuration Loading and Usage:

Configuration classes will be instantiated and passed to the relevant layers during system initialization. Configuration values can be accessed using standard attribute access (e.g., config.risk_config.drawdown_threshold).  While dataclasses are used for configuration structure, YAML or JSON files can still be used for storing the configuration data and loaded into these classes at runtime. Libraries like hydra can be considered in the future for more advanced configuration management.

Performance Optimization: Distributed Feature Engineering with Dask or Ray

To address potential performance bottlenecks in feature engineering, especially when processing large datasets or implementing computationally intensive Tier 2 and Tier 3 features, MCATS v4.9 will explore integrating distributed computing frameworks like Dask or Ray.

Dask Integration (Recommended for Prototype): Dask is a Python library for parallel computing that integrates well with Pandas and NumPy, making it a relatively straightforward option for distributing feature engineering workloads. Dask DataFrames can be used to parallelize feature calculations across multiple cores or machines.

Example Dask Integration in FeatureEngine (Illustrative):

Python

import dask.dataframe as dd
import pandas as pd

class FeatureEngine:
    # ... (rest of FeatureEngine class) ...

    def generate_features_distributed(self, ohlcv_data_vaex, feature_config):
        """
        Generates features using distributed computation with Dask.

        Args:
            ohlcv_data_vaex (vaex.DataFrame): Vaex DataFrame of OHLCV data.
            feature_config (dict): Feature configuration.

        Returns:
            vaex.DataFrame: Vaex DataFrame with features appended.
        """
        # Convert Vaex DataFrame to Pandas DataFrame (for Dask compatibility - optimize later if needed)
        ohlcv_data_pd = ohlcv_data_vaex.to_pandas_df()

        # Create Dask DataFrame from Pandas DataFrame, partition as needed
        ddf = dd.from_pandas(ohlcv_data_pd, npartitions=4) # Example: 4 partitions

        # Define a function to calculate features on a Pandas DataFrame partition
        def _calculate_features_partition(partition_df, feature_config):
            partition_vaex = vaex.from_pandas(partition_df) # Convert partition back to Vaex for feature calculations
            return self.generate_features(partition_vaex, feature_config).to_pandas_df() # Calculate features and return as Pandas

        # Use Dask's map_partitions to apply feature calculation in parallel
        features_ddf = ddf.map_partitions(_calculate_features_partition, feature_config, meta=pd.DataFrame(columns=ohlcv_data_vaex.column_names + list(feature_config.keys()))) # Define metadata for output DataFrame

        # Compute the Dask DataFrame to get a Pandas DataFrame
        features_pd = features_ddf.compute()

        # Convert the Pandas DataFrame back to Vaex DataFrame
        features_vaex = vaex.from_pandas(features_pd)
        return features_vaex

Ray Integration (Consider for Future Scalability): Ray is a more general-purpose distributed computing framework that is well-suited for scaling ML workloads, including RL training. Ray could be considered for future scalability, especially if the complexity of agent training or feature engineering increases significantly.  Ray offers more advanced features for distributed task scheduling, actor-based concurrency, and cluster management.

Experiment Tracking Integration: MLflow or Weights & Biases

To facilitate systematic experimentation, hyperparameter tuning, and model management, MCATS v4.9 will integrate an experiment tracking tool, either MLflow or Weights & Biases (W&B).

MLflow (Recommended for Prototype): MLflow is an open-source platform for managing the ML lifecycle. It offers features for tracking experiments, logging parameters, metrics, and artifacts (models, datasets), and comparing runs.  MLflow is a good choice for the prototype due to its open-source nature and ease of integration.

Example MLflow Integration in TrainingPipeline (Illustrative):

Python

import mlflow

class TrainingPipeline:
    def __init__(self, config, order_risk_management_layer, feature_engine, risk_manager, market_simulator):
        self.config = config # Store configuration
        self.order_risk_management_layer = order_risk_management_layer
        self.feature_engine = feature_engine
        self.risk_manager = risk_manager
        self.market_simulator = market_simulator
        self.agent_ids = []

        mlflow.set_experiment("MCATS_Training_Experiments") # Set MLflow experiment name

    def train_episode(self, episode_data):
        with mlflow.start_run(): # Start MLflow run for each episode
            mlflow.log_params({"rl_algorithm": self.config.training_pipeline_config.rl_algorithm,  # Log config parameters
                                "drawdown_threshold": self.config.risk_config.drawdown_threshold,
                                # ... other relevant parameters from config classes ...
                                })

            episode_summary = super().train_episode(episode_data) # Call base class train_episode

            mlflow.log_metrics({"episode_reward_aggregated": episode_summary['aggregated_episode_reward'], # Log metrics
                                "final_nlv_scalar_aggregated": episode_summary['aggregated_final_nlv_scalar'],
                                "max_episode_drawdown": episode_summary['max_episode_drawdown'],
                                "bar_count": episode_summary['bar_count']})

            # Log trained agent models as artifacts (future implementation)
            # mlflow.log_artifact("path/to/trained_agent_model.pth", artifact_path="models")

            return episode_summary
Weights & Biases (Consider for Advanced Experiment Tracking): Weights & Biases (W&B) is a more feature-rich experiment tracking platform, offering interactive dashboards, hyperparameter optimization tools, and collaboration features. W&B could be considered for more advanced experiment tracking and hyperparameter tuning in later stages of development.

Circuit Breakers in Order & Risk Management Layer

To enhance system resilience and prevent cascading failures, especially in scenarios with multiple agents or potential issues in downstream layers (e.g., Market Execution Layer), MCATS v4.9 will implement circuit breaker patterns in the Order & Risk Management Layer.

Circuit Breaker Logic: A circuit breaker will monitor the health and responsiveness of the Market Execution Layer (or other critical dependencies). If the Market Execution Layer becomes unresponsive, experiences errors, or exceeds a defined failure threshold, the circuit breaker will "open," preventing further order processing from being sent to the Market Execution Layer for a specified period.
Fallback Mechanism: When the circuit breaker is open, the Order & Risk Management Layer will implement a fallback mechanism, such as:
Generating Emergency Close Positions: Instructing the Market Execution Layer to close all existing positions for all agents (or a subset of agents).
Halting New Order Generation: Temporarily pausing the processing of new target_exposure_vectors from agents.
Logging and Alerting: Logging the circuit breaker activation event and triggering alerts to notify system administrators.
Circuit Breaker States: A typical circuit breaker implementation will have three states:
Closed: Normal operation. Requests are passed through to the Market Execution Layer.
Open: Circuit is broken due to failures. Requests are blocked, and fallback mechanism is activated.
Half-Open: After a timeout period in the "Open" state, the circuit breaker transitions to "Half-Open." A limited number of test requests are allowed to pass through to check if the downstream system has recovered. If requests succeed, the circuit breaker "closes" again. If requests fail, it returns to the "Open" state.
Example Circuit Breaker Implementation in OrderRiskManagementLayer (Illustrative - using a simplified circuit breaker class):

Python

class CircuitBreaker: # Simplified Circuit Breaker - Use a robust library for production
    def __init__(self, failure_threshold=5, recovery_timeout=30): # Failure threshold (consecutive failures), recovery timeout (seconds)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED" # "CLOSED", "OPEN", "HALF_OPEN"
        self.last_failure_time = None

    def is_open(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN" # Transition to half-open after timeout
                self.failure_count = 0 # Reset failure count for half-open test
            return True
        return False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN" # Open circuit breaker if failure threshold reached

    def record_success(self):
        if self.state == "HALF_OPEN":
            self.state = "CLOSED" # Close circuit breaker if half-open test succeeds
        self.failure_count = 0 # Reset failure count on success

    def reset(self): # For manual reset
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None


import time

class OrderRiskManagementLayer:
    def __init__(self, risk_manager, market_simulator, circuit_breaker=None): # Inject CircuitBreaker
        self.risk_manager = risk_manager
        self.market_simulator = market_simulator
        self.agents = {}
        self.circuit_breaker = circuit_breaker if circuit_breaker else CircuitBreaker() # Inject CircuitBreaker instance or use default

    def process_agent_actions(self, current_prices_normalized):
        if self.circuit_breaker.is_open():
            print("Circuit Breaker Open! Generating emergency close positions.") # Log circuit breaker event
            # Implement emergency close positions logic here (return zero exposure, or generate close orders)
            # ... fallback_aggregated_positions = self._generate_emergency_close_positions() ...
            # return {}, fallback_aggregated_positions # Return empty agent results, fallback positions

        aggregated_internal_positions = np.zeros(9, dtype=np.float32)
        agent_execution_results = {}

        for agent_id, agent_data in self.agents.items():
            # ... (rest of agent action processing logic from v4.8) ...

            try:
                # Simulate internal fills using MarketSimulator
                internal_fills, internal_transaction_costs = self.market_simulator._execute_orders(internal_orders, current_prices_normalized)
                self.circuit_breaker.record_success() # Record success if execution succeeds
            except Exception as e: # Catch potential exceptions from MarketSimulator
                print(f"Error during MarketSimulator execution: {e}") # Log error
                self.circuit_breaker.record_failure() # Record failure if exception occurs
                self.circuit_breaker.open() # Open circuit breaker immediately on exception
                # Handle error gracefully (e.g., return empty results, log error, trigger alerts)
                return {}, np.zeros(9, dtype=np.float32) # Return empty results and zero positions

            # ... (rest of NLV update and result aggregation logic) ...

        return agent_execution_results, aggregated_internal_positions
Dynamic Feature Normalization in Feature Engineering System

To improve the robustness and adaptability of the feature engineering system, especially in live trading scenarios where market data distributions can change over time, MCATS v4.9 will incorporate dynamic feature normalization.

Incremental Normalization: Instead of performing normalization on the entire dataset before training, feature normalization will be performed incrementally or online as new data arrives during training and live trading. This allows the normalization parameters (mean, standard deviation, min/max values, etc.) to adapt to the evolving data distribution.
Scalers and Partial Fitting: Scikit-learn scalers (e.g., StandardScaler, MinMaxScaler) will be used for feature normalization. These scalers support the partial_fit method, which allows for incremental updates of normalization parameters as new data batches are processed.
Example Dynamic Feature Normalization in FeatureEngine (Illustrative - using StandardScaler):

Python

from sklearn.preprocessing import StandardScaler
import pandas as pd
import vaex

class FeatureEngine:
    def __init__(self):
        self.feature_scalers = {} # Dictionary to store scalers per feature

    def generate_features(self, ohlcv_data, feature_config):
        # ... (feature generation logic from v4.8) ...

        # Normalize features dynamically
        feature_columns = list(feature_config.keys()) # Example: features to normalize are defined in config
        ohlcv_df_normalized = self.normalize_features(ohlcv_df, feature_columns)

        return ohlcv_df_normalized


    def normalize_features(self, features_vaex, feature_columns):
        """
        Normalizes specified feature columns dynamically using StandardScaler with partial_fit.

        Args:
            features_vaex (vaex.DataFrame): Vaex DataFrame with features.
            feature_columns (list): List of column names to normalize.

        Returns:
            vaex.DataFrame: Vaex DataFrame with normalized features.
        """
        features_pd = features_vaex[feature_columns].to_pandas_df() # Convert to Pandas for sklearn scalers

        for column in feature_columns:
            if column not in self.feature_scalers:
                self.feature_scalers[column] = StandardScaler() # Initialize scaler if not exists

            scaler = self.feature_scalers[column]
            features_pd[column] = scaler.partial_fit_transform(features_pd[[column]]) # Partial fit and transform in one step

        features_vaex_normalized = vaex.from_pandas(features_pd) # Convert back to Vaex
        return features_vaex_normalized

    # ... (rest of FeatureEngine class) ...
Currency Pair Universe (v4.9 - Explicitly Defined):

MCATS will focus on trading the following major currency pairs:

EURUSD
GBPUSD
USDJPY
USDCHF
AUDUSD
NZDUSD
USDCAD
USDNOK
USDSEK
These currency pairs are represented in the system using a consistent currency code order: ['EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK']. This order is used for vectors representing exposures, NLV changes, and other currency-specific data throughout the system.

2. Data Pipeline (v4.9 - Updated for Parquet and 1-Minute Bars)

Python

import pandas as pd
import numpy as np
import os
import glob
import parquet
import pytz  # For timezone handling

class DataPipeline:
    def __init__(self):
        self.raw_format = {
            'timestamp_utc': np.int64,  # millisecond precision
            'ask': np.float32,
            'bid': np.float32
        }
        self.tick_parquet_schema = { # Schema for tick data Parquet files
            'timestamp_utc': 'INT64',
            'ask': 'FLOAT32',
            'bid': 'FLOAT32'
        }
        self.ohlcv_output_format = { # Format of OHLCV data (in memory and Parquet)
            'timestamp_utc_ms': np.int64,
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32,
            'volume': np.int32 # Tick count
        }
        self.bar_size_default = '1min' # Default OHLCV bar size is now 1 minute
        self.valid_bar_sizes = ['1min', '5min', '10min', '1H'] # Supported bar sizes for aggregation
        self.est_timezone = pytz.timezone('US/Eastern') # EST timezone definition


    def process_raw_data(self, json_files, output_dir):
        """
        Convert raw JSON tick data to Parquet tick files, organized by currency pair and year.
        Parquet files will be partitioned by year for efficiency.
        """
        pass # Implementation in next instruction set

    def generate_tick_parquet(self, json_file, output_dir, currency_pair):
        """
        Processes a single JSON file to Parquet tick data and saves it.
        """
        pass # Implementation in next instruction set


    def generate_ohlcv(self, tick_data_dir, output_parquet_path, currency_pairs, bar_size='1min'):
        """
        Generate OHLCV bars from Parquet tick data files and save to Parquet.
        Generates 1-minute bars by default. Can aggregate to longer periods.
        Handles missing bars by forward-filling with previous close.
        """
        pass # Implementation in next instruction set


    def aggregate_ohlcv(self, ohlcv_1min_parquet_path, output_parquet_path, bar_size):
        """
        Aggregates 1-minute OHLCV Parquet data to longer period bars (e.g., 5min, 1H)
        and saves to a new Parquet file.
        """
        if bar_size not in self.valid_bar_sizes:
            raise ValueError(f"Invalid bar size: {bar_size}. Valid sizes are: {self.valid_bar_sizes}")
        pass # Implementation can be deferred to later stage if initial focus is on 1-min bars
3. Feature Engineering System

Python

from sklearn.preprocessing import StandardScaler
import pandas as pd
import vaex
import dask.dataframe as dd

class FeatureEngine:
    def __init__(self):
        self.base_features_tier1 = { # Tier 1 Base Features - Examples, to be extended
            'price_derived': ['log_returns', 'normalized_price'],
            'technical': ['RSI', 'MACD', 'BBANDS'], # Example TA-Lib indicators
            'volatility': ['ATR', 'historical_std_dev'] # Example volatility indicators
        }

        self.compound_features_tier2 = { # Tier 2 Compound Features - Initially empty, can be expanded
            'indicator_on_indicator': [], # Function list for indicator-on-indicator features (TBD)
            'multi_timeframe': [] # Deferred to future versions
        }

        self.ml_features_tier3 = { # Tier 3 ML-Generated Features - Placeholder for now
            'pattern_recognition': None, # Placeholder for PatternRecognitionCNN (Future)
            'range_prediction': None, # Placeholder for RangePredictor (Future)
            'regime_classification': None # Placeholder for RegimeClassifier (Future)
        }
        self.feature_scalers = {} # Dictionary to store scalers per feature

    def generate_features(self, ohlcv_data, feature_config):
        """
        Generates features based on OHLCV data and feature configuration.
        (Implementation details - see v4.8)
        """
        ohlcv_df_normalized = ohlcv_data # Placeholder - implementation in v4.8

        # Normalize features dynamically
        feature_columns = list(feature_config.keys()) # Example: features to normalize are defined in config
        ohlcv_df_normalized = self.normalize_features(ohlcv_df_normalized, feature_columns)


        return ohlcv_df_normalized # Placeholder - implementation in v4.8


    def generate_features_distributed(self, ohlcv_data_vaex, feature_config):
        """
        Generates features using distributed computation with Dask.

        Args:
            ohlcv_data_vaex (vaex.DataFrame): Vaex DataFrame of OHLCV data.
            feature_config (dict): Feature configuration.

        Returns:
            vaex.DataFrame: Vaex DataFrame with features appended.
        """
        # Convert Vaex DataFrame to Pandas DataFrame (for Dask compatibility - optimize later if needed)
        ohlcv_data_pd = ohlcv_data_vaex.to_pandas_df()

        # Create Dask DataFrame from Pandas DataFrame, partition as needed
        ddf = dd.from_pandas(ohlcv_data_pd, npartitions=4) # Example: 4 partitions

        # Define a function to calculate features on a Pandas DataFrame partition
        def _calculate_features_partition(partition_df, feature_config):
            partition_vaex = vaex.from_pandas(partition_df) # Convert partition back to Vaex for feature calculations
            return self.generate_features(partition_vaex, feature_config).to_pandas_df() # Calculate features and return as Pandas

        # Use Dask's map_partitions to apply feature calculation in parallel
        features_ddf = ddf.map_partitions(_calculate_features_partition, feature_config, meta=pd.DataFrame(columns=ohlcv_data_vaex.column_names + list(feature_config.keys()))) # Define metadata for output DataFrame

        # Compute the Dask DataFrame to get a Pandas DataFrame
        features_pd = features_ddf.compute()

        # Convert the Pandas DataFrame back to Vaex DataFrame
        features_vaex = vaex.from_pandas(features_pd)
        return features_vaex


    def normalize_features(self, features_vaex, feature_columns):
        """
        Normalizes specified feature columns dynamically using StandardScaler with partial_fit.

        Args:
            features_vaex (vaex.DataFrame): Vaex DataFrame with features.
            feature_columns (list): List of column names to normalize.

        Returns:
            vaex.DataFrame: Vaex DataFrame with normalized features.
        """
        features_pd = features_vaex[feature_columns].to_pandas_df() # Convert to Pandas for sklearn scalers

        for column in feature_columns:
            if column not in self.feature_scalers:
                self.feature_scalers[column] = StandardScaler() # Initialize scaler if not exists

            scaler = self.feature_scalers[column]
            features_pd[column] = scaler.partial_fit_transform(features_pd[[column]]) # Partial fit and transform in one step

        features_vaex_normalized = vaex.from_pandas(features_pd) # Convert back to Vaex
        return features_vaex_normalized


    def _calculate_log_returns(self, ohlcv_df):
        """Calculates log returns for 'close' price (example). Implementation needed."""
        return ohlcv_df # Placeholder - implementation in v4.8

    def _calculate_rsi(self, ohlcv_df):
        """Calculates RSI using TA-Lib (example). Needs TA-Lib integration and Vaex compatibility."""
        return ohlcv_df # Placeholder - implementation in v4.8

    def _calculate_macd(self, ohlcv_df):
        """Calculates MACD using TA-Lib (example). Needs TA-Lib integration and Vaex compatibility."""
        return ohlcv_df # Placeholder - implementation in v4.8
3.1 Compound Feature Generator (Placeholder - Deferred for Prototype) (No Change from v4.8)
3.2 ML Feature Generator (Placeholders - Deferred for Prototype) (No Change from v4.8)

4. Trading Agent Layer (Decision Layer)

Python

class TradingAgent:
    def __init__(self, allowed_currencies):
        self.exposure_vector = np.zeros(9, dtype=np.float32) # Shape (9,), initial exposure is zero
        self.nlv_vector = np.zeros(9, dtype=np.float32)     # Shape (9,), initial NLV change is zero
        self.max_nlv_scalar = 1.0  # Starting NLV scalar (used for drawdown calculation)
        self.drawdown_threshold = 0.02 # 2% drawdown threshold for reward penalty
        self.allowed_currencies = allowed_currencies # List of allowed currency codes (e.g., ['EUR', 'JPY', 'GBP'])
        self.disallowed_currencies_vector = self._create_disallowed_currency_vector(allowed_currencies) # Derived vector
        self.margin_rates = np.array([0.02, 0.01, 0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32) # Example margin rates (static for prototype)
        self.neural_network_model = None # Placeholder for neural network model (FFNN initially)

    def _create_disallowed_currency_vector(self, allowed_currencies):
        """Creates a boolean vector indicating disallowed currencies based on allowed_currencies list."""
        # (Implementation details - see v4.8)
        pass

    def get_action(self, state):
        """
        Generates a target exposure vector based on the current state using the neural network model.
        (Implementation details - see v4.8)
        """
        return np.zeros(9, dtype=np.float32) # Placeholder - implementation in v4.8

    def calculate_reward(self, new_nlv_vector_scalar): # Input is now scalar NLV change
        """
        Calculate reward based on Net Liquidation Value (NLV) change and drawdown penalty.
        (Implementation details - see v4.8)
        """
        return 0.0 # Placeholder - implementation in v4.8


    def update(self, state, action, reward, next_state): # State and next_state added for RL update
        """
        Updates the agent's neural network model based on the RL algorithm.
        (Algorithm TBD for prototype - placeholder for update mechanism)
        """
        pass # Placeholder - implementation in v4.8
5. Order & Risk Management Layer (Broker & Internal Fill Layer)

Python

class CircuitBreaker: # Simplified Circuit Breaker - Use a robust library for production
    def __init__(self, failure_threshold=5, recovery_timeout=30): # Failure threshold (consecutive failures), recovery timeout (seconds)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED" # "CLOSED", "OPEN", "HALF_OPEN"
        self.last_failure_time = None

    def is_open(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN" # Transition to half-open after timeout
                self.failure_count = 0 # Reset failure count for half-open test
            return True
        return False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN" # Open circuit breaker if failure threshold reached

    def record_success(self):
        if self.state == "HALF_OPEN":
            self.state = "CLOSED" # Close circuit breaker if half-open test succeeds
        self.failure_count = 0 # Reset failure count on success

    def reset(self): # For manual reset
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None


import time

class OrderRiskManagementLayer:
    def __init__(self, risk_manager, market_simulator, circuit_breaker=None): # Inject CircuitBreaker
        self.risk_manager = risk_manager
        self.market_simulator = market_simulator
        self.agents = {}
        self.circuit_breaker = circuit_breaker if circuit_breaker else CircuitBreaker() # Inject CircuitBreaker instance or use default

    def process_agent_actions(self, current_prices_normalized):
        if self.circuit_breaker.is_open():
            print("Circuit Breaker Open! Generating emergency close positions.") # Log circuit breaker event
            # Implement emergency close positions logic here (return zero exposure, or generate close orders)
            # ... fallback_aggregated_positions = self._generate_emergency_close_positions() ...
            # return {}, fallback_aggregated_positions # Return empty agent results, fallback positions

        aggregated_internal_positions = np.zeros(9, dtype=np.float32)
        agent_execution_results = {}

        for agent_id, agent_data in self.agents.items():
            agent = agent_data['agent']
            agent_state = agent_data # Access agent-specific tracking data
            previous_exposure_vector = agent_state['exposure_vector'].copy() # Get previous exposure

            target_exposure_vector = agent.get_action(agent_state) # Agent generates target exposure

            risk_check_result = self.risk_manager.check_risk_limits(agent) # Check risk limits

            if risk_check_result['stop_out']:
                action_exposure_vector = np.zeros(9, dtype=np.float32) # Zero exposure action if stop-out
            else:
                action_exposure_vector = target_exposure_vector # Use agent's target exposure if risk-compliant

            # Enforce disallowed currency constraints
            constrained_exposure_change_vector = (action_exposure_vector - previous_exposure_vector) * (1 - agent.disallowed_currencies_vector)
            internal_orders = self._convert_exposure_change_to_internal_orders(agent, constrained_exposure_change_vector) # Convert to internal orders

            try:
                # Simulate internal fills using MarketSimulator
                internal_fills, internal_transaction_costs = self.market_simulator._execute_orders(internal_orders, current_prices_normalized)
                self.circuit_breaker.record_success() # Record success if execution succeeds
            except Exception as e: # Catch potential exceptions from MarketSimulator
                print(f"Error during MarketSimulator execution: {e}") # Log error
                self.circuit_breaker.record_failure() # Record failure if exception occurs
                self.circuit_breaker.open() # Open circuit breaker immediately on exception
                # Handle error gracefully (e.g., return empty results, log error, trigger alerts)
                return {}, np.zeros(9, dtype=np.float32) # Return empty results and zero positions

            # Calculate NLV change based on internal fills
            nlv_change_vector = self.market_simulator._calculate_nlv_change(previous_exposure_vector, internal_fills, current_prices_normalized)
            nlv_change_scalar = np.sum(nlv_change_vector)

            # Update agent-specific state (NLV and exposure)
            agent_state['nlv_vector'] = nlv_change_vector
            agent_state['exposure_vector'] = self.market_simulator._update_exposure_vector(previous_exposure_vector, internal_fills)
            agent_state['max_nlv_scalar'] = max(agent_state['max_nlv_scalar'], np.sum(agent_state['nlv_vector'])) # Update max NLV


            agent_execution_results[agent_id] = { # Store results for each agent
                'fills': internal_fills,
                'transaction_costs': internal_transaction_costs,
                'nlv_change_vector': nlv_change_vector,
                'nlv_change_scalar': nlv_change_scalar,
                'target_exposure_vector': target_exposure_vector, # Store target exposure for analysis
                'actual_exposure_vector': agent_state['exposure_vector'].copy() # Store actual achieved exposure
            }

            aggregated_internal_positions += agent_state['exposure_vector'] # Accumulate agent exposures into aggregated positions

        return agent_execution_results, aggregated_internal_positions # Return agent results and aggregated positions



    def _convert_exposure_change_to_internal_orders(self, agent, exposure_change_vector):
        """
        Converts exposure change vector to a list of internal order objects.
        (Implementation details - see v4.8's _convert_exposure_to_orders, adapt for internal orders if needed)
        """
        orders = [] # Placeholder - Adapt from v4.8 Executor._convert_exposure_to_orders if order structure needs to be different for internal fills
        currency_order = self.market_simulator.currency_order # Use currency order from MarketSimulator

        for i, currency_code in enumerate(currency_order):
            exposure_delta = exposure_change_vector[i]
            if exposure_delta != 0:
                order = { # Example internal order structure - refine as needed
                    'currency_pair': self.market_simulator._get_currency_pair(currency_code),
                    'direction': 'BUY' if exposure_delta > 0 else 'SELL',
                    'volume': abs(exposure_delta)
                }
                orders.append(order)
        return orders
6. Market Execution Layer (Aggregation & Real Market Interface)

Python

class MarketExecutionLayer: # Renamed from Executor
    def __init__(self):
        self.spread_table = { # Example spread table - to be populated with realistic values
            'EURUSD': 0.0001, # 10 pips (in normalized price units, example)
            'GBPUSD': 0.00012,
            'USDJPY': 0.015,  # 1.5 pips (in normalized price units, example)
            'USDCHF': 0.009,
            'AUDUSD': 0.00011,
            'NZDUSD': 0.00013,
            'USDCAD': 0.00014,
            'USDNOK': 0.0018,
            'USDSEK': 0.0019
        }
        self.commission_rate = 0.000025  # 0.0025% commission per trade (example, fraction of traded normalized price)
        self.currency_order = ['EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK'] # Static currency order


    def execute_aggregated_positions(self, aggregated_internal_positions, current_prices_normalized):
        """
        Executes aggregated internal positions in the market (or simulator).

        Args:
            aggregated_internal_positions (np.array): Aggregated positions from OrderRiskManagementLayer (shape (9,)).
            current_prices_normalized (np.array): Current normalized prices.

        Returns:
            dict: Market execution summary (for prototype, might be simplified).
        """
        # For prototype, simulate execution of aggregated positions - Simple Pass-Through for now.
        # In future, implement more sophisticated execution strategies here.
        market_execution_results = {
            'status': 'SIMULATED_EXECUTION', # Or 'REAL_EXECUTION' in future
            'aggregated_positions_executed': aggregated_internal_positions.copy() # Return executed positions
        }
        return market_execution_results


    def _execute_orders(self, orders, current_prices_normalized): # Kept for internal use by OrderRiskManagementLayer for internal fills
        """
        Simulates order execution, calculates fills and transaction costs (spread, commission).
        (Implementation details - see v4.8's _execute_orders)
        """
        fills = []
        total_transaction_cost = 0

        for order in orders:
            currency_pair = order['currency_pair']
            direction = order['direction']
            volume = order['volume']
            spread = self.spread_table.get(currency_pair, 0) # Get spread from table, default to 0 if not found
            commission = volume * self.commission_rate

            price = self._get_price_for_execution(currency_pair, current_prices_normalized, direction, spread) # Price with spread

            fill = { # Example fill structure - refine as needed
                'currency_pair': currency_pair,
                'direction': direction,
                'volume': volume,
                'price': price # Execution price with spread
            }
            fills.append(fill)
            transaction_cost = spread * volume + commission # Transaction cost for this fill
            total_transaction_cost += transaction_cost

        return fills, total_transaction_cost


    def _calculate_nlv_change(self, previous_exposure_vector, fills, current_prices_normalized):
        """
        Calculates the currency-wise Net Liquidation Value (NLV) change based on fills and price movements.
        (Implementation details - see v4.8's _calculate_nlv_change)
        """
        nlv_change_vector = np.zeros(9, dtype=np.float32)
        # (Implementation details - see v4.8's _calculate_nlv_change)
        return nlv_change_vector

    def _update_exposure_vector(self, previous_exposure_vector, fills):
        """Updates agent's exposure vector based on executed fills."""
        updated_exposure_vector = previous_exposure_vector.copy()
        # (Implementation details - see v4.8's _update_exposure_vector)
        return updated_exposure_vector


    def _get_currency_pair(self, currency_code):
        """Helper function to get currency pair string (e.g., 'EURUSD' for 'EUR')."""
        return "" # Placeholder - implementation in v4.8

    def _get_base_currency_code(self, currency_pair):
        """Helper to extract base currency code from currency pair string."""
        return "" # Placeholder - implementation in v4.8

    def _get_quote_currency_code(self, currency_pair):
        """Helper to extract quote currency code from currency pair string."""
        return "" # Placeholder - implementation in v4.8

    def _get_price_for_execution(self, currency_pair, current_prices_normalized, direction, spread):
        """
        Helper to get execution price with spread.
        (For prototype, using midpoint + spread/2 for BUY, midpoint - spread/2 for SELL.
         More realistic spread modeling can be added later).
        """
        return 0.0 # Placeholder - implementation in v4.8

    def _create_price_map(self, current_prices_normalized):
        """Helper to create a dictionary mapping currency pairs to normalized prices."""
        return {} # Placeholder - implementation in v4.8
7. Training Pipeline

Python

import mlflow

class TrainingPipeline:
    def __init__(self, config, order_risk_management_layer, feature_engine, risk_manager, market_simulator): # Inject OrderRiskManagementLayer, FeatureEngine, RiskManager, MarketExecutionLayer, and Configuration
        self.config = config # Store configuration
        self.order_risk_management_layer = order_risk_management_layer # Renamed
        self.feature_engine = feature_engine
        self.risk_manager = risk_manager # Still need RiskManager instance for OrderRiskManagementLayer dependency injection
        self.market_simulator = market_simulator # Still need MarketSimulator instance for OrderRiskManagementLayer dependency injection

        self.agent_ids = [] # List to keep track of registered agent IDs

        mlflow.set_experiment("MCATS_Training_Experiments") # Set MLflow experiment name


    def register_agent(self, agent, agent_id):
        """Registers a trading agent for training."""
        self.order_risk_management_layer.register_agent(agent, agent_id) # Register agent with ORM Layer
        self.agent_ids.append(agent_id) # Keep track of agent IDs

    def train_episode(self, episode_data):
        """
        Train agents on one trading day episode.

        Args:
            episode_data (vaex.DataFrame): Vaex DataFrame containing data for a single trading day episode.
                                             Must include 'normalized_price' and time-based features.

        Returns:
            dict: Episode summary (e.g., total reward, final NLV, drawdown - aggregated across agents).
        """
        episode_reward_agents = {agent_id: 0 for agent_id in self.agent_ids} # Track episode reward per agent
        bar_count = 0
        episode_max_nlv_scalar_agents = {agent_id: self.order_risk_management_layer.agents[agent_id]['max_nlv_scalar'] for agent_id in self.agent_ids} # Track max NLV per agent


        with mlflow.start_run(): # Start MLflow run for each episode
            mlflow.log_params({"rl_algorithm": self.config.training_pipeline_config.rl_algorithm,  # Log config parameters
                                "drawdown_threshold": self.config.risk_config.drawdown_threshold,
                                # ... other relevant parameters from config classes ...
                                })


            for agent_id in self.agent_ids: # Reset agent NLV at start of episode
                 self.order_risk_management_layer.agents[agent_id]['nlv_vector'] = np.zeros(9, dtype=np.float32)

            state = self._get_initial_state(episode_data.row(0)) # Get initial state from first bar

            for bar_index in range(episode_data.count()):
                bar = episode_data.row(bar_index)
                current_prices_normalized = bar[['normalized_price']].to_numpy().flatten()

                features = self.feature_engine.generate_features(bar, feature_config={'tier1': {'price_derived': ['normalized_price']}}) # Generate features for current bar - Example config

                agent_execution_results, aggregated_internal_positions = self.order_risk_management_layer.process_agent_actions(current_prices_normalized) # Process actions from ALL agents

                market_execution_results = self.market_simulator.execute_aggregated_positions(aggregated_internal_positions, current_prices_normalized) # Market Execution Layer handles aggregated positions


                for agent_id in self.agent_ids: # Process results for each agent
                    agent_data = self.order_risk_management_layer.agents[agent_id]
                    agent = agent_data['agent']
                    execution_result = agent_execution_results[agent_id] # Get agent-specific execution results

                    reward = agent.calculate_reward(execution_result['nlv_change_scalar']) # Agent calculates reward

                    next_state = self._get_next_state(bar) # Get next state

                    agent.update(state, execution_result['target_exposure_vector'], reward, next_state) # Agent update step (using target exposure as action for now)

                    episode_reward_agents[agent_id] += reward # Accumulate agent reward
                    episode_max_nlv_scalar_agents[agent_id] = max(episode_max_nlv_scalar_agents[agent_id], np.sum(agent_data['nlv_vector'])) # Update agent max NLV


                bar_count += 1
                state = next_state # Transition to next state


            episode_summary = {} # Aggregate episode summary across agents
            total_episode_reward = 0
            total_final_nlv_scalar = 0
            max_episode_drawdown = 0 # Initialize max drawdown across agents to 0

            for agent_id in self.agent_ids: # Calculate aggregated episode summary
                agent_data = self.order_risk_management_layer.agents[agent_id]
                agent_final_nlv_scalar = np.sum(agent_data['nlv_vector'])
                agent_episode_drawdown = (episode_max_nlv_scalar_agents[agent_id] - agent_final_nlv_scalar) / episode_max_nlv_scalar_agents[agent_id] if episode_max_nlv_scalar_agents[agent_id] > 0 else 0

                episode_summary[agent_id] = {
                    'episode_reward': episode_reward_agents[agent_id],
                    'final_nlv_scalar': agent_final_nlv_scalar,
                    'episode_drawdown': agent_episode_drawdown
                }
                total_episode_reward += episode_reward_agents[agent_id]
                total_final_nlv_scalar += total_final_nlv_scalar
                max_episode_drawdown = max(max_episode_drawdown, agent_episode_drawdown) # Track max drawdown across agents

            episode_summary['aggregated_episode_reward'] = total_episode_reward
            episode_summary['aggregated_final_nlv_scalar'] = total_final_nlv_scalar
            episode_summary['max_episode_drawdown'] = max_episode_drawdown # Max drawdown across agents in the episode
            episode_summary['bar_count'] = bar_count

            mlflow.log_metrics({"episode_reward_aggregated": episode_summary['aggregated_episode_reward'], # Log metrics to MLflow
                                    "final_nlv_scalar_aggregated": episode_summary['aggregated_final_nlv_scalar'],
                                    "max_episode_drawdown": episode_summary['max_episode_drawdown'],
                                    "bar_count": episode_summary['bar_count']})


        return episode_summary


    def _get_initial_state(self, first_bar_data):
        """
        Gets the initial state at the beginning of a trading episode.
        (Implementation details - see v4.8)
        """
        return np.array([]) # Placeholder - implementation in v4.8

    def _get_next_state(self, bar_data):
        """
        Gets the next state at the next time step.
        (Implementation details - see v4.8)
        """
        return np.array([]) # Placeholder - implementation in v4.8
8. Implementation Notes (Updated - v4.9 - Reflecting Parquet Data Layer and 1-Minute Bars)

(All notes from v4.8 remain relevant where applicable).
Data Pipeline v2 Implementation: Implement the revised DataPipeline class (v4.9) that processes raw JSON directly to Parquet tick files and generates 1-minute OHLCV Parquet data as the default. Focus on robust error handling, timezone management (EST 17:15/17:00 trading days), and missing bar filling at the OHLCV stage.
Parquet Data Storage: Use Parquet format for storing both tick data and OHLCV data files. Ensure efficient partitioning by year for tick data files. Define clear file naming conventions for Parquet files (e.g., EURUSD_ticks_2023.parquet, EURUSD_1min_ohlcv.parquet).
1-Minute OHLCV as Default: Make 1-minute bars the default output of the generate_ohlcv method. Implement functionality to aggregate to longer period bars (aggregate_ohlcv method), but this can be deferred to a later sub-stage.
Timezone Handling: Pay close attention to timezone handling, especially for EST 17:15/17:00 trading day boundaries. Use pytz and Pandas' timezone functionalities to ensure correct time conversions and filtering.
Missing Bar Handling in OHLCV Generation: Implement missing bar filling (previous close) during OHLCV generation to ensure complete and consistently shaped OHLCV datasets for episodes.
(Other Implementation Notes from v4.8 remain - Configuration, Dask/Ray, MLflow, Circuit Breakers, Dynamic Normalization, Testing, CLI, Web UI, Compute Infrastructure).
Next Steps Priority (v4.9 - Updated for Data Layer v2 Focus):

Implement revised Data Layer (v4.9) with direct JSON to Parquet tick files and 1-minute OHLCV Parquet generation as the primary focus. Include error handling, timezone management, and missing bar filling.
Implement core layered architecture with basic functionality (following v4.9 spec - starting with Data Layer integration).
Add comprehensive logging and monitoring throughout the system (starting with Data Layer).
Implement class-based configuration system using dataclasses.
Develop Command Line Interface (CLI) for core Data Layer functionalities (data processing, OHLCV generation).
Integrate basic RL algorithm (suggest starting with Random Search or Hill Climbing for initial validation, then DQN or PPO).
Develop unit and integration testing framework alongside implementation (starting with Data Layer).
Integrate MLflow experiment tracking into the TrainingPipeline.
Evaluate Feature Engineering performance and implement Dask distribution if needed.
Implement Circuit Breaker in Order & Risk Management Layer.
Implement Dynamic Feature Normalization in Feature Engineering System.
Plan and begin initial design of the Web-based User Interface (UI) architecture.
Plan and prepare for migration to cloud-based compute infrastructure (AWS, GCP, Azure).
Enhance Feature Engineering System with more Tier 1 features and begin planning for Tier 2 and Tier 3 feature integration.
Currency Pair Universe (v4.9 - Explicitly Defined):

MCATS will focus on trading the following major currency pairs:

EURUSD
GBPUSD
USDJPY
USDCHF
AUDUSD
NZDUSD
USDCAD
USDNOK
USDSEK
These currency pairs are represented in the system using a consistent currency code order: ['EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK']. This order is used for vectors representing exposures, NLV changes, and other currency-specific data throughout the system.

9. User Interface (UI) and Command Line Interface (CLI)

MCATS will provide both a Command Line Interface (CLI) and a User Interface (UI) to interact with the system.  The CLI will be the primary interface for core functionalities and automation, while the UI will provide a more user-friendly and visually interactive experience.

Command Line Interface (CLI):

Purpose: Primary interface for developers, quantitative analysts, and for automation and scripting. Provides access to all core functionalities of MCATS in a programmatic and scriptable way.
Technology: Python's argparse or click libraries will be used for building a robust and user-friendly CLI.
Functionality (Initial): CLI will initially expose functionalities such as:
Data Pipeline: mcats data process, mcats data ohlcv
Feature Engineering: mcats features generate
Training Pipeline: mcats train episode, mcats train agent
Risk Management: mcats risk check
System Configuration: mcats config show, mcats config set
Abstraction: CLI commands will directly interact with the MCATS backend layers, providing a clear and direct interface to the system's core logic.
User Interface (UI) - Web-based (Planned):

Purpose: To provide a more user-friendly and visually interactive way to monitor, control, and analyze MCATS, especially for users who are not comfortable with command lines.
Technology (Proposed):
Frontend: React, Vue.js, or a similar modern JavaScript framework for building a rich and interactive web UI.
Backend (if needed): Flask (or Django) for a lightweight Python backend API to serve data to the frontend and orchestrate CLI commands. Alternatively, the frontend could directly interact with the MCATS backend (depending on deployment architecture).
Interaction Model: The Web UI will primarily interact with the MCATS backend via the CLI (or a REST API built on top of the CLI layer). This ensures a clear separation between the UI and the core backend logic. The UI will essentially act as a visual front-end for executing CLI commands and displaying their results.
Planned UI Functionality (Initial):
System Monitoring: Real-time monitoring of training progress, resource usage, and system health.
Experiment Management: Integration with MLflow or Weights & Biases UI for experiment tracking and comparison.
Configuration Management: Visual interface for viewing and editing system configurations.
Data Visualization: Interactive charts and graphs for visualizing market data, features, agent performance, and risk metrics.
Agent Control: Start, stop, and monitor trading agents.
Reporting: Generate reports on training episodes, agent performance, and system activity.
Jupyter Notebooks/Lab (Development and Exploration):

Jupyter Notebooks/Lab will be extensively used during development, prototyping, and experimentation for interactive data analysis, visualization, and rapid iteration. While not intended as the primary UI for end-users, Jupyter notebooks will serve as a valuable tool for developers and researchers working on MCATS.
Rationale for UI/CLI Strategy:

CLI-First Approach: Prioritizing CLI development ensures a robust, testable, and automatable core system. It allows for headless operation and provides a solid foundation for building UIs on top.
Abstraction and Decoupling: Abstracting the UI layer from the backend via the CLI (or API) makes the system more modular, maintainable, and future-proof. It allows for UI technology changes or the development of multiple UIs without impacting the core backend.
Web-based UI for Accessibility: A web-based UI provides broad accessibility across platforms and facilitates collaboration and deployment.
Jupyter for Development Efficiency: Jupyter notebooks accelerate development, experimentation, and data exploration, crucial for an iterative ML project.
10. Compute Infrastructure and Scalability

MCATS is anticipated to have escalating compute demands, particularly for feature engineering and reinforcement learning training.  This section outlines the planned compute infrastructure strategy.

Initial Development and Prototyping: Google Colab

Google Colab (Free Tier): For the initial phases of development and prototyping, Google Colab (Free) will be utilized. Colab provides a free and readily accessible environment with pre-installed libraries and basic compute resources, sufficient for getting the core architecture implemented and running.
Google Colab Pro/Pro+ (Transition): As development progresses and compute requirements increase (especially when training more complex RL algorithms and using larger datasets), MCATS will transition to Google Colab Pro or Pro+. These paid tiers offer more compute resources (RAM, disk, faster GPUs/TPUs), longer runtimes, and fewer interruptions, providing a more robust development and experimentation environment.
Benefits of Google Colab: Ease of setup, free (for basic tier), readily accessible GPUs/TPUs in Pro/Pro+, collaborative environment.
Limitations of Google Colab: Resource limits (even in Pro/Pro+), potential runtime limitations, less control over environment configuration than VMs.
Scalable Compute and Production Readiness: Cloud Virtual Machines (VMs)

Cloud Platforms (AWS, GCP, Azure): For scalability, demanding training runs, backtesting, and eventual production readiness (even simulated production), MCATS will migrate to cloud Virtual Machines (VMs) on platforms like Amazon Web Services (AWS), Google Cloud Platform (GCP), or Microsoft Azure.
VM Instance Types: Utilize cloud VMs with sufficient CPU and memory resources, and potentially GPU-accelerated VMs for training RL models. Specific instance types will be selected based on profiled compute needs.
Scalability and Elasticity: Cloud VMs offer excellent scalability and elasticity. Resources can be provisioned and de-provisioned on-demand, allowing MCATS to scale up compute power for heavy training runs and scale down during development or idle periods, optimizing cost efficiency.
Containerization (Docker) and Orchestration (Kubernetes - Future): For enhanced deployment, scalability, and manageability on cloud VMs, containerization using Docker and container orchestration using Kubernetes (or similar platforms) may be explored in future iterations, especially as the system scales and becomes more complex.
Cost Management in Cloud: Careful cost management in cloud environments is crucial. Implement strategies such as:
Right-sizing VMs: Selecting appropriate VM instance types based on actual resource needs.
Spot Instances/Preemptible VMs: Utilizing spot instances or preemptible VMs (if suitable for workloads) to reduce compute costs (at the risk of interruptions).
Auto-scaling: Implementing auto-scaling for VMs based on workload demands.
Resource monitoring and cost dashboards: Regularly monitoring cloud resource usage and costs to identify optimization opportunities.
Local Server/Workstation (Optional - Long-Term Evaluation):

For very long-term, sustained heavy compute needs, and if budget allows, the option of deploying MCATS on a dedicated local server or powerful workstation with GPUs can be evaluated. This option would need to be compared to the cost and scalability of cloud VMs based on long-term resource usage patterns.
Rationale for Compute Infrastructure Strategy:

Start with Colab for Accessibility and Prototyping: Google Colab provides a low-barrier entry point for initial development and experimentation, reducing setup complexity and cost.
Transition to Cloud VMs for Scalability and Production: Cloud VMs offer the necessary scalability, control, and production-readiness features for more demanding workloads and potential future deployment scenarios.
Cost-Effective Scaling: Cloud VMs enable cost-effective scaling by allowing resources to be adjusted based on demand.
Flexibility and Choice: The strategy allows for flexibility in choosing cloud platforms and VM instance types based on evolving project needs and cost considerations.
This Technical Specification Document v4.9 is now updated to reflect the revised data pipeline and 1-minute OHLCV default, along with other clarifications.  It is ready to guide the next phase of development, starting with the implementation of the Parquet-based Data Layer (Data Layer v2).
# MCATS - Multi-Currency Automated Trading System

## Project Overview

MCATS (Multi-Currency Automated Trading System) is a modular, scalable, and robust platform for developing and deploying reinforcement learning (RL) agents for automated trading in multi-currency foreign exchange (FX) markets.

### Project Goals

- **Develop a Reinforcement Learning based Trading System**: Leverage RL to create intelligent agents capable of learning optimal trading strategies directly from market data.
- **Multi-Currency Trading**: Enable agents to trade across a portfolio of major currency pairs, capturing potential diversification benefits and cross-currency trading opportunities.
- **Modular and Scalable Architecture**: Design a system with clearly defined layers and interfaces to facilitate independent development, testing, and future enhancements.
- **Realistic Simulation Environment**: Create a simulation environment that accurately models market dynamics, transaction costs (spread and commission), and risk factors (drawdown).
- **Production Readiness (Future Goal)**: Architect the system with considerations for eventual deployment in a production trading environment, including robust risk management and efficient market execution.

## Technical Specification

This implementation follows the MCATS Technical Specification Document v4.9. The specification provides detailed requirements for each layer of the system and defines the interfaces between them.

## System Architecture

MCATS employs a layered architecture to clearly separate concerns and promote modularity, scalability, and maintainability. The system is structured into the following distinct layers:

- **Data Layer**: Responsible for raw data ingestion and processing into Parquet tick data files, and generation of 1-minute OHLCV Parquet data files. Functionality to generate longer period OHLCV bars from 1-minute data is included.
- **Feature Engineering System**: Extracts relevant features from 1-minute OHLCV data, with potential for distributed computation (Dask/Ray) and dynamic feature normalization.
- **Trading Agent Layer (Decision Layer)**: Houses reinforcement learning agents.
- **Order & Risk Management Layer (Broker & Internal Fill Layer)**: Acts as an intermediary, enforcing risk management, simulating internal fills, and aggregating positions, incorporating circuit breaker patterns for resilience.
- **Market Execution Layer (Aggregation & Real Market Interface)**: Handles market execution of aggregated positions.
- **Training Pipeline**: Orchestrates the end-to-end training process, integrated with experiment tracking tools (MLflow/Weights & Biases).
- **User Interface (UI) & Command Line Interface (CLI)**: Provides user interaction and system control via both CLI and a planned Web-based UI.
- **Compute Infrastructure**: Plans for scalable compute infrastructure, starting with Google Colab and transitioning to Cloud VMs.

## Data Layer Functionality

The Data Layer is responsible for raw data ingestion, processing, and transformation into structured formats suitable for feature engineering and training. It provides the following key functionalities:

### Process Raw Data (`process_raw_data`)

Converts raw JSON tick data to Parquet tick files, organized by currency pair and year.

- **Input Parameters**:
  - `json_files` (List[str]): List of paths to JSON files containing raw tick data.
  - `output_dir` (str): Directory where processed Parquet files will be saved.
- **Output**: Parquet files organized by currency pair and year in the specified output directory.
- **Usage Scenario**: Process a collection of raw JSON tick data files (e.g., from a data provider) into an optimized Parquet format for faster subsequent processing.

### Generate Tick Parquet (`generate_tick_parquet`)

Processes a single JSON file to Parquet tick data and saves it.

- **Input Parameters**:
  - `json_file` (str): Path to the JSON file containing raw tick data.
  - `output_dir` (str): Directory where the processed Parquet file will be saved.
  - `currency_pair` (str): Currency pair code (e.g., 'EURUSD').
- **Output**: Path to the generated Parquet file, organized by year in the output directory.
- **Usage Scenario**: Convert an individual tick data file for a specific currency pair to Parquet format, typically called by `process_raw_data` for each file.

### Generate OHLCV Bars (`generate_ohlcv`)

Generates OHLCV (Open, High, Low, Close, Volume) bars from Parquet tick data files and saves them to a Parquet file.

- **Input Parameters**:
  - `tick_data_dir` (str): Directory containing Parquet tick data files.
  - `output_parquet_path` (str): Path where the OHLCV Parquet file will be saved.
  - `currency_pairs` (List[str]): List of currency pairs to process.
  - `bar_size` (str, optional): Size of bars to generate, defaults to '1min'. Must be one of the valid bar sizes.
- **Output**: Path to the generated OHLCV Parquet file containing bars for all specified currency pairs.
- **Usage Scenario**: Generate time-based OHLCV bars (e.g., 1-minute bars) from tick data for use in feature engineering and trading strategy development.

### Aggregate OHLCV (`aggregate_ohlcv`)

Aggregates 1-minute OHLCV Parquet data to longer period bars (e.g., 5min, 1H) and saves them to a new Parquet file.

- **Input Parameters**:
  - `ohlcv_1min_parquet_path` (str): Path to the 1-minute OHLCV Parquet file.
  - `output_parquet_path` (str): Path where the aggregated OHLCV Parquet file will be saved.
  - `bar_size` (str): Size of bars to generate. Must be one of the valid bar sizes and longer than '1min'.
- **Output**: Path to the generated aggregated OHLCV Parquet file.
- **Usage Scenario**: Create longer timeframe OHLCV bars from 1-minute data for multi-timeframe analysis or to reduce data frequency for certain trading strategies.

## Project Structure

```
MCATS/
├── configuration/       # Configuration management
├── data_layer/          # Data ingestion and processing
├── docs/                # Documentation
├── feature_engineering/ # Feature extraction and engineering
├── logs/                # Logging configuration and log files
├── market_execution/    # Market execution and simulation
├── mcats_main.py        # Main entry point
├── notebooks/           # Jupyter notebooks for exploration
├── order_risk_management/ # Order and risk management
├── tests/               # Unit and integration tests
├── trading_agent_layer/ # RL agent implementations
├── training_pipeline/   # Training orchestration
└── user_interface/      # CLI and Web UI
```

## Currency Pair Universe

MCATS focuses on trading the following major currency pairs:

- EURUSD
- GBPUSD
- USDJPY
- USDCHF
- AUDUSD
- NZDUSD
- USDCAD
- USDNOK
- USDSEK

These currency pairs are represented in the system using a consistent currency code order: ['EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK'].

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/MCATS.git
   cd MCATS
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Required Packages

- numpy
- pandas
- pytz
- pyarrow
- matplotlib
- scikit-learn
- tensorflow or pytorch (for RL agent implementation)
- mlflow (for experiment tracking)
- flask (for Web UI, future)

## Usage

Detailed usage instructions will be provided as the project development progresses.

## Contributing

Guidelines for contributing to the project will be added in the future.

## License

[License information to be determined] 
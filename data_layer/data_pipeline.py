#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Pipeline Module for MCATS (Multi-Currency Automated Trading System)

This module contains the DataPipeline class responsible for raw data ingestion,
processing into Parquet tick data files, and generation of OHLCV data files.
It follows the specifications defined in the MCATS Technical Specification v4.9.
"""

import os
import glob
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytz
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple


class DataPipeline:
    """
    DataPipeline class for the MCATS Data Layer.
    
    Responsible for:
    1. Raw data ingestion and processing into Parquet tick data files
    2. Generation of 1-minute OHLCV Parquet data files
    3. Aggregation of 1-minute data into longer period bars (e.g., 5min, 1H)
    
    The class handles timezone management, missing bar filling, and ensures data 
    integrity throughout the pipeline.
    """

    def __init__(self):
        """
        Initialize the DataPipeline with default configurations.
        
        Sets up data formats, schemas, default parameters, and timezone configurations
        for processing FX market data according to MCATS specifications v4.9.
        """
        # Format specification for raw tick data
        self.raw_format = {
            'timestamp_utc': np.int64,  # millisecond precision
            'ask': np.float32,
            'bid': np.float32
        }
        
        # Schema for tick data Parquet files
        self.tick_parquet_schema = {
            'timestamp_utc': 'INT64',
            'ask': 'FLOAT32',
            'bid': 'FLOAT32'
        }
        
        # Format specification for OHLCV data (in memory and Parquet)
        self.ohlcv_output_format = {
            'timestamp_utc_ms': np.int64,
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32,
            'volume': np.int32  # Tick count
        }
        
        # Default and supported bar sizes
        self.bar_size_default = '1min'  # Default OHLCV bar size is 1 minute
        self.valid_bar_sizes = ['1min', '5min', '10min', '1H']  # Supported bar sizes
        
        # Timezone configuration
        self.est_timezone = pytz.timezone('US/Eastern')  # EST timezone definition
        self.utc_timezone = pytz.UTC  # UTC timezone for consistency
        
        # Logger setup
        self.logger = logging.getLogger(__name__)

    def process_raw_data(self, json_files: List[str], output_dir: str) -> None:
        """
        Convert raw JSON tick data to Parquet tick files, organized by currency pair and year.
        
        This method processes multiple JSON files containing raw tick data and converts
        them to Parquet format, organizing the output by currency pair and year.
        
        Args:
            json_files (List[str]): List of paths to JSON files containing raw tick data.
            output_dir (str): Directory where processed Parquet files will be saved.
                              Will be created if it doesn't exist.
        
        Raises:
            FileNotFoundError: If any of the input JSON files don't exist.
            ValueError: If the JSON files contain invalid data.
            IOError: If there are issues reading the JSON files or writing Parquet files.
        """
        self.logger.info(f"Starting processing of {len(json_files)} raw JSON files to Parquet")
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Track processed files for summary
        processed_files = 0
        skipped_files = 0
        error_files = 0
        
        # Process each JSON file
        for json_file in json_files:
            try:
                # Log the current file being processed
                self.logger.info(f"Processing file: {json_file}")
                
                # Extract currency pair from filename
                file_name = Path(json_file).name
                currency_pair = self._extract_currency_pair_from_filename(file_name)
                
                if not currency_pair:
                    self.logger.warning(f"Could not extract currency pair from filename: {file_name}. Skipping file.")
                    skipped_files += 1
                    continue
                
                # Process the file with the extracted currency pair
                self.logger.info(f"Extracted currency pair: {currency_pair} from {file_name}")
                output_file = self.generate_tick_parquet(json_file, output_dir, currency_pair)
                
                self.logger.info(f"Successfully processed {json_file} to {output_file}")
                processed_files += 1
                
            except FileNotFoundError as e:
                self.logger.error(f"File not found error processing {json_file}: {str(e)}")
                error_files += 1
            except ValueError as e:
                self.logger.error(f"Invalid data error processing {json_file}: {str(e)}")
                error_files += 1
            except IOError as e:
                self.logger.error(f"I/O error processing {json_file}: {str(e)}")
                error_files += 1
            except Exception as e:
                self.logger.exception(f"Unexpected error processing {json_file}: {str(e)}")
                error_files += 1
                
        # Log summary of processing
        self.logger.info(f"Completed processing raw JSON files to Parquet")
        self.logger.info(f"Summary: Processed: {processed_files}, Skipped: {skipped_files}, Errors: {error_files}")

    def generate_tick_parquet(self, json_file: str, output_dir: str, currency_pair: str) -> str:
        """
        Process a single JSON file to Parquet tick data and save it.
        
        This method takes a single JSON file containing raw tick data for a specific
        currency pair, processes it into a standardized format, and saves it as a Parquet
        file organized by currency pair and year.
        
        Args:
            json_file (str): Path to the JSON file containing raw tick data.
            output_dir (str): Directory where the processed Parquet file will be saved.
            currency_pair (str): Currency pair code (e.g., 'EURUSD').
            
        Returns:
            str: Path to the generated Parquet file.
            
        Raises:
            FileNotFoundError: If the input JSON file doesn't exist.
            ValueError: If the JSON file contains invalid data.
            IOError: If there are issues reading the JSON file or writing the Parquet file.
        """
        self.logger.info(f"Processing {json_file} for {currency_pair}")
        
        # Validate that the input file exists
        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {json_file}")
        
        try:
            # Read the JSON file
            with open(json_path, 'r') as file:
                tick_data = json.load(file)
            
            # Validate the JSON data structure
            if not isinstance(tick_data, list):
                raise ValueError(f"Expected JSON data to be a list, got {type(tick_data).__name__}")
            
            if not tick_data:
                self.logger.warning(f"JSON file {json_file} contains no data")
                return None
            
            # Check if the required fields exist in the first item
            required_fields = ["timestamp_utc", "ask", "bid"]
            for field in required_fields:
                if field not in tick_data[0]:
                    raise ValueError(f"Required field '{field}' missing in JSON data")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(tick_data)
            
            # Ensure the data types match the raw_format
            for column, dtype in self.raw_format.items():
                if column in df.columns:
                    df[column] = df[column].astype(dtype)
            
            # Extract year from timestamp for partitioning
            # Timestamps are in milliseconds since epoch
            # Convert first to timestamp then extract year
            first_timestamp = df["timestamp_utc"].iloc[0]
            year = pd.to_datetime(first_timestamp, unit='ms').year
            
            # Create output filename
            output_path = Path(output_dir)
            year_dir = output_path / str(year)
            year_dir.mkdir(exist_ok=True, parents=True)
            
            output_file = year_dir / f"{currency_pair}_ticks_{year}.parquet"
            
            # Sort by timestamp to ensure data is chronologically ordered
            df = df.sort_values(by="timestamp_utc")
            
            # Check for and handle duplicate timestamps
            duplicates = df.duplicated(subset=["timestamp_utc"], keep=False)
            if duplicates.any():
                duplicate_count = duplicates.sum()
                self.logger.warning(f"Found {duplicate_count} duplicate timestamps in {json_file}")
                
                # For demonstration purposes, we'll keep the first occurrence
                # In a production system, you might want a more sophisticated approach
                df = df.drop_duplicates(subset=["timestamp_utc"], keep='first')
                self.logger.info(f"Removed duplicate timestamps, keeping first occurrence")
            
            # Create PyArrow Table and write to Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(output_file))
            
            self.logger.info(f"Successfully wrote {len(df)} tick records to {output_file}")
            
            return str(output_file)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_file}: {str(e)}")
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError, IOError)):
                raise
            raise IOError(f"Error processing {json_file} to Parquet: {str(e)}")

    def generate_ohlcv(self, 
                      tick_data_dir: str, 
                      output_parquet_path: str, 
                      currency_pairs: List[str], 
                      bar_size: str = '1min') -> str:
        """
        Generate OHLCV bars from Parquet tick data files and save to Parquet.
        
        This method reads tick data from Parquet files, generates OHLCV bars of the
        specified size (default 1-minute), handles missing bars by forward-filling
        with previous close, and saves the result to a Parquet file.
        
        Args:
            tick_data_dir (str): Directory containing Parquet tick data files.
            output_parquet_path (str): Path where the OHLCV Parquet file will be saved.
            currency_pairs (List[str]): List of currency pairs to process.
            bar_size (str, optional): Size of bars to generate. Defaults to '1min'.
                                     Must be one of the valid_bar_sizes.
                                     
        Returns:
            str: Path to the generated OHLCV Parquet file.
            
        Raises:
            ValueError: If bar_size is not valid or if tick data files are not found.
            IOError: If there are issues reading tick data or writing OHLCV data.
        """
        # Validate bar size
        if bar_size not in self.valid_bar_sizes:
            raise ValueError(f"Invalid bar size: {bar_size}. Valid sizes are: {self.valid_bar_sizes}")
            
        self.logger.info(f"Starting generation of {bar_size} OHLCV data for {currency_pairs}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_parquet_path).parent
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Dictionary to store OHLCV DataFrames for each currency pair
        ohlcv_dfs = {}
        
        # Process each currency pair
        for currency_pair in currency_pairs:
            self.logger.info(f"Processing OHLCV data for {currency_pair}")
            
            try:
                # Find all relevant Parquet tick data files for this currency pair
                tick_data_path = Path(tick_data_dir)
                
                # Search for year directories and tick files within them
                # Pattern matches: {year}/{currency_pair}_ticks_{year}.parquet
                tick_files = []
                for year_dir in tick_data_path.glob("*"):
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        file_pattern = f"{currency_pair}_ticks_{year_dir.name}.parquet"
                        tick_file = year_dir / file_pattern
                        if tick_file.exists():
                            tick_files.append(tick_file)
                
                if not tick_files:
                    self.logger.warning(f"No tick data files found for {currency_pair} in {tick_data_dir}")
                    continue
                
                self.logger.info(f"Found {len(tick_files)} tick data files for {currency_pair}")
                
                # Read and combine all tick data files for this currency pair
                dfs = []
                for tick_file in tick_files:
                    self.logger.info(f"Reading tick data from {tick_file}")
                    try:
                        tick_df = pd.read_parquet(tick_file)
                        dfs.append(tick_df)
                    except Exception as e:
                        self.logger.error(f"Error reading tick data file {tick_file}: {str(e)}")
                        continue
                
                if not dfs:
                    self.logger.warning(f"No valid tick data found for {currency_pair}")
                    continue
                
                # Combine all tick data for this currency pair
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Ensure no duplicates in timestamp
                combined_df = combined_df.drop_duplicates(subset=["timestamp_utc"], keep="first")
                
                # Sort by timestamp
                combined_df = combined_df.sort_values(by="timestamp_utc")
                
                # Convert timestamp to datetime for resampling
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp_utc"], unit="ms")
                combined_df = combined_df.set_index("timestamp")
                
                # Generate OHLCV bars
                self.logger.info(f"Generating {bar_size} OHLCV bars for {currency_pair}")
                
                # Calculate mid price (average of bid and ask)
                combined_df["mid"] = (combined_df["bid"] + combined_df["ask"]) / 2
                
                # Resample to the specified bar size
                ohlcv_resampled = combined_df.resample(bar_size)
                
                # Generate OHLCV data
                ohlcv_df = pd.DataFrame({
                    "open": ohlcv_resampled["mid"].first(),
                    "high": ohlcv_resampled["mid"].max(),
                    "low": ohlcv_resampled["mid"].min(),
                    "close": ohlcv_resampled["mid"].last(),
                    "volume": ohlcv_resampled["mid"].count()  # Count of ticks in each bar
                })
                
                # Check for missing bars (gaps in time series)
                expected_indices = pd.date_range(
                    start=ohlcv_df.index.min(),
                    end=ohlcv_df.index.max(),
                    freq=bar_size
                )
                
                missing_bars = expected_indices.difference(ohlcv_df.index)
                if len(missing_bars) > 0:
                    self.logger.warning(f"Found {len(missing_bars)} missing bars for {currency_pair} {bar_size}. Filling with previous close.")
                    
                    # Create a DataFrame with the full expected index
                    full_ohlcv_df = pd.DataFrame(index=expected_indices)
                    
                    # Join with the actual data
                    full_ohlcv_df = full_ohlcv_df.join(ohlcv_df)
                    
                    # Forward fill missing values with previous close
                    full_ohlcv_df = full_ohlcv_df.fillna(method="ffill")
                    
                    # If there are still NaN values at the beginning, fill with the first valid value
                    if full_ohlcv_df.isna().any().any():
                        first_valid = ohlcv_df.iloc[0]
                        full_ohlcv_df = full_ohlcv_df.fillna({
                            "open": first_valid["open"],
                            "high": first_valid["high"],
                            "low": first_valid["low"],
                            "close": first_valid["close"],
                            "volume": 0  # Use 0 for volume in filled bars
                        })
                    
                    ohlcv_df = full_ohlcv_df
                
                # Reset index to get timestamp as a column
                ohlcv_df = ohlcv_df.reset_index()
                
                # Convert timestamp to milliseconds (UTC) for OHLCV format
                ohlcv_df["timestamp_utc_ms"] = ohlcv_df["index"].astype(np.int64) // 10**6
                
                # Ensure OHLCV data has correct columns and types
                ohlcv_df = ohlcv_df[["timestamp_utc_ms", "open", "high", "low", "close", "volume"]]
                
                # Set correct data types
                for column, dtype in self.ohlcv_output_format.items():
                    ohlcv_df[column] = ohlcv_df[column].astype(dtype)
                
                # Store the OHLCV DataFrame
                ohlcv_dfs[currency_pair] = ohlcv_df
                
                self.logger.info(f"Successfully generated {len(ohlcv_df)} OHLCV bars for {currency_pair}")
                
            except FileNotFoundError as e:
                self.logger.error(f"File not found error for {currency_pair}: {str(e)}")
            except ValueError as e:
                self.logger.error(f"Invalid data error for {currency_pair}: {str(e)}")
            except IOError as e:
                self.logger.error(f"I/O error for {currency_pair}: {str(e)}")
            except Exception as e:
                self.logger.exception(f"Unexpected error processing {currency_pair}: {str(e)}")
        
        # Combine all OHLCV DataFrames into a single file with a currency_pair column
        if not ohlcv_dfs:
            raise ValueError(f"No OHLCV data generated for any currency pair. Check logs for details.")
        
        # Add currency_pair as a column to each DataFrame
        for currency_pair, df in ohlcv_dfs.items():
            df["currency_pair"] = currency_pair
        
        # Combine all DataFrames
        combined_ohlcv_df = pd.concat(ohlcv_dfs.values(), ignore_index=True)
        
        # Write to Parquet
        self.logger.info(f"Writing combined OHLCV data to {output_parquet_path}")
        table = pa.Table.from_pandas(combined_ohlcv_df)
        pq.write_table(table, output_parquet_path)
        
        self.logger.info(f"Successfully wrote {len(combined_ohlcv_df)} OHLCV bars to {output_parquet_path}")
        
        return output_parquet_path

    def aggregate_ohlcv(self, 
                       ohlcv_1min_parquet_path: str, 
                       output_parquet_path: str, 
                       bar_size: str) -> str:
        """
        Aggregate 1-minute OHLCV Parquet data to longer period bars and save to Parquet.
        
        This method takes 1-minute OHLCV data from a Parquet file, aggregates it to
        the specified longer period (e.g., 5min, 1H), and saves the result to a new
        Parquet file.
        
        Args:
            ohlcv_1min_parquet_path (str): Path to the 1-minute OHLCV Parquet file.
            output_parquet_path (str): Path where the aggregated OHLCV Parquet file will be saved.
            bar_size (str): Size of bars to generate. Must be one of the valid_bar_sizes
                           and longer than '1min'.
                           
        Returns:
            str: Path to the generated aggregated OHLCV Parquet file.
            
        Raises:
            ValueError: If bar_size is not valid or is '1min'.
            FileNotFoundError: If the input 1-minute OHLCV Parquet file doesn't exist.
            IOError: If there are issues reading or writing Parquet files.
        """
        if bar_size not in self.valid_bar_sizes:
            raise ValueError(f"Invalid bar size: {bar_size}. Valid sizes are: {self.valid_bar_sizes}")
            
        if bar_size == '1min':
            raise ValueError("Cannot aggregate to 1min as it's already the base timeframe")
            
        self.logger.info(f"Aggregating 1-minute OHLCV data to {bar_size}")
        
        # Implementation will be added in future iterations
        pass
        
    def _extract_currency_pair_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract currency pair from a filename.
        
        Helper method to extract the currency pair code from a filename. Assumes
        the filename follows a pattern like 'EURUSD_ticks_20231026.json' where the
        currency pair is the first part before the first underscore.
        
        Args:
            filename (str): The filename to extract currency pair from.
            
        Returns:
            Optional[str]: Extracted currency pair or None if it couldn't be extracted or
                          is not in the list of known currency pairs.
        """
        # Define known currency pairs according to the specification
        known_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
            'NZDUSD', 'USDCAD', 'USDNOK', 'USDSEK'
        ]
        
        try:
            # Try pattern like 'EURUSD_ticks_20231026.json'
            parts = filename.split('_')
            if len(parts) < 2:
                self.logger.debug(f"Filename does not match expected pattern with underscores: {filename}")
                
                # Try alternative pattern - may have no underscore but currency pair at start
                # For example: 'EURUSD-ticks-20231026.json' or 'EURUSD.json'
                for pair in known_pairs:
                    if filename.startswith(pair):
                        self.logger.debug(f"Found currency pair at start of filename: {pair}")
                        return pair
                
                return None
            
            # Check if first part is a known currency pair
            potential_pair = parts[0].upper()
            
            # Sometimes files might have the currency pair in a different format
            # e.g., EUR_USD_ticks.json instead of EURUSD_ticks.json
            if potential_pair not in known_pairs and len(potential_pair) == 3:
                # Try to see if next part is also a 3-letter currency code
                if len(parts) > 1 and len(parts[1]) == 3:
                    combined_pair = potential_pair + parts[1].upper()
                    if combined_pair in known_pairs:
                        self.logger.debug(f"Combined currency pair from parts: {combined_pair}")
                        return combined_pair
            
            # Check if the potential pair is in our known list
            if potential_pair in known_pairs:
                return potential_pair
            
            # It could be that the currency pair is embedded within the first part
            # e.g., filename-EURUSD-ticks.json
            for pair in known_pairs:
                if pair in potential_pair:
                    self.logger.debug(f"Extracted embedded currency pair: {pair} from {potential_pair}")
                    return pair
            
            self.logger.debug(f"Extracted currency pair '{potential_pair}' is not in the list of known pairs")
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting currency pair from filename {filename}: {str(e)}")
            return None 
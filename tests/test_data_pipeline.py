#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module for DataPipeline class in MCATS Data Layer.

This module contains tests for the DataPipeline class, ensuring that its methods
function correctly and handle edge cases appropriately.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock
import sys

# Add the parent directory of MCATS to the system 
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

# Import the DataPipeline class
from MCATS.data_layer.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Test cases for the DataPipeline class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.data_pipeline = DataPipeline()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_data_dir = Path(self.temp_dir.name)
        
        # Sample currency pairs for testing
        self.test_currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']

    def tearDown(self):
        """Clean up test environment after each test."""
        self.temp_dir.cleanup()

    def create_test_json_file(self, currency_pair='EURUSD'):
        """Helper method to create a test JSON file for tick data."""
        test_data = [
            {
                "timestamp_utc": 1625097600000,  # 2021-07-01 00:00:00 UTC
                "ask": 1.18500,
                "bid": 1.18480
            },
            {
                "timestamp_utc": 1625097601000,  # 2021-07-01 00:00:01 UTC
                "ask": 1.18510,
                "bid": 1.18490
            }
        ]
        
        file_path = self.test_data_dir / f"{currency_pair}_ticks_20210701.json"
        with open(file_path, 'w') as f:
            json.dump(test_data, f)
            
        return file_path

    def test_initialization(self):
        """Test that DataPipeline initializes with correct default values."""
        self.assertEqual(self.data_pipeline.bar_size_default, '1min')
        self.assertEqual(self.data_pipeline.valid_bar_sizes, ['1min', '5min', '10min', '1H'])
        self.assertEqual(self.data_pipeline.est_timezone.zone, 'US/Eastern')

    def test_extract_currency_pair_from_filename(self):
        """Test extraction of currency pair from filename."""
        # Test valid filenames
        self.assertEqual(
            self.data_pipeline._extract_currency_pair_from_filename('EURUSD_ticks_20210701.json'),
            'EURUSD'
        )
        self.assertEqual(
            self.data_pipeline._extract_currency_pair_from_filename('GBPUSD_something_else.json'),
            'GBPUSD'
        )
        
        # Test invalid filename
        self.assertIsNone(
            self.data_pipeline._extract_currency_pair_from_filename('invalid_filename.json')
        )

    @patch('MCATS.data_layer.data_pipeline.DataPipeline.generate_tick_parquet')
    def test_process_raw_data(self, mock_generate_tick_parquet):
        """Test processing of raw JSON files to Parquet."""
        # Create test JSON files
        json_file1 = self.create_test_json_file('EURUSD')
        json_file2 = self.create_test_json_file('GBPUSD')
        
        # Call the method under test
        self.data_pipeline.process_raw_data(
            [str(json_file1), str(json_file2)],
            str(self.test_data_dir / 'output')
        )
        
        # Check that generate_tick_parquet was called for each file
        self.assertEqual(mock_generate_tick_parquet.call_count, 2)

    def test_aggregate_ohlcv_invalid_bar_size(self):
        """Test that aggregate_ohlcv raises ValueError for invalid bar size."""
        with self.assertRaises(ValueError):
            self.data_pipeline.aggregate_ohlcv(
                'input.parquet',
                'output.parquet',
                'invalid_size'
            )

    def test_aggregate_ohlcv_1min_bar_size(self):
        """Test that aggregate_ohlcv raises ValueError for 1min bar size."""
        with self.assertRaises(ValueError):
            self.data_pipeline.aggregate_ohlcv(
                'input.parquet',
                'output.parquet',
                '1min'
            )

    def test_generate_ohlcv_invalid_bar_size(self):
        """Test that generate_ohlcv raises ValueError for invalid bar size."""
        with self.assertRaises(ValueError):
            self.data_pipeline.generate_ohlcv(
                'input_dir',
                'output.parquet',
                self.test_currency_pairs,
                'invalid_size'
            )


if __name__ == '__main__':
    unittest.main() 
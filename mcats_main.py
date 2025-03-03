#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Currency Automated Trading System (MCATS)
Main entry point for the MCATS system.

This file serves as the main entry point for the MCATS system, importing core modules
and providing a main() function that can be executed when the file is run directly.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

# Import modules from MCATS packages
from MCATS.logs.logging_config import setup_logging
from MCATS.configuration.config import load_config, SystemConfig
from MCATS.data_layer.data_pipeline import DataPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MCATS: Multi-Currency Automated Trading System')
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')
    
    # Data processing sub-command
    data_parser = subparsers.add_parser('data', help='Data processing commands')
    data_subparsers = data_parser.add_subparsers(dest='data_command', help='Data sub-command help')
    
    # Process raw data command
    process_parser = data_subparsers.add_parser('process', help='Process raw JSON data to Parquet tick files')
    process_parser.add_argument('--input-dir', type=str, required=True, help='Directory containing raw JSON files')
    process_parser.add_argument('--output-dir', type=str, required=True, help='Output directory for Parquet tick files')
    
    # Generate OHLCV command
    ohlcv_parser = data_subparsers.add_parser('ohlcv', help='Generate OHLCV data from tick data')
    ohlcv_parser.add_argument('--input-dir', type=str, required=True, help='Directory containing Parquet tick files')
    ohlcv_parser.add_argument('--output-file', type=str, required=True, help='Output Parquet file for OHLCV data')
    ohlcv_parser.add_argument('--currency-pairs', type=str, nargs='+', help='Currency pairs to process')
    ohlcv_parser.add_argument('--bar-size', type=str, default='1min', help='Bar size for OHLCV data')
    
    return parser.parse_args()


def main():
    """Main function to run the MCATS system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level)
    logger.info("Starting MCATS (Multi-Currency Automated Trading System)")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration")
    
    try:
        # Process commands
        if args.command == 'data':
            handle_data_commands(args, config, logger)
        else:
            logger.info("No command specified, exiting")
            
        logger.info("MCATS completed successfully")
            
    except Exception as e:
        logger.error(f"Error in MCATS: {e}", exc_info=True)
        sys.exit(1)


def handle_data_commands(args, config, logger):
    """Handle data processing commands."""
    # Initialize DataPipeline
    data_pipeline = DataPipeline()
    
    if args.data_command == 'process':
        # Process raw JSON data to Parquet tick files
        logger.info(f"Processing raw data from {args.input_dir} to {args.output_dir}")
        
        # Get list of JSON files in input directory
        input_dir = Path(args.input_dir)
        json_files = list(input_dir.glob('*.json'))
        
        if not json_files:
            logger.warning(f"No JSON files found in {args.input_dir}")
            return
            
        # Process raw data
        data_pipeline.process_raw_data([str(f) for f in json_files], args.output_dir)
        logger.info(f"Processed {len(json_files)} JSON files to Parquet tick files")
        
    elif args.data_command == 'ohlcv':
        # Generate OHLCV data from tick data
        logger.info(f"Generating OHLCV data from {args.input_dir} to {args.output_file}")
        
        # Use currency pairs from arguments or configuration
        currency_pairs = args.currency_pairs if args.currency_pairs else config.data_pipeline_config.currency_pairs
        
        # Generate OHLCV data
        data_pipeline.generate_ohlcv(
            args.input_dir,
            args.output_file,
            currency_pairs,
            args.bar_size
        )
        logger.info(f"Generated {args.bar_size} OHLCV data for {len(currency_pairs)} currency pairs")
    else:
        logger.warning(f"Unknown data command: {args.data_command}")


if __name__ == "__main__":
    main() 
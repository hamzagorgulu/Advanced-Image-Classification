"""
run.py: Main script to run experiments with various configurations.

This script is the entry point for running the experiment pipeline. It reads configuration
from a JSON file, sets up data preparation and transformations, and executes the experiment runner.
"""

import argparse
import json
from src.utils.experiment_runner import ExperimentRunner
from src.utils.data_preparation import DataPreparation
from src.models.model import FlexibleConvLayer, FlexibleResNet
from src.utils.helpers import load_config, setup_transformations

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load configuration from JSON file and run experiments.")
    parser.add_argument("--config", type=str, default="src/config/config.json",
                        help="Path to the JSON configuration file")

    args = parser.parse_args()
    config = load_config(args.config)

    # Set up data transformations
    train_transform, test_transform = setup_transformations()

    # Initialize data preparation
    data_prep = DataPreparation(
        config['general_config']['batch_size'],
        train_transform,
        test_transform
    )

    # Load data
    trainloader, testloader = data_prep.load_data()

    # Initialize experiment runner
    experiment_runner = ExperimentRunner(
        config['grid_search_config'],
        config['general_config'],
        trainloader,
        testloader
    )

    # Start experiment
    experiment_runner.run_experiments()

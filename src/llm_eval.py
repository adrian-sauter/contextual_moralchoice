import json
import pandas as pd
import numpy as np
import torch
import pickle
import os
import argparse
from tqdm import tqdm
from src.config import PATH_RESULTS


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Test Effect of Steering Vector on Model Capabilities")
parser.add_argument(
    "--model-name", 
    default="meta/llama-3.1-8b-instruct", 
    type=str
)

parser.add_argument(
    "--experiment-name", 
    type=str, 
    required=True
)

parser.add_argument(
    "--test-id-path",
    type=str,
    help="Path to a JSON file containing test IDs to exclude from training"
)

parser.add_argument(
    "--vector-type",
    default=["unweighted"],
    type=str,
    nargs="+",
    help="Type of vector to use for steering (e.g., 'unweighted', 'weighted')"
)

parser.add_argument(
    "--vector-type",
    default=["unweighted"],
    type=str,
    nargs="+",
    help="Type of vector to use for steering (e.g., 'unweighted', 'weighted')"
)

args = parser.parse_args()




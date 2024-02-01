import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import json
from src.utils.data_preparation import Transforms

def load_config(config_path: str) -> dict:
    """
    Load the configuration from a JSON file.

    Parameters:
    - config_path: The path to the JSON configuration file.

    Returns:
    - config: The configuration dictionary.
    """
    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        #logging.error("Configuration file not found.")
        raise
    except json.JSONDecodeError:
        #logging.error("Configuration file is not valid JSON.")
        raise
        
def setup_transformations():
    """
    Sets up the data transformations for training and testing.

    Returns:
    - train_transform: The transformations to apply to the training data.
    - test_transform: The transformations to apply to the test data.
    """
    train_transform = A.Compose([
        # Normalize (mean and std are per channel)
        A.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768)),

        # Random Horizontal Flip with a probability of 0.3
        A.HorizontalFlip(p=0.3),  # 73.88 - 77.68

        # Random Rotation within 30 degrees
        A.Rotate(limit=30, p=0.5), # 77.68 - 78.21

        # Random Erasing
        #A.CoarseDropout(max_holes=1, max_height=0.1, max_width=0.1, min_holes=1, min_height=0.02, min_width=0.02, fill_value=255, p=0.75), # so bad

        # Simulating Sharpness Adjustment
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5)
        ], p=0.1),  # Adjust 'p' for probability of applying this block of transformations

        # Convert to tensor
        ToTensorV2(),
    ])
    
    test_transform = A.Compose([
         A.Normalize(mean=(0.49139968, 0.48215827, 0.44653124), std=(0.24703233, 0.24348505, 0.26158768)),
         ToTensorV2(),
         ])
    
    return Transforms(train_transform), Transforms(test_transform)  
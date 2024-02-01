import torch.optim as optim
from typing import Type, Dict, List
from pydantic import BaseModel, Field
import torch
from src.models.model import FlexibleConvLayer, FlexibleResNet


class CNNModelConfig(BaseModel):
    """
    Configuration class for creating a PyTorch model.

    This class serves as a configuration container for instantiating PyTorch models.
    It holds parameters required to initialize a model and provides a method to create
    the model instance with these parameters.

    Attributes:
        model_class (Type[torch.nn.Module]): The class of the PyTorch model to be instantiated.
        initial_filters (int): The number of initial filters for the model, typically used in convolutional layers.
        fc_layer_sizes (List[int]): A list representing the sizes of fully connected layers in the model.
        num_classes (int): The number of output classes the model should predict.
        num_conv_layers(int): The number of convolution layers.

    Methods:
        create_model: Creates an instance of the PyTorch model using the stored configuration.
    """

    network_model_class: Type[torch.nn.Module]
    initial_filters: int = Field(..., gt=3) # select higher value than the channel size
    fc_layer_sizes: List[int] = Field(..., min_items=1)  # has to be specified
    num_classes: int = Field(..., gt=0)  
    num_conv_layers: int = Field(...,lt=20, gt=0) # high value of conv layers cause out of memory error

    def create_model(self):
        """
        Create an instance of the PyTorch model.

        This method instantiates the model defined in `model_class` with the parameters
        provided in the configuration (initial_filters, fc_layer_sizes, and num_classes).

        Returns:
            An instance of the specified PyTorch model class initialized with the given parameters.
        """
        return self.network_model_class(initial_filters=self.initial_filters,
                                fc_layer_sizes=self.fc_layer_sizes,
                                num_classes=self.num_classes,
                                num_conv_layers = self.num_conv_layers)
    
    
class ResNetModelConfig(BaseModel):
    """
    Configuration class for creating a PyTorch model.

    This class serves as a configuration container for instantiating PyTorch models.
    It holds parameters required to initialize a model and provides a method to create
    the model instance with these parameters.

    Attributes:
        network_model_class (Type[torch.nn.Module]): The class of the PyTorch model to be instantiated.
        num_blocks (int): Number of residual blocks.

    Methods:
        create_model: Creates an instance of the PyTorch model using the stored configuration.
    """
    
    network_model_class: Type[torch.nn.Module]
    num_blocks: int = Field(..., gt=0)  # Must be greater than 0

    def create_model(self):
        """
        Create an instance of the PyTorch model.

        This method instantiates the model defined in `network_model_class` with the num_blocks parameters.

        Returns:
            An instance of the specified PyTorch model class initialized with the given parameters.
        """
        return self.network_model_class(num_blocks = self.num_blocks)
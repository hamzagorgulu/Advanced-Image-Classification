import torch
import torch.optim as optim
from typing import Type, Union, Literal
from pydantic import BaseModel, validator, Field

class TrainingConfig(BaseModel):
    """
    Configuration class for setting up training parameters and validating them for a PyTorch model.

    This class contains the configuration necessary for training a PyTorch model, including
    the model itself, optimizer, number of epochs, and other training parameters. It also
    includes a method to setup the model and optimizer for training.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer to use for training the model.
        epochs (int): The number of training epochs. Must be greater than 0 and less than 1001.
        device (Union[Literal["cuda"], Literal["cpu"]]): The device to use for training ('cuda' or 'cpu').
        step_size (int): Step size for the optimizer, must be between 0 and 30 (inclusive).
        alpha (float): Alpha parameter for the optimizer, must be between 0 and 1 (inclusive).

    Methods:
        setup: Prepares the model and optimizer instances for training.
        validate_device: Validates if the specified device is available.

    Configuration:
        arbitrary_types_allowed: Allows arbitrary types (e.g., custom PyTorch modules) in Pydantic models.
    """

    model: torch.nn.Module
    optimizer: optim.Optimizer
    epochs: int = Field(..., gt=0, lt=1001)
    device: Union[Literal["cuda"], Literal["cpu"]]
    step_size: int = Field(..., ge=0, le=30)
    alpha: float = Field(..., ge=0, le=1)

    class Config:
        arbitrary_types_allowed = True

    @validator('device')
    def validate_device(cls, v):
        """
        Validates the selected device.

        Ensures that the device specified for training (either CUDA or CPU) is available.
        If CUDA is selected but not available, raises a ValueError. 
        Automatically called by pydantic during the instantiation of a TrainingConfig.

        Args:
            v (Union[Literal["cuda"], Literal["cpu"]]): The device to be validated.

        Returns:
            The validated device.

        Raises:
            ValueError: If CUDA is selected but not available.
        """
        if not torch.cuda.is_available() and v == "cuda":
            raise ValueError("CUDA is not available on this system.")
        return v

    def setup(self):
        """
        Sets up the training environment.

        Instantiates the model and optimizer based on the provided configuration and moves the model
        to the specified device (CUDA or CPU).

        Returns:
            tuple: A tuple containing the instantiated model and optimizer.
        """
        model_instance = self.model().to(self.device)
        optimizer_instance = self.optimizer(model_instance.parameters())
        return model_instance, optimizer_instance

import torch.optim as optim
from typing import Type, Dict
from pydantic import BaseModel, Field
import torch

class OptimizerConfig(BaseModel):
    """
    Configuration class for creating a PyTorch optimizer.

    This class serves as a configuration container for instantiating optimizers in PyTorch.
    It holds the optimizer class type, learning rate, and any additional optimizer-specific
    keyword arguments. It also provides a method to create an optimizer instance with these parameters.

    Attributes:
        optimizer_class (Type[optim.Optimizer]): The class of the PyTorch optimizer to be instantiated.
        lr (float): The learning rate for the optimizer.
        optimizer_kwargs (Dict[str, float]): A dictionary containing any additional keyword arguments
                                             required for the optimizer. The keys are the argument names
                                             and the values are their respective values.

    Methods:
        create_optimizer: Creates an instance of the PyTorch optimizer using the stored configuration.
    """

    optimizer_class: Type[optim.Optimizer]
    lr: float = Field(..., gt=0)  # Learning rate must be greater than 0
    optimizer_kwargs: Dict[str, float] = Field(default_factory=dict)  

    def create_optimizer(self, parameters):
        """
        Create an instance of the PyTorch optimizer.

        This method instantiates the optimizer defined in `optimizer_class` with the provided 
        parameters, learning rate (`lr`), and any additional keyword arguments (`optimizer_kwargs`).

        Args:
            parameters: The parameters of the model over which to optimize.

        Returns:
            An instance of the specified PyTorch optimizer class initialized with the given parameters
            and configuration.
        """
        return self.optimizer_class(parameters, lr=self.lr, **self.optimizer_kwargs)
        

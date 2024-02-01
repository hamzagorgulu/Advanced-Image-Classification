import itertools
from src.utils.model_wrapper import ModelWrapper 
from torch import optim
from src.models.model import FlexibleConvLayer, FlexibleResNet

class ExperimentRunner:
    """
    A class for running experiments with different combinations of models, hyperparameters, and data loaders.

    Parameters:
    grid_search_config (dict): Configuration dictionary containing lists of models, learning rates, step sizes, optimizers, and more.
    general_config (dict): General configuration dictionary containing experiment settings.
    trainloader (torch.utils.data.DataLoader): DataLoader for training data.
    testloader (torch.utils.data.DataLoader): DataLoader for test data.

    Attributes:
    models (list): List of model classes to be used in the experiments.
    learning_rates (list): List of learning rates to be tested.
    step_sizes (list): List of step sizes to be tested.
    optimizers (list): List of optimizer names and their respective keyword arguments to be tested.
    num_conv_layers (list): List of numbers of convolutional layers to be tested.
    trainloader (torch.utils.data.DataLoader): DataLoader for training data.
    testloader (torch.utils.data.DataLoader): DataLoader for test data.
    checkpoint_save_threshold (float): Threshold accuracy for saving model checkpoints.
    """
    def __init__(self, grid_search_config, general_config, trainloader, testloader):
        # pydantic
        
        # configs
        self.general_config = general_config
        self.grid_search_config = grid_search_config
        
        # grid search lists
        self.models = grid_search_config["models"]
        self.learning_rates = grid_search_config["learning_rates"]
        self.step_sizes = grid_search_config["step_sizes"]
        self.optimizers = grid_search_config["optimizers"]
        self.num_conv_layers = grid_search_config["num_conv_layers"]
        
        # dataloaders
        self.trainloader = trainloader
        self.testloader = testloader
        
        self.checkpoint_save_threshold = general_config["checkpoint_save_threshold"]

    def run_experiments(self):
        """
        Run experiments for various combinations of models, hyperparameters, and data loaders.

        This method iterates through all combinations specified in the grid search configuration and runs experiments
        with different model architectures, learning rates, step sizes, optimizers, and numbers of convolutional layers.
        It prints experiment details, trains the models, plots and saves metrics, and optionally saves model checkpoints.

        Returns:
        None
        """
        
        # Loop through all combinations and run experiments
        
        for network_model_class, lr, step_size, (opt_class, opt_kwargs), num_conv_layer in itertools.product(self.models, self.learning_rates, self.step_sizes, self.optimizers, self.num_conv_layers):
            print(f"Running experiment with Model: {network_model_class}, EPOCHS: {self.general_config['epoch_size']}, LR: {lr}, Step Size: {step_size}, Optimizer: {opt_class}, Num. conv layers: {num_conv_layer}")
            
            model = eval(network_model_class)
            optimizer = eval(f"optim.{opt_class}")
            
            experiment_config = {
                "model": model,
                "optimizer": optimizer,
                "optimizer_kwargs": opt_kwargs,
                "step_size": step_size,
                "lr": lr,
                "num_conv_layer": num_conv_layer
            }
            
            # Initialize ModelWrapper
            model_wrapper = ModelWrapper(self.general_config, experiment_config)
            
            train_losses, train_accuracies, test_losses, test_accuracies = model_wrapper.train_and_test(self.trainloader, self.testloader)
            #test_losses, test_accuracies = model_wrapper.evaluate(self.test_loader)

            # Plot and save metrics
            model_wrapper.plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)
            
            if test_accuracies[-1] > self.checkpoint_save_threshold:
                model_wrapper.save_model(f"./checkpoints/{model_wrapper.experiment_id}.pth")
            

            # You can also store or log results for later analysis
        

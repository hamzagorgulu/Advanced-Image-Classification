import matplotlib.pyplot as plt
import torch
import os
from src.config.model_config import CNNModelConfig, ResNetModelConfig
from src.config.optimizer_config import OptimizerConfig
from src.config.training_config import TrainingConfig
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

class ModelWrapper:
    """
    A wrapper class for training, testing, and visualizing deep learning models.

    Parameters:
    general_config (dict): General experiment settings.
    experiment_config (dict): Experiment-specific configurations.

    Attributes:
    model (torch.nn.Module): The model to be trained and tested.
    optimizer (torch.optim.Optimizer): The optimizer for training.
    training_config (TrainingConfig): Training settings.
    general_config (dict): General experiment settings.
    experiment_config (dict): Experiment-specific configurations.
    criterion (torch.nn.CrossEntropyLoss): Loss function.
    scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler.
    epoch_size (int): Number of training epochs.
    device (str): Training device (e.g., 'cuda' or 'cpu').
    use_amp (bool): Enable automatic mixed-precision training (AMP).
    num_conv_layer (int): Number of convolutional layers.
    experiment_id (str): Experiment identifier.

    Methods:
    generate_experiment_id(): Generate an experiment identifier.
    train_and_test(trainloader, testloader): Train and test the model.
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies): Plot and save metrics.
    save_model(path): Save model state.
    load_model(path, device): Load model state.
    get_n_params(model): Calculate trainable parameters.

    Example:
    ```python
    # Create a ModelWrapper instance and run experiments
    wrapper = ModelWrapper(general_config, experiment_config)
    train_losses, train_accuracies, test_losses, test_accuracies = wrapper.train_and_test(trainloader, testloader)
    wrapper.plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)
    wrapper.save_model("experiment_checkpoint.pth")
    ```
    """
    def __init__(self, general_config, experiment_config):
        # apply pydantic
        # Define the training configuration
        if experiment_config["model"].__name__.startswith("FlexibleConvLayer"): 
            self.model = CNNModelConfig(
                    network_model_class=experiment_config["model"],  # model_config: model_class, initial_filters, fc_layer_sizes, num_classes
                    initial_filters = general_config["initial_filters"],
                    fc_layer_sizes = general_config["fc_layer_sizes"],
                    num_classes = general_config["num_classes"],
                    num_conv_layers = experiment_config["num_conv_layer"]
                ).create_model().to(general_config["device"])
        
        elif experiment_config["model"].__name__.startswith("FlexibleResNet"):
            self.model = ResNetModelConfig(network_model_class=experiment_config["model"], 
                                           num_blocks = experiment_config["num_conv_layer"]
                                          ).create_model().to(general_config["device"])
        
        else:
            print("Check the model_wrapper")
        
        self.optimizer = OptimizerConfig(  
                optimizer_class = experiment_config["optimizer"],     # optimizer_config: optimizer_class, lr, optimizer_kwargs
                lr = experiment_config["lr"],
                optimizer_kwargs = experiment_config["optimizer_kwargs"]
            ).create_optimizer(self.model.parameters())
        
        self.training_config = TrainingConfig(model=self.model,
                                      optimizer=self.optimizer,
                                      epochs=general_config["epoch_size"],
                                      device=general_config["device"],
                                      step_size=experiment_config["step_size"],
                                      alpha=general_config["alpha"])
        
        self.general_config = general_config
        self.experiment_config = experiment_config
        
        self.criterion = eval(general_config["criterion"])()  # initialize cross entropy loss
        self.scheduler =eval(general_config["lr_scheduler"])(self.optimizer, step_size=experiment_config["step_size"], gamma=0.1)
        self.epoch_size = general_config["epoch_size"]
        self.device = general_config["device"]
        self.use_amp = general_config["use_amp"]
        self.num_conv_layer = experiment_config["num_conv_layer"]

        self.experiment_id = self.generate_experiment_id()
        
    def generate_experiment_id(self):
        """
        Generates a unique experiment identifier based on model, optimizer, and hyperparameter configurations.

        Returns:
        str: Experiment identifier.
        """
        optimizer_name = self.optimizer.__class__.__name__
        model_class_name = self.model.__class__.__name__
        num_conv_layer = self.num_conv_layer
        lr = self.optimizer.param_groups[0]['lr']
        step_size = self.scheduler.step_size
        epoch_size = self.epoch_size
        weight_decay = 'no_weight_decay' if 'weight_decay' not in self.optimizer.defaults else self.optimizer.defaults['weight_decay']
        return f"Model_{model_class_name}_Epochs_{epoch_size}_NumConvLayers_{num_conv_layer}_Optimizer_{optimizer_name}_LR_{lr}_Step_{step_size}_WeightDecay_{weight_decay}"

    def train_and_test(self, trainloader, testloader):
        """
        Train and test the model. At the end of each epoch, records the test performance results for both sets.

        Parameters:
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.

        Returns:
        list: Training losses over epochs.
        list: Training accuracies over epochs.
        list: Test losses over epochs.
        list: Test accuracies over epochs.
        """
        print(self.model)
        
        self.model.train()
        total_params = self.get_n_params(self.model)
        print(f"Total Number of params: {total_params}")
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        scaler = GradScaler()

        for epoch in range(self.epoch_size):
            train_loss = 0
            train_total = 0
            train_correct = 0
            
            test_loss = 0
            test_correct = 0
            test_total = 0

            for idx, (imgs, labels) in enumerate(trainloader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                if self.use_amp:
                    with autocast():
                        outputs = self.model(imgs)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                    self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()
            
            self.scheduler.step()

            # Record the average loss and accuracy for each epoch
            train_loss /= len(trainloader)
            train_accuracy = 100 * train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            
            
            # iterate over test set for each epoch
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    test_loss += self.criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_loss /= len(testloader)
            test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
            print(f'[{epoch + 1}, {self.epoch_size}] Train Accuracy: {train_accuracy}%,Train Loss: {train_loss} | Test Accuracy: {test_accuracy}% , Test Loss: {test_loss}')
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        return train_losses, train_accuracies, test_losses, test_accuracies
    
    def plot_metrics(self, train_losses, train_accuracies, test_losses, test_accuracies):
        """
        Plots and saves training and testing metrics, including loss and accuracy.

        Parameters:
        train_losses (list): Training losses over epochs.
        train_accuracies (list): Training accuracies over epochs.
        test_losses (list): Test losses over epochs.
        test_accuracies (list): Test accuracies over epochs.

        Returns:
        None
        """
        plt.figure(figsize=(15, 7))
        final_train_accuracy = train_accuracies[-1] if train_accuracies else 0
        final_test_accuracy = test_accuracies[-1] if test_accuracies else 0
        optimizer_name = self.optimizer.__class__.__name__
        model_class_name = self.model.__class__.__name__

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue', linestyle='--')
        plt.plot(test_losses, label='Test Loss', color='red', linestyle='-')
        plt.title(f"Loss (Final Train Acc: {round(final_train_accuracy, 3)})")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='blue', linestyle='--')
        plt.plot(test_accuracies, label='Test Accuracy', color='red', linestyle='-')
        plt.title(f"Accuracy (Final Test Acc: {round(final_test_accuracy, 3)})")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Enhanced information
        plt.figtext(0.5, 0.01, f"Model: {model_class_name}, Optimizer: {optimizer_name}, LR: {self.optimizer.param_groups[0]['lr']}, Step: {self.scheduler.step_size}, Weight Decay: {'no_weight_decay' if 'weight_decay' not in self.optimizer.defaults else self.optimizer.defaults['weight_decay']}", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        fig_name = f"results/graphs/{self.experiment_id}_TrainAcc_{round(final_train_accuracy, 3)}_TestAcc_{round(final_test_accuracy, 3)}.png"
        plt.savefig(fig_name)
        plt.close()
        
    def save_model(self, path):
        """
        Saves the model state to a specified path.

        Parameters:
        path (str): The file path where the model state will be saved.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path, device):
        """
        Loads the model state from a specified path.

        Parameters:
        path (str): The file path from where the model state will be loaded.
        device (torch.device): The device to load the model onto.
        """
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.model.to(device)
        print(f"Model loaded from {path}")
        
    def get_n_params(self, model):
        """
        Calculates the total number of trainable parameters in the deep learning model.

        Parameters:
        model (torch.nn.Module): The deep learning model.

        Returns:
        int: Total number of trainable parameters.
        """
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

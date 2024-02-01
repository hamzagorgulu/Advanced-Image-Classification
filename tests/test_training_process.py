import torch
import torch.optim as optim
import torch.nn as nn
import unittest
from src.models.model import FlexibleResNet, FlexibleConvLayer

BATCH_SIZE = 5
NUM_CLASSES = 10
DATASET_SIZE = 20

class TestModels(unittest.TestCase):
    def train_and_test_loss(self, model):
        """
        Trains a given model on a dummy dataset for a few epochs and verifies 
        that the loss decreases. This method creates a dummy dataset, defines a loss function
        and an optimizer, and then runs a training loop to update the model's weights.

        Args:
        model (torch.nn.Module): The PyTorch model to be trained and tested.

        Asserts:
        final_loss is less than initial_loss: Ensures that the model is learning from the dummy dataset.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create a dummy dataset
        inputs = torch.randn(DATASET_SIZE, 3, 32, 32)
        targets = torch.randint(0, NUM_CLASSES, (DATASET_SIZE,))

        initial_loss = None
        final_loss = None

        for epoch in range(5):
            for i in range(0, len(inputs), BATCH_SIZE):
                batch_inputs = inputs[i:i+BATCH_SIZE]
                batch_targets = targets[i:i+BATCH_SIZE]

                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                if initial_loss is None:
                    initial_loss = loss.item()
                final_loss = loss.item()

        self.assertTrue(final_loss < initial_loss)

    def test_multiple_models(self):
        """
        Tests multiple neural network models to ensure their training loss decreases. 
        This method iterates over a predefined list of models, using the `train_and_test_loss` 
        method to train each model on a dummy dataset and verify the decrease in loss.

        The method utilizes the `subTest` context manager for isolation and clearer debugging 
        in case of test failures.
        """
        models = [FlexibleResNet(), FlexibleConvLayer()]
        for model in models:
            with self.subTest(model=model):
                self.train_and_test_loss(model)

if __name__ == '__main__':
    unittest.main()

import torch
import unittest
import os
from src.models.model import FlexibleResNet, FlexibleConvLayer

class TestModelIO(unittest.TestCase):

    def save_and_load_model(self, model, save_path):
        """
        Saves and loads a model to ensure state preservation.

        Args:
            model (torch.nn.Module): The model to test.
            save_path (str): File path to save and load the model.
        """
        # Save the model
        torch.save(model.state_dict(), save_path)

        # Load the model into a new instance
        model_loaded = type(model)()  # Create a new instance of the same class as 'model'
        model_loaded.load_state_dict(torch.load(save_path))

        # Compare parameters of the original and loaded model
        for param, loaded_param in zip(model.parameters(), model_loaded.parameters()):
            self.assertTrue(torch.equal(param, loaded_param))

        # Delete the saved model file
        os.remove(save_path)

    def resnet_test_layer_io(self):
        """
        Test saving and loading the FlexibleResNet model.
        """
        model = FlexibleResNet()
        self.save_and_load_model(model, 'resnet_test_model.pth')
        print("FlexibleResNet save and load features are tested.")

    def convnet_layer_io(self):
        """
        Test saving and loading the FlexibleConvLayer model.
        """
        model = FlexibleConvLayer()
        self.save_and_load_model(model, 'convnet_test_model.pth')
        print("FlexibleConvLayer save and load features are tested.")

if __name__ == '__main__':
    unittest.main()

import torch
import unittest
from src.models.model import FlexibleResNet, FlexibleConvLayer

BATCH_SIZE = 5
NUM_CLASSES = 10

class TestModels(unittest.TestCase):
    def check_model_output_shape(self, model, input_shape, expected_output_shape):
        """
        Tests whether the output shape of a model is as expected.

        Args:
        model (torch.nn.Module): The PyTorch model to be tested.
        input_shape (tuple): The shape of the input tensor to be fed into the model.
        expected_output_shape (tuple): The expected shape of the output tensor from the model.

        Returns:
        None
        """
        dummy_input = torch.randn(input_shape)
        output = model(dummy_input)
        self.assertEqual(output.shape, expected_output_shape)

    def test_resnet_layer(self):
        """
        Tests the output shape of the FlexibleResNet model.
        """
        model = FlexibleResNet()
        self.check_model_output_shape(model, (BATCH_SIZE, 3, 32, 32), torch.Size([BATCH_SIZE, NUM_CLASSES]))
        print("FlexibleResNet output is tested")

    def test_convnet_layer(self):
        """
        Tests the output shape of the FlexibleConvLayer model.
        """
        model = FlexibleConvLayer()
        self.check_model_output_shape(model, (BATCH_SIZE, 3, 32, 32), torch.Size([BATCH_SIZE, NUM_CLASSES]))
        print("FlexibleConvLayer output is tested")

if __name__ == '__main__':
    unittest.main()

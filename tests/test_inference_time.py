import torch
import time
import unittest
from src.models.model import FlexibleResNet, FlexibleConvLayer

# to run the tests: python -m unittest discover tests -> run all test in "tests" directory

BATCH_SIZE = 5

class TestModelPerformance(unittest.TestCase):

    def measure_inference_time(self, model, input_shape):
        """
        Measures the inference time for a single image using the given model.

        Args:
            model (torch.nn.Module): The model to test.
            input_shape (tuple): The shape of the input tensor.

        Returns:
            float: Inference time for a single image.
        """
        dummy_input = torch.randn(input_shape)

        # Ensure model is in evaluation mode and no gradient computation
        model.eval()

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            model(dummy_input)
        elapsed_time = time.time() - start_time

        # Average inference time per image
        return elapsed_time / input_shape[0]

    def test_resnet_inference_time(self):
        """
        Test the inference time of the FlexibleResNet model.
        """
        model = FlexibleResNet()
        inference_time = self.measure_inference_time(model, (BATCH_SIZE, 3, 32, 32))
        print(f"Inference time for FlexibleResNet: {round(inference_time, 3)} seconds")
        self.assertLess(inference_time, 0.1)

    def test_convnet_inference_time(self):
        """
        Test the inference time of the FlexibleConvLayer model.
        """
        model = FlexibleConvLayer()
        inference_time = self.measure_inference_time(model, (BATCH_SIZE, 3, 32, 32))
        print(f"Inference time for FlexibleConvLayer: {round(inference_time, 3)} seconds")
        self.assertLess(inference_time, 0.1)


if __name__ == '__main__':
    unittest.main()

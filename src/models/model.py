import torch.nn as nn
import math
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ConvBlock(nn.Module):
    """
    A convolutional block module using PyTorch, which consists of a convolutional layer,
    a batch normalization layer, followed by an activation layer, and an optional max pooling layer.

    Parameters:
    in_channels (int): Number of input channels for the convolutional layer.
    out_channels (int): Number of output channels for the convolutional layer.
    use_leaky_relu (bool, optional): If True, uses LeakyReLU as the activation function; otherwise, uses ReLU. Default: False.
    leaky_relu_slope (float, optional): Negative slope coefficient for LeakyReLU. Default: 0.01.
    pool (bool, optional): If True, adds a max pooling layer after the activation layer. Default: False.
    """
    def __init__(self, in_channels, out_channels, use_leaky_relu=False, leaky_relu_slope=0.01, pool=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        ]

        if use_leaky_relu:
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        if pool:
            layers.append(nn.MaxPool2d(2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the convolutional block.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the convolutional block.
        """
        return self.block(x)

class FlexibleConvLayer(nn.Module):
    """
    A flexible convolutional layer module that can adapt its architecture based on the specified parameters.

    Parameters:
    num_channels (int, optional): Number of input channels. Default: 3.
    initial_filters (int, optional): Number of filters in the initial convolutional layer. Default: 32.
    num_conv_layers (int, optional): Number of convolutional layers in the model. Default: 4.
    fc_layer_sizes (list of int, optional): List of fully connected layer sizes. Default: [1024, 512, 128].
    num_classes (int, optional): Number of output classes. Default: 10.
    leaky_relu_slope (float, optional): Negative slope coefficient for LeakyReLU. Default: 0.01.
    """
    def __init__(self, num_channels=3, initial_filters=32, num_conv_layers=4, fc_layer_sizes=[1024, 512, 128], num_classes=10, leaky_relu_slope=0.01):
        super(FlexibleConvLayer, self).__init__()
        # total num. of layer: num_conv_layers + 3

        # Check to ensure a valid number of convolutional layers
        assert num_conv_layers >= 1, "Number of convolutional layers must be at least 1"

        # Convolutional layers using ConvBlock
        self.cnn_model = nn.Sequential()
        current_channels = num_channels
        for i in range(num_conv_layers):
            next_channels = initial_filters * (2 ** (i // 2))
            # Apply pooling less frequently
            use_pool = i % 2 == 1 and i < 6  # Example: Apply pooling on layers 1, 3, 5
            self.cnn_model.add_module(f"conv_block_{i}", ConvBlock(current_channels, next_channels, use_leaky_relu=True, leaky_relu_slope=leaky_relu_slope, pool=use_pool))
            current_channels = next_channels

        # Automatically calculate in_features
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, 32, 32) 
            dummy_output = self.cnn_model(dummy_input)
            in_features = dummy_output.view(1, -1).size(1)

        # Fully connected layers
        fc_layers = []
        for idx, layer_size in enumerate(fc_layer_sizes):
            if idx == 0:
                fc_layers.append(nn.Linear(in_features, layer_size))
            else:
                fc_layers.append(nn.Linear(fc_layer_sizes[idx-1], layer_size))

            fc_layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))
            fc_layers.append(nn.BatchNorm1d(num_features=layer_size))

        # Add the final layer
        fc_layers.append(nn.Linear(fc_layer_sizes[-1], num_classes))

        self.fc_model = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass through the flexible convolutional layer.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the flexible convolutional layer.
        """
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_model(x)
        return x
    
class FlexibleResNet(nn.Module):
    """
    A flexible ResNet model module that can adapt its architecture based on the specified parameters.

    Parameters:
    num_channels (int, optional): Number of input channels. Default: 3.
    num_classes (int, optional): Number of output classes. Default: 10.
    num_blocks (int, optional): Number of residual blocks in the model. Default: 2.
    initial_filters (int, optional): Number of filters in the initial convolutional layer. Default: 64.
    input_size (int, optional): Size of the input images. Default: 32.
    """
    def __init__(self, num_channels=3, num_classes=10, num_blocks=2, initial_filters=64, input_size=32):
        super(FlexibleResNet, self).__init__()
        # total num. of layer: 2*num_blocks + 3

        channels = initial_filters
        self.initial_conv = ConvBlock(num_channels, channels)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_pool = i % 2 == 1 and i < 6
            conv_block = nn.Sequential(
                ConvBlock(channels, channels * 2, pool=use_pool),
                ConvBlock(channels * 2, channels * 2)
            )
            res_block = nn.Sequential(
                ConvBlock(channels * 2, channels * 2),
                ConvBlock(channels * 2, channels * 2)
            )
            combined_block = nn.Sequential(conv_block, res_block)
            self.blocks.append(combined_block)
            channels *= 2

        # Initialize a dummy forward pass to determine the size
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, input_size, input_size)
            out_features = self.forward_conv(dummy_input).view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(out_features, num_classes)  
        )

    def forward_conv(self, xb):
        """
        Forward pass through the initial convolutional layers of the FlexibleResNet model,
        which helps to define the output shape of the residual blocks.

        Parameters:
        xb (torch.Tensor): Input tensor representing a batch of images.

        Returns:
        torch.Tensor: Output tensor after passing through the initial convolutional layers.
        """
        out = self.initial_conv(xb)
        for block in self.blocks:
            out = block(out)
        return out

    def forward(self, xb):
        """
        Forward pass through the flexible ResNet model.

        Parameters:
        xb (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the flexible ResNet model.
        """
        out = self.initial_conv(xb)
        for conv_block, res_block in self.blocks:
            out = conv_block(out)
            out = res_block(out) + out  # Residual connection
        out = self.classifier(out)
        return out

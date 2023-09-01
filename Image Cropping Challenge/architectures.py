"""
Different CNN architectures.
"""

import torch
import torch.nn as nn
from torchvision import models

# ----------------------------------------------------------------------------------------------------------------------
# Model A -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class ModelA(nn.Module):
    def __init__(self, targets: list, means: list, ids: list):
        """
        Model A: Fills the missing part with the mean of the image.
        """
        super(ModelA, self).__init__()

        self.targets = targets
        self.means = means
        self.ids = ids

    def forward(self):
        predictions = []

        means_selected = [self.means[id] for id in self.ids]

        for n, target in enumerate(self.targets):
            prediction = torch.empty_like(target)
            prediction = prediction.fill_(means_selected[n])
            predictions.append(prediction)

        return predictions

# ----------------------------------------------------------------------------------------------------------------------
# Model B --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class ModelB(torch.nn.Module):
    def __init__(self, n_in_channels: int = 2, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """
        Model B: A simple CNN that outputs a tensor with the same dimension as the input tensor.

        :param n_in_channels: number of input channels
        :param n_hidden_layers: number of hidden layers
        :param n_kernels: number of kernels
        :param kernel_size: size of kernels
        """
        super(ModelB, self).__init__()

        # create hidden layers
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size / 2)))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        # create output layer
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1, kernel_size=kernel_size,
                                            bias=True, padding=int(kernel_size / 2))

    def forward(self, x):

        cnn_out = self.hidden_layers(x)
        output = self.output_layer(cnn_out)

        return output

# ----------------------------------------------------------------------------------------------------------------------
# CustomCNN ------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class CustomCNN(nn.Module):
    def __init__(self, args: dict, x_dim: int, y_dim: int):
        """
        A CNN that can be modified through a dictionary 'args'. If -use_pooling-, a MaxPooling2d layer is added to every
        Conv2d layer.

        :param args: dictionary which has to consist of the following entries. 'n_input_channels': number of input
        channels, 'n_output_channels': number of output channels, 'n_layers': number of layers in total, 'kernel_sizes':
        list with sizes of kernel for each layer, 'num_kernels': list with number of kernels in each layer, 'paddings':
        list with paddings for each layer, 'strides': list with strides for each layer, 'use_pooling': set to True if
        a MaxPool2d layer should follow each convolutional layer, 'pool_kernel_size': sizes of kernel for each pooling
        layer, 'pool_strides': strides for each pooling layer, 'pool_paddings': paddings for each pooling layer
        :param x_dim: size of x dimension of input
        :param y_dim: size of y dimension of input
        """
        super(CustomCNN, self).__init__()

        self.n_input_channels = args['n_input_channels']
        self.n_output_channels = args['n_output_channels']
        self.n_layers = args['n_layers']

        self.kernel_sizes = args['kernel_size']
        self.num_kernels = args['num_kernels']
        self.paddings = args['padding']
        self.strides = args['stride']

        self.use_pooling = args['use_pooling']
        if self.use_pooling:
            self.pool_kernel_sizes = args['pool_kernel_size']
            self.pool_strides = args['pool_stride']
            self.pool_paddings = args['pool_padding']

        self.x_dim_input = x_dim
        self.y_dim_input = y_dim

        # validation ---------------------------------------------------------------------------------------------------

        # check if the number of entries in every list of -args- corresponds to the defined number of layers
        if (len(self.kernel_sizes) < self.n_layers) or (len(self.num_kernels) < self.n_layers) or \
                (len(self.paddings) < self.n_layers) or (len(self.strides) < self.n_layers):

            raise AssertionError('Please define the CNN parameters (kernel_size, num_kernels, padding, stride) for '
                                 'each layer!')

        if (len(self.pool_kernel_sizes) < self.n_layers - 1) or (len(self.pool_paddings) < self.n_layers - 1) or \
                (len(self.pool_strides) < self.n_layers - 1):

            raise AssertionError('Please define the MaxPooling parameters (kernel_size, padding, stride) '
                                 'for each pooling layer!')

        # check if the output dimensions equal the input dimensions

        # [(W − K + 2P) / S] + 1
        for n in range(self.n_layers):
            x_dim = int(((x_dim - self.kernel_sizes[n] + 2 * self.paddings[n]) / self.strides[n]) + 1)
            y_dim = int(((y_dim - self.kernel_sizes[n] + 2 * self.paddings[n]) / self.strides[n]) + 1)

            if self.use_pooling and (n < self.n_layers - 1):
                x_dim = int(((x_dim - self.pool_kernel_sizes[n] + 2 * self.pool_paddings[n]) / self.pool_strides[n])
                            + 1)
                y_dim = int(((y_dim - self.pool_kernel_sizes[n] + 2 * self.pool_paddings[n]) / self.pool_strides[n])
                            + 1)

        if x_dim != self.x_dim_input or y_dim != self.y_dim_input:
            raise AssertionError(f'out_dim = {(x_dim, y_dim)}: Shape of output with given parameters doesn\'t match '
                                 f'shape of input!')

        # layer definition ---------------------------------------------------------------------------------------------

        # define input layer, input activation function and input pooling
        self.input_layer = nn.Conv2d(in_channels=self.n_input_channels,
                                     out_channels=self.num_kernels[0],
                                     kernel_size=self.kernel_sizes[0],
                                     padding=self.paddings[0],
                                     stride=self.strides[0])

        self.input_activation_function = nn.ReLU()

        self.input_pooling = nn.MaxPool2d(kernel_size=self.pool_kernel_sizes[0],
                                          padding=self.pool_paddings[0],
                                          stride=self.pool_strides[0])

        # define hidden layers
        layers = []
        n_input_channels = self.num_kernels[0]
        for n in range(1, self.n_layers - 1):
            # Add a CNN layer
            layer = nn.Conv2d(in_channels=n_input_channels,
                              out_channels=self.num_kernels[n],
                              kernel_size=self.kernel_sizes[n],
                              padding=self.paddings[n],
                              stride=self.strides[n])
            layers.append(layer)
            # Add relu activation module to list of modules
            layers.append(nn.ReLU())

            if self.use_pooling:
                layers.append(nn.MaxPool2d(kernel_size=self.pool_kernel_sizes[n],
                                           padding=self.pool_paddings[n],
                                           stride=self.pool_strides[n]))

            n_input_channels = self.num_kernels[n]

        self.hidden_layers = nn.Sequential(*layers)

        # define output layer
        self.output_layer = nn.Conv2d(in_channels=self.num_kernels[-2],
                                      out_channels=self.n_output_channels,
                                      kernel_size=self.kernel_sizes[-1],
                                      padding=self.paddings[-1],
                                      stride=self.strides[-1])

    def forward(self, x):

        x = self.input_layer(x)
        x = self.input_activation_function(x)
        x = self.input_pooling(x)

        x = self.hidden_layers(x)

        output = self.output_layer(x)

        return output

# ----------------------------------------------------------------------------------------------------------------------
# MyScndCNN ------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class MyScndCNN(nn.Module):

    def __init__(self):
        """
        A CNN bases on basic principles for CNN design:
        - Start with smaller kernels and use bigger kernels in the middle
        - Start with a smaller number of kernels and use more in the middle
        - Use common number of kernels, in ths case: 32-64-128
        - Use common combination of layers, in this case: Conv2d - Conv2d - MaxPool2d
        """
        super(MyScndCNN, self).__init__()

        # define input layer and input activation function
        self.input_layer = nn.Conv2d(in_channels=2,
                                     out_channels=32,
                                     kernel_size=3)
        self.input_activation = nn.ReLU()

        # define hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=1),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=5,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=1),
            nn.Conv2d(in_channels=64,
                      out_channels=32,
                      kernel_size=5,
                      padding=5),
            nn.ReLU()
        )

        # define output layer
        self.output_layer = nn.Conv2d(in_channels=32,
                                      out_channels=1,
                                      kernel_size=3,
                                      padding=3)

    def forward(self, x):
        x = self.input_activation(self.input_layer(x))
        x = self.hidden_layers(x)
        output = self.output_layer(x)

        return output


# ----------------------------------------------------------------------------------------------------------------------
# Transfer Learning ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


class AlexNet(nn.Module):

    def __init__(self):
        """
        Try on transfer learning.
        """
        super(AlexNet, self).__init__()

        # load pretrained AlexNet
        axn = models.alexnet(pretrained=True)

        # combine pretrained layers with custom layers
        self.input_layer = nn.Conv2d(2, 3, kernel_size=3)
        self.input_activation = nn.ReLU()

        self.features = axn.features

        self.classifier = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x, x_dim, y_dim):

        avg = nn.AdaptiveAvgPool2d(output_size=(x_dim, y_dim))

        x = self.input_layer(x)
        x = self.input_activation(x)
        x = self.features(x)

        x = avg(x)

        output = self.classifier(x)

        return output

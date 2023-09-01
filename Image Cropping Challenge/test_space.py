"""
Test space to experiment with CNN architectures.
"""

from architectures import MyCNN, MyScndCNN, SimpleCNN, AlexNet
import numpy as np
import torch

architecture = {'n_input_channels': 2, 'n_output_channels': 1, 'n_layers': 4, 'kernel_size': [15, 7, 7, 5],
                'num_kernels': [32, 32, 32, 1], 'padding': [7, 4, 3, 4], 'stride': [1, 1, 1, 1], 'use_pooling': True,
                'pool_kernel_size': [5, 3, 9], 'pool_padding': [1, 1, 2], 'pool_stride': [1, 1, 1]}

architecture_2 = {'n_input_channels': 2, 'n_output_channels': 1, 'n_layers': 5, 'kernel_size': [3, 5, 7, 5, 3],
                  'num_kernels': [5, 15, 45, 9], 'padding': [7, 4, 3, 4], 'stride': [1, 1, 1, 1], 'use_pooling': True,
                  'pool_kernel_size': [5, 3, 9], 'pool_padding': [1, 1, 2], 'pool_stride': [1, 1, 1]}

# my_cnn = MyCNN(args=architecture, x_dim=100, y_dim=100)
my_cnn = AlexNet()
my_cnn.to('cuda:0')

test_np = np.random.rand(50, 2, 99, 99)
test_ten = torch.tensor(test_np, dtype=torch.float32, device='cuda:0')

output = my_cnn(test_ten)
print(output.shape)



"""
[(W − K + 2P) / S] + 1
a....Conv2d
b....MaxPooling2d

input: 80

1a. ((80 - 15 + 14) / 1) + 1 = 80
1b. ((80 - 5 + 2) / 1) + 1 = 78

2a. ((78 - 7 + 8) / 1) + 1 = 80
2b. ((80 - 3 + 2) / 1) + 1 = 80

3a. ((80 - 7 + 6) / 1) + 1 = 80
3b. ((80 - 9 + 4) / 1) + 1 = 76

4. ((76 - 5 + 8) / 1) + 1 = 80

"""
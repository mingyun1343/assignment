"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn


class ConvNet(nn.Module):
    """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

    def __init__(self, n_channels, n_classes):
        """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),  # conv1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # maxpool1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # conv2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # maxpool2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # conv3_a
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # conv3_b
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # maxpool3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # conv4_a
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # conv4_b
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # maxpool4
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # conv5_a
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # conv5_b
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # maxpool5
        )
        self.classifier = nn.Sequential(
            nn.Linear(512,n_classes)
        )
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.features(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out


# if __name__ == '__main__':
#   random_data = np.random.rand(2, 3, 32, 32)  # 调整数据形状为 (batch_size, channels, height, width)
#   random_data_tensor = torch.from_numpy(random_data.astype(np.float32))  # 将NumPy数组转换为PyTorch的Tensor类型，并确保数据类型为float32
#   print("输入数据的数据维度", random_data_tensor.size())  # 检查数据形状是否正确
#
#   # 创建VGG16网络实例
#   model = ConvNet(3,10)
#   output = model(random_data_tensor)
#   print("输出数据维度", output.shape)
#   print(output)

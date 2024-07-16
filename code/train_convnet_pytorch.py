"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torchvision.transforms import transforms

from convnet_pytorch import ConvNet
import cifar10_utils

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10'

FLAGS = None


def accuracy(predictions, targets):
    """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    predicted_classes = torch.argmax(predictions, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    correct_predictions = torch.sum(predicted_classes == true_classes)
    accuracy = correct_predictions.float() / targets.size(0)
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练数据集
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR_DEFAULT, train=True,
                                            download=False, transform=transform)

    # 创建训练数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # 加载测试数据集
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR_DEFAULT, train=False,
                                           download=False, transform=transform)

    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = ConvNet(3,10)  # 替换为你的模型
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # 训练两个周期
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入；数据是一个[inputs, labels]列表
            inputs, labels = data

            # 清零参数梯度
            optimizer.zero_grad()
            # print(inputs.shape)
            # 正向传播，反向传播，优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个批次打印一次
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    # digit = trainloader.dataset.data[0]
    # plt.imshow(digit, cmap=plt.cm.binary)
    # plt.show()
    # print(classes[trainloader.dataset.targets[0]])
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
  Main function
  """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()

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
LOAD_MODE = True

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
    correct_predictions = torch.sum(predicted_classes == targets)
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
    transform = transforms.Compose([            # 图像变换
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 数据读入
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR_DEFAULT, train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR_DEFAULT, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE_DEFAULT, shuffle=False, num_workers=2)

    # cpu/gpu，获取模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet(3, 10).to(device)

    # 交叉熵损失函数，Adam优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_DEFAULT)

    # 损失和准确率
    losses = []
    accuracies = []

    if not LOAD_MODE:       # 是否训练
        # 已经迭代更新次数
        steps = 0
        running_loss = 0.0
        while steps < MAX_STEPS_DEFAULT:    # 达到设定的次数即截止

            for i, data in enumerate(trainloader, 0):
                steps = steps + 1
                if steps >= MAX_STEPS_DEFAULT:
                    break
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)             # 结果
                loss = criterion(outputs, labels)   # 损失
                loss.backward()                     # 反向传播
                optimizer.step()

                running_loss += loss.item()
                if (steps + 1) % EVAL_FREQ_DEFAULT == 0:    # 达到特定次数计算准确度并加入损失

                    model.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for data in testloader:
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                        accuracy = 100 * correct / total
                        accuracies.append(accuracy)         # 加入历史
                    model.train()

                    loss = running_loss / 500
                    running_loss = 0
                    losses.append(loss)             # 加入历史

        # 绘制精准度和损失曲线图
        line = np.arange(EVAL_FREQ_DEFAULT,MAX_STEPS_DEFAULT+EVAL_FREQ_DEFAULT,EVAL_FREQ_DEFAULT)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(line,losses, label='Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(line,accuracies, label='Accuracy')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # 保存模型
        torch.save(model.state_dict(), "../model/model.pth")

    # 加载模型
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("../model/model.pth"))
    else:
        model.load_state_dict(torch.load("../model/model.pth", map_location=torch.device('cpu')))
    # 计算最终模型的准确度
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("Accuracy=",accuracy)

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
                        help='Directory for storing input data'),
    parser.add_argument('--load_model', type=str, default=LOAD_MODE,
                        help='load model to predict')
    FLAGS, unparsed = parser.parse_known_args()

    main()

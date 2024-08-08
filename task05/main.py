# dip lab task 05

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 100

# 归一化
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081))]
)

# 获取 MNIST 训练集和数据集
train_dataset = datasets.MNIST(
    root="./data/mnist", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data/mnist", train=False, download=True, transform=transform
)

# 载入数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层 1，有 10 个输出通道
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # 卷积层 2，有 20 个输出通道
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # 全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10)
        )

    # 前向传播函数
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 把输入通过第一个卷积层
        x = self.conv2(x)  # 把第一层的输出通过第二个卷积层
        x = x.view(batch_size, -1)  # 展平特征图成一维向量
        x = self.fc(x)  # 把展平的向量通过全连接层
        return x


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum
)


def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:
            print("[%d, %5d]: loss: %.3f, acc: %.3f %%" % (
                epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print("[%d / %d]: Accuracy on test set: %.1f %% " %
          (epoch + 1, EPOCH, 100 * acc))
    return acc


if __name__ == "__main__":
    acc_list_test = []
    acc_flag = 0
    for epoch in range(EPOCH):
        train(epoch)
        acc_test = test()
        acc_list_test.append(acc_test)
        if acc_test > acc_flag:
            acc_flag = acc_test
            torch.save(model.state_dict(), "./model/handwritten_digits.pt")
    plt.plot(acc_list_test)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy on Testset")
    plt.grid()
    plt.show()

    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(
            train_dataset.train_data[i],
            cmap="gray", interpolation="none"
        )
        plt.title("Label: {}".format(train_dataset.train_labels[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()

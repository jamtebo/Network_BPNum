import torch
import torch.nn as nn
import torch.nn.functional as F

#日志记录频率
log_interval = 50

class netCNN(nn.Module):

    # 初始化
    def __init__(self):
        # 调用父类的构造函数
        super(netCNN, self).__init__()
        # 卷积层 输入通道：1  输出通道：10  卷积核：5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 卷积层 输入通道：10  输出通道：20  卷积核：5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 防止过拟合？？？？
        self.conv2_drop = nn.Dropout2d()
        # 全连接层 输入节点：320  输出节点：50
        self.fc1 = nn.Linear(320, 50)
        # 全连接层 输入节点：50  输出节点：10
        self.fc2 = nn.Linear(50, 10)

    # 前向传播
    def forward(self, x):
        # 卷积 池化 1@28x28 -> 10@24x24 -> 10@12x12
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 卷积 池化 10@12x12 -> 20@8x8 -> 20@4x4
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 1x320
        x = x.view(-1, 320)
        # 全连接 1x50
        x = F.relu(self.fc1(x))
        # 防止过拟合？？？？？
        x = F.dropout(x, training=self.training)
        # 全连接
        x = self.fc2(x)
        # 将张量的每个元素缩放到（0,1）区间且和为1
        return F.log_softmax(x)

    #训练
    def trainData(self, epoch, optimizer, train_loader, train_losses, train_counter):
        super().train()
        for batch_idx, (data, target) in enumerate(train_loader):
            #把梯度设置为0
            optimizer.zero_grad()
            #前向传播，求估算值
            output = self(data)
            #损失值
            loss = F.nll_loss(output, target)
            #反向传播求梯度
            loss.backward()
            # 更新所有参数
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            #保存内部状态到文件中
            torch.save(self.state_dict(), './Weight/model.pth')
            torch.save(optimizer.state_dict(), './Weight/optimizer.pth')

    #测试
    def test(self, test_loader, test_losses):
        super().eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    #测试图片
    def testImage(self, imageData):
        result = []
        with torch.no_grad():
            for data in imageData:
                output = self(data)
                result.append(output.data.max(1, keepdim=True)[1].item())
        return result

    #加载网络内部状态
    def LoadState(self, optimizer):
        network_state_dict = torch.load('./Weight/model.pth')
        self.load_state_dict(network_state_dict)
        optimizer_state_dict = torch.load('./Weight/optimizer.pth')
        optimizer.load_state_dict(optimizer_state_dict)
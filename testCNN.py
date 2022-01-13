import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from netCNN import*
import time

#训练的世代
n_epochs = 3
#训练集每batch加载的数据数
batch_size_train = 64
#训练集每batch加载的数据数
batch_size_test = 1000
#学习率
learning_rate = 0.01
#权重衰减
momentum = 0.5
#日志记录频率
log_interval = 10
#随机种子
random_seed = 1
#设置随机种子（软件运行时的随机数一致）
torch.manual_seed(random_seed)
transformData = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
#加载训练测试集
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,transform=transformData),
  batch_size=batch_size_train, shuffle=True)
#加载测试训练集
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,transform=transformData),
  batch_size=batch_size_test, shuffle=True)

#查看测试数据的结构
#examples = enumerate(test_loader)
#batch_idx, (example_data, example_targets) = next(examples)
#print(example_targets)
#print(example_data.shape)

#打印图片
#fig = plt.figure()
#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.tight_layout()
#    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#    plt.title("Ground Truth: {}".format(example_targets[i]))
#    plt.xticks([])
#    plt.yticks([])
#plt.show()

# 初始化网络和优化器
network = netCNN()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

#加载网络内部结构
network.LoadState(optimizer)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(1, n_epochs + 1)]

#训练及测试 跑过两次：90s和120s
#start = time.clock()
#for epoch in range(1, n_epochs + 1):    
  #network.trainData(epoch, optimizer, train_loader, train_losses, train_counter)
  #network.test(test_loader, test_losses)
#end = time.clock()
#UseTime = end - start
#print("总耗时为：%s" % UseTime)

#绘制训练曲线
#fig = plt.figure()
#plt.plot(train_counter, train_losses, color='blue')
#plt.scatter(test_counter, test_losses, color='red')
#plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
#plt.xlabel('number of training examples seen')
#plt.ylabel('negative log likelihood loss')
#plt.show()
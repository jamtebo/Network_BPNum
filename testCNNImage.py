from netCNN import*
import torch.optim as optim
from ImageDataSet import*

#学习率
learning_rate = 0.01
#权重衰减
momentum = 0.5

# 初始化网络和优化器
network = netCNN()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#加载网络内部结构
network.LoadState(optimizer)

#加载测试训练集
test_loader = torch.utils.data.DataLoader(
  imageDataSet('./image/'),
  batch_size=10, shuffle=True)
test_losses = []

examples = enumerate(test_loader)
batch_idx, (example_data) = next(examples)

reslut = network.testImage(test_loader)
print(reslut)
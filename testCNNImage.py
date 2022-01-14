from netCNN import*
import torch.optim as optim
from ImageDataSet import*
import matplotlib.pyplot as plt

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
batch_size_tesst = 10
test_loader = torch.utils.data.DataLoader(
  imageDataSet('./image/'),
  batch_size=batch_size_tesst, shuffle=True)
test_losses = []

examples = enumerate(test_loader)
batch_idx, (example_data) = next(examples)

reslut = network.testImage(test_loader)
print(reslut)

test_examples = enumerate(test_loader)
batch_idx, (example_data) = next(test_examples)
for i in range(batch_size_tesst):
  plt.subplot(5,5,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("计算值: {}".format(reslut[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()
import imageio
import matplotlib.pyplot as plt

from neuralNetwork import*
from ImageOperate import*

img_data = GetImageData(28, 28, "image/2.png")

#初始化神经网络并从文件中加载权重
n = neuralNetwork(784, 100, 10, 0.1)
n.LoadWeight()
#显示图片
plt.imshow(img_data, cmap='Greys', interpolation='None')
plt.show()
outputs = n.query(img_data.reshape(784))
label = numpy.argmax(outputs)
print("输入图片数字为：", label)
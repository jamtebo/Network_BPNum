import numpy
import scipy.special
import matplotlib.pyplot as plt
import csv
#%matplotlib inline

from neuralNetwork import*

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#打开训练集文件
training_data_file = open("mnidata/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#训练
print("开始训练" )
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        # 调整输入
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 期望输出
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
print("结束训练")
n.SaveWeight()

#测试
test_data_file = open("mnidata/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
#测试结果
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print("正确值：", correct_label)
    input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(input)
    label = numpy.argmax(outputs)
    print("计算值：", label)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asfarray(scorecard)
print("正确率：", scorecard_array.sum() / scorecard_array.size * 100, "%")

#显示图片
#image_array = numpy.asfarray(all_values [1:]).reshape((28,28))
#plt.imshow(image_array, cmap='Greys', interpolation='None')
#plt.title('%s' % all_values[0])
#plt.show()




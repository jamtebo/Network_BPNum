import numpy
import scipy.special

class neuralNetwork:

    #初始化函数：设定输入层节点、隐藏层节点和输出层节点的数量
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #输入层、隐藏层、输出层的节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 权重：输入-隐藏[100, 784]   隐藏-输出[10, 100]
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        #学习率
        self.lr = learningrate
        #激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    #训练：学习给定训练集样本，优化权重
    def train(self, inputs_list, targets_list):
        #目标（标准答案）
        targets = numpy.array(targets_list, ndmin=2).T
        # 输入节点
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 隐藏层 [100, 784] . [784, 1] = [100, 1]
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层 [10, 100] . [100, 1] = [10, 1]
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #调整权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))


    #查询：给定输入，从输出节点给出答案
    def query(self, inputs_list):
        #输入节点
        inputs = numpy.array(inputs_list, ndmin=2).T
        #隐藏层 [100, 784] . [784, 1] = [100, 1]
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层 [10, 100] . [100, 1] = [10, 1]
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    #将权重记录到文件中
    def SaveWeight(self):
        numpy.save('Weight\InputHide', self.wih)
        numpy.save('Weight\HideOutput', self.who)
        pass

    #将权重从文件中读取出来
    def LoadWeight(self):
        self.wih = numpy.load('Weight\InputHide.npy')
        self.who = numpy.load('Weight\HideOutput.npy')
        pass
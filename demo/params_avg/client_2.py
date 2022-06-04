import utils
import conn
from torchvision import transforms as tf
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
import torch.optim as optim


batch_size = 50
data_size = 10000
batch_num = data_size / batch_size
transform = tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, transform=transform, download=True)
train_dataset_2 = Subset(train_dataset, range(data_size, data_size*2))
train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class NeuralNetwork(torch.nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=(5, 5)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2)
        )
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Linear(320, 160),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(160, 80),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(80, 40),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(40, 20),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(20, 10)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.view(x.size(0), -1)
        x = self.layer_3(x)
        return x


server_host = '192.168.1.104'
server_port = 666
epoch = 100
client_2 = conn.Client(server_host, server_port)  
model_2 = NeuralNetwork()
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model_params(model):

    '''获取模型参数
    :param model: 用于完成图像识别的模型，当前使用CNN
    :return: 返回一个list， 其中包含模型当前的参数，类型为torch.Tensor
    '''

    params = []

    for param in model.parameters():
        params.append(param.data.clone())

    return params


def get_model_grads(model):

    '''获取模型参数梯度

    :param model: 用于完成图像识别的模型，当前使用CNN

    :return: 返回一个list， 其中包含模型参数的梯度，类型为torch.Tensor

    '''

    grads = []

    for param in model.parameters():
        grads.append(param.grad.data.clone())

    return grads


def train(model, optimizer, train_loader, pattern='model', batch_index=None):

    '''模型训练
    :param model: 待训练的客户端模型
    :param optimizer: 对应于客户端模型的优化器
    :param train_loader:
            当采用模型平均时，train_loader代表客户端数据集的迭代器
            当采用梯度平均时，train_loader代表客户端数据集的batch列表
    :param pattern: 模型训练采用的模式，有‘model‘(模型平均)和'grad'(梯度平均)两种模式，默认为模型平均
    :param batch_index: 只有当采用梯度平均时，该变量才会有效，用于指示当前用于训练的batch
    :return:
            当采用模型平均时，返回模型参数
            当采用梯度平均时，返回模型参数梯度
    '''

    model.to(device)

    if pattern == 'model':

        for data in train_loader:

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return get_model_params(model)

    else:

        inputs, labels = train_loader[batch_index]
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        return get_model_grads(model)


def test(model, test_loader, participant, epoch):

    '''模型测试
    :param model: 待测试的客户端模型
    :param test_loader: 全部客户端统一的测试集
    :param participant: 待测试的客户端编号
    :param epoch: 当前迭代的轮数
    :return: 无返回值，函数内部完成测试结果(正确率)的输出
    '''
    correct = 0
    total = 0

    for data in test_loader:

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, prediction = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    print('[Epoch%d Participant%2d]Accuracy Rate on Test_Dataset: %.2f%%' % (epoch+1, participant, 100*correct/total))


for i in range(epoch):

    params = train(model_2, optimizer_2, train_loader_2, pattern='model')
    test(model_2, test_loader, 2, i)
    
    client_2.send(params)
    
  
    avg_params = client_2.recv()
   
    utils.update_model_params(model_2, avg_params, [])

import time
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np


from preprocess import preprocess
from preprocess import cross_validation


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 16, bias=False)
        self.L1 = nn.Linear(16, 10, bias=False)
        self.L2 = nn.Linear(10, 5, bias=False)
        self.output_layer = nn.Linear(5, output_size, bias=False)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.output_layer(x)
        return x


start_time = time.time()
# epoch, device = 5000, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch = 10000

input_matrix, target_matrix, features_counts = preprocess()
ratio = 10
input_arrays, target_arrays = cross_validation(ratio, input_matrix, target_matrix)
criterion = nn.CrossEntropyLoss()


def train(epoch, input_model, input_training, input_target):
    for idx in range(epoch):
        output_data = input_model(input_training)
        input_target = input_target.long()
        output_data = output_data.float()
        loss = criterion(output_data, input_target)
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            print('Train Epoch: {}/{}\tLoss: {:.6f}'.format(idx, epoch, loss.data[0]))
    return loss


def test(text, input_model, input_train, input_target):
    correct = 0
    for data, target in zip(input_train, input_target):
        output_data = input_model(data)
        index = output_data.max(0)[1]
        if index == target:
            correct += 1
    percentage = round(100 * (correct / len(input_target)))
    print('{} Percentage:{}%({}/{})'.format(text, percentage, correct, len(input_target)))
    return percentage


def init_weights(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


train_value, test_value = [], []
for idx, (x_test, y_test) in enumerate(zip(input_arrays, target_arrays)):
    model = NeuralNet(features_counts, 4)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    x_train = torch.Tensor([])
    y_train = torch.LongTensor([])
    for train_idx, (x, y) in enumerate(zip(input_arrays, target_arrays)):
        if train_idx != idx:
            x_train = torch.cat((x_train, x), 0)
            y_train = torch.cat((y_train, y), 0)
    loss = train(epoch, model, x_train, y_train)
    print('R{} Train Loss: {}'.format(idx+1, loss.data[0]))
    train_value.append(test('R{} Train'.format(idx+1), model, x_train, y_train))
    test_value.append(test('R{} Test'.format(idx+1), model, x_test, y_test))
    print()


print("time: {}".format(time.time()-start_time))
overall_train = round(sum(train_value) / ratio)
overall_test = round(sum(test_value) / ratio)
print("overall train's percentage: {}%, test's percentage: {}%".format(overall_train, overall_test))

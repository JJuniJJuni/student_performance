import time
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np


from preprocess import preprocess
from preprocess import cross_validation


start_time = time.time()
epoch, device = 5000, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_mat, target_mat, features_counts = preprocess()
input_matrix = torch.LongTensor(input_mat, device=device)
target_matrix = torch.LongTensor(target_mat, device=device)


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 80)
        self.L1 = nn.Linear(80, 50)
        self.L2 = nn.Linear(50, 30)
        # self.L4 = nn.Linear(60, 50)
        # self.L5 = nn.Linear(50, 40)
        # self.L6 = nn.Linear(40, 30)
        # self.L7 = nn.Linear(30, 20)
        # self.L8 = nn.Linear(20, 10)
        self.output_layer = nn.Linear(30, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        # x = F.relu(self.L3(x))
        # x = F.dropout(self.L4(x))
        # x = F.relu(self.L5(x))
        # x = F.relu(self.L6(x))
        # x = F.relu(self.L7(x))
        # x = F.relu(self.L8(x))
        # x = self.output_layer(x)
        return x


ratio = 10
input_arrays, target_arrays = cross_validation(ratio, input_matrix, target_matrix)
criterion = nn.CrossEntropyLoss()


def train(epoch, input_model, input_training, input_target):
    for idx in range(epoch):
        output_data = input_model(input_training.type(torch.float))
        loss = criterion(output_data, input_target)
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            print('Train Epoch: {}/{}\tLoss: {:.6f}'.format(idx, epoch, loss.data[0]))
    return loss.data[0]


def test(text, input_model, input_train, input_target):
    correct = 0
    for data, target in zip(input_train, input_target):
        output_data = input_model(data.type(torch.float))
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
    # print('R{} Model Structure'.format(idx+1))
    model = NeuralNet(features_counts, 4)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    x_train = torch.LongTensor([], device=device)
    y_train = torch.LongTensor([], device=device)
    for train_idx, (x, y) in enumerate(zip(input_arrays, target_arrays)):
        if train_idx != idx:
            x_train = torch.cat((x_train, x), 0)
            y_train = torch.cat((y_train, y), 0)
    loss = train(epoch, model, x_train, y_train)
    print('R{} Train Loss: {}'.format(idx+1, loss))
    train_value.append(test('R{} Train'.format(idx+1), model, x_train, y_train))
    test_value.append(test('R{} Test'.format(idx+1), model, x_test, y_test))
    print()
print("time: {}".format(time.time()-start_time))
overall_train = round(sum(train_value) / ratio)
overall_test = round(sum(test_value) / ratio)
print("overall train's percentage: {}%, test's percentage: {}%".format(overall_train, overall_test))

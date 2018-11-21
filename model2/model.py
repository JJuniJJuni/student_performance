import time
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np


from preprocess import preprocess
from preprocess import cross_validation


start_time = time.time()
epoch, device = 100000, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_mat, target_mat, features_counts = preprocess('./data/student-por.csv')
input_por, target_por, _ = preprocess('./data/student-mat.csv')
input_matrix = torch.tensor(np.concatenate((input_mat, input_por), 0), device=device,
                            dtype=torch.float, requires_grad=False)
target_matrix = torch.tensor(np.concatenate((target_mat, target_por), 0), device=device,
                             dtype=torch.float, requires_grad=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 16)
        self.L1 = nn.Linear(16, 12)
        self.L2 = nn.Linear(12, 8)
        self.L3 = nn.Linear(8, 4)
        self.output_layer = nn.Linear(4, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = torch.sigmoid(self.L1(x))
        x = torch.sigmoid(self.L2(x))
        x = torch.sigmoid(self.L3(x))
        x = self.output_layer(x)
        return x


input_arrays, target_arrays = cross_validation(10, input_matrix, target_matrix)
criterion = nn.MSELoss()


def train(epoch, input_model, input_training, input_target):
    for idx in range(epoch):
        output_data = input_model(input_training.to(device=torch.device('cpu')))
        loss = criterion(torch.squeeze(output_data), input_target.to(device=torch.device('cpu')))
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
        output_data = float(input_model(data.to(device=torch.device('cpu'))))
        if round(output_data) == target:
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
    model = NeuralNet(features_counts, 1)
    model.apply(init_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    x_train = torch.tensor([], device=device, dtype=torch.float, requires_grad=False)
    y_train = torch.tensor([], device=device, dtype=torch.float, requires_grad=False)
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
overall_train = round(sum(train_value) / 10)
overall_test = round(sum(test_value) / 10)
print("overall train's percentage: {}%, test's percentage: {}%".format(overall_train, overall_test))



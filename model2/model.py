import time
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np


from preprocess import preprocess
from preprocess import split_data
from preprocess import cross_validation


start_time = time.time()
epoch, device = 5000, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_mat, target_mat, features_counts = preprocess('./data/student-por.csv')
input_por, target_por, _ = preprocess('./data/student-mat.csv')
input_matrix = torch.tensor(np.concatenate((input_mat, input_por), 0), device=device,
                            dtype=torch.float, requires_grad=False)
target_matrix = torch.tensor(np.concatenate((target_mat, target_por), 0), device=device,
                             dtype=torch.float, requires_grad=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 22)
        self.L1 = nn.Linear(22, 16)
        self.L2 = nn.Linear(16, 10)
        # self.L3 = nn.Linear(20, 16)
        self.output_layer = nn.Linear(10, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.input_layer(x))
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        # x = F.relu(self.L3(x))
        x = self.output_layer(x)
        return x


# x_train, x_test, y_train, y_test = split_data(input_matrix, target_matrix)
# x_train = torch.tensor(x_train, device=device, dtype=torch.float, requires_grad=False)
# x_test = torch.tensor(x_test, device=device, dtype=torch.float, requires_grad=False)
# y_train = torch.tensor(y_train, device=device, dtype=torch.float, requires_grad=False)
# y_test = torch.tensor(y_test, device=device, dtype=torch.float, requires_grad=False)
input_arrays, target_arrays = cross_validation(10, input_matrix, target_matrix)
criterion = nn.MSELoss()


def train(epoch, input_model, input_training, input_target):
    for idx in range(epoch):
        # x_train.to(device), y_train.to(device)
        output_data = input_model(input_training.to(device=torch.device('cpu')))
        loss = criterion(torch.squeeze(output_data), input_target.to(device=torch.device('cpu')))
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            print('Train Epoch: {}/{}\tLoss: {:.6f}'.format(idx, epoch, loss.data[0]))


def test(text, input_model, input_train, input_target):
    correct = 0
    for data, target in zip(input_train, input_target):
        output_data = float(input_model(data.to(device=torch.device('cpu'))))
        # print(output_data, target)
        if round(output_data) == target:
            correct += 1
    print('{} Percentage:'.format(text), 100 * (correct / len(input_target)))


for idx, (x_test, y_test) in enumerate(zip(input_arrays, target_arrays)):
    print('R{} Model Structure'.format(idx+1))
    model = NeuralNet(features_counts, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    x_train = torch.tensor([], device=device, dtype=torch.float, requires_grad=False)
    # print(x_train)
    y_train = torch.tensor([], device=device, dtype=torch.float, requires_grad=False)
    for train_idx, (x, y) in enumerate(zip(input_arrays, target_arrays)):
        if train_idx != idx:
            x_train = torch.cat((x_train, x), 0)
            y_train = torch.cat((y_train, y), 0)
    train(epoch, model, x_train, y_train)
    test('R{} Train'.format(idx+1), model, x_train, y_train)
    test('R{} Test'.format(idx+1), model, x_test, y_test)
print("time: {}".format(time.time()-start_time))
# model.eval()
# print('[Test]')


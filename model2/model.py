import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from preprocess import preprocess
from preprocess import split_data


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


input_matrix, target_matrix = preprocess('./data/student-por.csv')
epoch, device = 12000, torch.device('cpu')
x_train, x_test, y_train, y_test = split_data(input_matrix, target_matrix)
x_train = torch.tensor(x_train, device=device, dtype=torch.float, requires_grad=False)
x_test = torch.tensor(x_test, device=device, dtype=torch.float, requires_grad=False)
y_train = torch.tensor(y_train, device=device, dtype=torch.float, requires_grad=False)
y_test = torch.tensor(y_test, device=device, dtype=torch.float, requires_grad=False)
model = NeuralNet(32, 1)
print('[Model Structure]')
print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for idx in range(epoch):
    # x_train.to(device), y_train.to(device)
    output_data = model(x_train.to(device=torch.device('cpu')))
    loss = criterion(torch.squeeze(output_data), y_train.to(device=torch.device('cpu')))
    model.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if idx % 50 == 0:
        print('Train Epoch: {}/{}\tLoss: {:.6f}'.format(
            idx, epoch, loss.data[0]))
train_correct = 0


for data, target in zip(x_train, y_train):
    output_data = float(model(data.to(device=torch.device('cpu'))))
    print(output_data, target)
    if round(output_data) == target:
        train_correct += 1
print('Train Percentage:', 100*(train_correct/len(y_train)))

# model.eval()
test_correct = 0

# print('[Test]')
for data, target in zip(x_test, y_test):
    output_data = float(model(data.to(device=torch.device('cpu'))))
    # print(output_data, target, bool(output_data == target))
    if round(output_data) == target:
        test_correct += 1
print('Test Percentage:', 100*(test_correct/len(y_test)))


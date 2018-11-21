import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from preprocess import preprocess
from preprocess import split_data


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 28)
        self.L1 = nn.Linear(28, 24)
        self.L2 = nn.Linear(24, 20)
        self.L3 = nn.Linear(20, 16)
        self.output_layer = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = self.output_layer(x)
        return x


input_matrix, target_matrix = preprocess('data/student-por.csv')
x_train, x_test, y_train, y_test = split_data(input_matrix, target_matrix)
x_train = torch.tensor(x_train, device=torch.device('cpu'), dtype=torch.float, requires_grad=False)
x_test = torch.tensor(x_test, device=torch.device('cpu'), dtype=torch.float, requires_grad=False)
y_train = torch.tensor(y_train, device=torch.device('cpu'), dtype=torch.float, requires_grad=False)
y_test = torch.tensor(y_test, device=torch.device('cpu'), dtype=torch.float, requires_grad=False)
model = NeuralNet(32, 1)
print('[Model Structure]')
print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for _ in range(5000):
    # x_train.to(device), y_train.to(device)
    output_data = model(x_train)
    loss = criterion(torch.squeeze(output_data), y_train)
    model.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
test_loss = 0
correct = 0

print('[Test]')
for data, target in zip(x_test, y_test):
    output_data = int(model(data))
    # print(output_data)
    # value = int(torch.argmax(output_data))
    print(output_data, target)
    if output_data == target:
        correct += 1
print(100*(correct/len(y_test)))
print('[Prediction]')


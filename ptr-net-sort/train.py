import numpy as np
import torch
import torch.utils.data as Data
from model import PointerNetwork

EPOCH = 100
BATCH_SIZE = 2
DATA_SIZE = 100
INPUT_SIZE = 1
HIDDEN_SIZE = 512
WEIGHT_SIZE = 256
LR = 0.001


def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def getdata(experiment=1, data_size=None):
    if experiment == 1:
        high = 10
        senlen = 5
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(data_size)])
        y = np.argsort(x)
        y = y[:, :-2]           ## 앞 3개를 정답으로
    elif experiment == 2:
        high = 100
        senlen = 10
        x = np.array([np.random.choice(range(high), senlen, replace=False)
                      for _ in range(data_size)])
        y = np.argsort(x)
    elif experiment == 3:
        senlen = 5
        x = np.array([np.random.random(senlen) for _ in range(data_size)])
        y = np.argsort(x)
    elif experiment == 4:
        senlen = 10
        x = np.array([np.random.random(senlen) for _ in range(data_size)])
        y = np.argsort(x)
    return x, y


def evaluate(model, X, Y):
    probs = model(X)
    prob, indices = torch.max(probs, 2)
    equal_cnt = sum([1 if torch.equal(index.detach(), y.detach()) else 0 for index, y in zip(indices, Y)])
    accuracy = equal_cnt / len(X)
    print('Acc: {:.2f}%'.format(accuracy * 100))


# Get Dataset
x, y = getdata(experiment=1, data_size=DATA_SIZE)
x = to_cuda(torch.FloatTensor(x).unsqueeze(2))
y = to_cuda(torch.LongTensor(y))
# Split Dataset
train_size = (int)(DATA_SIZE * 0.9)
train_X = x[:train_size]
train_Y = y[:train_size]
test_X = x[train_size:]
test_Y = y[train_size:]
# Build DataLoader
train_data = Data.TensorDataset(train_X, train_Y)
data_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Define the Model
model = PointerNetwork(INPUT_SIZE, HIDDEN_SIZE, WEIGHT_SIZE, is_GRU=False)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = torch.nn.CrossEntropyLoss()

# Training...
print('Training... ')
for epoch in range(EPOCH):
    for (batch_x, batch_y) in data_loader:
        probs = model(batch_x)
        outputs = probs.view(-1, batch_x.shape[1])
        batch_y = batch_y.view(-1)
        loss = loss_fun(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
        evaluate(model, train_X[:2], train_Y[:2])
# Test...
print('Test...')
evaluate(model, test_X, test_Y)



# Training...
# Epoch: 0, Loss: 0.99542
# Acc: 0.00%
# Epoch: 2, Loss: 0.84342
# Acc: 0.00%
# Epoch: 4, Loss: 0.90028
# Acc: 0.00%
# Epoch: 6, Loss: 0.95734
# Acc: 0.00%
# Epoch: 8, Loss: 0.74193
# Acc: 0.00%
# Epoch: 10, Loss: 0.85922
# Acc: 0.00%
# Epoch: 12, Loss: 0.72589
# Acc: 0.00%
# Epoch: 14, Loss: 0.88290
# Acc: 0.00%
# Epoch: 16, Loss: 0.74075
# Acc: 0.00%
# Epoch: 18, Loss: 0.75814
# Acc: 0.00%
# Epoch: 20, Loss: 0.80860
# Acc: 0.00%
# Epoch: 22, Loss: 0.74228
# Acc: 0.00%
# Epoch: 24, Loss: 0.70597
# Acc: 0.00%
# Epoch: 26, Loss: 0.67966
# Acc: 0.00%
# Epoch: 28, Loss: 0.80013
# Acc: 0.00%
# Epoch: 30, Loss: 0.70278
# Acc: 0.00%
# Epoch: 32, Loss: 0.73939
# Acc: 0.00%
# Epoch: 34, Loss: 0.74251
# Acc: 0.00%
# Epoch: 36, Loss: 0.62284
# Acc: 0.00%
# Epoch: 38, Loss: 0.66134
# Acc: 0.00%
# Epoch: 40, Loss: 0.66684
# Acc: 0.00%
# Epoch: 42, Loss: 0.62876
# Acc: 0.00%
# Epoch: 44, Loss: 0.68352
# Acc: 0.00%
# Epoch: 46, Loss: 0.69297
# Acc: 0.00%
# Epoch: 48, Loss: 0.68334
# Acc: 0.00%
# Epoch: 50, Loss: 0.62110
# Acc: 0.00%
# Epoch: 52, Loss: 0.66988
# Acc: 0.00%
# Epoch: 54, Loss: 0.59932
# Acc: 0.00%
# Epoch: 56, Loss: 0.62294
# Acc: 0.00%
# Epoch: 58, Loss: 0.60647
# Acc: 0.00%
# Epoch: 60, Loss: 0.75567
# Acc: 0.00%
# Epoch: 62, Loss: 0.60727
# Acc: 0.00%
# Epoch: 64, Loss: 0.63494
# Acc: 0.00%
# Epoch: 66, Loss: 0.60802
# Acc: 0.00%
# Epoch: 68, Loss: 0.62197
# Acc: 0.00%
# Epoch: 70, Loss: 0.60145
# Acc: 0.00%
# Epoch: 72, Loss: 0.60833
# Acc: 0.00%
# Epoch: 74, Loss: 0.60431
# Acc: 0.00%
# Epoch: 76, Loss: 0.60465
# Acc: 0.00%
# Epoch: 78, Loss: 0.47476
# Acc: 100.00%
# Epoch: 80, Loss: 0.35298
# Acc: 50.00%
# Epoch: 82, Loss: 0.08765
# Acc: 100.00%
# Epoch: 84, Loss: 0.01742
# Acc: 100.00%
# Epoch: 86, Loss: 0.01733
# Acc: 100.00%
# Epoch: 88, Loss: 0.03870
# Acc: 100.00%
# Epoch: 90, Loss: 0.00666
# Acc: 100.00%
# Epoch: 92, Loss: 0.00534
# Acc: 100.00%
# Epoch: 94, Loss: 0.01457
# Acc: 100.00%
# Epoch: 96, Loss: 0.00773
# Acc: 100.00%
# Epoch: 98, Loss: 0.00479
# Acc: 100.00%
# Test...
# Acc: 100.00%
#
# Process finished with exit code 0

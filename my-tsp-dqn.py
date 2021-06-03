import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random

learning_rate = 0.0005

gamma = 0.98
buffer_limit = 50000
batch_size = 32

num_episodes = 2000

time = np.array([[0, 6, 8, 7],
                 [3, 0, 4, 2],
                 [4, 6, 0, 3],
                 [4, 2, 9, 0]])


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()



def main():
    q = Qnet()

    for i in range(num_episodes):
        done = False
        total_reward = 0
        state = 0  # Initial state
        seq = [state]
        togo = [1, 2, 3]

        e = max(0.01, 0.20 - 0.01 * (i / 100))  # Linear annealing from 20% to 1%

        print(e)

        while not done:
            a = q.sample_action(torch.from_numpy(s))


if __name__ == '__main__':
    main()

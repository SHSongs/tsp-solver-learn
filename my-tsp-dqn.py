import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random

learning_rate = 0.005

gamma = 0.98
buffer_limit = 50000
batch_size = 32

num_episodes = 2000

time = np.array([[100, 6, 8, 7],
                 [3, 100, 4, 2],
                 [4, 6, 100, 3],
                 [4, 2, 9, 100]])


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
            # print('ramdom')
            return random.randint(0, 1), out
        else:
            return out.argmin().item(), out


def preprocessing(seq, state):
    seq_in = np.identity(4)[len(seq) - 1]  # [1 0 0 0] -> [0 0 0 1]
    state_in = np.identity(4)[state]
    return np.concatenate((seq_in, state_in))


def main():
    q = Qnet()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    print('시작')
    for i in range(num_episodes):
        done = False
        total_reward = 0
        state = 0  # Initial state
        seq = [state]
        togo = [1, 2, 3]

        cnt = 0
        e = max(0.01, 0.20 - 0.01 * (i / 100))  # Linear annealing from 20% to 1%

        while not done:
            cnt += 1
            s_in = torch.from_numpy(preprocessing(seq, state)).float()
            a, out = q.sample_action(s_in, e)

            if a in togo:
                seq.append(a)
                togo.remove(a)
                r = time[state][a]
                total_reward += r

                next_s_in = torch.from_numpy(preprocessing(seq, a)).float()

                next_out = q(next_s_in)

                if len(seq) < 4:
                    next_out_pos = next_out.take(torch.tensor(togo))
                else:
                    next_out_pos = next_out.take(torch.tensor([0]))

                next_q = torch.amin(next_out_pos)
                target = r + gamma * next_q
                # out[a] = target

                loss = F.smooth_l1_loss(out[a], target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = a


            elif a == 0 or a == state:  # 제자리 이동 또는 0으로의 이동 방지
                target = 20.0
                out[a] = target

                loss = F.smooth_l1_loss(out[a], torch.tensor(target).float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if len(seq) == 4:
                done = True

        s_in = preprocessing(seq, state)
        s_in = torch.from_numpy(s_in).float()
        out = q(s_in)

        a = 0
        r = time[state][a]
        total_reward += r
        target = r

        seq.append(a)

        loss = F.smooth_l1_loss(out[a], torch.tensor(target).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(cnt, total_reward)
        print(seq)


if __name__ == '__main__':
    main()

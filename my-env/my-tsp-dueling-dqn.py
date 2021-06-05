import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random

learning_rate = 0.005

gamma = 0.90
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
        self.fc1 = nn.Linear(8, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 4)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        value = self.relu(self.fc_value(x))
        adv = self.relu(self.fc_adv(x))

        value = self.value(value)
        adv = self.adv(adv)

        avdAverage = torch.mean(adv, dim=0, keepdim=True)
        Q = value + adv - avdAverage

        return Q

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return np.random.randint(0, 4), out
        else:
            return out.argmin().item(), out


def preprocessing(seq, state):
    seq_in = np.identity(4)[len(seq) - 1]  # [1 0 0 0] -> [0 0 0 1]
    state_in = np.identity(4)[state]
    return np.concatenate((seq_in, state_in))


def main():
    q = Qnet()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    criterion = F.smooth_l1_loss

    print('시작')
    for i in range(num_episodes):
        done = False
        total_reward = 0
        state = 0  # Initial state
        seq = [state]
        togo = [1, 2, 3]

        cnt = 0
        e = 0.85 if i < 0.98 * num_episodes else 0.01

        while not done:
            cnt += 1
            s_in = torch.from_numpy(preprocessing(seq, state)).float()
            a, out = q.sample_action(s_in, e)
            # print(a, out)
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

                next_min_q = torch.amin(next_out_pos)
                target = r + gamma * next_min_q
                # out[a] = target

                loss = criterion(out[a], target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(' ', len(seq) - 1, q(s_in))

                state = a
            elif a == 0 or a == state:  # 제자리 이동 또는 0으로의 이동 방지
                target = torch.tensor(20.0).float()

                loss = criterion(out[a], target)
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
        target = r
        total_reward += r

        loss = criterion(out[a], torch.tensor(target).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        seq.append(a)
        print(' ', len(seq) - 1, q(s_in))

        print(i, seq, total_reward, 'cnt', cnt)



if __name__ == '__main__':
    main()

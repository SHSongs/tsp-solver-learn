import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import random

from torch.distributions import Categorical
from collections import namedtuple

learning_rate = 0.005

gamma = 0.98

num_episodes = 500

time = np.array([[100, 6, 8, 7],
                 [3, 100, 4, 2],
                 [4, 6, 100, 3],
                 [4, 2, 9, 100]])

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

eps = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.relu = nn.ReLU()

        # actor's layer
        self.action_head = nn.Linear(128, 4)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.relu(self.fc1(x))

        # actor : choices action to take from state s_t
        # by returning probability of each action
        action_prob = self.action_head(x)
        action_prob = F.softmax(action_prob, dim=-1)

        # critic : evaluates being in the state
        state_values = self.value_head(x)

        return action_prob, state_values

    def sample_action(self, obs, epsilon):
        probs, state_value = self.forward(obs)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item(), m, action, state_value


def preprocessing(seq, state):
    seq_in = np.identity(4)[len(seq) - 1]  # [1 0 0 0] -> [0 0 0 1]
    state_in = np.identity(4)[state]
    return np.concatenate((seq_in, state_in))


def finish_episode(model, optimizer):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        R = R.float()
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            a, m, action, state_value = model.sample_action(s_in, e)

            r = time[state][a]
            model.rewards.append(r * -1)

            # print(a, out)
            if a in togo:
                seq.append(a)
                togo.remove(a)
                r = time[state][a]
                total_reward += r

                next_s_in = torch.from_numpy(preprocessing(seq, a)).float()

                next_out = model(next_s_in)

                state = a

            elif a == 0 or a == state:  # 제자리 이동 또는 0으로의 이동 방지
                target = torch.tensor(20.0).float()

            if len(seq) == 4:
                done = True
            print(' ', len(seq) - 1, model(s_in))

        s_in = preprocessing(seq, state)
        s_in = torch.from_numpy(s_in).float()

        a = 0
        r = time[state][a]
        target = r
        total_reward += r

        seq.append(a)

        print(' ', len(seq) - 1, model(s_in))

        print(i, seq, total_reward, 'cnt', cnt)

        finish_episode(model, optimizer)


if __name__ == '__main__':
    main()



## Result
# 시작
#   1 (tensor([0.2679, 0.2173, 0.2443, 0.2706], grad_fn=<SoftmaxBackward>), tensor([0.1545], grad_fn=<AddBackward0>))
#   1 (tensor([0.2410, 0.2418, 0.2317, 0.2854], grad_fn=<SoftmaxBackward>), tensor([-0.1473], grad_fn=<AddBackward0>))
#   1 (tensor([0.2410, 0.2418, 0.2317, 0.2854], grad_fn=<SoftmaxBackward>), tensor([-0.1473], grad_fn=<AddBackward0>))
#   2 (tensor([0.2410, 0.2418, 0.2317, 0.2854], grad_fn=<SoftmaxBackward>), tensor([-0.1473], grad_fn=<AddBackward0>))
#   2 (tensor([0.3136, 0.1962, 0.2122, 0.2780], grad_fn=<SoftmaxBackward>), tensor([0.0928], grad_fn=<AddBackward0>))
#   3 (tensor([0.3136, 0.1962, 0.2122, 0.2780], grad_fn=<SoftmaxBackward>), tensor([0.0928], grad_fn=<AddBackward0>))
#   4 (tensor([0.1948, 0.2373, 0.2351, 0.3328], grad_fn=<SoftmaxBackward>), tensor([0.0613], grad_fn=<AddBackward0>))
# 0 [0, 3, 2, 1, 0] 25 cnt 6
#   1 (tensor([0.2823, 0.2383, 0.2515, 0.2279], grad_fn=<SoftmaxBackward>), tensor([0.0898], grad_fn=<AddBackward0>))
#   1 (tensor([0.2115, 0.2549, 0.2800, 0.2535], grad_fn=<SoftmaxBackward>), tensor([-0.0718], grad_fn=<AddBackward0>))
#   1 (tensor([0.2115, 0.2549, 0.2800, 0.2535], grad_fn=<SoftmaxBackward>), tensor([-0.0718], grad_fn=<AddBackward0>))
#   1 (tensor([0.2115, 0.2549, 0.2800, 0.2535], grad_fn=<SoftmaxBackward>), tensor([-0.0718], grad_fn=<AddBackward0>))
#   2 (tensor([0.2115, 0.2549, 0.2800, 0.2535], grad_fn=<SoftmaxBackward>), tensor([-0.0718], grad_fn=<AddBackward0>))
#   2 (tensor([0.3259, 0.2177, 0.2056, 0.2508], grad_fn=<SoftmaxBackward>), tensor([0.1878], grad_fn=<AddBackward0>))
#   3 (tensor([0.3259, 0.2177, 0.2056, 0.2508], grad_fn=<SoftmaxBackward>), tensor([0.1878], grad_fn=<AddBackward0>))
#   4 (tensor([0.2067, 0.2560, 0.2298, 0.3075], grad_fn=<SoftmaxBackward>), tensor([-0.0796], grad_fn=<AddBackward0>))
# 1 [0, 1, 2, 3, 0] 17 cnt 7
#   1 (tensor([0.3044, 0.2182, 0.2661, 0.2113], grad_fn=<SoftmaxBackward>), tensor([0.0333], grad_fn=<AddBackward0>))
#   2 (tensor([0.2457, 0.2256, 0.2588, 0.2699], grad_fn=<SoftmaxBackward>), tensor([0.0740], grad_fn=<AddBackward0>))
#   2 (tensor([0.2836, 0.2052, 0.2320, 0.2792], grad_fn=<SoftmaxBackward>), tensor([0.0051], grad_fn=<AddBackward0>))
#   2 (tensor([0.2836, 0.2052, 0.2320, 0.2792], grad_fn=<SoftmaxBackward>), tensor([0.0051], grad_fn=<AddBackward0>))
#   3 (tensor([0.2836, 0.2052, 0.2320, 0.2792], grad_fn=<SoftmaxBackward>), tensor([0.0051], grad_fn=<AddBackward0>))
#   4 (tensor([0.2016, 0.2407, 0.2552, 0.3026], grad_fn=<SoftmaxBackward>), tensor([0.0440], grad_fn=<AddBackward0>))
# 2 [0, 2, 3, 1, 0] 16 cnt 5
#   0 (tensor([0.3280, 0.2128, 0.2610, 0.1981], grad_fn=<SoftmaxBackward>), tensor([-0.0551], grad_fn=<AddBackward0>))
#   1 (tensor([0.3280, 0.2128, 0.2610, 0.1981], grad_fn=<SoftmaxBackward>), tensor([-0.0551], grad_fn=<AddBackward0>))
#   2 (tensor([0.2164, 0.2287, 0.3113, 0.2436], grad_fn=<SoftmaxBackward>), tensor([-0.1264], grad_fn=<AddBackward0>))
#   2 (tensor([0.3613, 0.2132, 0.1941, 0.2315], grad_fn=<SoftmaxBackward>), tensor([0.3285], grad_fn=<AddBackward0>))
#   3 (tensor([0.3613, 0.2132, 0.1941, 0.2315], grad_fn=<SoftmaxBackward>), tensor([0.3285], grad_fn=<AddBackward0>))
#   4 (tensor([0.2102, 0.2529, 0.2419, 0.2950], grad_fn=<SoftmaxBackward>), tensor([-0.1012], grad_fn=<AddBackward0>))
# 3 [0, 1, 2, 3, 0] 17 cnt 5
#   1 (tensor([0.3186, 0.2178, 0.2691, 0.1945], grad_fn=<SoftmaxBackward>), tensor([-0.1355], grad_fn=<AddBackward0>))
#   1 (tensor([0.2109, 0.2590, 0.2740, 0.2561], grad_fn=<SoftmaxBackward>), tensor([-0.2082], grad_fn=<AddBackward0>))
#   1 (tensor([0.2109, 0.2590, 0.2740, 0.2561], grad_fn=<SoftmaxBackward>), tensor([-0.2082], grad_fn=<AddBackward0>))
#   1 (tensor([0.2109, 0.2590, 0.2740, 0.2561], grad_fn=<SoftmaxBackward>), tensor([-0.2082], grad_fn=<AddBackward0>))
#   1 (tensor([0.2109, 0.2590, 0.2740, 0.2561], grad_fn=<SoftmaxBackward>), tensor([-0.2082], grad_fn=<AddBackward0>))
#   2 (tensor([0.2109, 0.2590, 0.2740, 0.2561], grad_fn=<SoftmaxBackward>), tensor([-0.2082], grad_fn=<AddBackward0>))
#   3 (tensor([0.3612, 0.2189, 0.1922, 0.2277], grad_fn=<SoftmaxBackward>), tensor([0.3723], grad_fn=<AddBackward0>))
#   4 (tensor([0.2036, 0.2407, 0.2680, 0.2877], grad_fn=<SoftmaxBackward>), tensor([0.0121], grad_fn=<AddBackward0>))
# 4 [0, 3, 2, 1, 0] 25 cnt 7
#   1 (tensor([0.3158, 0.2227, 0.2791, 0.1824], grad_fn=<SoftmaxBackward>), tensor([-0.2132], grad_fn=<AddBackward0>))
#   1 (tensor([0.2533, 0.2268, 0.2851, 0.2349], grad_fn=<SoftmaxBackward>), tensor([0.0588], grad_fn=<AddBackward0>))
#   1 (tensor([0.2533, 0.2268, 0.2851, 0.2349], grad_fn=<SoftmaxBackward>), tensor([0.0588], grad_fn=<AddBackward0>))
#   2 (tensor([0.2533, 0.2268, 0.2851, 0.2349], grad_fn=<SoftmaxBackward>), tensor([0.0588], grad_fn=<AddBackward0>))
#   2 (tensor([0.2697, 0.2315, 0.2723, 0.2265], grad_fn=<SoftmaxBackward>), tensor([0.0710], grad_fn=<AddBackward0>))
#   3 (tensor([0.2697, 0.2315, 0.2723, 0.2265], grad_fn=<SoftmaxBackward>), tensor([0.0710], grad_fn=<AddBackward0>))
#   4 (tensor([0.2050, 0.2654, 0.2546, 0.2751], grad_fn=<SoftmaxBackward>), tensor([-0.0936], grad_fn=<AddBackward0>))
# 5 [0, 2, 1, 3, 0] 20 cnt 6
#   1 (tensor([0.3128, 0.2313, 0.2780, 0.1779], grad_fn=<SoftmaxBackward>), tensor([-0.2951], grad_fn=<AddBackward0>))
#   1 (tensor([0.2491, 0.2342, 0.2855, 0.2312], grad_fn=<SoftmaxBackward>), tensor([0.0385], grad_fn=<AddBackward0>))
#   1 (tensor([0.2491, 0.2342, 0.2855, 0.2312], grad_fn=<SoftmaxBackward>), tensor([0.0385], grad_fn=<AddBackward0>))
#   1 (tensor([0.2491, 0.2342, 0.2855, 0.2312], grad_fn=<SoftmaxBackward>), tensor([0.0385], grad_fn=<AddBackward0>))
#   2 (tensor([0.2491, 0.2342, 0.2855, 0.2312], grad_fn=<SoftmaxBackward>), tensor([0.0385], grad_fn=<AddBackward0>))
#   2 (tensor([0.2759, 0.2490, 0.2289, 0.2462], grad_fn=<SoftmaxBackward>), tensor([0.0995], grad_fn=<AddBackward0>))
#   2 (tensor([0.2759, 0.2490, 0.2289, 0.2462], grad_fn=<SoftmaxBackward>), tensor([0.0995], grad_fn=<AddBackward0>))
#   2 (tensor([0.2759, 0.2490, 0.2289, 0.2462], grad_fn=<SoftmaxBackward>), tensor([0.0995], grad_fn=<AddBackward0>))
#   3 (tensor([0.2759, 0.2490, 0.2289, 0.2462], grad_fn=<SoftmaxBackward>), tensor([0.0995], grad_fn=<AddBackward0>))
#   4 (tensor([0.2006, 0.2454, 0.2799, 0.2741], grad_fn=<SoftmaxBackward>), tensor([0.0181], grad_fn=<AddBackward0>))
# 6 [0, 2, 3, 1, 0] 16 cnt 9
# ....
#   1 (tensor([0.0029, 0.0629, 0.4286, 0.5057], grad_fn=<SoftmaxBackward>), tensor([-1.1133], grad_fn=<AddBackward0>))
#   2 (tensor([3.4220e-04, 5.2498e-03, 1.0372e-03, 9.9337e-01],
#        grad_fn=<SoftmaxBackward>), tensor([0.3026], grad_fn=<AddBackward0>))
#   3 (tensor([3.7402e-04, 9.9531e-01, 3.4406e-03, 8.7177e-04],
#        grad_fn=<SoftmaxBackward>), tensor([0.8113], grad_fn=<AddBackward0>))
#   4 (tensor([0.0032, 0.0048, 0.9008, 0.0911], grad_fn=<SoftmaxBackward>), tensor([0.4369], grad_fn=<AddBackward0>))
# 490 [0, 2, 3, 1, 0] 16 cnt 3
#   1 (tensor([0.0028, 0.0633, 0.4362, 0.4977], grad_fn=<SoftmaxBackward>), tensor([-1.1247], grad_fn=<AddBackward0>))
#   2 (tensor([3.3991e-04, 5.2584e-03, 1.0541e-03, 9.9335e-01],
#        grad_fn=<SoftmaxBackward>), tensor([0.2946], grad_fn=<AddBackward0>))
#   3 (tensor([3.6623e-04, 9.9529e-01, 3.4739e-03, 8.6664e-04],
#        grad_fn=<SoftmaxBackward>), tensor([0.8032], grad_fn=<AddBackward0>))
#   4 (tensor([0.0031, 0.0047, 0.9037, 0.0886], grad_fn=<SoftmaxBackward>), tensor([0.4255], grad_fn=<AddBackward0>))
# 491 [0, 2, 3, 1, 0] 16 cnt 3
#   1 (tensor([0.0028, 0.0636, 0.4433, 0.4903], grad_fn=<SoftmaxBackward>), tensor([-1.1347], grad_fn=<AddBackward0>))
#   2 (tensor([3.3793e-04, 5.2670e-03, 1.0699e-03, 9.9333e-01],
#        grad_fn=<SoftmaxBackward>), tensor([0.2877], grad_fn=<AddBackward0>))
#   3 (tensor([3.5939e-04, 9.9527e-01, 3.5054e-03, 8.6208e-04],
#        grad_fn=<SoftmaxBackward>), tensor([0.7963], grad_fn=<AddBackward0>))
#   4 (tensor([0.0029, 0.0045, 0.9063, 0.0863], grad_fn=<SoftmaxBackward>), tensor([0.4154], grad_fn=<AddBackward0>))
# 492 [0, 2, 3, 1, 0] 16 cnt 3
#   1 (tensor([0.0027, 0.0639, 0.4502, 0.4832], grad_fn=<SoftmaxBackward>), tensor([-1.1430], grad_fn=<AddBackward0>))
#   2 (tensor([2.7655e-04, 9.9730e-01, 1.3556e-03, 1.0675e-03],
#        grad_fn=<SoftmaxBackward>), tensor([0.3277], grad_fn=<AddBackward0>))
#   3 (tensor([0.0015, 0.0013, 0.9785, 0.0187], grad_fn=<SoftmaxBackward>), tensor([0.7509], grad_fn=<AddBackward0>))
#   4 (tensor([0.0014, 0.0352, 0.0047, 0.9587], grad_fn=<SoftmaxBackward>), tensor([0.4657], grad_fn=<AddBackward0>))
# 493 [0, 3, 1, 2, 0] 17 cnt 3
#   1 (tensor([0.0027, 0.0641, 0.4562, 0.4770], grad_fn=<SoftmaxBackward>), tensor([-1.1498], grad_fn=<AddBackward0>))
#   2 (tensor([2.7407e-04, 9.9730e-01, 1.3660e-03, 1.0643e-03],
#        grad_fn=<SoftmaxBackward>), tensor([0.3221], grad_fn=<AddBackward0>))
#   3 (tensor([0.0014, 0.0013, 0.9792, 0.0182], grad_fn=<SoftmaxBackward>), tensor([0.7428], grad_fn=<AddBackward0>))
#   4 (tensor([0.0014, 0.0352, 0.0047, 0.9586], grad_fn=<SoftmaxBackward>), tensor([0.4627], grad_fn=<AddBackward0>))
# 494 [0, 3, 1, 2, 0] 17 cnt 3
#   1 (tensor([0.0027, 0.0643, 0.4612, 0.4719], grad_fn=<SoftmaxBackward>), tensor([-1.1550], grad_fn=<AddBackward0>))
#   2 (tensor([0.0014, 0.0019, 0.9132, 0.0835], grad_fn=<SoftmaxBackward>), tensor([-0.0042], grad_fn=<AddBackward0>))
#   3 (tensor([0.0024, 0.0510, 0.0055, 0.9411], grad_fn=<SoftmaxBackward>), tensor([0.8157], grad_fn=<AddBackward0>))
#   4 (tensor([5.9408e-04, 9.9266e-01, 4.5739e-03, 2.1723e-03],
#        grad_fn=<SoftmaxBackward>), tensor([0.4376], grad_fn=<AddBackward0>))
# 495 [0, 1, 2, 3, 0] 17 cnt 3
#   1 (tensor([0.0027, 0.0655, 0.4655, 0.4663], grad_fn=<SoftmaxBackward>), tensor([-1.1551], grad_fn=<AddBackward0>))
#   2 (tensor([3.3218e-04, 5.3100e-03, 1.1193e-03, 9.9324e-01],
#        grad_fn=<SoftmaxBackward>), tensor([0.2769], grad_fn=<AddBackward0>))
#   3 (tensor([3.3875e-04, 9.9521e-01, 3.6033e-03, 8.4715e-04],
#        grad_fn=<SoftmaxBackward>), tensor([0.7827], grad_fn=<AddBackward0>))
#   4 (tensor([0.0024, 0.0041, 0.9140, 0.0795], grad_fn=<SoftmaxBackward>), tensor([0.3954], grad_fn=<AddBackward0>))
# 496 [0, 2, 3, 1, 0] 16 cnt 3
#   1 (tensor([0.0027, 0.0666, 0.4702, 0.4605], grad_fn=<SoftmaxBackward>), tensor([-1.1538], grad_fn=<AddBackward0>))
#   2 (tensor([3.3159e-04, 5.3341e-03, 1.1296e-03, 9.9320e-01],
#        grad_fn=<SoftmaxBackward>), tensor([0.2807], grad_fn=<AddBackward0>))
#   3 (tensor([3.3472e-04, 9.9520e-01, 3.6210e-03, 8.4312e-04],
#        grad_fn=<SoftmaxBackward>), tensor([0.7832], grad_fn=<AddBackward0>))
#   4 (tensor([0.0023, 0.0041, 0.9155, 0.0781], grad_fn=<SoftmaxBackward>), tensor([0.3969], grad_fn=<AddBackward0>))
# 497 [0, 2, 3, 1, 0] 16 cnt 3
#   1 (tensor([0.0027, 0.0675, 0.4753, 0.4545], grad_fn=<SoftmaxBackward>), tensor([-1.1512], grad_fn=<AddBackward0>))
#   2 (tensor([2.6528e-04, 9.9730e-01, 1.3888e-03, 1.0461e-03],
#        grad_fn=<SoftmaxBackward>), tensor([0.3153], grad_fn=<AddBackward0>))
#   3 (tensor([0.0012, 0.0011, 0.9810, 0.0166], grad_fn=<SoftmaxBackward>), tensor([0.7420], grad_fn=<AddBackward0>))
#   4 (tensor([0.0014, 0.0355, 0.0048, 0.9582], grad_fn=<SoftmaxBackward>), tensor([0.4711], grad_fn=<AddBackward0>))
# 498 [0, 3, 1, 2, 0] 17 cnt 3
#   1 (tensor([0.0027, 0.0683, 0.4794, 0.4496], grad_fn=<SoftmaxBackward>), tensor([-1.1479], grad_fn=<AddBackward0>))
#   2 (tensor([3.3076e-04, 5.3769e-03, 1.1485e-03, 9.9314e-01],
#        grad_fn=<SoftmaxBackward>), tensor([0.2903], grad_fn=<AddBackward0>))
#   3 (tensor([3.2811e-04, 9.9518e-01, 3.6573e-03, 8.3653e-04],
#        grad_fn=<SoftmaxBackward>), tensor([0.7871], grad_fn=<AddBackward0>))
#   4 (tensor([0.0022, 0.0040, 0.9182, 0.0756], grad_fn=<SoftmaxBackward>), tensor([0.4023], grad_fn=<AddBackward0>))
# 499 [0, 2, 3, 1, 0] 16 cnt 3
import or_gym
from solver import solver_RNN
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

episode = 10000
seq_len = 20

env_config = {'N': seq_len}
env = or_gym.make('TSP-v1', env_config=env_config)

embedding_size = 128
hidden_size = 128

model = solver_RNN(
    embedding_size,
    hidden_size,
    seq_len,
    2, 10)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

losses = []
episodes_length = []


def visualization(coords, tour_indices, title='None'):
    plt.close('all')

    num_plots = 3
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]  # 2dim -> 1dim

    for i, ax in enumerate(axes):
        # idx 의 좌표 가져오기
        idx = tour_indices[i].unsqueeze(0)
        idx = idx.expand(2, -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)
        data = coords[i].transpose(1, 0)
        data = data.gather(1, idx).cpu().numpy()

        # draw graph
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        # limit 설정
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.title(title)
    plt.show()


data_for_visual_coords = []
data_for_visual_actions = []


for i in range(episode):
    s = env.reset()

    print('-------------------------new game--------------------')
    print('coord')
    print(env.coords)

    coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0)

    rewards, log_probs, actions, value = model(coords)

    if i % 10 == 9:
        data_for_visual_coords.append(coords.squeeze(0))
        data_for_visual_actions.append(actions.squeeze(0))
    if i % 100 == 99:
        c = torch.stack(data_for_visual_coords)
        a = torch.stack(data_for_visual_actions)
        visualization(c, a, str(i))
        data_for_visual_coords.clear()
        data_for_visual_actions.clear()

    actions = actions.squeeze(0).tolist()

    start_idx = actions.index(s[0])
    a_1 = actions[start_idx + 1:]
    a_2 = actions[0:start_idx + 1]
    actions = a_1 + a_2

    print('first state', s)

    total_reward = 0
    cnt = 0
    done = False
    while not done:
        a = actions[cnt]
        next_state, reward, done, _ = env.step(a)
        total_reward += reward
        print('current node', env.current_node)
        print(next_state, reward, done)

        cnt += 1

    # to home
    total_reward += env.distance_matrix[actions[-2], actions[-1]]

    episodes_length.append(total_reward)
    print('total length', total_reward)

    # network train
    optimizer.zero_grad()

    advantage = (total_reward - value)
    actor_loss = -log_probs * advantage
    critic_loss = F.smooth_l1_loss(value.squeeze(0), torch.FloatTensor([total_reward]))

    loss = actor_loss.sum() + critic_loss

    print('loss : ', loss.item())
    losses.append(loss.item())

    loss.backward()
    optimizer.step()

plt.plot(range(len(losses)), losses, color="blue")
plt.title("A2C Loss")
plt.xlabel("episode")
plt.ylabel("loss")

plt.show()

plt.plot(range(len(episodes_length)), episodes_length, color="blue")
plt.title("Episodes length")
plt.xlabel("episode")
plt.ylabel("length")

plt.show()

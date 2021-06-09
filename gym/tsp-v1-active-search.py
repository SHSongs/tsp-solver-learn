import or_gym
from solver import solver_RNN
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from util import visualization

episode = 10000
seq_len = 7

env_config = {'N': seq_len}
env = or_gym.make('TSP-v1', env_config=env_config)

embedding_size = 128
hidden_size = 128
beta = 0.99
grad_clip = 1.5


model = solver_RNN(
    embedding_size,
    hidden_size,
    seq_len,
    2, 10)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

losses = []
episodes_length = []

data_for_visual_coords = []
data_for_visual_actions = []

moving_avg = torch.zeros(1)
first_step = True

for i in range(episode):
    s = env.reset()

    # print('-------------------------new game--------------------')
    # print('coord')
    # print(env.coords)

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
        # print('current node', env.current_node)
        # print(next_state, reward, done)

        cnt += 1

    # to home
    total_reward += env.distance_matrix[actions[-2], actions[-1]]

    if first_step:  # generating first baseline
        moving_avg = total_reward
        first_step = False
        continue

    episodes_length.append(total_reward)
    print('total length', total_reward)

    # network train

    moving_avg = moving_avg * beta + total_reward * (1.0 - beta)
    advantage = total_reward - moving_avg
    log_probs = torch.sum(log_probs, dim=-1)
    log_probs[log_probs < -100] = -100
    loss = (advantage * log_probs).mean()

    print('loss : ', loss.item())
    losses.append(loss.item())
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

plt.plot(range(len(losses)), losses, color="blue")
plt.title("Active search Loss")
plt.xlabel("episode")
plt.ylabel("loss")

plt.show()

plt.plot(range(len(episodes_length)), episodes_length, color="blue")
plt.title("Episodes length")
plt.xlabel("episode")
plt.ylabel("length")

plt.show()

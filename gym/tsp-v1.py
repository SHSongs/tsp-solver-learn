import or_gym
from solver import solver_RNN
import torch

seq_len = 4

env_config = {'N': seq_len}
env = or_gym.make('TSP-v1', env_config=env_config)

embedding_size = 128
hidden_size = 128

model = solver_RNN(
    embedding_size,
    hidden_size,
    seq_len,
    2, 10)

s = env.reset()

print('coord')
print(env.coords)

coords = torch.FloatTensor(env.coords).transpose(1, 0).unsqueeze(0)

rewards, log_probs, action = model(coords)

action = action.squeeze(0).tolist()

start_idx = action.index(s[0])
a_1 = action[start_idx + 1:]
a_2 = action[0:start_idx + 1]
action = a_1 + a_2

print('first state', s)

total_reward = 0
cnt = 0
done = False
while not done:
    a = action[cnt]
    next_state, reward, done, _ = env.step(a)
    total_reward += reward
    print('current node', env.current_node)
    print(next_state, reward, done)

    cnt += 1

print('total length', total_reward)

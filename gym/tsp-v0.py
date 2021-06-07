import elkai
import or_gym
import numpy as np
# env = or_gym.make('TSP-v0')
import matplotlib

matplotlib.use('Qt5Agg')

env_config = {'N': 50}

env = or_gym.make('TSP-v0', env_config=env_config)

# env.plot_network()

done = False
s = env.reset()

while s[0] != 0:
    s = env.reset()

print(env.node_dict)
matrix = env.adjacency_matrix
for i in range(len(matrix)):
    for j in range(len(matrix)):
        if i != j and matrix[i][j] == 0:
            matrix[i][j] = 999

print(matrix)

result = elkai.solve_float_matrix(matrix)
start_idx = result.index(s[0])
r_1 = result[start_idx + 1:]
r_2 = result[0:start_idx + 1]
result = r_1 + r_2

print(result)
print('first state', s)

cnt = 0
while not done:
    a = result[cnt]
    next_state, reward, done, _ = env.step(a)
    print('current node', env.current_node)

    print(next_state, reward, done)

    cnt += 1

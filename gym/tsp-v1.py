import elkai
import or_gym
import numpy as np
# env = or_gym.make('TSP-v0')
import matplotlib

env_config = {'N': 4}

env = or_gym.make('TSP-v1', env_config=env_config)

done = False
s = env.reset()

while s[0] != 0:
    s = env.reset()

print('coord')
print(env.coords)
matrix = env.distance_matrix

print('distance matrix')
print(matrix)

result = elkai.solve_float_matrix(matrix)
result = result[1:]

print(result)

print('-----------------------\n\n')

print('first state', s)

cnt = 0
while not done:
    a = result[cnt]
    next_state, reward, done, _ = env.step(a)
    print('current node', env.current_node)

    print(next_state, reward, done)

    cnt += 1

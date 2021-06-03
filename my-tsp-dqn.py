import numpy as np

num_episodes = 2000
time = np.array([[0, 6, 8, 7],
                 [3, 0, 4, 2],
                 [4, 6, 0, 3],
                 [4, 2, 9, 0]])

for i in range(num_episodes):
    done = False
    total_reward = 0
    state = 0  # Initial state
    seq = [state]
    togo = [1, 2, 3]

    e = max(0.01, 0.20 - 0.01 * (i / 100))  # Linear annealing from 20% to 1%

    print(e)


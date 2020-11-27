
import numpy as np
from MazeEnv import MazeEnv

env = MazeEnv(world_size=np.array([50,10]), \
              gold_pos=np.array([[1,2],[5,6]]), \
              bad_pos=np.array([[3,4],[7,8]]), \
              max_ite=10)

env.reset()
for i in range(20):
    next_state, reward, done = env.step(env.feasible_actions()[1])
    print(env.decode_s(next_state), next_state, reward, done)
    if done:
        break


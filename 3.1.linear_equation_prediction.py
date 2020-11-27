
import numpy as np
from MazeEnv import *

world_size=np.array([5,5])
env = MazeEnv(world_size=world_size, \
              gold_pos=np.array([[1,2]]), \
              bad_pos=np.array([[3,4]]), \
              max_ite=100)
env.reset()
# 四个动作各1/4
ACTION_PROB = 0.25
DISCOUNT = 0.9

A1 = -1 * np.eye(world_size[0] * world_size[1])
A2 = np.zeros((world_size[0] * world_size[1], world_size[0] * world_size[1]))
b = np.zeros(world_size[0] * world_size[1])

for s in env.feasible_states:
    for a in env.feasible_actions:
        s_, r, d = env.step_state(s, a)
        A2[s, s_] += ACTION_PROB * DISCOUNT
        b[s] += ACTION_PROB * r

V = np.linalg.solve(A1+A2, -b).reshape(world_size)
print(V)


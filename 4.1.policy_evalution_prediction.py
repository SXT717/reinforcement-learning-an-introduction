
import numpy as np
from MazeEnv import *
import copy

world_size=np.array([5,5])
env = MazeEnv(world_size=world_size, \
              gold_pos=np.array([[1,2]]), \
              bad_pos=np.array([[3,4]]), \
              max_ite=10)
env.reset()
# 四个动作各1/4
ACTION_PROB = 0.25
DISCOUNT = 0.9
V = np.zeros((world_size[0]*world_size[1]))
V_old = copy.deepcopy(V)

while True:
    V_old = copy.deepcopy(V)
    for s in env.feasible_states:
        values = []
        for a in env.feasible_actions:
            s_, r, d = env.step_state(s, a)
            values.append(ACTION_PROB*(r+DISCOUNT*V[s_]))
        V[s] = np.sum(np.array(values))
    if np.max(np.abs(V - V_old)) < 0.0001:
        break

print('env.ite = ', env.ite)
print(V.reshape(world_size))

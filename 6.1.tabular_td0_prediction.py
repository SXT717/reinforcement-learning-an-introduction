
import numpy as np
from MazeEnv import *
import copy

world_size=np.array([5,5])
env = MazeEnv(world_size=world_size, \
              gold_pos=np.array([[1,2]]), \
              bad_pos=np.array([[3,4]]), \
              max_ite=20)
env.reset()
DISCOUNT = 0.9
alpha = 0.01
V = np.zeros((world_size[0]*world_size[1]))
V_old = copy.deepcopy(V)

ite = 0
while True:  # episode循环
    ite += 1
    V_old = copy.deepcopy(V)
    s = env.reset()
    while True:  # step循环
        #随机策略
        a = env.random_action()
        s_, r, d = env.step(a)
        V[s] += alpha * (r + DISCOUNT*V[s_] - V[s])
        s = s_
        if d:
            break
    print(np.max(np.abs(V - V_old)))
    if np.max(np.abs(V - V_old)) < 0.01 and ite >= 1000:
        break

print(V.reshape(world_size))





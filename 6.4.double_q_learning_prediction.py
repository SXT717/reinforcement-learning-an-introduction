
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
epsilon = 0.1
alpha = 0.01
Q = np.zeros((world_size[0] * world_size[1], 4))
Q_old = copy.deepcopy(Q)
Q1 = np.zeros((world_size[0] * world_size[1], 4))
Q1_old = copy.deepcopy(Q1)
Q2 = np.zeros((world_size[0] * world_size[1], 4))
Q2_old = copy.deepcopy(Q2)
V = np.zeros((world_size[0]*world_size[1]))
V_old = copy.deepcopy(V)

ite = 0
while True:  # episode循环
    ite += 1
    Q_old = copy.deepcopy(Q)
    s = env.reset()
    while True:  # step循环
        a = np.argmax(Q1[s, :]+Q2[s, :]) if (np.random.rand() > epsilon) else env.random_action()
        s_, r, d = env.step(a)
        if np.random.rand() >= 0.5:
            Q1[s,a] += alpha * (r + DISCOUNT*Q2[s_, np.argmax(Q1[s_, :])] - Q1[s, a])
        else:
            Q2[s,a] += alpha * (r + DISCOUNT*Q1[s_, np.argmax(Q2[s_, :])] - Q2[s, a])
        s = s_
        if d:
            break
    Q = (Q1 + Q2)/2.0
    print(np.max(np.abs(Q - Q_old)))
    if np.max(np.abs(Q - Q_old)) < 0.0001 and ite >= 10000:
        break

print(ite)
print(Q)

for i in range(world_size[0]*world_size[1]):
    V[i] = np.max(Q[i,:])

print(V.reshape(world_size))





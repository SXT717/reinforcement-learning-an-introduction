
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
V = np.zeros((world_size[0]*world_size[1]))
V_old = copy.deepcopy(V)

ite = 0
while True:  # episode循环
    ite += 1
    Q_old = copy.deepcopy(Q)
    s = env.reset()
    a = np.argmax(Q[s, :]) if (np.random.rand() > epsilon) else env.random_action()
    while True:  # step循环
        s_, r, d = env.step(a)
        a_ = np.argmax(Q[s_, :]) if (np.random.rand() > epsilon) else env.random_action()
        Q[s,a] += alpha * (r + DISCOUNT*Q[s_, a_] - Q[s, a])
        s = s_
        a = a_
        if d:
            break
    print(np.max(np.abs(Q - Q_old)))
    if np.max(np.abs(Q - Q_old)) < 0.0001 and ite >= 10000:
        break

print(ite)
print(Q)

for i in range(world_size[0]*world_size[1]):
    V[i] = np.max(Q[i,:])

print(V.reshape(world_size))





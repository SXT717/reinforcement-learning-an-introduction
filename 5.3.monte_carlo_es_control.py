import numpy as np
from MazeEnv import *
import copy
from collections import defaultdict

world_size = np.array([5, 5])
env = MazeEnv(world_size=world_size, \
              gold_pos=np.array([[1, 2]]), \
              bad_pos=np.array([[3, 4]]), \
              max_ite=50)
env.reset()
DISCOUNT = 0.9
Q = np.zeros((world_size[0] * world_size[1], 4))
Q_old = copy.deepcopy(Q)
PI = np.zeros((world_size[0] * world_size[1]), dtype=np.int)  # 0,1,2,3分别代表4个动作
#
Return = defaultdict(list)

ite = 0
while True:  # episode循环
    ite += 1
    Q_old = copy.deepcopy(Q)
    for s0 in env.feasible_states:
        for a0 in env.feasible_actions:
            s_a_his = []
            r_his = []
            s = env.reset(s0)
            s_, r, d = env.step(a0)
            s_a_his.append(env.encode_s_a(s0, a0))
            r_his.append(r)
            s = s_
            while True:  # step循环
                a = PI[s]
                s_, r, d = env.step(a)
                s_a_his.append(env.encode_s_a(s, a))
                r_his.append(r)
                s = s_
                if d:
                    break
            G = 0
            for i in range(len(r_his) - 1, -1, -1):
                G = r_his[i] + DISCOUNT * G
                s_a = s_a_his[i]
                if s_a not in s_a_his[:i]:
                    Return[s_a].append(G)
                    s, a = env.decode_s_a(s_a)
                    Q[s, a] = np.mean(np.array(Return[s_a]))
                    PI[s] = np.argmax(Q[s, :])
    print(np.max(np.abs(Q - Q_old)))
    if np.max(np.abs(Q - Q_old)) < 0.01 and ite >= 100:
        break

print(ite)
print(Q)
print(PI.reshape(world_size))
PI_FIG = []
for i in range(world_size[0]):
    PI_FIG_SUB = []
    for j in range(world_size[1]):
        PI_FIG_SUB.append(env.actions_figs[PI[i * world_size[0] + j]])
    PI_FIG.append(PI_FIG_SUB)
    print(PI_FIG_SUB)

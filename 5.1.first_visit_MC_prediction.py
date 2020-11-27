
import numpy as np
from MazeEnv import *
import copy
from collections import defaultdict

world_size=np.array([5,5])
env = MazeEnv(world_size=world_size, \
              gold_pos=np.array([[1,2]]), \
              bad_pos=np.array([[3,4]]), \
              max_ite=20)
env.reset()
DISCOUNT = 0.9
V = np.zeros((world_size[0]*world_size[1]))
V_old = copy.deepcopy(V)
PI = np.zeros((world_size[0]*world_size[1]), dtype=np.int)#0,1,2,3分别代表4个动作
#
Return = defaultdict(list)

ite = 0
while True:#episode循环
    ite += 1
    V_old = copy.deepcopy(V)
    s = env.reset()
    s_his = []
    r_his = []
    while True:#step循环
        #随机策略
        a = env.random_action()
        s_, r, d = env.step(a)
        s_his.append(s)
        r_his.append(r)
        s = s_
        if d:
            break
    G = 0
    for i in range(len(r_his)-1, -1, -1):
        G = r_his[i] + DISCOUNT * G
        s = s_his[i]
        if s not in s_his[:i]:
            Return[s].append(G)
            V[s] = np.mean(np.array(Return[s]))
    if np.max(np.abs(V-V_old)) < 0.0001 and ite >= 100:
        break

def policy_improvement():
    for s in env.feasible_states:
        values = []
        for a in env.feasible_actions:
            s_, r, d = env.step_state(s, a)
            values.append(r + DISCOUNT * V[s_])
        PI[s] = np.array(values).argmax()

policy_improvement()

print(ite)
print(len(Return))
#print(Return)
print(V.reshape(world_size))
print(PI.reshape(world_size))
PI_FIG = []
for i in range(world_size[0]):
    PI_FIG_SUB = []
    for j in range(world_size[1]):
        PI_FIG_SUB.append(env.actions_figs[PI[i*world_size[0]+j]])
    PI_FIG.append(PI_FIG_SUB)
    print(PI_FIG_SUB)


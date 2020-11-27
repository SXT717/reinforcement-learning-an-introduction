
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
V = np.zeros((world_size[0]*world_size[1]))
Q = np.zeros((world_size[0] * world_size[1], 4))
Q_old = copy.deepcopy(Q)
PI = np.zeros((world_size[0] * world_size[1]), dtype=np.int)  # 0,1,2,3分别代表4个动作
C = np.zeros((world_size[0] * world_size[1], 4))

#每个动作概率 p=[0.1,0.2,0.3,0.4]
def generate_action(p):
    psum = 0
    randv = np.random.rand()#[0,1)
    for i in range(len(p)):
        if randv >= psum and randv < (psum + p[i]):
            return env.feasible_actions[i]
        else:
            psum += p[i]

#pt = np.array([0.25,0.25,0.25,0.25])
pb = np.array([0.25,0.25,0.25,0.25])
#设定 policy t 的 p=[0.25,0.25,0.25,0.25]
#设定 policy b 的 p=[0.1,0.2,0.3,0.4]

ite = 0
while True:  # episode循环
    ite += 1
    Q_old = copy.deepcopy(Q)
    s = env.reset()
    s_his = []
    a_his = []
    r_his = []
    while True:  # step循环
        #根据 policy b 产生数据
        a = generate_action(pb)
        s_, r, d = env.step(a)
        s_his.append(s)
        a_his.append(a)
        r_his.append(r)
        s = s_
        if d:
            break
    G = 0
    W = 1
    for i in range(len(r_his) - 1, -1, -1):
        G = r_his[i] + DISCOUNT * G
        s = s_his[i]
        a = a_his[i]
        #
        C[s, a] += W
        Q[s, a] += (W/C[s, a])*(G-Q[s, a])
        PI[s] = np.argmax(Q[s, :])
        if a != PI[s]: break
        W *= 1.0/pb[a]
    print(np.max(np.abs(Q - Q_old)))
    if np.max(np.abs(Q - Q_old)) < 0.0001 and ite >= 1000:
        break

print(ite)
print(Q)

for i in range(world_size[0]*world_size[1]):
    V[i] = np.max(Q[i,:])

print(V.reshape(world_size))

print(PI.reshape(world_size))
PI_FIG = []
for i in range(world_size[0]):
    PI_FIG_SUB = []
    for j in range(world_size[1]):
        PI_FIG_SUB.append(env.actions_figs[PI[i * world_size[0] + j]])
    PI_FIG.append(PI_FIG_SUB)
    print(PI_FIG_SUB)







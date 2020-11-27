
import numpy as np
from MazeEnv import *
import copy

world_size=np.array([5,5])
env = MazeEnv(world_size=world_size, \
              gold_pos=np.array([[1,2]]), \
              bad_pos=np.array([[3,4]]), \
              max_ite=10)
env.reset()
DISCOUNT = 0.9
V = np.zeros((world_size[0]*world_size[1]))
V_old = copy.deepcopy(V)
PI = np.random.randint(0, 4, (world_size[0]*world_size[1]), dtype=np.int)#0,1,2,3分别代表4个动作
PI_old = copy.deepcopy(PI)

def policy_evaluation():
    while True:
        V_old = copy.deepcopy(V)
        for s in env.feasible_states:
            a = PI[s]
            s_, r, d = env.step_state(s, a)
            V[s] = r + DISCOUNT * V[s_]
        if np.max(np.abs(V - V_old)) < 0.0001:
            break

def policy_improvement():
    PI_old = copy.deepcopy(PI)
    for s in env.feasible_states:
        values = []
        for a in env.feasible_actions:
            s_, r, d = env.step_state(s, a)
            values.append(r + DISCOUNT * V[s_])
        PI[s] = np.array(values).argmax()

    return np.array_equal(PI, PI_old)

while True:
    policy_evaluation()
    res = policy_improvement()
    if res:
        break

print(V.reshape(world_size))
print(PI.reshape(world_size))
PI_FIG = []
for i in range(world_size[0]):
    PI_FIG_SUB = []
    for j in range(world_size[1]):
        PI_FIG_SUB.append(env.actions_figs[PI[i*world_size[0]+j]])
    PI_FIG.append(PI_FIG_SUB)
    print(PI_FIG_SUB)

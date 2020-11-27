import numpy as np

#输出的状态都是一个一维的数字，动作的标号
#输入的动作也需要是一个一维的数组，动作的标号

AXXX = 1.2

class MazeEnv():

    def __init__(self, world_size, gold_pos, bad_pos, max_ite):
        self.world_size = world_size  # [5,5],np.array类型
        self.gold_pos = gold_pos  # [[1,2],[3,4]],np.array类型
        self.bad_pos = bad_pos  # [[1,2],[3,4]],np.array类型
        self.max_ite = max_ite
        self.actions = [np.array([0, -1]),
                        np.array([-1, 0]),
                        np.array([0, 1]),
                        np.array([1, 0])]
        self.actions_figs = ['←', '↑', '→', '↓']
        #--→Y+
        #|
        #↓
        #X+
        self.feasible_states = self.generate_feasible_states()
        self.feasible_actions = self.generate_feasible_actions()

    def generate_feasible_states(self):#num
        states = []
        for i in range(self.world_size[0]):
            for j in range(self.world_size[1]):
                s = np.array([i, j])
                if not self.is_done_state(s):
                    states.append(self.encode_s(s))
        return states

    def generate_feasible_actions(self):#num
        return [i for i in range(len(self.actions))]

    def random_state(self):
        return np.random.choice(self.feasible_states)

    def random_action(self):
        return np.random.choice(self.feasible_actions)

    def is_done_state(self, state):
        for goad_pos in self.gold_pos:
            if np.array_equal(goad_pos, state):
                return True
        for bad_pos in self.bad_pos:
            if np.array_equal(bad_pos, state):
                return True

        return False

    def reset(self, state_init=None):
        if state_init is None:
            self.state = self.feasible_states[np.random.randint(0,len(self.feasible_states))] # num
            self.state = self.decode_s(self.state) # np.array
        else:
            self.state = self.decode_s(state_init) # array

        self.ite = 0

        return self.encode_s(self.state)

    def step(self, action):#num
        next_state = self.state + self.actions[action]
        # 超过边界不能动
        reward = 0
        done = False
        x, y = next_state
        if x < 0 or x >= self.world_size[0] or y < 0 or y >= self.world_size[1]:
            reward -= 1.0
            next_state = self.state
        else:
            self.state = next_state
        # 判断有没有到达目标点，或者障碍点
        for goad_pos in self.gold_pos:
            if np.array_equal(goad_pos, self.state):
                reward += 10.0
                done = True
        for bad_pos in self.bad_pos:
            if np.array_equal(bad_pos, self.state):
                reward -= 10.0
                done = True
        # 判断有没有超过最大迭代次数
        self.ite += 1
        if self.ite >= self.max_ite:
            done = True

        return self.encode_s(self.state), reward, done

    def step_state(self, state, action):
        self.state = self.decode_s(state)
        return self.step(action)

    def decode_s(self, num):#num->array
        return np.array([int(num / self.world_size[1]), num % self.world_size[1]])

    def encode_s(self, state):#array->num
        return state[0] * self.world_size[1] + state[1]

    def decode_s_a(self, num):
        state_num = (self.world_size[0] * self.world_size[1])
        #return np.array([int(num / state_num), num % state_num])
        return np.array([int(num / 10000), num % state_num])

    def encode_s_a(self, state, action):
        #return state * (self.world_size[0] * self.world_size[1]) + action
        return state * 10000 + action

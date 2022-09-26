
from cmath import isnan
import numpy as np
from gym import spaces
import numpy as np
import gym
import math


class AtariEnv(gym.Env):
    
    def __init__(self, seed = 100):
        self.max = 100
        self.min = -100
        self.env = gym.make('BreakoutDeterministic-v4', render_mode='human') 
        self.seed = seed
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.min, self.max, (2,), dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.normalizer=[0.01, 0.01]
        self.denormalizer=[100, 100]
        self.previous_y = None
        self.action_names = ['idle', 'right', 'left']
        self.unsafe_areas = []
        self.starting = True
        self.violating_actions_samples = []


    def test_agent_accuracy(self, agent):
        return 0

    def normalize(self, state):
        return (state[0] * self.normalizer[0], state[1] * self.normalizer[1])

    def denormalize(self, state):
        return (state[0] * self.denormalizer[0], state[1] * self.denormalizer[1])

    def __check_violation(self, state):
        return False

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        while(self.starting == True):
            result_state = self.env.step(1)
            self.state_converter(result_state)
            if not math.isnan(self.state[1]):
                self.starting = False


        result_state = self.env.step(action+1)
        self.state_converter(result_state)

        done = False
        print(self.previous_y)
        if math.isnan(self.state[1]):
            if self.previous_y < 0:
                self.state[1] = -100
            else:
                done = True
                self.state[1] = 100
        else:
            self.previous_y = self.state[1]

        if not done:
            reward = 1.0
        else: 
            reward = -1.0

        return np.array(self.normalize(self.state), dtype=np.float32), reward, done, {}

    def reset(self):
        self.env.reset(seed=self.seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.step_count = 0
        self.previous_y = self.state[1]
        self.starting = True
        return np.array(self.normalize(self.state), dtype=np.float32)

    def render(self, mode="human"):
        self.env.render()

    def state_converter(self, s):
        # first dimention is y
        # ball territory
        # index x min = 8
        # index x max = 151
        # index y min = 93
        # index y max = 188
        # x length = 2
        # y length = 4

        # handler territory
        # index x min = 8
        # index x max = 151
        # index y = 190
        # x length = 16

        os = s[0]

        ball_y = None
        ball_x = None
        sy = os[93:-(len(os) -1 -188)]
        for y, xx in enumerate(sy):
            sx = xx[8:-(len(os) -1 -151)]
            for i, c in enumerate(sx):
                if c[0] != 0:
                    # found the ball
                    ball_y = y + 1
                    ball_x = i + 1
                    break
            if ball_y != None: break

        handler_x = None
        counter = 0
        for index, c in enumerate(os[190]):
            if c[0] != 0:
                counter += 1
            else:
                counter = 0
            if counter >= 12:
                # found the hanlder
                handler_x = index - 8
                break

        # align = False
        # if ((ball_x -1) >= (handler_x - 8)) and ((ball_x + 1) <= (handler_x + 8)):
        #     align = True 
        x = ((handler_x/len(os[190])) - 0.5) * 200
        y = ((ball_y/len(sy)) - 0.5) * 200 if ball_y != None else None
        self.state[0] = x
        self.state[1] = y

        # return align

# if __name__ == "__main__":
    
#     env = gym.make('BreakoutDeterministic-v4', render_mode='human') 
#     env.reset()
#     for _ in range(1000):
#         e = env.step(env.action_space.sample())
#         env.render()
#     env.close()
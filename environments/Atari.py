from asyncio.sslproto import _create_transport_context
from itertools import count
import math
import random
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import gym
from gym import envs


class RoverEnv(gym.Env):
    
    def __init__(self, seed = None):
        self.max = 100
        self.min = -100
        self.env = gym.make('BreakoutDeterministic-v4', render_mode='human') 
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.min, self.max, (2,), dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.step_count = 0
        self.rendering_size = 600
        self.rendering_scale = self.rendering_size / (self.max - self.min)
        self.normalizer=[0.01, 0.01]
        self.denormalizer=[100, 100]
        self.previous_location = (-2,-2)
        self.action_names = []
        self.unsafe_areas = []
        self.violating_actions_samples = []


    def test_agent_accuracy(self, agent):
        return 0

    def normalize(self, state):
        return (state[0] * self.normalizer[0], state[1] * self.normalizer[1])

    def denormalize(self, state):
        return (state[0] * self.denormalizer[0], state[1] * self.denormalizer[1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __check_violation(self, state):
        return False

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.state = self.move(action, self.state)

        done = self.__check_violation(self.state)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = -1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.normalize(self.state), dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.steps_beyond_done = None
        self.step_count = 0
        return np.array(self.normalize(self.state), dtype=np.float32)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.rendering_size, self.rendering_size)
            rover = rendering.FilledPolygon([(-10, -10), (-10, 10), (10, 10), (10, -10)])
            ghost = rendering.FilledPolygon([(-10, -10), (-10, 10), (10, 10), (10, -10)])
            rover.set_color(0, 0, 0)
            ghost.set_color(50, 50, 50)
            self.rovertrans = rendering.Transform()
            self.ghosttrans = rendering.Transform()
            rover.add_attr(self.rovertrans)
            ghost.add_attr(self.ghosttrans)
            self.viewer.add_geom(rover)
            self.viewer.add_geom(ghost)

            for area in self.unsafe_areas:
                left_bot = self.loc_to_screen((area[0], area[1]))
                left_top = self.loc_to_screen((area[0], area[3]))
                right_top = self.loc_to_screen((area[2], area[3]))
                right_bot = self.loc_to_screen((area[2], area[1]))
                a = rendering.FilledPolygon([left_bot, left_top, right_top, right_bot])
                a.set_color(255, 0, 0)
                self.viewer.add_geom(a)
        

        if self.state is None:
            return None

        x, y = self.loc_to_screen(self.state)
        self.rovertrans.set_translation(x, y)
        self.ghosttrans.set_translation(self.previous_location[0], self.previous_location[1])
        self.previous_location = (x, y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    
    env = gym.make('BreakoutDeterministic-v4', render_mode='human') 
    env.reset()
    for _ in range(1000):
        e = env.step(env.action_space.sample())
        env.render()
    env.close()
from asyncio.sslproto import _create_transport_context
from itertools import count
import math
import gym
import random
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class RoverEnv(gym.Env):
    """
    Description:
        A Rover moving randomly in closed environment with certain areas marked as Unsafe.

    Observation:
        Type: Float(2)
        Num     Observation              Min     Max
        0       Rover Position Hor       -100    +100
        1       Rover Position Ver       -100    +100

    Actions:
        Type: Discrete(4)
        Num   Action
        0     Move DOWN 30 units
        1     Move LEFT 30 units
        2     Move RIGHT 30 units
        3     Move UP 30 units

    Reward:
        Reward is 1 for every step taken.

    Starting State:
        Center of the field, 0,0

    Episode Termination:
        Hitting safety violation. Safety specifications are areas in the field that rover should not enter.
        Done would be true
        These areas are ajdustable. 
    """

    def __init__(self, seed = None):
        self.max = 100
        self.min = -100
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
        # MIN X MIN Y MAX X MAX Y
        self.unsafe_areas = [
            (-70, -70, -45, -45), 
            (50, 25, 100, 80), 
            (-100, 80, -80, 100)
            ]
        # state, forbidden action
        self.violating_actions_samples = [
            ((-90, -30), 0),
            ((-60, -30), 0),
            ((-30, -30), 0),
            ((0, -30), 0),
            ((30, -30), 0),
            ((60, -30), 0),
            ((90, -30), 0),
            ((90, -30), 2),
            ((90, 0), 2),
            ((30, 30), 2),
            ((30, 60), 2),
            ((90, 90), 0),
            ((90, 90), 2),
            ((90, 90), 3),
            ((60, 90), 3),
            ((30, 90), 3),
            ((0, 90), 3),
            ((-30, 90), 3),
            ((-60, 90), 3),
            ((-90, 60), 1),
            ((-90, 30), 1),
            ((-90, 0), 1),
            ((-90, -30), 1),
        ]
        self.action_names = ['down', 'left', 'right', 'up']
        self.safe_samples = [
            ((-90, -30), [2,3]),
            ((-90, 0), [0,2,3]),
            ((-90, 30), [0,2,3]),
            ((-90, 60), [0,2]),
            ((-60, -30), [1,2,3]),
            ((-60, 90), [0,2]),
            ((-30, -30), [1,2,3]),
            ((-30, 90), [0,1,2]),
            ((0, -30), [1,2,3]),
            ((0, 90), [0,1,2]),
            ((30, -30), [1,2,3]),
            ((30, 30), [0,1,3]),
            ((30, 60), [0,1,3]),
            ((30, 90), [0,1,2]),
            ((60, -30), [1,2,3]),
            ((90, -30), [1,3]),
            ((90, 0), [0,1]),
            ((90, 90), [1])
        ]


    def test_agent_accuracy(self, agent):
        error = 0

        for state, action in self.violating_actions_samples:
            a = agent.get_action(self.normalize(state))
            if a == action: error += 1

        for state, actions in self.safe_samples:
            a = agent.get_action(self.normalize(state))
            if a not in actions: error += 1

        length = len(self.violating_actions_samples) + len(self.safe_samples)
        return (1.0 - error/length) * 100

    def normalize(self, state):
        return (state[0] * self.normalizer[0], state[1] * self.normalizer[1])

    def denormalize(self, state):
        return (state[0] * self.denormalizer[0], state[1] * self.denormalizer[1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __check_violation(self, state):
        x, y = state
        for minx, miny, maxx, maxy in self.unsafe_areas:
            if x >= minx and x <= maxx and y >= miny and y <= maxy:
                return True

        if x < self.min or x > self.max or y < self.min or y > self.max:
            return True
        return False 

    def move(self, action, state):
        x, y = state

        r = 30

        # 0 bot, 1 left,  2 right, 3 top
        if y < -50:
            y -= r

        else:
            if action == 0: y -= r
            elif action == 1: x -= r
            elif action == 2: x += r
            else: y += r

        return (x,y)

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

    def loc_to_screen(self, state):
        x, y = state
        return ((x + 100) * self.rendering_scale, (y + 100) * self.rendering_scale)

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
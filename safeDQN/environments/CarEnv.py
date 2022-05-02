from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.core.defchararray import array
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from random import random

dt = 0.5

def get_safety_distance(vf):
    return max( (vf ** 2) / (2 * 6) + 5, 0)

class CarEnv(py_environment.PyEnvironment):
    """
        Action: discrete, single action that can take five values
            0 = idle
            1 = push the acceleration pedal by a certain degree
            2 = release the acceleration pedal by a certain degree
            3 = push the brake pedal by a certain degree
            4 = release the brake pedal by a certain degree 

        Observation: continuous, 5 values
            al = leader vehicle acceleration
            vl = leader vehicle velocity
            s = spacing
            af = following vehicle acceleration
            vf = following vehicle velocity
    """

    def __init__ (self):
        self._action_spec = array_spec.BoundedArraySpec(shape = (), dtype =np.int32, minimum=0, maximum=4)
        self._observation_spec = array_spec.ArraySpec(shape = (5,), dtype=np.float64)
        self._initialization()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _initialization(self):
        self._episode_ended = False
        self.time = 0
        # initial values
        self.state = [0, 10, 0, 10, 10] # al, vl, af, vf, s
        self.k = max(random(), 0.1) 

    def _reset(self):
        self._initialization()
        return ts.restart(np.array(self.state, dtype=np.float64))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.time += dt # progress the time
        t = self.time

        # dynamic is based on the Euler's numerical method
        al, vl, af, vf, s = self.state

        #al = 0
        #al = -2 * np.sin(self.k * (0.2 * t + 0.5 * np.sin(t) ))
        al = np.sin(0.2 * t + self.k)
        vl = vl + al * dt

        # maximum change in acceleration is +1 per second or -4 per second
        if action == 1:             # acceleration pedal push
            af = min(af+0.5, 2.5)       # upperbound
        elif action == 2:           # acceleration pedal release
            af = max(af-1, 0)         # lowerbound 
        elif action == 3:           # brake pedal push
            af = max(af-2, -6)        # lowerbound 
        elif action == 4:           # brake pedal release
            af = min(af+3, 0)         #  upperbound

        vf = max(vf + af * dt, 0)   # velocity cannot go below zero
        s = s + (vl - vf) * dt

        self.state = [al, vl, af, vf, s]

        # pedal protection violation
        if (af > 0 and action == 3) or (af < 0 and action == 1):
            self._episode_ended = True
            reward = 0
            return ts.termination(np.array(self.state, dtype= np.float64), reward)

        # collision-free violation 
        if s < 0:
            self._episode_ended = True
            reward = 0
            return ts.termination(np.array(self.state, dtype= np.float64), reward)


        if (0 <= s) and (s <= 10):
            reward = 1
        else: # if (10 <= s)
            reward = 10/s 

        if self.time >= 30:
            self._episode_ended = True
            return ts.termination(np.array(self.state, dtype= np.float64), reward)
        else:
            return ts.transition(np.array(self.state, dtype=np.float64), reward, discount=1.0)





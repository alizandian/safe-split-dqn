from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import random
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from MapInfo import MapInfo

class Ball(object):
    def __init__(self, id, color, init_pos):
        self.id = id
        self.color = color
        self.pos = init_pos
        self.next_pos = init_pos

    def get_id(self):
        return self.id

    def get_pos(self):
        return self.pos

    def get_color(self):
        return self.color

    def get_next_pos(self):
        return self.get_next_pos

    def move(self):
        self.pos = self.next_pos
    
    def set_next_pos(self, pos):
        self.next_pos = pos

    def collected(self):
        # this method can later return more data 
        return self.color


class GameEngine(py_environment.PyEnvironment):

    def __init__ (self):
        # here we need to define action and observation specifications
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(14,), dtype = np.int32, minimum=0, maximum=5, name='observation'
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(15,), dtype = np.int32, 
            minimum=0, 
            maximum=2,
            name='action'
        )
        self._episode_ended = False
        self._state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self._output = {} # to track how many outputs we yield at each step
        """ 
        System state is a vector of the value of each node in the graph
        Each value can be 0 to 5:
        0 = Empty
        1 = Red ball is occupying
        2 = Green ball is occupying
        3 = Blue ball is occupying
        4 = Yellow ball is occupying
        5 = Collision (or unknown state)
        """

        self.mapinfo = MapInfo()
        self.mapinfo.load_scenario1()

        self.ball_instances = []
        self.ball_id = 0 # increasing id given to each ball as they spawn

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # reset environment
        self._state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def get_ball_index(self, pos):
        i = 0
        detected = False
        for ball in self.ball_instances:
            if ball.get_pos() == pos: # if this position is the output pos
                detected = True
                break
            i += 1
        return i, detected

    def update_output(self, pos):
        if pos in self._output:
            self._output[pos] += 1
        else:
            self._output[pos] = 1

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        end_of_game = 0 # end of game indicate which condition is satisfied for ending the game
        # 0 means the game is not finished
        # 1 means the game is finished because there is a collision
        # 2 means the game is finished because there is no ball left 
        # 3 means the game is finished because wrong ball is passed to the output site

        # Step 0. despawn balls that are at the output site
        for pos, color in self.mapinfo.get_output_dict().items():
            # the output site is empty
            if self._state[pos] == 0:
                continue

            ball_index, _ = self.get_ball_index(pos)

            if self.ball_instances[ball_index].get_color() == color:
                self.update_output(pos)
            else:
                end_of_game = 3
            
            self.ball_instances.pop(ball_index) # remove the ball instance

        # Step 1. move the balls according to the action
        for ball in self.ball_instances:
            pos = ball.get_pos()                                     # get the ball position
            [for_e, rev_e] = self.mapinfo.get_connected_edges(pos)   # get all the associated edges
            avail_e = []
            for e_id in for_e:
                if action[e_id] == 1:
                    avail_e.append(e_id)
            for e_id in rev_e:
                if action[e_id] == 2:
                    avail_e.append(e_id)

            if len(avail_e) == 0: # this ball cannot move (since no enabled edge)
                continue

            # select one randomly from the available edges
            (s,d) = self.mapinfo.get_edges()[random.choice(avail_e)]
            if s == pos:
                ball.set_next_pos(d)
            else:
                ball.set_next_pos(s)
            ball.move()


        # Step 2. spawn balls at the input site by consuming the task queue
        for pos, tasks in self.mapinfo.get_tasks_dict().items():
            if len(tasks) == 0: # no task 
                continue
            if self._state[pos] == 0: # the input site is clear to put a ball
                color = int(tasks.pop(0))
                ball = Ball(self.ball_id, color, pos)
                self.ball_instances.append(ball)
                self.ball_id += 1 


        # update the state (occupancy of the ball in the map)
        # this must be done after spawning the ball
        next_state = np.zeros(14)
        for ball in self.ball_instances:
            pos = ball.get_pos()
            # check if this position is already occupied
            if next_state[pos] > 0:
                next_state[pos] = 5
                end_of_game = 1
            else:
                next_state[pos] = ball.get_color()

        self._state = next_state # update the state        

        if len(self.ball_instances) == 0:
            end_of_game = 2

        # return the transition data
        if end_of_game == 0:    # game should continue
            return ts.transition(np.array(self._state, dtype=np.int32), reward=-1, discount=1.0)
        elif end_of_game == 1:  # collision detected
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.int32), -100) # the reward is -100
        elif end_of_game == 2:  # no ball left
            self._episode_ended = True
            return ts.transition(np.array(self._state, dtype=np.int32), reward=-1)
        else:                   # wrong color ball at the output site
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.int32), -100) # the reward is -100



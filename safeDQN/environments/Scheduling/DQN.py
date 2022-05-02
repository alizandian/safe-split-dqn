from collections import deque
import random
import tensorflow as tf

"""
Based on the DQN theory, each learning iteration requires a transition
Each transition is a tuple <s, a, r, s'>, where:
s = state
a = action
r = reward
s' = next state
The replay buffer stores this tuple and provides interface to access them
"""
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
    
    def append(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        size = batch_size if self.buffer_size > batch_size else self.buffer_size
        if len(self.buffer) < batch_size:
            return self.buffer
        return random.sample(self.buffer, size)

    def clear(self):
        self.buffer.clear()
        

class Agent(object):

    def __init__(self, action_spec):

        pass
            


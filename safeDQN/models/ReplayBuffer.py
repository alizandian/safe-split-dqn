from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
    
    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_buffer(self):
        return self.buffer

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
    
    def sample_sequential(self, data_count):
        if len(self.buffer) < data_count:
            return [], False
        else:
            random_index = random.randint(0, len(self.buffer) - data_count)
            return self.buffer[random_index : random_index+data_count-1 ], True
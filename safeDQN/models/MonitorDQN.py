from AgentDQN import AgentDQN
import tensorflow as tf
import numpy as np
from ReplayBuffer import ReplayBuffer

class MonitorDQN(AgentDQN):

    def __init__(self, input_dim, output_dim, nn_model, batch_size=32, epoch=10, gamma=1, replay_buffer_size=500, counterexample_buffer_size=50, update_target_interval=2, verbose=False):
        super().__init__(input_dim, output_dim, nn_model, batch_size=batch_size, epoch=epoch, gamma=gamma, replay_buffer_size=replay_buffer_size, update_target_interval=update_target_interval, verbose=verbose)
        
        self.counterexample_buffer = ReplayBuffer(counterexample_buffer_size)

    # the monitor has two buffers, one for safe data points, and the other for counterexamples
    def add_transition(self, trans):
        counterexample_flag = trans[-1]
        if counterexample_flag > 0:
            self.counterexample_buffer.append(trans)
        else:
            self.replay_buffer.append(trans)
        
    def Qnet_train(self):
        safe_batch = min(len(self.replay_buffer), int(self.batch_size/2))
        unsafe_batch = min(len(self.counterexample_buffer), int(self.batch_size/2))
        if unsafe_batch == [None]: unsafe_batch = 0

        loss = []
        for i in range(self.epoch):
            safe_points = self.replay_buffer.sample(safe_batch)
            unsafe_points = self.counterexample_buffer.sample(unsafe_batch)
            data_points = safe_points + unsafe_points
            l = self.learn(data_points)
            loss.append(l)
        average_loss = np.average(loss)
        self.update_q_target()
        
        return True, average_loss
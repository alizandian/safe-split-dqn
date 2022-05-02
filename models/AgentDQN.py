from models.ReplayBuffer import ReplayBuffer
from models.DQN import DQN
import numpy as np

class AgentDQN(object):
    def __init__(self, input_dim, output_dim, nn_model, gamma = 1.0, replay_buffer_size = 500, target_update_interval = 100, verbose = False):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
        self.frame_count = 0
        self.epsilon_random_frames = 3500
        self.epsilon_greedy_frames = 50000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.update_interval = target_update_interval
        self.update_counter = 0
        self.dqn = DQN(input_dim = input_dim, output_dim = output_dim,nn_model = nn_model, gamma = gamma, verbose = verbose)
        self.previous_action_type = -1

    def get_action(self, input):
        self.frame_count += 1
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            if self.previous_action_type != 0: 
                print("--------------------------RANDOM---------------------------")
                self.previous_action_type = 0
            return np.random.randint(self.output_dim)
        else:
            if self.previous_action_type != 1: 
                print("--------------------------CHOSE----------------------------")
                self.previous_action_type = 1
            q_values = self.dqn.get_q_values(np.array(input)[np.newaxis])[0]
            return np.argmax(q_values)

    def train(self, batch_size=64, epoch=1):
        batch_size_ = min(batch_size, len(self.replay_buffer))
        epoch_ = min(epoch, int(len(self.replay_buffer)/batch_size_)+1)

        loss = 0
        for i in range(0, epoch_):
            transitions = self.replay_buffer.sample(batch_size_)
            loss += self.dqn.learn(transitions, batch_size)

        loss = loss / epoch_ 
        self.update_counter += 1

        if self.update_counter >= self.update_interval:
            self.dqn.update_q_target()
            self.update_counter = 0

        return loss 

    def add_transition(self, trans):
        self.replay_buffer.append(trans)

    def random_action(self):
        return np.random.randint(self.output_dim)

    def save_model(self, file_name):
        self.dqn.save(file_name)

    def load_model(self, file_name):
        self.dqn.load(file_name)

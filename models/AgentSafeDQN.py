from models.ReplayBuffer import ReplayBuffer
from models.DQN_Base import DQN_Base as DQN 
import numpy as np
import random

class AgentSafeDQN(object):
    def __init__(self, input_dim, output_dim, nn_model, monitor_nn_model,
                gamma = 1.0, epsilon = 0.01,
                replay_buffer_size = 128, target_update_interval = 100,
                verbose = False):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.base_dqn = DQN(input_dim = input_dim, 
                        output_dim = output_dim,
                        nn_model = nn_model,
                        gamma = gamma,
                        verbose = verbose)

        self.dqn = DQN(input_dim = input_dim, 
                        output_dim = output_dim,
                        nn_model = monitor_nn_model,
                        gamma = gamma,
                        verbose = verbose)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_max = 1.0
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
        self.frame_count = 0
        self.epsilon_random_frames = 100
        self.epsilon_greedy_frames = 100
        self.previous_action_type = -1

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.counterexample_buffer = ReplayBuffer(replay_buffer_size)

        self.update_interval = target_update_interval
        self.update_counter = 0

    def add_experience(self, experience):
        s, a, r, n, v = experience
        e = (s, a, r, n, v, v)
        # each transition data must be in the format below:
        # [state, action, reward, new_state, done, violation]

        # first of all, put the data into the replay buffer 
        self.replay_buffer.append(e)

        # if this is a counterexample, put into the counterexample buffer too
        violation = e[-1]
        if violation != 0:
            self.counterexample_buffer.append(e)

    def get_action(self, input, theta = 0.0):
        self.frame_count += 1
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        #if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
        if False:
            if self.previous_action_type != 0: 
                print("--------------------------RANDOM---------------------------")
                self.previous_action_type = 0
            return np.random.choice(self.output_dim)

        else:
            if self.previous_action_type != 1: 
                print("--------------------------CHOSE----------------------------")
                self.previous_action_type = 1

            safety_q_val = self.dqn.get_q_values(np.array(input)[np.newaxis])[0]
            mask = [ 1 if val >= theta else 0 for val in safety_q_val]
            action_q_val = self.base_dqn.get_q_values(np.array(input)[np.newaxis])[0]
            final_action_q_val = [ q * m for q, m in zip(action_q_val, mask) ]
            if mask.count(1) == 0:
                return np.argmax(action_q_val)

            return np.argmax(final_action_q_val)

    def train(self, batch_size=32, epoch=1):
        batch_size_ = min(batch_size, len(self.replay_buffer))
        epoch_ = min(epoch, int(len(self.replay_buffer)/batch_size_))+1

        loss_action = 0
        loss_monitor = 0
        for i in range(0, epoch_):
            transitions = self.replay_buffer.sample(batch_size_)

            # We need to convert each transition
            #   [state, action, reward, new_state, done, violation]
            # into two structures
            #   [state, action, reward, new_state, done]
            #   and
            #   [state, action, violation, new_state, done]
            transitions_action = [ [s,a,r,n,d] for s,a,r,n,d,v in transitions]
            transitions_monitor = [ [s,a,v,n,d] for s,a,r,n,d,v in transitions]

            loss_action += self.base_dqn.learn(transitions_action)
            loss_monitor += self.dqn.learn(transitions_monitor)

        # Repeat the monitor DQN training only with the counterexample data
        batch_size_ = min(batch_size, len(self.counterexample_buffer))
        for i in range(0, epoch_):
            transitions = self.counterexample_buffer.sample(batch_size_) 
            transitions_monitor = [ [s,a,v,n,d] for s,a,r,n,d,v in transitions]
            loss_monitor += self.dqn.learn(transitions_monitor)

        loss_action /= epoch_
        loss_monitor /= (epoch_ * 2)

        self.update_counter += 1
        if self.update_counter >= self.update_interval:
            self.base_dqn.update_q_target()
            self.dqn.update_q_target()
            self.update_counter = 0

        return loss_action, loss_monitor

    def random_action(self):
        return np.random.randint(self.output_dim)

    def save_model(self, file_name_dqn, file_name_monitor):
        self.base_dqn.save(file_name_dqn)
        self.dqn.save(file_name_monitor)

    def load_model(self, file_name_dqn, file_name_monitor):
        self.base_dqn.load(file_name_dqn)
        self.dqn.load(file_name_monitor)

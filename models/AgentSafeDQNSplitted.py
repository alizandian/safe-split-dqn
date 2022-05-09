from models.ReplayBuffer import ReplayBuffer
from models.DQNSplited import DQNSplited
from models.DQN import DQN
import numpy as np

class AgentSafeDQNSplitted(object):
    def __init__(self, input_dim, output_dim, nn_model, estimator_nn_model, gamma = 1.0, replay_buffer_size = 500, target_update_interval = 100, verbose = False):

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Explore vs Exploit parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
        self.frame_count = 0
        self.epsilon_random_frames = 3500
        self.epsilon_greedy_frames = 50000

        self.update_interval = target_update_interval
        self.update_counter = 0
        self.previous_action_type = -1

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.counterexample_buffer = ReplayBuffer(replay_buffer_size)

        self.dqn = DQN(input_dim = input_dim, output_dim = output_dim,nn_model = nn_model, gamma = gamma, verbose = verbose)
        self.estimator_dqn = DQNSplited(input_dim = input_dim, output_dim = output_dim, nn_model = estimator_nn_model, gamma = gamma, verbose = verbose)


    def get_action(self, input, theta = 0.9):
        safety_q_val = self.estimator_dqn.get_q_values(np.array(input)[np.newaxis])[0]
        mask = [ 1 if val >= theta else 0 for val in safety_q_val]
        action_q_val = self.dqn.get_q_values(np.array(input)[np.newaxis])[0]
        final_action_q_val = [ q * m for q, m in zip(action_q_val, mask) ]

        if mask.count(1) == 0:
            return np.argmax(action_q_val)

        self.frame_count += 1
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            if self.previous_action_type != 0: 
                print("--------------------------RANDOM---------------------------")
                self.previous_action_type = 0
            valid_actions = [ i for i, x in enumerate(mask) if x == 1]
            return np.random.choice(valid_actions)

        else:
            if self.previous_action_type != 1: 
                print("--------------------------CHOSE----------------------------")
                self.previous_action_type = 1
            return np.argmax(final_action_q_val)

    def train(self, batch_size=64, epoch=1):
        batch_size_ = min(batch_size, len(self.replay_buffer))
        epoch_ = min(epoch, int(len(self.replay_buffer)/batch_size_)+1)

        loss = 0
        loss_estimator = 0
        for i in range(0, epoch_):
            transitions = self.replay_buffer.sample(batch_size_)
            loss += self.dqn.learn(transitions, batch_size)
            estimator_transitions = [ [s, a, -10 if d == True else r, n, d] for s,a,r,n,d in transitions]
            loss_estimator += self.estimator_dqn.learn(estimator_transitions)

        # Repeat the monitor DQN training only with the counterexample data
        batch_size_ = min(batch_size, len(self.counterexample_buffer))
        for i in range(0, epoch_):
            transitions = self.counterexample_buffer.sample(batch_size_) 
            estimator_transitions = [ [s, a, -10 if d == True else r, n, d] for s,a,r,n,d in transitions]
            loss_estimator += self.estimator_dqn.learn(estimator_transitions)

        loss = loss / epoch_ 
        loss_estimator /= (epoch_ * 2)

        self.update_counter += 1

        if self.update_counter >= self.update_interval:
            self.dqn.update_q_target()
            self.estimator_dqn.update_q_target()
            self.update_counter = 0

        return loss, loss_estimator

    def add_transition(self, trans):
        self.replay_buffer.append(trans)

        violation = trans[-1]
        if violation != 0:
            self.counterexample_buffer.append(trans)

    def random_action(self):
        return np.random.randint(self.output_dim)

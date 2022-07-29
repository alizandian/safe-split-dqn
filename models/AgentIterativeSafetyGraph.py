from classes.graph import Graph
from models.ReplayBuffer import ReplayBuffer
from models.DQN import DQN
from typing import List, Tuple
import numpy as np
import configparser

class AgentIterativeSafetyGraph(object):
    def __init__(self, input_dim, output_dim, nn_model, dimention, do_enhance_transitions=True, gamma = 1.0, replay_buffer_size = 200, verbose = False):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.do_enhance_transitions = do_enhance_transitions
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
        self.frame_count = 0
        self.epsilon_random_frames = 3500
        self.epsilon_greedy_frames = 50000
        self.previous_action_type = -1
        self.history_buffer = ReplayBuffer(1000)
        self.transition_buffer = ReplayBuffer(replay_buffer_size)
        self.dqn = DQN(input_dim = input_dim, output_dim = output_dim, nn_model = nn_model, gamma = gamma, verbose = verbose)
        self.safety_graph = Graph((dimention), (-1, -1), (1, 1))

    def __clamp(self, n, smallest, largest): return max(smallest, min(n, largest))

    def predict(self, s):
        return self.dqn.Q_target.predict(s)

    def get_action(self, input, theta = -8.0):
        self.frame_count += 1
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
            if self.previous_action_type != 0: 
                print("------------------------  RANDOM  -------------------------")
                self.previous_action_type = 0
            return np.random.choice(self.output_dim)

        else:
            if self.previous_action_type != 1: 
                print("------------------------  CHOSE  -------------------------")
                self.previous_action_type = 1
            q_val = self.dqn.get_q_values(np.array(input)[np.newaxis])[0]
            mask = [ 1 if val >= theta else 0 for val in q_val]
            final_action_q_val = [ q * m for q, m in zip(q_val, mask)]
            if mask.count(1) == 0: return np.argmax(q_val)
                
            return np.argmax(final_action_q_val)

    def enhance_transitions(self, transitions: List[Tuple]):
        return [[s,a, self.__clamp((-1 * self.safety_graph.proximity_to_unsafe_states((s,a,r,n,d))) * 100, -100, 1), n,d] for s,a,r,n,d in transitions]

    def train(self):
        hb = self.history_buffer.get_buffer()
        tb = self.transition_buffer.get_buffer()
        transitions = self.enhance_transitions(list(tb)) if self.do_enhance_transitions else tb
        if len(hb) >= 500:
            history = self.enhance_transitions(self.history_buffer.sample(500))
            self.dqn.learn(history, len(history))

        self.dqn.learn(transitions, len(tb))
        self.dqn.learn(transitions, len(tb))
        self.dqn.update_q_target()
        self.update_counter = 0
        hb.extend(tb)
        self.transition_buffer.clear()
        #self.safety_graph.visualize()

    def add_transition(self, trans):
        self.transition_buffer.append(trans)
        self.safety_graph.add_transitions(trans[0], trans[3], trans[4])

    def random_action(self):
        return np.random.randint(self.output_dim)

from classes.graph import Graph
from models.ReplayBuffer import ReplayBuffer
from models.DQN import DQN
from typing import List, Tuple
import numpy as np
import configparser

class AgentIterativeSafetyGraph(object):
    def __init__(self, input_dim, output_dim, nn_model, dimention, refined_experiences=True, gamma = 0.8, replay_buffer_size = 200, verbose = False):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.refined_experiences = refined_experiences
        self.dimention = dimention
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_max = 1.0
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
        self.frame_count = 0
        self.epsilon_random_frames = 0
        self.epsilon_greedy_frames = 4000
        self.update_counter = 0
        self.update_target_interval = 200
        self.previous_action_type = -1
        self.history_buffer = ReplayBuffer(1000)
        self.experience_buffer = ReplayBuffer(replay_buffer_size)
        self.dqn = DQN(input_dim = input_dim, output_dim = output_dim, nn_model = nn_model, gamma = gamma, verbose = verbose)
        self.safety_graph = Graph(output_dim, dimention, (-1, -1), (1, 1))

    def __clamp(self, n, smallest, largest): return max(smallest, min(n, largest))

    def predict(self, s):
        return self.dqn.Q_target.predict(s)

    def get_action(self, input, theta = 0.0):
        self.frame_count += 1
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        # if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
        if self.frame_count < self.epsilon_random_frames:
            if self.previous_action_type != 0: 
                print("------------------------  RANDOM  -------------------------")
                self.previous_action_type = 0
            return np.random.choice(self.output_dim)

        else:
            if self.previous_action_type != 1: 
                print("------------------------  CHOSE  -------------------------")
                self.previous_action_type = 1
            q_val = self.dqn.get_q_values(np.array(input)[np.newaxis])[0]
            # mask = [ 1 if val >= theta else 0 for val in q_val]
            # final_action_q_val = [ q * m for q, m in zip(q_val, mask)]
            # if mask.count(1) == 0: return np.argmax(q_val)
                
            # return np.argmax(final_action_q_val)

            possibles = self.safety_graph.get_safe_actions(input)
            if len(possibles) == 0:
                return np.argmax(q_val)
            else:
                return np.random.choice(possibles)


    def refine_experiences(self, experiences: List[Tuple], only_last_items = 20):
        t = experiences if len(experiences) < only_last_items else experiences[-only_last_items+1:]
        return [[s,a, self.safety_graph.transition_safety((s,a,r,n,d)), n,d] for s,a,r,n,d in t]

    def train(self):
        if self.refined_experiences: self.safety_graph.update()

        hb = self.history_buffer.get_buffer()
        tb = self.experience_buffer.get_buffer()
        experiences = self.refine_experiences(list(tb)) if self.refined_experiences else tb
        if len(hb) >= 100:
            history_b = self.history_buffer.sample(100)
            history = self.refine_experiences(history_b, 100) if self.refined_experiences else history_b
            self.dqn.learn(history, len(history), self.refined_experiences)
            self.update_counter += len(history)

        self.dqn.learn(experiences, len(tb), self.refined_experiences)
        self.update_counter += len(tb)

        if self.update_counter >= self.update_target_interval:
            self.dqn.update_q_target()
            self.update_counter = 0

        hb.extend(tb)
        self.experience_buffer.clear()

        if self.refined_experiences: self.safety_graph.feed_neural_network_feedback(self.dqn.get_snapshot(self.dimention)[0])

    def add_experience(self, e):
        self.experience_buffer.append(e)
        self.safety_graph.add_experience(e)

    def random_action(self):
        return np.random.randint(self.output_dim)

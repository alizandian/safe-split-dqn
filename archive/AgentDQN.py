from safeDQN.models.ReplayBuffer import ReplayBuffer
import tensorflow as tf
import numpy as np
import random

class AgentDQN(object):

    def __init__(self, input_dim, output_dim, nn_model, batch_size=32, epoch=10, 
                    gamma = 1.0, replay_buffer_size=500, update_target_interval=2, 
                    verbose=False):

        self.input_dim = input_dim
        self.output_dim = output_dim
            
        self.Q_main = nn_model
        self.Q_target = tf.keras.models.clone_model(nn_model)

        # further parameterisation can be implemented here
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.batch_size = batch_size
        self.epoch = epoch

        self.gamma = gamma
        self.epsilon = 0.01
        self.replay_buffer_size = replay_buffer_size
        self.update_q_target_interval = update_target_interval
        self.update_q_target_count = 0

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # compile the neural networks
        self.Q_main.compile(optimizer = self.optimizer, loss = self.loss)
        self.Q_target.compile(optimizer = self.optimizer, loss = self.loss)
        if verbose: self.Q_main.summary()

        self.bellman_mode = 'max'

    def select_action(self, obs):
        action_q_values = self.Q_main.predict( np.array(obs)[np.newaxis] )
        return np.argmax(action_q_values[0])

    def action_q_values(self, obs):
        return self.Q_main.predict( np.array(obs)[np.newaxis] )[0]

    def get_q_values(self, obs):
        return self.Q_main.predict( obs )

    def update_q_target(self):
        self.Q_target.set_weights(self.Q_main.get_weights() )
        self.update_q_target_count = 0

    def masked_action_selection(self, obs, estimate, theta):
        q_values = self.action_q_values(obs)
        mask = [ 1 if e >= theta else 0 for e in estimate ]
        filtered_actions = [ a * b for a,b in zip(q_values, mask)] 

        if mask.count(1) == 0:
            return np.argmax(q_values)
            #return np.argmax(estimate)

        if np.random.uniform() <= self.epsilon:
            valid_index = [i for i, x in enumerate(mask) if x == 1]
            return random.choice(valid_index)
        else:
            return np.argmax(filtered_actions)


    def greedy_action_selection(self, obs):
        # greedy action selection
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(self.output_dim)
        else:
            return self.select_action(obs)

    def set_bellman_mode(self, mode):
        if mode == 'max' or mode =='avg':
            self.bellman_mode = mode
            print("Bellman mode: " + self.bellman_mode)
        return self.bellman_mode

    def custom_training(self, input, target, batch_size=32, epoch=10):
        for i in range(epoch):
            self.Q_main.fit(input, target, verbose=0, batch_size = batch_size)
        self.update_q_target()

    def random_action(self):
        return np.random.randint(self.output_dim)

    def random_run(self, env, iterations, visualize=False, learning=False, verbose=False):
        done_count = 0
        while done_count < iterations:
            state = env.reset()
            while True:
                action = self.random_action()
                next_state, reward, done, _ = env.step(action)
                self.add_transition([state, action, reward, next_state, done])
                state = next_state
                if verbose: print(reward)
                if done: break
            if learning: self.Qnet_train()
            done_count += 1

    # this method helps on the automation of the training loop
    def run(self, env, iterations, visualize=False, learning=True, verbose=False):

        rewards = []

        for ep_count in range(0, iterations):
            state = env.reset()
            ep_reward = 0
            if visualize: env.render()

            while True:
                action = self.greedy_action_selection(state)
                next_state, reward, done, _ = env.step(action)
                trans = [state, action, reward, next_state, done]
                self.add_transition(trans)
                ep_reward += reward
                state = next_state
                if visualize: env.render()
                if done: break

            if learning: self.Qnet_train()
            rewards.append(ep_reward)
            if verbose: print("Episode {0}/{1} -- reward {2}".format(ep_count+1, iterations, ep_reward)) 

        return rewards

    def Qnet_train(self):
        if len(self.replay_buffer) < self.batch_size * 2:
            return False, 0
        else:
            loss = []
            for i in range(self.epoch):
                transitions = self.replay_buffer.sample(self.batch_size)
                l = self.learn(transitions)
                loss.append(l)

            average_loss = np.average(loss)
            self.update_q_target_count += 1

        if self.update_q_target_count >= self.update_q_target_interval:
            self.update_q_target()
        
        return True, average_loss

    def learn(self, transitions):
        # transitions is a array which must have the first four element:
        # transitions = [ tras1, tras2, tras3, ... ], where:
        # tras = [state, action, reward, next_state, done]

        current_state = np.stack([traj[0] for traj in transitions])
        next_state = np.stack([traj[3] for traj in transitions])
        current_q_values = self.Q_main.predict(current_state)
        next_q_values = self.Q_target.predict(next_state)

        for i, (s, a, r, sp, done) in enumerate(transitions):
            if done:
                target_q = r
            else:
                if self.bellman_mode == 'max':
                    target_q = r + self.gamma * np.max(next_q_values[i])
                elif self.bellman_mode == 'avg':
                    target_q = r + self.gamma * np.average(next_q_values[i])
            # update the current q-value
            current_q_values[i, a] = target_q

        result = self.Q_main.fit(current_state, current_q_values, 
                                    shuffle=False, 
                                    verbose=0, 
                                    batch_size = self.batch_size)

        loss = result.history['loss'][0]
        return loss

    def add_transition(self, trans):
        self.replay_buffer.append(trans)

    def save(self, model_filepath):
        self.Q_main.save(model_filepath)

    def load(self, model_filepath):
        self.Q_main = tf.keras.models.load_model(model_filepath)
        self.update_q_target()



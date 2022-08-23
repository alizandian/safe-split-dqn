import tensorflow as tf
import numpy as np


class DQN(object):

    def __init__(self, input_dim, output_dim, nn_model, 
                optimizer = 'adam', loss_function = 'mse', gamma = 0.5,
                verbose=False):

        self.input_dim = input_dim
        self.output_dim = output_dim
            
        self.Q_main = nn_model
        self.Q_target = tf.keras.models.clone_model(nn_model)

        # further parameterisation can be implemented here
        self.optimizer = optimizer
        self.loss = loss_function
        self.gamma = gamma

        # compile the neural networks
        self.Q_main.compile(optimizer = self.optimizer, loss = self.loss)
        self.Q_target.compile(optimizer = self.optimizer, loss = self.loss)

        if verbose: self.Q_main.summary()

    def get_q_values(self, input):
        return self.Q_main.predict(input)

    def update_q_target(self):
        self.Q_target.set_weights(self.Q_main.get_weights())

    """
    If the user can provide target data, supervised learning can run
    """
    def supervised_learning(self, input, target, batch_size=32, epoch=10):
        for i in range(epoch):
            self.Q_main.fit(input, target, verbose=0, batch_size = batch_size)
        self.update_q_target()

    def get_snapshot(self, reso, violation_check_func=None):
        values = [[0]*reso for i in range(reso)]
        states = []
        for y in range(reso):
            for x in range(reso):
                d = 2 / reso
                dd = d / 2
                states.append(((x * d) + dd - 1, (y * d) + dd - 1))

        results = self.Q_main.predict(np.stack(states))
        r = np.reshape(results, (-1, reso, 4))

        min = np.min(r)
        max = np.max(r)

        min_threshold = min + ((max - min) * 0.5)

        error = 0
        for y in range(reso):
            for x in range(reso):
                v = r[y][x]
                value = np.min(v)
                values[reso-y-1][x] = v
                if violation_check_func != None:
                    violation = violation_check_func(states[y*reso+x])
                    value_estimation = True if value < min_threshold else False
                    if violation != value_estimation: error += 1
        accuracy = (1 - (error / (reso * reso))) * 100

        return values, accuracy


    def learn(self, transitions, batch_size = 64, is_enhanced = False):
        # transitions is a array which must have the first four element:
        # transitions = [ tras1, tras2, tras3, ... ], where:
        # tras = [state, action, reward, next_state, done]

        current_state = np.stack([trans[0] for trans in transitions])
        next_state = np.stack([trans[3] for trans in transitions])
        current_q_values = self.Q_main.predict(current_state)
        next_q_values = self.Q_target.predict(next_state)

        for i, (_, a, r, _, done) in enumerate(transitions):
            if is_enhanced:
                current_q_values[i, a] = r
            else:
                if done: 
                    target_q = r
                else: 
                    target_q = r + self.gamma * np.min(next_q_values[i])
                # update the current q-value
                current_q_values[i, a] = target_q
            
        result = self.Q_main.fit(current_state, current_q_values, shuffle=False, verbose=0, batch_size = batch_size)
        loss = result.history['loss'][0]
        return loss


    def save(self, model_filepath):
        self.Q_main.save(model_filepath)

    def load(self, model_filepath):
        self.Q_main = tf.keras.models.load_model(model_filepath)
        self.update_q_target()


from typing import Iterator, Tuple
import tensorflow as tf
import numpy as np
from keras.models import Model

class DQNSplit(object):

    def __init__(self, input_dim, output_dim, nn_model, dimention,
                optimizer = 'adam', loss_function = 'mse', gamma = 1.0, 
                verbose=False):

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Models are in 2d grid, and each entity is a tuple, first is the main and the second is the target
        self.x_axis_model_count = dimention
        self.y_axis_model_count = dimention
        self.models: list[list[tuple[Model, Model]]] = [[(tf.keras.models.clone_model(nn_model), tf.keras.models.clone_model(nn_model)) for j in range(self.y_axis_model_count)] for i in range(self.x_axis_model_count)]

        # further parameterisation can be implemented here
        self.optimizer = optimizer
        self.loss = loss_function
        self.gamma = gamma

        # compile the neural networks
        for (main, target) in self.get_models_iterator():
            main.compile(optimizer = self.optimizer, loss = self.loss)
            target.compile(optimizer = self.optimizer, loss = self.loss)


    def get_models_iterator(self) -> Iterator[Tuple[Model, Model]]:
        for i in range(self.x_axis_model_count):
            for j in range(self.y_axis_model_count):
                yield self.models[i][j]

    def get_model(self, state) -> Tuple[Model, Model]:
        t = self.get_model_indecis(state)
        return self.models[t[0]][t[1]]

    def clamp(self, n, smallest, largest): return max(smallest, min(n, largest))

    def get_model_indecis(self, state) -> Tuple[int, int]:
        xl = 2.0 / self.x_axis_model_count
        yl = 2.0 / self.y_axis_model_count

        xi = self.clamp(int((state[0] + 1) / xl), 0, self.x_axis_model_count-1)
        yi = self.clamp(int((state[1] + 1) / yl), 0, self.x_axis_model_count-1)

        return (xi, yi)

    def get_q_values(self, state):
        return self.get_model(state[0])[0].predict(state)

    def update_q_target(self):
        for (main, target) in self.get_models_iterator():
            target.set_weights(main.get_weights())

    def learn(self, transitions, batch_size = 32):
        model__s_q= {}
        ti__index__qv = {i:{j:None for j in range(2)} for i in range(len(transitions))}
        mi__mti__ti = {i:{j:{z:[] for z in range(2)} for j in range(self.y_axis_model_count)} for i in range(self.x_axis_model_count)}
        
        for i, (s, a, r, sp, done) in enumerate(transitions):
            s_indecis = self.get_model_indecis(s)
            sp_indecis = self.get_model_indecis(sp)

            mi__mti__ti[s_indecis[0]][s_indecis[1]][0].append(i)
            mi__mti__ti[sp_indecis[0]][sp_indecis[1]][1].append(i)

        for i in range(self.x_axis_model_count):
            for j in range(self.y_axis_model_count):
                for z in range(2):
                    tindecis = mi__mti__ti[i][j][z]
                    if len(tindecis) > 0:
                        x = 0 if z == 0 else 3
                        states = np.array([t[x] for t in transitions])[tindecis]
                        results = self.models[i][j][z].predict(states)
                        for k, ti in enumerate(tindecis):
                            ti__index__qv[ti][z] = results[k]


        for i, (s, a, r, sp, done) in enumerate(transitions):
            current_q_values = ti__index__qv[i][0]
            next_q_values = ti__index__qv[i][1]
            if done:  target_q = r
            else: target_q = r + self.gamma * np.max(next_q_values)
            current_q_values[a] = target_q
            model = self.get_model(s)[0]
            if model not in model__s_q: model__s_q[model] = []
            model__s_q[model].append((s, current_q_values))
            
        loss = 0 
        for m in model__s_q:
            sqvalues = list(map(list, zip(*model__s_q[m]))) 
            result = m.fit(np.stack(sqvalues[0]), np.stack(sqvalues[1]), shuffle=False, verbose=0, batch_size = batch_size)
            loss += result.history['loss'][0]

        return loss


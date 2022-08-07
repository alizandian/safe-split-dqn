import tensorflow as tf
import numpy as np

def SimplifiedCartPole_DQN_NN(input_dim=2, output_dim=2):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=np.random.randint(1000))

    inputs = tf.keras.Input(shape=(input_dim,))
    hidden1 = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer=initializer, name='hidden1')(inputs)
    hidden2 = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer=initializer, name='hidden2')(hidden1)
    outputs = tf.keras.layers.Dense(output_dim, activation = 'relu', kernel_initializer=initializer, name='output')(hidden2)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return input_dim, output_dim, model

def Smaller8x_SimplifiedCartPole_DQN_NN(input_dim=2, output_dim=2):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=np.random.randint(1000))

    inputs = tf.keras.Input(shape=(input_dim,))
    hidden1 = tf.keras.layers.Dense(25, activation = 'relu', kernel_initializer=initializer, name='hidden1')(inputs)
    hidden2 = tf.keras.layers.Dense(25, activation = 'relu', kernel_initializer=initializer, name='hidden2')(hidden1)
    outputs = tf.keras.layers.Dense(output_dim, activation = 'relu', kernel_initializer=initializer, name='output')(hidden2)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return input_dim, output_dim, model

def SimplifiedCartPole_SafetyMonitor_NN(input_dim = 2, output_dim = 2):
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(100, activation = tf.keras.activations.linear, kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(100, activation = tf.keras.activations.sigmoid, kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(100, activation = tf.keras.activations.linear, kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(output_dim, activation = tf.keras.activations.tanh, kernel_initializer=initializer, name='output'),
    ])

    return input_dim, output_dim, model


def SingleIntersection_DQN_NN():
    input_dim = 1 + 8 + 8
    output_dim = 4
    initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(500, activation = tf.keras.activations.relu, kernel_initializer=initializer),
        tf.keras.layers.Dense(500, activation = tf.keras.activations.relu, kernel_initializer=initializer),
        tf.keras.layers.Dense(500, activation = tf.keras.activations.relu, kernel_initializer=initializer),
        tf.keras.layers.Dense(output_dim, activation = tf.keras.activations.linear, kernel_initializer=initializer, name='output'),
    ])

    return input_dim, output_dim, model

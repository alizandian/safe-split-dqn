from models.AgentSafeDQNSplit import AgentSafeDQNSplit
from models.AgentDQN import AgentDQN
from environments.FixedCartPoleEnv import FixedCartPoleEnv
from experiment.nn_config import *
from matplotlib import pyplot as plt

from classes.grid import Grid

MAX_EPISODE = 1001
VISUALISATION = True
PLOT_INTERVAL = 20
normalizers=[2.5, 0.5]
denormalizers=[0.40, 2.0]

def normalize(state):
    state[0] = state[0] * normalizers[0]
    state[1] = state[1] * normalizers[1]
    return state

def denormalize(state):
    state[0] = state[0] * denormalizers[0]
    state[1] = state[1] * denormalizers[1]
    return state

def DQN_experiment():
    env = FixedCartPoleEnv()
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_DQN_NN()
    i_dim, o_dim, ES_DQN_nn = Smaller8x_SimplifiedCartPole_DQN_NN()
    agent = AgentSafeDQNSplit(i_dim, o_dim, DQN_nn, ES_DQN_nn)

    rewards = []

    for i in range(0, MAX_EPISODE):
        state = normalize(env.reset())
        episode_reward = 0
        if VISUALISATION: env.render()

        while True:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = normalize(next_state)
            trans = [state, action, reward, next_state, done]
            agent.add_transition(trans)
            episode_reward += reward
            state = next_state
            if VISUALISATION: env.render()
            if done: 
                agent.train()
                break

        if i % PLOT_INTERVAL == 0 and i != 0:
            values = [[0]*10 for i in range(10)]
            for degree in range(10):
                for velocity in range(10):
                    next_state[0] = ((degree - 5) / 5) 
                    next_state[1] = ((velocity - 5) / 5)
                    s = np.stack([next_state])
                    v = agent.estimator_dqn.get_model(next_state)[1].predict(s)
                    #v = agent.dqn.Q_target.predict(s)
                    values[degree][9-velocity] = np.max(v)

            plt.imshow(values, cmap='hot', interpolation='bicubic')
            plt.show()

        rewards.append(episode_reward) 
        print("Episode {0}/{1} -- reward {2}".format(i+1, MAX_EPISODE, episode_reward)) 


def SafeDQN_experiment():
    pass

if __name__ == "__main__":
    # DQN_experiment()
    # print("experiment done")
    s = Grid()
    s.grid_to_graph()
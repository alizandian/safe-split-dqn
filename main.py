from environments.RoverEnv import RoverEnv
from models.AgentSafeDQNSplit import AgentSafeDQNSplit
from models.AgentIterativeSafetyGraph import AgentIterativeSafetyGraph
from experiment.nn_config import *
from matplotlib import pyplot as plt
from typing import Dict, Tuple


MAX_EPISODE = 5
VISUALISATION = True
PLOT_INTERVAL = 2
plot_values: Dict[str, Dict[int, Tuple[list, float]]] = {} # values and accurace (tuple) of each episode (second dict) of each experiment (first dict).
# for FixedCartPole
# normalizers=[2.5, 0.5]
# denormalizers=[0.40, 2.0]
normalizers=[0.01, 0.01]
denormalizers=[100, 100]

def normalize(state):
    state[0] = state[0] * normalizers[0]
    state[1] = state[1] * normalizers[1]
    return state

def denormalize(state):
    state[0] = state[0] * denormalizers[0]
    state[1] = state[1] * denormalizers[1]
    return state

def experiment_AgentSafeDQNSplit(actions_xp = None, dimentions = (8,8)):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_DQN_NN(2,4)
    i_dim, o_dim, ES_DQN_nn = Smaller8x_SimplifiedCartPole_DQN_NN(2,4)
    agent = AgentSafeDQNSplit(i_dim, o_dim, DQN_nn, ES_DQN_nn, dimentions=dimentions)
    actions, rewards = run_experiment("split", actions_xp, agent, env)
    return actions

def experiment_AgentIterativeSafetyGraph(actions_xp = None, dimentions = (8,8)):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_DQN_NN(2,4)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, dimentions=dimentions)
    actions, rewards = run_experiment("iterative", actions_xp, agent, env)
    return actions

def run_experiment(experiment_name, actions_xp, agent, env):
    print(f"Running Experimen {experiment_name}") 
    action_index = 0
    actions = []
    rewards = []
    for i in range(0, MAX_EPISODE):
        state = normalize(env.reset())
        episode_reward = 0

        while True:
            action = None
            if actions_xp != None and action_index < len(actions_xp):
                action = actions_xp[action_index]
            else:
                action = agent.get_action(state)
                actions.append(action)
            action_index += 1
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

        if i % PLOT_INTERVAL == 0 and i != 0: record(experiment_name, i, agent, env, next_state)

        rewards.append(episode_reward) 
        print("Episode {0}/{1} -- reward {2}".format(i+1, MAX_EPISODE, episode_reward)) 
    return actions, rewards

def record(experiment, episode, agent, env, state):
    reso = 10
    error = 0
    values = [[0]*reso for i in range(reso)]
    for x in range(reso):
        for y in range(reso):
            state[0] = ((x - reso/2) / reso/2) 
            state[1] = ((y - reso/2) / reso/2)
            s = np.stack([state])
            v = agent.dqn.Q_target.predict(s)
            value = np.max(v)
            values[x][reso-1-y] = value
            violation = env.check_violation_solution(denormalize(state))
            value_estimation = 1 if value < 0.1 else 0
            if violation != value_estimation: error += 1

    accuracy = 100 - (error / reso * reso)
    if experiment not in plot_values: plot_values[experiment] = {}
    plot_values[experiment][episode] = (values, accuracy)

def plot():
    if len(plot_values.keys()) == 0:
        for experiment in plot_values.keys():
            for episode, (values, accuracy) in plot_values[experiment].items():
                plt.imshow(values, cmap='hot', interpolation='bicubic')
                plt.legend()
                plt.show()
    else:
        episodes = {}
        accuracies = {}
        for experiment in plot_values.keys():
            episodes[experiment] = []
            accuracies[experiment] = []
            for episode, (values, accuracy) in plot_values[experiment].items():
                episodes[experiment].append(episode)
                accuracies[experiment].append(accuracy)
        for experiment in plot_values.keys():
            plt.plot(episodes[experiment], accuracies[experiment], label = experiment, linestyle="-.")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    actions = experiment_AgentIterativeSafetyGraph()
    experiment_AgentSafeDQNSplit(actions_xp=actions)
    plot()
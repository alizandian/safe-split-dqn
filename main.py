from environments.RoverEnv import RoverEnv
from models.AgentSafeDQNSplit import AgentSafeDQNSplit
from models.AgentIterativeSafetyGraph import AgentIterativeSafetyGraph
from experiment.nn_config import *
from matplotlib import pyplot as plt
from typing import Dict, Tuple


MAX_EPISODE = 101
VISUALISATION = True
PLOT_INTERVAL = 20
plot_values: Dict[str, Dict[int, Tuple[list, float]]] = {} # values and accurace (tuple) of each episode (second dict) of each experiment (first dict).

def experiment_AgentSafeDQNSplit(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_DQN_NN(2,4)
    i_dim, o_dim, ES_DQN_nn = Smaller8x_SimplifiedCartPole_DQN_NN(2,4)
    agent = AgentSafeDQNSplit(i_dim, o_dim, DQN_nn, ES_DQN_nn, 1)
    actions, rewards = run_experiment("base", predefined_actions, agent, env)
    return actions

def experiment_AgentIterativeSafetyGraph(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_DQN_NN(2,4)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, 5)
    actions, rewards = run_experiment("iterative", predefined_actions, agent, env)
    return actions

def run_experiment(experiment_name, predefined_actions, agent, env):
    print(f"Running Experimen {experiment_name}") 
    action_index = 0
    actions = []
    rewards = []
    for i in range(0, MAX_EPISODE):
        state = env.reset()
        episode_reward = 0

        while True:
            action = None
            if predefined_actions != None and action_index < len(predefined_actions):
                action = predefined_actions[action_index]
            else:
                action = agent.get_action(state)
                actions.append(action)
            action_index += 1
            next_state, reward, done, _ = env.step(action)
            trans = [state, action, reward, next_state, done]
            agent.add_transition(trans)
            episode_reward += reward
            state = next_state
            if VISUALISATION: env.render()
            if done: 
                agent.train()
                break

        if i % PLOT_INTERVAL == 0 and i != 0: 
            record(experiment_name, i, agent, env, next_state)
            plot(only_updates=True, only_accuracy=True)

        rewards.append(episode_reward) 
        print("Episode {0}/{1} -- reward {2}".format(i+1, MAX_EPISODE, episode_reward)) 
    return actions, rewards

def record(experiment, episode, agent, env, state):
    values, accuracy = agent.dqn.get_snapshot(10, env.check_violation_solution, 5)
    if experiment not in plot_values: plot_values[experiment] = {}
    plot_values[experiment][episode] = (values, accuracy)

def plot_output(values, accuracy, only_accuracy=False):
    if only_accuracy:
        print(f"accuracy: {accuracy}")
        return

    plt.imshow(values, cmap='hot', interpolation='bicubic')
    plt.legend()
    plt.show()

def plot(only_updates=False, only_accuracy=False): 
    if only_updates:
        last_experiment = list(plot_values.values())[-1]
        values, accuracy = list(last_experiment.values())[-1]
        plot_output(values, accuracy, only_accuracy)
    else:
        episodes = {}
        accuracies = {}
        for experiment in plot_values.keys():
            episodes[experiment] = []
            accuracies[experiment] = []
            for episode, (values, accuracy) in plot_values[experiment].items():
                episodes[experiment].append(episode)
                accuracies[experiment].append(accuracy)
                plot_output(values, accuracy, only_accuracy)
        for experiment in plot_values.keys():
            plt.plot(episodes[experiment], accuracies[experiment], label = experiment, linestyle="-.")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    actions = experiment_AgentIterativeSafetyGraph()
    actions = experiment_AgentSafeDQNSplit(predefined_actions=actions)
    plot()
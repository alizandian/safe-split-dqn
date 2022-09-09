from environments.FixedCartPoleEnv import FixedCartPoleEnv
from environments.RoverEnv import RoverEnv
from models.AgentSafeDQN import AgentSafeDQN
from models.AgentSafeDQNSplit import AgentSafeDQNSplit
from models.AgentIterativeSafetyGraph import AgentIterativeSafetyGraph
from experiment.nn_config import *
from matplotlib import pyplot as plt
from typing import Dict, Tuple
import time


MAX_EPISODE = 401
VISUALISATION = True
PLOT_INTERVAL = 20
ARTIFICIAL_DELAY = -0.1
plot_values: Dict[str, Dict[int, Tuple[list, float]]] = {} # values, env and accuracy (tuple) of each episode (second dict) of each experiment (first dict).

def experiment_base(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    _, _, MON_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentSafeDQN(i_dim, o_dim, DQN_nn, MON_nn)
    actions, rewards = run_experiment("base", predefined_actions, agent, env)
    return actions

def experiment_refined_experiences(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, 15)
    actions, rewards = run_experiment("refined", predefined_actions, agent, env)
    return actions

def experiment_refined_experiences_fixed_cartPole(predefined_actions = None):
    env = FixedCartPoleEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,2)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, 10)
    actions, rewards = run_experiment("refined", predefined_actions, agent, env)
    return actions

def run_experiment(experiment_name, predefined_actions, agent, env):
    print(f"Running Experiment {experiment_name}") 
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
            next_state, reward, violation, _ = env.step(action)
            experience = [state, action, reward, next_state, violation]
            agent.add_experience(experience)
            episode_reward += reward
            state = next_state
            if ARTIFICIAL_DELAY >= 0: time.sleep(ARTIFICIAL_DELAY)
            if VISUALISATION: env.render()
            if violation: 
                agent.train()
                break


        if i % PLOT_INTERVAL == 0 and i != 0: 
            record(experiment_name, i, agent, env, next_state)
            if hasattr(agent, "safety_graph"):
                agent.safety_graph.visualize()
            plot(only_updates=True, only_accuracy=False)

        rewards.append(episode_reward) 
        print("Episode {0}/{1} -- reward {2}".format(i+1, MAX_EPISODE, episode_reward)) 
    return actions, rewards

def record(experiment, episode, agent, env, state):
    reso = 15
    c = agent.input_dim
    values = agent.dqn.get_snapshot(reso)
    accuracy = env.test_agent_accuracy(agent)
    if experiment not in plot_values: plot_values[experiment] = {}

    rvs = []
    for i in range(c):
        rvs.append([[0]*reso for i in range(reso)])

    for y in range(reso):
        for x in range(reso):
            for i in range(c):
                rvs[i][reso-y-1][x] = values[reso-y-1][x][i]
                rvs[i] = np.interp(rvs[i], [np.min(rvs[i]), np.max(rvs[i])], [-1, +1])

    plot_values[experiment][episode] = (rvs, env, accuracy)

def plot_output(values, accuracy, only_accuracy=False):
    if only_accuracy:
        print(f"accuracy: {accuracy}")
        return

    print(f"accuracy: {accuracy}")
    plt.imshow(values, cmap='hot', interpolation='bicubic')
    plt.legend()
    plt.colorbar()
    plt.show()

def plot_output_4(experiment_name, values, accuracy, only_accuracy, env, cmap='hot', interpolation='bicubic'):
    if only_accuracy:
        print(f"accuracy: {accuracy}")
        return

    print(f"accuracy: {accuracy}")
    _, ax = plt.subplots(len(env.action_names))


    for i, name in enumerate(env.action_names):
        ax[i].imshow(values[i], cmap='hot')
        ax[i].set_title(name)

    plt.title(experiment_name)
    plt.show() 

def plot(only_updates=False, only_accuracy=False): 
    if only_updates:
        last_experiment = list(plot_values.values())[-1]
        values, env, accuracy = list(last_experiment.values())[-1]
        plot_output_4(list(plot_values.keys())[-1], values, accuracy, only_accuracy, env)
    else:
        episodes = {}
        accuracies = {}
        for experiment in plot_values.keys():
            episodes[experiment] = []
            accuracies[experiment] = []
            for episode, (values, env, accuracy) in plot_values[experiment].items():
                episodes[experiment].append(episode)
                accuracies[experiment].append(accuracy)
                plot_output_4(experiment, values, accuracy, only_accuracy, env)
        for experiment in plot_values.keys():
            plt.plot(episodes[experiment], accuracies[experiment], label = experiment, linestyle="-.")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    actions = experiment_refined_experiences_fixed_cartPole()
    actions = experiment_base(predefined_actions=actions)
    plot()
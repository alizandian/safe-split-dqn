import gym
from environments.FixedCartPoleEnv import FixedCartPoleEnv
from environments.RoverEnv import RoverEnv
from environments.AtariEnv import AtariEnv
from models.AgentSafeDQN import AgentSafeDQN
from models.AgentDQN import AgentDQN
from models.AgentIterativeSafetyGraph import AgentIterativeSafetyGraph
from experiment.nn_config import *
from matplotlib import pyplot as plt
from typing import Dict, Tuple, List
import time


MAX_EPISODE = 200
VISUALISATION = False
PLOT_INTERVAL = 20
ARTIFICIAL_DELAY = -0.1
plot_values: Dict[str, Dict[int, Tuple[list, gym.Env, float]]] = {} # values, env and accuracy (tuple) of each episode (second dict) of each experiment (first dict).
episode_info: Dict[str, list] = {}

def experiment_rover_base(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    _, _, MON_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentSafeDQN(i_dim, o_dim, DQN_nn, MON_nn)
    actions, rewards = run_experiment("base", predefined_actions, agent, env)
    return actions

def experiment_rover_vanilla(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentDQN(i_dim, o_dim, DQN_nn)
    actions, rewards = run_experiment("vanilla", predefined_actions, agent, env)
    return actions

def experiment_rover_refined(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, (10, 10))
    actions, rewards = run_experiment("refined", predefined_actions, agent, env)
    return actions

def experiment_rover_refined_no_feedback(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, (10, 10), feedback=False)
    actions, rewards = run_experiment("no-feedback", predefined_actions, agent, env)
    return actions

def experiment_pole_refined(predefined_actions = None):
    env = FixedCartPoleEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,2)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, (16, 16))
    actions, rewards = run_experiment("refined", predefined_actions, agent, env)
    return actions

def experiment_pole_base(predefined_actions = None):
    env = FixedCartPoleEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,2)
    _, _, MON_nn = SimplifiedCartPole_SafetyMonitor_NN(2,2)
    agent = AgentSafeDQN(i_dim, o_dim, DQN_nn, MON_nn)
    actions, rewards = run_experiment("base", predefined_actions, agent, env)
    return actions

def experiment_pole_vanilla(predefined_actions = None):
    env = FixedCartPoleEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,2)
    agent = AgentDQN(i_dim, o_dim, DQN_nn)
    actions, rewards = run_experiment("vanilla", predefined_actions, agent, env)
    return actions

def experiment_refined_experiences_atari(predefined_actions = None):
    env = AtariEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,3)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, (20, 20))
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

        e = 0
        maxE = 100
        while True:
            e += 1
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
                env.reset()
                break
            if e >= maxE:
                env.reset()
                break

        if experiment_name not in episode_info: episode_info[experiment_name] = {}
        episode_info[experiment_name][i] = e

        if i % PLOT_INTERVAL == 0: 
            record(experiment_name, i, agent, env)
            #if hasattr(agent, "safety_graph"):
                #agent.safety_graph.visualize()
            #plot(only_updates=True, only_accuracy=False)

        rewards.append(episode_reward) 
        print("Episode {0}/{1} -- reward {2}".format(i+1, MAX_EPISODE, episode_reward)) 
    return actions, rewards

def record(experiment, episode, agent, env):
    reso = (15, 15)
    c = agent.output_dim
    values = agent.dqn.get_snapshot(reso)
    accuracy = env.test_agent_accuracy(agent)
    if experiment not in plot_values: plot_values[experiment] = {}

    rvs = []
    for _ in range(c):
        rvs.append([[0]*reso[0] for _ in range(reso[1])])

    for y in range(reso[1]):
        for x in range(reso[0]):
            for i in range(c):
                rvs[i][reso[1]-y-1][x] = values[reso[1]-y-1][x][i]

    for i in range(c):
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

    plt.show() 

def plot_episode_info():
    for experiment in episode_info.keys():
        plt.plot(episode_info[experiment].keys(), episode_info[experiment].values(), label = experiment, linestyle="-.")
    plt.legend()
    plt.show()

def plot_episode_info_accuracy():
    for experiment in episode_info.keys():
        accuracies = []
        ap = {}
        for c in episode_info[experiment].values():
            accuracies.append(c/100)
        for i in range(0, len(accuracies), PLOT_INTERVAL):
            l = accuracies[i:i+PLOT_INTERVAL]
            a = sum(l)/len(l)
            ap[i] = a

        plt.plot(ap.keys(), ap.values(), label = experiment, linestyle="-.")
    plt.legend()
    plt.show()


def plot(only_updates=False, only_accuracy=False, only_comparision=False): 
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
                if not only_comparision:plot_output_4(experiment, values, accuracy, only_accuracy, env)
        for experiment in plot_values.keys():
            plt.plot(episodes[experiment], accuracies[experiment], label = experiment, linestyle="-.")
        plt.legend()
        plt.show()

if __name__ == "__main__":

    experiment_rover_refined()
    experiment_rover_refined_no_feedback()

    # experiment_pole_refined()
    # experiment_pole_base()
    # experiment_pole_vanilla()

    plot(only_comparision=True)
    plot_episode_info()
    plot_episode_info_accuracy()
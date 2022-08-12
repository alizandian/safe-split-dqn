from environments.RoverEnv import RoverEnv
from models.AgentSafeDQNSplit import AgentSafeDQNSplit
from models.AgentIterativeSafetyGraph import AgentIterativeSafetyGraph
from experiment.nn_config import *
from matplotlib import pyplot as plt
from typing import Dict, Tuple
import time


MAX_EPISODE = 401
VISUALISATION = True
PLOT_INTERVAL = 25
ARTIFICIAL_DELAY = -0.1
plot_values: Dict[str, Dict[int, Tuple[list, float]]] = {} # values and accurace (tuple) of each episode (second dict) of each experiment (first dict).

def experiment_base(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, 7, do_enhance_transitions=False)
    actions, rewards = run_experiment("base", predefined_actions, agent, env)
    return actions

def experiment_enhanced_transitions(predefined_actions = None):
    env = RoverEnv(seed=100)
    i_dim, o_dim, DQN_nn = SimplifiedCartPole_SafetyMonitor_NN(2,4)
    agent = AgentIterativeSafetyGraph(i_dim, o_dim, DQN_nn, 7)
    actions, rewards = run_experiment("enhanced", predefined_actions, agent, env)
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
            if ARTIFICIAL_DELAY >= 0: time.sleep(ARTIFICIAL_DELAY)
            if VISUALISATION: env.render()
            if done: 
                agent.train()
                break


        if i % PLOT_INTERVAL == 0 and i != 0: 
            record(experiment_name, i, agent, env, next_state)
            agent.safety_graph.visualize()
            plot(only_updates=True, only_accuracy=True)

        rewards.append(episode_reward) 
        print("Episode {0}/{1} -- reward {2}".format(i+1, MAX_EPISODE, episode_reward)) 
    return actions, rewards

def record(experiment, episode, agent, env, state):
    reso = 15
    values, accuracy = agent.dqn.get_snapshot(reso, env.check_violation_solution, 0)
    if experiment not in plot_values: plot_values[experiment] = {}

    values_up = [[0]*reso for i in range(reso)]
    values_down = [[0]*reso for i in range(reso)]
    values_right = [[0]*reso for i in range(reso)]
    values_left = [[0]*reso for i in range(reso)]
    values_min = [[0]*reso for i in range(reso)]
    values_max = [[0]*reso for i in range(reso)]
    values_avg = [[0]*reso for i in range(reso)]
    for y in range(reso):
        for x in range(reso):
            values_up[reso-y-1][x] = values[reso-y-1][x][0]
            values_down[reso-y-1][x] = values[reso-y-1][x][1]
            values_right[reso-y-1][x] = values[reso-y-1][x][2]
            values_left[reso-y-1][x] = values[reso-y-1][x][3]
            values_min[reso-y-1][x] = np.min(values[reso-y-1][x])
            values_max[reso-y-1][x] = np.max(values[reso-y-1][x])
            values_avg[reso-y-1][x] = np.average(values[reso-y-1][x])

    values_up = np.interp(values_up, [np.min(values_up), np.max(values_up)], [-1, +1])
    values_down = np.interp(values_down, [np.min(values_down), np.max(values_down)], [-1, +1])
    values_right = np.interp(values_right, [np.min(values_right), np.max(values_right)], [-1, +1])
    values_left = np.interp(values_left, [np.min(values_left), np.max(values_left)], [-1, +1])
    plot_values[experiment][episode] = ((values_up, values_down, values_right, values_left), accuracy)
    plot()

def plot_output(values, accuracy, only_accuracy=False):
    if only_accuracy:
        print(f"accuracy: {accuracy}")
        return

    plt.imshow(values, cmap='hot', interpolation='bicubic')
    plt.legend()
    plt.colorbar()
    plt.show()

def plot_output_4(values, cmap='hot', interpolation='bicubic'):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(values[0], cmap='hot', interpolation='sinc')
    ax[1, 0].imshow(values[1], cmap='hot', interpolation='sinc')
    ax[0, 1].imshow(values[2], cmap='hot', interpolation='sinc')
    ax[1, 1].imshow(values[3], cmap='hot', interpolation='sinc')
    plt.show() 

def plot(only_updates=False, only_accuracy=False): 
    if only_updates:
        last_experiment = list(plot_values.values())[-1]
        values, accuracy = list(last_experiment.values())[-1]
        plot_output_4(values, accuracy, only_accuracy)
    else:
        episodes = {}
        accuracies = {}
        for experiment in plot_values.keys():
            episodes[experiment] = []
            accuracies[experiment] = []
            for episode, (values, accuracy) in plot_values[experiment].items():
                episodes[experiment].append(episode)
                accuracies[experiment].append(accuracy)
                plot_output_4(values, accuracy, only_accuracy)
        for experiment in plot_values.keys():
            plt.plot(episodes[experiment], accuracies[experiment], label = experiment, linestyle="-.")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    actions = experiment_enhanced_transitions()
    actions = experiment_base(predefined_actions=actions)
    plot()
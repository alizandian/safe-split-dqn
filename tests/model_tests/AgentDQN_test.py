from safeDQN.models.AgentDQN import AgentDQN
from experiments.nn_config import *

if __name__ == '__main__':
    idim, odim, nn = SimplifiedCartPole_DQN_NN()
    agent = AgentDQN(idim, odim, nn, verbose=True)
    print("agent DQN instantiated")


Safe Reinforcement Learning with Iterative Refinement of Variable Domains
This repository contains the code project for the Master's thesis titled "Safe Reinforcement Learning with Iterative Refinement of Variable Domains." The project introduces a novel approach to enhance safety in Reinforcement Learning (RL) systems, particularly in environments with unknown or hidden unsafe states.

Abstract
In systems interacting with physical environments, failures are often unavoidable due to uncertainties and the inability to perfectly capture real-world phenomena in computer models. This is especially critical in safety-critical systems where failures can lead to catastrophic consequences. This work proposes a module that leverages Reinforcement Learning to observe system interactions with the environment and iteratively gain knowledge to restrict the system from failing. The acquired knowledge is represented as a model that classifies system variable domains into safe and unsafe regions, becoming more accurate with continued interaction. Ultimately, our method uses this knowledge model to prevent systems from entering unsafe domains and consequently failing.

Features
This project introduces significant improvements over existing approaches, particularly in handling safety in Reinforcement Learning:

Enhanced Accuracy: Achieves improved accuracy in detecting and avoiding unsafe states compared to preceding works and conventional RL methods.

Reduced Failures: Significantly lowers the number of failures during the early stages of learning, which is crucial for safety-critical applications.

Knowledge Extraction: Enables the extraction of learned safety knowledge in a formal, mathematical language, making the system more transparent and modular.

Generalization: Incorporates a generalization mechanism within the Safety Graph module, allowing the system to learn about continuous and infinite state spaces from a finite number of experiences, drastically speeding up the learning process.

Robustness: Overcomes limitations of previous methods, such as unstable accuracy and unextractable learned knowledge.

Methodology
The proposed system is composed of two primary modules:

Safety Graph Module:

Maintains a graph model that classifies the state space into Safe, Unsafe, and Unsure regions.

Iteratively updates and refines this graph model based on trajectories (experience data) from the agent's interactions with the environment.

Refines raw experience data to improve data quality and uniformity for training the RL module.

Utilizes the Reinforcement Learning module for generalization (feedback operation) to estimate the safety of unexplored states.

Allows for the export of the learned safety knowledge at any time in a formal language (e.g., mathematical formulas).

Reinforcement Learning Module:

Employs a modified Q-Learning algorithm. Instead of maximizing future rewards, it minimizes them, effectively prioritizing the avoidance of unsafe states (which are assigned a minimum reward of -1).

Its Neural Network is trained by the Safety Graph module with refined experience data, learning to predict the current safety model over the state space.

This modular design allows for the replacement of the core RL algorithm with newer versions while maintaining the safety framework.

Case Studies and Results
The effectiveness of the proposed approach has been demonstrated through five comprehensive case studies:

Case Study 1 & 2 (Benchmarking): Showed consistent accuracy gains in both a custom "Rover" environment and the "Cartpole" environment, outperforming previous work and vanilla Reinforcement Learning agents.

Case Study 3 (Failures Count Improvement): Highlighted that the proposed approach makes the fewest failures in the early learning stages, emphasizing its efficiency in safety-critical settings.

Case Study 4 (Generalization Proof of Concept): Empirically proved that the generalization mechanism significantly speeds up the learning process (by almost 200%) without compromising accuracy.

Case Study 5 (Knowledge Extraction): Visually demonstrated the Safety Graph module's ability to learn and represent hidden unsafe states, even starting with zero prior knowledge, and export this knowledge into a comprehensible format.

These results collectively confirm the drastic accuracy improvements, reduced training episodes, fewer system failures, and the successful extraction of learned unsafe states.

Implementation Details
The project leverages established libraries and frameworks for its implementation:

Environments: Utilizes OpenAI Gym for creating and standardizing reinforcement learning environments.

Reinforcement Learning Agents: Implemented using TensorFlow and its Keras API for Neural Network configurations.

Program Structure: Designed to simulate experiments, allowing for benchmarking of different agent and environment configurations.

Installation and Usage
To set up and run the project, follow these steps:

Clone the repository:

git clone https://github.com/alizandian/safe-split-dqn.git
cd safe-split-dqn

Create a virtual environment (recommended):

python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install dependencies:
It is assumed that a requirements.txt file exists in the repository.

pip install -r requirements.txt

Run experiments:
The thesis mentions a program structure overview (Figure 11) with a Run Experiments entry point. You would typically run the main script to start simulations.

# Example: This command might vary based on your project's main entry point
python main.py

Refer to the project's internal documentation or scripts for specific commands to run different case studies or experiments.

Future Work
While the proposed approach shows significant effectiveness, particularly in guiding Neural Networks, the current graph model might face scalability issues with environments involving thousands or more system variables. Future work could focus on developing models with diminishing complexity concerning the number of system variables, enabling deployment in environments with extremely large state spaces.

License
This project is open-source. Please consider adding a LICENSE file to specify the terms under which others can use, modify, and distribute your work (e.g., MIT, Apache 2.0, GPL).

Contact
For any questions or inquiries, please refer to the thesis author or the repository maintainers.

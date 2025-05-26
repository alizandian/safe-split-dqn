# Safe Reinforcement Learning with Iterative Refinement of Variable Domains

This repository contains the official implementation for the Master's thesis titled **"Safe Reinforcement Learning with Iterative Refinement of Variable Domains."** The project presents a novel method to improve safety in Reinforcement Learning (RL), particularly in environments containing unknown or hidden unsafe states.

---

## 📜 Abstract

In physical systems, failure is often unavoidable due to uncertainty and incomplete modeling. This is especially dangerous in safety-critical environments, where failure can result in catastrophic consequences. This work proposes a modular RL-based approach that incrementally learns to avoid unsafe states by refining its understanding of variable domains. The system develops a safety model that becomes more accurate through interaction, classifying states as **Safe**, **Unsafe**, or **Unsure**. This knowledge is used to prevent the agent from entering unsafe regions, significantly reducing failure rates during learning.

---

## ✨ Features

- **Enhanced Accuracy**: Improves accuracy in detecting and avoiding unsafe states over baseline RL methods.
- **Reduced Failures**: Decreases the number of critical failures during early learning stages.
- **Knowledge Extraction**: Converts learned safety knowledge into interpretable mathematical formulas.
- **Generalization**: Learns from finite experiences and generalizes to infinite/continuous state spaces.
- **Robustness**: Solves instability issues and improves explainability over previous approaches.

---

## 🧠 Methodology

The system is composed of two primary modules:

### 1. Safety Graph Module

- Classifies states as **Safe**, **Unsafe**, or **Unsure** using a graph-based representation.
- Refines the safety model using trajectory data from agent-environment interactions.
- Enhances training data quality for the RL module.
- Uses feedback from the RL module to estimate safety of unexplored states.
- Exports learned knowledge as formal expressions (e.g., logical/mathematical constraints).

### 2. Reinforcement Learning Module

- Utilizes a modified **Q-Learning** algorithm that **minimizes rewards**, aiming to avoid unsafe states (reward = -1).
- Neural Network is trained on refined safety data from the graph module.
- The RL module remains swappable, allowing future integration of newer algorithms while preserving the safety layer.

---

## 📊 Case Studies and Results

### ✅ Benchmarking (Case Study 1 & 2)
- Tested on custom **Rover** and classic **Cartpole** environments.
- Demonstrated superior accuracy and safety over baseline RL.

### 🛡️ Failure Reduction (Case Study 3)
- Achieved the **lowest failure rates** during early training.

### 🧩 Generalization (Case Study 4)
- Accelerated learning by ~200% while maintaining high safety accuracy.

### 📤 Knowledge Extraction (Case Study 5)
- Visualized and exported learned unsafe states even with zero prior knowledge.

---

## 🛠️ Implementation Details

- **Environments**: [OpenAI Gym](https://www.gymlibrary.dev/)
- **RL Framework**: [TensorFlow](https://www.tensorflow.org/) + Keras API
- **Simulation Structure**: Designed to support benchmarking across multiple agents/environments.

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/alizandian/safe-split-dqn.git
cd safe-split-dqn
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Make sure `requirements.txt` exists in the root directory.

```bash
pip install -r requirements.txt
```

### 4. Run Experiments

Run the main experiment script (update with your actual entry point if needed):

```bash
python main.py
```

> Refer to project scripts or documentation to run specific case studies or configurations.

---

## 🔭 Future Work

- **Scalability Improvements**: The current graph model may face limitations in high-dimensional spaces (e.g., thousands of variables).
- **Complexity Reduction**: Future research could focus on graph representations that scale sub-linearly with the number of state variables.

---

## ⚖️ License

> _No license file has been provided yet._  
Please consider adding an open-source license (e.g., MIT, Apache 2.0) to clarify usage rights.

---

## 📬 Contact

For questions or collaboration inquiries, please reach out to the thesis author or repository maintainers.  
> You can also open an issue on GitHub for technical questions or bug reports.

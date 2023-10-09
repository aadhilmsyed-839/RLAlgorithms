# Reinforcement Learning Algorithm Testing and Visualization

This repository contains Python code for testing and visualizing the performance of various reinforcement learning (RL) algorithms using the Stable Baselines3 library. It also includes data analysis and visualization tools to assess the training results of these algorithms.

## Prerequisites

Before running the code, make sure you have the following libraries installed:

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3): A library for RL algorithms.
- [Gymnasium](https://github.com/DLR-RM/gymnasium): A collection of environments for testing RL algorithms.
- [NumPy](https://numpy.org/): For numerical operations.
- [Pandas](https://pandas.pydata.org/): For data analysis.
- [Matplotlib](https://matplotlib.org/): For data visualization.

You can install these libraries using pip:

```bash
pip install stable-baselines3 gymnasium numpy pandas matplotlib
```

## Usage

The code in this repository allows you to:

1. Test multiple RL algorithms (e.g., A2C, DQN, PPO, TRPO) on a specific Gym environment (e.g., LunarLander-v2).
2. Train each algorithm for a specified number of timesteps and record the results, including total rewards and execution time.
3. Visualize the training progress and performance of each algorithm using matplotlib.

To customize the testing and training parameters, modify the following variables in the script:

- `env_name`: Name of the Gym environment to run.
- `train_steps`: Number of timesteps for training.
- `rl_algs`: List of RL algorithms to test.

Run the script to execute the tests and generate visualizations for each algorithm.

## Results

The code generates the following results:

- Training progress logs saved in the `./{env_name}/{alg.__name__}/log/` directory.
- Video recordings of trained agents in the `./{env_name}/{alg.__name__}/videos/` directory.
- Data visualization of training performance in a matplotlib plot.

## License

This code is provided under the [MIT License](LICENSE).

## Acknowledgments

- The code uses the Stable Baselines3 library for RL algorithms.
- Gymnasium provides the RL environments for testing.

Feel free to modify and extend this code for your specific RL experiments and analysis.

For more information about Stable Baselines3 and Gymnasium, please refer to their respective GitHub repositories linked above.
```

You can copy the entire content above and save it as a README.md file in your GitHub repository.

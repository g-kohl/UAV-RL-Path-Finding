# Project Description

This project contains a custom [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) environment designed to train UAV's (Unmanned Aerial Vehicles) to find paths to a target while avoiding static and mobile obstacles.
The environment is represented as a 2D grid, where the UAV, the target, and the obstacles occupy individual cells.
Although simplified, you can map the model's input and output to more realistic UAV simulators like [Gazebo](https://gazebosim.org/).

# Installation

Before anything, consider creating a Python virtual environment to install the dependencies locally.
You can learn about it in [this website](https://www.w3schools.com/python/python_virtualenv.asp).

Once your virtual environment is active, install the dependencies by running:

```sh
pip install -r requirements.txt
```

Alternatively, you can install everything manually via:

```sh
pip install stable-baselines3[extra]
```

Both methods will install all required packages.

# How to Use

## Training

To start the training, run:

```sh
python train.py --timesteps NUMBER_OF_TIMESTEPS --grid_size ROWS COLUMNS
```

- Replace *NUMBER_OF_TIMESTEPS* with the desired number of timesteps.
- Replace *ROWS* and *COLUMNS* with the grid dimensions.

You can also include two flags, `--static_obstacles` and/or `--mobile_obstacles`, to add obstacles to the training.

### Retraining

In order to retrain your model, you have the following options:
- **Continue training the same model:**
If the number of timesteps was insufficient and you don't want to change your training strategy or start from scratch, make sure the model file still exists in the [`models`](./models/) folder with the name `last_model`.
Then, simply run the training command again (this won't change the hyperparameters).

- **Transfer learning:**
If you trained your model in one specific scenario and want to adapt it to a new environment (while keeping what it has already learned), rename the file to `pretrained_model`.
When using a pre-trained model, the script will retrain it with a smaller learning rate and a higher exploration rate to encourage adaptation.

- **Start from scratch:**
If you want to train a completely new model, ensure there is no file named `last_model` in the folder.
In this case, the script will automatically create a fresh model and start training from the beginning.

## Evaluation

To evaluate a trained model, run:

```sh
python evaluate.py
```

As well as in training, you can specify some parameters by the command line:
`--episodes NUMBER_OF_EPISODES`, `--model MODEL_FILE_NAME`, `--algorithm DQN or PPO` and the three other parameters that were available for the training script (grid size, static obstacles and mobile obstacles).

If you specify a number of episodes higher then 10, the program assumes you just want to test the model and won't render any window.
Otherwise, the program assumes you want to see the behavior of the agent and will render a window.

## Visualizing Training Logs

To take a look on the model progress while training, run:

```sh
python visualize_logs.py
```

This will plot a graph showing the model's reward evolution during training.

# Developer Notes

If you want to build upon this project, here is a detailing of my implementation:
- The environment is a 2D grid.
- The UAV can move in eight directions (the eight surrounding cells).
- To simplify the neural network, the UAV has limited perception. Its observation space contains:
    1. A 5x5 window centered around itself
    2. Relative distance (ΔX and ΔY) to the target
- The distance is normalized based on the grid size, making the model agnostic to scenario scale (helping generalization).

## Environment Implementation
A custom stable-baselines3 environment needs to override some methods:
- `__init__()`: define the observation and action space
- `reset()`: reset the environment every episode
- `step()`: how a agent action modifies the environment

All other methods in the [`environment.py`](./environment.py) script are auxiliary.

## Obstacles and Maps
- Static obstacles: 20 predefined [`maps`](./maps/) are available.
For training, there are 3 levels of difficulty, each with 5 maps.
There is also 5 maps for testing.
All are 15x15 grids, so make sure your grid size matches these dimensions.
- Mobile obstacles: These are represented by singular cells moving randomly, simulating other UAV's.

## Training Algorithm
The training is based on the DQN algorithm implemented by Stable-Baselines3.
The hyperparameters can be changed edited in the [`train.py`](./train.py) script, as well as the callback for evaluation.

# References

This project was deeply inspired by [this article](https://ieeexplore.ieee.org/document/9564258).
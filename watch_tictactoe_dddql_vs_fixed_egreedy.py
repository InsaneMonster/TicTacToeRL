# Import packages

import tensorflow
import logging
import os

# Import usienarl

from usienarl import Config, LayerType, watch_experiment, command_line_parse
from usienarl.td_models import DuelingDeepQLearning
from usienarl.exploration_policies import EpsilonGreedyExplorationPolicy

# Import required src

from src.dddql_tictactoe_agent import DDDQLTicTacToeAgent
from src.tictactoe_experiment import TicTacToeExperiment
from src.tictactoe_environment_fixed import TicTacToeEnvironmentFixed, Player
from src.tictactoe_pass_through_interface import TicTacToePassThroughInterface

# Define utility functions to run the experiment


def _define_dddqn_model(config: Config) -> DuelingDeepQLearning:
    # Define attributes
    learning_rate: float = 0.000001
    discount_factor: float = 0.99
    buffer_capacity: int = 100000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    error_clip: bool = True
    # Return the model
    return DuelingDeepQLearning("model",
                                learning_rate, discount_factor,
                                buffer_capacity,
                                minimum_sample_probability, random_sample_trade_off,
                                importance_sampling_value, importance_sampling_value_increment,
                                config, error_clip)


def _define_epsilon_greedy_exploration_policy() -> EpsilonGreedyExplorationPolicy:
    # Define attributes
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 0.1
    exploration_rate_decay: float = 0.00002
    # Return the explorer
    return EpsilonGreedyExplorationPolicy(exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_epsilon_greedy_agent(model: DuelingDeepQLearning, exploration_policy: EpsilonGreedyExplorationPolicy) -> DDDQLTicTacToeAgent:
    # Define attributes
    weight_copy_step_interval: int = 100
    batch_size: int = 150
    # Return the agent
    return DDDQLTicTacToeAgent("dddqn_egreedy_agent", model, exploration_policy, weight_copy_step_interval, batch_size)


if __name__ == "__main__":
    # Parse the command line arguments
    checkpoint_path, iteration_number, cuda_devices, render = command_line_parse(True)
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Tic Tac Toe random environment:
    #       - success threshold to consider both the training completed and the experiment successful is around 95% of match won by the agent (depending on reward assigned)
    environment_name: str = 'TicTacToeRandom'
    # Generate Tic Tac Toe environments with fixed environment player and using the O player as the environment player with only low reward type
    environment_low_reward: TicTacToeEnvironmentFixed = TicTacToeEnvironmentFixed(environment_name, Player.o,
                                                                                  1.0, -0.1, 0.0)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [1024, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [1024, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [1024, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model: DuelingDeepQLearning = _define_dddqn_model(nn_config)
    # Define exploration policy
    epsilon_greedy_exploration_policy: EpsilonGreedyExplorationPolicy = _define_epsilon_greedy_exploration_policy()
    # Define agent
    dddqn_epsilon_greedy_agent: DDDQLTicTacToeAgent = _define_epsilon_greedy_agent(inner_model, epsilon_greedy_exploration_policy)
    # Define interface
    interface_low_reward: TicTacToePassThroughInterface = TicTacToePassThroughInterface(environment_low_reward)
    # Define experiment
    success_threshold: float = 0.65
    experiment_egreedy_low_reward: TicTacToeExperiment = TicTacToeExperiment("eg_experiment_low_reward", success_threshold,
                                                                             environment_low_reward,
                                                                             dddqn_epsilon_greedy_agent, interface_low_reward)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    episode_length_max: int = 20
    # Run epsilon greedy experiment for low reward
    watch_experiment(experiment_egreedy_low_reward,
                     testing_episodes, test_cycles,
                     episode_length_max,
                     render,
                     logger, checkpoint_path)



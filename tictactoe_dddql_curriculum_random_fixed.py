#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# TicTacToeRL project is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import tensorflow
import logging
import os

# Import usienarl

from usienarl import Config, LayerType, run_experiment, command_line_parse
from usienarl.td_models import DuelingDeepQLearning
from usienarl.exploration_policies import BoltzmannExplorationPolicy

# Import required src

from src.dddql_tictactoe_agent import DDDQLTicTacToeAgent
from src.tictactoe_experiment import TicTacToeExperiment
from src.tictactoe_environment_random import TicTacToeEnvironmentRandom, Player
from src.tictactoe_environment_fixed import TicTacToeEnvironmentFixed
from src.tictactoe_pass_through_interface import TicTacToePassThroughInterface

# Define utility functions to run the experiment


def _define_dddqn_model(config: Config, learning_rate: float) -> DuelingDeepQLearning:
    # Define attributes
    discount_factor: float = 0.99
    buffer_capacity: int = 100000
    minimum_sample_probability: float = 0.01
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 0.001
    error_clip: bool = True
    # Return the _model
    return DuelingDeepQLearning("model",
                                learning_rate, discount_factor,
                                buffer_capacity,
                                minimum_sample_probability, random_sample_trade_off,
                                importance_sampling_value, importance_sampling_value_increment,
                                config, error_clip)


def _define_boltzmann_exploration_policy(temperature_max: float, temperature_min: float) -> BoltzmannExplorationPolicy:
    # Define attributes
    temperature_decay: float = 0.00002
    # Return the explorer
    return BoltzmannExplorationPolicy(temperature_max, temperature_min, temperature_decay)


def _define_curriculum_agent(model: DuelingDeepQLearning,
                             exploration_policy: BoltzmannExplorationPolicy,
                             warmup_random_action_probability: float) -> DDDQLTicTacToeAgent:
    # Define attributes
    weight_copy_step_interval: int = 100
    batch_size: int = 150
    # Return the agent
    return DDDQLTicTacToeAgent("dddqn_curriculum_agent", model, exploration_policy, weight_copy_step_interval, batch_size, warmup_random_action_probability)


if __name__ == "__main__":
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Generate Tic Tac Toe environments with random and fixed environment player and using the O player as the environment player with two reward types
    # Tic Tac Toe random environment:
    #       - success threshold to consider both the training completed and the experiment successful is around 95% of match won by the agent (depending on reward assigned)
    environment_name: str = 'TicTacToeRandom'
    environment_random_low_reward: TicTacToeEnvironmentRandom = TicTacToeEnvironmentRandom(environment_name, Player.o,
                                                                                           1.0, -0.1, 0.0)
    environment_random_high_reward: TicTacToeEnvironmentRandom = TicTacToeEnvironmentRandom(environment_name, Player.o,
                                                                                            100.0, -10.0, 0.0)
    # Tic Tac Toe fixed environment:
    #       - success threshold to consider both the training completed and the experiment successful is around 65% of match won by the agent (depending on reward assigned)
    environment_name: str = 'TicTacToeFixed'
    environment_fixed_low_reward: TicTacToeEnvironmentFixed = TicTacToeEnvironmentFixed(environment_name, Player.o,
                                                                                        1.0, -0.1, 0.0)
    environment_fixed_high_reward: TicTacToeEnvironmentFixed = TicTacToeEnvironmentFixed(environment_name, Player.o,
                                                                                         100.0, -10.0, 0.0)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [1024, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [1024, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    nn_config.add_hidden_layer(LayerType.dense, [1024, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()])
    # Define model
    inner_model_first: DuelingDeepQLearning = _define_dddqn_model(nn_config, 0.000001)
    inner_model_second: DuelingDeepQLearning = _define_dddqn_model(nn_config, 0.000001)
    # Define exploration policies
    exploration_policy_first: BoltzmannExplorationPolicy = _define_boltzmann_exploration_policy(1.0, 0.1)
    exploration_policy_second: BoltzmannExplorationPolicy = _define_boltzmann_exploration_policy(0.85, 0.1)
    # Define agents
    dddqn_curriculum_agent_first: DDDQLTicTacToeAgent = _define_curriculum_agent(inner_model_first,
                                                                                 exploration_policy_first,
                                                                                 1.0)
    dddqn_curriculum_agent_second: DDDQLTicTacToeAgent = _define_curriculum_agent(inner_model_second,
                                                                                  exploration_policy_second,
                                                                                  0.25)
    # Define interfaces
    interface_low_reward_random: TicTacToePassThroughInterface = TicTacToePassThroughInterface(environment_random_low_reward)
    interface_high_reward_random: TicTacToePassThroughInterface = TicTacToePassThroughInterface(environment_random_high_reward)
    interface_low_reward_fixed: TicTacToePassThroughInterface = TicTacToePassThroughInterface(environment_fixed_low_reward)
    interface_high_reward_fixed: TicTacToePassThroughInterface = TicTacToePassThroughInterface(environment_fixed_high_reward)
    # Define experiments
    success_threshold: float = 0.95
    experiment_low_reward_random: TicTacToeExperiment = TicTacToeExperiment("experiment_low_reward", success_threshold,
                                                                            environment_random_low_reward,
                                                                            dddqn_curriculum_agent_first, interface_low_reward_random)
    success_threshold: float = 0.65
    experiment_low_reward_fixed: TicTacToeExperiment = TicTacToeExperiment("experiment_low_reward", success_threshold,
                                                                           environment_fixed_low_reward,
                                                                           dddqn_curriculum_agent_second, interface_low_reward_fixed)
    success_threshold: float = 95.0
    experiment_high_reward_random: TicTacToeExperiment = TicTacToeExperiment("experiment_high_reward", success_threshold,
                                                                             environment_random_high_reward,
                                                                             dddqn_curriculum_agent_first, interface_high_reward_random)
    success_threshold: float = 65.0
    experiment_high_reward_fixed: TicTacToeExperiment = TicTacToeExperiment("experiment_low_reward", success_threshold,
                                                                            environment_fixed_high_reward,
                                                                            dddqn_curriculum_agent_second, interface_high_reward_fixed)
    # Define experiments data
    testing_episodes: int = 100
    test_cycles: int = 10
    training_episodes: int = 1000
    validation_episodes: int = 100
    max_training_episodes: int = 50000
    episode_length_max: int = 20
    # Run curriculum experiments for low reward
    saved_metagraph_paths: [] = run_experiment(experiment_low_reward_random,
                                               training_episodes,
                                               max_training_episodes, episode_length_max,
                                               validation_episodes,
                                               testing_episodes, test_cycles,
                                               render_during_training, render_during_validation, render_during_test,
                                               workspace_path, __file__,
                                               logger, None, experiment_iterations_number)
    for metagraph_path in saved_metagraph_paths:
        run_experiment(experiment_low_reward_fixed,
                       training_episodes,
                       max_training_episodes, episode_length_max,
                       validation_episodes,
                       testing_episodes, test_cycles,
                       render_during_training, render_during_validation, render_during_test,
                       workspace_path, __file__,
                       logger, metagraph_path)
    # Run curriculum experiments for high reward
    saved_metagraph_paths: [] = run_experiment(experiment_high_reward_random,
                                               training_episodes,
                                               max_training_episodes, episode_length_max,
                                               validation_episodes,
                                               testing_episodes, test_cycles,
                                               render_during_training, render_during_validation, render_during_test,
                                               workspace_path, __file__,
                                               logger, None, experiment_iterations_number)
    for metagraph_path in saved_metagraph_paths:
        run_experiment(experiment_high_reward_fixed,
                       training_episodes,
                       max_training_episodes, episode_length_max,
                       validation_episodes,
                       testing_episodes, test_cycles,
                       render_during_training, render_during_validation, render_during_test,
                       workspace_path, __file__,
                       logger, metagraph_path)



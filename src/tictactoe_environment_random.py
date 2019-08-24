# Import packages

import logging

# Import src

from src.tictactoe_environment import TicTacToeEnvironment, Player


class TicTacToeEnvironmentRandom(TicTacToeEnvironment):
    """
    Tic-Tac-Toe environment in which the environment player plays with a random policy.
    """

    def __init__(self,
                 name: str,
                 environment_player: Player,
                 agent_player_win_reward: float,
                 environment_player_win_reward: float,
                 draw_reward: float):
        # Generate the base tic tac toe environment
        super(TicTacToeEnvironmentRandom, self).__init__(name, environment_player, agent_player_win_reward, environment_player_win_reward, draw_reward)

    def get_environment_player_first_action(self,
                                            logger: logging.Logger,
                                            session) -> int:
        # Just return a random action
        return self.get_random_action(logger, session)

    def get_environment_player_action(self,
                                      logger: logging.Logger,
                                      session) -> int:
        return self.get_random_action(logger, session)

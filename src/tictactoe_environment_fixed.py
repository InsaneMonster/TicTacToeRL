# Import packages

import logging

# Import src

from src.tictactoe_environment import TicTacToeEnvironment, Player


class TicTacToeEnvironmentFixed(TicTacToeEnvironment):
    """
    Tic-Tac-Toe environment in which the environment player plays with a fixed policy.
    """

    def __init__(self,
                 name: str,
                 environment_player: Player,
                 agent_player_win_reward: float,
                 environment_player_win_reward: float,
                 draw_reward: float):
        # Generate the base tic tac toe environment
        super(TicTacToeEnvironmentFixed, self).__init__(name, environment_player, agent_player_win_reward, environment_player_win_reward, draw_reward)

    def get_environment_player_first_action(self,
                                            logger: logging.Logger,
                                            session) -> int:
        # Just return a random action
        return self.get_random_action(logger, session)

    def get_environment_player_action(self,
                                      logger: logging.Logger,
                                      session) -> int:
        # Get all the possible actions at the current state
        possible_actions: [] = self.get_possible_actions(logger, session)
        # Check if anyone of the possible actions can achieve victory for the environment player
        for action in possible_actions:
            position, player = divmod(action, 2)
            # Set the player to its defined value
            if player == 0:
                player = Player.o
            else:
                player = Player.x
            # Check if the chosen player is valid
            if player == self.environment_player:
                # Return the action if the chosen position would make the environment player win at this move
                # Match in positions 0 - 1 - 2
                if position == 0 and self._state[1] == self.environment_player and self._state[1] == self._state[2]:
                    return action
                if position == 1 and self._state[0] == self.environment_player and self._state[0] == self._state[2]:
                    return action
                if position == 2 and self._state[0] == self.environment_player and self._state[0] == self._state[1]:
                    return action
                # Match in positions 0 - 3 - 6
                if position == 0 and self._state[3] == self.environment_player and self._state[3] == self._state[6]:
                    return action
                if position == 3 and self._state[0] == self.environment_player and self._state[0] == self._state[6]:
                    return action
                if position == 6 and self._state[0] == self.environment_player and self._state[0] == self._state[3]:
                    return action
                # Match in positions 0 - 4 - 8
                if position == 0 and self._state[4] == self.environment_player and self._state[4] == self._state[8]:
                    return action
                if position == 4 and self._state[0] == self.environment_player and self._state[0] == self._state[8]:
                    return action
                if position == 8 and self._state[0] == self.environment_player and self._state[0] == self._state[4]:
                    return action
                # Match in positions 2 - 5 - 8
                if position == 2 and self._state[5] == self.environment_player and self._state[5] == self._state[8]:
                    return action
                if position == 5 and self._state[2] == self.environment_player and self._state[2] == self._state[8]:
                    return action
                if position == 8 and self._state[2] == self.environment_player and self._state[2] == self._state[5]:
                    return action
                # Match in positions 2 - 4 - 6
                if position == 2 and self._state[4] == self.environment_player and self._state[4] == self._state[6]:
                    return action
                if position == 4 and self._state[2] == self.environment_player and self._state[2] == self._state[6]:
                    return action
                if position == 6 and self._state[2] == self.environment_player and self._state[2] == self._state[4]:
                    return action
                # Match in positions 1 - 4 - 7
                if position == 1 and self._state[4] == self.environment_player and self._state[4] == self._state[7]:
                    return action
                if position == 4 and self._state[1] == self.environment_player and self._state[1] == self._state[7]:
                    return action
                if position == 7 and self._state[1] == self.environment_player and self._state[1] == self._state[4]:
                    return action
                # Match in positions 3 - 4 - 5
                if position == 3 and self._state[4] == self.environment_player and self._state[4] == self._state[5]:
                    return action
                if position == 4 and self._state[3] == self.environment_player and self._state[3] == self._state[5]:
                    return action
                if position == 5 and self._state[3] == self.environment_player and self._state[3] == self._state[4]:
                    return action
                # Match in positions 6 - 7 - 8
                if position == 6 and self._state[7] == self.environment_player and self._state[7] == self._state[8]:
                    return action
                if position == 7 and self._state[6] == self.environment_player and self._state[6] == self._state[8]:
                    return action
                if position == 8 and self._state[6] == self.environment_player and self._state[6] == self._state[7]:
                    return action
            else:
                logger.error("Environment player asked to play when it's not its turn!")
                break
        # Check if anyone of the possible actions can avoid victory for the agent player
        for action in possible_actions:
            position, player = divmod(action, 2)
            # Set the player to its defined value
            if player == 0:
                player = Player.o
            else:
                player = Player.x
            # Check if the chosen player is valid
            if player == self.environment_player:
                # Return the action if it can block the agent player to win
                # Match in positions 0 - 1 - 2
                if position == 0 and self._state[1] == self.agent_player and self._state[1] == self._state[2]:
                    return action
                if position == 1 and self._state[0] == self.agent_player and self._state[0] == self._state[2]:
                    return action
                if position == 2 and self._state[0] == self.agent_player and self._state[0] == self._state[1]:
                    return action
                # Match in positions 0 - 3 - 6
                if position == 0 and self._state[3] == self.agent_player and self._state[3] == self._state[6]:
                    return action
                if position == 3 and self._state[0] == self.agent_player and self._state[0] == self._state[6]:
                    return action
                if position == 6 and self._state[0] == self.agent_player and self._state[0] == self._state[3]:
                    return action
                # Match in positions 0 - 4 - 8
                if position == 0 and self._state[4] == self.agent_player and self._state[4] == self._state[8]:
                    return action
                if position == 4 and self._state[0] == self.agent_player and self._state[0] == self._state[8]:
                    return action
                if position == 8 and self._state[0] == self.agent_player and self._state[0] == self._state[4]:
                    return action
                # Match in positions 2 - 5 - 8
                if position == 2 and self._state[5] == self.agent_player and self._state[5] == self._state[8]:
                    return action
                if position == 5 and self._state[2] == self.agent_player and self._state[2] == self._state[8]:
                    return action
                if position == 8 and self._state[2] == self.agent_player and self._state[2] == self._state[5]:
                    return action
                # Match in positions 2 - 4 - 6
                if position == 2 and self._state[4] == self.agent_player and self._state[4] == self._state[6]:
                    return action
                if position == 4 and self._state[2] == self.agent_player and self._state[2] == self._state[6]:
                    return action
                if position == 6 and self._state[2] == self.agent_player and self._state[2] == self._state[4]:
                    return action
                # Match in positions 1 - 4 - 7
                if position == 1 and self._state[4] == self.agent_player and self._state[4] == self._state[7]:
                    return action
                if position == 4 and self._state[1] == self.agent_player and self._state[1] == self._state[7]:
                    return action
                if position == 7 and self._state[1] == self.agent_player and self._state[1] == self._state[4]:
                    return action
                # Match in positions 3 - 4 - 5
                if position == 3 and self._state[4] == self.agent_player and self._state[4] == self._state[5]:
                    return action
                if position == 4 and self._state[3] == self.agent_player and self._state[3] == self._state[5]:
                    return action
                if position == 5 and self._state[3] == self.agent_player and self._state[3] == self._state[4]:
                    return action
                # Match in positions 6 - 7 - 8
                if position == 6 and self._state[7] == self.agent_player and self._state[7] == self._state[8]:
                    return action
                if position == 7 and self._state[6] == self.agent_player and self._state[6] == self._state[8]:
                    return action
                if position == 8 and self._state[6] == self.agent_player and self._state[6] == self._state[7]:
                    return action
            else:
                logger.error("Environment player asked to play when it's not its turn!")
                break
        # Otherwise just return a random action
        return self.get_random_action(logger, session)

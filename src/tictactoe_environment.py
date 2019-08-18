# Import packages

import logging
import numpy
import random
import time
import enum
import copy

# Import usienarl

from usienarl import Environment, SpaceType


# Define player type class

class Player(enum.Enum):
    x = 1
    o = -1
    none = 0


class TicTacToeEnvironment(Environment):
    """
    Tic-Tac-Toe abstract environment.

    The state is a vector representing the board with:
        - 0 is the empty cell
        - 1 is the X cell
        - -1 is the O cell

    The action is a number where:
        - 0 => X in 0
        - 1 => O in 0
        - 4 => X in 2
        - 5 => O in 2
        - 2n => X in n
        - 2n+1 => O in n
    """

    def __init__(self,
                 name: str,
                 environment_player: Player,
                 agent_player_win_reward: float,
                 environment_player_win_reward: float,
                 draw_reward: float):
        # Define attributes
        self.winner: Player = Player.none
        self.last_player: Player = Player.none
        self.current_player: Player = Player.none
        self.environment_player: Player = environment_player
        if self.environment_player == Player.x:
            self.agent_player: Player = Player.o
        else:
            self.agent_player: Player = Player.x
        self.agent_player_win_reward: float = agent_player_win_reward
        self.environment_player_win_reward: float = environment_player_win_reward
        self.draw_reward: float = draw_reward
        # Define internal attributes
        self._move: int = 0
        self._episode_done: bool = False
        # Define internal empty attributes
        self._state: numpy.ndarray = None
        self._intermediate_state: numpy.ndarray = None
        self._flipped_state: numpy.ndarray = None
        # Generate the base environment
        super(TicTacToeEnvironment, self).__init__(name)

    def setup(self,
              logger: logging.Logger) -> bool:
        # The environment setup is always successful
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        pass

    def close(self,
              logger: logging.Logger,
              session):
        pass

    def reset(self,
              logger: logging.Logger,
              session):
        # Reset attributes
        self.winner = Player.none
        self.last_player = Player.none
        # Reset internal attributes
        self._move: int = 0
        self._episode_done = False
        # Reset state
        self._state = numpy.array([Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none, Player.none])
        self._flipped_state = self._state.copy()
        # Choose a random starting player
        self.current_player = Player.o
        if random.uniform(0, 1) <= 0.5:
            self.current_player = Player.x
        # If the current player is the environment player, let it decide how to play
        if self.current_player == self.environment_player:
            # Increase move count
            self._move += 1
            # Save the current representation of the board for rendering purpose
            self._intermediate_state = copy.deepcopy(self._state)
            # Get the environment player action
            environment_player_action: int = self.get_environment_player_first_action(logger, session)
            position, player = divmod(environment_player_action, 2)
            # Set the player to its defined value and update the state
            if player == 0:
                player = Player.o
                flipped_player = Player.x
            else:
                player = Player.x
                flipped_player = Player.o
            self._state[position] = player
            self._flipped_state[position] = flipped_player
            # Update the last player and current player
            self.last_player = self.current_player
            if self.current_player == Player.x:
                self.current_player = Player.o
            else:
                self.current_player = Player.x
        # Return the first state encoded
        return self._encode_state_int(self._state)

    def step(self,
             logger: logging.Logger,
             action,
             session):
        # Increase move count
        self._move += 1
        # Change the state with the given action
        position, player = divmod(action, 2)
        # Set the player to its defined value and update the state
        if player == 0:
            player = Player.o
            flipped_player = Player.x
        else:
            player = Player.x
            flipped_player = Player.o
        self._state[position] = player
        self._flipped_state[position] = flipped_player
        # Update the last player and current player
        self.last_player = self.current_player
        if self.current_player == Player.x:
            self.current_player = Player.o
        else:
            self.current_player = Player.x
        # Reset the current intermediate state
        self._intermediate_state = None
        # Check for winner and episode completion flag
        self._episode_done, self.winner = self._check_if_final(self._state)
        # If the current player is the environment player, let it decide how to play
        if not self._episode_done and self.current_player == self.environment_player:
            # Increase move count
            self._move += 1
            # Save the current representation of the board for rendering purpose
            self._intermediate_state = copy.deepcopy(self._state)
            # Get the environment player action
            environment_player_action: int = self.get_environment_player_action(logger, session)
            position, player = divmod(environment_player_action, 2)
            # Set the player to its defined value and update the state
            if player == 0:
                player = Player.o
                flipped_player = Player.x
            else:
                player = Player.x
                flipped_player = Player.o
            self._state[position] = player
            self._flipped_state[position] = flipped_player
            # Update the last player and current player
            self.last_player = self.current_player
            if self.current_player == Player.x:
                self.current_player = Player.o
            else:
                self.current_player = Player.x
            # Check for winner and episode completion flag
            self._episode_done, self.winner = self._check_if_final(self._state)
        # Assign rewards
        reward: float = 0.0
        if self._episode_done:
            if self.winner == Player.x:
                reward = self.agent_player_win_reward
            elif self.winner == Player.o:
                reward = self.environment_player_win_reward
            else:
                reward = self.draw_reward
        # Return the encoded state, the reward and the episode completion flag
        return self._encode_state_int(self._state), reward, self._episode_done

    def render(self,
               logger: logging.Logger,
               session):
        # Print the intermediate board, if any
        if self._intermediate_state is not None:
            self._print_board(self._intermediate_state)
            # Print separator
            print("____________________")
            time.sleep(1.0)
        # Print the current state
        self._print_board(self._state)
        # Print separator
        print("____________________")
        # Print end of episode footer and results
        if self._episode_done:
            print("MATCH END")
            print("Played moves: " + str(self._move))
            if self.winner != Player.none:
                print("Winner player is " + ("X" if self.winner == Player.x else "O"))
            else:
                print("There is no winner: it's a draw!")
            print("____________________")
        time.sleep(1.0)

    def get_random_action(self,
                          logger: logging.Logger,
                          session):
        # Choose a random action in the possible actions
        return random.choice(self.get_possible_actions(logger, session))

    @property
    def state_space_type(self):
        return SpaceType.continuous

    @property
    def state_space_shape(self):
        return 9,

    @property
    def action_space_type(self):
        return SpaceType.discrete

    @property
    def action_space_shape(self):
        return 9 * 2,

    def get_possible_actions(self,
                             logger: logging.Logger,
                             session):
        """
        Return all the possible action at the current state in the environment.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: the list of possible actions
        """
        # Get all the possible actions according to position
        possible_actions: [] = []
        for action in range(*self.action_space_shape):
            position, player = divmod(action, 2)
            # Set the player to its defined value
            if player == 0:
                player = Player.o
            else:
                player = Player.x
            if self._state[position] == Player.none and player == self.current_player:
                possible_actions.append(action)
        return possible_actions

    @staticmethod
    def _check_if_final(state: numpy.ndarray):
        """
        Check if the given state is final and also return the winner.

        :return: True if final, False otherwise and the winner
        """
        # Check if state is final and compute winner
        winner: Player = Player.none
        episode_done: bool = False
        if state[0] != Player.none and state[0] == state[1] == state[2]:
            episode_done = True
            winner = state[0]
        if state[0] != Player.none and state[0] == state[3] == state[6]:
            episode_done = True
            winner = state[0]
        if state[0] != Player.none and state[0] == state[4] == state[8]:
            episode_done = True
            winner = state[0]
        if state[2] != Player.none and state[2] == state[5] == state[8]:
            episode_done = True
            winner = state[2]
        if state[2] != Player.none and state[2] == state[4] == state[6]:
            episode_done = True
            winner = state[2]
        if state[1] != Player.none and state[1] == state[4] == state[7]:
            episode_done = True
            winner = state[1]
        if state[3] != Player.none and state[3] == state[4] == state[5]:
            episode_done = True
            winner = state[3]
        if state[6] != Player.none and state[6] == state[7] == state[8]:
            episode_done = True
            winner = state[6]
        # Check for draw: no empty element in the board and no winner
        if winner == Player.none:
            empty_found: bool = False
            for element in state:
                if element == Player.none:
                    empty_found = True
                    break
            if not empty_found:
                episode_done = True
        # Return completion flag and winner
        return episode_done, winner

    @staticmethod
    def _print_board(state: numpy.ndarray):
        """
        Print the given state of the board.
        """
        # Get all the rows in the board state
        first_row: [] = []
        second_row: [] = []
        third_row: [] = []
        for i in range(state.size):
            # Convert the state to its graphical representation
            graphical_element: str = " "
            if state[i] == Player.x:
                graphical_element = "X"
            elif state[i] == Player.o:
                graphical_element = "O"
            if i < 3:
                first_row.append(graphical_element)
            elif 3 <= i < 6:
                second_row.append(graphical_element)
            else:
                third_row.append(graphical_element)
        # Print each row
        print(first_row)
        print(second_row)
        print(third_row)

    @staticmethod
    def _encode_state_int(state: numpy.ndarray):
        """
        Encode the given state of the board (expressed in player occupied cells) with an integer sequence.

        :param state: the state to encode
        :return: the encoded state
        """
        encoded_state: numpy.ndarray = numpy.zeros(state.size, dtype=int)
        for i in range(state.size):
            encoded_state[i] = state[i].value
        return encoded_state

    def get_environment_player_first_action(self,
                                            logger: logging.Logger,
                                            session) -> int:
        """
        Get the first action from the environment player, if any.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: the action of the environment agent at the first state
        """
        raise NotImplementedError()

    def get_environment_player_action(self,
                                      logger: logging.Logger,
                                      session) -> int:
        """
        Get the action from the environment player, if any.

        :param logger: the logger used to print the environment information, warnings and errors
        :param session: the session of tensorflow currently running, if any
        :return: the action of the environment agent at the current state
        """
        raise NotImplementedError()

from Game import Game
import numpy as np

class State:
    def __init__(self):
        self.current_player = 0 #Player 0 is -1, player 1 is 1 
        self.board = np.zeros((3, 3))

class Game:
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.squares = self.rows * self.cols

    def __repr__(self) -> str:
        return "TicTacToe"

    def get_number_of_players(self):
        return 2

    def get_action_count(self):
        return 9

    def get_initial_state(self):
        return State()

    def get_next_state(self, state: State, action):
        row = action // self.cols
        col = action % self.cols
        state.board[row, col] = -1 if state.current_player == 0 else 1
        state.current_player = (state.current_player + 1) % 2

    def get_legal_actions(self, state: State): #There is definitely a more pythonic way to do this but it's fine
        legal = np.reshape(state.board, 9)
        return legal == 0

    def has_won(self, state: State):
        pass

    def is_game_over(self, state):
        pass

    def get_canonical_state(self, state):
        pass

    def get_encoded_state(self, state):
        return np.stack()

    def get_current_player(self, state):
        pass

    #The following functions SHOULD be implemented by children
    def get_string_representation(self, state) -> str:
        pass

    
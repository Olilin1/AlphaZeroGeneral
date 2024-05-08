from Game import Game
import numpy as np
import torch

class State:
    def __init__(self):
        self.current_player = 0 #Player 0 is -1, player 1 is 1 
        self.board = np.zeros((3, 3))

    def copy(self):
        newState = State()
        newState.board = self.board.copy()
        newState.current_player = self.current_player
        return newState

class TicTacToe(Game):
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.squares = self.rows * self.cols

    def __repr__(self) -> str:
        return "TicTacToe"

    def get_number_of_symmetries(self): #Not true but leave it be for now
        return 1

    def get_symmetrical_state(self, state, symmetry):
        return state

    def get_number_of_players(self):
        return 2

    def get_action_count(self):
        return 9

    def get_initial_state(self):
        state = State()
        # state = self.get_next_state(state, 0)
        # state = self.get_next_state(state, 1)
        # state = self.get_next_state(state, 3)
        # state = self.get_next_state(state, 4)
        # state = self.get_next_state(state, 2)
        return state

    def get_next_state(self, state: State, action):
        row = action // self.cols
        col = action % self.cols
        state.board[row, col] = -1 if state.current_player == 0 else 1
        state.current_player = (state.current_player + 1) % 2
        return state

    def get_legal_actions(self, state: State): #There is definitely a more pythonic way to do this but it's fine
        legal = np.reshape(state.board, 9)
        return legal == 0

    def has_won(self, state: State):
        prev_player = 1 if state.current_player == 0 else -1
        for i in range(0, 3):
            if np.sum(state.board[i,:]) == 3*prev_player:
                return [-prev_player, prev_player]
            if np.sum(state.board[:,i]) == 3*prev_player:
                return [-prev_player, prev_player]

        if (np.sum(np.diag(state.board)) == prev_player * 3 or 
            np.sum(np.diag(np.flip(state.board, axis=0))) == prev_player * 3):
            return [-prev_player, prev_player]
        return [0, 0]

    def is_game_over(self, state: State):
        val = self.has_won(state)
        val = np.array(val, dtype=np.float32)
        if val[0] == 0:
            return np.prod(state.board) != 0, val
        
        return True, val

    def get_encoded_state(self, state: State):
        if state.current_player == 0:
            return torch.tensor(np.stack((state.board == -1, state.board == 1, state.board == 0), dtype=np.float32)).unsqueeze(0)
        else:
            return torch.tensor(np.stack((state.board == 1, state.board == -1, state.board == 0), dtype=np.float32)).unsqueeze(0)

    def get_current_player(self, state: State):
        return state.current_player

    #The following functions SHOULD be implemented by children
    def get_string_representation(self, state: State) -> str:
        rep = ""
        for i in range(0, 3):
            for j in range(0, 3):
                rep += f"{int(state.board[i, j])} "
            rep += '\n'
        return rep

    
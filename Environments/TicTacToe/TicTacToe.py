import numpy as np
from Game import Game

args = {
    'c': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'num_hidden_layers': 4,
    'hidden_layer_size': 64,
    'lr': 0.001,
    'weight_decay': 0.0001
}

class TicTacToeState(np.ndarray):
    def __new__(cls, input_array, action=None):        
        obj = np.asarray(input_array).view(cls)
        obj.action = action
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.action = getattr(obj, 'your_new_attr', None)

class TicTacToe(Game):
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.action_count = self.rows * self.cols

    def __repr__(self) -> str:
        return 'TicTacToe'

    def get_action_count(self):
        return 9

    def get_initial_state(self):
        state = TicTacToeState(
            np.zeros((self.rows, self.cols))
        )
        state.action = None
        return state

    def get_next_state(self, state, action):
        player = self.get_current_player(state)
        row = action // self.cols
        col = action % self.cols
        state[row, col] = player
        state.action = action
        return state

    def get_legal_actions(self, state):
        legal_actions = np.zeros(9)
        for action in range(0, 9):
            row = action // self.cols
            col = action % self.rows
            if(state[row, col] == 0): 
                legal_actions[action] = 1
        
        return legal_actions

    def get_legal_indicies(self, state):
        legal_actions = self.get_legal_actions(state)
        return [idx for (idx, x) in enumerate(legal_actions) if x == 1]

    def check_win(self, state, action, player):
        row = action // self.cols
        col = action % self.rows

        has_won = (
            np.sum(state[row, :]) == player * 3 or 
            np.sum(state[:, col]) == player * 3 or 
            np.sum(np.diag(state)) == player * 3 or 
            np.sum(np.diag(np.flip(state, axis=0))) == player * 3
        )
        return has_won

    def is_game_over(self, state):
        action = state.action
        if action == None:
            return False, 0
        row = action // self.cols
        col = action % self.rows
        player = state[row, col]
        if self.check_win(state, action, player):
            return True, 1
        elif np.sum(self.get_legal_actions(state)) == 0:
            return True, 0
        else:
            return False, 0

    def get_state_from_perspective(self, state, player):
        action = state.action
        state = state * player
        state.action = action
        return state

    def get_canonical_state(self, state):
        canon_state = state * self.get_current_player(state)
        canon_state.action = state.action
        return canon_state

    def get_encoded_state(self, state):
        return np.stack((state == -1, state == 0, state == 1)).astype(np.float32)

    def get_current_player(self, state):
        return 1 if np.sum(state) <= 0 else -1

    def get_string_representation(self, state) -> str:
        return str(state)
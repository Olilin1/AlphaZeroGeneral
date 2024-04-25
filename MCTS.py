from sre_parse import State
from Game import Game
import numpy as np
import math

class Node:
    def __init__(self, game: Game, args, state, parent=None, action=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action

        self.children = []
        self.expandable_moves = game.get_legal_actions(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb_score = -np.inf

        for child in self.children: 
            ucb = self.get_ucb(child)
            if ucb > best_ucb_score:
                best_ucb_score = ucb
                best_child = child
        return best_child

    def get_ucb(self, child):
        win_percent = 1 - ((child.value_sum / child.visit_count) + 1) / 2 #This needs to be changed when generalising for more players
        ucb = win_percent + self.args['c'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
        return ucb

    def expand(self):
        move = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[move] = 0

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, move)
        
        child_state = self.game.get_state_from_perspective(child_state, -1)

        child = Node(self.game, self.args, child_state, self, move)
        self.children.append(child)

        return child

    def simulate(self):
        terminal, value = self.game.is_game_over(self.state)
        value = -value

        if terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            legal_actions = self.game.get_legal_actions(rollout_state)
            valid_moves = [i for i, val in enumerate(legal_actions) if val == 1]
            
            action = np.random.choice(valid_moves)
            rollout_state = self.game.get_next_state(rollout_state, action)
            terminal, value = self.game.is_game_over(rollout_state)
            if terminal:
                if rollout_player == -1:
                    value = -value
                return value

            rollout_player = self.game.get_current_player(rollout_state)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, game: Game, args):
        self.game = game
        self.args = args
    
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()

            terminal, value = self.game.is_game_over(node.state)
            value = -value
            if not terminal:
                node = node.expand()
                value = node.simulate()
            node.backpropagate(value)

        action_probailities = np.zeros(self.game.get_action_count())
        for child in root.children:
            action_probailities[child.action] = child.visit_count
        
        action_probailities /= np.sum(action_probailities)
        return action_probailities
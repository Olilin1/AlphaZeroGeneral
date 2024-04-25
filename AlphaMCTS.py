from Game import Game
import numpy as np
import math
import torch

class Node:
    def __init__(self, game: Game, args, state, parent=None, action=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

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
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 #This needs to be changed when generalising for more players
        ucb = q_value + self.args['c'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
        return ucb

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)
                child_state = self.game.get_state_from_perspective(child_state, -1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)

class AlphaMCTS:
    def __init__(self, game: Game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.get_action_count())
        
        valid_moves = self.game.get_legal_actions(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()

            terminal, value = self.game.is_game_over(node.state)
            value = -value
            if not terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis = 1).squeeze(0).cpu().numpy()

                valid_actions = self.game.get_legal_actions(node.state)
                policy *= valid_actions
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)
                
            node.backpropagate(value)

        action_probailities = np.zeros(self.game.get_action_count())
        for child in root.children:
            action_probailities[child.action] = child.visit_count
        
        action_probailities /= np.sum(action_probailities)
        return action_probailities
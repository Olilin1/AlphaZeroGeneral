from copy import deepcopy
from Game import Game
import numpy as np
import torch

class Node:
    def __init__(self, args, game: Game, state, parent, action, prior_probability) -> None:
        self.state = state
        self.parent: Node = parent
        self.game = game
        self.action = action
        self.args = args

        self.total_action_value = 0 #W
        self.visit_count = 0        #N
        self.mean_action_value = 0  #Q
        self.prior_probability = prior_probability #P

        self.children: list[Node] = []


    def select(self):
        if len(self.children) == 0:
            return self
        
        max_util = -np.inf
        best_child = None
        visit_value_sum = 0
        for child in self.children:
            visit_value_sum += child.visit_count

        for child in self.children:
            util = self.args['c_puct'] * child.prior_probability * (np.sqrt(visit_value_sum) / (1 + child.visit_count))
            util += child.mean_action_value
            if util > max_util:
                max_util = util
                best_child = child

        return best_child.select()

    def expand(self, policy):
        for idx, probability in enumerate(policy):
            if probability != 0:
                child_state = deepcopy(self.state)
                child_state = self.game.get_next_state(child_state, idx)
                self.children.append(Node(self.args, self.game, child_state, self, idx, probability))

    def backup(self, value):
        if self.parent == None:
            return

        player = self.game.get_current_player(self.parent.state)
        self.visit_count += 1
        self.total_action_value += value[player]
        self.mean_action_value = self.total_action_value / self.visit_count
        self.parent.backup(value)



class MCTS:
    def __init__(self, args, game: Game, model, state) -> None:
        self.args = args
        self.game = game
        self.model = model
        self.root = Node(self.args, self.game, state, None, None, 1)

    def take_step(self, action):
        for node in self.root.children:
            if node.action == action:
                self.root = node
                return

    def get_policy(self, temperature):
        root = self.root
        
        for i in range(self.args["num_searches"]):
            self.search(root)
        
        policy = np.zeros(self.game.get_action_count())
        
        for child in root.children:
            policy[child.action] = child.visit_count
        
        if temperature == 0:
            best = np.argmax(policy)
            policy = np.zeros(self.game.get_action_count())
            policy[best] = 1
        else:
            policy /= np.sum(policy)
            policy ** (1 / temperature)
            policy /= np.sum(policy)
        
        
        return policy

    def search(self, root: Node):
        node = root.select()

        terminal, value = self.game.is_game_over(node.state)

        if not terminal:
            enc = self.game.get_encoded_state(node.state)
            policy, value = self.model(enc)
            value = value.squeeze(0).detach().cpu().numpy()
            policy = policy.squeeze(0).detach().cpu().numpy()
            
            player = self.game.get_current_player(node.state)
            value = np.roll(value, player)
            legal_moves = self.game.get_legal_actions(node.state)
            policy *= legal_moves
            sum = np.sum(policy)
            if sum != 0:
                policy /= sum
            else:
                policy = legal_moves / np.sum(legal_moves)

            node.expand(policy)

        node.backup(value) 
        
        
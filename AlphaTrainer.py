from email import policy
from Game import Game
from MCTS import MCTS
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AlphaTrainer:
    def __init__(self, args, model: nn.Module, optimizer, game: Game):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.game = game

    def selfPlay(self):
        memory: list[list] = []
        state = self.game.get_initial_state()
        mcts = MCTS(self.game, self.args)
        turn = 0
        while True:
            temperature = 1 #For now
            if turn > self.args["temperature_threshold"]:
                temperature = 0

            action_probabilities = mcts.get_policy(state, temperature) #Next step, make MCTS persistant. For example, have it return a node as well or similar

            memory.append([state.copy(), action_probabilities])

            action = np.random.choice(self.game.get_action_count(), action_probabilities)
            state = self.game.get_next_state(state, action)

            terminal, value = self.game.is_game_over(state)

            if terminal:
                for item in memory:
                    item.append(value)

                return memory

    def learn(self, memory):
        np.random.shuffle(memory)
        for item in memory:
            state = item[0]
            state = self.game.get_symmetrical_state(state, np.random.choice(self.game.get_number_of_symmetries()))
            target_probabilities = item[1]
            target_value = item[2]
            
            measured_probabilities, measured_value = self.model(self.game.get_encoded_state(state))

            policy_loss = F.cross_entropy(measured_probabilities, target_probabilities)
            value_loss = F.mse_loss(measured_value, target_value, reduction='mean')

            total_loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()



    def train(self):
        memory = []
        self.model.eval()

        for iteration in range(self.args["num_selfplay_games"]):
            memory += self.selfPlay()

        self.model.train()
        for epoc in range(self.args['num_training_epochs']):
            self.learn(memory)







            
            

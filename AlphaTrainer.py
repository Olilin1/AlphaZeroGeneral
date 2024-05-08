from copy import deepcopy
from Game import Game
from MCTS import MCTS
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class AlphaTrainer:
    def __init__(self, args, model: nn.Module, optimizer, game: Game):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.game = game

        self.last_loss = 0
        self.loss_arr = []

    def selfPlay(self):
        memory: list[list] = []
        state = self.game.get_initial_state()
        mcts = MCTS(self.args, self.game, self.model, state)
        turn = 0
        while True:
            temperature = 2 #For now
            turn += 1
            print("Turn: ", turn)
            if turn > self.args["temperature_threshold"]:
                temperature = 0

            action_probabilities = mcts.get_policy(temperature) 

            memory.append([deepcopy(state), action_probabilities])

            
            action = np.random.choice(self.game.get_action_count(), p=action_probabilities)
            state = self.game.get_next_state(state, action)

            terminal, value = self.game.is_game_over(state)

            if terminal:
                for item in memory:
                    item.append(value)

                return memory
            mcts.take_step(action)

    def learn(self, memory):
        np.random.shuffle(memory)
        for item in memory:
            state = item[0]
            state = self.game.get_symmetrical_state(state, np.random.choice(self.game.get_number_of_symmetries()))
            target_probabilities = item[1]
            target_values = item[2]

            target_values = np.roll(target_values, -self.game.get_current_player(state))
            
            measured_probabilities, measured_value = self.model(self.game.get_encoded_state(state))

            target_probabilities = torch.tensor(target_probabilities).unsqueeze(0)
            target_values = torch.tensor(target_values).unsqueeze(0)

            policy_loss = F.cross_entropy(measured_probabilities, target_probabilities)
            value_loss = F.mse_loss(measured_value, target_values, reduction='mean')

            total_loss = policy_loss + value_loss

            self.last_loss += total_loss.item()

            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


    def train(self):
        memory = []
        self.model.eval()

        for i in range(self.args["num_selfplay_games"]):
            print("Game ", i)
            memory += self.selfPlay()

        self.model.train()
        self.last_loss = 0
        for i in range(self.args['num_training_epochs']):
            print("Batch ", i)
            self.learn(memory)
        self.loss_arr.append(self.last_loss)
        print("Loss: ", self.loss_arr)







            
            

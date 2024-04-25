from AlphaMCTS import AlphaMCTS
from Game import Game
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TorchModel import ResNet
from tqdm import tqdm

class AlphaZero:
    def __init__(self, model, optimizer, game: Game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        while True:
            neutral_state = self.game.get_canonical_state(state)
            action_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state, action_probs, player))

            temperature_probs = action_probs ** (1 / self.args['temperature'])
            temperature_probs /= np.sum(temperature_probs)
            action = np.random.choice(self.game.action_count, p = temperature_probs)
            state = self.game.get_next_state(state, action)
            terminal, value = self.game.is_game_over(state)

            if terminal:
                returnMemory =[]
                for hist_neutral, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral), 
                        hist_action_probs, 
                        hist_outcome
                    ))
                return returnMemory
            else:
                player = self.game.get_current_player(state)

    def train(self, memory):
        np.random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_target = zip(*sample)

            state, policy_targets, value_target = np.array(state), np.array(policy_targets), np.array(value_target).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_target = torch.tensor(value_target, dtype=torch.float32, device=self.model.device)
            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_target)

            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'])):
                memory += self.selfPlay()
            
            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict, f"optimizer_{iteration}.pt")
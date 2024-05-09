from copy import deepcopy
from Game import Game
from MCTS import MCTS
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import multiprocessing
import time

class AlphaTrainer:
    def __init__(self, args, model: nn.Module, optimizer, game: Game):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.game = game

        self.queue = multiprocessing.Queue()

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
            #print("Turn: ", turn)
            # if turn == 2:
            #     stime = time.time()
            # elif turn == 4:
            #     print((time.time()-stime)/2)
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

                self.queue.put(memory)

                return
            mcts.take_step(action)

    def learn(self, memory):
        np.random.shuffle(memory)
        for item in memory:
            state = item[0]
            state = self.game.get_symmetrical_state(state, np.random.choice(self.game.get_number_of_symmetries()))
            target_probabilities = item[1]
            target_values = item[2]

            target_values = np.roll(target_values, -self.game.get_current_player(state))
            
            measured_probabilities, measured_value = self.model(self.game.get_encoded_state(state).to(torch.device('cuda')))

            target_probabilities = torch.tensor(target_probabilities).unsqueeze(0).to(torch.device('cuda'))
            target_values = torch.tensor(target_values).unsqueeze(0).to(torch.device('cuda'))

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
        #Yes I am aware that this code is horrible, it will be fixed
        # for i in range(self.args["num_selfplay_games"]):
        #     print("Game ", i)

        num_procs = 8
        procs = [multiprocessing.Process(target=self.selfPlay) for _ in range(0, num_procs)]


        for proc in procs:
            proc.start()
        
        for _ in range(0, num_procs):
            memory += self.queue.get()

        for proc in procs:
            proc.join()


        self.model.train()
        self.last_loss = 0
        for i in range(self.args['num_training_epochs']):
            print("Batch ", i)
            self.learn(memory)
        self.loss_arr.append(self.last_loss)
        print("Loss: ", self.loss_arr)







            
            

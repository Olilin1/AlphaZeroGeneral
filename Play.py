import torch
from AlphaTrainer import AlphaTrainer
# from Environments.TicTacToe.TicTacToe import TicTacToe
# from Environments.TicTacToe.Model import TicTacToeModel
from Environments.Towers2P.Towers2P import Towers2P
from Environments.Towers2P.Model import Towers2PModel
import numpy as np
from MCTS import MCTS

useMCTS = True

args = {
    'c_puct': 2,
    'num_searches': 100,
    "num_selfplay_games": 100,
    "num_training_epochs": 4,
    "temperature_threshold": 100
}
checkpoint = torch.load("Trained/model_2.pt")

values = []
p1s = 0
p2s = 0

for i in range (0, 10):
    print(i)
    terminal = False
    turn = 0

    game = Towers2P()
    state = game.get_initial_state()
    model1 = Towers2PModel()



    model1.load_state_dict(checkpoint["model_state_dict"])



    mcts1 = MCTS(args, game, model1, state)


    while not terminal:
        turn += 1
        if(turn % 10 == 0):
            print(turn)


        policy1 = mcts1.get_policy(0)


        if(game.get_current_player(state) == 0):
            action = np.argmax(policy1)
        else:
            legal = game.get_legal_actions(state)
            pick = [i for i, val in enumerate(legal) if val == 1]
            action = np.random.choice(pick)

        state = game.get_next_state(state, action)
        terminal, value = game.is_game_over(state)
        mcts1.take_step(action)

        if(terminal):
            p1s += value[0]
            p2s += value[1]
            values.append(value)
            print(values)
print(values)
print(p1s)
print(p2s)


    



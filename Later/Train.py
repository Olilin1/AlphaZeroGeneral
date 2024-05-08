import sys
from Game import Game
import torch
from AlphaTrainer import AlphaTrainer
from importlib import import_module



if __name__ != '__main__':
    exit()

arg_dict = dict()
for arg in sys.argv[1:]:
    v, k = arg.split('=')
    arg_dict[v] = k

game_path = "Environments." + arg_dict['game'] + "." + arg_dict['game']
args_path = "Environments." + arg_dict['game'] + "." + "Params"
module = import_module(game_path)

game = getattr(module, arg_dict['game'])()
args = getattr(module, 'args')

device = torch.device(arg_dict['device'])
model = ResNet(game, args['num_hidden_layers'], args['hidden_layer_size'])

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
trainer = AlphaZero(model, optimizer, game, args)

trainer.learn()
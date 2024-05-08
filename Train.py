import torch
from AlphaTrainer import AlphaTrainer
# from Environments.TicTacToe.TicTacToe import TicTacToe
# from Environments.TicTacToe.Model import TicTacToeModel
from Environments.Towers2P.Towers2P import Towers2P
from Environments.Towers2P.Model import Towers2PModel

args = {
    'c_puct': 2,
    'num_searches': 1000,
    "num_selfplay_games": 1,
    "num_training_epochs": 1,
    "temperature_threshold": 100
}

game = Towers2P()
model = Towers2PModel()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0001)

checkpoint = torch.load("Trained/model_0.pt")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

trainer = AlphaTrainer(args, model, optimizer, game)

iteration = 1
while True:
    trainer.train()
    print("Iteration ", iteration, " complete")
    
    torch.save(
        {"iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": trainer.last_loss},
        f"Trained/model_{iteration}.pt"
    )
    iteration += 1


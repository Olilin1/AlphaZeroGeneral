import pygame
from Environments.Towers2P.Towers2P import State, Towers2P
import numpy as np
import time
import torch
from AlphaTrainer import AlphaTrainer
# from Environments.TicTacToe.TicTacToe import TicTacToe
# from Environments.TicTacToe.Model import TicTacToeModel
from Environments.Towers2P.Towers2P import Towers2P
from Environments.Towers2P.Model import Towers2PModel
import numpy as np
from MCTS import MCTS

args = {
    'c_puct': 2,
    'num_searches': 100,
    "num_selfplay_games": 100,
    "num_training_epochs": 4,
    "temperature_threshold": 100
}
WHITE = (255, 255, 255)
BLACK = (0, 0,0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 106, 167)
YELLOW = (254, 204, 0)

WIDTH = 500
HEIGHT = 500
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
running = True

checkpoint = torch.load("Trained/model_4.pt")
game = Towers2P()
state = game.get_initial_state()
model1 = Towers2PModel()



model1.load_state_dict(checkpoint["model_state_dict"])



mcts1 = MCTS(args, game, model1, state)
number_font = pygame.font.SysFont( None, 32 )
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0,0,0))
    for i in range(0, 5):
        for j in range(0, 5):
            match state.board_flags[i, j]:
                case 0:
                    color = WHITE
                case 1:
                    color = BLUE
                case 2:
                    color = RED
                case 3:
                    color = YELLOW
                case 4:
                    color = GREEN
            rect = pygame.Rect(j * 100 + 5, i * 100 + 5, 90, 90)
            
            number_text  = str(int(state.board_height[i, j]))
            number_image = number_font.render( number_text, True, BLACK)
            tex_rec = number_image.get_rect()
            tex_rec.center = (j* 100 + 50, i * 100 + 50)
            

            pygame.draw.rect(screen, color, rect, 0)
            if state.board_pieces[i, j] != 0:
                pygame.draw.circle(screen, BLACK, (j * 100 + 50, i * 100 + 50), 25, 8)
            screen.blit(number_image, tex_rec)
    
    screen.set_at((int(WIDTH/2), int(HEIGHT/2)), (0,255,0))
    pygame.display.flip()

    terminal, value = game.is_game_over(state)
    if terminal:
        print(value)
        time.sleep(1000)

    policy = mcts1.get_policy(0)
    if game.get_current_player(state) == 0:
        action = np.argmax(policy)
    else:
        legal = game.get_legal_actions(state)
        pick = [i for i, val in enumerate(legal) if val == 1]
        action = np.random.choice(pick)
    
    state = game.get_next_state(state, action)
    mcts1.take_step(action)
    

    time.sleep(2)
pygame.quit()
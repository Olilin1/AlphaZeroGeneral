from shutil import move
from Environments.Towers2P.Towers2P import Towers2P
import numpy as np
game = Towers2P()
state = game.get_initial_state()

while True:
    print(game.get_string_representation(state))
    action = None
    legal = game.get_legal_actions(state)
    moves = [i for i, val in enumerate(legal) if val == 1]
    print(moves)
    if game.get_current_player(state) == 0:
        type = int(input("1 - Build, 2 - Spawn, 3 - Move. Your action:"))

        if type == 1:
            row = int(input("Row: "))
            col = int(input("Col: "))
            action = row * 5
            action += col
        elif type == 2:
            row = int(input("Row: "))
            col = int(input("Col: "))
            action = row * 5
            action += col
            action += 25
        else:
            row = int(input("Row: "))
            col = int(input("Col: "))
            direction = int(input("0 - Down, 1 - Up, 2 - Right, 3 - Left. Direction:"))
            action = row * 5
            action += col
            action *= 4
            action += direction
            action += 50
    else:
        action = np.random.choice(moves)
        print("Ai Action: ", action)
    state = game.get_next_state(state, action)

    terminal, val = game.is_game_over(state)
    if(terminal):
        print("Game over! ", val)
        exit()

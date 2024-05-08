
from Game import Game
import numpy as np
import torch



class State:
    def __init__(self):
        self.current_player = 1 #Not represented, will go from 1 to 4, despite only 2 players
        self.blocks_left = 50
        self.turn = 0
        self.prev_action = None
        self.prev_move = None
        self.setup_phase = True
        self.forced_placement = False
        self.action_left = False
        self.start_team = None #Represent as, am I starting or no
        self.pieces_left = [6, 6, 6, 6]
        self.flags_left = [9, 9, 9, 9]
        self.captured_pieces = [0, 0]
        self.board_pieces = np.zeros((5, 5)) #5 layers
        self.board_flags = np.zeros((5, 5)) #5 layers
        self.board_height = np.zeros((5, 5)) #1 layer

        #Non-commented fields can be represented using an additional 7 layers
        #Or by using some sort of deep network something


# In this version of the game you are allowed to capture any pieces.
# That includes your own pieces of different colors, or even your own pieces
# that are the same color as the capturing piece.
# Furthermore, if the game lasts for 50 * 4 consecutive turns, the game is determined to be over
# If one person places a pawn in setup, then players will still alternate moves


class Towers2P(Game):
    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.squares = self.rows * self.cols
    
    def __repr__(self):
        return "Towers2P"
    
    def get_number_of_symmetries(self):
        return 8

    def get_symmetrical_state(self, state: State, symmetry):
        return state #Forgot about that prev_action needs to change as_well
        if symmetry <= 3:
            for _ in range(0, symmetry):
                state.board_flags = np.rot90(state.board_flags)
                state.board_pieces = np.rot90(state.board_pieces)
                state.board_height = np.rot90(state.board_height)
        elif symmetry == 4:
            state.board_flags = np.flipud(state.board_flags)
            state.board_pieces = np.flipud(state.board_pieces)
            state.board_height = np.flipud(state.board_height)
        elif symmetry == 5:
            state.board_flags = np.fliplr(state.board_flags)
            state.board_pieces = np.fliplr(state.board_pieces)
            state.board_height = np.fliplr(state.board_height)
        elif symmetry == 6:
            state.board_flags = np.rot90(np.flipud(state.board_flags))
            state.board_pieces = np.rot90(np.flipud(state.board_pieces))
            state.board_height = np.rot90(np.flipud(state.board_height))
        elif symmetry == 7:
            state.board_flags = np.rot90(np.fliplr(state.board_flags))
            state.board_pieces = np.rot90(np.fliplr(state.board_pieces))
            state.board_height = np.rot90(np.fliplr(state.board_height))
        return state

    def get_action_count(self):
        return (
            self.squares +      #Build
            self.squares +      #Spawn
            self.squares * 4    #Move
            )

    def get_number_of_players(self):
        return 2

    def get_initial_state(self):
        state = State()
        return state

    def get_next_state(self, state: State, action):
        legal = self.get_legal_actions(state)

        if(legal[action] != 1):
            print("Legal_Action_Missmatch! ", action)

        state.turn += 1
        if action < self.squares: #Build
            row = action  // self.cols
            col = action % self.cols

            state.board_height[row, col] += 1
            state.blocks_left -= 1
        elif action < self.squares * 2: #Spawn
            if state.setup_phase:
                state.flags_left[state.current_player-1] -= 1
            if state.start_team == None:
                state.start_team = state.current_player
            if state.current_player > 2:
                state.forced_placement = True
            action -= self.squares
            row = action  // self.cols
            col = action % self.cols
            action += self.squares

            state.board_pieces[row, col] = state.current_player
            state.board_flags[row, col] = state.current_player
            state.pieces_left[state.current_player-1] -= 1
        else: #Move
            action -= self.squares * 2
            direction = action % 4
            square = action // 4
            action += self.squares * 2

            row = square  // self.cols
            col = square % self.cols

            state.board_pieces[row, col] = 0

            if direction == 0:
                row += 1
            elif direction == 1:
                row -= 1
            elif direction == 2:
                col += 1
            else:
                col -= 1

            if row == 5 or col == 5:
                print(action)
            state.prev_move = (row, col)
            if state.board_flags[row, col] != 0:
                state.flags_left[int(state.board_flags[row, col]-1)] += 1
            
            state.board_flags[row, col] = state.current_player
            state.flags_left[state.current_player-1] -= 1

            if state.board_pieces[row, col] != 0:
                state.captured_pieces[state.current_player % 2] += 1
            state.board_pieces[row, col] = state.current_player
        
        if state.setup_phase:
            if np.sum(state.pieces_left) <= 5 * 4:
                state.setup_phase = False
                state.current_player = state.start_team
                state.action_left = True
            elif state.forced_placement:
                for i in range(1, 5):
                    if state.pieces_left[i-1] == 6:
                        state.current_player = i
                        break
            else:
                state.current_player = (state.current_player % 2) + 1
                if state.pieces_left[state.current_player - 1] != 6:
                    state.current_player += 2

        elif state.action_left:
            state.prev_action = action
            state.action_left = False
        
        else:
            state.action_left = True
            state.prev_action = None
            state.prev_move = None
            state.current_player = (state.current_player % 4) + 1

        return state

    def get_legal_actions(self, state: State):
        legal_actions = np.zeros(self.get_action_count())
        
        for action in range(0, self.squares): #Build
            row = action  // self.cols
            col = action % self.cols

            if state.setup_phase and not state.forced_placement and state.board_pieces[row, col] == 0: #Maybe you are allowed to build under spawned pieces? Need to ask Titouan
                legal_actions[action] = 1

            elif state.board_pieces[row, col] == state.current_player and action != state.prev_action:
                legal_actions[action] = 1

        for action in range(self.squares, self.squares * 2): #Spawn
            action -= self.squares
            row = action  // self.cols
            col = action % self.cols
            action += self.squares
            
            if state.setup_phase: #work here
                allowed = True
                if row == 0 or row == 4 or col == 0 or col == 4:
                    allowed = True
                else:
                    allowed = False

                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if row + i < 0 or row + i >= self.rows or col + j < 0 or col +j >= self.cols or abs(i) + abs(j) > 2:
                            continue
                        if state.board_pieces[row + i, col + j] != 0:
                            allowed = False
                if allowed:
                    legal_actions[action] = 1

            elif (state.board_flags[row, col] == state.current_player and 
                state.board_pieces[row, col] == 0 and 
                state.pieces_left[state.current_player - 1] > 0 and
                action != state.prev_action):
                
                legal_actions[action] = 1
        
        for action in range(self.squares * 2, self.squares * 6): #Move
            if state.setup_phase:
                break
            action -= self.squares * 2
            direction = action % 4
            square = action // 4

            row = square  // self.cols
            col = square % self.cols
            action += self.squares * 2

            if state.board_pieces[row, col] != state.current_player:
                continue

            if state.prev_move == (row, col):
                continue

            orgRow = row
            orgCol = col
            
            if direction == 0:
                row += 1
            elif direction == 1:
                row -= 1
            elif direction == 2:
                col += 1
            else:
                col -= 1

            if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                continue
            
            if row == 5 or col == 5:
                print (action)
            
            if state.board_height[orgRow, orgCol] > state.board_height[row, col] or state.board_pieces[row, col] == 0:
                legal_actions[action] = 1
        return legal_actions

    def compute_winner(self, state: State):
        if state.flags_left[0] + state.flags_left[2] > state.flags_left[1] + state.flags_left[3]:
            #print("c1")
            return [-1, 1]
        elif state.flags_left[0] + state.flags_left[2] < state.flags_left[1] + state.flags_left[3]:
           # print("c2")
            return [1, -1]
        else:
            #print("c3")
            if state.captured_pieces[0] > state.captured_pieces[1]:
                return [1, -1]
            elif state.captured_pieces[0] < state.captured_pieces[1]:
                return [-1, 1]
            else:
                return [0, 0]
              
    def is_game_over(self, state: State):
        if np.prod(state.flags_left) == 0 or state.blocks_left == 0 or np.sum(self.get_legal_actions(state)) == 0 or state.turn == 200:
            return True, np.array(self.compute_winner(state), dtype=np.float32)
        else:
            return False, np.array([0, 0], dtype=np.float32)
        
    def get_encoded_state(self, state: State):
        first = state.current_player
        second = first % 4 + 1
        third = second % 4 + 1
        fourth = third % 4 + 1


        curr_team = (state.current_player - 1) % 2
        opp_team = (curr_team + 1) % 2

        move_map = np.zeros((self.rows, self.cols))
        build_map = np.zeros((self.rows, self.cols))

        

        (row, col, ac) = self.format_action(state.prev_action)
        if state.action_left or state.setup_phase:
            pass
        elif ac == "Build":
            build_map[row, col] = 1
        elif ac == "Move":
            if row == 5 or col == 5:
                print(state.prev_action)
                print(row, " ", col)
                print(state.prev_move)
            move_map[row, col] = 1

        map_data = np.stack((
            state.board_flags == 0,
            state.board_flags == first,
            state.board_flags == second,
            state.board_flags == third,
            state.board_flags == fourth,
            state.board_pieces == 0,
            state.board_pieces == first,
            state.board_pieces == second,
            state.board_pieces == third,
            state.board_pieces == fourth,
            state.board_height,
            build_map,
            move_map,
            np.full((self.rows, self.cols), 0 if state.start_team == None else (state.start_team % 2 == state.current_player % 2)),
            np.full((self.rows, self.cols), state.turn),
            np.full((self.rows, self.cols), state.action_left),
            np.full((self.rows, self.cols), state.pieces_left[first - 1]),
            np.full((self.rows, self.cols), state.pieces_left[second - 1]),
            np.full((self.rows, self.cols), state.pieces_left[third - 1]),
            np.full((self.rows, self.cols), state.pieces_left[fourth - 1]),
            np.full((self.rows, self.cols), state.captured_pieces[curr_team]),
            np.full((self.rows, self.cols), state.captured_pieces[opp_team])
        ), dtype=np.float32) #22 planes. 5 X 5 X 22 network
        


        return torch.tensor(map_data).unsqueeze(0)
        
    def get_current_player(self, state: State):
        return (state.current_player-1) % 2

    def get_string_representation(self, state) -> str:
        rep = ""
        for row in range(0, 5):
            for col in range(0, 5):
                rep += f"({state.board_flags[row, col]}, {state.board_pieces[row, col]}, {state.board_height[row, col]}) "
            rep += "\n"
        return rep

    def format_action(self, action):
        if action == None:
            return (0, 0, None)
        if action < self.squares:
            row = action // self.cols
            col = action // self.cols
            return (row, col, "Build")
        elif action < self.squares * 2:
            action -= self.squares
            row = action // self.cols
            col = action // self.cols
            return (row, col, "Spawn")
        else:
            action -= self.squares * 2
            direction = action % 4
            square = action // 4

            row = square  // self.cols
            col = square % self.cols

            if direction == 0:
                row += 1
            elif direction == 1:
                row -= 1
            elif direction == 2:
                col += 1
            else:
                col -= 1
            
            return (row, col, "Move")


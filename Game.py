class Game:
    #The following functions SHALL be implemented by children
    def __init__(self):
        pass

    def __repr__(self) -> str:
        pass

    def get_action_count(self):
        pass

    def get_initial_state(self):
        pass

    def get_next_state(self, state, action):
        pass

    def get_legal_actions(self, state):
        pass

    def is_game_over(self, state):
        pass

    def get_canonical_state(self, state):
        pass

    def get_encoded_state(self, state):
        pass

    def get_current_player(self, state):
        pass

    def get_state_from_perspective(self, state, player):
        pass

    #The following functions SHOULD be implemented by children
    def get_string_representation(self, state) -> str:
        pass

    
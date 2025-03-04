class BlokusGame:
    BOARD_SIZE = 14

    def __init__(self):
        self.board = np.zeros((14,14))
        self.player = 1
        self.first_move = {1: True, 2: True}
        self.starting_positions = {
            1: (4, 4),
            2: (self.BOARD_SIZE - 5, self.BOARD_SIZE - 5),
        }
        self.available_pieces = {
            1: set(pieces.keys()),
            2: set(pieces.keys()),
        }
        self.move_history = []


    def get_valid_placements(self, piece, position):
        # Returns coordinates of possible piece placements on spot, or empty list if no valid placements
        # player = 1 or 2 for current player trying to place piece.
        # check if piece can be placed on grid at coord based on blokus rules
        # position is a tuple for (r, c)

        ##### IMPORTANT: MAKE SURE TO CHECK THAT CORNER AND BOUNDARY CONDITIONS ARE FILTERED!!!!!!
        ################################## OTHERWISE, IT WILL NOT WORK ##############################
        possible_placements = []
        # for piece in self.available_pieces[self.player]:
        placement_options = pcco[piece] + position
        # Remove any options out of bounds
        mask = np.any((placement_options < 0) | (placement_options >= 14), axis=(1, 2))
        placement_options = placement_options[~mask]

        boundary_options = pcb[piece] + position
        boundary_options = boundary_options[~mask]
        corner_options = pcc[piece] + position
        corner_options = corner_options[~mask]
        for option, boundary, corner in zip(placement_options, boundary_options, corner_options):
            b_mask = np.any((boundary < 0) | (boundary >= 14), axis=1)
            boundary = tuple(boundary[~b_mask].T)
            c_mask = np.any((corner < 0) | (corner >= 14), axis=1)
            corner = tuple(corner[~c_mask].T)
            # Check space that piece will occupy is empty
            if np.all(self.board[tuple(option.T)] == 0):
                # Check at least one of corners is equal to player OR IF FIRST MOVE
                if np.any(self.board[corner] == self.player) or self.first_move[self.player]:
                    # Check if all boundary are not equal to player
                    if np.all(self.board[boundary] != self.player):
                        possible_placements.append(option)
        return possible_placements

    def place_piece(self, coords):
        self.board[tuple(coords.T)] = self.player
        # # Place piece on board at position
        # # player = 1 or 2 for current player trying to place piece.
        # # position is a tuple for (r, c)
        # # piece is the name of the piece to place
        # valid_placements = self.get_valid_placements(piece, position)
        # if len(valid_placements) == 0:
        #     return False
        # self.board[valid_placements[0]] = self.player
        # self.available_pieces[self.player].remove(piece)
        # self.move_history.append((piece, position))
        # return True
    def reset_board(self):
        self.board = np.zeros((14,14))
        # self.player = 1
        # self.first_move = {1: True, 2: True}
        # self.available_pieces = {
        #     1: set(pieces.keys()),
        #     2: set(pieces.keys()),
        # }
        # self.move_history = []
    
    def __str__(self):
        return "\n".join(" ".join(f"{int(cell)}" for cell in row) for row in self.board)
import numpy as np
from gameconfig import piece_bdd_corner_pos_dict as pdcpd
from gameconfig import pcco, pcc, pcb


class BlokusGame:
    BOARD_SIZE = 14

    def __init__(self):
        self.board = np.zeros((14, 14))
        self.player = 1
        self.first_move = {1: True, 2: True}
        self.starting_positions = {
            1: (4, 4),
            2: (self.BOARD_SIZE - 5, self.BOARD_SIZE - 5),
        }
        # self.corner_positions = {1: [(4, 4)], 2: [(self.BOARD_SIZE - 5, self.BOARD_SIZE - 5)]}
        self.available_pieces = {
            1: set(pcco.keys()),
            2: set(pcco.keys()),
        }
        self.move_history = []

    def get_legal_actions(self):
        """
        Returns list of tuple of (piece_name, piece_coords, position) for all legal actions.
        """
        legal_actions = []
        for piece in self.available_pieces[self.player]:
            for position in self.get_candidate_positions():
                for coords in self.get_valid_placements(piece, position):
                    legal_actions.append((piece, coords, position))
        if not legal_actions:
            legal_actions.append(("pass", None, None))
        return legal_actions

    def get_valid_placements(self, piece, position):
        # Assuming position is corner to some piece or starting position
        # Returns coordinates of possible piece placements on spot, or empty list if no valid placements
        # player = 1 or 2 for current player trying to place piece.
        # check if piece can be placed on grid at coord based on blokus rules
        # position is a tuple for (r, c)
        possible_placements = []
        # for piece in self.available_pieces[self.player]:
        # placement_options = pcco[piece] + position
        # # Remove any options out of bounds
        # mask = np.any((placement_options < 0) | (placement_options >= 14), axis=(1, 2))
        # placement_options = placement_options[~mask]

        # boundary_options = pcb[piece] + position
        # boundary_options = boundary_options[~mask]
        # corner_options = pcc[piece] + position
        # corner_options = corner_options[~mask]

        # for option, boundary, corner in zip(placement_options, boundary_options, corner_options):
        #     b_mask = np.any((boundary < 0) | (boundary >= 14), axis=1)
        #     boundary = tuple(boundary[~b_mask].T)
        #     c_mask = np.any((corner < 0) | (corner >= 14), axis=1)
        #     corner = tuple(corner[~c_mask].T)
        # Check space that piece will occupy is empty
        for option, boundary, corner in pdcpd[position][piece]:
            if np.all(self.board[boundary] != self.player):
                if np.all(self.board[tuple(option.T)] == 0):
                    possible_placements.append(option)
        return possible_placements

    def get_candidate_positions(self):
        """
        Finds all valid candidate positions where a player can place a piece.
        A position is valid if:
        - It is diagonally adjacent (corner-adjacent) to at least one existing piece of the player.
        - It is NOT orthogonally adjacent to any existing piece of the player.
        """
        candidate_positions = set()
        player_cells = np.argwhere(self.board == self.player)

        if not len(player_cells):  # If no pieces have been placed yet, return the starting position
            return {self.starting_positions[self.player]}

        for r, c in player_cells:
            # Check diagonal (corner-adjacent) positions
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and self.board[nr, nc] == 0:
                    # Ensure it is NOT orthogonally adjacent to any of the player's own pieces
                    orthogonally_adjacent = any(
                        0 <= nr + odr < self.BOARD_SIZE
                        and 0 <= nc + odc < self.BOARD_SIZE
                        and self.board[nr + odr, nc + odc] == self.player
                        for odr, odc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    )
                    if not orthogonally_adjacent:
                        candidate_positions.add((nr, nc))

        return candidate_positions

    def apply_move(self, piece_name, coords, position):
        if piece_name != "pass":
            self.board[tuple(coords.T)] = self.player
            self.available_pieces[self.player].remove(piece_name)
        self.move_history.append((self.player, piece_name, coords))

        self.first_move[self.player] = False
        self.player = 3 - self.player  # Switch player

    def game_over(self):
        if len(self.move_history) >= 2:
            if self.move_history[-1][1] == "pass" and self.move_history[-2][1] == "pass":
                return True
        return False

    def get_winner(self):
        if not self.game_over():
            return 0
        score1, score2 = self.get_score()
        if score1 > score2:
            return 1
        elif score2 > score1:
            return 2
        else:
            return 0

    def get_score(self):
        score1 = np.sum(self.board == 1)
        score2 = np.sum(self.board == 2)
        return score1, score2

    def clone(self):
        """
        Creates and returns a deep copy of the game state.
        This is essential for simulations in MCTS.
        """
        cloned = BlokusGame()
        cloned.board = np.copy(self.board)
        cloned.player = self.player
        cloned.first_move = self.first_move.copy()
        cloned.starting_positions = self.starting_positions.copy()
        cloned.available_pieces = {p: self.available_pieces[p].copy() for p in self.available_pieces}
        cloned.move_history = self.move_history[:]
        return cloned

    def encode_board(self):
        """
        Encodes the current game state into a multi-channel NumPy array,
        which can be used as input for a neural network.
        Channels:
          - Channel 0: Binary map of player 1's pieces.
          - Channel 1: Binary map of player 2's pieces.
          - Channel 2: Board filled with a turn indicator (1.0 if player 1's turn, 0.0 otherwise).
          - Channel 3: (Optional) Additional features (currently zeros).
        Output shape: (4, BOARD_SIZE, BOARD_SIZE)
        """
        channel1 = (self.board == 1).astype(np.float32)
        channel2 = (self.board == 2).astype(np.float32)
        channel3 = np.full((self.BOARD_SIZE, self.BOARD_SIZE), 1.0 if self.player == 1 else 0.0, dtype=np.float32)
        # TODO: Make channel4 representative of actions left for each player
        channel4 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        return np.stack([channel1, channel2, channel3, channel4], axis=0)

    def __str__(self):
        # return "\n".join(" ".join(f"{int(cell)}" for cell in row) for row in self.board)
        board_str = "\n".join(
            " ".join("X" if cell == 1 else "O" if cell == 2 else "." for cell in row) for row in self.board
        )
        return board_str

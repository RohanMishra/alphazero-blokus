import numpy as np

from gameconfig import (
    board_size,
    action_size,
    piecenames,
    id_to_boundary,
    pos_piece_to_id,
    id_to_piece_coords,
    reverse_id_perspective,
    id_to_coords,
    p1_start,
    p2_start,
)


class BlokusGame:
    BOARD_SIZE = board_size

    def __init__(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.player = 1
        self.flipped = False
        self.first_move = {1: True, -1: True}
        self.starting_positions = {
            1: p1_start,
            -1: p2_start,
        }
        self.available_pieces = {
            1: piecenames.copy(),
            -1: piecenames.copy(),
        }
        self.passed = {1: False, -1: False}
        self.action_size = action_size

        # Area of board that a piece can validly cover. Initially the entire board
        # Dictionary for player to its valid area, a 14x14 numpy mask of T/F
        self.valid_area = {
            1: np.ones((self.BOARD_SIZE, self.BOARD_SIZE), dtype=bool),
            -1: np.ones((self.BOARD_SIZE, self.BOARD_SIZE), dtype=bool),
        }

        self.turnnum = 1

    def get_legal_actions(self, dropout_rate=0):
        legal_actions = []
        positions = self.get_candidate_positions()
        for piece in self.available_pieces[self.player]:
            for position in positions:
                possible_placements_at_pos = []
                for action_id in pos_piece_to_id[position][piece]:
                    if np.random.rand() < dropout_rate:
                        continue
                    coords = id_to_coords[action_id]
                    if np.all(self.valid_area[self.player][coords]):
                        possible_placements_at_pos.append(action_id)
                legal_actions += possible_placements_at_pos
        if not legal_actions:
            legal_actions.append(0)
        return legal_actions

    def get_candidate_positions(self):
        if self.first_move[self.player]:
            return {self.starting_positions[self.player]}

        # Create a boolean mask of where the player's pieces are
        player_mask = self.board == self.player

        # Create a mask that is True for cells that are diagonally adjacent
        diag_candidate = np.zeros_like(self.board, dtype=bool)
        # Check each diagonal direction by shifting the player's mask.
        # A candidate cell (i, j) is diagonally adjacent if, for instance, (i-1, j-1) holds a player's piece.
        diag_candidate[1:, 1:] |= player_mask[:-1, :-1]  # top-left shifted to bottom-right candidate
        diag_candidate[1:, :-1] |= player_mask[:-1, 1:]  # top-right shifted to bottom-left candidate
        diag_candidate[:-1, 1:] |= player_mask[1:, :-1]  # bottom-left shifted to top-right candidate
        diag_candidate[:-1, :-1] |= player_mask[1:, 1:]  # bottom-right shifted to top-left candidate

        # Create a mask for cells that are orthogonally adjacent to a player's piece.
        ortho_mask = np.zeros_like(self.board, dtype=bool)
        ortho_mask[1:, :] |= player_mask[:-1, :]  # above neighbor: mark cell below if above is player's piece
        ortho_mask[:-1, :] |= player_mask[1:, :]  # below neighbor: mark cell above if below is player's piece
        ortho_mask[:, 1:] |= player_mask[:, :-1]  # left neighbor: mark cell right if left is player's piece
        ortho_mask[:, :-1] |= player_mask[:, 1:]  # right neighbor: mark cell left if right is player's piece

        candidate_mask = (self.board == 0) & diag_candidate & (~ortho_mask)

        candidate_positions = {tuple(pos) for pos in np.argwhere(candidate_mask)}

        return candidate_positions

    def apply_move(self, action_id):
        if action_id != 0:
            piecename, coords = id_to_piece_coords[action_id]
            self.board[coords] = self.player
            self.available_pieces[self.player].remove(piecename)
            self.passed[self.player] = False

            # Remove from valid area for current player boundary of piece placed
            piece_boundary = id_to_boundary[action_id]
            self.valid_area[self.player][piece_boundary] = False
            self.valid_area[self.player][coords] = False
            # Remove from valid area for other player the area covered by the piece placed
            self.valid_area[-self.player][coords] = False

        else:
            self.passed[self.player] = True

        self.first_move[self.player] = False
        self.player *= -1  # Switch player
        self.turnnum += 1

    def game_over(self):
        return self.passed[1] and self.passed[-1]

    def get_winner(self):
        if self.game_over():
            score1, score2 = self.get_score()
            if score1 > score2:
                return 1
            elif score2 > score1:
                return -1
            else:
                return 0
        else:
            raise Exception("Game is not over yet.")

    def get_score(self):
        score1 = np.sum(self.board == 1)
        score2 = np.sum(self.board == -1)
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
        cloned.passed = self.passed.copy()
        cloned.valid_area = {p: self.valid_area[p].copy() for p in self.valid_area}
        cloned.turnnum = self.turnnum
        cloned.flipped = self.flipped

        return cloned

    def encode_board(self):
        """
        Encodes the current game state into a multi-channel NumPy array,
        which can be used as input for a neural network.
        Channels:
          - Channel 0: Binary map of player 1's pieces.
          - Channel 1: Binary map of player 2's pieces.
          - Channel 3+: (Optional) Additional features.
        Output shape: (?, BOARD_SIZE, BOARD_SIZE)
        """
        channels = []
        channels.append((self.board == 1).astype(np.float32))
        channels.append((self.board == -1).astype(np.float32))
        # channel3 = (board == 0).astype(np.float32)
        # channel3 = np.full((self.BOARD_SIZE, self.BOARD_SIZE), 1.0 if self.player == 1 else 0.0, dtype=np.float32)
        # Channel 4 should be a mask of the valid area for the current player
        channels.append(self.valid_area[self.player].astype(np.float32))
        # Channel 5 mask of valid positions
        channel4 = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32)
        for pos in self.get_candidate_positions():
            channel4[pos] = 1.0
        channels.append(channel4)
        # Channels to encode remaining pieces - iterate through all piece names, and set channel 1 if available, 0 otherwise
        for piece in piecenames:
            if piece in self.available_pieces[self.player]:
                channels.append(np.ones((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32))
            else:
                channels.append(np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32))
        for piece in piecenames:
            if piece in self.available_pieces[-self.player]:
                channels.append(np.ones((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32))
            else:
                channels.append(np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.float32))

        return np.stack(channels, axis=0)

    def flip_action(self, action_id):
        return reverse_id_perspective[action_id] if self.flipped else action_id

    def flip_perspective(self, player):
        assert player == self.player
        self.flipped = not self.flipped if player == -1 else self.flipped
        self.board *= player
        self.board = self.board[::player, ::player]
        self.player *= player

        self.first_move = {1: self.first_move[player], -1: self.first_move[-player]}
        self.available_pieces = {1: self.available_pieces[player], -1: self.available_pieces[-player]}
        self.passed = {1: self.passed[player], -1: self.passed[-player]}
        self.valid_area = {
            1: self.valid_area[player][::player, ::player],
            -1: self.valid_area[-player][::player, ::player],
        }

    def __str__(self):
        board_str = f"Player {self.player}'s Turn\n" + "\n".join(
            " ".join("X" if cell == 1 else "O" if cell == -1 else "." for cell in row) for row in self.board
        )
        return board_str

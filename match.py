import pygame
import sys
import numpy as np
import time
from gameconfig import board_size
from blokus import BlokusGame, id_to_piece_coords

# Constants
CELL_SIZE = 40  # Size of each square on the board
BOARD_SIZE = board_size  # Typically 14 for Blokus Duo
MARGIN = 20  # Margin around the board
WINDOW_SIZE = CELL_SIZE * BOARD_SIZE + 2 * MARGIN
FPS = 10  # Frames per second for visualization
FINAL_PAUSE = 1000  # Milliseconds to pause on final board
TOP_K = 15  # Number of top actions to consider for heatmap


# Color definitions (R, G, B)
COLORS = {
    "EMPTY": (255, 255, 255),  # White
    "PLAYER1": (0, 0, 255),  # Blue
    "PLAYER2": (255, 0, 0),  # Red
    "CANDIDATE": (0, 255, 0),  # Green for candidate highlights
    "GRID_LINE": (0, 0, 0),  # Black grid lines
}


class BlokusGUI:
    def __init__(self, game):
        """
        Initialize the GUI with a given BlokusGame instance.
        """
        pygame.init()
        self.game = game
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Blokus Duo")

    def draw_board(self, scores={}):
        """Draw the board grid, pieces, and highlight valid positions."""
        self.screen.fill(COLORS["GRID_LINE"])
        candidates = set(self.game.get_candidate_positions())

        if scores:
            vals = scores.values()
            min_s, max_s = min(vals), max(vals)
        else:
            min_s, max_s = 0, 1

        curr_player_color = COLORS["PLAYER1"] if self.game.player == 1 else COLORS["PLAYER2"]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x = MARGIN + c * CELL_SIZE
                y = MARGIN + r * CELL_SIZE
                val = self.game.board[r][c]
                if val == 0:
                    color = COLORS["EMPTY"]
                elif val == 1:
                    color = COLORS["PLAYER1"]
                else:
                    color = COLORS["PLAYER2"]
                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                if (r, c) in scores:
                    score = scores.get((r, c), 0)
                    # Normalize to [0,1]
                    norm = (score - min_s) / (max_s - min_s) if max_s > min_s else 1.0
                    alpha = int(norm * 200)
                    overlay = pygame.Surface((CELL_SIZE - 8, CELL_SIZE - 8), pygame.SRCALPHA)
                    overlay.fill((*curr_player_color, alpha))  # red heatmap
                    self.screen.blit(overlay, (x + 4, y + 4))
                # Outline fallback for candidates when no scores
                if (r, c) in candidates:
                    pygame.draw.rect(self.screen, COLORS["CANDIDATE"], (x + 4, y + 4, CELL_SIZE - 8, CELL_SIZE - 8), 3)

                pygame.draw.rect(self.screen, COLORS["GRID_LINE"], (x, y, CELL_SIZE, CELL_SIZE), 1)

    def get_scores(self, probs):
        ind = np.argpartition(probs, -TOP_K)[-TOP_K:]
        ind = ind[np.argsort(probs[ind])][::-1]
        scores = {}
        for i in ind:
            if i == 0:
                return scores
            piece_name, coords = id_to_piece_coords[i]
            positions = list(zip(coords[0], coords[1]))
            for pos in positions:
                if pos not in scores:
                    scores[pos] = 0
                scores[pos] += probs[i]
        return scores

    def run_game(self, agent1, agent2):
        """
        Run one full game with two agents, visualizing each move in real time.
        Gracefully handle window close without crashing Jupyter.
        """
        clock = pygame.time.Clock()
        running = True
        try:
            # Play until game over or window closed
            while running and not self.game.game_over():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                if not running:
                    break
                # Agent selects move
                mover = agent1 if self.game.player == 1 else agent2
                move, probs = mover.select(self.game)
                scores = self.get_scores(probs)
                self.draw_board(scores=scores)
                pygame.display.flip()
                pygame.time.wait(2000)
                self.game.apply_move(move)
                # Draw updated board
                self.draw_board(scores=scores)
                pygame.display.flip()
                clock.tick(FPS)
        finally:
            # Final draw if game ended normally
            if self.game.game_over():
                self.draw_board()
                pygame.display.flip()
                pygame.time.wait(FINAL_PAUSE)
            # Clean up
            pygame.display.quit()
            pygame.quit()


class Match:
    def __init__(self, player1, player2, num_games):
        self.agent1 = player1
        self.agent2 = player2
        self.agent1_wins = 0
        self.agent2_wins = 0
        self.agent1_margin = 0
        self.agent2_margin = 0
        self.ties = 0
        self.num_games = num_games

    def play(self, printfinalboard=False, visualize=False):
        """Play a series of games, optionally visualizing each one."""
        start_time = time.time()
        for _ in range(self.num_games):
            game = BlokusGame()
            if visualize:
                gui = BlokusGUI(game)
                gui.run_game(self.agent1, self.agent2)
            else:
                while not game.game_over():
                    mover = self.agent1 if game.player == 1 else self.agent2
                    game.apply_move(mover.select(game))
            score1, score2 = game.get_score()
            winner = game.get_winner()
            print(f"{score1}, {score2} -> {winner} wins")
            if winner == 1:
                self.agent1_wins += 1
            elif winner == 2:
                self.agent2_wins += 1
            else:
                self.ties += 1
            self.agent1_margin += score1
            self.agent2_margin += score2
            if printfinalboard and not visualize:
                print("Game Final Board:")
                print(game)
        elapsed = (time.time() - start_time) / self.num_games
        print(f"P1 wins: {self.agent1_wins}/{self.num_games}")
        print(f"P2 wins: {self.agent2_wins}/{self.num_games}")
        print(f"Ties: {self.ties}")
        print(
            f"Avg margins (P1, P2): ({self.agent1_margin/self.num_games:.2f}, {self.agent2_margin/self.num_games:.2f})"
        )
        print(f"Avg game time: {elapsed:.3f}s")

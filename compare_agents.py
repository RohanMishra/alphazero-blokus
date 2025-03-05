from blokus import BlokusGame
import time


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

    def play(self):
        start_time = time.time()
        for _ in range(self.num_games):
            game = BlokusGame()
            while not game.game_over():
                if game.player == 1:
                    move = self.agent1.select_move(game)
                else:
                    move = self.agent2.select_move(game)
                game.apply_move(*move)
            score1, score2 = game.get_score()
            winner = game.get_winner()
            print(f"{score1}, {score2} -> {winner} wins")
            if winner == 1:
                self.agent1_wins += 1
            elif winner == 2:
                self.agent2_wins += 1
            self.agent1_margin += score1
            self.agent2_margin += score2
        print(f"{self.agent1_wins}/{self.num_games} won by P1")
        print(f"({self.agent1_margin/self.num_games}, {self.agent2_margin/self.num_games}) Avg. Margin")
        print(f"Avg. Match Time: {round((time.time() - start_time)/self.num_games, 3)}s")

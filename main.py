from compare_agents import Match
from agent import RandomAgent, MCTSAgent, MCTSAgentDropout

# m = Match(MCTSAgent(), RandomAgent(), num_games=5)
# m.play()
print("Random vs. Random 100 Games")
m = Match(RandomAgent(), RandomAgent(), num_games=100)
m.play()

print("MCTS(0.2s) vs. Random 100 Games")
m = Match(MCTSAgent(time_limit=0.2), RandomAgent(), num_games=100)
m.play()

print("Random vs. MCTS(0.2s) 100 Games")
m = Match(RandomAgent(), MCTSAgent(time_limit=0.2), num_games=100)
m.play()

print("MCTS(3s) vs. Random 30 Games")
m = Match(MCTSAgent(time_limit=3), RandomAgent(), num_games=30)
m.play()

print("Random vs. MCTS(3s) 30 Games")
m = Match(RandomAgent(), MCTSAgent(time_limit=3), num_games=30)
m.play()

print("MCTS(3s) vs. MCTS(0.2s) 30 Games")
m = Match(MCTSAgent(time_limit=3), MCTSAgent(time_limit=0.2), num_games=30)
m.play()

print("MCTS(10s) vs. Random 30 Games")
m = Match(MCTSAgent(time_limit=10), RandomAgent(), num_games=30)
m.play()

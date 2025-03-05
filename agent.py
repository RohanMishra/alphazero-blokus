import random
import numpy as np
import math
import time


class MCTSNode:
    def __init__(self, game_state, max_children=None, parent=None, move=None, player=None):
        """
        game_state: A clone of the BlokusGame state after the move is applied.
        parent: Parent node in the tree.
        move: The move that got us here (None for the root).
        player: The player who made the move to reach this state (None for the root).
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.player = player  # player who made the move; for root, leave as None.
        self.children = []
        # List of moves that haven't been tried from this state.
        self.untried_moves = game_state.get_legal_actions()
        if max_children:
            self.untried_moves = random.sample(self.untried_moves, min(max_children, len(self.untried_moves)))
        self.wins = 0
        self.visits = 0

    def ucb1(self, exploration=1.41):
        """Calculate the UCB1 score for a child node."""
        return self.wins / self.visits + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTSAgentDropout:
    def __init__(self, time_limit=5, max_children=20):
        """
        iterations: Number of MCTS iterations to run per move.
        """
        self.time_limit = time_limit
        self.max_children = max_children

    def play_random_game(self, state):
        while not state.game_over():
            legal_moves = state.get_legal_actions()
            # Randomly select a legal move.
            move = random.choice(legal_moves)
            state.apply_move(*move)
        return state.get_winner()

    def select_move(self, root_state):
        """
        Runs MCTS starting from the given root_state (a BlokusGame instance).
        Returns the move with the highest visit count from the root.
        """
        # Create the root node from a cloned state.
        root_node = MCTSNode(root_state.clone(), max_children=self.max_children, parent=None, move=None, player=None)

        end_time = time.time() + self.time_limit
        while time.time() < end_time:
            node = root_node
            state = root_state.clone()

            # 1. Selection: Traverse the tree until a node with untried moves or a terminal node is reached.
            while node.untried_moves == [] and node.children:
                node = self.select_child(node)
                state.apply_move(*node.move)

            # 2. Expansion: If node is non-terminal and has untried moves, expand one.
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state.apply_move(*move)
                # node.untried_moves.remove(move)

                # Remove the move from untried_moves by comparing elements manually.
                found_index = None
                for i, m in enumerate(node.untried_moves):
                    # Compare piece_id and offset directly, and use np.array_equal for orientation.
                    if m[0] == move[0] and np.array_equal(m[1], move[1]) and m[2] == move[2]:
                        found_index = i
                        break
                if found_index is not None:
                    del node.untried_moves[found_index]

                # The player who made the move is the one whose turn it was before move application.
                child_node = MCTSNode(
                    state.clone(), max_children=self.max_children, parent=node, move=move, player=node.game_state.player
                )
                node.children.append(child_node)
                node = child_node

            # 3. Simulation: Run a random playout from the current state until the game ends.
            winner = self.play_random_game(state)

            # 4. Backpropagation: Propagate the simulation result back up the tree.
            while node is not None:
                node.visits += 1
                # If this node represents a move (i.e. node.player is not None),
                # update win count from that player's perspective.
                if node.player is not None:
                    if winner == node.player:
                        node.wins += 1
                    elif winner == 0:  # draw
                        node.wins += 0.5
                    else:
                        node.wins -= 1
                node = node.parent

        # Choose the move from the root with the highest visit count.
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.move

    def select_child(self, node):
        """
        Selects and returns a child node using the UCB1 formula.
        """
        return max(node.children, key=lambda child: child.ucb1())


class MCTSAgent:
    def __init__(self, time_limit=5):
        """
        iterations: Number of MCTS iterations to run per move.
        """
        self.time_limit = time_limit

    def play_random_game(self, state):
        while not state.game_over():
            legal_moves = state.get_legal_actions()
            # Randomly select a legal move.
            move = random.choice(legal_moves)
            state.apply_move(*move)
        return state.get_winner()

    def select_move(self, root_state):
        """
        Runs MCTS starting from the given root_state (a BlokusGame instance).
        Returns the move with the highest visit count from the root.
        """
        # Create the root node from a cloned state.
        root_node = MCTSNode(root_state.clone(), parent=None, move=None, player=None)

        end_time = time.time() + self.time_limit
        while time.time() < end_time:
            node = root_node
            state = root_state.clone()

            # 1. Selection: Traverse the tree until a node with untried moves or a terminal node is reached.
            while node.untried_moves == [] and node.children:
                node = self.select_child(node)
                state.apply_move(*node.move)

            # 2. Expansion: If node is non-terminal and has untried moves, expand one.
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state.apply_move(*move)
                # node.untried_moves.remove(move)

                # Remove the move from untried_moves by comparing elements manually.
                found_index = None
                for i, m in enumerate(node.untried_moves):
                    # Compare piece_id and offset directly, and use np.array_equal for orientation.
                    if m[0] == move[0] and np.array_equal(m[1], move[1]) and m[2] == move[2]:
                        found_index = i
                        break
                if found_index is not None:
                    del node.untried_moves[found_index]

                # The player who made the move is the one whose turn it was before move application.
                child_node = MCTSNode(state.clone(), parent=node, move=move, player=node.game_state.player)
                node.children.append(child_node)
                node = child_node

            # 3. Simulation: Run a random playout from the current state until the game ends.
            winner = self.play_random_game(state)

            # 4. Backpropagation: Propagate the simulation result back up the tree.
            while node is not None:
                node.visits += 1
                # If this node represents a move (i.e. node.player is not None),
                # update win count from that player's perspective.
                if node.player is not None:
                    if winner == node.player:
                        node.wins += 1
                    elif winner == 0:  # draw
                        node.wins += 0.5
                    else:
                        node.wins -= 1
                node = node.parent

        # Choose the move from the root with the highest visit count.
        best_child = max(root_node.children, key=lambda c: c.visits)
        return best_child.move

    def select_child(self, node):
        """
        Selects and returns a child node using the UCB1 formula.
        """
        return max(node.children, key=lambda child: child.ucb1())


class RandomAgent:
    def __init__(self):
        pass

    def select_move(self, state):
        legal_moves = state.get_legal_actions()
        # Randomly select a legal move.
        return random.choice(legal_moves)

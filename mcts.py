import time
import math
import random
import numpy as np


class MCTS_Node:
    def __init__(self, game, args, parent=None, action=None):
        self.game = game
        self.args = args
        self.parent = parent
        self.action = action

        self.children = []

        self.unvisited_moves = np.zeros(game.action_size)
        legal_actions = self.game.get_legal_actions()
        self.unvisited_moves[legal_actions] = 1

        self.visit_count = 0
        self.value_sum = 0

    def is_expanded(self):
        return np.sum(self.unvisited_moves) == 0 and len(self.children) > 0

    def calc_ucb(self, child):
        exploit = 1 - ((child.value_sum / child.visit_count) + 1) / 2

        explore = math.sqrt(math.log(self.visit_count) / child.visit_count)
        return exploit + self.args["c"] * explore

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.calc_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def expand(self):
        action = np.random.choice(np.where(self.unvisited_moves == 1)[0])
        self.unvisited_moves[action] = 0

        child_state = self.game.clone()
        child_state.apply_move(action)
        child_state.flip_perspective(-1)

        child_node = MCTS_Node(child_state, self.args, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        game_copy = self.game.clone()
        while not game_copy.game_over():
            legal_moves = game_copy.get_legal_actions()
            move = random.choice(legal_moves)
            game_copy.apply_move(move)
        return game_copy.get_winner()

    def backprop(self, value):
        self.visit_count += 1
        self.value_sum += value

        if self.parent is not None:
            value = -value
            self.parent.backprop(value)


class MCTSNormal:
    def __init__(self, args):
        self.args = args

    def select(self, state):
        end_time = time.time() + self.args["time_limit"]

        # Ensure acting as player 1:
        state = state.clone()
        state.flip_perspective(state.player)
        root = MCTS_Node(state, self.args)

        # num_rollouts = 0

        while time.time() < end_time:
            # Selection
            # num_rollouts += 1
            node = root
            while node.is_expanded():
                node = node.select()

            # Expansion
            # Check if node terminal, if not, expand it
            is_terminal = node.game.game_over()
            if not is_terminal:
                node = node.expand()
                # Simulation
            value = node.simulate()

            # Backpropagation
            node.backprop(value)
        # print(f"Num rollouts: {num_rollouts}, Root Moves: {len(state.get_legal_actions())}")

        action_probs = np.zeros(root.game.action_size)
        for child in root.children:
            action_probs[child.action] = child.visit_count
        action_probs /= np.sum(action_probs)
        action = np.argmax(action_probs)

        if self.args["playmode"]:
            flipped_action_probs = np.zeros(root.game.action_size)
            for i in range(len(action_probs)):
                flipped_action_probs[state.flip_action(i)] = action_probs[i]
            return state.flip_action(np.argmax(action_probs)), flipped_action_probs
        else:
            return action_probs


class RandomAgent:
    def __init__(self):
        pass

    def select(self, state):
        legal_moves = state.get_legal_actions()
        # Randomly select a legal move.
        return random.choice(legal_moves)

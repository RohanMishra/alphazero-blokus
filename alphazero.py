import math
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
from mcts import MCTSNormal


class MCTS_Node:
    def __init__(self, game, args, parent=None, action=None, prior=0):
        self.game = game
        self.args = args
        # self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    def is_expanded(self):
        return len(self.children) > 0

    def calc_ucb(self, child):
        if child.visit_count == 0:
            exploit = 0
        else:
            exploit = -child.value_sum / child.visit_count
        explore = math.sqrt(self.visit_count) / (child.visit_count + 1) * child.prior
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

    def expand(self, policy):
        positive_prob_indices = np.where(policy > 0)[0]

        for action in positive_prob_indices:
            prob = policy[action]

            child_state = self.game.clone()
            child_state.apply_move(action)
            child_state.flip_perspective(-1)

            child_node = MCTS_Node(child_state, self.args, parent=self, action=action, prior=prob)
            self.children.append(child_node)

    def backprop(self, value):
        self.visit_count += 1
        self.value_sum += value

        if self.parent is not None:
            value = -value
            self.parent.backprop(value)


class MCTS:
    def __init__(self, args, model):
        self.args = args
        self.model = model

    @torch.no_grad()
    def select(self, state):
        end_time = time.time() + self.args["time_limit"]

        # Ensure acting as player 1:
        state = state.clone()
        state.flip_perspective(state.player)
        root = MCTS_Node(state, self.args)
        root.visit_count = 1

        policy, _ = self.model(torch.tensor(state.encode_board()).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * state.action_size
        )
        valid_moves = np.zeros(state.action_size)
        valid_moves[state.get_legal_actions()] = 1
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        root_num_children = len(root.children)
        # print("Init Policy: ", policy[policy > 0])
        # print("PreSearchPol: ")
        # top_k_ind_pol(policy, 4)
        # print("\n")

        num_rollouts = 0
        while time.time() < end_time and num_rollouts < self.args["max_rollouts"]:
            # Selection
            num_rollouts += 1
            node = root
            while node.is_expanded():
                node = node.select()

            # Expansion
            # Check if node terminal, if not, expand it
            is_terminal = node.game.game_over()
            if not is_terminal:

                ## TODO: Check why policy sometimes div by 0
                policy, value = self.model(torch.tensor(node.game.encode_board()).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = np.zeros(node.game.action_size)
                valid_moves[node.game.get_legal_actions()] = 1
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)
            else:
                value = node.game.get_winner()

            # Backpropagation
            node.backprop(value)

        # print(f"Num rollouts: {num_rollouts}, Root Num Children: {len(root.children)}")
        action_probs = np.zeros(root.game.action_size)
        for child in root.children:
            action_probs[child.action] = child.visit_count
        # print("Final Policy: ", action_probs[action_probs > 0])

        action_probs /= np.sum(action_probs)
        # print("PostSearchPol: ")
        # top_k_ind_pol(action_probs, 4)

        if self.args["playmode"]:
            flipped_action_probs = np.zeros(root.game.action_size)
            for i in range(len(action_probs)):
                flipped_action_probs[state.flip_action(i)] = action_probs[i]
            return state.flip_action(np.argmax(action_probs)), flipped_action_probs
        else:
            return action_probs


class AlphaZero:
    def __init__(self, model, optimizer, game_cls, args):
        self.model = model
        self.optimizer = optimizer
        self.game_cls = game_cls
        self.args = args
        self.mcts = MCTS(args, model)
        self.normal_mcts = MCTSNormal(args)

    def selfplay(self):
        memory = []
        game = self.game_cls()
        player = 1

        while True:
            neutral_state = game.clone()
            neutral_state.flip_perspective(player)

            if np.random.rand() < self.args["prob_normal_mcts"]:
                action_probs = self.normal_mcts.select(neutral_state)
            else:
                action_probs = self.mcts.select(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(game.action_size, p=temperature_action_probs)

            game.apply_move(neutral_state.flip_action(action))

            is_terminal = game.game_over()

            if is_terminal:
                winner = game.get_winner()
                returnmemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = winner * hist_player
                    returnmemory.append((hist_neutral_state.encode_board(), hist_action_probs, hist_outcome))
                return returnmemory

            player *= -1

    def train(self, memory):
        random.shuffle(memory)
        losses = []
        pol_losses = []
        val_losses = []
        for batchidx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchidx : min(len(memory), batchidx + self.args["batch_size"])]
            state, policy_target, value_target = zip(*sample)

            state, policy_target, value_target = (
                np.array(state),
                np.array(policy_target),
                np.array(value_target).reshape(-1, 1),
            )

            state = torch.tensor(state, dtype=torch.float32)
            policy_target = torch.tensor(policy_target, dtype=torch.float32)
            value_target = torch.tensor(value_target, dtype=torch.float32)

            out_policy, out_value = self.model(state)

            value_loss = F.mse_loss(out_value, value_target)
            policy_loss = F.cross_entropy(out_policy, policy_target)
            loss = self.args["pol_lambda"] * policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pol_losses.append(policy_loss.item())
            val_losses.append(value_loss.item())
            losses.append(loss.item())
        # print("Policy Loss: {:.4f}, Value Loss: {:.4f}".format(np.mean(pol_losses), np.mean(val_losses)))
        return losses, pol_losses, val_losses

    def learn(self):
        for i in trange(self.args["num_iterations"]):
            memory = []
            all_losses = []
            all_pol_losses = []
            all_val_losses = []

            self.model.eval()
            for selfplay_iter in trange(self.args["num_selfplay_iters"]):
                memory += self.selfplay()
            print("Collected {} samples".format(len(memory)))
            with open(f"memory_{i}.pkl", "wb") as f:
                pickle.dump(memory, f)
            self.model.train()
            for epoch in trange(self.args["num_epochs"]):
                losses, pol_losses, val_losses = self.train(memory)
                all_losses.append(losses)
                all_pol_losses.append(pol_losses)
                all_val_losses.append(val_losses)
                print("Epoch {}: Loss {:.4f}".format(epoch, np.mean(losses)))

            torch.save(self.model.state_dict(), f"model_{i}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{i}.pt")

            with open(f"losses_{i}.pkl", "wb") as f:
                pickle.dump((all_losses, all_pol_losses, all_val_losses), f)


class ResNet(nn.Module):
    def __init__(self, board_size, num_channels, action_size, num_resblocks, num_hidden):
        super().__init__()
        self.startblock = nn.Sequential(
            nn.Conv2d(num_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backbone = nn.ModuleList([ResBlock(num_hidden) for i in range(num_resblocks)])

        self.policyhead = nn.Sequential(
            nn.Conv2d(num_hidden, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * board_size * board_size, action_size),
        )

        self.valuehead = nn.Sequential(
            nn.Conv2d(num_hidden, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_channels * board_size * board_size, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.startblock(x)
        for resBlock in self.backbone:
            x = resBlock(x)
        policy = self.policyhead(x)
        value = self.valuehead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

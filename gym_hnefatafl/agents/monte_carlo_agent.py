import cProfile
import copy
import math
import random

import numpy as np

from gym_hnefatafl.agents.evaluation import evaluate, quick_evaluate, ANGLE_INTERVALS_3, calculate_angle_intervals
from gym_hnefatafl.agents.minimax_agent import MinimaxAgent
from gym_hnefatafl.envs import HnefataflEnv
from gym_hnefatafl.envs.board import Outcome, Player

QUICK_EVALUATION = True     # whether the nodes calls evaluate or quick_evaluate
USE_MINIMAX = True          # whether the algorithm uses the minimax algorithm to finish simulating a game

MONTE_CARLO_ITERATIONS = 100
MIN_NUM_VISITS_INTERNAL = 5  # may have to be much higher go uses 9*9
DEFAULT_SIGMA_SQUARED = 1

OUTCOME_BLACK_VALUE = -1
OUTCOME_WHITE_VALUE = 1
OUTCOME_DRAW_VALUE = 0


# represents a monte carlo search tree
class Tree(object):
    # board: the current board
    # player: the player that this agent represents
    def __init__(self, board, player):
        self.root = Node(player, None)
        self.board = board
        self.player = player
        self.white_minimax = MinimaxAgent(Player.white)
        self.black_minimax = MinimaxAgent(Player.black)

    # simulates an entire game
    def simulate_game(self):
        simulation_board_copy = copy.deepcopy(self.board)
        current_node = self.root

        # simulate actions within the tree until we are no longer at a stored node
        while simulation_board_copy.outcome == Outcome.ongoing:
            self.player = current_node.player
            next_node, action = current_node.choose_and_simulate_action(simulation_board_copy)
            if next_node is None:
                self.player = other_player(self.player)
                break
            else:
                current_node = next_node

        back_up_board_copy = copy.deepcopy(simulation_board_copy)

        # finish game
        while simulation_board_copy.outcome == Outcome.ongoing:
            self.__choose_and_simulate_action__(simulation_board_copy)
            self.player = other_player(self.player)

        # calculate game value
        game_value = OUTCOME_BLACK_VALUE if simulation_board_copy.outcome == Outcome.black \
            else OUTCOME_WHITE_VALUE if simulation_board_copy.outcome == Outcome.white \
            else OUTCOME_DRAW_VALUE

        # back up value
        while current_node is not None:
            back_up_board_copy.undo_last_action()
            current_node.back_up(game_value, back_up_board_copy)
            current_node = current_node.parent
        print("game simulated")

    # chooses an action and simulates it.
    def __choose_and_simulate_action__(self, board):
        if USE_MINIMAX:
            if self.player == Player.white:
                board.do_action(self.white_minimax.make_move(board), self.player)
            else:
                board.do_action(self.black_minimax.make_move(board), self.player)
        else:
            board.do_action(random.choice(board.get_valid_actions(self.player)), self.player)

    # returns the best action found
    def get_best_action(self):
        best_action = None
        if self.root.player == Player.white:
            best_mu = -math.inf
            for child, action in self.root.children_with_actions:
                if child.mean > best_mu:
                    best_mu = child.mean
                    best_action = action
        else:
            best_mu = math.inf
            for child, action in self.root.children_with_actions:
                if child.mean < best_mu:
                    best_mu = child.mean
                    best_action = action
        return best_action


# represents a node within the monte carlo search tree (that is actually stored in memory -> see the paper)
class Node(object):
    def __init__(self, player, parent_node):
        self.parent = parent_node
        self.children_with_actions = []  # list of (children, action) pairs
        self.action_to_children_dict = {}
        self.number_of_visits = 0
        self.sum_of_values = 0
        self.sum_of_squared_values = 0

        # sign of this needs to return something meaningful, so instead of 0 this is set to machine epsilon
        self.mean = np.finfo(float).eps if player == Player.white else -np.finfo(float).eps

        self.variance = 0.0
        self.player = player
        self.is_internal = False

    # chooses and simulates an action
    def choose_and_simulate_action(self, board):
        self.number_of_visits += 1
        self.is_internal = self.number_of_visits > MIN_NUM_VISITS_INTERNAL
        action = self.__choose_action__(board)
        board.do_action(action, self.player)

        # If this node has already been visited, either create a child node or return an existing one.
        # Otherwise return None
        if self.number_of_visits > 1:
            if action not in self.action_to_children_dict.keys():
                child_node = Node(other_player(self.player), self)
                self.children_with_actions.append((child_node, action))
                self.action_to_children_dict[action] = child_node
                return child_node, action
            else:
                return self.action_to_children_dict[action], action
        else:
            return None, action

    # chooses an action
    def __choose_action__(self, board):
        actions = board.get_valid_actions(self.player)
        probabilities = self.get_action_probabilities(actions, board)

        # Nach einer Stunde googeln herausgefunden, dass numpy nicht in der Lage ist,
        # ein Array aus Tupeln zu machen und man stattdessen diesen Quatsch machen muss.
        actions_array = np.empty(len(actions), dtype=object)
        actions_array[:] = actions

        action = np.random.choice(actions_array, p=probabilities)
        return action

    # returns the probabilities for each action
    # actions: the possible actions
    # board: the current board
    def get_action_probabilities(self, actions, board):
        mus = np.empty(len(actions))
        sigmas_squared = np.empty(len(actions))
        for i, action in enumerate(actions):
            if action in self.action_to_children_dict.keys():
                child = self.action_to_children_dict[action]
                mus[i] = child.mean
                sigmas_squared[i] = child.variance
            else:
                board.do_action(action, self.player)
                evaluation = quick_evaluate(board, self.player) if QUICK_EVALUATION else evaluate(board, self.player)
                mus[i] = 1 if evaluation == math.inf else -1 if evaluation == -math.inf else evaluation / len(
                    actions)
                board.undo_last_action()
                sigmas_squared[i] = DEFAULT_SIGMA_SQUARED

        mu_0_index = np.argmax(mus)
        mu_0 = mus[mu_0_index]
        sigma_0_squared = sigmas_squared[mu_0_index]

        # (returns the indices that would sort the array in ascending order). Quicksort is specified so that
        # it is NOT stable because we don't want to see the same behaviour in every game
        sorted_indices = np.argsort(mus, kind='quicksort')

        # cap big values so that their powers don't explode. (when overflow happens,
        # the value becomes 0 which results in infinite probabilities)
        sorted_indices[sorted_indices < len(actions) - 20] = len(actions) - 20

        ########################################################################################################
        # parameter that somehow needs to reflect "urgency of a move". mustn't be zero
        # could possibly be chosen as "pieces captured"
        a = 1
        eps = (0.1 + 1 / np.power(2, len(actions) - 1 - sorted_indices) + a) / len(actions)
        ########################################################################################################

        probabilities = np.exp(-2.4 * (mu_0 - mus) / np.sqrt(2 * (sigma_0_squared + sigmas_squared))) + eps
        probabilities /= np.sum(probabilities)
        return probabilities

    def back_up(self, value, board):
        self.sum_of_values += value
        self.sum_of_squared_values += value * value
        self.update_mean_variance(board)

    def update_mean_variance(self, board):
        ################################################################################################
        # parameter that somehow needs to reflect "points on the board", i. e. empty intersections in go
        # could possibly be chosen as "number of pieces on the board"
        p = 11 * 11 * (board.white_pieces + board.black_pieces)
        ################################################################################################
        if self.parent is None:
            if self.player == Player.white:
                sign = +1
            else:
                sign = -1
            self.mean = self.sum_of_values/self.number_of_visits * sign
        else:
            self.mean = self.sum_of_values/self.number_of_visits * -np.sign(self.parent.mean)
        self.variance = (self.sum_of_squared_values - self.number_of_visits*self.mean*self.mean + 4*p*p) \
                        / (self.number_of_visits + 1)


# returns the opponent of the given player
def other_player(player):
    return Player.white if player == Player.black else Player.white


class MonteCarloAgent(object):
    def __init__(self, player):
        self.player = player
        if not ANGLE_INTERVALS_3:
            calculate_angle_intervals()

    # chooses a random move and returns it
    # in order for games to finish in a reasonable amount of time,
    # the agent always sends the king to one of the corners if able
    # (this causes white to win basically all the time)
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):

        prof = cProfile.Profile()
        prof.enable()
        tree = Tree(env.get_board(), self.player)
        for i in range(MONTE_CARLO_ITERATIONS):
            tree.simulate_game()
        best_action = tree.get_best_action()
        prof.disable()
        prof.print_stats(sort=2)

        return best_action

    # does nothing in this agent, but is here because other agents need it
    def give_reward(self, reward):
        pass

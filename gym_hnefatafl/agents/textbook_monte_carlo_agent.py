import cProfile
import copy
import math
import random

from gym_hnefatafl.agents.evaluation import ANGLE_INTERVALS_3, calculate_angle_intervals
from gym_hnefatafl.agents.minimax_agent import MinimaxAgent
from gym_hnefatafl.envs.board import Player, Outcome

USE_MINIMAX = True          # whether the algorithm uses the minimax algorithm to finish simulating a game
PROFILE = True
MONTE_CARLO_ITERATIONS = 100
EXPLORATION_PARAMETER = math.sqrt(2)


class Tree(object):
    def __init__(self, board, player):
        self.root = Node(player)
        self.board = board
        self.player = player
        self.white_minimax = MinimaxAgent(Player.white)
        self.black_minimax = MinimaxAgent(Player.black)
        self.total_simulations = 0

    def simulate_game(self):
        simulation_board_copy = copy.deepcopy(self.board)
        current_node = self.root
        game_history = []
        self.total_simulations += 1

        # selection and expansion
        while simulation_board_copy.outcome == Outcome.ongoing:
            self.player = current_node.player
            # selection
            if current_node.children_dict != {}:
                action = current_node.select(self.total_simulations)
                current_node = current_node.children_dict[action]
                game_history.append(action)

            # expansion
            else:
                current_node.expand(simulation_board_copy)
                action = current_node.select(self.total_simulations)
                game_history.append(action)
                break

        # rollout
        while simulation_board_copy.outcome == Outcome.ongoing:
            self.__select_rollout_move__(simulation_board_copy)
            self.player = other_player(self.player)

        print(str(simulation_board_copy.outcome))

        # backpropagation
        current_node = self.root
        for action in game_history:
            current_node.update(simulation_board_copy.outcome)
            current_node = current_node.children_dict[action]
        current_node.update(simulation_board_copy.outcome)

    # makes moves until the game is decided
    def __select_rollout_move__(self, board):
        if USE_MINIMAX:
            if self.player == Player.white:
                board.do_action(self.white_minimax.make_move(board), self.player)
            else:
                board.do_action(self.black_minimax.make_move(board), self.player)
        else:
            board.do_action(random.choice(board.get_valid_actions(self.player)), self.player)

    def get_best_action(self):
        most_simulations = 0
        most_simulated_action = []
        for action in self.root.children_dict:
            child = self.root.children_dict[action]
            if child.simulations > most_simulations:
                most_simulations = child.simulations
                most_simulated_action = [action]
            elif child.simulations == most_simulations:
                most_simulated_action.append(action)
        return random.choice(most_simulated_action)


class Node(object):
    def __init__(self, player):
        self.player = player
        self.results = {Outcome.black: 0, Outcome.white: 0, Outcome.draw: 0}
        self.simulations = 0
        self.children_dict = {}

    def win_outcome(self):
        return Outcome.black if self.player == Player.black else Outcome.white

    def update(self, outcome):
        self.results[outcome] += 1
        self.simulations += 1

    def select(self, total_simulations):
        best_actions = []   # this is a list so that we can later draw an action
        #                     randomly out of all actions with the best value
        best_value = -math.inf
        for action in self.children_dict.keys():
            child = self.children_dict[action]
            # the following formula is adapted from  here: https://en.wikipedia.org/w/
            #                       index.php?title=Monte_Carlo_tree_search&oldid=871362180#Exploration_and_exploitation
            value = (child.results[self.win_outcome()]) / (self.simulations + 1)\
                    + EXPLORATION_PARAMETER * math.sqrt(2 * math.log(total_simulations) / (self.simulations + 1))
            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        return random.choice(best_actions)

    def expand(self, board):
        for action in board.get_valid_actions(self.player):
            child = Node(other_player(self.player))
            self.children_dict[action] = child


class TextbookMonteCarloAgent(object):
    def __init__(self, player):
        self.player = player
        if not ANGLE_INTERVALS_3:
            calculate_angle_intervals()

    # chooses a random move and returns it
    # in order for games to finish in a reasonable amount of time,
    # the agent always sends the king to one of the corners if able
    # (this causes white to win basically all the time)
    def make_move(self, env) -> ((int, int), (int, int)):
        if PROFILE:
            prof = cProfile.Profile()
            prof.enable()
        tree = Tree(env.get_board(), self.player)
        for i in range(MONTE_CARLO_ITERATIONS):
            tree.simulate_game()
            if i % 10 == 9:
                print(str(i + 1) + " games simulated")
        best_action = tree.get_best_action()
        if PROFILE:
            prof.disable()
            prof.print_stats(sort=2)

        return best_action

    # does nothing in this agent, but is here because other agents need it
    def give_reward(self, reward):
        pass


# returns the opponent of the given player
def other_player(player):
    return Player.white if player == Player.black else Player.white

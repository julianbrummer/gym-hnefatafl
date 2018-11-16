import copy
import math
import operator
import random
from bisect import bisect_left

from gym_hnefatafl.agents.Evaluation import evaluate
from gym_hnefatafl.envs import HnefataflEnv
from gym_hnefatafl.envs.board import Player, HnefataflBoard, Outcome

MINIMAX_SEARCH_DEPTH = 3


# returns the other player
def other_player(this_player):
    return Player.black if this_player == Player.white else Player.white


# returns "<" for black and ">" for white
def minimax_comp(this_player):
    return operator.__lt__ if this_player == Player.black else operator.__gt__


# makes a copy of the board for each action and executes the action on the board.
# The (action, board) pairs are inserted into a list that is sorted by the number
# of pieces that are captured when performing the action
def reordered_boards_after_action(board, turn_player):
    boards = []
    captures = []
    for action in board.get_valid_actions(turn_player):
        board_copy = copy.deepcopy(board)
        captured = -len(board_copy.do_action(action, turn_player))
        insertion_index = bisect_left(captures, captured)
        boards.insert(insertion_index, (action, board_copy))
    return boards


# an agent that uses a minimax search for estimating which move is best
class MinimaxAgent(object):
    def __init__(self, player):
        self.player = player

    # chooses a move based on a minimax search with the __evaluate__ heuristic further below
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        # minimax_action, minimax_value = self.minimax_search(env.get_board(), self.player, 0)
        minimax_action, minimax_value = self.alphabeta(env.get_board(), 0, -math.inf, math.inf, self.player)
        return random.choice(env.action_space) if minimax_action is None else minimax_action

    # does nothing yet
    def give_reward(self, reward):
        pass

    # returns the minimax action and minimax value for the given board and the turn player.
    # The calculation is cut off at the depth that is specified at the top of this file
    # white is maximizer, black is minimizer
    def minimax_search(self, board: HnefataflBoard, turn_player, depth):
        # evaluate this node using the heuristic if the max depth is reached
        if depth == MINIMAX_SEARCH_DEPTH or board.outcome != Outcome.ongoing:
            return None, evaluate(board, turn_player)

        # initialize minimax value with either positive or negative infinity
        best_minimax_value_found = math.inf if turn_player == Player.black else -math.inf
        best_action_found = None

        # loop over all actions and calculate the action with best minimax value
        for action, next_board in reordered_boards_after_action(board, turn_player):
            subtree_minimax_action, subtree_minimax_value = self.minimax_search(next_board, other_player(turn_player),
                                                                                depth + 1)
            # if better play is found, update minimax action and minimax value
            if minimax_comp(turn_player)(subtree_minimax_value, best_minimax_value_found):
                best_minimax_value_found = subtree_minimax_value
                best_action_found = action
        return best_action_found, best_minimax_value_found

    # does the same as minimax_search, but uses alpha-beta-pruning to make it faster. initialize with
    # depth = 0, alpha = -math.inf, beta = math.inf
    def alphabeta(self, board, depth, alpha, beta, turn_player):
        if depth == MINIMAX_SEARCH_DEPTH or board.outcome != Outcome.ongoing:
            return None, evaluate(board, turn_player)
        if turn_player == Player.white:
            value = -math.inf
            best_action = None
            for action, next_board in reordered_boards_after_action(board, Player.white):
                subtree_best_action, subtree_alpha = self.alphabeta(next_board, depth + 1, alpha, beta, Player.black)
                if value < subtree_alpha:
                    value = subtree_alpha
                    best_action = action
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            return best_action, value
        else:
            value = math.inf
            best_action = None
            for action, next_board in reordered_boards_after_action(board, Player.black):
                subtree_best_action, subtree_beta = self.alphabeta(next_board, depth + 1, alpha, beta, Player.white)
                if value > subtree_beta:
                    value = subtree_beta
                    best_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_action, value

import copy
import math
import operator

from gym_hnefatafl.envs import HnefataflEnv
from gym_hnefatafl.envs.board import Player, TileState, HnefataflBoard

MINIMAX_SEARCH_DEPTH = 2


# returns the other player
def other_player(this_player):
    return Player.black if this_player == Player.white else Player.white


# returns "<" for black and ">" for white
def minimax_comp(this_player, a, b):
    return operator.__lt__(a, b) if this_player == Player.black else operator.__gt__(a, b)


# returns (number of black pieces, number of white pieces) on the given board
def calculate_number_of_pieces(board):
    white = 0
    black = 0
    for row in board.board:
        for tile in row:
            if tile == TileState.white:
                white += 1
            elif tile == TileState.black:
                black += 1
    return black, white


# an agent that uses a minimax search for estimating which move is best
class MinimaxAgent(object):
    def __init__(self, player):
        self.player = player

    # chooses a move based on a minimax search with the __evaluate__ heuristic further below
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        minimax_action, minimax_value = self.minimax_search(env.get_board(), self.player, 0)
        return minimax_action

    # does nothing yet
    def give_reward(self, reward):
        pass

    # returns the minimax action and minimax value for the given board and the turn player.
    # The calculation is cut off at the depth that is specified at the top of this file
    # white is maximizer, black is minimizer
    def minimax_search(self, board: HnefataflBoard, turn_player, depth):
        # evaluate this node using the heuristic if the max depth is reached
        if depth == MINIMAX_SEARCH_DEPTH:
            return None, self.__evaluate__(board)

        # initialize minimax value with either positive or negative infinity
        best_minimax_value_found = math.inf if turn_player == Player.black else -math.inf
        best_action_found = None

        # loop over all actions and calculate the action with best minimax value
        for action in board.get_valid_actions(turn_player):
            # deep copy so that nothing on the real board is changed
            board_copy = copy.deepcopy(board)
            board_copy.do_action(action, turn_player)
            subtree_minimax_action, subtree_minimax_value = self.minimax_search(board_copy, other_player(turn_player),
                                                                                depth + 1)
            # if better play is found, update minimax action and minimax value
            if minimax_comp(turn_player, subtree_minimax_value, best_minimax_value_found):
                best_minimax_value_found = subtree_minimax_value
                best_action_found = action
        return best_action_found, best_minimax_value_found

    # estimates the value of the given board
    def __evaluate__(self, board):
        black_pieces, white_pieces = calculate_number_of_pieces(board)
        return white_pieces - black_pieces

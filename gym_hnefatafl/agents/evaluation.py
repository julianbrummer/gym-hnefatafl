import itertools
import math
import numpy as np
from enum import IntEnum

from gym_hnefatafl.envs.board import Outcome, TileState, Player

BOARD_PRESENCE_WEIGHT = 4
SUPERIORITY_WEIGHT = 5
KING_IN_TROUBLE_WEIGHT = 10
KING_TURNS_TO_CORNER_WEIGHT = 3

# board presence
KING_BONUS_FACTOR = 2.5
AREA_FACTOR = [1.3, 1.1, 1.3, 1.1, 1, 1.1, 1.3, 1.1, 1.3]

# king in trouble
KING_IN_TROUBLE_EXP_BASE = 1.4

# king turns to corner
KING_TURNS_TO_CORNER_EXP_BASE = 50


# evaluates the given board and returns a number based on its value
def evaluate(board, player):
    if board.outcome == Outcome.white:
        return math.inf
    if board.outcome == Outcome.black:
        return -math.inf
    if board.outcome == Outcome.draw:
        return math.inf if player == Player.black else -math.inf
    return superiority_rating(board) + king_in_trouble_rating(board) + king_turns_to_corner(board)\
            + board_presence_rating(board)\


# does a material comparison and returns a number based on that
def superiority_rating(board):
    return SUPERIORITY_WEIGHT*(2*board.white_pieces - board.black_pieces)


# returns (number of black pieces, number of white pieces) on the entire given board or in only an area
def number_of_pieces(board, area=None):
    if area is None:
        return board.black_pieces, board.white_pieces
    else:
        white = 0
        black = 0
        for i in area.indices():
            if board.board[i] == TileState.white:
                white += 1
            elif board.board[i] == TileState.black:
                black += 1
        return black, white


# enum that describes an area on the board. Corner areas are 4x4, the middle is 3x3 and edge areas are 4x3 or 3x4
class Area(IntEnum):
    top_left = 0
    top = 1
    top_right = 2
    left = 3
    middle = 4
    right = 5
    bottom_left = 6
    bottom = 7
    bottom_right = 8

    # returns the indices of all board tiles that this enum covers
    def indices(self):
        first = range(1, 5)
        middle = range(5, 8)
        last = range(8, 12)
        indices = {
            # itertools.product returns the cartesian product of the lists
            0: itertools.product(first, first),
            1: itertools.product(first, middle),
            2: itertools.product(first, last),
            3: itertools.product(middle, first),
            4: itertools.product(middle, middle),
            5: itertools.product(middle, last),
            6: itertools.product(last, first),
            7: itertools.product(last, middle),
            8: itertools.product(last, last),
        }
        return indices[self]


# divides the board into areas and calculates an area presence value between -1 and 1 for each one
# the area that the king is in gets an additional bonus
# the area ratings are then weighted by the values in the list above and summed up
def board_presence_rating(board):
    total_value = 0
    for area in Area:
        white, black = number_of_pieces(board, area)
        # make values not exceed 1 and -1
        area_value = max(min(white - black/2, 1), -1)
        if board.king_position in area.indices():
            area_value *= KING_BONUS_FACTOR
        total_value += area_value * AREA_FACTOR[area]
    return total_value


# calculates a value for the king's threat of capture as an exponential
# function in the number of the black pieces around the king
# at 0, the function returns 0
# at more pieces, it will return exponentially more (well, less because it's negative..)
def king_in_trouble_rating(board):
    black_pieces_around_king = 0
    king_x, king_y = board.king_position
    if board.board[king_x - 1, king_y] == TileState.black:
        black_pieces_around_king += 1
    if board.board[king_x + 1, king_y] == TileState.black:
        black_pieces_around_king += 1
    if board.board[king_x, king_y - 1] == TileState.black:
        black_pieces_around_king += 1
    if board.board[king_x, king_y + 1] == TileState.black:
        black_pieces_around_king += 1
    return -KING_IN_TROUBLE_WEIGHT*(KING_IN_TROUBLE_EXP_BASE**black_pieces_around_king - 1)


# calculates how many moves the king needs in a row to reach the nearest corner
# (just by movement alone, not checking for capture of black pieces)
def king_turns_to_corner(board):
    turns_from = np.full((13, 13), -1)
    this_list = [(1, 1), (1, 11), (11, 1), (11, 11)]
    for position in this_list:
        turns_from[position] = 0
    next_list = []
    current_step_count = 1
    while True:
        for to_position in this_list:
            for end_position, from_position in board.get_valid_actions_for_piece(to_position):
                if board.king_position == from_position:
                    return -KING_TURNS_TO_CORNER_WEIGHT*KING_TURNS_TO_CORNER_EXP_BASE**(-current_step_count)
                if turns_from[from_position] == -1:
                    turns_from[from_position] = current_step_count
                    next_list.append(from_position)
        if len(next_list) == 0:
            return 0
        this_list = next_list
        next_list = []
        current_step_count += 1

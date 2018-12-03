import itertools
import math
import random

import numpy as np
from enum import IntEnum

from gym_hnefatafl.envs.board import Outcome, TileState, Player, TileMoveState

BOARD_PRESENCE_WEIGHT = 4
SUPERIORITY_WEIGHT = 5
KING_IN_TROUBLE_WEIGHT = 10
KING_TURNS_TO_CORNER_WEIGHT = 6
COVERED_ANGLE_WEIGHT = 1
SAME_AXIS_AS_KING_WEIGHT = 1

# board presence
KING_BONUS_FACTOR = 2.5
AREA_FACTOR = [1.3, 1.1, 1.3, 1.1, 1, 1.1, 1.3, 1.1, 1.3]

# king in trouble
KING_IN_TROUBLE_EXP_BASE = 1.4

# king turns to corner
KING_TURNS_TO_CORNER_EXP_BASE = 50

# penalty for squares that are not on the same axis as the king
# (must range from 0 to 1 with 1 being no penalty and 0 meaning not counted)
SQUARE_OFF_AXIS_FACTOR = 0.5


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



# does only the stuff that can be calculated quickly
def quick_evaluate(board, player):
    if board.outcome == Outcome.white:
        return math.inf
    if board.outcome == Outcome.black:
        return -math.inf
    if board.outcome == Outcome.draw:
        return math.inf if player == Player.black else -math.inf
    return superiority_rating(board) + king_in_trouble_rating(board) + covered_angle_rating(board) \
           + same_axis_as_king_rating(board)


# currently tested, add stuff as you like, but don't make it too slow
def king_centered_evaluation(board, player):
    if board.outcome == Outcome.white:
        return math.inf
    if board.outcome == Outcome.black:
        return -math.inf
    if board.outcome == Outcome.draw:
        return math.inf if player == Player.black else -math.inf
    return random_jiggle() + covered_angle_rating(board) + same_axis_as_king_rating(board) + superiority_rating(board)\
           + king_turns_to_corner(board)


# adds a bit of randomness when the best actions evaluate to the same value
def random_jiggle():
    return random.random()*2**(-20)


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


# the relative squares and their evaluation order in a circle around the king with radius 3
# these are sorted by the left ends (looking counter-clockwise) of the angle intervals in ascending order
# in other words like this:
ANGLE_CALCULATION_ORDER_3 = [
    (3, 0),     (2, 0),     (3, 1),     (1, 0),     (2, 1),     (2, 2),     (1, 1),     (1, 2),     (1, 3),     # NE
    (0, 3),     (0, 2),     (-1, 3),    (0, 1),     (-1, 2),    (-2, 2),    (-1, 1),    (-2, 1),    (-3, 1),    # NW
    (-3, 0),    (-2, 0),    (-3, -1),   (-1, 0),    (-2, -1),   (-2, -2),   (-1, -1),   (-1, -2),   (-1, -3),   # SW
    (0, -3),    (0, -2),    (1, -3),    (0, -1),    (1, -2),    (2, -2),    (1, -1),    (2, -1),    (3, -1)     # SE
]

# this intervals that the respective squares cover
# method below populates this list
ANGLE_INTERVALS_3 = []


# does the calculation for the list above
def calculate_angle_intervals():
    for relative_x, relative_y in ANGLE_CALCULATION_ORDER_3:
        if relative_x < 0:
            if relative_y < 0:
                right_angle = np.angle(complex(relative_x - .5 * SQUARE_OFF_AXIS_FACTOR, relative_y + .5 * SQUARE_OFF_AXIS_FACTOR))
                left_angle = np.angle(complex(relative_x + .5 * SQUARE_OFF_AXIS_FACTOR, relative_y - .5 * SQUARE_OFF_AXIS_FACTOR))
            elif relative_y == 0:
                right_angle = np.angle(complex(relative_x + .5, relative_y + .5))
                left_angle = np.angle(complex(relative_x + .5, relative_y - .5))
            else:
                right_angle = np.angle(complex(relative_x + .5 * SQUARE_OFF_AXIS_FACTOR, relative_y + .5 * SQUARE_OFF_AXIS_FACTOR))
                left_angle = np.angle(complex(relative_x - .5 * SQUARE_OFF_AXIS_FACTOR, relative_y - .5 * SQUARE_OFF_AXIS_FACTOR))
        elif relative_x == 0:
            if relative_y < 0:
                right_angle = np.angle(complex(relative_x - .5, relative_y + .5))
                left_angle = np.angle(complex(relative_x + .5, relative_y + .5))
            else:
                right_angle = np.angle(complex(relative_x + .5, relative_y - .5))
                left_angle = np.angle(complex(relative_x - .5, relative_y - .5))
        else:
            if relative_y < 0:
                right_angle = np.angle(complex(relative_x - .5 * SQUARE_OFF_AXIS_FACTOR, relative_y - .5 * SQUARE_OFF_AXIS_FACTOR))
                left_angle = np.angle(complex(relative_x + .5 * SQUARE_OFF_AXIS_FACTOR, relative_y + .5 * SQUARE_OFF_AXIS_FACTOR))
            elif relative_y == 0:
                right_angle = np.angle(complex(relative_x - .5, relative_y - .5))
                left_angle = np.angle(complex(relative_x - .5, relative_y + .5))
            else:
                right_angle = np.angle(complex(relative_x + .5 * SQUARE_OFF_AXIS_FACTOR, relative_y - .5 * SQUARE_OFF_AXIS_FACTOR))
                left_angle = np.angle(complex(relative_x - .5 * SQUARE_OFF_AXIS_FACTOR, relative_y + .5 * SQUARE_OFF_AXIS_FACTOR))
        if right_angle < 0:
            right_angle += 2 * math.pi
        if left_angle < 0:
            left_angle += 2 * math.pi
        ANGLE_INTERVALS_3.append((right_angle, left_angle))


# calculates the angle that the squares of all black pieces in a circle of radius 3 around the king cover
# returns the negative of that scaled between 0 and -1
def covered_angle_rating(board):
    union = []
    edge_pieces = []
    king_x, king_y = board.king_position
    for (relative_x, relative_y), (right, left) in zip(ANGLE_CALCULATION_ORDER_3, ANGLE_INTERVALS_3):
        pos_x = king_x + relative_x
        pos_y = king_y + relative_y
        if 1 <= pos_x <= 11 and 1 <= pos_y <= 11:
            if board.board[(pos_x, pos_y)] == TileState.black:
                # split interval if it crosses over 0
                if right > left:
                    edge_pieces.append(right)
                    right = 0
                if not union:
                    union.append((right, left))
                else:
                    # find the first interval that is not fully included in the current one
                    first_subinterval_index = len(union) - 1
                    while first_subinterval_index > 0 and union[first_subinterval_index - 1][0] > right:
                        first_subinterval_index -= 1
                    # delete all elements after the found index
                    del union[first_subinterval_index + 1:]
                    # if last interval and the current one are disjunct, append it. Else join them.
                    if union[first_subinterval_index][1] <= right:
                        union.append((right, left))
                    else:
                        union[first_subinterval_index] = (union[first_subinterval_index][0], left)

    # do the same thing for the right pieces of the intervals where the angle has crossed over 0
    left = 2 * math.pi
    for right in edge_pieces:
        if not union:
            union.append((right, left))
        else:
            first_subinterval_index = len(union) - 1
            while first_subinterval_index > 0 and union[first_subinterval_index - 1][0] > right:
                first_subinterval_index -= 1
            del union[first_subinterval_index + 1:]
            if union[first_subinterval_index][1] <= right:
                union.append((right, left))
            else:
                del union[first_subinterval_index + 1:]
                union[first_subinterval_index] = (union[first_subinterval_index][0], left)

    # calculate the covered angle
    covered_angle = 0
    for interval in union:
        covered_angle += interval[1] - interval[0]
    return -covered_angle * COVERED_ANGLE_WEIGHT / (2 * math.pi)


# consider the distance from the king along the horizontal or vertical axis to a black
# piece with no other piece in between. This function calculates the sum of reciprocals
# of these distances in each of the four directions and returns the negative of that value
# (scaled between 0 and -1)
def same_axis_as_king_rating(board):
    axis_sum = 0
    king_x, king_y = board.king_position
    for x_other in reversed(range(1, king_x)):
        if board.move_board[x_other, king_y] == TileMoveState.traversable:
            continue
        elif board.board[x_other, king_y] == TileState.black:
            axis_sum += 1/(king_x - x_other)
        break
    # second direction
    for x_other in range(king_x + 1, 12):
        if board.move_board[x_other, king_y] == TileMoveState.traversable:
            continue
        elif board.board[x_other, king_y] == TileState.black:
            axis_sum += 1/(x_other - king_x)
        break
    # third direction
    for y_other in reversed(range(1, king_y)):
        if board.move_board[king_x, y_other] == TileMoveState.traversable:
            continue
        elif board.board[king_x, y_other] == TileState.black:
            axis_sum += 1/(king_y - y_other)
        break
    # forth direction
    for y_other in range(king_y + 1, 12):
        if board.move_board[king_x, y_other] == TileMoveState.traversable:
            continue
        elif board.board[king_x, y_other] == TileState.black:
            axis_sum += 1/(y_other - king_y)
        break
    return -axis_sum / 4 * SAME_AXIS_AS_KING_WEIGHT

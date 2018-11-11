import numpy as np
import copy
from enum import IntEnum

from gym_hnefatafl.envs.rule_config import MAX_NUMBER_OF_TURNS, MAX_NUMBER_OF_TURNS_WITHOUT_CAPTURE


class Player(IntEnum):
    white = 1
    black = 2


class TileState(IntEnum):
    empty = 0   # neutral
    white = 1   # hostile to black
    black = 2   # hostile to white and king
    king = 3    # unarmed, hostile to no one
    throne = 4  # the empty throne is hostile to any piece
    corner = 5  # target tile for the king, hostile to any piece
    border = 6  # not a reachable tile, but hostile to king


class TileBattleState(IntEnum):
    hostile = -1
    neutral = 0
    allied = 1


# Describes whether a tile is traversable.
# Note that the empty throne is traversable by any piece but can not be occupied by a soldier
# Corners and the empty throne can only be occupied by the king
class TileMoveState(IntEnum):
    traversable = 0    # empty, throne
    blocking = 1  # white, black, border, corner


class Outcome(IntEnum):
    ongoing = 0
    white = 1
    black = 2
    draw = 3


class HnefataflBoard:

    def __init__(self):
        # empty
        self.board = np.zeros((13, 13))         # TileStates
        self.white_board = np.zeros((13, 13))   # TileBattleStates
        self.black_board = np.zeros((13, 13))   # TileBattleStates
        self.move_board = np.zeros((13, 13))    # TileMoveStates
        self.player_board = np.zeros((13, 13))  # Players

        self.king_position = (6, 6)

        # holds all board states and the frequency how often they occurred
        # with the current number of pieces on the board (meaning it is reset
        # every time a piece is captured). If we wouldn't reset it, then
        # especially with the random agents we could get a very large memory
        # footprint. Since pieces can't get back onto the board, it is
        # possible to reset this whenever a piece is captured
        self.board_states_dict = {self.board.tobytes(): 1}

        # the outcome of the current match
        self.outcome = Outcome.ongoing

        # turn counts for draw condition
        self.turn_count = 0
        self.turns_without_capture_count = 0

        # Whether this board prints moves and captures to the console. This shouldn't
        # happen when the game tree is searched because during that time only
        # possibilities are considered, but no actual move is made. Turn this off for
        # all instances of this class that are a copy of the original class.
        self.print_to_console = True

        self.reset_board()

    def reset_board(self):
        # empty
        self.board = np.zeros((13, 13))        # TileStates
        self.white_board = np.zeros((13, 13))  # TileBattleStates
        self.black_board = np.zeros((13, 13))  # TileBattleStates
        self.move_board = np.zeros((13, 13))   # TileMoveStates
        self.player_board = np.zeros((13, 13)) # Player

        # black (four battalions)
        self.board[1, 4:9] = TileState.black
        self.board[2, 6] = TileState.black
        self.board[11, 4:9] = TileState.black
        self.board[10, 6] = TileState.black
        self.board[4:9, 1] = TileState.black
        self.board[6, 2] = TileState.black
        self.board[4:9, 11] = TileState.black
        self.board[6, 10] = TileState.black
        # white (fortress)
        self.board[5:8, 5:8] = TileState.white
        self.board[4, 6] = TileState.white
        self.board[6, 8] = TileState.white
        self.board[6, 4] = TileState.white
        self.board[8, 6] = TileState.white
        # king
        self.board[6, 6] = TileState.king
        self.king_position = (6, 6)
        # border
        self.board[0, :] = TileState.border
        self.board[12, :] = TileState.border
        self.board[:, 0] = TileState.border
        self.board[:, 12] = TileState.border
        # corner
        self.board[1, 1] = TileState.corner
        self.board[1, 11] = TileState.corner
        self.board[11, 11] = TileState.corner
        self.board[11, 1] = TileState.corner

        self.update_board_states()
        self.board_states_dict = {self.board.tobytes(): 1}
        self.outcome = Outcome.ongoing
        self.turn_count = 0
        self.turns_without_capture_count = 0

    # creates a deep copy of this board
    def copy(self):
        board_copy = HnefataflBoard()
        board_copy.board = np.copy(self.board)
        board_copy.king_position = self.king_position
        board_copy.board_states_dict = copy.deepcopy(self.board_states_dict)
        board_copy.outcome = self.outcome
        board_copy.turn_count = self.turn_count
        board_copy.turns_without_capture_count = self.turns_without_capture_count
        board_copy.print_to_console = False
        board_copy.update_board_states()
        return board_copy

    def update_board_states(self):

        # battle state for white soldiers (corners, empty throne and black soldiers are hostile)
        # white soldiers are allied and the king is neutral, since he is unarmed
        self.white_board = np.zeros((13, 13))
        hostile_mask = (self.board == TileState.black) | (self.board == TileState.corner) \
                       | (self.board == TileState.throne)
        np.place(self.white_board, self.board == TileState.white, TileBattleState.allied)
        np.place(self.white_board, hostile_mask, TileBattleState.hostile)

        # battle state for black soldiers (corners, empty throne and white soldiers are hostile)
        # black soldiers are allied
        self.black_board = np.zeros((13, 13))
        hostile_mask = (self.board == TileState.white) | (self.board == TileState.corner) \
                       | (self.board == TileState.throne) | (self.board == TileState.king)
        np.place(self.black_board, self.board == TileState.black, TileBattleState.allied)
        np.place(self.black_board, hostile_mask, TileBattleState.hostile)

        # movable state for any player (borders, corners, and soldiers are blocking)
        # anything else is traversable
        self.move_board = np.zeros((13, 13))
        blocking_mask = (self.board == TileState.border) | (self.board == TileState.white) \
                        | (self.board == TileState.black) | (self.board == TileState.king)
        np.place(self.move_board, blocking_mask, TileMoveState.blocking)

        # player state
        self.player_board = np.zeros((13, 13))
        np.place(self.player_board, self.board == TileState.black, Player.black)
        np.place(self.player_board, (self.board == TileState.white) | (self.board == TileState.king), Player.white)

    #  Checks whether "player" can do action "move".
    #  move = ((fromX,fromY),(toX,toY))
    def can_do_action(self, move, player):
        (position_from, position_to) = move
        return self.player_board[position_from] == player and move in self.get_valid_actions_for_piece(position_from)

    # returns all valid actions for a player as a list of actions
    def get_valid_actions(self, turn_player):
        valid_actions = []
        for position, player in np.ndenumerate(self.player_board):
            if player == turn_player:
                valid_actions.extend(self.get_valid_actions_for_piece(position))
        if len(valid_actions) == 0:
            self.outcome = Outcome.white if turn_player == Player.black else Outcome.black
            if self.print_to_console:
                print("It is " + str(turn_player) + "'s turn, but they can't make any moves. "
                      + str(Player.white if turn_player == Player.black else Player.black) + " wins!")
        return valid_actions

    # returns all valid actions for a piece at a given position as a list of actions
    def get_valid_actions_for_piece(self, position):
        x, y = position
        is_king = self.board[x, y] == TileState.king
        valid_actions = []
        # first direction
        for x_other in reversed(range(0, x)):
            if self.move_board[x_other, y] == 0 and (is_king or self.board[x_other, y] != TileState.corner):
                if (not (x_other, y) == (6, 6)) or is_king:  # exclude throne if the piece is not the king
                    valid_actions.append(((x, y), (x_other, y)))
            else:
                break
        # second direction
        for x_other in range(x + 1, 13):
            if self.move_board[x_other, y] == 0 and (is_king or self.board[x_other, y] != TileState.corner):
                if (not (x_other, y) == (6, 6)) or is_king:  # exclude throne if the piece is not the king
                    valid_actions.append(((x, y), (x_other, y)))
            else:
                break
        # third direction
        for y_other in reversed(range(0, y)):
            if self.move_board[x, y_other] == 0 and (is_king or self.board[x, y_other] != TileState.corner):
                if (not (x, y_other) == (6, 6)) or is_king:  # exclude throne if the piece is not the king
                    valid_actions.append(((x, y), (x, y_other)))
            else:
                break
        # forth direction
        for y_other in range(y + 1, 13):
            if self.move_board[x, y_other] == 0 and (is_king or self.board[x, y_other] != TileState.corner):
                if (not (x, y_other) == (6, 6)) or is_king:  # exclude throne if the piece is not the king
                    valid_actions.append(((x, y), (x, y_other)))
            else:
                break
        return valid_actions

    # executes "move" for the player "player" whose turn it is
    def do_action(self, move, player):
        (from_x, from_y), (to_x, to_y) = move
        if self.can_do_action(move, player):
            # increase turn counts
            self.turn_count += 1
            self.turns_without_capture_count += 1
            if self.print_to_console:
                print(str(player) + " moves a piece from " + str((from_x, from_y)) + " to " + str((to_x, to_y)))

            # if king is moving: update king position and check if he reached a corner
            if self.board[from_x, from_y] == TileState.king:
                self.king_position = (to_x, to_y)
                if self.board[self.king_position] == TileState.corner:
                    self.outcome = Outcome.white
                    if self.print_to_console:
                        print("The king escapes to corner " + str((to_x, to_y)) + ". White wins!")
                    return []

            # update the board itself and capture pieces if applicable
            self.board[to_x, to_y] = self.board[from_x, from_y]
            self.board[from_x, from_y] = TileState.empty if (from_x, from_y) != (6, 6) else TileState.throne
            captured_pieces = self.capture((to_x, to_y), player)

            # update the board_states_dictionary so that we know whether the present board has occurred for the 3rd time
            if self.board.tobytes() in self.board_states_dict:
                self.board_states_dict[self.board.tobytes()] += 1
                if self.board_states_dict[self.board.tobytes()] == 3:
                    self.outcome = Outcome.draw
                    if self.print_to_console:
                        print("The same board state has occurred three times. The game ends in a draw!")
                    return []
            else:
                self.board_states_dict[self.board.tobytes()] = 1

            # check if draw conditions by turn count are met
            if self.turn_count == MAX_NUMBER_OF_TURNS and self.outcome == Outcome.ongoing:
                self.outcome = Outcome.draw
            if self.turns_without_capture_count == MAX_NUMBER_OF_TURNS_WITHOUT_CAPTURE \
                    and self.outcome == Outcome.ongoing:
                self.outcome = Outcome.draw

            self.update_board_states()
            return captured_pieces
        else:
            raise Exception(str(player) + " tried to make move " + str(move) + ", but that move is not possible.")

    # captures all enemy pieces around the position "position_to" that the player "player" has just moved a piece to
    def capture(self, position_to, turn_player):
        x, y = position_to
        other_player_board = self.get_other_player_board(turn_player)
        captured_pieces = []

        # TileState.white for Player.black, TileState.black for Player.white
        # this way is necessary because capturing the king works differently and is done further below
        opponent_pawn_tile_state = TileState.white if turn_player == Player.black else TileState.black

        # this flag marks if at least one piece has been captured
        # in this case the board_states_dict is reset after capture calculation
        has_captured = False

        # check capture right
        if self.board[x + 1, y] == opponent_pawn_tile_state and other_player_board[x + 2, y] == TileBattleState.hostile:
            self.board[x + 1, y] = TileState.empty
            captured_pieces.append((x + 1, y))
            has_captured = True
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x + 1, y)))
        # check capture left
        if self.board[x - 1, y] == opponent_pawn_tile_state and other_player_board[x - 2, y] == TileBattleState.hostile:
            self.board[x - 1, y] = TileState.empty
            captured_pieces.append((x - 1, y))
            has_captured = True
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x - 1, y)))
        # check capture bottom
        if self.board[x, y + 1] == opponent_pawn_tile_state and other_player_board[x, y + 2] == TileBattleState.hostile:
            self.board[x, y + 1] = TileState.empty
            captured_pieces.append((x, y + 1))
            has_captured = True
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x, y + 1)))
        # check capture top
        if self.board[x, y - 1] == opponent_pawn_tile_state and other_player_board[x, y - 2] == TileBattleState.hostile:
            self.board[x, y - 1] = TileState.empty
            captured_pieces.append((x, y - 1))
            has_captured = True
            if self.print_to_console:
                print(str(turn_player) + " captures piece at " + str((x, y - 1)))

        # check capture king
        king_x, king_y = self.king_position
        if self.white_board[king_x + 1, king_y] == TileBattleState.hostile \
                and self.white_board[king_x - 1, king_y] == TileBattleState.hostile \
                and self.white_board[king_x, king_y + 1] == TileBattleState.hostile \
                and self.white_board[king_x, king_y - 1] == TileBattleState.hostile:
            self.outcome = Outcome.black
            has_captured = True
            captured_pieces.append((king_x, king_y))
            if self.print_to_console:
                print("Black wins by capturing the king at " + str(self.king_position) + "!")

        # reset turn count and board_states_dict
        if has_captured:
            self.turns_without_capture_count = 0
            self.board_states_dict.clear()

        return captured_pieces

    # returns the player's board who is not "player":
    # white_board if player is black
    # black_board if player is white
    def get_other_player_board(self, player):
        return self.black_board if player == Player.white else self.white_board

    def __str__(self):
        return "Gameboard: \n" + str(self.board) + \
               "\nWhiteboard: \n" + str(self.white_board) + \
               "\nBlackboard: \n" + str(self.black_board) + \
               "\nMoveboard: \n" + str(self.move_board) + \
               "\nPlayerboard: \n" + str(self.player_board)


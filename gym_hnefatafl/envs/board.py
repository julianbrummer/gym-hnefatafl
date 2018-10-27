import numpy as np
from enum import IntEnum


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


class HnefataflBoard:

    def __init__(self):
        # empty
        self.board = np.zeros((13, 13))        # TileStates
        self.white_board = np.zeros((13, 13))  # TileBattleStates
        self.black_board = np.zeros((13, 13))  # TileBattleStates
        self.king_board = np.zeros((13, 13))   # TileBattleStates
        self.move_board = np.zeros((13, 13))   # TileMoveStates
        self.player_board = np.zeros((13, 13)) # Player

        self.reset_board()

    def reset_board(self):
        # empty
        self.board = np.zeros((13, 13))        # TileStates
        self.white_board = np.zeros((13, 13))  # TileBattleStates
        self.black_board = np.zeros((13, 13))  # TileBattleStates
        self.king_board = np.zeros((13, 13))   # TileBattleStates
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

    def update_board_states(self):

        # battle state for white soldiers (corners, empty throne and black soldiers are hostile)
        # white soldiers are allied and the king is neutral, since he is unarmed
        self.white_board = np.zeros((13, 13))
        hostile_mask = (self.board == TileState.black) | (self.board == TileState.corner) | (self.board == TileState.throne)
        np.place(self.white_board, self.board == TileState.white, TileBattleState.allied)
        np.place(self.white_board, hostile_mask, TileBattleState.hostile)

        # battle state for black soldiers (corners, empty throne and white soldiers are hostile)
        # black soldiers are allied and the king is neutral, since he is unarmed
        self.black_board = np.zeros((13, 13))
        hostile_mask = (self.board == TileState.white) | (self.board == TileState.corner) | (self.board == TileState.throne)
        np.place(self.black_board, self.board == TileState.black, TileBattleState.allied)
        np.place(self.black_board, hostile_mask, TileBattleState.hostile)

        # battle state for king (borders, corners, empty throne and black soldiers are hostile)
        # anything else is neutral
        self.king_board = np.zeros((13, 13))
        hostile_mask = (self.board == TileState.border) | (self.board == TileState.corner) \
                       | (self.board == TileState.throne) | (self.board == TileState.black)
        np.place(self.king_board, hostile_mask, TileBattleState.hostile)

        # movable state for any player (borders, corners, and soldiers are blocking)
        # anything else is traversable
        self.move_board = np.zeros((13, 13))
        blocking_mask = (self.board == TileState.border) | (self.board == TileState.corner) \
                       | (self.board == TileState.white) | (self.board == TileState.black) \
                       | (self.board == TileState.king)
        np.place(self.move_board, blocking_mask, TileMoveState.blocking)

        # player state
        self.player_board = np.zeros((13, 13))
        np.place(self.player_board, self.board == TileState.black, Player.black)
        np.place(self.player_board, (self.board == TileState.white) | (self.board == TileState.king), Player.white)

    #  Checks whether "player" can do action "move".
    #  move = ((fromX,fromY),(toX,toY))
    def can_do_action(self, move, player):
        (position_from, position_to) = move
        return self.get_player_board(player)[position_from] == TileBattleState.allied \
               and move in self.get_valid_actions_for_piece(position_from)

    # returns all valid actions for a player as a list of actions
    def get_valid_actions(self, player):
        valid_actions = []
        for position, tileBattleState in np.ndenumerate(self.get_player_board(player)):
            if tileBattleState == TileBattleState.allied:
                valid_actions.extend(self.get_valid_actions_for_piece(position))
        return valid_actions

    # returns all valid actions for a piece at a given position as a list of actions
    def get_valid_actions_for_piece(self, position):
        x, y = position
        is_king = self.board[x, y] == TileState.king
        valid_actions = []
        # first direction
        for x_other in reversed(range(0, x)):
            if self.move_board[x_other, y] == 0 or is_king and self.board[x_other, y] == TileState.corner:
                valid_actions.append(((x, y), (x_other, y)))
            else:
                break
        # second direction
        for x_other in range(x + 1, 13):
            if self.move_board[x_other, y] == 0 or is_king and self.board[x_other, y] == TileState.corner:
                valid_actions.append(((x, y), (x_other, y)))
            else:
                break
        # third direction
        for y_other in reversed(range(0, y)):
            if self.move_board[x, y_other] == 0 or is_king and self.board[x, y_other] == TileState.corner:
                valid_actions.append(((x, y), (x, y_other)))
            else:
                break
        # forth direction
        for y_other in range(y + 1, 13):
            if self.move_board[x, y_other] == 0 or is_king and self.board[x, y_other] == TileState.corner:
                valid_actions.append(((x, y), (x, y_other)))
            else:
                break
        return valid_actions

    # executes "move" for the player "player" whose turn it is
    def do_action(self, move, player):
        print(str(move))
        (from_x, from_y), (to_x, to_y) = move
        if self.can_do_action(move, player):
            print(str(player) + " moves a piece from " + str((from_x, from_y)) + " to " + str((to_x, to_y)))
            self.board[to_x, to_y] = self.board[from_x, from_y]
            self.board[from_x, from_y] = TileState.empty if (from_x, from_y) != (6, 6) else TileState.throne
            self.capture((to_x, to_y), player)
            # TODO check for game over
            # TODO check whether board state has occured three times
            self.update_board_states()
        else:
            raise Exception(str(player) + " tried to make move " + str(move) + ", but that move is not possible.")

    # captures all enemy pieces around the position that the player "player" has just moved a piece to
    def capture(self, position_to, turn_player):
        x, y = position_to
        other_player_board = self.get_other_player_board(turn_player)
        # capture right
        if other_player_board[x + 1, y] == TileBattleState.allied and other_player_board[x + 2, y] == TileBattleState.hostile:
            self.board[x + 1, y] = TileState.empty
            print(str(turn_player) + " captures piece at " + str((x + 1, y)))
        # capture left
        if other_player_board[x - 1, y] == TileBattleState.allied and other_player_board[x - 2, y] == TileBattleState.hostile:
            self.board[x - 1, y] = TileState.empty
            print(str(turn_player) + " captures piece at " + str((x - 1, y)))
        # capture bottom
        if other_player_board[x, y + 1] == TileBattleState.allied and other_player_board[x, y + 2] == TileBattleState.hostile:
            self.board[x, y + 1] = TileState.empty
            print(str(turn_player) + " captures piece at " + str((x, y + 1)))
        # capture top
        if other_player_board[x, y - 1] == TileBattleState.allied and other_player_board[x, y - 2] == TileBattleState.hostile:
            self.board[x, y - 1] = TileState.empty
            print(str(turn_player) + " captures piece at " + str((x, y - 1)))

    # returns either self.black_board or self.white_board depending on whether player is Player.black or player.white
    def get_player_board(self, player):
        return self.black_board if player == Player.black else self.white_board

    # returns the opposite of self.get_player_board, i. e. the player's board who is not "player"
    def get_other_player_board(self, player):
        return self.black_board if player == Player.white else self.white_board

    def __str__(self):
        return "Gameboard: \n" + str(self.board) + \
               "\nWhiteboard: \n" + str(self.white_board) + \
               "\nBlackboard: \n" + str(self.black_board) + \
               "\nKingboard: \n" + str(self.king_board) + \
               "\nMoveboard: \n" + str(self.move_board) + \
               "\nPlayerboard: \n" + str(self.player_board)


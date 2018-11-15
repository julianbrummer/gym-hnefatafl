from gym_hnefatafl.envs.board import Outcome, TileState


# evaluates the given board and returns a number based on its value
def evaluate(board):
    if board.outcome == Outcome.white:
        return 100
    if board.outcome == Outcome.black:
        return -100
    if board.outcome == Outcome.draw:
        return 0
    black_pieces, white_pieces = calculate_number_of_pieces(board)
    return white_pieces - black_pieces


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
